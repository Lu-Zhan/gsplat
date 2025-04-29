from bdb import Breakpoint
import decimal
import wandb
import json
import math
import os
import time
from typing import Tuple
from einops import rearrange

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

# from datasets.colmap_with_intrinsics import Dataset, Parser
# from datasets.blender_with_intrinsics import Dataset, Parser
# from datasets.blender_with_intrinsics_2 import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.utils import CameraOptModule, colormap, set_random_seed
from gsplat.strategy import DefaultStrategy
from gsplat.cuda._wrapper import spherical_harmonics

from pbr.light import CubemapLight
from pbr.surface_rendering import SurfaceRenderer, hdr_to_ldr, tonemap

from gs_model import create_splats_with_optimizers, create_backgrpound_splats_with_optimizers
from renderer import rasterize_splats, render_reflection, render_envmap, obtain_irradiance

from utils.geo_utils import transform_normals_to_image_coord #, obtain_surface_position
from utils.losses import get_tv_loss, anisotropy_loss

os.environ['TORCH_CUDA_ARCH_LIST'] = ''


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        # self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # wandb
        wandb.init(
            project="scene_gen",
            name=os.path.basename(cfg.result_dir),
            config=cfg,
        )

        if cfg.use_dataset_type == "colmap":
            from datasets.colmap_with_intrinsics import Dataset, Parser
        elif cfg.use_dataset_type == "blender":
            from datasets.blender_with_intrinsics import Dataset, Parser
        else:
            raise NotImplementedError(f"Dataset type {cfg.use_dataset_type} is not supported.")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
            load_cubemap=cfg.load_cubemap,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=True,
            load_intrinsics=True,    # luzhan: loading intrinsics
        )
        self.valset = Dataset(
            self.parser, 
            split="val",
            load_depths=True,
            load_intrinsics=True,    # luzhan: loading intrinsics
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            random_drop_pts=cfg.random_drop_pts,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        # Background Model
        self.splats_bg, self.optimizers_bg = create_backgrpound_splats_with_optimizers(
            init_num_pts=cfg.init_num_pts_bg,
            radius_of_sphere=cfg.radius_of_sphere_bg,
            init_scale=0.2,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            device=self.device,
        )

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d, 
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
        
        self.hdr_scaler = torch.nn.Parameter(torch.tensor([0.], requires_grad=True).to(self.device))
        self.hdr_scaler_optimizer = torch.optim.Adam([self.hdr_scaler], lr=1e-5)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
        
        self.light_model = CubemapLight(height=256)
        self.surface_renderer = SurfaceRenderer()

        if cfg.init_lighting:
            self.init_lighting(cubemap=self.parser.cubemap)
    
    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=32,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                gt_depths = data["depths"].to(device)  # [1, M]

            if cfg.direct_depth_loss:
                gt_depth_map = data["depth_map"].to(device)   # [1, H, W, 1]
            
            # luzhan: load gt intrinsics and normals
            if cfg.intrinsics_loss:
                gt_intrinsics = data["intrinsics"].to(device)
            
            if cfg.direct_normal_loss:
                gt_normals = data["normals"].to(device)
            
            if cfg.irradiance_loss:
                gt_irradiance = data["irradiance"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # luzhan: use bg splats only enabling surface rendering
            render_with_bg = self.cfg.render_with_bg \
                and (cfg.irradiance_loss or cfg.surface_rendering_loss) \
                and (step > cfg.surface_rendering_start_iter)

            # forward
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = rasterize_splats(
                splats=self.splats,
                splats_bg=self.splats_bg if render_with_bg else None,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
                render_with_bg=False, # whether to render with bg splats
            )

            # luzhan: unpack intrinsics and depths from renders
            if renders.shape[-1] == 4:
                colors, intrinsics, depths = renders[..., 0:3], None, renders[..., 3:4]
            elif renders.shape[-1] == 9:
                colors, intrinsics, depths = renders[..., 0:3], renders[..., 3:-1], renders[..., -1:]
            else:
                colors, intrinsics, depths = renders, None, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                pixels = pixels * masks[..., None]
                colors = colors * masks[..., None]

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if step % 2000 == 0:
                wandb.log({
                    'train/vol': wandb.Image(colors.data.cpu().numpy()), 
                    'train/albedo': wandb.Image(intrinsics[..., :3].data.cpu().numpy()),
                    }, step,
                )

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths_tensor = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths_tensor = depths_tensor.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths_tensor > 0.0, 1.0 / depths_tensor, torch.zeros_like(depths_tensor))
                disp_gt = 1.0 / gt_depths  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            # luzhan: new depth loss
            if cfg.direct_depth_loss:
                if step > cfg.depth_start_iter:
                    curr_direct_depth_lambda = cfg.direct_depth_lambda
                else:
                    curr_direct_depth_lambda = 0.0

                gt_depth_map = (gt_depth_map - gt_depth_map.min()) / (gt_depth_map.max() - gt_depth_map.min() + 1e-8)
                depth_map = torch.clamp(depths, min=0)
                depth_map = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)

                gt_depth_median = torch.median(gt_depth_map[gt_depth_map > 0.0])
                depth_median = torch.median(depth_map[depth_map > 0.0])
                depth_map = depth_map * gt_depth_median / (depth_median + 1e-8)

                directdepthloss = F.l1_loss(depth_map, gt_depth_map) * self.scene_scale
                loss += directdepthloss * curr_direct_depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals_tensor = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals_tensor * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda
            
            if cfg.anisotropy_loss and step % 10 == 0:
                anisotropyloss = anisotropy_loss(torch.exp(self.splats['scales']), th=cfg.anisotropy_th)
                loss += anisotropyloss * cfg.anisotropy_lambda
            
            if cfg.normals_tv_loss:
                normals_tv_loss = get_tv_loss(
                    gt_image=pixels[0].permute(2, 0, 1),
                    prediction=normals[0].permute(2, 0, 1),
                )

                loss += normals_tv_loss * cfg.normals_tv_lambda
            
            if cfg.normal_dir_loss:
                normals_cam = transform_normals_to_image_coord(normals, camtoworlds)
                normals_cam = -torch.nn.functional.normalize(normals_cam, dim=-1)[..., -1]   # (1, H, W)
                normal_dir_loss = torch.where(normals_cam > 0.0, normals_cam, torch.zeros_like(normals_cam)).mean()
                loss += normal_dir_loss * cfg.normal_dir_lambda
            
            if cfg.intrinsics_tv_loss:
                intrinsics_tv_loss = get_tv_loss(
                    gt_image=pixels[0].permute(2, 0, 1),
                    prediction=intrinsics[0].permute(2, 0, 1),
                )

                loss += intrinsics_tv_loss * cfg.intrinsics_tv_lambda

            # luzhan: add more losses, including intrinsics loss, direct normal loss
            if cfg.intrinsics_loss:
                if step > cfg.intrinsics_start_iter:
                    curr_intrinsics_lambda = cfg.intrinsics_lambda
                else:
                    curr_intrinsics_lambda = 0.0
                intrinsics_loss = F.l1_loss(intrinsics, gt_intrinsics)
                loss += intrinsics_loss * curr_intrinsics_lambda
            
            if cfg.direct_normal_loss:
                if step > cfg.direct_normal_start_iter:
                    curr_normal_lambda = cfg.direct_normal_lambda
                else:
                    curr_normal_lambda = 0.0
                normals_tensor = transform_normals_to_image_coord(normals, camtoworlds)
                normals_tensor = F.normalize(normals_tensor, dim=-1)
                direct_normal_loss = (1 - (normals_tensor * gt_normals).sum(dim=-1).mean())   
                loss += direct_normal_loss * curr_normal_lambda

            if (cfg.irradiance_loss or cfg.surface_rendering_loss) and step >= cfg.surface_rendering_start_iter:
                pbr_result = render_reflection(
                    splats=self.splats,
                    surface_renderer=self.surface_renderer,
                    light_model=self.light_model,
                    hdr_scaler=self.hdr_scaler,
                    Ks=Ks,
                    hw=(height, width),
                    camtoworlds=camtoworlds,
                    depth_map=depths,
                    normal_map=normals,
                    albedo_map=intrinsics[..., :3],
                    roughness_map=intrinsics[..., 3:4] * (1.0 - 0.04) + 0.04,   # roughness in [0.04, 1.0], as GSIR
                    metallic_map=intrinsics[..., 4:5],
                    render_with_bg=render_with_bg,
                    splats_bg=self.splats_bg if render_with_bg else None,
                    distance_to_surface=cfg.distance_to_surface * self.scene_scale,
                )

                irradiance = pbr_result["diffuse_light"][None, ...]
                rendered_image = pbr_result["render_rgb"][None, ...]

                if step % 2000 == 0:
                    wandb.log({
                        'train/surf': wandb.Image(rendered_image.data.cpu().numpy()), 
                        'train/irr': wandb.Image(irradiance.data.cpu().numpy())
                        }, step,
                    )

                if step == cfg.surface_rendering_start_iter:
                    self.update_hdr_scaler(init_scaler=(pixels.mean() / (rendered_image.mean() + 1e-8)))
                else:
                    if cfg.irradiance_loss:
                        irradiance_loss = F.l1_loss(irradiance, gt_irradiance)
                        loss += irradiance_loss * cfg.irradiance_lambda
                    
                    if cfg.surface_rendering_loss:
                        surf_l1loss = F.l1_loss(rendered_image, pixels)
                        surf_ssimloss = 1.0 - self.ssim(
                            pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
                        )
                        surface_rendering_loss = surf_l1loss * (1.0 - cfg.ssim_lambda) + surf_ssimloss * cfg.ssim_lambda
                        loss += surface_rendering_loss * cfg.surface_rendering_lambda

            loss.backward()

            desc = f"loss={loss.data:.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"dep loss={depthloss.data:.4f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.data:.4f}"
            if cfg.direct_depth_loss and step > cfg.depth_start_iter:
                desc += f"ddep loss={directdepthloss.data:.4f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.data:.6f}| "
            if step > cfg.surface_rendering_start_iter:
                if cfg.surface_rendering_loss:
                    desc += f"surf loss={surface_rendering_loss.data:.4f}| "
                if cfg.irradiance_loss:
                    desc += f"irr loss={irradiance_loss.data:.4f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                # self.writer.add_scalar("train/loss", loss.data, step)
                # self.writer.add_scalar("train/l1loss", l1loss.data, step)
                # self.writer.add_scalar("train/ssimloss", ssimloss.data, step)
                # self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                # self.writer.add_scalar("train/mem", mem, step)
                wandb.log(
                    {
                        "train/loss": loss.data,
                        "train/l1loss": l1loss.data,
                        "train/ssimloss": ssimloss.data,
                        "train/num_GS": len(self.splats["means"]),
                        "train/mem": mem,
                    }, step
                )
                if cfg.depth_loss:
                    # self.writer.add_scalar("train/depthloss", depthloss.data, step)
                    wandb.log({"train/depthloss": depthloss.data}, step)
                if cfg.normal_loss:
                    # self.writer.add_scalar("train/normalloss", normalloss.data, step)
                    wandb.log({"train/normalloss": normalloss.data}, step)
                if cfg.dist_loss:
                    # self.writer.add_scalar("train/distloss", distloss.data, step)
                    wandb.log({"train/distloss": distloss.data}, step)
                if cfg.direct_depth_loss:
                    # self.writer.add_scalar("train/directdepthloss", directdepthloss.data, step)
                    wandb.log({"train/directdepthloss": directdepthloss.data}, step)
                
                # luzhan: add more losses, including intrinsics loss, direct normal loss
                if cfg.intrinsics_loss:
                    # self.writer.add_scalar("train/intrinsics_loss", intrinsics_loss.data, step)
                    wandb.log({"train/intrinsics_loss": intrinsics_loss.data}, step)
                if cfg.direct_normal_loss:
                    # self.writer.add_scalar("train/direct_normal_loss", direct_normal_loss.data, step)
                    wandb.log({"train/direct_normal_loss": direct_normal_loss.data}, step)
                
                if step > cfg.surface_rendering_start_iter:
                    if cfg.irradiance_loss:
                        # self.writer.add_scalar("train/irradiance_loss", irradiance_loss.data, step)
                        wandb.log({"train/irradiance_loss": irradiance_loss.data}, step)
                    if cfg.surface_rendering_loss:
                        # self.writer.add_scalar("train/surface_rendering_loss", surface_rendering_loss.data, step)    
                        wandb.log({"train/surface_rendering_loss": surface_rendering_loss.data}, step)

                # if cfg.tb_save_image:
                #     canvas = (
                #         torch.cat([pixels, colors[..., :3]], dim=2)
                #         .detach()
                #         .cpu()
                #         .numpy()
                #     )
                #     canvas = canvas.reshape(-1, *canvas.shape[2:])
                #     self.writer.add_image("train/render", canvas, step)
                # self.writer.flush()
            
            # luzhan: use only the front splats for densification
            if render_with_bg:
                num_gs_front = self.splats["means"].shape[0]
                grad_info = info["gradient_2dgs"].grad.clone()[:, :num_gs_front]

                for k in [
                    'radii', 'means2d', 'depths', 'ray_transforms', 
                    'opacities', 'normals', 'tiles_per_gauss', 'gradient_2dgs',
                ]:
                    info[k] = info[k][:, :num_gs_front]
                
                info['gradient_2dgs'].grad = grad_info

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # luzhan: optimize hdr scaler
            self.hdr_scaler_optimizer.step()
            self.hdr_scaler_optimizer.zero_grad(set_to_none=True)

            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                        "splats_bg": self.splats_bg.state_dict(),
                        "hdr_scaler": self.hdr_scaler.data,
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        dataset = self.trainset if cfg.eval_trainset else self.valset
        valloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1,
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "psnr_surf": [], "ssim_surf": [], "lpips_surf": []}

        # luzhan: update render_dir
        curr_render_dir = f"{self.render_dir}/step_{step:05d}" if cfg.eval_trainset else f"{self.render_dir}_eval/step_{step:05d}"
        os.makedirs(curr_render_dir, exist_ok=True)
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            # breakpoint()
            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = rasterize_splats(
                splats=self.splats,
                splats_bg=self.splats_bg if cfg.render_with_bg else None,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                render_with_bg=cfg.render_with_bg,
            )  # [1, H, W, 3]
            excepted_depths = colors[..., -1:]
            depths_tensor = excepted_depths.clone()
            colors = torch.clamp(colors, 0.0, 1.0)
    
            # luzhan: take intrinsics
            intrinsics = colors[..., 3:-1]  # (1, H, W, 5)

            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            save_path = f"{curr_render_dir}/images/val_{i:04d}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))
            if i == 0: wandb.log({"val/colors": wandb.Image((canvas * 255).astype(np.uint8))})

            # write depths
            # render_median = (render_median - render_median.min()) / (render_median.max() - render_median.min())
            # render_median = render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()

            gt_depths = data["depth_map"]
            gt_depths = (gt_depths - gt_depths.min()) / (gt_depths.max() - gt_depths.min())
            gt_depths = gt_depths.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()

            excepted_depths = torch.where(excepted_depths > 0.0, 1 / excepted_depths, torch.zeros_like(excepted_depths))
            excepted_depths = (excepted_depths - excepted_depths.min()) / (excepted_depths.max() - excepted_depths.min())
            excepted_depths = excepted_depths.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()

            # align the median of depths
            gt_median = np.median(gt_depths)
            excepted_median = np.median(excepted_depths)
            excepted_depths = excepted_depths * gt_median / (excepted_median + 1e-8)

            canvas = np.concatenate([gt_depths, excepted_depths], axis=1)

            save_path = f"{curr_render_dir}/depths/val_{i:04d}_depth_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # write normals
            normals_tensor = normals.clone()
            normals = transform_normals_to_image_coord(normals, camtoworlds)
            normals = torch.nn.functional.normalize(normals, dim=-1)
            # normals_tensor = normals.clone()
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = transform_normals_to_image_coord(normals_from_depth, camtoworlds)
            normals_from_depth = torch.nn.functional.normalize(normals_from_depth, dim=-1)
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).squeeze().cpu().numpy()
            
            gt_normals = torch.nn.functional.normalize(data["normals"], dim=-1)
            gt_normals = (gt_normals * 0.5 + 0.5).detach().squeeze().cpu().numpy()

            canvas = np.concatenate([gt_normals, normals, normals_from_depth], axis=1)
            canvas = (canvas * 255).astype(np.uint8)

            save_path = f"{curr_render_dir}/normals/val_{i:04d}_normal_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, canvas)

            # write distortions
            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                colormap(render_dist.cpu().numpy()[0])
                .permute((1, 2, 0))
                .numpy()
                .astype(np.uint8)
            )
            save_path = f"{curr_render_dir}/distortions/val_{i:04d}_distortions_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, render_dist)
  
            # write alphas
            alphas = alphas.repeat(1, 1, 1, 3).squeeze(0).detach().cpu().numpy()
            alphas = (alphas - np.min(alphas)) / (np.max(alphas) - np.min(alphas) + 1e-8)
            alphas = (alphas * 255).astype(np.uint8)
            save_path = f"{curr_render_dir}/alphas/val_{i:04d}_alphas_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, alphas)

            # luzhan: write intrinsics
            albedo = intrinsics[..., :3]    # (1, H, W, 3)
            roughness =  intrinsics[..., 3:4].repeat(1, 1, 1, 3)   # (1, H, W, 3)
            metallic = intrinsics[..., 4:].repeat(1, 1, 1, 3)   # (1, H, W, 3)

            gt_intrinsics = data["intrinsics"]   # (1, H, W, 5)
            gt_albedo = gt_intrinsics[..., :3]    # (1, H, W, 3)
            gt_roughness = gt_intrinsics[..., 3:4].repeat(1, 1, 1, 3)   # (1, H, W, 3)
            gt_metallic = gt_intrinsics[..., 4:].repeat(1, 1, 1, 3)   # (1, H, W, 3)

            canvas_top = torch.cat([gt_albedo, gt_roughness, gt_metallic], dim=2).squeeze(0).cpu().numpy()
            canvas_bottom = torch.cat([albedo, roughness, metallic], dim=2).squeeze(0).cpu().numpy()    # 
            canvas = np.concatenate([canvas_top, canvas_bottom], axis=0)
            save_path = f"{curr_render_dir}/intrinsics/val_{i:04d}_intrinsics_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))
            if i == 0: wandb.log({"val/intrinsics": wandb.Image((canvas * 255).astype(np.uint8))})

            pbr_result = render_reflection(
                splats=self.splats,
                surface_renderer=self.surface_renderer,
                light_model=self.light_model,
                hdr_scaler=self.hdr_scaler,
                Ks=Ks,
                hw=(height, width),
                camtoworlds=camtoworlds,
                depth_map=depths_tensor,
                normal_map=normals_tensor,
                albedo_map=albedo,
                roughness_map=roughness * (1 - 0.04) + 0.04,
                metallic_map=metallic,
                render_with_bg=cfg.render_with_bg,
                splats_bg=self.splats_bg if cfg.render_with_bg else None,
                distance_to_surface=cfg.distance_to_surface * self.scene_scale,
            )
            
            diffuse_image = pbr_result["diffuse_rgb"]
            specular_image = pbr_result["specular_rgb"]
            rendered_image = pbr_result["render_rgb"]

            canvas = torch.cat([diffuse_image, specular_image, rendered_image, colors[0]], dim=1).cpu().numpy()
            save_path = f"{curr_render_dir}/surface_renderer/val_{i:04d}_surface_renderer_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # envmap = self.light_model.export_envmap(return_img=True, res=[32, 64])
            # normals_tensor = transform_normals_to_image_coord(normals_tensor, camtoworlds)
            # normals_tensor = torch.nn.functional.normalize(normals_tensor, dim=-1)
            # simple_irr = obtain_irradiance(envmap=envmap, normals=normals_tensor[0])

            irradiace = pbr_result["diffuse_light"]
            gt_irradiance = data["irradiance"][0].to(irradiace)

            min_irradiance = torch.min(irradiace)
            min_gt_irradiance = torch.min(gt_irradiance)
            range_gt_irradiance = torch.max(gt_irradiance) - min_gt_irradiance
            range_irradiance = torch.max(irradiace) - min_irradiance
            irradiace = (irradiace - min_irradiance) / range_irradiance * range_gt_irradiance + min_gt_irradiance

            # min_irradiance = torch.min(simple_irr)
            # range_irradiance = torch.max(simple_irr) - min_irradiance
            # simple_irr = (simple_irr - min_irradiance) / range_irradiance * range_gt_irradiance + min_gt_irradiance

            canvas = torch.cat([gt_irradiance, irradiace], dim=1).cpu().numpy()
            save_path = f"{curr_render_dir}/irradiance/val_{i:04d}_irradiance_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # save cubemap and envmap
            cubemap = rearrange(self.light_model.cubemap, 'n h w c -> h (n w) c')
            cubemap = hdr_to_ldr(cubemap)
            cubemap = (cubemap.cpu().numpy() * 255).astype(np.uint8)
            save_path = f"{curr_render_dir}/cubemap/val_{i:04d}_cubemap_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, cubemap)

            envmap = self.light_model.export_envmap(return_img=True)
            envmap = hdr_to_ldr(envmap)
            envmap = (envmap.cpu().numpy() * 255).astype(np.uint8)
            save_path = f"{curr_render_dir}/envmap/val_{i:04d}_envmap_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, envmap)

            if i == 0:
                # save envmap of intrinsics
                w2c = torch.linalg.inv(camtoworlds)
                w2c[:, :3, -1] = 0.
                c2w = torch.linalg.inv(w2c) 

                env_colors = render_envmap(
                    splats=self.splats,
                    splats_bg=self.splats_bg,
                    point_xyz=torch.tensor([0., 0., 0.]).to(c2w), 
                    c2w=c2w,
                    light_model=self.light_model,
                    render_with_bg=cfg.render_with_bg,
                    model='albedo',
                )
                self.light_model.update_cubemap(env_colors)

                albedo_map = self.light_model.export_envmap(return_img=True)
                albedo_map = hdr_to_ldr(albedo_map)
                albedo_map = (albedo_map.cpu().numpy() * 255).astype(np.uint8)
                save_path = f"{curr_render_dir}/overall/albedo.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, albedo_map)

                env_colors = render_envmap(
                    splats=self.splats,
                    splats_bg=self.splats_bg,
                    point_xyz=torch.tensor([0., 0., 0.]).to(c2w), 
                    c2w=c2w,
                    light_model=self.light_model,
                    render_with_bg=cfg.render_with_bg,
                    model='color',
                )
                self.light_model.update_cubemap(env_colors)

                envmap = self.light_model.export_envmap(return_img=True)
                envmap = hdr_to_ldr(envmap)
                envmap = (envmap.cpu().numpy() * 255).astype(np.uint8)
                save_path = f"{curr_render_dir}/overall/color.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, envmap)

                env_colors = render_envmap(
                    splats=self.splats,
                    splats_bg=self.splats_bg,
                    point_xyz=torch.tensor([0., 0., 0.]).to(c2w), 
                    c2w=c2w,
                    light_model=self.light_model,
                    render_with_bg=cfg.render_with_bg,
                    model='normal',
                )
                self.light_model.update_cubemap(env_colors * 0.5 + 0.5)

                normal_pano = self.light_model.export_envmap(return_img=True)
                normal_pano = tonemap(normal_pano) * 2 - 1
                normal_pano = torch.nn.functional.normalize(normal_pano, dim=-1)
                normal_pano = transform_normals_to_image_coord(normal_pano, camtoworlds)
                normal_pano = (normal_pano * 0.5 + 0.5).squeeze(0)

                normal_pano = (normal_pano.cpu().numpy() * 255).astype(np.uint8)
                save_path = f"{curr_render_dir}/overall/normal.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, normal_pano)

                env_colors = render_envmap(
                    splats=self.splats,
                    splats_bg=self.splats_bg,
                    point_xyz=torch.tensor([0., 0., 0.]).to(c2w), 
                    c2w=c2w,
                    light_model=self.light_model,
                    render_with_bg=cfg.render_with_bg,
                    model='depth',
                )
                self.light_model.update_cubemap(env_colors)

                depth_pano = self.light_model.export_envmap(return_img=True)
                depth_pano = tonemap(depth_pano)
                depth_pano = (depth_pano - depth_pano.min()) / (depth_pano.max() - depth_pano.min())
                depth_pano = (depth_pano.cpu().numpy() * 255).astype(np.uint8)
                save_path = f"{curr_render_dir}/overall/depth.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, depth_pano)

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

            rendered_image = rendered_image[None, ...].permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr_surf"].append(self.psnr(rendered_image, pixels))
            metrics["ssim_surf"].append(self.ssim(rendered_image, pixels))
            metrics["lpips_surf"].append(self.lpips(rendered_image, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()

        psnr_surf = torch.stack(metrics["psnr_surf"]).mean()
        ssim_surf = torch.stack(metrics["ssim_surf"]).mean()
        lpips_surf = torch.stack(metrics["lpips_surf"]).mean()

        print(
            f"PSNR: {psnr.data:.3f}, SSIM: {ssim.data:.4f}, LPIPS: {lpips.data:.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )

        print(
            f"==Surface Rendering==",
            f"PSNR: {psnr_surf.data:.3f}, SSIM: {ssim_surf.data:.4f}, LPIPS: {lpips_surf.data:.3f}",
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "psnr_surf": psnr_surf.item(),
            "ssim_surf": ssim_surf.item(),
            "lpips_surf": lpips_surf.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # # save stats to tensorboard
        # for k, v in stats.items():
            # self.writer.add_scalar(f"val/{k}", v, step)
        # self.writer.flush()
        wandb.log({f"val/{k}": v for k, v in stats.items()}, step)

        # save splats as point cloud (.ply file)
        self.save_splats(f"{curr_render_dir}/pointcloud/points_{step:05d}.ply")
    
    def save_splats(self, save_path):
        import trimesh
        print(f"Saving splats to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        points = self.splats["means"]

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        dirs = points[None, ...]
        colors = spherical_harmonics(0, dirs[0], colors)   # [N, 3]
        colors = torch.clamp_min(colors + 0.5, 0.0) # [N, 3]

        pc = trimesh.PointCloud(points.data.cpu().numpy(), colors=colors.data.cpu().numpy())
        pc.export(save_path, "ply")

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = self.parser.camtoworlds
        camtoworlds = generate_interpolated_path(camtoworlds, 2)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        # w2cs = torch.linalg.inv(camtoworlds)    # [N, 4, 4]
        # w2cs[:, :3, -1] *= 0.8
        # camtoworlds = torch.linalg.inv(w2cs)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _ = rasterize_splats(
                splats=self.splats,
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                render_with_bg=cfg.render_with_bg,
                splats_bg=self.splats_bg if cfg.render_with_bg else None,
            )  # [1, H, W, 4]

            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., -1:]  # [H, W, 1]
            depths = torch.where(depths > 0, 1 / depths, torch.zeros_like(depths))
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # luzhan: take intrinsics
            albedos = renders[0, ..., 3:6]
            roughness =  renders[0, ..., 6:7].repeat(1, 1, 3)
            metallicity = renders[0, ..., 7:8].repeat(1, 1, 3)

            surf_normals = transform_normals_to_image_coord(surf_normals, camtoworlds)
            surf_normals = torch.nn.functional.normalize(surf_normals, dim=-1)
            surf_normals = (surf_normals * 0.5 + 0.5).squeeze(0)

            # luzhan: write images, including colors, depths, normals, albedo, roughness, metallicity
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3), surf_normals, albedos, roughness, metallicity], dim=1
            )

            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=10)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _, _, _ = rasterize_splats(
            splats=self.splats,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
            render_with_bg=self.cfg.render_with_bg,
            splats_bg=self.splats_bg if self.cfg.render_with_bg else None,
        )  # [1, H, W, 3]

        return render_colors[0].cpu().numpy()
    
    def update_hdr_scaler(self, init_scaler):
        self.hdr_scaler.data = torch.log(init_scaler).view((1,)).to(self.device)
        print(f"Initial hdr scaler: {init_scaler}")

    def init_lighting(self, cubemap):
        cfg = self.cfg
        device = self.device

        cubemap = hdr_to_ldr(cubemap.to(device))

        max_steps = 400
        init_step = 0

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        data = next(trainloader_iter)   # get first image, c2w
        camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]

        # Training loop.
        for step in tqdm.tqdm(range(init_step, max_steps), desc="Init Lit"):
            env_colors = render_envmap(
                splats=self.splats_bg,
                splats_bg=None,
                point_xyz=torch.tensor([0., 0., 0.]).to(camtoworlds), 
                c2w=camtoworlds,
                light_model=self.light_model,
                render_with_bg=False,
                only_bg=True,
            )
        
            # loss
            l1loss = F.l1_loss(env_colors, cubemap)
            ssimloss = 1.0 - self.ssim(
                cubemap.permute(0, 3, 1, 2), env_colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            loss.backward()

            # luzhan: optimize bg splats
            for optimizer in self.optimizers_bg.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if step % 10 == 0:
                wandb.log(
                    {
                        "env/loss": loss.data,
                        "env/l1loss": l1loss.data,
                        "env/ssimloss": ssimloss.data,
                    }
                )

            if step % 100 == 0:
                self.light_model.update_cubemap(env_colors)

                envmap = self.light_model.export_envmap(return_img=True)
                envmap = hdr_to_ldr(envmap)

                envmap = (envmap.data.cpu().numpy() * 255).astype(np.uint8)
                wandb.log({"env/envmap": wandb.Image(envmap)})
                save_path = os.path.join(self.render_dir, "init_envmap", f"env_{step:04d}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, envmap)

                self.light_model.update_cubemap(cubemap)
                envmap = self.light_model.export_envmap(return_img=True)
                envmap = hdr_to_ldr(envmap)

                envmap = (envmap.data.cpu().numpy() * 255).astype(np.uint8)
                wandb.log({"env/gt_envmap": wandb.Image(envmap)})
                save_path = os.path.join(self.render_dir, "init_envmap", f"gt_{step:04d}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.imwrite(save_path, envmap)
