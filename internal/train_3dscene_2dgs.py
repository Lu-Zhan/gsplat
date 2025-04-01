import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from einops import rearrange

import imageio
import nerfview
import numpy as np
from pycolmap import rotation
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
# from datasets.colmap import Dataset, Parser
# luzhan: using colmap_with_intrinsics as dataloader
# from datasets.colmap_with_intrinsics import Dataset, Parser
from datasets.blender_with_intrinsics import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy

from pbr.light import CubemapLight
from pbr.surface_rendering import SurfaceRenderer, hdr_to_ldr


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # luzhan: Initialization strategy using random
    init_type: str = "random"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # luzhan: add Initialization for env map
    init_num_pts_bg: int = 1_000
    radius_of_sphere_bg: float = 100.0
    render_with_bg: int = 1

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 5e-1

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # luzhan: intrinsics loss and direct normal loss
    intrinsics_loss: bool = False
    intrinsics_lambda: float = 5e-1
    direct_normal_loss: bool = False
    direct_normal_lambda: float = 1e-1

    # luzhan: add surface rendering as a regularizer
    surface_rendering_loss: bool = False
    surface_rendering_lambda: float = 5e-1
    irradiance_loss: bool = False
    irradiance_lambda: float = 5e-1
    surface_rendering_start_iter: int = 100

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
    
    # luzhan: add intrinsics for albedo, roughness, metallic, irradiance
    intrinsics = torch.logit(torch.rand((N, 5)))
    params.append(("intrinsics", torch.nn.Parameter(intrinsics), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def create_backgrpound_splats_with_optimizers(
    # parser: Parser,
    init_num_pts: int = 100_000,
    radius_of_sphere: float = 100.0,
    init_opacity: float = 1,
    init_scale: float = 1.0,
    sparse_grad: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # Initialize points using Fibonacci lattice on a sphere with a radius of 10 meters
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    indices = torch.arange(0, init_num_pts, dtype=torch.float) + 0.5
    theta = 2 * math.pi * indices / phi
    z = 1 - (2 * indices / init_num_pts)
    radius = torch.sqrt(1 - z * z)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack((x, y, z), dim=-1) * radius_of_sphere  # Scale to 10m sphere

    # init rgbs
    rgbs = torch.ones((init_num_pts, 3)) * 0.01

    # init geometry
    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    quats = torch.cat([torch.ones((N, 1)), torch.zeros((N, 3))], dim=-1) # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("means", points, 0),
        ("opacities", opacities, 0),
        # ("means", torch.nn.Parameter(points), 0),
        # ("opacities", torch.nn.Parameter(opacities), 0),
    ]

    colors = torch.logit(rgbs)  # [N, 3]
    params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params if name in ["scales", "quats"]
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
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
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            load_intrinsics=cfg.intrinsics_loss,    # luzhan: loading intrinsics
        )
        self.valset = Dataset(
            self.parser, 
            split="val",
            load_depths=cfg.depth_loss,
            load_intrinsics=cfg.intrinsics_loss,    # luzhan: loading intrinsics
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
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        # Background Model
        self.splats_bg, self.optimizers_bg = create_backgrpound_splats_with_optimizers(
            init_num_pts=cfg.init_num_pts_bg,
            radius_of_sphere=cfg.radius_of_sphere_bg,
            init_scale=cfg.init_scale,
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

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

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
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        render_with_bg: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        # luzhan: concat intrinsics into colors
        ## step1: transfer sh to rgb 
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
        sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree)
        colors = spherical_harmonics(sh_degree, dirs[0], colors)   # [N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0) # [N, 3]
        intrinsics = torch.sigmoid(self.splats["intrinsics"]) # [N, 5]
        colors = torch.cat([colors, intrinsics], dim=-1) # [N, 3+5]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        # luzhan: add bg splats
        if render_with_bg:
            means_bg = self.splats_bg["means"]  # [K, 3]
            quats_bg = self.splats_bg["quats"]  # [K, 4]
            scales_bg = torch.exp(self.splats_bg["scales"])  # [K, 3]
            opacities_bg = torch.sigmoid(self.splats_bg["opacities"])  # [K,]
            colors_bg = torch.sigmoid(self.splats_bg["colors"])  # [K, 3]
            colors_bg = torch.cat([colors_bg, torch.zeros([colors_bg.shape[0], 5]).to(colors_bg)], dim=-1)  # [K, 3+5]

            means = torch.cat([means, means_bg], dim=0)
            quats = torch.cat([quats, quats_bg], dim=0)
            scales = torch.cat([scales, scales_bg], dim=0)
            opacities = torch.cat([opacities, opacities_bg], dim=0)
            colors = torch.cat([colors, colors_bg], dim=0)

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = rasterization_2dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            renders, info = rasterization_2dgs_inria_wrapper(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=self.cfg.packed,
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )
    
    # luzhan: render env at given point
    def render_envmap(self, point_xyz):
        Ks, c2w_cubemap = self.light_model.get_Ks_c2w(point_xyz)

        colors = self.rasterize_splats(
            camtoworlds=c2w_cubemap,
            Ks=Ks,
            width=self.light_model.height,
            height=self.light_model.height,
            near_plane=0.01,
            far_plane=self.cfg.far_plane,
            image_ids=None,
            render_mode="RGB",
            distloss=False,
            render_with_bg=self.cfg.render_with_bg, # whether to render with bg splats
        )[0][..., :3]   # [6, H, W, 3]

        self.light_model.update_cubemap(colors)
    
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
            num_workers=27,
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
                # points = data["points"].to(device)  # [1, M, 2]
                # depths_gt = data["depths"].to(device)  # [1, M]

                gt_depths = data["depths"].to(device)   # [1, H, W, 1]
            
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
            ) = self.rasterize_splats(
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
                render_with_bg=render_with_bg, # whether to render with bg splats
            )

            # luzhan: unpack intrinsics and depths from renders
            if renders.shape[-1] == 4:
                colors, intrinsics, depths = renders[..., 0:3], None, renders[..., 3:4]
            elif renders.shape[-1] == 9:
                colors, intrinsics, depths = renders[..., 0:3], renders[..., 3:-1], renders[..., -1:]
                # luzhan: formulate roughness, roughness = roughness * (rmax - rmin) + rmin from GS-IR
                # rmax, rmin = 1.0, 0.04
                # intrinsics[..., 3:4] = intrinsics[..., 3:4] * (rmax - rmin) + rmin
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
            if cfg.depth_loss:
                # # query depths from depth map
                # points = torch.stack(
                #     [
                #         points[:, :, 0] / (width - 1) * 2 - 1,
                #         points[:, :, 1] / (height - 1) * 2 - 1,
                #     ],
                #     dim=-1,
                # )  # normalize to [-1, 1]
                # grid = points.unsqueeze(2)  # [1, M, 1, 2]
                # depths = F.grid_sample(
                #     depths.permute(0, 3, 1, 2), grid, align_corners=True
                # )  # [1, 1, M, 1]
                # depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # # calculate loss in disparity space
                # disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                # disp_gt = 1.0 / depths_gt  # [1, M]
                # depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                # loss += depthloss * cfg.depth_lambda

                # luzhan: new depth loss
                gt_depths = (gt_depths - gt_depths.min()) / (gt_depths.max() - gt_depths.min())
                depths = torch.clamp(depths, min=0.)
                depths = (depths - depths.min()) / (depths.max() - depths.min())

                depthloss = F.l1_loss(depths, gt_depths) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            # luzhan: add more losses, including intrinsics loss, direct normal loss
            if cfg.intrinsics_loss:
                intrinsics_loss = F.l1_loss(intrinsics, gt_intrinsics)
                loss += intrinsics_loss * cfg.intrinsics_lambda
            
            if cfg.direct_normal_loss:
                direct_normal_loss = (1 - (normals * gt_normals).sum(dim=0).mean()) / 2       
                loss += direct_normal_loss * cfg.direct_normal_lambda

            if (cfg.irradiance_loss or cfg.surface_rendering_loss) and step > cfg.surface_rendering_start_iter:
                _, h, w, _ = depths.shape
                depths_center_patch = depths[0, h // 4:-h // 4, w // 4:-w // 4]
                distance = max(depths_center_patch.min(), depths_center_patch[h // 4, w // 4]) + 0.1
                point_xyz = camtoworlds[0] @ torch.tensor([0, 0, distance, 1], device=device).t()
                self.render_envmap(point_xyz[:3])

                albedo = intrinsics[0, ..., :3]
                roughness = intrinsics[0, ..., 3:4]
                metallic = intrinsics[0, ..., 4:5]

                if self.surface_renderer.camera_dirs is None:
                    self.surface_renderer.update_params(ref_Ks=Ks, hw=[h, w])

                pbr_result = self.surface_renderer.render(
                    c2w=camtoworlds[0],
                    normals=normals[0],   # 
                    albedo=albedo,
                    roughness=roughness,
                    metallic=metallic,
                    light_model=self.light_model,
                )

                irradiance = pbr_result["diffuse_light"]
                colors_surf = pbr_result["render_rgb"]

                if cfg.irradiance_loss:
                    irradiance_loss = F.l1_loss(irradiance, gt_irradiance[0])
                    loss += irradiance_loss * cfg.irradiance_lambda
                
                if cfg.surface_rendering_loss:
                    surface_rendering_loss = F.l1_loss(colors_surf, pixels)
                    loss += surface_rendering_loss * cfg.surface_rendering_lambda

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                
                # luzhan: add more losses, including intrinsics loss, direct normal loss
                if cfg.intrinsics_loss:
                    self.writer.add_scalar("train/intrinsics_loss", intrinsics_loss.item(), step)
                if cfg.direct_normal_loss:
                    self.writer.add_scalar("train/direct_normal_loss", direct_normal_loss.item(), step)
                
                if step > cfg.surface_rendering_start_iter:
                    if cfg.irradiance_loss:
                        self.writer.add_scalar("train/irradiance_loss", irradiance_loss.item(), step)
                    if cfg.surface_rendering_loss:
                        self.writer.add_scalar("train/surface_rendering_loss", surface_rendering_loss.item(), step)    

                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            
            # luzhan: use only the front splats for densification
            if self.cfg.render_with_bg:
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
            # luzhan: optimize bg splats
            for optimizer in self.optimizers_bg.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}

        # luzhan: update render_dir
        self.render_dir = f"{self.render_dir}/step_{step:05d}"
        os.makedirs(self.render_dir, exist_ok=True)
        
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
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)

            # luzhan: take intrinsics
            intrinsics = colors[..., 3:-1]  # (1, H, W, 5)

            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            save_path = f"{self.render_dir}/images/val_{i:04d}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # write median depths
            render_median = (render_median - render_median.min()) / (render_median.max() - render_median.min())
            render_median = render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()

            gt_depths = data["depths"]
            gt_depths = (gt_depths - gt_depths.min()) / (gt_depths.max() - gt_depths.min())
            gt_depths = gt_depths.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()

            canvas = np.concatenate([gt_depths, render_median], axis=1)

            save_path = f"{self.render_dir}/depths/val_{i:04d}_median_depth_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # write normals
            normals_tensor = normals.clone()
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            save_path = f"{self.render_dir}/normals/val_{i:04d}_normal_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, normals_output)

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            save_path = f"{self.render_dir}/normals_from_depth/val_{i:04d}_normals_from_depth_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, normals_from_depth_output)

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
            save_path = f"{self.render_dir}/distortions/val_{i:04d}_distortions_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, render_dist)
  
            # write alphas
            alphas = alphas.repeat(1, 1, 1, 3).squeeze(0).detach().cpu().numpy()
            alphas = (alphas - np.min(alphas)) / (np.max(alphas) - np.min(alphas))
            alphas = (alphas * 255).astype(np.uint8)
            save_path = f"{self.render_dir}/alphas/val_{i:04d}_alphas_{step}.png"
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
            save_path = f"{self.render_dir}/intrinsics/val_{i:04d}_intrinsics_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            # luzhan: render and write env map at current camera
            # point_xyz = torch.linalg.inv(camtoworlds)[0, :3, 3]
            point_xyz = torch.zeros(3).to(camtoworlds)
            self.render_envmap(point_xyz=point_xyz)
            cubemap = rearrange(self.light_model.cubemap, 'n h w c -> h (n w) c')
            cubemap = hdr_to_ldr(cubemap)
            cubemap = (cubemap.cpu().numpy() * 255).astype(np.uint8)
            save_path = f"{self.render_dir}/cubemap/val_{i:04d}_cubemap_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, cubemap)

            envmap = self.light_model.export_envmap(return_img=True)
            envmap = hdr_to_ldr(envmap)
            envmap = (envmap.cpu().numpy() * 255).astype(np.uint8)
            save_path = f"{self.render_dir}/envmap/val_{i:04d}_envmap_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, envmap)

            # luzhan: surface renderer
            if self.surface_renderer.camera_dirs is None:
                self.surface_renderer.update_params(Ks, hw=(height, width))

            pbr_result = self.surface_renderer.render(
                c2w=camtoworlds[0],
                normals=normals_tensor[0],
                albedo=albedo[0],
                roughness=roughness[0, ..., :1],
                metallic=metallic[0, ..., :1],
                light_model=self.light_model,
            )
            
            diffuse_image = pbr_result["diffuse_rgb"]
            specular_image = pbr_result["specular_rgb"]
            rendered_image = pbr_result["render_rgb"]

            canvas = torch.cat([diffuse_image, specular_image, rendered_image, colors[0]], dim=1).cpu().numpy()
            save_path = f"{self.render_dir}/surface_renderer/val_{i:04d}_surface_renderer_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            irradiace = pbr_result["diffuse_light"]
            gt_irradiance = data["irradiance"].to(irradiace)

            canvas = torch.cat([gt_irradiance[0], irradiace], dim=1).cpu().numpy()
            save_path = f"{self.render_dir}/irradiance/val_{i:04d}_irradiance_{step}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, (canvas * 255).astype(np.uint8))

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # luzhan: take intrinsics
            albedos = renders[0, ..., 3:6]
            roughness =  renders[0, ..., 6:7].repeat(1, 1, 3)
            metallicity = renders[0, ..., 7:8].repeat(1, 1, 3)

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            # canvas = torch.cat(
            #     [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            # )

            # luzhan: write images, including colors, depths, normals, albedo, roughness, metallicity
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3), surf_normals, albedos, roughness, metallicity], dim=1
            )

            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
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

        render_colors, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)
    # luzhan: convert render_with_bg to a boolean
    cfg.render_with_bg = cfg.render_with_bg == 1
    print(f"render_with_bg: {cfg.render_with_bg}")

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
