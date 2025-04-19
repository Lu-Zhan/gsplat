import torch

from torch import Tensor
from torch.nn import functional as F
from typing import Dict, Optional, Tuple

from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.rendering import rasterization_2dgs

from utils.geo_utils import transform_normals_to_image_coord, obtain_surface_position


def rasterize_splats(
        splats,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        render_with_bg=False,
        splats_bg=None,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    means = splats["means"]  # [N, 3]
    # quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
    # rasterization does normalization internally
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    image_ids = kwargs.pop("image_ids", None)
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
    
    # luzhan: concat intrinsics into colors
    ## step1: transfer sh to rgb 
    dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
    # sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree)
    colors = spherical_harmonics(sh_degree, dirs[0], colors)   # [N, 3]
    # make it apple-to-apple with Inria's CUDA Backend.
    colors = torch.clamp_min(colors + 0.5, 0.0) # [N, 3]
    intrinsics = torch.sigmoid(splats["intrinsics"]) # [N, 5]
    colors = torch.cat([colors, intrinsics], dim=-1) # [N, 3+5]
    backgrounds = torch.zeros_like(colors[:1])

    # assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

    # luzhan: add bg splats
    if render_with_bg:
        assert splats_bg is not None, "Please provide splats_bg when render_with_bg is True."

        means_bg = splats_bg["means"]  # [K, 3]
        quats_bg = splats_bg["quats"]  # [K, 4]
        scales_bg = torch.exp(splats_bg["scales"])  # [K, 3]
        opacities_bg = torch.sigmoid(splats_bg["opacities"])  # [K,]
        colors_bg = torch.sigmoid(splats_bg["colors"])  # [K, 3]
        colors_bg = torch.cat([colors_bg, torch.zeros([colors_bg.shape[0], 5]).to(colors_bg)], dim=-1)  # [K, 3+5]

        means = torch.cat([means, means_bg], dim=0)
        quats = torch.cat([quats, quats_bg], dim=0)
        scales = torch.cat([scales, scales_bg], dim=0)
        opacities = torch.cat([opacities, opacities_bg], dim=0)
        colors = torch.cat([colors, colors_bg], dim=0)

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
        packed=packed,
        absgrad=absgrad,
        sparse_grad=sparse_grad,
        backgrounds=backgrounds,
        **kwargs,
    )

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
def render_envmap(splats, point_xyz, c2w, light_model, render_with_bg, splats_bg=None):
    Ks, c2w_cubemaps = light_model.get_Ks_c2w(point_xyz)

    # rotate c2w to align with camera forward direction
    ref_w2c = torch.linalg.inv(c2w[0])

    for i, c2w_cubemap in enumerate(c2w_cubemaps):
        w2c = torch.linalg.inv(c2w_cubemap.clone())
        w2c = w2c @ ref_w2c

        c2w_cubemaps[i] = torch.linalg.inv(w2c)

    colors = []
    for i, c2w_cubemap in enumerate(c2w_cubemaps):
        color = rasterize_splats(
            splats=splats,
            splats_bg=splats_bg,
            sh_degree=3,
            camtoworlds=c2w_cubemap[None, ...],
            Ks=Ks[:1],
            width=light_model.height,
            height=light_model.height,
            near_plane=0.02,
            far_plane=200,
            image_ids=None,
            render_mode="RGB",
            distloss=False,
            render_with_bg=render_with_bg, # whether to render with bg splats
        )[0][..., :3]   # [6, H, W, 3]
        
        colors.append(color)
    
    colors = torch.cat(colors, dim=0)
    colors = torch.clamp(colors, 0.0, 1.0)

    return colors
    

def render_reflection(
        splats,
        surface_renderer,
        light_model,
        hdr_scaler,
        Ks,
        hw,
        camtoworlds,
        depths_tensor,
        normals_tensor, # (1, h, w, 3)
        albedo, # (1, h, w, 3)
        roughness, # (1, h, w, 1)
        metallic, # (1, h, w, 1)
        render_with_bg,
        splats_bg=None,
        distance_to_surface=0,
    ):
    # luzhan: render and write env map at current camera
    point_xyz = obtain_surface_position(
        depth_map=depths_tensor,
        distance_to_surface=distance_to_surface,
    )

    env_colors = render_envmap(
        splats=splats,
        splats_bg=splats_bg,
        point_xyz=point_xyz, 
        c2w=camtoworlds,
        light_model=light_model,
        render_with_bg=render_with_bg,
    )
    light_model.update_cubemap(env_colors, hdr_scaler=torch.exp(hdr_scaler))

    # luzhan: surface renderer
    if surface_renderer.camera_dirs is None:
        surface_renderer.update_params(Ks, hw=hw)

    # normals_tensor = torch.nn.functional.normalize(data["normals"], dim=-1).to(normals_tensor)
    normals_tensor = transform_normals_to_image_coord(normals_tensor, camtoworlds)
    normals_tensor = torch.nn.functional.normalize(normals_tensor, dim=-1)
    normals_tensor[..., 0] *= -1 # flip x axis: opengl -> cubemap 

    pbr_result = surface_renderer.render(
        c2w=camtoworlds[0],
        normals=normals_tensor[0],
        albedo=albedo[0],
        roughness=roughness[0, ..., :1],
        metallic=metallic[0, ..., :1],
        light_model=light_model,
    )

    return pbr_result