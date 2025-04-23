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
        filter_3D=None,
        only_bg=False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    means = splats["means"]  # [N, 3]
    # quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
    # rasterization does normalization internally
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    image_ids = kwargs.pop("image_ids", None)

    if only_bg:
        colors = torch.sigmoid(splats["colors"])  # [K, 3]
    else:
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
    if render_with_bg and not only_bg:
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
    
    if filter_3D is not None:
        scales, opacities = get_scaling_opacity_with_3D_filter(scales, opacities, filter_3D)

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
def render_envmap(splats, point_xyz, c2w, light_model, render_with_bg, splats_bg=None, model='color', only_bg=False):
    Ks, c2w_cubemaps = light_model.get_Ks_c2w(point_xyz)

    # rotate c2w to align with camera forward direction
    ref_w2c = torch.linalg.inv(c2w[0])

    for i, c2w_cubemap in enumerate(c2w_cubemaps):
        w2c = torch.linalg.inv(c2w_cubemap.clone())
        w2c = w2c @ ref_w2c

        c2w_cubemaps[i] = torch.linalg.inv(w2c)

    # compute filter3D
    means = splats["means"]  # [N, 3]
    if render_with_bg:
        means = torch.cat([means, splats_bg["means"]], dim=0)
    filter_3D = compute_3D_filter(means, Ks, c2w_cubemaps)

    if model == 'color':
        start_idx, end_idx = 0, 3
        mode_idx = 0
    elif model == 'albedo':
        start_idx, end_idx = 3, 6
        mode_idx = 0
    elif model == 'normal':
        start_idx, end_idx = 0, 3
        mode_idx = 2
    elif model == 'depth':
        start_idx, end_idx = -2, -1
        mode_idx = 0

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
            filter_3D=filter_3D,
            only_bg=only_bg,
        )[mode_idx][..., start_idx:end_idx]   # [6, H, W, 3]

        # unit test
        # color = unit_test_color(i, color)
        if model == 'depth':
            color = color.repeat((1, 1, 1, 3))
        colors.append(color)
    
    colors = torch.cat(colors, dim=0)
    colors = torch.clamp(colors, 0.0, 1.0)

    return colors   # [6, H, W, 3]
    

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
        distance_to_surface=0.1,
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

    normals_tensor = transform_normals_to_image_coord(normals_tensor, camtoworlds)
    normals_tensor = torch.nn.functional.normalize(normals_tensor, dim=-1)
    # normals_tensor = torch.zeros_like(normals_tensor)
    # normals_tensor[..., -1] = -1.

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


@torch.no_grad()
def compute_3D_filter(xyz, Ks, cam2worlds):
    # print("Computing 3D filter")
    xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    
    # we should use the focal length of the highest resolution camera
    focal_length = 0.

    for c2w, K in zip(cam2worlds, Ks):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        h = cy * 2
        w = cx * 2

        w2c = torch.linalg.inv(c2w)
        xyz_cam = xyz @ w2c.T
        xyz_cam = xyz_cam[..., :3]
        # xyz_to_cam = torch.norm(xyz_cam, dim=1)
        
        # project to screen space
        valid_depth = xyz_cam[:, 2] > 0.2
        
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        z = torch.clamp(z, min=0.001)
        
        x = x / z * fx + w / 2.0
        y = y / z * fy + h / 2.0
     
        # use similar tangent space filtering as in the paper
        in_screen = torch.logical_and(
            torch.logical_and(
                x >= -0.15 * w, x <= w * 1.15
            ), 
            torch.logical_and(
                y >= -0.15 * h, y <= 1.15 * h
            )
        )
        
        valid = torch.logical_and(valid_depth, in_screen)
        
        # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
        distance[valid] = torch.min(distance[valid], z[valid])
        valid_points = torch.logical_or(valid_points, valid)
        if focal_length < fx:
            focal_length = fx
    
    if valid_points.sum() > 0:
        distance[~valid_points] = distance[valid_points].max()
    filter_3D = distance / focal_length * (0.5 ** 0.5)

    return filter_3D[:, None]


def get_scaling_opacity_with_3D_filter(scales, opacity, filter_3D):
    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)

    scales_after_square = scales_square + torch.square(filter_3D) 
    det2 = scales_after_square.prod(dim=1) 
    coef = torch.sqrt(det1 / det2)
    aa_opacity = opacity * coef

    aa_scales = torch.sqrt(scales_after_square)

    return aa_scales, aa_opacity


def unit_test_color(i, color):
    color = torch.zeros_like(color)
    # if i == 0:
    #     color[..., 0] = 1
    # if i == 2:
    #     color[..., 1] = 1
    if i == 4:
        color[..., 2] = 1
    # if i == 1:
    #     color[..., 0] = 1
    #     color[..., 1] = 1
    # if i == 3:
    #     color[..., 1] = 1
    #     color[..., 2] = 1
    # if i == 5:
    #     color[..., 0] = 1
    #     color[..., 2] = 1
    
    return color