import torch
import torch.nn.functional as F
from skybox_utils import obtain_dirs_for_skybox

from pbr.shade import pbr_shading


def get_canonical_rays(Ks):
    cen_x = Ks[0, -1]
    cen_y = Ks[1, -1]
    focal_x = Ks[0, 0]
    focal_y = Ks[1, 1]

    h = cen_x * 2
    w = cen_y * 2

    x, y = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing="xy",
    ).to(Ks)
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    
    camera_dirs = F.normalize(camera_dirs, dim=-1)

    return camera_dirs.reshape(h, w, 3), h, w


class SurfaceRenderer:
    def __init__(self, ref_Ks, height=256):
        # self.env_dirs = obtain_dirs_for_skybox(height=height)
        self.tonemap = True
        self.gamma = 2.2
        self.camera_dirs, self.h, self.w = get_canonical_rays(ref_Ks)
    
    def load_brdf_lut(self, path):
        self.brdf_lut = torch.from_numpy(np.load(path)).float()
    
    def get_view_dirs(self, c2w):
        # (h, w, 1, 3) * (1, 1, 3, 3) -> (h, w, 3, (3)) -> (h, w, 3)
        return -(self.camera_dirs[..., None, :] * c2w[None, None, :3, :3]).sum(dim=-1)

    def render(
        self, 
        c2w,
        normal_map,
        albedo_map,
        roughness_map,
        metallic_map,
        lighting,
    ):
        occlusion = torch.ones_like(roughness_map)  # [H, W, 1]
        irradiance = torch.zeros_like(roughness_map) # [H, W, 1]
        normal_mask = torch.ones_like(roughness_map)

        view_dirs = self.get_view_dirs(c2w=c2w)
        lighting.build_mips()

        pbr_result = pbr_shading(
            light=lighting,
            normals=normal_map.permute(1, 2, 0).detach(),  # [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
            metallic=metallic_map.permute(1, 2, 0),  # [H, W, 1]
            tone=self.tonemap,
            gamma=self.gamma,
            occlusion=occlusion,
            irradiance=irradiance,
            brdf_lut=self.brdf_lut,
        )

        return pbr_result