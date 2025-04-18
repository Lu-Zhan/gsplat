import torch
import torch.nn.functional as F

from .shade import pbr_shading, get_brdf_lut
from load_image import tonemap
from .shade import linear_to_srgb


def hdr_to_ldr(hdr):
    return linear_to_srgb(tonemap(hdr))


def get_canonical_rays(Ks, hw):
    cen_x = Ks[0, -1]
    cen_y = Ks[1, -1]
    focal_x = Ks[0, 0]
    focal_y = Ks[1, 1]

    h, w = hw

    x, y = torch.meshgrid(
        torch.arange(w).to(Ks),
        torch.arange(h).to(Ks),
        indexing="xy",
    )
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

    return camera_dirs.reshape((h, w, 3))
    # return camera_dirs


class SurfaceRenderer:
    def __init__(self):
        self.brdf_lut = get_brdf_lut()
        self.tonemap = True
        self.gamma = 2.2
        self.camera_dirs = None

    def update_params(self, ref_Ks, hw):
        self.camera_dirs = get_canonical_rays(ref_Ks[0], hw)
        self.h, self.w = hw
    
    # def get_view_dirs(self, c2w):
    #     view_dirs = -(
    #         (F.normalize(self.camera_dirs[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
    #         .sum(dim=-1)
    #         .reshape(self.h, self.w, 3)
    #     )  # [H, W, 3]

    #     return view_dirs

    def render(
        self, 
        c2w,
        normals,
        albedo,
        roughness,
        metallic,
        light_model,
    ):  
        normal_mask = torch.ones_like(roughness).bool()

        # view_dirs = self.get_view_dirs(c2w=c2w)
        view_dirs = self.camera_dirs
        light_model.build_mips()

        pbr_result = pbr_shading(
            light=light_model,
            normals=normals,  # should detach [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask,  # [H, W, 1]
            albedo=albedo,  # [H, W, 3]
            roughness=roughness,  # [H, W, 1]
            metallic=metallic,  # [H, W, 1]
            tone=self.tonemap,
            gamma=self.gamma,
            brdf_lut=self.brdf_lut.to(normals),
        )

        return pbr_result