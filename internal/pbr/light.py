import cv2
import torch
import numpy as np
import nvdiffrast.torch as dr
import torch.nn.functional as F

from typing import List, Optional

from load_image import tonemap, inverse_tonemap
from .renderutils import diffuse_cubemap, specular_cubemap
from .cubemap import create_cubemap_c2w
from .shade import linear_to_srgb, srgb_to_linear


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # directions: +x, -x, +y, -y, +z, -z
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap: torch.Tensor) -> torch.Tensor:
        # avg_pool_nhwc
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> torch.Tensor:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing="ij",
            )
            v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out


class CubemapLight():
    # for nvdiffrec
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, height=256):
        # prepare Ks representing the image with 90 degree field of view
        Ks = torch.zeros((1, 3, 3))
        Ks[0, 0, -1] = height / 2
        Ks[0, 1, -1] = height / 2
        Ks[0, -1, -1] = 1
        Ks[0, 0, 0] = height / 2
        Ks[0, 1, 1] = height / 2

        self.height = height
        self.Ks = Ks.repeat(6, 1, 1)    # (6, 3, 3)
    
    def get_Ks_c2w(self, point_xyz):
        return self.Ks.to(point_xyz), create_cubemap_c2w(pos=point_xyz).to(point_xyz)

    def update_cubemap(self, cubemap):
        # cubemap is ldr -> linear -> inv tonemap
        self.cubemap = inverse_tonemap(srgb_to_linear(cubemap)).contiguous()

    # processing
    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )

    def build_mips(self, cutoff: float = 0.99) -> None:
        self.specular = [self.cubemap]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: List[int] = [512, 1024],
        return_img: bool = False,
    ) -> Optional[torch.Tensor]:
        # cubemap_to_latlong
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )

        sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
        sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

        reflvec = torch.stack(
            (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
        )  # [H, W, 3]

        color = dr.texture(
            self.cubemap[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[0]  # [H, W, 3]

        if return_img:
            return color    # [H, W, 3]
        else:
            color = linear_to_srgb(tonemap(color))
            cv2.imwrite(filename, color.clamp(min=0).cpu().numpy()[..., ::-1])


if __name__ == "__main__":
    import torchvision
    res = 256
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        v = (v+1) / 2
        torchvision.utils.save_image(v.permute(2, 0, 1), f"envmap_{s}.png")