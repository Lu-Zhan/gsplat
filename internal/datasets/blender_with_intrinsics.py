import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange

from .load_image import read_exr
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


class Parser:
    """Blender parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_cubemap: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        with open(os.path.join(data_dir, "transforms.json"), "r") as f:
            frames = json.load(f)['frames']

        image_names = []
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for idx, frame in enumerate(frames):
            image_path = frame['file_path']
            image_names.append(image_path)
            w2c = np.array(frame['transform_matrix'])
            # w2c = np.concatenate([w2c, bottom], axis=0)

            params = np.empty(0, dtype=np.float32)
            # camtype = "perspective"

            height, width = int(frame['h']), int(frame['w'])
            cx, cy = int(frame['cx']), int(frame['cy'])
            fx, fy = float(frame['fl_x']), float(frame['fl_y'])

            w2c_mats.append(w2c)
            camera_ids.append(idx)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[idx] = K
            params_dict[idx] = params
            imsize_dict[idx] = (width // factor, height // factor)
            mask_dict[idx] = None

        print(f"[Parser] {len(image_names)} images")

        w2c_mats = np.stack(w2c_mats, axis=0)
        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])

        image_dir = data_dir
        image_paths = [os.path.join(image_dir, f) for f in image_names]

        # load ply
        # pc = trimesh.load(os.path.join(data_dir, 'points.ply'))
        # points = np.array(pc.vertices)
        # points_rgb = np.array(pc.colors)[:, :3]

        splats_path = os.path.join(data_dir, 'init_splats.pth')
        self.init_splats = torch.load(splats_path)
        points = self.init_splats['means'].numpy()
        
        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)

        # self.points = points  # np.ndarray, (num_points, 3)
        # self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.init_splats['means'] = torch.from_numpy(points).float()

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # load cubemap
        if load_cubemap:
            cubemap_path = os.path.join(data_dir, 'light_info/cubemap.exr')
            cubemap = read_exr(cubemap_path)   
            self.cubemap = rearrange(cubemap, 'h (n w) c -> n h w c', n=6)
            self.cubemap = torch.from_numpy(self.cubemap).float()
        
        # load splats
        # self.splats_path = os.path.join(data_dir, 'init_splats.pth')



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
        value=-1.0,
    )  # [H * W, 3]
    
    camera_dirs = F.normalize(camera_dirs, dim=-1)

    return camera_dirs.reshape((h, w, 3))

class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_intrinsics: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        # if split == "train":
        #     # self.indices = indices[indices % self.parser.test_every != 0]
        # else:
        #     self.indices = indices[indices % self.parser.test_every == 0]
        self.indices = indices

        # luzhan: add intrinsics
        self.load_intrinsics = load_intrinsics
        print(f"Load intrinsics: {self.load_intrinsics}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        # params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        # if len(params) > 0:
        #     # Images are distorted. Undistort them.
        #     mapx, mapy = (
        #         self.parser.mapx_dict[camera_id],
        #         self.parser.mapy_dict[camera_id],
        #     )
        #     image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        #     x, y, w, h = self.parser.roi_undist_dict[camera_id]
        #     image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        
        image_dir_name = 'images'
        # luzhan: add intrinsics
        if self.load_intrinsics:
            # image = imageio.imread(self.parser.image_paths[index])[..., :3]
            albedo_path = self.parser.image_paths[index].replace(image_dir_name, 'intrinsics/albedo_maps').replace('.png', '.exr')
            normal_path = self.parser.image_paths[index].replace(image_dir_name, 'intrinsics/normal_maps').replace('.png', '.exr')
            roughness_path = self.parser.image_paths[index].replace(image_dir_name, 'intrinsics/roughness_maps').replace('.png', '.exr')
            metallic_path = self.parser.image_paths[index].replace(image_dir_name, 'intrinsics/metallic_maps').replace('.png', '.exr')
            irrdiance_path = self.parser.image_paths[index].replace(image_dir_name, 'intrinsics/irradiance_maps').replace('.png', '.exr')

            albedo = read_exr(albedo_path)
            normals = read_exr(normal_path)
            roughness = read_exr(roughness_path)[..., :1]
            metallic = read_exr(metallic_path)[..., :1]
            irradiance = read_exr(irrdiance_path)

            irradiance = torch.from_numpy(irradiance).float()
            normals = torch.from_numpy(normals).float() * 2.0 - 1.0 # [0, 1] -> [-1, 1]
            intrinsics = torch.cat(
                [torch.from_numpy(albedo), torch.from_numpy(roughness), torch.from_numpy(metallic)], dim=-1
            ).float()
            
            # resize to the same shape as the image
            h, w = image.shape[:2]
            # print(intrinsics.permute(2, 0, 1)[None, ...].shape)
            intrinsics = torch.nn.functional.interpolate(
                intrinsics.permute(2, 0, 1)[None, ...], size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)
            irradiance = torch.nn.functional.interpolate(
                irradiance.permute(2, 0, 1)[None, ...], size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)
            normals = torch.nn.functional.interpolate(
                normals.permute(2, 0, 1)[None, ...], size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0)

            # normalize normal vectors
            normals = torch.nn.functional.normalize(normals, dim=-1)

            data['irradiance'] = irradiance
            data['normals'] = normals
            data['intrinsics'] = intrinsics   

        if self.load_depths:
            # # projected points to image plane to get depths
            # worldtocams = np.linalg.inv(camtoworlds)
            # image_name = self.parser.image_names[index]
            # point_indices = self.parser.point_indices[image_name]
            # points_world = self.parser.points[point_indices]
            # points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            # points_proj = (K @ points_cam.T).T
            # points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            # depths = points_cam[:, 2]  # (M,)
            # # filter out points outside the image
            # selector = (
            #     (points[:, 0] >= 0)
            #     & (points[:, 0] < image.shape[1])
            #     & (points[:, 1] >= 0)
            #     & (points[:, 1] < image.shape[0])
            #     & (depths > 0)
            # )
            # points = points[selector]
            # depths = depths[selector]
            # data["points"] = torch.from_numpy(points).float()
            # data["depths"] = torch.from_numpy(depths).float()

            # load depth map
            depths_path = self.parser.image_paths[index].replace(image_dir_name, 'depths').replace('.png', '.npy')
            depths = np.load(depths_path)[..., None]
            depths = torch.from_numpy(depths).float()

            data['depth_map'] = depths

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/luzhan/Projects/scene_gen/consistent_3dscene/stable-virtual-camera/work_dirs/demo/img2trajvid_s-prob/llff-room")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse Blender data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    # writer = imageio.get_writer("results/points.mp4", fps=30)
    # for data in tqdm(dataset, desc="Plotting points"):
    #     image = data["image"].numpy().astype(np.uint8)
    #     points = data["points"].numpy()
    #     depths = data["depths"].numpy()
    #     for x, y in points:
    #         cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
    #     writer.append_data(image)
    # writer.close()
