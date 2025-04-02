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

from .load_image import read_exr
from .normalize import (
    # align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    # transform_points,
)


class Parser:
    """Blender parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
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

        height, width = frames[0]['h'], frames[0]['w']
        cx, cy = frames[0]['cx'], frames[0]['cy']
        fx, fy = frames[0]['fl_x'], frames[0]['fl_y']

        for idx, frame in enumerate(frames):
            image_path = frame['file_path']
            image_names.append(image_path)
            w2c = np.array(frame['transform_matrix'])
            w2c = np.concatenate([w2c, bottom], axis=0)

            params = np.empty(0, dtype=np.float32)
            # camtype = "perspective"

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

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            transform = T1
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

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


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
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
        
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
        
        # luzhan: add intrinsics
        if self.load_intrinsics:
            # image = imageio.imread(self.parser.image_paths[index])[..., :3]
            albedo_path = self.parser.image_paths[index].replace('samples-rgb', 'intrinsics/albedo_maps').replace('.png', '.exr')
            normal_path = self.parser.image_paths[index].replace('samples-rgb', 'intrinsics/normal_maps').replace('.png', '.exr')
            roughness_path = self.parser.image_paths[index].replace('samples-rgb', 'intrinsics/roughness_maps').replace('.png', '.exr')
            metallic_path = self.parser.image_paths[index].replace('samples-rgb', 'intrinsics/metallic_maps').replace('.png', '.exr')
            irrdiance_path = self.parser.image_paths[index].replace('samples-rgb', 'intrinsics/irradiance_maps').replace('.png', '.exr')

            albedo = read_exr(albedo_path)
            normals = read_exr(normal_path)
            roughness = read_exr(roughness_path)[..., :1]
            metallic = read_exr(metallic_path)[..., :1]
            irradiance = read_exr(irrdiance_path)

            irradiance = torch.from_numpy(irradiance).float()
            normals = torch.from_numpy(normals).float()
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
            # image = imageio.imread(self.parser.image_paths[index])[..., :3]
            depths_path = self.parser.image_paths[index].replace('samples-rgb', 'depths').replace('.png', '.exr')
            depths = read_exr(depths_path, channel=1)
            depths = torch.from_numpy(depths).float()[..., None]

            data['depths'] = depths

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
