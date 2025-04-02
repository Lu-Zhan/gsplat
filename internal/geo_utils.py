import torch


def transform_normals_to_image_coord(global_normal_map, c2w):
    """
    Transforms a normal map from global coordinates to camera view coordinates.

    Parameters:
    - global_normal_map (torch.Tensor): The normal map in global coordinates with shape (1, H, W, 3).
    - c2w (torch.Tensor): The camera-to-world transformation matrix with shape (1, 4, 4).

    Returns:
    - torch.Tensor: The normal map in camera view coordinates with shape (1, H, W, 3).
    """
    # Extract rotation part of the camera-to-world matrix
    rotation_matrix = c2w[0, :3, :3]

    # Invert the rotation matrix to get world-to-camera rotation
    world_to_camera_rotation = torch.linalg.inv(rotation_matrix)

    # Reshape the normal map for matrix multiplication
    if len(global_normal_map.shape) == 3:
        global_normal_map = global_normal_map.unsqueeze(0)

    H, W = global_normal_map.shape[1:3]
    reshaped_normals = global_normal_map.view(-1, 3).T  # (3, H*W)

    # Transform normals to camera view, 
    camera_view_normals = world_to_camera_rotation @ reshaped_normals

    # Reshape back to original normal map shape
    camera_view_normal_map = camera_view_normals.T.view(1, H, W, 3)

    # Transform normals to image coordinates
    camera_view_normal_map[..., 1:] *= -1

    return camera_view_normal_map