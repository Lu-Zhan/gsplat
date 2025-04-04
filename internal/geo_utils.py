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

    # Transform normals to image coordinates, cam coord: x -> right, y -> down, z -> forward, image coord: x -> right, y -> up, z -> backward
    camera_view_normal_map[..., 1:] *= -1

    return camera_view_normal_map


def obtain_surface_position(depth_map, distance_to_surface=0.1):
    """
    Returns the center position of the surface.
    """
    
    h, w = depth_map.shape[1:3]
    start_h, end_h = h // 2 - h // 20, h // 2 + h // 20
    start_w, end_w = w // 2 - w // 20, w // 2 + w // 20
    center_depth_map = depth_map[0, start_h:end_h, start_w:end_w, 0]
    center_distance = torch.min(center_depth_map) - distance_to_surface
    center_distance = torch.clamp(center_distance, min=0.0)

    # move camera to the center of the surface
    vector_z = torch.tensor([0.0, 0.0, center_distance]).to(depth_map.device)

    return vector_z