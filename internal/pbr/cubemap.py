from locale import normalize
import torch
from torch.nn.functional import normalize


def look_at(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    根据相机位置 eye, 目标点 center, 以及 up 向量，
    返回一个 Camera-to-World 变换矩阵 (4x4)。
    
    约定：相机自身坐标系的 -Z 为“前向”， +Y 为“上”。
    """
    # breakpoint()
    forward = normalize(center - eye, dim=0)  # 世界空间下“相机朝前方向”（相机想看的方向）
    right   = normalize(torch.cross(forward, up, dim=-1), dim=0)  # +X
    new_up  = torch.cross(right, forward, dim=-1)          # +Y
    
    # 在约定中：相机 -Z 对应 "forward"
    # 因此第三列应是 -forward
    R = torch.stack([right, new_up, -forward], dim=-1)  # [3,3]
    
    c2w = torch.eye(4, dtype=eye.dtype, device=eye.device)
    c2w[:3, :3] = R
    c2w[:3, 3]  = eye
    return c2w


def create_cubemap_c2w(pos):
    directions_and_ups_local = [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),  # +X
        (torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),  # -X
        (torch.tensor([ 0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0])),  # +Y, up of +X
        (torch.tensor([ 0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),  # -Y, down of +X
        (torch.tensor([ 0.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])),  # +Z, right of +X
        (torch.tensor([ 0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0])),  # -Z, left of +X
    ]
    
    # 4. 对每个方向，用 R_fwd 把局部空间的 direction / up 转到世界空间
    c2w_list = []
    for (dir_local, up_local) in directions_and_ups_local:
        dir_world = dir_local.to(pos)  # [3]
        up_world  = up_local.to(pos)  # [3]
        
        center = pos + dir_world       # “朝这个方向看”的目标点
        c2w = look_at(pos, center, up_world)
        c2w_list.append(c2w)
    
    # 拼到一起: [6,4,4]
    return torch.stack(c2w_list, dim=0)


def create_six_c2w_from_c2w_forward(c2w_forward: torch.Tensor) -> torch.Tensor:
    """
    输入:
        c2w_forward: [4,4] 的相机到世界变换矩阵。
    输出:
        c2w_six: [6,4,4]，分别为渲染 +X, -X, +Y, -Y, +Z, -Z 六个方向的 c2w。
    """
    assert c2w_forward.shape == (4,4), "c2w_forward 必须是 4x4"
    
    # 1. 取出相机在世界坐标系下的位置
    pos = c2w_forward[:3, 3]  # [3]
    
    # 2. 取出旋转部分 (相机局部 -> 世界坐标)
    R_fwd = c2w_forward[:3, :3]  # [3,3]
    
    # 3. 定义在相机局部坐标系下的 6 个方向和它们对应的 up
    #    这些通常是做 CubeMap / Skybox 时常用的设置
    directions_and_ups_local = [
        (torch.tensor([ 1.0,  0.0,  0.0]), torch.tensor([0.0, 1.0,  0.0])),  # +X
        (torch.tensor([-1.0,  0.0,  0.0]), torch.tensor([0.0, 1.0,  0.0])),  # -X
        (torch.tensor([ 0.0,  1.0,  0.0]), torch.tensor([0.0, 0.0, -1.0])),  # +Y
        (torch.tensor([ 0.0, -1.0,  0.0]), torch.tensor([0.0, 0.0,  1.0])),  # -Y
        (torch.tensor([ 0.0,  0.0,  1.0]), torch.tensor([0.0, 1.0,  0.0])),  # +Z
        (torch.tensor([ 0.0,  0.0, -1.0]), torch.tensor([0.0, 1.0,  0.0])),  # -Z
    ]
    
    # 4. 对每个方向，用 R_fwd 把局部空间的 direction / up 转到世界空间
    c2w_list = []
    for (dir_local, up_local) in directions_and_ups_local:
        dir_world = R_fwd @ dir_local.to(R_fwd)  # [3]
        up_world  = R_fwd @ up_local.to(R_fwd)  # [3]
        
        center = pos + dir_world       # “朝这个方向看”的目标点
        c2w = look_at(pos, center, up_world)
        c2w_list.append(c2w)
    
    # 拼到一起: [6,4,4]
    return torch.stack(c2w_list, dim=0)


def obtain_dirs_for_skybox(height=256) -> torch.Tensor:
    """
    计算skybox中每个像素点代表的光源方向。使用简单相机模型，fx=h/2，cx=h/2。
    按照 +x, -x, +y, -y, +z, -z 的顺序排列六个面。
    
    Args:
        height: skybox的高度（宽度相同），默认256
        
    Returns:
        dirs: [6, height, height, 3] 张量，表示每个像素对应的归一化方向向量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成像素坐标网格
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(height, device=device),
        indexing='ij'
    )
    
    # 相机参数
    f = height / 2  # 焦距
    c = height / 2  # 主点
    
    # 计算归一化的相机坐标
    x = (x - c) / f
    y = (y - c) / f
    z = torch.ones_like(x)
    
    # 堆叠成方向向量 [H, W, 3]
    dirs = torch.stack([x, y, z], dim=-1)
    
    # 为六个面创建旋转矩阵
    rotations = [
        torch.tensor([[0, 0, 1],   # +x: 将z轴转到x轴
                     [0, 1, 0],
                     [-1, 0, 0]], device=device),
        
        torch.tensor([[0, 0, -1],  # -x: 将z轴转到-x轴
                     [0, 1, 0],
                     [1, 0, 0]], device=device),
        
        torch.tensor([[1, 0, 0],   # +y: 将z轴转到y轴
                     [0, 0, 1],
                     [0, -1, 0]], device=device),
        
        torch.tensor([[1, 0, 0],   # -y: 将z轴转到-y轴
                     [0, 0, -1],
                     [0, 1, 0]], device=device),
        
        torch.tensor([[1, 0, 0],   # +z: 保持原样
                     [0, 1, 0],
                     [0, 0, 1]], device=device),
        
        torch.tensor([[-1, 0, 0],  # -z: 180度旋转
                     [0, -1, 0],
                     [0, 0, -1]], device=device),
    ]
    
    # 对每个面应用旋转
    all_dirs = []
    for R in rotations:
        rotated_dirs = torch.einsum('ij,hwj->hwi', R.to(dirs), dirs)
        normalized_dirs = normalize(rotated_dirs, dim=-1)
        all_dirs.append(normalized_dirs)
    
    # 堆叠所有面 [6, H, W, 3]
    return torch.stack(all_dirs, dim=0)


if __name__ == "__main__":
    # 测试obtain_dirs_for_skybox并可视化
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 获取方向向量
    dirs = obtain_dirs_for_skybox(height=256)  # [6, 256, 256, 3]
    print("Direction vectors shape:", dirs.shape)
    
    # 将方向向量转换为RGB颜色 (将[-1,1]映射到[0,1])
    dirs_rgb = dirs.cpu().numpy() * 0.5 + 0.5
    
    # 创建3x2的图像布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    for idx, (face, name) in enumerate(zip(dirs_rgb, face_names)):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(face)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test.png')
    print("Normal map saved as test.png")

    # 可视化dirs的mo