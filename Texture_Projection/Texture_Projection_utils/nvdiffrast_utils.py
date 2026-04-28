import torch
import math
import numpy as np
from typing import Union, Optional, Tuple, List
import trimesh


def intr_to_proj(intr_mtx:torch.Tensor, near=0.01, far=1000.0, perspective=False):
    proj_mtx = torch.zeros((*intr_mtx.shape[:-2], 4, 4), dtype=intr_mtx.dtype, device=intr_mtx.device)
    intr_mtx = intr_mtx.clone()
    if perspective:
        proj_mtx[..., 0, 0] = 2 * intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = 2 * intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -(far + near) / (far - near)
        proj_mtx[..., 0, 2] = 2 * intr_mtx[..., 0, 2] - 1
        proj_mtx[..., 1, 2] = 2 * intr_mtx[..., 1, 2] - 1
        proj_mtx[..., 3, 2] = -1.0
        proj_mtx[..., 2, 3] = -2.0 * far * near / (far - near)
    else:
        proj_mtx[..., 0, 0] = intr_mtx[..., 0, 0]
        proj_mtx[..., 1, 1] = intr_mtx[..., 1, 1]
        proj_mtx[..., 2, 2] = -2.0 / (far - near)
        proj_mtx[..., 3, 3] = 1.0
        proj_mtx[..., 0, 3] = -(2 * intr_mtx[..., 0, 2] - 1)
        proj_mtx[..., 1, 3] = -(2 * intr_mtx[..., 1, 2] - 1)
        proj_mtx[..., 2, 3] = - (far + near) / (far - near)
    proj_mtx[..., 1, :] = -proj_mtx[..., 1, :]  # for nvdiffrast
    return proj_mtx


def generate_intrinsics(f_x: float, f_y: float, fov=True, degree=False):
    if fov:
        if degree:
            f_x = math.radians(f_x)
            f_y = math.radians(f_y)
        f_x_div_W = 1 / (2 * math.tan(f_x / 2))
        f_y_div_H = 1 / (2 * math.tan(f_y / 2))
    else:
        f_x_div_W = f_x
        f_y_div_H = f_y
    return torch.as_tensor([
        [f_x_div_W, 0.0, 0.5],
        [0.0, f_y_div_H, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

def c2w_to_w2c(c2w:torch.Tensor):
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    c2w = c2w.clone()
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c


def lookat_to_matrix(lookat:torch.Tensor) -> torch.Tensor:
    batch_shape = lookat.shape[:-1]
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    xyz_to_zxy = torch.as_tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=lookat.dtype, device=lookat.device)
    z_axis = torch.nn.functional.normalize(lookat, dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis_mask = (x_axis == 0).all(dim=-1, keepdim=True)
    if x_axis_mask.sum() > 0:
        x_axis = torch.where(x_axis_mask, e2, x_axis)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    c2ws = torch.cat([
        torch.cat([rots, lookat.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    c2ws = torch.matmul(xyz_to_zxy, c2ws)
    return c2ws


def lookat_to_matrix_fixed(lookat: torch.Tensor) -> torch.Tensor:
    batch_shape = lookat.shape[:-1]
    e2 = torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device)
    e3 = torch.as_tensor([0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    zzzo = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=lookat.dtype, device=lookat.device)
    xyz_to_zxy = torch.as_tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=lookat.dtype, device=lookat.device)
    z_axis = torch.nn.functional.normalize(lookat, dim=-1)
    x_axis = torch.linalg.cross(e3.expand_as(z_axis), z_axis, dim=-1)
    x_axis = torch.nn.functional.normalize(x_axis, dim=-1)
    x_axis_mask = (torch.sum(x_axis * x_axis, dim=-1, keepdim=True) < 1e-6)
    if x_axis_mask.sum() > 0:
        x_axis = torch.where(x_axis_mask, e2, x_axis)
        x_axis = torch.nn.functional.normalize(x_axis, dim=-1)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = torch.nn.functional.normalize(y_axis, dim=-1)
    
    rots = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    c2ws = torch.cat([
        torch.cat([rots, lookat.unsqueeze(-1)], dim=-1),
        zzzo.expand(batch_shape + (1, -1)),
    ], dim=1)
    
    c2ws = torch.matmul(xyz_to_zxy, c2ws)
    return c2ws


def generate_orbit_views_c2ws(num_views: int, radius: float = 1.0, height: float = 0.0, theta_0: float = 0.0, degree=False):
    if degree:
        theta_0 = math.radians(theta_0)
    projected_radius = math.sqrt(radius ** 2 - height ** 2)
    theta = torch.linspace(theta_0, 2.0 * math.pi + theta_0, num_views, dtype=torch.float32)
    x = projected_radius * torch.cos(theta)
    y = projected_radius * torch.sin(theta)
    z = torch.full((num_views,), fill_value=height, dtype=torch.float32)
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix(xyz)
    return c2ws


def sample_point_on_sphere(radius: float, theta: float = None, phi: float = None):
    theta = torch.as_tensor(theta, dtype=torch.float32)
    phi = torch.as_tensor(phi, dtype=torch.float32)
    x = radius * torch.cos(phi) * torch.sin(theta)
    y = radius * torch.sin(phi)
    z = radius * torch.cos(phi) * torch.cos(theta)
    x, y, z = z, x, y
    return x, y, z


def generate_orbit_views_c2ws_from_elev_azim(radius: float = 2.0, elevation: List[float] = None, azimuth: List[float] = None):
    ele = np.deg2rad(elevation)
    azi = np.deg2rad(azimuth)
    x, y, z = sample_point_on_sphere(radius, theta=azi, phi=ele)
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix_fixed(xyz)
    return c2ws
