import math
import torch
import numpy as np
from typing import List, Union
from .rotation import euler_angles_to_matrix

def lookat_to_matrix(lookat:torch.Tensor) -> torch.Tensor:
    batch_shape = lookat.shape[:-1]
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
        x_axis = torch.where(x_axis_mask, torch.as_tensor([0.0, 1.0, 0.0], dtype=lookat.dtype, device=lookat.device), x_axis)
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

def sample_point_on_sphere(radius: float, theta: torch.Tensor, phi: torch.Tensor):
    x = radius * torch.cos(phi) * torch.sin(theta)
    y = radius * torch.sin(phi)
    z = radius * torch.cos(phi) * torch.cos(theta)
    # pytorch3d (up-axis: Y, forward-axis: -Z) -> blender (up-axis: Z, forward-axis: Y)
    x, y, z = z, x, y
    return x, y, z

def generate_orbit_views_c2ws_from_elev_azim(radius: Union[float, List[float]] = 2.0, elevation: List[float] = None, azimuth: List[float] = None):
    ele = torch.deg2rad(torch.as_tensor(elevation, dtype=torch.float32))
    azi = torch.deg2rad(torch.as_tensor(azimuth, dtype=torch.float32))
    
    if isinstance(radius, (list, tuple)):
        radius = torch.as_tensor(radius, dtype=torch.float32)
    
    x, y, z = sample_point_on_sphere(radius, theta=azi, phi=ele)
    xyz = torch.stack([x, y, z], dim=-1)
    c2ws = lookat_to_matrix_fixed(xyz)
    return c2ws

def generate_box_views_c2ws(radius=2.8):
    return torch.tensor([
        [[ 1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [-1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[-1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [-0.0000, -0.0000, -1.0000, -radius],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[ 0.0000,  0.0000, -1.0000, -radius],
        [ 0.0000,  1.0000,  0.0000, -0.0000],
        [ 1.0000, -0.0000,  0.0000, -0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[ 1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000,  radius],
        [ 0.0000, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]],
        [[-1.0000,  0.0000, -0.0000, -0.0000],
        [-0.0000, -0.0000, -1.0000, -radius],
        [-0.0000, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    ], dtype=torch.float32)
