# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

import cv2
import torch
import trimesh
import numpy as np
import os
import sys
from PIL import Image
import torch.nn.functional as F
from typing import Union, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Standard Grid Camera Utilities
from .camera_utils import (
    transform_pos,
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
)

try:
    from .mesh_utils import load_mesh, save_mesh
except ImportError:
    print("Bpy IO CAN NOT BE Imported!!!")

try:
    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor
    INPAINT_AVAILABLE = True
except ImportError:
    print("InPaint Function CAN NOT BE Imported!!!")
    INPAINT_AVAILABLE = False

class RenderMode(Enum):
    """Rendering mode enumeration."""
    NORMAL = "normal"
    POSITION = "position"
    ALPHA = "alpha"
    UV_POS = "uvpos"
    ALBEDO = "albedo"

class ReturnType(Enum):
    """Return type enumeration."""
    TENSOR = "th"
    NUMPY = "np"
    PIL = "pl"

class TextureType(Enum):
    """Texture type enumeration."""
    DIFFUSE = "diffuse"
    METALLIC_ROUGHNESS = "mr"
    NORMAL = "normal"

@dataclass
class RenderConfig:
    """Unified rendering configuration."""
    elev: float = 0
    azim: float = 0
    camera_distance: Optional[float] = None
    center: Optional[List[float]] = None
    resolution: Optional[Union[int, Tuple[int, int]]] = None
    bg_color: List[float] = None
    return_type: str = "th"
    
    def __post_init__(self):
        if self.bg_color is None:
            self.bg_color = [1, 1, 1]

@dataclass
class ViewState:
    """Camera view state for rendering pipeline."""
    proj_mat: torch.Tensor
    mv_mat: torch.Tensor
    pos_camera: torch.Tensor
    pos_clip: torch.Tensor
    resolution: Tuple[int, int]

def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x)
    return list(reversed(stride))

def scatter_add_nd_with_count(input, count, indices, values, weights=None):
    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)
    assert len(size) == D
    input = input.view(-1, C)
    count = count.view(-1, 1)
    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)
    if weights is None:
        weights = torch.ones_like(values[..., :1])
    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)
    count.scatter_add_(0, flatten_indices.unsqueeze(1), weights)
    return input.view(*size, C), count.view(*size, 1)

def linear_grid_put_2d(H, W, coords, values, return_count=False):
    C = values.shape[-1]
    indices = coords * torch.tensor([H - 1, W - 1], dtype=torch.float32, device=coords.device)
    indices_00 = indices.floor().long()
    indices_00[:, 0].clamp_(0, H - 2)
    indices_00[:, 1].clamp_(0, W - 2)
    indices_01 = indices_00 + torch.tensor([0, 1], dtype=torch.long, device=indices.device)
    indices_10 = indices_00 + torch.tensor([1, 0], dtype=torch.long, device=indices.device)
    indices_11 = indices_00 + torch.tensor([1, 1], dtype=torch.long, device=indices.device)
    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w
    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)
    weights = torch.ones_like(values[..., :1])
    result, count = scatter_add_nd_with_count(result, count, indices_00, values * w_00.unsqueeze(1), weights * w_00.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_01, values * w_01.unsqueeze(1), weights * w_01.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_10, values * w_10.unsqueeze(1), weights * w_10.unsqueeze(1))
    result, count = scatter_add_nd_with_count(result, count, indices_11, values * w_11.unsqueeze(1), weights * w_11.unsqueeze(1))
    if return_count:
        return result, count
    mask = count.squeeze(-1) > 0
    result[mask] = result[mask] / count[mask].repeat(1, C)
    return result

def mipmap_linear_grid_put_2d(H, W, coords, values, min_resolution=128, return_count=False):
    C = values.shape[-1]
    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)
    cur_H, cur_W = H, W
    while min(cur_H, cur_W) > min_resolution:
        mask = count.squeeze(-1) == 0
        if not mask.any():
            break
        cur_result, cur_count = linear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        result[mask] = (result[mask] + F.interpolate(cur_result.permute(2, 0, 1).unsqueeze(0).contiguous(), (H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()[mask])
        count[mask] = (count[mask] + F.interpolate(cur_count.view(1, 1, cur_H, cur_W), (H, W), mode="bilinear", align_corners=False).view(H, W, 1)[mask])
        cur_H //= 2
        cur_W //= 2
    if return_count:
        return result, count
    mask = count.squeeze(-1) > 0
    result[mask] = result[mask] / count[mask].repeat(1, C)
    return result

def _normalize_image_input(image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(image, Image.Image):
        return np.array(image) / 255.0
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy() if image.is_cuda else image
    return image

def _convert_texture_format(tex: Union[np.ndarray, torch.Tensor, Image.Image], texture_size: Tuple[int, int], device: str, force_set: bool = False) -> torch.Tensor:
    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex_np = tex.cpu().numpy()
            tex = Image.fromarray((tex_np * 255).astype(np.uint8))
        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).to(device).float()
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.to(device).float()

def _format_output(image: torch.Tensor, return_type: str) -> Union[torch.Tensor, np.ndarray, Image.Image]:
    if return_type == ReturnType.NUMPY.value:
        return image.cpu().numpy()
    elif return_type == ReturnType.PIL.value:
        img_np = image.cpu().numpy() * 255
        return Image.fromarray(img_np.astype(np.uint8))
    return image

def _ensure_resolution_format(resolution: Optional[Union[int, Tuple[int, int]]], default: Tuple[int, int]) -> Tuple[int, int]:
    if resolution is None: return default
    if isinstance(resolution, (int, float)): return (int(resolution), int(resolution))
    return tuple(resolution)

def _apply_background_mask(content: torch.Tensor, visible_mask: torch.Tensor, bg_color: List[float], device: str) -> torch.Tensor:
    bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)
    return content * visible_mask + bg_tensor * (1 - visible_mask)

class MeshRender:
    def __init__(self, camera_distance=1.45, camera_type="orth", default_resolution=1024, texture_size=1024, use_antialias=True, max_mip_level=None, filter_mode="linear-mipmap-linear", bake_mode="back_sample", raster_mode="cr", shader_type="face", use_opengl=False, device="cuda"):
        self.device = device
        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)
        self.camera_distance = camera_distance
        self.use_antialias = use_antialias
        self.max_mip_level = max_mip_level
        self.tex_position = None
        self.tex_normal = None
        self.tex_grid = None
        self.texture_indices = None
        self.vtx_uv = None
        self.uv_idx = None
        self.filter_mode = filter_mode
        self.bake_angle_thres = 75
        self.set_boundary_unreliable_scale(2)
        self.bake_mode = bake_mode
        self.shader_type = shader_type
        self.raster_mode = raster_mode
        if self.raster_mode == "cr":
            # Standalone path hack for custom_rasterizer
            try:
                import custom_rasterizer as cr
            except ImportError:
                temp_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sys.path.append(os.path.join(temp_path, "custom_rasterizer"))
                import custom_rasterizer as cr
            self.raster = cr
        else:
            raise ValueError(f"No raster named {self.raster_mode}")
        if camera_type == "orth":
            self.set_orth_scale(1.0)
        elif camera_type == "perspective":
            self.camera_proj_mat = get_perspective_projection_matrix(49.13, self.default_resolution[1] / self.default_resolution[0], 0.01, 100.0)
        else:
            raise ValueError(f"No camera type {camera_type}")

    def _create_view_state(self, config: RenderConfig) -> ViewState:
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(elev=config.elev, azim=config.azim, camera_distance=self.camera_distance if config.camera_distance is None else config.camera_distance, center=config.center)
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        resolution = _ensure_resolution_format(config.resolution, self.default_resolution)
        return ViewState(proj, r_mv, pos_camera, pos_clip, resolution)

    def _compute_face_normals(self, triangles: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=-1), dim=-1)

    def _get_normals_for_shading(self, view_state: ViewState, use_abs_coor: bool = False) -> torch.Tensor:
        if use_abs_coor:
            mesh_triangles = self.vtx_pos[self.pos_idx[:, :3], :]
        else:
            pos_camera = view_state.pos_camera[:, :3] / view_state.pos_camera[:, 3:4]
            mesh_triangles = pos_camera[self.pos_idx[:, :3], :]
        face_normals = self._compute_face_normals(mesh_triangles)
        rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
        if self.shader_type == "vertex":
            vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=self.vtx_pos.shape[0], faces=self.pos_idx.cpu(), face_normals=face_normals.cpu())
            vertex_normals = torch.from_numpy(vertex_normals).float().to(self.device).contiguous()
            normal, _ = self.raster_interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        elif self.shader_type == "face":
            tri_ids = rast_out[..., 3]
            tri_ids_mask = tri_ids > 0
            tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
            normal = torch.zeros(rast_out.shape[0], rast_out.shape[1], rast_out.shape[2], 3).to(rast_out)
            normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[tri_ids[tri_ids_mask].view(-1)]
        return normal, rast_out

    def _unified_render_pipeline(self, config: RenderConfig, mode: RenderMode, **kwargs) -> torch.Tensor:
        view_state = self._create_view_state(config)
        if mode == RenderMode.ALPHA:
            rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
            return rast_out[..., -1:].long()
        elif mode == RenderMode.UV_POS:
            return self.uv_feature_map(self.vtx_pos * 0.5 + 0.5)
        elif mode == RenderMode.NORMAL:
            use_abs_coor = kwargs.get('use_abs_coor', False)
            normalize_rgb = kwargs.get('normalize_rgb', True)
            normal, rast_out = self._get_normals_for_shading(view_state, use_abs_coor)
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
            result = _apply_background_mask(normal, visible_mask, config.bg_color, self.device)
            if normalize_rgb: result = (result + 1) * 0.5
            if self.use_antialias: result = self.raster_antialias(result, rast_out, view_state.pos_clip, self.pos_idx)
            return result[0, ...]
        elif mode == RenderMode.POSITION:
            rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
            tex_position = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
            tex_position = tex_position.contiguous()
            position, _ = self.raster_interpolate(tex_position[None, ...], rast_out, self.pos_idx)
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
            result = _apply_background_mask(position, visible_mask, config.bg_color, self.device)
            if self.use_antialias: result = self.raster_antialias(result, rast_out, view_state.pos_clip, self.pos_idx)
            return result[0, ...]
        elif mode == RenderMode.ALBEDO:
            rast_out, _ = self.raster_rasterize(view_state.pos_clip, self.pos_idx, resolution=view_state.resolution)
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)
            if not hasattr(self, 'tex') or self.tex is None:
                default_color = torch.ones(view_state.resolution + (3,), device=self.device) * 0.8
                result = _apply_background_mask(default_color, visible_mask, config.bg_color, self.device)
            else:
                if self.vtx_uv is not None and self.uv_idx is not None:
                    uv_coords, _ = self.raster_interpolate(self.vtx_uv[None, ...], rast_out, self.uv_idx)
                    uv_coords = uv_coords[0, ...]
                    albedo_color = self._sample_texture_grid(uv_coords)
                else:
                    if hasattr(self, 'vertex_colors') and self.vertex_colors is not None:
                        albedo_color, _ = self.raster_interpolate(self.vertex_colors[None, ...], rast_out, self.pos_idx)
                        albedo_color = albedo_color[0, ...]
                    else:
                        albedo_color = torch.ones(view_state.resolution + (3,), device=self.device) * 0.8
                result = _apply_background_mask(albedo_color, visible_mask, config.bg_color, self.device)
            if self.use_antialias: result = self.raster_antialias(result, rast_out, view_state.pos_clip, self.pos_idx)
            return result[0, ...]

    def set_orth_scale(self, ortho_scale):
        self.ortho_scale = ortho_scale
        self.camera_proj_mat = get_orthographic_projection_matrix(left=-self.ortho_scale * 0.5, right=self.ortho_scale * 0.5, bottom=-self.ortho_scale * 0.5, top=self.ortho_scale * 0.5, near=0.1, far=100)

    def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        if self.raster_mode == "cr":
            rast_out_db = None
            if pos.dim() == 2: pos = pos.unsqueeze(0)
            if pos.dtype == torch.float64: pos = pos.to(torch.float32)
            if tri.dtype == torch.int64: tri = tri.to(torch.int32)
            findices, barycentric = self.raster.rasterize(pos, tri, resolution)
            rast_out = torch.cat((barycentric, findices.unsqueeze(-1)), dim=-1).unsqueeze(0)
        else: raise ValueError(f"No raster named {self.raster_mode}")
        return rast_out, rast_out_db

    def raster_interpolate(self, uv, rast_out, uv_idx):
        if self.raster_mode == "cr":
            textd = None
            barycentric = rast_out[0, ..., :-1]
            findices = rast_out[0, ..., -1]
            if uv.dim() == 2: uv = uv.unsqueeze(0)
            textc = self.raster.interpolate(uv, findices, barycentric, uv_idx)
        else: raise ValueError(f"No raster named {self.raster_mode}")
        return textc, textd

    def raster_antialias(self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
        return color

    def set_boundary_unreliable_scale(self, scale):
        self.bake_unreliable_kernel_size = int((scale / 512) * max(self.default_resolution[0], self.default_resolution[1]))

    def load_mesh(self, mesh, scale_factor=1.15, auto_center=True):
        vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data = load_mesh(mesh)
        self.set_mesh(vtx_pos, pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx, scale_factor=scale_factor, auto_center=auto_center)
        if texture_data is not None: self.set_texture(texture_data)

    def save_mesh(self, mesh_path, downsample=False):
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh(normalize=False)
        texture_data = self.get_texture()
        texture_metallic, texture_roughness = self.get_texture_mr()
        texture_normal = self.get_texture_normal()
        if downsample:
            texture_data = cv2.resize(texture_data, (texture_data.shape[1]//2, texture_data.shape[0]//2))
            if texture_metallic is not None: texture_metallic = cv2.resize(texture_metallic, (texture_metallic.shape[1]//2, texture_metallic.shape[0]//2))
            if texture_roughness is not None: texture_roughness = cv2.resize(texture_roughness, (texture_roughness.shape[1]//2, texture_roughness.shape[0]//2))
            if texture_normal is not None: texture_normal = cv2.resize(texture_normal, (texture_normal.shape[1]//2, texture_normal.shape[0]//2))
        save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data, metallic=texture_metallic, roughness=texture_roughness, normal=texture_normal)

    def set_mesh(self, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None, scale_factor=1.15, auto_center=True):
        self.vtx_pos = torch.from_numpy(vtx_pos).to(self.device).float()
        self.pos_idx = torch.from_numpy(pos_idx).to(self.device).int()
        if (vtx_uv is not None) and (uv_idx is not None):
            self.vtx_uv = torch.from_numpy(vtx_uv).to(self.device).float()
            self.uv_idx = torch.from_numpy(uv_idx).to(self.device).int()
        else:
            self.vtx_uv, self.uv_idx = None, None
        self.vtx_pos[:, [0, 1]] = -self.vtx_pos[:, [0, 1]]
        self.vtx_pos[:, [1, 2]] = self.vtx_pos[:, [2, 1]]
        if self.vtx_uv is not None: self.vtx_uv[:, 1] = 1.0 - self.vtx_uv[:, 1]
        if auto_center:
            max_bb, min_bb = self.vtx_pos.max(0)[0], self.vtx_pos.min(0)[0]
            center = (max_bb + min_bb) / 2
            scale = torch.norm(self.vtx_pos - center, dim=1).max() * 2.0
            self.vtx_pos = (self.vtx_pos - center) * (scale_factor / float(scale))
            self.scale_factor = scale_factor
            self.mesh_normalize_scale_factor = scale_factor / float(scale)
            self.mesh_normalize_scale_center = center.unsqueeze(0).cpu().numpy()
        else:
            self.scale_factor, self.mesh_normalize_scale_factor, self.mesh_normalize_scale_center = 1.0, 1.0, np.array([[0, 0, 0]])
        if self.vtx_uv is not None: self.extract_textiles()

    def set_texture(self, tex, force_set=False):
        self.tex = _convert_texture_format(tex, self.texture_size, self.device, force_set)

    def set_texture_mr(self, mr, force_set=False):
        self.tex_mr = _convert_texture_format(mr, self.texture_size, self.device, force_set)

    def set_texture_normal(self, normal, force_set=False):
        self.tex_normalMap = _convert_texture_format(normal, self.texture_size, self.device, force_set)

    def set_default_render_resolution(self, default_resolution):
        if isinstance(default_resolution, int): default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_default_texture_resolution(self, texture_size):
        if isinstance(texture_size, int): texture_size = (texture_size, texture_size)
        self.texture_size = texture_size

    def get_face_areas(self, from_one_index=False):
        v0, v1, v2 = self.vtx_pos[self.pos_idx[:, 0], :], self.vtx_pos[self.pos_idx[:, 1], :], self.vtx_pos[self.pos_idx[:, 2], :]
        areas = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1) * 0.5
        areas = areas.cpu().numpy()
        if from_one_index: areas = np.insert(areas, 0, 0)
        return areas

    def get_mesh(self, normalize=True):
        vtx_pos, pos_idx = self.vtx_pos.cpu().numpy(), self.pos_idx.cpu().numpy()
        vtx_uv, uv_idx = self.vtx_uv.cpu().numpy(), self.uv_idx.cpu().numpy()
        if not normalize:
            vtx_pos = vtx_pos / self.mesh_normalize_scale_factor
            vtx_pos = vtx_pos + self.mesh_normalize_scale_center
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]
        vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def get_texture(self):
        return self.tex.cpu().numpy() if hasattr(self, 'tex') else None

    def get_texture_mr(self):
        if not hasattr(self, "tex_mr"): return None, None
        mr = self.tex_mr.cpu().numpy()
        return np.repeat(mr[:, :, 0:1], 3, 2), np.repeat(mr[:, :, 1:2], 3, 2)

    def get_texture_normal(self):
        return self.tex_normalMap.cpu().numpy() if hasattr(self, "tex_normalMap") else None

    def extract_textiles(self):
        if self.vtx_uv is None:
            print("MeshRender: No UV data available to extract textiles.")
            return
        vnum = self.vtx_uv.shape[0]
        vtx_uv_ext = torch.cat((self.vtx_uv, torch.zeros_like(self.vtx_uv[:, 0:1]), torch.ones_like(self.vtx_uv[:, 0:1])), axis=1)
        vtx_uv_ext = vtx_uv_ext.view(1, vnum, 4) * 2 - 1
        rast_out, _ = self.raster_rasterize(vtx_uv_ext, self.uv_idx, resolution=self.texture_size)
        position, _ = self.raster_interpolate(self.vtx_pos, rast_out, self.pos_idx)
        v0, v1, v2 = self.vtx_pos[self.pos_idx[:, 0], :], self.vtx_pos[self.pos_idx[:, 1], :], self.vtx_pos[self.pos_idx[:, 2], :]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
        vertex_normals = trimesh.geometry.mean_vertex_normals(vertex_count=self.vtx_pos.shape[0], faces=self.pos_idx.cpu(), face_normals=face_normals.cpu())
        vertex_normals = torch.from_numpy(vertex_normals).to(self.vtx_pos).contiguous()
        position_normal, _ = self.raster_interpolate(vertex_normals[None, ...], rast_out, self.pos_idx)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ..., 0]
        position, position_normal = position[0], position_normal[0]
        tri_ids = rast_out[0, ..., 3]
        tri_ids_mask = tri_ids > 0
        tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
        position_normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[tri_ids[tri_ids_mask].view(-1)]
        row, col = torch.arange(position.shape[0], device=self.device), torch.arange(position.shape[1], device=self.device)
        grid_i, grid_j = torch.meshgrid(row, col, indexing="ij")
        mask = visible_mask.reshape(-1) > 0
        position, position_normal = position.reshape(-1, 3)[mask], position_normal.reshape(-1, 3)[mask]
        position = torch.cat((position, torch.ones_like(position[:, :1])), axis=-1)
        grid = torch.stack((grid_i, grid_j), -1).reshape(-1, 2)[mask]
        texture_indices = (torch.ones(self.texture_size[0], self.texture_size[1], device=self.device, dtype=torch.long) * -1)
        texture_indices.view(-1)[grid[:, 0] * self.texture_size[1] + grid[:, 1]] = torch.arange(grid.shape[0]).to(device=self.device, dtype=torch.long)
        self.tex_position, self.tex_normal, self.tex_grid, self.texture_indices = position, position_normal, grid, texture_indices

    def render_normal(self, elev, azim, camera_distance=None, center=None, resolution=None, bg_color=[1, 1, 1], use_abs_coor=False, normalize_rgb=True, return_type="th"):
        config = RenderConfig(elev, azim, camera_distance, center, resolution, bg_color, return_type)
        image = self._unified_render_pipeline(config, RenderMode.NORMAL, use_abs_coor=use_abs_coor, normalize_rgb=normalize_rgb)
        return _format_output(image, return_type)

    def render_position(self, elev, azim, camera_distance=None, center=None, resolution=None, bg_color=[1, 1, 1], return_type="th"):
        config = RenderConfig(elev, azim, camera_distance, center, resolution, bg_color, return_type)
        image = self._unified_render_pipeline(config, RenderMode.POSITION)
        return _format_output(image, return_type)

    def render_alpha(self, elev, azim, camera_distance=None, center=None, resolution=None, return_type="th"):
        config = RenderConfig(elev, azim, camera_distance, center, resolution, return_type=return_type)
        image = self._unified_render_pipeline(config, RenderMode.ALPHA)
        return _format_output(image, return_type)

    def uv_feature_map(self, vert_feat, bg=None):
        vtx_uv = self.vtx_uv * 2 - 1.0
        vtx_uv = torch.cat([vtx_uv, torch.zeros_like(self.vtx_uv)], dim=1).unsqueeze(0)
        vtx_uv[..., -1] = 1
        rast_out, _ = self.raster_rasterize(vtx_uv, self.uv_idx, resolution=self.texture_size)
        feat_map, _ = self.raster_interpolate(vert_feat[None, ...], rast_out, self.uv_idx)
        feat_map = feat_map[0, ...]
        if bg is not None:
            visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]
            feat_map[visible_mask == 0] = bg
        return feat_map

    def render_sketch_from_depth(self, depth_image):
        depth_image_np = (depth_image.cpu().numpy() * 255).astype(np.uint8)
        depth_edges = cv2.Canny(depth_image_np, 30, 80)
        sketch_image = torch.from_numpy(depth_edges).to(depth_image.device).float() / 255.0
        return sketch_image.unsqueeze(-1)

    def back_project(self, image, elev, azim, camera_distance=None, center=None, method=None):
        if isinstance(image, Image.Image): image = torch.tensor(np.array(image)/255.0)
        elif isinstance(image, np.ndarray): image = torch.tensor(image)
        image = image.float().to(self.device)
        if image.dim() == 2: image = image.unsqueeze(-1)
        resolution, channel = image.shape[:2], image.shape[-1]
        
        proj = self.camera_proj_mat
        r_mv = get_mv_matrix(elev=elev, azim=azim, camera_distance=self.camera_distance if camera_distance is None else camera_distance, center=center)
        pos_camera = transform_pos(r_mv, self.vtx_pos, keepdim=True)
        pos_clip = transform_pos(proj, pos_camera)
        pos_camera = pos_camera[:, :3] / pos_camera[:, 3:4]
        
        v0, v1, v2 = pos_camera[self.pos_idx[:, 0], :], pos_camera[self.pos_idx[:, 1], :], pos_camera[self.pos_idx[:, 2], :]
        face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
        
        rast_out, _ = self.raster_rasterize(pos_clip, self.pos_idx, resolution=resolution)
        visible_mask = torch.clamp(rast_out[..., -1:], 0, 1)[0, ...]
        
        if self.shader_type == "face":
            tri_ids = rast_out[..., 3]
            tri_ids_mask = tri_ids > 0
            tri_ids = ((tri_ids - 1) * tri_ids_mask).long()
            normal = torch.zeros(rast_out.shape[0], rast_out.shape[1], rast_out.shape[2], 3).to(rast_out)
            normal.reshape(-1, 3)[tri_ids_mask.view(-1)] = face_normals.reshape(-1, 3)[tri_ids[tri_ids_mask].view(-1)]
        normal = normal[0, ...]
        depth, _ = self.raster_interpolate(pos_camera[:, 2].reshape(1, -1, 1).contiguous(), rast_out, self.pos_idx)
        depth = depth[0, ...]
        
        depth_m = depth[visible_mask > 0]
        depth_normalized = (depth - depth_m.min()) / (depth_m.max() - depth_m.min()) * visible_mask
        sketch_image = self.render_sketch_from_depth(depth_normalized)
        
        lookat = torch.tensor([[0, 0, -1]], device=self.device)
        cos_image = torch.nn.functional.cosine_similarity(lookat, normal.view(-1, 3)).view(normal.shape[0], normal.shape[1], 1)
        cos_image[cos_image < np.cos(self.bake_angle_thres / 180 * np.pi)] = 0
        
        if self.bake_unreliable_kernel_size > 0:
            k = self.bake_unreliable_kernel_size * 2 + 1
            kernel = torch.ones((1, 1, k, k), device=self.device)
            vm = visible_mask.permute(2,0,1).unsqueeze(0).float()
            vm = 1.0 - (F.conv2d(1.0 - vm, kernel, padding=k//2) > 0).float()
            sk = sketch_image.permute(2,0,1).unsqueeze(0)
            sk = (F.conv2d(sk, kernel, padding=k//2) > 0).float()
            visible_mask = vm.squeeze(0).permute(1,2,0) * (sk.squeeze(0).permute(1,2,0) < 0.5)
        cos_image[visible_mask == 0] = 0
        
        method = self.bake_mode if method is None else method
        if method == "back_sample":
            if self.tex_position is None:
                raise RuntimeError("Texture_Projection Error: Attempting to bake/back_project on a mesh with no UV data or uninitialized textiles. Please ensure your mesh has UV coordinates.")
            img_proj = torch.tensor([[proj[0,0],0,0,0],[0,proj[1,1],0,0],[0,0,1,0],[0,0,0,1]], device=self.device)
            w2c = torch.from_numpy(r_mv).to(self.device)
            v_proj = self.tex_position @ w2c.T @ img_proj
            img_x = torch.clamp(((v_proj[:, 0].clamp(-1,1)*0.5+0.5)*resolution[0]).long(), 0, resolution[0]-1)
            img_y = torch.clamp(((v_proj[:, 1].clamp(-1,1)*0.5+0.5)*resolution[1]).long(), 0, resolution[1]-1)
            indices = img_y * resolution[0] + img_x
            v_z = v_proj[:, 2]
            valid_idx = torch.where((torch.abs(v_z - depth.reshape(-1)[indices]) < 3e-3) & (visible_mask.reshape(-1)[indices]*cos_image.reshape(-1)[indices]>0))[0]
            indices, valid_idx = indices[valid_idx], valid_idx
            
            texture = torch.zeros(self.texture_size[0] * self.texture_size[1], channel, device=self.device)
            cos_map = torch.zeros(self.texture_size[0] * self.texture_size[1], 1, device=self.device)
            boundary_map = torch.zeros(self.texture_size[0] * self.texture_size[1], 1, device=self.device)
            
            valid_tex_indices = self.tex_grid[valid_idx, 0] * self.texture_size[1] + self.tex_grid[valid_idx, 1]
            texture[valid_tex_indices] = image.reshape(-1, channel)[indices]
            cos_map[valid_tex_indices, 0] = cos_image.reshape(-1)[indices]
            boundary_map[valid_tex_indices, 0] = sketch_image.reshape(-1)[indices]
            
            return texture.view(self.texture_size[0], self.texture_size[1], channel), cos_map.view(self.texture_size[0], self.texture_size[1], 1), boundary_map.view(self.texture_size[0], self.texture_size[1], 1)
        else: raise ValueError(f"No bake mode {method}")

    @torch.no_grad()
    def fast_bake_texture(self, textures, cos_maps):
        channel = textures[0].shape[-1]
        texture_merge = torch.zeros(self.texture_size + (channel,), device=self.device)
        trust_map_merge = torch.zeros(self.texture_size + (1,), device=self.device)
        for texture, cos_map in zip(textures, cos_maps):
            if ((cos_map > 0) * (trust_map_merge > 0)).sum() / (cos_map > 0).sum() > 0.99: continue
            texture_merge += texture * cos_map
            trust_map_merge += cos_map
        return texture_merge / torch.clamp(trust_map_merge, min=1e-8), trust_map_merge > 1e-8

    @torch.no_grad()
    def uv_inpaint(self, texture, mask, vertex_inpaint=True, method="NS", return_float=False):
        if isinstance(texture, torch.Tensor): texture_np = texture.cpu().numpy()
        else: texture_np = texture
        if isinstance(mask, torch.Tensor): mask = (mask.squeeze(-1).cpu().numpy()*255).astype(np.uint8)
        
        if vertex_inpaint and INPAINT_AVAILABLE:
            verbose = False
            try:
                vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
                texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
            except Exception as e:
                print(f"InPaint Error: {e}")
                
        if method == "NS":
            texture_np = cv2.inpaint((texture_np*255).astype(np.uint8), 255-mask, 3, cv2.INPAINT_NS)
        return texture_np

    def _sample_texture_grid(self, uv_coords: torch.Tensor) -> torch.Tensor:
        tex = self.tex.permute(2,0,1).unsqueeze(0)
        grid = (uv_coords * 2.0 - 1.0).unsqueeze(0)
        sampled = F.grid_sample(tex, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return sampled.squeeze(0).permute(1, 2, 0)

    def render_albedo(self, elev, azim, camera_distance=None, center=None, resolution=None, bg_color=[1, 1, 1], return_type="th"):
        config = RenderConfig(elev, azim, camera_distance, center, resolution, bg_color, return_type)
        image = self._unified_render_pipeline(config, RenderMode.ALBEDO)
        return _format_output(image, return_type)
