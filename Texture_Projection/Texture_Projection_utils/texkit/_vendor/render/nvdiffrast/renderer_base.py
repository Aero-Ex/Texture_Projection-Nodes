import math
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from ...camera.conversion import (
    intr_to_proj,
    c2w_to_w2c,
    discretize,
    undiscretize,
)
from ...geometry.triangle_topology.topology import erode_face

class NVDiffRendererBase(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.enable_nvdiffrast_cuda_ctx()
        self.enable_perspective()
        self.erode_neighbor = 0

    def enable_orthogonal(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=False)
        self.perspective = False

    def enable_perspective(self):
        self.intr_to_proj = lambda intr: intr_to_proj(intr, perspective=True)
        self.perspective = True

    def enable_nvdiffrast_cuda_ctx(self):
        self.ctx = dr.RasterizeCudaContext(device=self.device)

    def simple_rendering(
        self, mesh, v_attr:torch.Tensor, map_attr:Union[torch.Tensor, Tuple[torch.Tensor], Callable], voxel_attr:Union[torch.Tensor, Tuple[torch.Tensor], Callable], 
        c2ws:torch.Tensor, intrinsics:torch.Tensor, render_size:Union[int, Tuple[int]], 
        render_z_depth=False,
        render_world_normal=False,
        render_camera_normal=False, 
        render_world_position=False,
        render_camera_position=False,
        render_ray_direction=False,
        render_v_attr=False,
        render_uv=False,
        render_map_attr=False,
        map_normal:Optional[torch.Tensor]=None,
        normal_map_strength:float=1.0,
        background=None,
        grid_interpolate_mode="bilinear",
        enable_antialis=True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = c2ws.shape[0]
        height, width = (render_size, render_size) if isinstance(render_size, int) else render_size

        v_pos_homo = torch.cat([mesh.v_pos, torch.ones_like(mesh.v_pos[..., :1])], dim=-1)
        w2cs_mtx = c2w_to_w2c(c2ws)
        proj_mtx = self.intr_to_proj(intrinsics)
        mvp_mtx = torch.matmul(proj_mtx, w2cs_mtx)
        v_pos_clip = torch.matmul(v_pos_homo, mvp_mtx.permute(0, 2, 1))
        t_pos_idx = mesh.t_pos_idx.to(dtype=torch.int32)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip, t_pos_idx, (height, width))
        mask = rast[..., [3]] > 0
        if enable_antialis:
            alpha = dr.antialias(mask.float(), rast, v_pos_clip, t_pos_idx)
        else:
            alpha = mask.float()
        out = {"mask": mask, "alpha": alpha}

        if render_world_normal:
            world_normal, _ = dr.interpolate(mesh.v_nrm.contiguous(), rast, t_pos_idx)
            world_normal = F.normalize(world_normal, dim=-1)
            world_normal = torch.lerp(torch.full_like(world_normal, fill_value=-1.0), world_normal, alpha)
            if enable_antialis:
                world_normal = dr.antialias(world_normal, rast, v_pos_clip, t_pos_idx)
            out.update({"world_normal": world_normal})

            # Render world normal with bump map effect
            has_normal_map = (map_normal is not None and 
                            hasattr(mesh, 'v_tex') and mesh.v_tex is not None and
                            hasattr(mesh, 't_tex_idx') and mesh.t_tex_idx is not None)
            
            if has_normal_map:
                # Interpolate UVs
                v_tex_ndc = mesh.v_tex * 2.0 - 1.0
                t_tex_idx = mesh.t_tex_idx.to(torch.int32)
                gb_uv, _ = dr.interpolate(v_tex_ndc, rast, t_tex_idx)
                
                # Sample normal map
                map_normal_expanded = map_normal.expand(batch_size, *map_normal.shape[-3:])
                if grid_interpolate_mode == 'nvdiffrast':
                    sampled_normal = dr.texture(map_normal_expanded.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
                else:
                    sampled_normal = torch.nn.functional.grid_sample(
                        map_normal_expanded.permute(0, 3, 1, 2).contiguous(), 
                        gb_uv, mode=grid_interpolate_mode, align_corners=False
                    ).permute(0, 2, 3, 1)
                
                sampled_normal = sampled_normal * 2.0 - 1.0
                sampled_normal[..., :2] *= normal_map_strength
                sampled_normal = F.normalize(sampled_normal, dim=-1)
                
                # Compute TBN
                v_normal, _ = dr.interpolate(mesh.v_nrm.contiguous(), rast, t_pos_idx)
                v_normal = F.normalize(v_normal, dim=-1)
                
                v_tangent, _ = dr.interpolate(mesh.v_tng.contiguous(), rast, t_pos_idx)
                v_tangent = F.normalize(v_tangent, dim=-1)
                
                v_bitangent = torch.linalg.cross(v_normal, v_tangent)
                v_bitangent = F.normalize(v_bitangent, dim=-1)
                
                world_normal_bump = (v_tangent * sampled_normal[..., [0]] + 
                                   v_bitangent * sampled_normal[..., [1]] + 
                                   v_normal * sampled_normal[..., [2]])
                world_normal_bump = F.normalize(world_normal_bump, dim=-1)
            else:
                world_normal_bump = world_normal.clone()
            
            world_normal_bump = torch.lerp(torch.full_like(world_normal_bump, fill_value=-1.0), world_normal_bump, alpha)
            if enable_antialis:
                world_normal_bump = dr.antialias(world_normal_bump, rast, v_pos_clip, t_pos_idx)
            out.update({"world_normal_bump": world_normal_bump})
            out.update({"normal_map_texture": map_normal})

        if render_camera_normal:
            v_nrm_cam = torch.matmul(mesh.v_nrm.contiguous(), c2ws[:, :3, :3])
            v_nrm_cam = torch.nn.functional.normalize(v_nrm_cam, dim=-1)
            camera_normal, _ = dr.interpolate(v_nrm_cam, rast, t_pos_idx)
            camera_normal = torch.nn.functional.normalize(camera_normal, dim=-1)
            camera_normal = torch.lerp(torch.full_like(camera_normal, fill_value=-1.0), camera_normal, alpha)
            if enable_antialis:
                camera_normal = dr.antialias(camera_normal, rast, v_pos_clip, t_pos_idx)
            out.update({"camera_normal": camera_normal})

        if render_world_position:
            gb_ccm, _ = dr.interpolate(mesh.v_pos, rast, t_pos_idx)
            gb_bg = torch.full_like(gb_ccm, fill_value=-1.0)
            if enable_antialis:
                gb_ccm_aa = torch.lerp(gb_bg, gb_ccm, alpha)
                gb_ccm_aa = dr.antialias(gb_ccm_aa, rast, v_pos_clip, t_pos_idx)
                out.update({"world_position": gb_ccm_aa})
            else:
                gb_ccm = torch.lerp(gb_bg, gb_ccm, mask.float())
                out.update({"world_position": gb_ccm})

        if render_camera_position:
            v_pos_cam = torch.matmul(v_pos_homo, w2cs_mtx.permute(0, 2, 1))[:, :, :3].contiguous()
            camera_position, _ = dr.interpolate(v_pos_cam, rast, t_pos_idx)
            camera_position = torch.lerp(torch.full_like(camera_position, fill_value=0.0), camera_position, alpha)
            if enable_antialis:
                camera_position = dr.antialias(camera_position, rast, v_pos_clip, t_pos_idx)
            out.update({"camera_position": camera_position})

        if kwargs.get('render_map_kd', False) and kwargs.get('map_kd') is not None:
            v_tex_ndc = mesh.v_tex * 2.0 - 1.0
            t_tex_idx = mesh.t_tex_idx.to(torch.int32)
            gb_uv, _ = dr.interpolate(v_tex_ndc, rast, t_tex_idx)
            map_kd = kwargs['map_kd']
            map_kd_expanded = map_kd.expand(batch_size, *map_kd.shape[-3:])
            sampled_kd = dr.texture(map_kd_expanded.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
            sampled_kd = torch.lerp(torch.zeros_like(sampled_kd), sampled_kd, alpha)
            if enable_antialis:
                sampled_kd = dr.antialias(sampled_kd, rast, v_pos_clip, t_pos_idx)
            out.update({"map_kd": sampled_kd})

        if kwargs.get('render_map_ks', False) and kwargs.get('map_ks') is not None:
            v_tex_ndc = mesh.v_tex * 2.0 - 1.0
            t_tex_idx = mesh.t_tex_idx.to(torch.int32)
            gb_uv, _ = dr.interpolate(v_tex_ndc, rast, t_tex_idx)
            map_ks = kwargs['map_ks']
            map_ks_expanded = map_ks.expand(batch_size, *map_ks.shape[-3:])
            sampled_ks = dr.texture(map_ks_expanded.contiguous(), gb_uv.mul(0.5).add(0.5), filter_mode='linear')
            sampled_ks = torch.lerp(torch.zeros_like(sampled_ks), sampled_ks, alpha)
            if enable_antialis:
                sampled_ks = dr.antialias(sampled_ks, rast, v_pos_clip, t_pos_idx)
            out.update({"map_ks": sampled_ks})

        return out

    def geometry_rendering(self, mesh, c2ws:torch.Tensor, intrinsics:torch.Tensor, render_size:Union[int, Tuple[int]], **kwargs):
        return self.simple_rendering(
            mesh, None, None, None,
            c2ws, intrinsics, render_size,
            render_world_normal=True,
            render_camera_normal=True,
            render_world_position=True,
            render_camera_position=True,
            **kwargs,
        )
