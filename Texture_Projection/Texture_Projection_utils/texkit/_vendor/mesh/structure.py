import math
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial
from typing import Dict, Optional, Tuple, Union
from .utils import dot
from .trimesh_utils import parse_texture_visuals

class DeviceMixin:
    attr_list = []
    def to(self, device):
        device = torch.device(device)
        for key in self.attr_list:
            value = getattr(self, key, None)
            if value is not None:
                if isinstance(value, torch.Tensor) or (hasattr(value, 'device') and hasattr(value, 'to')):
                    if value.device != device:
                        setattr(self, key, value.to(device))
        return self

class CoordinateSystemMixin:
    def __init__(self) -> None:
        self._identity = torch.eye(4, dtype=torch.float32)
        self._transform = None
        self.attr_list.extend(['_identity', '_transform'])
    
    @property
    def identity(self) -> torch.Tensor: return self._identity
    
    def init_transform(self, transform:Optional[torch.Tensor]=None):
        self._transform = self.identity.clone() if transform is None else transform
        return self
    
    def apply_transform(self, clear_transform=True):
        if self._transform is not None:
            v_pos_homo = torch.cat([self.v_pos, torch.ones_like(self.v_pos[:, [0]])], dim=-1)
            v_pos_homo = torch.matmul(v_pos_homo, self._transform.T.to(v_pos_homo))
            self.v_pos = v_pos_homo[:, :3].contiguous()
            v_nrm = getattr(self, '_v_nrm', None)
            if v_nrm is not None:
                v_nrm = torch.matmul(v_nrm, self._transform[:3, :3].T.to(v_nrm))
                v_nrm = torch.nn.functional.normalize(v_nrm, dim=-1)
                self._v_nrm = v_nrm.contiguous()
            if clear_transform: self._transform = None
        return self
    
    def compose_transform(self, transform:torch.Tensor, after=True):
        if self._transform is None: self.init_transform()
        if after: self._transform = torch.matmul(transform.to(self._transform), self._transform)
        else: self._transform = torch.matmul(self._transform, transform.to(self._transform))
        return self

    def scale_to_bbox(self, largest=True, scale=1.0):
        bbox = torch.stack([self.v_pos.min(dim=0).values, self.v_pos.max(dim=0).values], dim=0)
        ccc = bbox.mean(dim=0)
        sss = (bbox[1, :] - bbox[0, :]) / (2.0 * scale)
        sss = sss.max() if largest else sss.min()
        transform = self.identity.clone()
        transform[[0, 1, 2], [0, 1, 2]] = 1 / sss
        transform[:3, 3] = - ccc / sss
        self.compose_transform(transform)
        return self

class Mesh(DeviceMixin, CoordinateSystemMixin):
    def __init__(self, v_pos:torch.Tensor, t_pos_idx:torch.Tensor, v_tex:Optional[torch.Tensor]=None, t_tex_idx:Optional[torch.Tensor]=None, **kwargs):
        self.attr_list = ['v_pos', 't_pos_idx', '_v_nrm', '_v_tng', '_v_tex', '_t_tex_idx']
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self._v_nrm = None
        self._v_tng = None
        self._v_tex = v_tex
        self._t_tex_idx = t_tex_idx
        CoordinateSystemMixin.__init__(self)
    
    @property
    def device(self): return self.v_pos.device

    @property
    def v_nrm(self) -> torch.Tensor:
        if self._v_nrm is None:
            i0, i1, i2 = self.t_pos_idx[:, 0].long(), self.t_pos_idx[:, 1].long(), self.t_pos_idx[:, 2].long()
            面法线 = torch.linalg.cross(self.v_pos[i1] - self.v_pos[i0], self.v_pos[i2] - self.v_pos[i0])
            self._v_nrm = torch.zeros_like(self.v_pos)
            self._v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), 面法线)
            self._v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), 面法线)
            self._v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), 面法线)
            self._v_nrm = F.normalize(self._v_nrm, dim=1)
        return self._v_nrm

    @property
    def v_tng(self) -> torch.Tensor:
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)
            tansum.scatter_add_(0, idx, torch.ones_like(tang))
        tangents = tangents / tansum

        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)
        return tangents

    @property
    def v_tex(self): return self._v_tex
    @property
    def t_tex_idx(self): return self._t_tex_idx

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        v_pos = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        t_pos_idx = torch.as_tensor(mesh.faces, dtype=torch.int64)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)

class Texture(DeviceMixin):
    def __init__(self, mesh:Mesh, v_rgb:Optional[torch.Tensor]=None, map_Kd:Optional[torch.Tensor]=None, map_Ks:Optional[torch.Tensor]=None, map_normal:Optional[torch.Tensor]=None, **kwargs):
        self.attr_list = ['mesh', 'v_rgb', 'map_Kd', 'map_Ks', 'map_normal']
        self.mesh = mesh
        self.v_rgb, self.map_Kd, self.map_Ks, self.map_normal = v_rgb, map_Kd, map_Ks, map_normal
    
    @property
    def device(self): return self.mesh.device

    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        m = mesh.copy()
        if isinstance(m.visual, TextureVisuals):
            map_Kd, map_Ks, map_normal = parse_texture_visuals(m.visual)
            if map_Kd is not None: map_Kd = torch.as_tensor(np.array(map_Kd.convert('RGBA'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            if map_Ks is not None: map_Ks = torch.as_tensor(np.array(map_Ks.convert('RGB'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            if map_normal is not None: map_normal = torch.as_tensor(np.array(map_normal.convert('RGB'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            v_tex = torch.as_tensor(m.visual.uv, dtype=torch.float32) if m.visual.uv is not None else None
            t_tex_idx = torch.as_tensor(m.faces, dtype=torch.int64) if m.visual.uv is not None else None
        else:
            map_Kd, map_Ks, map_normal, v_tex, t_tex_idx = None, None, None, None, None
        mesh_obj = Mesh.from_trimesh(m)
        mesh_obj._v_tex, mesh_obj._t_tex_idx = v_tex, t_tex_idx
        return Texture(mesh=mesh_obj, map_Kd=map_Kd, map_Ks=map_Ks, map_normal=map_normal)
