from typing import List, Tuple
import torch


def intr_to_proj(intr_mtx:torch.Tensor, near=0.01, far=1000.0, perspective=True):
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


def c2w_to_w2c(c2w:torch.Tensor):
    w2c = torch.zeros((*c2w.shape[:-2], 4, 4), dtype=c2w.dtype, device=c2w.device)
    c2w = c2w.clone()
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, 3:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, 3:]
    w2c[..., 3, 3] = 1.0
    return w2c

def discretize(v_pos_ndc:torch.Tensor, H:int, W:int, ndc=True, align_corner=False, to_int=False):
    uf, vf = v_pos_ndc.unbind(-1)
    if ndc:
        uf = uf * 0.5 + 0.5
        vf = vf * 0.5 + 0.5
    if not align_corner:
        ui = uf * W
        vi = vf * H
    else:
        ui = uf * (W - 1) + 0.5
        vi = vf * (H - 1) + 0.5
    v_pos_pix = torch.stack([ui, vi], dim=-1)
    if to_int:
        v_pos_pix = torch.floor(v_pos_pix).to(dtype=torch.int64)
    return v_pos_pix

def undiscretize(v_pos_pix:torch.Tensor, H:int, W:int, ndc=True, align_corner=False, from_int=False):
    if from_int:
        v_pos_pix = v_pos_pix.to(dtype=torch.float32)
    ui, vi = v_pos_pix.unbind(-1)
    if not align_corner:
        uf = (ui + 0.5) / W
        vf = (vi + 0.5) / H
    else:
        uf = ui / (W - 1)
        vf = vi / (H - 1)
    if ndc:
        uf = uf * 2.0 - 1.0
        vf = vf * 2.0 - 1.0
    v_pos_ndc = torch.stack([uf, vf], dim=-1)
    return v_pos_ndc
