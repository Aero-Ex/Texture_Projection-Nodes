import torch

def dilate_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    if depth <= 0:
        return f_mask
    v_value = torch.zeros((V,), dtype=torch.int64, device=f_v_idx.device)
    f_ones = torch.ones((f_v_idx.shape[0],), dtype=torch.int64, device=f_v_idx.device)
    f_mask_v_idx = torch.masked_select(f_v_idx, f_mask.unsqueeze(-1)).reshape(-1, 3)
    v_value = v_value.scatter_add(0, f_mask_v_idx[:, 0], f_ones).scatter_add(0, f_mask_v_idx[:, 1], f_ones).scatter_add(0, f_mask_v_idx[:, 2], f_ones)
    f_v_value = torch.gather(v_value.unsqueeze(-1).tile(1, 3), 0, f_v_idx)
    f_mask = (f_v_value.sum(dim=-1) > 0)
    return dilate_face(f_v_idx, f_mask, V, depth=depth-1)

def erode_face(f_v_idx:torch.Tensor, f_mask:torch.Tensor, V:int, depth=1):
    return ~dilate_face(f_v_idx, ~f_mask, V, depth=depth)
