import torch

def dot(x, y, dim=-1):
    return torch.sum(x * y, dim=dim, keepdim=True)

def length(x, dim=-1, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x, dim=dim), min=eps))

def safe_normalize(x, dim=-1, eps=1e-20):
    return x / length(x, dim=dim, eps=eps)
