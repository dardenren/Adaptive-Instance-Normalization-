import torch

def to_float_4d(x, device):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        raise TypeError("Expected torch.Tensor")
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(device)