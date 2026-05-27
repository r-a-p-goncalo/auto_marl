import torch

def nearest_multiple(Y, W):
    lower = (Y // W) * W
    upper = lower + W
    
    if abs(Y - lower) <= abs(upper - Y):
        return lower
    else:
        return upper
    

def nearest_highest_multiple(Y, W):
    lower = (Y // W) * W
    upper = lower + W
    
    return upper


def to_float_scalar(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tuple(v.shape)}")
        return float(v.item())

    if isinstance(v, np.ndarray):
        if v.size != 1:
            raise ValueError(f"Expected scalar ndarray, got shape {v.shape}")
        return float(v.item())

    return float(v)