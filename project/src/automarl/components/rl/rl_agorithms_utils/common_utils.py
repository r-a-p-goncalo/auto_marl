import torch

def ensure_batch_agent(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)      # [B] -> [B,1]
    if x.dim() == 2:
        return x                    # already [B,A]
    raise ValueError(f"Expected [B] or [B,A], got shape {tuple(x.shape)}")