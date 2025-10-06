import torch
from torch import nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
