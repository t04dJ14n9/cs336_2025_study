import torch
from torch import nn
from typing import override

# RMSNorm is a layer normalization layer that normalizes the inputs based on their root mean square (RMS) value.
class RMSNorm(nn.Module):
    # eps: epsilon value for numerical stability
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.device: torch.device | None = device
        self.dtype: torch.dtype | None = dtype
        self.d_model: int = d_model
        self.eps: float = eps
        # initialize weights of layer norm to 1
        self.g: nn.Parameter = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    # Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)/self.d_model + self.eps)
        result =  x / mean_square * self.g
        return result.to(in_dtype)
        