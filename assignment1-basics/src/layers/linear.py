import math
import torch
from torch import nn
from einops import einsum
from typing_extensions import override

class Linear(nn.Module):
    """
    Construct a linear transformation module.

    This module performs a linear transformation: y = xW^T + b
    where W is the weight matrix and b is the bias vector.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        device (torch.device, optional): Device to place the parameters on
        dtype (torch.dtype, optional): Data type of the parameters

    Example:
        >>> linear = Linear(10, 5)
        >>> x = torch.randn(32, 10)  # batch_size=32, in_features=10
        >>> output = linear(x)       # shape: (32, 5)
    """

    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None, bias: bool=False) -> None:
        super().__init__()
        # Create weight parameter and initialize it
        self.weights: nn.Parameter = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        # Initialize weights using truncated normal distribution, truncated at [-3*std, 3std]
        fan_in_out: float = float(in_features + out_features)
        std: float = math.sqrt(2.0 / fan_in_out)
        _ = nn.init.trunc_normal_(self.weights, mean=0, std=std, a=(-3 * std), b=(3 * std))
        self.need_bias: bool = False

        # Create bias parameter and initialize it
        if bias:
            self.need_bias = True
            self.bias: nn.Parameter = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))


    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer. Only applies to the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        if self.need_bias:
            return einsum(x, self.weights, "... in, out in -> ... out") + self.bias

        return einsum(x, self.weights, "... in, out in -> ... out")
