import torch
from torch import nn
from einops import einsum

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
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, bias: bool=False):
        super().__init__()
        # Create weight parameter and initialize it
        self.weights = nn.Parameter(torch.rand(out_features, in_features, device=device, dtype=dtype))
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weights)
        self.need_bias = False

        # Create bias parameter and initialize it
        if bias:
            self.need_bias = True
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        if self.need_bias:
            return einsum(x, self.weights, "... in, out in -> ... out") + self.bias

        return einsum(x, self.weights, "... in, out in -> ... out")
