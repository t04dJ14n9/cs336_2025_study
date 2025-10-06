import torch
from torch import nn
from einops import einsum

class FeedForward(nn.Module):
    """
    FeedForward layer with SwiGLU activation using einops for clear tensor operations.
    
    SwiGLU: SiLU(W1 @ x) ⊙ (W3 @ x) @ W2
    where ⊙ denotes element-wise multiplication
    
    Args:
        w1: First projection weight matrix [d_ff, d_model]
        w2: Output projection weight matrix [d_model, d_ff] 
        w3: Gate projection weight matrix [d_ff, d_model]
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(nn.init.trunc_normal_(torch.rand(d_ff, d_model)))
        self.w2 = nn.Parameter(nn.init.trunc_normal_(torch.rand(d_model, d_ff)))
        self.w3 = nn.Parameter(nn.init.trunc_normal_(torch.rand(d_ff, d_model)))


    def _load_weight(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        """Validate that weight matrices have compatible dimensions"""
        d_ff_1, d_model_1 = w1.shape
        d_model_2, d_ff_2 = w2.shape
        d_ff_3, d_model_3 = w3.shape
        
        assert d_model_1 == d_model_2 == d_model_3, f"d_model mismatch: {d_model_1}, {d_model_2}, {d_model_3}"
        assert d_ff_1 == d_ff_2 == d_ff_3, f"d_ff mismatch: {d_ff_1}, {d_ff_2}, {d_ff_3}"

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU (Swish) activation function: x * sigmoid(x)"""
        return x * torch.sigmoid(x)

    def _swiglu_einops(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation using einops for clear tensor operations.
        
        Args:
            x: Input tensor [..., d_model]
            
        Returns:
            Output tensor [..., d_model]
        """
        # Project input through w1 and w3 gates
        # x: [..., d_model], w1/w3: [d_ff, d_model] -> gate1/gate3: [..., d_ff]
        gate1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        gate3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        
        # Apply SiLU to first gate and multiply with second gate
        activated = self._silu(gate1) * gate3
        
        # Project back to d_model using w2
        # activated: [..., d_ff], w2: [d_model, d_ff] -> output: [..., d_model]
        output = einsum(activated, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        
        return output

    def _swiglu_traditional(self, x: torch.Tensor) -> torch.Tensor:
        """
        Traditional implementation for comparison (without einops).
        
        Args:
            x: Input tensor [..., d_model]
            
        Returns:
            Output tensor [..., d_model]
        """
        # Reshape for matrix multiplication: [..., d_model] -> [..., 1, d_model]
        x_reshaped = x.unsqueeze(-2)
        
        # Matrix multiplications
        gate1 = torch.matmul(x_reshaped, self.w1.T).squeeze(-2)  # [..., d_ff]
        gate3 = torch.matmul(x_reshaped, self.w3.T).squeeze(-2)  # [..., d_ff]
        
        # SwiGLU activation
        activated = self._silu(gate1) * gate3  # [..., d_ff]
        
        # Output projection
        activated_reshaped = activated.unsqueeze(-2)  # [..., 1, d_ff]
        output = torch.matmul(activated_reshaped, self.w2.T).squeeze(-2)  # [..., d_model]
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network using einops.
        
        Args:
            x: Input tensor of shape [..., d_model]
            
        Returns:
            Output tensor of shape [..., d_model]
        """
        return self._swiglu_einops(x)
