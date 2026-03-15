import torch
from torch import nn
from einops import rearrange, einsum
import math
from jaxtyping import Float, Bool, Int
from typing import override
from .softmax import softmax
from .positional_encoding import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None=None, pos_encoding: bool=False, theta: float=0, max_seq_len: int=1024): 
        """
        Initialize the multi-head attention layer.
        Args:
            d_model: The dimension of the model, equals the dimension of embedding vector for a token.
            num_heads: The number of heads.
        """
        super().__init__()
        self.h: int = num_heads
        self.d_k: int = d_model // num_heads
        self.d_v: int = d_model // num_heads

        # initialize projection weights
        self.w_q: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(torch.rand(num_heads * self.d_k, d_model, device=device)))
        self.w_k: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(torch.rand(num_heads * self.d_k, d_model, device=device)))
        self.w_v: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(torch.rand(num_heads * self.d_v, d_model, device=device)))
        self.w_o: nn.Parameter = nn.Parameter(nn.init.trunc_normal_(torch.rand(d_model, num_heads * self.d_v, device=device)))
        
        # initialize positional encoding
        self.RoPE: RoPE | None = None
        if pos_encoding:
            self.RoPE = RoPE(theta, self.d_k, max_seq_len=max_seq_len, device=device)
        

    @override
    def forward(self, 
        Q: Float[torch.Tensor, "... seq_len d_model"],
        K: Float[torch.Tensor, "... seq_len d_model"],
        V: Float[torch.Tensor, "... seq_len d_model"],
        mask: Bool[torch.Tensor, "... seq_len seq_len"] | None=None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None=None,
        ) -> Float[torch.Tensor, "... seq_len d_model"]:

        # Batched approach: project all heads at once
        # w_q, w_k, w_v have shape (num_heads * d_k, d_model)
        # After projection, we get shape (..., seq_len, num_heads * d_k)
        Q_proj = einsum(Q, self.w_q, "... seq_len d_model, hd_k d_model -> ... seq_len hd_k")
        K_proj = einsum(K, self.w_k, "... seq_len d_model, hd_k d_model -> ... seq_len hd_k")
        V_proj = einsum(V, self.w_v, "... seq_len d_model, hd_v d_model -> ... seq_len hd_v")
        
        # Reshape to separate heads: (..., seq_len, num_heads * d_k) -> (..., seq_len, num_heads, d_k)
        # Rearrange to put heads in batch dimension: (..., seq_len, h, d_k) -> (..., h, seq_len, d_k)
        Q_proj = rearrange(Q_proj, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.h)
        K_proj = rearrange(K_proj, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.h)
        V_proj = rearrange(V_proj, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.h)

        # If need to perform positional encoding (apply after reshaping to separate heads)
        if self.RoPE is not None:
            # Generate token_positions if not provided
            if token_positions is None:
                # Create token positions with same batch dimensions as Q_proj
                batch_shape = Q_proj.shape[:-2]  # Get all dimensions except seq_len and d_k
                seq_len = Q_proj.shape[-2]
                token_positions = torch.arange(seq_len, device=Q.device)
                # Expand to match batch dimensions: (..., seq_len)
                for _ in range(len(batch_shape)):
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.expand(*batch_shape, seq_len)

            # Perform encoding on separated heads: (..., h, seq_len, d_k)
            Q_proj = self.RoPE.forward(Q_proj, token_positions)
            K_proj = self.RoPE.forward(K_proj, token_positions)
        
        # Compute attention for all heads at once
        # scaled_dot_product_attention expects (..., seq_len, d)
        attention = scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask)
        # attention shape: (..., h, seq_len, d_v)
        
        # Rearrange back: (..., h, seq_len, d_v) -> (..., seq_len, h, d_v)
        # Concatenate heads: (..., seq_len, h, d_v) -> (..., seq_len, h * d_v)
        attention = rearrange(attention, "... h seq_len d_v -> ... seq_len (h d_v)")
        
        # Apply output projection: w_o has shape (d_model, h * d_v)
        output = einsum(attention, self.w_o, "... seq_len hd_v, d_model hd_v -> ... seq_len d_model")
        
        return output




def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... seq_len d_k"],
    V: Float[torch.Tensor, " ... seq_len d_v"],
    mask: Bool[torch.Tensor, " ... queries seq_len"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... seq_len d_k"]): Key tensor
        V (Float[Tensor, " ... seq_len d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries seq_len"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    # Q * K^T / sqrt(d_k)
    scores = einsum(Q, K, "... queries d_k, ... seq_len d_k -> ... queries seq_len") / math.sqrt(d_k)
    # Apply mask (mask=True means allow attention, so we mask where mask=False)
    if mask is not None:
        scores = torch.where(mask, scores, torch.tensor(float('-inf')))
    # Apply softmax
    scores = softmax(scores, dim=-1)
    # scores * V
    return einsum(scores, V, "... queries seq_len, ... seq_len d_v -> ... queries d_v")
