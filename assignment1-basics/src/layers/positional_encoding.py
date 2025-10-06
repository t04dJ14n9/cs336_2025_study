import torch
from torch import nn
from einops import rearrange, einsum
import math

class RoPE(nn.Module):
    """Rotary Positional Embedding (RoPE) module.

    Args:
        theta (float): The base frequency for the rotary embeddings.
        d_k (int): Dimension of query and key vectors.
        max_seq_len (int): Maximum sequence length for positional embeddings.
        device (torch.device, optional): Device to store the embeddings on. Defaults to None.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # pre-compute the rotation matrix
        self.register_buffer('rot_mats', self._compute_rot_mat(), persistent=False)
    
    def _compute_rot_mat(self):
        """Compute the rotation matrix for rotary embeddings."""
        rot_mats = torch.zeros(self.max_seq_len, self.d_k, self.d_k)
        # i is the token position
        for i in range(self.max_seq_len):
            mat_list = []
            # k denote the index of the 2 by 2 matrix on the diagonal
            for k in range(self.d_k//2):
                theta_i_k = i / (self.theta ** (2 * k / self.d_k))
                mat_list.append(torch.tensor([
                    [math.cos(theta_i_k), -math.sin(theta_i_k)],
                    [math.sin(theta_i_k), math.cos(theta_i_k)],
                ]))
            stacked_mat = torch.stack(mat_list)  # Shape: [d_k//2, 2, 2]
            # Create block diagonal matrix
            rot_mats[i] = torch.block_diag(*mat_list)
        return rot_mats


        

    """
    Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
    Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume
    that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x 
    along the sequence dimension.  You should use the token positions to slice your (possibly precomputed) 
    cos and sin tensors along the sequence dimension.
    """
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get rotation matrices for the specified token positions
        rot_mat = self.get_buffer('rot_mats')[token_positions]  # Shape: (..., seq_len, d_k, d_k)
        
        # Apply rotation: rot_mat @ x where x has shape (..., seq_len, d_k)
        # We need to add a dimension to x for matrix multiplication
        x_expanded = x.unsqueeze(-1)  # Shape: (..., seq_len, d_k, 1)
        rotated = torch.matmul(rot_mat, x_expanded)  # Shape: (..., seq_len, d_k, 1)
        return rotated.squeeze(-1)  # Shape: (..., seq_len, d_k)

        
