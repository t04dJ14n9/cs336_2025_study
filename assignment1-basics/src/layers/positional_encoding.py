import torch
from torch import nn
from einops import einsum
import math
from jaxtyping import Float, Int

class RoPE(nn.Module):
    """Rotary Positional Embedding (RoPE) module.

    Args:
        theta (float): The base frequency for the rotary embeddings.
        d_k (int): Dimension of query and key vectors.
        max_seq_len (int): Maximum sequence length for positional embeddings.
        device (torch.device, optional): Device to store the embeddings on. Defaults to None.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int=1024, device: torch.device|None=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

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
                ], device=self.device))
            # Create block diagonal matrix
            rot_mats[i] = torch.block_diag(*mat_list) # the usage of '*' unpacks the list, similar to ... in Golang
        return rot_mats

    def forward(
        self, 
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]: 
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x 
        along the sequence dimension.  You should use the token positions to slice your (possibly precomputed) 
        cos and sin tensors along the sequence dimension.
        """
        # Get rotation matrices for the specified token positions
        # note that token_positions is a tensor of length seq_len and elements in rot_mats has shape (d_k, d_k)
        # so rot_mat is of shape (... seq_len, d_k, d_k)
        rot_mat = self.get_buffer('rot_mats')[token_positions]  
        
        # raise Exception(f'rot_mat shape: {rot_mat.shape}, x shape: {x.shape}')
        # Apply rotation matrices to input tensor
        return einsum(rot_mat, x, "... seq_len i j, ... seq_len j -> ... seq_len i")

        
