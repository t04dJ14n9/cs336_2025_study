import torch
from torch import nn
from einops import rearrange, einsum

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask=None):
        einsum(Q, K, 'batch_size, ..., seq_len, d_k, batch_size, ..., seq_len, d_v -> ')
