import torch
import torch.nn as nn
from jaxtyping import Float

from src.layers import MultiHeadAttention, FeedForward, RMSNorm, RoPE

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int=1024, pos_encoding: bool=False, theta: float=0, device: torch.device|None=None):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)).bool()
        self.MHA = MultiHeadAttention(d_model, num_heads, device=device, theta=theta, max_seq_len=max_seq_len, pos_encoding=pos_encoding) 
        self.FF = FeedForward(d_model, d_ff, device=device) 
        self.LN = [RMSNorm(d_model, device=device) for _ in range(2)]
        

    def forward(self, q: Float[torch.Tensor, " batch seq_len d_model"],
                 k: Float[torch.Tensor, " batch seq_len d_model"],
                 v: Float[torch.Tensor, " batch seq_len d_model"]):
        # compute the mask
        seq_len = q.shape[-2]
        mask = self.mask[:seq_len, :seq_len]

        # compute MHA y = x + MultiHeadSelfAttention(RMSNorm(x))
        q = q + self.MHA.forward(self.LN[0].forward(q), self.LN[0].forward(k), self.LN[0].forward(v), mask=mask)

        # compute fast forward layer
        q = q + self.FF.forward(self.LN[1].forward(q))

        return q
