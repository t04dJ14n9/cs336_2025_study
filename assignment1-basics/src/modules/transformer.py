import torch
import torch.nn as nn

from jaxtyping import Float, Int
from typing_extensions import override

from src.layers import MultiHeadAttention, FeedForward, RMSNorm, Embedding, Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int=1024, pos_encoding: bool=False, theta: float=0, device: torch.device|None=None) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.d_model: int = d_model
        self.h: int = num_heads
        self.d_ff: int = d_ff
        self.max_seq_len: int = max_seq_len
        self.mask: torch.Tensor = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)).bool()
        self.MHA: MultiHeadAttention = MultiHeadAttention(d_model, num_heads, device=device, theta=theta, max_seq_len=max_seq_len, pos_encoding=pos_encoding)
        self.FF: FeedForward = FeedForward(d_model, d_ff, device=device)
        # Use nn.ModuleList to properly register layer norms as submodules
        self.LN: nn.ModuleList = nn.ModuleList([RMSNorm(d_model, device=device) for _ in range(2)])


    @override
    def forward(self, q: Float[torch.Tensor, " batch seq_len d_model"],
                 k: Float[torch.Tensor, " batch seq_len d_model"],
                 v: Float[torch.Tensor, " batch seq_len d_model"]
        ) -> Float[torch.Tensor, " batch seq_len d_model"]:
        # compute the mask
        seq_len = q.shape[-2]
        mask = self.mask[:seq_len, :seq_len]
        _ = 'a'.__repr__()
        # compute MHA y = x + MultiHeadSelfAttention(RMSNorm(x))
        x: torch.Tensor = q + self.MHA.forward(self.LN[0].forward(q), self.LN[0].forward(k), self.LN[0].forward(v), mask=mask)  # pyright: ignore[reportAny]

        # compute fast forward layer
        x = x + self.FF.forward(self.LN[1].forward(x))  # pyright: ignore[reportAny]

        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, context_length: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None=None, dtype: torch.dtype | None=None) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Fix: embedding_dim should be d_model, not context_length
        self.embedding: Embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        # All transformer blocks use RoPE positional encoding
        self.transformer_blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, pos_encoding=True, theta=rope_theta, device=device)
            for _ in range(num_layers)
        ])
        self.layer_norm: RMSNorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear: Linear = Linear(d_model, vocab_size, device=device)

    @override
    def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        # Embed input token indices: (batch, seq_len) -> (batch, seq_len, d_model)
        x: torch.Tensor = self.embedding.forward(in_indices)

        # Pass through transformer blocks (self-attention)
        for block in self.transformer_blocks:
            x = block.forward(x, x, x)  # pyright: ignore[reportAny]

        # Apply final layer normalization
        x = self.layer_norm.forward(x)

        # Project to vocabulary size: (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        x = self.linear.forward(x)

        # Return logits (unnormalized), not probabilities
        return x
