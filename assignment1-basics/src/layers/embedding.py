from torch import nn
import torch
from typing_extensions import override

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None=None) -> None:
        super().__init__()
        self.weights: nn.Parameter = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype), 
            mean=0, std=1, a=-3, b=3
        ))
    
    @override
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
