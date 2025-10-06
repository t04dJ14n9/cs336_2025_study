from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, weights=None):
        super().__init__()
        if weights is not None:
            self.weights = nn.Parameter(weights)
        else:
            self.weights = nn.Parameter(torch.nn.init.trunc_normal_(torch.rand()))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
