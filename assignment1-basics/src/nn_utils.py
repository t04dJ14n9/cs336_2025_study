"""Neural network utility functions: SiLU, cross-entropy, gradient clipping, get_batch."""

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from collections.abc import Iterable


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def cross_entropy(
    inputs: Tensor,  # (batch_size, vocab_size) unnormalized logits
    targets: Tensor,  # (batch_size,) class indices
) -> Tensor:
    """Numerically stable cross-entropy loss (mean over batch)."""
    # Subtract max for numerical stability (log-sum-exp trick)
    max_vals = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_vals
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    # Gather the logits for the target classes
    target_logits = shifted[torch.arange(inputs.size(0), device=inputs.device), targets]
    loss = -target_logits + log_sum_exp
    return loss.mean()


def gradient_clipping(parameters: Iterable[Tensor], max_l2_norm: float) -> None:
    """Clip combined gradient L2 norm to max_l2_norm, in-place."""
    # Collect parameters that have gradients
    params_with_grad: list[Tensor] = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return

    # Compute total L2 norm of all gradients
    # Initialize with tensor to ensure type consistency
    total_norm_sq: Tensor = torch.tensor(0.0, device=params_with_grad[0].device)
    for p in params_with_grad:
        grad = p.grad
        assert grad is not None
        total_norm_sq += (grad.detach().float() ** 2).sum()

    # Compute total norm
    total_norm: Tensor = total_norm_sq.sqrt()

    # Clip factor
    clip_coef: Tensor = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            grad = p.grad
            assert grad is not None
            grad.detach().mul_(clip_coef)


def get_batch(
    dataset: npt.NDArray[np.int64],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample random (x, y) pairs from dataset for language modeling."""
    max_start = len(dataset) - context_length
    start_indices = np.random.randint(0, max_start, size=(batch_size,))

    x = np.stack([dataset[i : i + context_length] for i in start_indices])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in start_indices])

    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return x_tensor, y_tensor
