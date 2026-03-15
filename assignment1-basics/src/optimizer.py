"""AdamW optimizer and cosine learning rate schedule."""

import math
from typing import Callable, Iterable
from collections.abc import Iterable as AbcIterable
import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(self, params: AbcIterable[Tensor], lr: float=1e-3, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None=None) -> float | None:  # type: ignore[override, reportIncompatibleMethodOverride]
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
            Note: PyTorch's base Optimizer.step() returns None, but many
            optimizers (including AdamW) return Optional[float] for closure support.
        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad

                state: dict[str, Tensor | int] = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                step: int = int(state["step"]) + 1
                state["step"] = step
                t: int = step

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                assert isinstance(exp_avg, Tensor)
                assert isinstance(exp_avg_sq, Tensor)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1: float = 1 - beta1 ** t
                bias_correction2: float = 1 - beta2 ** t

                step_size: float = lr / bias_correction1
                denom: Tensor = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Parameter update (Adam part)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    - it < warmup_iters: linear warmup from 0 to max_lr
    - warmup_iters <= it < cosine_cycle_iters: cosine decay from max_lr to min_lr
    - it >= cosine_cycle_iters: constant min_lr
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it >= cosine_cycle_iters:
        return min_learning_rate
    else:
        # Cosine annealing phase
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
