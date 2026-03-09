"""Adapters for CS336 Systems assignment tests."""
import torch
import torch.distributed as dist
from typing import Type
from torch import Tensor

from cs336_basics.model import BasicsTransformerLM as TransformerLM


def get_transformer_lm() -> type:
    """Returns the TransformerLM class (not an instance)."""
    return TransformerLM


def get_adamw(
    params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), 
    eps: float = 1e-8, weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """Returns an AdamW optimizer instance."""
    from cs336_basics.optimizer import AdamW
    return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def get_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute cross-entropy loss."""
    from cs336_basics.nn_utils import cross_entropy
    return cross_entropy(inputs, targets)


def get_gradient_clipping(parameters, max_l2_norm: float) -> None:
    """Clip gradient L2 norm."""
    from cs336_basics.nn_utils import gradient_clipping
    gradient_clipping(parameters, max_l2_norm)


def get_get_batch(
    dataset, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    """Get a batch of data."""
    from cs336_basics.nn_utils import get_batch
    return get_batch(dataset, batch_size, context_length, device)


def get_checkpoint_saver():
    """Returns the save_checkpoint function."""
    from cs336_basics.serialization import save_checkpoint
    return save_checkpoint


def get_checkpoint_loader():
    """Returns the load_checkpoint function."""
    from cs336_basics.serialization import load_checkpoint
    return load_checkpoint


def get_bpe_tokenizer():
    """Returns the BPETokenizer class."""
    from cs336_basics.bpe_tokenizer import BPETokenizer
    return BPETokenizer


def get_bpe_trainer():
    """Returns the train_bpe function."""
    from cs336_basics.bpe_trainer import train_bpe
    return train_bpe


def get_flashattention_autograd_function_pytorch():
    """Returns FlashAttention PyTorch autograd function."""
    from cs336_systems.flash_attention_pytorch import get_flashattention_autograd_function_pytorch
    return get_flashattention_autograd_function_pytorch()


def get_flashattention_autograd_function_triton():
    """Returns FlashAttention Triton implementation."""
    # Triton requires GPU - return None for CPU testing
    return None


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should immediately synchronize gradients
    (via all-reduce) as they are computed in the backward pass.
    This is achieved by registering backward hooks on each parameter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    from cs336_systems.ddp_individual import DDPIndividualParameters
    return DDPIndividualParameters(module)


def ddp_individual_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    from cs336_systems.ddp_individual import ddp_individual_parameters_on_after_backward as _on_after_backward
    _on_after_backward(ddp_model, optimizer)


def ddp_individual_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    from cs336_systems.ddp_individual import ddp_individual_on_train_batch_start as _on_train_batch_start
    _on_train_batch_start(ddp_model, optimizer)


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    from cs336_systems.ddp_bucketed import DDPBucketed
    return DDPBucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    from cs336_systems.ddp_bucketed import ddp_bucketed_on_after_backward as _on_after_backward
    _on_after_backward(ddp_model, optimizer)


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    from cs336_systems.ddp_bucketed import ddp_bucketed_on_train_batch_start as _on_train_batch_start
    _on_train_batch_start(ddp_model, optimizer)


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    from cs336_systems.sharded_optimizer import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
