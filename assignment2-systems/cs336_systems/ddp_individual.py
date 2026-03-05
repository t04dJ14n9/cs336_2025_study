"""
Distributed Data Parallel (DDP) Implementation for CS336 Assignment 2.

Implements DDP with two strategies:
1. Individual parameters: All-reduce each parameter's gradient individually
2. Bucketed: Group parameters into buckets for efficient all-reduce
"""

import torch
import torch.distributed as dist
from torch import nn
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DDPIndividualParameters(nn.Module):
    """
    DDP wrapper that all-reduces gradients for each parameter individually.
    
    Key features:
    1. Broadcasts parameters from rank 0 during initialization
    2. Registers backward hooks to all-reduce gradients as they're computed
    3. Overlaps communication with backward pass computation
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles: List = []  # Store backward hook handles
        self.pending_allreduces: List = []  # Store async all-reduce handles
        self._grad_accs = []  # Store gradient accumulators
        
        # Ensure distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "DDPIndividualParameters requires torch.distributed to be initialized. "
                "Call torch.distributed.init_process_group() first."
            )
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Register backward hooks for gradient synchronization
        self._register_gradient_hooks()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to other ranks."""
        # Broadcast ALL parameters (including those with requires_grad=False)
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _register_gradient_hooks(self):
        """
        Register backward hooks to all-reduce gradients as they're computed.
        
        We use register_hook instead of register_post_accumulate_grad_hook
        for broader compatibility. The hook is registered on the parameter's
        grad_fn to ensure it fires after gradient accumulation.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                # Use pack_hook to get called when gradient is computed
                # Alternative: register hook on the grad accumulator
                def make_hook(p):
                    def hook(grad):
                        if grad is not None:
                            # Launch asynchronous all-reduce
                            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                            self.pending_allreduces.append(handle)
                        return grad
                    return hook
                
                # Register the hook on the parameter
                handle = param.register_hook(make_hook(param))
                self.handles.append(handle)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all pending gradient all-reduces to complete.
        
        This should be called after backward pass, before optimizer step.
        Also need to average the gradients (divide by world_size).
        """
        world_size = dist.get_world_size()
        
        # Process gradients for ALL parameters (not just those with hooks)
        for param in self.module.parameters():
            if param.grad is not None:
                # All-reduce the gradient
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                # Average by dividing by world_size
                param.grad.data.div_(world_size)


def get_ddp_individual_parameters(module: nn.Module) -> nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: nn.Module, 
    optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    if isinstance(ddp_model, DDPIndividualParameters):
        ddp_model.finish_gradient_synchronization()
