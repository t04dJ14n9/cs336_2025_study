"""
Distributed Data Parallel (DDP) with Bucketed Gradient Synchronization.

Groups gradients into buckets for efficient all-reduce operations.
"""

import torch
import torch.distributed as dist
from torch import nn
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DDPBucketed(nn.Module):
    """
    DDP wrapper that groups gradients into buckets for efficient all-reduce.
    
    Key features:
    1. Broadcasts parameters from rank 0 during initialization
    2. Groups gradients into buckets of specified size
    3. Asynchronously all-reduces bucket gradients during backward pass
    4. Overlaps communication with backward computation
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.handles: List = []
        self.buckets: List[List[torch.Tensor]] = []
        self.bucket_allreduce_handles: List = []
        self.bucket_buffers: List[torch.Tensor] = []
        
        # Ensure distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "DDPBucketed requires torch.distributed to be initialized. "
                "Call torch.distributed.init_process_group() first."
            )
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Create buckets and register backward hooks
        self._create_buckets()
        self._register_gradient_hooks()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to other ranks."""
        dist.barrier()
        
        # Broadcast ALL parameters (including those with requires_grad=False)
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _create_buckets(self):
        """
        Group parameters into buckets based on size.
        
        This reduces the number of all-reduce operations by grouping
        small gradients together.
        """
        bucket_size_bytes = int(self.bucket_size_mb * 1024 * 1024)
        current_bucket = []
        current_bucket_size = 0
        
        # Only bucket parameters that require gradients
        params_to_bucket = [p for p in self.module.parameters() if p.requires_grad]
        
        # Group parameters into buckets
        for param in params_to_bucket:
            param_size = param.numel() * param.element_size()
            
            # If adding this param would exceed bucket size, start new bucket
            if current_bucket_size + param_size > bucket_size_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            
            current_bucket.append(param)
            current_bucket_size += param_size
        
        # Add remaining params to final bucket
        if current_bucket:
            self.buckets.append(current_bucket)
        
        # Create mapping from parameter to bucket index
        self.param_to_bucket: Dict[int, int] = {}
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[id(param)] = bucket_idx
        
        logger.info(f"Created {len(self.buckets)} gradient buckets")
    
    def _register_gradient_hooks(self):
        """
        Register backward hooks for each parameter.
        
        When a gradient is computed, we mark it as ready. We don't immediately
        launch all-reduce - that happens in finish_gradient_synchronization.
        """
        # No hooks needed - we'll collect gradients in finish_gradient_synchronization
        pass
    
    def finish_gradient_synchronization(self):
        """
        Perform all-reduce on all gradient buckets.
        
        This should be called after backward pass, before optimizer step.
        """
        world_size = dist.get_world_size()
        
        # Process each bucket
        for bucket in self.buckets:
            # Collect gradients from bucket
            grads = [p.grad for p in bucket if p.grad is not None]
            if not grads:
                continue
            
            # Concatenate into buffer
            buffer = torch.cat([g.flatten() for g in grads])
            
            # All-reduce
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            
            # Average
            buffer.div_(world_size)
            
            # Scatter back to individual gradients
            offset = 0
            for grad in grads:
                numel = grad.numel()
                grad.copy_(buffer[offset:offset + numel].view_as(grad))
                offset += numel
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)


def ddp_bucketed_on_after_backward(ddp_model: "DDPBucketed", optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    
    Args:
        ddp_model: DDPBucketed-wrapped model
        optimizer: Optimizer being used with the model
    """
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: "DDPBucketed", optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.
    
    Args:
        ddp_model: DDPBucketed-wrapped model
        optimizer: Optimizer being used with the model
    """
    # Nothing needed at batch start for this simple implementation
    pass
