"""
Sharded Optimizer for distributed training.

Implements optimizer state sharding across ranks to reduce memory footprint.
Each rank stores optimizer state only for its subset of parameters.
"""

import torch
import torch.distributed as dist
from typing import Iterator


class ShardedOptimizer:
    """
    Sharded optimizer that stores optimizer state only for local parameters.
    
    Each rank stores optimizer state only for parameters it "owns",
    reducing memory from O(N) to O(N/world_size).
    """
    
    def __init__(
        self,
        params,
        optimizer_class: type[torch.optim.Optimizer],
        **kwargs
    ):
        """
        Initialize sharded optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            optimizer_class: Base optimizer class (e.g., torch.optim.AdamW)
            **kwargs: Arguments to pass to optimizer constructor
        """
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Convert params to list
        params_list = list(params)
        all_params = params_list
        self.all_params = all_params
        
        # Assign each parameter to a rank
        self.param_owners = {}
        for idx, param in enumerate(all_params):
            owner_rank = idx % self.world_size
            self.param_owners[id(param)] = owner_rank
        
        # Get parameters owned by this rank
        owned_params = [
            p for p in all_params 
            if self.param_owners[id(p)] == self.rank
        ]
        
        # Initialize base optimizer with only local params
        self.local_optimizer = optimizer_class(owned_params, **kwargs)
        
        # Store optimizer class and kwargs for reference
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = kwargs
        
        # Log initialization
        print(
            f"Rank {self.rank}: ShardedOptimizer managing "
            f"{len(owned_params)}/{len(params_list)} parameters"
        )
    
    def step(self, closure=None) -> float | None:
        """
        Perform one optimization step with gradient synchronization.
        
        Args:
            closure: Optional closure for reevaluating model
            
        Returns:
            Loss value if closure is provided, None otherwise
        """
        # Synchronize gradients across ranks
        # For sharded optimizer, we need to:
        # 1. All-reduce gradients for all parameters
        # 2. Each rank only updates its owned parameters
        # 3. Broadcast updated parameters to other ranks
        
        # Step 1: All-reduce gradients
        for param in self.all_params:
            if param.grad is not None:
                # Average gradient across all ranks
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
        
        # Step 2: Local optimizer step (only updates owned params)
        loss = self.local_optimizer.step(closure)
        
        # Step 3: Broadcast updated parameters from owners to other ranks
        for param in self.all_params:
            owner_rank = self.param_owners[id(param)]
            dist.broadcast(param.data, src=owner_rank)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all parameters."""
        for param in self.all_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()
    
    def state_dict(self):
        """Return the state dict of the local optimizer."""
        return self.local_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into the local optimizer."""
        self.local_optimizer.load_state_dict(state_dict)
    
    def add_param_group(self, param_group):
        """Add a parameter group to the local optimizer."""
        self.local_optimizer.add_param_group(param_group)


def get_sharded_optimizer(
    params,
    optimizer_class: type[torch.optim.Optimizer],
    **kwargs
) -> ShardedOptimizer:
    """
    Factory function to create a ShardedOptimizer.
    
    Args:
        params: Iterable of parameters to optimize
        optimizer_class: Base optimizer class (e.g., torch.optim.AdamW)
        **kwargs: Arguments to pass to optimizer constructor
    
    Returns:
        ShardedOptimizer instance
    """
    return ShardedOptimizer(params, optimizer_class, **kwargs)
