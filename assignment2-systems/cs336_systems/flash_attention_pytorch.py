"""FlashAttention implementation in pure PyTorch (no Triton)."""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class FlashAttentionPytorch(torch.autograd.Function):
    """
    Memory-efficient attention using PyTorch operations.
    
    Key idea: Don't store the full (batch, n_queries, n_keys) attention matrix.
    Instead, compute it on-the-fly during backward pass.
    """
    
    @staticmethod
    def forward(
        ctx,  # type: ignore
        q: Tensor,  # (batch, n_queries, d)
        k: Tensor,  # (batch, n_keys, d)
        v: Tensor,  # (batch, n_keys, d)
        is_causal: bool = False,
    ) -> Tensor:
        """
        Forward pass: Compute attention output.
        
        Args:
            q: Queries (batch, n_queries, d)
            k: Keys (batch, n_keys, d)
            v: Values (batch, n_keys, d)
            is_causal: Whether to apply causal masking
            
        Returns:
            output: (batch, n_queries, d)
        """
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        
        # Compute attention scores
        scale = 1.0 / (d ** 0.5)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (batch, n_queries, n_keys)
        
        # Apply causal mask if needed
        if is_causal:
            # Create causal mask: upper triangular part is -inf
            mask = torch.triu(
                torch.ones(n_queries, n_keys, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax with numerical stability
        # Instead of storing full attention matrix, store log-sum-exp for backward
        max_scores = scores.max(dim=-1, keepdim=True).values  # (batch, n_queries, 1)
        scores_stable = scores - max_scores
        attn_weights = F.softmax(scores_stable, dim=-1)  # (batch, n_queries, n_keys)
        
        # Compute output
        output = torch.bmm(attn_weights, v)  # (batch, n_queries, d)
        
        # Save for backward:
        # - Don't save full attention matrix (saves memory)
        # - Save Q, K, V for backward recomputation
        ctx.save_for_backward(q, k, v, max_scores.squeeze(-1))
        ctx.is_causal = is_causal
        ctx.scale = scale
        
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        """
        Backward pass: Recompute attention and compute gradients.
        
        Instead of storing the full attention matrix from forward, we recompute it.
        This trades compute for memory.
        
        Args:
            grad_outputs: Gradient of loss w.r.t. output (batch, n_queries, d)
            
        Returns:
            Tuple of gradients for (q, k, v, is_causal)
        """
        grad_output = grad_outputs[0]
        q, k, v, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        
        # Recompute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        
        # Apply causal mask if needed
        if is_causal:
            mask = torch.triu(
                torch.ones(n_queries, n_keys, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Recompute attention weights
        # Use the saved L for numerical stability
        # P = exp(scores - L.unsqueeze(-1))
        attn_weights = torch.exp(scores - L.unsqueeze(-1))  # (batch, n_queries, n_keys)
        
        # Backward through attention mechanism
        # Given: output = P @ V
        # grad_output: dL/dO (batch, n_queries, d)
        
        # Gradient w.r.t. V: dL/dV = P^T @ dL/dO
        grad_v = torch.bmm(attn_weights.transpose(1, 2), grad_output)  # (batch, n_keys, d)
        
        # Gradient w.r.t. P: dL/dP = dL/dO @ V^T
        grad_p = torch.bmm(grad_output, v.transpose(1, 2))  # (batch, n_queries, n_keys)
        
        # Backward through softmax
        # For softmax: dL/dS = P * (dL/dP - sum(dL/dP * P, dim=-1, keepdim=True))
        # This is the standard softmax backward formula
        dS = attn_weights * (grad_p - (grad_p * attn_weights).sum(dim=-1, keepdim=True))
        
        # Gradient w.r.t. Q: dL/dQ = dL/dS @ K * scale
        grad_q = torch.bmm(dS, k) * scale  # (batch, n_queries, d)
        
        # Gradient w.r.t. K: dL/dK = dL/dS^T @ Q * scale  
        grad_k = torch.bmm(dS.transpose(1, 2), q) * scale  # (batch, n_keys, d)
        
        # Return gradients (None for is_causal as it's not a tensor)
        return grad_q, grad_k, grad_v, None


def get_flashattention_autograd_function_pytorch():
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).
    
    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionPytorch
