"""
CS336 Systems - Flash Attention and DDP implementations.
"""

from .flash_attention_pytorch import get_flashattention_autograd_function_pytorch

__all__ = [
    "get_flashattention_autograd_function_pytorch",
]
