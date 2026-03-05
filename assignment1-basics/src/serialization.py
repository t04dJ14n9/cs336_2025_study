"""Checkpoint save/load utilities."""

import os
from typing import IO, Any, BinaryIO, TypedDict

import torch


class CheckpointDict(TypedDict):
    """Type definition for checkpoint dictionary."""
    model_state_dict: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    optimizer_state_dict: dict[str, Any]  # pyright: ignore[reportExplicitAny]
    iteration: int


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration to disk."""
    checkpoint: CheckpointDict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint and restore model/optimizer state. Returns iteration."""
    # torch.load returns Any by design - unavoidable for checkpoint loading
    checkpoint = torch.load(src, weights_only=False)  # pyright: ignore[reportAny]
    _ = model.load_state_dict(checkpoint["model_state_dict"])  # pyright: ignore[reportAny]
    _ = optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # pyright: ignore[reportAny]
    return checkpoint["iteration"]  # pyright: ignore[reportAny]
