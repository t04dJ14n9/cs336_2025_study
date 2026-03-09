# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is the Stanford CS336 Spring 2025 course repository containing four independent assignments, each in its own directory with its own Python package and `pyproject.toml`.

```
cs336/
├── assignment1-basics/    # Transformer LM from scratch
├── assignment2-systems/   # Systems optimizations (Flash Attention, DDP, ZeRO)
├── assignment3-scaling/   # Scaling laws analysis
└── assignment4-data/      # Data filtering and processing pipeline
```

## Package Management

All assignments use `uv` for dependency management. **Never use `pip install` directly.**

```sh
uv run <command>          # Run any command in the managed environment
uv run pytest             # Run all tests
uv run pytest tests/test_model.py  # Run a single test file
uv run pytest -k "test_name"       # Run a specific test by name
uv add <package>          # Add a dependency
```

## Assignment 1: Basics (`assignment1-basics/`)

**Package**: `cs336_basics` | **Module root**: repo root (not inside `cs336_basics/`)

**Source layout** (your implementation goes in `src/`):
- `src/layers/` — Linear, Embedding, RMSNorm, RoPE, FeedForward, MultiHeadAttention
- `src/modules/` — TransformerBlock, Transformer
- `src/tokenization/` — BPETrainer, BPETokenizer
- `src/nn_utils.py` — silu, cross_entropy, gradient_clipping, get_batch
- `src/optimizer.py` — AdamW, get_lr_cosine_schedule
- `src/serialization.py` — save_checkpoint, load_checkpoint

**Test adapter**: `tests/adapters.py` — bridges tests to your implementation. Must be completed before tests pass.

**Key constraints**: Cannot use `torch.nn.*` (except `Parameter`, `Module`, `ModuleList`, `Sequential`). RMSNorm must upcast to float32. SwiGLU FFN uses `d_ff ≈ (8/3)*d_model` rounded to multiple of 64.

**Submission**: `./make_submission.sh`

## Assignment 2: Systems (`assignment2-systems/`)

**Package**: `cs336_systems` | Depends on `cs336-basics` (staff implementation in `./cs336-basics/`)

**Your code goes in**: `cs336_systems/`
- `flash_attention_pytorch.py` — FlashAttention via PyTorch autograd
- `ddp_individual.py` — DDP with per-parameter gradient all-reduce hooks
- `ddp_bucketed.py` — DDP with bucketed gradient communication
- `sharded_optimizer.py` — ZeRO-style sharded optimizer

**Tests**: `tests/` covers attention correctness, DDP correctness, and sharded optimizer.

**Submission**: `./test_and_make_submission.sh`

## Assignment 3: Scaling (`assignment3-scaling/`)

**Package**: `cs336_scaling`

- `cs336_scaling/model.py` — model definition
- `cs336_scaling/data_pipeline.py` — data pipeline
- `scaling_analysis.py` — scaling laws analysis script

No submission script; this is an analysis/writeup assignment.

## Assignment 4: Data (`assignment4-data/`)

**Package**: `cs336_data` | Depends on `cs336-basics` (staff training code in `./cs336-basics/`)

**Your code goes in**: `cs336_data/` — implement data filtering and processing.

**Submission**: `./test_and_make_submission.sh`

## Cross-Assignment Dependencies

- A2, A3, A4 all include a `cs336-basics/` subdirectory with the staff LM implementation.
- If you want to use your A1 implementation in A2/A4, edit the outer `pyproject.toml`'s `[tool.uv.sources]` to point to your A1 directory.

## Linting

Ruff is configured with `line-length = 120`. Run with:
```sh
uv run ruff check .
uv run ruff format .
```
