#!/usr/bin/env python3
"""
Transformer Language Model Training Script for CS336 Assignment 1.

Trains a decoder-only Transformer on tokenized data (.npy) using:
- BPE tokenization (pre-trained vocab loaded for vocab_size)
- AdamW optimizer with cosine LR schedule + linear warmup
- Gradient clipping
- Periodic eval, logging, and checkpointing

Usage:
    # Train on TinyStories (default)
    python3 train.py

    # Train on TinyStories with custom hyperparams
    python3 train.py --d_model 512 --num_layers 4 --num_heads 8 --d_ff 1024 \
                     --batch_size 64 --context_length 256 --max_iters 10000

    # Train on OWT
    python3 train.py --dataset owt --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048

    # Resume from checkpoint
    python3 train.py --resume ./checkpoints/tiny_stories/checkpoint_5000.pt
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.modules.transformer import Transformer
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.nn_utils import cross_entropy, gradient_clipping, get_batch
from src.optimizer import AdamW, get_lr_cosine_schedule
from src.serialization import save_checkpoint, load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "tiny_stories": {
        "tokenizer_path": "src/tokenization/saved_bpe_tiny_story_train.json",
        "train_tokens": "data/TinyStoriesV2-GPT4-train_tokens.npy",
        "valid_tokens": "data/TinyStoriesV2-GPT4-valid_tokens.npy",
    },
    "owt": {
        "tokenizer_path": "src/tokenization/saved_bpe_owt_train.json",
        "train_tokens": "data/owt_train_tokens.npy",
        "valid_tokens": "data/owt_valid_tokens.npy",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, context_length, device, eval_iters=50):
    """Estimate loss on a dataset by averaging over eval_iters batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str, temperature: float = 0.8):
    """Simple autoregressive generation for sanity checking."""
    model.eval()
    token_ids = tokenizer.encode(prompt)
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to context length (take the last context_length tokens)
        context = ids[:, -256:]
        logits = model(context)                # (1, T, V)
        logits = logits[:, -1, :] / temperature  # (1, V) last position
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

    model.train()
    return tokenizer.decode(ids[0].tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    parser.add_argument("--dataset", type=str, default="tiny_stories", choices=list(DATASET_CONFIGS.keys()))

    # Model architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    # Eval & checkpointing
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--generate_interval", type=int, default=500,
                        help="Generate sample text every N iters (0 to disable)")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    ckpt_dir = os.path.join(args.checkpoint_dir, args.dataset)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Load tokenizer (for vocab_size + generation) ─────────────────────────
    print(f"Loading tokenizer from {cfg['tokenizer_path']}")
    tokenizer = BPETokenizer.load(cfg["tokenizer_path"])
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # ── Load tokenized data ──────────────────────────────────────────────────
    print(f"\nLoading tokenized data...")
    if not os.path.exists(cfg["train_tokens"]):
        print(f"ERROR: {cfg['train_tokens']} not found. Run tokenize_data.py first:")
        print(f"  python3 tokenize_data.py --dataset {args.dataset}")
        sys.exit(1)

    train_data = np.load(cfg["train_tokens"]).astype(np.int64)
    print(f"  Train: {len(train_data):,} tokens")

    val_data = None
    if os.path.exists(cfg["valid_tokens"]):
        val_data = np.load(cfg["valid_tokens"]).astype(np.int64)
        print(f"  Valid: {len(val_data):,} tokens")
    else:
        print(f"  Valid: not found, will use train data for eval")
        val_data = train_data

    # ── Create model ─────────────────────────────────────────────────────────
    device = args.device
    print(f"\nCreating model on {device} ...")
    model = Transformer(
        d_model=args.d_model,
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps,
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_iter = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed at iteration {start_iter}")

    # ── Print config ─────────────────────────────────────────────────────────
    cosine_cycle_iters = args.max_iters
    print(f"\n{'='*70}")
    print(f"  TRAINING CONFIG")
    print(f"{'='*70}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Model            : d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, d_ff={args.d_ff}")
    print(f"  Context length   : {args.context_length}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Parameters       : {num_params:,}")
    print(f"  Max LR           : {args.max_lr}")
    print(f"  Min LR           : {args.min_lr}")
    print(f"  Warmup iters     : {args.warmup_iters}")
    print(f"  Max iters        : {args.max_iters}")
    print(f"  Grad clip norm   : {args.max_grad_norm}")
    print(f"  Weight decay     : {args.weight_decay}")
    print(f"  Device           : {device}")
    print(f"  Checkpoint dir   : {ckpt_dir}")
    print(f"{'='*70}\n")

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    running_loss = 0.0
    log_count = 0

    for it in range(start_iter, args.max_iters):
        # -- LR schedule --
        lr = get_lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # -- Eval --
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.context_length, device, args.eval_iters)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            elapsed = time.time() - t0
            print(
                f"[eval] iter {it:>6d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"ppl {math.exp(val_loss):.1f} | lr {lr:.6f} | {elapsed:.0f}s"
            )
            model.train()

        # -- Generate sample --
        if args.generate_interval > 0 and it > 0 and it % args.generate_interval == 0:
            prompt = "Once upon a time"
            sample = generate(model, tokenizer, prompt, max_new_tokens=100, device=device)
            print(f"[gen]  iter {it}: {sample[:300]}")
            model.train()

        # -- Forward --
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        B, T, V = logits.shape
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))

        # -- Backward --
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # -- Logging --
        running_loss += loss.item()
        log_count += 1
        if (it + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_count
            elapsed = time.time() - t0
            tokens_per_sec = (log_count * args.batch_size * args.context_length) / max(elapsed - (t0 if it < args.log_interval else 0), 1e-9)
            print(f"[train] iter {it+1:>6d} | loss {avg_loss:.4f} | lr {lr:.6f} | {elapsed:.0f}s")
            running_loss = 0.0
            log_count = 0

        # -- Checkpoint --
        if (it + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

    # ── Final ────────────────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)

    val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Final val loss: {val_loss:.4f}  |  Perplexity: {math.exp(val_loss):.2f}")
    print(f"  Checkpoint: {final_path}")
    print(f"{'='*70}")

    # Final generation sample
    prompt = "Once upon a time"
    sample = generate(model, tokenizer, prompt, max_new_tokens=200, device=device)
    print(f"\n[sample] {sample[:500]}")


if __name__ == "__main__":
    main()
