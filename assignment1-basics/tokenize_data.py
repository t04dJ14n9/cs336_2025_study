#!/usr/bin/env python3
"""
Tokenize Data Script for CS336 Assignment 1.

This script handles:
1. Training a BPE tokenizer on a corpus (if no pre-trained weights exist)
2. Encoding train/valid splits into .npy token arrays for fast loading

Usage:
    # Tokenize TinyStories (default, uses existing tokenizer weights)
    python3 tokenize_data.py --dataset tiny_stories

    # Tokenize OpenWebText
    python3 tokenize_data.py --dataset owt

    # Train a new tokenizer from scratch then tokenize
    python3 tokenize_data.py --dataset tiny_stories --train_tokenizer --vocab_size 10000

    # Custom paths
    python3 tokenize_data.py --dataset tiny_stories --data_dir ./data --tokenizer_dir ./tokenizers
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.tokenization.bpe_trainer import BPETrainer
from src.tokenization.bpe_tokenizer import BPETokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "tiny_stories": {
        "tokenizer_path": "src/tokenization/saved_bpe_tiny_story_train.json",
        "train_file": "TinyStoriesV2-GPT4-train.txt",
        "valid_file": "TinyStoriesV2-GPT4-valid.txt",
        "default_vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    },
    "owt": {
        "tokenizer_path": "src/tokenization/saved_bpe_owt_train.json",
        "train_file": "owt_train.txt",
        "valid_file": "owt_valid.txt",
        "default_vocab_size": 32000,
        "special_tokens": ["<|endoftext|>"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_bpe_tokenizer(corpus_path: str, vocab_size: int, special_tokens: list[str], save_path: str) -> BPETokenizer:
    """Train a BPE tokenizer from scratch and save weights."""
    print(f"\n{'='*60}")
    print(f"Training BPE tokenizer")
    print(f"  corpus:         {corpus_path}")
    print(f"  vocab_size:     {vocab_size}")
    print(f"  special_tokens: {special_tokens}")
    print(f"{'='*60}")

    t0 = time.time()
    trainer = BPETrainer(input_path=corpus_path, vocab_size=vocab_size, special_tokens=special_tokens)
    trainer.preprocess()
    vocab, merges = trainer.train()
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s  |  vocab size: {len(vocab)}  |  merges: {len(merges)}")

    # Save trainer weights
    trainer.save(save_path)

    # Return a BPETokenizer built from the trained vocab/merges
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    return tokenizer


def encode_file(tokenizer: BPETokenizer, txt_path: str, out_path: str, num_processes: int = 4):
    """Tokenize a text file and save as .npy."""
    if os.path.exists(out_path):
        tokens = np.load(out_path)
        print(f"  [cached] {out_path}  ({len(tokens):,} tokens)")
        return

    print(f"  Tokenizing {txt_path} ...")
    t0 = time.time()

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text, use_parallel=True, num_processes=num_processes)
    tokens = np.array(tokens, dtype=np.uint16)
    np.save(out_path, tokens)
    elapsed = time.time() - t0
    print(f"  Saved {len(tokens):,} tokens to {out_path}  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tokenize data for CS336 transformer training")
    parser.add_argument("--dataset", type=str, default="tiny_stories", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--tokenizer_dir", type=str, default="./src/tokenization",
                        help="Directory where tokenizer weights are stored/saved")
    parser.add_argument("--train_tokenizer", action="store_true",
                        help="Force train a new BPE tokenizer even if weights exist")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocab size for BPE training (default: dataset-specific)")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of parallel processes for encoding")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    tokenizer_path = os.path.join(args.tokenizer_dir, os.path.basename(cfg["tokenizer_path"]))
    vocab_size = args.vocab_size or cfg["default_vocab_size"]
    train_txt = os.path.join(args.data_dir, cfg["train_file"])
    valid_txt = os.path.join(args.data_dir, cfg["valid_file"])

    # ── Step 1: Get tokenizer ────────────────────────────────────────────────
    if args.train_tokenizer or not os.path.exists(tokenizer_path):
        if not os.path.exists(tokenizer_path):
            print(f"No tokenizer weights found at {tokenizer_path}, training from scratch...")
        tokenizer = train_bpe_tokenizer(
            corpus_path=train_txt,
            vocab_size=vocab_size,
            special_tokens=cfg["special_tokens"],
            save_path=tokenizer_path,
        )
    else:
        print(f"Loading pre-trained tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Step 2: Encode train/valid splits ────────────────────────────────────
    print(f"\nEncoding data files:")

    if os.path.exists(train_txt):
        train_npy = train_txt.replace(".txt", "_tokens.npy")
        encode_file(tokenizer, train_txt, train_npy, args.num_processes)
    else:
        print(f"  [skip] {train_txt} not found")

    if os.path.exists(valid_txt):
        valid_npy = valid_txt.replace(".txt", "_tokens.npy")
        encode_file(tokenizer, valid_txt, valid_npy, args.num_processes)
    else:
        print(f"  [skip] {valid_txt} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
