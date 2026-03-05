#!/usr/bin/env python3
"""
Example demonstrating multi-threaded BPE training.

This script shows how to use the multi-threaded BPE trainer
to speed up training on large datasets.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.tokenization.bpe_trainer import BPETrainer


def train_with_threads(input_path: str, vocab_size: int, num_threads: int):
    """
    Train a BPE tokenizer with specified number of threads.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        num_threads: Number of threads to use
        
    Returns:
        Training time in seconds
    """
    print(f"\n{'='*80}")
    print(f"Training with {num_threads} thread(s)")
    print(f"{'='*80}")
    
    # Create trainer
    trainer = BPETrainer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>'],
        num_threads=num_threads
    )
    
    # Train
    start_time = time.time()
    trainer.preprocess()
    vocab, merges = trainer.train()
    elapsed = time.time() - start_time
    
    print(f"Training completed in {elapsed:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    return elapsed


def main():
    """
    Demonstrate multi-threaded BPE training with different thread counts.
    """
    print("="*80)
    print("Multi-threaded BPE Training Example")
    print("="*80)
    
    # Configuration
    input_path = './data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 512  # Small vocab for quick demo
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"\nError: Training file not found at {input_path}")
        print("Please ensure the training data is available.")
        return
    
    # Get file size
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"\nTraining file: {input_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Target vocabulary size: {vocab_size}")
    
    # Example 1: Single-threaded training
    print("\n" + "="*80)
    print("Example 1: Single-threaded Training")
    print("="*80)
    print("This is the baseline for comparison.")
    
    time_single = train_with_threads(input_path, vocab_size, num_threads=1)
    
    # Example 2: Multi-threaded training (4 threads)
    print("\n" + "="*80)
    print("Example 2: Multi-threaded Training (4 threads)")
    print("="*80)
    print("Using 4 threads to parallelize preprocessing.")
    
    time_multi = train_with_threads(input_path, vocab_size, num_threads=4)
    
    # Example 3: Default (automatic thread count)
    print("\n" + "="*80)
    print("Example 3: Automatic Thread Count")
    print("="*80)
    print("Let the trainer automatically choose optimal thread count.")
    
    trainer_auto = BPETrainer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=['<|endoftext|>']
        # num_threads not specified - uses default
    )
    
    print(f"Automatically selected {trainer_auto.num_threads} threads")
    
    start_time = time.time()
    trainer_auto.preprocess()
    vocab, merges = trainer_auto.train()
    time_auto = time.time() - start_time
    
    print(f"Training completed in {time_auto:.2f} seconds")
    
    # Save the trained tokenizer
    output_path = './src/tokenization/example_tokenizer.json'
    trainer_auto.save(output_path)
    print(f"\nTokenizer saved to {output_path}")
    
    # Performance comparison
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    
    speedup_4 = time_single / time_multi
    speedup_auto = time_single / time_auto
    
    print(f"Single-threaded:  {time_single:.2f}s (baseline)")
    print(f"4 threads:        {time_multi:.2f}s ({speedup_4:.2f}x speedup)")
    print(f"Auto threads:     {time_auto:.2f}s ({speedup_auto:.2f}x speedup)")
    
    # Recommendations
    print("\n" + "="*80)
    print("Recommendations")
    print("="*80)
    print("✅ For large datasets (>10MB): Use multi-threading (4-8 threads)")
    print("✅ For small datasets (<1MB): Single-threaded is sufficient")
    print("✅ For production: Let the trainer auto-select thread count")
    print("✅ For debugging: Use num_threads=1 for deterministic behavior")
    
    # Usage tips
    print("\n" + "="*80)
    print("Usage Tips")
    print("="*80)
    print("""
# Default (recommended for most cases)
trainer = BPETrainer('corpus.txt', vocab_size=10000)

# Explicit thread count
trainer = BPETrainer('corpus.txt', vocab_size=10000, num_threads=4)

# Single-threaded (for debugging)
trainer = BPETrainer('corpus.txt', vocab_size=10000, num_threads=1)

# Maximum parallelism
from multiprocessing import cpu_count
trainer = BPETrainer('corpus.txt', vocab_size=10000, num_threads=cpu_count())
    """)


if __name__ == "__main__":
    main()
