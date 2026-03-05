#!/usr/bin/env python3
"""
Benchmark script to compare BPE training performance with different thread counts.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.tokenization.bpe_trainer import BPETrainer


def benchmark_training(input_path: str, vocab_size: int, num_threads: int, special_tokens: list | None = None):
    """
    Benchmark BPE training with specified number of threads.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        num_threads: Number of threads to use
        special_tokens: List of special tokens
        
    Returns:
        Tuple of (preprocess_time, train_time, total_time)
    """
    if special_tokens is None:
        special_tokens = ['<|endoftext|>']
    
    print(f"\n{'='*80}")
    print(f"Benchmarking with {num_threads} thread(s)")
    print(f"{'='*80}")
    
    # Create trainer
    trainer = BPETrainer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_threads=num_threads
    )
    
    # Benchmark preprocessing
    print("Starting preprocessing...")
    start_time = time.time()
    trainer.preprocess()
    preprocess_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Benchmark training
    print("Starting training...")
    start_time = time.time()
    vocab, merges = trainer.train()
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    total_time = preprocess_time + train_time
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    return preprocess_time, train_time, total_time


def main():
    """
    Run benchmarks with different thread counts and compare results.
    """
    # Configuration
    input_path = './data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 512  # Smaller vocab for faster benchmarking
    special_tokens = ['<|endoftext|>']
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Training file not found at {input_path}")
        print("Please ensure the training data is available.")
        return
    
    # Get file size
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"BPE Training Performance Benchmark")
    print(f"{'='*80}")
    print(f"Training file: {input_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Target vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    for num_threads in thread_counts:
        try:
            preprocess_time, train_time, total_time = benchmark_training(
                input_path=input_path,
                vocab_size=vocab_size,
                num_threads=num_threads,
                special_tokens=special_tokens
            )
            results[num_threads] = {
                'preprocess': preprocess_time,
                'train': train_time,
                'total': total_time
            }
        except Exception as e:
            print(f"Error with {num_threads} threads: {e}")
            continue
    
    # Print comparison
    print(f"\n{'='*80}")
    print("Performance Comparison")
    print(f"{'='*80}")
    print(f"{'Threads':<10} {'Preprocess':<15} {'Train':<15} {'Total':<15} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    baseline_time = results[1]['total'] if 1 in results else None
    
    for num_threads in sorted(results.keys()):
        result = results[num_threads]
        speedup = baseline_time / result['total'] if baseline_time else 1.0
        print(f"{num_threads:<10} {result['preprocess']:<15.2f} {result['train']:<15.2f} "
              f"{result['total']:<15.2f} {speedup:<10.2f}x")
    
    # Analysis
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("Analysis")
        print(f"{'='*80}")
        
        best_threads = min(results.keys(), key=lambda k: results[k]['total'])
        best_time = results[best_threads]['total']
        
        print(f"✅ Best performance: {best_threads} thread(s) with {best_time:.2f}s total time")
        
        if baseline_time:
            best_speedup = baseline_time / best_time
            print(f"✅ Maximum speedup: {best_speedup:.2f}x compared to single-threaded")
        
        # Calculate preprocessing speedup
        if 1 in results and best_threads in results:
            preprocess_speedup = results[1]['preprocess'] / results[best_threads]['preprocess']
            print(f"✅ Preprocessing speedup: {preprocess_speedup:.2f}x")
            
            # Calculate throughput
            throughput_single = file_size_mb / results[1]['preprocess']
            throughput_multi = file_size_mb / results[best_threads]['preprocess']
            print(f"\nThroughput comparison:")
            print(f"  Single-threaded: {throughput_single:.2f} MB/s")
            print(f"  Multi-threaded ({best_threads} threads): {throughput_multi:.2f} MB/s")
    
    print(f"\n{'='*80}")
    print("Recommendations")
    print(f"{'='*80}")
    print("✅ Use multi-threading for large datasets (>10MB)")
    print("✅ Optimal thread count is typically 4-8 for most systems")
    print("✅ Preprocessing benefits most from parallelization")
    print("✅ Training (merge operations) remains sequential")


if __name__ == "__main__":
    main()
