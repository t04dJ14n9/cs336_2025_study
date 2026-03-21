#!/usr/bin/env python3
"""
Example usage of parallel encoding for BPE tokenizer.

This demonstrates how to use the parallel pre-tokenization feature
within the encode() method to speed up encoding of large texts.
"""

from src.tokenization.bpe_tokenizer import BPETokenizer
import time


def main():
    # Load a trained tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load('./saved_bpe_owt_train.json')
    
    # Create a large test text
    print("\n" + "="*80)
    print("Encoding Performance Comparison")
    print("="*80)
    
    # Create a large text sample
    large_text = ("Hello world! This is a test of the BPE tokenizer. " * 200 + 
                  "<|endoftext|>" + 
                  "Another document with different content. " * 200 +
                  "<|endoftext|>" +
                  "Yet another document to tokenize. " * 200)
    
    text_size_kb = len(large_text.encode('utf-8')) / 1024
    print(f"\nTest text size: {text_size_kb:.2f} KB")
    
    # Example 1: Sequential encoding (default)
    print("\n" + "-"*80)
    print("Example 1: Sequential Encoding (default)")
    print("-"*80)
    
    start_time = time.time()
    tokens_sequential = tokenizer.encode(large_text)
    time_sequential = time.time() - start_time
    
    print(f"Time: {time_sequential:.3f} seconds")
    print(f"Tokens generated: {len(tokens_sequential):,}")
    print(f"Throughput: {text_size_kb / time_sequential:.2f} KB/sec")
    
    # Example 2: Parallel encoding
    print("\n" + "-"*80)
    print("Example 2: Parallel Encoding (use_parallel=True)")
    print("-"*80)
    
    start_time = time.time()
    tokens_parallel = tokenizer.encode(large_text, use_parallel=True, num_processes=4)
    time_parallel = time.time() - start_time
    
    print(f"Time: {time_parallel:.3f} seconds")
    print(f"Tokens generated: {len(tokens_parallel):,}")
    print(f"Throughput: {text_size_kb / time_parallel:.2f} KB/sec")
    
    # Verify results are identical
    print("\n" + "-"*80)
    print("Verification")
    print("-"*80)
    print(f"Results match: {tokens_sequential == tokens_parallel}")
    
    if time_sequential > time_parallel:
        speedup = time_sequential / time_parallel
        print(f"Speedup: {speedup:.2f}x faster with parallel processing")
    else:
        print(f"Note: For this text size, sequential may be faster due to multiprocessing overhead")
    
    # Example 3: Automatic parallel decision
    print("\n" + "="*80)
    print("Example 3: When to Use Parallel Processing")
    print("="*80)
    
    print("\n✅ Use parallel processing (use_parallel=True) when:")
    print("   - Text is large (>10KB)")
    print("   - Multiple CPU cores available")
    print("   - Processing many documents")
    
    print("\n❌ Use sequential processing (default) when:")
    print("   - Text is small (<10KB)")
    print("   - Limited CPU cores")
    print("   - Real-time/interactive applications")
    
    # Example 4: Batch processing
    print("\n" + "="*80)
    print("Example 4: Batch Processing Multiple Documents")
    print("="*80)
    
    documents = [
        "Document 1 content " * 100,
        "Document 2 content " * 100,
        "Document 3 content " * 100,
    ]
    
    print(f"\nProcessing {len(documents)} documents...")
    
    start_time = time.time()
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc, use_parallel=True, num_processes=2)
        all_tokens.extend(tokens)
    elapsed = time.time() - start_time
    
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Time: {elapsed:.3f} seconds")
    print(f"Average per document: {elapsed/len(documents):.3f} seconds")


if __name__ == "__main__":
    main()
