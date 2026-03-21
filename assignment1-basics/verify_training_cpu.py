#!/usr/bin/env python3
"""
Quick training verification script for CPU.
Trains a tiny model on a small subset to verify implementation correctness.
"""

import os
import sys
import time

import numpy as np
import numpy.typing as npt
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.modules.transformer import Transformer
from src.tokenization.bpe_tokenizer import BPETokenizer
from src.nn_utils import cross_entropy, gradient_clipping, get_batch  # pyright: ignore[reportUnknownVariableType]
from src.optimizer import AdamW


def create_tiny_dataset(vocab_size: int, num_docs: int = 100, doc_len: int = 50) -> npt.NDArray[np.int64]:
    """Create a tiny synthetic dataset for quick testing."""
    print(f"Creating synthetic dataset: {num_docs} docs × {doc_len} tokens")
    # Generate random token sequences (simple language model task)
    tokens: list[int] = []
    for _ in range(num_docs):
        # Each "document" is a random sequence
        doc = np.random.randint(0, vocab_size, size=doc_len)
        tokens.extend(doc.tolist())

    result = np.array(tokens, dtype=np.int64)
    print(f"Total tokens: {len(result):,}")
    return result


def verify_model_training():
    """Train a tiny model and verify loss decreases."""
    print("\n" + "="*80)
    print("TRAINING VERIFICATION TEST")
    print("="*80)
    
    # Load tokenizer to get vocab size
    tokenizer_path = "src/tokenization/saved_bpe_tiny_story_train.json"
    if os.path.exists(tokenizer_path):
        print(f"\nLoading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
        vocab_size = tokenizer.vocab_size
        print(f"Vocab size: {vocab_size}")
    else:
        print("\nNo tokenizer found, using synthetic vocab")
        vocab_size = 1000
    
    # Create tiny model config
    config = {
        "vocab_size": vocab_size,
        "context_length": 64,  # Small context
        "d_model": 64,  # Tiny embedding
        "num_layers": 2,  # Few layers
        "num_heads": 2,  # Few heads
        "d_ff": 128,  # Small FFN
        "rope_theta": 10000.0,
    }
    
    print("\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Initialize model
    device = torch.device("cpu")
    print(f"\nInitializing model on {device}...")
    model = Transformer(
        d_model=int(config["d_model"]),
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=config["rope_theta"],
        device=device,
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create synthetic dataset
    train_data = create_tiny_dataset(vocab_size, num_docs=500, doc_len=64)
    
    # Training setup
    batch_size = 8
    learning_rate = 1e-3
    max_iters = 100
    eval_interval = 20
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max iterations: {max_iters}")
    print(f"Device: {device}")
    print()
    
    losses = []
    _ = model.train()  # PyTorch method returns self for chaining
    
    for iter in range(max_iters):
        t0 = time.time()
        
        # Sample batch
        x, y = get_batch(train_data, batch_size, int(config["context_length"]), str(device))
        
        # Forward pass
        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape
        
        # Compute loss
        loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
        
        # Backward pass
        optimizer.zero_grad()
        _ = loss.backward()  # PyTorch method returns optional gradient
        
        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        # Optimizer step
        _ = optimizer.step()  # PyTorch optimizer returns optional closure result
        
        elapsed = time.time() - t0
        losses.append(loss.item())
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            avg_loss = np.mean(losses[-eval_interval:])
            print(f"Iter {iter:4d} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | Time: {elapsed*1000:.1f}ms")
    
    # Verify loss decreased
    initial_loss = float(np.mean(losses[:10]))  # type: ignore
    final_loss = float(np.mean(losses[-10:]))  # type: ignore
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    print(f"Initial loss (first 10 iters): {initial_loss:.4f}")
    print(f"Final loss (last 10 iters):    {final_loss:.4f}")
    print(f"Loss reduction:                {initial_loss - final_loss:.4f}")
    print(f"Reduction percentage:          {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
    
    # Test generation
    print("\n" + "="*80)
    print("GENERATION TEST")
    print("="*80)
    _ = model.eval()  # PyTorch method returns self for chaining
    
    # Generate a few tokens
    prompt_tokens = [0, 1, 2, 3]  # Simple prompt
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"Prompt tokens: {prompt_tokens}")
    print("Generating 20 tokens...")
    
    with torch.no_grad():
        for _ in range(20):
            # Get last context_length tokens
            ctx_len = int(config["context_length"])  # type: ignore
            context = prompt[:, -ctx_len:]
            logits: torch.Tensor = model(context)
            logits = logits[:, -1, :]  # Last position
            probs = torch.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)

    generated: list[int] = prompt[0].tolist()  # type: ignore
    print(f"Generated tokens: {generated}")

    # Check if model produces valid outputs
    valid_tokens: list[bool] = [0 <= t < vocab_size for t in generated]  # type: ignore
    print(f"\nAll generated tokens in vocab range: {all(valid_tokens)}")
    
    # Final verdict
    print("\n" + "="*80)
    all_valid = all(0 <= t < vocab_size for t in generated)  # type: ignore
    if final_loss < initial_loss and all_valid:
        print("✅ VERIFICATION PASSED")
        print("   - Loss decreased during training")
        print("   - Model generates valid tokens")
        print("   - Implementation appears correct")
    else:
        print("❌ VERIFICATION FAILED")
        if final_loss >= initial_loss:
            print("   - Loss did not decrease")
        if not all_valid:
            print("   - Generated invalid tokens")
    print("="*80)

    return final_loss < initial_loss


if __name__ == "__main__":
    success = verify_model_training()
    sys.exit(0 if success else 1)
