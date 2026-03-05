# CS336 Assignment 1: Quick Start Guide

## Overview

This assignment implements a transformer language model from scratch with:
- **BPE Tokenization**: Train a custom tokenizer on your dataset
- **Transformer Architecture**: Decoder-only transformer with RoPE embeddings
- **Training Pipeline**: AdamW optimizer with cosine learning rate schedule

## Quick Start

### 1. Basic Training (Recommended)
```bash
# Tokenize data and train transformer on TinyStories (default)
./run.sh
```

### 2. Quick Testing
```bash
# Fast training for testing (small model, few iterations)
./run.sh --quick
```

### 3. Full Training
```bash
# Full training with larger model
./run.sh --full
```

### 4. OpenWebText Dataset
```bash
# Train on OpenWebText instead
./run.sh --dataset owt
```

## Advanced Usage

### Force Retrain Tokenizer
```bash
# Retrain BPE tokenizer from scratch
./run.sh --train-tokenizer
```

### Specify Device
```bash
# Use specific GPU
./run.sh --device cuda:0

# Force CPU training
./run.sh --device cpu
```

### Resume Training
```bash
# Resume from checkpoint
python3 train.py --resume checkpoints/tiny_stories/checkpoint_5000.pt
```

## Individual Scripts

### Tokenization Only
```bash
# Tokenize TinyStories
python3 tokenize_data.py --dataset tiny_stories

# Tokenize OpenWebText
python3 tokenize_data.py --dataset owt

# Train new tokenizer
python3 tokenize_data.py --dataset tiny_stories --train_tokenizer --vocab_size 10000
```

### Training Only
```bash
# Custom hyperparameters
python3 train.py \
    --dataset tiny_stories \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 1024 \
    --batch_size 64 \
    --context_length 512 \
    --max_iters 10000 \
    --device cuda
```

## Training Modes

| Mode | Model Size | Iterations | Time (GPU) | Use Case |
|------|-----------|------------|------------|----------|
| **Quick** | Small (128d, 2L) | 1K | ~5 min | Testing pipeline |
| **Default** | Medium (256d, 4L) | 5K | ~30 min | Development |
| **Full** | Large (512d, 6L) | 20K | ~2 hours | Final training |

## Output Structure

```
assignment1-basics/
├── data/
│   ├── TinyStoriesV2-GPT4-train.txt         # Raw training data
│   ├── TinyStoriesV2-GPT4-train_tokens.npy  # Tokenized data
│   └── TinyStoriesV2-GPT4-valid_tokens.npy  # Validation tokens
├── checkpoints/
│   └── tiny_stories/
│       ├── checkpoint_1000.pt
│       ├── checkpoint_2000.pt
│       └── checkpoint_5000.pt
├── logs/
│   └── tiny_stories_train.log
└── src/tokenization/
    └── saved_bpe_tiny_story_train.json      # Trained tokenizer
```

## Monitoring Training

### View Training Logs
```bash
tail -f logs/tiny_stories_train.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Expected Results

### Training Loss
- **TinyStories**: Should decrease from ~9.2 to ~7.5 (quick mode) or ~6.5 (full mode)
- **OpenWebText**: Should decrease from ~9.5 to ~7.0

### Model Checkpoints
- Saved every `checkpoint_interval` iterations
- Can be loaded for generation or continued training

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or model size
python3 train.py --batch_size 16 --d_model 128
```

### Slow Training on CPU
```bash
# Use quick mode with small model
./run.sh --quick --device cpu
```

### Missing Data Files
```bash
# Ensure data files are in data/ directory
ls -lh data/TinyStoriesV2-GPT4-*.txt
```

## Testing Your Implementation

### Run Unit Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_transformer.py -v
```

### Verify Training Works
```bash
# Quick verification on CPU
python3 verify_training_cpu.py
```

## Files Created by This Script

1. **Tokenized Data** (`.npy` files)
   - Fast loading during training
   - Reused across multiple training runs

2. **Trained Tokenizer** (`.json` file)
   - BPE vocabulary and merges
   - Can be loaded for inference

3. **Model Checkpoints** (`.pt` files)
   - Model weights and optimizer state
   - Training iteration counter

4. **Training Logs**
   - Loss values over time
   - Evaluation metrics

## Next Steps

After training completes:
1. **Generate text**: `python3 generate.py --checkpoint checkpoints/tiny_stories/checkpoint_final.pt`
2. **Evaluate perplexity**: `python3 evaluate.py --checkpoint checkpoints/tiny_stories/checkpoint_final.pt`
3. **Analyze outputs**: Check `logs/` for training curves

## Resources

- **Course PDF**: `cs336_spring2025_assignment1_basics.pdf`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Test Cases**: `tests/` directory
