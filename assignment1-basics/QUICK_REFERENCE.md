# BPE Tokenizer - Quick Reference

## Save & Load Tokenizer

### Save
```python
from src.tokenization.bpe import BPE

bpe = BPE('corpus.txt', vocab_size=10000, special_tokens=['<|endoftext|>'])
bpe.preprocess()
bpe.train()
bpe.save('tokenizer.json')  # ✅ Saves to JSON file
```

### Load
```python
from src.tokenization.bpe import BPE

bpe = BPE.load('tokenizer.json')  # ✅ Loads from JSON file
# Ready to use!
```

### Get Dictionary (without saving)
```python
data = bpe.serialize()  # Returns dict with vocab, merges, special_tokens
```

## What Gets Saved?

- ✅ **Vocabulary** (`vocab`): All token IDs → byte mappings
- ✅ **Merges** (`merges`): All BPE merge operations in order
- ✅ **Special tokens** (`special_tokens`): List of special tokens
- ✅ **Vocab size** (`vocab_size`): Current vocabulary size
- ✅ **Target vocab size** (`target_vocab_size`): Original target size

## File Format

JSON file with base64-encoded bytes for cross-platform compatibility.

## Common Use Cases

### 1. Train once, use many times
```python
# Training (do once)
bpe = BPE('large_corpus.txt', vocab_size=50000)
bpe.preprocess()
bpe.train()
bpe.save('production_tokenizer.json')

# Usage (do many times)
bpe = BPE.load('production_tokenizer.json')
# Use for encoding/decoding
```

### 2. Version control
```python
bpe.save(f'tokenizer_v{version}_{date}.json')
```

### 3. Share with team
```python
# Person A trains
bpe.save('shared_tokenizer.json')

# Person B uses
bpe = BPE.load('shared_tokenizer.json')
```

## Tips

- 💾 JSON format is human-readable and portable
- 🔄 Loaded tokenizer is identical to original
- 📦 File size ≈ vocab_size × 50 bytes (approximate)
- ⚡ Loading is fast (< 1 second for 50K vocab)

## See Full Documentation

- [TOKENIZER_PERSISTENCE.md](TOKENIZER_PERSISTENCE.md) - Complete guide
- [example_save_load_tokenizer.py](example_save_load_tokenizer.py) - Working example
