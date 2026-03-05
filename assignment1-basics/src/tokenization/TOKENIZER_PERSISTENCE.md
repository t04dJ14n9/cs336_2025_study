# BPE Tokenizer Persistence Guide

This guide explains how to save and load your trained BPE tokenizer to/from persistent storage.

## Overview

The BPE tokenizer provides three methods for persistence:

1. **`serialize()`** - Returns a dictionary representation of the tokenizer
2. **`save(path)`** - Saves the tokenizer to a JSON file
3. **`load(path)`** - Class method to load a tokenizer from a JSON file

## Quick Start

### Training and Saving

```python
from src.tokenization.bpe import BPE

# Train a tokenizer
bpe = BPE(
    input_path='corpus.txt',
    vocab_size=10000,
    special_tokens=['<|endoftext|>', '<|pad|>']
)

bpe.preprocess()
vocab, merges = bpe.train()

# Save to file
bpe.save('my_tokenizer.json')
```

### Loading

```python
from src.tokenization.bpe import BPE

# Load a previously saved tokenizer
bpe = BPE.load('my_tokenizer.json')

# Now you can use it for encoding/decoding
# (assuming you have encode/decode methods implemented)
```

## Method Details

### `serialize()` → dict

Returns a dictionary containing:
- `vocab`: Dictionary mapping token IDs to base64-encoded byte strings
- `merges`: List of merge operations as [base64_str, base64_str] pairs
- `special_tokens`: List of special token strings
- `vocab_size`: Current vocabulary size
- `target_vocab_size`: Target vocabulary size

**Example:**
```python
data = bpe.serialize()
print(data.keys())  # dict_keys(['vocab', 'merges', 'special_tokens', 'vocab_size', 'target_vocab_size'])
```

### `save(output_path: str | os.PathLike)`

Saves the tokenizer to a JSON file at the specified path.

**Parameters:**
- `output_path`: Path where the tokenizer will be saved (e.g., 'tokenizer.json')

**Example:**
```python
bpe.save('models/my_tokenizer.json')
# Output: Tokenizer saved to models/my_tokenizer.json
```

### `load(input_path: str | os.PathLike)` → BPE (classmethod)

Loads a tokenizer from a JSON file.

**Parameters:**
- `input_path`: Path to the saved tokenizer file

**Returns:**
- A BPE tokenizer instance with loaded vocabulary and merges

**Example:**
```python
bpe = BPE.load('models/my_tokenizer.json')
# Output: Tokenizer loaded from models/my_tokenizer.json
```

## File Format

The saved JSON file contains:

```json
{
  "vocab": {
    "0": "AA==",      // Base64-encoded byte for token ID 0
    "1": "AQ==",      // Base64-encoded byte for token ID 1
    ...
    "256": "c3Q=",    // Base64 for "st" (first merged token)
    "257": "ZXN0"     // Base64 for "est" (second merged token)
  },
  "merges": [
    ["cw==", "dA=="],  // Base64 pair for first merge: 's' + 't'
    ["ZQ==", "c3Q="]   // Base64 pair for second merge: 'e' + 'st'
  ],
  "special_tokens": ["<|endoftext|>"],
  "vocab_size": 272,
  "target_vocab_size": 300
}
```

**Why Base64?**
- Bytes cannot be directly serialized to JSON
- Base64 encoding converts bytes to ASCII strings
- Ensures compatibility across different systems and platforms

## Complete Example

```python
from src.tokenization.bpe import BPE

# Step 1: Train a tokenizer
print("Training tokenizer...")
bpe = BPE(
    input_path='data/corpus.txt',
    vocab_size=5000,
    special_tokens=['<|endoftext|>', '<|pad|>', '<|unk|>']
)

bpe.preprocess()
vocab, merges = bpe.train()

print(f"Trained vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

# Step 2: Save the tokenizer
tokenizer_path = 'models/tokenizer_v1.json'
bpe.save(tokenizer_path)

# Step 3: Later, load the tokenizer
print("\nLoading tokenizer...")
loaded_bpe = BPE.load(tokenizer_path)

# Step 4: Verify it loaded correctly
assert loaded_bpe.vocab == bpe.vocab
assert loaded_bpe.merges == bpe.merges
assert loaded_bpe.special_tokens == bpe.special_tokens

print("✅ Tokenizer successfully saved and loaded!")
```

## Best Practices

1. **Version your tokenizers**: Include version numbers in filenames
   ```python
   bpe.save(f'tokenizer_v{version}.json')
   ```

2. **Store metadata**: Consider adding training date, corpus info, etc.
   ```python
   import json
   from datetime import datetime
   
   data = bpe.serialize()
   data['metadata'] = {
       'trained_on': datetime.now().isoformat(),
       'corpus': 'wikipedia_en',
       'description': 'English Wikipedia tokenizer'
   }
   
   with open('tokenizer.json', 'w') as f:
       json.dump(data, f, indent=2)
   ```

3. **Backup important tokenizers**: Keep copies of production tokenizers

4. **Test after loading**: Always verify the loaded tokenizer works as expected

## Troubleshooting

### File not found error
```python
try:
    bpe = BPE.load('tokenizer.json')
except FileNotFoundError:
    print("Tokenizer file not found. Please train a new one.")
```

### JSON decode error
```python
import json

try:
    bpe = BPE.load('tokenizer.json')
except json.JSONDecodeError:
    print("Corrupted tokenizer file. Please retrain.")
```

### Memory issues with large vocabularies
For very large vocabularies (>100K tokens), consider:
- Using pickle instead of JSON (faster but less portable)
- Compressing the JSON file (gzip)
- Storing only essential data

## Alternative: Using Pickle (for Python-only projects)

If you don't need cross-language compatibility, you can use pickle:

```python
import pickle

# Save
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(bpe, f)

# Load
with open('tokenizer.pkl', 'rb') as f:
    bpe = pickle.load(f)
```

**Pros:** Faster, more compact
**Cons:** Python-only, potential security risks, version compatibility issues

## See Also

- [example_save_load_tokenizer.py](example_save_load_tokenizer.py) - Working example
- [bpe.py](src/tokenization/bpe.py) - Implementation details
