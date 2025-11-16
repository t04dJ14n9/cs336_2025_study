#!/usr/bin/env python3

from src.tokenization.bpe_tokenizer import BPETokenizer

# Load the tokenizer
bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')

# Test text
text = "test villager"

print(f"Input text: '{text}'")
print(f"Vocab size: {len(bpe_tokenizer.vocab)}")
print(f"Number of merges: {len(bpe_tokenizer.merges)}")

# Step 1: Pre-tokenize
process_text = bpe_tokenizer.pre_tokenize(text)
print(f"After pre_tokenize: {process_text}")

# Check what type of data we have
print(f"Type of process_text[0]: {type(process_text[0])}")
print(f"Type of process_text[0][0]: {type(process_text[0][0])}")

# Check if individual bytes are in vocab
print(f"Is byte 116 in token_to_id? {116 in [k for k in bpe_tokenizer.vocab.keys()]}")
print(f"Is bytes([116]) in token_to_id? {bytes([116]) in bpe_tokenizer.token_to_id}")

# Let's see what's in the vocab (first few entries)
print("First 10 vocab entries:")
for i, (k, v) in enumerate(list(bpe_tokenizer.vocab.items())[:10]):
    print(f"  {k}: {v}")

print("First 10 token_to_id entries:")
for i, (k, v) in enumerate(list(bpe_tokenizer.token_to_id.items())[:10]):
    print(f"  {k}: {v}")
