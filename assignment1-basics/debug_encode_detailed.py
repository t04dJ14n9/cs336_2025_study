#!/usr/bin/env python3

from src.tokenization.bpe_tokenizer import BPETokenizer

# Load the tokenizer
bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')

# Test text
text = "test villager"

print(f"Input text: '{text}'")

# Step 1: Pre-tokenize
process_text = bpe_tokenizer.pre_tokenize(text)
print(f"After pre_tokenize: {process_text}")

# Convert integers to bytes for proper processing
print("\nConverting integers to bytes objects:")
for i, word in enumerate(process_text):
    bytes_word = [bytes([b]) for b in word]
    print(f"Word {i}: {word} -> {bytes_word}")
    process_text[i] = bytes_word

print(f"After conversion: {process_text}")

# Now try the merge process
print(f"\nTesting first few merges:")
for i, merge in enumerate(bpe_tokenizer.merges[:3]):
    print(f"Merge {i}: {merge}")
    # Apply this merge to all words
    for j in range(len(process_text)):
        old_word = process_text[j].copy()
        process_text[j] = bpe_tokenizer._encode_word(process_text[j], merge)
        if old_word != process_text[j]:
            print(f"  Word {j} changed: {old_word} -> {process_text[j]}")

print(f"After first 3 merges: {process_text}")

# Check final token lookup
print(f"\nFinal token lookup:")
for word in process_text:
    for token in word:
        if token in bpe_tokenizer.token_to_id:
            print(f"  {token} -> {bpe_tokenizer.token_to_id[token]}")
        else:
            print(f"  {token} NOT FOUND in token_to_id")
