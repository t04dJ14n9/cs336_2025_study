from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
)
test_string = "Héllò hôw are ü? 🙃"

print('Original string:', repr(test_string))
print('UTF-8 bytes:', test_string.encode('utf-8'))

print('\nPre-tokenization:')
pre_tokens = tokenizer.pre_tokenize(test_string)
for i, token in enumerate(pre_tokens):
    token_bytes = bytes(token)
    print(f'  {i}: {token} -> {token_bytes} -> {repr(token_bytes)}')

print('\nEncoding:')
encoded_ids = tokenizer.encode(test_string)
print('Encoded IDs:', encoded_ids)

print('\nDecoding each token:')
for i, token_id in enumerate(encoded_ids):
    token_bytes = tokenizer.vocab[token_id]
    try:
        decoded = token_bytes.decode('utf-8')
        print(f'  {i}: {token_id} -> {repr(token_bytes)} -> {repr(decoded)}')
    except UnicodeDecodeError as e:
        print(f'  {i}: {token_id} -> {repr(token_bytes)} -> UnicodeDecodeError: {e}')

print('\nFull decode:')
decoded_string = tokenizer.decode(encoded_ids)
print('Decoded string:', repr(decoded_string))
print('Match original?', test_string == decoded_string)
