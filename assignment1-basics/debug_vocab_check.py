from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
)

# Check if the expected tokens exist in vocabulary
test_chars = ['é', 'ò', 'ô', 'ü']
print('Checking for Unicode characters in vocabulary:')
for char in test_chars:
    char_bytes = char.encode('utf-8')
    if char_bytes in tokenizer.token_to_id:
        token_id = tokenizer.token_to_id[char_bytes]
        print(f'  {repr(char)} -> {char_bytes} -> token_id {token_id}')
    else:
        print(f'  {repr(char)} -> {char_bytes} -> NOT FOUND in vocabulary')

# Check what tokens tiktoken expects
import tiktoken
reference_tokenizer = tiktoken.get_encoding('gpt2')
expected_ids = [2634, 27083]  # 'é' and 'ô' from tiktoken
print(f'\nChecking tiktoken expected tokens:')
for token_id in expected_ids:
    if token_id in tokenizer.vocab:
        token_bytes = tokenizer.vocab[token_id]
        try:
            decoded = token_bytes.decode('utf-8')
            print(f'  token_id {token_id} -> {token_bytes} -> {repr(decoded)}')
        except:
            print(f'  token_id {token_id} -> {token_bytes} -> (decode error)')
    else:
        print(f'  token_id {token_id} -> NOT FOUND in our vocabulary')

# Let's also check what our current pre-tokenization is missing
print(f'\nOur current pre-tokenization result:')
pre_tokens = tokenizer.pre_tokenize('Héllò hôw are ü? 🙃')
for i, token in enumerate(pre_tokens):
    token_bytes = bytes(token)
    print(f'  {i}: {token} -> {token_bytes}')
    
print(f'\nWhat tiktoken expects (first few tokens):')
tiktoken_ids = reference_tokenizer.encode('Héllò hôw are ü? 🙃')
for i, token_id in enumerate(tiktoken_ids[:8]):
    decoded = reference_tokenizer.decode([token_id])
    print(f'  {i}: token_id {token_id} -> {repr(decoded)}')
