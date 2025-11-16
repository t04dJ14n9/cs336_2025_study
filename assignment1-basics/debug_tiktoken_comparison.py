import tiktoken
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

reference_tokenizer = tiktoken.get_encoding('gpt2')
tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=['<|endoftext|>']
)
test_string = 'Héllò hôw are ü? 🙃'

reference_ids = reference_tokenizer.encode(test_string)
ids = tokenizer.encode(test_string)

print('Test string:', repr(test_string))
print('Reference (tiktoken):', reference_ids)
print('Our tokenizer:', ids)
print()
print('Reference tokens:')
for i, token_id in enumerate(reference_ids):
    print(f'  {i}: {token_id} -> {repr(reference_tokenizer.decode([token_id]))}')
print()
print('Our tokens:')
for i, token_id in enumerate(ids):
    print(f'  {i}: {token_id} -> {repr(tokenizer.decode([token_id]))}')
