import sys
sys.path.append('.')

from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH

tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path=VOCAB_PATH,
    merges_path=MERGES_PATH,
    special_tokens=['<|endoftext|>', '<|endoftext|><|endoftext|>'],
)

test_string = 'Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>'
print('Test string:', test_string)
print('Pre-tokenize result:', tokenizer.pre_tokenize(test_string))

print('\nSpecial tokens in vocab:')
for token_id, token_bytes in tokenizer.vocab.items():
    try:
        decoded = token_bytes.decode('utf-8')
        if '<|endoftext|>' in decoded:
            print(f'  {token_id}: {repr(decoded)}')
    except:
        pass

print('\nEncoding result:')
ids = tokenizer.encode(test_string)
print('IDs:', ids)

print('\nDecoding each token:')
tokenized_string = []
for i, token_id in enumerate(ids):
    decoded_token = tokenizer.decode([token_id])
    tokenized_string.append(decoded_token)
    print(f'  {i}: {token_id} -> {repr(decoded_token)}')

print('\nFull tokenized string:', tokenized_string)
print('Count of <|endoftext|>:', tokenized_string.count('<|endoftext|>'))
print('Count of <|endoftext|><|endoftext|>:', tokenized_string.count('<|endoftext|><|endoftext|>'))
