from tests.common import gpt2_bytes_to_unicode

# Debug the GPT-2 encoding/decoding process
test_string = "é"
print('Test character:', repr(test_string))
print('UTF-8 bytes:', test_string.encode('utf-8'))

byte_encoder = gpt2_bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Step 1: Convert to GPT-2 encoding
gpt2_encoded = ''.join(byte_encoder[b] for b in test_string.encode('utf-8'))
print('GPT-2 encoded:', repr(gpt2_encoded))

# Step 2: What should the regex match?
import re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
print('Original PAT:', repr(PAT))
matches = re.findall(PAT, gpt2_encoded)
print('Original regex matches:', matches)

# Test the modified pattern
unicode_pattern = PAT.replace('[a-zA-Z]', r'\w')
print('Modified PAT:', repr(unicode_pattern))
unicode_matches = re.findall(unicode_pattern, gpt2_encoded)
print('Modified regex matches:', unicode_matches)

# Step 3: Convert each match back to bytes
print('Converting matches back to bytes:')
for i, match in enumerate(matches):
    print(f'  Match {i}: {repr(match)}')
    byte_list = []
    for char in match:
        if char in byte_decoder:
            byte_val = byte_decoder[char]
            byte_list.append(byte_val)
            print(f'    {repr(char)} -> byte {byte_val}')
        else:
            print(f'    {repr(char)} -> NOT FOUND in decoder')
    print(f'    Final bytes: {byte_list} -> {bytes(byte_list)}')

# Let's also check what each byte maps to in GPT-2 encoding
print(f'\nGPT-2 byte mapping for é (\\xc3\\xa9):')
for byte_val in [0xc3, 0xa9]:
    if byte_val in byte_encoder:
        gpt2_char = byte_encoder[byte_val]
        print(f'  byte {byte_val} (0x{byte_val:02x}) -> {repr(gpt2_char)}')
    else:
        print(f'  byte {byte_val} (0x{byte_val:02x}) -> NOT FOUND')
