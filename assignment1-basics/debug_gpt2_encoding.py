from tests.common import gpt2_bytes_to_unicode
import re
from src.tokenization.bpe_trainer import PAT

# Test the GPT-2 encoding approach
test_string = "Héllò hôw are ü? 🙃"
print('Original string:', repr(test_string))
print('UTF-8 bytes:', test_string.encode('utf-8'))

# Create the GPT-2 byte-to-unicode mapping
byte_encoder = gpt2_bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Convert text to GPT-2 Unicode encoding
gpt2_encoded = ''.join(byte_encoder[b] for b in test_string.encode('utf-8'))
print('GPT-2 encoded:', repr(gpt2_encoded))

# Apply regex to GPT-2 encoded text
print('PAT pattern:', repr(PAT))
str_list = re.findall(PAT, gpt2_encoded)
print('Regex matches on GPT-2 encoded:', str_list)

# Convert back to bytes
print('Converting back to bytes:')
for i, gpt2_str in enumerate(str_list):
    byte_in_str = []
    for char in gpt2_str:
        if char in byte_decoder:
            byte_in_str.append(byte_decoder[char])
    print(f'  {i}: {repr(gpt2_str)} -> {byte_in_str} -> {bytes(byte_in_str)}')
