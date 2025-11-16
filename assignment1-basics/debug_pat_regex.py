import re
from src.tokenization.bpe_trainer import PAT

test_string = "Héllò hôw are ü? 🙃"
print('Original string:', repr(test_string))
print('PAT pattern:', repr(PAT))

matches = re.findall(PAT, test_string)
print('Regex matches:', matches)

# Let's see what characters are being missed
matched_chars = ''.join(matches)
print('Matched characters:', repr(matched_chars))
print('Missing characters:', repr(''.join(c for c in test_string if c not in matched_chars)))
