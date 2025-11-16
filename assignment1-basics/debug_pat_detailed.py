import re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

test_string = 'HÃ©llÃ²ĠhÃ´wĠareĠÃ¼?ĠðŁĻĥ'
print('GPT-2 encoded string:', repr(test_string))
print('PAT pattern:', repr(PAT))

# Test each part of the regex separately
patterns = [
    r"'(?:[sdmt]|ll|ve|re)",  # Contractions
    r" ?[a-zA-Z]+",           # Optional space + ASCII letters
    r" ?[0-9]+",              # Optional space + digits
    r" ?[^\s\w]+",            # Optional space + non-word, non-space chars
    r"\s+(?!\S)",             # Whitespace not followed by non-whitespace
    r"\s+"                    # Whitespace
]

pattern_names = [
    "Contractions",
    "ASCII letters", 
    "Digits",
    "Non-word chars",
    "Whitespace (special)",
    "Whitespace"
]

print('\nTesting each pattern part:')
for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
    matches = re.findall(pattern, test_string)
    print(f'{i+1}. {name}: {matches}')

print('\nFull pattern matches:', re.findall(PAT, test_string))

# Let's see what characters are not being matched
full_matches = re.findall(PAT, test_string)
matched_text = ''.join(full_matches)
print(f'\nMatched text: {repr(matched_text)}')
print(f'Original text: {repr(test_string)}')
print(f'Missing chars: {repr("".join(c for c in test_string if c not in matched_text))}')
