#!/usr/bin/env python3

# Let's demonstrate why bytes([byte]) is needed

text = "test"
print(f"Original text: '{text}'")
print()

# When we encode a string to UTF-8, we get bytes
encoded_bytes = text.encode('utf-8')
print(f"text.encode('utf-8') = {encoded_bytes}")
print(f"Type: {type(encoded_bytes)}")
print()

# When we iterate over bytes, we get integers (0-255)
print("Iterating over bytes gives us integers:")
for i, byte in enumerate(encoded_bytes):
    print(f"  byte[{i}] = {byte} (type: {type(byte)})")
print()

# The BPE algorithm needs each byte as a separate bytes object, not as integers
print("What we need for BPE (each byte as a bytes object):")
byte_list = []
for byte in encoded_bytes:
    byte_obj = bytes([byte])  # Convert integer back to bytes object
    byte_list.append(byte_obj)
    print(f"  bytes([{byte}]) = {byte_obj} (type: {type(byte_obj)})")

print()
print(f"Final byte_list: {byte_list}")
print()

# Let's show what happens if we don't use bytes([byte])
print("What happens if we try to use integers directly:")
try:
    # This would fail in the BPE merge operations because we need bytes objects
    integer_list = []
    for byte in encoded_bytes:
        integer_list.append(byte)  # Just the integer
    print(f"Integer list: {integer_list}")
    
    # Try to concatenate integers (this won't work for BPE merging)
    print("Trying to 'merge' integers 116 + 101:")
    print(f"116 + 101 = {116 + 101} (arithmetic addition, not byte concatenation)")
    
    # But with bytes objects, we can concatenate properly
    print("With bytes objects:")
    print(f"b't' + b'e' = {b't' + b'e'}")
    
except Exception as e:
    print(f"Error: {e}")

print()
print("Summary:")
print("- text.encode('utf-8') returns a bytes object")
print("- Iterating over bytes gives integers (0-255)")
print("- BPE needs each byte as a separate bytes object for merging")
print("- bytes([integer]) converts an integer back to a single-byte bytes object")
print("- This allows proper concatenation during BPE merge operations")
