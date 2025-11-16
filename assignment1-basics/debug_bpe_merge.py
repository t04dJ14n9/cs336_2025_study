#!/usr/bin/env python3

# Let's demonstrate why BPE merge operations need bytes objects

print("=== BPE Merge Operation Demo ===")
print()

# Simulate what happens in the _encode_word method
word_as_integers = [116, 101, 115, 116]  # 'test' as integers
word_as_bytes = [b't', b'e', b's', b't']  # 'test' as bytes objects

print(f"Word as integers: {word_as_integers}")
print(f"Word as bytes objects: {word_as_bytes}")
print()

# Let's say we want to merge 't' + 'e' -> 'te'
merge_pair = (b't', b'e')
print(f"Merge pair: {merge_pair}")
print()

print("=== Attempting merge with integers (FAILS) ===")
try:
    # This is what would happen if we stored integers
    if (word_as_integers[0], word_as_integers[1]) == (116, 101):
        # We can't concatenate integers meaningfully for BPE
        merged_integer = word_as_integers[0] + word_as_integers[1]  # 217
        print(f"Integer 'merge': {word_as_integers[0]} + {word_as_integers[1]} = {merged_integer}")
        print("Problem: 217 is not a valid representation of 'te'!")
        print("We lose the ability to represent multi-byte tokens properly.")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== Attempting merge with bytes objects (WORKS) ===")
try:
    # This is what actually happens with bytes objects
    if (word_as_bytes[0], word_as_bytes[1]) == merge_pair:
        merged_bytes = word_as_bytes[0] + word_as_bytes[1]  # b'te'
        print(f"Bytes merge: {word_as_bytes[0]} + {word_as_bytes[1]} = {merged_bytes}")
        print(f"Result type: {type(merged_bytes)}")
        print(f"Can decode back to string: '{merged_bytes.decode('utf-8')}'")
        
        # Show the new word after merge
        new_word = [merged_bytes] + word_as_bytes[2:]
        print(f"New word after merge: {new_word}")
        print(f"Represents: {[token.decode('utf-8') for token in new_word]}")
except Exception as e:
    print(f"Error: {e}")

print()
print("=== Why bytes([byte]) specifically? ===")
print()

# Show different ways to create bytes objects
byte_value = 116  # ASCII value for 't'

print(f"byte_value = {byte_value} (integer)")
print()

# Method 1: bytes([byte]) - what we use
method1 = bytes([byte_value])
print(f"bytes([{byte_value}]) = {method1}")

# Method 2: chr() then encode - alternative but more complex
method2 = chr(byte_value).encode('utf-8')
print(f"chr({byte_value}).encode('utf-8') = {method2}")

# Method 3: Direct bytes creation - doesn't work with single integer
try:
    method3 = bytes(byte_value)
    print(f"bytes({byte_value}) = {method3}")
except Exception as e:
    print(f"bytes({byte_value}) fails: {e}")

print()
print("Summary:")
print("- bytes([integer]) creates a bytes object containing one byte")
print("- This allows BPE merge operations to concatenate bytes properly")
print("- Alternative methods are more complex or don't work")
print("- The result can be used in vocabulary lookups and further merges")
