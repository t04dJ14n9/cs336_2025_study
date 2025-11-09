import re
from src.tokenization.bpe import PAT

def test_pattern_basic():
    """Simple test function similar to Go's test functions"""
    
    # Test basic word matching
    text = "hello world"
    matches = re.findall(PAT, text)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}: {matches}"
    assert matches[0] == "hello", f"Expected 'hello', got '{matches[0]}'"
    assert "world" in matches[1], f"Expected 'world' in second match, got '{matches[1]}'"
    
    # Test contractions
    text = "don't"
    matches = re.findall(PAT, text)
    assert len(matches) == 2, f"Expected 2 matches for contraction, got {len(matches)}: {matches}"
    assert "don" in matches[0], f"Expected 'don' in first match, got '{matches[0]}'"
    assert "'t" in matches[1], f"Expected \"'t\" in second match, got '{matches[1]}'"
    
    # Test numbers and letters
    text = "123abc"
    matches = re.findall(PAT, text)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}: {matches}"
    assert "123" in matches, f"Expected '123' in matches, got {matches}"
    assert "abc" in matches, f"Expected 'abc' in matches, got {matches}"
    
    # Test empty string
    text = ""
    matches = re.findall(PAT, text)
    assert len(matches) == 0, f"Expected 0 matches for empty string, got {len(matches)}: {matches}"
    
    print("✓ All basic tests passed!")

def test_pattern_table_driven():
    """Table-driven tests similar to Go's approach"""
    
    test_cases = [
        ("hello", ["hello"]),
        ("hello world", ["hello", " world"]),
        ("don't", ["don", "'t"]),
        ("I'll", ["I", "'ll"]),
        ("we're", ["we", "'re"]),
        ("123", ["123"]),
        ("abc", ["abc"]),
        ("123abc", ["123", "abc"]),
        ("", []),
        ("   ", ["   "]),
        ("!@#", ["!@#"]),
    ]
    
    for input_text, expected in test_cases:
        matches = re.findall(PAT, input_text)
        assert matches == expected, f"Input: '{input_text}' - Expected: {expected}, Got: {matches}"
    
    print("✓ All table-driven tests passed!")

def test_init():
    pass
def run_all_tests():
    """Run all test functions"""
    try:
        test_pattern_basic()
        test_pattern_table_driven()
        print("\n🎉 All tests passed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()
