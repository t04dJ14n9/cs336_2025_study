# Python Testing Approaches (Similar to Go Testing)

This directory contains several approaches to testing your regex pattern, similar to Go's testing style.

## Quick Start

### Simple Assert-Based Tests (Most Go-like)

```bash
python3 bpe_simple_test.py
```

### Unittest Framework (Python standard)

```bash
python3 bpe_test.py
```

### Test Runner (Similar to `go test`)

```bash
python3 test_runner.py              # Run simple tests
python3 test_runner.py simple       # Run simple tests
python3 test_runner.py unittest     # Run unittest tests
python3 test_runner.py all          # Run both
```

## Testing Approaches Explained

### 1. Simple Assert-Based Tests (`bpe_simple_test.py`)

This is the most similar to Go's testing approach:

```python
def test_pattern_basic():
    text = "hello world"
    matches = re.findall(PAT, text)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
    assert matches[0] == "hello", f"Expected 'hello', got '{matches[0]}'"
```

**Pros:**

- Very similar to Go's testing style
- Simple and straightforward
- No external dependencies
- Easy to understand for Go developers

**Cons:**

- Less detailed output on failures
- No test discovery
- Manual test organization

### 2. Unittest Framework (`bpe_test.py`)

Python's built-in testing framework:

```python
class TestBPEPattern(unittest.TestCase):
    def test_pattern_matching(self):
        test_cases = [
            ("hello world", ["hello", " world"]),
            ("don't", ["don", "'t"]),
        ]

        for input_str, expected in test_cases:
            matches = re.findall(PAT, input_str)
            self.assertEqual(matches, expected)
```

**Pros:**

- Built into Python standard library
- Detailed test output and failure messages
- Test discovery and organization
- Similar to Go's testing framework structure

**Cons:**

- More verbose than simple asserts
- Requires understanding of unittest conventions

### 3. Test Runner (`test_runner.py`)

Custom test runner similar to `go test`:

```bash
# Run all tests
python3 test_runner.py all

# Run specific test file pattern
python3 test_runner.py *simple*
```

## Understanding the Regex Pattern

The pattern `PAT` in `bpe.py` is designed for tokenization:

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
```

This pattern matches:

1. Contractions: `'s`, `'d`, `'m`, `'t`, `'ll`, `'ve`, `'re`
2. Words: `[a-zA-Z]+`
3. Numbers: `[0-9]+`
4. Special characters: `[^\s\w]+`
5. Whitespace patterns

## Adding New Tests

### For Simple Tests:

Add functions to `bpe_simple_test.py`:

```python
def test_my_new_case():
    text = "your test input"
    matches = re.findall(PAT, text)
    assert matches == ["expected", "result"]
    print("✓ My new test passed!")
```

### For Unittest:

Add methods to `TestBPEPattern` class in `bpe_test.py`:

```python
def test_my_new_case(self):
    matches = re.findall(PAT, "your test input")
    self.assertEqual(matches, ["expected", "result"])
```

## Go vs Python Testing Comparison

| Go                      | Python Equivalent        |
| ----------------------- | ------------------------ |
| `go test`               | `python3 test_runner.py` |
| `testing.T`             | `unittest.TestCase`      |
| `t.Errorf()`            | `self.assertEqual()`     |
| Table-driven tests      | List of tuples in loop   |
| `t.Run()`               | `self.subTest()`         |
| Test files: `*_test.go` | `*_test.py`              |

## Recommendations

1. **For quick testing**: Use `bpe_simple_test.py` (most Go-like)
2. **For comprehensive testing**: Use `bpe_test.py` with unittest
3. **For CI/CD**: Use `python3 -m unittest discover` or `pytest`

The simple assert-based approach will feel most familiar as a Go developer, while unittest provides more Python-idiomatic testing features.
