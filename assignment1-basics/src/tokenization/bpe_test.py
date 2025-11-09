import re
import unittest
from .bpe import PAT, BPE

class TestBPEPattern(unittest.TestCase):
    """Test cases for the BPE regex pattern, similar to Go's table-driven tests"""
    
    def test_pattern_matching(self):
        """Test the regex pattern against various input strings"""
        test_cases = [
            # (input_string, expected_matches, description)
            ("hello world", ["hello", " world"], "Basic words with space"),
            ("don't", ["don", "'t"], "Contraction with apostrophe"),
            ("123abc", ["123", "abc"], "Numbers and letters"),
            ("hello123world", ["hello", "123", "world"], "Mixed alphanumeric - splits on number boundaries"),
            ("   spaces   ", ["  ", " spaces", "   "], "Multiple spaces - complex splitting"),
            ("hello   world", ["hello", "  ", " world"], "Multiple spaces between words"),
            ("test@email.com", ["test", "@", "email", ".", "com"], "Email-like string - splits on special chars"),
            ("it's", ["it", "'s"], "Contraction 's"),
            ("I'll", ["I", "'ll"], "Contraction 'll"),
            ("we're", ["we", "'re"], "Contraction 're"),
            ("I've", ["I", "'ve"], "Contraction 've"),
            ("can't", ["can", "'t"], "Contraction 't"),
            ("", [], "Empty string"),
            ("   ", ["   "], "Only spaces"),
            ("!@#$%", ["!@#$%"], "Only special characters"),
        ]
        
        for input_str, expected, description in test_cases:
            with self.subTest(input=input_str, description=description):
                matches = re.findall(PAT, input_str)
                self.assertEqual(matches, expected, f"Failed for input '{input_str}': expected {expected}, got {matches}")
    
    def test_pre_processing(self):
        bpe =  BPE('/data/workspace/Code/cs336/assignment1-basics/src/tokenization/corpus_test.txt', special_tokens=['<|endoftext|>'])
        bpe.preprocess()
        
        self.assertEqual(bpe.word_map['low'].count,1, 'count of low is 1')
        self.assertEqual(bpe.word_map[' low'].count,4, 'count of  low is 4')
        self.assertEqual(bpe.word_map[' lower'].count, 2, 'count of lower is 2')
        self.assertEqual(bpe.loc_map[(ord('l'), ord('o'))], ['low', ' low', ' lower'])
        self.assertEqual(bpe.count_map[(ord(' '), ord('l'))], 6)

        vocab, merges = bpe.train()
        self.assertEqual(vocab[256], b'es')
if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
