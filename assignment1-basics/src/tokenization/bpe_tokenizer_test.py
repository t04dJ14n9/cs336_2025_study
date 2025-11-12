import unittest
from .bpe_tokenizer import PAT, BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_pre_tokenize(self):
        bpe_tokenizer = BPETokenizer(special_tokens=['<special token>'])
        bytes_list = bpe_tokenizer.pre_tokenize("te st <special token> asd")
        self.assertEqual(bytes_list, [b'te', b' st', b' ', b' asd'])
    def test_encode(self):
        bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')
        res = bpe_tokenizer.encode("test villager")
        print(f'res={res}')
        self.assertEqual(res, [13213, 13412])
    def test_decode(self):
        bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')
        res = bpe_tokenizer.decode([13213, 13412])
        self.assertEqual(res, "test villager")

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
