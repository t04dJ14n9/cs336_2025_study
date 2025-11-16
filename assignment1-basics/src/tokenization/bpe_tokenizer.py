import os
from typing import Tuple, Dict, List, Iterable
import re
import base64
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes]={}, merges: List[Tuple[bytes, bytes]]=[], special_tokens: List[str]|None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_size = len(vocab)

        self.token_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}

        
    def pre_tokenize(self, text: str) -> List[List[bytes]]:
        """
        pre_tokenize convert text to list of strings, which is a list of UTF-8 bytes
        """
        # split on special tokens before preprocess
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            docs: List[str] = [doc for doc in re.split(pattern, text) if doc]
        else:
            docs: List[str] = [text]
        process_text = []
        for doc in docs:
            words_in_doc = re.findall(PAT, doc)
            for word in words_in_doc:
                byte_list = []
                # byte_value is the integer value for the byte
                for byte_value in word.encode('utf-8'):
                    # append the byte to the list
                    byte_list.append(bytes([byte_value]))
                process_text.append(byte_list)
        return process_text

    def encode(self, text: str) -> List[int]:
        """
        encode convert text to list of token IDs
        """
        token_ids = []
        process_text = self.pre_tokenize(text)
        
        # Apply all merges
        for merge in self.merges:
            # scan through the byte_list to see if there is a match. If there is, merge them.
            for i in range(len(process_text)):
                process_text[i] = self._encode_word(process_text[i], merge)
        
        # all merge completed, now calculate the token IDs
        for word in process_text:
            for token in word:
                token_ids.append(self.token_to_id[token])
        return token_ids
                
    def _encode_word(self, word: List[bytes], attempted_merge: Tuple[bytes, bytes]) -> List[bytes]:
        if len(word) <= 1:
            return word
            
        new_word = []
        i = 0
        while i < len(word):
            # Check if we can merge current and next token
            if i < len(word) - 1 and (word[i], word[i+1]) == attempted_merge:
                # Merge the two tokens
                new_word.append(word[i] + word[i+1])
                i += 2  # Skip the next token since we merged it
            else:
                # No merge, just add the current token
                new_word.append(word[i])
                i += 1
        return new_word
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
                
    def decode(self, IDs: List[int]) -> str:
        """
        decode converts list of token IDs back to the original text
        """        
        byte_list = []
        for ID in IDs:
            byte_list.append(self.vocab[ID])
        return b''.join(byte_list).decode('utf-8')
        

                
    @classmethod
    def load(cls, input_path: str | os.PathLike):
        """
        Load a tokenizer from a JSON file.
        
        Args:
            input_path: Path to the saved tokenizer file
            
        Returns:
            BPE: A BPE tokenizer instance with loaded vocabulary and merges
            
        Example:
            bpe = BPE.load('my_tokenizer.json')
            # Now you can use bpe.encode() or bpe.decode()
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a new BPE instance (dummy input_path since we're loading)
        tokenizer = cls()
        
        # Restore vocab: convert base64 strings back to bytes
        tokenizer.vocab = {
            int(k): base64.b64decode(v.encode('utf-8'))
            for k, v in data['vocab'].items()
        }
        
        # Restore merges: convert base64 strings back to bytes tuples
        tokenizer.merges = [
            (base64.b64decode(t1.encode('utf-8')), base64.b64decode(t2.encode('utf-8')))
            for t1, t2 in data['merges']
        ]
        
        # Restore other attributes
        tokenizer.special_tokens = data.get('special_tokens', [])
        tokenizer.vocab_size = data.get('vocab_size', 256)
        tokenizer.token_to_id = {v: k for k, v in tokenizer.vocab.items()}

        print(f"Tokenizer loaded from {input_path}")
        return tokenizer
