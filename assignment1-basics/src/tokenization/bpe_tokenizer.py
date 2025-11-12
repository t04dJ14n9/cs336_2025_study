import os
from typing import Tuple, Dict, List
from .bpe_trainer import PAT
import re
import base64
import json

MERGE_NOT_EXIST = -1

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes]={}, merges: List[Tuple[bytes, bytes]]=[], special_tokens: List[str]|None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_size = len(vocab)

        self.token_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        # create a mapping from two tokens IDs to the ID of the merged token, -1 if it doesn't exist
        self.merge_exist = {}
        for merge in merges:
            left_token_id = self.token_to_id[merge[0]]
            right_token_id = self.token_to_id[merge[1]]
            self.merge_exist[(left_token_id, right_token_id)] = self.token_to_id[merge[0] + merge[1]]
        
    def pre_tokenize(self, text: str) -> List[bytes]:
        """
        pre_tokenize convert text to list of list of UTF-8 bytes
        """
        byte_list = []
        def split_chunk_to_bytes_list(pattern: str, chunk: str) -> List[bytes]:
            # embedded function to convert string to list of UTF-8 bytes
            str_list = re.findall(pattern, chunk)
            return [str.encode() for str in str_list]

        if self.special_tokens is not None: 
            # split on special token, then regex split on patttern
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            for chunk in re.split(pattern, text):
                if chunk == "":
                    continue
                byte_list += split_chunk_to_bytes_list(PAT, chunk)
            return byte_list
        return split_chunk_to_bytes_list(PAT, text)

    def encode(self, text: str) -> List[int]:
        """
        encode convert text to list of token IDs
        """
        token_id_list = []
        pre_token_list = self.pre_tokenize(text)
        # prev_token represent the token examined before, None if not exist
        prev_token: bytes|None = None
        for token in pre_token_list:
            if prev_token == None:
                prev_token = token 
                continue
            # if prev_token exist, examine if they can be merged
            merge_token_id = self.merge_exist.get((self.token_to_id[prev_token], self.token_to_id[token]), MERGE_NOT_EXIST)
            if merge_token_id != MERGE_NOT_EXIST:
                # there is a merge
                prev_token = prev_token + token
                continue
            # merge not exist
            token_id_list.append(self.token_to_id[prev_token])
            prev_token = token
        # end of list, check if there is remaining token unmerged
        if prev_token is not None:
            token_id_list.append(self.token_to_id[prev_token])
        return token_id_list
                
    def decode(self, IDs: List[int]) -> str:
        """
        decode converts list of token IDs back to the original text
        """        
        text = ""
        for ID in IDs:
            text += self.vocab[ID].decode()
        return text

                
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
        # create a mapping from two tokens IDs to the ID of the merged token, -1 if it doesn't exist
        tokenizer.merge_exist = {}
        for merge in tokenizer.merges:
            left_token_id = tokenizer.token_to_id[merge[0]]
            right_token_id = tokenizer.token_to_id[merge[1]]
            tokenizer.merge_exist[(left_token_id,right_token_id)] = tokenizer.token_to_id[merge[0] + merge[1]]
        
        print(f"Tokenizer loaded from {input_path}")
        return tokenizer
