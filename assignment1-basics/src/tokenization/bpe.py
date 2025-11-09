import re
from typing import List, Dict, Tuple
from queue import PriorityQueue
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

class PriorityItem:
    """Wrapper class for priority queue items with custom comparison logic"""
    def __init__(self, count: int, pair: Tuple[int, int]):
        self.count = count
        self.pair = pair
    
    def __lt__(self, other):
        # For max heap behavior: higher count has higher priority (comes first)
        # When counts are equal, use lexicographic ordering (smaller pair comes first)
        if self.count != other.count:
            return self.count > other.count  # Reverse for max heap
        return self.pair < other.pair  # Lexicographic ordering when counts are equal
    
    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair

class Word():
    def __init__(self, raw: str):
        self.raw = raw
        self.token_list = [ord(c) for c in raw]
        self.count = 1

class BPE():
    def __init__(self, input_path: str | os.PathLike, vocab_size: int=10000, special_tokens: List[str]=[]) -> None:
        self.vocab_size = 256 # number of bytes
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.input_path = input_path
        self.vocab: Dict[int, bytes] = {i: i.to_bytes(1, 'big') for i in range(256)}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.loc_map: Dict[Tuple[int, int], List[str]] = {} # maps from adjacent tokens to words that it appears
        self.word_map: Dict[str, Word] = {} # maps from raw string to Word struct

    # preprocess separates raw corpus into words for training
    def preprocess(self):
        # read corpus
        with open(self.input_path, "r") as f:
            corpus = f.read()
        # TODO optimize using multi threading
        # split on special tokens before preprocess
        self.docs: List[str] = re.split("|".join(self.special_tokens), corpus)
        self.count_map: Dict[Tuple[int, int], int] = {} # count_map maps adjacent token to their count
        self.count_queue: PriorityQueue = PriorityQueue()
        self.loc_map = {}

        # preprocess corpus, store the count of each adjacent pairs in corpus in count_map
        for i in range(len(self.docs)):
            doc = self.docs[i]
            words_in_doc = re.findall(PAT, doc)
            for j in range(len(words_in_doc)):
                word = words_in_doc[j]
                if word in self.word_map: # already processed, increment its count
                    self.word_map[word].count += 1
                    continue
                # create word instance
                word_instance = Word(raw=word)
                self.word_map[word] = word_instance
                for k in range(len(word_instance.token_list) - 1):
                    pair = (word_instance.token_list[k], word_instance.token_list[k+1])
                    self.count_map[pair] = self.count_map.get(pair, 0) + 1
                    if pair not in self.loc_map:
                        self.loc_map[pair] = []
                    self.loc_map[pair].append(word)


        for pairs, count in self.count_map.items():
            self.count_queue.put(PriorityItem(count, pairs)) # priority queue will pop maximum item first




    # train is the function to train BPE tokenization to preprocessed corpus
    def train(self) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, 
        <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        """
        while self.vocab_size < self.target_vocab_size and not self.count_queue.empty():
            item = self.count_queue.get()
            count, (t1, t2) = item.count, item.pair # t1, t2 is the index of the token need to be merged
            if self.count_map.get((t1, t2)) != count:
                # tombstone item
                continue
            self.merge_tokens(t1, t2)
        return self.vocab, self.merges
            
    def merge_tokens(self, token1: int, token2: int):
        """
        merge token1 and token2 into a new token
        
        """
        self.merges.append((self.vocab[token1], self.vocab[token2]))

        # assign a new number to the merged token
        new_token_index = self.vocab_size
        self.vocab[new_token_index] = self.vocab[token1] + self.vocab[token2]

        # update the count of tokens
        changed_pairs = {(token1, token2): 0}
        for word_raw in self.loc_map[(token1, token2)]:
            # find the index of (token1, token2)
            word = self.word_map[word_raw]
            for i in range(len(word.token_list)-1):
                if word.token_list[i] == token1 and word.token_list[i+1] == token2:
                    if i-1 >= 0:
                        # update (previous_token, token1)
                        prev_pair = (word.token_list[i-1], token1)
                        if prev_pair not in changed_pairs:
                            changed_pairs[prev_pair] = self.count_map.pop(prev_pair) # remove merged token in count_map
                        changed_pairs[prev_pair] -= word.count
                        
                        # insert new prev pair
                        new_prev_pair = (word.token_list[i-1], new_token_index)
                        if new_prev_pair not in changed_pairs:
                            changed_pairs[new_prev_pair] = 0
                        changed_pairs[new_prev_pair] += word.count
                        if new_prev_pair not in self.loc_map:
                            self.loc_map[new_prev_pair] = []
                        self.loc_map[new_prev_pair].append(word_raw)
                    if i+2 < len(word.token_list):
                        # update (token2, latter_token)
                        latter_pair = (token2, word.token_list[i+2])
                        if latter_pair not in changed_pairs:
                            changed_pairs[latter_pair] = self.count_map.pop(latter_pair)
                        changed_pairs[latter_pair] -= word.count
                        # insert new later pair
                        new_latter_pair = (new_token_index, word.token_list[i+2])
                        if new_latter_pair not in changed_pairs:
                            changed_pairs[new_latter_pair] = 0
                        changed_pairs[new_latter_pair] += word.count
                        if new_latter_pair not in self.loc_map:
                            self.loc_map[new_latter_pair] = []
                        self.loc_map[new_latter_pair].append(word_raw)
                    # update word's token list
                    word.token_list = word.token_list[:i] + [new_token_index] + word.token_list[i+2:]
        self.apply_token_count_change(changed_pairs)

    def apply_token_count_change(self, changed_pairs: Dict[Tuple[int, int], int]):
        """
        the count of tokens in changed_pairs has been altered due to merge action, update internal states
        """
        for pair, new_count in changed_pairs.items():
            self.count_map[pair] = new_count
            self.count_queue.put(PriorityItem(new_count, pair))
         
