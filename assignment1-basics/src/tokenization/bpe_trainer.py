import regex as re
from typing import List, Dict, Tuple
from queue import PriorityQueue
import os
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from multiprocessing import cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class PriorityItem:
    """Wrapper class for priority queue items with custom comparison logic"""
    def __init__(self, count: int, pair: Tuple[int, int], pair_bytes: Tuple[bytes, bytes]):
        self.count = count
        self.pair = pair
        self.pair_bytes = pair_bytes  # Store byte values for comparison
    
    def __lt__(self, other):
        # For max heap behavior: higher count has higher priority (comes first)
        # When counts are equal, prefer lexicographically greater pair (reverse ordering)
        if self.count != other.count:
            return self.count > other.count  # Reverse for max heap
        # Prefer lexicographically greater pair for tie-breaking
        return self.pair_bytes > other.pair_bytes  # Reverse: greater comes first
    
    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair

class Word():
    def __init__(self, raw: str):
        self.raw = raw
        # Convert string to UTF-8 bytes, then to list of byte values (0-255)
        self.token_list = list(raw.encode('utf-8'))
        self.count = 1

class BPETrainer():
    def __init__(self, input_path: str | os.PathLike, vocab_size: int=10000, special_tokens: List[str]=[], num_threads: int | None = None) -> None:
        self.vocab_size = 256 # number of bytes
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.input_path = input_path
        self.vocab: Dict[int, bytes] = {i: i.to_bytes(1, 'big') for i in range(256)}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.loc_map: Dict[Tuple[int, int], List[str]] = {} # maps from adjacent tokens to words that it appears
        self.word_map: Dict[str, Word] = {} # maps from raw string to Word struct
        self.num_threads = num_threads if num_threads is not None else min(cpu_count(), 8)
        # Thread-safety locks for shared data structures
        self._word_map_lock = Lock()
        self._count_map_lock = Lock()
        self._loc_map_lock = Lock()

    def _process_document_chunk(self, doc: str) -> Tuple[Dict[str, Word], Dict[Tuple[int, int], int], Dict[Tuple[int, int], List[str]]]:
        """
        Process a single document chunk and return local word_map, count_map, and loc_map.
        This function is thread-safe as it only works on local data structures.
        """
        local_word_map = {}
        local_count_map = {}
        local_loc_map = {}
        
        words_in_doc = re.findall(PAT, doc)
        for word in words_in_doc:
            if word in local_word_map:
                # Already processed in this chunk, increment its count
                local_word_map[word].count += 1
                # Also increment the pair counts for this word occurrence
                word_instance = local_word_map[word]
                for k in range(len(word_instance.token_list) - 1):
                    pair = (word_instance.token_list[k], word_instance.token_list[k+1])
                    local_count_map[pair] = local_count_map.get(pair, 0) + 1
            else:
                # Create word instance
                word_instance = Word(raw=word)
                local_word_map[word] = word_instance
                for k in range(len(word_instance.token_list) - 1):
                    pair = (word_instance.token_list[k], word_instance.token_list[k+1])
                    local_count_map[pair] = local_count_map.get(pair, 0) + 1
                    if pair not in local_loc_map:
                        local_loc_map[pair] = []
                    local_loc_map[pair].append(word)
        
        return local_word_map, local_count_map, local_loc_map
    
    def _merge_local_results(self, local_word_map: Dict[str, Word], local_count_map: Dict[Tuple[int, int], int], 
                            local_loc_map: Dict[Tuple[int, int], List[str]]):
        """
        Merge local results from a thread into the global data structures.
        This function uses locks to ensure thread-safety.
        """
        # Merge word_map
        with self._word_map_lock:
            for word, word_instance in local_word_map.items():
                if word in self.word_map:
                    self.word_map[word].count += word_instance.count
                else:
                    self.word_map[word] = word_instance
        
        # Merge count_map
        with self._count_map_lock:
            for pair, count in local_count_map.items():
                self.count_map[pair] = self.count_map.get(pair, 0) + count
        
        # Merge loc_map
        with self._loc_map_lock:
            for pair, words in local_loc_map.items():
                if pair not in self.loc_map:
                    self.loc_map[pair] = []
                self.loc_map[pair].extend(words)

    # preprocess separates raw corpus into words for training
    def preprocess(self):
        """
        Preprocess the corpus using multi-threading for improved performance.
        Documents are processed in parallel, and results are merged into global data structures.
        """
        # read corpus
        with open(self.input_path, "r") as f:
            corpus = f.read()
        
        # split on special tokens before preprocess
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.docs: List[str] = [doc for doc in re.split(pattern, corpus) if doc]
        else:
            self.docs: List[str] = [corpus]
        
        # Initialize data structures
        self.count_map: Dict[Tuple[int, int], int] = {}
        self.count_queue: PriorityQueue = PriorityQueue()
        self.loc_map = {}

        # Process documents in parallel using ThreadPoolExecutor
        print(f"Processing {len(self.docs)} documents using {self.num_threads} threads...")
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all documents for processing
            futures = [executor.submit(self._process_document_chunk, doc) for doc in self.docs]
            
            # Collect and merge results as they complete
            for future in as_completed(futures):
                local_word_map, local_count_map, local_loc_map = future.result()
                self._merge_local_results(local_word_map, local_count_map, local_loc_map)
        
        print(f"Processed {len(self.word_map)} unique words")
        
        # Build priority queue from count_map
        for pairs, count in self.count_map.items():
            pair_bytes = (self.vocab[pairs[0]], self.vocab[pairs[1]])
            self.count_queue.put(PriorityItem(count, pairs, pair_bytes))




    # train is the function to train BPE tokenization to preprocessed corpus
    def train(self) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """
        Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, 
        <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
        """
        # Reserve space for special tokens
        max_vocab_size = self.target_vocab_size - len(self.special_tokens)
        while self.vocab_size < max_vocab_size and not self.count_queue.empty():
            item = self.count_queue.get()
            count, (t1, t2) = item.count, item.pair # t1, t2 is the index of the token need to be merged
            if self.count_map.get((t1, t2)) != count:
                # tombstone item
                continue
            self.merge_tokens(t1, t2)
        self.add_special_token_to_vocab()
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
        changed_pairs = {(token1, token2): self.count_map[(token1, token2)]}
        for word_raw in self.loc_map[(token1, token2)]:
            # find the index of (token1, token2)
            word = self.word_map[word_raw]
            old_token_list = word.token_list
            new_token_list = []
            i = 0
            while i < len(old_token_list):
                # Check if we have a matching pair
                if i < len(old_token_list) - 1 and old_token_list[i] == token1 and old_token_list[i+1] == token2:
                    # Decrement the count for the merged pair
                    changed_pairs[(token1, token2)] -= word.count
                    # Handle previous token pair updates
                    if len(new_token_list) > 0:
                        # update (previous_token, token1)
                        prev_pair = (new_token_list[-1], token1)
                        if prev_pair not in changed_pairs:
                            changed_pairs[prev_pair] = self.count_map[prev_pair]
                        changed_pairs[prev_pair] -= word.count
                        
                        # insert new prev pair
                        new_prev_pair = (new_token_list[-1], new_token_index)
                        if new_prev_pair not in changed_pairs:
                            changed_pairs[new_prev_pair] = 0
                        changed_pairs[new_prev_pair] += word.count
                        if new_prev_pair not in self.loc_map:
                            self.loc_map[new_prev_pair] = []
                        self.loc_map[new_prev_pair].append(word_raw)
                    
                    # Handle next token pair updates
                    if i + 2 < len(old_token_list):
                        # update (token2, latter_token)
                        latter_pair = (token2, old_token_list[i+2])
                        if latter_pair not in changed_pairs:
                            changed_pairs[latter_pair] = self.count_map[latter_pair]
                        changed_pairs[latter_pair] -= word.count
                        # insert new later pair
                        new_latter_pair = (new_token_index, old_token_list[i+2])
                        if new_latter_pair not in changed_pairs:
                            changed_pairs[new_latter_pair] = 0
                        changed_pairs[new_latter_pair] += word.count
                        if new_latter_pair not in self.loc_map:
                            self.loc_map[new_latter_pair] = []
                        self.loc_map[new_latter_pair].append(word_raw)
                    
                    # Add the merged token
                    new_token_list.append(new_token_index)
                    i += 2  # Skip both tokens that were merged
                else:
                    # No match, just copy the token
                    new_token_list.append(old_token_list[i])
                    i += 1
            
            # Update word's token list
            word.token_list = new_token_list
        
        # Increment vocab_size for the newly created token
        self.vocab_size += 1
        self.apply_token_count_change(changed_pairs)

    def apply_token_count_change(self, changed_pairs: Dict[Tuple[int, int], int]):
        """
        the count of tokens in changed_pairs has been altered due to merge action, update internal states
        """
        for pair, new_count in changed_pairs.items():
            self.count_map[pair] = new_count
            # Only add pairs with positive counts to avoid infinite queue growth
            if new_count > 0:
                pair_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                self.count_queue.put(PriorityItem(new_count, pair, pair_bytes))
         
    def add_special_token_to_vocab(self):
        for special_token in self.special_tokens:
            self.vocab[self.vocab_size] = special_token.encode('utf-8')
            self.vocab_size += 1

    def serialize(self):
        """
        Serialize the tokenizer to a dictionary format.
        Converts bytes to base64 strings for JSON compatibility.
        """
        # Convert vocab: bytes values to base64 strings
        vocab_serialized = {
            str(k): base64.b64encode(v).decode('utf-8') 
            for k, v in self.vocab.items()
        }
        
        # Convert merges: list of (bytes, bytes) tuples to list of [base64_str, base64_str]
        merges_serialized = [
            [base64.b64encode(t1).decode('utf-8'), base64.b64encode(t2).decode('utf-8')]
            for t1, t2 in self.merges
        ]
        
        return {
            'vocab': vocab_serialized,
            'merges': merges_serialized,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'target_vocab_size': self.target_vocab_size
        }
    
    def save(self, output_path: str | os.PathLike):
        """
        Save the tokenizer to a JSON file.
        
        Args:
            output_path: Path where the tokenizer will be saved (e.g., 'tokenizer.json')
        
        Example:
            bpe = BPE('corpus.txt', vocab_size=10000)
            bpe.preprocess()
            bpe.train()
            bpe.save('my_tokenizer.json')
        """
        serialized_data = self.serialize()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {output_path}")
    
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
        tokenizer = cls(input_path='', vocab_size=data.get('target_vocab_size', 10000))
        
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
        tokenizer.target_vocab_size = data.get('target_vocab_size', 10000)
        
        print(f"Tokenizer loaded from {input_path}")
        return tokenizer
