import os
from typing import Tuple, Dict, List, Iterable, BinaryIO
import regex as re
import base64
import json
import time
from threading import Thread
from queue import Queue
from multiprocessing import cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    
    Args:
        file: Binary file handle opened in 'rb' mode
        desired_num_chunks: Target number of chunks to create
        split_special_token: Special token (as bytes) to use as chunk boundary
        
    Returns:
        List of byte positions representing chunk boundaries (including 0 and file_size)
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk_worker(job_queue: Queue, result_queue: Queue, tokenizer_state: Tuple):
    """
    Worker goroutine (thread) for parallel processing of file chunks.
    Mimics Go's goroutine pattern: reads jobs from a channel (queue),
    processes them, and sends results to another channel.
    
    Args:
        job_queue: Queue containing (job_id, file_path, start, end) tuples
        result_queue: Queue to send (job_id, result) tuples
        tokenizer_state: Tuple of (vocab, merges, special_tokens)
    """
    # Reconstruct tokenizer from state (done once per goroutine)
    vocab, merges, special_tokens = tokenizer_state
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # Keep processing jobs until receiving None (sentinel value)
    while True:
        job = job_queue.get()
        if job is None:  # Sentinel value to stop the goroutine
            job_queue.task_done()
            break
        
        job_id, file_path, start, end = job
        
        try:
            # Read the chunk
            with open(file_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
            
            # Pre-tokenize the chunk
            result = tokenizer.pre_tokenize(chunk)
            result_queue.put((job_id, result))
        except Exception as e:
            # Send error result
            result_queue.put((job_id, None, str(e)))
        finally:
            job_queue.task_done()


def _process_text_chunk_worker(job_queue: Queue, result_queue: Queue, tokenizer_state: Tuple):
    """
    Worker goroutine (thread) for parallel processing of text chunks.
    Mimics Go's goroutine pattern with channels (queues).
    
    Args:
        job_queue: Queue containing (job_id, text_chunk) tuples
        result_queue: Queue to send (job_id, result) tuples
        tokenizer_state: Tuple of (vocab, merges, special_tokens)
    """
    # Reconstruct tokenizer from state (done once per goroutine)
    vocab, merges, special_tokens = tokenizer_state
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # Keep processing jobs until receiving None (sentinel value)
    while True:
        job = job_queue.get()
        if job is None:  # Sentinel value to stop the goroutine
            job_queue.task_done()
            break
        
        job_id, text_chunk = job
        
        try:
            # Pre-tokenize the chunk
            result = tokenizer.pre_tokenize(text_chunk)
            result_queue.put((job_id, result))
        except Exception as e:
            # Send error result
            result_queue.put((job_id, None, str(e)))
        finally:
            job_queue.task_done()


class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes]={}, merges: List[Tuple[bytes, bytes]]=[], special_tokens: List[str]|None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_size = len(vocab)
        if special_tokens is not None and len(special_tokens) > 0:
            for special_token in special_tokens:
                # print(special_token)
                if special_token.encode("utf-8") not in vocab.values():
                    self.vocab[self.vocab_size] = special_token.encode("utf-8")
                    self.vocab_size += 1

        self.token_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        
        # Build merge priority map for efficient lookup
        # Lower priority value means the merge should be applied earlier
        self.merge_priority: Dict[Tuple[bytes, bytes], int] = {
            merge: idx for idx, merge in enumerate(merges)
        }

        
    def pre_tokenize(self, text: str) -> List[List[bytes]]:
        """
        pre_tokenize convert text to list of strings, which is a list of UTF-8 bytes
        """
        # split on special tokens before preprocess
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens correctly
            # this makes sure longer special tokens are matched before shorter ones.
            # also notice that capturing group are used, so the special tokens are captured as groups.
            # example: 
            # With capturing group:
            # re.split("(<|endoftext|>)", "Hello <|endoftext|> World") 
            # Returns: ["Hello ", "<|endoftext|>", " World"]  # Special token preserved! 
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(token) for token in special_tokens_sorted) + ")"
            docs: List[str] = [doc for doc in re.split(pattern, text) if doc]
        else:
            docs: List[str] = [text]
        
        process_text = []
        special_tokens_set = set(self.special_tokens) if self.special_tokens else set()
        
        for doc in docs:
            # since capturing group is used, the special tokens are captured as groups.
            # requires atomic treatment to handle special tokens
            if doc in special_tokens_set:
                # This is a special token, treat it as a single token
                byte_list = [doc.encode('utf-8')]
                process_text.append(byte_list)
            else:
                # Regular text, apply pre-tokenization pattern
                words_in_doc = re.findall(PAT, doc)
                for word in words_in_doc:
                    byte_list = []
                    # byte_value is the integer value for the byte
                    for byte_value in word.encode('utf-8'):
                        # append the byte to the list
                        byte_list.append(bytes([byte_value]))
                    process_text.append(byte_list)
        return process_text

    def pre_tokenize_parallel(
        self,
        text: str,
        num_processes: int = None,
        split_special_token: str = "<|endoftext|>",
        min_chunk_size: int = 10000
    ) -> List[List[bytes]]:
        """
        Pre-tokenize text using parallel processing.
        The text is split into chunks at special token boundaries, and each chunk
        is processed in parallel.
        
        Args:
            text: Text to pre-tokenize
            num_processes: Number of parallel processes to use (default: CPU count)
            split_special_token: Special token to use as chunk boundary
            min_chunk_size: Minimum characters per chunk (default: 10000)
            
        Returns:
            List of pre-tokenized words (each word is a list of bytes)
        """
        # For small texts, use sequential processing to avoid overhead
        if len(text) < min_chunk_size * 2:
            return self.pre_tokenize(text)
        
        if num_processes is None:
            num_processes = cpu_count()
        
        # Use regex to split while preserving the special token
        # This creates a pattern that captures the special token
        pattern = f"({re.escape(split_special_token)})"
        parts = re.split(pattern, text)
        
        # Reconstruct chunks with special tokens
        chunks = []
        for i in range(0, len(parts)):
            if parts[i]:  # Skip empty strings
                chunks.append(parts[i])
        
        # Calculate approximate chunk size
        total_chars = sum(len(chunk) for chunk in chunks)
        target_chunk_size = total_chars // num_processes
        
        # Group chunks to achieve target size while preserving special tokens
        grouped_chunks = []
        current_group = []
        current_size = 0
        
        for chunk in chunks:
            current_group.append(chunk)
            current_size += len(chunk)
            
            # Create a new group if we've reached target size and haven't reached max groups
            if current_size >= target_chunk_size and len(grouped_chunks) < num_processes - 1:
                grouped_chunks.append("".join(current_group))
                current_group = []
                current_size = 0
        
        # Add remaining chunks
        if current_group:
            grouped_chunks.append("".join(current_group))
        
        # If we have fewer chunks than processes, just use sequential processing
        if len(grouped_chunks) <= 1:
            return self.pre_tokenize(text)
        
        # Go-style concurrency: Create channels (queues) for communication
        job_queue = Queue()
        result_queue = Queue()
        
        # Prepare tokenizer state for workers
        tokenizer_state = (self.vocab, self.merges, self.special_tokens)
        
        # Launch goroutines (threads) - similar to Go's "go func()"
        num_workers = min(num_processes, len(grouped_chunks))
        workers = []
        for _ in range(num_workers):
            worker = Thread(
                target=_process_text_chunk_worker,
                args=(job_queue, result_queue, tokenizer_state)
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Send jobs to the job channel (queue)
        for job_id, chunk in enumerate(grouped_chunks):
            job_queue.put((job_id, chunk))
        
        # Send sentinel values to stop workers (like closing a channel in Go)
        for _ in range(num_workers):
            job_queue.put(None)
        
        # Collect results from the result channel (queue)
        # Store results with their job_id to maintain order
        results = {}
        for _ in range(len(grouped_chunks)):
            job_id, result = result_queue.get()
            results[job_id] = result
        
        # Wait for all goroutines to finish
        for worker in workers:
            worker.join()
        
        # Flatten results in order
        all_pre_tokens = []
        for job_id in sorted(results.keys()):
            all_pre_tokens.extend(results[job_id])
        
        return all_pre_tokens
    
    def encode(self, text: str, use_parallel: bool = False, num_processes: int = 0) -> List[int]:
        """
        encode convert text to list of token IDs
        
        Args:
            text: Text to encode
            use_parallel: Whether to use parallel pre-tokenization (default: False)
            num_processes: Number of parallel processes (only used if use_parallel=True)
            
        Returns:
            List of token IDs
        """
        token_ids = []
        
        # Use parallel or sequential pre-tokenization
        if use_parallel:
            process_text = self.pre_tokenize_parallel(text, num_processes=num_processes)
        else:
            process_text = self.pre_tokenize(text)

        print(f'{time.time()}: Pre-tokenization completed')
        
        # Apply merges efficiently using priority-based approach
        for i in range(len(process_text)):
            if i % 50000 == 0:
                print(f'{time.time()}: Encoding word {i}')
            process_text[i] = self._encode_word_optimized(process_text[i])
        
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
    
    def _encode_word_optimized(self, word: List[bytes]) -> List[bytes]:
        """
        Optimized word encoding using priority-based merging.
        Instead of iterating through all merges, we find possible merges
        and apply them in priority order.
        """
        if len(word) <= 1:
            return word
        
        # Keep merging until no more merges are possible
        while len(word) > 1:
            # Find all possible merge pairs in the current word with their priorities
            possible_merges = {}
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    # Store the position with the lowest priority (earliest merge)
                    if pair not in possible_merges or priority < possible_merges[pair][0]:
                        possible_merges[pair] = (priority, i)
            
            # If no merges are possible, we're done
            if not possible_merges:
                break
            
            # Find the merge with the highest priority (lowest priority value)
            best_pair = min(possible_merges.items(), key=lambda x: x[1][0])
            merge_pair = best_pair[0]
            merge_pos = best_pair[1][1]
            
            # Apply the best merge at its position
            new_word = []
            i = 0
            while i < len(word):
                if i == merge_pos:
                    # Merge at this position
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        return word
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def pre_tokenize_file_parallel(
        self, 
        file_path: str | os.PathLike, 
        num_processes: int = None,
        split_special_token: str = "<|endoftext|>"
    ) -> List[List[bytes]]:
        """
        Pre-tokenize a large file using parallel processing.
        The file is split into chunks at special token boundaries, and each chunk
        is processed in parallel.
        
        Args:
            file_path: Path to the file to pre-tokenize
            num_processes: Number of parallel processes to use (default: CPU count)
            split_special_token: Special token to use as chunk boundary
            
        Returns:
            List of pre-tokenized words (each word is a list of bytes)
            
        Example:
            tokenizer = BPETokenizer.load('tokenizer.json')
            pre_tokens = tokenizer.pre_tokenize_file_parallel('large_file.txt', num_processes=4)
        """
        if num_processes is None:
            num_processes = cpu_count()
        
        # Find chunk boundaries
        with open(file_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token.encode('utf-8'))
        
        # Go-style concurrency: Create channels (queues) for communication
        job_queue = Queue()
        result_queue = Queue()
        
        # Prepare tokenizer state for workers
        tokenizer_state = (self.vocab, self.merges, self.special_tokens)
        
        # Calculate number of chunks
        chunks = list(zip(boundaries[:-1], boundaries[1:]))
        num_chunks = len(chunks)
        
        # Launch goroutines (threads) - similar to Go's "go func()"
        num_workers = min(num_processes, num_chunks)
        workers = []
        for _ in range(num_workers):
            worker = Thread(
                target=_process_chunk_worker,
                args=(job_queue, result_queue, tokenizer_state)
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Send jobs to the job channel (queue)
        for job_id, (start, end) in enumerate(chunks):
            job_queue.put((job_id, file_path, start, end))
        
        # Send sentinel values to stop workers (like closing a channel in Go)
        for _ in range(num_workers):
            job_queue.put(None)
        
        # Collect results from the result channel (queue)
        # Store results with their job_id to maintain order
        results = {}
        for _ in range(num_chunks):
            job_id, result = result_queue.get()
            results[job_id] = result
        
        # Wait for all goroutines to finish
        for worker in workers:
            worker.join()
        
        # Flatten results in order
        all_pre_tokens = []
        for job_id in sorted(results.keys()):
            all_pre_tokens.extend(results[job_id])
        
        return all_pre_tokens
    
    def encode_file_parallel(
        self,
        file_path: str | os.PathLike,
        num_processes: int = None,
        split_special_token: str = "<|endoftext|>"
    ) -> List[int]:
        """
        Encode a large file using parallel pre-tokenization.
        
        Args:
            file_path: Path to the file to encode
            num_processes: Number of parallel processes to use (default: CPU count)
            split_special_token: Special token to use as chunk boundary
            
        Returns:
            List of token IDs
            
        Example:
            tokenizer = BPETokenizer.load('tokenizer.json')
            token_ids = tokenizer.encode_file_parallel('large_file.txt', num_processes=4)
        """
        # Pre-tokenize in parallel
        process_text = self.pre_tokenize_file_parallel(file_path, num_processes, split_special_token)
        
        # Apply merges (this part is still sequential, but pre-tokenization is the bottleneck)
        for i in range(len(process_text)):
            process_text[i] = self._encode_word_optimized(process_text[i])
        
        # Convert to token IDs
        token_ids = []
        for word in process_text:
            for token in word:
                token_ids.append(self.token_to_id[token])
        
        return token_ids
                
    def decode(self, IDs: List[int]) -> str:
        """
        decode converts list of token IDs back to the original text
        """        
        byte_list = []
        for ID in IDs:
            byte_list.append(self.vocab[ID])
        return b''.join(byte_list).decode('utf-8', errors='replace')
        

                
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
        
        # Rebuild merge priority map for efficient lookup
        tokenizer.merge_priority = {
            merge: idx for idx, merge in enumerate(tokenizer.merges)
        }

        print(f"Tokenizer loaded from {input_path}")
        return tokenizer
