import unittest
import random
import time
from .bpe_tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_pre_tokenize(self):
        bpe_tokenizer = BPETokenizer(special_tokens=['<special token>'])
        bytes_list = bpe_tokenizer.pre_tokenize("test villager")
        print(f'byte_list = {bytes_list}')
    
    def test_encode(self):
        bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')
        res = bpe_tokenizer.encode("test villager")
        print(f'res={res}')
    
    def test_encode_parallel(self):
        """Test that parallel encoding produces the same results as sequential encoding."""
        bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')
        
        # Test with a larger text
        test_text = "test villager " * 1000 + "<|endoftext|>" + "hello world " * 1000
        
        # Sequential encoding
        res_sequential = bpe_tokenizer.encode(test_text, use_parallel=False)
        
        # Parallel encoding
        res_parallel = bpe_tokenizer.encode(test_text, use_parallel=True, num_processes=2)
        
        # They should produce identical results
        self.assertEqual(res_sequential, res_parallel)
        print(f'Sequential tokens: {len(res_sequential)}, Parallel tokens: {len(res_parallel)}')
        print(f'Results match: {res_sequential == res_parallel}')
    
    def test_decode(self):
        bpe_tokenizer = BPETokenizer.load('./src/tokenization/saved_bpe_owt_train.json')
        res = bpe_tokenizer.decode([13213, 13412])
        self.assertEqual(res, "test villager")


class BPETokenizerExperiment(unittest.TestCase):
    def compute_compression_ratio(self, weight_path, sample_size, validation_data_path):
        bpe_tokenizer = BPETokenizer.load(weight_path)
        with open(validation_data_path, 'r') as f:
            data = f.read()
            # sample 10 docs from data
            docs = data.split("<|endoftext|>")
            # Remove empty strings
            docs = [doc for doc in docs if doc.strip()]
            # Randomly sample documents
            sampled_docs = random.sample(docs, min(sample_size, len(docs)))
            
            # Calculate compression ratio
            total_bytes = 0
            total_tokens = 0
            for doc in sampled_docs:
                # Count bytes (UTF-8 encoding)
                total_bytes += len(doc.encode('utf-8'))
                # Encode and count tokens
                tokens = bpe_tokenizer.encode(doc)
                total_tokens += len(tokens)
            
            # Compression ratio = bytes/token
            compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
            return compression_ratio
    """
    Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
    encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
    """
    def test_compute_compression_ratio(self):
        test_data = {
            "tiny_stories": {
                "weight_path": "./src/tokenization/saved_bpe_tiny_story_train.json",
                "sample_size": 10,
                "validation_data_path": "./data/TinyStoriesV2-GPT4-valid.txt"
            },
            "owt": {
                "weight_path": "./src/tokenization/saved_bpe_owt_train.json",
                "sample_size": 10,
                "validation_data_path": "./data/owt_valid.txt"
            }
        }

        
        # Test both tokenizers
        for name, config in test_data.items():
            ratio = self.compute_compression_ratio(
                config["weight_path"],
                config["sample_size"],
                config["validation_data_path"]
            )
            print(f"{name} compression ratio: {ratio:.4f} bytes/token")


    def test_cross_tokenization(self):
        """
        What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
        """
        test_data = {
            "tiny_stories data, owt tokenizer": {
                "weight_path": "./src/tokenization/saved_bpe_owt_train.json",
                "sample_size": 10,
                "validation_data_path": "./data/TinyStoriesV2-GPT4-valid.txt"
            },
            "owt data, tiny stories tokenizer": {
                "weight_path": "./src/tokenization/saved_bpe_tiny_story_train.json",
                "sample_size": 10,
                "validation_data_path": "./data/owt_valid.txt"
            }
        }

        
        # Test both tokenizers
        for name, config in test_data.items():
            ratio = self.compute_compression_ratio(
                config["weight_path"],
                config["sample_size"],
                config["validation_data_path"]
            )
            print(f"{name} compression ratio: {ratio:.4f} bytes/token")


class BPETokenizerBenchmark(unittest.TestCase):
    """
    Benchmark the tokenizer throughput to estimate how long it would take to tokenize large datasets.
    """
    def test_parallel_pretokenization(self):
        """
        Test and benchmark parallel pre-tokenization vs sequential pre-tokenization.
        """
        print("\n" + "="*80)
        print("PARALLEL PRE-TOKENIZATION BENCHMARK")
        print("="*80)
        
        # Load tokenizer
        tokenizer_path = "./src/tokenization/saved_bpe_owt_train.json"
        bpe_tokenizer = BPETokenizer.load(tokenizer_path)
        
        # Create a temporary test file with a subset of data
        import os
        import tempfile
        
        test_file = "./data/owt_valid.txt"
        
        # Read first 5MB for testing
        with open(test_file, 'rb') as f:
            sample_data = f.read(5 * 1024 * 1024)  # 5MB
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(sample_data)
            tmp_file_path = tmp_file.name
        
        try:
            file_size = len(sample_data)
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"\nTest file: {tmp_file_path} (sample from {test_file})")
            print(f"File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
            
            # Initialize baseline_time for tracking speedup
            baseline_time = 0.0
            
            # Test with different numbers of processes
            for num_processes in [1, 2, 4]:
                print(f"\n{'='*80}")
                print(f"Testing with {num_processes} process(es)")
                print(f"{'='*80}")
                
                start_time = time.time()
                pre_tokens = bpe_tokenizer.pre_tokenize_file_parallel(
                    tmp_file_path, 
                    num_processes=num_processes,
                    split_special_token="<|endoftext|>"
                )
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                throughput_bytes_per_sec = file_size / elapsed_time
                throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
                
                print(f"Time taken: {elapsed_time:.2f} seconds")
                print(f"Pre-tokens generated: {len(pre_tokens):,}")
                print(f"Throughput: {throughput_bytes_per_sec:,.0f} bytes/sec ({throughput_mb_per_sec:.2f} MB/sec)")
                
                # Initialize baseline_time before the comparison
                if num_processes == 1:
                    baseline_time = elapsed_time
                    speedup = 1.0  # No speedup for baseline
                else:
                    speedup = baseline_time / elapsed_time
                    print(f"Speedup vs 1 process: {speedup:.2f}x")
            
            print("\n" + "="*80)
            print("PARALLEL PRE-TOKENIZATION BENCHMARK COMPLETE")
            print("="*80 + "\n")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def test_tokenizer_throughput(self):
        """
        Measure the throughput of the tokenizer in bytes/second.
        Estimate how long it would take to tokenize the Pile dataset (825GB).
        """
        # Load both tokenizers
        tokenizers = {
            "TinyStories (10K vocab)": "./src/tokenization/saved_bpe_tiny_story_train.json",
            "OpenWebText (32K vocab)": "./src/tokenization/saved_bpe_owt_train.json"
        }
        
        # Test data - use a reasonable sample size for benchmarking
        test_files = {
            "TinyStories": "./data/TinyStoriesV2-GPT4-valid.txt",
            "OpenWebText": "./data/owt_valid.txt"
        }
        
        print("\n" + "="*80)
        print("TOKENIZER THROUGHPUT BENCHMARK")
        print("="*80)
        
        for tokenizer_name, tokenizer_path in tokenizers.items():
            print(f"\n{tokenizer_name}:")
            print("-" * 80)
            
            bpe_tokenizer = BPETokenizer.load(tokenizer_path)
            
            for data_name, data_path in test_files.items():
                # Read a sample of data for benchmarking
                with open(data_path, 'r') as f:
                    data = f.read()
                
                # Use first 1MB of data for consistent benchmarking
                benchmark_size = min(1_000_000, len(data))  # 1MB or less
                benchmark_text = data[:benchmark_size]
                benchmark_bytes = len(benchmark_text.encode('utf-8'))
                
                # Warm-up run (to avoid cold start effects)
                _ = bpe_tokenizer.encode(benchmark_text[:10000])
                
                # Actual benchmark
                start_time = time.time()
                tokens = bpe_tokenizer.encode(benchmark_text)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                throughput_bytes_per_sec = benchmark_bytes / elapsed_time
                throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)
                
                # Calculate time to tokenize the Pile dataset (825GB)
                pile_size_bytes = 825 * 1024 * 1024 * 1024  # 825GB in bytes
                time_for_pile_seconds = pile_size_bytes / throughput_bytes_per_sec
                time_for_pile_hours = time_for_pile_seconds / 3600
                time_for_pile_days = time_for_pile_hours / 24
                
                print(f"\n  Testing on {data_name} data:")
                print(f"    Benchmark size: {benchmark_bytes:,} bytes ({benchmark_bytes/(1024*1024):.2f} MB)")
                print(f"    Time taken: {elapsed_time:.2f} seconds")
                print(f"    Tokens generated: {len(tokens):,}")
                print(f"    Throughput: {throughput_bytes_per_sec:,.0f} bytes/sec ({throughput_mb_per_sec:.2f} MB/sec)")
                print(f"    Compression ratio: {benchmark_bytes/len(tokens):.2f} bytes/token")
                print(f"\n    Estimated time to tokenize the Pile (825GB):")
                print(f"      {time_for_pile_seconds:,.0f} seconds")
                print(f"      {time_for_pile_hours:,.1f} hours")
                print(f"      {time_for_pile_days:.1f} days")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80 + "\n")
    

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
