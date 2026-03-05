# Go-Style Concurrency in BPE Tokenizer

## Overview

The BPE tokenizer now uses **Go-style concurrency patterns** instead of Python's traditional multiprocessing Pool. This implementation mimics Go's goroutines and channels using Python's threading and queues, providing a more elegant and efficient concurrent processing model.

## Key Concepts

### Go vs Python Equivalents

| Go Concept      | Python Equivalent             | Description                                |
| --------------- | ----------------------------- | ------------------------------------------ |
| `goroutine`     | `Thread`                      | Lightweight concurrent execution unit      |
| `channel`       | `Queue`                       | Communication mechanism between goroutines |
| `go func()`     | `Thread(target=func).start()` | Launch a goroutine                         |
| `ch <- value`   | `queue.put(value)`            | Send to channel                            |
| `value := <-ch` | `value = queue.get()`         | Receive from channel                       |
| `close(ch)`     | `queue.put(None)`             | Close channel (sentinel value)             |

## Architecture

### Traditional Pool-based Approach (Old)

```python
# Old approach: Process pool with map
with Pool(processes=num_processes) as pool:
    results = pool.map(worker_func, args_list)
```

**Limitations:**

- ❌ High overhead for process creation
- ❌ Less flexible communication patterns
- ❌ Difficult to implement complex workflows
- ❌ No streaming results

### Go-Style Approach (New)

```python
# New approach: Goroutines with channels
job_queue = Queue()      # Input channel
result_queue = Queue()   # Output channel

# Launch goroutines
for _ in range(num_workers):
    Thread(target=worker, args=(job_queue, result_queue)).start()

# Send jobs
for job in jobs:
    job_queue.put(job)

# Collect results
for _ in range(len(jobs)):
    result = result_queue.get()
```

**Advantages:**

- ✅ Lower overhead (threads vs processes)
- ✅ Flexible communication patterns
- ✅ Easy to implement complex workflows
- ✅ Supports streaming results
- ✅ Better resource utilization

## Implementation Details

### Worker Goroutine Pattern

The worker goroutines follow a standard Go pattern:

```python
def _process_chunk_worker(job_queue: Queue, result_queue: Queue, tokenizer_state: Tuple):
    """
    Worker goroutine that processes jobs from a channel.
    Mimics Go's pattern:

    func worker(jobs <-chan Job, results chan<- Result) {
        for job := range jobs {
            result := process(job)
            results <- result
        }
    }
    """
    # Initialize worker state (done once per goroutine)
    vocab, merges, special_tokens = tokenizer_state
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # Process jobs until channel is closed
    while True:
        job = job_queue.get()
        if job is None:  # Sentinel value = channel closed
            job_queue.task_done()
            break

        job_id, file_path, start, end = job

        try:
            # Process the job
            with open(file_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')

            result = tokenizer.pre_tokenize(chunk)

            # Send result to output channel
            result_queue.put((job_id, result))
        except Exception as e:
            result_queue.put((job_id, None, str(e)))
        finally:
            job_queue.task_done()
```

### Main Coordinator Pattern

The main function coordinates goroutines:

```python
def pre_tokenize_file_parallel(self, file_path, num_processes=None):
    """
    Coordinator that launches goroutines and manages channels.
    Mimics Go's pattern:

    func main() {
        jobs := make(chan Job, 100)
        results := make(chan Result, 100)

        // Launch workers
        for i := 0; i < numWorkers; i++ {
            go worker(jobs, results)
        }

        // Send jobs
        for _, job := range jobList {
            jobs <- job
        }
        close(jobs)

        // Collect results
        for i := 0; i < len(jobList); i++ {
            result := <-results
            process(result)
        }
    }
    """
    # Create channels (queues)
    job_queue = Queue()
    result_queue = Queue()

    # Prepare state
    tokenizer_state = (self.vocab, self.merges, self.special_tokens)
    boundaries = find_chunk_boundaries(...)
    chunks = list(zip(boundaries[:-1], boundaries[1:]))

    # Launch goroutines
    num_workers = min(num_processes, len(chunks))
    workers = []
    for _ in range(num_workers):
        worker = Thread(
            target=_process_chunk_worker,
            args=(job_queue, result_queue, tokenizer_state)
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Send jobs to channel
    for job_id, (start, end) in enumerate(chunks):
        job_queue.put((job_id, file_path, start, end))

    # Close channel (send sentinel values)
    for _ in range(num_workers):
        job_queue.put(None)

    # Collect results from channel
    results = {}
    for _ in range(len(chunks)):
        job_id, result = result_queue.get()
        results[job_id] = result

    # Wait for all goroutines to finish
    for worker in workers:
        worker.join()

    # Process results in order
    all_pre_tokens = []
    for job_id in sorted(results.keys()):
        all_pre_tokens.extend(results[job_id])

    return all_pre_tokens
```

## Key Features

### 1. Job Ordering

Jobs are tagged with IDs to maintain order:

```python
# Send jobs with IDs
for job_id, chunk in enumerate(chunks):
    job_queue.put((job_id, chunk))

# Collect results with IDs
results = {}
for _ in range(len(chunks)):
    job_id, result = result_queue.get()
    results[job_id] = result

# Process in order
for job_id in sorted(results.keys()):
    process(results[job_id])
```

This ensures that even though goroutines process jobs concurrently, the final output maintains the correct order.

### 2. Channel Closing

Go-style channel closing using sentinel values:

```python
# Send sentinel values to close channel
for _ in range(num_workers):
    job_queue.put(None)

# Worker detects closed channel
while True:
    job = job_queue.get()
    if job is None:  # Channel closed
        break
    process(job)
```

### 3. Error Handling

Errors are sent through the result channel:

```python
try:
    result = process(job)
    result_queue.put((job_id, result))
except Exception as e:
    result_queue.put((job_id, None, str(e)))
```

### 4. Graceful Shutdown

Workers are properly shut down:

```python
# Send sentinel values
for _ in range(num_workers):
    job_queue.put(None)

# Wait for all workers to finish
for worker in workers:
    worker.join()
```

## Comparison: Go vs Python

### Go Implementation

```go
package main

import (
    "sync"
)

func worker(jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()

    for job := range jobs {
        result := process(job)
        results <- result
    }
}

func main() {
    jobs := make(chan Job, 100)
    results := make(chan Result, 100)

    var wg sync.WaitGroup

    // Launch goroutines
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go worker(jobs, results, &wg)
    }

    // Send jobs
    for _, job := range jobList {
        jobs <- job
    }
    close(jobs)

    // Close results channel when all workers done
    go func() {
        wg.Wait()
        close(results)
    }()

    // Collect results
    for result := range results {
        process(result)
    }
}
```

### Python Implementation (Our Code)

```python
from threading import Thread
from queue import Queue

def worker(job_queue, result_queue, state):
    while True:
        job = job_queue.get()
        if job is None:  # Channel closed
            job_queue.task_done()
            break

        result = process(job, state)
        result_queue.put(result)
        job_queue.task_done()

def main():
    job_queue = Queue()
    result_queue = Queue()

    # Launch goroutines
    workers = []
    for _ in range(num_workers):
        worker_thread = Thread(target=worker, args=(job_queue, result_queue, state))
        worker_thread.daemon = True
        worker_thread.start()
        workers.append(worker_thread)

    # Send jobs
    for job in job_list:
        job_queue.put(job)

    # Close channel
    for _ in range(num_workers):
        job_queue.put(None)

    # Collect results
    results = []
    for _ in range(len(job_list)):
        result = result_queue.get()
        results.append(result)

    # Wait for workers
    for worker in workers:
        worker.join()

    return results
```

## Performance Characteristics

### Threading vs Multiprocessing

| Aspect            | Threading (Go-style)   | Multiprocessing (Pool)      |
| ----------------- | ---------------------- | --------------------------- |
| **Overhead**      | Low (shared memory)    | High (process creation)     |
| **Startup Time**  | Fast (~1ms per thread) | Slow (~50ms per process)    |
| **Memory**        | Shared (efficient)     | Copied (expensive)          |
| **GIL Impact**    | Limited by GIL         | No GIL (separate processes) |
| **Communication** | Fast (shared memory)   | Slow (IPC)                  |
| **Best For**      | I/O-bound tasks        | CPU-bound tasks             |

### When to Use Go-Style Concurrency

✅ **Use Go-style (Threading) when:**

- I/O-bound operations (file reading, network)
- Need fast startup and low overhead
- Frequent communication between workers
- Shared state is beneficial
- Task granularity is fine

❌ **Use Multiprocessing when:**

- CPU-bound operations (heavy computation)
- Need true parallelism (bypass GIL)
- Tasks are completely independent
- Task granularity is coarse

### Our Use Case: File I/O

The BPE tokenizer's parallel pre-tokenization is **I/O-bound**:

1. **Read file chunks** (I/O operation)
2. **Regex matching** (moderate CPU)
3. **Byte conversion** (light CPU)

This makes threading ideal:

- ✅ File I/O releases GIL
- ✅ Regex operations are fast
- ✅ Low overhead for many small chunks
- ✅ Fast communication for results

## Usage Examples

### Example 1: Parallel File Pre-tokenization

```python
from bpe_tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer.load('tokenizer.json')

# Pre-tokenize large file with Go-style concurrency
pre_tokens = tokenizer.pre_tokenize_file_parallel(
    file_path='large_corpus.txt',
    num_processes=8,  # Number of goroutines
    split_special_token='<|endoftext|>'
)

print(f"Generated {len(pre_tokens)} pre-tokens")
```

### Example 2: Parallel Text Pre-tokenization

```python
# Pre-tokenize text with Go-style concurrency
text = "Large text content..."
pre_tokens = tokenizer.pre_tokenize_parallel(
    text=text,
    num_processes=4,
    split_special_token='<|endoftext|>'
)
```

### Example 3: Full Parallel Encoding

```python
# Encode file with parallel pre-tokenization
token_ids = tokenizer.encode_file_parallel(
    file_path='corpus.txt',
    num_processes=8
)

print(f"Generated {len(token_ids)} tokens")
```

## Benefits of Go-Style Approach

### 1. **Simplicity**

Go-style code is easier to understand:

```python
# Clear flow: create channels → launch workers → send jobs → collect results
job_queue = Queue()
result_queue = Queue()

for _ in range(num_workers):
    Thread(target=worker, args=(job_queue, result_queue)).start()

for job in jobs:
    job_queue.put(job)

for _ in range(len(jobs)):
    result = result_queue.get()
```

### 2. **Flexibility**

Easy to implement complex patterns:

```python
# Pipeline pattern
stage1_out = Queue()
stage2_out = Queue()

Thread(target=stage1_worker, args=(input_queue, stage1_out)).start()
Thread(target=stage2_worker, args=(stage1_out, stage2_out)).start()
```

### 3. **Efficiency**

Lower overhead for I/O-bound tasks:

```
Threading:  ~1ms startup, shared memory
Multiprocessing: ~50ms startup, copied memory
```

### 4. **Scalability**

Easy to scale to many workers:

```python
# Can easily launch 100+ threads
for _ in range(100):
    Thread(target=worker, args=(job_queue, result_queue)).start()
```

## Testing

All tests pass with the Go-style implementation:

```bash
# Run tests
uv run pytest src/tokenization/bpe_tokenizer_test.py -v

# All tests pass:
# ✅ test_pre_tokenize
# ✅ test_encode
# ✅ test_encode_parallel
# ✅ test_decode
# ✅ test_compute_compression_ratio
# ✅ test_cross_tokenization
# ✅ test_parallel_pretokenization
# ✅ test_tokenizer_throughput
```

## Conclusion

The Go-style concurrency implementation provides:

- ✅ **Cleaner code**: Easier to understand and maintain
- ✅ **Better performance**: Lower overhead for I/O-bound tasks
- ✅ **More flexible**: Easy to implement complex patterns
- ✅ **Scalable**: Can handle many concurrent workers
- ✅ **Pythonic**: Uses standard library (threading, queue)

This approach demonstrates how Go's elegant concurrency patterns can be successfully applied in Python, resulting in more maintainable and efficient code.

## References

- [Go Concurrency Patterns](https://go.dev/blog/pipelines)
- [Python Threading](https://docs.python.org/3/library/threading.html)
- [Python Queue](https://docs.python.org/3/library/queue.html)
- [Effective Go - Concurrency](https://go.dev/doc/effective_go#concurrency)
