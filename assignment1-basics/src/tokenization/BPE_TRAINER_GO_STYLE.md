# Go-Style Concurrency in BPE Trainer

## Overview

The BPE trainer now uses **Go-style concurrency patterns** for the preprocessing phase. This implementation replaces Python's `ThreadPoolExecutor` with a more elegant goroutine-and-channel pattern using `Thread` and `Queue`, providing better control flow and clearer code structure.

## Architecture Comparison

### Old Approach: ThreadPoolExecutor with Futures

```python
# Old approach: ThreadPoolExecutor with futures
with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
    # Submit all documents for processing
    futures = [executor.submit(self._process_document_chunk, doc) for doc in self.docs]

    # Collect and merge results as they complete
    for future in as_completed(futures):
        local_word_map, local_count_map, local_loc_map = future.result()
        self._merge_local_results(local_word_map, local_count_map, local_loc_map)
```

**Limitations:**

- ❌ Less explicit control flow
- ❌ Harder to understand execution order
- ❌ Futures abstraction adds complexity
- ❌ Difficult to implement custom patterns

### New Approach: Go-Style Goroutines with Channels

```python
# New approach: Goroutines with channels
job_queue = Queue()      # Input channel
result_queue = Queue()   # Output channel

# Launch goroutines
for _ in range(num_workers):
    Thread(target=worker, args=(job_queue, result_queue)).start()

# Send jobs
for job_id, doc in enumerate(docs):
    job_queue.put((job_id, doc))

# Close channel
for _ in range(num_workers):
    job_queue.put(None)

# Collect results
for _ in range(len(docs)):
    result = result_queue.get()
    process(result)
```

**Advantages:**

- ✅ Explicit control flow (easy to follow)
- ✅ Clear communication pattern
- ✅ Simple to understand and maintain
- ✅ Easy to extend with custom patterns
- ✅ Better debugging experience

## Implementation Details

### Worker Goroutine Pattern

The worker goroutine processes documents from the job channel:

```python
def _process_document_worker(self, job_queue: Queue, result_queue: Queue):
    """
    Worker goroutine (thread) for parallel document processing.
    Mimics Go's goroutine pattern:

    func worker(jobs <-chan Job, results chan<- Result) {
        for job := range jobs {
            result := process(job)
            results <- result
        }
    }
    """
    while True:
        job = job_queue.get()
        if job is None:  # Sentinel value = channel closed
            job_queue.task_done()
            break

        job_id, doc = job

        try:
            # Process document chunk - local data structures (thread-safe)
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

            # Send result to output channel
            result_queue.put((job_id, local_word_map, local_count_map, local_loc_map))
        except Exception as e:
            # Send error result
            result_queue.put((job_id, None, None, None, str(e)))
        finally:
            job_queue.task_done()
```

### Coordinator Pattern in Preprocess

The `preprocess()` method coordinates the goroutines:

```python
def preprocess(self):
    """
    Preprocess the corpus using Go-style multi-threading.

    Go-style pattern:
    1. Create job and result channels (queues)
    2. Launch worker goroutines (threads)
    3. Send jobs to job channel
    4. Close job channel (send sentinel values)
    5. Collect results from result channel
    6. Wait for all goroutines to finish
    """
    # Read and split corpus
    with open(self.input_path, "r") as f:
        corpus = f.read()

    if self.special_tokens:
        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        self.docs = [doc for doc in re.split(pattern, corpus) if doc]
    else:
        self.docs = [corpus]

    # Initialize data structures
    self.count_map = {}
    self.count_queue = PriorityQueue()
    self.loc_map = {}

    print(f"Processing {len(self.docs)} documents using {self.num_threads} threads (Go-style)...")

    # Go-style concurrency: Create channels (queues)
    job_queue = Queue()
    result_queue = Queue()

    # Calculate number of workers
    num_workers = min(self.num_threads, len(self.docs))

    # Launch goroutines (threads)
    workers = []
    for _ in range(num_workers):
        worker = Thread(
            target=self._process_document_worker,
            args=(job_queue, result_queue)
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Send jobs to the job channel
    for job_id, doc in enumerate(self.docs):
        job_queue.put((job_id, doc))

    # Close channel (send sentinel values)
    for _ in range(num_workers):
        job_queue.put(None)

    # Collect results from the result channel
    results = {}
    for _ in range(len(self.docs)):
        result = result_queue.get()
        if len(result) == 5:  # Error case
            job_id, _, _, _, error = result
            print(f"Error processing document {job_id}: {error}")
            continue
        job_id, local_word_map, local_count_map, local_loc_map = result
        results[job_id] = (local_word_map, local_count_map, local_loc_map)

    # Wait for all goroutines to finish
    for worker in workers:
        worker.join()

    # Merge results in order
    for job_id in sorted(results.keys()):
        local_word_map, local_count_map, local_loc_map = results[job_id]
        self._merge_local_results(local_word_map, local_count_map, local_loc_map)

    print(f"Processed {len(self.word_map)} unique words")

    # Build priority queue from count_map
    for pairs, count in self.count_map.items():
        pair_bytes = (self.vocab[pairs[0]], self.vocab[pairs[1]])
        self.count_queue.put(PriorityItem(count, pairs, pair_bytes))
```

## Key Features

### 1. Job Ordering

Jobs are tagged with IDs to maintain processing order:

```python
# Send jobs with IDs
for job_id, doc in enumerate(self.docs):
    job_queue.put((job_id, doc))

# Collect results with IDs
results = {}
for _ in range(len(self.docs)):
    job_id, local_word_map, local_count_map, local_loc_map = result_queue.get()
    results[job_id] = (local_word_map, local_count_map, local_loc_map)

# Merge in order
for job_id in sorted(results.keys()):
    local_word_map, local_count_map, local_loc_map = results[job_id]
    self._merge_local_results(local_word_map, local_count_map, local_loc_map)
```

This ensures deterministic results even with concurrent processing.

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
        job_queue.task_done()
        break
    process(job)
```

### 3. Error Handling

Errors are sent through the result channel:

```python
try:
    # Process document
    result = process(doc)
    result_queue.put((job_id, result))
except Exception as e:
    # Send error result with extra field
    result_queue.put((job_id, None, None, None, str(e)))

# Coordinator detects errors
result = result_queue.get()
if len(result) == 5:  # Error case
    job_id, _, _, _, error = result
    print(f"Error processing document {job_id}: {error}")
    continue
```

### 4. Thread-Safe Merging

Results are merged using locks to ensure thread-safety:

```python
def _merge_local_results(self, local_word_map, local_count_map, local_loc_map):
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
```

## Comparison: Go vs Python

### Go Implementation

```go
package main

import (
    "sync"
)

type Job struct {
    ID  int
    Doc string
}

type Result struct {
    ID           int
    WordMap      map[string]*Word
    CountMap     map[Pair]int
    LocMap       map[Pair][]string
}

func worker(jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()

    for job := range jobs {
        // Process document
        wordMap, countMap, locMap := processDocument(job.Doc)

        results <- Result{
            ID:       job.ID,
            WordMap:  wordMap,
            CountMap: countMap,
            LocMap:   locMap,
        }
    }
}

func preprocess(docs []string, numWorkers int) {
    jobs := make(chan Job, len(docs))
    results := make(chan Result, len(docs))

    var wg sync.WaitGroup

    // Launch goroutines
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go worker(jobs, results, &wg)
    }

    // Send jobs
    for id, doc := range docs {
        jobs <- Job{ID: id, Doc: doc}
    }
    close(jobs)

    // Close results channel when all workers done
    go func() {
        wg.Wait()
        close(results)
    }()

    // Collect results
    resultMap := make(map[int]Result)
    for result := range results {
        resultMap[result.ID] = result
    }

    // Merge results in order
    for id := 0; id < len(docs); id++ {
        result := resultMap[id]
        mergeResults(result)
    }
}
```

### Python Implementation (Our Code)

```python
from threading import Thread
from queue import Queue

def _process_document_worker(self, job_queue, result_queue):
    while True:
        job = job_queue.get()
        if job is None:  # Channel closed
            job_queue.task_done()
            break

        job_id, doc = job

        try:
            # Process document
            word_map, count_map, loc_map = process_document(doc)
            result_queue.put((job_id, word_map, count_map, loc_map))
        except Exception as e:
            result_queue.put((job_id, None, None, None, str(e)))
        finally:
            job_queue.task_done()

def preprocess(self):
    job_queue = Queue()
    result_queue = Queue()

    # Launch goroutines
    workers = []
    for _ in range(num_workers):
        worker = Thread(target=self._process_document_worker,
                       args=(job_queue, result_queue))
        worker.daemon = True
        worker.start()
        workers.append(worker)

    # Send jobs
    for job_id, doc in enumerate(self.docs):
        job_queue.put((job_id, doc))

    # Close channel
    for _ in range(num_workers):
        job_queue.put(None)

    # Collect results
    results = {}
    for _ in range(len(self.docs)):
        result = result_queue.get()
        if len(result) == 5:  # Error
            continue
        job_id, word_map, count_map, loc_map = result
        results[job_id] = (word_map, count_map, loc_map)

    # Wait for workers
    for worker in workers:
        worker.join()

    # Merge results in order
    for job_id in sorted(results.keys()):
        word_map, count_map, loc_map = results[job_id]
        self._merge_local_results(word_map, count_map, loc_map)
```

## Performance Characteristics

### Why Threading Works Well Here

The BPE trainer preprocessing is **I/O and regex-bound**:

1. **File I/O**: Reading corpus from disk (releases GIL)
2. **Regex matching**: `re.findall(PAT, doc)` (moderate CPU, releases GIL)
3. **Dictionary operations**: Building local maps (light CPU)

This makes threading ideal:

- ✅ Regex operations release GIL
- ✅ Dictionary operations are fast
- ✅ Low overhead for many documents
- ✅ Fast communication for results

### Expected Performance Gains

| Number of Threads | Expected Speedup |
| ----------------- | ---------------- |
| 1 (sequential)    | 1.0x (baseline)  |
| 2                 | 1.5-1.7x         |
| 4                 | 2.0-2.5x         |
| 8                 | 2.5-3.5x         |

**Note**: Speedup is sublinear due to:

- GIL contention during dictionary operations
- Lock contention during result merging
- Overhead of thread coordination

## Usage Examples

### Example 1: Basic Training with Go-Style Concurrency

```python
from bpe_trainer import BPETrainer

# Create trainer with default thread count (min(cpu_count(), 8))
trainer = BPETrainer(
    input_path='corpus.txt',
    vocab_size=10000,
    special_tokens=['<|endoftext|>']
)

# Preprocess with Go-style concurrency
trainer.preprocess()  # Uses Go-style goroutines and channels

# Train BPE
vocab, merges = trainer.train()

# Save tokenizer
trainer.save('tokenizer.json')
```

### Example 2: Custom Thread Count

```python
# Use more threads for large corpus
trainer = BPETrainer(
    input_path='large_corpus.txt',
    vocab_size=32000,
    special_tokens=['<|endoftext|>'],
    num_threads=16  # Custom thread count
)

trainer.preprocess()
vocab, merges = trainer.train()
```

### Example 3: Single-Threaded (for debugging)

```python
# Use single thread for debugging
trainer = BPETrainer(
    input_path='corpus.txt',
    vocab_size=10000,
    num_threads=1  # Sequential processing
)

trainer.preprocess()
vocab, merges = trainer.train()
```

## Benefits of Go-Style Approach

### 1. **Clarity**

The execution flow is explicit and easy to follow:

```python
# Clear 6-step pattern
1. Create channels
2. Launch goroutines
3. Send jobs
4. Close channels
5. Collect results
6. Wait for completion
```

### 2. **Maintainability**

Easy to modify and extend:

```python
# Easy to add progress tracking
progress_queue = Queue()

def worker(job_queue, result_queue, progress_queue):
    while True:
        job = job_queue.get()
        if job is None:
            break
        result = process(job)
        result_queue.put(result)
        progress_queue.put(1)  # Report progress
```

### 3. **Debuggability**

Easy to add logging and debugging:

```python
def worker(job_queue, result_queue):
    while True:
        job = job_queue.get()
        if job is None:
            break

        print(f"Processing job {job[0]}")  # Easy to add logging
        result = process(job)
        print(f"Completed job {job[0]}")
        result_queue.put(result)
```

### 4. **Testability**

Easy to test individual components:

```python
# Test worker in isolation
job_queue = Queue()
result_queue = Queue()

job_queue.put((0, "test document"))
job_queue.put(None)

worker(job_queue, result_queue)

result = result_queue.get()
assert result[0] == 0  # Check job ID
```

## Comparison with ThreadPoolExecutor

| Aspect                | Go-Style (Thread+Queue) | ThreadPoolExecutor       |
| --------------------- | ----------------------- | ------------------------ |
| **Code Clarity**      | ✅ Explicit flow        | ❌ Implicit with futures |
| **Control**           | ✅ Full control         | ❌ Limited control       |
| **Debugging**         | ✅ Easy to debug        | ❌ Harder to debug       |
| **Extensibility**     | ✅ Easy to extend       | ❌ Limited patterns      |
| **Performance**       | ✅ Similar              | ✅ Similar               |
| **Error Handling**    | ✅ Explicit             | ❌ Exception handling    |
| **Progress Tracking** | ✅ Easy to add          | ❌ Requires callbacks    |

## Testing

The Go-style implementation maintains compatibility with all existing tests:

```bash
# Run all trainer tests
uv run pytest src/tokenization/bpe_trainer_test.py -v

# Expected output:
# ✅ test_bpe
# ✅ test_pattern_matching
# ✅ test_train_owt
# ✅ test_train_tiny_stories
# ✅ test_save_load
```

## Conclusion

The Go-style concurrency implementation in the BPE trainer provides:

- ✅ **Cleaner code**: Explicit control flow, easy to understand
- ✅ **Better maintainability**: Simple to modify and extend
- ✅ **Easier debugging**: Clear execution path, easy to add logging
- ✅ **Same performance**: Similar speedup to ThreadPoolExecutor
- ✅ **More flexible**: Easy to implement custom patterns
- ✅ **Pythonic**: Uses standard library (threading, queue)

This approach demonstrates how Go's elegant concurrency patterns can be successfully applied to Python's BPE training preprocessing, resulting in more maintainable and understandable code without sacrificing performance.

## References

- [Go Concurrency Patterns](https://go.dev/blog/pipelines)
- [Python Threading](https://docs.python.org/3/library/threading.html)
- [Python Queue](https://docs.python.org/3/library/queue.html)
- [Effective Go - Concurrency](https://go.dev/doc/effective_go#concurrency)
- [BPE Tokenizer Go-Style Documentation](./GO_STYLE_CONCURRENCY.md)
