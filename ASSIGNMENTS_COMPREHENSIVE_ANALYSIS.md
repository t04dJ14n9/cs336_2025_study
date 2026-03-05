# CS336 2025 Assignments - Comprehensive Analysis & Implementation Guide

## Executive Summary

This document provides a complete analysis of all four CS336 assignments, their interconnections, dependencies, and implementation requirements. Assignment 1 is **COMPLETE** with all 51 tests passing. The remaining assignments (A2, A3, A4) require implementation.

---

## Assignment 1: Building a Transformer Language Model

### Status: ✅ **COMPLETE** - All 51 tests passing

### Core Requirements

#### 1. BPE Tokenizer Implementation (Section 2)
**Objective**: Build a byte-level BPE tokenizer from scratch

**Key Components**:
- Pre-tokenization using GPT-2 regex pattern (PAT)
- Byte-level vocabulary initialization (256 byte values)
- Iterative merge computation
- Special token handling (<|endoftext|>)
- Encoding and decoding functions
- Memory-efficient streaming via `encode_iterable`

**Constraints**:
- Must handle large files via chunking
- Special tokens must not be split
- Ties broken lexicographically (greater pair wins)
- Target: <2 minutes training on TinyStories with multiprocessing

**Deliverables**:
- `BPETrainer` class for training
- `BPETokenizer` class for encode/decode
- Tests: `tests/test_train_bpe.py`, `tests/test_tokenizer.py`

**Current Implementation Status**: ✅ Complete
- Located in: `src/tokenization/bpe_tokenizer.py`, `src/tokenization/bpe_trainer.py`
- Features: Parallel pre-tokenization, optimized merge priority queue, memory-efficient encoding

#### 2. Transformer Language Model (Section 3)
**Objective**: Build a complete Transformer LM from scratch

**Architecture Components**:

##### 2.1 Basic Building Blocks
- **Linear Layer** (no bias): `src/layers/linear.py`
  - Initialization: Truncated normal σ² = 2/(d_in + d_out)
  - Constraint: Cannot use `nn.Linear` or `F.linear`
  
- **Embedding Layer**: `src/layers/embedding.py`
  - Shape: (vocab_size, d_model)
  - Constraint: Cannot use `nn.Embedding`

##### 2.2 Normalization
- **RMSNorm** (Root Mean Square Layer Normalization):
  - Formula: `RMSNorm(a_i) = a_i / RMS(a) * g_i`
  - Must upcast to float32 for numerical stability
  - No learnable bias, only gain parameter g

##### 2.3 Position Encoding
- **RoPE** (Rotary Position Embeddings):
  - Applies rotation to Q and K vectors (not V)
  - Rotation matrix: Block-diagonal with 2x2 rotation blocks
  - θ_i,k = i * Θ^(2k-2)/d for k ∈ {1, ..., d/2}
  - Can precompute cos/sin buffers for efficiency

##### 2.4 Attention Mechanism
- **Scaled Dot-Product Attention**:
  - Formula: `Attention(Q, K, V) = softmax(Q^T K / √d_k) V`
  - Must support masking (True = attend, False = mask)
  - Numerical stability: subtract max before softmax
  
- **Multi-Head Self-Attention**:
  - Projects Q, K, V in single matrix multiply each
  - Applies RoPE after reshaping to separate heads
  - Causal masking for autoregressive generation
  - Returns: `W_o @ Concat(head_1, ..., head_h)`

##### 2.5 Feed-Forward Network
- **SwiGLU Activation**:
  - Formula: `FFN(x) = W2(SiLU(W1 x) ⊙ W3 x)`
  - SiLU(x) = x * sigmoid(x)
  - d_ff ≈ (8/3) * d_model, rounded to multiple of 64

##### 2.6 Transformer Block
- **Pre-norm architecture**:
  ```
  y = x + MultiHeadSelfAttention(RMSNorm(x))
  y = y + FeedForward(RMSNorm(y))
  ```

##### 2.7 Full Transformer LM
```
Input tokens → Embedding → [Transformer Block] × num_layers 
            → RMSNorm → Linear(vocab_size) → Logits
```

**Constraint Checklist**:
- ❌ Cannot use: `torch.nn.*` (except Parameter, Module, ModuleList, Sequential)
- ❌ Cannot use: `torch.nn.functional.*`
- ❌ Cannot use: `torch.optim.*` (except Optimizer base class)
- ✅ Can use: `torch.sigmoid`, `torch.tril`, basic tensor ops

**Current Implementation Status**: ✅ Complete
- All layers implemented in: `src/layers/`
- Full model in: `src/modules/transformer.py`
- All tests passing in `tests/test_model.py`

#### 3. Training Components (Section 4)

##### 3.1 Loss Function
- **Cross-Entropy**:
  - Numerically stable implementation (log-sum-exp trick)
  - Returns average loss over batch
  - `src/nn_utils.py` ✅

##### 3.2 Optimizer
- **AdamW** (Adam with decoupled weight decay):
  - Maintains first and second moment estimates (m, v)
  - Bias correction: `α_t = α * √(1-β2^t) / (1-β1^t)`
  - Decoupled weight decay: `θ = θ - α λ θ`
  - Default hyperparameters: β1=0.9, β2=0.999, ε=1e-8, λ=0.01
  - `src/optimizer.py` ✅

##### 3.3 Learning Rate Schedule
- **Cosine Annealing with Warmup**:
  - Warmup: `α_t = (t/T_w) * α_max` for t < T_w
  - Cosine: `α_t = α_min + 0.5(1 + cos(π(t-T_w)/(T_c-T_w)))(α_max - α_min)`
  - Post: `α_t = α_min` for t > T_c
  - `src/optimizer.py` ✅

##### 3.4 Gradient Clipping
- L2 norm clipping with threshold M
- Scale factor: `M / (||g||_2 + ε)`
- In-place modification
- `src/nn_utils.py` ✅

##### 3.5 Data Loading
- **`get_batch` function**:
  - Returns (input_ids, target_ids) of shape (batch_size, context_length)
  - Memory-mapped loading for large datasets
  - `src/nn_utils.py` ✅

##### 3.6 Checkpointing
- **`save_checkpoint`**: Serialize model, optimizer, iteration
- **`load_checkpoint`**: Restore state, return iteration
- `src/serialization.py` ✅

**Current Implementation Status**: ✅ Complete
- All training utilities implemented
- Tests passing: `tests/test_optimizer.py`, `tests/test_nn_utils.py`, `tests/test_serialization.py`

#### 4. Training Loop & Experiments (Section 5)

**Required Experiments**:
1. Train BPE tokenizer on TinyStories (vocab_size=10,000)
2. Train BPE tokenizer on OpenWebText (vocab_size=32,000)
3. Tokenize datasets and save as uint16 arrays
4. Train Transformer LM on TinyStories
5. Generate samples and evaluate perplexity
6. Train on OpenWebText for leaderboard submission

**Resource Requirements**:
- TinyStories BPE training: ≤30 minutes, ≤30GB RAM
- OpenWebText BPE training: ≤12 hours, ≤100GB RAM
- Model training: Varies by configuration

**Current Implementation Status**: ✅ Code complete, needs execution on GPU
- Training script: `train.py` exists
- Tokenization script: `tokenize_data.py` exists

### Assignment 1 - Key Constraints Summary

1. **"From Scratch" Philosophy**: 
   - No use of high-level PyTorch abstractions
   - Forces deep understanding of each component
   
2. **Numerical Stability**:
   - Must handle overflow/underflow in softmax, cross-entropy
   - Upcasting for normalization operations
   
3. **Memory Efficiency**:
   - Streaming tokenization for large files
   - Memory-mapped dataset loading
   - Gradient checkpointing support

4. **Correctness Over Optimization**:
   - Must match reference implementations exactly
   - Tests verify mathematical correctness

---

## Assignment 2: Systems and Parallelism

### Status: 🔴 **NOT STARTED** - Requires implementation

### Overview
Focuses on **optimizing and scaling** the Transformer from A1:
- Single-GPU optimization (Flash Attention)
- Multi-GPU training (DDP)
- Memory optimization (optimizer sharding)

### Key Deliverables

#### 1. Benchmarking & Profiling (Section 1)
- **Objective**: Profile A1 model to identify bottlenecks
- Create benchmarking script for forward/backward passes
- Measure memory usage, FLOPs, throughput
- **Problem**: `benchmarking_script` (4 points)

#### 2. Flash Attention 2 (Section 2)
- **Objective**: Implement memory-efficient attention kernel

**Two Implementations Required**:
1. **PyTorch-only version**:
   - `get_flashattention_autograd_function_pytorch()`
   - Custom autograd.Function with memory-efficient backward
   - No Triton allowed
   
2. **Triton kernel version**:
   - `get_flashattention_autograd_function_triton()`
   - Custom CUDA kernels via Triton
   - Must match PyTorch version behavior

**Key Concepts**:
- Tiling to avoid materializing full attention matrix
- Online softmax computation
- Memory: O(N) instead of O(N²)
- Backward pass recomputation strategy

**Tests**: `tests/test_attention.py`

#### 3. Distributed Data Parallel (Section 3)
- **Objective**: Implement DDP for multi-GPU training

**Two Strategies**:

1. **Individual Parameter DDP**:
   - `get_ddp_individual_parameters(module)`
   - Broadcast parameters at init
   - All-reduce gradients individually as they're computed
   - Overlaps communication with backward pass
   - Hook: `ddp_individual_parameters_on_after_backward()`

2. **Bucketed DDP**:
   - `get_ddp_bucketed(module, bucket_size_mb)`
   - Groups parameters into buckets by size
   - All-reduce bucket when all gradients ready
   - Better bandwidth utilization
   - Hooks: `ddp_bucketed_on_after_backward()`, `ddp_bucketed_on_train_batch_start()`

**Key Concepts**:
- Gradient synchronization hooks
- Communication-computation overlap
- Bucket size tradeoffs

**Tests**: `tests/test_ddp*.py`

#### 4. Sharded Optimizer (Section 4)
- **Objective**: Shard optimizer state across GPUs

- **`get_sharded_optimizer(params, optimizer_cls, **kwargs)`**:
  - Each GPU stores only 1/N of optimizer states
  - Reduces memory from O(params) to O(params/N)
  - Must handle parameter-local state (Adam's m, v)
  
**Tests**: `tests/test_sharded_optimizer.py`

### Assignment 2 - Dependencies on A1

- **Reuses**: Transformer model from A1
- **Extends**: Adds performance optimizations
- **Tests**: Uses `tests/adapters.py` to call your implementations
- **Critical**: Must maintain mathematical correctness from A1

### Implementation Strategy for A2

1. **Flash Attention** (Hardest):
   - Start with PyTorch version first
   - Understand tiling strategy from Flash Attention 2 paper
   - Then implement Triton version
   - CPU testing: verify numerical correctness

2. **DDP** (Medium):
   - Start with individual parameter version (simpler)
   - Use `torch.distributed` primitives
   - Test with `torch.multiprocessing.spawn`
   - CPU testing: mock distributed environment

3. **Sharded Optimizer** (Medium):
   - Wrap existing AdamW from A1
   - Implement state partitioning logic
   - CPU testing: single-process simulation

---

## Assignment 3: Scaling Laws

### Status: 🔴 **NOT STARTED** - Requires implementation

### Overview
Study **how model performance scales** with compute, data, and parameters using the Chinchilla scaling laws.

### Key Deliverables

#### 1. IsoFLOP Analysis (Section 1)
- **Objective**: Fit scaling laws to training run data

**Given**:
- `data/isoflops_curves.json`: Synthetic training data
- Contains: model sizes, dataset sizes, compute budgets, final losses

**Tasks**:
1. Fit Chinchilla-style scaling law:
   ```
   L(N, D) = E + A/N^α + B/D^β
   ```
   where N = parameters, D = tokens, E/A/B/α/β are fit

2. Plot optimal model size vs compute budget
3. Plot optimal dataset size vs compute budget

**Problem**: `chinchilla_isoflops` (5 points)
**Deliverables**: Plots showing scaling relationships

#### 2. Extrapolation to 10^19 FLOPs (Section 2)
- **Objective**: Predict optimal configuration for large compute budget

**Tasks**:
1. Use fitted scaling law to predict:
   - Optimal model size at 10^19 FLOPs
   - Optimal dataset size at 10^19 FLOPs
   - Expected final loss
   
2. Compare to Chinchilla paper predictions
3. Analyze uncertainty/confidence

**Problem**: `scaling_laws` (50 points - MAJOR)
**Deliverable**: Comprehensive write-up with:
- Fitting methodology
- Coefficient values (α, β, E, A, B)
- Plots and predictions
- Uncertainty analysis
- Comparison to literature

### Assignment 3 - Dependencies on A1/A2

- **Uses**: Model architecture knowledge from A1
- **Does NOT require**: Running actual training
- **Input**: Provided synthetic data (no GPU needed)
- **Focus**: Data analysis, curve fitting, statistics

### Implementation Strategy for A3

1. **Data Analysis**:
   - Load and explore `isoflops_curves.json`
   - Visualize existing data points
   
2. **Fitting**:
   - Implement log-space regression
   - Use `scipy.optimize.curve_fit` or similar
   - Bootstrap for uncertainty quantification
   
3. **Prediction**:
   - Extrapolate using fitted parameters
   - Visualize predictions with confidence intervals
   
4. **Write-up**:
   - Document methodology
   - Include all plots and tables
   - Discuss limitations

**CPU Testing**: All analysis can run on CPU (no training required)

---

## Assignment 4: Data Processing at Scale

### Status: 🔴 **NOT STARTED** - Requires implementation

### Overview
Build **data filtering and preprocessing pipelines** for training language models on web-scale data.

### Key Deliverables

#### 1. Text Extraction (Section 2)
- **Objective**: Extract text from HTML

- **`run_extract_text_from_html_bytes(html_bytes)`**:
  - Input: Raw HTML bytes
  - Output: Plain text string
  - Use Resiliparse library
  - Handle encoding issues

**Problem**: `extract_text` (3 points)

#### 2. Language Identification (Section 3)
- **Objective**: Identify language of text

- **`run_identify_language(text)`**:
  - Input: Text string
  - Output: Language code (e.g., 'en', 'zh')
  - Use fastText or similar

**Tests**: `tests/test_langid.py`

#### 3. PII Masking (Section 4)
- **Objective**: Detect and mask personally identifiable information

- **`run_mask_pii(text)`**:
  - Detect: emails, phone numbers, addresses, etc.
  - Replace with placeholders (e.g., `<EMAIL>`, `<PHONE>`)
  - Return masked text

**Tests**: `tests/test_pii.py`

#### 4. Quality Filtering (Section 5)

**Multiple Classifiers**:

1. **NSFW Detection**:
   - `run_nsfw_detection(text)` → bool
   
2. **Toxicity Detection**:
   - `run_toxicity_detection(text)` → bool
   
3. **Gopher Quality Rules**:
   - Implement rules from Gopher paper
   - Word count, sentence length, symbol ratios, etc.
   - `run_gopher_quality_filter(text)` → bool

**Tests**: `tests/test_quality.py`, `tests/test_toxicity.py`

#### 5. Deduplication (Section 6)

**Two Methods**:

1. **Exact Deduplication**:
   - `run_exact_deduplication(documents)` → deduplicated docs
   - Use hash-based approach
   
2. **MinHash LSH Deduplication**:
   - `run_minhash_deduplication(documents, threshold)` → deduplicated docs
   - Near-duplicate detection
   - Implement MinHash + Locality Sensitive Hashing

**Tests**: `tests/test_dedup.py`

### Assignment 4 - Dependencies on A1/A2

- **Independent**: Can be developed standalone
- **Uses**: BPE tokenizer from A1 for final tokenization
- **Feeds into**: Training data for models in A1/A2
- **Focus**: Data engineering, not ML model building

### Implementation Strategy for A4

1. **Text Extraction** (Easy):
   - Use Resiliparse library
   - Handle edge cases (encoding, malformed HTML)
   
2. **Language ID** (Easy):
   - Use pre-trained fastText model
   - Download and load model weights
   
3. **PII Masking** (Medium):
   - Use regex patterns
   - Consider using presidio library
   
4. **Quality Filtering** (Medium):
   - NSFW/Toxicity: Use pre-trained classifiers
   - Gopher rules: Implement manually
   
5. **Deduplication** (Hard):
   - Exact: Hash-based (easy)
   - MinHash: Implement from scratch or use datasketch library
   - Optimize for streaming/large-scale

**CPU Testing**: All components testable on CPU with small samples

---

## Cross-Assignment Dependencies

```
A1 (Transformer Basics)
 ├── Provides: Model architecture, tokenizer, optimizer
 ├── Used by: A2 (for optimization), A3 (for understanding), A4 (for tokenization)
 │
A2 (Systems & Parallelism)
 ├── Requires: A1 model implementation
 ├── Extends: Adds Flash Attention, DDP, Sharded Optimizer
 ├── Enables: Faster training for A1 experiments
 │
A3 (Scaling Laws)
 ├── Requires: Understanding of A1 model (no code dependency)
 ├── Uses: Synthetic data (provided)
 ├── Independent: Can be done in parallel with A2/A4
 │
A4 (Data Processing)
 ├── Requires: A1 tokenizer (for final step)
 ├── Independent: Core logic standalone
 ├── Feeds: Training data to A1/A2
```

## Implementation Priority

### Phase 1: Foundation (Already Complete ✅)
- **A1**: All components implemented and tested

### Phase 2: Systems Optimization
- **A2 Flash Attention**: PyTorch version first, then Triton
- **A2 DDP**: Individual parameters version first
- **A2 Sharded Optimizer**: Wrap A1's AdamW

### Phase 3: Data Pipeline
- **A4**: Can be done in parallel with A2
- **A4 Text Extraction**: Start here
- **A4 Language ID & PII**: Then these
- **A4 Quality Filters**: Then these
- **A4 Deduplication**: Last (most complex)

### Phase 4: Analysis
- **A3**: Can be done anytime after understanding A1
- **A3 Fitting**: Core task
- **A3 Prediction**: Final deliverable

## Resource Requirements Summary

| Assignment | GPU Required? | Memory | Compute Time |
|------------|---------------|--------|--------------|
| A1 - Tests | ❌ No | <8GB | Minutes |
| A1 - Training | ✅ Yes | 16-80GB | Hours-Days |
| A2 - Tests | ❌ No | <8GB | Minutes |
| A2 - Flash Attn | ✅ Yes | 16GB+ | Hours |
| A2 - DDP | ✅ Yes (multi-GPU) | 32GB+ | Hours |
| A3 - All | ❌ No | <8GB | Minutes |
| A4 - Tests | ❌ No | <8GB | Minutes |
| A4 - Scale | ❌ No (CPU OK) | 32-100GB | Hours-Days |

## Testing Strategy

### CPU-Only Testing (Local Machine)
1. **A1**: All 51 unit tests pass ✅
2. **A2**: 
   - Numerical correctness tests (no actual multi-GPU)
   - Mock distributed environment
   - Single-GPU Flash Attention correctness
3. **A3**: All analysis and fitting
4. **A4**: Small sample tests for all components

### GPU Testing (Cluster)
1. **A1**: Training runs, perplexity evaluation
2. **A2**: 
   - Flash Attention performance benchmarks
   - Actual multi-GPU DDP tests
   - Distributed optimizer tests
3. **A3**: N/A (no GPU needed)
4. **A4**: Large-scale deduplication (if needed)

## Success Criteria

### Assignment 1 ✅
- [x] All 51 unit tests passing
- [x] BPE tokenizer trains in <2 minutes on TinyStories
- [x] Model forward pass matches reference
- [x] Optimizer converges on toy task
- [ ] Train on TinyStories (GPU)
- [ ] Evaluate perplexity (GPU)
- [ ] Train on OpenWebText for leaderboard (GPU)

### Assignment 2
- [ ] Flash Attention PyTorch version passes tests
- [ ] Flash Attention Triton version passes tests
- [ ] Individual parameter DDP passes tests
- [ ] Bucketed DDP passes tests
- [ ] Sharded optimizer passes tests
- [ ] Benchmarking script runs

### Assignment 3
- [ ] Fit scaling law to isoflops data
- [ ] Generate required plots
- [ ] Predict optimal config at 10^19 FLOPs
- [ ] Comprehensive write-up complete

### Assignment 4
- [ ] Text extraction passes tests
- [ ] Language ID passes tests
- [ ] PII masking passes tests
- [ ] Quality filters pass tests
- [ ] Exact deduplication passes tests
- [ ] MinHash deduplication passes tests

## Key Insights for Implementation

### Assignment 1 Lessons (Learned)
1. **Numerical stability is critical**: Always use log-sum-exp trick
2. **Batch dimensions are tricky**: Use einsum for clarity
3. **Memory efficiency matters**: Streaming is essential for large data
4. **Test coverage is excellent**: 51 tests catch many edge cases

### Assignment 2 Anticipated Challenges
1. **Flash Attention**: Understanding tiling strategy is non-trivial
2. **DDP**: Distributed debugging is hard; test locally first
3. **Triton**: Learning curve for GPU programming
4. **Performance**: Must balance correctness and speed

### Assignment 3 Anticipated Challenges
1. **Fitting**: Log-space regression can be numerically sensitive
2. **Extrapolation**: Uncertainty grows with distance from data
3. **Interpretation**: Results must align with Chinchilla paper
4. **Statistics**: Need confidence intervals, not point estimates

### Assignment 4 Anticipated Challenges
1. **Scale**: Pipelines must handle millions of documents
2. **Quality**: False positives/negatives in classifiers
3. **Deduplication**: MinHash parameter tuning
4. **Integration**: All components must work together

## Next Steps

1. **Immediate**: Start A2 implementation
   - Begin with Flash Attention PyTorch version
   - Set up DDP testing framework
   
2. **Parallel**: Begin A4 implementation
   - Text extraction is independent
   - Can make progress while A2 is in progress

3. **Later**: A3 analysis
   - Can be done anytime
   - Requires careful statistical analysis

4. **Final**: Integration and GPU testing
   - Run full training experiments
   - Generate final report

---

## Appendix: Test Coverage Summary

### A1 Test Files (All Passing ✅)
- `tests/test_data.py`: Data loading tests
- `tests/test_model.py`: Model architecture tests (12 tests)
- `tests/test_nn_utils.py`: Softmax, cross-entropy, gradient clipping
- `tests/test_optimizer.py`: AdamW, learning rate schedule
- `tests/test_serialization.py`: Checkpointing
- `tests/test_tokenizer.py`: BPE tokenizer tests (27 tests)
- `tests/test_train_bpe.py`: BPE training tests (3 tests)

**Total**: 51 tests, all passing ✅

### A2 Test Files (Not Yet Implemented)
- `tests/test_attention.py`: Flash Attention tests
- `tests/test_ddp_individual.py`: Individual parameter DDP
- `tests/test_ddp_bucketed.py`: Bucketed DDP
- `tests/test_sharded_optimizer.py`: Sharded optimizer

### A3 Test Files
- No automated tests; manual evaluation via plots and write-up

### A4 Test Files (Not Yet Implemented)
- `tests/test_extract.py`: Text extraction
- `tests/test_langid.py`: Language identification
- `tests/test_pii.py`: PII masking
- `tests/test_toxicity.py`: Toxicity detection
- `tests/test_quality.py`: Quality filtering
- `tests/test_dedup.py`: Deduplication

---

**Document Version**: 1.0
**Last Updated**: March 5, 2026
**Author**: CS336 Implementation Team
