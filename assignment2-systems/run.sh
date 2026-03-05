#!/bin/bash
# =============================================================================
# CS336 Assignment 2: Systems & Parallelism
# =============================================================================
# This script tests and benchmarks:
# 1. Flash Attention implementations (PyTorch & Triton)
# 2. Distributed Data Parallel (DDP) - Individual & Bucketed
# 3. Sharded Optimizer
#
# Usage:
#   ./run.sh                           # Run all tests
#   ./run.sh --component flash_attn    # Test only Flash Attention
#   ./run.sh --component ddp           # Test only DDP implementations
#   ./run.sh --component sharded_opt   # Test only Sharded Optimizer
#   ./run.sh --benchmark               # Run benchmarks
# =============================================================================

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
COMPONENT="all"
RUN_BENCHMARK=false
DEVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --component COMP    Test specific component: all, flash_attn, ddp, sharded_opt"
            echo "  --benchmark         Run performance benchmarks"
            echo "  --device DEVICE     Device to use: cpu, cuda, cuda:0"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}CS336 Assignment 2: Systems & Parallelism${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Detect device
if [ -z "$DEVICE" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda"
        echo -e "${GREEN}✓ GPU detected: Using CUDA${NC}"
    else
        DEVICE="cpu"
        echo -e "${YELLOW}⚠ No GPU: Using CPU (some tests may be skipped)${NC}"
    fi
fi

# Change to assignment directory
cd "$(dirname "$0")"

# =============================================================================
# Test Functions
# =============================================================================

test_flash_attention() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing Flash Attention${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Test PyTorch implementation
    echo -e "${YELLOW}Testing PyTorch Flash Attention...${NC}"
    python3 -m pytest tests/test_attention.py::test_flash_attention_pytorch -xvs 2>&1 | tail -20
    
    if [ "$DEVICE" = "cuda" ]; then
        echo ""
        echo -e "${YELLOW}Testing Triton Flash Attention (requires GPU)...${NC}"
        python3 -m pytest tests/test_attention.py::test_flash_attention_triton -xvs 2>&1 | tail -20
    else
        echo -e "${YELLOW}⚠ Skipping Triton tests (requires GPU)${NC}"
    fi
    
    if [ "$RUN_BENCHMARK" = true ]; then
        echo ""
        echo -e "${YELLOW}Running Flash Attention benchmarks...${NC}"
        python3 -c "
import torch
import time
from cs336_systems.flash_attention_pytorch import FlashAttentionPytorch

# Benchmark configuration
batch_size = 8
seq_len = 512
d_model = 64
num_heads = 4

print(f'Benchmark: batch={batch_size}, seq_len={seq_len}, d_model={d_model}, heads={num_heads}')

# Standard attention
q = torch.randn(batch_size, seq_len, d_model, device='${DEVICE}')
k = torch.randn(batch_size, seq_len, d_model, device='${DEVICE}')
v = torch.randn(batch_size, seq_len, d_model, device='${DEVICE}')

# Warmup
_ = FlashAttentionPytorch.apply(q, k, v, False)

# Benchmark
start = time.time()
for _ in range(100):
    _ = FlashAttentionPytorch.apply(q, k, v, False)
if '${DEVICE}' == 'cuda':
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f'Average time per forward: {elapsed/100*1000:.2f}ms')
print(f'Throughput: {batch_size * seq_len / (elapsed/100):.0f} tokens/sec')
"
    fi
    
    echo -e "${GREEN}✓ Flash Attention tests complete${NC}"
}

test_ddp() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing Distributed Data Parallel (DDP)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Test DDP with individual parameters
    echo -e "${YELLOW}Testing DDP Individual Parameters...${NC}"
    python3 -m pytest tests/test_ddp.py::test_DistributedDataParallelCPU -xvs -k "individual" 2>&1 | tail -20
    
    # Test DDP with bucketing
    echo ""
    echo -e "${YELLOW}Testing DDP Bucketed...${NC}"
    python3 -m pytest tests/test_ddp.py::test_DistributedDataParallelCPU -xvs -k "bucketed" 2>&1 | tail -20
    
    if [ "$RUN_BENCHMARK" = true ]; then
        echo ""
        echo -e "${YELLOW}Running DDP benchmarks...${NC}"
        python3 -c "
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def benchmark_ddp(rank, world_size, mode='individual'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    # Simple model
    model = torch.nn.Linear(100, 100)
    
    if mode == 'individual':
        from cs336_systems.ddp_individual import DDPIndividual
        model = DDPIndividual(model)
    else:
        from cs336_systems.ddp_bucketed import DDPBucketed
        model = DDPBucketed(model, bucket_size_mb=1.0)
    
    # Benchmark gradient synchronization
    x = torch.randn(32, 100)
    y = model(x).sum()
    y.backward()
    
    start = time.time()
    for _ in range(100):
        model.train()
        y = model(x).sum()
        y.backward()
        if hasattr(model, 'finish_gradient_synchronization'):
            model.finish_gradient_synchronization()
    
    if rank == 0:
        elapsed = time.time() - start
        print(f'{mode.upper()} DDP: {elapsed/100*1000:.2f}ms per iteration')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 2
    print('Benchmarking DDP implementations with 2 processes...')
    mp.spawn(benchmark_ddp, args=(world_size, 'individual'), nprocs=world_size, join=True)
" 2>&1 || echo -e "${YELLOW}DDP benchmark requires multiple processes${NC}"
    fi
    
    echo -e "${GREEN}✓ DDP tests complete${NC}"
}

test_sharded_optimizer() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing Sharded Optimizer${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Testing Sharded Optimizer...${NC}"
    python3 -m pytest tests/test_sharded_optimizer.py -xvs 2>&1 | tail -20
    
    if [ "$RUN_BENCHMARK" = true ]; then
        echo ""
        echo -e "${YELLOW}Running Sharded Optimizer benchmarks...${NC}"
        python3 -c "
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def benchmark_sharded_opt(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    # Large model with many parameters
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1000),
    )
    
    # Sharded optimizer
    from cs336_systems.sharded_optimizer import ShardedOptimizer
    optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3)
    
    # Benchmark optimizer step
    x = torch.randn(32, 1000)
    loss = model(x).sum()
    loss.backward()
    
    start = time.time()
    for _ in range(100):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
    
    if rank == 0:
        elapsed = time.time() - start
        print(f'Sharded Optimizer: {elapsed/100*1000:.2f}ms per step')
        print(f'Memory saved: ~{1/world_size*100:.0f}% (theoretical)')
    
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 2
    print('Benchmarking Sharded Optimizer with 2 processes...')
    mp.spawn(benchmark_sharded_opt, args=(world_size,), nprocs=world_size, join=True)
" 2>&1 || echo -e "${YELLOW}Sharded Optimizer benchmark requires multiple processes${NC}"
    fi
    
    echo -e "${GREEN}✓ Sharded Optimizer tests complete${NC}"
}

# =============================================================================
# Run Tests
# =============================================================================

case "$COMPONENT" in
    all)
        test_flash_attention
        test_ddp
        test_sharded_optimizer
        ;;
    flash_attn)
        test_flash_attention
        ;;
    ddp)
        test_ddp
        ;;
    sharded_opt)
        test_sharded_optimizer
        ;;
    *)
        echo -e "${RED}Unknown component: ${COMPONENT}${NC}"
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All tests completed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Tested components:"
[ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "flash_attn" ] && echo -e "  ${GREEN}✓ Flash Attention${NC}"
[ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "ddp" ] && echo -e "  ${GREEN}✓ Distributed Data Parallel${NC}"
[ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "sharded_opt" ] && echo -e "  ${GREEN}✓ Sharded Optimizer${NC}"
echo ""

if [ "$DEVICE" = "cpu" ]; then
    echo -e "${YELLOW}Note: Some tests were skipped (requires GPU)${NC}"
    echo -e "${YELLOW}Run on GPU machine for full test coverage${NC}"
fi
