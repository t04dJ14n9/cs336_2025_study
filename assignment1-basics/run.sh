#!/bin/bash
# =============================================================================
# CS336 Assignment 1: Tokenization and Transformer Training
# =============================================================================
# This script handles the complete pipeline:
# 1. Check dependencies and setup
# 2. Tokenize training data (if not already done)
# 3. Train transformer language model
# 
# Usage:
#   ./run.sh                           # Default: TinyStories dataset
#   ./run.sh --dataset owt             # OpenWebText dataset
#   ./run.sh --train-tokenizer         # Force retrain tokenizer
#   ./run.sh --quick                   # Quick training for testing
#   ./run.sh --full                    # Full training (larger model)
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DATASET="tiny_stories"
TRAIN_TOKENIZER=false
QUICK_MODE=false
FULL_MODE=false
DEVICE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --train-tokenizer)
            TRAIN_TOKENIZER=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
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
            echo "  --dataset DATASET       Dataset to use: tiny_stories (default) or owt"
            echo "  --train-tokenizer       Force retrain the BPE tokenizer"
            echo "  --quick                 Quick training mode (small model, fewer iterations)"
            echo "  --full                  Full training mode (larger model, more iterations)"
            echo "  --device DEVICE         Device to use: cpu, cuda, cuda:0, etc."
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                # Train on TinyStories (default)"
            echo "  $0 --dataset owt                  # Train on OpenWebText"
            echo "  $0 --quick                        # Quick test run"
            echo "  $0 --full --dataset owt           # Full training on OWT"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Environment Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python version: ${PYTHON_VERSION}${NC}"

# Check if we're in the right directory
if [ ! -f "tokenize_data.py" ] || [ ! -f "train.py" ]; then
    echo -e "${RED}Error: Please run this script from the assignment1-basics directory${NC}"
    exit 1
fi

# Detect device
if [ -z "$DEVICE" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda"
        echo -e "${GREEN}✓ GPU detected: Using CUDA${NC}"
    else
        DEVICE="cpu"
        echo -e "${YELLOW}⚠ No GPU detected: Using CPU (training will be slower)${NC}"
    fi
else
    echo -e "${GREEN}✓ Using device: ${DEVICE}${NC}"
fi

# Create necessary directories
mkdir -p data checkpoints logs

# =============================================================================
# Step 2: Data Preparation
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Data Preparation & Tokenization${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if raw data files exist
if [ "$DATASET" = "tiny_stories" ]; then
    TRAIN_FILE="data/TinyStoriesV2-GPT4-train.txt"
    VALID_FILE="data/TinyStoriesV2-GPT4-valid.txt"
    TOKENIZED_TRAIN="data/TinyStoriesV2-GPT4-train_tokens.npy"
    TOKENIZED_VALID="data/TinyStoriesV2-GPT4-valid_tokens.npy"
elif [ "$DATASET" = "owt" ]; then
    TRAIN_FILE="data/owt_train.txt"
    VALID_FILE="data/owt_valid.txt"
    TOKENIZED_TRAIN="data/owt_train_tokens.npy"
    TOKENIZED_VALID="data/owt_valid_tokens.npy"
else
    echo -e "${RED}Error: Unknown dataset: ${DATASET}${NC}"
    exit 1
fi

# Check if raw data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "${RED}Error: Training data not found: ${TRAIN_FILE}${NC}"
    echo -e "${YELLOW}Please ensure data files are in the data/ directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found training data: ${TRAIN_FILE}${NC}"

# Check if we need to tokenize
TOKENIZE_NEEDED=false
if [ ! -f "$TOKENIZED_TRAIN" ] || [ ! -f "$TOKENIZED_VALID" ]; then
    TOKENIZE_NEEDED=true
    echo -e "${YELLOW}Tokenized data not found, will tokenize...${NC}"
fi

if [ "$TRAIN_TOKENIZER" = true ]; then
    TOKENIZE_NEEDED=true
    echo -e "${YELLOW}Retraining tokenizer as requested...${NC}"
fi

# Run tokenization if needed
if [ "$TOKENIZE_NEEDED" = true ]; then
    echo ""
    echo -e "${YELLOW}Running tokenization (this may take a few minutes)...${NC}"
    
    TOKENIZE_CMD="python3 tokenize_data.py --dataset ${DATASET} --num_processes 4"
    if [ "$TRAIN_TOKENIZER" = true ]; then
        TOKENIZE_CMD="${TOKENIZE_CMD} --train_tokenizer"
    fi
    
    echo -e "${BLUE}Command: ${TOKENIZE_CMD}${NC}"
    eval $TOKENIZE_CMD
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Tokenization complete${NC}"
    else
        echo -e "${RED}Error: Tokenization failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Tokenized data already exists, skipping tokenization${NC}"
    echo -e "  ${TOKENIZED_TRAIN}"
    echo -e "  ${TOKENIZED_VALID}"
fi

# =============================================================================
# Step 3: Transformer Training
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Transformer Training${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Set training parameters based on mode
if [ "$QUICK_MODE" = true ]; then
    # Quick mode: small model, few iterations for testing
    D_MODEL=128
    NUM_LAYERS=2
    NUM_HEADS=4
    D_FF=256
    BATCH_SIZE=16
    CONTEXT_LENGTH=128
    MAX_ITERS=1000
    EVAL_INTERVAL=100
    CHECKPOINT_INTERVAL=500
    echo -e "${YELLOW}Training in QUICK mode (small model, few iterations)${NC}"
elif [ "$FULL_MODE" = true ]; then
    # Full mode: larger model for better performance
    if [ "$DATASET" = "tiny_stories" ]; then
        D_MODEL=512
        NUM_LAYERS=6
        NUM_HEADS=8
        D_FF=1024
        BATCH_SIZE=64
        CONTEXT_LENGTH=256
        MAX_ITERS=20000
        EVAL_INTERVAL=500
        CHECKPOINT_INTERVAL=2000
    else  # owt
        D_MODEL=768
        NUM_LAYERS=12
        NUM_HEADS=12
        D_FF=2048
        BATCH_SIZE=32
        CONTEXT_LENGTH=512
        MAX_ITERS=50000
        EVAL_INTERVAL=1000
        CHECKPOINT_INTERVAL=5000
    fi
    echo -e "${YELLOW}Training in FULL mode (large model, many iterations)${NC}"
else
    # Default mode: balanced
    D_MODEL=256
    NUM_LAYERS=4
    NUM_HEADS=4
    D_FF=512
    BATCH_SIZE=32
    CONTEXT_LENGTH=256
    MAX_ITERS=5000
    EVAL_INTERVAL=250
    CHECKPOINT_INTERVAL=1000
    echo -e "${GREEN}Training in DEFAULT mode (balanced)${NC}"
fi

# Display training configuration
echo ""
echo -e "${BLUE}Training Configuration:${NC}"
echo -e "  Dataset:         ${DATASET}"
echo -e "  Model dimension:  ${D_MODEL}"
echo -e "  Layers:           ${NUM_LAYERS}"
echo -e "  Attention heads:  ${NUM_HEADS}"
echo -e "  FFN dimension:    ${D_FF}"
echo -e "  Batch size:       ${BATCH_SIZE}"
echo -e "  Context length:   ${CONTEXT_LENGTH}"
echo -e "  Max iterations:   ${MAX_ITERS}"
echo -e "  Device:           ${DEVICE}"
echo ""

# Build training command
TRAIN_CMD="python3 train.py \
    --dataset ${DATASET} \
    --d_model ${D_MODEL} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --d_ff ${D_FF} \
    --batch_size ${BATCH_SIZE} \
    --context_length ${CONTEXT_LENGTH} \
    --max_iters ${MAX_ITERS} \
    --eval_interval ${EVAL_INTERVAL} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --device ${DEVICE}"

# Run training
echo -e "${BLUE}Starting training...${NC}"
echo -e "${BLUE}Command: ${TRAIN_CMD}${NC}"
echo ""

eval $TRAIN_CMD

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "Checkpoints saved in: ${BLUE}checkpoints/${DATASET}/${NC}"
    echo -e "Logs saved in:        ${BLUE}logs/${NC}"
    echo ""
    echo -e "To generate text with your trained model:"
    echo -e "  ${BLUE}python3 generate.py --checkpoint checkpoints/${DATASET}/checkpoint_${MAX_ITERS}.pt${NC}"
else
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ Training failed${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Pipeline Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✓ Step 1: Environment setup complete${NC}"
echo -e "${GREEN}✓ Step 2: Data tokenization complete${NC}"
echo -e "${GREEN}✓ Step 3: Transformer training complete${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Monitor training:       ${BLUE}tail -f logs/${DATASET}_train.log${NC}"
echo -e "  2. Generate text:          ${BLUE}python3 generate.py --checkpoint checkpoints/${DATASET}/checkpoint_${MAX_ITERS}.pt${NC}"
echo -e "  3. Resume training:        ${BLUE}./run.sh --resume checkpoints/${DATASET}/checkpoint_5000.pt${NC}"
echo ""
