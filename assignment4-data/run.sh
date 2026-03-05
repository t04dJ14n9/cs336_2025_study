#!/bin/bash
# =============================================================================
# CS336 Assignment 4: Data Processing Pipeline
# =============================================================================
# This script runs the complete data processing pipeline:
# 1. Text extraction from HTML
# 2. Language identification
# 3. PII masking
# 4. Quality filtering (NSFW, toxicity, Gopher)
# 5. Deduplication (exact & MinHash)
#
# Usage:
#   ./run.sh                           # Run full pipeline with sample data
#   ./run.sh --input data/raw          # Process custom input directory
#   ./run.sh --output data/processed   # Custom output directory
#   ./run.sh --steps extract,lang,pii  # Run specific steps only
# =============================================================================

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
INPUT_DIR="data/sample"
OUTPUT_DIR="data/processed"
STEPS="all"
NUM_WORKERS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input DIR        Input directory with raw data (default: data/sample)"
            echo "  --output DIR       Output directory for processed data (default: data/processed)"
            echo "  --steps STEPS      Comma-separated steps (default: all)"
            echo "                     Options: extract,lang,pii,quality,dedup,all"
            echo "  --workers N        Number of parallel workers (default: 4)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Pipeline Steps:"
            echo "  1. extract    - Extract text from HTML files"
            echo "  2. lang       - Identify and filter by language"
            echo "  3. pii        - Mask PII (emails, phones, etc.)"
            echo "  4. quality    - Apply quality filters (NSFW, toxicity, Gopher)"
            echo "  5. dedup      - Remove duplicates (exact + MinHash)"
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
echo -e "${BLUE}CS336 Assignment 4: Data Processing Pipeline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Change to assignment directory
cd "$(dirname "$0")"

# Check for external dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

check_dependency() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓ $1 installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $1 not installed (required for $2)${NC}"
        return 1
    fi
}

# Check optional dependencies
DEPS_OK=true
check_dependency "resiliparse" "HTML extraction" || DEPS_OK=false
check_dependency "fasttext" "language identification" || DEPS_OK=false
check_dependency "datasketch" "MinHash deduplication" || DEPS_OK=false

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}Some dependencies are missing. Install with:${NC}"
    echo -e "  ${BLUE}pip install resiliparse fasttext datasketch${NC}"
    echo ""
    echo -e "${YELLOW}Continuing with available components...${NC}"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# =============================================================================
# Pipeline Functions
# =============================================================================

run_extraction() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 1: Text Extraction from HTML${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Extracting text from HTML files...${NC}"
    python3 -c "
import os
import json
from cs336_data.data_pipeline import extract_text_from_html

input_dir = '${INPUT_DIR}'
output_file = '${OUTPUT_DIR}/extracted.jsonl'

print(f'Processing HTML files from {input_dir}...')
extracted_count = 0

with open(output_file, 'w') as out_f:
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.html') or file.endswith('.htm'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    text = extract_text_from_html(html_content)
                    if text and len(text) > 100:  # Filter empty/short texts
                        out_f.write(json.dumps({'text': text, 'source': filepath}) + '\\n')
                        extracted_count += 1
                except Exception as e:
                    print(f'Error processing {filepath}: {e}')

print(f'Extracted {extracted_count} documents to {output_file}')
" 2>&1 | tee logs/extraction.log
    
    echo -e "${GREEN}✓ Text extraction complete${NC}"
}

run_language_id() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 2: Language Identification${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Identifying languages and filtering English documents...${NC}"
    python3 -c "
import json
from cs336_data.data_pipeline import identify_language

input_file = '${OUTPUT_DIR}/extracted.jsonl'
output_file = '${OUTPUT_DIR}/lang_filtered.jsonl'

print(f'Filtering documents by language...')
kept_count = 0
total_count = 0

with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
    for line in in_f:
        doc = json.loads(line)
        total_count += 1
        
        try:
            lang = identify_language(doc['text'])
            if lang == 'en':  # Keep only English
                doc['language'] = lang
                out_f.write(json.dumps(doc) + '\\n')
                kept_count += 1
        except Exception as e:
            print(f'Language ID error: {e}')

print(f'Kept {kept_count}/{total_count} documents ({kept_count/total_count*100:.1f}%)')
" 2>&1 | tee logs/language_id.log
    
    echo -e "${GREEN}✓ Language identification complete${NC}"
}

run_pii_masking() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 3: PII Masking${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Masking PII (emails, phones, IPs, etc.)...${NC}"
    python3 -c "
import json
from cs336_data.data_pipeline import mask_pii

input_file = '${OUTPUT_DIR}/lang_filtered.jsonl'
output_file = '${OUTPUT_DIR}/pii_masked.jsonl'

print(f'Masking PII in documents...')
processed_count = 0
pii_instances = 0

with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
    for line in in_f:
        doc = json.loads(line)
        
        try:
            masked_text, count = mask_pii(doc['text'])
            doc['text'] = masked_text
            doc['pii_count'] = count
            pii_instances += count
            out_f.write(json.dumps(doc) + '\\n')
            processed_count += 1
        except Exception as e:
            print(f'PII masking error: {e}')

print(f'Processed {processed_count} documents, masked {pii_instances} PII instances')
" 2>&1 | tee logs/pii_masking.log
    
    echo -e "${GREEN}✓ PII masking complete${NC}"
}

run_quality_filters() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 4: Quality Filtering${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Applying quality filters (NSFW, toxicity, Gopher)...${NC}"
    python3 -c "
import json
from cs336_data.data_pipeline import (
    filter_nswf,
    filter_toxicity,
    filter_gopher_quality
)

input_file = '${OUTPUT_DIR}/pii_masked.jsonl'
output_file = '${OUTPUT_DIR}/quality_filtered.jsonl'

print(f'Applying quality filters...')
kept_count = 0
total_count = 0
rejected = {'nsfw': 0, 'toxicity': 0, 'gopher': 0}

with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
    for line in in_f:
        doc = json.loads(line)
        total_count += 1
        text = doc['text']
        
        # Apply filters
        if filter_nswf(text):
            rejected['nsfw'] += 1
            continue
        if filter_toxicity(text):
            rejected['toxicity'] += 1
            continue
        if not filter_gopher_quality(text):
            rejected['gopher'] += 1
            continue
        
        doc['quality_filtered'] = True
        out_f.write(json.dumps(doc) + '\\n')
        kept_count += 1

print(f'Kept {kept_count}/{total_count} documents ({kept_count/total_count*100:.1f}%)')
print(f'Rejected: NSFW={rejected[\"nsfw\"]}, Toxicity={rejected[\"toxicity\"]}, Gopher={rejected[\"gopher\"]}')
" 2>&1 | tee logs/quality_filters.log
    
    echo -e "${GREEN}✓ Quality filtering complete${NC}"
}

run_deduplication() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 5: Deduplication${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}Removing duplicates (exact + MinHash)...${NC}"
    python3 -c "
import json
from cs336_data.data_pipeline import deduplicate_exact, deduplicate_minhash

input_file = '${OUTPUT_DIR}/quality_filtered.jsonl'
output_exact = '${OUTPUT_DIR}/dedup_exact.jsonl'
output_final = '${OUTPUT_DIR}/final.jsonl'

print(f'Removing exact duplicates...')
# Exact deduplication
seen_hashes = set()
kept_count = 0
total_count = 0

with open(input_file, 'r') as in_f, open(output_exact, 'w') as out_f:
    for line in in_f:
        doc = json.loads(line)
        total_count += 1
        
        try:
            is_dup, hash_val = deduplicate_exact(doc['text'])
            if not is_dup:
                seen_hashes.add(hash_val)
                out_f.write(json.dumps(doc) + '\\n')
                kept_count += 1
        except Exception as e:
            print(f'Exact dedup error: {e}')

print(f'After exact dedup: {kept_count}/{total_count} documents')

print(f'\\nRemoving near-duplicates with MinHash...')
# MinHash deduplication
documents = []
with open(output_exact, 'r') as f:
    for line in f:
        documents.append(json.loads(line))

try:
    unique_docs = deduplicate_minhash(documents, threshold=0.8)
    
    with open(output_final, 'w') as out_f:
        for doc in unique_docs:
            out_f.write(json.dumps(doc) + '\\n')
    
    print(f'After MinHash dedup: {len(unique_docs)}/{len(documents)} documents')
except Exception as e:
    print(f'MinHash dedup error: {e}')
    print('Using exact dedup results as final output')
    import shutil
    shutil.copy(output_exact, output_final)
" 2>&1 | tee logs/deduplication.log
    
    echo -e "${GREEN}✓ Deduplication complete${NC}"
}

# =============================================================================
# Run Pipeline
# =============================================================================

echo -e "${BLUE}Pipeline Configuration:${NC}"
echo -e "  Input:          ${INPUT_DIR}"
echo -e "  Output:         ${OUTPUT_DIR}"
echo -e "  Steps:          ${STEPS}"
echo -e "  Workers:        ${NUM_WORKERS}"
echo ""

# Parse and run steps
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"

if [ "$STEPS" = "all" ]; then
    run_extraction
    run_language_id
    run_pii_masking
    run_quality_filters
    run_deduplication
else
    for step in "${STEP_ARRAY[@]}"; do
        case "$step" in
            extract) run_extraction ;;
            lang) run_language_id ;;
            pii) run_pii_masking ;;
            quality) run_quality_filters ;;
            dedup) run_deduplication ;;
            *) echo -e "${RED}Unknown step: $step${NC}" ;;
        esac
    done
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Count final documents
if [ -f "${OUTPUT_DIR}/final.jsonl" ]; then
    FINAL_COUNT=$(wc -l < "${OUTPUT_DIR}/final.jsonl")
    echo -e "${GREEN}Final dataset: ${FINAL_COUNT} documents${NC}"
fi

echo -e "Output files:"
echo -e "  ${BLUE}${OUTPUT_DIR}/final.jsonl${NC}           - Final processed dataset"
echo -e "  ${BLUE}${OUTPUT_DIR}/extracted.jsonl${NC}        - After extraction"
echo -e "  ${BLUE}${OUTPUT_DIR}/lang_filtered.jsonl${NC}    - After language filtering"
echo -e "  ${BLUE}${OUTPUT_DIR}/pii_masked.jsonl${NC}       - After PII masking"
echo -e "  ${BLUE}${OUTPUT_DIR}/quality_filtered.jsonl${NC} - After quality filters"
echo ""
echo -e "Logs:"
echo -e "  ${BLUE}logs/extraction.log${NC}"
echo -e "  ${BLUE}logs/language_id.log${NC}"
echo -e "  ${BLUE}logs/pii_masking.log${NC}"
echo -e "  ${BLUE}logs/quality_filters.log${NC}"
echo -e "  ${BLUE}logs/deduplication.log${NC}"
echo ""
