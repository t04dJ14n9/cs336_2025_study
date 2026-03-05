#!/bin/bash
# =============================================================================
# CS336 Assignment 3: Scaling Laws
# =============================================================================
# This script analyzes transformer scaling laws:
# 1. Load isoFLOP curves data
# 2. Fit scaling law parameters (E, A, B, α, β)
# 3. Generate predictions for optimal compute
# 4. Create visualizations
#
# Usage:
#   ./run.sh                           # Run full analysis
#   ./run.sh --predict 1e19            # Predict optimal config at 10^19 FLOPs
#   ./run.sh --predict 1e20            # Predict optimal config at 10^20 FLOPs
#   ./run.sh --visualize               # Generate plots only
# =============================================================================

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
PREDICT_FLOPS=""
VISUALIZE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --predict)
            PREDICT_FLOPS="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --predict FLOPS    Predict optimal config at given FLOPs (e.g., 1e19)"
            echo "  --visualize        Generate visualization plots only"
            echo "  -h, --help         Show this help"
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
echo -e "${BLUE}CS336 Assignment 3: Scaling Laws Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Change to assignment directory
cd "$(dirname "$0")"

# Check if data exists
if [ ! -f "data/isoflops_curves.json" ]; then
    echo -e "${RED}Error: Data file not found: data/isoflops_curves.json${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found isoFLOP curves data${NC}"

# Create output directories
mkdir -p output plots

# =============================================================================
# Run Analysis
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Running Scaling Law Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$VISUALIZE_ONLY" = true ]; then
    # Generate plots only
    echo -e "${YELLOW}Generating visualization plots...${NC}"
    python3 scaling_analysis.py --visualize-only 2>&1 || {
        echo -e "${YELLOW}Note: --visualize-only flag not implemented, running full analysis${NC}"
        python3 scaling_analysis.py 2>&1 | tail -40
    }
elif [ -n "$PREDICT_FLOPS" ]; then
    # Predict at specific compute
    echo -e "${YELLOW}Predicting optimal configuration at ${PREDICT_FLOPS} FLOPs...${NC}"
    python3 scaling_analysis.py --predict "$PREDICT_FLOPS" 2>&1 || {
        echo -e "${YELLOW}Note: --predict flag not implemented, running full analysis${NC}"
        python3 scaling_analysis.py 2>&1 | tail -40
    }
else
    # Run full analysis
    echo -e "${YELLOW}Running full scaling law analysis...${NC}"
    python3 scaling_analysis.py 2>&1 | tee output/analysis.log | tail -40
fi

# Check if analysis succeeded
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Analysis completed successfully${NC}"
else
    echo -e "${RED}✗ Analysis failed${NC}"
    exit 1
fi

# =============================================================================
# Display Results
# =============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Results${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Show fitted parameters
if [ -f "output/fitted_parameters.json" ]; then
    echo -e "${GREEN}Fitted Scaling Law Parameters:${NC}"
    cat output/fitted_parameters.json | python3 -m json.tool 2>/dev/null || cat output/fitted_parameters.json
    echo ""
fi

# Show predictions
if [ -f "output/predictions.json" ]; then
    echo -e "${GREEN}Predictions:${NC}"
    cat output/predictions.json | python3 -m json.tool 2>/dev/null || cat output/predictions.json
    echo ""
fi

# Check for plots
if [ -f "plots/scaling_laws.png" ]; then
    echo -e "${GREEN}✓ Visualization saved: plots/scaling_laws.png${NC}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Analysis Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Output files:"
echo -e "  ${BLUE}output/analysis.log${NC}           - Full analysis log"
echo -e "  ${BLUE}output/fitted_parameters.json${NC} - Fitted scaling law parameters"
echo -e "  ${BLUE}output/predictions.json${NC}       - Optimal config predictions"
echo -e "  ${BLUE}plots/scaling_laws.png${NC}        - Visualization plots"
echo ""
echo -e "Next steps:"
echo -e "  ${BLUE}./run.sh --predict 1e20${NC}       - Predict at 10^20 FLOPs"
echo -e "  ${BLUE}python3 scaling_analysis.py${NC}   - Run analysis with custom parameters"
echo ""
