#!/bin/bash
# run_on_phobos.sh — Run D spec Variant B on Cayuga Phobos node
#
# Usage:
#   1. Copy project to Phobos scratch
#   2. SSH to Phobos
#   3. Run this script in a tmux/screen session
#
# Requirements: outbound HTTPS to api.anthropic.com

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
CONDA_INIT="${HOME}/miniconda3/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="d_spec_env"

# ============================================================================
# Step 1: Setup environment
# ============================================================================
echo "=== Setting up environment ==="

# Initialize conda
if [ -f "${CONDA_INIT}" ]; then
    source "${CONDA_INIT}"
else
    echo "ERROR: Conda not found at ${CONDA_INIT}"
    echo "Try: source ~/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

# Create env if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"

# Install dependencies
pip install --quiet anthropic

echo "Python: $(python3 --version)"
echo "anthropic: $(python3 -c 'import anthropic; print(anthropic.__version__)')"

# ============================================================================
# Step 2: Check API key
# ============================================================================
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo ""
    echo "Option A — export directly:"
    echo "  export ANTHROPIC_API_KEY='<your-anthropic-api-key>'"
    echo ""
    echo "Option B — create a .env file:"
    echo "  echo 'export ANTHROPIC_API_KEY=<your-anthropic-api-key>' > ${PROJECT_DIR}/.env"
    echo "  source ${PROJECT_DIR}/.env"
    echo ""
    exit 1
fi

echo "API key: set"

# ============================================================================
# Step 3: Check network access
# ============================================================================
echo ""
echo "=== Checking network access to api.anthropic.com ==="
if curl -s --max-time 10 https://api.anthropic.com/ > /dev/null 2>&1; then
    echo "✓ api.anthropic.com reachable"
else
    echo "✗ api.anthropic.com NOT reachable"
    echo ""
    echo "Phobos may not have outbound internet. Try:"
    echo "  1. Login node instead (cayuga-login1)"
    echo "  2. Or check proxy: echo \$https_proxy"
    echo "  3. Or ask sysadmin about firewall rules"
    exit 1
fi

# ============================================================================
# Step 4: Run experiment
# ============================================================================
echo ""
echo "=== Running D Spec Variant B ==="
echo "Working directory: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Dry run first
echo ""
echo "--- Dry run (no API calls) ---"
python3 d_spec_variant_b.py --dry-run 2>&1 | head -20
echo "..."
echo ""

read -p "Proceed with actual experiment? (y/N): " confirm
if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# Actual run
echo ""
echo "--- Starting experiment (600 API calls, ~30 min) ---"
echo "Started: $(date)"
echo ""

python3 d_spec_variant_b.py 2>&1 | tee "variant_b_run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Finished: $(date)"
echo "Results saved to results/ directory"
echo ""
echo "=== Done ==="
