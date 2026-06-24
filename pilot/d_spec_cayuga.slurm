#!/bin/bash
#SBATCH --job-name=d_spec_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d_spec_%x_%j.out
#SBATCH --error=logs/d_spec_%x_%j.err
#
# d_spec_cayuga.slurm — Submit D spec Variant B job on Cayuga GPU node
#
# Usage:
#   sbatch d_spec_cayuga.slurm <model_key>
#   sbatch d_spec_cayuga.slurm mistral
#   sbatch d_spec_cayuga.slurm qwen
#   sbatch d_spec_cayuga.slurm llama   # requires HF_TOKEN + license acceptance
#
# Submit with corrected SLURM binary:
#   /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch d_spec_cayuga.slurm mistral

set -euo pipefail

MODEL_KEY="${1:-mistral}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

# Create log directory
mkdir -p logs

echo "=============================================="
echo "D Spec Variant B — Cayuga GPU Inference"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Setup conda environment
# ---------------------------------------------------------------------------
source ~/miniconda3/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y

    conda activate "${ENV_NAME}"

    echo "Installing vllm and dependencies..."
    pip install --upgrade pip
    pip install vllm requests
else
    conda activate "${ENV_NAME}"
fi

echo "Python: $(python3 --version)"
echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo

# ---------------------------------------------------------------------------
# Setup HuggingFace cache (use scratch)
# ---------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

# Set HF_TOKEN if available (needed for Llama, Gemma)
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (OK for Mistral, Qwen)"
fi
echo

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
ls -la
echo

# Check sequence file exists
if [ ! -f d_spec_sequences.json ]; then
    echo "WARNING: d_spec_sequences.json not found"
    echo "Running setup script..."
    python3 d_spec_setup_sequences.py
fi

echo "Starting vLLM inference for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 d_spec_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo
echo "End time: $(date)"
echo "Done."
