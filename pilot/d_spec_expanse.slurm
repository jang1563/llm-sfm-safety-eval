#!/bin/bash
#SBATCH --job-name=d_spec_vllm
#SBATCH --partition=gpu-shared
#SBATCH --account=__ALLOCATION__
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/d_spec_%x_%j.out
#SBATCH --error=logs/d_spec_%x_%j.err
#
# d_spec_expanse.slurm — Submit D spec Variant B job on SDSC Expanse
#
# Expanse GPU resources:
#   gpu          — full A100 80GB nodes (4 GPUs per node, exclusive)
#   gpu-shared   — shared A100 80GB GPUs (request --gpus=N)
#
# Usage:
#   1. Edit __ALLOCATION__ above with your XSEDE/ACCESS allocation code
#   2. Edit paths below for your Expanse scratch
#   3. Submit: sbatch d_spec_expanse.slurm <model_key>
#
# Recommended GPUs per model:
#   - 7B/8B models:   --gpus=1
#   - 27B–30B:        --gpus=1 (A100 80GB) or --gpus=2 (smaller GPU)
#   - 70B–72B bf16:   --gpus=2
#   - 70B AWQ 4-bit:  --gpus=1
#   - Mixtral 8x22B:  --gpus=4
#
# Example invocations:
#   sbatch --gpus=1 d_spec_expanse.slurm qwen           # 7B
#   sbatch --gpus=2 d_spec_expanse.slurm llama33-70     # 70B bf16
#   sbatch --gpus=1 d_spec_expanse.slurm llama33-70-awq # 70B AWQ
#   sbatch --gpus=4 d_spec_expanse.slurm mixtral-8x22b  # MoE 141B

set -euo pipefail

MODEL_KEY="${1:-qwen}"

# ---------------------------------------------------------------------------
# Expanse-specific paths (EDIT THESE FOR YOUR ACCOUNT)
# ---------------------------------------------------------------------------
SCRATCH="/expanse/lustre/scratch/${USER}/temp_project"
PROJECT_DIR="${SCRATCH}/d_spec_experiment"
HF_CACHE="${SCRATCH}/hf_cache"
ENV_NAME="d_spec_vllm"
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
mkdir -p logs

echo "=============================================="
echo "D Spec Variant B — Expanse GPU Inference"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs: ${SLURM_GPUS:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Load modules (Expanse-specific)
# ---------------------------------------------------------------------------
module purge
module load gpu/0.15.4 cuda/11.7.1 anaconda3/2021.05 2>/dev/null || true

# Use the local conda if available
if [ -f "${CONDA_INIT}" ]; then
    source "${CONDA_INIT}"
else
    echo "WARNING: Conda init not found at ${CONDA_INIT}"
fi

# ---------------------------------------------------------------------------
# Setup conda env
# ---------------------------------------------------------------------------
if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
    conda activate "${ENV_NAME}"
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
# HuggingFace cache (use Expanse scratch — fast Lustre)
# ---------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (OK for non-gated models)"
fi
echo

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"

echo "Working directory: ${PROJECT_DIR}"
ls -la
echo

if [ ! -f d_spec_sequences.json ]; then
    echo "WARNING: d_spec_sequences.json not found — running setup..."
    python3 d_spec_setup_sequences.py
fi

echo "Starting vLLM inference for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 d_spec_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo
echo "End time: $(date)"
echo "Done."
