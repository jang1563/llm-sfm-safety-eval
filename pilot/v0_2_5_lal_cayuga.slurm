#!/bin/bash
#SBATCH --job-name=v025_lal
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v025_lal_%x_%j.out
#SBATCH --error=logs/v025_lal_%x_%j.err
#
# v0_2_5_lal_cayuga.slurm — LAL multi-turn loop on Cayuga GPU
#
# Usage (use corrected SLURM binary):
#   /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch v0_2_5_lal_cayuga.slurm <model_key>
#
# Single A40 (48GB):
#   sbatch v0_2_5_lal_cayuga.slurm mistral         # 7B
#   sbatch v0_2_5_lal_cayuga.slurm qwen             # 7B
#   sbatch v0_2_5_lal_cayuga.slurm qwen3-4b         # 4B
#   sbatch v0_2_5_lal_cayuga.slurm phi4-mini         # 3.8B
#   sbatch v0_2_5_lal_cayuga.slurm ministral3-8b     # 8B
#   sbatch v0_2_5_lal_cayuga.slurm ministral3-14b    # 14B
#   sbatch v0_2_5_lal_cayuga.slurm gemma4-31b        # 31B
#
# For >48GB models, override gres:
#   sbatch --gres=gpu:a100:1 v0_2_5_lal_cayuga.slurm gemma4-moe
#   sbatch --gres=gpu:a100:1 v0_2_5_lal_cayuga.slurm qwen3-30b-moe

set -euo pipefail

MODEL_KEY="${1:-mistral}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "v0.2.5 LAL — Cayuga GPU Multi-Turn"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================================="
echo

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh

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

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (OK for Mistral, Qwen, Phi)"
fi
echo

cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"
ls -la v0_2_5_lal_vllm.py d_spec_vllm.py 2>&1 | head -5
echo

echo "Starting v0.2.5 LAL vLLM for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 v0_2_5_lal_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo
echo "End time: $(date)"
echo "Done."
