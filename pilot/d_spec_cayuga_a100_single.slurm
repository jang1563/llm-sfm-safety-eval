#!/bin/bash
#SBATCH --job-name=d_spec_a100_1
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d_spec_a100_1_%x_%j.out
#SBATCH --error=logs/d_spec_a100_1_%x_%j.err
#
# d_spec_cayuga_a100_single.slurm — Single A100 80GB job
# For models in 20-65 GB VRAM range:
#   gemma3-12b, gemma3-27b, gemma4-31b, gemma4-moe, llama70-awq, qwen72-awq

set -euo pipefail

MODEL_KEY="${1:-gemma4-31b}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "D Spec Variant B — 1× A100 80GB"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=============================================="
echo

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

echo "Python: $(python3 --version)"
echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

cd "${PROJECT_DIR}"

if [ ! -f d_spec_sequences.json ]; then
    python3 d_spec_setup_sequences.py
fi

echo "Starting vLLM inference for ${MODEL_KEY}..."
echo "Start time: $(date)"

python3 d_spec_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo "End time: $(date)"
echo "Done."
