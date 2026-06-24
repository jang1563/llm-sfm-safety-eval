#!/bin/bash
#SBATCH --job-name=d_spec_a100
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/d_spec_a100_%x_%j.out
#SBATCH --error=logs/d_spec_a100_%x_%j.err
#
# d_spec_cayuga_a100.slurm — Submit large-model D spec on 2× A100 80GB
#
# Recommended models (tensor_parallel_size=2):
#   - llama33-70:   Llama 3.3 70B Instruct (bf16)  → requires HF license accept
#   - qwen72:       Qwen 2.5 72B Instruct (bf16)   → no license needed
#   - deepseek-r1-70: DeepSeek-R1-Distill-Llama-70B → no license needed
#   - llama70:      Llama 3.1 70B Instruct (bf16)  → requires HF license accept
#
# Usage:
#   sbatch d_spec_cayuga_a100.slurm qwen72
#   sbatch d_spec_cayuga_a100.slurm deepseek-r1-70

set -euo pipefail

MODEL_KEY="${1:-qwen72}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "D Spec Variant B — Cayuga A100 Inference"
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
