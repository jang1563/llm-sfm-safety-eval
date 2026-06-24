#!/bin/bash
#SBATCH --job-name=d_spec_mut
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/d_spec_mut_%x_%j.out
#SBATCH --error=logs/d_spec_mut_%x_%j.err

set -euo pipefail
MODEL_KEY="${1:-gemma4-moe}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs
source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
export HF_HOME="${HF_CACHE}" TRANSFORMERS_CACHE="${HF_CACHE}" HF_HUB_CACHE="${HF_CACHE}"

cd "${PROJECT_DIR}"
echo "Model: ${MODEL_KEY}, Node: $(hostname), Date: $(date)"
python3 d_spec_mutation_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"
echo "Done: $(date)"
