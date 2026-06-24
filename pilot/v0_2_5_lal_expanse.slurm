#!/bin/bash
#SBATCH --job-name=v0_2_5_lal
#SBATCH --partition=gpu-shared
#SBATCH --account=__ALLOCATION__
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v0_2_5_lal_%x_%j.out
#SBATCH --error=logs/v0_2_5_lal_%x_%j.err
#
# v0_2_5_lal_expanse.slurm — vLLM v0.2.5 LAL multi-turn loop on SDSC Expanse
#
# Usage:
#   sbatch v0_2_5_lal_expanse.slurm <model_key>
#
# Examples:
#   sbatch --gpus=1 v0_2_5_lal_expanse.slurm qwen           # 7B
#   sbatch --gpus=1 v0_2_5_lal_expanse.slurm mistral        # 7B
#   sbatch --gpus=1 v0_2_5_lal_expanse.slurm gemma3-12b     # 12B
#   sbatch --gpus=1 v0_2_5_lal_expanse.slurm gemma3-27b     # 27B (A100 80GB)
#   sbatch --gpus=1 v0_2_5_lal_expanse.slurm llama33-70-awq # 70B AWQ
#   sbatch --gpus=2 v0_2_5_lal_expanse.slurm llama33-70     # 70B bf16
#   sbatch --gpus=2 v0_2_5_lal_expanse.slurm qwen72         # 72B
#   sbatch --gpus=4 v0_2_5_lal_expanse.slurm mixtral-8x22b  # 141B MoE

set -euo pipefail

MODEL_KEY="${1:-qwen}"

# ---------------------------------------------------------------------------
# Expanse paths (EDIT for your account)
# ---------------------------------------------------------------------------
SCRATCH="/expanse/lustre/scratch/${USER}/temp_project"
PROJECT_DIR="${SCRATCH}/d_spec_experiment"   # reuse same project dir as d_spec
HF_CACHE="${SCRATCH}/hf_cache"
ENV_NAME="d_spec_vllm"
CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"

mkdir -p logs

echo "=============================================="
echo "v0.2.5 LAL — Expanse GPU Multi-Turn"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs: ${SLURM_GPUS:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo

module purge
module load gpu/0.15.4 cuda/11.7.1 anaconda3/2021.05 2>/dev/null || true

if [ -f "${CONDA_INIT}" ]; then
    source "${CONDA_INIT}"
fi

if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "Creating env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
    conda activate "${ENV_NAME}"
    pip install --upgrade pip vllm requests
else
    conda activate "${ENV_NAME}"
fi

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"
echo "Files:"
ls -la v0_2_5_lal_vllm.py d_spec_vllm.py 2>&1 | head -5
echo

python3 v0_2_5_lal_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo
echo "End: $(date)"
echo "Done."
