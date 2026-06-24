#!/bin/bash
#SBATCH --job-name=e1_vllm_a100
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e1_vllm_a100_%x_%A_%a.out
#SBATCH --error=logs/e1_vllm_a100_%x_%A_%a.err
#
# E1_vllm_cayuga_a100.slurm — E1 over-trust open-weight subject collection
#                              (single A100 80GB, 26-32B models)
#
# Companion to E1_vllm_cayuga.slurm (A40, <=14B). Same experiment, same one-model-
# per-process pattern; these models just need 80GB VRAM. All are gate-open per
# Finding 9, so they probe whether the Finding-16 over-trust dissociation generalizes
# across vendors and scales (3.8B-32B once both scripts are run).
#
# LARGE_PANEL (26-32B, single A100 80GB; first download of a 32B model is slow):
#   0 gemma4-31b      Gemma 4 31B       (strong)
#   1 gemma4-moe      Gemma 4 26B-A4B   (strong, MoE)
#   2 qwen3-30b-moe   Qwen 3 30B MoE    (moderate, MoE)
#   3 qwen3-32b       Qwen 3 32B dense  (moderate, large)
#   4 olmo2-32b       OLMo-2 32B        (safety-tuned, fully open)
#
# Usage (corrected SLURM binary — /usr/bin/sbatch v22.05.2 is BROKEN):
#   SB=/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch
#   $SB --array=0-4 E1_vllm_cayuga_a100.slurm   # sweep all 5 large models
#   $SB E1_vllm_cayuga_a100.slurm qwen3-32b     # single model (positional override)

set -euo pipefail

LARGE_PANEL=(gemma4-31b gemma4-moe qwen3-30b-moe qwen3-32b olmo2-32b)

# Model selection: positional arg wins; else SLURM array index; else first panel entry.
if [ -n "${1:-}" ]; then
    MODEL_KEY="$1"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_KEY="${LARGE_PANEL[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_KEY="${LARGE_PANEL[0]}"
fi

SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
# E1 files are staged alongside the proven d_spec_vllm.py / d_spec_config.py.
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "E1 Over-Trust — Cayuga 1x A100 80GB (open-weight subject)"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Array: ${SLURM_ARRAY_TASK_ID:-<none>} / job ${SLURM_JOB_ID:-<none>}"
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

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (Gemma + OLMo gated repos may need it)"
fi
echo

cd "${PROJECT_DIR}"

echo "Working directory: $(pwd)"
if [ ! -f results/E1_value_table_20260530.json ]; then
    echo "ERROR: results/E1_value_table_20260530.json not found — sync the value table first."
    exit 1
fi

echo "Starting vLLM inference for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 E1_overtrust_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}"

echo
echo "End time: $(date)"
echo "Done."
