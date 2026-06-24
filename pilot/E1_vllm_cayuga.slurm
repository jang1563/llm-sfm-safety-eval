#!/bin/bash
#SBATCH --job-name=e1_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/e1_vllm_%x_%A_%a.out
#SBATCH --error=logs/e1_vllm_%x_%A_%a.err
#
# E1_vllm_cayuga.slurm — E1 over-trust open-weight subject collection (A40, small models)
#
# Tests whether Finding 16's reliability-discrimination-without-hazard-relevance
# dissociation (the sole gate-open Claude, Sonnet 4) is Claude-specific or a general
# property of the LLM x SFM interpretation interface. Open-weight models are all
# gate-open (Finding 9), so they are the natural substrate. One model / process.
#
# SMALL_PANEL (<=14B, fits A40 48GB):
#   0 mistral         Mistral 7B v0.3   (minimal safety)
#   1 qwen            Qwen 2.5 7B       (moderate)
#   2 qwen3-4b        Qwen 3 4B         (moderate, small)
#   3 phi4-mini       Phi-4 mini 3.8B   (moderate, tiny)
#   4 ministral3-14b  Ministral 3 14B   (minimal safety, newer)
#
# Usage (corrected SLURM binary — /usr/bin/sbatch v22.05.2 is BROKEN):
#   SB=/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch
#   $SB --array=0-4 E1_vllm_cayuga.slurm        # sweep all 5 small models
#   $SB E1_vllm_cayuga.slurm qwen               # single model (positional override)
#
# Large models (gemma4-31b, gemma4-moe, qwen3-30b-moe, qwen3-32b, olmo2-32b):
#   use E1_vllm_cayuga_a100.slurm (single A100 80GB).

set -euo pipefail

SMALL_PANEL=(mistral qwen qwen3-4b phi4-mini ministral3-14b)

# Model selection: positional arg wins; else SLURM array index; else first panel entry.
if [ -n "${1:-}" ]; then
    MODEL_KEY="$1"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_KEY="${SMALL_PANEL[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_KEY="${SMALL_PANEL[0]}"
fi

SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
# E1 files are staged alongside the proven d_spec_vllm.py / d_spec_config.py.
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "E1 Over-Trust — Cayuga A40 (open-weight subject)"
echo "=============================================="
echo "Model: ${MODEL_KEY}"
echo "Array: ${SLURM_ARRAY_TASK_ID:-<none>} / job ${SLURM_JOB_ID:-<none>}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Setup conda environment (reuse the proven d_spec_vllm env)
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
# HuggingFace cache (scratch)
# ---------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (OK for Mistral, Qwen, Phi; Gemma/OLMo may need it)"
fi
echo

# ---------------------------------------------------------------------------
# Run experiment (E1 files live in pilot/; runner reads results/<value table>)
# ---------------------------------------------------------------------------
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
