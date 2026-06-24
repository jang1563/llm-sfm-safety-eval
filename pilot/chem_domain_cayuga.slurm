#!/bin/bash
#SBATCH --job-name=chem_domain_vllm
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chem_domain_%x_%j.out
#SBATCH --error=logs/chem_domain_%x_%j.err
#
# chem_domain_cayuga.slurm — open-weight chemistry identifier-gradient probe.
#
# Tests the MODEL HALF of the mirror-image finding (F-C6): do open-weight models
# also fail to resolve the CWC PubChem CIDs (ID-3) while resolving fentanyl's
# CID 3345?  Reuses the d_spec_vllm model panel + conda env for parity with the
# protein open-weight run.
#
# Requires in PROJECT_DIR (same dir as the protein run):
#   chem_config.py, chem_domain_probe_vllm.py   (this experiment)
#   d_spec_vllm.py, d_spec_config.py            (imported: MODELS, is_refusal)
#
# Usage:
#   /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch chem_domain_cayuga.slurm mistral
#   ... qwen | gemma4-31b | gemma4-moe | qwen3-30b-moe | ministral3-14b |
#       ministral3-8b | qwen3-4b | phi4-mini      (the 9-model protein panel)

set -euo pipefail

MODEL_KEY="${1:-mistral}"
REPEATS="${2:-25}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "Chem Domain — Cayuga GPU Inference (open-weight)"
echo "=============================================="
echo "Model: ${MODEL_KEY}   Repeats: ${REPEATS}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Conda env (reuse the protein vLLM env)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# HuggingFace cache on scratch
# ---------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4}"
else
    echo "HF_TOKEN: not set (OK for Mistral, Qwen, Phi, Gemma-Apache, Ministral)"
fi
echo

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"

for f in chem_config.py chem_domain_probe_vllm.py d_spec_vllm.py d_spec_config.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file $f not found in ${PROJECT_DIR}. rsync it from the laptop pilot/ dir."
        exit 1
    fi
done

echo "Starting vLLM chemistry probe for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 chem_domain_probe_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}" --repeats "${REPEATS}"

echo
echo "End time: $(date)"
echo "Done."
