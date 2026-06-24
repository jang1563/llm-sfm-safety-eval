#!/bin/bash
#SBATCH --job-name=dna_domain_a100
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dna_domain_%x_%j.out
#SBATCH --error=logs/dna_domain_%x_%j.err
#
# dna_domain_cayuga_a100.slurm — A100 variant of dna_domain_cayuga.slurm.
#
# For the LARGE open-weight models that OOM on a 44GB A40 but fit on a single
# 80GB A100 at tensor_parallel_size=1 (per d_spec_vllm.MODELS comments):
#   gemma4-31b      62GB bf16
#   gemma4-moe      52GB bf16
#   qwen3-30b-moe   60GB bf16
# The small models (mistral, qwen, ministral3-*, qwen3-4b, phi4-mini) use the
# a40 script (dna_domain_cayuga.slurm). The runner reads tensor_parallel_size
# from the model config, so a single A100 with TP=1 is sufficient here.
#
# Usage:
#   /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch dna_domain_cayuga_a100.slurm gemma4-31b
#   ... gemma4-moe | qwen3-30b-moe

set -euo pipefail

MODEL_KEY="${1:-gemma4-31b}"
REPEATS="${2:-25}"
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
ENV_NAME="d_spec_vllm"

mkdir -p logs

echo "=============================================="
echo "DNA Domain — Cayuga A100 Inference (open-weight, large models)"
echo "=============================================="
echo "Model: ${MODEL_KEY}   Repeats: ${REPEATS}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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
    echo "HF_TOKEN: not set"
fi
echo

cd "${PROJECT_DIR}"
echo "Working directory: ${PROJECT_DIR}"

for f in dna_config.py dna_sequences.json dna_domain_probe_vllm.py d_spec_vllm.py d_spec_config.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file $f not found in ${PROJECT_DIR}. rsync it from the laptop pilot/ dir."
        exit 1
    fi
done

echo "Starting vLLM DNA probe for ${MODEL_KEY}..."
echo "Start time: $(date)"
echo

python3 dna_domain_probe_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}" --repeats "${REPEATS}"

echo
echo "End time: $(date)"
echo "Done."
