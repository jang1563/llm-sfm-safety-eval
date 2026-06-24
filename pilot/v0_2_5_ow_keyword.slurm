#!/bin/bash
#SBATCH --job-name=ow_kw_probe
#SBATCH --output=slurm_ow_kw_%j_%a.out
#SBATCH --error=slurm_ow_kw_%j_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=scu-gpu

SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/Science_FM_Safety}"

MODEL=${1:-qwen}
echo "=== OW Keyword Probe: ${MODEL} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate d_spec_vllm

export HF_HOME=${SCRATCH}/hf_cache
export HF_CACHE=${SCRATCH}/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache

cd "${PROJECT_DIR}/pilot"
python3 v0_2_5_ow_keyword_probe.py --model ${MODEL}
echo "=== DONE ==="
