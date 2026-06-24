#!/bin/bash
#SBATCH --job-name=chem_smoke
#SBATCH --partition=nairr-gpu-shared
#SBATCH --account=crl195
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%x_%j.out
#SBATCH --error=logs/smoke_%x_%j.err
#
# chem_smoke_expanse.slurm — fast 1-GPU validation of the d_spec_vllm_cu124 env
# BEFORE committing to the full 222GB command-a run. Runs on nairr-gpu-shared
# (1 GPU <= the 3-GPU shared-QOS cap). See smoke_cu124.py for the 4 checks.
#
# KEY: do NOT load a cuda toolkit module. torch 2.6.0+cu124 ships its own CUDA-12.4
# runtime; it talks to the node's CUDA-12.0 driver via minor-version compatibility.
# Loading cuda/11.7.1 (as the original SLURM did) would shadow torch's bundled
# libcudart with 11.7 libs and break the ABI.

set -euo pipefail

SCRATCH="/expanse/lustre/scratch/${USER}/temp_project"
PROJECT_DIR="${SCRATCH}/d_spec_experiment"
HF_CACHE="${SCRATCH}/hf_cache"
ENV_PATH="/expanse/lustre/projects/crl195/${USER}/conda_envs/d_spec_vllm_cu124"

mkdir -p logs

# conda init
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: no conda init found"; exit 1
fi

module purge 2>/dev/null || true
module load gpu/0.15.4 anaconda3/2021.05 2>/dev/null || true   # NO cuda toolkit module (see header)

if [ ! -x "${ENV_PATH}/bin/python" ]; then
    echo "ERROR: cu124 env not found at ${ENV_PATH} — run build_d_spec_cu124.sh first."
    exit 1
fi
conda activate "${ENV_PATH}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"
mkdir -p "${HF_CACHE}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

echo "=============================================="
echo "cu124 env SMOKE TEST — $(hostname) — $(date)"
echo "=============================================="
python3 --version
echo "--- nvidia-smi (driver version is the key datum: confirms CUDA-12.0 driver) ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

cd "${PROJECT_DIR}"
python3 smoke_cu124.py
echo "smoke wrapper done: $(date)"
