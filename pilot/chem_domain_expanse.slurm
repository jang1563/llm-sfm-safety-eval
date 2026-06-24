#!/bin/bash
#SBATCH --job-name=chem_cmda_fp8
#SBATCH --partition=nairr-gpu
#SBATCH --account=crl195
#SBATCH --qos=nairr-gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/chem_cmda_%x_%j.out
#SBATCH --error=logs/chem_cmda_%x_%j.err
#
# chem_domain_expanse.slurm — FAITHFUL fp8 command-a chemistry re-run on SDSC Expanse.
#
# Why this exists: command-a was run in the 12-model chemistry panel at nf4 4-bit
# (~3.8B effective capacity) and gave a FLAT / non-load-bearing H1 (CWC PubChem
# CID-resolution). This re-run tests whether that was a quantization artifact, at
# full fp8 precision. fp8 e4m3 needs NATIVE fp8 → Hopper; on Ampere (A100/Cayuga)
# vLLM emulates via Marlin (int32 double-copy) → ~114GB/GPU OOM. So this targets
# the NAIRR H100 partition only.
#
# Target: nairr-gpu (EXCLUSIVE full-node), 4x H100 80GB, tp=4 (command-a-fp8 config in d_spec_vllm.py).
# NOTE: nairr-gpu-shared QOS caps a job at 3 GPUs — too few for tp=4. The exclusive
# nairr-gpu partition (QOS nairr-gpu-normal) allows 4 GPU/node and had 25+ idle H100
# nodes at submit time, so going exclusive costs nothing in queue latency.
# Online dynamic quant: downloads the gated bf16 repo (~222GB) and converts to fp8
# at load. Requires an HF token that has accepted the CohereLabs CC-BY-NC license.
#
# Requires in PROJECT_DIR:
#   chem_config.py, chem_domain_probe_vllm.py   (the experiment)
#   d_spec_vllm.py, d_spec_config.py            (imported: MODELS[command-a-fp8], is_refusal)
#
# Usage:
#   /path/to/sbatch chem_domain_expanse.slurm                 # defaults: command-a-fp8, 25 reps
#   /path/to/sbatch chem_domain_expanse.slurm command-a-fp8 25

set -euo pipefail

MODEL_KEY="${1:-command-a-fp8}"
REPEATS="${2:-25}"
SCRATCH="/expanse/lustre/scratch/${USER}/temp_project"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
# Existing env on the PERSISTENT projects filesystem (survives scratch purge).
# Path-based (outside default envs_dirs) → must activate by full path, not name.
ENV_PATH="/expanse/lustre/projects/crl195/${USER}/conda_envs/d_spec_vllm_cu124"
CONDA_INIT_A="${HOME}/miniconda3/etc/profile.d/conda.sh"
CONDA_INIT_B="${HOME}/miniconda3/miniconda3/etc/profile.d/conda.sh"

mkdir -p logs

echo "=============================================="
echo "Chem Domain — Expanse NAIRR H100 (command-a fp8 faithful)"
echo "=============================================="
echo "Model: ${MODEL_KEY}   Repeats: ${REPEATS}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="
echo

# ---------------------------------------------------------------------------
# Modules + conda (mirror the known-good Expanse vLLM templates)
# ---------------------------------------------------------------------------
module purge 2>/dev/null || true
# NO cuda toolkit module: torch 2.6.0+cu124 bundles its own CUDA-12.4 runtime and
# talks to the node's CUDA-12.0 driver via minor-version compat. Loading cuda/11.7.1
# (as before) would prepend 11.7 libs that shadow torch's bundled libcudart -> ABI break.
module load gpu/0.15.4 anaconda3/2021.05 2>/dev/null || true

if [ -f "${CONDA_INIT_A}" ]; then
    source "${CONDA_INIT_A}"
elif [ -f "${CONDA_INIT_B}" ]; then
    source "${CONDA_INIT_B}"
else
    echo "ERROR: conda init not found (${CONDA_INIT_A} | ${CONDA_INIT_B})"; exit 1
fi

if [ ! -x "${ENV_PATH}/bin/python" ]; then
    echo "ERROR: conda env not found at ${ENV_PATH}"
    exit 1
fi
conda activate "${ENV_PATH}"

echo "Python: $(python3 --version)"
echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo

# ---------------------------------------------------------------------------
# Hard GPU guard — fp8 command-a 111B needs tp=2; tp=1 fallback would OOM.
# ---------------------------------------------------------------------------
echo "GPUs visible:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "${N_GPU}" -lt 4 ]; then
    echo "ERROR: only ${N_GPU} GPU(s) visible but command-a-fp8 needs tp=4. Aborting before OOM."
    exit 1
fi
echo "GPU count OK: ${N_GPU}"
echo

# ---------------------------------------------------------------------------
# Fail fast if compute node is air-gapped and model is not pre-staged.
# (~222GB bf16 download; aborting in ~15s beats dying deep into setup.)
# ---------------------------------------------------------------------------
# Go offline ONLY if the actual weight shards are present. A bare `[ -d ]` check is
# unsafe: the smoke test's hf_hub_download(config.json) leaves a cache dir with ONLY
# config.json, which previously tricked offline mode -> tokenizer/weights 404 offline.
if ls "${HF_CACHE}"/models--CohereLabs--c4ai-command-a-03-2025/snapshots/*/*.safetensors >/dev/null 2>&1; then
    echo "Full weight shards present in cache — loading offline."
    export HF_HUB_OFFLINE=1
else
    echo "Weights not fully cached — online mode (vLLM fetches tokenizer + ~222GB shards into cache)."
    if ! curl -sI --max-time 15 https://huggingface.co >/dev/null 2>&1; then
        echo "ERROR: no internet from compute node AND weights not pre-staged."
        echo "  -> Pre-stage on the LOGIN node, then resubmit:"
        echo "     huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --cache-dir ${HF_CACHE}"
        exit 1
    fi
    echo "Internet OK — first run downloads ~222GB (incremental: reuses the cached config.json)."
fi
echo

# ---------------------------------------------------------------------------
# HuggingFace cache + gated token
# ---------------------------------------------------------------------------
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN: ...${HF_TOKEN: -4} (required for gated CohereLabs repo)"
else
    echo "ERROR: HF_TOKEN not set and no token file — gated command-a download will 403."
    exit 1
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

echo "Starting vLLM chemistry probe for ${MODEL_KEY} (this downloads ~222GB bf16 on first run)..."
echo "Start time: $(date)"
echo

python3 chem_domain_probe_vllm.py --model "${MODEL_KEY}" --hf-cache "${HF_CACHE}" --repeats "${REPEATS}"

echo
echo "End time: $(date)"
echo "Done."
