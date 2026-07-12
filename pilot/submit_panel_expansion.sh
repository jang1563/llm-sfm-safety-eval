#!/bin/bash
# submit_panel_expansion.sh — run on a SLURM login node.
#
# Submits the 2026-05-30 chemistry open-weight panel expansion:
#   - ministral3-8b : re-run after wiping its corrupted cache (garbage decode)
#   - chemdfm-14b   : chemistry-specialist (Qwen2.5 base)      [A40]
#   - chemllm-7b    : chemistry-specialist (InternLM2 base)    [A40]
#   - olmo2-32b     : fully-open, documented safety post-train [A100 x1]
#   - qwen3-32b     : largest dense Qwen3                       [A100 x1]
#   - command-a     : Cohere 111B, fp8 online quant            [A100 x2, GATED]
#
# PREREQ (set REMOTE_HOST and REMOTE_PROJECT_DIR, then run from a workstation):
#   rsync -avz d_spec_vllm.py submit_panel_expansion.sh \
#       "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"
#
# PREREQ for command-a (gated CC-BY-NC):
#   1. Accept the license at https://hf.co/CohereLabs/c4ai-command-a-03-2025
#   2. export HF_TOKEN=hf_...   (before running this script; sbatch --export=ALL
#      propagates it into the job). First run downloads the full 222GB bf16
#      checkpoint, then vLLM quantizes to fp8 on load → ~111GB across 2xA100.

set -euo pipefail

SBATCH="${SBATCH:-sbatch}"
SCRATCH="${SCRATCH:-${HOME}/scratch}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/d_spec_experiment}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
REPEATS="${1:-25}"

cd "${PROJECT_DIR}"
echo "PROJECT_DIR=${PROJECT_DIR}  HF_CACHE=${HF_CACHE}  REPEATS=${REPEATS}"

# 1) Wipe the corrupted ministral3-8b snapshot so it re-downloads clean.
MIN8B="${HF_CACHE}/models--mistralai--Ministral-3-8B-Instruct-2512"
if [ -d "${MIN8B}" ]; then
    echo "Wiping corrupted ministral3-8b cache: ${MIN8B}"
    rm -rf "${MIN8B}"
fi

# 2) command-a gate: ensure HF_TOKEN reaches the job. The SLURM submission
#    inherits the explicitly provided environment via --export=ALL.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN unset — command-a (gated) will fail."
    echo "         The other 5 jobs are ungated and run fine regardless."
fi

# 3) Submit. <=14B on A40 (slurm default a40:1); 32B on A100x1; command-a A100x2.
#    Site-specific queue and GPU limits are controlled by the scheduler.
echo "Submitting jobs..."
${SBATCH}                                              chem_domain_cayuga.slurm ministral3-8b "${REPEATS}"
${SBATCH}                                              chem_domain_cayuga.slurm chemdfm-14b   "${REPEATS}"
${SBATCH}                                              chem_domain_cayuga.slurm chemllm-7b    "${REPEATS}"
${SBATCH} --gres=gpu:a100:1 --time=03:00:00            chem_domain_cayuga.slurm olmo2-32b     "${REPEATS}"
${SBATCH} --gres=gpu:a100:1 --time=03:00:00            chem_domain_cayuga.slurm qwen3-32b     "${REPEATS}"
${SBATCH} --gres=gpu:a100:2 --mem=128G --time=05:00:00 chem_domain_cayuga.slurm command-a     "${REPEATS}"

echo
echo "Submitted 6 jobs. Monitor with:"
echo "  squeue -u ${USER}"
