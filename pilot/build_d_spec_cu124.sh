#!/bin/bash
# Build d_spec_vllm_cu124 conda env on Expanse (login node). Idempotent, sentinel-tracked.
#
# WHY: the original env (vllm 0.21.0 / torch 2.11.0+cu130 = CUDA 13.0) is forward-
# INCOMPATIBLE with Expanse's NAIRR H100 driver (CUDA 12.0; newest toolkit/compat on
# the cluster is 12.2). torch reported `is_available()=False` + "driver too old
# (found version 12000)". CUDA major 13 > 12 and no cuda-13 compat shim exists.
#
# FIX: pin vLLM 0.8.5 -> torch 2.6.0+cu124 (CUDA major 12). cu124 runs on the 12.0
# driver via CUDA minor-version compatibility. 0.8.5 (Apr 2025) postdates command-a
# (Mar 2025) so it supports Cohere2ForCausalLM (command-a-03-2025's architecture).
set -e

ENV_PATH=/expanse/lustre/projects/crl195/jkim61/conda_envs/d_spec_vllm_cu124
PROJ=/expanse/lustre/scratch/jkim61/temp_project/d_spec_experiment
SENTINEL="${PROJ}/logs/env_cu124.done"
mkdir -p "${PROJ}/logs"
rm -f "${SENTINEL}"

# conda init (mirror chem_domain_expanse.slurm logic)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: no conda init found"; exit 1
fi

if [ ! -x "${ENV_PATH}/bin/python" ]; then
    echo "[build] creating env at ${ENV_PATH} (python 3.10)..."
    # Recent conda gates the anaconda 'defaults' channels (pkgs/main, pkgs/r) behind a
    # Terms-of-Service accept. Build from conda-forge with --override-channels to avoid
    # the gated channels entirely (no global conda config change).
    # NOTE: conda-forge's `python` does NOT bundle pip (anaconda defaults' did) — must
    # request `pip` explicitly or the later `python -m pip` calls fail with No module pip.
    conda create -y -p "${ENV_PATH}" -c conda-forge --override-channels python=3.10 pip
else
    echo "[build] env already exists at ${ENV_PATH}"
fi

conda activate "${ENV_PATH}"
# Safety net: ensure pip exists even if a pre-existing env was created without it.
if ! python -m pip --version >/dev/null 2>&1; then
    echo "[build] pip missing in env — installing via conda..."
    conda install -y -p "${ENV_PATH}" -c conda-forge --override-channels pip || python -m ensurepip --upgrade
fi
python -m pip install --upgrade pip
# vllm 0.8.5 pins torch 2.6.0; PyPI default torch 2.6.0 wheel = +cu124. No special index.
# Pin transformers==4.51.3: vllm 0.8.5 only floors transformers>=4.51.1 (no upper bound),
# so an unpinned install pulls transformers 5.x, which removed `all_special_tokens_extended`
# and breaks vllm's get_cached_tokenizer at LLM() init. 4.51.3 is vllm 0.8.5's tested pairing.
python -m pip install "vllm==0.8.5" "transformers==4.51.3" requests huggingface_hub

echo "[build] === lightweight verify (NO heavy import: login nodes are memory-capped;"
echo "        importing torch+vllm here segfaults with 'failed to map segment'. The real"
echo "        torch.cuda + Cohere2 arch + fp8 checks run on a GPU node via chem_smoke_expanse.slurm) ==="
# Read torch's build string straight from version.py WITHOUT importing torch.
TORCH_VER=$(grep -oE "__version__ = '[^']+'" "${ENV_PATH}/lib/python3.10/site-packages/torch/version.py" | head -1 | cut -d"'" -f2)
VLLM_VER=$("${ENV_PATH}/bin/pip" show vllm 2>/dev/null | awk -F': ' '/^Version:/{print $2}')
echo "vllm ${VLLM_VER}   torch ${TORCH_VER}"
case "${TORCH_VER}" in
  *+cu12*) echo "CUDA-major-12 OK (minor-version-compatible with the node's CUDA-12.0 driver)";;
  *) echo "FATAL: torch build is '${TORCH_VER}', need +cu12x for the 12.0 driver"; exit 1;;
esac
[ -f "${ENV_PATH}/lib/python3.10/site-packages/vllm/__init__.py" ] || { echo "FATAL: vllm package missing"; exit 1; }

echo OK > "${SENTINEL}"
echo "ENV_CU124_DONE_OK"
