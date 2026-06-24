#!/bin/bash
# Build d_spec_vllm conda env on Expanse (login node). Idempotent.
set -e
cd "$HOME"

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    echo "[build] downloading Miniconda3..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "/tmp/mc_${USER}.sh"
    bash "/tmp/mc_${USER}.sh" -b -p "$HOME/miniconda3"
    rm -f "/tmp/mc_${USER}.sh"
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^d_spec_vllm[[:space:]]|/d_spec_vllm$"; then
    echo "[build] creating env d_spec_vllm (python 3.10)..."
    conda create -n d_spec_vllm python=3.10 -y
fi

conda activate d_spec_vllm
pip install --upgrade pip
pip install vllm requests huggingface_hub

echo "[build] versions:"
python -c 'import vllm, torch; print("vllm", vllm.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda)'
echo ENV_SETUP_DONE_OK
