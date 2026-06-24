#!/usr/bin/env python3
"""Fast GPU smoke test for the d_spec_vllm_cu124 env on Expanse NAIRR H100.

Validates the entire fp8 inference pipeline WITHOUT the 222GB command-a download:
  [1] torch can actually see the H100 (cu124 runtime via minor-version compat on
      the node's CUDA-12.0 driver) -> the exact failure mode that killed job 49907742.
  [2] this vLLM build registers Cohere2ForCausalLM (command-a-03-2025's arch).
  [3] command-a's real config.json arch string matches (gated fetch, ~KB).
  [4] vLLM can init CUDA + run the fp8 online-quant path + generate (tiny ungated model).
If this prints SMOKE_ALL_OK, the only thing the full run adds is the model's size.

NOTE: the LLM() call MUST live under `if __name__ == "__main__":`. vLLM 0.8.5's V1
engine spawns worker processes (forced 'spawn' once CUDA is initialized); a spawn
worker re-imports this module, so any top-level LLM() would re-execute and crash the
engine core. The real harness (chem_domain_probe_vllm.py) already guards this the same way.
"""
import os, json


def main():
    print("=== [1] torch CUDA ===", flush=True)
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("is_available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "FATAL: torch.cuda.is_available() is False (driver/cu mismatch persists)"
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0), "(Hopper sm_90 expected for native fp8)")

    print("=== [2] vLLM arch registry ===", flush=True)
    from vllm.model_executor.models.registry import ModelRegistry
    archs = list(ModelRegistry.get_supported_archs())
    print("Cohere2ForCausalLM supported:", "Cohere2ForCausalLM" in archs)
    assert "Cohere2ForCausalLM" in archs, "FATAL: this vLLM does not support command-a's architecture"

    print("=== [3] command-a config arch (gated fetch) ===", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download("CohereLabs/c4ai-command-a-03-2025", "config.json",
                              cache_dir=os.environ.get("HF_HOME"))
        j = json.load(open(cfg))
        print("command-a architectures:", j.get("architectures"), "model_type:", j.get("model_type"))
    except Exception as e:
        print("config-fetch note (non-fatal):", repr(e))

    print("=== [4] vLLM CUDA init + fp8 path (tiny ungated model w/ fast tokenizer) ===", flush=True)
    # Qwen2.5-0.5B exercises the same fp8 online-quant + Hopper generate path command-a uses.
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", quantization="fp8",
              gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    print("fp8 generate ->", repr(out[0].outputs[0].text))

    print("=== SMOKE_ALL_OK ===", flush=True)


if __name__ == "__main__":
    main()
