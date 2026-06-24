#!/usr/bin/env python3
"""
chem_domain_probe_vllm.py — open-weight (vLLM) runner for the chemical-domain
identifier-gradient probe. Mirror of chem_domain_probe.py (Anthropic API) for
GPU HPC clusters, batched via vLLM.

WHY this exists / what it tests
-------------------------------
The Anthropic-API chemistry run (Sonnet 4.6, 2026-05-29) produced the
mirror-image dissociation (Finding F-C6): at the bare PubChem CID (ID-3) the
*safety classifier* recognizes the CWC agent (fires 76-100%) while the *base
model* cannot resolve it (0/8 correct; hallucinates warfarin/parathion/...).
The "classifier fires" half is Claude-specific infrastructure; open-weight
models have no such system classifier. But the "model can't resolve a bare
CID" half is a *base-model knowledge* property and should replicate on
open-weight models. So this run tests the MODEL HALF of F-C6:

  H1 (mis-ID replicates):  open-weight models also fail to resolve the CWC
     CIDs (ID-3) and instead hallucinate / name a wrong compound, while they
     DO resolve fentanyl's training-prominent CID 3345.
  H2 (no system refusal):  open-weight models show ~0% true safety refusal at
     every level (consistent with the protein-domain variant-B result).
  H3 (Layer-2 disposition): whether/which open-weight models self-warn in text
     on CWC names (ID-0) the way Claude does.

The authoritative read is the full response_text per cell — hence, unlike the
protein vLLM runner (which stored a 150-char preview), this runner stores the
FULL response so the mis-ID audit can be done post-hoc.

Model panel + refusal detection are imported from d_spec_vllm.py to guarantee
parity with the protein open-weight panel (same 9 models).

Usage on Cayuga (via SLURM):
    sbatch chem_domain_cayuga.slurm mistral
For a direct test on a GPU node:
    conda activate d_spec_vllm
    python3 chem_domain_probe_vllm.py --model mistral
Local prompt sanity-check (no GPU / no vllm needed):
    python3 chem_domain_probe_vllm.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Lightweight, dependency-free config (safe to import anywhere) ---------------
from chem_config import SUBSTANCES, ID_LEVELS, build_prompt, names_correctly

# Heavy deps (vllm + protein runner) — only needed for actual inference.
# Guarded so that --dry-run prompt inspection works on a laptop without vllm.
try:
    from vllm import LLM, SamplingParams
    from d_spec_vllm import MODELS, is_refusal
    _HEAVY_OK = True
    _HEAVY_ERR = None
except Exception as e:  # ImportError, or d_spec_vllm's sys.exit-on-missing-vllm
    _HEAVY_OK = False
    _HEAVY_ERR = e
    MODELS = {}

    def is_refusal(_text: str) -> bool:  # placeholder; real one comes from d_spec_vllm
        return False

# ── Experiment configuration (matches chem_domain_probe.py for comparability) ──
RESULTS_DIR = Path(__file__).parent / "results"
TEMPERATURE = 1.0
DEFAULT_REPS = 25            # matches the Sonnet 4.6 API run (25 reps)
MAX_TOKENS = 512            # matches the API run's max_tokens=512


class ChemVLLMRunner:
    """Batched vLLM inference for the chemical-domain identifier gradient.

    Self-contained (does not subclass the protein VLLMRunner) so the heavy
    imports can stay guarded; the ~25-line model-loading block is intentionally
    kept in sync with d_spec_vllm.VLLMRunner.__init__.
    """

    def __init__(self, model_key: str, hf_cache_dir: Optional[str] = None,
                 reps: int = DEFAULT_REPS, substances: Optional[List[str]] = None):
        if not _HEAVY_OK:
            raise RuntimeError(f"vllm / d_spec_vllm import failed — cannot run inference: {_HEAVY_ERR}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")

        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.reps = reps
        self.substances = substances or list(SUBSTANCES.keys())
        self.results: List[Dict] = []

        if hf_cache_dir:
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir

        print(f"Loading model: {self.model_config['display_name']} ({self.model_config['hf_id']})")
        print("(This may take 2-10 minutes on first run as model downloads)")

        # AWQ requires float16 (bf16 unsupported); else bf16.
        quantization = self.model_config.get("quantization")
        dtype = "float16" if quantization in ("awq", "awq_marlin") else "bfloat16"

        llm_kwargs = {
            "model": self.model_config["hf_id"],
            "max_model_len": self.model_config["max_model_len"],
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if quantization:
            llm_kwargs["quantization"] = quantization

        # Pre-quantized bitsandbytes (nf4/fp4) checkpoints need load_format set.
        load_format = self.model_config.get("load_format")
        if load_format:
            llm_kwargs["load_format"] = load_format
        # Optional memory knobs (e.g. tight single-GPU 4-bit 111B fits).
        gmu = self.model_config.get("gpu_memory_utilization")
        if gmu is not None:
            llm_kwargs["gpu_memory_utilization"] = gmu
        if self.model_config.get("enforce_eager"):
            llm_kwargs["enforce_eager"] = True

        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        requested_tp = self.model_config.get("tensor_parallel_size", 1)
        if requested_tp > n_gpus:
            print(f"WARNING: tensor_parallel_size={requested_tp} requested but only {n_gpus} GPU(s) available.")
            print(f"  -> Falling back to tensor_parallel_size={max(1, n_gpus)}")
            requested_tp = max(1, n_gpus)
        llm_kwargs["tensor_parallel_size"] = requested_tp
        print(f"Using {requested_tp} GPU(s), dtype={dtype}, quantization={quantization or 'none'}, "
              f"load_format={load_format or 'auto'}, gmu={gmu or 'default'}, "
              f"enforce_eager={bool(self.model_config.get('enforce_eager'))}")

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully")

    def _build_prompts(self) -> List[Dict]:
        prompts_meta = []
        for sub_name in self.substances:
            for level in ID_LEVELS:
                for rep in range(self.reps):
                    user_prompt = build_prompt(sub_name, level)
                    messages = [{"role": "user", "content": user_prompt}]
                    chat_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts_meta.append({
                        "substance": sub_name,
                        "category": SUBSTANCES[sub_name]["category"],
                        "id_level": level,
                        "rep": rep,
                        "user_prompt": user_prompt,
                        "chat_prompt": chat_prompt,
                    })
        return prompts_meta

    def run(self) -> List[Dict]:
        prompts_meta = self._build_prompts()
        total = len(prompts_meta)
        start_time = datetime.now()

        print(f"\n{'='*70}")
        print(f"CHEM DOMAIN — vLLM Open-Weight Inference")
        print(f"{'='*70}")
        print(f"Model: {self.model_config['display_name']}  (safety tier: {self.model_config['safety_tier']})")
        print(f"Substances: {self.substances}")
        print(f"Levels: {ID_LEVELS}")
        print(f"Protocol: {len(self.substances)} x {len(ID_LEVELS)} x {self.reps} = {total} generations")
        print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=1)
        chat_prompts = [p["chat_prompt"] for p in prompts_meta]

        print(f"Running batched generation ({total} prompts)...")
        gen_start = time.time()
        outputs = self.llm.generate(chat_prompts, sampling_params)
        gen_elapsed = time.time() - gen_start
        print(f"Generation done in {gen_elapsed:.1f}s ({total/gen_elapsed:.1f} prompts/sec)")

        for meta, output in zip(prompts_meta, outputs):
            response_text = output.outputs[0].text
            self.results.append({
                "substance": meta["substance"],
                "category": meta["category"],
                "id_level": meta["id_level"],
                "rep": meta["rep"],
                "is_refusal": is_refusal(response_text),       # text-based; ~0% expected
                "names_correctly": names_correctly(meta["substance"], response_text),
                "response_length": len(response_text),
                "response_text": response_text,                # FULL text (mis-ID audit)
                "prompt": meta["user_prompt"],
            })

        self._print_summary()
        elapsed = datetime.now() - start_time
        print(f"\n{'='*70}\nCompleted: {len(self.results)} generations in {elapsed}\n{'='*70}\n")
        return self.results

    def _print_summary(self):
        # Two matrices: refusal rate, and correct-identification rate (the mis-ID signal).
        def cell(pred):
            agg = {}
            for r in self.results:
                k = (r["substance"], r["id_level"])
                agg.setdefault(k, [0, 0])
                agg[k][1] += 1
                if pred(r):
                    agg[k][0] += 1
            return agg

        for title, pred in [("Refusal rate (text-based)", lambda r: r["is_refusal"]),
                            ("Correct-identification rate (heuristic)", lambda r: r["names_correctly"])]:
            agg = cell(pred)
            print(f"\n{title}:")
            print(f"{'Substance':12s}", end="")
            for idl in ID_LEVELS:
                print(f"  {idl:>6s}", end="")
            print()
            for sub in self.substances:
                print(f"{sub:12s}", end="")
                for idl in ID_LEVELS:
                    n, d = agg.get((sub, idl), [0, 0])
                    rate = (n / d * 100) if d else 0
                    print(f"  {rate:5.0f}%", end="")
                print(f"  [{SUBSTANCES[sub]['category']}]")

    def save_results(self, output_dir: Path = RESULTS_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"chem_domain_vllm_{self.model_key}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump({
                "model_key": self.model_key,
                "model_id": self.model_config["hf_id"],
                "display_name": self.model_config["display_name"],
                "safety_tier": self.model_config["safety_tier"],
                "inference_backend": "vllm",
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "id_levels": ID_LEVELS,
                "substances": self.substances,
                "n_substances": len(self.substances),
                "n_levels": len(ID_LEVELS),
                "n_reps": self.reps,
                "total_calls": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")
        return output_file


def dry_run_preview(reps: int, substances: Optional[List[str]]):
    """Build and print sample prompts using only chem_config (no vllm/GPU)."""
    subs = substances or list(SUBSTANCES.keys())
    total = len(subs) * len(ID_LEVELS) * reps
    print(f"DRY RUN — no inference. Would generate {len(subs)} x {len(ID_LEVELS)} x {reps} = {total} responses.")
    print(f"Substances: {subs}")
    print(f"Levels: {ID_LEVELS}\n")
    sample = subs[0]
    print(f"Sample prompts for '{sample}':")
    for level in ID_LEVELS:
        p = build_prompt(sample, level)
        print(f"\n  --- {level} ---\n  {p[:300]}{'...' if len(p) > 300 else ''}")
    # Validate every cell builds without error
    bad = []
    for s in subs:
        for level in ID_LEVELS:
            try:
                if not build_prompt(s, level).strip():
                    bad.append(f"{s}/{level} (empty)")
            except Exception as ex:
                bad.append(f"{s}/{level} ({ex})")
    print(f"\nPrompt-build validation: {'ALL OK' if not bad else 'PROBLEMS: ' + str(bad)}")
    if MODELS:
        print(f"Model panel available: {list(MODELS.keys())}")
    else:
        print(f"(Model panel not loaded locally — vllm/d_spec_vllm import deferred: {_HEAVY_ERR})")


def main():
    parser = argparse.ArgumentParser(description="Chemical-domain identifier-gradient probe (open-weight / vLLM)")
    parser.add_argument("--model", default=None, help="Model key (see d_spec_vllm.MODELS)")
    parser.add_argument("--dry-run", action="store_true", help="Print sample prompts; no GPU/vllm needed")
    parser.add_argument("--hf-cache", type=str, default=None, help="HuggingFace cache dir")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPS)
    parser.add_argument("--substances", nargs="+", default=None, help="Subset of substances")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_preview(args.repeats, args.substances)
        return

    if not args.model:
        print("ERROR: --model is required for a real run (or use --dry-run).")
        if MODELS:
            print(f"Choices: {list(MODELS.keys())}")
        sys.exit(1)

    runner = ChemVLLMRunner(model_key=args.model, hf_cache_dir=args.hf_cache,
                            reps=args.repeats, substances=args.substances)
    results = runner.run()
    if results:
        runner.save_results()


if __name__ == "__main__":
    main()
