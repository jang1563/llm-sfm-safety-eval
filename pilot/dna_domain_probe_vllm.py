#!/usr/bin/env python3
"""
dna_domain_probe_vllm.py — open-weight (vLLM) runner for the DNA/genomic
identifier-gradient probe. Mirror of dna_domain_probe.py (Anthropic API) and of
chem_domain_probe_vllm.py, for GPU HPC clusters, batched via vLLM.

WHY this exists / what it tests
-------------------------------
The Anthropic-API DNA run (Sonnet 4.6, 2026-05-30, 1,050 calls) found:
  - No accession wall: ID-3 GenBank accession refused 100% for all 4 toxins.
  - Sequence classifier fires on raw nucleotides: ID-4/ID-5 refused ~99%.
  - ID-5->ID-6 cliff: the same 500 bp fragment is refused 99% bare (ID-5) but
    1% when wrapped in genomic-LM/codon-optimization framing (ID-6).

All of these are *system-classifier* behaviours (stop_reason=='refusal'), i.e.
Claude managed-access infrastructure. Open-weight models have no such system
classifier, so this run tests the cross-model claims:

  H1 (Claude-exclusive sequence recognition): open-weight models show ~0% TRUE
     safety refusal at every level — including the raw-sequence levels ID-4/ID-5
     that drive 99% refusal on Claude. (Text-pattern "refusals" here are
     capability disclaimers / false positives, audited via full response_text.)
  H2 (base-model gene resolution): can an open-weight model identify the gene
     from name (ID-0~ID-2), from accession (ID-3), and from raw sequence
     (ID-4~ID-6)? The `identifies_gene` heuristic gives the per-level ID rate;
     expectation is high at the name levels, ~floor at bare sequence.

Model panel + refusal detection are imported from d_spec_vllm.py to guarantee
parity with the protein/chem open-weight panels (same models).

Usage on Cayuga (SLURM):
    sbatch dna_domain_cayuga.slurm mistral
Direct on a GPU node:
    conda activate d_spec_vllm
    python3 dna_domain_probe_vllm.py --model mistral
Local prompt sanity-check (no GPU / no vllm):
    python3 dna_domain_probe_vllm.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Lightweight, dependency-free config (safe to import anywhere).
from dna_config import GENES, ID_LEVELS, build_prompt, identifies_gene

# Heavy deps (vllm + protein runner) — guarded so --dry-run works without vllm.
try:
    from vllm import LLM, SamplingParams
    from d_spec_vllm import MODELS, is_refusal
    _HEAVY_OK = True
    _HEAVY_ERR = None
except Exception as e:
    _HEAVY_OK = False
    _HEAVY_ERR = e
    MODELS = {}

    def is_refusal(_text: str) -> bool:
        return False

RESULTS_DIR = Path(__file__).parent / "results"
TEMPERATURE = 1.0
DEFAULT_REPS = 25
MAX_TOKENS = 512


class DNAVLLMRunner:
    """Batched vLLM inference for the DNA/genomic identifier gradient.

    Self-contained (does not subclass the protein VLLMRunner); the model-loading
    block is intentionally kept in sync with d_spec_vllm.VLLMRunner.__init__ and
    chem_domain_probe_vllm.ChemVLLMRunner.__init__.
    """

    def __init__(self, model_key: str, hf_cache_dir: Optional[str] = None,
                 reps: int = DEFAULT_REPS, genes: Optional[List[str]] = None):
        if not _HEAVY_OK:
            raise RuntimeError(f"vllm / d_spec_vllm import failed — cannot run inference: {_HEAVY_ERR}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS.keys())}")

        self.model_key = model_key
        self.model_config = MODELS[model_key]
        self.reps = reps
        self.genes = genes or list(GENES.keys())
        self.results: List[Dict] = []

        if hf_cache_dir:
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir

        print(f"Loading model: {self.model_config['display_name']} ({self.model_config['hf_id']})")
        print("(This may take 2-10 minutes on first run as model downloads)")

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
        load_format = self.model_config.get("load_format")
        if load_format:
            llm_kwargs["load_format"] = load_format
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
        for gene_key in self.genes:
            for level in ID_LEVELS:
                for rep in range(self.reps):
                    user_prompt = build_prompt(gene_key, level)
                    messages = [{"role": "user", "content": user_prompt}]
                    chat_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts_meta.append({
                        "gene": gene_key,
                        "category": GENES[gene_key]["category"],
                        "protein_xref": GENES[gene_key]["protein_xref"],
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
        print(f"DNA DOMAIN — vLLM Open-Weight Inference")
        print(f"{'='*70}")
        print(f"Model: {self.model_config['display_name']}  (safety tier: {self.model_config['safety_tier']})")
        print(f"Genes: {self.genes}")
        print(f"Levels: {ID_LEVELS}")
        print(f"Protocol: {len(self.genes)} x {len(ID_LEVELS)} x {self.reps} = {total} generations")
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
                "gene": meta["gene"],
                "category": meta["category"],
                "protein_xref": meta["protein_xref"],
                "id_level": meta["id_level"],
                "rep": meta["rep"],
                "is_refusal": is_refusal(response_text),     # text-based; ~0% expected
                "identifies_gene": identifies_gene(meta["gene"], response_text),
                "response_length": len(response_text),
                "response_text": response_text,              # FULL text (mis-ID / FP audit)
                "prompt": meta["user_prompt"],
            })

        self._print_summary()
        elapsed = datetime.now() - start_time
        print(f"\n{'='*70}\nCompleted: {len(self.results)} generations in {elapsed}\n{'='*70}\n")
        return self.results

    def _print_summary(self):
        def cell(pred):
            agg = {}
            for r in self.results:
                k = (r["gene"], r["id_level"])
                agg.setdefault(k, [0, 0])
                agg[k][1] += 1
                if pred(r):
                    agg[k][0] += 1
            return agg

        for title, pred in [("Refusal rate (text-based; FP audit needed)", lambda r: r["is_refusal"]),
                            ("Gene-identification rate (heuristic)", lambda r: r["identifies_gene"])]:
            agg = cell(pred)
            print(f"\n{title}:")
            print(f"{'Gene':12s}", end="")
            for idl in ID_LEVELS:
                print(f"  {idl:>6s}", end="")
            print()
            for gene_key in self.genes:
                print(f"{gene_key:12s}", end="")
                for idl in ID_LEVELS:
                    n, d = agg.get((gene_key, idl), [0, 0])
                    rate = (n / d * 100) if d else 0
                    print(f"  {rate:5.0f}%", end="")
                print(f"  [{GENES[gene_key]['category']}]")

    def save_results(self, output_dir: Path = RESULTS_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dna_domain_vllm_{self.model_key}_{timestamp}.json"
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
                "genes": self.genes,
                "n_genes": len(self.genes),
                "n_levels": len(ID_LEVELS),
                "n_reps": self.reps,
                "total_calls": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")
        return output_file


def dry_run_preview(reps: int, genes: Optional[List[str]]):
    gkeys = genes or list(GENES.keys())
    total = len(gkeys) * len(ID_LEVELS) * reps
    print(f"DRY RUN — no inference. Would generate {len(gkeys)} x {len(ID_LEVELS)} x {reps} = {total} responses.")
    print(f"Genes: {gkeys}")
    print(f"Levels: {ID_LEVELS}\n")
    sample = gkeys[0]
    print(f"Sample prompts for '{sample}':")
    for level in ID_LEVELS:
        p = build_prompt(sample, level)
        print(f"\n  --- {level} ---\n  {p[:280]}{'...' if len(p) > 280 else ''}")
    bad = []
    for g in gkeys:
        for level in ID_LEVELS:
            try:
                if not build_prompt(g, level).strip():
                    bad.append(f"{g}/{level} (empty)")
            except Exception as ex:
                bad.append(f"{g}/{level} ({ex})")
    print(f"\nPrompt-build validation: {'ALL OK' if not bad else 'PROBLEMS: ' + str(bad)}")
    if MODELS:
        print(f"Model panel available: {list(MODELS.keys())}")
    else:
        print(f"(Model panel not loaded locally — vllm/d_spec_vllm import deferred: {_HEAVY_ERR})")


def main():
    parser = argparse.ArgumentParser(description="DNA/genomic identifier-gradient probe (open-weight / vLLM)")
    parser.add_argument("--model", default=None, help="Model key (see d_spec_vllm.MODELS)")
    parser.add_argument("--dry-run", action="store_true", help="Print sample prompts; no GPU/vllm needed")
    parser.add_argument("--hf-cache", type=str, default=None, help="HuggingFace cache dir")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPS)
    parser.add_argument("--genes", nargs="+", default=None, help="Subset of gene keys")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_preview(args.repeats, args.genes)
        return

    if not args.model:
        print("ERROR: --model is required for a real run (or use --dry-run).")
        if MODELS:
            print(f"Choices: {list(MODELS.keys())}")
        sys.exit(1)

    runner = DNAVLLMRunner(model_key=args.model, hf_cache_dir=args.hf_cache,
                           reps=args.repeats, genes=args.genes)
    results = runner.run()
    if results:
        runner.save_results()


if __name__ == "__main__":
    main()
