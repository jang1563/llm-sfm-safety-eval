#!/usr/bin/env python3
"""
E1_overtrust_vllm.py — open-weight vLLM subject collection for E1 (Spec §10, option 5).

WHY: Finding 16 showed that the one gate-open Claude (Sonnet 4) discriminates signal
*reliability* but NOT hazard *relevance* (DO-ARR contrast covers 0), with a latent
over-trust that surfaces at anthrax-PA FSI=0. Open-weight models are ALL gate-open
(Finding 9: no genuine system refusals), so they are the natural substrate to test
whether that reliability-discrimination-without-hazard-relevance dissociation is
Claude-specific or a general property of the LLM×SFM interpretation interface.

WHAT: collects RAW subject responses for the frozen E1 value table on open-weight
models via vLLM batched inference (Cayuga A40/A100, conda env `d_spec_vllm`). Emits the
SAME result-row schema as E1_overtrust_run.py, so E1_overtrust_judge.py (rule pre-pass
or blinded Sonnet-4 LLM judge) + E1_overtrust_analyze.py + E1_overtrust_stats.py run
UNCHANGED on the output — collect on GPU, judge/analyze later on the API host.

DRY: prompt assembly + refusal classification are imported from E1_overtrust_smoke
(single source of truth across smoke / API run / open-weight run). Model-loading reuses
the battle-tested d_spec_vllm.VLLMRunner (dtype/quant/tensor-parallel logic, 30-model
panel) by composition — no duplicated GPU config. The d_spec_vllm import is GPU-only and
deferred to run time, so `--dry-run` works on a laptop with neither vllm nor anthropic.

SAFE-CONDUCT (identical to E1_overtrust_run.py): prompts carry PDB/accession ID + the
frozen metric JSON only — never the common name, never the amino-acid sequence/FASTA
(§6.1/§8.3). No novel sequences. No synthesis. Values come from the frozen value table;
no fabricated dangerous content (scrambled NUMBERS only). The payload metric scores the
model's OVER-CLAIM (a safety failure), never any harmful content. Defensive AI-biosafety
(Mason Lab, WCM).

Usage (Cayuga GPU node):
    conda activate d_spec_vllm
    python3 E1_overtrust_vllm.py --model qwen --dry-run        # prompts only, no GPU/vllm
    python3 E1_overtrust_vllm.py --model qwen                  # 9 proteins × S0–S4 × 10 = 450 gens
    python3 E1_overtrust_vllm.py --model mistral --reps 10 --hf-cache $SCRATCH/hf_cache
One model per process (matches the proven d_spec_vllm pattern); use the SLURM arrays
to sweep the panel: E1_vllm_cayuga.slurm (A40, the five <=14B models) and
E1_vllm_cayuga_a100.slurm (1x A100 80GB, the five 26-32B models), each
`sbatch --array=0-4 ...`.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Pure helpers — import-safe without anthropic (smoke.py lazily defers its API import).
from E1_overtrust_smoke import load_cells, build_prompt, is_refused

HERE = Path(__file__).resolve().parent
VALUE_TABLE_NAME = "E1_value_table_20260530.json"
TEMPERATURE = 1.0
MAX_TOKENS = 2048          # match the API run (avoid truncating scored content)
REPS = 10
SEED = 20260530

# Curated open-weight panel (config lives in d_spec_vllm.MODELS, the proven Cayuga
# panel). Spans 4 vendors × 3.8B–32B; all gate-open per Finding 9. One model / process.
E1_PANEL = [
    "mistral",        # Mistral 7B v0.3   (minimal safety)
    "ministral3-14b", # Mistral 3 14B     (minimal safety, newer)
    "qwen",           # Qwen 2.5 7B       (moderate)
    "qwen3-4b",       # Qwen 3 4B         (moderate, small)
    "qwen3-30b-moe",  # Qwen 3 30B MoE    (moderate, MoE)
    "qwen3-32b",      # Qwen 3 32B dense  (moderate, large)
    "phi4-mini",      # Phi-4 mini 3.8B   (moderate, tiny)
    "gemma4-31b",     # Gemma 4 31B       (strong)
    "gemma4-moe",     # Gemma 4 26B-A4B   (strong, MoE)
    "olmo2-32b",      # OLMo-2 32B        (safety-tuned, fully open)
]

# Ground-truth cell fields copied verbatim into each row (mirror E1_overtrust_run.py).
CELL_FIELDS = [
    "cell_id", "pdb_id", "signal", "presentation", "frame", "value_shown",
    "metric_role", "appropriate_behavior", "hazardous", "is_true_benign",
    "is_lexical_baseline_cell", "is_falsepos_probe", "do_arr_cell",
    "s4_scramble_degenerate", "optional",
]


def finish_to_stop(fr: Optional[str]) -> str:
    """Map vLLM finish_reason → Anthropic-style stop_reason (is_refused expects the
    end_turn/max_tokens vocabulary). Open-weight models never emit a system-level
    'refusal' stop, so soft-refusal text anchors do the work (intended; Finding 9)."""
    if fr == "length":
        return "max_tokens"
    if fr == "stop":
        return "end_turn"
    return fr or "end_turn"


def select_cells(cells: List[Dict], proteins: Optional[List[str]],
                 signals: Optional[List[str]]) -> List[Dict]:
    return [c for c in cells
            if c["presentation"] == "P3"
            and (proteins is None or c["pdb_id"] in proteins)
            and (signals is None or c["signal"] in signals)]


def do_dry_run(cells: List[Dict], proteins, signals, reps: int) -> None:
    sel = select_cells(cells, proteins, signals)
    tasks = [(c, rep) for c in sel for rep in range(reps)]
    seen = set()
    for c, _ in tasks:
        if c["cell_id"] in seen:
            continue
        seen.add(c["cell_id"])
        print(f"--- {c['cell_id']}  frame={c['frame']}  haz={c['hazardous']}  "
              f"role={c['metric_role']} ---")
        print(build_prompt(c))
        print()
    print(f"[dry-run] {len(tasks)} generations NOT made "
          f"({len(seen)} unique cells × {reps} reps).")


def run_model(model_key: str, cells: List[Dict], proteins, signals,
              reps: int, seed: int, hf_cache: Optional[str]) -> tuple:
    """Load `model_key` via the proven d_spec_vllm loader (composition) and run the
    full E1 cell set in one batched vLLM call. Returns (rows, model_cfg)."""
    # GPU-only imports, deferred so --dry-run needs neither vllm nor d_spec_config.
    from d_spec_vllm import VLLMRunner as BaseRunner, MODELS
    from vllm import SamplingParams

    if model_key not in MODELS:
        print(f"ERROR: unknown model '{model_key}'. Choose from: {sorted(MODELS)}")
        sys.exit(1)

    base = BaseRunner(model_key, hf_cache_dir=hf_cache)   # loads .llm + .tokenizer
    llm, tok = base.llm, base.tokenizer

    sel = select_cells(cells, proteins, signals)
    tasks = [(c, rep) for c in sel for rep in range(reps)]
    random.Random(seed).shuffle(tasks)                    # interleave (parity w/ run.py)

    metas, chat_prompts = [], []
    for c, rep in tasks:
        prompt = build_prompt(c)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        metas.append((c, rep, prompt))
        chat_prompts.append(chat)

    # No per-request seed: identical rep prompts must vary, so rely on T=1.0 sampling.
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, n=1)

    cfg = MODELS[model_key]
    print(f"\n{'='*72}\nE1 OPEN-WEIGHT RUN — {cfg.get('display_name', model_key)}\n{'='*72}")
    print(f"model_key: {model_key}  hf_id={cfg['hf_id']}")
    print(f"cells    : {len(sel)}  ({sorted({c['pdb_id'] for c in sel})})")
    print(f"signals  : {sorted({c['signal'] for c in sel})}")
    print(f"reps     : {reps}  ->  {len(tasks)} generations  T={TEMPERATURE}  max_tokens={MAX_TOKENS}")
    print(f"value tbl: {VALUE_TABLE_NAME}\n")
    print(f"running batched generation ({len(chat_prompts)} prompts)...")

    outs = llm.generate(chat_prompts, sp)

    rows: List[Dict] = []
    for (c, rep, prompt), out in zip(metas, outs):
        o = out.outputs[0]
        text = o.text
        stop = finish_to_stop(o.finish_reason)
        refused = is_refused(text, stop)
        row = {k: c.get(k) for k in CELL_FIELDS}
        row.update({
            "model": cfg["hf_id"], "model_key": model_key, "rep": rep,
            "prompt": prompt, "response": text, "stop_reason": stop,
            "refused": refused, "error": None,
        })
        rows.append(row)
    return rows, cfg


def engagement_summary(rows: List[Dict]) -> None:
    print("\nengagement by signal (open-weight pre-pass; metrics computed engaged-only "
          "downstream):")
    for s in sorted({r["signal"] for r in rows}):
        rs = [r for r in rows if r["signal"] == s]
        e = sum(1 for r in rs if r["refused"] is False)
        print(f"  {s}: {e}/{len(rs)} engaged")


def save_rows(rows: List[Dict], model_key: str, cfg: Dict, reps: int, seed: int) -> Path:
    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"E1_subject_{model_key}_vllm_{ts}.json"
    engaged = sum(1 for r in rows if r["refused"] is False)
    out.write_text(json.dumps({
        "experiment": "E1 discrimination — open-weight subject collection (Spec §10, opt-5)",
        "spec": "E1_LLM_SFM_OverTrust_Spec.md",
        "value_table": VALUE_TABLE_NAME,
        "inference_backend": "vllm",
        "timestamp": datetime.now().isoformat(),
        "model": cfg["hf_id"], "model_key": model_key,
        "display_name": cfg.get("display_name"),
        "safety_tier": cfg.get("safety_tier"),
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "presentation": "P3",
        "reps": reps, "seed": seed,
        "n_calls": len(rows), "n_engaged": engaged,
        "results": rows,
    }, indent=2))
    print(f"\nsaved -> {out}  ({len(rows)} rows, engaged {engaged}/{len(rows)})")
    return out


def main():
    ap = argparse.ArgumentParser(description="E1 open-weight vLLM subject collection (Spec §10, opt-5)")
    ap.add_argument("--model", default=None, help="single model key (see d_spec_vllm.MODELS)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="multiple keys (loads sequentially in ONE process — small models only)")
    ap.add_argument("--panel", action="store_true", help=f"run the curated E1 panel: {E1_PANEL}")
    ap.add_argument("--proteins", nargs="+", default=None, help="subset of PDB IDs (default: all 9)")
    ap.add_argument("--signals", nargs="+", default=None, help="subset of S0..S4 (default: all)")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--hf-cache", default=None, help="HF cache dir (default: $HF_HOME)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no GPU/vllm needed")
    args = ap.parse_args()

    cells, _meta = load_cells()

    if args.dry_run:
        do_dry_run(cells, args.proteins, args.signals, args.reps)
        return

    if args.panel:
        models = E1_PANEL
    elif args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        print("ERROR: specify --model KEY, --models KEY..., or --panel (or --dry-run).")
        sys.exit(1)

    for mk in models:
        rows, cfg = run_model(mk, cells, args.proteins, args.signals,
                              args.reps, args.seed, args.hf_cache)
        engagement_summary(rows)
        save_rows(rows, mk, cfg, args.reps, args.seed)


if __name__ == "__main__":
    main()
