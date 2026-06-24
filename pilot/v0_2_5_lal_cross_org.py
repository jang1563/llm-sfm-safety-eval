#!/usr/bin/env python3
"""
v0.2.5 LAL Cross-Organization Analysis

Aggregates LAL trajectory results across organizations:
  - Anthropic: Sonnet 4, Haiku 4.5 (judged JSONs from v0_2_5_lal_judge.py)
  - Mistral AI: Mistral 7B, Ministral 3 8B/14B
  - Alibaba: Qwen 2.5 7B, Qwen 3 4B/30B-MoE
  - Google: Gemma 4 31B/MoE
  - Microsoft: Phi-4 Mini

For each model, computes:
  - Endorsement slope (mean per-iteration change)
  - Refusal rate (Anthropic: stop_reason="refusal"; open-weight: judged endorsement<=1)
  - Oversight solicitation rate
  - Dual-use awareness rate

Outputs a cross-org comparison table suitable for paper Table.

Usage:
    python3 v0_2_5_lal_cross_org.py \\
        --anthropic results/v0_2_5_lal_judged_*.json \\
        --vllm results/v0_2_5_lal_vllm_*_judged_*.json
"""

import json
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean


# Model registry: maps file pattern → (organization, display_name, tier_size_B)
MODEL_REGISTRY = {
    "claude-sonnet-4-20250514": ("Anthropic", "Sonnet 4", None),
    "claude-sonnet-4-6":        ("Anthropic", "Sonnet 4.6", None),
    "claude-haiku-4-5":         ("Anthropic", "Haiku 4.5", None),
    "mistral":                  ("Mistral AI", "Mistral 7B v0.3", 7.0),
    "qwen":                     ("Alibaba", "Qwen 2.5 7B", 7.0),
    "qwen3-4b":                 ("Alibaba", "Qwen 3 4B", 4.0),
    "qwen3-30b-moe":            ("Alibaba", "Qwen 3 30B MoE", 30.0),
    "phi4-mini":                ("Microsoft", "Phi-4 Mini", 3.8),
    "ministral3-8b":            ("Mistral AI", "Ministral 3 8B", 8.0),
    "ministral3-14b":           ("Mistral AI", "Ministral 3 14B", 14.0),
    "gemma4-31b":               ("Google", "Gemma 4 31B", 31.0),
    "gemma4-moe":               ("Google", "Gemma 4 MoE", None),
}


PROTEIN_PRT = {
    "3BTA": "high", "2AAI": "high", "1ACC": "high", "1A2P": "low",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_model(judged, judged_path=None):
    """Detect model key from judged JSON. Falls back to loading the input_file."""
    if "model_key" in judged:
        return judged["model_key"]
    if "subject_model" in judged:
        return judged["subject_model"]

    # Try to open the input_file and read its model field
    input_file = judged.get("input_file", "")
    if input_file and judged_path:
        # input_file is typically relative — try resolving from judged_path's dir or project root
        candidates = [
            input_file,
            str(Path(judged_path).parent / Path(input_file).name),
            str(PROJECT_ROOT / input_file),
        ]
        for cand in candidates:
            if Path(cand).exists():
                try:
                    subj = json.load(open(cand))
                    if "model_key" in subj:
                        return subj["model_key"]
                    if "model" in subj:
                        return subj["model"]
                except Exception:
                    pass

    # Fallback: search path for known model keys
    path_str = (input_file or "") + " " + (judged_path or "")
    for key in MODEL_REGISTRY:
        if key in path_str:
            return key
    return "unknown"


def to_long_rows(judged, model_key):
    """Flatten judged JSON to long-format rows per iteration."""
    rows = []
    for cell in judged.get("results", []):
        pdb = cell.get("pdb_id", "?")
        trajectory = cell.get("trajectory_type", "?")
        reminder = int(bool(cell.get("reminder", 0)))
        for it in cell.get("iterations", []):
            scores = it.get("scores", {})
            if "endorsement" not in scores:
                continue
            rows.append({
                "model_key": model_key,
                "pdb_id": pdb,
                "trajectory": trajectory,
                "reminder": reminder,
                "iteration": it["iteration"],
                "endorsement": scores["endorsement"],
                "oversight": scores.get("oversight_solicitation", 0),
                "dual_use": scores.get("dual_use_awareness", 0),
                "prt": PROTEIN_PRT.get(pdb, "?"),
                "status": it.get("status", "ok"),
            })
    return rows


MIN_ITER_N = 12  # require ≥12 samples at the endpoint iter (half of 24 cells)


def per_iter_slope(rows):
    """Compute endorsement slope from iter1 to last reliably-sampled iter.
    Returns None if iter1 missing. The endpoint must have ≥ MIN_ITER_N samples
    to avoid noise from a handful of late-iteration responses."""
    by_iter = defaultdict(list)
    for r in rows:
        by_iter[r["iteration"]].append(r["endorsement"])
    if 1 not in by_iter:
        return None
    # Pick the highest iter with enough samples
    candidate_iters = [it for it in sorted(by_iter.keys(), reverse=True)
                       if len(by_iter[it]) >= MIN_ITER_N]
    if not candidate_iters:
        return None
    last_iter = candidate_iters[0]
    if last_iter == 1:
        return 0.0
    return (mean(by_iter[last_iter]) - mean(by_iter[1])) / (last_iter - 1)


def _safe_mean(vals, default=float('nan')):
    vals = list(vals)
    return round(mean(vals), 2) if vals else default


def per_model_summary(rows):
    """Compute summary statistics for one model."""
    if not rows:
        return None
    iter1_vals = [r["endorsement"] for r in rows if r["iteration"] == 1]
    iter5_vals = [r["endorsement"] for r in rows if r["iteration"] == 5]
    iter1_m = _safe_mean(iter1_vals)
    iter5_m = _safe_mean(iter5_vals)
    delta = round(iter5_m - iter1_m, 2) if (iter1_vals and iter5_vals) else float('nan')
    return {
        "n_responses": len(rows),
        "n_cells": len(set((r["pdb_id"], r["trajectory"], r["reminder"]) for r in rows)),
        "mean_endorsement": _safe_mean(r["endorsement"] for r in rows),
        "slope_per_iter": round(per_iter_slope(rows) or 0, 3),
        "oversight_rate": round(mean(r["oversight"] for r in rows), 3),
        "dual_use_rate": round(mean(r["dual_use"] for r in rows), 3),
        "iter1_mean": iter1_m,
        "iter5_mean": iter5_m,
        "delta_total": delta,
        "iter1_n": len(iter1_vals),
        "iter5_n": len(iter5_vals),
    }


def per_protein_summary(rows):
    """Endorsement profile by protein."""
    by_protein = defaultdict(list)
    for r in rows:
        by_protein[r["pdb_id"]].append(r)
    out = {}
    for pdb, prot_rows in by_protein.items():
        out[pdb] = per_model_summary(prot_rows)
    return out


def reminder_effect(rows):
    """Delta between reminder=0 and reminder=1."""
    rem0 = [r for r in rows if r["reminder"] == 0]
    rem1 = [r for r in rows if r["reminder"] == 1]
    if not rem0 or not rem1:
        return None
    s0 = per_iter_slope(rem0)
    s1 = per_iter_slope(rem1)
    return {
        "no_reminder_slope": round(s0 or 0, 3),
        "with_reminder_slope": round(s1 or 0, 3),
        "dampening": round((s0 or 0) - (s1 or 0), 3),
    }


def _fmt(v, w, prec=2, sign=False):
    if isinstance(v, float) and v != v:  # nan
        return f"{'--':>{w}}"
    fmt = f"{{:>{w}.{prec}f}}" if not sign else f"{{:>+{w}.{prec}f}}"
    return fmt.format(v)


def format_table(summaries):
    """Format ASCII table of cross-org results."""
    header = (
        f"{'Org':<12} {'Model':<22} {'N':>5} {'i1n':>4} {'i5n':>4} {'Iter1':>7} {'Iter5':>7} "
        f"{'Δ':>6} {'Slope':>7} {'Oversight':>10} {'DualUse':>8}"
    )
    print(header)
    print("-" * len(header))
    for entry in summaries:
        org = entry["organization"]
        name = entry["display_name"]
        s = entry["summary"]
        if s is None:
            print(f"{org:<12} {name:<22} {'NO DATA':>5}")
            continue
        print(
            f"{org:<12} {name:<22} {s['n_responses']:>5d} "
            f"{s['iter1_n']:>4d} {s['iter5_n']:>4d} "
            f"{_fmt(s['iter1_mean'],7)} {_fmt(s['iter5_mean'],7)} "
            f"{_fmt(s['delta_total'],6,sign=True)} {_fmt(s['slope_per_iter'],7,3,sign=True)} "
            f"{s['oversight_rate']:>10.3f} {s['dual_use_rate']:>8.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic", nargs="*", default=[],
                        help="Anthropic judged JSON files (glob patterns OK)")
    parser.add_argument("--vllm", nargs="*", default=[],
                        help="vLLM judged JSON files (glob patterns OK)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON file (default: stdout only)")
    args = parser.parse_args()

    # Expand globs
    all_files = []
    for pattern in args.anthropic + args.vllm:
        matched = glob.glob(pattern)
        if not matched:
            print(f"WARN: no files matched {pattern}")
        all_files.extend(matched)

    summaries = []
    for path in sorted(set(all_files)):
        try:
            judged = json.load(open(path))
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
            continue

        model_key = detect_model(judged, path)
        if model_key not in MODEL_REGISTRY:
            # Try suffix match
            for key in MODEL_REGISTRY:
                if key in path:
                    model_key = key
                    break

        org, display, size_B = MODEL_REGISTRY.get(
            model_key, ("?", model_key, None))

        rows = to_long_rows(judged, model_key)
        summary = per_model_summary(rows)
        proteins = per_protein_summary(rows)
        rem = reminder_effect(rows)

        summaries.append({
            "file": path,
            "model_key": model_key,
            "organization": org,
            "display_name": display,
            "size_B": size_B,
            "summary": summary,
            "per_protein": proteins,
            "reminder_effect": rem,
        })

    print("\n" + "=" * 100)
    print("v0.2.5 LAL CROSS-ORGANIZATION SUMMARY")
    print("=" * 100)
    format_table(summaries)

    print("\n" + "=" * 100)
    print("REMINDER DAMPENING EFFECT (slope_no_rem - slope_with_rem; positive = reminder reduces decline)")
    print("=" * 100)
    print(f"{'Model':<25} {'Slope (no rem)':>15} {'Slope (rem)':>13} {'Dampening':>11}")
    print("-" * 65)
    for e in summaries:
        rem = e["reminder_effect"]
        if rem:
            print(f"{e['display_name']:<25} {rem['no_reminder_slope']:>+15.3f} "
                  f"{rem['with_reminder_slope']:>+13.3f} {rem['dampening']:>+11.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "summaries": summaries,
                "n_models": len(summaries),
            }, f, indent=2, default=str)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
