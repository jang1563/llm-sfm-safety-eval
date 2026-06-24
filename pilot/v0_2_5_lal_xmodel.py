#!/usr/bin/env python3
"""
v0.2.5 LAL — Cross-model mixed-effects analysis

Ingests multiple per-model judged JSON files (output of v0_2_5_lal_judge.py
applied to each model's v0_2_5_lal_subject_*.json or v0_2_5_lal_hf_*.json),
fits a unified model with `model` as a categorical fixed effect, and reports:
  - Per-model slope estimates (with bootstrap CIs)
  - Cross-model contrasts (each open-weight vs Haiku reference)
  - Heterogeneity test (does drift differ across models?)
  - Reminder × model interaction

Usage:
    python3 v0_2_5_lal_xmodel.py file1.json file2.json file3.json ...
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

PROTEIN_PRT = {
    "3BTA": "high", "2AAI": "high", "1ACC": "high", "1A2P": "low",
}


def model_label_from_path(path):
    """Extract model display name from filename."""
    p = Path(path).name.lower()
    if "haiku" in p:
        return "Haiku-4.5"
    # Qwen variants (order matters — check most specific first)
    if "qwen3-30b-moe" in p or "qwen3_30b_moe" in p:
        return "Qwen3-30B-MoE"
    if "qwen3-4b" in p or "qwen3_4b" in p:
        return "Qwen3-4B"
    if "qwen3" in p:
        return "Qwen3"
    if "qwen" in p:
        return "Qwen-2.5-7B"
    # Mistral variants
    if "ministral3-8b" in p:
        return "Ministral3-8B"
    if "ministral3-14b" in p:
        return "Ministral3-14B"
    if "mistral" in p or "ministral" in p:
        return "Mistral-7B"
    # Llama variants
    if "llama" in p:
        if "70" in p:
            return "Llama-3.3-70B"
        return "Llama-3.x"
    # Gemma variants
    if "gemma4-moe" in p:
        return "Gemma4-MoE"
    if "gemma4-31b" in p:
        return "Gemma4-31B"
    if "gemma" in p:
        return "Gemma"
    if "phi4-mini" in p:
        return "Phi4-mini"
    if "phi" in p:
        return "Phi"
    if "deepseek" in p or "r1" in p:
        return "DeepSeek-R1"
    if "subject_2026" in p:
        return "Haiku-4.5"
    return p[:30]


def to_long_df(judged_path):
    """Flatten one judged JSON to long-format rows."""
    with open(judged_path) as f:
        d = json.load(f)
    rows = []
    model_label = model_label_from_path(judged_path)
    # If file has explicit model_id field, prefer that
    if "model_key" in d:
        model_label_from_meta = d.get("display_name", d["model_key"])
        if model_label_from_meta:
            model_label = model_label_from_meta
    # Determine provider
    provider_map = {
        "Haiku": "Anthropic", "Sonnet": "Anthropic", "Opus": "Anthropic",
        "Qwen": "Alibaba", "Mistral": "Mistral", "Mixtral": "Mistral",
        "Llama": "Meta", "Gemma": "Google", "Phi": "Microsoft",
        "DeepSeek": "DeepSeek", "R1": "DeepSeek",
    }
    provider = "Unknown"
    for k, v in provider_map.items():
        if k in model_label:
            provider = v
            break

    for cell in d["results"]:
        for it in cell["iterations"]:
            scores = it.get("scores", {})
            if "endorsement" not in scores:
                continue
            rows.append({
                "model": model_label,
                "provider": provider,
                "pdb_id": cell["pdb_id"],
                "trajectory_type": cell["trajectory_type"],
                "reminder": int(bool(cell["reminder"])),
                "iteration": it["iteration"],
                "endorsement": scores["endorsement"],
                "oversight": scores["oversight_solicitation"],
                "dual_use_aware": scores["dual_use_awareness"],
                "protein_prt": PROTEIN_PRT.get(cell["pdb_id"], "unknown"),
            })
    return rows


def descriptive_table(df):
    print("\n" + "=" * 80)
    print("PER-MODEL × TRAJECTORY DESCRIPTIVES")
    print("=" * 80)
    print(f"{'model':25s} {'provider':12s} {'traj':12s} " +
          " ".join(f"iter{i}" for i in range(1, 6)) +
          "   Δ(5-1)")
    print("-" * 90)
    for model in sorted(df["model"].unique()):
        provider = df[df["model"] == model]["provider"].iloc[0]
        for tt in ["control", "escalation", "saturation"]:
            sub = df[(df["model"] == model) & (df["trajectory_type"] == tt)]
            if len(sub) == 0:
                continue
            means = []
            for i in range(1, 6):
                v = sub[sub["iteration"] == i]["endorsement"]
                means.append(v.mean() if len(v) else float("nan"))
            delta = means[4] - means[0] if not np.isnan(means[0] + means[4]) else float("nan")
            print(f"{model:25s} {provider:12s} {tt:12s} " +
                  " ".join(f"{m:5.2f}" for m in means) +
                  f"   {delta:+.2f}")


def cross_model_models(df):
    if not HAVE_STATSMODELS:
        print("[!] statsmodels missing")
        return
    print("\n" + "=" * 80)
    print("CROSS-MODEL MIXED-EFFECTS (Haiku-4.5 reference)")
    print("=" * 80)

    df["iter_c"] = df["iteration"] - 1
    # Set reference category to Haiku
    models_in_data = list(df["model"].unique())
    if "Haiku-4.5" in models_in_data:
        ref = "Haiku-4.5"
    else:
        ref = sorted(models_in_data)[0]
    df["model"] = pd.Categorical(df["model"], categories=[ref] +
                                  [m for m in models_in_data if m != ref])

    # H_xmodel: endorsement ~ iteration * model * trajectory_type * reminder
    # Simplified to: iteration * model + trajectory_type + reminder + (1+iter|protein)
    print("\nModel 1: endorsement ~ iteration * model + trajectory_type + reminder")
    print("          random: iteration | protein")
    try:
        m1 = smf.mixedlm(
            "endorsement ~ iter_c * C(model) + C(trajectory_type) + reminder",
            df, groups=df["pdb_id"], re_formula="~iter_c"
        ).fit(method="lbfgs", disp=False)
        print(m1.summary().tables[1])
    except Exception as e:
        print(f"  Failed: {e}")
        try:
            m1 = smf.ols(
                "endorsement ~ iter_c * C(model) + C(trajectory_type) + reminder", df
            ).fit()
            print("  Fallback OLS:")
            print(m1.summary().tables[1])
        except Exception as e2:
            print(f"  OLS fallback: {e2}")

    # Heterogeneity test
    print("\nModel 2: heterogeneity test (iter_c × model × trajectory)")
    try:
        m2 = smf.ols(
            "endorsement ~ iter_c * C(model) * C(trajectory_type) + reminder", df
        ).fit()
        # Print only iter_c × model × trajectory interaction terms
        params = m2.params
        print("  Three-way interaction coefficients (iter_c:model:trajectory):")
        for name, val in params.items():
            if "iter_c" in name and ":" in name and "T." in name:
                ci = m2.conf_int().loc[name]
                print(f"    {name[:60]:60s}  {val:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    except Exception as e:
        print(f"  Failed: {e}")


def reminder_effect_by_model(df):
    print("\n" + "=" * 80)
    print("REMINDER × MODEL INTERACTION (does reminder dampen drift more in some models?)")
    print("=" * 80)
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        for rem in [0, 1]:
            s = sub[sub["reminder"] == rem]
            # avg slope iter1 → iter5 for this (model, reminder) cell
            if len(s) == 0:
                continue
            mean_i1 = s[s["iteration"] == 1]["endorsement"].mean()
            mean_i5 = s[s["iteration"] == 5]["endorsement"].mean()
            print(f"  {model:25s} reminder={rem}  iter1={mean_i1:.2f}  iter5={mean_i5:.2f}  Δ={mean_i5-mean_i1:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="Judged JSON files")
    p.add_argument("--no-models", action="store_true")
    args = p.parse_args()

    all_rows = []
    for f in args.files:
        rows = to_long_df(f)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {Path(f).name}  (model={rows[0]['model'] if rows else '?'})")

    if not HAVE_PANDAS:
        print("ERROR: pandas required")
        sys.exit(1)
    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} rows across {df['model'].nunique()} models")
    print(f"Models: {sorted(df['model'].unique())}")

    descriptive_table(df)
    reminder_effect_by_model(df)
    if not args.no_models:
        cross_model_models(df)


if __name__ == "__main__":
    main()
