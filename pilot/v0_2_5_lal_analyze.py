#!/usr/bin/env python3
"""
v0.2.5 LAL Analysis — slope, oversight decay, reminder effect

Per v0.2.5_LAL_Pilot_Protocol.md §5.

Reads a judged JSON file (output of v0_2_5_lal_judge.py) and runs:
  - H1: endorsement slope across iteration × trajectory_type × reminder
  - H2: oversight solicitation decay
  - H3: reminder × iteration interaction
  - H4: trajectory_type × protein_PRT interaction
  - Quick descriptive summary tables

Uses statsmodels for mixed-effects; falls back to ordinary least squares
if statsmodels missing.
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

# Protein PRT class for H4
PROTEIN_PRT = {
    "3BTA": "high", "2AAI": "high", "1ACC": "high", "1A2P": "low",
}


def to_long_df(judged):
    """Flatten judged JSON to long-format rows (one per iteration)."""
    rows = []
    for cell in judged["results"]:
        for it in cell["iterations"]:
            scores = it.get("scores", {})
            if "endorsement" not in scores:
                continue  # skip errored or unscored
            rows.append({
                "pdb_id": cell["pdb_id"],
                "trajectory_type": cell["trajectory_type"],
                "reminder": int(bool(cell["reminder"])),
                "iteration": it["iteration"],
                "endorsement": scores["endorsement"],
                "oversight": scores["oversight_solicitation"],
                "dual_use_aware": scores["dual_use_awareness"],
                "protein_prt": PROTEIN_PRT.get(cell["pdb_id"], "unknown"),
                "n_tool_calls": len(it.get("tool_calls", [])),
                "status": it["status"],
            })
    return rows


def descriptive_summary(rows):
    print("\n" + "=" * 70)
    print("DESCRIPTIVE SUMMARY")
    print("=" * 70)

    # By trajectory_type × iteration: mean endorsement
    by_tt_iter = defaultdict(list)
    for r in rows:
        by_tt_iter[(r["trajectory_type"], r["iteration"])].append(r["endorsement"])

    print("\nMean endorsement by trajectory_type × iteration:")
    print(f"{'trajectory':12s} | " + " | ".join(f"iter{i}" for i in range(1, 6)))
    print("-" * 60)
    for tt in ["control", "escalation", "saturation"]:
        vals = []
        for i in range(1, 6):
            v = by_tt_iter.get((tt, i), [])
            vals.append(f"{sum(v)/len(v):.2f}" if v else "  -  ")
        print(f"{tt:12s} | " + " | ".join(f"{v:>5s}" for v in vals))

    # Oversight solicitation
    by_tt_iter_o = defaultdict(list)
    for r in rows:
        by_tt_iter_o[(r["trajectory_type"], r["iteration"])].append(r["oversight"])

    print("\nOversight solicitation rate by trajectory_type × iteration:")
    print(f"{'trajectory':12s} | " + " | ".join(f"iter{i}" for i in range(1, 6)))
    print("-" * 60)
    for tt in ["control", "escalation", "saturation"]:
        vals = []
        for i in range(1, 6):
            v = by_tt_iter_o.get((tt, i), [])
            vals.append(f"{sum(v)/len(v):.2f}" if v else "  -  ")
        print(f"{tt:12s} | " + " | ".join(f"{v:>5s}" for v in vals))

    # By reminder
    print("\nMean endorsement by reminder × iteration:")
    print(f"{'reminder':12s} | " + " | ".join(f"iter{i}" for i in range(1, 6)))
    print("-" * 60)
    for rem in [0, 1]:
        vals = []
        for i in range(1, 6):
            v = [r["endorsement"] for r in rows
                 if r["reminder"] == rem and r["iteration"] == i]
            vals.append(f"{sum(v)/len(v):.2f}" if v else "  -  ")
        print(f"{'reminder=' + str(rem):12s} | " + " | ".join(f"{v:>5s}" for v in vals))

    # By protein
    print("\nMean endorsement by protein × iteration:")
    print(f"{'protein':12s} | " + " | ".join(f"iter{i}" for i in range(1, 6)))
    print("-" * 60)
    for pdb in ["3BTA", "2AAI", "1ACC", "1A2P"]:
        vals = []
        for i in range(1, 6):
            v = [r["endorsement"] for r in rows
                 if r["pdb_id"] == pdb and r["iteration"] == i]
            vals.append(f"{sum(v)/len(v):.2f}" if v else "  -  ")
        print(f"{pdb:12s} | " + " | ".join(f"{v:>5s}" for v in vals))


def fit_models(rows):
    if not HAVE_PANDAS or not HAVE_STATSMODELS:
        print("\n[!] pandas/statsmodels not installed; skipping models.")
        print("    Install: pip install pandas statsmodels")
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("MIXED-EFFECTS MODELS (statsmodels)")
    print("=" * 70)

    # H1: endorsement slope (linear mixed-effects, random slope on iter | protein)
    print("\nH1: Endorsement ~ iteration * trajectory_type * reminder")
    print("    Random: iteration | protein")
    try:
        # statsmodels MixedLM uses formula
        # Center iteration around 1 so intercept = iter1 mean
        df["iter_c"] = df["iteration"] - 1
        mdl = smf.mixedlm(
            "endorsement ~ iter_c * C(trajectory_type) * reminder",
            df, groups=df["pdb_id"],
            re_formula="~iter_c",
        )
        res = mdl.fit(method="lbfgs", disp=False)
        print(res.summary().tables[1])  # only fixed effects table
    except Exception as e:
        print(f"  H1 model failed: {e}")
        # Fallback: simple OLS for slope
        try:
            mdl = smf.ols("endorsement ~ iter_c * C(trajectory_type) * reminder", df).fit()
            print("  Fallback OLS (no random effects):")
            print(mdl.summary().tables[1])
        except Exception as e2:
            print(f"  OLS fallback also failed: {e2}")

    # H2: oversight decay (GLMM-binomial; statsmodels has BinomialBayesMixedGLM
    # but mostly we'd use R / glmmTMB; here we do GEE or GLM with cluster CIs)
    print("\nH2: Oversight ~ iteration + trajectory_type + reminder (GEE-binomial)")
    try:
        mdl = smf.gee(
            "oversight ~ iter_c + C(trajectory_type) + reminder",
            "pdb_id", df, family=sm.families.Binomial(),
        ).fit()
        print(mdl.summary().tables[1])
    except Exception as e:
        print(f"  H2 model failed: {e}")
        try:
            mdl = smf.glm("oversight ~ iter_c + C(trajectory_type) + reminder",
                          df, family=sm.families.Binomial()).fit()
            print("  Fallback plain GLM (no clustering):")
            print(mdl.summary().tables[1])
        except Exception as e2:
            print(f"  GLM fallback also failed: {e2}")

    # H3 is the reminder*iter interaction term in H1 above (already estimated)
    print("\nH3: see iter_c:reminder term in H1 model")

    # H4: trajectory × protein_PRT
    print("\nH4: Endorsement ~ iter_c * trajectory_type * protein_prt (PRT-only)")
    try:
        mdl = smf.mixedlm(
            "endorsement ~ iter_c * C(trajectory_type) * C(protein_prt)",
            df, groups=df["pdb_id"], re_formula="~iter_c",
        ).fit(method="lbfgs", disp=False)
        print(mdl.summary().tables[1])
    except Exception as e:
        print(f"  H4 model failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to judged JSON (output of v0_2_5_lal_judge.py)")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip statistical models, descriptives only")
    args = parser.parse_args()

    with open(args.input) as f:
        judged = json.load(f)
    rows = to_long_df(judged)
    if not rows:
        print("No scored rows found.")
        sys.exit(1)
    print(f"Loaded {len(rows)} iteration rows from {Path(args.input).name}")

    descriptive_summary(rows)
    if not args.no_models:
        fit_models(rows)


if __name__ == "__main__":
    main()
