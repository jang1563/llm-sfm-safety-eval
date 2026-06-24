#!/usr/bin/env python3
"""
Phase 4 analysis — aggregates C, A, B results into summary statistics and findings.

Usage:
    python3 pilot/analyze_phase4.py [--c FILE] [--a FILE] [--b FILE]

If no files specified, finds the latest result files matching each pattern.
"""

import json, sys, argparse
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

def find_latest(pattern):
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None

# ── C: Natural Homolog Analysis ──

def analyze_c(path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"\n{'='*60}")
    print(f"C: NATURAL HOMOLOG PROBE ({len(results)} calls)")
    print(f"{'='*60}")

    # Group by (family, identity%, id_level)
    groups = defaultdict(lambda: {"ref": 0, "ok": 0, "err": 0})
    for r in results:
        key = (r["family"], r["identity_pct"], r["id_level"])
        if r.get("error"):
            groups[key]["err"] += 1
        elif r["stop_reason"] == "refusal":
            groups[key]["ref"] += 1
        else:
            groups[key]["ok"] += 1

    print(f"\n  {'Family':8s} {'Identity':10s} {'ID':5s}  {'Refusal':8s} {'OK':5s} {'Err':5s} {'Rate':8s}")
    print(f"  {'-'*55}")
    for (fam, ident, id_level), counts in sorted(groups.items()):
        total = counts["ref"] + counts["ok"]
        rate = counts["ref"] / total * 100 if total > 0 else 0
        print(f"  {fam:8s} {ident:6.1f}%     {id_level:5s}  {counts['ref']:5d}    {counts['ok']:5d} {counts['err']:5d}  {rate:5.1f}%")

    # Summary by identity level
    print(f"\n  Summary by identity level:")
    by_ident = defaultdict(lambda: {"ref": 0, "total": 0})
    for (fam, ident, id_level), counts in groups.items():
        key = f"{ident:.0f}%"
        total = counts["ref"] + counts["ok"]
        by_ident[key]["ref"] += counts["ref"]
        by_ident[key]["total"] += total

    for key, v in sorted(by_ident.items(), key=lambda x: float(x[0].rstrip('%'))):
        rate = v["ref"] / v["total"] * 100 if v["total"] > 0 else 0
        print(f"    {key:6s}: {v['ref']}/{v['total']} = {rate:.1f}%")

    return groups

# ── A: Multi-turn ID Transition Analysis ──

def analyze_a(path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"\n{'='*60}")
    print(f"A: MULTI-TURN ID TRANSITION ({len(results)} conversations)")
    print(f"{'='*60}")

    # Per-turn refusal rate by pattern
    by_pattern_turn = defaultdict(lambda: defaultdict(lambda: {"ref": 0, "total": 0}))
    by_protein_pattern = defaultdict(lambda: {"ref_total": 0, "turn_total": 0})

    for r in results:
        pattern = r["pattern"]
        protein = r["protein"]
        for t in r["turns"]:
            turn = t["turn"]
            by_pattern_turn[pattern][turn]["total"] += 1
            if t["stop_reason"] == "refusal":
                by_pattern_turn[pattern][turn]["ref"] += 1
                by_protein_pattern[(protein, pattern)]["ref_total"] += 1
            by_protein_pattern[(protein, pattern)]["turn_total"] += 1

    # Detect turn count from data
    max_turn = max(t["turn"] for r in results for t in r["turns"]) + 1
    print(f"\n  Refusal rate by pattern × turn (0-indexed, {max_turn} turns):")
    header = f"  {'Pattern':12s}" + "".join(f"  {'T'+str(i):8s}" for i in range(max_turn))
    print(header)
    print(f"  {'-'*len(header)}")
    for pattern in ["ascending", "descending", "mixed", "control"]:
        if pattern not in by_pattern_turn:
            continue
        row = []
        for turn in range(max_turn):
            d = by_pattern_turn[pattern][turn]
            rate = d["ref"] / d["total"] * 100 if d["total"] > 0 else 0
            row.append(f"{rate:5.1f}%")
        print(f"  {pattern:12s}  {'  '.join(row)}")

    # Turn-state carrying: does refusal at turn N predict refusal at turn N+1?
    print(f"\n  Turn-state carrying (P(refusal@T+1 | refusal@T)):")
    for pattern in ["ascending", "descending", "mixed", "control"]:
        if pattern not in by_pattern_turn:
            continue
        carry_count = 0
        carry_total = 0
        for r in results:
            if r["pattern"] != pattern:
                continue
            turns = r["turns"]
            for i in range(len(turns) - 1):
                if turns[i]["stop_reason"] == "refusal":
                    carry_total += 1
                    if turns[i+1]["stop_reason"] == "refusal":
                        carry_count += 1
        rate = carry_count / carry_total * 100 if carry_total > 0 else 0
        print(f"    {pattern:12s}: {carry_count}/{carry_total} = {rate:.1f}%")

    # By protein
    print(f"\n  Refusal rate by protein × pattern:")
    for (protein, pattern), d in sorted(by_protein_pattern.items()):
        rate = d["ref_total"] / d["turn_total"] * 100 if d["turn_total"] > 0 else 0
        print(f"    {protein:15s} {pattern:12s}: {d['ref_total']}/{d['turn_total']} = {rate:.1f}%")

    return by_pattern_turn

# ── B: Channel × ID Matrix Analysis ──

def analyze_b(path):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"\n{'='*60}")
    print(f"B: CHANNEL x ID MATRIX ({len(results)} calls)")
    print(f"{'='*60}")

    # Refusal rate by arm × id_level
    by_arm_id = defaultdict(lambda: {"ref": 0, "total": 0})
    by_protein_arm = defaultdict(lambda: {"ref": 0, "total": 0})

    for r in results:
        arm = r["arm"]
        id_level = r["id_level"]
        protein = r["protein"]
        by_arm_id[(arm, id_level)]["total"] += 1
        by_protein_arm[(protein, arm)]["total"] += 1
        if r["stop_reason"] == "refusal":
            by_arm_id[(arm, id_level)]["ref"] += 1
            by_protein_arm[(protein, arm)]["ref"] += 1

    arms = sorted(set(r["arm"] for r in results))
    ids = sorted(set(r["id_level"] for r in results))

    print(f"\n  Refusal rate: Arm × ID level")
    header = f"  {'Arm':20s}" + "".join(f"  {id_l:8s}" for id_l in ids)
    print(header)
    print(f"  {'-'*len(header)}")
    for arm in arms:
        row = []
        for id_l in ids:
            d = by_arm_id[(arm, id_l)]
            rate = d["ref"] / d["total"] * 100 if d["total"] > 0 else 0
            row.append(f"  {rate:6.1f}%")
        print(f"  {arm:20s}{'  '.join(row)}")

    # By protein × arm
    proteins = sorted(set(r["protein"] for r in results))
    print(f"\n  Refusal rate: Protein × Arm")
    header = f"  {'Protein':15s}" + "".join(f"  {arm:15s}" for arm in arms)
    print(header)
    print(f"  {'-'*len(header)}")
    for protein in proteins:
        row = []
        for arm in arms:
            d = by_protein_arm[(protein, arm)]
            rate = d["ref"] / d["total"] * 100 if d["total"] > 0 else 0
            row.append(f"  {rate:13.1f}%")
        print(f"  {protein:15s}{'  '.join(row)}")

    return by_arm_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", help="C probe result file")
    parser.add_argument("--a", help="A multi-turn result file")
    parser.add_argument("--b", help="B matrix result file")
    args = parser.parse_args()

    c_path = args.c or find_latest("c_homolog_probe_*.json")
    a_path = args.a or find_latest("a_multiturn_id_*.json")
    b_path = args.b or find_latest("b_channel_id_matrix_*.json")

    if c_path:
        print(f"C: {c_path}")
        analyze_c(c_path)
    else:
        print("C: no result file found")

    if a_path:
        print(f"\nA: {a_path}")
        analyze_a(a_path)
    else:
        print("A: no result file found")

    if b_path:
        print(f"\nB: {b_path}")
        analyze_b(b_path)
    else:
        print("B: no result file found")

if __name__ == "__main__":
    main()
