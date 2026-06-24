#!/usr/bin/env python3
"""Merge v0.2.6 powered-run shard checkpoints into one subject file.

Each shard wrote results/v0_2_6_lal_subject_powered_shard{i}of{n}_<ts>.json.
This concatenates them, dedupes by (pdb_id, trajectory_type, mode, rep)
keeping the most-recently-written copy, validates per-stratum completeness,
and writes a single judge-compatible subject JSON.

Usage:
    python3 v0_2_6_merge_shards.py [--glob PATTERN] [--expect-reps 10]
"""
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_GLOB = "v0_2_6_lal_subject_powered_shard*of*_*.json"
# The completeness matrix discovers its protein roster from the merged cell-runs
# (each carries pdb_id/pair/role), so this generalizes to the full 15-pair panel.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--expect-reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(str(RESULTS_DIR / args.glob)))
    if not paths:
        print(f"No shard files match {args.glob}")
        return
    print(f"Merging {len(paths)} shard file(s):")
    for p in paths:
        print(f"  {Path(p).name}")

    # Dedupe by key; later files (mtime) win. Sort by mtime ascending so the
    # newest write of any duplicated cell-run overwrites earlier ones.
    paths_by_mtime = sorted(paths, key=lambda p: Path(p).stat().st_mtime)
    by_key = {}
    meta = {}
    for p in paths_by_mtime:
        d = json.load(open(p))
        meta = {k: d.get(k) for k in ("model", "tau", "wt_ref_fsi", "version", "surfaces")}
        for r in d.get("results", []):
            key = (r["pdb_id"], r["trajectory_type"], r["mode"], r.get("rep", 1))
            by_key[key] = r
    results = list(by_key.values())

    # Completeness matrix: (pdb, traj, mode) -> rep count
    counts = {}
    for r in results:
        k = (r["pdb_id"], r["trajectory_type"], r["mode"])
        counts[k] = counts.get(k, 0) + 1

    # Discover the protein roster from the data, ordered by pair then role.
    pdb_meta = {r["pdb_id"]: (r.get("pair", ""), r.get("role", "")) for r in results}

    def _ord(pdb):
        pair, role = pdb_meta[pdb]
        pn = int(pair[1:]) if pair[1:].isdigit() else 10**9
        return (pn, 0 if role == "hazard" else 1, pdb)

    pdbs = sorted(pdb_meta, key=_ord)
    trajs = sorted({k[1] for k in counts}) or ["escalation", "control"]
    modes = sorted({k[2] for k in counts}) or ["matched", "natural"]

    print(f"\nCompleteness (cells with rep-count; target N={args.expect_reps}):")
    print(f"  {'pdb':6}", end="")
    hdr = [f"{t[:4]}/{m[:4]}" for t in trajs for m in modes]
    print("".join(f"{h:>12}" for h in hdr))
    total, complete = 0, 0
    for pdb in pdbs:
        print(f"  {pdb:6}", end="")
        for t in trajs:
            for m in modes:
                c = counts.get((pdb, t, m), 0)
                total += 1
                if c >= args.expect_reps:
                    complete += 1
                mark = "" if c >= args.expect_reps else " *"
                print(f"{str(c)+mark:>12}", end="")
        print()
    print(f"\n  strata complete: {complete}/{total}  "
          f"(* = under target; {len(results)} total cell-runs)")

    # Refusal sanity
    refused = sum(r.get("refusal_count", 0) for r in results)
    print(f"  total refused iters across all reps: {refused}")

    out = Path(args.out) if args.out else \
        RESULTS_DIR / f"v0_2_6_lal_subject_powered_merged_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "version": meta.get("version", "v0.2.6-P1"),
            "surfaces": meta.get("surfaces", ["S1_prime_differential", "S4_prime_ccs"]),
            "model": meta.get("model"), "tau": meta.get("tau"),
            "wt_ref_fsi": meta.get("wt_ref_fsi"),
            "modes": modes,
            "merged_from": [Path(p).name for p in paths],
            "timestamp": datetime.now().isoformat(),
            "n_cells": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nMerged -> {out}")


if __name__ == "__main__":
    main()
