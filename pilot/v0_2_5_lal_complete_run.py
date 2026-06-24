#!/usr/bin/env python3
"""
Complete remaining 12 cells (1ACC + 1A2P) of the 24-cell pilot on Sonnet 4.
Cells 1-12 (3BTA, 2AAI) already in snapshot_20260525_103604.json.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
import v0_2_5_lal_pilot as pilot
import anthropic

MODEL = "claude-sonnet-4-20250514"


def main():
    # Override model in pilot module
    pilot.MODEL = MODEL

    client = anthropic.Anthropic()

    # Only run remaining proteins
    remaining_pdbs = ["1ACC", "1A2P"]
    cells = []
    for pdb in remaining_pdbs:
        for traj in pilot.TRAJECTORIES:
            for reminder in [False, True]:
                cells.append((pdb, traj, reminder))

    print(f"Completing remaining {len(cells)} cells on {MODEL}\n")

    results = []
    t_start = time.time()

    for idx, (pdb, traj, reminder) in enumerate(cells, 1):
        rem_str = "REMIND" if reminder else "no-rem"
        print(f"[{idx:2d}/{len(cells)}] {pdb} × {traj:10s} × {rem_str}")
        rec = pilot.run_single_trajectory(client, pdb, traj, reminder,
                                          framing="A", verbose=True)
        results.append(rec)

    elapsed = time.time() - t_start
    print(f"\nDone. {elapsed:.1f} sec ({elapsed/60:.1f} min)")

    # Save completion batch
    out = pilot.RESULTS_DIR / f"v0_2_5_lal_complete_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "model": MODEL,
            "batch": "completion_cells_13_24",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cells_completed": len(results),
            "elapsed_sec": elapsed,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Saved: {out}")

    # Merge with existing snapshot
    snapshot = pilot.RESULTS_DIR / "v0_2_5_lal_snapshot_20260525_103604.json"
    if snapshot.exists():
        with open(snapshot) as f:
            existing = json.load(f)
        merged = existing["results"] + results
        merged_out = pilot.RESULTS_DIR / f"v0_2_5_lal_subject_merged_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(merged_out, "w") as f:
            json.dump({
                "model": MODEL,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cells_completed": len(merged),
                "elapsed_sec": elapsed,
                "results": merged,
            }, f, indent=2, default=str)
        print(f"Merged 24 cells: {merged_out}")

    # Summary
    print("\n--- Completion batch ---")
    for r in results:
        ok = sum(1 for i in r["iterations"] if i["status"] == "ok")
        ref = sum(1 for i in r["iterations"] if i["status"] == "refusal")
        rem = "R" if r["reminder"] else "-"
        print(f"  {r['pdb_id']:5} × {r['trajectory_type']:11} × {rem}  OK={ok} REF={ref}")


if __name__ == "__main__":
    main()
