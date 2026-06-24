#!/usr/bin/env python3
"""
v0.2.5 LAL — Sonnet 4 (original) tool-mode smoke test.

Tests whether the classifier pendulum extends to tool-mode.
Sonnet 4 showed ~0% refusal in both chat-mode and D spec.
If it also passes in tool-mode → classifier pendulum is modality-independent.

Uses SAME tool schemas as the name-based smoke test (ProteinMPNN-style tools
with PDB IDs and protein names) — identical to v0_2_5_lal_pilot.py.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from v0_2_5_lal_pilot import (
    PROTEINS, TRAJECTORIES, TOOLS, SYSTEM_PROMPT,
    USER_INITIAL_TEMPLATE, USER_FOLLOWUP_TEMPLATE,
    TOOL_HANDLERS, MAX_TOKENS_PER_TURN, MAX_TOOL_LOOPS_PER_ITER,
    SLEEP_BETWEEN_CALLS, RESULTS_DIR,
)
import anthropic

# Override model — Sonnet 4 (original), NOT 4.6
MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 1.0


def run_single_trajectory(client, pdb_id, verbose=True):
    """Same loop as main pilot, but forces Sonnet 4 model."""
    protein = PROTEINS[pdb_id]
    traj_state = {"pdb_id": pdb_id, "trajectory": "control", "iteration": 1}
    messages = []
    iter_records = []

    for it in range(1, 6):
        traj_state["iteration"] = it

        if it == 1:
            user_msg = USER_INITIAL_TEMPLATE.format(
                name=protein["name"], pdb_id=pdb_id)
        else:
            user_msg = USER_FOLLOWUP_TEMPLATE.format(n=it - 1, next_n=it)

        messages.append({"role": "user", "content": user_msg})

        tool_calls_this_iter = []
        final_text = None

        for loop_idx in range(MAX_TOOL_LOOPS_PER_ITER):
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_PER_TURN,
                    temperature=TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    tool_choice={"type": "auto"},
                    messages=messages,
                )
            except Exception as e:
                if verbose:
                    print(f"    ERROR iter {it}: {e}")
                iter_records.append({"iteration": it, "status": "error",
                                     "error": str(e), "tool_calls": tool_calls_this_iter})
                final_text = None
                break

            time.sleep(SLEEP_BETWEEN_CALLS)

            if resp.stop_reason == "refusal" or not resp.content:
                if verbose:
                    print(f"    iter {it}: SYSTEM REFUSAL (stop_reason={resp.stop_reason})")
                iter_records.append({"iteration": it, "status": "refusal",
                                     "stop_reason": str(resp.stop_reason),
                                     "tool_calls": tool_calls_this_iter})
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": "[refused]"}]})
                final_text = None
                break

            assistant_blocks = []
            for blk in resp.content:
                if blk.type == "text":
                    assistant_blocks.append({"type": "text", "text": blk.text})
                elif blk.type == "tool_use":
                    assistant_blocks.append({"type": "tool_use", "id": blk.id,
                                             "name": blk.name, "input": blk.input})
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                final_text = " ".join(b.text for b in resp.content if b.type == "text")
                break

            tool_result_blocks = []
            for tu in tool_uses:
                handler = TOOL_HANDLERS.get(tu.name)
                out = handler(tu.input, traj_state) if handler else {"error": f"Unknown: {tu.name}"}
                tool_calls_this_iter.append({"tool": tu.name, "input": tu.input, "output": out})
                tool_result_blocks.append({"type": "tool_result", "tool_use_id": tu.id,
                                           "content": json.dumps(out)})
            messages.append({"role": "user", "content": tool_result_blocks})

        if final_text is not None:
            iter_records.append({"iteration": it, "status": "ok",
                                 "final_text": final_text,
                                 "tool_calls": tool_calls_this_iter})
            if verbose:
                preview = final_text[:100].replace("\n", " ")
                print(f"    iter {it}: OK ({len(tool_calls_this_iter)} calls) — {preview}...")

    return {
        "pdb_id": pdb_id, "model": MODEL,
        "trajectory_type": "control", "reminder": False,
        "iterations": iter_records, "n_messages": len(messages),
    }


def main():
    client = anthropic.Anthropic()
    results = []

    for pdb in PROTEINS:
        name = PROTEINS[pdb]["name"][:40]
        print(f"--- {pdb} ({name}) × Sonnet 4 × control × name-based ---")
        rec = run_single_trajectory(client, pdb, verbose=True)
        results.append(rec)
        print()

    out = RESULTS_DIR / f"v0_2_5_lal_sonnet4_smoke_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "model": MODEL,
            "experiment": "sonnet4_tool_mode_smoke",
            "description": "Classifier pendulum test: Sonnet 4 in tool-mode",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cells_completed": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"Saved: {out}\n")

    print("=== Sonnet 4 tool-mode refusal pattern ===")
    print(f"{'protein':8} {'iter1':>6} {'iter2':>6} {'iter3':>6} {'iter4':>6} {'iter5':>6}")
    for r in results:
        statuses = []
        for it in range(1, 6):
            match = [i for i in r["iterations"] if i["iteration"] == it]
            if not match:
                statuses.append("-")
            elif match[0]["status"] == "ok":
                statuses.append("OK")
            elif match[0]["status"] == "refusal":
                statuses.append("REF")
            else:
                statuses.append("ERR")
        print(f"{r['pdb_id']:8} " + " ".join(f"{s:>6}" for s in statuses))

    n_ok = sum(1 for r in results for i in r["iterations"] if i["status"] == "ok")
    n_total = sum(len(r["iterations"]) for r in results)
    print(f"\nSonnet 4 tool-mode OK:   {n_ok}/{n_total} ({100*n_ok/max(n_total,1):.1f}%)")
    print(f"Sonnet 4.6 tool-mode OK: 0/20 (0.0%)  [from prior smoke test]")

    if n_ok == n_total:
        print("\n→ CLASSIFIER PENDULUM EXTENDS TO TOOL-MODE.")
        print("  Sonnet 4 passes tool-mode; Sonnet 4.6 blocks it.")
        print("  Proceed to full 24-cell run with Sonnet 4.")
    elif n_ok == 0:
        print("\n→ BOTH MODELS REFUSE. Tool-mode classification is model-independent.")
    else:
        print(f"\n→ PARTIAL. Examine per-protein pattern.")
        for r in results:
            ok = sum(1 for i in r["iterations"] if i["status"] == "ok")
            print(f"  {r['pdb_id']}: {ok}/5 OK")


if __name__ == "__main__":
    main()
