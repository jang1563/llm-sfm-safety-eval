#!/usr/bin/env python3
"""
E1_overtrust_run.py — E1 discrimination subject-collection runner (Spec §10).

Collects RAW subject responses for the over-trust discrimination study. Default:
the single gradable Claude model from the §8.4 smoke (claude-sonnet-4-20250514),
all 45 value-table cells (8 core proteins x S0-S4 + barnase block) at P3, 10 reps
= 450 calls. Order is randomized (cell x rep interleaved, fixed seed) per §10 so
all 10 reps of a cell are not run back-to-back.

This runner does NOT score. Caveat code (0/1/2), uptake direction (-1/0/+1), and
payload are computed downstream by E1_overtrust_judge.py + a rule-based pre-pass on
the *engaged* subset. Here we only attach an engagement pre-pass flag (refused vs
engaged, identical logic to the smoke) and persist everything (prompt, full response,
stop_reason, ground-truth cell flags) so re-judging needs no re-querying.

Safe-conduct: prompts carry PDB ID + metric JSON only (never the common name, never
the amino-acid sequence/FASTA; §6.1/§8.3). No novel sequences. No synthesis. Values
come from the frozen value table. The payload metric scores the model's OVER-CLAIM
(a safety failure), never any harmful content. Defensive AI-biosafety (Mason Lab, WCM).

Usage:
  source ~/.api_keys && python3 E1_overtrust_run.py --dry-run
  source ~/.api_keys && python3 E1_overtrust_run.py                       # 450 calls, Sonnet 4
  source ~/.api_keys && python3 E1_overtrust_run.py --reps 1 --proteins 3BTA --signals S0 S1 S3
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
except ImportError:
    print("ERROR: anthropic not installed. pip install anthropic")
    sys.exit(1)

# Reuse the smoke module's helpers so the prompt assembly + refusal classification
# are a SINGLE source of truth across smoke and full run.
from E1_overtrust_smoke import build_prompt, is_refused, load_cells, INTER_CALL_DELAY

# ----------------------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-sonnet-4-20250514"   # sole gradable Claude model (§8.4 smoke)
TEMPERATURE = 1.0
MAX_TOKENS = 2048                            # generous: avoid truncating scored content
REPS = 10
SEED = 20260530

# Cell fields copied verbatim into each result row (the ground truth the judge/analysis need).
CELL_FIELDS = [
    "cell_id", "pdb_id", "signal", "presentation", "frame", "value_shown",
    "metric_role", "appropriate_behavior", "hazardous", "is_true_benign",
    "is_lexical_baseline_cell", "is_falsepos_probe", "do_arr_cell",
    "s4_scramble_degenerate", "optional",
]


# ----------------------------------------------------------------------------- runner
class SubjectRunner:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS):
        self.client: Optional[Anthropic] = None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.results: List[Dict] = []
        self.call_count = 0
        self.error_count = 0

    def _c(self) -> Anthropic:
        if self.client is None:
            self.client = Anthropic()
        return self.client

    def call_api(self, prompt: str, max_retries: int = 3) -> Tuple[str, str]:
        delay = 1.0
        for attempt in range(max_retries):
            try:
                r = self._c().messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.call_count += 1
                text = r.content[0].text if r.content else ""
                time.sleep(INTER_CALL_DELAY)
                return text, r.stop_reason
            except (RateLimitError, APIConnectionError, APIError) as e:
                self.error_count += 1
                if attempt < max_retries - 1:
                    print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}; {delay:.0f}s")
                    time.sleep(delay); delay *= 2.0
                else:
                    raise

    def _dump(self, path: Path, reps: int, seed: int) -> None:
        scored = [r for r in self.results if r["refused"] is not None]
        path.write_text(json.dumps({
            "experiment": "E1 discrimination — subject collection (Spec §10)",
            "spec": "E1_LLM_SFM_OverTrust_Spec.md",
            "value_table": VALUE_TABLE_NAME,
            "timestamp": datetime.now().isoformat(),
            "model": self.model, "temperature": self.temperature,
            "max_tokens": self.max_tokens, "presentation": "P3",
            "reps": reps, "seed": seed,
            "n_calls": self.call_count, "n_errors": self.error_count,
            "n_engaged": sum(1 for r in scored if not r["refused"]),
            "results": self.results,
        }, indent=2))

    def run(self, cells: List[Dict], reps: int, seed: int, dry_run: bool = False,
            checkpoint_every: int = 50) -> None:
        tasks = [(c, rep) for c in cells for rep in range(reps)]
        random.Random(seed).shuffle(tasks)          # interleave S/P/protein (§10)
        total_all = len(tasks)

        # resume: skip (cell_id, rep) already collected (preloaded via load_resume).
        # Rows with refused=None (API errors) are NOT preloaded, so they get retried.
        done_keys = {(r["cell_id"], r["rep"]) for r in self.results if r.get("refused") is not None}
        if done_keys:
            tasks = [(c, rep) for (c, rep) in tasks if (c["cell_id"], rep) not in done_keys]
        total = len(tasks)

        print(f"\n{'='*72}\nE1 DISCRIMINATION RUN (subject collection, Spec §10)\n{'='*72}")
        print(f"model    : {self.model}  T={self.temperature}  max_tokens={self.max_tokens}")
        print(f"cells    : {len(cells)}  ({sorted({c['pdb_id'] for c in cells})})")
        print(f"signals  : {sorted({c['signal'] for c in cells})}")
        print(f"reps     : {reps}  seed={seed}  ->  {total_all} subject calls")
        if done_keys:
            print(f"resume   : {len(done_keys)} already collected  ->  {total} remaining")
        print(f"value tbl: {VALUE_TABLE_NAME}\n")

        if dry_run:
            seen = set()
            for c, _ in tasks:
                if c["cell_id"] in seen:
                    continue
                seen.add(c["cell_id"])
                print(f"--- {c['cell_id']}  frame={c['frame']}  haz={c['hazardous']}  "
                      f"role={c['metric_role']} ---")
                print(build_prompt(c))
                print()
            print(f"[dry-run] {total} calls NOT made ({len(seen)} unique cells).")
            return

        done = len(self.results)                       # preloaded completed rows (resume)
        eng = sum(1 for r in self.results if r.get("refused") is False)
        t0 = datetime.now()
        for i, (c, rep) in enumerate(tasks, 1):
            prompt = build_prompt(c)
            row = {k: c.get(k) for k in CELL_FIELDS}
            row.update({"model": self.model, "rep": rep, "prompt": prompt})
            try:
                text, stop = self.call_api(prompt)
            except Exception as e:
                row.update({"response": None, "stop_reason": "ERROR",
                            "refused": None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                self.results.append(row)
                print(f"  [{i}/{total}] {c['cell_id']} rep{rep}: FAILED {type(e).__name__}")
                continue
            refused = is_refused(text, stop)
            if not refused:
                eng += 1
            row.update({"response": text, "stop_reason": stop,
                        "refused": refused, "error": None})
            self.results.append(row)
            if i % 25 == 0 or i == total:
                overall = done + i
                print(f"  [{i}/{total} new | {overall}/{total_all}] engaged: {eng}/{overall} "
                      f"({eng/overall:.0%})  elapsed={str(datetime.now()-t0).split('.')[0]}")
            if checkpoint_every and i % checkpoint_every == 0 and i < total:
                self._dump(HERE / "results" / "E1_subject_partial.json", reps, seed)

        print(f"\n{'='*72}\ncalls={self.call_count}  errors={self.error_count}  "
              f"engaged={eng}/{len(self.results)}\n{'='*72}")
        self._engagement_by_signal()

    def _engagement_by_signal(self) -> None:
        scored = [r for r in self.results if r["refused"] is not None]
        sigs = sorted({r["signal"] for r in scored})
        print("engagement by signal (pre-pass; metrics computed engaged-only downstream):")
        for s in sigs:
            rs = [r for r in scored if r["signal"] == s]
            e = sum(1 for r in rs if not r["refused"])
            print(f"  {s}: {e}/{len(rs)} engaged")

    def save(self, reps: int, seed: int, out_dir: Path = None) -> Path:
        out_dir = out_dir or (HERE / "results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"E1_subject_{ts}.json"
        self._dump(out, reps, seed)
        print(f"\nsaved -> {out}  ({len(self.results)} rows)")
        return out

    def load_resume(self, path: Path) -> int:
        """Preload COMPLETED rows (refused is not None) from a prior subject/partial JSON.
        Errored rows (refused=None) are dropped so run() retries them. Returns # loaded.
        Idempotent on (cell_id, rep): later files win if keys overlap."""
        prior = json.loads(path.read_text())
        seen = {(r["cell_id"], r["rep"]) for r in self.results}
        keep = [r for r in prior.get("results", [])
                if r.get("refused") is not None and (r["cell_id"], r["rep"]) not in seen]
        self.results.extend(keep)
        return len(keep)


# ----------------------------------------------------------------------------- main
VALUE_TABLE_NAME = "E1_value_table_20260530.json"


def main():
    ap = argparse.ArgumentParser(description="E1 discrimination subject-collection runner")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--proteins", nargs="+", default=None, help="subset of PDB IDs (default: all)")
    ap.add_argument("--signals", nargs="+", default=None, help="subset of S0..S4 (default: all)")
    ap.add_argument("--presentation", default="P3")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", default=None,
                    help="prior subject/partial JSON; skip already-collected (cell_id,rep), retry errors")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells, _meta = load_cells()
    sel = [c for c in cells
           if c["presentation"] == args.presentation
           and (args.proteins is None or c["pdb_id"] in args.proteins)
           and (args.signals is None or c["signal"] in args.signals)]
    if not sel:
        print("ERROR: no cells match the selection.")
        sys.exit(1)

    runner = SubjectRunner(model=args.model, max_tokens=args.max_tokens)
    if args.resume:
        rp = Path(args.resume)
        if not rp.is_absolute():
            rp = HERE / rp
        n = runner.load_resume(rp)
        print(f"[resume] preloaded {n} completed rows from {rp.name}")
    runner.run(sel, args.reps, args.seed, dry_run=args.dry_run)
    if not args.dry_run:
        runner.save(args.reps, args.seed)


if __name__ == "__main__":
    main()
