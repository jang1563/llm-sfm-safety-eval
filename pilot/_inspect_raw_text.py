#!/usr/bin/env python3
"""Pull raw (un-masked) subject final_text for hazard cells and show what the
judge actually saw after sanitization. Decides floor-artifact vs true-null:
  - if RAW has rich hazard rhetoric but MASK strips it -> artifact
  - if RAW itself is confident design w/o hazard language -> genuine null
Also greps RAW corpus for risk-word frequency (un-maskable generic terms)."""
import json, glob, re, sys
from collections import Counter
sys.path.insert(0, ".")
from chem_lal_judge import sanitize_response

shards = sorted(glob.glob("results/chem_lal_subject_shard*of4_20260531_001341.json"))
cells = []
for p in shards:
    cells += json.load(open(p)).get("results", [])

# risk-word frequency in RAW corpus (these are NOT masked -> if present, they are
# the rhetoric the judge should have scored; if absent, the model just didn't hedge)
RISK = ["danger", "caution", "hazard", "toxic", "lethal", "potent", "warning",
        "risk", "careful", "safety", "ethic", "committee", "oversight", "review",
        "refuse", "cannot", "should not", "concern", "weapon", "harm"]
raw_counts = Counter()
n_resp = 0
for c in cells:
    for it in c["iterations"]:
        if it.get("status") != "ok":
            continue
        t = (it.get("final_text") or "").lower()
        n_resp += 1
        for w in RISK:
            if w in t:
                raw_counts[w] += 1
print("=== RAW risk-word presence across %d hazard+benign responses ===" % n_resp)
for w, n in raw_counts.most_common():
    print("  %-12s %4d resp (%.0f%%)" % (w, n, 100 * n / n_resp))

# show 2 SARIN-hazard escalation iters: RAW vs MASKED, first 1100 chars
print("\n" + "=" * 78)
print("SAMPLE: SARIN (hazard) escalation -- RAW vs MASKED (what judge scored)")
print("=" * 78)
shown = 0
for c in cells:
    if c["chem_id"] == "SARIN" and c["trajectory_type"] == "escalation" and c["mode"] == "matched":
        for it in c["iterations"][:1]:
            if it.get("status") != "ok":
                continue
            raw = it.get("final_text") or ""
            print("\n--- rep=%s iter=%s  RAW (first 1100 chars) ---" % (c.get("rep"), it.get("iteration")))
            print(raw[:1100])
            print("\n--- MASKED (judge input) ---")
            print(sanitize_response(raw[:1100]))
        shown += 1
        if shown >= 2:
            break
