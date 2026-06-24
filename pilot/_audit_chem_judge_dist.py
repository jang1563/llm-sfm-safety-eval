#!/usr/bin/env python3
"""Floor-artifact vs true-null audit of the chem judge scores.
If every dim is pinned to one value (e.g. H==0, E==5 everywhere) the S1' null
is a measurement floor, not discrimination failure. A TRUE null shows score
VARIANCE across cells while hazard-mean ~ benign-mean."""
import json, sys
from collections import Counter, defaultdict
from statistics import mean, pstdev

path = sys.argv[1] if len(sys.argv) > 1 else "results/chem_lal_judged_20260531_094114.json"
d = json.load(open(path))
cells = d["results"]
DIMS = ["endorsement", "hazard_recognition", "caution_specificity",
        "oversight_solicitation", "action_modification"]

dist = {k: Counter() for k in DIMS}
by_role = defaultdict(lambda: defaultdict(list))
by_chem = defaultdict(lambda: defaultdict(list))
n_iter = 0
for c in cells:
    role = c["role"]; chem = c["chem_id"]
    for it in c["iterations"]:
        sc = it.get("scores", {})
        if "_error" in sc or "endorsement" not in sc:
            continue
        n_iter += 1
        for k in DIMS:
            dist[k][sc[k]] += 1
            by_role[role][k].append(sc[k])
            by_chem[chem][k].append(sc[k])

print("scored iters: %d  (cells=%d)\n" % (n_iter, len(cells)))
print("=== per-dimension VALUE distribution (all iters) ===")
for k in DIMS:
    items = sorted(dist[k].items())
    tot = sum(v for _, v in items)
    s = "  ".join("%s:%d(%.0f%%)" % (val, cnt, 100 * cnt / tot) for val, cnt in items)
    print("  %-22s %s" % (k, s))

print("\n=== per-ROLE per-dim MEAN +/- SD (hazard vs benign, all strata pooled) ===")
print("  %-22s %10s %10s %9s" % ("dim", "hazard", "benign", "haz-ben"))
for k in DIMS:
    H = by_role["hazard"][k]; B = by_role["benign"][k]
    h, b = mean(H), mean(B)
    print("  %-22s %5.2f+-%4.2f %5.2f+-%4.2f %+9.3f" %
          (k, h, pstdev(H), b, pstdev(B), h - b))

print("\n=== per-CHEM mean (endorsement / hazard_recognition / caution_spec) ===")
print("  %-7s %6s %6s %6s" % ("chem", "E", "H", "C"))
for chem in sorted(by_chem):
    e = mean(by_chem[chem]["endorsement"])
    h = mean(by_chem[chem]["hazard_recognition"])
    cc = mean(by_chem[chem]["caution_specificity"])
    print("  %-7s %6.2f %6.2f %6.2f" % (chem, e, h, cc))

# floor verdict
floors = []
for k in DIMS:
    vals = list(dist[k].elements())
    if len(set(vals)) == 1:
        floors.append("%s pinned@%s" % (k, vals[0]))
print("\nFLOOR CHECK:", ("; ".join(floors) if floors else
      "no dim fully pinned -- scores vary, null is genuine discrimination-failure"))
