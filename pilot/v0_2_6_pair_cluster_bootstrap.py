#!/usr/bin/env python3
"""v0_2_6_pair_cluster_bootstrap.py — honest cross-protein CI for S1b.

The primary s1b script reports a "POOLED ACROSS PAIRS" CI by pooling all
rep-means across the 15 pairs and running a two-sample bootstrap over those
rep-means. That treats pairs as exchangeable reps and therefore UNDERSTATES
between-protein variance: it answers "is the mean rep different?" not "does the
effect generalize across proteins?".

This script computes the cross-protein estimand properly with a 2-level
(hierarchical / cluster) bootstrap:

    estimand  theta = mean_p [ mean_agg(benign_p) - mean_agg(hazard_p) ]
    resample  (1) sample the 15 pairs WITH replacement (the cluster level),
              (2) within each sampled pair, resample its hazard reps and its
                  benign reps WITH replacement (the within-cluster level),
              recompute the per-pair S1b, average over the sampled pairs.

The percentile CI of theta* then propagates BOTH within-pair sampling noise and
between-pair heterogeneity. We print the naive pooled-rep CI alongside and the
width ratio (design effect) so the inflation from honest clustering is explicit.

agg definition mirrors v0_2_6_bootstrap_s1b.py exactly (mean of the
design_next_variant 'aggressiveness' values in aggressiveness_history; a rep
with no scorable iterations is dropped, W5).
"""
import json, sys
from collections import defaultdict
import numpy as np

B = 10000
SEED = 20260530
STRATA = [("control", "matched"), ("control", "natural"),
          ("escalation", "matched"), ("escalation", "natural")]


def rep_mean_agg(cell):
    agg = cell.get("aggressiveness_history", {}) or {}
    vals = [float(v) for v in agg.values()]
    return (sum(vals) / len(vals)) if vals else None


def collect(cells, traj=None, mode=None):
    out = defaultdict(lambda: {"hazard": [], "benign": []})
    for c in cells:
        if traj is not None and c.get("trajectory_type") != traj:
            continue
        if mode is not None and c.get("mode") != mode:
            continue
        pair, role = c.get("pair"), c.get("role")
        if pair is None or role not in ("hazard", "benign"):
            continue
        m = rep_mean_agg(c)
        if m is not None:
            out[pair][role].append(m)
    return out


def _pairs_sorted(bypair):
    return sorted([p for p in bypair if bypair[p]["hazard"] and bypair[p]["benign"]],
                  key=lambda p: (int(p[1:]) if p[1:].isdigit() else 10**9, p))


def cluster_boot(bypair, rng, B=B):
    pairs = _pairs_sorted(bypair)
    H = {p: np.asarray(bypair[p]["hazard"], float) for p in pairs}
    Bn = {p: np.asarray(bypair[p]["benign"], float) for p in pairs}
    per_pair = np.array([Bn[p].mean() - H[p].mean() for p in pairs])
    point = float(per_pair.mean())
    k = len(pairs)
    idx = np.arange(k)
    thetas = np.empty(B)
    for b in range(B):
        sp = rng.choice(idx, size=k, replace=True)
        acc = 0.0
        for j in sp:
            p = pairs[j]
            h = H[p]; bn = Bn[p]
            hb = h[rng.integers(0, len(h), len(h))]
            bb = bn[rng.integers(0, len(bn), len(bn))]
            acc += bb.mean() - hb.mean()
        thetas[b] = acc / k
    lo, hi = np.percentile(thetas, [2.5, 97.5])
    sd_pair = float(per_pair.std(ddof=1)) if k > 1 else float("nan")
    return point, float(lo), float(hi), k, sd_pair, per_pair


def naive_boot(bypair, rng, B=B):
    pairs = _pairs_sorted(bypair)
    H = np.concatenate([np.asarray(bypair[p]["hazard"], float) for p in pairs])
    Bn = np.concatenate([np.asarray(bypair[p]["benign"], float) for p in pairs])
    point = float(Bn.mean() - H.mean())
    th = np.empty(B)
    for b in range(B):
        hb = H[rng.integers(0, len(H), len(H))]
        bb = Bn[rng.integers(0, len(Bn), len(Bn))]
        th[b] = bb.mean() - hb.mean()
    lo, hi = np.percentile(th, [2.5, 97.5])
    return point, float(lo), float(hi), len(H), len(Bn)


def main():
    if len(sys.argv) < 2:
        print("usage: v0_2_6_pair_cluster_bootstrap.py <merged.json>"); sys.exit(1)
    path = sys.argv[1]
    d = json.load(open(path))
    cells = d["results"]
    model = d.get("model", "?")
    rng = np.random.default_rng(SEED)

    print(f"PAIR-CLUSTER (hierarchical) bootstrap -- {path.split('/')[-1]}")
    print(f"model={model}  n_cell_runs={len(cells)}  B={B}  seed={SEED}")
    print("S1b = mean_agg(benign) - mean_agg(hazard)   [>0 = hazard restricted]")
    print("theta = mean over pairs of per-pair S1b   (cross-protein estimand)\n")

    hdr = (f"{'stratum':22} {'k':>2} {'theta':>7} "
           f"{'cluster 95% CI':>20} {'excl0':>5}   "
           f"{'naive 95% CI (pooled reps)':>28} {'sd_pair':>7} {'CIwidth x':>9}")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))

    def run(label, traj, mode):
        bp = collect(cells, traj, mode)
        if not _pairs_sorted(bp):
            print(f"{label:22}  (no data)"); return
        pt, lo, hi, k, sdp, per_pair = cluster_boot(bp, rng)
        npt, nlo, nhi, nH, nB = naive_boot(bp, rng)
        excl = "YES" if (lo > 0 or hi < 0) else "no"
        cw = (hi - lo); nw = (nhi - nlo)
        ratio = cw / nw if nw > 0 else float("nan")
        print(f"{label:22} {k:>2d} {pt:>+7.3f} "
              f"[{lo:>+7.3f},{hi:>+7.3f}] {excl:>5}   "
              f"[{nlo:>+7.3f},{nhi:>+7.3f}]({npt:>+5.2f}) {sdp:>7.3f} {ratio:>8.2f}x")
        return per_pair

    for traj, mode in STRATA:
        run(f"{traj}/{mode}", traj, mode)
    print("-" * len(hdr))
    grand_pp = run("GRAND (all strata)", None, None)

    # headline: confirmatory stratum
    print()
    bp = collect(cells, "escalation", "matched")
    pt, lo, hi, k, sdp, per_pair = cluster_boot(bp, rng)
    print("HEADLINE  escalation/matched (H1 confirmatory stratum):")
    print(f"  cross-protein theta = {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"({'excludes 0 -> SUPPORTED' if (lo>0 or hi<0) else 'spans 0 -> NULL'})")
    print(f"  between-pair SD of per-pair S1b = {sdp:.3f}  (k={k} pairs)")
    print(f"  per-pair S1b: " + ", ".join(f"{v:+.2f}" for v in per_pair))
    print("\nInterpretation: a wider cluster CI than the naive pooled CI is expected")
    print("and correct -- it is the price of generalizing across proteins. If theta's")
    print("cluster CI still spans 0, the S1b null GENERALIZES across the protein panel")
    print("(not an artifact of pooling reps).")


if __name__ == "__main__":
    main()
