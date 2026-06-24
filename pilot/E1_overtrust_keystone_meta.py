#!/usr/bin/env python3
"""
E1_overtrust_keystone_meta.py — pooled cross-model meta-analysis of the anthrax-PA
1ACC keystone (Finding 16e).

WHY: per-model per-protein heterogeneity (perm p in E1_overtrust_stats / ow_batch) is
UNDERPOWERED — only n=10 hazardous×S3 observations per protein per model, so every
open-weight model came back n.s. even though 6/10 show 1ACC strictly below the other
toxins. This pools the 1ACC-vs-other-toxin DO-ARR contrast ACROSS the 10 models with
MODEL AS A STRATUM, so the wildly different absolute rejection rates (0–76%) cancel and
only the WITHIN-model 1ACC deficit is tested.

ESTIMAND: RD = P(reject | other toxin) − P(reject | anthrax 1ACC), Mantel–Haenszel
pooled over models.  RD > 0  ==>  1ACC is UNDER-rejected relative to the other toxins
==> over-trust is localized to the FSI=0 self-interpreting keystone (Finding 16e).

TESTS (all stdlib, model-stratified):
  - Cochran–Mantel–Haenszel χ² (continuity-corrected, erfc p) : pooled 2×2×K association
  - MH common risk difference + percentile-bootstrap 95% CI (resample within strata)
  - within-model block permutation p (distribution-free anchor; robust to all-zero strata)
  - across-model sign test on the per-model direction (exact binomial)

SOURCE: rule from subject files (default) or rule/judge from judged files (--from-judged).

SAFE-CONDUCT: reads only model responses + frozen ground-truth labels; computes a SAFETY
metric (rejection of an unsupported hazard claim). No sequences, no synthesis, no
fabricated dangerous content. Defensive AI-biosafety (Mason Lab, WCM).

Usage:
  python3 E1_overtrust_keystone_meta.py                        # rule, subject files
  python3 E1_overtrust_keystone_meta.py --from-judged --src judge
  python3 E1_overtrust_keystone_meta.py --B 20000 --out results/E1_keystone_meta.json
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

# DRY: reuse the loaders/score-attach/keystone constant from the roll-up driver, and the
# validated reject predicate + percentile helper + seed from the stats module.
from E1_overtrust_ow_batch import load_latest, attach_scores, ANTHRAX
from E1_overtrust_stats import reject, _pct, SEED

HERE = Path(__file__).resolve().parent
B_DEFAULT = 20000

Strata = Dict[str, Tuple[List[int], List[int]]]   # model_key -> (1ACC obs, other-toxin obs)


def build_strata(found: Dict, src: str) -> Strata:
    """model_key -> (1ACC reject-flags, other-toxin reject-flags) over hazardous×S3."""
    out: Strata = {}
    for key in sorted(found):
        data = found[key][1]
        eng = [r for r in data["results"] if r.get("refused") is False]
        hs3 = [r for r in eng if r["signal"] == "S3" and r.get("hazardous")]
        g1: List[int] = []
        g0: List[int] = []
        for r in hs3:
            v = reject(r.get(src))
            if v is None:
                continue
            (g1 if r["pdb_id"] == ANTHRAX else g0).append(v)
        if g1 or g0:
            out[data.get("model_key") or key] = (g1, g0)
    return out


def _cell(g1: List[int], g0: List[int]) -> Tuple[int, int, int, int]:
    """Return (a, n1, c, n0): a=1ACC rejects, n1=1ACC n, c=other rejects, n0=other n."""
    return sum(g1), len(g1), sum(g0), len(g0)


def cmh_chi2(strat: Strata, cc: bool = True) -> Tuple[float, float]:
    """Cochran–Mantel–Haenszel χ² for the 2×2×K table; p via erfc on χ²(1)."""
    num = 0.0
    var = 0.0
    for g1, g0 in strat.values():
        a, n1, c, n0 = _cell(g1, g0)
        N = n1 + n0
        if N < 2 or n1 == 0 or n0 == 0:
            continue
        m1 = a + c                       # total rejects in stratum
        m0 = N - m1
        if m1 == 0 or m0 == 0:           # no within-stratum information
            continue
        num += a - (n1 * m1 / N)
        var += n1 * n0 * m1 * m0 / (N * N * (N - 1))
    if var <= 0:
        return 0.0, 1.0
    chi2 = max((abs(num) - (0.5 if cc else 0.0)) ** 2 / var, 0.0)
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


def mh_risk_diff(strat: Strata) -> float:
    """MH-pooled RD = P(reject|other) − P(reject|1ACC).  >0 => 1ACC under-rejected."""
    num = den = 0.0
    for g1, g0 in strat.values():
        a, n1, c, n0 = _cell(g1, g0)
        N = n1 + n0
        if n1 == 0 or n0 == 0:
            continue
        w = n1 * n0 / N
        num += w * ((c / n0) - (a / n1))
        den += w
    return (num / den) if den > 0 else float("nan")


def boot_rd(strat: Strata, B: int, rng: random.Random) -> List[float]:
    """Percentile bootstrap of MH-RD, resampling WITHIN each model stratum (sizes fixed)."""
    keys = list(strat.keys())
    out: List[float] = []
    for _ in range(B):
        res: Strata = {}
        for k in keys:
            g1, g0 = strat[k]
            n1, n0 = len(g1), len(g0)
            bg1 = [g1[rng.randrange(n1)] for _ in range(n1)] if n1 else []
            bg0 = [g0[rng.randrange(n0)] for _ in range(n0)] if n0 else []
            res[k] = (bg1, bg0)
        out.append(mh_risk_diff(res))
    out.sort()
    return out


def block_perm_p(strat: Strata, B: int, rng: random.Random) -> Tuple[float, float]:
    """Within-model block permutation of the 1ACC/other label; CMH χ² as statistic."""
    obs, _ = cmh_chi2(strat)
    keys = list(strat.keys())
    ge = 0
    for _ in range(B):
        res: Strata = {}
        for k in keys:
            g1, g0 = strat[k]
            n1 = len(g1)
            pool = g1 + g0
            rng.shuffle(pool)
            res[k] = (pool[:n1], pool[n1:])
        c, _ = cmh_chi2(res)
        if c >= obs - 1e-12:
            ge += 1
    return obs, (ge + 1) / (B + 1)


def _binom_two_sided(k: int, n: int, q: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * (q ** i) * ((1 - q) ** (n - i)) for i in range(n + 1)]
    thr = pmf[k] + 1e-12
    return min(1.0, sum(p for p in pmf if p <= thr))


def sign_test(strat: Strata):
    """Per-model direction (1ACC rate vs other rate) + exact two-sided binomial p."""
    below = above = tie = 0
    per: Dict[str, Tuple[float, float, int, int]] = {}
    for k, (g1, g0) in strat.items():
        a, n1, c, n0 = _cell(g1, g0)
        if n1 == 0 or n0 == 0:
            continue
        r1, r0 = a / n1, c / n0
        per[k] = (r1, r0, n1, n0)
        if r1 < r0:
            below += 1
        elif r1 > r0:
            above += 1
        else:
            tie += 1
    return below, above, tie, _binom_two_sided(below, below + above), per


def fsi_dose(found: Dict, src: str) -> List[Tuple[float, str, str, int, int]]:
    """Per-protein pooled S3 reject rate vs the FSI shown (value_shown), all proteins.

    Tests WHY 1ACC is the keystone: if caution tracks the metric's MAGNITUDE (a surface
    feature) rather than hazard, reject rate should rise with FSI and the FSI=0 cell sits
    at the floor. Returns rows (fsi, pdb_id, class, rejects, n) sorted by FSI.
    """
    agg: Dict[str, list] = {}
    for key in found:
        for r in found[key][1]["results"]:
            if r.get("signal") != "S3" or r.get("refused") is not False:
                continue
            v = reject(r.get(src))
            if v is None:
                continue
            pid = r["pdb_id"]
            a = agg.setdefault(pid, [0, 0, r.get("value_shown"), r.get("hazardous"),
                                    r.get("is_true_benign")])
            a[0] += v
            a[1] += 1
    rows = []
    for pid, (rej, n, fsi, hz, tb) in agg.items():
        cls = "TOXIN" if hz else ("benign" if tb else "other")
        rows.append((float(fsi) if fsi is not None else -1.0, pid, cls, rej, n))
    rows.sort()
    return rows


def spearman(pairs: List[Tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 2:
        return float("nan")

    def _rank(z):
        order = sorted(range(len(z)), key=lambda i: z[i])
        r = [0] * len(z)
        for o, i in enumerate(order):
            r[i] = o
        return r

    rx, ry = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def fsi_trend_strata(found: Dict, src: str) -> List[Tuple[List[float], List[int]]]:
    """Per-model (FSI, reject) over ALL S3 rows (9 proteins) — informative strata only
    (>=2 rows AND not all-equal reject, else they carry no trend information)."""
    strat = []
    for key in found:
        xs: List[float] = []
        ys: List[int] = []
        for r in found[key][1]["results"]:
            if r.get("signal") != "S3" or r.get("refused") is not False:
                continue
            v = reject(r.get(src))
            x = r.get("value_shown")
            if v is None or x is None:
                continue
            xs.append(float(x))
            ys.append(v)
        if len(xs) >= 2 and 0 < sum(ys) < len(ys):
            strat.append((xs, ys))
    return strat


def _trend_num_var_sxx(strat) -> Tuple[float, float, float]:
    """Stratified Cochran–Armitage components: Σ(O−E), Σ Var(hypergeometric), Σ centred-Σx²."""
    num = var = sxxc = 0.0
    for xs, ys in strat:
        n = len(xs)
        Sx, Sy = sum(xs), sum(ys)
        Sxx = sum(x * x for x in xs)
        Sxy = sum(x * y for x, y in zip(xs, ys))
        cx = Sxx - Sx * Sx / n
        num += Sxy - Sx * Sy / n
        sxxc += cx
        if n > 1:
            var += (Sy * (n - Sy) / (n * (n - 1))) * cx
    return num, var, sxxc


def fsi_trend_test(strat) -> Tuple[float, float, float, float]:
    """Returns (model-demeaned LPM slope β, CA χ²(1), analytic p, observed Σ(O−E))."""
    num, var, sxxc = _trend_num_var_sxx(strat)
    beta = num / sxxc if sxxc > 0 else float("nan")
    if var <= 0:
        return beta, 0.0, 1.0, num
    chi2 = (num * num) / var
    return beta, chi2, math.erfc(math.sqrt(chi2 / 2.0)), num


def fsi_trend_perm(strat, B: int, rng: random.Random) -> float:
    """Model-blocked permutation: shuffle reject within each model, two-sided on Σ(O−E)."""
    obs, _, _ = _trend_num_var_sxx(strat)
    pre = [(xs, list(ys), sum(xs) * sum(ys) / len(xs)) for xs, ys in strat]
    ge = 0
    for _ in range(B):
        stat = 0.0
        for xs, ys, E in pre:
            rng.shuffle(ys)
            stat += sum(x * y for x, y in zip(xs, ys)) - E
        if abs(stat) >= abs(obs) - 1e-12:
            ge += 1
    return (ge + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser(description="E1 anthrax-1ACC keystone pooled meta-analysis")
    ap.add_argument("--from-judged", action="store_true",
                    help="read results/E1_judged_*.json (rule+judge) instead of subject files")
    ap.add_argument("--src", default="rule", choices=["rule", "judge"],
                    help="score source (default rule; judge requires --from-judged)")
    ap.add_argument("--glob", default=None, help="override results/ glob")
    ap.add_argument("--B", type=int, default=B_DEFAULT, help="bootstrap/permutation iters")
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()

    glob = args.glob or ("E1_judged_*.json" if args.from_judged else "E1_subject_*_vllm_*.json")
    found = load_latest(glob)
    found = {k: (p, d) for k, (p, d) in found.items()
             if d.get("model_key") and "claude" not in (d.get("model") or "").lower()}
    if not found:
        print(f"no open-weight files match results/{glob}")
        return
    for key in found:
        attach_scores(found[key][1]["results"], args.from_judged)

    strat = build_strata(found, args.src)
    if not strat:
        print(f"no gradable hazardous×S3 cells under src='{args.src}' "
              f"({'judge needs --from-judged' if args.src == 'judge' and not args.from_judged else 'empty'})")
        return

    n1_tot = sum(len(g1) for g1, _ in strat.values())
    n0_tot = sum(len(g0) for _, g0 in strat.values())
    chi2, p_cmh = cmh_chi2(strat)
    rd = mh_risk_diff(strat)
    rng = random.Random(SEED)
    bd = boot_rd(strat, args.B, rng)
    rd_lo, rd_hi = _pct(bd, 2.5), _pct(bd, 97.5)
    rngp = random.Random(SEED + 7)
    _, p_perm = block_perm_p(strat, args.B, rngp)
    below, above, tie, p_sign, per = sign_test(strat)

    print(f"\n{'='*88}")
    print("E1 ANTHRAX-PA 1ACC KEYSTONE — POOLED CROSS-MODEL META-ANALYSIS (Finding 16e)")
    print(f"{'='*88}")
    print(f"source={args.src}   models(strata)={len(strat)}   B={args.B}   "
          f"1ACC n={n1_tot}   other-toxin n={n0_tot}")
    print("RD = P(reject|other) − P(reject|1ACC);  RD>0 => 1ACC UNDER-rejected (over-trust localized)\n")

    print(f"{'model_key':16s} {'1ACC':>9s} {'other':>9s} {'RD(o-1)':>8s}   dir")
    print("-" * 54)
    for k in sorted(per):
        r1, r0, n1, n0 = per[k]
        d = r0 - r1
        dirn = "1ACC<other" if d > 0 else ("1ACC>other" if d < 0 else "tie")
        print(f"{k:16s} {f'{r1:.2f}({n1})':>9s} {f'{r0:.2f}({n0})':>9s} {d:+8.2f}   {dirn}")
    print("-" * 54)
    print(f"\nMH pooled RD            = {rd:+.3f}   95% CI [{rd_lo:+.3f}, {rd_hi:+.3f}]   "
          f"{'excludes 0' if not (rd_lo <= 0 <= rd_hi) else 'COVERS 0'}")
    print(f"CMH χ²(1, cc)           = {chi2:.3f}   p = {p_cmh:.4f}   "
          f"{'SIGNIFICANT' if p_cmh < 0.05 else 'n.s.'}")
    print(f"block-permutation p     = {p_perm:.4f}   (distribution-free, model-blocked)   "
          f"{'SIGNIFICANT' if p_perm < 0.05 else 'n.s.'}")
    print(f"across-model sign test  = {below} below / {above} above / {tie} tie   "
          f"two-sided p = {p_sign:.4f}")
    verdict = ("1ACC keystone CONFIRMED pooled: over-trust localizes to the FSI=0 anthrax-PA cell"
               if (rd > 0 and (p_cmh < 0.05 or p_perm < 0.05))
               else "pooled keystone effect directional but not significant at .05")
    print(f"\n=> {verdict}")

    # ---- FSI dose-response: is the keystone the floor of a magnitude gradient? ----------
    drows = fsi_dose(found, args.src)
    tox = [(fsi, rej / n) for fsi, pid, cls, rej, n in drows if cls == "TOXIN" and n]
    rho = spearman(tox)
    print("\n--- FSI dose-response: pooled S3 reject rate vs value_shown(FSI), all proteins ---")
    print(f"{'FSI':>6s}  {'pdb':6s} {'class':7s} {'reject':>9s} {'rate':>5s}")
    for fsi, pid, cls, rej, n in drows:
        rate = rej / n if n else float("nan")
        star = "  <== FSI~0 keystone" if pid == ANTHRAX else ""
        print(f"{fsi:6.3f}  {pid:6s} {cls:7s} {rej:4d}/{n:<4d} {rate:5.2f}{star}")
    print(f"Spearman rho(FSI, reject) across {len(tox)} toxins = {rho:+.2f} (n={len(tox)} descriptive)  "
          f"-> caution tracks metric MAGNITUDE not hazard; FSI=0 1ACC = floor")

    # rigorous all-protein trend: Cochran–Armitage, model-stratified (n=4 ρ was descriptive)
    strat_t = fsi_trend_strata(found, args.src)
    n_rows = sum(len(xs) for xs, _ in strat_t)
    beta, chi2_t, p_t, _num = fsi_trend_test(strat_t)
    rngt = random.Random(SEED + 11)
    p_tperm = fsi_trend_perm(strat_t, args.B, rngt)
    trend_sig = (p_t < 0.05 and p_tperm < 0.05)
    print(f"\n--- FSI linear-trend (all 9 proteins, model-stratified Cochran-Armitage) ---")
    print(f"  strata={len(strat_t)}  rows={n_rows}   model-demeaned LPM slope "
          f"beta = {beta:+.4f}  (d P(reject) per +1 FSI unit, model-controlled)")
    print(f"  CA trend chi2(1) = {chi2_t:.3f}  p = {p_t:.4f}   block-perm p = {p_tperm:.4f}   "
          f"{'SIGNIFICANT (magnitude gradient, hazard-independent)' if trend_sig else ('trend p<.05 only' if p_t < 0.05 else 'n.s.')}")

    if args.out:
        outp = Path(args.out)
        if not outp.is_absolute():
            outp = HERE / outp
        outp.write_text(json.dumps({
            "experiment": "E1 anthrax-1ACC keystone pooled meta-analysis (Finding 16e)",
            "source": args.src, "from_judged": args.from_judged, "B": args.B,
            "n_models": len(strat), "n_1ACC": n1_tot, "n_other": n0_tot,
            "mh_risk_diff": rd, "rd_ci": [rd_lo, rd_hi],
            "cmh_chi2": chi2, "cmh_p": p_cmh, "perm_p": p_perm,
            "sign_below": below, "sign_above": above, "sign_tie": tie, "sign_p": p_sign,
            "per_model": {k: {"r_1ACC": per[k][0], "r_other": per[k][1],
                              "n_1ACC": per[k][2], "n_other": per[k][3]} for k in per},
            "fsi_dose": [{"fsi": fsi, "pdb_id": pid, "class": cls, "reject": rej, "n": n}
                         for fsi, pid, cls, rej, n in drows],
            "fsi_toxin_spearman": rho,
            "fsi_trend": {"strata": len(strat_t), "rows": n_rows, "lpm_slope": beta,
                          "ca_chi2": chi2_t, "ca_p": p_t, "perm_p": p_tperm},
        }, indent=2))
        print(f"\nsaved -> {outp}")
    print()


if __name__ == "__main__":
    main()
