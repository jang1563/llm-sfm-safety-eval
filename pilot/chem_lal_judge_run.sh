#!/usr/bin/env bash
# chem_lal_judge_run.sh -- detached judge S1' pass over the powered 240 cells.
# ============================================================================
# JUDGE-ONLY pass (no subject re-run): chem_lal_judge.py blind-scores every
# preserved iteration final_text on 5 dims, then v0_2_6_bootstrap_s1prime.py
# (the UNCHANGED protein tool -- modality-agnostic field reads) computes the
# powered S1'-action / S1'-rhetoric with bootstrap CIs.
#
# Resume-safe: the judge checkpoints per-cell to results/chem_lal_judged_ckpt.jsonl;
# re-running this script continues where it left off. Judge = Sonnet 4 (the
# scorable classifier; 4.6 system-refuses named CW/opioid content ~100% and must
# NOT be used as judge here). Single-thread, ~1200 judge calls, ~30-40 min.
#
# Launch detached (survives turn boundaries):
#   nohup caffeinate -i bash chem_lal_judge_run.sh > logs/chem_lal_judge_orch.log 2>&1 &
# ============================================================================
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PILOT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${PILOT_DIR}/logs}"
JUDGE="${PILOT_DIR}/chem_lal_judge.py"
BOOT="${PILOT_DIR}/v0_2_6_bootstrap_s1prime.py"
SHARD_GLOB="${RESULTS_DIR}/chem_lal_subject_shard*of4_20260531_001341.json"
CKPT="${RESULTS_DIR}/chem_lal_judged_ckpt.jsonl"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DONE="${RESULTS_DIR}/chem_lal_judge.done"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
rm -f "$DONE"

# ---- API key ---------------------------------------------------------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.api_keys" 2>/dev/null || source "$HOME/.zshrc" 2>/dev/null
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "EXIT=1 (no ANTHROPIC_API_KEY)" > "$DONE"; exit 1
fi

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

echo "=== chem LAL judge S1' run ==="
echo "  judge   : ${JUDGE}"
echo "  shards  : ${SHARD_GLOB}"
echo "  ckpt    : ${CKPT}  (resume-safe)"
echo "  started : $(date '+%Y-%m-%d %H:%M:%S')   RUN_TS=${RUN_TS}"
echo

# ---- 1. JUDGE (blind-score 5 dims; checkpoint+resume) ----------------------
JUDGE_LOG="${LOGS_DIR}/chem_lal_judge_${RUN_TS}.log"
# shellcheck disable=SC2086
python3 -u "$JUDGE" $SHARD_GLOB --ckpt "$CKPT" 2>&1 | tee "$JUDGE_LOG"
jrc=${PIPESTATUS[0]}

# locate the judged JSON the judge just wrote (newest)
JUDGED="$(ls -t "${RESULTS_DIR}"/chem_lal_judged_*.json 2>/dev/null | grep -v ckpt | head -1)"
echo
echo "  judge rc=${jrc}   judged=$(basename "${JUDGED:-NONE}")"

# ---- 2. BOOTSTRAP (powered S1' + CIs; UNCHANGED protein tool) --------------
BOOT_OUT="${RESULTS_DIR}/CHEM_LAL_S1PRIME_BOOTSTRAP_${RUN_TS}.txt"
brc=1
if [ -n "${JUDGED:-}" ] && [ -f "${JUDGED}" ]; then
    echo "=== S1' bootstrap ==="
    python3 "$BOOT" "$JUDGED" 2>&1 | tee "$BOOT_OUT"
    brc=${PIPESTATUS[0]}
else
    echo "  SKIP bootstrap (no judged JSON found)"
fi

echo "EXIT=$(( jrc || brc )) TS=${RUN_TS} JUDGED=$(basename "${JUDGED:-NONE}") BOOT=$(basename "$BOOT_OUT")" > "$DONE"
echo
echo "=== chem LAL judge DONE (jrc=${jrc} brc=${brc}) at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  judged    -> $(basename "${JUDGED:-NONE}")"
echo "  bootstrap -> $(basename "$BOOT_OUT")"
