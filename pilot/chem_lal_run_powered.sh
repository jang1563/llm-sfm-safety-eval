#!/usr/bin/env bash
# chem_lal_run_powered.sh — powered chem LAL run (3 pairs / 6 substances).
# ============================================================================
# Self-contained orchestration of the chem LAL S4' powered run:
#   1. launch N parallel API shards of chem_lal_pilot.py (resume-aware)
#   2. wait for all shards
#   3. run analyze_chem_lal.py over the shard checkpoints (hazard-discrim read)
#   4. drop a completion sentinel (results/chem_lal_powered.done) for the agent
#
# API-bound, NOT GPU. Substances referenced by COMMON NAME only; tool outputs
# are pre-scripted synthetic fiction; potency is an abstract scalar; no live
# REINVENT4 is ever run. Subject = Sonnet 4 (the only scorable model; 4.6
# system-refuses named CW/opioid tools ~100%).
# ============================================================================
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PILOT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${PILOT_DIR}/logs}"
PILOT="${PILOT_DIR}/chem_lal_pilot.py"
NSHARDS="${NSHARDS:-4}"            # concurrent API processes (good-neighbor to in-flight protein run)
REPS="${REPS:-10}"                # powered = 10 reps -> 240 cell-runs (6 chem x 2 traj x 2 mode x 10)
MODEL="${MODEL:-claude-sonnet-4-20250514}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DONE="${RESULTS_DIR}/chem_lal_powered.done"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
rm -f "$DONE"

# ---- API key (already in env if sourced; else source profile) --------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.api_keys" 2>/dev/null || source "$HOME/.zshrc" 2>/dev/null
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "EXIT=1 (no ANTHROPIC_API_KEY)" > "$DONE"; exit 1
fi

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

echo "=== chem LAL powered run ==="
echo "  pilot   : ${PILOT}"
echo "  results : ${RESULTS_DIR}"
echo "  model   : ${MODEL}"
echo "  shards  : ${NSHARDS}   reps/cell: ${REPS}   (expect $((6*2*2*REPS)) cell-runs total)"
echo "  started : $(date '+%Y-%m-%d %H:%M:%S')   RUN_TS=${RUN_TS}"
echo

# ---- Launch shards in parallel (resume-aware) ------------------------------
pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
    log="${LOGS_DIR}/chem_lal_shard${i}of${NSHARDS}_${RUN_TS}.log"
    ckpt="$(ls -t "${RESULTS_DIR}"/chem_lal_subject_shard${i}of${NSHARDS}_*.json 2>/dev/null | head -1)"
    resume_arg=""
    if [ -n "$ckpt" ]; then
        resume_arg="--resume $ckpt"
        echo "  shard ${i}/${NSHARDS} resuming $(basename "$ckpt")"
    fi
    # shellcheck disable=SC2086
    python3 -u "$PILOT" --reps "$REPS" --shard "${i}/${NSHARDS}" --model "$MODEL" $resume_arg \
        > "$log" 2>&1 &
    pids+=("$!")
    echo "  shard ${i}/${NSHARDS} -> PID $! (log $(basename "$log"))"
done

echo "  waiting for ${NSHARDS} shards..."
fail=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "  shard ${idx} OK"
    else
        echo "  shard ${idx} FAILED (see log); resume: python3 $PILOT --reps $REPS --shard ${idx}/${NSHARDS} --resume <ckpt>"
        fail=1
    fi
done
echo

# ---- Analyze (hazard-discrimination read; judge-independent) ---------------
echo "=== Analyzing shard checkpoints ==="
ANALYSIS="${RESULTS_DIR}/CHEM_LAL_POWERED_ANALYSIS_${RUN_TS}.txt"
# Only this run's shard files (avoid mixing in the n=1 smoke / old of12 probe).
python3 "${PILOT_DIR}/analyze_chem_lal.py" \
    "${RESULTS_DIR}"/chem_lal_subject_shard*of${NSHARDS}_*.json 2>&1 | tee "$ANALYSIS"

echo "EXIT=${fail} TS=${RUN_TS} ANALYSIS=$(basename "$ANALYSIS")" > "$DONE"
echo
echo "=== chem LAL powered DONE (fail=${fail}) at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  analysis -> $(basename "$ANALYSIS")"
