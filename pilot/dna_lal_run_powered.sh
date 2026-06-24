#!/usr/bin/env bash
# dna_lal_run_powered.sh — powered DNA/genomic LAL run (4 pairs / 8 genes).
# ============================================================================
# Mirror of chem_lal_run_powered.sh for the genomic modality:
#   1. launch N parallel API shards of dna_lal_pilot.py (resume-aware)
#   2. wait for all shards
#   3. drop a completion sentinel (results/dna_lal_powered.done)
# Analysis (rep-pool + matched-pair bootstrap) + judge S1' are run SEPARATELY
# after this completes (dna_lal_judge.py port + dna bootstrap), mirroring chem.
#
# API-bound, NOT GPU. Genes referenced by COMMON NAME / accession only; tool
# outputs are pre-scripted synthetic fiction; expression score is an abstract
# scalar (TAU=0.85); no live Evo2/NT-500M/codon-optimizer is run. Subject =
# Sonnet 4 (the gradable model for DNA; pendulum: 0/1050 refusals on S4).
# Panel: 8 genes x 2 traj x 2 mode x REPS = cell-runs (320 at REPS=10).
# ============================================================================
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PILOT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${PILOT_DIR}/logs}"
PILOT="${PILOT_DIR}/dna_lal_pilot.py"
NSHARDS="${NSHARDS:-4}"
REPS="${REPS:-10}"                # 8 genes x 2 traj x 2 mode x 10 = 320 cell-runs
MODEL="${MODEL:-claude-sonnet-4-20250514}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DONE="${RESULTS_DIR}/dna_lal_powered.done"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
rm -f "$DONE"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.api_keys" 2>/dev/null || source "$HOME/.zshrc" 2>/dev/null
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "EXIT=1 (no ANTHROPIC_API_KEY)" > "$DONE"; exit 1
fi

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

echo "=== DNA LAL powered run ==="
echo "  pilot   : ${PILOT}"
echo "  model   : ${MODEL}"
echo "  shards  : ${NSHARDS}   reps/cell: ${REPS}   (expect $((8*2*2*REPS)) cell-runs)"
echo "  started : $(date '+%Y-%m-%d %H:%M:%S')   RUN_TS=${RUN_TS}"
echo

pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
    log="${LOGS_DIR}/dna_lal_shard${i}of${NSHARDS}_${RUN_TS}.log"
    ckpt="$(ls -t "${RESULTS_DIR}"/dna_lal_subject_shard${i}of${NSHARDS}_*.json 2>/dev/null | head -1)"
    resume_arg=""
    if [ -n "$ckpt" ]; then
        resume_arg="--resume $ckpt"
        echo "  shard ${i}/${NSHARDS} resuming $(basename "$ckpt")"
    fi
    # shellcheck disable=SC2086
    python3 -u "$PILOT" --trajectories escalation,control --mode both \
        --reps "$REPS" --shard "${i}/${NSHARDS}" --model "$MODEL" $resume_arg \
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
        echo "  shard ${idx} FAILED (see log); resume with --shard ${idx}/${NSHARDS} --resume <ckpt>"
        fail=1
    fi
done

echo "EXIT=${fail} TS=${RUN_TS}" > "$DONE"
echo
echo "=== DNA LAL powered DONE (fail=${fail}) at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "  shard files: ${RESULTS_DIR}/dna_lal_subject_shard*of${NSHARDS}_${RUN_TS}.json"
