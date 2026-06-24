#!/usr/bin/env bash
# v0_2_6_lal_run_powered.sh — Track 2 powered LAL run (15 pairs / 30 proteins)
# ============================================================================
# End-to-end orchestration of the v0.2.6 LAL S1'/S4' powered run on Cayuga:
#   1. launch N parallel API shards of v0_2_6_lal_pilot.py
#   2. merge shard checkpoints                 (v0_2_6_merge_shards.py)
#   3. judge the merged subject file           (v0_2_6_lal_judge.py)
#   4. run the four analyses (S1b primary, escalation, S4' CCS, S1' secondary)
#
# Run inside tmux/screen on a CPU node with outbound HTTPS to api.anthropic.com
# (Phobos/login). It is API-bound, NOT GPU — no SLURM allocation needed.
#
#   tmux new -s lal
#   export ANTHROPIC_API_KEY=...        # required
#   ./v0_2_6_lal_run_powered.sh --plan  # 0-cost cell plan (expect 1200)
#   ./v0_2_6_lal_run_powered.sh         # full run (several hours)
#
# Scope: all proteins are referenced by published PDB id / name only. Tool
# outputs are pre-scripted synthetic fiction; the hazard predicate H is an
# abstract threshold on a synthetic FSI scalar. No live SFM is ever run.
# ============================================================================
set -uo pipefail

# ---- Config (override via env) ---------------------------------------------
SCRATCH="${SCRATCH:-/athena/masonlab/scratch/users/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${SCRATCH}/Science_FM_Safety_LAL}"
PILOT_DIR="${PILOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-${PILOT_DIR}/results}"
CONDA_INIT="${CONDA_INIT:-${HOME}/miniconda3/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-narrow_model_safety}"
NSHARDS="${NSHARDS:-8}"          # parallel API processes (rate-limit dependent)
REPS="${REPS:-10}"              # reps/cell (powered = 10 -> 1200 cell-runs)
MODEL="${MODEL:-claude-sonnet-4-20250514}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-20250514}"

DO_JUDGE=1
DO_PLAN=0
DO_RESUME=0
while [ $# -gt 0 ]; do
    case "$1" in
        --plan) DO_PLAN=1 ;;
        --resume) DO_RESUME=1 ;;
        --no-judge) DO_JUDGE=0 ;;
        --shards) NSHARDS="$2"; shift ;;
        --reps) REPS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --judge-model) JUDGE_MODEL="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

RUN_TS="$(date +%Y%m%d_%H%M%S)"
PILOT="${PILOT_DIR}/v0_2_6_lal_pilot.py"
mkdir -p "$RESULTS_DIR"

echo "=== v0.2.6 LAL powered run (Track 2) ==="
echo "  pilot dir : ${PILOT_DIR}"
echo "  results   : ${RESULTS_DIR}"
echo "  model     : ${MODEL}   judge: ${JUDGE_MODEL}"
echo "  shards    : ${NSHARDS}   reps/cell: ${REPS}"
echo

# ---- Environment -----------------------------------------------------------
if [ -f "${CONDA_INIT}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${ENV_NAME}" 2>/dev/null || {
        echo "  (env ${ENV_NAME} not found; creating minimal env)"
        conda create -n "${ENV_NAME}" python=3.11 -y && conda activate "${ENV_NAME}"
    }
else
    echo "WARN: conda init not at ${CONDA_INIT}; using current python"
fi
python3 -c 'import anthropic' 2>/dev/null || pip install --quiet anthropic
echo "  python    : $(python3 --version 2>&1)   anthropic: $(python3 -c 'import anthropic; print(anthropic.__version__)' 2>&1)"
echo

# ---- Plan-only (zero API cost) ---------------------------------------------
if [ "$DO_PLAN" = "1" ]; then
    echo "=== PLAN (no API calls) ==="
    python3 "$PILOT" --plan --reps "$REPS" | head -8
    echo "  ..."
    total=$(python3 "$PILOT" --plan --reps "$REPS" | grep -c '^\s*[0-9]')
    echo "  total cell-runs planned: ${total}  (expect 30 proteins x 2 traj x 2 mode x ${REPS} reps = $((30*2*2*REPS)))"
    echo "  CCS sanity (synthetic trajectories, 0-cost):"
    python3 "$PILOT" --ccs-only | head -12
    exit 0
fi

# ---- Preflight: API key + network ------------------------------------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."; exit 1
fi
if ! curl -s --max-time 10 https://api.anthropic.com/ > /dev/null 2>&1; then
    echo "ERROR: api.anthropic.com not reachable from this node."
    echo "  Use the login node, or check \$https_proxy / firewall."
    exit 1
fi
echo "  API key set; api.anthropic.com reachable."
echo

# ---- Clean prior shard checkpoints (avoid merging stale runs) --------------
# --resume KEEPS existing shard checkpoints (each shard continues its own via
# pilot --resume); only a fresh run archives them to avoid merging stale data.
if [ "$DO_RESUME" = "1" ]; then
    echo "  (resume mode: keeping existing shard checkpoints; shards continue them)"
else
    shopt -s nullglob
    prev=( "${RESULTS_DIR}"/v0_2_6_lal_subject_powered_shard*of*_*.json )
    if [ ${#prev[@]} -gt 0 ]; then
        archive="${RESULTS_DIR}/_prev_shards_${RUN_TS}"
        mkdir -p "$archive"
        echo "  Archiving ${#prev[@]} pre-existing shard checkpoint(s) -> $(basename "$archive")/"
        mv "${prev[@]}" "$archive"/
    fi
    shopt -u nullglob
fi

# ---- Launch shards in parallel ---------------------------------------------
echo "=== Launching ${NSHARDS} shards (model=${MODEL}, reps=${REPS}) ==="
pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
    log="${RESULTS_DIR}/shard_${i}of${NSHARDS}_${RUN_TS}.log"
    resume_arg=""
    if [ "$DO_RESUME" = "1" ]; then
        ckpt="$(ls -t "${RESULTS_DIR}"/v0_2_6_lal_subject_powered_shard${i}of${NSHARDS}_*.json 2>/dev/null | head -1)"
        if [ -n "$ckpt" ]; then
            resume_arg="--resume $ckpt"
            echo "  shard ${i}/${NSHARDS} resuming $(basename "$ckpt")"
        fi
    fi
    # shellcheck disable=SC2086
    python3 "$PILOT" --reps "$REPS" --shard "${i}/${NSHARDS}" --model "$MODEL" $resume_arg \
        > "$log" 2>&1 &
    pids+=("$!")
    echo "  shard ${i}/${NSHARDS} -> PID $! (log: $(basename "$log"))"
done

echo "  waiting for ${NSHARDS} shards..."
fail=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "  shard ${idx} OK"
    else
        echo "  shard ${idx} FAILED (see log); rerun with: python3 $PILOT --reps $REPS --shard ${idx}/${NSHARDS} --resume <ckpt>"
        fail=1
    fi
done
[ "$fail" = "1" ] && echo "WARN: >=1 shard failed; merge will use whatever completed."
echo

# ---- Merge -----------------------------------------------------------------
echo "=== Merging shards ==="
MERGED="${RESULTS_DIR}/v0_2_6_lal_subject_powered_merged_${RUN_TS}.json"
python3 "${PILOT_DIR}/v0_2_6_merge_shards.py" --expect-reps "$REPS" --out "$MERGED"
if [ ! -f "$MERGED" ]; then
    echo "ERROR: merge produced no output; aborting."; exit 1
fi
echo "  merged -> $(basename "$MERGED")"
echo

# ---- Primary (judge-independent) analyses ----------------------------------
echo "=== S1b primary + escalation + S4' CCS (judge-independent) ==="
python3 "${PILOT_DIR}/v0_2_6_bootstrap_s1b.py"        "$MERGED" | tee "${RESULTS_DIR}/H1_H4_H5_bootstrap_${RUN_TS}.txt"
python3 "${PILOT_DIR}/v0_2_6_escalation_contrast.py"  "$MERGED" | tee "${RESULTS_DIR}/escalation_contrast_${RUN_TS}.txt"
python3 "${PILOT_DIR}/v0_2_6_ccs_analysis.py"         "$MERGED" | tee "${RESULTS_DIR}/S4prime_ccs_analysis_${RUN_TS}.txt"
echo

# ---- Judge + secondary S1' -------------------------------------------------
if [ "$DO_JUDGE" = "1" ]; then
    echo "=== Judging merged subject file (judge=${JUDGE_MODEL}) ==="
    JUDGE_MODEL="$JUDGE_MODEL" python3 "${PILOT_DIR}/v0_2_6_lal_judge.py" "$MERGED" --judge-model "$JUDGE_MODEL" 2>&1 | tail -20
    JUDGED="$(ls -t "${RESULTS_DIR}"/v0_2_6_lal_judged_*.json 2>/dev/null | head -1)"
    if [ -n "$JUDGED" ]; then
        echo "  judged -> $(basename "$JUDGED")"
        echo "=== S1' secondary (powered bootstrap, judge-scored) ==="
        python3 "${PILOT_DIR}/v0_2_6_bootstrap_s1prime.py" "$JUDGED" | tee "${RESULTS_DIR}/S1prime_bootstrap_${RUN_TS}.txt"
    else
        echo "WARN: no judged file found; skipping S1' secondary."
    fi
fi

echo
echo "=== Done (${RUN_TS}) ==="
echo "  subject : $(basename "$MERGED")"
echo "  outputs : H1_H4_H5_bootstrap_${RUN_TS}.txt, escalation_contrast_${RUN_TS}.txt,"
echo "            S4prime_ccs_analysis_${RUN_TS}.txt, S1prime_bootstrap_${RUN_TS}.txt"
