#!/usr/bin/env bash
# _judge_lal_track2.sh — detached judge pass for the Track 2 powered merged file.
# Mirrors _launch_lal_track2.sh: sources the key (never echoes it), caffeinate,
# logs to a file, drops a sentinel. Run detached so it survives turn boundaries:
#   nohup bash _judge_lal_track2.sh >/dev/null 2>&1 &
# Scores 1200 cells x 5 iters = 6000 judge calls (no resume; single process).
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PILOT_DIR}/results"
LOG="${RESULTS_DIR}/judge_console_track2.log"
DONE="${RESULTS_DIR}/judge_track2.done"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-20250514}"
mkdir -p "$RESULTS_DIR"
rm -f "$DONE"

# newest merged subject file
MERGED="$(ls -t "${RESULTS_DIR}"/v0_2_6_lal_subject_powered_merged_*.json 2>/dev/null | head -1)"
if [ -z "$MERGED" ]; then echo "EXIT=2 (no merged file)" > "$DONE"; exit 2; fi

# shellcheck disable=SC1090
source "$HOME/.api_keys" 2>/dev/null

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i python3 -u v0_2_6_lal_judge.py "$MERGED" --judge-model "$JUDGE_MODEL" > "$LOG" 2>&1
else
    python3 -u v0_2_6_lal_judge.py "$MERGED" --judge-model "$JUDGE_MODEL" > "$LOG" 2>&1
fi
rc=$?
echo "EXIT=${rc}" > "$DONE"
