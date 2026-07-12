#!/usr/bin/env bash
# _launch_lal_track2.sh — detached-tmux launcher for the Track 2 powered run.
# Requires ANTHROPIC_API_KEY in the calling environment, keeps the host awake
# with caffeinate when available, and drops a completion sentinel.
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PILOT_DIR}/results"
LOG="${RESULTS_DIR}/run_console_track2.log"
DONE="${RESULTS_DIR}/run_track2.done"
mkdir -p "$RESULTS_DIR"
rm -f "$DONE"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "EXIT=1 (no ANTHROPIC_API_KEY)" > "$DONE"; exit 1
fi

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i ./v0_2_6_lal_run_powered.sh --resume > "$LOG" 2>&1
else
    ./v0_2_6_lal_run_powered.sh --resume > "$LOG" 2>&1
fi
rc=$?
echo "EXIT=${rc}" > "$DONE"
