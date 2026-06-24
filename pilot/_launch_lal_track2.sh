#!/usr/bin/env bash
# _launch_lal_track2.sh — detached-tmux launcher for the Track 2 powered run.
# Sources the API key from ~/.api_keys (never echoes it), keeps the Mac awake
# with caffeinate, and drops a completion sentinel so the agent can detect end.
set -uo pipefail

PILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PILOT_DIR}/results"
LOG="${RESULTS_DIR}/run_console_track2.log"
DONE="${RESULTS_DIR}/run_track2.done"
mkdir -p "$RESULTS_DIR"
rm -f "$DONE"

# Bring the key into the environment (file is shell-sourceable; value not printed).
# shellcheck disable=SC1090
source "$HOME/.api_keys" 2>/dev/null

cd "$PILOT_DIR" || { echo "EXIT=127 (cd failed)" > "$DONE"; exit 127; }

if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i ./v0_2_6_lal_run_powered.sh --resume > "$LOG" 2>&1
else
    ./v0_2_6_lal_run_powered.sh --resume > "$LOG" 2>&1
fi
rc=$?
echo "EXIT=${rc}" > "$DONE"
