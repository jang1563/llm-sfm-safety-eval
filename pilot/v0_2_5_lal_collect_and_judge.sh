#!/usr/bin/env bash
# Pull v0.2.5 LAL vLLM results from Cayuga and run judge on any not yet judged.
#
# Idempotent: run as often as you like. Skips already-judged inputs.
#
# Usage:
#   ./v0_2_5_lal_collect_and_judge.sh                    # collect + judge all
#   ./v0_2_5_lal_collect_and_judge.sh --no-judge         # collect only
#   ./v0_2_5_lal_collect_and_judge.sh --no-rsync         # judge already-pulled files only

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAYUGA_HOST="${CAYUGA_HOST:-cayuga-login1}"
CAYUGA_USER="${CAYUGA_USER:-$USER}"
CAYUGA_DIR="${CAYUGA_DIR:-/athena/masonlab/scratch/users/${CAYUGA_USER}/d_spec_experiment/results}"
LOCAL_DIR="${LOCAL_DIR:-${PROJECT_ROOT}/pilot/results}"
SSH_CONTROL_DIR="${SSH_CONTROL_DIR:-${HOME}/.ssh/sockets}"
SSH_OPTS="${SSH_OPTS:--o ControlPath=${SSH_CONTROL_DIR}/%r@%h-%p}"

DO_RSYNC=1
DO_JUDGE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --no-rsync) DO_RSYNC=0 ;;
        --no-judge) DO_JUDGE=0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

cd "$LOCAL_DIR"

if [ "$DO_RSYNC" = "1" ]; then
    echo "=== Pulling v0_2_5_lal_vllm_*.json from $CAYUGA_HOST ==="
    # Enumerate via ssh, then rsync each file individually (rsync globs are awkward)
    files=$(ssh $SSH_OPTS "$CAYUGA_HOST" "ls $CAYUGA_DIR/v0_2_5_lal_vllm_*.json 2>/dev/null")
    if [ -z "$files" ]; then
        echo "  (no vLLM files on remote yet)"
    else
        for remote_f in $files; do
            local_f="${LOCAL_DIR}/$(basename "$remote_f")"
            if [ -f "$local_f" ]; then
                echo "  EXISTS $(basename "$remote_f")"
            else
                echo "  PULL   $(basename "$remote_f")"
                scp -q $SSH_OPTS "${CAYUGA_HOST}:${remote_f}" "${local_f}"
            fi
        done
    fi
    echo
fi

if [ "$DO_JUDGE" = "0" ]; then
    echo "Skipping judge step."
    exit 0
fi

api_key="${ANTHROPIC_API_KEY:-}"
if [ -z "$api_key" ]; then
    api_key=$(grep '^export ANTHROPIC_API_KEY' ~/.zshrc 2>/dev/null | sed 's/.*="\(.*\)".*/\1/')
fi
if [ -z "$api_key" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set and not found in ~/.zshrc"
    exit 1
fi
export ANTHROPIC_API_KEY="$api_key"

JUDGE_SCRIPT="$LOCAL_DIR/../v0_2_5_lal_judge.py"

echo "=== Running judge on un-judged v0_2_5_lal_vllm_*.json ==="
for subj in "$LOCAL_DIR"/v0_2_5_lal_vllm_*.json; do
    [ -f "$subj" ] || continue
    base=$(basename "$subj" .json)
    # Extract model key: v0_2_5_lal_vllm_<model>_<ts>.json -> <model>
    model_key=$(echo "$base" | sed 's/v0_2_5_lal_vllm_//' | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')

    # Skip only if a judged file references THIS exact subject path
    subj_base=$(basename "$subj")
    already=$(grep -l "\"input_file\":.*${subj_base}" "$LOCAL_DIR"/v0_2_5_lal_judged_*.json 2>/dev/null | head -1)
    if [ -n "$already" ]; then
        echo "  SKIP $model_key — already judged ($(basename "$already"))"
        continue
    fi

    echo "  JUDGE $model_key ($subj)"
    python3 "$JUDGE_SCRIPT" "$subj" 2>&1 | tail -5
    echo
done

echo "=== Done. Judged files: ==="
ls -la "$LOCAL_DIR"/v0_2_5_lal_judged_*.json 2>/dev/null | tail -15
