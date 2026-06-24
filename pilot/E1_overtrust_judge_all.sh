#!/bin/bash
# E1_overtrust_judge_all.sh — blinded Sonnet-4 LLM judge over the 10 open-weight
# subject runs (Spec §7), concurrency-pooled. Produces results/E1_judged_<model>.json
# with both rule + judge scores per row, for the cross-vendor payload ladder + κ gate.
#
# Usage:  bash E1_overtrust_judge_all.sh [MAXPAR]
set -u

cd "$(dirname "$0")"
source ~/.api_keys           # sets ANTHROPIC_API_KEY (inherited by & subshells)

MODELS=(mistral qwen qwen3-4b phi4-mini ministral3-14b \
        gemma4-31b gemma4-moe qwen3-30b-moe qwen3-32b olmo2-32b)
MAXPAR="${1:-4}"

echo "=== E1 batch LLM judge — $(date) ==="
echo "models: ${#MODELS[@]}   max parallel: ${MAXPAR}   judge: claude-sonnet-4-20250514"
echo

judge_one() {
    local mk="$1"
    local subj
    subj=$(ls -t results/E1_subject_${mk}_vllm_*.json 2>/dev/null | head -1)
    if [ -z "$subj" ]; then echo "[${mk}] NO SUBJECT FILE — skipped"; return 1; fi
    local log="results/_judge_${mk}.log"
    local out="results/E1_judged_${mk}.json"
    echo "[${mk}] START  subj=$(basename "$subj")"
    if python3 E1_overtrust_judge.py "$subj" --out "$out" > "$log" 2>&1; then
        local pf
        pf=$(grep -oE "parse-fail=[0-9]+" "$log" | tail -1)
        local nc
        nc=$(grep -oE "calls=[0-9]+" "$log" | tail -1)
        echo "[${mk}] DONE  -> $(basename "$out")  ${nc} ${pf}"
    else
        echo "[${mk}] FAILED (exit $?) — see ${log}; tail:"
        tail -3 "$log" | sed 's/^/    /'
    fi
}

t0=$(date +%s)
for mk in "${MODELS[@]}"; do
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXPAR" ]; do sleep 3; done
    judge_one "$mk" &
done
wait
t1=$(date +%s)

echo
echo "=== judged files produced ==="
ls -1 results/E1_judged_*.json 2>/dev/null | grep -E "E1_judged_(mistral|qwen|qwen3-4b|phi4-mini|ministral3-14b|gemma4-31b|gemma4-moe|qwen3-30b-moe|qwen3-32b|olmo2-32b)\.json"
echo
echo "ALL JUDGE DONE in $(( (t1 - t0) / 60 ))m$(( (t1 - t0) % 60 ))s"
