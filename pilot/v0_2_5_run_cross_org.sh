#!/usr/bin/env bash
# Run cross-org analysis on all available judged files for v0.2.5 LAL.
# Excludes early/incomplete judged files; picks the latest per model_key.

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS="${RESULTS:-${PROJECT_ROOT}/pilot/results}"
SCRIPT="${SCRIPT:-${PROJECT_ROOT}/pilot/v0_2_5_lal_cross_org.py}"

# Anthropic: pick the two full 24-cell pilots
ANTH1="$RESULTS/v0_2_5_lal_judged_20260525_123220.json"  # Sonnet 4 (24 cells × 5 iters)
ANTH2="$RESULTS/v0_2_5_lal_judged_20260525_124652.json"  # Haiku 4.5 (24 cells × 5 iters)

# vLLM/HF: pick LATEST judged file per model_key
declare -A LATEST
for f in "$RESULTS"/v0_2_5_lal_judged_*.json; do
    # Extract model_key from filename if present (vllm) or skip
    name=$(basename "$f" .json)
    # Skip Anthropic timestamped ones
    if [[ "$name" == "v0_2_5_lal_judged_20260525_"* && ! "$name" == *"haiku45"* ]]; then
        continue
    fi
    # Read input_file to get subject path → derive model_key
    input=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('input_file',''))" "$f" 2>/dev/null)
    case "$input" in
        *v0_2_5_lal_vllm_*|*v0_2_5_lal_hf_*)
            mk=$(echo "$input" | sed -E 's/.*v0_2_5_lal_(vllm|hf)_([a-z0-9_-]+)_[0-9]{8}_.*/\2/')
            # Prefer vllm over hf if both exist
            if [[ "$input" == *"_vllm_"* ]]; then
                LATEST["$mk"]="$f"
            elif [ -z "${LATEST[$mk]:-}" ]; then
                LATEST["$mk"]="$f"
            fi
            ;;
    esac
done

# Build vllm args
VLLM_ARGS=""
for mk in "${!LATEST[@]}"; do
    VLLM_ARGS="$VLLM_ARGS ${LATEST[$mk]}"
done

echo "=== Files used ==="
echo "Anthropic Sonnet 4: $(basename "$ANTH1")"
echo "Anthropic Haiku 4.5: $(basename "$ANTH2")"
for mk in "${!LATEST[@]}"; do
    echo "Open-weight $mk: $(basename "${LATEST[$mk]}")"
done
echo

python3 "$SCRIPT" \
    --anthropic "$ANTH1" "$ANTH2" \
    --vllm $VLLM_ARGS \
    --out "$RESULTS/v0_2_5_cross_org_$(date +%H%M%S).json"
