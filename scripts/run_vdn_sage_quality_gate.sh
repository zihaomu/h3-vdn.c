#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "usage: $0 BASE_MODEL CHECKPOINT PROMPT_DIRECTORY OUTPUT_DIRECTORY [SAGE_MODE]" >&2
    echo "Runs three production prompt gates sequentially on physical GPU 4." >&2
}

if (( $# < 4 || $# > 5 )); then
    usage
    exit 2
fi

base_model=$1
checkpoint=$2
prompt_directory=$3
output_directory=$4
sage_mode=${5:-sage-i8-bf16}

case $sage_mode in
    sage-*) ;;
    *)
        echo "SAGE_MODE must be an explicit sage-* mode" >&2
        exit 2
        ;;
esac

for required in ./h3 scripts/profile_vdn_gpu4.sh \
        scripts/validate_vdn_profile.sh scripts/compare_vdn_media.py; do
    if [[ ! -x $required ]]; then
        echo "required executable is unavailable: $required" >&2
        exit 2
    fi
done
if [[ ! -d $base_model || ! -d $checkpoint || ! -d $prompt_directory ]]; then
    echo "model, checkpoint, and prompt paths must be directories" >&2
    exit 2
fi
if [[ -e $output_directory ]] &&
   [[ -n $(find "$output_directory" -mindepth 1 -print -quit 2>/dev/null) ]]; then
    echo "refusing to overwrite non-empty output directory: $output_directory" >&2
    exit 2
fi

# shellcheck source=/dev/null
. scripts/use_vdn_tools.sh
mkdir -p "$output_directory"
summary=$output_directory/summary.tsv
printf 'prompt\treference_seconds\tcandidate_mean_seconds\tspeedup_percent\tcandidate_sha256\tquality\n' >"$summary"

run_render() {
    local label=$1
    local mode=$2
    local prompt=$3
    local output=$4
    local telemetry=$output_directory/telemetry-$label
    H3_TELEMETRY_QUIET=1 scripts/profile_vdn_gpu4.sh "$telemetry" -- \
        env -u H3_VDN_RESIDENT_GIB H3_PROFILE=1 H3_VDN_SDPA="$mode" \
        ./h3 --profile --device 0 \
        --model-dir "$base_model" \
        --vdn-checkpoint "$checkpoint" \
        --prompt-embeds "$prompt" \
        --width 512 --height 512 --frames 56 --seed 0 \
        --output "$output"
    scripts/validate_vdn_profile.sh "$output.inference.json"
}

for prompt_index in 0 1 2; do
    prompt_name=example_$prompt_index
    prompt=$prompt_directory/$prompt_name.safetensors
    if [[ ! -f $prompt ]]; then
        echo "missing fixed prompt input: $prompt" >&2
        exit 2
    fi
    reference=$output_directory/$prompt_name-wave32.mp4
    candidate=$output_directory/$prompt_name-$sage_mode.mp4
    repeat=$output_directory/$prompt_name-$sage_mode-repeat.mp4

    run_render "$prompt_name-wave32" wave32 "$prompt" "$reference"
    run_render "$prompt_name-candidate" "$sage_mode" "$prompt" "$candidate"
    python3 scripts/compare_vdn_media.py "$reference" "$candidate" \
        --output "$candidate.quality.json" >/dev/null
    run_render "$prompt_name-candidate-repeat" "$sage_mode" "$prompt" "$repeat"
    python3 scripts/compare_vdn_media.py "$reference" "$repeat" \
        --output "$repeat.quality.json" >/dev/null

    first_hashes=$(jq -cS '.output_hashes_fnv1a64' \
        "$candidate.inference.json")
    second_hashes=$(jq -cS '.output_hashes_fnv1a64' \
        "$repeat.inference.json")
    first_sha=$(sha256sum "$candidate" | awk '{print $1}')
    second_sha=$(sha256sum "$repeat" | awk '{print $1}')
    if [[ $first_hashes != "$second_hashes" || $first_sha != "$second_sha" ]]; then
        echo "$prompt_name candidate is not deterministic" >&2
        exit 1
    fi

    reference_seconds=$(jq -er '.timing_seconds.total' \
        "$reference.inference.json")
    first_seconds=$(jq -er '.timing_seconds.total' \
        "$candidate.inference.json")
    second_seconds=$(jq -er '.timing_seconds.total' \
        "$repeat.inference.json")
    candidate_mean=$(awk -v a="$first_seconds" -v b="$second_seconds" \
        'BEGIN { printf "%.6f", (a + b) / 2.0 }')
    speedup=$(awk -v reference="$reference_seconds" -v candidate="$candidate_mean" \
        'BEGIN { printf "%.3f", 100.0 * (reference - candidate) / reference }')
    if ! awk -v reference="$reference_seconds" -v candidate="$candidate_mean" \
        'BEGIN { exit !(candidate <= reference * 0.90) }'; then
        echo "$prompt_name candidate missed the 10% production speed gate" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\tPASS\n' \
        "$prompt_name" "$reference_seconds" "$candidate_mean" "$speedup" \
        "$first_sha" >>"$summary"
done

echo "VDN Sage three-prompt production gate passed: $summary"
