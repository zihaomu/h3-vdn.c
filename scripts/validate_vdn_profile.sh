#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $0 OUTPUT.mp4.inference.json" >&2
    exit 2
fi

record=$1
if [[ ! -s $record ]]; then
    echo "missing or empty inference record: $record" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate an inference profile" >&2
    exit 2
fi

jq -e '
    . as $root |
    .schema_version == 2 and
    .backend == "hip" and
    (.pci_bus_id | test("^[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\\.[[:xdigit:]]$")) and
    .gpu_profile_enabled == true and
    (.nfe | type == "number" and . >= 1) and
    (.nfe_timings | type == "array" and length == $root.nfe) and
    ([.nfe_timings[] |
        .wall_seconds > 0 and
        .critical_path_seconds.forward > 0 and
        .critical_path_seconds.blocks > 0 and
        .gpu_profile_calls.sdpa == 50 and
        .weight_stream.read_bytes > 0 and
        .weight_stream.read_bytes == .weight_stream.h2d_bytes
    ] | all) and
    (.output_hashes_fnv1a64 | type == "object") and
    (.output_hashes_fnv1a64 | length == 5) and
    ([.output_hashes_fnv1a64[] | test("^[[:xdigit:]]{16}$")] | all) and
    .timing_seconds.total > 0 and
    .timing_seconds.denoise > 0 and
    .timing_seconds.video_vae > 0 and
    .timing_seconds.audio_vae > 0 and
    .timing_seconds.mux >= 0 and
    .timing_seconds.coverage >= 0.95 and
    .timing_seconds.coverage <= 1.001 and
    .timing_seconds.residual >= 0 and
    .weight_stream.read_bytes > 0 and
    .weight_stream.read_bytes == .weight_stream.h2d_bytes
' "$record" >/dev/null

echo "VDN inference profile passed: $record"
