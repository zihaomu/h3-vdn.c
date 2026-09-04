#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
# shellcheck source=use_vdn_tools.sh
. "$script_dir/use_vdn_tools.sh"
# shellcheck source=vdn_release.env
. "$script_dir/vdn_release.env"

destination=${1:-"$repo_dir/models/vdn-minimax-h3"}

if [[ $destination != /* ]]; then
    destination="$PWD/$destination"
fi

mkdir -p "$destination"

printf 'Downloading %s at revision %s\n' \
    "$VDN_MODEL_REPOSITORY" "$VDN_MODEL_REVISION"
printf 'Destination: %s\n' "$destination"

if command -v hf >/dev/null 2>&1; then
    hf download "$VDN_MODEL_REPOSITORY" \
        --revision "$VDN_MODEL_REVISION" --local-dir "$destination"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$VDN_MODEL_REPOSITORY" \
        --revision "$VDN_MODEL_REVISION" --local-dir "$destination"
elif python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    python3 - "$VDN_MODEL_REPOSITORY" "$VDN_MODEL_REVISION" "$destination" <<'PY'
import sys
from huggingface_hub import snapshot_download

repository, revision, destination = sys.argv[1:]
snapshot_download(repository, revision=revision, local_dir=destination)
PY
else
    printf 'error: hf, huggingface-cli, or the huggingface_hub Python package is required\n' >&2
    exit 2
fi

actual_bytes=$(find "$destination" -type f -printf '%s\n' | awk '{ total += $1 } END { print total + 0 }')
if (( actual_bytes < VDN_MODEL_EXPECTED_BYTES )); then
    printf 'error: downloaded %s bytes, expected at least %s\n' \
        "$actual_bytes" "$VDN_MODEL_EXPECTED_BYTES" >&2
    exit 1
fi

revision_file="$destination/.h3-vdn-revision"
{
    printf 'repository=%s\n' "$VDN_MODEL_REPOSITORY"
    printf 'revision=%s\n' "$VDN_MODEL_REVISION"
    printf 'bytes=%s\n' "$actual_bytes"
} >"$revision_file"

printf 'Verified byte count: %s\n' "$actual_bytes"
printf 'Revision marker: %s\n' "$revision_file"
