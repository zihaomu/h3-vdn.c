#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
output_dir="$repo_dir/doc/baseline"
mkdir -p "$output_dir"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$output_dir/rocm-$timestamp.txt"

{
    printf 'captured_utc=%s\n' "$timestamp"
    printf 'git_head=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git -C "$repo_dir" branch --show-current)"
    printf '\n[uname]\n'
    uname -a
    printf '\n[release]\n'
    # shellcheck source=vdn_release.env
    . "$script_dir/vdn_release.env"
    LC_ALL=C sort "$script_dir/vdn_release.env"
    printf '\n[hipcc]\n'
    hipcc --version 2>&1 || true
    printf '\n[gpus]\n'
    rocm-smi --showproductname --showmeminfo vram --showdriverversion 2>&1 || true
    printf '\n[memory]\n'
    free -b 2>&1 || true
    printf '\n[disk]\n'
    df -PB1 "$repo_dir" 2>&1 || true
    printf '\n[numa]\n'
    lscpu 2>&1 || true
    printf '\n[preflight]\n'
    "$script_dir/check_vdn_env.sh" 2>&1 || true
} >"$output"

printf '%s\n' "$output"
