#!/usr/bin/env bash

set -uo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=use_vdn_tools.sh
. "$script_dir/use_vdn_tools.sh"
# shellcheck source=vdn_release.env
. "$script_dir/vdn_release.env"

status=0

pass() {
    printf 'PASS  %s\n' "$1"
}

warn() {
    printf 'WARN  %s\n' "$1"
}

fail() {
    printf 'FAIL  %s\n' "$1"
    status=1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

bytes_available() {
    awk '/^MemAvailable:/ { print $2 * 1024; found = 1 } END { if (!found) print 0 }' \
        /proc/meminfo 2>/dev/null
}

bytes_free_at() {
    df -PB1 "$1" 2>/dev/null | awk 'NR == 2 { print $4 }'
}

gpu_count() {
    rocminfo 2>/dev/null | awk '
        /^[[:space:]]*Name:[[:space:]]+gfx[0-9]/ { count++ }
        END { print count + 0 }
    '
}

printf 'VDN native runtime preflight\n'
printf '  code revision   %s\n' "$VDN_CODE_REVISION"
printf '  model revision  %s\n' "$VDN_MODEL_REVISION"
printf '  target arch     %s\n\n' "$VDN_TARGET_ARCH"

if [[ $(uname -s) == Linux ]]; then
    pass "Linux host ($(uname -m))"
else
    fail "HIP build requires Linux; found $(uname -s)"
fi

if have make; then
    pass "make: $(command -v make)"
else
    fail "make is missing"
fi

if have hipcc; then
    hip_version=$(hipcc --version 2>/dev/null | awk '/HIP version:/ { print $3; exit }')
    pass "hipcc: $(command -v hipcc) (HIP ${hip_version:-unknown})"
else
    fail "hipcc is missing from PATH"
fi

if have rocminfo; then
    detected_gpus=$(gpu_count)
    if (( detected_gpus >= VDN_MIN_GPU_COUNT )); then
        pass "ROCm agents: $detected_gpus GPU(s)"
    else
        fail "ROCm exposes $detected_gpus GPU(s); need at least $VDN_MIN_GPU_COUNT"
    fi
    # Do not use grep -q here: with pipefail it can close the pipe early and make
    # rocminfo report SIGPIPE even though the architecture matched.
    if rocminfo 2>/dev/null | grep "Name:[[:space:]]*$VDN_TARGET_ARCH" >/dev/null; then
        pass "target architecture $VDN_TARGET_ARCH present"
    else
        fail "target architecture $VDN_TARGET_ARCH is not present"
    fi
else
    fail "rocminfo is missing"
fi

if have rocm-smi; then
    min_vram=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '
        /VRAM Total Memory \(B\):/ {
            value = $NF + 0
            if (minimum == 0 || value < minimum) minimum = value
        }
        END { print minimum + 0 }
    ')
    if (( min_vram >= VDN_MIN_GPU_BYTES )); then
        pass "minimum GPU VRAM: $min_vram bytes"
    else
        fail "minimum GPU VRAM is $min_vram bytes; need $VDN_MIN_GPU_BYTES"
    fi
else
    warn "rocm-smi is missing; per-device VRAM was not checked"
fi

available_ram=$(bytes_available)
if (( available_ram >= VDN_MIN_AVAILABLE_RAM_BYTES )); then
    pass "available host RAM: $available_ram bytes"
else
    fail "available host RAM is $available_ram bytes; need $VDN_MIN_AVAILABLE_RAM_BYTES before a model run"
fi

workspace_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
free_disk=$(bytes_free_at "$workspace_root")
if [[ -n $free_disk ]] && (( free_disk >= VDN_MODEL_MIN_FREE_BYTES )); then
    pass "workspace free disk: $free_disk bytes"
else
    fail "workspace needs at least $VDN_MODEL_MIN_FREE_BYTES free bytes"
fi

if have ffmpeg && have ffprobe; then
    pass "ffmpeg and ffprobe are available"
else
    fail "ffmpeg/ffprobe executables are required for the MP4 acceptance test"
fi

if have hf; then
    pass "Hugging Face CLI: $(command -v hf)"
elif have huggingface-cli; then
    pass "Hugging Face CLI: $(command -v huggingface-cli)"
elif python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    pass "huggingface_hub Python package is available"
else
    fail "install the Hugging Face CLI before downloading the pinned model"
fi

printf '\n'
if (( status == 0 )); then
    printf 'READY  environment satisfies the current VDN preflight\n'
else
    printf 'BLOCKED fix the FAIL entries before downloading/running the full model\n'
fi

exit "$status"
