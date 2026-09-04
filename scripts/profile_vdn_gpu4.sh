#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "usage: $0 OUTPUT_DIRECTORY -- COMMAND [ARGUMENT ...]" >&2
    echo "Runs COMMAND with only HIP ordinal 4 visible and captures 1 Hz GPU, disk, and process telemetry." >&2
    echo "Set H3_TELEMETRY_DISABLE=1 for a preflighted no-sampling control." >&2
}

if (( $# < 3 )) || [[ $2 != -- ]]; then
    usage
    exit 2
fi

output_directory=$1
shift 2
command=("$@")
physical_gpu=4
sample_interval=${H3_TELEMETRY_INTERVAL:-1}
telemetry_disabled=${H3_TELEMETRY_DISABLE:-0}
amd_smi=${AMD_SMI:-/opt/rocm/bin/amd-smi}
device_probe=${H3_DEVICE_PROBE:-./h3}

if [[ ! $sample_interval =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid H3_TELEMETRY_INTERVAL=$sample_interval; use whole seconds >= 1" >&2
    exit 2
fi
if [[ $telemetry_disabled != 0 && $telemetry_disabled != 1 ]]; then
    echo "invalid H3_TELEMETRY_DISABLE=$telemetry_disabled; use 0 or 1" >&2
    exit 2
fi
if [[ ! -x $amd_smi ]]; then
    echo "AMD SMI is unavailable at $amd_smi" >&2
    exit 2
fi
if [[ ! -x $device_probe ]]; then
    echo "h3 device probe is unavailable at $device_probe" >&2
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate the physical GPU 4 identity and idle state" >&2
    exit 2
fi

refuse_other_h3_workloads() {
    # A second h3 process can select a different physical card and therefore
    # evade the target-card utilization check below. This workspace permits
    # only one h3 GPU workload at a time.
    for process_directory in /proc/[0-9]*; do
        process_pid=${process_directory##*/}
        [[ $process_pid == $$ ]] && continue
        process_executable=$(readlink "$process_directory/exe" 2>/dev/null || true)
        case $process_executable in
            "$PWD/h3"|"$PWD"/h3_*)
                echo "another h3 workload is already running (PID=$process_pid, executable=$process_executable); refusing concurrent GPU use" >&2
                return 1
                ;;
        esac
    done
}

refuse_other_h3_workloads

if [[ -e $output_directory ]] && [[ -n $(find "$output_directory" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null) ]]; then
    echo "refusing to overwrite non-empty telemetry directory: $output_directory" >&2
    exit 2
fi
mkdir -p "$output_directory"

export HIP_VISIBLE_DEVICES=$physical_gpu
selected_device=$($device_probe --list-devices)
selected_bdf=$(awk -F '\t' '$1 == "0" {print $NF}' <<<"$selected_device")
amd_smi_gpu=$($amd_smi list --json | jq -r --arg bdf "$selected_bdf" \
    '.[] | select(.bdf == $bdf) | .gpu')
if [[ -z $selected_bdf || $selected_bdf == "-" ||
      ! $amd_smi_gpu =~ ^[0-9]+$ ]]; then
    echo "cannot map HIP ordinal 4 to an AMD SMI GPU; refusing to start" >&2
    exit 1
fi
identity_json=$($amd_smi static --gpu "$amd_smi_gpu" --asic --bus --json)
initial_json=$($amd_smi metric --gpu "$amd_smi_gpu" --usage --mem-usage --json)
identity_gpu=$(jq -r '.gpu_data[0].gpu // empty' <<<"$identity_json")
identity_bdf=$(jq -r '.gpu_data[0].bus.bdf // empty' <<<"$identity_json")
identity_arch=$(jq -r '.gpu_data[0].asic.target_graphics_version // empty' <<<"$identity_json")
gpu_use=$(jq -r '.gpu_data[0].usage.gfx_activity.value // empty' <<<"$initial_json")
gpu_memory_mb=$(jq -r '.gpu_data[0].mem_usage.used_vram.value // empty' <<<"$initial_json")
if [[ $identity_gpu != "$amd_smi_gpu" || $identity_bdf != "$selected_bdf" ||
      -z $identity_arch || -z $gpu_use || -z $gpu_memory_mb ]]; then
    echo "cannot parse physical GPU 4 utilization; refusing to start" >&2
    exit 1
fi
if (( gpu_use > 10 || gpu_memory_mb > 256 )); then
    echo "physical GPU 4 is busy (GPU=${gpu_use}%, used VRAM=${gpu_memory_mb} MiB); refusing to use another GPU" >&2
    exit 1
fi

# Repeat after device discovery so a process started during the BDF/SMI probe
# cannot slip through the first workspace-wide check.
refuse_other_h3_workloads

{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "physical_gpu=$physical_gpu"
    echo "physical_gpu_bdf=$identity_bdf"
    echo "amd_smi_gpu=$amd_smi_gpu"
    echo "physical_gpu_arch=$identity_arch"
    echo "hip_visible_devices=$HIP_VISIBLE_DEVICES"
    echo "sample_interval_seconds=$sample_interval"
    echo "telemetry_enabled=$((1 - telemetry_disabled))"
    echo "working_directory=$PWD"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srmo)"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_commit=$(git rev-parse HEAD)"
        echo "git_status_begin"
        git status --short
        echo "git_status_end"
    fi
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
    echo "initial_gpu_identity_json=$identity_json"
    echo "initial_gpu_metric_json=$initial_json"
} >"$output_directory/run.meta"

workload_pid=
gpu_sampler_pid=
disk_sampler_pid=
process_sampler_pid=
workspace_guard_pid=

stop_sampler() {
    local pid=$1
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    stop_sampler "$gpu_sampler_pid"
    stop_sampler "$disk_sampler_pid"
    stop_sampler "$process_sampler_pid"
    stop_sampler "$workspace_guard_pid"
}
trap cleanup EXIT INT TERM

if [[ ${H3_TELEMETRY_QUIET:-0} == 1 ]]; then
    "${command[@]}" >"$output_directory/stdout.log" \
        2>"$output_directory/stderr.log" &
else
    "${command[@]}" \
        > >(tee "$output_directory/stdout.log") \
        2> >(tee "$output_directory/stderr.log" >&2) &
fi
workload_pid=$!
echo "workload_pid=$workload_pid" >>"$output_directory/run.meta"

# Close the remaining post-launch race with direct invocations that do not use
# this wrapper. If another h3 executable appears, invalidate this run by
# terminating our own workload; never migrate it to a different GPU.
(
    while kill -0 "$workload_pid" 2>/dev/null; do
        for process_directory in /proc/[0-9]*; do
            process_pid=${process_directory##*/}
            [[ $process_pid == "$workload_pid" ]] && continue
            process_executable=$(readlink "$process_directory/exe" 2>/dev/null || true)
            case $process_executable in
                "$PWD/h3"|"$PWD"/h3_*)
                    echo "detected concurrent h3 workload PID=$process_pid; terminating guarded PID=$workload_pid"
                    kill -TERM "$workload_pid" 2>/dev/null || true
                    exit 0
                    ;;
            esac
        done
        sleep 1
    done
) >"$output_directory/concurrency.guard.log" 2>&1 &
workspace_guard_pid=$!

if [[ $telemetry_disabled == 0 ]]; then
    (
        while kill -0 "$workload_pid" 2>/dev/null; do
            timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            if metric_json=$($amd_smi metric --gpu "$amd_smi_gpu" --usage \
                    --mem-usage --temperature --power --clock --pcie --json); then
                jq -c --arg timestamp "$timestamp" \
                    '{sample_utc: $timestamp, gpu: .gpu_data[0]}' \
                    <<<"$metric_json"
            else
                printf '{"sample_utc":"%s","error":"amd-smi metric failed"}\n' \
                    "$timestamp"
            fi
            sleep "$sample_interval"
        done
    ) >"$output_directory/gpu4.telemetry.log" 2>&1 &
    gpu_sampler_pid=$!

    if command -v iostat >/dev/null 2>&1; then
        iostat -dx "$sample_interval" >"$output_directory/disk.iostat.log" 2>&1 &
        disk_sampler_pid=$!
    fi

    (
        echo -e "sample_utc\tpid\tread_bytes\tcancelled_write_bytes\tvoluntary_ctxt\tinvoluntary_ctxt\tschedstat\tproc_stat"
        while kill -0 "$workload_pid" 2>/dev/null; do
            timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            if [[ -r /proc/$workload_pid/stat ]]; then
                stat_line=$(<"/proc/$workload_pid/stat")
                schedstat_line=$(<"/proc/$workload_pid/schedstat")
                read_bytes=$(awk '$1 == "read_bytes:" {print $2}' "/proc/$workload_pid/io" 2>/dev/null || true)
                cancelled_write_bytes=$(awk '$1 == "cancelled_write_bytes:" {print $2}' "/proc/$workload_pid/io" 2>/dev/null || true)
                voluntary=$(awk '$1 == "voluntary_ctxt_switches:" {print $2}' "/proc/$workload_pid/status" 2>/dev/null || true)
                involuntary=$(awk '$1 == "nonvoluntary_ctxt_switches:" {print $2}' "/proc/$workload_pid/status" 2>/dev/null || true)
                echo -e "$timestamp\t$workload_pid\t${read_bytes:-0}\t${cancelled_write_bytes:-0}\t${voluntary:-0}\t${involuntary:-0}\t$schedstat_line\t$stat_line"
            fi
            sleep "$sample_interval"
        done
    ) >"$output_directory/process.telemetry.tsv" 2>&1 &
    process_sampler_pid=$!
fi

set +e
wait "$workload_pid"
status=$?
set -e
workload_pid=
cleanup
trap - EXIT INT TERM

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_status=$status"
} >>"$output_directory/run.meta"

exit "$status"
