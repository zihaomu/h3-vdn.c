#!/usr/bin/env python3
"""Strict preflight for original MiniMax-H3 correctness runs.

Unlike the developer smoke targets, every missing dependency is a failure.
The script never downloads weights and never starts a model workload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


FL2VA_FIXTURES = (
    "h3_dit.safetensors",
    "h3_dit_bf16.safetensors",
    "h3_text_bf16.safetensors",
    "h3_upstream_fl2va_text_v1.safetensors",
    "h3_upstream_fl2va_text_v1.safetensors.json",
    "h3_real_dit_block0_bf16.safetensors",
    "h3_real_dit_step0_bf16.safetensors",
    "h3_real_dit_denoise20_bf16.safetensors",
    "h3_real_prompt_bf16.safetensors",
    "h3_semantic_256x22_seed42.safetensors",
    "h3_upstream_fl2va_video_vae_v1.safetensors",
    "h3_upstream_fl2va_video_vae_v1.safetensors.json",
    "h3_upstream_fl2va_audio_vae_v1.safetensors",
    "h3_upstream_fl2va_audio_vae_v1.safetensors.json",
    "h3_upstream_fl2va_vision_64_v1.safetensors",
    "h3_upstream_fl2va_vision_64_v1.safetensors.json",
    "h3_upstream_fl2va_multimodal_64_v2.safetensors",
    "h3_upstream_fl2va_multimodal_64_v2.safetensors.json",
)

REF2VA_FIXTURES = (
    "h3_real_ref_video_text_64.safetensors",
)

MODEL_FILES = {
    "FL2VA": (
        "FL2VA/transformer/config.json",
        "FL2VA/tokenizer/tokenizer.json",
    ),
    "Ref2VA": (
        "Ref2VA/transformer/config.json",
        "Ref2VA/tokenizer/tokenizer.json",
    ),
}

MODEL_COMPONENTS = {
    "FL2VA": (
        "FL2VA/text_encoder",
        "FL2VA/transformer",
        "FL2VA/video_vae/source",
        "FL2VA/audio_vae",
    ),
    "Ref2VA": (
        "Ref2VA/text_encoder",
        "Ref2VA/transformer",
        "Ref2VA/video_vae/source",
        "Ref2VA/audio_vae",
    ),
}


class Checks:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def passed(self, message: str) -> None:
        print(f"PASS  {message}")

    def failed(self, message: str) -> None:
        print(f"FAIL  {message}")
        self.failures += 1

    def warned(self, message: str) -> None:
        print(f"WARN  {message}")
        self.warnings += 1


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="strict original H3 model, fixture, tool, and GPU preflight"
    )
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--model-root", type=Path, default=Path("MiniMax-H3"))
    parser.add_argument("--fixture-dir", type=Path, default=Path("misc/fixtures"))
    parser.add_argument("--physical-gpu", default="4")
    parser.add_argument("--expected-bdf", default="0000:e3:00.0")
    parser.add_argument("--expected-arch", default="gfx1201")
    parser.add_argument(
        "--fl2va-only",
        action="store_true",
        help="do not require the optional Ref2VA tree during staged bring-up",
    )
    parser.add_argument(
        "--skip-device",
        action="store_true",
        help="test-only: skip HIP device identity and idle-state checks",
    )
    return parser.parse_args()


def absolute(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def executable(repo_root: Path, name: str, local: str | None = None) -> str | None:
    if local:
        candidate = repo_root / local
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def require_file(checks: Checks, path: Path, label: str) -> None:
    if path.is_file() and path.stat().st_size > 0:
        checks.passed(f"{label}: {path}")
    else:
        checks.failed(f"missing or empty {label}: {path}")


def check_component(checks: Checks, root: Path, relative: str) -> None:
    component = root / relative
    index = component / "model.safetensors.index.json"
    single = component / "model.safetensors"
    if index.is_file():
        try:
            document = json.loads(index.read_text(encoding="utf-8"))
            weight_map = document.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is empty")
            shards = sorted(set(weight_map.values()))
            if not all(isinstance(item, str) and item for item in shards):
                raise ValueError("weight_map contains an invalid shard name")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            checks.failed(f"invalid model index {index}: {error}")
            return
        missing = [str(component / shard) for shard in shards if not (component / shard).is_file()]
        empty = [
            str(component / shard)
            for shard in shards
            if (component / shard).is_file() and (component / shard).stat().st_size == 0
        ]
        if missing or empty:
            detail = ", ".join(missing + empty)
            checks.failed(f"incomplete {relative} shards: {detail}")
        else:
            checks.passed(f"{relative}: indexed component with {len(shards)} shard(s)")
    elif single.is_file() and single.stat().st_size > 0:
        checks.passed(f"{relative}: single safetensors component")
    else:
        checks.failed(f"missing safetensors component: {component}")


def run_json(command: list[str], env: dict[str, str] | None = None) -> object:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    return json.loads(result.stdout)


def check_device(
    checks: Checks,
    repo_root: Path,
    physical_gpu: str,
    expected_bdf: str,
    expected_arch: str,
) -> None:
    probe = repo_root / "h3"
    amd_smi = Path(os.environ.get("AMD_SMI", "/opt/rocm/bin/amd-smi"))
    if not probe.is_file() or not os.access(probe, os.X_OK):
        checks.failed(f"device probe is not built: {probe}")
        return
    if not amd_smi.is_file() or not os.access(amd_smi, os.X_OK):
        checks.failed(f"AMD SMI is unavailable: {amd_smi}")
        return

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = physical_gpu
    try:
        result = subprocess.run(
            [str(probe), "--list-devices"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as error:
        checks.failed(f"cannot probe HIP physical GPU {physical_gpu}: {error}")
        return
    rows = [line.split("\t") for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) < 6:
        checks.failed(f"expected one visible HIP device, got: {result.stdout.strip()!r}")
        return
    logical, backend, arch, name, memory, bdf = rows[0][:6]
    if logical != "0" or backend != "hip":
        checks.failed(f"unexpected visible device row: {result.stdout.strip()}")
        return
    if bdf.lower() != expected_bdf.lower() or arch != expected_arch:
        checks.failed(
            f"physical GPU {physical_gpu} resolved to {bdf}/{arch}; "
            f"expected {expected_bdf}/{expected_arch}"
        )
        return
    checks.passed(
        f"HIP physical GPU {physical_gpu}: {name}, {memory}, {arch}, {bdf}"
    )

    try:
        devices = run_json([str(amd_smi), "list", "--json"])
        matches = [
            item
            for item in devices
            if str(item.get("bdf", "")).lower() == expected_bdf.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"AMD SMI mapping count is {len(matches)}")
        smi_gpu = str(matches[0]["gpu"])
        metric = run_json(
            [str(amd_smi), "metric", "--gpu", smi_gpu, "--usage", "--mem-usage", "--json"]
        )
        gpu_data = metric["gpu_data"][0]
        activity = int(gpu_data["usage"]["gfx_activity"]["value"])
        used_vram = int(gpu_data["mem_usage"]["used_vram"]["value"])
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        checks.failed(f"cannot validate target GPU idle state: {error}")
        return
    if activity > 10 or used_vram > 256:
        checks.failed(
            f"target GPU is busy: activity={activity}%, used_vram={used_vram} MiB"
        )
    else:
        checks.passed(
            f"target GPU is idle: activity={activity}%, used_vram={used_vram} MiB"
        )


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    model_root = absolute(repo_root, args.model_root).resolve()
    fixture_dir = absolute(repo_root, args.fixture_dir).resolve()
    checks = Checks()

    print("Original MiniMax-H3 reference preflight")
    print(f"  repository      {repo_root}")
    print(f"  model root      {model_root}")
    print(f"  fixture dir     {fixture_dir}")
    print(f"  physical GPU    {args.physical_gpu}")
    print(f"  expected BDF    {args.expected_bdf}")
    print()

    if sys.platform.startswith("linux"):
        checks.passed(f"Linux host: {os.uname().machine}")
    else:
        checks.failed(f"HIP reference run requires Linux; found {sys.platform}")

    hipcc = executable(repo_root, "hipcc", "/opt/rocm/bin/hipcc")
    make = executable(repo_root, "make")
    ffmpeg = executable(repo_root, "ffmpeg", ".tools/ffmpeg/usr/bin/ffmpeg")
    ffprobe = executable(repo_root, "ffprobe", ".tools/ffmpeg/usr/bin/ffprobe")
    for name, path in (("make", make), ("hipcc", hipcc), ("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        if path:
            checks.passed(f"{name}: {path}")
        else:
            checks.failed(f"{name} is unavailable")

    variants = ("FL2VA",) if args.fl2va_only else ("FL2VA", "Ref2VA")
    for variant in variants:
        for relative in MODEL_FILES[variant]:
            require_file(checks, model_root / relative, f"{variant} model file")
        for relative in MODEL_COMPONENTS[variant]:
            check_component(checks, model_root, relative)

    expected_fixtures = FL2VA_FIXTURES + (() if args.fl2va_only else REF2VA_FIXTURES)
    missing_fixtures = [name for name in expected_fixtures if not (fixture_dir / name).is_file()]
    empty_fixtures = [
        name
        for name in expected_fixtures
        if (fixture_dir / name).is_file() and (fixture_dir / name).stat().st_size == 0
    ]
    if missing_fixtures or empty_fixtures:
        for name in missing_fixtures:
            checks.failed(f"missing H3 fixture: {fixture_dir / name}")
        for name in empty_fixtures:
            checks.failed(f"empty H3 fixture: {fixture_dir / name}")
    else:
        checks.passed(f"all {len(expected_fixtures)} original H3 fixtures are present")

    if args.skip_device:
        checks.warned("device checks skipped by explicit test-only option")
    else:
        check_device(
            checks,
            repo_root,
            args.physical_gpu,
            args.expected_bdf,
            args.expected_arch,
        )

    print()
    print(f"SUMMARY failures={checks.failures} warnings={checks.warnings}")
    if checks.failures:
        print("BLOCKED original H3 reference prerequisites are incomplete")
        return 1
    print("READY original H3 reference prerequisites are complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
