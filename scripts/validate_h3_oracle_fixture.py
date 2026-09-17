#!/usr/bin/env python3
"""Validate an original-H3 safetensors oracle and its provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


SCHEMA = "h3-upstream-oracle-v1"
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
REQUIRED_PROVENANCE = (
    "upstream_repo", "upstream_revision", "model_repo", "model_revision",
    "task", "case_id", "seed", "rng_policy", "exporter_sha256",
)


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def sha256_range(path: Path, start: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(start)
        remaining = length
        while remaining is None or remaining:
            block_size = 1 << 20 if remaining is None else min(1 << 20, remaining)
            block = source.read(block_size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
        if remaining not in (None, 0):
            fail(f"short read while hashing {path}")
    return digest.hexdigest()


def read_header(path: Path) -> tuple[dict, int, int]:
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            fail(f"{path}: missing safetensors length prefix")
        (header_size,) = struct.unpack("<Q", prefix)
        if header_size == 0 or header_size > size - 8:
            fail(f"{path}: invalid safetensors header length {header_size}")
        encoded = source.read(header_size)
    try:
        header = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{path}: invalid safetensors JSON header: {exc}")
    if not isinstance(header, dict):
        fail(f"{path}: safetensors header must be an object")
    return header, 8 + header_size, size


def validate_header(path: Path, header: dict, data_start: int, file_size: int) -> dict:
    result = {}
    intervals = []
    data_size = file_size - data_start
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                fail(f"{path}: __metadata__ must be an object")
            continue
        if not isinstance(descriptor, dict):
            fail(f"{path}: tensor {name!r} descriptor is not an object")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            fail(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or value < 0 for value in shape
        ):
            fail(f"{path}: tensor {name!r} has invalid shape")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > data_size
        ):
            fail(f"{path}: tensor {name!r} has invalid data offsets")
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        actual = offsets[1] - offsets[0]
        if actual != expected:
            fail(
                f"{path}: tensor {name!r} payload is {actual} bytes, expected {expected}"
            )
        intervals.append((offsets[0], offsets[1], name))
        result[name] = {
            "dtype": dtype,
            "shape": shape,
            "sha256": sha256_range(path, data_start + offsets[0], actual),
        }
    if not result:
        fail(f"{path}: fixture contains no tensors")
    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            fail(f"{path}: tensor {name!r} leaves a gap or overlaps at byte {cursor}")
        cursor = end
    if cursor != data_size:
        fail(f"{path}: unclaimed trailing payload bytes")
    return result


def require_string(mapping: dict, key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        fail(f"{where}: missing non-empty string {key!r}")
    return value


def validate_manifest(path: Path, fixture: Path, tensors: dict) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")
    if not isinstance(manifest, dict):
        fail(f"{path}: manifest must be an object")
    if manifest.get("schema") != SCHEMA:
        fail(f"{path}: schema must be {SCHEMA!r}")
    require_string(manifest, "scope", str(path))
    fixture_info = manifest.get("fixture")
    if not isinstance(fixture_info, dict):
        fail(f"{path}: fixture must be an object")
    expected_fixture_hash = require_string(fixture_info, "sha256", str(path))
    actual_fixture_hash = sha256_range(fixture)
    if expected_fixture_hash != actual_fixture_hash:
        fail(f"{path}: whole-fixture SHA-256 mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        fail(f"{path}: provenance must be an object")
    for key in REQUIRED_PROVENANCE:
        require_string(provenance, key, str(path))
    manifest_tensors = manifest.get("tensors")
    if not isinstance(manifest_tensors, dict):
        fail(f"{path}: tensors must be an object")
    if set(manifest_tensors) != set(tensors):
        missing = sorted(set(tensors) - set(manifest_tensors))
        extra = sorted(set(manifest_tensors) - set(tensors))
        fail(f"{path}: tensor set mismatch; missing={missing}, extra={extra}")
    for name, actual in tensors.items():
        expected = manifest_tensors[name]
        if not isinstance(expected, dict):
            fail(f"{path}: tensor manifest {name!r} must be an object")
        for field in ("dtype", "shape", "sha256"):
            if expected.get(field) != actual[field]:
                fail(f"{path}: tensor {name!r} {field} mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate original-H3 oracle safetensors and manifest"
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        help="default: FIXTURE.safetensors.json",
    )
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    manifest_path = (
        args.manifest.resolve() if args.manifest
        else fixture.with_suffix(fixture.suffix + ".json")
    )
    if not fixture.is_file():
        fail(f"fixture does not exist: {fixture}")
    if not manifest_path.is_file():
        fail(f"manifest does not exist: {manifest_path}")
    header, data_start, file_size = read_header(fixture)
    tensors = validate_header(fixture, header, data_start, file_size)
    manifest = validate_manifest(manifest_path, fixture, tensors)
    print(
        f"PASS {fixture.name}: schema={manifest['schema']} "
        f"scope={manifest['scope']} tensors={len(tensors)} "
        f"sha256={manifest['fixture']['sha256']}"
    )


if __name__ == "__main__":
    main()
