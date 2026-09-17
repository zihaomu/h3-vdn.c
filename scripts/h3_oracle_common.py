"""Shared deterministic writer for original MiniMax-H3 oracle fixtures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct


SCHEMA = "h3-upstream-oracle-v1"


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(dtype=torch.uint8).numpy().tobytes()


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def write_canonical_safetensors(
    path: Path, tensors: dict[str, "torch.Tensor"], metadata: dict[str, str]
) -> None:
    dtype_names = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    payloads = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in dtype_names:
            fail(f"canonical writer does not support {name} dtype {tensor.dtype}")
        payload = tensor_bytes(tensor)
        header[name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    encoded = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for payload in payloads:
            output.write(payload)


def write_fixture(
    output: Path,
    tensors: dict[str, "torch.Tensor"],
    scope: str,
    provenance: dict[str, str],
    execution: dict[str, str],
) -> tuple[Path, Path]:
    required = {
        "upstream_repo", "upstream_revision", "model_repo", "model_revision",
        "task", "case_id", "seed", "rng_policy", "exporter_sha256",
    }
    missing = sorted(required - set(provenance))
    if missing:
        fail(f"missing provenance fields: {missing}")
    if output.suffix != ".safetensors":
        fail("oracle output must end in .safetensors")
    if not tensors:
        fail("oracle fixture must contain at least one tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    metadata = {
        "schema": SCHEMA,
        "scope": scope,
        **{f"provenance.{key}": str(value) for key, value in provenance.items()},
    }
    write_canonical_safetensors(temporary, tensors, metadata)
    os.replace(temporary, output)
    manifest = {
        "schema": SCHEMA,
        "scope": scope,
        "fixture": {
            "path": output.name,
            "sha256": sha256_file(output),
        },
        "provenance": dict(sorted((key, str(value)) for key, value in provenance.items())),
        "execution": dict(sorted((key, str(value)) for key, value in execution.items())),
        "tensors": {
            name: {
                "dtype": {
                    torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8",
                    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
                    torch.bfloat16: "BF16", torch.float16: "F16",
                    torch.float32: "F32", torch.float64: "F64",
                }[tensor.dtype],
                "shape": list(tensor.shape),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in sorted(tensors.items())
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    return output, manifest_path


try:
    import torch
except ImportError as exc:  # pragma: no cover - actionable runtime message
    fail(f"PyTorch is required to export H3 oracle fixtures: {exc}")
