#!/usr/bin/env python3
"""Export the first OpenVDN semantic-parity oracle fixture.

This stage intentionally stops before loading the 33B transformer.  It uses the
pinned, patched upstream Diffusers implementation to export the exact prompt,
packed layout, explicit random inputs, and paired eight-NFE schedule consumed by
both the PyTorch reference and the native C/HIP implementation.

The fixture owns its random input bytes.  Consumers must read these tensors
instead of trying to reproduce a backend-specific RNG stream from the seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Dict


SCHEMA = "h3-vdn-upstream-oracle-v1"
PATCH_SIZE = (1, 2, 2)
VIDEO_CHANNELS = 24
VIDEO_PATCH = 96
AUDIO_CHANNELS = 2
AUDIO_WIDTH = 32
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def git_revision(directory: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(dtype=torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write_canonical_safetensors(path: Path, tensors: Dict[str, "torch.Tensor"],
                                metadata: Dict[str, str]) -> None:
    """Write deterministic safetensors bytes.

    safetensors validates tensor data but does not promise a stable JSON object
    order in its Rust writer.  Oracle files are release evidence, so use a
    sorted compact header and sorted payload order.  The format is deliberately
    small here: every initial-scope dtype is explicit and little-endian.
    """
    dtype_names = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int64: "I64",
    }
    header = {"__metadata__": dict(sorted(metadata.items()))}
    payloads = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.dtype not in dtype_names:
            fail(f"canonical writer does not support {name} dtype {tensor.dtype}")
        payload = tensor.view(dtype=torch.uint8).numpy().tobytes()
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


def load_prompt(path: Path):
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            fail(f"{path} is not a prompt dictionary")
        embedding = value.get("prompt_embeds")
        tags = value.get("text_token_tags")
    elif path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as archive:
            keys = set(archive.keys())
            if keys != {"prompt_embeds", "token_tags"}:
                fail(f"{path} has unexpected tensors {sorted(keys)}")
            embedding = archive.get_tensor("prompt_embeds")
            tags = archive.get_tensor("token_tags")
    else:
        fail("prompt must be an upstream .pt or native .safetensors file")
    if not isinstance(embedding, torch.Tensor) or not isinstance(tags, torch.Tensor):
        fail(f"{path} does not contain prompt tensors")
    if embedding.dtype != torch.bfloat16 or embedding.ndim != 2 or embedding.shape[1] != 5120:
        fail(f"prompt_embeds must be BF16 [L,5120], got {embedding.dtype} {tuple(embedding.shape)}")
    if tags.dtype != torch.int64 or tags.shape != (embedding.shape[0],):
        fail(f"text token tags must be I64 [L], got {tags.dtype} {tuple(tags.shape)}")
    if tags.numel() == 0 or bool(((tags < 0) | (tags > 255)).any()):
        fail("text token tags are empty or outside the native U8 contract")
    return embedding.cpu().contiguous(), tags.cpu().contiguous()


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export OpenVDN prompt/layout/input/schedule oracle tensors"
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=workspace / "vdn-minimax-h3-upstream",
        help="pinned OpenVDN checkout containing the patched diffusers checkout",
    )
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=17)
    parser.add_argument("--latent-height", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--audio-latents", type=int, default=3)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.upstream_dir / ".git").is_dir():
        fail(f"not an OpenVDN checkout: {args.upstream_dir}")
    if not (args.upstream_dir / "diffusers" / ".git").is_dir():
        fail(
            "patched Diffusers checkout is missing; run upstream "
            "scripts/setup_diffusers.sh in the oracle environment"
        )
    if not args.prompt.is_file():
        fail(f"prompt does not exist: {args.prompt}")
    if args.latent_frames < 1:
        fail("--latent-frames must be positive")
    if args.latent_height < 2 or args.latent_height % 2:
        fail("--latent-height must be a positive multiple of 2")
    if args.latent_width < 2 or args.latent_width % 2:
        fail("--latent-width must be a positive multiple of 2")
    if args.audio_latents < 0:
        fail("--audio-latents must be nonnegative")
    if args.nfe < 1 or args.nfe > 64:
        fail("--nfe must be in [1,64]")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")


def add_schedule(tensors: Dict[str, "torch.Tensor"], token_count: int,
                 audio_indices, nfe: int) -> None:
    video_scheduler = MiniMaxH3Scheduler(shift=12.0)
    audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
    video_scheduler.set_timesteps(nfe, device="cpu")
    audio_scheduler.set_timesteps(nfe, device="cpu")
    tensors["schedule.video_sigmas"] = video_scheduler.sigmas.cpu().contiguous()
    tensors["schedule.audio_sigmas"] = audio_scheduler.sigmas.cpu().contiguous()
    tensors["schedule.video_timesteps"] = video_scheduler.timesteps.cpu().contiguous()
    tensors["schedule.audio_timesteps"] = audio_scheduler.timesteps.cpu().contiguous()
    video_scales = []
    audio_scales = []
    for step, (video_t, audio_t) in enumerate(
        zip(video_scheduler.timesteps, audio_scheduler.timesteps)
    ):
        row_timesteps = torch.full((token_count,), float(video_t), dtype=torch.float32)
        row_timesteps[audio_indices] = float(audio_t)
        unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        tensors[f"schedule.nfe_{step}.unique_timesteps"] = unique.contiguous()
        tensors[f"schedule.nfe_{step}.timestep_indices"] = inverse.to(torch.int64).contiguous()
        video_ratio = video_scheduler.sigmas[step + 1] / video_scheduler.sigmas[step]
        audio_ratio = audio_scheduler.sigmas[step + 1] / audio_scheduler.sigmas[step]
        video_scales.append((1.0 - video_ratio) * (1.0 - video_t))
        audio_scales.append((1.0 - audio_ratio) * (1.0 - audio_t))
    tensors["schedule.video_euler_scales"] = torch.stack(video_scales).to(torch.float32)
    tensors["schedule.audio_euler_scales"] = torch.stack(audio_scales).to(torch.float32)


def main() -> None:
    args = parse_args()
    validate_args(args)
    embedding, text_tags = load_prompt(args.prompt)

    (position_ids, token_tags, video_indices, audio_indices, text_indices,
     num_cond_video, num_cond_audio) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        args.audio_latents,
        PATCH_SIZE,
        AUDIO_CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
        keyframe_anchors=(),
    )
    if num_cond_video or num_cond_audio:
        fail("text-to-video oracle unexpectedly contains conditioning rows")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video_latents = torch.randn(
        (1, VIDEO_CHANNELS, args.latent_frames,
         args.latent_height, args.latent_width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    video_rows = patchify_video_latents(video_latents, PATCH_SIZE).contiguous()
    audio_rows = torch.randn(
        (args.audio_latents * AUDIO_CHANNELS, AUDIO_WIDTH),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    expected_video_rows = (
        args.latent_frames * (args.latent_height // 2) * (args.latent_width // 2)
    )
    if video_rows.shape != (expected_video_rows, VIDEO_PATCH):
        fail(f"unexpected patched video shape {tuple(video_rows.shape)}")

    tensors = {
        "prompt.embeds": embedding,
        "prompt.text_token_tags": text_tags,
        "layout.position_ids": position_ids.cpu().contiguous(),
        "layout.token_tags": token_tags.cpu().contiguous(),
        "layout.video_indices": video_indices.cpu().contiguous(),
        "layout.audio_indices": audio_indices.cpu().contiguous(),
        "layout.text_indices": text_indices.cpu().contiguous(),
        "input.video_latents": video_latents.contiguous(),
        "input.video_rows": video_rows,
        "input.audio_rows": audio_rows.contiguous(),
    }
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)
    rope_cos, rope_sin = rope(position_ids)
    tensors["layout.rope_cos_half_bf16"] = rope_cos[:, :48].to(torch.bfloat16).contiguous()
    tensors["layout.rope_sin_half_bf16"] = rope_sin[:, :48].to(torch.bfloat16).contiguous()
    add_schedule(tensors, position_ids.shape[0], audio_indices, args.nfe)

    upstream_revision = git_revision(args.upstream_dir)
    diffusers_revision = git_revision(args.upstream_dir / "diffusers")
    metadata = {
        "schema": SCHEMA,
        "scope": "prompt-layout-input-schedule",
        "openvdn_commit": upstream_revision,
        "diffusers_commit": diffusers_revision,
        "torch_version": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256_file(args.prompt),
        "seed": str(args.seed),
        "rng": "torch.Generator(cpu); fixture bytes are authoritative",
        "latent_frames": str(args.latent_frames),
        "latent_height": str(args.latent_height),
        "latent_width": str(args.latent_width),
        "audio_latents": str(args.audio_latents),
        "nfe": str(args.nfe),
        "video_shift": "12.0",
        "audio_shift": "3.0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    write_canonical_safetensors(temporary, tensors, metadata)
    os.replace(temporary, args.output)

    manifest = {
        "schema": SCHEMA,
        "scope": metadata["scope"],
        "fixture": str(args.output.resolve()),
        "fixture_sha256": sha256_file(args.output),
        "provenance": metadata,
        "tensors": {
            name: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in sorted(tensors.items())
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_temp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_temp, manifest_path)
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    try:
        import torch
        import diffusers
        from safetensors import safe_open
        from diffusers import MiniMaxH3Scheduler
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
            patchify_video_latents,
        )
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3RotaryPosEmbed,
        )
    except ImportError as exc:
        fail(
            "oracle dependencies are unavailable; use the project .venv with "
            f"the pinned patched Diffusers checkout ({exc})"
        )
    main()
