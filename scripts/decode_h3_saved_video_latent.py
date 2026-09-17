#!/usr/bin/env python3
"""Decode a saved official MiniMax-H3 video latent with only the Video VAE loaded."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import wave

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from diffusers.models import AutoencoderKLMiniMaxH3
from diffusers.video_processor import VideoProcessor

from run_h3_artifact_diagnosis import compare_videos, write_diff_contact
from run_h3_diffusers_reference import require_single_gpu, sha256, write_video_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("MiniMax-H3"))
    parser.add_argument("--latent-bundle", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--reference-frames", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("untiled", "custom-tiled"), default="untiled")
    parser.add_argument("--tile-height", type=int, default=192)
    parser.add_argument("--tile-width", type=int, default=192)
    parser.add_argument("--overlap-height", type=int, default=64)
    parser.add_argument("--overlap-width", type=int, default=64)
    return parser.parse_args()


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(os.fspath(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        if source.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit PCM input audio, got {source.getsampwidth()} bytes")
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return (pcm.reshape(-1, channels).T.astype(np.float32) / 32767.0), rate


def load_frames(path: Path) -> np.ndarray:
    names = sorted(path.glob("frame_*.png"))
    if not names:
        raise RuntimeError(f"no reference frames in {path}")
    return np.stack([np.asarray(Image.open(name).convert("RGB"), dtype=np.float32) / 255.0 for name in names])


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    physical_gpu, bdf = require_single_gpu()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "status": "LOADING_VAE",
        "mode": args.mode,
        "physical_gpu": physical_gpu,
        "physical_gpu_bdf": bdf,
        "latent_bundle": os.fspath(args.latent_bundle),
        "latent_bundle_sha256": sha256(args.latent_bundle),
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    write_json(manifest_path, manifest)

    started = time.monotonic()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        os.fspath(args.model_dir / "vae"), local_files_only=True, dtype=torch.bfloat16
    )
    vae.requires_grad_(False)
    vae.to("cuda:0")
    processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
    dtype_counts: dict[str, int] = {}
    for parameter in vae.parameters():
        name = str(parameter.dtype)
        dtype_counts[name] = dtype_counts.get(name, 0) + parameter.numel()
    manifest["vae_load_seconds"] = time.monotonic() - started
    manifest["vae_parameter_dtype_counts"] = dtype_counts

    if args.mode == "untiled":
        vae.disable_tiling()
    else:
        vae.enable_tiling(
            tile_sample_min_height=args.tile_height,
            tile_sample_min_width=args.tile_width,
            tile_sample_min_overlap_height=args.overlap_height,
            tile_sample_min_overlap_width=args.overlap_width,
        )

    latent = load_file(os.fspath(args.latent_bundle), device="cpu")["video_latent"].to("cuda:0")
    mean = torch.tensor(vae.config.latents_mean, device="cuda:0").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda:0").view(1, -1, 1, 1, 1)
    latent = latent * std + mean
    manifest["status"] = "DECODING"
    write_json(manifest_path, manifest)
    torch.cuda.reset_peak_memory_stats()
    decode_started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            decoded = vae.decode(latent, return_dict=False)[0]
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device="cuda:0").view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device="cuda:0").view(1, -1, 1, 1, 1)
        decoded = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
        video = np.asarray(processor.postprocess_video(decoded, output_type="np")[0], dtype=np.float32)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        manifest.update(
            {
                "status": "OOM",
                "decode_seconds_before_failure": time.monotonic() - decode_started,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "error": str(exc),
            }
        )
        write_json(manifest_path, manifest)
        raise

    audio, sample_rate = load_audio(args.audio_wav)
    mp4, media = write_video_artifacts(args.output_dir, video, audio, sample_rate)
    manifest.update(
        {
            "status": "COMPLETE",
            "decode_seconds": time.monotonic() - decode_started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "mp4": os.fspath(mp4),
            "media": media,
            "tile_geometry": {
                "tile_height": args.tile_height if args.mode == "custom-tiled" else None,
                "tile_width": args.tile_width if args.mode == "custom-tiled" else None,
                "overlap_height": args.overlap_height if args.mode == "custom-tiled" else None,
                "overlap_width": args.overlap_width if args.mode == "custom-tiled" else None,
            },
        }
    )
    if args.reference_frames:
        reference = load_frames(args.reference_frames)
        manifest["reference_comparison"] = compare_videos(reference, video)
        diff_path = args.output_dir / "reference-diff-4x.png"
        write_diff_contact(diff_path, reference, video)
        manifest["reference_diff_4x"] = os.fspath(diff_path)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
