#!/usr/bin/env python3
"""Decode the verified eight-NFE rows with the upstream OpenVDN VAEs."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

from export_vdn_upstream_refiner_oracle import (
    git_revision,
    sha256_file,
    tensor_sha256,
    write_canonical_safetensors,
)


SCHEMA = "h3-vdn-upstream-vae-oracle-v1"
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export OpenVDN video/audio VAE outputs from final rows"
    )
    parser.add_argument("--upstream-dir", type=Path,
                        default=workspace / "vdn-minimax-h3-upstream")
    parser.add_argument("--model-root", type=Path,
                        default=repo / "models" / "vdn-minimax-h3" / "h3-base")
    parser.add_argument(
        "--denoise-fixture", type=Path,
        default=repo / "misc/fixtures/vdn_upstream_denoise8_stage_dmd_small.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        fail("set HIP_VISIBLE_DEVICES=4; the oracle is restricted to physical GPU 4")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")
    if not args.denoise_fixture.is_file():
        fail(f"missing denoise fixture: {args.denoise_fixture}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        fail(f"ROCm CUDA-compatible device is unavailable: {device}")
    props = torch.cuda.get_device_properties(device)
    if int(props.pci_bus_id) != 0xE3:
        fail(f"visible device PCI bus is {int(props.pci_bus_id):#x}, expected 0xe3")

    with safe_open(args.denoise_fixture, framework="pt", device="cpu") as archive:
        video_rows = archive.get_tensor("final.video_rows")
        audio_rows = archive.get_tensor("final.audio_rows")
    if tuple(video_rows.shape) != (34, 96):
        fail(f"unexpected video row shape {tuple(video_rows.shape)}")
    if tuple(audio_rows.shape) != (6, 32):
        fail(f"unexpected audio row shape {tuple(audio_rows.shape)}")

    rows = video_rows.reshape(1, 17, 1, 2, 24, 1, 2, 2)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    video_latent = rows.reshape(1, 24, 17, 2, 4).contiguous()
    audio_latent = audio_rows.reshape(2, 3, 32).permute(0, 2, 1).contiguous()
    captured = {
        "input.video_latent_normalized": video_latent,
        "input.audio_latent_normalized": audio_latent,
    }

    video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model_root, subfolder="vae"
    ).to(device).eval().requires_grad_(False)
    video_mean = torch.tensor(
        video_vae.config.latents_mean, device=device
    ).view(1, -1, 1, 1, 1)
    video_std = torch.tensor(
        video_vae.config.latents_std, device=device
    ).view(1, -1, 1, 1, 1)
    video_input = video_latent.to(device) * video_std + video_mean
    captured["input.video_latent_unnormalized"] = \
        video_input.cpu().contiguous()
    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16):
        video = video_vae.decode(video_input, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    video_rgb = video[0].permute(1, 2, 3, 0).cpu().contiguous()
    captured["video.rgb_f32"] = video_rgb
    torch.cuda.synchronize(device)
    print(f"video={tuple(video_rgb.shape)}", flush=True)
    del video, video_rgb, video_input, video_mean, video_std, video_vae
    gc.collect()
    torch.cuda.empty_cache()

    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        args.model_root, subfolder="audio_vae"
    ).to(device).eval().requires_grad_(False)
    audio_mean = torch.tensor(
        audio_vae.config.latents_mean, device=device
    ).view(1, -1, 1)
    audio_std = torch.tensor(
        audio_vae.config.latents_std, device=device
    ).view(1, -1, 1)
    audio_input = audio_latent.to(device) * audio_std + audio_mean
    captured["input.audio_latent_unnormalized"] = \
        audio_input.cpu().contiguous()
    with torch.inference_mode():
        audio = audio_vae.decode(audio_input, return_dict=False)[0]
    audio_pcm = audio.float().permute(1, 0, 2)[0].cpu().contiguous()
    captured["audio.pcm_f32"] = audio_pcm
    torch.cuda.synchronize(device)
    print(f"audio={tuple(audio_pcm.shape)}", flush=True)

    metadata = {
        "schema": SCHEMA,
        "scope": "verified-eight-nfe-dual-vae-closure",
        "openvdn_commit": git_revision(args.upstream_dir),
        "diffusers_commit": git_revision(args.upstream_dir / "diffusers"),
        "torch_version": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "device_name": props.name,
        "device_pci_bus": f"0x{int(props.pci_bus_id):02x}",
        "video_autocast": "float16",
        "audio_dtype": "float32",
        "denoise_fixture_sha256": sha256_file(args.denoise_fixture),
        "video_config_sha256": sha256_file(args.model_root / "vae/config.json"),
        "audio_config_sha256": sha256_file(
            args.model_root / "audio_vae/config.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    write_canonical_safetensors(temporary, captured, metadata)
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
            for name, tensor in sorted(captured.items())
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
