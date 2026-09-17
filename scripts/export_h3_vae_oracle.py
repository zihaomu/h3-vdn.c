#!/usr/bin/env python3
"""Export an official original-H3 Audio VAE encoder/decoder oracle.

Video VAE support intentionally remains a separate next step: the first
version closes the smaller Audio VAE path with the official bundled source and
checkpoint while the full 134 GiB FL2VA snapshot cannot fit on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from h3_oracle_common import fail, sha256_file, write_fixture


MODEL_REPO = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export original MiniMax-H3 official Audio VAE oracle"
    )
    parser.add_argument("--model-root", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--task", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-length", type=int, default=37)
    parser.add_argument("--samples", type=int, default=64000)
    parser.add_argument("--physical-gpu", type=int, default=4)
    return parser.parse_args()


def verify_environment(args: argparse.Namespace, component: Path) -> None:
    if args.latent_length < 1:
        fail("--latent-length must be positive")
    if args.samples < 1 or args.samples > 32000 * 15:
        fail("--samples must be in [1,480000]")
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        fail(
            f"HIP_VISIBLE_DEVICES must be exactly {args.physical_gpu}; "
            "run through scripts/profile_vdn_gpu4.sh"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("the exporter requires exactly one visible ROCm device")
    if not (component / "model.safetensors").is_file():
        fail(f"missing Audio VAE checkpoint: {component / 'model.safetensors'}")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")


def deterministic_waveform(samples: int) -> "torch.Tensor":
    # Analytic stereo signal avoids relying on backend RNG for encoder input.
    time = torch.arange(samples, dtype=torch.float32) / 32000.0
    left = 0.35 * torch.sin(2.0 * torch.pi * 220.0 * time)
    left += 0.08 * torch.sin(2.0 * torch.pi * 880.0 * time + 0.25)
    right = 0.31 * torch.sin(2.0 * torch.pi * 330.0 * time + 0.5)
    right += 0.07 * torch.sin(2.0 * torch.pi * 1320.0 * time)
    return torch.stack((left, right)).contiguous()


def main() -> None:
    args = parse_args()
    component = args.model_root.resolve() / args.task / "audio_vae"
    verify_environment(args, component)
    sys.path.insert(0, str(component.parent))
    try:
        from audio_vae.minimax_h3_audio_vae import MiniMaxH3AudioVAE
    except ImportError as exc:
        fail(f"cannot import official Audio VAE source: {exc}")

    device = torch.device("cuda:0")
    model = MiniMaxH3AudioVAE.from_pretrained(str(component)).model
    model = model.to(device=device, dtype=torch.float32).eval()
    config = json.loads((component / "config.json").read_text(encoding="utf-8"))
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32, device=device)
    deviation = torch.tensor(config["latents_std"], dtype=torch.float32, device=device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    normalized_latent = torch.randn(
        (32, 2, args.latent_length), generator=generator, dtype=torch.float32
    )
    waveform = deterministic_waveform(args.samples)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # pragma: no cover - retained for older Torch
        from contextlib import nullcontext
        attention_context = nullcontext()

    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode(), attention_context:
        decoder_input = normalized_latent.to(device).permute(1, 0, 2)
        decoder_input = decoder_input * deviation[None, :, None] + mean[None, :, None]
        decoded = model.decode(decoder_input)[:, 0, :].float().cpu().contiguous()

        encoder_input = waveform.to(device)[:, None, :]
        encoder_input = model.preprocess(encoder_input, 32000)
        hidden = model.encoder(encoder_input)
        projected = model.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        encoded = model.mean_proj(projected)
        encoded = (encoded - mean[None, :, None]) / deviation[None, :, None]
        encoded = encoded.permute(1, 0, 2).unsqueeze(0).float().cpu().contiguous()

    expected_samples = args.latent_length * 800
    if decoded.shape != (2, expected_samples):
        fail(f"official decoder returned unexpected shape {tuple(decoded.shape)}")
    expected_length = (args.samples + 799) // 800
    if encoded.shape != (1, 32, 2, expected_length):
        fail(f"official encoder returned unexpected shape {tuple(encoded.shape)}")
    if not torch.isfinite(decoded).all() or not torch.isfinite(encoded).all():
        fail("official Audio VAE returned non-finite values")

    tensors = {
        "audio.decode.normalized_latent": normalized_latent,
        "audio.decode.pcm_f32": decoded,
        "audio.encode.pcm_f32": waveform.unsqueeze(0),
        "audio.encode.normalized_latent": encoded,
        # Legacy aliases keep the existing focused C tests consumable.
        "x.latent": normalized_latent,
        "x.waveform": decoded,
    }
    exporter = Path(__file__).resolve()
    output, manifest = write_fixture(
        args.output.resolve(), tensors, "vae-audio",
        {
            "upstream_repo": MODEL_REPO,
            "upstream_revision": MODEL_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "task": args.task,
            "case_id": f"{args.task.lower()}-audio-vae-v1",
            "seed": str(args.seed),
            "rng_policy": "explicit fixture tensor bytes are authoritative",
            "exporter_sha256": sha256_file(exporter),
            "model_weight_sha256": sha256_file(component / "model.safetensors"),
        },
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "device": torch.cuda.get_device_name(0),
            "physical_gpu": str(args.physical_gpu),
            "pci_bdf": "0000:e3:00.0",
            "attention_backend": "torch SDPA math",
            "dtype": "float32",
        },
    )
    print(output)
    print(manifest)


if __name__ == "__main__":
    try:
        import torch
    except ImportError as exc:
        fail(f"PyTorch ROCm environment is required: {exc}")
    main()
