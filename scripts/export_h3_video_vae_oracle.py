#!/usr/bin/env python3
"""Export minimal official original-H3 Video VAE decoder/encoder oracles."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import sys
import types

from h3_oracle_common import fail, sha256_file, write_fixture


MODEL_REPO = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


class Normalize:
    """Exact tensor subset of torchvision.transforms.Normalize used upstream.

    The matching PyPI torchvision wheel cannot load with the ROCm Torch build
    in this project.  The official Video VAE imports only Normalize; this shim
    preserves that operation without loading unrelated torchvision C++ ops.
    """

    def __init__(self, mean, std, inplace: bool = False) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.inplace = inplace

    def __call__(self, tensor):
        if tensor.ndim < 3:
            raise ValueError("Normalize expects (..., C, H, W)")
        output = tensor if self.inplace else tensor.clone()
        shape = [1] * output.ndim
        shape[-3] = -1
        mean = torch.as_tensor(
            self.mean, dtype=output.dtype, device=output.device
        ).reshape(shape)
        std = torch.as_tensor(
            self.std, dtype=output.dtype, device=output.device
        ).reshape(shape)
        return output.sub_(mean).div_(std)


def install_torchvision_normalize_shim() -> None:
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    torchvision.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision", loader=None
    )
    transforms.__spec__ = importlib.machinery.ModuleSpec(
        "torchvision.transforms", loader=None
    )
    transforms.Normalize = Normalize
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export original MiniMax-H3 official Video VAE oracle"
    )
    parser.add_argument("--model-root", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--task", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def verify_environment(args: argparse.Namespace, component: Path) -> None:
    if args.image_size < 32 or args.image_size % 16:
        fail("--image-size must be a multiple of 16 and at least 32")
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        fail(
            f"HIP_VISIBLE_DEVICES must be exactly {args.physical_gpu}; "
            "run through scripts/profile_vdn_gpu4.sh"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("the exporter requires exactly one visible ROCm device")
    weights = component / "source" / "model.safetensors"
    if not weights.is_file():
        fail(f"missing Video VAE checkpoint: {weights}")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")


def deterministic_image(size: int) -> "torch.Tensor":
    y = torch.arange(size, dtype=torch.float32)[:, None]
    x = torch.arange(size, dtype=torch.float32)[None, :]
    red = (x / max(size - 1, 1)).expand(size, size)
    green = (y / max(size - 1, 1)).expand(size, size)
    blue = 0.5 + 0.25 * torch.sin(x / 11.0) * torch.cos(y / 17.0)
    return torch.stack((red, green, blue)).unsqueeze(0).contiguous()


def main() -> None:
    args = parse_args()
    component = args.model_root.resolve() / args.task / "video_vae"
    verify_environment(args, component)
    install_torchvision_normalize_shim()
    sys.path.insert(0, str(component.parent))
    try:
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
        from video_vae.vae_module import DiagonalGaussianDistribution
    except ImportError as exc:
        fail(f"cannot import official Video VAE source: {exc}")

    device = torch.device("cuda:0")
    model = MiniMaxH3VideoVAE.from_pretrained(str(component)).model
    model = model.to(device=device, dtype=torch.float32).eval()
    config = json.loads((component / "config.json").read_text(encoding="utf-8"))
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32, device=device)
    deviation = torch.tensor(config["latents_std"], dtype=torch.float32, device=device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    normalized_latent = torch.randn(
        (1, 24, 2, 2, 2), generator=generator, dtype=torch.float32
    )
    # The existing focused C test consumes BF16 latent bytes.  Round before
    # both upstream and native execution so the comparison owns one input.
    normalized_latent = normalized_latent.to(torch.bfloat16).to(torch.float32)
    image = deterministic_image(args.image_size)

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
        decoder_input = normalized_latent.to(device)
        decoder_input = decoder_input * deviation[None, :, None, None, None]
        decoder_input = decoder_input + mean[None, :, None, None, None]
        # The native diagnostic path accepts two latent tokens, repeats the
        # final token to the decoder's seven-token first chunk, then exposes
        # decoded frames 3..7.  Calling decode_base() directly with T=2 is not
        # equivalent: the upstream temporal scheduler subtracts token_drop=3
        # and produces no chunk.  Drive the official low-level decoder with
        # the exact native padded tensor and select the same output indices.
        decoder_input = torch.cat(
            (
                decoder_input,
                decoder_input[:, :, -1:, :, :].repeat(1, 1, 5, 1, 1),
            ),
            dim=2,
        )
        decoded_full = model.decode(decoder_input)
        if decoded_full.shape[2] < 8:
            fail(
                "official low-level decoder returned fewer than eight "
                f"frames: {tuple(decoded_full.shape)}"
            )
        decoded = decoded_full[:, :, 3:8, :, :]
        decoded = model.processor.revert_tensor(decoded)
        decoded = decoded[0].permute(1, 2, 3, 0).float().cpu().contiguous()

        image_device = image.to(device)
        image_normalized = model.processor.transform(image_device).unsqueeze(2)
        moments = model._adaptive_encode(image_normalized)
        encoded = DiagonalGaussianDistribution(moments).mean
        encoded = model.trim_code(encoded, 1)
        encoded = (encoded - mean[None, :, None, None, None])
        encoded = encoded / deviation[None, :, None, None, None]
        encoded = encoded.float().cpu().contiguous()

    expected_rgb = (5, 32, 32, 3)
    if tuple(decoded.shape) != expected_rgb:
        fail(f"official decoder returned {tuple(decoded.shape)}, expected {expected_rgb}")
    expected_latent = (1, 24, 1, args.image_size // 16, args.image_size // 16)
    if tuple(encoded.shape) != expected_latent:
        fail(f"official encoder returned {tuple(encoded.shape)}, expected {expected_latent}")
    if not torch.isfinite(decoded).all() or not torch.isfinite(encoded).all():
        fail("official Video VAE returned non-finite values")

    legacy_frames = decoded.permute(3, 0, 1, 2).contiguous()
    tensors = {
        "video.decode.normalized_latent": normalized_latent,
        "video.decode.rgb_f32": decoded,
        "video.encode.rgb_f32": image,
        "video.encode.normalized_latent": encoded,
        "x.video_after_20": normalized_latent.to(torch.bfloat16).squeeze(0),
        "x.frames": legacy_frames,
        "x.pixels": image,
        "x.latent": encoded,
    }
    exporter = Path(__file__).resolve()
    output, manifest = write_fixture(
        args.output.resolve(), tensors, "vae-video",
        {
            "upstream_repo": MODEL_REPO,
            "upstream_revision": MODEL_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "task": args.task,
            "case_id": f"{args.task.lower()}-video-vae-tiny-v1",
            "seed": str(args.seed),
            "rng_policy": "explicit fixture tensor bytes are authoritative",
            "exporter_sha256": sha256_file(exporter),
            "model_weight_sha256": sha256_file(
                component / "source" / "model.safetensors"
            ),
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
            "torchvision_normalize": "local exact tensor shim; no C++ ops",
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
