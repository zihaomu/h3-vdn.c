#!/usr/bin/env python3
"""Export an official FL2VA Qwen3-VL 64x64 vision-tower oracle."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform

from h3_oracle_common import fail, sha256_file, write_fixture


MODEL_REPO = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
HEIGHT = 64
WIDTH = 64
PATCH = 16
MERGE = 2
TEMPORAL_PATCH = 2


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--task", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    return parser.parse_args()


def verify_environment(args: argparse.Namespace, component: Path) -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_gpu):
        fail(
            f"HIP_VISIBLE_DEVICES must be exactly {args.physical_gpu}; "
            "run through scripts/profile_vdn_gpu4.sh"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("the exporter requires exactly one visible ROCm device")
    index = component / "model.safetensors.index.json"
    if not index.is_file():
        fail(f"missing Text/Vision Encoder checkpoint index: {index}")
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        fail(f"invalid Text/Vision Encoder checkpoint index: {exc}")
    missing = sorted({name for name in weight_map.values() if not (component / name).is_file()})
    if missing:
        fail(f"missing Text/Vision Encoder shards: {missing}")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")


def fixed_pixels() -> torch.Tensor:
    """Return deterministic display-range F32 pixels in [0,1], [1,3,H,W]."""
    channel = torch.arange(3, dtype=torch.int64).view(1, 3, 1, 1)
    row = torch.arange(HEIGHT, dtype=torch.int64).view(1, 1, HEIGHT, 1)
    column = torch.arange(WIDTH, dtype=torch.int64).view(1, 1, 1, WIDTH)
    values = (channel * 67 + row * 13 + column * 29 + row * column * 3) % 257
    return (values.to(torch.float32) / 256.0).contiguous()


def patch_rows(pixels: torch.Tensor) -> torch.Tensor:
    """Match the released processor's spatial-merge-major patch row order."""
    frames = pixels.repeat(TEMPORAL_PATCH, 1, 1, 1)
    grid_h = HEIGHT // PATCH
    grid_w = WIDTH // PATCH
    rows = (
        frames.reshape(
            TEMPORAL_PATCH,
            3,
            grid_h // MERGE,
            MERGE,
            PATCH,
            grid_w // MERGE,
            MERGE,
            PATCH,
        )
        .permute(2, 5, 3, 6, 1, 0, 4, 7)
        .reshape(grid_h * grid_w, 3 * TEMPORAL_PATCH * PATCH * PATCH)
    )
    # MiniMax's preprocessor config uses mean/std 0.5. The native C path
    # performs this normalization immediately before BF16 patch projection.
    return (rows * 2.0 - 1.0).contiguous()


def main() -> None:
    args = parse_args()
    component = args.model_root.resolve() / args.task / "text_encoder"
    verify_environment(args, component)

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:
        fail(
            "Transformers 4.57 vision-oracle environment is required; prepend "
            f".text-oracle-deps to PYTHONPATH: {exc}"
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(component),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    visual = model.model.visual
    visual.config._attn_implementation = "eager"
    if len(visual.blocks) != 27 or visual.deepstack_visual_indexes != [8, 16, 24]:
        fail("unexpected released Qwen3-VL vision configuration")

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    pixels = fixed_pixels()
    rows = patch_rows(pixels).to(device)
    grid_thw = torch.tensor([[1, HEIGHT // PATCH, WIDTH // PATCH]], dtype=torch.long, device=device)
    visual.to(device)
    with torch.inference_mode():
        merged, deepstack = visual(rows, grid_thw)
    if len(deepstack) != 3:
        fail("official vision tower returned the wrong deepstack count")
    merged = merged.to("cpu").contiguous()
    deepstack = [item.to("cpu").contiguous() for item in deepstack]
    visual.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    expected_shape = (4, 5120)
    if tuple(merged.shape) != expected_shape or merged.dtype != torch.bfloat16:
        fail(f"official merged vision output is malformed: {merged.shape} {merged.dtype}")
    if any(tuple(item.shape) != expected_shape or item.dtype != torch.bfloat16 for item in deepstack):
        fail("official deepstack vision output is malformed")
    if not torch.isfinite(merged.float()).all() or any(
        not torch.isfinite(item.float()).all() for item in deepstack
    ):
        fail("official vision output contains non-finite values")

    tensors = {
        "media.pixels_f32": pixels,
        "vision.merged": merged,
        "vision.deepstack_0": deepstack[0],
        "vision.deepstack_1": deepstack[1],
        "vision.deepstack_2": deepstack[2],
        # Transitional aliases for existing focused tests.
        "x.pixels": pixels,
        "x.merged": merged,
        "x.deepstack_0": deepstack[0],
        "x.deepstack_1": deepstack[1],
        "x.deepstack_2": deepstack[2],
    }
    exporter = Path(__file__).resolve()
    output, manifest = write_fixture(
        args.output.resolve(),
        tensors,
        "input-conditioning",
        {
            "upstream_repo": MODEL_REPO,
            "upstream_revision": MODEL_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "task": args.task,
            "case_id": f"{args.task.lower()}-vision-64-v1",
            "seed": "not-used",
            "rng_policy": "explicit deterministic pixel tensor bytes are authoritative",
            "exporter_sha256": sha256_file(exporter),
            "model_index_sha256": sha256_file(component / "model.safetensors.index.json"),
        },
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "physical_gpu": str(args.physical_gpu),
            "pci_bdf": "0000:e3:00.0",
            "attention_backend": "Transformers eager",
            "dtype": "bfloat16",
            "image_shape": f"1x3x{HEIGHT}x{WIDTH}",
            "normalization": "pixels * 2 - 1",
        },
    )
    print(output)
    print(manifest)


if __name__ == "__main__":
    try:
        import torch
        import transformers
    except ImportError as exc:
        fail(f"PyTorch/Transformers environment is required: {exc}")
    main()
