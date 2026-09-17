#!/usr/bin/env python3
"""Export an official FL2VA image-presentation plus 50-layer Qwen oracle."""

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
DIFFUSERS_REVISION = "ae2e4c7907ccc21e52ebe86b349d67e9b0b9f316"
PROMPT = "A red fox walking through snow"
PREFIX = "<Picture 1>: "
HEIGHT = 64
WIDTH = 64
PATCH = 16
MERGE = 2
LAYERS = 50
VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--task", choices=("FL2VA",), default="FL2VA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument(
        "--cpu-diagnostic",
        action="store_true",
        help="run without a GPU; output is diagnostic rather than release evidence",
    )
    parser.add_argument(
        "--zero-deepstack",
        action="store_true",
        help="diagnostic: zero all deepstack residuals on both sides",
    )
    parser.add_argument(
        "--zero-vision",
        action="store_true",
        help="diagnostic: zero merged vision and all deepstack tensors",
    )
    return parser.parse_args()


def verify_environment(args: argparse.Namespace, component: Path) -> None:
    if args.cpu_diagnostic:
        if os.environ.get("HIP_VISIBLE_DEVICES", "") not in ("", "-1"):
            fail("CPU diagnostic requires HIP_VISIBLE_DEVICES to be empty or -1")
    elif os.environ.get("HIP_VISIBLE_DEVICES") != str(args.physical_gpu):
        fail(
            f"HIP_VISIBLE_DEVICES must be exactly {args.physical_gpu}; "
            "run through scripts/profile_vdn_gpu4.sh"
        )
    if not args.cpu_diagnostic and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
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
    channel = torch.arange(3, dtype=torch.int64).view(1, 3, 1, 1)
    row = torch.arange(HEIGHT, dtype=torch.int64).view(1, 1, HEIGHT, 1)
    column = torch.arange(WIDTH, dtype=torch.int64).view(1, 1, 1, WIDTH)
    values = (channel * 67 + row * 13 + column * 29 + row * column * 3) % 257
    return (values.to(torch.float32) / 256.0).contiguous()


def patch_rows(pixels: torch.Tensor) -> torch.Tensor:
    frames = pixels.repeat(2, 1, 1, 1)
    grid_h = HEIGHT // PATCH
    grid_w = WIDTH // PATCH
    rows = (
        frames.reshape(2, 3, grid_h // MERGE, MERGE, PATCH,
                       grid_w // MERGE, MERGE, PATCH)
        .permute(2, 5, 3, 6, 1, 0, 4, 7)
        .reshape(grid_h * grid_w, 3 * 2 * PATCH * PATCH)
    )
    return (rows * 2.0 - 1.0).contiguous()


def build_presentation(tokenizer: object, visual_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    ids = prefix_ids + [VISION_START] + [IMAGE_PAD] * visual_tokens + [VISION_END] + prompt_ids
    start = len(prefix_ids) + 1
    end = start + visual_tokens
    count = len(ids)
    positions = torch.zeros((3, count), dtype=torch.int64)
    for axis in range(3):
        positions[axis, :start] = torch.arange(start, dtype=torch.int64)
    merged_h = HEIGHT // (PATCH * MERGE)
    merged_w = WIDTH // (PATCH * MERGE)
    base = start
    next_position = start + max(merged_h, merged_w)
    positions[0, start:end] = base
    cursor = start
    for row in range(merged_h):
        for column in range(merged_w):
            positions[1, cursor] = base + row
            positions[2, cursor] = base + column
            cursor += 1
    for axis in range(3):
        positions[axis, end:] = next_position + torch.arange(count - end, dtype=torch.int64)
    tags = torch.ones(count, dtype=torch.int32)
    tags[start - 1 : end + 1] = 0
    return torch.tensor(ids, dtype=torch.int32), positions.to(torch.int32), tags, start


def main() -> None:
    args = parse_args()
    if args.layers < 1 or args.layers > LAYERS:
        fail(f"--layers must be in [1,{LAYERS}]")
    component = args.model_root.resolve() / args.task / "text_encoder"
    verify_environment(args, component)
    try:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError as exc:
        fail(
            "Transformers 4.57 multimodal-oracle environment is required; "
            f"prepend .text-oracle-deps to PYTHONPATH: {exc}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(component), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(component), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True
    ).eval()
    language = model.model.language_model
    visual = model.model.visual
    language.config._attn_implementation = "eager"
    visual.config._attn_implementation = "eager"
    if len(language.layers) < args.layers:
        fail(f"checkpoint exposes only {len(language.layers)} language layers")

    device = torch.device("cpu" if args.cpu_diagnostic else "cuda:0")
    if not args.cpu_diagnostic:
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    pixels = fixed_pixels()
    rows = patch_rows(pixels).to(device)
    grid_thw = torch.tensor([[1, HEIGHT // PATCH, WIDTH // PATCH]], dtype=torch.long, device=device)
    visual.to(device)
    snapshots: dict[int, torch.Tensor] = {}
    attention_diagnostics: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        merged, deepstack = visual(rows, grid_thw)
    visual.to("cpu")
    if not args.cpu_diagnostic:
        torch.cuda.empty_cache()
    if len(deepstack) != 3:
        fail("official vision tower returned the wrong deepstack count")
    if args.zero_vision:
        merged = torch.zeros_like(merged)
        deepstack = [torch.zeros_like(item) for item in deepstack]
    elif args.zero_deepstack:
        deepstack = [torch.zeros_like(item) for item in deepstack]

    ids_i32, positions_i32, tags_i32, visual_start = build_presentation(tokenizer, merged.shape[0])
    ids = ids_i32.to(torch.long).unsqueeze(0).to(device)
    positions = positions_i32.to(torch.long).unsqueeze(1).to(device)
    # The released Diffusers conditioner always passes an explicit all-ones
    # attention mask.  This is semantically important for multimodal mRoPE:
    # without it, Transformers interprets repeated temporal position IDs in a
    # vision block as packed-sequence boundaries instead of one causal stream.
    attention_mask_2d = torch.ones_like(ids)
    cache_position = torch.arange(ids.shape[1], device=device)
    with torch.inference_mode():
        embedding = language.embed_tokens.to(device)
        hidden = embedding(ids)
        embedding.to("cpu")
        hidden[0, visual_start : visual_start + merged.shape[0]] = merged
        if not args.cpu_diagnostic:
            torch.cuda.empty_cache()
        attention_mask = create_causal_mask(
            config=language.config,
            input_embeds=hidden,
            attention_mask=attention_mask_2d,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=positions[0],
        )
        rotary = language.rotary_emb.to(device)
        position_embeddings = rotary(hidden, positions)
        rotary.to("cpu")
        if not args.cpu_diagnostic:
            torch.cuda.empty_cache()
        for layer_index in range(args.layers):
            layer = language.layers[layer_index].to(device)
            if layer_index == 0:
                attention = layer.self_attn
                normalized = layer.input_layernorm(hidden)
                hidden_shape = (
                    *normalized.shape[:-1],
                    -1,
                    attention.head_dim,
                )
                query = attention.q_norm(
                    attention.q_proj(normalized).view(hidden_shape)
                ).transpose(1, 2)
                key = attention.k_norm(
                    attention.k_proj(normalized).view(hidden_shape)
                ).transpose(1, 2)
                value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
                query, key = apply_rotary_pos_emb(
                    query, key, *position_embeddings
                )
                attention_heads, _ = eager_attention_forward(
                    attention,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling=attention.scaling,
                    dropout=0.0,
                    is_causal=False,
                )
                attention_diagnostics = {
                    "diagnostic.text.layer_01.rope_q": query[0]
                    .transpose(0, 1)
                    .to("cpu")
                    .contiguous(),
                    "diagnostic.text.layer_01.rope_k": key[0]
                    .transpose(0, 1)
                    .to("cpu")
                    .contiguous(),
                    "diagnostic.text.layer_01.value": value[0]
                    .transpose(0, 1)
                    .to("cpu")
                    .contiguous(),
                    "diagnostic.text.layer_01.attention_heads": attention_heads[0]
                    .to("cpu")
                    .contiguous(),
                }
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=positions[0],
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if layer_index < len(deepstack):
                hidden[0, visual_start : visual_start + merged.shape[0]] += deepstack[layer_index]
            snapshots[layer_index + 1] = hidden[0].to("cpu").contiguous()
            layer.to("cpu")
            if not args.cpu_diagnostic:
                torch.cuda.empty_cache()
            print(
                f"official Qwen multimodal oracle: {layer_index + 1}/{args.layers}",
                flush=True,
            )

    output = snapshots[args.layers]
    merged = merged.to("cpu").contiguous()
    deepstack = [item.to("cpu").contiguous() for item in deepstack]
    del model
    gc.collect()
    if not args.cpu_diagnostic:
        torch.cuda.empty_cache()
    if output.dtype != torch.bfloat16 or not torch.isfinite(output.float()).all():
        fail("official multimodal output is invalid")

    tensors = {
        "input.prompt_utf8": torch.tensor(list(PROMPT.encode("utf-8")), dtype=torch.uint8),
        "input.token_ids": ids_i32.unsqueeze(0),
        "input.attention_mask": torch.ones_like(ids_i32).unsqueeze(0),
        "layout.position_ids": positions_i32,
        "layout.tags": tags_i32,
        "media.pixels_f32": pixels,
        "vision.merged": merged,
        "vision.deepstack_0": deepstack[0],
        "vision.deepstack_1": deepstack[1],
        "vision.deepstack_2": deepstack[2],
        f"text.layer_{args.layers:02d}.output": output,
        # Transitional aliases for the existing focused multimodal test.
        "x.ids": ids_i32.unsqueeze(0),
        "x.position_ids": positions_i32,
        "x.tags": tags_i32,
        "x.vision_merged": merged,
        "x.vision_deepstack_0": deepstack[0],
        "x.vision_deepstack_1": deepstack[1],
        "x.vision_deepstack_2": deepstack[2],
        f"x.layer_{args.layers - 1}": output,
    }
    for layer, snapshot in snapshots.items():
        tensors[f"text.layer_{layer:02d}.output"] = snapshot
        tensors[f"x.layer_{layer - 1}"] = snapshot
    tensors.update(attention_diagnostics)
    exporter = Path(__file__).resolve()
    output_path, manifest = write_fixture(
        args.output.resolve(), tensors, "input-conditioning",
        {
            "upstream_repo": MODEL_REPO,
            "upstream_revision": MODEL_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "diffusers_revision": DIFFUSERS_REVISION,
            "task": args.task,
            "case_id": (
                f"fl2va-picture-1-vision-64-layer-{args.layers}-v2"
                + ("-cpu-diagnostic" if args.cpu_diagnostic else "")
                + ("-zero-deepstack" if args.zero_deepstack else "")
                + ("-zero-vision" if args.zero_vision else "")
            ),
            "seed": "not-used",
            "rng_policy": "explicit pixel, token, layout, and tensor bytes are authoritative",
            "exporter_sha256": sha256_file(exporter),
            "model_index_sha256": sha256_file(component / "model.safetensors.index.json"),
        },
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "transformers": transformers.__version__,
            "device": "cpu" if args.cpu_diagnostic else torch.cuda.get_device_name(0),
            "physical_gpu": "not-used" if args.cpu_diagnostic else str(args.physical_gpu),
            "pci_bdf": "not-used" if args.cpu_diagnostic else "0000:e3:00.0",
            "attention_backend": "Transformers eager",
            "attention_mask": "explicit-all-ones-as-released-diffusers",
            "dtype": "bfloat16",
            "layer_count": str(args.layers),
            "presentation": PREFIX + "<vision>" + PROMPT,
            "evidence_class": (
                "diagnostic"
                if args.cpu_diagnostic or args.zero_deepstack or args.zero_vision
                else "release-candidate"
            ),
            "deepstack": (
                "zeroed-diagnostic"
                if args.zero_deepstack or args.zero_vision
                else "official"
            ),
            "vision": "zeroed-diagnostic" if args.zero_vision else "official",
        },
    )
    print(output_path)
    print(manifest)


if __name__ == "__main__":
    try:
        import torch
        import transformers
    except ImportError as exc:
        fail(f"PyTorch/Transformers environment is required: {exc}")
    main()
