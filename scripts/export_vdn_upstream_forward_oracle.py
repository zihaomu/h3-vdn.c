#!/usr/bin/env python3
"""Export a streamed real-weight 50-layer OpenVDN Stage-DMD oracle.

Only one transformed block is resident at a time.  This keeps the reference
below the single-card memory limit while preserving the exact upstream eager
hybrid-attention, default-LoRA, and turbo-LoRA semantics.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from export_vdn_upstream_refiner_oracle import (
    git_revision,
    sha256_file,
    tensor_sha256,
    write_canonical_safetensors,
)

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)


SCHEMA = "h3-vdn-upstream-forward-oracle-v1"


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export a streamed 50-layer Stage-DMD forward oracle"
    )
    parser.add_argument("--upstream-dir", type=Path,
                        default=workspace / "vdn-minimax-h3-upstream")
    parser.add_argument("--model-root", type=Path,
                        default=repo / "models" / "vdn-minimax-h3")
    parser.add_argument(
        "--block0-fixture", type=Path,
        default=repo / "misc/fixtures/vdn_upstream_block0_stage_dmd_small.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sdpa-backend", choices=("math", "native"),
                        default="math")
    return parser.parse_args()


class SingleBlockModule(torch.nn.Module):
    def __init__(self, hybrid_cls) -> None:
        super().__init__()
        block = MiniMaxH3TransformerBlock(
            hidden_size=5376,
            num_attention_heads=56,
            attention_head_dim=128,
            ffn_dim=14336,
            time_embed_dim=2688,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
        )
        block.attn = hybrid_cls(
            block.attn,
            hidden_size=5376,
            delta_rule="vdn_solve",
            linear_head_dim=128,
            bridge="alpha",
            a_fp32=True,
            short_conv=("k", "v"),
            enable_text_state=True,
            radius=1,
            chunk=5,
            anchor_frames="both",
            enable_softmax_gate=True,
            softmax_impl="ref",
        )
        self.block = block


class FinalHeadModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=5376, time_embed_dim=2688, eps=1e-5
        )
        self.proj_out = torch.nn.Linear(5376, 96, bias=True)
        self.audio_proj_out = torch.nn.Linear(5376, 32, bias=True)


def load_base_state(module: torch.nn.Module, source_prefix: str,
                    transformer_dir: Path, weight_map: dict[str, str]):
    expected = set(module.state_dict())
    state = {}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for target_name in sorted(expected):
        if target_name.startswith("block.attn.") and not \
                target_name.startswith("block.attn.orig."):
            continue
        suffix = target_name
        if target_name.startswith("block."):
            suffix = target_name[len("block."):]
        source_name = source_prefix + suffix
        if ".attn." in source_name and ".attn.orig." in source_name:
            source_name = source_name.replace(".attn.orig.", ".attn.")
        if source_name not in weight_map:
            fail(f"transformer index is missing {source_name}")
        by_shard.setdefault(weight_map[source_name], []).append(
            (target_name, source_name)
        )
    for shard_name, names in sorted(by_shard.items()):
        with safe_open(transformer_dir / shard_name,
                       framework="pt", device="cpu") as archive:
            for target_name, source_name in names:
                state[target_name] = archive.get_tensor(source_name)
    return state, sorted(by_shard)


def load_linear_state(module: SingleBlockModule, layer: int,
                      linear_path: Path, state: dict[str, torch.Tensor]) -> None:
    prefix = f"transformer_blocks.{layer}."
    with safe_open(linear_path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        for target_name in sorted(module.state_dict()):
            if not target_name.startswith("block.attn.") or \
                    target_name.startswith("block.attn.orig."):
                continue
            source_name = prefix + target_name[len("block."):]
            if source_name not in keys:
                fail(f"linear checkpoint is missing {source_name}")
            state[target_name] = archive.get_tensor(source_name)


def load_adapter_state(path: Path, adapter_name: str,
                       source_prefix: str, target_prefix: str,
                       device: torch.device):
    marker = f".lora_A.{adapter_name}."
    state = {}
    with safe_open(path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        for source_name in sorted(keys):
            if not source_name.startswith(source_prefix) or marker not in source_name:
                continue
            other = source_name.replace(".lora_A.", ".lora_B.")
            if other not in keys:
                fail(f"adapter pair is incomplete for {source_name}")
            target_name = target_prefix + source_name[len(source_prefix):]
            target_other = target_prefix + other[len(source_prefix):]
            state[target_name] = archive.get_tensor(source_name).to(device)
            state[target_other] = archive.get_tensor(other).to(device)
    return state


def main() -> None:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        fail("set HIP_VISIBLE_DEVICES=4; the oracle is restricted to physical GPU 4")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")
    if not args.block0_fixture.is_file():
        fail(f"missing block-0 fixture: {args.block0_fixture}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        fail(f"ROCm CUDA-compatible device is unavailable: {device}")
    props = torch.cuda.get_device_properties(device)
    if int(props.pci_bus_id) != 0xE3:
        fail(f"visible device PCI bus is {int(props.pci_bus_id):#x}, expected 0xe3")

    sys.path.insert(0, str(args.upstream_dir))
    from src.inference.lora import merge_lora_state
    from src.models.hybrid_attention import HybridAttention
    from src.models.linear_attention import features as feature_module
    from src.models.sequence_layout import layout_from_indices

    feature_module._TCONV["fn"] = feature_module._temporal_shift

    checkpoint = args.model_root / "stage-dmd-step-250"
    transformer_dir = args.model_root / "h3-base/transformer"
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    linear_path = checkpoint / "linear_branch/model.safetensors"
    default_path = checkpoint / "adapters/default/adapter_model.safetensors"
    turbo_path = checkpoint / "adapters/turbo/adapter_model.safetensors"

    names = (
        "input.packed_hidden", "input.time_embedding",
        "input.timestep_indices", "input.adaln_indices",
        "input.position_ids", "input.video_projection",
        "input.audio_projection",
    )
    with safe_open(args.block0_fixture, framework="pt", device="cpu") as archive:
        tensors = {name: archive.get_tensor(name) for name in names}
    hidden = tensors["input.packed_hidden"].to(device)
    temb = tensors["input.time_embedding"].to(device)
    timestep_indices = tensors["input.timestep_indices"].to(device)
    adaln_indices = tensors["input.adaln_indices"].to(device)
    position_ids = tensors["input.position_ids"].to(device)
    video_rows = tensors["input.video_projection"].shape[1]
    audio_rows = tensors["input.audio_projection"].shape[1]
    text_rows = hidden.shape[1] - video_rows - audio_rows
    text_indices = torch.arange(text_rows, device=device, dtype=torch.int64)
    audio_indices = torch.arange(
        text_rows, text_rows + audio_rows, device=device, dtype=torch.int64
    )
    video_indices = torch.arange(
        text_rows + audio_rows, hidden.shape[1], device=device, dtype=torch.int64
    )
    rope = MiniMaxH3RotaryPosEmbed(
        rope_freq_dim=16, rope_theta=10000.0
    ).to(device)
    rotary = rope(position_ids)
    layout = layout_from_indices(
        video_indices, 17, 2, seq_len=hidden.shape[1],
        frame_size=(1, 2), text_indices=text_indices,
    )

    captured = {"input.packed_hidden": hidden.cpu().contiguous()}
    base_shards = set()
    default_pairs = []
    turbo_pairs = []
    sdpa_context = (sdpa_kernel(SDPBackend.MATH)
                    if args.sdpa_backend == "math" else nullcontext())
    with torch.inference_mode(), sdpa_context:
        for layer in range(50):
            with torch.device("meta"):
                model = SingleBlockModule(HybridAttention)
            prefix = f"transformer_blocks.{layer}."
            state, shards = load_base_state(
                model, prefix, transformer_dir, weight_map
            )
            base_shards.update(shards)
            load_linear_state(model, layer, linear_path, state)
            incompatible = model.load_state_dict(state, strict=True, assign=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                fail(f"block {layer} load mismatch: {incompatible}")
            del state
            model = model.to(device).eval().requires_grad_(False)
            default_state = load_adapter_state(
                default_path, "default", prefix, "block.", device
            )
            default_pairs.append(merge_lora_state(model, default_state, 1.0))
            del default_state
            turbo_state = load_adapter_state(
                turbo_path, "turbo", prefix, "block.", device
            )
            turbo_pairs.append(merge_lora_state(model, turbo_state, 1.0))
            del turbo_state
            model.block.attn.layout = layout
            hidden = model.block(hidden, temb, adaln_indices, rotary)
            torch.cuda.synchronize(device)
            captured[f"block_{layer:02d}.output"] = hidden.cpu().contiguous()
            print(f"block={layer:02d} default_pairs={default_pairs[-1]} "
                  f"turbo_pairs={turbo_pairs[-1]}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        with torch.device("meta"):
            final = FinalHeadModule()
        final_state, shards = load_base_state(
            final, "", transformer_dir, weight_map
        )
        base_shards.update(shards)
        incompatible = final.load_state_dict(final_state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            fail(f"final head load mismatch: {incompatible}")
        del final_state
        final = final.to(device).eval().requires_grad_(False)
        final_default = load_adapter_state(
            default_path, "default", "norm_out.", "norm_out.", device
        )
        final_default_pairs = merge_lora_state(final, final_default, 1.0)
        del final_default
        final_turbo = load_adapter_state(
            turbo_path, "turbo", "norm_out.", "norm_out.", device
        )
        final_turbo_pairs = merge_lora_state(final, final_turbo, 1.0)
        del final_turbo
        normalized = final.norm_out(hidden, temb, timestep_indices)
        captured["final.normalized"] = normalized.cpu().contiguous()
        normalized_f32 = normalized.to(final.proj_out.weight.dtype)
        captured["final.video_velocity"] = final.proj_out(
            normalized_f32
        ).index_select(1, video_indices).cpu().contiguous()
        captured["final.audio_velocity"] = final.audio_proj_out(
            normalized_f32
        ).index_select(1, audio_indices).cpu().contiguous()
        torch.cuda.synchronize(device)

    metadata = {
        "schema": SCHEMA,
        "scope": "real-weight-stage-dmd-50-layer-forward-eager",
        "openvdn_commit": git_revision(args.upstream_dir),
        "diffusers_commit": git_revision(args.upstream_dir / "diffusers"),
        "checkpoint": "stage-dmd-step-250",
        "torch_version": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "device_name": props.name,
        "device_pci_bus": f"0x{int(props.pci_bus_id):02x}",
        "softmax_backend": "ref",
        "sdpa_backend": args.sdpa_backend,
        "hybrid_inference_mode": "false",
        "transformer_index_sha256": sha256_file(index_path),
        "linear_branch_sha256": sha256_file(linear_path),
        "default_adapter_sha256": sha256_file(default_path),
        "turbo_adapter_sha256": sha256_file(turbo_path),
        "base_shards": ",".join(sorted(base_shards)),
        "default_pairs_by_block": ",".join(map(str, default_pairs)),
        "turbo_pairs_by_block": ",".join(map(str, turbo_pairs)),
        "final_default_pairs": str(final_default_pairs),
        "final_turbo_pairs": str(final_turbo_pairs),
        "block0_fixture_sha256": sha256_file(args.block0_fixture),
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
