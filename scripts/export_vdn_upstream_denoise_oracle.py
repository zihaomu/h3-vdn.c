#!/usr/bin/env python3
"""Export an eight-NFE streamed OpenVDN Stage-DMD latent oracle."""

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

from export_vdn_upstream_forward_oracle import (
    FinalHeadModule,
    SingleBlockModule,
    load_adapter_state,
    load_base_state,
    load_linear_state,
)
from export_vdn_upstream_refiner_oracle import (
    git_revision,
    sha256_file,
    tensor_sha256,
    write_canonical_safetensors,
)

from diffusers import MiniMaxH3Scheduler
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed,
)


SCHEMA = "h3-vdn-upstream-denoise-oracle-v1"


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export a streamed eight-NFE Stage-DMD latent oracle"
    )
    parser.add_argument("--upstream-dir", type=Path,
                        default=workspace / "vdn-minimax-h3-upstream")
    parser.add_argument("--model-root", type=Path,
                        default=repo / "models" / "vdn-minimax-h3")
    parser.add_argument(
        "--input-fixture", type=Path,
        default=repo / "misc/fixtures/vdn_upstream_input_small.safetensors",
    )
    parser.add_argument(
        "--block0-fixture", type=Path,
        default=repo / "misc/fixtures/vdn_upstream_block0_stage_dmd_small.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sdpa-backend", choices=("math", "native"),
                        default="math")
    return parser.parse_args()


class InputModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Linear(96, 5376, bias=True)
        self.audio_proj_in = torch.nn.Linear(32, 5376, bias=True)
        self.time_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=5376, out_dim=2688
        )


def prepare_module(module: torch.nn.Module, prefix: str,
                   transformer_dir: Path, weight_map: dict[str, str],
                   device: torch.device):
    state, shards = load_base_state(
        module, prefix, transformer_dir, weight_map
    )
    incompatible = module.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail(f"component load mismatch: {incompatible}")
    del state
    return module.to(device).eval().requires_grad_(False), shards


def main() -> None:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        fail("set HIP_VISIBLE_DEVICES=4; the oracle is restricted to physical GPU 4")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")
    for path in (args.input_fixture, args.block0_fixture):
        if not path.is_file():
            fail(f"missing oracle input: {path}")
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

    schedule_names = [
        "layout.position_ids", "layout.token_tags", "layout.video_indices",
        "layout.audio_indices", "layout.text_indices", "input.video_rows",
        "input.audio_rows", "schedule.video_timesteps",
        "schedule.audio_timesteps",
    ]
    for step in range(8):
        schedule_names.extend((
            f"schedule.nfe_{step}.unique_timesteps",
            f"schedule.nfe_{step}.timestep_indices",
        ))
    with safe_open(args.input_fixture, framework="pt", device="cpu") as archive:
        schedule = {name: archive.get_tensor(name) for name in schedule_names}
    with safe_open(args.block0_fixture, framework="pt", device="cpu") as archive:
        refined_prompt = archive.get_tensor("input.refined_prompt")

    position_ids = schedule["layout.position_ids"].to(device)
    token_tags = schedule["layout.token_tags"].to(device)
    video_indices = schedule["layout.video_indices"].to(device)
    audio_indices = schedule["layout.audio_indices"].to(device)
    text_indices = schedule["layout.text_indices"].to(device)
    video_rows = schedule["input.video_rows"].to(device)
    audio_rows = schedule["input.audio_rows"].to(device)
    refined_prompt = refined_prompt.to(device)
    rope = MiniMaxH3RotaryPosEmbed(
        rope_freq_dim=16, rope_theta=10000.0
    ).to(device)
    rotary = rope(position_ids)
    layout = layout_from_indices(
        video_indices, 17, 2, seq_len=position_ids.shape[0],
        frame_size=(1, 2), text_indices=text_indices,
    )
    time_proj = Timesteps(
        num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
    ).to(device)

    with torch.device("meta"):
        input_module = InputModule()
        final = FinalHeadModule()
    input_module, input_shards = prepare_module(
        input_module, "", transformer_dir, weight_map, device
    )
    final, final_shards = prepare_module(
        final, "", transformer_dir, weight_map, device
    )
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

    video_scheduler = MiniMaxH3Scheduler(shift=12.0)
    audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
    video_scheduler.set_timesteps(8, device=device)
    audio_scheduler.set_timesteps(8, device=device)
    captured = {
        "input.video_rows": video_rows.cpu().contiguous(),
        "input.audio_rows": audio_rows.cpu().contiguous(),
    }
    base_shards = set(input_shards + final_shards)
    default_pairs = []
    turbo_pairs = []
    sdpa_context = (sdpa_kernel(SDPBackend.MATH)
                    if args.sdpa_backend == "math" else nullcontext())
    with torch.inference_mode(), sdpa_context:
        for step in range(8):
            unique_timesteps = schedule[
                f"schedule.nfe_{step}.unique_timesteps"
            ].to(device)
            timestep_indices = schedule[
                f"schedule.nfe_{step}.timestep_indices"
            ].to(device)
            video_embeds = input_module.proj_in(video_rows.unsqueeze(0))
            audio_embeds = input_module.audio_proj_in(audio_rows.unsqueeze(0))
            hidden = refined_prompt.new_zeros(
                (1, position_ids.shape[0], refined_prompt.shape[-1])
            )
            hidden.index_copy_(1, text_indices, refined_prompt)
            hidden.index_copy_(1, video_indices,
                               video_embeds.to(refined_prompt.dtype))
            hidden.index_copy_(1, audio_indices,
                               audio_embeds.to(refined_prompt.dtype))
            temb = input_module.time_embedder(time_proj(unique_timesteps))
            adaln_indices = timestep_indices * 3 + token_tags

            for layer in range(50):
                with torch.device("meta"):
                    model = SingleBlockModule(HybridAttention)
                prefix = f"transformer_blocks.{layer}."
                state, shards = load_base_state(
                    model, prefix, transformer_dir, weight_map
                )
                base_shards.update(shards)
                load_linear_state(model, layer, linear_path, state)
                incompatible = model.load_state_dict(
                    state, strict=True, assign=True
                )
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    fail(f"block {layer} load mismatch: {incompatible}")
                del state
                model = model.to(device).eval().requires_grad_(False)
                default_state = load_adapter_state(
                    default_path, "default", prefix, "block.", device
                )
                default_count = merge_lora_state(
                    model, default_state, 1.0
                )
                del default_state
                turbo_state = load_adapter_state(
                    turbo_path, "turbo", prefix, "block.", device
                )
                turbo_count = merge_lora_state(model, turbo_state, 1.0)
                del turbo_state
                if step == 0:
                    default_pairs.append(default_count)
                    turbo_pairs.append(turbo_count)
                elif default_count != default_pairs[layer] or \
                        turbo_count != turbo_pairs[layer]:
                    fail(f"adapter pair count changed at NFE {step} block {layer}")
                model.block.attn.layout = layout
                hidden = model.block(hidden, temb, adaln_indices, rotary)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                if layer == 0 or (layer + 1) % 10 == 0:
                    print(f"nfe={step} block={layer:02d}", flush=True)

            normalized = final.norm_out(hidden, temb, timestep_indices)
            normalized_f32 = normalized.to(final.proj_out.weight.dtype)
            video_velocity = final.proj_out(normalized_f32).index_select(
                1, video_indices
            )
            audio_velocity = final.audio_proj_out(normalized_f32).index_select(
                1, audio_indices
            )
            captured[f"nfe_{step}.video_velocity"] = \
                video_velocity.cpu().contiguous()
            captured[f"nfe_{step}.audio_velocity"] = \
                audio_velocity.cpu().contiguous()
            video_rows = video_scheduler.step(
                video_velocity.squeeze(0),
                schedule["schedule.video_timesteps"][step].to(device),
                video_rows,
            ).prev_sample
            audio_rows = audio_scheduler.step(
                audio_velocity.squeeze(0),
                schedule["schedule.audio_timesteps"][step].to(device),
                audio_rows,
            ).prev_sample
            torch.cuda.synchronize(device)
            captured[f"nfe_{step}.video_rows"] = video_rows.cpu().contiguous()
            captured[f"nfe_{step}.audio_rows"] = audio_rows.cpu().contiguous()
            print(f"nfe={step} complete", flush=True)

    captured["final.video_rows"] = video_rows.cpu().contiguous()
    captured["final.audio_rows"] = audio_rows.cpu().contiguous()
    metadata = {
        "schema": SCHEMA,
        "scope": "real-weight-stage-dmd-eight-nfe-latent",
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
        "nfe": "8",
        "video_shift": "12.0",
        "audio_shift": "3.0",
        "transformer_index_sha256": sha256_file(index_path),
        "linear_branch_sha256": sha256_file(linear_path),
        "default_adapter_sha256": sha256_file(default_path),
        "turbo_adapter_sha256": sha256_file(turbo_path),
        "base_shards": ",".join(sorted(base_shards)),
        "default_pairs_by_block": ",".join(map(str, default_pairs)),
        "turbo_pairs_by_block": ",".join(map(str, turbo_pairs)),
        "final_default_pairs": str(final_default_pairs),
        "final_turbo_pairs": str(final_turbo_pairs),
        "input_fixture_sha256": sha256_file(args.input_fixture),
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
