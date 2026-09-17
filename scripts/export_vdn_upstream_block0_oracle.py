#!/usr/bin/env python3
"""Export an eager real-weight OpenVDN Stage-DMD hybrid block-0 oracle."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from export_vdn_upstream_refiner_oracle import (
    git_revision,
    sha256_file,
    tensor_sha256,
    write_canonical_safetensors,
)

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)


SCHEMA = "h3-vdn-upstream-block0-oracle-v1"


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export a real Stage-DMD eager hybrid block-0 oracle"
    )
    parser.add_argument("--upstream-dir", type=Path,
                        default=workspace / "vdn-minimax-h3-upstream")
    parser.add_argument("--model-root", type=Path,
                        default=repo / "models" / "vdn-minimax-h3")
    parser.add_argument("--input-fixture", type=Path,
                        default=repo / "misc/fixtures/vdn_upstream_input_small.safetensors")
    parser.add_argument("--refiner-fixture", type=Path,
                        default=repo / "misc/fixtures/vdn_upstream_refiner_stage_dmd_example0.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sdpa-backend", choices=("math", "native"),
                        default="math")
    return parser.parse_args()


class Block0OracleModule(torch.nn.Module):
    def __init__(self, hybrid_cls) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=5376)
        self.proj_in = torch.nn.Linear(96, 5376, bias=True)
        self.audio_proj_in = torch.nn.Linear(32, 5376, bias=True)
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=5376, out_dim=2688
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)
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
        self.transformer_blocks = torch.nn.ModuleList([block])


def load_component_state(model, transformer_dir: Path, linear_path: Path):
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    expected = set(model.state_dict())
    state = {}
    base_names = {}
    linear_names = []
    for name in sorted(expected):
        if ".attn." in name and ".attn.orig." not in name:
            linear_names.append(name)
            continue
        base_name = name.replace(".attn.orig.", ".attn.")
        if base_name not in weight_map:
            fail(f"transformer index is missing {base_name}")
        base_names.setdefault(weight_map[base_name], []).append((name, base_name))
    for shard_name, names in sorted(base_names.items()):
        with safe_open(transformer_dir / shard_name,
                       framework="pt", device="cpu") as archive:
            for target_name, base_name in names:
                state[target_name] = archive.get_tensor(base_name)
    with safe_open(linear_path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        missing = sorted(set(linear_names) - keys)
        if missing:
            fail(f"linear checkpoint is missing {missing[:4]}")
        for name in linear_names:
            state[name] = archive.get_tensor(name)
    missing = sorted(expected - set(state))
    if missing:
        fail(f"component state is incomplete: {missing[:4]}")
    return state, index_path, sorted(base_names)


def load_adapter_block(path: Path, adapter_name: str, device: torch.device):
    marker = f".lora_A.{adapter_name}."
    state = {}
    with safe_open(path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        for name in sorted(keys):
            if not name.startswith("transformer_blocks.0.") or marker not in name:
                continue
            other = name.replace(".lora_A.", ".lora_B.")
            if other not in keys:
                fail(f"adapter pair is incomplete for {name}")
            state[name] = archive.get_tensor(name).to(device)
            state[other] = archive.get_tensor(other).to(device)
    return state


def capture_hooks(model: Block0OracleModule, captured):
    handles = []

    def save(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                for index, value in enumerate(output):
                    captured[f"{name}.{index}"] = value.detach().cpu().contiguous()
            else:
                captured[name] = output.detach().cpu().contiguous()
        return hook

    def save_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().cpu().contiguous()
        return hook

    block = model.transformer_blocks[0]
    hybrid = block.attn
    handles.append(block.adaln_proj.register_forward_hook(save("block_0.modulation")))
    handles.append(block.norm1.register_forward_hook(save("block_0.norm1")))
    handles.append(hybrid.register_forward_pre_hook(save_input("block_0.attention_adaln")))
    handles.append(hybrid.orig.to_q.register_forward_hook(save("block_0.q_raw")))
    handles.append(hybrid.orig.to_k.register_forward_hook(save("block_0.k_raw")))
    handles.append(hybrid.orig.to_v.register_forward_hook(save("block_0.v_raw")))
    handles.append(hybrid.orig.norm_q.register_forward_hook(save("block_0.q_norm")))
    handles.append(hybrid.orig.norm_k.register_forward_hook(save("block_0.k_norm")))
    handles.append(hybrid.softmax_gate.register_forward_hook(save("block_0.softmax_gate")))
    handles.append(hybrid.orig.to_out[0].register_forward_hook(
        save("block_0.softmax_projection")))
    handles.append(hybrid.linear_attention.output_gate.register_forward_hook(
        save("block_0.linear_output_gate")))
    handles.append(hybrid.linear_attention.register_forward_hook(
        save("block_0.linear_readout")))
    handles.append(hybrid.to_out_linear.register_forward_hook(
        save("block_0.linear_projection")))
    handles.append(hybrid.register_forward_hook(save("block_0.attention_output")))
    handles.append(block.norm2.register_forward_pre_hook(
        save_input("block_0.attention_residual")))
    handles.append(block.norm2.register_forward_hook(save("block_0.norm2")))
    handles.append(block.ff.register_forward_pre_hook(save_input("block_0.mlp_adaln")))
    handles.append(block.ff.net[0].proj.register_forward_hook(save("block_0.ff_fused")))
    handles.append(block.ff.net[0].register_forward_hook(save("block_0.ff_activated")))
    handles.append(block.ff.register_forward_hook(save("block_0.ff_output")))
    handles.append(block.register_forward_hook(save("block_0.output")))
    return handles


def main() -> None:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        fail("set HIP_VISIBLE_DEVICES=4; the oracle is restricted to physical GPU 4")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")
    for path in (args.input_fixture, args.refiner_fixture):
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

    # The eager feature body is the reference. Avoid torch.compile selecting a
    # fused inference implementation merely because the reference is on ROCm.
    feature_module._TCONV["fn"] = feature_module._temporal_shift

    with torch.device("meta"):
        model = Block0OracleModule(HybridAttention)
    checkpoint = args.model_root / "stage-dmd-step-250"
    transformer_dir = args.model_root / "h3-base/transformer"
    linear_path = checkpoint / "linear_branch/model.safetensors"
    state, index_path, base_shards = load_component_state(
        model, transformer_dir, linear_path
    )
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail(f"component load mismatch: {incompatible}")
    del state
    # These two modules own computed, non-persistent buffers and therefore have
    # no entry in load_state_dict to replace their meta storage.
    model.time_proj = Timesteps(
        num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
    )
    model.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)
    model = model.to(device).eval().requires_grad_(False)

    default_path = checkpoint / "adapters/default/adapter_model.safetensors"
    turbo_path = checkpoint / "adapters/turbo/adapter_model.safetensors"
    default_state = load_adapter_block(default_path, "default", device)
    default_pairs = merge_lora_state(model, default_state, 1.0)
    del default_state
    turbo_state = load_adapter_block(turbo_path, "turbo", device)
    turbo_pairs = merge_lora_state(model, turbo_state, 1.0)
    del turbo_state
    torch.cuda.empty_cache()

    with safe_open(args.input_fixture, framework="pt", device="cpu") as archive:
        position_ids = archive.get_tensor("layout.position_ids")
        token_tags = archive.get_tensor("layout.token_tags")
        video_indices = archive.get_tensor("layout.video_indices")
        audio_indices = archive.get_tensor("layout.audio_indices")
        text_indices = archive.get_tensor("layout.text_indices")
        video_rows = archive.get_tensor("input.video_rows")
        audio_rows = archive.get_tensor("input.audio_rows")
        timestep = archive.get_tensor("schedule.nfe_0.unique_timesteps")
        timestep_indices = archive.get_tensor("schedule.nfe_0.timestep_indices")
    with safe_open(args.refiner_fixture, framework="pt", device="cpu") as archive:
        refined_prompt = archive.get_tensor("refiner.final")

    position_ids = position_ids.to(device)
    token_tags = token_tags.to(device)
    video_indices = video_indices.to(device)
    audio_indices = audio_indices.to(device)
    text_indices = text_indices.to(device)
    timestep_indices = timestep_indices.to(device)
    block = model.transformer_blocks[0]
    block.attn.layout = layout_from_indices(
        video_indices, 17, 2, seq_len=position_ids.shape[0],
        frame_size=(1, 2), text_indices=text_indices,
    )

    captured = {
        "input.video_rows": video_rows.contiguous(),
        "input.audio_rows": audio_rows.contiguous(),
        "input.refined_prompt": refined_prompt.contiguous(),
        "input.position_ids": position_ids.cpu().contiguous(),
        "input.token_tags": token_tags.cpu().contiguous(),
        "input.timestep_indices": timestep_indices.cpu().contiguous(),
        "input.unique_timesteps": timestep.contiguous(),
    }
    handles = capture_hooks(model, captured)
    sdpa_context = (sdpa_kernel(SDPBackend.MATH)
                    if args.sdpa_backend == "math" else nullcontext())
    with torch.inference_mode(), sdpa_context:
        video_embeds = model.proj_in(video_rows.to(device).unsqueeze(0))
        audio_embeds = model.audio_proj_in(audio_rows.to(device).unsqueeze(0))
        text_embeds = refined_prompt.to(device)
        hidden = text_embeds.new_zeros(
            (1, position_ids.shape[0], text_embeds.shape[-1])
        )
        hidden.index_copy_(1, text_indices, text_embeds)
        hidden.index_copy_(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden.index_copy_(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        time_features = model.time_proj(timestep.to(device))
        temb = model.time_embedder(time_features)
        adaln_indices = timestep_indices * 3 + token_tags
        rotary = model.rope(position_ids)
        captured["input.video_projection"] = video_embeds.cpu().contiguous()
        captured["input.audio_projection"] = audio_embeds.cpu().contiguous()
        captured["input.packed_hidden"] = hidden.cpu().contiguous()
        captured["input.time_features"] = time_features.cpu().contiguous()
        captured["input.time_embedding"] = temb.cpu().contiguous()
        captured["input.adaln_indices"] = adaln_indices.cpu().contiguous()
        captured["input.rope_cos_half_bf16"] = rotary[0][:, :48].to(
            torch.bfloat16).cpu().contiguous()
        captured["input.rope_sin_half_bf16"] = rotary[1][:, :48].to(
            torch.bfloat16).cpu().contiguous()
        block(hidden, temb, adaln_indices, rotary)
        torch.cuda.synchronize(device)
    for handle in handles:
        handle.remove()

    metadata = {
        "schema": SCHEMA,
        "scope": "real-weight-stage-dmd-hybrid-block-0-eager",
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
        "base_shards": ",".join(base_shards),
        "default_pairs": str(default_pairs),
        "turbo_pairs": str(turbo_pairs),
        "input_fixture_sha256": sha256_file(args.input_fixture),
        "refiner_fixture_sha256": sha256_file(args.refiner_fixture),
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
    print(f"default_pairs={default_pairs} turbo_pairs={turbo_pairs}")
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
