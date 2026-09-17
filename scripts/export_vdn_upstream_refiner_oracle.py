#!/usr/bin/env python3
"""Export a real-weight OpenVDN prompt-refiner oracle on one ROCm device.

Only the context projection and two token-refiner blocks are materialized.  This
keeps the reference below the single-card memory limit while executing the exact
patched Diffusers modules and the released Stage-DMD default+turbo LoRA merge.
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

try:
    import torch
    from safetensors import safe_open
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TokenRefiner,
    )
except ImportError as exc:
    raise SystemExit(f"error: oracle dependencies are unavailable ({exc})")


SCHEMA = "h3-vdn-upstream-refiner-oracle-v1"


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(directory: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def tensor_sha256(tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write_canonical_safetensors(path: Path, tensors: Dict[str, "torch.Tensor"],
                                metadata: Dict[str, str]) -> None:
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
        payload = tensor.view(torch.uint8).numpy().tobytes()
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


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser(
        description="Export the real Stage-DMD MiniMax-H3 prompt-refiner oracle"
    )
    parser.add_argument(
        "--upstream-dir", type=Path,
        default=workspace / "vdn-minimax-h3-upstream",
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=repo / "models" / "vdn-minimax-h3",
    )
    parser.add_argument(
        "--prompt", type=Path,
        default=repo / "models" / "vdn-minimax-h3" / "prompts" /
        "example_0.safetensors",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="stage-dmd-step-250",
        choices=("stage-b-step-2000", "stage-dmd-step-250"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-backend", default="_native_math",
        choices=("_native_math", "native"),
    )
    return parser.parse_args()


class RefinerOracleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_embedder = torch.nn.Linear(5120, 5376, bias=True)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=5376,
            num_attention_heads=56,
            attention_head_dim=128,
            ffn_dim=14336,
            num_layers=2,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        )


def load_base_state(model: "torch.nn.Module", transformer_dir: Path):
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        fail(f"missing transformer index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    wanted = set(model.state_dict())
    missing = sorted(wanted - set(weight_map))
    if missing:
        fail(f"base index is missing refiner tensors: {missing[:4]}")
    by_shard: Dict[str, list[str]] = {}
    for name in wanted:
        by_shard.setdefault(weight_map[name], []).append(name)
    state = {}
    for shard_name, names in sorted(by_shard.items()):
        shard_path = transformer_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as archive:
            for name in sorted(names):
                state[name] = archive.get_tensor(name)
    return state, index_path, sorted(by_shard)


def load_adapter_refiner(path: Path, adapter_name: str, device: str):
    state = {}
    marker = f".lora_A.{adapter_name}."
    with safe_open(path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        for name in sorted(keys):
            if not name.startswith("token_refiner.") or marker not in name:
                continue
            other = name.replace(".lora_A.", ".lora_B.")
            if other not in keys:
                fail(f"adapter pair is incomplete for {name}")
            state[name] = archive.get_tensor(name).to(device)
            state[other] = archive.get_tensor(other).to(device)
    return state


def register_captures(model: RefinerOracleModule,
                      captured: Dict[str, "torch.Tensor"]):
    handles = []

    def save(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                fail(f"capture {name} unexpectedly returned a tuple")
            captured[name] = output.detach().cpu().contiguous()
        return hook

    def save_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().cpu().contiguous()
        return hook

    handles.append(model.context_embedder.register_forward_hook(
        save("refiner.context_projection")))
    block0 = model.token_refiner.refiner_blocks[0]
    handles.append(block0.norm1.register_forward_hook(save("refiner.block_0.norm1")))
    handles.append(block0.attn.to_q.register_forward_hook(save("refiner.block_0.q_raw")))
    handles.append(block0.attn.to_k.register_forward_hook(save("refiner.block_0.k_raw")))
    handles.append(block0.attn.to_v.register_forward_hook(save("refiner.block_0.v_raw")))
    handles.append(block0.attn.norm_q.register_forward_hook(save("refiner.block_0.q_norm")))
    handles.append(block0.attn.norm_k.register_forward_hook(save("refiner.block_0.k_norm")))
    handles.append(block0.attn.register_forward_hook(save("refiner.block_0.attention_output")))
    handles.append(block0.norm2.register_forward_pre_hook(
        save_input("refiner.block_0.attention_residual")))
    handles.append(block0.norm2.register_forward_hook(save("refiner.block_0.norm2")))
    handles.append(block0.ff.net[0].proj.register_forward_hook(
        save("refiner.block_0.ff_fused")))
    handles.append(block0.ff.net[0].register_forward_hook(
        save("refiner.block_0.ff_activated")))
    handles.append(block0.ff.register_forward_hook(save("refiner.block_0.ff_output")))
    handles.append(block0.register_forward_hook(save("refiner.block_0.output")))
    handles.append(model.token_refiner.refiner_blocks[1].register_forward_hook(
        save("refiner.block_1.output")))
    handles.append(model.token_refiner.final_norm.register_forward_hook(
        save("refiner.final")))
    return handles


def main() -> None:
    args = parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") != "4":
        fail("set HIP_VISIBLE_DEVICES=4; the oracle is restricted to physical GPU 4")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")
    if not args.prompt.is_file():
        fail(f"missing prompt: {args.prompt}")
    if not (args.upstream_dir / ".git").is_dir():
        fail(f"not an OpenVDN checkout: {args.upstream_dir}")
    checkpoint_dir = args.model_root / args.checkpoint
    transformer_dir = args.model_root / "h3-base" / "transformer"
    default_adapter = checkpoint_dir / "adapters/default/adapter_model.safetensors"
    turbo_adapter = checkpoint_dir / "adapters/turbo/adapter_model.safetensors"
    if not default_adapter.is_file():
        fail(f"missing default adapter: {default_adapter}")
    use_turbo = args.checkpoint == "stage-dmd-step-250"
    if use_turbo and not turbo_adapter.is_file():
        fail(f"missing turbo adapter: {turbo_adapter}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        fail(f"ROCm CUDA-compatible device is unavailable: {device}")
    props = torch.cuda.get_device_properties(device)
    if int(props.pci_bus_id) != 0xE3:
        fail(f"visible device PCI bus is {int(props.pci_bus_id):#x}, expected 0xe3")

    with safe_open(args.prompt, framework="pt", device="cpu") as archive:
        prompt = archive.get_tensor("prompt_embeds")
    if prompt.dtype != torch.bfloat16 or prompt.ndim != 2 or prompt.shape[1] != 5120:
        fail(f"prompt must be BF16 [L,5120], got {prompt.dtype} {tuple(prompt.shape)}")

    with torch.device("meta"):
        model = RefinerOracleModule()
    base_state, index_path, shard_names = load_base_state(model, transformer_dir)
    incompatible = model.load_state_dict(base_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail(f"base load mismatch: {incompatible}")
    del base_state
    model = model.to(device).eval().requires_grad_(False)
    for block in model.token_refiner.refiner_blocks:
        block.attn.set_attention_backend(args.attention_backend)

    sys.path.insert(0, str(args.upstream_dir))
    from src.inference.lora import merge_lora_state

    default_state = load_adapter_refiner(default_adapter, "default", str(device))
    default_pairs = merge_lora_state(model, default_state, 1.0)
    del default_state
    turbo_pairs = 0
    if use_turbo:
        turbo_state = load_adapter_refiner(turbo_adapter, "turbo", str(device))
        turbo_pairs = merge_lora_state(model, turbo_state, 1.0)
        del turbo_state
    torch.cuda.empty_cache()

    captured: Dict[str, torch.Tensor] = {
        "input.prompt_embeds": prompt.contiguous(),
    }
    handles = register_captures(model, captured)
    with torch.inference_mode():
        projected = model.context_embedder(prompt.to(device).unsqueeze(0))
        result = model.token_refiner(projected)
        torch.cuda.synchronize(device)
    if "refiner.final" not in captured:
        captured["refiner.final"] = result.detach().cpu().contiguous()
    for handle in handles:
        handle.remove()

    upstream_revision = git_revision(args.upstream_dir)
    diffusers_revision = git_revision(args.upstream_dir / "diffusers")
    revision_file = args.model_root / ".h3-vdn-revision"
    metadata = {
        "schema": SCHEMA,
        "scope": "real-weight-stage-dmd-prompt-refiner",
        "openvdn_commit": upstream_revision,
        "diffusers_commit": diffusers_revision,
        "model_revision": revision_file.read_text().strip().replace("\n", ";"),
        "checkpoint": args.checkpoint,
        "torch_version": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "device_name": props.name,
        "device_pci_bus": f"0x{int(props.pci_bus_id):02x}",
        "attention_backend": args.attention_backend,
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": sha256_file(args.prompt),
        "transformer_index_sha256": sha256_file(index_path),
        "base_shards": ",".join(shard_names),
        "default_adapter_sha256": sha256_file(default_adapter),
        "turbo_adapter_sha256": sha256_file(turbo_adapter) if use_turbo else "disabled",
        "default_pairs": str(default_pairs),
        "turbo_pairs": str(turbo_pairs),
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
    manifest_temp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_temp, manifest_path)
    print(f"default_pairs={default_pairs} turbo_pairs={turbo_pairs}")
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
