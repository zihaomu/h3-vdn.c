#!/usr/bin/env python3
"""Export an official FL2VA Qwen3-VL first-50-layer text oracle."""

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
LAYERS = 50


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--task", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="A red fox walking through snow"
    )
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
        fail(f"missing Text Encoder checkpoint index: {index}")
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        fail(f"invalid Text Encoder checkpoint index: {exc}")
    missing = sorted(
        {
            name
            for name in weight_map.values()
            if not (component / name).is_file()
        }
    )
    if missing:
        fail(f"missing Text Encoder shards: {missing}")
    if args.output.suffix != ".safetensors":
        fail("--output must end in .safetensors")


def main() -> None:
    args = parse_args()
    component = args.model_root.resolve() / args.task / "text_encoder"
    verify_environment(args, component)

    try:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        from transformers.masking_utils import create_causal_mask
    except ImportError as exc:
        fail(
            "Transformers 4.57 text-oracle environment is required; prepend "
            f".text-oracle-deps to PYTHONPATH: {exc}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(component), local_files_only=True
    )
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    if not token_ids:
        fail("the fixed prompt produced no tokens")

    # Keep the official checkpoint resident in host RAM, then migrate only one
    # decoder layer at a time.  This executes the released Transformers
    # modules verbatim while staying below the 31.9 GiB device limit.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(component),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    language = model.model.language_model
    if len(language.layers) < LAYERS:
        fail(f"checkpoint exposes only {len(language.layers)} language layers")
    language.config._attn_implementation = "eager"

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    ids_cpu = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    ids = ids_cpu.to(device)

    snapshots: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        embedding = language.embed_tokens.to(device)
        hidden = embedding(ids)
        embedding.to("cpu")
        torch.cuda.empty_cache()

        sequence = hidden.shape[1]
        cache_position = torch.arange(sequence, device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        attention_mask = create_causal_mask(
            config=language.config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids[0],
        )
        rotary = language.rotary_emb.to(device)
        position_embeddings = rotary(hidden, position_ids)
        rotary.to("cpu")
        torch.cuda.empty_cache()

        for layer_index in range(LAYERS):
            layer = language.layers[layer_index].to(device)
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            layer.to("cpu")
            torch.cuda.empty_cache()
            completed = layer_index + 1
            if completed in (1, 25, 50):
                snapshots[completed] = hidden[0].to("cpu").contiguous()
            print(f"official Qwen text oracle: {completed}/{LAYERS}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    output_50 = snapshots[50]
    if output_50.dtype != torch.bfloat16 or not torch.isfinite(
        output_50.float()
    ).all():
        fail("official first-50-layer output is invalid")
    ids_i32 = ids_cpu.to(torch.int32).squeeze(0)
    prompt_utf8 = torch.tensor(
        list(args.prompt.encode("utf-8")), dtype=torch.uint8
    )
    tensors = {
        "input.prompt_utf8": prompt_utf8,
        "input.token_ids": ids_i32,
        "text.layer_01.output": snapshots[1],
        "text.layer_25.output": snapshots[25],
        "text.layer_50.output": output_50,
        "x.ids": ids_i32,
        "x.output": output_50,
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
            "case_id": f"{args.task.lower()}-text-tiny-v1",
            "seed": "not-used",
            "rng_policy": "explicit token IDs and tensor bytes are authoritative",
            "exporter_sha256": sha256_file(exporter),
            "model_index_sha256": sha256_file(
                component / "model.safetensors.index.json"
            ),
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
            "layer_count": str(LAYERS),
            "weight_residency": "host RAM; one official layer at a time on GPU",
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
