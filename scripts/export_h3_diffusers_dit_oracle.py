#!/usr/bin/env python3
"""Export a compact official-Diffusers MiniMax-H3 DiT parity fixture.

The fixture contains inputs in native h3.c channel-major layout, the first
50-layer velocity, the final Euler latents, and selected hidden states. It is
a local test artifact and must never be committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h3_diffusers_reference import load_pipeline, require_single_gpu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("MiniMax-H3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox walks through snow")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unpatch_video(rows: torch.Tensor, time: int, height: int, width: int) -> torch.Tensor:
    rows = rows.reshape(1, time, height // 2, width // 2, 24, 1, 2, 2)
    return rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, time, height, width)


def native_audio(value: torch.Tensor) -> torch.Tensor:
    # Diffusers decoder layout [stereo, latent_channels, time] -> h3.c
    # host boundary [latent_channels, stereo, time].
    return value.permute(1, 0, 2).contiguous()


def main() -> None:
    args = parse_args()
    if args.output.suffix != ".safetensors":
        raise SystemExit("--output must end in .safetensors")
    if args.width % 32 or args.height % 32 or args.frames % 17 != 5:
        raise SystemExit("geometry must use 32-pixel multiples and 17*n+5 frames")
    physical_gpu, bdf = require_single_gpu()
    pipe = load_pipeline(args.model_dir.resolve(), use_stream=True)

    latent_t = (args.frames // 17) * 5 + 2
    latent_h = args.height // 16
    latent_w = args.width // 16
    audio_t = round(args.frames / 24 * 40)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    video_initial = torch.randn(
        (1, 24, latent_t, latent_h, latent_w), generator=generator,
        dtype=torch.float32,
    )
    audio_initial = torch.randn(
        (2, 32, audio_t), generator=generator, dtype=torch.float32,
    )

    captured: dict[str, torch.Tensor] = {}

    def refiner_hook(_module, _inputs, output):
        if "refiner" not in captured:
            captured["refiner"] = output.detach().to("cpu").contiguous()

    def block_hook(index: int):
        def capture(_module, _inputs, output):
            name = f"block_{index:02d}"
            if name not in captured:
                captured[name] = output.detach().to("cpu").contiguous()
        return capture

    def transformer_hook(_module, _inputs, kwargs, output):
        if "prompt" in captured:
            return
        captured["prompt"] = kwargs["encoder_hidden_states"].detach().to("cpu").contiguous()
        captured["token_tags"] = kwargs["token_tags"].detach().to("cpu").to(torch.int32).contiguous()
        captured["position_ids"] = kwargs["position_ids"].detach().to("cpu").float().contiguous()
        captured["timestep"] = kwargs["timestep"].detach().to("cpu").float().contiguous()
        captured["timestep_indices"] = kwargs["timestep_indices"].detach().to("cpu").to(torch.int32).contiguous()
        video_rows, audio_rows = output
        captured["video_velocity"] = unpatch_video(
            video_rows.detach().to("cpu").float(), latent_t, latent_h, latent_w
        )[0].contiguous()
        audio = audio_rows.detach().to("cpu").float().reshape(2, audio_t, 32).permute(0, 2, 1)
        captured["audio_velocity"] = native_audio(audio.contiguous())

    handles = [pipe.transformer.token_refiner.register_forward_hook(refiner_hook)]
    handles.extend(
        block.register_forward_hook(block_hook(index))
        for index, block in enumerate(pipe.transformer.transformer_blocks)
    )
    handles.append(pipe.transformer.register_forward_hook(transformer_hook, with_kwargs=True))
    try:
        result = pipe(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            latents=video_initial,
            audio_latents=audio_initial,
            output=["latents", "audio_latents", "timesteps"],
        )
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    required = {"prompt", "refiner", "video_velocity", "audio_velocity"}
    required.update(f"block_{index:02d}" for index in range(50))
    missing = sorted(required - captured.keys())
    if missing:
        raise RuntimeError(f"hooks did not capture: {missing}")
    video_final = result["latents"].detach().to("cpu").float()[0].contiguous()
    audio_final = native_audio(
        result["audio_latents"].detach().to("cpu").float().contiguous()
    )
    tensors = {
        "input.video": video_initial[0].contiguous(),
        "input.audio": native_audio(audio_initial),
        "input.prompt": captured["prompt"][0].to(torch.bfloat16),
        "input.token_tags": captured["token_tags"],
        "input.position_ids": captured["position_ids"],
        "input.timestep": captured["timestep"],
        "input.timestep_indices": captured["timestep_indices"],
        "first.refiner": captured["refiner"][0].to(torch.bfloat16),
        "first.video_velocity": captured["video_velocity"],
        "first.audio_velocity": captured["audio_velocity"],
        "final.video": video_final,
        "final.audio": audio_final,
    }
    tensors.update(
        {
            f"first.block_{index:02d}": captured[f"block_{index:02d}"][0].to(torch.bfloat16)
            for index in range(50)
        }
    )
    metadata = {
        "schema": "h3-diffusers-dit-oracle-v1",
        "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
        "prompt": args.prompt,
        "seed": str(args.seed),
        "width": str(args.width),
        "height": str(args.height),
        "frames": str(args.frames),
        "steps": str(args.steps),
        "physical_gpu": str(physical_gpu),
        "pci_bdf": bdf,
        "dtype": "bfloat16 transformer / float32 latent boundary",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(manifest)


if __name__ == "__main__":
    main()
