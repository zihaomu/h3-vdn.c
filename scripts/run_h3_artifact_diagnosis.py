#!/usr/bin/env python3
"""Diagnose MiniMax-H3 color blocks on one guarded physical GPU.

The experiment deliberately loads the official Diffusers pipeline once, then
runs, in order:

1. one 50-NFE trajectory and two Video-VAE decodes of its exact final latent
   (the released tiled path and the non-tiled path);
2. the same fixed request with native SDPA and ``_native_math``, capturing the
   first transformer input/output and comparing the final video/audio latents;
3. one native-SDPA 640x384, 50-NFE trajectory.

It never increases the NFE count.  That decision belongs after the first three
experiments have been inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

from run_h3_diffusers_reference import (
    load_pipeline,
    require_single_gpu,
    sha256,
    write_video_artifacts,
)


DEFAULT_PROMPT = (
    "A red fox walks slowly through a snowy pine forest, cinematic natural light, detailed fur"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("MiniMax-H3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--probe-width", type=int, default=640)
    parser.add_argument("--probe-height", type=int, default=384)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--resume-native-dir",
        type=Path,
        help="Reuse a completed native run directory and execute only stages 2 and 3.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, value in (
        ("width", args.width),
        ("height", args.height),
        ("probe-width", args.probe_width),
        ("probe-height", args.probe_height),
    ):
        if value <= 0 or value % 32:
            raise SystemExit(f"{label} must be a positive multiple of 32")
    if args.frames < 120 or args.frames > 360 or args.frames % 17 != 5:
        raise SystemExit("frames must be in [120, 360] and have the form 17*n+5")
    if args.steps != 50:
        raise SystemExit("this controlled diagnosis requires exactly 50 NFE")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha256(tensor),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
    }


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left_f32 = left.detach().float().cpu()
    right_f32 = right.detach().float().cpu()
    delta = left_f32 - right_f32
    denominator = max(float(torch.linalg.vector_norm(left_f32)), 1.0e-30)
    return {
        "same_shape": True,
        "bitwise_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "relative_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "nonfinite": int((~torch.isfinite(delta)).sum()),
    }


class FirstNfeCapture:
    def __init__(self, transformer: torch.nn.Module):
        self.call_count = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self.handle = transformer.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, _module, _args, kwargs, output) -> None:
        self.call_count += 1
        if self.call_count != 1:
            return
        for source, destination in (
            ("hidden_states", "video_input"),
            ("audio_hidden_states", "audio_input"),
            ("encoder_hidden_states", "prompt_input"),
            ("timestep", "timestep"),
            ("timestep_indices", "timestep_indices"),
        ):
            if source in kwargs:
                self.tensors[destination] = capture_cpu(kwargs[source])
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"unexpected transformer output type: {type(output)!r}")
        self.tensors["video_prediction"] = capture_cpu(output[0])
        self.tensors["audio_prediction"] = capture_cpu(output[1])

    def close(self) -> None:
        self.handle.remove()


def run_trajectory(
    pipe,
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, Any]]:
    capture = FirstNfeCapture(pipe.transformer)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=frames,
            num_inference_steps=steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="np",
            output=["videos", "audio", "sampling_rate", "timesteps", "latents", "audio_latents"],
        )
        torch.cuda.synchronize()
    finally:
        capture.close()
    elapsed = time.monotonic() - started
    if capture.call_count != steps:
        raise RuntimeError(f"expected {steps} transformer calls, observed {capture.call_count}")
    metadata = {
        "elapsed_seconds": elapsed,
        "nfe_observed": capture.call_count,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "width": width,
        "height": height,
        "frames": frames,
    }
    print(
        f"trajectory complete: {width}x{height} NFE={capture.call_count} "
        f"elapsed={elapsed:.3f}s peak_vram={metadata['peak_vram_gib']:.3f} GiB",
        flush=True,
    )
    return result, capture.tensors, metadata


def decode_video_latent(pipe, normalized_latent: torch.Tensor, *, tiled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if tiled:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.disable_tiling()
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    latent = normalized_latent.to(device)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(pipe.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    latent = latent * latents_std + latents_mean
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        decoded = pipe.vae.decode(latent, return_dict=False)[0]
    pixel_mean = torch.tensor(pipe.pixel_mean, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(pipe.pixel_std, device=device).view(1, -1, 1, 1, 1)
    decoded = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
    video = np.asarray(pipe.video_processor.postprocess_video(decoded, output_type="np")[0], dtype=np.float32)
    torch.cuda.synchronize()
    metadata = {
        "tiled": tiled,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "video_shape": list(video.shape),
    }
    del latent, decoded
    torch.cuda.empty_cache()
    print(
        f"VAE decode complete: tiled={tiled} elapsed={metadata['elapsed_seconds']:.3f}s "
        f"peak_vram={metadata['peak_vram_gib']:.3f} GiB",
        flush=True,
    )
    return video, metadata


def compare_videos(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    absolute = np.abs(delta)
    left_norm = max(float(np.linalg.norm(left.astype(np.float64).ravel())), 1.0e-30)
    column_mae = absolute.mean(axis=(0, 1, 3))
    row_mae = absolute.mean(axis=(0, 2, 3))
    top_columns = np.argsort(column_mae)[-10:][::-1]
    top_rows = np.argsort(row_mae)[-10:][::-1]
    left_u8 = np.rint(np.clip(left, 0, 1) * 255).astype(np.uint8)
    right_u8 = np.rint(np.clip(right, 0, 1) * 255).astype(np.uint8)
    return {
        "same_shape": True,
        "max_abs_float01": float(absolute.max()),
        "mean_abs_float01": float(absolute.mean()),
        "rmse_float01": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "relative_l2": float(np.linalg.norm(delta.astype(np.float64).ravel()) / left_norm),
        "changed_u8_fraction": float(np.mean(left_u8 != right_u8)),
        "max_abs_u8": int(np.abs(left_u8.astype(np.int16) - right_u8.astype(np.int16)).max()),
        "top_columns_by_mae": [[int(i), float(column_mae[i])] for i in top_columns],
        "top_rows_by_mae": [[int(i), float(row_mae[i])] for i in top_rows],
    }


def write_diff_contact(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
    # Fourfold amplification preserves absolute spatial structure without a
    # per-frame normalization that could make negligible noise look severe.
    visual = np.rint(np.clip(difference * 4.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    selected = sorted({0, len(visual) // 4, len(visual) // 2, 3 * len(visual) // 4, len(visual) - 1})
    contact = Image.new("RGB", (visual.shape[2] * len(selected), visual.shape[1]))
    for column, index in enumerate(selected):
        contact.paste(Image.fromarray(visual[index], "RGB"), (column * visual.shape[2], 0))
    contact.save(path)


def save_tensor_bundle(path: Path, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    canonical = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(canonical, os.fspath(path))
    return {
        "path": os.fspath(path),
        "file_sha256": sha256(path),
        "tensors": {name: tensor_summary(tensor) for name, tensor in canonical.items()},
    }


def update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_run_artifacts(
    root: Path,
    label: str,
    result: dict[str, object],
    first_nfe: dict[str, torch.Tensor],
    trajectory: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, np.ndarray]:
    output_dir = root / label
    output_dir.mkdir()
    video_latent = capture_cpu(result["latents"])
    audio_latent = capture_cpu(result["audio_latents"])
    latent_bundle = save_tensor_bundle(
        output_dir / "final-latents.safetensors",
        {"video_latent": video_latent, "audio_latent": audio_latent},
    )
    first_bundle = save_tensor_bundle(output_dir / "first-nfe.safetensors", first_nfe)
    # Persist the irreplaceable model outputs before PNG/WAV/ffmpeg packaging.
    # A media-tool failure must never force another 50-NFE trajectory merely
    # because its final latent had only existed in process memory.
    mp4_path, media = write_video_artifacts(
        output_dir, result["videos"][0], result["audio"], int(result["sampling_rate"])
    )
    stage = {
        **trajectory,
        "attention_backend": label,
        "media": media,
        "mp4": os.fspath(mp4_path),
        "final_latents": latent_bundle,
        "first_nfe": first_bundle,
    }
    return stage, video_latent, audio_latent, np.asarray(result["videos"][0], dtype=np.float32)


def main() -> None:
    args = parse_args()
    validate_args(args)
    physical_gpu, physical_gpu_bdf = require_single_gpu()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    if not (model_dir / "modular_model_index.json").is_file():
        raise SystemExit(f"not a local MiniMax-H3 modular snapshot: {model_dir}")

    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment_order": [
            "same-latent-tiled-vs-untiled-vae",
            "native-vs-native-math-first-nfe-and-final-latent",
            "native-640x384-50-nfe",
        ],
        "model_dir": os.fspath(model_dir),
        "model_index_sha256": sha256(model_dir / "modular_model_index.json"),
        "prompt": args.prompt,
        "seed": args.seed,
        "frames": args.frames,
        "steps": args.steps,
        "physical_gpu": physical_gpu,
        "physical_gpu_bdf": physical_gpu_bdf,
        "logical_gpu": 0,
        "stages": {},
    }
    update_manifest(manifest_path, manifest)
    pipe = load_pipeline(model_dir, use_stream=not args.no_stream)

    if args.resume_native_dir:
        latent_path = args.resume_native_dir / "final-latents.safetensors"
        first_path = args.resume_native_dir / "first-nfe.safetensors"
        if not latent_path.is_file() or not first_path.is_file():
            raise SystemExit(f"incomplete native resume directory: {args.resume_native_dir}")
        native_bundle = load_file(os.fspath(latent_path), device="cpu")
        native_latent = native_bundle["video_latent"]
        native_audio_latent = native_bundle["audio_latent"]
        native_first = load_file(os.fspath(first_path), device="cpu")
        manifest["stages"]["stage1_vae_ab"] = {
            "status": "COMPLETE_EXTERNAL",
            "resume_native_dir": os.fspath(args.resume_native_dir),
            "final_latents_sha256": sha256(latent_path),
            "first_nfe_sha256": sha256(first_path),
            "untiled_result": "recorded by the VAE-only companion run",
        }
        update_manifest(manifest_path, manifest)
        print(f"STAGE 1/3: reusing persisted native evidence from {args.resume_native_dir}", flush=True)
    else:
        # Stage 1: the released tiled decode and an untiled decode receive the
        # exact same persisted final latent.  Native is the default SDPA dispatch,
        # selected explicitly so no inherited process-global backend can leak in.
        print("STAGE 1/3: native 50-NFE + same-latent VAE tiled/untiled A/B", flush=True)
        pipe.transformer.set_attention_backend("native")
        pipe.vae.enable_tiling()
        native_result, native_first, native_timing = run_trajectory(
            pipe,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            seed=args.seed,
        )
        native_stage, native_latent, native_audio_latent, tiled_video = save_run_artifacts(
            args.output_dir, "native-tiled-320x192", native_result, native_first, native_timing
        )
        _, tile_widths, x_overlaps = pipe.vae._split_tiles(
            args.width, pipe.vae.tile_sample_min_width, pipe.vae.tile_sample_min_overlap_width
        )
        _, tile_heights, y_overlaps = pipe.vae._split_tiles(
            args.height, pipe.vae.tile_sample_min_height, pipe.vae.tile_sample_min_overlap_height
        )
        torch.cuda.empty_cache()
        untiled_video, untiled_timing = decode_video_latent(pipe, native_latent, tiled=False)
        untiled_dir = args.output_dir / "same-latent-untiled-320x192"
        untiled_dir.mkdir()
        untiled_mp4, untiled_media = write_video_artifacts(
            untiled_dir, untiled_video, native_result["audio"], int(native_result["sampling_rate"])
        )
        vae_comparison = compare_videos(tiled_video, untiled_video)
        diff_contact = args.output_dir / "vae-tiled-vs-untiled-diff-4x.png"
        write_diff_contact(diff_contact, tiled_video, untiled_video)
        manifest["stages"]["stage1_vae_ab"] = {
            "status": "COMPLETE",
            "native_tiled": native_stage,
            "untiled": {**untiled_timing, "media": untiled_media, "mp4": os.fspath(untiled_mp4)},
            "same_video_latent_sha256": tensor_sha256(native_latent),
            "tile_geometry": {
                "tile_widths": tile_widths,
                "tile_heights": tile_heights,
                "x_overlaps": x_overlaps,
                "y_overlaps": y_overlaps,
            },
            "comparison": vae_comparison,
            "diff_contact_4x": os.fspath(diff_contact),
        }
        update_manifest(manifest_path, manifest)
        print("STAGE 1/3 COMPLETE", flush=True)
        pipe.vae.enable_tiling()
        del tiled_video, untiled_video
        torch.cuda.empty_cache()

    # Stage 2: repeat the exact request and seed, changing only Diffusers'
    # attention backend.  Capturing first inputs proves whether the A/B really
    # starts from the same transformer state.
    print("STAGE 2/3: _native_math fixed-input 50-NFE comparison", flush=True)
    pipe.transformer.set_attention_backend("_native_math")
    math_result, math_first, math_timing = run_trajectory(
        pipe,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )
    math_stage, math_latent, math_audio_latent, _ = save_run_artifacts(
        args.output_dir, "native-math-tiled-320x192", math_result, math_first, math_timing
    )
    first_names = sorted(set(native_first) & set(math_first))
    first_comparisons = {name: compare_tensors(native_first[name], math_first[name]) for name in first_names}
    manifest["stages"]["stage2_attention_ab"] = {
        "status": "COMPLETE",
        "native_math": math_stage,
        "first_nfe": first_comparisons,
        "final_video_latent": compare_tensors(native_latent, math_latent),
        "final_audio_latent": compare_tensors(native_audio_latent, math_audio_latent),
    }
    update_manifest(manifest_path, manifest)
    print("STAGE 2/3 COMPLETE", flush=True)
    del math_result, math_latent, math_audio_latent
    torch.cuda.empty_cache()

    # Stage 3: resolution probe with the released tiled VAE.  Explicit native
    # restores the default backend after _native_math's process-global setting.
    print("STAGE 3/3: native 640x384 50-NFE resolution probe", flush=True)
    pipe.transformer.set_attention_backend("native")
    pipe.vae.enable_tiling()
    probe_result, probe_first, probe_timing = run_trajectory(
        pipe,
        prompt=args.prompt,
        width=args.probe_width,
        height=args.probe_height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )
    probe_stage, _, _, _ = save_run_artifacts(
        args.output_dir, "native-tiled-640x384", probe_result, probe_first, probe_timing
    )
    _, probe_tile_widths, probe_x_overlaps = pipe.vae._split_tiles(
        args.probe_width, pipe.vae.tile_sample_min_width, pipe.vae.tile_sample_min_overlap_width
    )
    _, probe_tile_heights, probe_y_overlaps = pipe.vae._split_tiles(
        args.probe_height, pipe.vae.tile_sample_min_height, pipe.vae.tile_sample_min_overlap_height
    )
    manifest["stages"]["stage3_resolution_probe"] = {
        "status": "COMPLETE",
        **probe_stage,
        "tile_geometry": {
            "tile_widths": probe_tile_widths,
            "tile_heights": probe_tile_heights,
            "x_overlaps": probe_x_overlaps,
            "y_overlaps": probe_y_overlaps,
        },
    }
    manifest["status"] = "COMPLETE_AWAITING_VISUAL_ASSESSMENT"
    manifest["step_increase_executed"] = False
    update_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
