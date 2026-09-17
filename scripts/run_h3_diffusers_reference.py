#!/usr/bin/env python3
"""Run a local, upstream-Diffusers MiniMax-H3 reference on one guarded GPU.

The script deliberately uses the released BF16 base weights, not the VDN
adapters.  It is intended to answer whether a semantic failure is already
present in the official PyTorch path or was introduced by the native C/ROCm
port.  Large modules remain in host RAM and are streamed one block at a time
to the single visible GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import numpy as np
from PIL import Image
import torch
from transformers import Qwen3VLProcessor

from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("MiniMax-H3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="A red fox walks slowly through a snowy pine forest, cinematic natural light, detailed fur",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the 32x32, two-NFE smoke run before the requested run.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable asynchronous group-offload transfers if pinned-memory streaming is unsupported.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_gpu() -> tuple[int, str]:
    expected_text = os.environ.get("H3_PHYSICAL_GPU", "4")
    try:
        expected = int(expected_text)
    except ValueError as exc:
        raise SystemExit(f"invalid H3_PHYSICAL_GPU={expected_text!r}") from exc
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible != expected_text:
        raise SystemExit(
            "refusing to run: launch through scripts/profile_vdn_gpu4.sh so "
            f"HIP_VISIBLE_DEVICES is exactly H3_PHYSICAL_GPU={expected_text}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            f"refusing to run: expected one visible HIP device, got {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "")
    if not architecture.startswith("gfx1201"):
        raise SystemExit(
            f"refusing to run: expected physical GPU {expected} / gfx1201, "
            f"got {properties.name} / {architecture}"
        )
    bdf = os.environ.get("H3_PHYSICAL_GPU_BDF", "unknown")
    print(
        f"device: logical=0 physical={expected} bdf={bdf} "
        f"name={properties.name!r} architecture={architecture}",
        flush=True,
    )
    return expected, bdf


def validate_request(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise SystemExit("width and height must be positive multiples of 32")
    if args.frames < 120 or args.frames > 360:
        raise SystemExit("the released MiniMax-H3 supports 5-15 seconds (120-360 frames at 24 fps)")
    if args.frames % 17 != 5:
        raise SystemExit("frames must have the form 17*n+5 (124 is the shortest supported value)")
    if args.steps < 2:
        raise SystemExit("steps must be at least 2")


def load_pipeline(model_dir: Path, use_stream: bool) -> ModularPipeline:
    started = time.monotonic()
    pipe = ModularPipeline.from_pretrained(
        os.fspath(model_dir), workflow="t2va", local_files_only=True
    )
    # Override the repository recorded in modular_model_index.json so every
    # component is resolved from the assembled local snapshot, never the Hub.
    # from_pretrained(..., workflow="t2va") has already pruned the blocks to
    # a SequentialPipelineBlocks instance, so load the components directly;
    # asking that pruned object for a workflow a second time is invalid.
    load_names = [name for name in pipe.component_names if name != "processor"]
    pipe.load_components(
        names=load_names,
        pretrained_model_name_or_path=os.fspath(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
    )

    # This checkout's Transformers 4.57 predates ProcessorMixin's
    # create_mm_token_type_ids(), while this Diffusers integration was written
    # against the newer API.  T2VA does not process images or video: it only
    # asks the processor to label the already-tokenized prompt.  Reproduce the
    # upstream method exactly without constructing unused torchvision-backed
    # media processors.  Every token in an ordinary text prompt remains type 0.
    processor = object.__new__(Qwen3VLProcessor)
    processor.image_token_ids = [pipe.tokenizer.convert_tokens_to_ids("<|image_pad|>")]
    processor.video_token_ids = [pipe.tokenizer.convert_tokens_to_ids("<|video_pad|>")]
    processor.audio_token_ids = []

    def create_mm_token_type_ids(input_ids: list) -> list[list[int]]:
        result = []
        for tokenizer_input in input_ids:
            if not isinstance(tokenizer_input, list):
                tokenizer_input = tokenizer_input.tolist()
            tokenizer_input = np.array(tokenizer_input)
            token_types = np.zeros_like(tokenizer_input)
            token_types[np.isin(tokenizer_input, processor.image_token_ids)] = 1
            token_types[np.isin(tokenizer_input, processor.video_token_ids)] = 2
            token_types[np.isin(tokenizer_input, processor.audio_token_ids)] = 3
            result.append(token_types.tolist())
        return result

    processor.create_mm_token_type_ids = create_mm_token_type_ids
    pipe.update_components(processor=processor)
    missing = [name for name in pipe.component_names if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"failed to load pipeline components: {missing}")
    print(f"loaded all t2va components in {time.monotonic() - started:.3f}s", flush=True)

    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    offload = {
        "onload_device": torch.device("cuda:0"),
        "offload_device": torch.device("cpu"),
        "use_stream": use_stream,
    }
    pipe.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1, **offload
    )
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)

    # Together these released decoders occupy far less than the 31.9 GiB card;
    # keeping them resident avoids another offload mechanism during decode.
    pipe.vae.to("cuda:0")
    pipe.audio_vae.to("cuda:0")
    pipe.set_progress_bar_config(desc="MiniMax-H3 denoise", dynamic_ncols=True)
    print(
        f"offload configured: transformer=block text_encoder=leaf stream={use_stream}; "
        f"resident decoders on cuda:0; allocated={torch.cuda.memory_allocated() / 2**30:.3f} GiB",
        flush=True,
    )
    return pipe


def run_request(
    pipe: ModularPipeline,
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
) -> tuple[dict[str, object], float]:
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=frames,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
        output=["videos", "audio", "sampling_rate", "timesteps"],
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    print(
        f"generation complete: {width}x{height} frames={frames} steps={steps} "
        f"elapsed={elapsed:.3f}s peak_vram={torch.cuda.max_memory_allocated() / 2**30:.3f} GiB",
        flush=True,
    )
    return result, elapsed


def to_uint8_video(value: object) -> np.ndarray:
    video = np.asarray(value)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise RuntimeError(f"unexpected decoded video shape: {video.shape}")
    if not np.isfinite(video).all():
        raise RuntimeError("decoded video contains NaN or infinity")
    if np.issubdtype(video.dtype, np.floating):
        video = np.rint(np.clip(video, 0.0, 1.0) * 255.0)
    return video.astype(np.uint8)


def write_wav(path: Path, audio_value: object, sample_rate: int) -> dict[str, object]:
    if isinstance(audio_value, torch.Tensor):
        audio = audio_value.detach().float().cpu().numpy()
    else:
        audio = np.asarray(audio_value, dtype=np.float32)
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 2:
        raise RuntimeError(f"unexpected decoded audio shape: {audio.shape}")
    if not np.isfinite(audio).all():
        raise RuntimeError("decoded audio contains NaN or infinity")
    channels, samples = audio.shape
    pcm = np.rint(np.clip(audio, -1.0, 1.0).T * 32767.0).astype("<i2")
    with wave.open(os.fspath(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return {
        "shape": list(audio.shape),
        "min": float(audio.min()),
        "max": float(audio.max()),
        "mean": float(audio.mean()),
        "std": float(audio.std()),
        "samples": int(samples),
        "channels": int(channels),
    }


def write_video_artifacts(
    output_dir: Path, video_value: object, audio_value: object, sample_rate: int
) -> tuple[Path, dict[str, object]]:
    video = to_uint8_video(video_value)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()
    for index, frame in enumerate(video):
        Image.fromarray(frame, "RGB").save(frames_dir / f"frame_{index:04d}.png")

    selected = sorted({0, len(video) // 4, len(video) // 2, 3 * len(video) // 4, len(video) - 1})
    contact = Image.new("RGB", (video.shape[2] * len(selected), video.shape[1]))
    for column, index in enumerate(selected):
        contact.paste(Image.fromarray(video[index], "RGB"), (column * video.shape[2], 0))
    contact_path = output_dir / "contact.png"
    contact.save(contact_path)

    wav_path = output_dir / "audio.wav"
    audio_stats = write_wav(wav_path, audio_value, sample_rate)
    mp4_path = output_dir / "minimax-h3-reference.mp4"
    ffmpeg = Path(os.environ.get("H3_FFMPEG", ".tools/ffmpeg/usr/bin/ffmpeg"))
    if not ffmpeg.is_file():
        raise RuntimeError(f"ffmpeg executable is unavailable: {ffmpeg}")
    ffmpeg_environment = os.environ.copy()
    private_library_dir = ffmpeg.resolve().parents[1] / "lib" / "x86_64-linux-gnu"
    if private_library_dir.is_dir():
        existing_library_path = ffmpeg_environment.get("LD_LIBRARY_PATH", "")
        ffmpeg_environment["LD_LIBRARY_PATH"] = os.fspath(private_library_dir) + (
            f":{existing_library_path}" if existing_library_path else ""
        )
    subprocess.run(
        [
            os.fspath(ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "24",
            "-i",
            os.fspath(frames_dir / "frame_%04d.png"),
            "-i",
            os.fspath(wav_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            os.fspath(mp4_path),
        ],
        check=True,
        env=ffmpeg_environment,
    )
    frame_f32 = video.astype(np.float32)
    stats = {
        "shape": list(video.shape),
        "min": int(video.min()),
        "max": int(video.max()),
        "mean": float(frame_f32.mean()),
        "std": float(frame_f32.std()),
        "mean_abs_temporal_delta": float(
            np.abs(np.diff(frame_f32, axis=0)).mean() if len(video) > 1 else 0.0
        ),
        "selected_contact_frames": selected,
        "audio": audio_stats,
        "mp4_sha256": sha256(mp4_path),
        "contact_sha256": sha256(contact_path),
    }
    return mp4_path, stats


def main() -> None:
    args = parse_args()
    validate_request(args)
    physical_gpu, physical_gpu_bdf = require_single_gpu()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir.resolve()
    if not (model_dir / "modular_model_index.json").is_file():
        raise SystemExit(f"not a local MiniMax-H3 modular snapshot: {model_dir}")
    print(f"model: {model_dir}", flush=True)
    pipe = load_pipeline(model_dir, use_stream=not args.no_stream)

    smoke_elapsed = None
    if not args.skip_smoke:
        print("starting supported-duration 32x32 / two-NFE execution-chain smoke", flush=True)
        smoke, smoke_elapsed = run_request(
            pipe,
            prompt="a red fox",
            width=32,
            height=32,
            frames=124,
            steps=2,
            seed=args.seed,
        )
        smoke_video = to_uint8_video(smoke["videos"][0])
        print(
            f"smoke finite: shape={tuple(smoke_video.shape)} min={smoke_video.min()} "
            f"max={smoke_video.max()} std={smoke_video.std():.6f}",
            flush=True,
        )
        del smoke, smoke_video
        torch.cuda.empty_cache()

    print("starting requested semantic reference generation", flush=True)
    result, elapsed = run_request(
        pipe,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )
    mp4_path, stats = write_video_artifacts(
        args.output_dir,
        result["videos"][0],
        result["audio"],
        int(result["sampling_rate"]),
    )
    manifest = {
        "model_dir": os.fspath(model_dir),
        "model_index_sha256": sha256(model_dir / "modular_model_index.json"),
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": 24,
        "num_inference_steps": args.steps,
        "model_evaluations": int(result["timesteps"].numel()),
        "dtype": "bfloat16",
        "physical_gpu": physical_gpu,
        "physical_gpu_bdf": physical_gpu_bdf,
        "logical_gpu": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_architecture": getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
        "group_offload_stream": not args.no_stream,
        "smoke_elapsed_seconds": smoke_elapsed,
        "generation_elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "output_mp4": os.fspath(mp4_path),
        "stats": stats,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
