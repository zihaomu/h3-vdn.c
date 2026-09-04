#!/usr/bin/env python3
"""Compare a production VDN candidate with its exact BF16 media reference."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


THRESHOLDS = {
    "video_psnr_mean_min_db": 32.0,
    "video_psnr_frame_min_db": 28.0,
    "video_ssim_mean_min": 0.95,
    "video_ssim_frame_min": 0.90,
    "video_luma_mean_delta_max": 2.0 / 255.0,
    "temporal_energy_ratio_min": 0.90,
    "temporal_energy_ratio_max": 1.10,
    "temporal_cosine_min": 0.95,
    "audio_cosine_min": 0.99,
    "audio_relative_rmse_max": 0.15,
    "audio_si_sdr_min_db": 15.0,
    "audio_rms_delta_max_db": 1.0,
    "audio_clipping_delta_max": 0.001,
    "audio_silence_delta_max": 0.01,
    "av_duration_delta_max_seconds": 1.0 / 24.0,
}


def run(command: list[str], *, stdout: bool = False) -> bytes:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result.stdout if stdout else b""


def probe(ffprobe: str, path: Path) -> dict:
    return json.loads(
        run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                os.fspath(path),
            ],
            stdout=True,
        )
    )


def selected_stream(metadata: dict, codec_type: str) -> dict:
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == codec_type:
            return stream
    raise RuntimeError(f"missing {codec_type} stream")


def stream_duration(stream: dict, metadata: dict) -> float:
    if stream.get("duration") not in (None, "N/A"):
        return float(stream["duration"])
    if stream.get("duration_ts") not in (None, "N/A") and stream.get(
        "time_base"
    ):
        return float(stream["duration_ts"]) * float(Fraction(stream["time_base"]))
    duration = metadata.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        raise RuntimeError("stream and container duration are unavailable")
    return float(duration)


def media_contract(metadata: dict) -> dict:
    video = selected_stream(metadata, "video")
    audio = selected_stream(metadata, "audio")
    return {
        "video_codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frames": int(video.get("nb_frames", 0)),
        "fps": float(Fraction(video["r_frame_rate"])),
        "video_duration_seconds": stream_duration(video, metadata),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "audio_duration_seconds": stream_duration(audio, metadata),
    }


def parse_stats(path: Path, key: str) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = {}
        for item in line.split():
            if ":" in item:
                name, value = item.split(":", 1)
                fields[name] = value
        if key in fields:
            values.append(float(fields[key]))
    if not values:
        raise RuntimeError(f"no {key} values in {path}")
    return values


def ffmpeg_frame_metrics(
    ffmpeg: str, reference: Path, candidate: Path, directory: Path
) -> tuple[list[float], list[float]]:
    psnr_path = directory / "psnr.log"
    ssim_path = directory / "ssim.log"
    common = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        os.fspath(reference),
        "-i",
        os.fspath(candidate),
    ]
    prefix = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[r];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv444p[c];"
    )
    run(
        common
        + [
            "-filter_complex",
            prefix + f"[r][c]psnr=stats_file={psnr_path}",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    run(
        common
        + [
            "-filter_complex",
            prefix + f"[r][c]ssim=stats_file={ssim_path}",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return parse_stats(psnr_path, "psnr_avg"), parse_stats(ssim_path, "All")


def decode_gray(ffmpeg: str, path: Path) -> bytes:
    return run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            os.fspath(path),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        stdout=True,
    )


def video_temporal_metrics(
    reference: bytes, candidate: bytes, width: int, height: int, frames: int
) -> dict:
    frame_bytes = width * height
    expected = frame_bytes * frames
    if len(reference) != expected or len(candidate) != expected:
        raise RuntimeError(
            "decoded video size mismatch: "
            f"reference={len(reference)} candidate={len(candidate)} "
            f"expected={expected}"
        )
    reference_luma = sum(reference) / (255.0 * len(reference))
    candidate_luma = sum(candidate) / (255.0 * len(candidate))
    reference_energy = 0
    candidate_energy = 0
    dot_product = 0
    reference_view = memoryview(reference)
    candidate_view = memoryview(candidate)
    for frame in range(1, frames):
        begin = frame * frame_bytes
        previous = begin - frame_bytes
        for offset in range(frame_bytes):
            left = reference_view[begin + offset] - reference_view[previous + offset]
            right = candidate_view[begin + offset] - candidate_view[previous + offset]
            reference_energy += left * left
            candidate_energy += right * right
            dot_product += left * right
    energy_ratio = (
        candidate_energy / reference_energy if reference_energy else math.inf
    )
    cosine = (
        dot_product / math.sqrt(reference_energy * candidate_energy)
        if reference_energy and candidate_energy
        else 0.0
    )
    return {
        "reference_luma_mean": reference_luma,
        "candidate_luma_mean": candidate_luma,
        "luma_mean_delta": abs(candidate_luma - reference_luma),
        "reference_delta_energy": reference_energy,
        "candidate_delta_energy": candidate_energy,
        "delta_energy_ratio": energy_ratio,
        "delta_cosine": cosine,
    }


def decode_audio(ffmpeg: str, path: Path) -> array.array:
    payload = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            os.fspath(path),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=True,
    )
    if len(payload) % 4:
        raise RuntimeError("decoded F32LE audio has a partial sample")
    samples = array.array("f")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def audio_metrics(reference: array.array, candidate: array.array) -> dict:
    if not reference or len(reference) != len(candidate):
        raise RuntimeError(
            "decoded audio size mismatch: "
            f"reference={len(reference)} candidate={len(candidate)}"
        )
    count = len(reference)
    reference_mean = math.fsum(reference) / count
    candidate_mean = math.fsum(candidate) / count
    reference_energy = 0.0
    candidate_energy = 0.0
    dot_product = 0.0
    error_energy = 0.0
    clipping_reference = 0
    clipping_candidate = 0
    silence_reference = 0
    silence_candidate = 0
    invalid = 0
    for left_raw, right_raw in zip(reference, candidate):
        left = float(left_raw)
        right = float(right_raw)
        invalid += not math.isfinite(left) or not math.isfinite(right)
        left_zero_mean = left - reference_mean
        right_zero_mean = right - candidate_mean
        difference = right_zero_mean - left_zero_mean
        reference_energy += left_zero_mean * left_zero_mean
        candidate_energy += right_zero_mean * right_zero_mean
        dot_product += left_zero_mean * right_zero_mean
        error_energy += difference * difference
        clipping_reference += abs(left) >= 0.999
        clipping_candidate += abs(right) >= 0.999
        silence_reference += abs(left) < 0.001
        silence_candidate += abs(right) < 0.001
    cosine = (
        dot_product / math.sqrt(reference_energy * candidate_energy)
        if reference_energy and candidate_energy
        else 0.0
    )
    relative_rmse = (
        math.sqrt(error_energy / reference_energy)
        if reference_energy
        else math.inf
    )
    projection = dot_product / reference_energy
    target_energy = projection * projection * reference_energy
    noise_energy = 0.0
    for left_raw, right_raw in zip(reference, candidate):
        target = projection * (float(left_raw) - reference_mean)
        residual = (float(right_raw) - candidate_mean) - target
        noise_energy += residual * residual
    si_sdr = (
        10.0 * math.log10(target_energy / noise_energy)
        if noise_energy
        else math.inf
    )
    reference_rms = math.sqrt(reference_energy / count)
    candidate_rms = math.sqrt(candidate_energy / count)
    rms_delta_db = (
        abs(20.0 * math.log10(candidate_rms / reference_rms))
        if reference_rms and candidate_rms
        else math.inf
    )
    return {
        "samples": count,
        "invalid": invalid,
        "cosine": cosine,
        "relative_rmse": relative_rmse,
        "si_sdr_db": si_sdr,
        "reference_rms_dbfs": (
            20.0 * math.log10(reference_rms) if reference_rms else -math.inf
        ),
        "candidate_rms_dbfs": (
            20.0 * math.log10(candidate_rms) if candidate_rms else -math.inf
        ),
        "rms_delta_db": rms_delta_db,
        "reference_clipping_ratio": clipping_reference / count,
        "candidate_clipping_ratio": clipping_candidate / count,
        "clipping_ratio_delta": max(
            0.0, (clipping_candidate - clipping_reference) / count
        ),
        "reference_silence_ratio": silence_reference / count,
        "candidate_silence_ratio": silence_candidate / count,
        "silence_ratio_delta": abs(silence_candidate - silence_reference) / count,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    reference_probe = probe(args.ffprobe, args.reference)
    candidate_probe = probe(args.ffprobe, args.candidate)
    reference_contract = media_contract(reference_probe)
    candidate_contract = media_contract(candidate_probe)
    with tempfile.TemporaryDirectory(prefix="vdn-quality-") as temporary:
        psnr, ssim = ffmpeg_frame_metrics(
            args.ffmpeg, args.reference, args.candidate, Path(temporary)
        )
    if len(psnr) != len(ssim):
        raise RuntimeError("PSNR and SSIM frame counts differ")
    width = reference_contract["width"]
    height = reference_contract["height"]
    frames = reference_contract["frames"]
    temporal = video_temporal_metrics(
        decode_gray(args.ffmpeg, args.reference),
        decode_gray(args.ffmpeg, args.candidate),
        width,
        height,
        frames,
    )
    audio = audio_metrics(
        decode_audio(args.ffmpeg, args.reference),
        decode_audio(args.ffmpeg, args.candidate),
    )
    video = {
        "frames_compared": len(psnr),
        "psnr_mean_db": sum(psnr) / len(psnr),
        "psnr_min_db": min(psnr),
        "ssim_mean": sum(ssim) / len(ssim),
        "ssim_min": min(ssim),
        **temporal,
    }
    av_delta = abs(
        candidate_contract["video_duration_seconds"]
        - candidate_contract["audio_duration_seconds"]
    )
    failures: list[str] = []
    check(reference_contract == candidate_contract, "container contract differs", failures)
    check(
        candidate_contract["width"] == 512
        and candidate_contract["height"] == 512
        and candidate_contract["frames"] == 56
        and abs(candidate_contract["fps"] - 24.0) < 1e-9
        and candidate_contract["sample_rate"] == 32000
        and candidate_contract["channels"] == 2,
        "candidate is not the production media geometry",
        failures,
    )
    check(len(psnr) == 56, "comparison did not produce 56 frame metrics", failures)
    check(video["psnr_mean_db"] >= THRESHOLDS["video_psnr_mean_min_db"], "mean PSNR", failures)
    check(video["psnr_min_db"] >= THRESHOLDS["video_psnr_frame_min_db"], "minimum frame PSNR", failures)
    check(video["ssim_mean"] >= THRESHOLDS["video_ssim_mean_min"], "mean SSIM", failures)
    check(video["ssim_min"] >= THRESHOLDS["video_ssim_frame_min"], "minimum frame SSIM", failures)
    check(video["luma_mean_delta"] <= THRESHOLDS["video_luma_mean_delta_max"], "mean luma delta", failures)
    check(
        THRESHOLDS["temporal_energy_ratio_min"]
        <= video["delta_energy_ratio"]
        <= THRESHOLDS["temporal_energy_ratio_max"],
        "temporal delta energy ratio",
        failures,
    )
    check(video["delta_cosine"] >= THRESHOLDS["temporal_cosine_min"], "temporal delta cosine", failures)
    check(audio["invalid"] == 0, "non-finite audio", failures)
    check(audio["cosine"] >= THRESHOLDS["audio_cosine_min"], "audio cosine", failures)
    check(audio["relative_rmse"] <= THRESHOLDS["audio_relative_rmse_max"], "audio relative RMSE", failures)
    check(audio["si_sdr_db"] >= THRESHOLDS["audio_si_sdr_min_db"], "audio SI-SDR", failures)
    check(audio["rms_delta_db"] <= THRESHOLDS["audio_rms_delta_max_db"], "audio RMS delta", failures)
    check(audio["clipping_ratio_delta"] <= THRESHOLDS["audio_clipping_delta_max"], "audio clipping delta", failures)
    check(audio["silence_ratio_delta"] <= THRESHOLDS["audio_silence_delta_max"], "audio silence delta", failures)
    check(av_delta <= THRESHOLDS["av_duration_delta_max_seconds"], "candidate A/V duration delta", failures)

    report = {
        "schema": "h3-vdn-media-quality-v1",
        "reference": {
            "path": os.fspath(args.reference),
            "sha256": sha256(args.reference),
            "contract": reference_contract,
        },
        "candidate": {
            "path": os.fspath(args.candidate),
            "sha256": sha256(args.candidate),
            "contract": candidate_contract,
        },
        "thresholds": THRESHOLDS,
        "video": video,
        "audio": audio,
        "candidate_av_duration_delta_seconds": av_delta,
        "passed": not failures,
        "failures": failures,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, ZeroDivisionError) as error:
        print(f"VDN media comparison failed: {error}", file=sys.stderr)
        raise SystemExit(2)
