#!/usr/bin/env python3
"""Strict ffprobe contract gate for the original-H3 deterministic A/V mux."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def fail(message: str) -> None:
    print(f"FAIL h3 mux: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=Fraction, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--max-av-drift", type=float, default=1.0 / 24.0)
    return parser.parse_args()


def stream_duration(stream: dict[str, object], document: dict[str, object]) -> float:
    raw = stream.get("duration")
    if raw not in (None, "N/A"):
        return float(raw)
    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts not in (None, "N/A") and time_base not in (None, "N/A"):
        return float(duration_ts) * float(Fraction(str(time_base)))
    format_data = document.get("format")
    if isinstance(format_data, dict) and format_data.get("duration") not in (
        None,
        "N/A",
    ):
        return float(format_data["duration"])
    fail(f"duration is unavailable for {stream.get('codec_type')} stream")
    return math.nan


def main() -> None:
    args = parse_args()
    if not args.path.is_file() or args.path.stat().st_size < 1000:
        fail(f"missing or empty container: {args.path}")
    ffprobe = os.environ.get("H3_FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        fail("ffprobe is unavailable")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            os.fspath(args.path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        fail(f"ffprobe exited {result.returncode}: {result.stderr.strip()}")
    try:
        document = json.loads(result.stdout)
        streams = document["streams"]
        video = next(item for item in streams if item.get("codec_type") == "video")
        audio = next(item for item in streams if item.get("codec_type") == "audio")
        video_duration = stream_duration(video, document)
        audio_duration = stream_duration(audio, document)
        fps = Fraction(str(video["avg_frame_rate"]))
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        fail(f"malformed ffprobe document: {exc}")

    expected = {
        "video_codec": "h264",
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "audio_codec": "aac",
        "sample_rate": args.sample_rate,
        "channels": args.channels,
    }
    actual = {
        "video_codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frames": int(video.get("nb_frames", 0)),
        "fps": fps,
        "audio_codec": audio.get("codec_name"),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
    }
    if actual != expected:
        fail(f"container contract mismatch: actual={actual}, expected={expected}")
    if abs(video_duration - args.duration) > 1.0 / float(args.fps):
        fail(f"video duration {video_duration:.9f}s differs from {args.duration:.9f}s")
    if abs(audio_duration - args.duration) > 1.0 / args.sample_rate:
        fail(f"audio duration {audio_duration:.9f}s differs from {args.duration:.9f}s")
    drift = abs(video_duration - audio_duration)
    if drift > args.max_av_drift:
        fail(f"A/V drift {drift:.9f}s exceeds {args.max_av_drift:.9f}s")
    format_data = document.get("format", {})
    names = str(format_data.get("format_name", "")).split(",")
    if "mp4" not in names:
        fail(f"container is not MP4: {format_data.get('format_name')}")
    print(
        "PASS h3 mux: "
        f"h264 {args.width}x{args.height} {args.frames}@{float(args.fps):g}, "
        f"aac {args.channels}ch/{args.sample_rate}Hz, "
        f"video={video_duration:.6f}s audio={audio_duration:.6f}s "
        f"drift={drift:.6f}s"
    )


if __name__ == "__main__":
    main()
