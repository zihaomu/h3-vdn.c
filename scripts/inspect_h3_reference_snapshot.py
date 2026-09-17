#!/usr/bin/env python3
"""Inspect the pinned official MiniMax-H3 snapshot before downloading it.

The official repository is much larger than the free space on many developer
machines.  This helper uses the public Hugging Face tree API, reports logical
and content-addressed sizes, and can fail before ``hf download`` starts.
It never downloads checkpoint payloads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


DEFAULT_REPO = "MiniMaxAI/MiniMax-H3"
PINNED_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
GIB = 1024 ** 3


def fail(message: str) -> "None":
    raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Report official MiniMax-H3 snapshot sizes and disk fit"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument(
        "--scope",
        choices=(
            "metadata", "audio-vae", "video-vae", "text-encoder",
            "transformer", "fl2va", "ref2va", "both",
        ),
        default="both",
    )
    parser.add_argument("--destination", type=Path, default=repo / "MiniMax-H3")
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--require-fits", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args()


def fetch_tree(repo_id: str, revision: str) -> list[dict]:
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = (
        f"https://huggingface.co/api/models/{encoded_repo}/tree/"
        f"{encoded_revision}?recursive=true&limit=1000"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "h3.c-reference-planner/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except Exception as exc:
        fail(f"cannot read Hugging Face tree API: {exc}")
    if not isinstance(result, list):
        fail("Hugging Face tree API returned an unexpected document")
    files = [item for item in result if item.get("type") == "file"]
    if len(result) >= 1000:
        fail("snapshot tree reached the API page limit; pagination is required")
    if not files:
        fail("snapshot tree contains no files")
    return files


def selected(path: str, scope: str) -> bool:
    if scope == "metadata":
        return not path.endswith(".safetensors")
    families = ("FL2VA", "Ref2VA") if scope == "both" else (
        ("FL2VA",) if scope != "ref2va" else ("Ref2VA",)
    )
    if scope in ("fl2va", "ref2va", "both"):
        return any(path.startswith(f"{family}/") for family in families)
    component = {
        "audio-vae": "audio_vae",
        "video-vae": "video_vae",
        "text-encoder": "text_encoder",
        "transformer": "transformer",
    }[scope]
    return any(path.startswith(f"{family}/{component}/") for family in families)


def main() -> None:
    args = parse_args()
    if args.reserve_gib < 0:
        fail("--reserve-gib cannot be negative")
    files = fetch_tree(args.repo_id, args.revision)
    chosen = [item for item in files if selected(str(item.get("path", "")), args.scope)]
    if not chosen:
        fail(f"scope {args.scope!r} selected no files")

    logical_bytes = sum(int(item.get("size", 0)) for item in chosen)
    unique = {}
    for item in chosen:
        oid = str(item.get("oid") or item.get("path"))
        unique.setdefault(oid, item)
    unique_bytes = sum(int(item.get("size", 0)) for item in unique.values())
    by_component: dict[str, int] = defaultdict(int)
    for item in chosen:
        parts = str(item.get("path", "")).split("/")
        key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        by_component[key] += int(item.get("size", 0))

    probe = args.destination
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    reserve_bytes = int(args.reserve_gib * GIB)
    fits = unique_bytes + reserve_bytes <= free_bytes
    report = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "scope": args.scope,
        "destination": str(args.destination.resolve()),
        "file_count": len(chosen),
        "logical_bytes": logical_bytes,
        "unique_oid_bytes": unique_bytes,
        "free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "fits": fits,
        "components": dict(sorted(by_component.items())),
    }
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Official MiniMax-H3 snapshot plan")
        print(f"  repository       {args.repo_id}")
        print(f"  revision         {args.revision}")
        print(f"  scope            {args.scope}")
        print(f"  destination      {args.destination.resolve()}")
        print(f"  selected files   {len(chosen)}")
        print(f"  logical size     {logical_bytes / GIB:.3f} GiB")
        print(f"  unique OID size  {unique_bytes / GIB:.3f} GiB")
        print(f"  free space       {free_bytes / GIB:.3f} GiB")
        print(f"  safety reserve   {reserve_bytes / GIB:.3f} GiB")
        print(f"  fits             {'yes' if fits else 'NO'}")
        print("  components")
        for name, size in sorted(by_component.items()):
            print(f"    {name:<24} {size / GIB:8.3f} GiB")
    if args.require_fits and not fits:
        sys.exit(1)


if __name__ == "__main__":
    main()
