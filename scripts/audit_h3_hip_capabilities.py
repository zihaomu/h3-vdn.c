#!/usr/bin/env python3
"""Audit original MiniMax-H3 GPU APIs against the HIP backend.

This is intentionally a source audit rather than a runtime smoke test.  It
answers two questions before a multi-hour model run is attempted:

1. Which h3_gpu.h entry points are missing or explicit HIP stubs?
2. Which of those entry points are called by the original FL2VA/Ref2VA path?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


COMPONENT_SOURCES = {
    "dit": "h3_dit.c",
    "text_encoder": "h3_text_encoder.c",
    "vision_encoder": "h3_vision_encoder.c",
    "video_encoder": "h3_video_encoder.c",
    "video_vae": "h3_video_vae.c",
    "audio_vae": "h3_audio_vae.c",
    "multimodal": "h3_multimodal.c",
}

# These APIs are referenced by h3_dit.c but belong only to optional fast or
# approximate paths.  A BF16 reference run has an unfused/non-quantized
# fallback and therefore must not be blocked by these entries.
OPTIONAL_ACCELERATION_APIS = {
    "h3_gpu_gate_adaln_quantize_int8",
    "h3_gpu_grouped_qkv_linear_rope_int8",
    "h3_gpu_linear_int8_head_major_bf16",
    "h3_gpu_mlp_int8_bf16",
    "h3_gpu_mlp_nax_bf16",
    "h3_gpu_sdpa_bf16_head_major_output",
    "h3_gpu_token_expand_adaln_bf16",
    "h3_gpu_token_expand_delta_bf16",
    "h3_gpu_token_pool_adaln_bf16",
    "h3_gpu_token_pool_bf16",
}

FUNCTION_RE = re.compile(r"\b(h3_gpu_[A-Za-z0-9_]+)\s*\(")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="report original H3 API coverage in h3_gpu_hip.cpp"
    )
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--fail-on-reference-gaps",
        action="store_true",
        help="return 1 when a BF16 reference API is missing or stubbed",
    )
    return parser.parse_args()


def closing_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    state = "code"
    index = start
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return index
            elif char == '"':
                state = "string"
            elif char == "'":
                state = "character"
            elif char == "/" and next_char == "/":
                state = "line_comment"
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                index += 1
        elif state == "string":
            if char == "\\":
                index += 1
            elif char == '"':
                state = "code"
        elif state == "character":
            if char == "\\":
                index += 1
            elif char == "'":
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment" and char == "*" and next_char == "/":
            state = "code"
            index += 1
        index += 1
    return -1


def definition_body(source: str, name: str) -> tuple[str | None, int | None]:
    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", source):
        open_paren = source.find("(", match.start())
        close_paren = closing_delimiter(source, open_paren, "(", ")")
        if close_paren < 0:
            continue
        cursor = close_paren + 1
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or source[cursor] != "{":
            continue
        close_brace = closing_delimiter(source, cursor, "{", "}")
        if close_brace < 0:
            continue
        line = source.count("\n", 0, match.start()) + 1
        return source[cursor + 1 : close_brace], line
    return None, None


def caller_map(repo_root: Path) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    by_api: dict[str, set[str]] = {}
    line_details: dict[str, list[str]] = {}
    for component, relative in COMPONENT_SOURCES.items():
        path = repo_root / relative
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), 1):
            for name in FUNCTION_RE.findall(line):
                by_api.setdefault(name, set()).add(component)
                line_details.setdefault(name, []).append(f"{relative}:{line_number}")
    return by_api, line_details


def collect(repo_root: Path) -> dict[str, object]:
    header_path = repo_root / "h3_gpu.h"
    hip_path = repo_root / "h3_gpu_hip.cpp"
    header = header_path.read_text(encoding="utf-8")
    hip = hip_path.read_text(encoding="utf-8")
    declarations = sorted(set(FUNCTION_RE.findall(header)))
    callers, call_lines = caller_map(repo_root)

    entries = []
    for name in declarations:
        body, definition_line = definition_body(hip, name)
        if body is None:
            status = "missing"
        elif "h3_gpu_unsupported(" in body:
            status = "unsupported_stub"
        else:
            status = "implemented"
        components = sorted(callers.get(name, set()))
        original_path = bool(components)
        optional = name in OPTIONAL_ACCELERATION_APIS
        reference_required = original_path and not optional
        entries.append(
            {
                "api": name,
                "status": status,
                "hip_definition_line": definition_line,
                "original_components": components,
                "call_sites": call_lines.get(name, []),
                "optional_acceleration": optional,
                "reference_required": reference_required,
            }
        )

    gaps = [
        entry
        for entry in entries
        if entry["reference_required"] and entry["status"] != "implemented"
    ]
    optional_gaps = [
        entry
        for entry in entries
        if entry["optional_acceleration"] and entry["status"] != "implemented"
    ]
    original_entries = [entry for entry in entries if entry["original_components"]]
    return {
        "schema": "h3.hip-capability-audit.v1",
        "header": str(header_path.relative_to(repo_root)),
        "backend": str(hip_path.relative_to(repo_root)),
        "declared_api_count": len(entries),
        "original_path_api_count": len(original_entries),
        "reference_gap_count": len(gaps),
        "optional_acceleration_gap_count": len(optional_gaps),
        "reference_ready": not gaps,
        "reference_gaps": gaps,
        "optional_acceleration_gaps": optional_gaps,
        "apis": entries,
    }


def markdown(report: dict[str, object]) -> str:
    gaps = report["reference_gaps"]
    optional = report["optional_acceleration_gaps"]
    lines = [
        "# Original H3 HIP capability audit",
        "",
        f"- Declared GPU APIs: {report['declared_api_count']}",
        f"- APIs referenced by original H3 sources: {report['original_path_api_count']}",
        f"- BF16 reference gaps: {report['reference_gap_count']}",
        f"- Optional acceleration gaps: {report['optional_acceleration_gap_count']}",
        f"- Reference ready: `{'yes' if report['reference_ready'] else 'no'}`",
        "",
        "## BF16 reference gaps",
        "",
        "| API | HIP status | Component | Call sites |",
        "|---|---|---|---|",
    ]
    if gaps:
        for entry in gaps:
            lines.append(
                "| `{}` | `{}` | {} | {} |".format(
                    entry["api"],
                    entry["status"],
                    ", ".join(entry["original_components"]),
                    "<br>".join(f"`{site}`" for site in entry["call_sites"]),
                )
            )
    else:
        lines.append("| _none_ | `implemented` | - | - |")

    lines.extend(
        [
            "",
            "## Optional acceleration gaps",
            "",
            "These do not block the BF16 reference profile.",
            "",
            "| API | HIP status | Component |",
            "|---|---|---|",
        ]
    )
    if optional:
        for entry in optional:
            lines.append(
                "| `{}` | `{}` | {} |".format(
                    entry["api"],
                    entry["status"],
                    ", ".join(entry["original_components"]),
                )
            )
    else:
        lines.append("| _none_ | `implemented` | - |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    try:
        report = collect(repo_root)
    except (OSError, UnicodeError) as error:
        print(f"audit error: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(markdown(report))
    if args.fail_on_reference_gaps and not report["reference_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
