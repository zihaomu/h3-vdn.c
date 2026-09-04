# OpenVDN ROCm v0.1.0 Stable Release

Validation date: 2026-09-04  
Release branch: `vdn-h3-rocm`  
Upstream OpenVDN code: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`  
Model revision: `18be6bcc4ee72585eee322ba28b5ccac2cf85ef0`

## Supported scope

This release runs OpenVDN `stage-dmd-step-250` end to end on one selected AMD
Radeon AI PRO R9700 (`gfx1201`) with ROCm 7.2.3. It merges the default and turbo
adapters, evaluates all 50 hybrid-attention blocks for each of 8 NFE, decodes
video and stereo audio, and muxes an MP4.

The stable VDN prompt API is `--prompt-embeds` using one safetensors file with
exactly two tensors: BF16 `prompt_embeds[L,5120]` and I64 `token_tags[L]`. The
no-unpickle converter supports all three upstream examples (`L=800`, `821`, and
`1299`) and output from the pinned upstream `encode_prompt.py`.

## Build and run

```sh
make clean
make BACKEND=hip HIP_ARCHS=gfx1201 -j16

HIP_VISIBLE_DEVICES=4 ./h3 \
  --model-dir models/vdn-minimax-h3/h3-base \
  --vdn-checkpoint models/vdn-minimax-h3/stage-dmd-step-250 \
  --prompt-embeds models/vdn-minimax-h3/prompts/example_0.safetensors \
  --width 512 --height 512 --frames 56 --seed 0 \
  --output outputs/vdn-stable.mp4
```

`HIP_VISIBLE_DEVICES=4` above records the validation machine's physical-device
selection. On another machine, select exactly the intended GPU and use its
logical device 0 inside the process.

## Release evidence

| Gate | Result |
|---|---|
| Clean `gfx1201` build | Pass, no compiler warning/error |
| Host/API | 1774 checks plus JSON, metadata, reference, prompt, and fail-fast input-contract suites pass |
| GPU operators | backend/storage/general DiT and VDN ops/features/solve/scan pass; scalar fallback also passes |
| Real loader | base + default + turbo parity passes with staging cache enabled and disabled; no live allocation leak |
| Production single NFE | sequence 5338; video `b3d3500676d3fb12`; audio `5fbd7afb3d78a277`; peak 4.969 GiB |
| Prompt lengths | 800/821/1299 convert and load; 821/1299 refine; 821 completes all 50 blocks |
| Production E2E | two consecutive 512×512, 56-frame, 8-NFE runs pass and are byte-identical |
| Release-rehearsal E2E | clean-build 64×32 fixture passes 8 NFE, both VAEs, and mux; exact historical SHA retained |

The two production runs took 486.49 and 487.20 seconds. Their internal hashes
were video rows `e77bfd64f14b695c`, audio rows `bdde023376238608`, decoded video
F32 `127cba9e701bb2c4`, audio PCM F32 `5cfe75130b41efad`, and RGB24
`0338db1c620814e3`. Both 2,315,918-byte MP4 files have SHA-256
`ee267508d2c988629811ce86db8d6ac7a1a8291957b792583348dc0be90eea43`.

The clean-build smoke artifact is 73,528 bytes with SHA-256
`7a447fe6f63697ad1bbb2df8a74f385b432ce3a1d5855caee5f8c1531a955c5b`.
FFprobe reports 56 H.264 frames at 64×32 and 24 fps plus stereo AAC at 32 kHz.

The historical one-off old lane-0 hash anomaly did not reproduce in a 26-run
matrix: distributed/lane-0 attention each passed 10/10 with pipelined staging
and 3/3 with serial staging. Kernel logs showed no reset/fault/ECC report and
GPU 4 UMC RAS counters were 0/0.

## Platform matrix

| Target | Release status |
|---|---|
| R9700 / `gfx1201`, ROCm 7.2.3 | Supported and runtime tested |
| `gfx90a`, `gfx942`, `gfx1030`, `gfx1100`, `gfx1151` | Compile-only; runtime experimental/unvalidated |
| Other ROCm targets | Unsupported |

The build defaults to `HIP_ARCHS=gfx1201`; multiple code objects may be built
with a space-separated list. Compilation alone is not a support claim.
`H3_VDN_SCALAR_SDPA=1` retains the scalar correctness fallback.

## Intentional limitations

- One selected GPU only; no multi-GPU layer sharding.
- `stage-dmd-step-250` with exactly 8 NFE is the end-to-end stable checkpoint.
  `stage-b-step-2000` has metadata validation but no 50-NFE E2E acceptance.
- Raw `-p` text is not encoded inside this binary. The VDN download omits its
  approximately 62 GB Qwen3-VL-32B processor/text encoder, so raw text is first
  encoded with the pinned upstream environment and then converted.
- OpenVDN upstream inference defines no first/last-frame or ordered-media
  conditioning. Those flags fail before large weights load and direct users to
  the separate MiniMax-H3 FL2VA/Ref2VA checkpoints.
- VDN denoising previews are unsupported.

These boundaries are enforced with nonzero, actionable failures; unsupported
native MiniMax-H3 features are not silently ignored or mapped to incompatible
VDN semantics.
