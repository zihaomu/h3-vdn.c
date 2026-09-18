# MiniMax-H3 Sage E33 640×384 experimental baseline

Status: `CODE_FROZEN_KEEP_EXPERIMENTAL`

Model: [`MiniMaxAI/MiniMax-H3` revision
`42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)

This document freezes the quality and end-to-end timing evidence for the first
formal Stage 3 SageAttention-AMD E33 render. It is a benchmark for subsequent
optimization, not a claim that Sage E33 is the default or release-quality
attention backend.

Only the implementation, submodule pin, tests, README, and this textual
contract are tracked. The model, log, frames, contact sheet, PCM, and MP4 stay
local and are not committed.

## Frozen case

| Field | Value |
|---|---|
| Parent source | `65a6b350d24bcc8af54ed0b035c70011e8f64224` |
| SageAttention-AMD pin | `133b53168e3e8f9a3a059cd31aac7e69d802e89e` |
| Attention | explicit `H3_BF16_SDPA=sage-e33` |
| Resident DiT blocks | `20` |
| Prompt | `A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind.` |
| Seed | `42` |
| Geometry | `640×384`, 124 frames, 24 fps |
| Schedule | 50 requested and observed NFEs, 50 DiT blocks, reuse 1 |
| Precision | original BF16 Transformer; E33 BF16 Q/K/V with F32 online softmax/accumulation |
| Device | one `gfx1201`, physical card 2, PCI `0000:43:00.0` |
| Peak device allocation | `18.474 GiB` |

## Output and quality

The local MP4 is H.264 High at `640×384`, 124 frames at 24 fps, with AAC-LC
stereo audio at 32 kHz. The video stream is `5.166667 s`; the muxed container
duration is `5.175000 s`. Its size is `1,191,837 bytes` and its SHA-256 is:

```text
8abdf79b11886af448b428f1350f9592c4a592f896b1006f33da85d21ad37bc9
```

The media and manual quality gates produced the following evidence:

- all 124 decoded frames have unique content hashes;
- the ordered raw-frame aggregate SHA-256 is
  `9abf7a47a69a904c1c64035210d6485e9d3da0db69372826be00399f6ad1dd72`;
- the three formal E2E runs produced identical raw frames and identical MP4
  hashes;
- manual contact-sheet review found one coherent fox, snow, pine forest,
  continuous motion and lighting, with no checkerboard, color blocks, flicker,
  or subject collapse;
- decoded audio is non-silent, with mean/max levels of `-41.7/-15.7 dB`.

This visual pass does **not** satisfy the frozen matrix-render parity gate.
Full-video SSIM/PSNR against that render are `0.378892 / 13.944662 dB`, below
the immutable promotion threshold of `0.933314 / 31.089467 dB`. The subject
position and pose drift over 50 denoising passes. Therefore Sage E33 remains
an explicit experimental backend and `auto` continues to use rocBLAS.

## End-to-end timing

No rerun was used to establish this result. The original `run.log` timestamps
span `2026-09-18 10:38:28.422 +08:00` through
`2026-09-18 10:51:09.248 +08:00`, or `760.826 s`, reported as `761 s`
(`12:41`) end to end.

| Logged phase | Wall time |
|---|---:|
| Qwen text encoder | `4.010 s` |
| DiT initial load | `5.591 s` |
| 50-NFE Euler denoise | `502.222 s` |
| Audio VAE decode | `3.494 s` |
| Video VAE decode | `241.430 s` |
| Remaining setup, frame output, mux, and logging | approximately `4.079 s` |
| **Complete process** | **`760.826 s` (`12:41`)** |

Within denoising, cumulative Sage SDPA GPU-event time was `180.267 s`.
The DiT streamed `1,077.378 GiB` in `359.642 s` and recorded `150.043 s` of
unhidden wait. This distinguishes the complete generation wall time from the
attention-kernel-only measurement.

The same frozen input completed three times in `761 / 772 / 774 s`; the median
was `772 s` (`12:52`). Compared with the frozen rocBLAS matrix E2E baseline of
`849.46 s` (`14:09.46`), the first run was `10.41%` faster and the three-run
median was `9.12%` faster.

At `5.175 s` of delivered media, the first run operated at approximately
`147×` slower than real time.

## Local evidence boundary

The reviewed local artifact was:

```text
/tmp/h3-sage-stage3-frozen-run1.4lpYag/minimax-h3-sage-e33-50nfe.mp4
```

That path is intentionally ephemeral and is recorded only to identify the
reviewed run. Reproduction and future comparisons must rely on the frozen
model revision, source commits, input contract, hashes, and metrics above;
they must not require this `/tmp` directory to exist.

## Optimization comparison rule

Future performance claims must keep the same model revision, prompt, seed,
geometry, frame count, NFE count, block count, reuse factor, single-GPU scope,
and output checks. Report both complete process wall time and phase timing.
An optimization may replace this performance baseline only when it also
passes the existing tensor, media, determinism, and quality gates; wall-time
improvement alone is insufficient.
