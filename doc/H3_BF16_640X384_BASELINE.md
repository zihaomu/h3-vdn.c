# MiniMax-H3 upstream BF16 640×384 frozen baseline

Status: `CODE_FROZEN_HUMAN_ACCEPTED`
Model: [`MiniMaxAI/MiniMax-H3` revision
`42ed227ee7df40d41602854ae760620d6eb651fe`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)

This document freezes the code and reproducibility contract for the accepted
upstream MiniMax-H3 T2VA baseline. It is intentionally independent of the
native C/HIP and OpenVDN execution paths. The user visually accepted the
640×384 MP4 and five-frame contact sheet on 2026-09-17.

No model shard, generated video, image, audio, latent, first-NFE tensor, or
other file under `MiniMax-H3/`, `models/`, `misc/`, or `outputs/` is committed.
Those directories remain local and ignored by Git. The commit freezes only
the implementation, tests, reproduction scripts, upstream source patches, and
this textual contract.

## Frozen case

| Field | Value |
|---|---|
| Prompt | `A red fox walks slowly through a snowy pine forest, cinematic natural light, detailed fur` |
| Seed | `42` |
| Geometry | `640×384`, 124 frames, 24 fps |
| Schedule | 50 requested steps, 50 observed Transformer NFEs |
| Precision | released mixed BF16/F32 checkpoint; BF16 Transformer |
| Attention | Diffusers `native` SDPA |
| Video VAE | released spatial tiling |
| Device | one `gfx1201`, physical ordinal 1, PCI `0000:43:00.0` |
| Pipeline wall | `345.6100926749641 s` |
| Torch peak VRAM | `16.196913719177246 GiB` |

## Frozen code provenance

The accepted run used the local Diffusers source at
`c41d51befdaef751cbd98f07a8eb4f4279d1578c`, based on upstream revision
`3a2f35d4efa4c059c8bfb3bc0d6c906264895c81`. The two MiniMax-H3-specific
changes are committed as source patches under
`third_party/patches/diffusers-minimax-h3/`:

1. force the AdaLN SiLU inputs to F32 before the released BF16 projection;
2. make requested inference steps count actual Transformer NFEs.

The reference and diagnosis entry points committed with this contract are:

- `scripts/run_h3_diffusers_reference.py`;
- `scripts/run_h3_artifact_diagnosis.py`;
- `scripts/decode_h3_saved_video_latent.py`;
- `scripts/profile_vdn_gpu4.sh`.

The model identity is frozen by the official repository URL and immutable
revision above. The local `modular_model_index.json` SHA-256 used by the run was
`a2b6a210e482ffb78e613b553f570c44e101afce6741bd4ed91429d0559af031`;
the Transformer safetensors index SHA-256 was
`ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e`.
No model bytes are mirrored by this repository.

## Reproduction

Apply the two committed Diffusers patches to the pinned base revision, install
that source into the local environment, obtain the official model revision at
the link above, and run on one guarded GPU:

```sh
H3_PHYSICAL_GPU=1 PYTHONPATH=.text-oracle-deps \
  scripts/profile_vdn_gpu4.sh outputs/h3-bf16-baseline-gpu1 -- \
  .venv/bin/python scripts/run_h3_artifact_diagnosis.py \
  --model-dir MiniMax-H3 \
  --output-dir outputs/h3-bf16-baseline-640x384 \
  --width 320 --height 192 --probe-width 640 --probe-height 384 \
  --frames 124 --steps 50 --seed 42
```

The frozen accepted local evidence remains outside Git at
`outputs/h3-artifact-diagnosis-20260917-r4/native-tiled-640x384/`:

| Local evidence | SHA-256 |
|---|---|
| `first-nfe.safetensors` | `7685b680a05c66d0c6bb4b12cc00a9b8c67059aed18a07aeefda6c2a75ee3da7` |
| `final-latents.safetensors` | `827e2b859eccf15d52882773f10849d2516e42ac673560aff4b99932637952eb` |
| `minimax-h3-reference.mp4` | `fa179519f8aeb4587254c914f8c1b7867f123eada2ddd29807ffd39f42faa50b` |
| `contact.png` | `0dca137d1e4eadde076b52ba35291bfff192b0086e40089010038d95d3e59762` |

The tensor and latent bytes are the numerical gates. The MP4 is a visual and
container acceptance artifact, not a replacement for tensor comparison.

## Scope boundary

This baseline proves that the pinned official checkpoint and pinned Diffusers
implementation produce an accepted BF16 result on ROCm. It does not yet prove
native C/HIP parity, VDN semantic correctness, or stable-release readiness.
The next gate is native modern 14-shard loader support followed by block-0,
50-layer, single-NFE, final-latent, raw RGB/PCM, and media parity against this
fixture.
