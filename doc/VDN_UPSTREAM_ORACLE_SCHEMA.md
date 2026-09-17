# OpenVDN upstream oracle fixture schema

Status: input through streamed eight-NFE and dual-VAE scopes implemented

Schema identifier: `h3-vdn-upstream-oracle-v1`
Created: 2026-09-16

## Purpose

These fixtures make the upstream OpenVDN PyTorch implementation, rather than a
second C formula, the source of truth for semantic recovery. The first scope
freezes prompt bytes, packed layout, explicit random inputs, and the paired
video/audio scheduler. Later scopes will add prompt-refiner, hybrid-block,
final-head, Euler, and full denoising activations without changing the input
contract.

The fixture tensors are authoritative. A consumer must not recreate random
inputs from `seed`, because CPU, CUDA, and HIP generators do not promise the
same byte stream.

## Provenance contract

Every safetensors file contains string metadata for:

- schema and scope;
- OpenVDN and patched Diffusers commits;
- Torch and HIP versions;
- source prompt path and SHA-256;
- geometry, NFE, shifts, seed, and RNG policy.

A sibling `.safetensors.json` manifest records the fixture SHA-256 plus dtype,
shape, and raw-byte SHA-256 for every tensor. A fixture without this provenance
must not be promoted into a correctness gate.

## Initial scope: `prompt-layout-input-schedule`

| Tensor | Dtype | Meaning |
|---|---|---|
| `prompt.embeds` | BF16 | Official `[L,5120]` prompt hidden state |
| `prompt.text_token_tags` | I64 | Upstream text tags `[L]` |
| `layout.position_ids` | F64 | Upstream packed `[S,3]` positions |
| `layout.token_tags` | I64 | Packed modality tags `[S]` |
| `layout.video_indices` | I64 | Contiguous target video rows |
| `layout.audio_indices` | I64 | Target audio rows |
| `layout.text_indices` | I64 | Prompt rows |
| `layout.rope_cos_half_bf16` | BF16 | First 48 columns of upstream repeated RoPE cosine table |
| `layout.rope_sin_half_bf16` | BF16 | First 48 columns of upstream repeated RoPE sine table |
| `input.video_latents` | F32 | Explicit unpatched `[1,24,T,H,W]` noise |
| `input.video_rows` | F32 | Upstream patchified `[Tv*(H/2)*(W/2),96]` input |
| `input.audio_rows` | F32 | Explicit `[2*Ta,32]` audio input |
| `schedule.video_sigmas` | F32 | Video sigma grid, terminal zero included |
| `schedule.audio_sigmas` | F32 | Audio sigma grid, terminal zero included |
| `schedule.video_timesteps` | F32 | Model video timesteps `1-sigma` |
| `schedule.audio_timesteps` | F32 | Model audio timesteps `1-sigma` |
| `schedule.video_euler_scales` | F32 | Native additive velocity scale per NFE |
| `schedule.audio_euler_scales` | F32 | Native additive velocity scale per NFE |
| `schedule.nfe_N.unique_timesteps` | F32 | Sorted unique modality timesteps for NFE N |
| `schedule.nfe_N.timestep_indices` | I64 | Packed row to unique-timestep map |

The initial small parity geometry is `T=17, H=2, W=4, audio=3`. It is a math
fixture, not a quality workload. Official semantic acceptance remains
`T=102, H=48, W=84, 345 frames`.

## Planned activation scopes

1. `prompt-refiner`: refiner inputs, both block outputs, final norm.
2. `hybrid-block-0`: AdaLN, Q/K/V, QK norm/RoPE, window branch, linear
   features, solve, scans, readout, branch merge, attention residual, and MLP.
3. `transformer-forward`: selected block boundaries plus final modulation and
   video/audio velocities.
4. `single-euler`: identical input rows, velocities, and post-step rows.
5. `eight-nfe-latent`: final patched and unpatched normalized latents.
6. `vae-closure`: decoded video F32/RGB24 and audio F32/PCM from identical
   upstream latents.

Each new scope must preserve the same provenance fields and tensor-hash rules.

## Export command

From the H3 repository after installing Torch ROCm and the pinned patched
Diffusers checkout:

```sh
HIP_VISIBLE_DEVICES=4 .venv/bin/python scripts/export_vdn_upstream_oracle.py \
  --upstream-dir ../vdn-minimax-h3-upstream \
  --prompt ../vdn-minimax-h3-upstream/prompts/example_0.pt \
  --output misc/fixtures/vdn_upstream_input_small.safetensors
```

This initial scope performs CPU tensor construction and does not submit GPU
work. `HIP_VISIBLE_DEVICES=4` is still specified so all later scopes inherit the
single-device contract.

## Real-weight prompt-refiner scope

The second exporter materializes only the released context projection and two
token-refiner blocks, then applies the Stage-DMD default and turbo adapters with
the same upstream merge implementation. It captures the first block at each
RMSNorm, Q/K/V projection, QK normalization, attention output/residual, fused
FFN projection, SwiGLU, FFN output and residual, plus block 1 and final outputs.
This is a GPU workload and must run through the physical-GPU-4 guard:

```sh
scripts/profile_vdn_gpu4.sh outputs/oracle-refiner-gpu4 -- \
  .venv/bin/python scripts/export_vdn_upstream_refiner_oracle.py \
  --upstream-dir ../vdn-minimax-h3-upstream \
  --model-root models/vdn-minimax-h3 \
  --prompt models/vdn-minimax-h3/prompts/example_0.safetensors \
  --checkpoint stage-dmd-step-250 \
  --attention-backend _native_math \
  --output misc/fixtures/vdn_upstream_refiner_stage_dmd_example0.safetensors
```

The reference-math fixture is compared with a separate upstream `native`
attention export before setting C/HIP thresholds. For the current pinned
environment, the two correct backends differ at the final refiner output by
relative RMSE `0.00438570` and cosine `0.999990383`; the C gate is fixed at
relative RMSE `<= 0.01`, cosine `>= 0.9999`, max absolute error `<= 2`, and no
non-finite values.

## Real-weight hybrid block-0 scope

`export_vdn_upstream_block0_oracle.py` materializes one transformed Stage-DMD
block and runs the eager hybrid algorithm with the small deterministic packed
fixture. The MATH reference command is:

```sh
scripts/profile_vdn_gpu4.sh outputs/oracle-block0-gpu4 -- \
  .venv/bin/python scripts/export_vdn_upstream_block0_oracle.py \
  --upstream-dir ../vdn-minimax-h3-upstream \
  --model-root models/vdn-minimax-h3 \
  --input-fixture misc/fixtures/vdn_upstream_input_small.safetensors \
  --refiner-fixture misc/fixtures/vdn_upstream_refiner_stage_dmd_example0.safetensors \
  --sdpa-backend math \
  --output misc/fixtures/vdn_upstream_block0_stage_dmd_small.safetensors
```

The fixture captures both input preparation and block substages. A second
`--sdpa-backend native` export establishes the legitimate upstream block-final
variation (relative RMSE `0.00164886`, cosine `0.999998642`) before testing C.
The C block-final gate is relative RMSE `<= 0.01`, cosine `>= 0.9999`, and no
non-finite values.

## Streamed 50-layer forward scope

`export_vdn_upstream_forward_oracle.py` runs all 50 real Stage-DMD blocks while
keeping only one block resident on the GPU. It applies the per-layer default
and turbo LoRA pairs, stores every BF16 block boundary, and evaluates the final
AdaLN plus video/audio heads. Run it on the guarded physical GPU 4:

```sh
scripts/profile_vdn_gpu4.sh outputs/oracle-forward50-gpu4 -- \
  .venv/bin/python scripts/export_vdn_upstream_forward_oracle.py \
  --upstream-dir ../vdn-minimax-h3-upstream \
  --model-root models/vdn-minimax-h3 \
  --block0-fixture misc/fixtures/vdn_upstream_block0_stage_dmd_small.safetensors \
  --sdpa-backend math \
  --output misc/fixtures/vdn_upstream_forward50_stage_dmd_small.safetensors
```

The scope adds `block_00.output` through `block_49.output`,
`final.normalized`, `final.video_velocity`, and `final.audio_velocity`. The
MATH fixture SHA-256 is
`244d85f6c4c0cbce5a5dcc6e4180224f65c7d68f2325479a5704e18fba0feb97`.
The corresponding native-SDPA fixture SHA-256 is
`c1ece14cdb4ef7a3915a69ffff4fd277ad4774b21d1f6c691bf592723554d8bc`.

MATH/native upstream divergence reaches relative RMSE `0.08449` and cosine
`0.996919` at the deepest block, while final video/audio velocity relative
RMSE is `0.02472` / `0.05698`. Before running C, the scalar C gate was fixed at
block relative RMSE `<=0.15`, block cosine `>=0.99`, video relative RMSE
`<=0.05` / cosine `>=0.995`, audio relative RMSE `<=0.12` / cosine `>=0.99`,
and zero non-finite values. Run `make vdn-forward-oracle-test` after generating
the local fixture.

## Streamed eight-NFE latent scope

`export_vdn_upstream_denoise_oracle.py` retains only the small input/time/final
heads and streams one real block at a time for every NFE. Each step uses the
official `MiniMaxH3Scheduler.step()` independently for video shift 12 and audio
shift 3. It records both velocity outputs and post-Euler latent rows:

```sh
scripts/profile_vdn_gpu4.sh outputs/oracle-denoise8-gpu4 -- \
  .venv/bin/python scripts/export_vdn_upstream_denoise_oracle.py \
  --upstream-dir ../vdn-minimax-h3-upstream \
  --model-root models/vdn-minimax-h3 \
  --input-fixture misc/fixtures/vdn_upstream_input_small.safetensors \
  --block0-fixture misc/fixtures/vdn_upstream_block0_stage_dmd_small.safetensors \
  --sdpa-backend math \
  --output misc/fixtures/vdn_upstream_denoise8_stage_dmd_small.safetensors
```

The MATH and native fixture SHA-256 values are respectively
`2d894ba0173954ffc6f72bb091b76cb191eb674139c3a10cf91afeff13c578c4` and
`413314aee889acd4e6b31b6b08a1c5c289978a243ac62b3c8d5f1b0e8634c4f4`.
Their final video/audio latent-row relative RMSE values are `0.0340827` and
`0.0150450`. The pre-registered C gates are video relative RMSE `<=0.08`,
audio `<=0.05`, cosine `>=0.995`, and no non-finite values. Native C validation
uses `h3_vdn_denoise_observed` and `make vdn-denoise-oracle-test`.

## Dual-VAE closure scope

`export_vdn_upstream_vae_oracle.py` unpatches the verified MATH eight-NFE rows,
applies the released per-channel latent normalization, and runs the upstream
video VAE with FP16 autocast and the audio VAE in F32. The fixture contains the
normalized and decoder-input latents plus final RGB F32 `[56,32,64,3]` and PCM
F32 `[2,2400]`:

```sh
scripts/profile_vdn_gpu4.sh outputs/oracle-dual-vae-gpu4 -- \
  .venv/bin/python scripts/export_vdn_upstream_vae_oracle.py \
  --denoise-fixture misc/fixtures/vdn_upstream_denoise8_stage_dmd_small.safetensors \
  --output misc/fixtures/vdn_upstream_dual_vae_stage_dmd_small.safetensors
```

The fixture SHA-256 is
`73fca1e1ee69f893149986f142c953431d1fb766615f0b18e7d5b19d5182a552`.
`make vdn-vae-oracle-test` feeds the same normalized values to both C decoders.
The test explicitly converts upstream audio `[stereo,channel,time]` into the C
API's documented `[channel,stereo,time]` layout. Current C results are video
relative RMSE `0.00101465`, cosine `0.999999488`, and audio relative RMSE
`5.85675e-05`, cosine `0.999999998`, with no non-finite values.
