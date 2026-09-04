# OpenVDN ROCm optimization ledger

This document is the repository-local performance ledger for the
`vdn-h3-rocm` branch. Correctness remains the first gate: a candidate is kept
only after the VDN operator tests pass and a complete 50-layer forward is
compared with the scalar BF16 path.

## Fixed inputs

- Target start: `3cc97ec5f0c880486bdbb85335ba5d44ca5df562`
- GPU: AMD Radeon AI PRO R9700, `gfx1201`, wave32
- ROCm: 7.2
- Checkpoint: OpenVDN `stage-dmd-step-250`, default and turbo adapters
- Reference implementation: `alexhegit/h3-hip.c` at
  `377ad3698d9d82fdb9ab5a22afc9fdc2c521adc3`
- Microbenchmark: 17 latent frames, 2x4 latent canvas, 6 audio rows,
  sequence 840, all 50 blocks, GPU 0

The reference implementation is used to identify optimization techniques, not
as a source of performance claims for this model. Its FD cache, parallel
`pread`, event profiling, reusable workspaces, wave32 SDPA, INT8, and GPU
sampler must each pass an independent A/B on R9700 before adoption.

## Baseline

The initial HIP `--profile` hooks were no-ops. Event-based timing now records
linear, SDPA, VDN solve, and VDN scan GPU time after stream synchronization,
while load timing records bytes and the separate `pread` and H2D intervals.
Profiling is enabled only by `H3_PROFILE=1` or `--profile`.

Initial complete-forward profile:

| Metric | Baseline |
|---|---:|
| Wall inside HIP context | 18.60 s |
| Measured GPU classes | 7.710 s |
| Window/full SDPA | 6.275 s (81.4%) |
| Linear | 0.981 s (12.7%) |
| VDN solve | 0.352 s (4.6%) |
| VDN scan | 0.102 s (1.3%) |
| Weight read | 66.818 GiB / 4.496 s |
| Weight H2D | 66.818 GiB / 2.858 s |
| Peak live GPU allocation | 3.438 GiB |

Scalar output hashes are `d8b9ef8c887aa44c` for video and
`0be36a1038984ac9` for audio.

## Results

### KEEP: wave32 VDN window SDPA

One wave now owns a query/head pair. Each lane retains four output dimensions
for the production D=128 shape, and wave shuffles replace repeated
256-thread-wide barriers. The D=128 dot product deliberately follows the old
reduction tree, `(d0+d64)+(d32+d96)` followed by 16/8/4/2/1 shuffle steps.

| Metric | Scalar | wave32 | Change |
|---|---:|---:|---:|
| 50-layer wall, median of 3 | 19.12 s | 13.41 s | -29.9% |
| SDPA GPU time | 6.275 s | 1.045 s | -83.3% |
| Peak live allocation | 3.438 GiB | 3.438 GiB | unchanged |

The same-process full-forward comparison reports `max_abs=0`, `RMSE=0`, and
identical video/audio hashes. `H3_VDN_SCALAR_SDPA=1` retains the oracle and
diagnostic fallback. Devices that are not wave32, and head dimensions above
256, automatically use the scalar implementation.

The final 8-NFE latent test completed all 400 real transformer blocks in 94.37
seconds with video hash `356f91a5a7163d3b`, audio hash
`ff1eacbc4bd3584a`, and a 3.438 GiB peak. The full VAE/mux acceptance completed
in 141.72 seconds and reproduced the existing MP4 SHA-256 exactly:
`7a447fe6f63697ad1bbb2df8a74f385b432ce3a1d5855caee5f8c1531a955c5b`.
The container contains 56 H.264 frames at 64x32/24 fps and stereo AAC at
32 kHz.

### REJECT: two-buffer weight upload pipeline

A two-buffer 8 MiB pipeline overlapped `pread` with H2D copies and preserved
both output hashes, but wall time regressed from 18.04 s to 19.33 s (+7.2%).
Per-tensor event and pinned-buffer lifecycle increased system time. The code was
removed. A future I/O attempt must use persistent staging/FD caches and report
cold and warm cache separately.

### REJECT: combined solver-info synchronization

Combining POTRF/POTRI info readback reduced two synchronization points to one,
but crossed profiles were indistinguishable: 12.637 s deferred versus 12.599 s
eager, and solve GPU time was 0.369 versus 0.373 s. It also weakened failure
ordering by submitting POTRI before confirming POTRF. The code was removed.

## Test gates

Build and run the local gates with:

```sh
make BACKEND=hip -j16 \
  h3_vdn_gpu_ops_tests h3_vdn_feature_tests \
  h3_vdn_solve_tests h3_vdn_scan_tests h3_vdn_forward_smoke_tests

HIP_VISIBLE_DEVICES=0 ./h3_vdn_gpu_ops_tests
HIP_VISIBLE_DEVICES=0 ./h3_vdn_feature_tests
HIP_VISIBLE_DEVICES=0 ./h3_vdn_solve_tests
HIP_VISIBLE_DEVICES=0 ./h3_vdn_scan_tests

HIP_VISIBLE_DEVICES=0 VDN_SMOKE_COMPARE_SDPA=1 \
  ./h3_vdn_forward_smoke_tests \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/prompts/example_0.safetensors
```

For timing, alternate scalar and wave32 runs rather than executing all samples
of one arm first. Record the exact commit, environment, shape, GPU, cache state,
wall time, per-class GPU time, peak memory, hashes, and KEEP/REJECT decision.

## Next priorities

1. Profile production sequence lengths; determine when scalar, wave, or a tiled
   matrix-instruction kernel wins.
2. Replace per-block activation allocation with a shape-keyed workspace only if
   allocation count and wall time both improve without increasing peak memory.
3. Evaluate a persistent weight FD/staging cache. Do not revive per-tensor
   events or page-lock the entire streamed model.
4. Fuse the bidirectional scan launches, preserving the CPU oracle.
5. Evaluate BF16/INT8/FP8 only behind the existing BF16 scalar oracle and add
   perceptual output gates before making reduced precision the default.
6. Add topology-aware multi-GPU layer sharding after the single-GPU kernels and
   streaming lifetime are stable.
