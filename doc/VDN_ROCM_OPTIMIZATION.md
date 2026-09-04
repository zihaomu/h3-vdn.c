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

A standalone production-length benchmark covers the 512x512/56-frame attention
geometry: sequence 5338, video start 986, 17 latent frames, and 256 video tokens
per frame. Three crossed samples produced scalar times of 4.709, 5.756, and
4.740 seconds (median 4.740) versus wave32 averages of 0.8280, 0.8277, and
0.8295 seconds (median 0.8280), a 5.72x kernel speedup. All six runs hashed the
complete 38,262,784-element BF16 output to `3d65eea81de34693`.

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

### REJECT: transparent exact-size allocation pool

A 3 GiB exact-size device allocation pool recorded 4,965 reuse hits during one
50-layer forward. Three crossed samples had medians of 13.41 seconds without
the pool and 13.33 seconds with it, only 0.6% apart, while the pool retained the
full 3.0 GiB limit. The memory cost and lifetime/locking complexity are not
justified by a sub-noise result, so the implementation and opt-out were removed.

### KEEP: persistent pinned weight staging

Weight reads still use the original 8 MiB chunks and preserve their ordering,
but a thread-safe per-context cache now recycles the pinned staging buffer
instead of calling `hipHostMalloc`/`hipHostFree` for every tensor. It retains at
most 32 buffers for concurrent loaders; the sequential VDN path uses one, so
the observed pinned-memory cost is 8 MiB. `H3_HIP_STAGING_CACHE=0` restores the
old lifecycle.

Three crossed 50-layer samples produced 13.83, 13.90, and 12.81 seconds without
the cache (median 13.83) versus 12.49, 12.51, and 12.41 seconds with it (median
12.49), a 9.7% reduction. The cache recorded 2,579 hits in 2,580 acquisitions.
Disk and H2D payloads are unchanged, and all runs retained both output hashes.
The full 8-NFE VAE/mux test fell from 141.72 seconds on the wave32-only commit
to 95.95 seconds with staging reuse (-32.3%), and from the original 170.7-second
acceptance baseline by about 43.8%. Maximum host RSS increased from roughly
652 MiB to 660 MiB, matching one retained staging buffer. The MP4 SHA-256 and
stream metadata remained identical.

### REJECT: per-context weight FD cache

A bounded, thread-safe cache retained descriptors for the weight shards and
exposed `H3_HIP_FD_CACHE=0` for same-binary comparison. It recorded 2,563 hits
in 2,580 opens (17 shard misses), but open/close was not a meaningful part of
the remaining load time. Three crossed samples without the cache were 12.01,
13.08, and 13.93 seconds (median 13.08); with the cache they were 12.97, 13.97,
and 12.97 seconds (median 12.97), only 0.8% apart and well inside run-to-run
noise. All six runs retained both output hashes and used about 658 MiB maximum
RSS. The candidate code and opt-out were removed.

### Production-shape full-forward baseline

The forward smoke test accepts `VDN_SMOKE_FRAMES`,
`VDN_SMOKE_LATENT_H`, `VDN_SMOKE_LATENT_W`, and
`VDN_SMOKE_AUDIO_LATENTS`, while retaining its original small defaults. One
profile at the real 512x512/56-frame token geometry completed all 50 layers at
sequence 5338 in 53.48 seconds with a 4.969 GiB device-allocation peak. The
measured GPU classes totaled 39.595 seconds: SDPA 35.546 (89.8%), linear 3.587,
solve 0.358, and scan 0.104 seconds. Weight read and upload took 4.603 and 4.058
seconds for the unchanged 66.818 GiB payload. The full output hashes are video
`b3d3500676d3fb12` and audio `5fbd7afb3d78a277`.

### REJECT: multiple query waves per block

Packing 2, 4, or 8 query waves into each block was bitwise identical but did
not improve K/V locality enough to offset the larger block. Production-shape
averages for 1, 2, 4, and 8 waves were 0.8146, 0.8293, 0.8277, and 0.8343
seconds. The kernel remains one query/head wave per block.

### KEEP: precomputed window mask and specialized D=128 SDPA

The wave32 kernel now computes each query's video window once, replacing the
hot-loop frame divisions with ordered row-range comparisons. A same-binary
crossed A/B reduced the production-shape median from 0.8297 to 0.8038 seconds
(-3.1%). A compile-time D=128 path then removes the unused D<=256 accumulator
slots and dynamic value loop. Its crossed median was 0.6542 seconds versus
0.8024 for the generic path (-18.5%); code-object metadata shows 24 versus 29
VGPRs with no spills. Combined with the earlier 0.8280 production median, the
new 0.6542 median is 21.0% faster. Every result retained BF16 hash
`3d65eea81de34693`; other dimensions still use the generic wave32 path, and
`H3_VDN_SCALAR_SDPA=1` remains the oracle.

On the complete production-token 50-layer profile, wall time fell from 53.48
to 37.42 seconds (-30.0%) and measured SDPA time from 35.546 to 20.884 seconds
(-41.2%). Peak allocation stayed at 4.969 GiB, weight traffic was unchanged,
and both complete output hashes matched the pre-optimization baseline exactly.
The small-shape native 8-NFE VAE/mux acceptance completed in 92.66 seconds
versus 95.95 seconds before this change and reproduced the exact 73,528-byte
MP4 SHA-256 `7a447fe6f63697ad1bbb2df8a74f385b432ce3a1d5855caee5f8c1531a955c5b`.

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

# 512x512 / 56-frame attention geometry; add H3_VDN_SCALAR_SDPA=1 for oracle
make BACKEND=hip h3_vdn_sdpa_bench
HIP_VISIBLE_DEVICES=0 ./h3_vdn_sdpa_bench

# One complete 50-layer production-token profile (one NFE, no VAE/mux)
HIP_VISIBLE_DEVICES=0 H3_PROFILE=1 \
  VDN_SMOKE_FRAMES=17 VDN_SMOKE_LATENT_H=32 \
  VDN_SMOKE_LATENT_W=32 VDN_SMOKE_AUDIO_LATENTS=93 \
  ./h3_vdn_forward_smoke_tests \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/prompts/example_0.safetensors
```

For timing, alternate scalar and wave32 runs rather than executing all samples
of one arm first. Record the exact commit, environment, shape, GPU, cache state,
wall time, per-class GPU time, peak memory, hashes, and KEEP/REJECT decision.

## Next priorities

1. Evaluate a tiled/matrix-instruction production SDPA kernel against the new
   0.6542-second wave32 baseline, preserving the scalar oracle.
2. Replace per-block activation allocation with an explicit block workspace
   only if allocation count and wall time both improve without increasing peak
   memory; a transparent 3 GiB allocation pool has already been rejected.
3. Do not add an FD-only cache: its crossed A/B improved the median by just
   0.8%. Further I/O work must reduce the 66.8 GiB payload itself. Do not revive
   per-tensor events or page-lock the entire streamed model.
4. Fuse the bidirectional scan launches, preserving the CPU oracle.
5. Evaluate BF16/INT8/FP8 only behind the existing BF16 scalar oracle and add
   perceptual output gates before making reduced precision the default.
6. Add topology-aware multi-GPU layer sharding and persistent per-device block
   weights to remove the repeated 66.8 GiB read/upload payload from each NFE.
