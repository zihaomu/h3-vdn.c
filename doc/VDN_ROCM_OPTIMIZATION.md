# OpenVDN ROCm optimization ledger

This document is the repository-local performance ledger for the
`vdn-h3-rocm` branch. Correctness remains the first gate: a candidate is kept
only after the VDN operator tests pass and a complete 50-layer forward is
compared with the scalar BF16 path.

## Fixed inputs

- Target start: `3cc97ec5f0c880486bdbb85335ba5d44ca5df562`
- GPU: physical GPU 4, AMD Radeon AI PRO R9700, `gfx1201`, wave32;
  `HIP_VISIBLE_DEVICES=4` exposes it as logical device 0, PCI BDF
  `0000:e3:00.0` (`amd-smi` GPU 7 on the test host)
- ROCm: 7.2.3; HIP 7.2.53211; AMD clang 22.0.0git
- Checkpoint: OpenVDN `stage-dmd-step-250`, default and turbo adapters
- Reference implementation: `alexhegit/h3-hip.c` at
  `377ad3698d9d82fdb9ab5a22afc9fdc2c521adc3`
- Microbenchmark: 17 latent frames, 2x4 latent canvas, 6 audio rows,
  sequence 840, all 50 blocks, logical GPU 0 after the physical-GPU-4 filter

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

### REJECT: one-exp online softmax update

For a maximum update, one of the two online-softmax exponent arguments is zero,
so a candidate replaced that `expf(0)` with 1.0. The standalone kernel improved
by about 1.2%, and only 1,233 of 38,262,784 BF16 outputs differed, with maximum
absolute error `1.52587891e-05` and RMSE `1.75812545e-08`. Those tiny local
changes accumulated across 50 layers, however: final maximum absolute error was
`0.483652472`, RMSE `0.00351053542`, and relative RMSE `1.111%`. The candidate
and its temporary comparison hooks were removed because it failed the exact
output gate.

### KEEP: cached D=128 query and direct mask-gap jumps

The fixed-D=128 kernel now loads each wave's four invariant query values before
the key loop. `H3_VDN_RELOAD_QUERY=1` retains the old behavior for same-binary
A/B. Five crossed groups of five production-shape iterations measured medians
of 0.660494 seconds for reload and 0.653752 seconds for cached query (-1.0%),
with identical `3d65eea81de34693` output hashes.

When both endpoint frames are anchors, the allowed video-key region is the
ordered union of the first frame, local window, and last frame. The kernel now
jumps directly across the two possible gaps instead of evaluating every masked
row; this preserves the exact key and floating-point reduction order.
`H3_VDN_SCAN_MASK=1` retains the legacy scan. Five crossed groups of five
iterations measured medians of 0.654632 and 0.639143 seconds (-2.37%). Tests
cover prefix and suffix text, zero prefix, one token per frame, and both window
gaps; every old/new output was bitwise identical.

With both legacy switches enabled versus the combined default, five final
crossed groups of five iterations measured medians of 0.661281 and 0.640123
seconds, a 3.20% end-to-end kernel improvement. All ten hashes were again
`3d65eea81de34693`.

The combined production path uses 31 VGPRs, no VGPR or SGPR spills, and no
private segment. A complete 50-layer run retained video/audio hashes
`b3d3500676d3fb12` and `5fbd7afb3d78a277`, kept the 4.969 GiB peak, and measured
18.647 seconds of SDPA GPU-event time versus 20.884 seconds before this batch.
Its total wall time was excluded from the comparison because concurrent disk
traffic reduced weight-read throughput to 3.74 GiB/s. The final native 8-NFE,
dual-VAE, mux acceptance completed in about 83.8 seconds and reproduced the
exact 73,528-byte MP4 and SHA-256 above.

### REJECT: multiple head waves per workgroup

Packing adjacent heads for the same query into one workgroup appeared highly
effective on the synthetic benchmark: a final crossed 1-versus-16-wave median
fell from 0.644220 to 0.397790 seconds (-38.25%), with bitwise-identical output.
The 512-thread kernel used 35 VGPRs, no spills or private segment, and reported
occupancy 16. Real weights reversed the result. After a profile mark excluded
prompt/setup operations, the same-process production 50-layer comparison was
19.409 seconds of SDPA and 34.851 seconds wall for one wave, versus 19.973 and
35.447 seconds for 16 waves (+2.91% SDPA, +1.71% wall). Both full outputs were
bitwise identical. Real-model data takes precedence, so every candidate kernel,
environment switch, and comparison hook was removed.

### REJECT: staging chunk-size sweep and explicit block workspace

A real block-0 plus top-level/refiner loader streamed 2.946 GiB per run while
sweeping 4, 8, 16, 32, 64, and 128 MiB pinned chunks. The 8/16 MiB cases read
at roughly 13.2--13.4 GiB/s; 32/64/128 MiB fell to roughly 12.4/10.5/10.0
GiB/s, while RSS rose from about 427 to 550 MiB. Four MiB matched 8 MiB reads
but was slightly slower for upload. The fixed 8 MiB chunk remains the best
tradeoff; the temporary sizing code was removed.

A `rocprofv3` HIP-runtime trace measured 5,217 `hipMalloc` and 5,217 `hipFree`
calls in a complete small forward. They totaled only 105.7 and 147.5 ms. The
call count is shape-independent, putting the production workspace's theoretical
upper bound below 1% of wall time. This confirms the earlier allocation-pool
rejection, so a large explicit block-workspace refactor was not implemented.

### KEEP: double-buffered weight staging

Weights spanning multiple 8 MiB chunks now alternate between two cached pinned
buffers. Per-buffer HIP events prevent reuse before completion, allowing the
next `pread` to overlap the previous H2D copy on the existing single GPU stream.
Timing-enabled events keep upload profiling meaningful. Small tensors retain
the one-buffer path; `H3_HIP_SERIAL_STAGING=1` restores it for same-binary A/B,
and disabling the staging cache automatically selects serialized behavior.

On production geometry, three crossed external wall samples had medians of
37.01 seconds serialized and 35.51 seconds pipelined (-4.05%). Internal profile
wall medians were 36.163 and 34.288 seconds (-5.19%). The payload remained
66.818 GiB, device peak stayed 4.969 GiB, host RSS rose by only 8--9 MiB, and
every video/audio hash stayed `b3d3500676d3fb12` / `5fbd7afb3d78a277`.
Default and cache-disabled loader paths, all GPU operator gates, scalar/wave32
comparison, and 1,774 host checks passed. The final 8-NFE dual-VAE/mux run took
88.39 seconds under the then-current system load and reproduced the exact
73,528-byte MP4 SHA-256.

### REJECT: split full/window query launches

The production query rows were split into three launches for the prefix plus
first anchor, window-restricted middle frames, and last anchor plus suffix. This
removed query-category and anchor branches while preserving each query's key
and reduction order, but separating long and short rows lost useful scheduling
overlap. Five crossed groups of five iterations regressed from a 0.638988-second
median to 0.810594 seconds (+26.9%). Every output hash remained exact; the
candidate and its diagnostic switch were removed without running the 50-layer
gate.

### KEEP: distributed wave online-softmax state

After lane 0 finishes the dot-product reduction, the D=128 production kernel
now broadcasts the score once. Every lane performs the same online-softmax
max/sum update, so the value accumulators consume local scales instead of two
separate wave shuffles from lane 0. The floating-point operation order is
unchanged. `H3_VDN_LANE0_SOFTMAX=1` restores the previous path for same-binary
A/B.

Five crossed standalone groups of five production-shape iterations reduced the
median from 0.638640 to 0.415342 seconds (-35.0%). All ten full-output hashes
were `3d65eea81de34693`, and prefix/suffix, zero-prefix, one-token-per-frame, and
both-gap boundary shapes were bitwise identical. The accepted code object uses
34 VGPRs versus 31 for the legacy path, with no VGPR/SGPR spills and no private
segment.

On real weights, pipeline-staging samples had median SDPA event times of 18.834
seconds for lane-0 softmax and 16.372 seconds for the distributed path (-13.1%).
Median external wall time fell from 35.51 to 33.00 seconds (-7.1%), while the
device peak remained 4.969 GiB and the 66.818 GiB payload was unchanged. Three
distributed runs reproduced video/audio hashes `b3d3500676d3fb12` /
`5fbd7afb3d78a277`. One legacy sample produced a transient different hash; one
serial-staging and one additional pipeline legacy run both reproduced the
baseline, and all candidate runs remained exact, so it was recorded but did not
invalidate the candidate comparison.

The default and cache-disabled real loaders, GPU operator/feature/solve/scan
gates, small 50-layer forward, and 1,774 host checks passed. The final native
8-NFE dual-VAE/mux acceptance completed in 74.71 seconds and reproduced the
exact 73,528-byte MP4 and SHA-256. Its streams remain 56-frame H.264 at
64x32/24 fps and stereo AAC at 32 kHz.

### KEEP: parameterized production E2E release gate

The native E2E harness now accepts requested frames, latent height/width, audio
latents, and NFE through `VDN_E2E_*` variables, with overflow, even-canvas, NFE,
and synchronized audio/video geometry validation. Its existing 64x32 defaults
remain unchanged. It also reports FNV-1a hashes for denoised video/audio rows,
decoded F32 video, decoded PCM, and RGB24 output.

Two consecutive single-GPU runs used 56 requested frames, a 17x32x32 video
latent, 93 audio latents, and 8 NFE. They completed in 486.49 and 487.20 seconds
with roughly 873 MiB maximum host RSS. All five internal hashes matched:

- video rows `e77bfd64f14b695c`
- audio rows `bdde023376238608`
- decoded video F32 `127cba9e701bb2c4`
- decoded audio PCM `5cfe75130b41efad`
- RGB24 `0338db1c620814e3`

Both outputs were byte-identical 2,315,918-byte MP4 files with SHA-256
`ee267508d2c988629811ce86db8d6ac7a1a8291957b792583348dc0be90eea43`.
`ffprobe` reports H.264, 56 frames, 512x512 at 24 fps, and stereo AAC at 32 kHz.
All 56 decoded frame hashes were distinct; audio mean/peak volume was
-23.5/-9.8 dB.

### KEEP: configurable HIP code-object targets

The HIP build now accepts a space-separated `HIP_ARCHS` list while retaining
`gfx1201` as the default. On ROCm 7.2.3 the complete HIP translation unit built
successfully in compile-only checks for `gfx90a`, `gfx942`, `gfx1030`,
`gfx1100`, `gfx1151`, and `gfx1201`. Only the R9700 `gfx1201` target has runtime
evidence and is part of the stable support guarantee; the others remain
compile-only rather than being advertised as supported.

The validated GPU reports wavefront size 32, 31.9 GiB VRAM, and zero UMC RAS
correctable/uncorrectable errors after the release stress run. Forced scalar
and default wave32 production-shape attention both produced hash
`3d65eea81de34693`; their single-iteration times were 4.8031 and 0.4146 seconds.
The runtime continues to require wave size 32 before selecting the optimized
kernel, and `H3_VDN_SCALAR_SDPA=1` remains the explicit fallback.

### KEEP: official variable-length prompt contract

OpenVDN upstream commit `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`
uses a separate Qwen3-VL-32B conditioner: verbatim tokenization without special
tokens, decoder hidden state 50, BF16 `[L,5120]`, and I64 `[L]` text tags. The
release checkpoint contains no processor, tokenizer, or text-encoder files.
Its three official prompts have 800, 821, and 1299 rows, which exposed the old
800-row-only converter as an incomplete contract.

The safe no-unpickle converter, safetensors loader, prompt refiner, and packed
layout now carry `L` dynamically. All three official prompts pass conversion
and loading; the 821- and 1299-row prompts pass real GPU refinement, and the
821-row prompt passes a complete 50-layer forward on `gfx1201`. The existing
800-row conversion is byte-identical and retains its established hashes.

Raw text remains an explicit upstream preprocessing step: embedding the 62 GB
PyTorch/Transformers conditioner into this native runtime is outside the stable
binary's dependency and checkpoint boundary. Upstream OpenVDN inference defines
only `render.prompt_file`; it has no first/last-frame or ordered-media config.
The native API tests these unsupported paths for early, nonzero, actionable
failure rather than borrowing the different FL2VA/Ref2VA model semantics.

### KEEP: continuous per-NFE and pipeline profiling

The cumulative HIP profile API now survives human-readable profile marks, so
two snapshots can delimit one NFE without losing event totals. Schema-v2
`<output>.inference.json` records the PCI BDF, five FNV-1a output hashes,
critical-path phases, residual/coverage, per-NFE forward subphases, GPU command
counters, inclusive linear/SDPA/solve/scan events, read/H2D bytes and time, RSS,
faults, and context switches. `scripts/profile_vdn_gpu4.sh` enforces the physical
GPU-4 filter, maps the HIP BDF to the matching `amd-smi` ID, rejects a busy
device, and captures approximately 1 Hz GPU, disk, and process telemetry.

The first 512x512/56-frame public-CLI profile explained 99.9999997% of its
486.528789-second critical path:

| Phase | Wall |
|---|---:|
| setup + readback/teardown | 1.513 s |
| 8-NFE denoise | 241.253 s |
| video VAE | 241.640 s |
| audio VAE | 1.579 s |
| RGB + mux | 0.543 s |

The eight NFE times were 30.752, 30.026, 30.047, 30.082, 30.071, 30.099,
30.083, and 30.093 seconds. Each true NFE streamed exactly 65.176 GiB read and
H2D; aggregate DiT read/H2D was 523.048 GiB in 39.463/33.094 seconds. DiT SDPA
was 130.219 seconds inclusive, while the video VAE independently spent
184.609 seconds in SDPA and 54.095 seconds in linear operations. This proves
that the historical 60.9 seconds/NFE quotient includes a nearly equal-size,
one-time video-VAE phase; it is not evidence that sustained NFE latency doubles.

The correct BDF telemetry showed no monotonic NFE slowdown, no major faults,
and no sustained clock collapse as the hotspot warmed. A no-telemetry single-NFE
control was not slower than the approximately 1 Hz diagnostic samples. The
full no-continuous-sampling control was 488.349333 seconds versus a
486.857837-second sampled-run mean: it was 0.31% slower, so no >1% sampler
penalty was detected. Two post-schema production outputs passed the PCI BDF,
five frozen internal hashes, 2,315,918-byte MP4 SHA-256, and ffprobe gates.
P0 is complete; all later candidates use this schema and baseline.

### KEEP: exact wave32 F32/D64 video-VAE SDPA

The production video VAE was the largest single measured hotspot: its generic
F32 attention used a 256-thread block for D=64 and synchronized the complete
block once per key. The kept specialization assigns one wave32 to each
query/head pair. Every lane owns dimensions `d` and `d+32`; it combines those
two products at the same node where the scalar 256-thread tree reaches stride
32, then performs the identical 16/8/4/2/1 reduction. Key traversal, online
softmax, and output FMA order are unchanged, so the result is bitwise exact.

Five crossed `S=2273, H=32, D=64` microbenchmark groups produced scalar and
wave32 medians of 0.422021 and 0.123266 seconds (3.42x, -70.8%). Every one of
23,275,520 F32 outputs matched bit for bit. Boundary fixtures at S=1, 7, 65,
and 509 with nonstandard head counts and tails also reported zero mismatches.

| Production result | Scalar | wave32 | Change |
|---|---:|---:|---:|
| Isolated video-VAE wall, crossed mean | 243.587705 s | 109.497752 s | -55.05% |
| Video-VAE SDPA event, crossed mean | ~186.684 s | ~52.954 s | -71.6% |
| Full 8-NFE E2E, crossed mean | 486.705699 s | 354.399810 s | -27.184% / 1.373x |
| Peak live GPU allocation | 9.454 GiB | 9.454 GiB | unchanged |

Both production candidate renders reproduced the five frozen internal hashes,
the 2,315,918-byte MP4, and SHA-256
`ee267508d2c988629811ce86db8d6ac7a1a8291957b792583348dc0be90eea43`.
The small VAE scalar/wave decoded-F32 hash was identically
`4e1406b60b207415`; the production isolated hash was identically
`aafcf45d65a16b31`.

The gfx1201 code object reports wavefront size 32, 39 VGPR, 30 SGPR, no VGPR
or SGPR spills, zero private segment, and no dynamic stack. Disassembly
contains wave permutation and exponential instructions and no block barrier.
The exact specialization is the default only for non-causal F32/D64 on a
wave32 device. Unsupported cases automatically retain the generic kernel;
`H3_F32_SDPA_SCALAR=1` forces that oracle explicitly.

After the default switch, a clean-build public CLI run with no SDPA environment
override completed the 64x32/56-frame/8-NFE gate in 73.402279 seconds. Its
video VAE was 1.181242 seconds and reported 0.004 seconds of SDPA. Schema v2,
PCI BDF `0000:e3:00.0`, all five frozen small-output hashes, the 73,528-byte
MP4 SHA-256 `7a447fe6f63697ad1bbb2df8a74f385b432ce3a1d5855caee5f8c1531a955c5b`,
and ffprobe all passed. A separate smoke run forced the scalar fallback and
reproduced decoded-F32 hash `4e1406b60b207415`.

## Test gates

Build and run the local gates with:

```sh
make BACKEND=hip -j16 \
  h3_vdn_gpu_ops_tests h3_vdn_feature_tests \
  h3_vdn_solve_tests h3_vdn_scan_tests h3_vdn_forward_smoke_tests \
  h3_f32_sdpa_bench h3_vdn_video_vae_smoke_tests

HIP_VISIBLE_DEVICES=4 ./h3_vdn_gpu_ops_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_feature_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_solve_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_scan_tests

# Generic F32/D64 oracle versus the default exact wave32 specialization.
HIP_VISIBLE_DEVICES=4 ./h3_f32_sdpa_bench
HIP_VISIBLE_DEVICES=4 ./h3_vdn_video_vae_smoke_tests \
  models/vdn-minimax-h3/h3-base/vae
HIP_VISIBLE_DEVICES=4 H3_F32_SDPA_SCALAR=1 \
  ./h3_vdn_video_vae_smoke_tests models/vdn-minimax-h3/h3-base/vae

HIP_VISIBLE_DEVICES=4 VDN_SMOKE_COMPARE_SDPA=1 \
  ./h3_vdn_forward_smoke_tests \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/prompts/example_0.safetensors

# 512x512 / 56-frame attention geometry; add H3_VDN_SCALAR_SDPA=1 for oracle
make BACKEND=hip h3_vdn_sdpa_bench
HIP_VISIBLE_DEVICES=4 ./h3_vdn_sdpa_bench

# One complete 50-layer production-token profile (one NFE, no VAE/mux)
HIP_VISIBLE_DEVICES=4 H3_PROFILE=1 \
  VDN_SMOKE_FRAMES=17 VDN_SMOKE_LATENT_H=32 \
  VDN_SMOKE_LATENT_W=32 VDN_SMOKE_AUDIO_LATENTS=93 \
  ./h3_vdn_forward_smoke_tests \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/prompts/example_0.safetensors

# Full stable-release production E2E (56 frames, 512x512, 8 NFE)
HIP_VISIBLE_DEVICES=4 \
  VDN_E2E_FRAMES=56 VDN_E2E_LATENT_H=32 VDN_E2E_LATENT_W=32 \
  VDN_E2E_AUDIO_LATENTS=93 VDN_E2E_NFE=8 \
  ./h3_vdn_e2e_tests \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/prompts/example_0.safetensors \
  outputs/vdn-e2e-production.mp4
```

For timing, alternate scalar and wave32 runs rather than executing all samples
of one arm first. Record the exact commit, environment, shape, GPU, cache state,
wall time, per-class GPU time, peak memory, hashes, and KEEP/REJECT decision.

## Next priorities

1. Evaluate a tiled/matrix-instruction VDN SDPA kernel against the new
   0.4153-second wave32 baseline, preserving the scalar oracle.
2. Consider layer-level host prefetch only if it preserves bounded memory and
   improves on double-buffered per-tensor staging; do not pursue an explicit
   block workspace unless allocator behavior changes materially.
3. Do not add an FD-only cache: its crossed A/B improved the median by just
   0.8%. Further I/O work must reduce the 66.8 GiB payload itself. Do not revive
   per-tensor events or page-lock the entire streamed model.
4. Fuse the bidirectional scan launches, preserving the CPU oracle.
5. Evaluate BF16/INT8/FP8 only behind the existing BF16 scalar oracle and add
   perceptual output gates before making reduced precision the default.
6. Keep this phase single-card. Multi-GPU sharding is explicitly deferred; use
   bounded hot-block residency and prefetch only after measuring available VRAM.
