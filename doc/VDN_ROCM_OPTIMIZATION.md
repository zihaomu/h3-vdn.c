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

### REJECT: exact LDS-tiled BF16/D128 VDN SDPA

An opt-in experiment grouped eight same-frame queries in one 256-thread block
and shared eight D=128 K/V rows through 4 KiB of LDS. Every wave retained the
default QK reduction tree, online-softmax key order, and PV update order.
Whole key tiles outside the common VDN mask were skipped. Five same-process,
crossed production-geometry groups appeared positive in isolation: the
default median was 0.413741 seconds and the tiled median was 0.397547 seconds
(-3.91%, 1.041x), with zero mismatches across every BF16 output. Prefix,
suffix, frame-size, and query/key tail fixtures also matched exactly.

The required real-weight gate reversed that result:

| 50-layer production-token metric | Default mean | LDS tile mean | Change |
|---|---:|---:|---:|
| SDPA event time | 16.2315 s | 19.5890 s | +20.69% |
| Forward wall | 30.398459 s | 33.825738 s | +11.27% |
| Output hashes | `b3d350...fb12` / `5fbd7a...a277` | identical | exact |
| Peak live GPU allocation | 4.969 GiB | 4.969 GiB | unchanged |

The crossed order reproduced the regression, while read/H2D payload and times
were effectively unchanged. The standalone synthetic residency/cache context
therefore did not predict sustained layer execution. The candidate and its
environment switch were removed; the existing one-wave exact kernel remains
the default.

ROCm exposes the gfx12 BF16 intrinsic
`wmma_f32_16x16x16_bf16_w32_gfx12`, but its 16-wide matrix accumulation cannot
preserve the scalar D=128 reduction tree. A WMMA candidate is consequently not
eligible for the exact BF16 default track and moves to the explicitly
non-bitwise SageAttention/low-precision research gate instead.

### RESEARCH KEEP: native gfx12 SageAttention-style SDPA

The native gfx12 design was informed by SageAttention PR #368 fixed at commit
`66f5e64c9e36084c863a4480e570069245e58f90` (Apache-2.0), without importing its
PyTorch extension runtime. The implementation remains C/C++17 and HIP-only.
The maintained implementation now comes from the fixed
`third_party/sageattention-amd` submodule at
`18d949018cec1467ac6d30c12b3494f2f51bb552`; H3 owns only the thin dispatch and
context bridge. Planner, quantization, kernel, ISA and operator contracts must
be changed and validated in that upstream repository before its gitlink is
advanced here.
`H3_VDN_SDPA=sage-i8-bf16` is an explicit experimental mode; `auto` continues
to select the bitwise-exact wave32 implementation. The incomplete F16 and FP8
modes fail explicitly instead of silently falling back.

Q and K are symmetrically quantized to signed I8 `[-127,127]`, using one scale
per head and 32 Q rows or 64 K rows. Rounding is round-to-nearest-even and a
zero group uses scale 1. GPU output matched the CPU quantization oracle exactly,
including non-aligned tails and zero groups. The production S=5338/H=56/D=128
quantization itself averaged 0.001098 seconds.

The current E27 gfx12 kernel maps one 64-thread workgroup to two 16-query waves
from one head. Host-built query tasks split arbitrary text/audio/video geometry
into at most 32 query rows and an ordered list of allowed key intervals, then
cache that table in the device workspace. Each wave uses I8 WMMA for QK,
updates the online softmax, converts probabilities with RNE, and immediately
uses BF16 WMMA for all eight 16-column P*V fragments. No SxS score or mask
tensor is materialized, and masked gaps are skipped rather than tested in the
hot score loop. A targeted test crosses frame and Q32/K64 scale boundaries and
changes only values in a masked frame; the protected output reports zero BF16
mismatches.

Five crossed production-shape standalone groups gave:

| Metric | exact wave32 | I8-QK/BF16-PV Sage | Change |
|---|---:|---:|---:|
| Median, including Q/K quantization | 0.412219 s | 0.097357 s | -76.38%, 4.234x |
| Synthetic relative RMSE | — | 0.00379933 | approximate |
| Synthetic cosine | — | 0.999992786 | finite |
| Stable output hash | `2a54af9f76d9adbe` | `98fc5281025a8eba` | repeatable |

The later E27 interval-task rewrite was revalidated after a clean build. One
GPU-4 production-shape B-then-S run measured 0.411735 seconds for exact wave32
and 0.014938 seconds for Sage including Q/K quantization (27.562x). Its maximum
absolute error was 0.00012207, relative RMSE 0.00361013, cosine 0.999993586,
and there were no non-finite values. The E27 approximate hash was
`b9a74fa3e1008c63`; the full 8-NFE gate below is the acceptance-level speed and
quality result.

Disassembly contains both `v_wmma_i32_16x16x16_iu8` and
`v_wmma_f32_16x16x16_bf16`. The fused kernel uses wave32, 72 VGPR, 68 SGPR,
18,048 bytes of LDS, no VGPR/SGPR spills, no private segment, and no dynamic
stack. A combined gfx90a/gfx1100/gfx1201 bundle compiles; the public entry point
rejects non-gfx12 hardware before dispatch.

Three crossed real-weight production-token samples retained the speedup:

| 50-layer metric | exact wave32 | Sage | Change |
|---|---:|---:|---:|
| Forward median | 30.120520 s | 18.788200 s | -37.62%, 1.603x |
| Independent SDPA samples | 16.318/16.382 s | 5.171/5.220 s | about -68.2% |
| Peak live allocation | 4.969 GiB | 5.040 GiB | +71 MiB |

The Sage video/audio hashes were stable at `9f08559b684bc7ff` and
`b27024068a0fe1a6`. Relative to exact wave32, final F32 velocity had maximum
absolute error 0.420409739, relative RMSE 0.0152040993, cosine
0.999884424423, and no non-finite values. A separate diagnostic retained all
50 exact hidden states on the same GPU and compared every BF16 element at each
candidate layer. Relative RMSE was 0.00274554 at layer 1, peaked at 0.01931865
at layer 22, and ended at 0.00896422 at layer 50; all layers were finite.

This is a research keep, not a stable default. It must next pass the three-
prompt video, audio, and complete E2E quality gate. BF16 exact fallback remains
mandatory regardless of that result.

#### SageAttention-AMD submodule ownership cutover

The 2026-09-08 cutover removed H3's duplicate planner, Q/K quantization,
rocWMMA experiments, E27 kernel, and low-level test API. The fixed submodule now
owns those components; `h3_vdn_sage_bridge.cpp` is the only H3-to-upstream
adapter, while `h3_vdn_sdpa_mode.c` retains H3's dispatch policy. A clean HIP
build places the three upstream runtime objects directly in `libh3.a`; symbol
inspection found a single E27 archive member. Missing submodules fail with an
explicit `git submodule update --init --recursive` instruction and never cause
a build-time download.

The physical-GPU-4 migration gate produced:

| Gate | Result |
|---|---|
| Upstream GPU contract | generic/H3 oracle, targeted interval shapes, guard canaries, determinism and finite checks passed on `gfx1201` |
| Upstream production operator | 14.665 ms profiled total; 14.792 ms GPU median; Q/K quant 0.474/0.397 ms; hash `d8fccefb0ea98938` |
| H3 crossed public dispatch | 14.953/15.217 ms Sage including bridge and quant; exact pre-cutover hashes `2a54af9f76d9adbe` / `b9a74fa3e1008c63` |
| Production 50 layers | wave32 30.447261 s; Sage 14.438533 s; exact wave32 output hashes retained |
| Prompt-0 8-NFE | wave32 246.579859 s; Sage 132.308016 s; migrated Sage latent hashes exactly matched pre-cutover E27 |
| Prompt-0/1 full media | migrated MP4 SHA-256 values exactly matched pre-cutover E27: `5a4cc484205211255c2c6b9a1e71e7256e338ebca830f033284507408b4b2f77` and `798f8a8c7f6f437522340205a7157dca58a418d4d8f842b348df9a88bc365bd1` |

This validates the ownership migration, not a quality promotion. Prompt 0 and
1 retain their decoded-audio failures. Prompt 2 stopped at the staged 8-NFE
gate with video relative RMSE/cosine `0.00221533361/0.999997546` but audio
`0.105619612/0.994429938`, outside the frozen 5%/0.999 limits. Therefore
`sage-i8-bf16` remains explicit and experimental, and `auto` remains exact
wave32.

### REJECT: offline default/turbo LoRA premerge

Dedicated profiling measured 571 LoRA merge calls in one production-token
forward (550 in the 50 block stream) at 0.728726 seconds total. Exact
safetensors accounting found only 1.066780 GiB of adapter payload per NFE,
1.64% of the 65.176 GiB stream. Materializing all effective BF16 matrices would
require 60.1135 GiB for the blocks and about 61.6029 GiB including setup, a
roughly 56x storage amplification for a critical-path ceiling near 0.8 seconds
per NFE. The cache builder and its invalidation/manifest surface are not
justified, so no derived checkpoint was generated.

### KEEP (opt-in): bounded resident effective-weight sources

`H3_VDN_RESIDENT_GIB=N` enables a generation-lifetime cache with an integer
budget from 1 through 20 GiB. Lowest-numbered blocks fill deterministically;
admission queries runtime-visible free VRAM and always reserves at least 6 GiB.
An unsupported query or insufficient headroom stops cache growth and retains
normal streaming. Unset or `0` preserves the existing default path. Profile
output records the budget, resident bytes, blocks, hits, misses, and whether
admission was limited.

A first implementation directly reused cached GPU tensor pointers. Although it
was fast, it produced one changed final hash in a three-run 12 GiB stress and
one text-state Cholesky failure in two 16 GiB runs. Ten independent one-forward
runs were exact; keeping 11.540 GiB allocated while bypassing reuse was exact
3/3. Full fingerprints showed the nine cached blocks unchanged after 8 NFE and
bitwise identical to independently reloaded/merged blocks. Direct pointer reuse
is permanently rejected.

The kept design treats resident tensors only as immutable sources. Every hit
allocates the normal per-block working tensors, clones source data device to
device in one ordered submission, and releases those tensors after the block.
It still removes disk read, H2D, and LoRA merge for a hit while preserving the
original temporary-buffer lifecycle. The final 12 GiB small-token 8-NFE stress
passed 10/10 with fixed hashes, `admission_limited=0`, 9 blocks/11.540 GiB,
63 hits/337 misses, and a 14.977 GiB peak. Its wall median was 61.594 seconds
versus a 69.396-second streaming control median (-11.24%).

Two production 8-NFE DiT runs were 238.027 and 237.353 seconds (237.690-second
mean), 1.48% below the P0 241.253-second reference. Logical read/H2D fell from
523.048 to 440.927 GiB (-15.70%); DiT peak rose from 4.969 to 16.509 GiB. A
complete 512x512/56-frame render passed all five frozen internal hashes, the
2,315,918-byte MP4, ffprobe, and SHA-256
`ee267508d2c988629811ce86db8d6ac7a1a8291957b792583348dc0be90eea43`.
Because exact production is compute-dominated and gains only about 1.5% in
DiT, the cache stays opt-in rather than consuming 11.54 GiB by default. It may
become more valuable with the experimental faster attention path.

### REJECT: block prefetch and userspace host weight cache

The production resident run issued 440.927 GiB of logical reads, while
`/proc/<pid>/io` recorded only 382,509,056 bytes of physical reads for the whole
E2E and zero major faults. Disk samples were normally idle. Linux page cache is
already serving more than 99.9% of the stream, so readahead cannot hide a
material physical-I/O wait.

The remaining `pread` cost is page-cache-to-host/pinned copying. A next-block
buffer is about 1.28 GiB: ordinary memory would add another copy before H2D,
while pinning the block would expand the proven 16 MiB staging footprint about
80x. Upload also needs new cross-stream ordering. A separate 8/16/32 GiB
userspace host cache would duplicate Linux page cache, increase RSS and NUMA
complexity, and leave H2D bytes unchanged. Neither design has a safe expected
gain over the kept per-tensor double staging, so both are rejected without
adding runtime code.

### P4 REJECT for stable: Sage 8-NFE audio latent gate

The reduced-precision gate was frozen before inspecting production candidate
media. Each video and audio latent arm after 8 NFE requires cosine at least
0.999 and relative RMSE at most 5%. A GPU-4-only same-process run reset the
production latent and evaluated exact wave32 followed by the current E27
`sage-i8-bf16` mode:

| Metric | exact wave32 | E27 Sage | Result |
|---|---:|---:|---|
| 8-NFE DiT wall | 241.002963 s | 118.216247 s | -50.95% |
| Stable NFE wall | 30.0--30.3 s | 14.77--14.79 s | about -51% |
| Stable SDPA event/NFE | 16.1--16.3 s | 0.783--0.789 s | about -95% |
| Video relative RMSE / cosine | — | 0.233513% / 0.999997274256 | pass |
| Audio relative RMSE / cosine | — | 11.499490% / 0.993377193997 | **fail RMSE** |
| Peak live allocation | 4.969 GiB | 5.040 GiB | +71 MiB |

The wrapper recorded BDF `0000:e3:00.0`, `gfx1201`, exit status zero, and an
empty concurrency guard. No output was non-finite. Performance is more than
sufficient, but the audio latent error fails before decoded-media evaluation;
the three-prompt production render was therefore not run. Sage remains an
explicit experimental/research option and `auto` remains exact BF16 wave32.

`scripts/compare_vdn_media.py` provides the frozen decoded-video, temporal,
PCM/SI-SDR, container, and A/V checks. `scripts/run_vdn_sage_quality_gate.sh`
runs exact/candidate/candidate-repeat sequentially through the physical-GPU-4
guard, but the higher-level runner is intentionally not invoked after a lower
8-NFE latent gate fails.

### P4 KEEP as research: ROCm INT8 model-weight GEMM probe

The fixed checkpoint carries 65.175770 GiB/NFE of block payload. AdalN and MLP
account for about 45.76 GiB/NFE; offline per-output-channel INT8 plus F32 scale
would halve those bytes. Before defining a derived checkpoint, a gfx1201 probe
measured rocBLAS BF16 against per-row activation quantization, I8xI8-to-I32
GEMM, and output-scale dequantization:

| Matrix `(M,N,K)` | BF16 | INT8 all-in | Speedup | BF16/I8 weight GiB |
|---|---:|---:|---:|---:|
| AdalN `(3,96768,2688)` | 0.887813 ms | 0.502589 ms | 1.766479x | 0.484497 / 0.242609 |
| FC1 `(5338,28672,5376)` | 9.786150 ms | 7.603232 ms | 1.287104x | 0.287109 / 0.143661 |
| FC2 `(5338,5376,14336)` | 5.629123 ms | 3.784621 ms | 1.487368x | 0.143555 / 0.071797 |

All three were isolated GPU 4 runs with empty concurrency guards. The three
GEMMs alone save only an estimated 0.220632 seconds/NFE across 50 blocks; the
larger opportunity is halving loader copies and H2D bytes. This is a GO for a
future versioned offline cache/loader experiment, not a shipped runtime format
or speed claim. BF16 stays the default and fallback.

### P5 REJECT: separate-C bidirectional scan

The low-risk scan candidate used separate C/D pointers in
`rocblas_gemm_strided_batched_ex`, reading injection directly instead of doing
two D2D copies per frame. Operator prefix/suffix and two production 50-layer
outputs were bitwise exact. Crossed measurements were:

| Order | separate-C scan | legacy scan | first wall | second wall |
|---|---:|---:|---:|---:|
| legacy then separate-C | 0.091 s | 0.106 s | 30.272 s | 30.484 s |
| separate-C then legacy | 0.090 s | 0.108 s | 29.967 s | 30.210 s |

The GPU event saves 15--18 ms per 50-layer forward, only about 0.05% of wall.
Whichever arm ran first was about 0.2 seconds faster because SDPA drift was
larger than the scan signal, so the candidate did not meet the requirement for
both stable event and end-to-end wall improvement. Its runtime switch and A/B
hooks were removed. More complex dual-stream or custom scan GEMM work is not
justified by the roughly 0.1-second/NFE ceiling.

### KEEP: gfx1201 rocSOLVER POTRF verification and retry

Post-release cross-prompt stress exposed two failure modes in ROCm 7.2.3's
rocSOLVER 3.32 `spotrf_strided_batched` path on the R9700: successful 50-layer
runs could silently return changed video/audio hashes, and other runs stopped
at a random Cholesky leading minor. A stream synchronization before POTRF was
rejected: it initially produced six correct real-weight runs, then failed at
layer 37, video batch 71, leading minor 80.

This matches the independently reported
[ROCm issue #6623](https://github.com/ROCm/legacy-rocm-build/issues/6623): on
the same `gfx1201` GPU family and rocSOLVER 3.32, strided-batched POTRF can
return a corrupted factor with `info=0`, and launch/device serialization does
not resolve it. The kept backend workaround is limited to `gfx1201`:

- copy the original `A+I` into the otherwise-unused transition output;
- verify the complete lower triangle of `L*L^T` on the GPU using a relative
  residual threshold of `1e-4 * max(max_diag(A+I), 1)`;
- restore and refactorize only failed matrices, for at most two retries;
- report retry counts in human-readable profiles and schema-v2 inference JSON.

A deterministic `H3_TEST_VDN_CORRUPT_POTRF=1` test hook corrupts one factor
after a successful POTRF while leaving `info=0`. The verifier detected and
retried all 66 injected calls in the final small-plus-production test process;
two production-batch iterations retained hash `6367b2dcf68759d5`. Without
fault injection, 64 synthetic production-batch iterations were identical.
Ten consecutive real-weight, production-token 50-layer runs all reproduced
video/audio hashes `b3d3500676d3fb12` and `5fbd7afb3d78a277`; the old detached
binary silently changed both hashes in two of three comparison runs.

The real-weight verifier/retry build measured a median solve event near
0.504 seconds/NFE versus 0.381 seconds for the one valid old-binary sample,
about 0.12 seconds or 0.4% of a 30-second forward. That cost is accepted for
detecting a library failure that otherwise reports success. The guard remains
independent of exact wave32 versus experimental Sage attention.

The post-workaround production E2E ran on physical GPU 4/BDF `e3:00.0` with
an empty concurrency guard. Eight-NFE DiT took 268.504 seconds and the exact
video VAE took 110.749 seconds; the wrapper wall was about 382 seconds. The
run reproduced all five frozen internal hashes and the 2,315,918-byte MP4
SHA-256 `ee267508d2c988629811ce86db8d6ac7a1a8291957b792583348dc0be90eea43`.
Its 68.329-second weight-read total was materially slower than the earlier
39.463-second profile, so this run is a correctness/reliability gate rather
than a replacement crossed performance baseline. It nevertheless remains
below the 438.16-second phase target.

### P7 REJECT: Sage E27 plus exact text/audio-query prefix

The first post-P6 audio-quality candidate kept the existing E27 Sage result for
video queries and overwrote `[0, video_start)` with the exact wave32 result on
every attention call. This made text and audio query outputs bitwise equal to
wave32 for the current hidden state while preserving E27 output bitwise for the
video suffix. A GPU operator composition test reported zero prefix and suffix
mismatches. Two production-shape samples, including quantization and the exact
prefix recomputation, took 0.084386 and 0.082121 seconds; the stable crossed
wave32 sample took 0.411999 seconds.

Prompt 2's real 50-layer comparison initially passed the output-level gates:
wall fell from 35.187573 to 28.542355 seconds, video relative RMSE/cosine were
0.00617807/0.999980939, and audio relative RMSE/cosine were
0.0378569/0.999283377. The complete same-process eight-NFE latent gate then
rejected the candidate:

| Metric | exact wave32 | exact-prefix candidate | Decision |
|---|---:|---:|---|
| DiT wall | 280.602851 s | 184.934060 s | -34.09%, performance passes |
| steady SDPA/NFE | about 20.0 s | about 6.8 s | passes |
| video relative RMSE / cosine | - | 0.222981% / 0.999997515 | passes |
| audio relative RMSE / cosine | - | 8.671439% / 0.996251677 | **fails both gates** |

Both outputs were finite, all natural POTRF retry counts were zero, and the
physical GPU 4 concurrency guard was empty. Exact-prefix improved audio error
over pure E27's prompt-2 12.108549% relative RMSE, but did not meet the frozen
5%/0.999 audio limits. The mode, kernel range support, and composition test were
therefore removed; no dead runtime switch remains, no three-prompt decoded-media
run was started, and exact BF16 remains the stable default. The next accepted
research direction is the already-qualified, versioned offline INT8 model-weight
cache rather than another unversioned runtime format.

### P7 RESEARCH KEEP / RUNTIME REJECT: versioned INT8 weight cache

The follow-up investigated the largest remaining streaming target without
changing the supported checkpoint. A research-only builder now merges the
turbo LoRA into each block's AdaLN, FC1, and FC2 matrices, applies symmetric
per-output INT8 quantization (`amax/127`, RNE, `[-127,127]`, zero scale 1), and
writes a separate cache. It does not overwrite the source model or expose an
inference switch.

Cache v1 has a strict, fixed layout:

- `manifest.json` plus `block-00.safetensors` through
  `block-49.safetensors`;
- six tensors per block: I8 weight and F32 output-row scale for
  `96768x2688` AdaLN, `28672x5376` FC1, and `5376x14336` FC2;
- a canonical source digest over the names, dtypes, shapes, and payloads of
  every participating base and turbo-LoRA tensor;
- exact file sizes and SHA-256 digests for all 50 payload files;
- schema verification for all 300 tensors and non-overwriting, atomic
  directory publication only after full validation.

The physical-GPU-4 build produced 491,847,424 bytes per block, about 23 GiB in
total. Its source digest was
`2c3da2ccd63ef7cc27493c8ee8d2800a0f28c0570f22fa0965b9eb4172439060`.
All files and tensors passed the post-build verifier, and the complete build
took 401.713 seconds. The first portable SHA implementation spent 184.322
seconds hashing source tensors and 90.267 seconds rechecking cache files. The
final implementation optionally resolves libcrypto EVP at runtime, uses the
host's SHA-NI path when available, and retains the self-contained C fallback.
The final read-only verification took 30.695 seconds for the source, 14.324
seconds for the cache, and 45.06 seconds wall in total. Both accelerated and
forced-portable known-vector/range tests pass.

The HIP operator foundation remains useful research infrastructure:
per-row BF16-to-I8 quantization, rocBLAS I8xI8-to-I32 GEMM, BF16 dequantization,
and a bounded reusable I32 workspace. Its CPU oracle is bitwise for input and
weight I8 values, F32 scales, and final BF16 output on GPU 4. It remains
unreachable from the normal VDN and generic H3 runtime paths.

Four same-process production-shape 50-layer comparisons determined the runtime
decision. Percentages below are relative RMSE; the frozen combined 50-layer
gate is at most 2.5% with cosine at least 0.9995, and final production E2E must
be at least 10% faster.

| Candidate | BF16 -> candidate wall | Combined RMSE / cosine | Video RMSE / cosine | Audio RMSE / cosine | Decision |
|---|---:|---:|---:|---:|---|
| INT8 weights + dynamic INT8 activations | 33.073876 -> 28.660565 s (-13.34%) | 16.4616% / 0.988139 | 12.4701% / 0.993238 | 46.3774% / 0.922521 | reject quality |
| all cached weights dequantized to BF16 | 31.373132 -> 28.391895 s (-9.50%) | 3.20946% / 0.999512 | 2.58371% / 0.999699 | 8.32546% / 0.996534 | reject RMSE/performance |
| MLP-only I8-to-BF16 | 29.767053 -> 29.120509 s (-2.17%) | 3.04473% / 0.999562 | 2.55066% / 0.999705 | 7.36706% / 0.997294 | reject RMSE/performance |
| AdaLN-only I8-to-BF16 | 30.514340 -> 28.838376 s (-5.49%) | 1.68684% / 0.999858 | 0.611624% / 0.999982 | 6.56324% / 0.997846 | early quality pass, reject E2E potential |

The full-INT8 and MLP candidates failed the 50-layer gate, so the hierarchy
stopped before eight-NFE and decoded-media runs. AdaLN-only passed the combined
early quality threshold, but its 5.49% forward gain cannot produce a 10%
production E2E gain once the roughly 110-second video VAE and 45-second strict
cache/source verification are included. Running another roughly nine-minute
latent A/B could not change that performance conclusion, so it was also
stopped at the prescribed gate.

All temporary runtime cache parsing, component selection, block fields,
AdaLN/MLP branches, and comparison hooks were removed. The cache implementation
is linked only by its builder and tests, not by `h3`. A clean-build default
production forward subsequently took 30.664280 seconds and reproduced the
frozen video/audio hashes `b3d3500676d3fb12` and `5fbd7afb3d78a277`, with zero
natural POTRF retries. The GPU was physical card 4/BDF `e3:00.0`, and the
concurrency guard was empty.

Reproduce the retained research artifacts with:

```sh
make BACKEND=hip HIP_ARCHS=gfx1201 \
  h3_vdn_int8_tests h3_vdn_int8_cache_tests h3_vdn_int8_cache_builder

scripts/profile_vdn_gpu4.sh outputs/int8-operator-gpu4 -- \
  ./h3_vdn_int8_tests

scripts/profile_vdn_gpu4.sh outputs/int8-cache-build-gpu4 -- \
  ./h3_vdn_int8_cache_builder \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/int8-cache-stage-dmd-turbo-v1 1

./h3_vdn_int8_cache_builder --verify \
  models/vdn-minimax-h3/h3-base \
  models/vdn-minimax-h3/stage-dmd-step-250 \
  models/vdn-minimax-h3/int8-cache-stage-dmd-turbo-v1 1
```

### FP8 model-weight feasibility on gfx1201

The remaining FP8 half of the reduced-precision plan was evaluated separately
after the INT8 cache decision. ROCm 7.2.3 rocBLAS 5.2 exposes no FP8 datatype,
but the installed hipBLASLt 1.2.2 and gfx1201 wave32 ISA do support OCP E4M3.
`h3_vdn_fp8_gemm_bench` therefore adds an explicit hipBLASLt link only to an
isolated research executable. The main `h3` link line and direct `DT_NEEDED`
set remain unchanged; this ROCm package's rocBLAS already loads hipBLASLt
transitively.

The operator oracle uses non-uniform inputs and explicit zero rows; CPU and
GPU E4M3 bytes, F32 row scales, and final BF16 output all had zero mismatches.
The benchmark uses the runtime's row-major tensors through an equivalent
column-major view, quantizes input and weight per output row with E4M3
`amax/448`, runs FP8xFP8-to-F32, applies both row scales, and converts the
result to BF16. A 16x16x128 capability check returned 32 algorithms and
reproduced the analytical and BF16 value exactly. Production-shape results on
physical GPU 4/BDF `e3:00.0` were:

| Matrix | BF16 rocBLAS | FP8 quant + hipBLASLt + dequant | Speedup |
|---|---:|---:|---:|
| AdaLN, `3x2688` by `96768x2688` | 0.867067 ms | 0.437810 ms | 1.980462x |
| FC1, `5338x5376` by `28672x5376` | 10.321590 ms | 7.833940 ms | 1.317548x |
| FC2, `5338x14336` by `5376x14336` | 5.557861 ms | 4.232268 ms | 1.313211x |

All four runs, including the capability check, exited zero on gfx1201 and had
empty concurrency guards. Each result is the median of five crossed groups;
all raw group times are emitted by the benchmark. The three GEMM deltas total
only about 0.212 seconds per 50-layer NFE. Even an optimistic additive estimate
that also halves the historical read and H2D time for the 45.76 GiB/NFE
AdaLN/MLP payload saves only about 3.3 seconds/NFE, or roughly 7.5% of the
354.4-second production E2E baseline over eight NFEs. Real overlap and cache
verification can only reduce that bound, so this format cannot independently
meet the 10% E2E gate.

Quality was checked against the actual turbo-merged effective weights without
creating another 23 GiB cache: a temporary loader hook performed the exact
per-row E4M3 quantize/dequantize operation that such a cache would introduce,
then ran a same-process wave32 50-layer comparison. These timings include the
temporary BF16 load/roundtrip and are not used as performance evidence.

| Quantized component | Combined RMSE / cosine | Video RMSE / cosine | Audio RMSE / cosine | Decision |
|---|---:|---:|---:|---|
| AdaLN + FC1 + FC2 | 4.79686% / 0.998927651 | 4.51494% / 0.999062897 | 8.10835% / 0.996736490 | reject |
| FC1 + FC2 only | 4.93898% / 0.998862535 | 4.55417% / 0.999058255 | 9.15756% / 0.995801518 | reject |
| AdaLN only | 2.57055% / 0.999670840 | 0.833963% / 0.999967403 | 10.1416% / 0.994864878 | reject |

Every output was finite and all natural POTRF retry counts were zero. The full
and MLP-only candidates clearly exceed the 2.5% combined 50-layer gate;
AdaLN-only also exceeds it narrowly and has severe modality-specific audio
error. The hierarchy therefore stopped before eight-NFE and decoded-media
tests. All temporary loader, environment, GPU API, and forward-test hooks were
removed after attribution. No FP8 cache v2 was generated, no FP8 runtime mode
is exposed, and the supported BF16 path is unchanged.

Reproduce the retained performance probe with:

```sh
make BACKEND=hip HIP_ARCHS=gfx1201 \
  h3_vdn_fp8_tests h3_vdn_fp8_gemm_bench
scripts/profile_vdn_gpu4.sh outputs/fp8-operator-gpu4 -- \
  ./h3_vdn_fp8_tests
scripts/profile_vdn_gpu4.sh outputs/fp8-adaln-gpu4 -- \
  ./h3_vdn_fp8_gemm_bench 3 96768 2688 10
scripts/profile_vdn_gpu4.sh outputs/fp8-fc1-gpu4 -- \
  ./h3_vdn_fp8_gemm_bench 5338 28672 5376 5
scripts/profile_vdn_gpu4.sh outputs/fp8-fc2-gpu4 -- \
  ./h3_vdn_fp8_gemm_bench 5338 5376 14336 5
```

### KEEP research-only: video-VAE F32 GEMM fast solution

Production-shape profiling showed that FC1, FC2, QKV, and attention-output
projections account for 99.2% of video-VAE linear event time. Exact rocBLAS and
hipBLASLt solution searches found at most a 4.8% bitwise-exact FC1 gain, below
the operator gate. The gfx1201/rocBLAS-5.2.0 solution-index 91217 path is not
bitwise exact, so it is available only through explicit research modes; unset,
`standard`, and `exact` preserve `rocblas_gemm_algo_standard`.

`H3_VAE_F32_GEMM=fast-all` applies only to the four registered production
`M=2273` shapes. Three crossed VAE runs reduced wall median from 108.871191 to
86.176618 seconds (-20.85%) and linear event median from 53.713 to 31.244
seconds (-41.83%). The candidate RGB-F32 hash was deterministic in all three
runs. Its paired raw decoded-video relative RMSE/cosine were
0.000122535%/0.999999999999232. Three full prompt renders passed video
PSNR/SSIM/temporal, exact audio, container, and A/V-sync gates. This saves about
22.7 seconds, or roughly 6.4% of the frozen 354.4-second production E2E, so it
does not independently satisfy the 10% release-promotion target. It remains
explicit and approximate.

A production-shape `rocprofv3` trace also bounded all proposed neighboring
fusion work. Bias, QKV prep, SwiGLU, scale/residual-add, and RMSNorm together
were only 185.877 ms, 0.709% of GPU kernel event and 0.659% of wall. Those
fusion candidates were rejected by upper bound instead of adding high-risk
custom GEMM code.

### REJECT runtime: E33 compensated BF16 SageAttention

SageAttention-AMD commit `c74ea30` introduced the clean E33 operator, and
commit `288fbca` recorded immutable GPU4 evidence. The final H3 gitlink points
to `37e838b`, which also records the downstream failure. Clean E33 operator
event median was 19.247 ms versus 462.824 ms for the paired exact wave32 sample;
the H3 bridge measured 411.463 to 19.191 ms (21.440x), with relative RMSE
0.000190124 and cosine 0.999999982.

Prompt 2 passed the 50-layer gate, including audio RMSE/cosine
4.02281%/0.999193881. The required eight-NFE comparison then reduced DiT wall
from 305.908 to 154.313 seconds and passed video at
0.21219%/0.999997758, but audio accumulated to
10.68372%/0.994280444 and failed both limits. A third BF16 probability residual
was tested as E33-R4; it passed operator correctness but slowed 19.247 to
23.209 ms and worsened RMSE from 0.000215745 to 0.000266111. R4 was reverted,
three-prompt media was not run, and H3's temporary E33 runtime mode/bridge
branch was removed. `auto`, exact wave32, and the existing E27 research mode
retain their previous semantics.

### REJECT operator: block-scaled weight-only I8/FP8

The follow-up deliberately changed both the v1 per-row error model and its
full-BF16 materialization path. An isolated gfx12 probe keeps activations BF16,
stores output-channel/K-group I8 weights and F32 scales, dequantizes only a
`16x256` tile into LDS, and consumes it immediately with BF16 WMMA. Its small
GPU4 contract passed quant byte/scale oracle, output/metadata canaries, finite,
and bitwise determinism. Installed hipBLASLt 1.2.2 returned zero heuristics for
both mixed BF16-activation/block-FP8-weight and dual-block-FP8 descriptors on
gfx1201.

The fused fallback failed the production operator gate by a large margin:
AdaLN was 3.034--3.089 ms versus 0.866--0.873 ms BF16; FC1 was 242.500 versus
12.332 ms; FC2 was 126.227--126.936 versus 6.579--6.704 ms. K-groups
16/32/64/128/256 were screened on AdaLN, and 32/128 on FC2. Expanding LDS
staging from K=16 to K=256 reduced barrier count sixteen-fold without improving
FC1, showing that repeated I8-to-BF16 conversion per M tile dominates. No
plausible tile adjustment can bridge the observed roughly 19--20x deficit in
the absence of a mixed native instruction/library kernel.

The candidate therefore stopped before single-block, 50-layer, 8-NFE, and
media gates. No cache or runtime mode was created; the existing BF16 loader and
default are unchanged. `h3_vdn_block_weight_gemm_bench` is retained so a future
ROCm/hipBLASLt gfx1201 mixed block-scale implementation can be retested without
rebuilding the experiment.

```sh
make BACKEND=hip HIP_ARCHS=gfx1201 h3_vdn_block_weight_gemm_bench

# Small byte/scale/canary/finite/determinism contract and native capability.
scripts/profile_vdn_gpu4.sh outputs/block-weight-contract-gpu4 -- \
  env H3_BLOCK_WEIGHT_QUERY_NATIVE=1 \
  ./h3_vdn_block_weight_gemm_bench 17 19 48 16 2

# Representative production operator gates (M N K group iterations).
scripts/profile_vdn_gpu4.sh outputs/block-weight-fc1-gpu4 -- \
  ./h3_vdn_block_weight_gemm_bench 5338 28672 5376 128 3
scripts/profile_vdn_gpu4.sh outputs/block-weight-fc2-gpu4 -- \
  ./h3_vdn_block_weight_gemm_bench 5338 5376 14336 32 3
```

## Test gates

Build and run the local gates with:

```sh
make BACKEND=hip -j16 \
  h3_vdn_gpu_ops_tests h3_vdn_feature_tests \
  h3_vdn_solve_tests h3_vdn_scan_tests h3_vdn_forward_smoke_tests \
  h3_f32_sdpa_bench h3_vdn_video_vae_smoke_tests \
  h3_vdn_sage_tests h3_vdn_sage_sdpa_bench
make BACKEND=hip HIP_ARCHS=gfx1201 \
  sage-upstream-contract-test sage-upstream-isa-test

HIP_VISIBLE_DEVICES=4 ./h3_vdn_gpu_ops_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_feature_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_solve_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_scan_tests
./h3_vdn_sage_tests
HIP_VISIBLE_DEVICES=4 ./h3_vdn_sage_sdpa_bench

scripts/profile_vdn_gpu4.sh outputs/sage-upstream-gpu-test-gpu4 -- \
  make BACKEND=hip HIP_ARCHS=gfx1201 H3_PHYSICAL_GPU=4 \
  sage-upstream-gpu-test

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

The final clean-build regression passed 1,774 core checks, 28,512,043 Sage
mask/workspace checks, backend/storage/core/DiT/VDN operator suites, INT8
quantization and WMMA oracles, default and 2 GiB resident-loader parity, input
contracts, refiner/block smoke, and both VAEs. The default production 50-layer
forward completed in 29.612809 seconds with frozen video/audio hashes
`b3d3500676d3fb12` and `5fbd7afb3d78a277`. The public 64x32/56-frame/8-NFE
CLI completed in 72.754850 seconds and reproduced all five frozen internal
hashes plus the 73,528-byte MP4 SHA-256
`7a447fe6f63697ad1bbb2df8a74f385b432ce3a1d5855caee5f8c1531a955c5b`.
Its schema-v2 record reported `attention.requested_mode=auto`,
`approximate=false`, eight NFE entries, LoRA timing, and a zeroed default
resident-cache record. Every formal GPU gate used BDF `0000:e3:00.0` and had
an empty concurrency guard.

The 2026-09-08 POTRF-guard clean-build regression repeated the complete
backend/storage/DiT/VDN operator group, including deterministic silent
corruption/retry injection. Input contracts, both loader modes, refiner, block,
the production-shape 50-layer forward, both VAEs, and the 64x32 8-NFE E2E all
returned status 0 on physical GPU 4 with empty concurrency guards. The final
small E2E retained the five frozen hashes and the same 73,528-byte MP4 SHA-256;
the 50-layer run reported zero natural POTRF retries.

The subsequent P7 INT8-cache study also ended with the runtime candidate fully
removed. Its final clean build passed the 1,774 host checks, JSON, accelerated
and forced-portable SHA-256, VDN metadata/reference/prompt/cache contracts, the
GPU-4 INT8 CPU oracle, and default real block loading. The restored production
BF16 forward took 30.664280 seconds, reproduced hashes
`b3d3500676d3fb12`/`5fbd7afb3d78a277`, and reported zero natural retries on
BDF `e3:00.0` with an empty guard.

## Next priorities

1. Keep exact BF16 as the stable default; any new Sage candidate must first
   reduce 8-NFE audio latent RMSE below 5% before decoded-media work resumes.
2. Do not expose the v1 per-output INT8 weight cache at runtime. Any future
   groupwise INT8 or FP8 design must define a new version, pass the 50-layer
   audio-sensitive gate first, and retain credible 10% production-E2E potential
   after verification and VAE costs.
3. Do not revisit scan fusion without a new profile showing a materially larger
   hotspot than the current roughly 0.1 seconds/NFE.
4. Keep all formal acceptance on physical GPU 4. Multi-GPU sharding remains
   deferred.
