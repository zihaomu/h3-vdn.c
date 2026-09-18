# MiniMax-H3 Sage E33 760 秒性能剖析

更新日期：2026-09-18

状态：`FROZEN_PROFILE_BASELINE_OPERATOR_CONTRACTS_AUDITED`

关联基线：
[`H3_SAGE_E33_640X384_EXPERIMENTAL_BASELINE.md`](H3_SAGE_E33_640X384_EXPERIMENTAL_BASELINE.md)

## 1. 分析对象

本文根据已有运行日志分析以下本地实验，不重新生成视频：

```text
/tmp/h3-sage-stage3-frozen-run1.4lpYag/run.log
/tmp/h3-sage-stage3-frozen-run1.4lpYag/minimax-h3-sage-e33-50nfe.mp4
```

冻结条件：

| 项目 | 值 |
|---|---|
| Parent source | `65a6b350d24bcc8af54ed0b035c70011e8f64224` |
| SageAttention-AMD pin | `133b53168e3e8f9a3a059cd31aac7e69d802e89e` |
| Attention | `H3_BF16_SDPA=sage-e33` |
| 模型 | MiniMax-H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe` |
| 输出 | 640×384，124 帧，24 fps，容器 5.175 秒 |
| 推理 | 50 NFE，50 DiT blocks，reuse 1 |
| Resident blocks | 20 |
| GPU | 单张 `gfx1201`，物理 card 2，PCI `0000:43:00.0` |

`run.log` 文件时间从 `2026-09-18 10:38:28.422 +08:00` 到
`2026-09-18 10:51:09.248 +08:00`，完整端到端 wall time 为
`760.826 秒`，对外取整为 `761 秒（12:41）`。下文所有“整体占比”均以
`760.826 秒`为分母。

## 2. 端到端模块耗时

以下模块 wall time 互斥，可加总回完整运行时间：

| 模块 | Wall time | 整体占比 |
|---|---:|---:|
| Qwen 文本编码 | `4.010 s` | `0.53%` |
| DiT 初始加载 | `5.591 s` | `0.73%` |
| 50-NFE DiT Euler 去噪 | `502.222 s` | `66.01%` |
| Audio VAE 解码 | `3.494 s` | `0.46%` |
| Video VAE 解码 | `241.430 s` | `31.73%` |
| 初始化、帧输出、mux、日志及舍入差 | `4.079 s` | `0.54%` |
| **完整进程** | **`760.826 s`** | **`100.00%`** |

核心结论：DiT 去噪与 Video VAE 合计 `743.652 秒`，占完整生成的
`97.74%`。文本编码、Audio VAE 和 mux 均不是当前主要瓶颈。

## 3. DiT 去噪内部构成

50-NFE DiT 去噪的 `502.222 秒`可近似拆为：

| DiT 项目 | 时间 | 占 DiT 去噪 | 占完整生成 |
|---|---:|---:|---:|
| Sage E33 SDPA | `180.267 s` | `35.89%` | `23.69%` |
| Linear/GEMM | `148.555 s` | `29.58%` | `19.53%` |
| BF16 SSD 权重流未隐藏等待 | `150.043 s` | `29.88%` | `19.72%` |
| 调度及其他开销 | `23.357 s` | `4.65%` | `3.07%` |
| **DiT 去噪** | **`502.222 s`** | **`100.00%`** | **`66.01%`** |

对应原始日志摘要：

```text
h3 profile: H3 DiT Euler denoise wall=502.222s
h3 profile: H3 DiT gpu-op-classes measured=328.823s
  linear=148.555s sdpa=180.267s
h3: BF16 SSD stream 1077.378 GiB read in 359.642s,
  unhidden wait 150.043s; resident blocks=20 (14.355 GiB)
```

Sage E33 当前直接优化的是 DiT SDPA，其时间占完整生成的 `23.69%`，而不是整个
DiT 的 `66.01%`。权重流未隐藏等待已经与 SDPA、Linear/GEMM 处于同一量级。

## 4. Video VAE 内部构成

| Video VAE 项目 | 时间 | 占 Video VAE | 占完整生成 |
|---|---:|---:|---:|
| Linear/GEMM | `147.850 s` | `61.24%` | `19.43%` |
| SDPA | `89.713 s` | `37.16%` | `11.79%` |
| 其他开销 | `3.867 s` | `1.60%` | `0.51%` |
| **Video VAE** | **`241.430 s`** | **`100.00%`** | **`31.73%`** |

对应原始日志摘要：

```text
h3 profile: video VAE decoder total wall=241.430s
h3 profile: video VAE decoder gpu-op-classes measured=237.562s
  linear=147.850s sdpa=89.713s
```

Video VAE 本身接近完整运行的三分之一，其中 `98.40%` 的模块 wall time 可由已记录的
Linear/GEMM 和 SDPA GPU event 解释。只优化原始 H3 DiT attention 不会自动减少这部分
VAE 时间。

## 5. 全链路已计时 GPU 算子

将 Qwen、DiT load、DiT denoise 和 Video VAE 的 GPU event 合并：

| 算子类别 | 累计时间 | 整体占比 |
|---|---:|---:|
| Linear/GEMM | `297.382 s` | `39.09%` |
| SDPA/Attention | `270.389 s` | `35.54%` |
| **已计时 GPU 算子** | **`567.772 s`** | **`74.63%`** |

SDPA/Attention 的来源为：

| Attention 来源 | 时间 | 整体占比 |
|---|---:|---:|
| DiT Sage E33 | `180.267 s` | `23.69%` |
| Video VAE SDPA | `89.713 s` | `11.79%` |
| Qwen text attention | `0.408 s` | `0.05%` |
| DiT load/refiner attention | `0.001 s` | `<0.01%` |
| **合计** | **`270.389 s`** | **`35.54%`** |

Linear/GEMM 的来源为：

| Linear 来源 | 时间 | 整体占比 |
|---|---:|---:|
| DiT denoise | `148.555 s` | `19.53%` |
| Video VAE | `147.850 s` | `19.43%` |
| Qwen text encoder | `0.909 s` | `0.12%` |
| DiT load/refiner | `0.068 s` | `0.01%` |
| **合计** | **`297.382 s`** | **`39.09%`** |

由于日志按毫秒级小数输出，分类相加与 `gpu-op-classes measured` 之间可能有约
`0.001 秒`的舍入差。

## 6. 瓶颈排序

按完整生成时间占比排序：

| 优先级 | 瓶颈 | 时间 | 整体占比 |
|---:|---|---:|---:|
| 1 | DiT Sage E33 SDPA | `180.267 s` | `23.69%` |
| 2 | DiT SSD 未隐藏等待 | `150.043 s` | `19.72%` |
| 3 | DiT Linear/GEMM | `148.555 s` | `19.53%` |
| 4 | Video VAE Linear/GEMM | `147.850 s` | `19.43%` |
| 5 | Video VAE SDPA | `89.713 s` | `11.79%` |

前五项合计约占完整运行的 `94.16%`。后续优化必须报告完整 E2E wall time，不能只用
attention microbenchmark 推算最终收益。

## 7. 计时口径说明

以下指标不能直接相加：

- `weight-load read` 是多个预取 lane 的累计读取时间；
- `upload` 可能与 GPU 计算、CPU 读取并发；
- `host-wait` 包含不同加载阶段和同步点；
- GPU event 时间可能与 CPU/SSD 工作重叠；
- `encode`、`wait` 是 profiler 内部子口径，不是额外添加到模块 wall time 的独立阶段。

例如 Qwen 的日志同时报告 `read=10.407 s`、`upload=3.325 s`，但模块 wall time 只有
`4.010 s`，这正是并行预取和累计 lane 计时的结果。把这些数字相加会重复计算。

本文只在模块层使用互斥 wall time；在算子层使用 GPU event；只有日志明确标记为
`unhidden wait` 的 `150.043 秒`才作为 DiT wall 的可见 I/O 等待参与近似拆分。

## 8. 后续优化基线

任何后续优化结果应至少与下列数字比较：

- 完整 E2E：`760.826 s`；
- DiT denoise：`502.222 s`；
- DiT Sage SDPA：`180.267 s`；
- DiT Linear/GEMM：`148.555 s`；
- DiT SSD unhidden wait：`150.043 s`；
- Video VAE：`241.430 s`；
- Video VAE Linear/GEMM：`147.850 s`；
- Video VAE SDPA：`89.713 s`。

比较时必须保持同一模型 revision、prompt、seed、640×384、124 帧、50 NFE、50
blocks、reuse 1、20 resident blocks、单 GPU 和相同质量门禁。性能提升不能替代
tensor、媒体、确定性和视觉质量验证。

## 9. 热点 shape 的来源

### 9.1 DiT production sequence

本次长 prompt 的 DiT 实际 sequence 为 `9322`，组成如下：

| Token 类型 | 行数 | 来源 |
|---|---:|---|
| Text | `28` | Qwen 编码后的 prompt tokens |
| Audio | `414` | 双声道 audio latent rows |
| Video | `8880` | `37 × 12 × 20` video latent patch rows |
| **合计** | **`9322`** | dense multimodal self-attention sequence |

因此，Stage 3 standalone registry 中的 `S=9300` 是接入阶段使用的固定 production
近似 shape；本文分析的正式长 prompt E2E 实际 shape 是 `S=9322`。后续 kernel
benchmark 必须新增或显式传入 `S=9322`，不能用 S9300 的结果代替。

DiT 固定模型维度为：

```text
hidden = 5376
heads = 56
head_dim = 128
attention inner = 7168
FFN hidden = 14336
```

### 9.2 Video VAE production sequence

日志显示空间 tiling 为 `3×2`、每 tile `256×256` 像素：

- 每个空间 tile 对应 `16×16` latent；
- 每个 temporal chunk 使用 7 个 latent time slices；
- patch rows 为 `7 × 16 × 16 = 1792`；
- 再加入 4 个 register tokens 和 1 个 suffix token；
- 每次 VAE Transformer 的实际 sequence 为 `1792 + 5 = 1797`；
- 124 帧对应 7 个重叠 temporal chunks；
- 7 chunks × 6 spatial tiles = 42 次 tile/chunk decoder pass；
- 每次 pass 运行 36 个 VAE Transformer blocks，所以主算子各调用
  `42 × 36 = 1512` 次。

空间 tile 的 X 方向 overlap 为 64 像素，Y 方向 overlap 为 128 像素。Video VAE
固定模型维度为：

```text
hidden = 2048
heads = 32
head_dim = 64
attention inner = 2048
FFN hidden = 8192
```

当前代码中已有的 VAE fast GEMM 和 split-score 自动登记仅覆盖
`S/M=2273`。本次 `S/M=1797` 不会自动使用那组优化，必须作为新的独立 shape
重新验证。

## 10. GEMM 的统一输入输出契约

日志的 `M/N/K` 对应以下逻辑 row-major 运算：

```text
X: [M, K]
W: [N, K]
B: [N]，仅 bias=1 时存在
Y: [M, N]
Y = X × W^T + B
```

HIP 后端使用 `rocblas_gemm_ex`。当前热点 GEMM 均使用
`rocblas_gemm_algo_standard`、`solution=0`，乘加 compute type 为 F32：

- BF16 GEMM：X/W/Y 存储为 BF16，内部乘加为 F32，结果舍入写回 BF16；
- F32 GEMM：X/W/Y/B 均为 F32，内部乘加及结果均为 F32；
- 权重逻辑布局为连续 `[N,K]`；
- activation 逻辑布局为连续 `[M,K]` 或 `[M,N]`；
- 日志中的 event time 包括 GEMM；`bias=1` 时还包含随后独立执行的 bias kernel。

后续替代算子必须显式说明：输入、权重、累加和输出 dtype；转置语义；bias 是否融合；
以及是否改变 reduction order。只写“BF16 GEMM”或“F32 GEMM”不足以定义数值契约。

## 11. 子任务 A：DiT Sage E33 dense SDPA

### 11.1 输入输出契约

| 字段 | 冻结值 |
|---|---|
| 语义 | dense、non-causal、self-attention |
| Batch | `1` |
| Q/K/V shape | `[9322,56,128]`，逻辑 NHD，可写为 `[1,9322,56,128]` |
| Q/K/V storage | BF16 |
| Output shape/storage | `[9322,56,128]`，BF16 |
| Scale | `1/sqrt(128) = 0.0883883476...` |
| Mask | 每个 query 允许完整 key 区间 `[0,9322)` |
| Output alias | output 不允许与 Q、K 或 V 指针重叠 |
| 调用次数 | `50 NFE × 50 blocks = 2500` |
| 累计 event | `180.267 s` |
| 单次平均 | `72.107 ms` |
| 每 NFE | `3.605 s` |
| 整体占比 | `23.69%` |

张量无 padding，D 是最内层连续维度。每个 head stride 为 128 BF16 elements
（256 bytes），每个 sequence row stride 为 7168 BF16 elements（14,336 bytes）。
Q、K、V、Output 每个包含 `66,820,096` 个 BF16 elements，即 `127.449 MiB`。

Q/K/V 不是原始 linear 输出。进入 E33 前，当前 H3 已经完成：

- Q/K per-head RMS normalization，epsilon `1e-5`；
- Q/K RoPE，D128 中前 96 维参与旋转（`ROPE_HALF=48`）；
- fused QKV BF16 buffer 拆分为连续 NHD Q、K、V；
- V 不进行 RMSNorm/RoPE。

任何独立 attention 子任务应从这些“已处理完成”的 Q/K/V 开始，不能重复执行 QK norm
或 RoPE。

### 11.2 当前 E33 数值与 kernel 结构

当前 kernel id 为 `e33_bf16_qk_gfx12_d128`，仅登记：

```text
gfx1201 + wave32 + batch1 + NHD + BF16 input/output
D128 + self-attention + ordered-interval-v1
```

当前执行细节：

- QK：BF16 WMMA，F32 accumulate；
- softmax：F32 streaming online softmax；
- PV probability：先转为 BF16 high，再计算 BF16 residual；
- PV：high 和 residual 分别执行一次 BF16 WMMA，两次均 F32 accumulate；
- 最终除以 F32 denominator 后舍入为 BF16 output；
- 不物化完整 `S×S` score matrix；
- Q tile 为 32 rows，K tile 为 16 rows；
- `ceil(9322/32)=292` 个 query tasks，最后一个 task 为 10 rows；
- 每 task 只有一个 `[0,9322)` interval；
- launch grid 为 `(292,56)`，共 16,352 blocks/call，block 为 64 threads；
- task workspace 为 `15,360 bytes`，geometry 未变化时跨 2500 次调用复用；
- 每次理论 QK+PV 主乘加量约 `2.492 TFLOP`，按 72.107 ms 计算的等效值约
  `34.55 TFLOP/s`。该值不含 softmax、边界与访存操作，不能当硬件峰值利用率。

当前正确性证据：

- production same-QKV E33 对 rocBLAS：max abs `0.000122070312`、
  relative-L2 `0.00366749022`、cosine `0.999993331063`、non-finite 0；
- E33 重复运行 bitwise deterministic；
- official 50-block/2-NFE DiT oracle 通过；
- 完整视频人工质量通过，但对 frozen matrix render 的
  SSIM/PSNR `0.378892 / 13.944662 dB` 未过 promotion gate。

### 11.3 子任务交付要求

建议该子任务只修改 SageAttention-AMD 的独立 kernel id 或 H3 dense bridge：

1. benchmark 必须使用真实 `S=9322,H=56,D=128`；
2. 报告 kernel event、单 NFE event、workspace、VGPR、LDS、occupancy 和 ISA；
3. 至少 3 个独立 session，每 session warm-up ≥3、iterations ≥10；
4. output canary、workspace canary、non-finite、20 次 hash determinism 必须通过；
5. exact 变体应对当前 E33 bitwise；非 exact 变体必须重新跑 official DiT 和完整媒体门禁；
6. 以单次 `72.107 ms` 为基线，第一阶段 10% 目标为 `≤64.896 ms`；
7. 即使更快，也不得因当前 frozen-render 质量门禁失败而自动提升为默认 backend。

## 12. 子任务 B：DiT BF16 Linear/GEMM

四个主 GEMM 占 DiT linear event 的 `148.540 / 148.555 s`，其余 final heads 等
linear 仅约 `0.015 s`。

| 语义 | X `[M,K]` | W `[N,K]` | Y `[M,N]` | Bias | Calls | Event | 平均/call | 等效 TFLOP/s | 整体占比 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| QKV projection | `[9322,5376]` | `[21504,5376]` | `[9322,21504]` | 无 | 2500 | `44.322 s` | `17.729 ms` | `121.57` | `5.83%` |
| Attention output | `[9322,7168]` | `[5376,7168]` | `[9322,5376]` | 无 | 2500 | `15.026 s` | `6.010 ms` | `119.53` | `1.98%` |
| FFN input/FC1 | `[9322,5376]` | `[28672,5376]` | `[9322,28672]` | 无 | 2500 | `56.846 s` | `22.738 ms` | `126.39` | `7.47%` |
| FFN output/FC2 | `[9322,14336]` | `[5376,14336]` | `[9322,5376]` | 无 | 2500 | `32.346 s` | `12.938 ms` | `111.06` | `4.25%` |

所有四项的 X/W/Y 均为 BF16，rocBLAS compute type 为 F32。QKV 的 N=21504 是
`3 × 56 × 128`。FC1 的 `[9322,28672]` 输出按 `[value,gate]` 两半存储，随后
计算 `value * SiLU(gate)` 得到 `[9322,14336]` BF16 activation，再进入 FC2。

每次调用的存储规模：

| 语义 | X | W | Y |
|---|---:|---:|---:|
| QKV | `95.587 MiB` | `220.500 MiB` | `382.348 MiB` |
| Attention output | `127.449 MiB` | `73.500 MiB` | `95.587 MiB` |
| FC1 | `95.587 MiB` | `294.000 MiB` | `509.797 MiB` |
| FC2 | `254.898 MiB` | `147.000 MiB` | `95.587 MiB` |

四项合计 `2.971 s/NFE`。建议子任务优先级为 FC1、QKV、FC2、attention output。
交付要求：

1. 四个 shape 分开 benchmark，不能只测其中一个后外推；
2. 同 session 比较 `rocblas_gemm_algo_standard/solution=0`；
3. 报告 cold/warm、workspace、峰值显存和 effective TFLOP/s；
4. 若保持 BF16 output bitwise，应通过完整 hash gate；若改变 reduction order，必须报告
   max abs、relative-L2、cosine、non-finite 并跑 50-block/2-NFE/完整视频；
5. aggregate linear event 的 10% 目标为 `≤133.700 s`，不能用单个 FC1 的提升代表整体；
6. 不得把 INT8/FP8 结果混入 BF16 exact 结果；量化必须作为独立数值模式和独立子任务。

## 13. 子任务 C：DiT SSD 权重流

该项不是 GPU 数学算子，但其 `150.043 s` unhidden wait 是第二大 wall hotspot。

每个 DiT block 流式读取四组 BF16 matrices：

| Weight | Shape | 单 block 大小 |
|---|---|---:|
| fused QKV | `[21504,5376]`；现代 checkpoint 实际由三个 `[7168,5376]` 拼接 | `220.500 MiB` |
| attention output | `[5376,7168]` | `73.500 MiB` |
| FC1 | `[28672,5376]` | `294.000 MiB` |
| FC2 | `[5376,14336]` | `147.000 MiB` |
| **每 block 合计** | 4 个逻辑矩阵、现代格式为 6 个文件区间 | **`735.000 MiB` / `0.717773 GiB`** |

当前配置：

- 前 20 个 active blocks 常驻，占 `14.355 GiB`；
- 后 30 个 blocks 使用两个交替 device stream slots；
- 每层读取按 safetensors 文件路径和 file offset 排序；
- 每轮计算当前 block 时，pthread 预取下一非 resident block；
- 50 NFE 的 30 个流式 blocks 加一次 prime，共 1501 block-equivalent reads；
- `1501 × 0.717773 GiB = 1077.378 GiB`，与日志完全一致；
- 累计 read span `359.642 s`，平均 `2.996 GiB/s`；
- 真正暴露在 wall 上的 join wait 为 `150.043 s`，即 `3.001 s/NFE`、完整生成
  `19.72%`。

该子任务必须保持数学路径不变，因此输出应 bitwise exact。建议分别探索：

1. `H3_DIT_RESIDENT_BLOCKS` 安全上调 sweep；当前 peak `18.474 GiB`，每增加一个
   resident block 原始矩阵预算约增加 `0.718 GiB`，但必须保留 activation、双 stream
   slots、Sage workspace 和驱动余量；
2. block 级 host cache/pinned cache，避免重复磁盘读取；
3. 将同 shard 连续区间合并读取，减少小请求和线程调度；
4. 提前超过一层的有界 prefetch，评估 SSD queue depth；
5. 将 disk read、page-cache hit、host staging、H2D、join wait 分开计时。

第一阶段成功标准：`unhidden wait ≤120.034 s`（下降至少 20%），完整 tensor、latent、
raw frame 和 MP4 hash 不变，单卡 peak allocation 受控。`read=359.642 s` 是累计/并发
口径，优化是否成功仍以 `unhidden wait` 和 E2E wall 为准。

## 14. 子任务 D：Video VAE F32 Linear/GEMM

### 14.1 主 GEMM 契约

| 语义 | X `[M,K]` | W `[N,K]` | Y `[M,N]` | Bias | Calls | Event | 平均/call | 等效 TFLOP/s | 整体占比 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| QKV projection | `[1797,2048]` | `[6144,2048]` | `[1797,6144]` | F32 `[6144]` | 1512 | `25.800 s` | `17.063 ms` | `2.650` | `3.39%` |
| Attention output | `[1797,2048]` | `[2048,2048]` | `[1797,2048]` | F32 `[2048]` | 1512 | `8.373 s` | `5.538 ms` | `2.722` | `1.10%` |
| FFN input/FC1 | `[1797,2048]` | `[16384,2048]` | `[1797,16384]` | F32 `[16384]` | 1512 | `78.347 s` | `51.817 ms` | `2.327` | `10.30%` |
| FFN output/FC2 | `[1797,8192]` | `[2048,8192]` | `[1797,2048]` | F32 `[2048]` | 1512 | `34.943 s` | `23.110 ms` | `2.609` | `4.59%` |

X/W/Y/B、SwiGLU activation 和 residual 全部保持 F32。FC1 的
`[1797,16384]` 输出按 `[gate,value]` 两半存储，计算
`SiLU(gate) * value` 后得到 `[1797,8192]` F32 activation。

每次调用的存储规模：

| 语义 | X | W | Y |
|---|---:|---:|---:|
| QKV | `14.039 MiB` | `48.000 MiB` | `42.117 MiB` |
| Attention output | `14.039 MiB` | `16.000 MiB` | `14.039 MiB` |
| FC1 | `14.039 MiB` | `128.000 MiB` | `112.313 MiB` |
| FC2 | `56.156 MiB` | `64.000 MiB` | `14.039 MiB` |

四个主 GEMM 合计 `147.463 s`，占 VAE linear event 的 `99.74%`。剩余约
`0.387 s` 主要是每 tile/chunk 的 latent embedding 和 `[1797,2048] ->
[1797,3072]` final patch projection，不应先单独优化。

### 14.2 现有优化不能直接复用

`H3_VAE_F32_GEMM=fast-all` 当前仅在 `rows==2273` 且匹配固定 gfx1201/rocBLAS
版本时选择 solution 91217。本次 `rows=1797` 会继续使用 standard solution 0。
既有 M2273 的 41.83% linear event 提升和微小 RGB 误差不能外推到 M1797。

该子任务应对四个 M1797 shape 重新执行 solution search。优先寻找 bitwise-exact
solution；approximate solution 必须独立命名并重新执行 Video VAE RGB oracle、完整视频
SSIM/PSNR/temporal、音频不变和 mux gate。aggregate linear 10% 目标为
`<=133.065 s`，其中 FC1 是单一最大目标。

## 15. 子任务 E：Video VAE F32/D64 SDPA

### 15.1 输入输出契约

| 字段 | 冻结值 |
|---|---|
| 语义 | dense、non-causal、self-attention |
| Batch | `1` |
| Q/K/V shape | `[1797,32,64]`，逻辑 NHD |
| Q/K/V/Output storage | F32 |
| Scale | `1/sqrt(64) = 0.125` |
| Calls | `1512` |
| 累计 event | `89.713 s` |
| 单次平均 | `59.334 ms` |
| 每 tile/chunk pass | `2.136 s` |
| 整体占比 | `11.79%` |

张量无 padding，D 是最内层连续维度。每个 head stride 为 64 F32 elements
（256 bytes），每个 sequence row stride 为 2048 F32 elements（8,192 bytes）。
Q、K、V、Output 每个包含 `3,680,256` 个 F32 elements，即 `14.039 MiB`。

QKV projection 后先执行 per-head Q/K RMS normalization（epsilon `1e-5`）和 RoPE；
D64 中前 48 维参与 RoPE（`ROPE_HALF=24`）。V 保持 projection 原值。处理后的 Q、K、V
和 SDPA output 都是连续 F32 NHD。

### 15.2 当前 kernel

本 shape 走 `h3_hip_sdpa_f32_d64_wave32_kernel`：

- grid `(1797,32)`，每次 57,504 blocks；
- block 32 threads，即一个 wave 负责一个 query/head；
- 每 block 动态 LDS 为 `1797 × 4 = 7,188 bytes`，保存一行 F32 scores；
- QK、max、exp、sum 和 PV 全部 F32；
- lane 0 按 key 顺序执行 max/exp/denominator，PV 保持 key/FMA 顺序；
- 不物化全局 score matrix；
- 每次理论 QK+PV 主乘加约 `26.454 GFLOP`，按 59.334 ms 仅约
  `0.446 TFLOP/s`，调度、串行 softmax 和逐 query 重读 K/V 是明显研究点。

现有 exact split-score kernel 只自动登记 `S=2273,H=32,D=64`。若扩展到 S1797，
所需全局 score+inverse workspace 为：

```text
(32 * 1797 * 1797 + 32 * 1797) * sizeof(float)
= 413,568,768 bytes
= 0.385166 GiB
```

该候选可复用现有相同 QK reduction、softmax traversal 和 PV FMA order，理论上能够做到
bitwise exact，但必须在 S1797 上重新证明。不能直接引用 S2273 的 2.05x 结果。

子任务交付要求：

1. 第一候选为 S1797 exact split-score 注册与 A/B；
2. 比较 one-kernel `59.334 ms`，10% 目标为 `<=53.401 ms`；
3. 全部 `1797*32*64 = 3,680,256` 个 F32 output 必须 bitwise；
4. 检查 workspace/output canary、determinism、peak VRAM 和 42-pass VAE wall；
5. 若研究 Sage 风格版本，必须新增 F32/D64 kernel id 和 registry domain，不能把
   BF16/D128 E33 宣称为可直接复用；
6. 完成 isolated VAE、RGB oracle 和完整 124-frame E2E gate。

## 16. 不应优先拆分的算子

以下项在本次 E2E 中上限太低，暂不建议成为首轮独立优化任务：

| 项目 | 时间/占比 | 原因 |
|---|---:|---|
| Qwen text encoder 整体 | `4.010 s / 0.53%` | 即使完全消除也无法显著改变 E2E |
| Audio VAE 整体 | `3.494 s / 0.46%` | 当前正确性风险高于收益上限 |
| mux/帧输出/其他 | `4.079 s / 0.54%` | 需要先补细分计时，绝对上限低 |
| Video VAE 小投影 | `~0.387 s / 0.05%` | 主 GEMM 已覆盖 99.74% linear event |
| DiT final audio/video heads | `~0.016 s` | 非性能瓶颈 |

## 17. 建议的并行子任务边界

| 子任务 | 独立工作区/代码边界 | 冻结 baseline | 首轮目标 | 必要正确性门禁 |
|---|---|---:|---:|---|
| [A. DiT E33 SDPA](../../kernel_workspace/01-dit-sage-e33-sdpa/01-dit-sage-e33-sdpa.md) | SageAttention-AMD 新 kernel id + H3 dense bridge | `180.267 s` | `-10%` | canary、determinism、operator、DiT oracle、media |
| [B. DiT BF16 GEMM](../../kernel_workspace/02-dit-bf16-gemm/02-dit-bf16-gemm.md) | HIP linear dispatch/独立 benchmark | `148.555 s` | `-10%` | BF16 output、50-block、2-NFE、media |
| [C. DiT weight stream](../../kernel_workspace/03-dit-ssd-weight-stream/03-dit-ssd-weight-stream.md) | loader/prefetch/resident policy | `150.043 s` unhidden | `-20%` | 所有 tensor/media hash exact |
| [D. VAE F32 GEMM](../../kernel_workspace/04-video-vae-f32-gemm/04-video-vae-f32-gemm.md) | M1797 solution registry/benchmark | `147.850 s` | `-10%` | F32/RGB oracle、temporal、media |
| [E. VAE F32 SDPA](../../kernel_workspace/05-video-vae-f32-sdpa/05-video-vae-f32-sdpa.md) | S1797 exact split或新 kernel | `89.713 s` | `-10%` | 3,680,256 outputs bitwise、RGB/media |

每个子任务应交付：

1. 独立分支和最小代码边界；
2. frozen input generator 或 fixture identity；
3. baseline/candidate 同进程 crossed A/B；
4. warm-up、iteration、3-session median/min/max；
5. dtype、shape、layout、stride、scale、mask、workspace、stream semantics；
6. max abs、relative-L2/RMSE、cosine、non-finite、hash、canary、determinism；
7. kernel event 与所属模块 wall，禁止只给 microbenchmark；
8. 显存峰值、GPU BDF、ROCm/compiler/库版本；
9. `KEEP`、`KEEP_EXPERIMENTAL` 或 `REJECT` 结论；
10. 不提交模型、latent、frames、MP4、日志或 profiler 数据库。

## 18. 主分支集成顺序

各子任务完成后不要一次合并全部候选。建议按以下顺序逐项集成并重新建立基线：

1. C：纯 I/O/驻留优化，要求 hash exact；
2. E：Video VAE exact SDPA，要求 F32 bitwise；
3. D：Video VAE GEMM，exact 优先，approximate 独立开关；
4. B：DiT BF16 GEMM，exact 优先；
5. A：DiT Sage kernel，保持显式实验模式直到 frozen-render gate 恢复。

每合入一项，都要以新的完整 E2E wall 作为下一项的分母，并保留本文件的
`760.826 s` 原始基线不覆盖。最终组合必须重新运行同一 prompt/seed/50-NFE/124-frame
三轮压力测试，避免把各自 isolated speedup 直接相加。

## 19. 代码定位与证据等级

子任务开始前应先查看以下代码边界：

| 范围 | 主文件/入口 |
|---|---|
| DiT shape、activation、block 调用 | `h3_dit.c`：`allocate_activations()`、`run_block()` |
| DiT SSD stream | `h3_dit.c`：`prepare_stream_layer()`、`read_stream_layer()`、forward block loop |
| 通用 HIP GEMM | `h3_gpu_hip.cpp`：`h3_gpu_linear()` |
| BF16 SDPA dispatch | `h3_gpu_hip.cpp`：`h3_gpu_sdpa()`、`h3_gpu_sdpa_bf16_d128_sage()` |
| H3 dense Sage bridge | `h3_sage_dense_bridge.cpp` |
| E33 API/registry | `third_party/sageattention-amd/include/sage_attention.hpp`、`registry/kernels/e33-bf16-qk-gfx12-d128.json` |
| E33 device kernel | `third_party/sageattention-amd/src/h3_vdn_sage_gfx12.hip` |
| Video VAE shape/block/tile | `h3_video_vae.c`：constants、`run_block()`、tile/chunk decoder |
| F32/D64 VAE SDPA | `h3_gpu_hip.cpp`：`h3_hip_sdpa_f32_d64_wave32_kernel`、split QK/PV kernels |

本文字段的证据等级：

- wall、event、M/N/K、calls、dtype、bias、tile count：直接来自冻结 `run.log`；
- tensor 语义、layout、stride、accumulate dtype、scale、mask、workspace 和 kernel
  dispatch：由当前 source/submodule pin 静态审计确认；
- sequence 组成、平均时间、比例、MiB、TFLOP/s 和 split workspace：由上述冻结值计算；
- TFLOP/s 是按 dense 主乘加公式得到的等效值，不是硬件 counter；
- S9322/S1797 上未来 candidate 的性能和数值结论仍是未知项，必须由各子任务实测，
  不得从 S9300/S2273 外推。
