# Kernel workspace 产物集成评估

状态：`IMPLEMENTATION_COMPLETE`（原始评估保留于第 1–10 节，落地结果见第 11 节）

评估日期：2026-09-18

## 1. 结论

当前 5 个算子工作区中，**没有一项默认性能改动适合在不补门禁的情况下直接进入
production 默认路径**。但已经有三类可复用产物：

1. `05-video-vae-f32-sdpa` 是唯一达到算子级性能门槛的候选；它可在完成
   42-pass VAE 和 124-frame E2E 门禁后，以一行 dispatch 登记的形式进入默认
   `gfx1201 + S1797/H32/D64/F32` 路径。
2. `03-dit-ssd-weight-stream` 中的结构化 stream 计数器可以直接整理后合入；
   `resident=32` 显示出明显收益，但仍缺完整候选跑次，不应现在改默认值。
3. `01`/`02`/`04` 的主要产物是有价值的负结果和可复现 harness；不应合入被拒绝的
   kernel/solution，但应将结论写入优化 ledger，避免重复搜索。

按目前证据，建议集成顺序是：

```text
05 的正式 VAE/E2E 门禁
  -> 05 的 S1797 auto-dispatch 登记
  -> 03 resident=32 的三轮 50-NFE/E2E
  -> 03 结构化统计 API
  -> 01/02/04 负结果归档
```

## 2. 评估边界

- 主仓库分支：`vdn-h3-rocm`
- 主仓库当前 HEAD：`ed49054`
- 5 个算子 worktree 的共同基线：`65a6b35`
- `ed49054` 相对 `65a6b35` 没有修改本次涉及的 `Makefile`、`h3_dit.c`、
  `h3_dit.h`、`h3_gpu_hip.cpp` 和 `tests/bench_f32_sdpa.c`，因此当前没有源码冲突风险。
- 5 个 worktree 都没有新的可 cherry-pick commit；产物是 uncommitted diff 或 untracked
  实验目录。正式集成必须从明确的小 diff 重新组织 commit，不能整目录复制。
- 本次仅审核代码、证据和可集成性，没有占用 GPU，没有修改 production 源码。

## 3. 总体分级

| 序号 | 工作区 | 实验结果 | 可直接取用 | 默认性能路径判定 |
|---:|---|---|---|---|
| 01 | DiT Sage E33 SDPA | 4 个候选全部拒绝 | 仅负结果摘要 | `REJECT` |
| 02 | DiT BF16 GEMM | 7 个候选全部拒绝 | 仅负结果摘要；harness 需通用化 | `REJECT` |
| 03 | DiT SSD weight stream | resident=32 单 NFE 有明显收益，整轮未完成 | stream stats API | `HOLD` |
| 04 | Video VAE F32 GEMM | exact 91216/91218 都不快 | 仅负结果摘要；harness 需通用化 | `REJECT` |
| 05 | Video VAE F32 SDPA | exact split S1797 算子级 `1.714×` | auto-route 测试；dispatch 需补 E2E | `PROMOTE_AFTER_GATES` |

## 4. 可以直接整理后进主仓库的内容

### 4.1 03 的结构化 SSD stream 统计 API

工作区 diff 在 `h3_dit.c`/`h3_dit.h` 中新增：

- block read 数；
- source range 数；
- 8 MiB 分块下的 pread request 数；
- 累计 bytes/read seconds/wait seconds；
- resident block 和 stream slot 数；
- 只读 `h3_dit_get_stream_stats()` 接口。

这些改动不更改权重、数学算子、预取顺序或同步语义；`h3_dit` 通过 `calloc`
创建，新计数器会正确从 0 开始。这部分可作为独立 instrumentation commit 合入。

不应同时照搬工作区的 `Makefile` 改动，因为它引用未跟踪、机器绑定的
`asmevo03/ssd_stream_tool.c`。如果主仓库需要该工具，应先去除硬编码的提交号、PCI 地址、
GPU ordinal、模型绝对路径和 SageAttention 绝对路径。

### 4.2 05 的 auto-route 验证用例

`tests/bench_f32_sdpa.c` 的 diff 为现有 scalar/wave32/split A/B 增加了不设 override 的
`auto` 路径，并校验 `auto` 输出 hash/bitwise 结果。这个测试能直接用于防止
dispatch 边界退化，可作为 test-only commit 合入。

建议在合入时保留以下用例：

- `S1797/H32`：目标 auto route；
- `S509/H3`：未登记 sequence/head fallback；
- `S1797/H31`：错误 head-count fallback；
- `S2273/H32`：已有 split 路径不退化。

### 4.3 负结果摘要

01、02 和 04 的 summary 可以摘要到主仓库优化 ledger，但不应原样复制整个实验
目录。应保留的是：工作负载、平台、候选、正确性结果、性能结果和拒绝原因。

## 5. 最接近合入的性能产物：05 F32 SDPA

### 5.1 已完成的证据

`split-score-s1797` 在冻结的 `S1797/H32/D64/F32`、batch 1、dense non-causal 负载上：

| 项目 | K0/fresh original | split candidate |
|---|---:|---:|
| Controller median | `52.4604187 ms` | `30.5982933 ms` |
| Speedup | `1.0×` | `1.7144883927×` |
| Candidate CV | - | `0.2739%` |
| 1512 calls 投影 | `79.3202 s` | `46.2646 s` |

以交叉 session 计算，预计可节省约 `33.06 s`，相对 760.826 s 基线约为 `4.35%`。
以历史 production event `59.334 ms/call` 估算，上限信号约为节省 `43.45 s`，
约占 E2E `5.71%`。两者都是投影，不是完整 VAE/E2E 实测。

正确性已覆盖：

- 独立 scalar oracle；
- `S/H=1/3, 7/5, 65/7, 509/3, 1797/32`；
- 相对 wave32 的 bitwise exact；
- 20 次 repeat hash；
- output/workspace canary；
- non-finite 为 0；
- native public API route 测试；
- `make h3 -j4` 和 `make gpu-ops-test` 通过。

候选需要 `413,568,768` bytes（`0.385166 GiB`）split workspace。这是可接受的局部
开销信号，但完整进程峰值显存尚未测量。

### 5.2 为什么还不能直接改 auto 默认路径

当前 production diff 仅将：

```c
sequence == 2273
```

扩展为：

```c
sequence == 1797 || sequence == 2273
```

技术上该 diff 可干净应用到当前主分支，但它会直接改变用户未设置环境变量时的
production 路由。以下门禁仍未完成：

- 42-pass Video VAE decoded F32/RGB exact hash；
- 124-frame 完整视频和 MP4 hash；
- Video VAE 真实 wall time；
- 完整进程 peak VRAM；
- 媒体质量、时序、音频和 A/V sync 门禁。

主仓库已经支持 `H3_F32_SDPA_SPLIT_SCORES=1` 强制 split path，因此不需要先改默认
dispatch 就可完成上述验收。这是下一步最低风险的验证方式。

### 5.3 推荐的合入条件

只有当强制 split 的 production 跑次同时满足以下条件，才合入一行 auto registration：

1. final latent、decoded F32/RGB、PCM、raw frames 和 MP4 与 wave32 基线完全一致；
2. 至少 3 个独立 session，Video VAE wall 无退化且 SDPA 收益可复现；
3. 峰值显存保持在 31.9 GiB 设备的安全边界内；
4. `S1797/H31`、`S509/H3` 等未登记 shape 仍使用 wave32 fallback；
5. 只在登记的 `gfx1201` 域内自动生效，其他环境保留原路径。

## 6. 03 SSD resident=32 的集成判定

### 6.1 已有信号

工作区对相同生产 shape 的单 NFE 校准显示：

| 配置 | Block reads | Stream GiB | Unhidden wait | Wall | Peak live bytes |
|---|---:|---:|---:|---:|---:|
| resident=20（K0-1） | 31 | `22.251` | `3.0529 s` | `9.7105 s` | `18.84 GB` |
| resident=20（K0-2） | 31 | `22.251` | `2.8772 s` | `9.6240 s` | `18.84 GB` |
| resident=32 | 19 | `13.638` | `1.8885 s` | `8.6786 s` | `29.08 GB` |

resident=32 与 K0 的 raw F32 video/audio velocity SHA-256 完全一致，说明它是强的
exact-only 候选。主仓库已有 `H3_DIT_RESIDENT_BLOCKS=32` 配置入口，候选本身不需要
新的 production 算子代码。

### 6.2 尚缺的证据

- c001/resident=32 完整 50-NFE controller timing；
- 3 个独立 K0 session 和 3 个独立 candidate session；
- final latent、RGB、PCM、MP4 exact gate；
- 持续负载下的 peak VRAM 和 OOM 余量；
- 冷/热 page-cache 拆分。

因此现在可以使用显式环境变量做下一轮实验，不能将 32 改为 31.9 GiB 设备的
默认值。

## 7. 不应合入的候选

### 7.1 01 DiT Sage E33 SDPA

- K0 median：`69.187576 ms`。
- `c001-wave32-workgroups`：`73.560783 ms`，变慢。
- `c002`/`c003` alpha fast path：与 K0 exact replay 发散。
- `c004-q-nontemporal`：`69.998222 ms`，变慢。
- worktree 没有 production source diff。

判定：保留当前 Sage E33 K0；只归档负结果，不复制被拒绝候选。

### 7.2 02 DiT BF16 GEMM

- K0 四 shape aggregate median：`45.92227385 ms`。
- 7 个 controller 候选的 speedup 均小于 `1.0×`，全部拒绝。
- 局部 FC2 收益在聚焦复现中消失。
- tracked production source diff 为空；只有 untracked `asmevo02/` 实验工具。

判定：继续使用 rocBLAS runtime-selected solution 0/71949。实验工具硬绑定 GPU/PCI/
commit，需通用化后才能作为主仓库 benchmark，不直接复制。

### 7.3 04 Video VAE F32 GEMM

- K0 aggregate median：`96.704139709 ms`。
- exact 91216：`0.998291×`，中性/略慢。
- exact 91218：`0.918996×`，明显变慢。
- 91217/91219 不是 bitwise exact，且未进入严格性能 controller。
- 没有 production source diff。

判定：不登记 M1797 solution override。保留 standard solution 0，下一轮应转向 fusion
或 custom kernel，不重复搜索这两个 exact solution。

## 8. 禁止整目录带入主仓库的产物

以下内容是本地实验证据或构建副产物，不应进入主仓库 commit：

- `*.o`、`*.d`、本地 executable 和 static library；
- `.rocprofv3/`、`__pycache__/` 和 profiler database；
- `run/`、`run-v2/`、`private-run*/`、`evidence*/` 中包含绝对路径和机器身份的原始
  controller 状态；
- `frozen/` 下的已编译 runner/library；
- 硬编码 PCI bus ID、`HIP_VISIBLE_DEVICES` ordinal、用户目录和模型绝对路径的 adapter；
- 模型、latent、原始帧、音频、MP4 和任何 `outputs/` 内容。

如需保存原始证据，应放在仓库外的归档位置；主仓库只保存去机器化的摘要、
复现契约和必要的源码测试。

## 9. 建议的下一轮执行清单

### Phase A：先验证 05，不改默认路由

1. 在当前主分支使用 `H3_F32_SDPA_SPLIT_SCORES=1`。
2. 在同一张空闲 GPU 上跑 wave32/split 交叉的 42-pass VAE 和 124-frame E2E。
3. 记录 Video VAE wall、SDPA event、peak VRAM 及全部 exact hashes。
4. 三轮全部通过后，再合入 `S1797` auto predicate 和 auto-route 测试。

### Phase B：完成 03 resident=32

1. 使用已有 `H3_DIT_RESIDENT_BLOCKS=32` 执行 3 个独立 50-NFE session。
2. 与 resident=20 的 3 个 session 交叉，分离冷/热 page cache。
3. 完成 final latent/RGB/PCM/MP4 exact gate 和 peak VRAM 门禁。
4. 若达标，再设计基于显存容量的 admission policy，不把 32 对所有 GPU 硬编码。

### Phase C：整理可维护产物

1. 独立合入 03 stream stats API。
2. 将 01/02/04 的负结果追加到优化 ledger。
3. 若要保留 harness，将 GPU 选择、PCI、commit 和模型路径改为参数，并移除
   生成二进制与 controller state。

## 10. 本轮审核证据索引

- 01：`../../kernel_workspace/01-dit-sage-e33-sdpa/asmevo-private/SUMMARY.md`
- 02：`../../kernel_workspace/02-dit-bf16-gemm/worktree/asmevo02/SUMMARY.md`
- 03：`../../kernel_workspace/03-dit-ssd-weight-stream/worktree/asmevo03/STARTED_INTERIM_SUMMARY.md`
- 04：`../../kernel_workspace/04-video-vae-f32-gemm/private-run-v2/SUMMARY.md`
- 05：`../../kernel_workspace/05-video-vae-f32-sdpa/evidence-v3/SUMMARY.md`
- 05 native route：
  `../../kernel_workspace/05-video-vae-f32-sdpa/evidence-v3/native-verification.json`

这些路径是当前本地 workspace 的审核来源，不是主仓库 release 产物。

## 11. 实施记录

本节从 2026-09-18 开始实时更新。在所有验收完成前，不把实验路径宣布为
production 默认。

### 11.1 执行状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 工作树与基线确认 | `DONE` | 主分支 `vdn-h3-rocm`，HEAD `ed49054` |
| 单 GPU 选择 | `DONE` | PCI23 主验证；被外部长期占用后切换 PCI43 做配对复验，全程一次只暴露一张卡 |
| 05 standalone/native route 复验 | `DONE` | split 相对 wave32 `1.843×`，四条输出均 bitwise exact；S1797 auto 仍走 wave32 |
| 05 三轮 production split E2E | `DONE` | R2/R3/R4 完整 E2E 均 exact；R1 计算 exact 但不计完整跑次 |
| 05 auto predicate 与 route 测试合入 | `DONE` | S1797/S2273 auto 走 split；S509/H3、S1797/H31 保持 wave32；全 exact |
| 03 resident=20/32 交叉验收 | `DONE` | resident=32 三轮 exact；中位 E2E 663 s，峰值 29138 MiB |
| 03 容量准入策略 | `DONE` | `auto` 已按实时 free VRAM、6 GiB reserve、双 stream slot 和 cap=32 准入；完整 smoke PASS |
| 03 stream stats API 合入 | `DONE` | 仅合入通用计数和只读 API，未带入私有 harness |
| 01/02/04 负结果归档 | `DONE` | 本文保留 02/04；Sage 子仓 ledger 已追加 01 |

### 11.2 GPU 预检

- `HIP_VISIBLE_DEVICES=1 ./h3 --list-devices` 只返回一张卡：
  `gfx1201`, AMD Radeon AI PRO R9700, 31.9 GiB, PCI `0000:43:00.0`。
- AMD SMI 对应 GPU 2，开始时 gfx activity `3%`，已用 VRAM `59 MiB`，
  edge/hotspot/memory 温度为 `30/31/30°C`。
- 后续所有测试都通过 `scripts/profile_vdn_gpu4.sh` 的 BDF、空闲、单卡和外部
  contention guard 执行，不使用其他 GPU。

第一次 production 尝试在 PCI `0000:43:00.0` 执行到 `27/50` NFE 时，guard
发现外部 PID `800855` 新增 2 MiB VRAM 占用，因此只终止本次 h3 PID 并将该跑次
标记为 `INVALID_EXTERNAL_CONTENTION`。未终止外部进程，未保留任何部分性能结论。
对 PCI `0000:63:00.0` 的后续 preflight 同样发现已有外部 2 MiB GPU 上下文，
因此该卡也未进入正式跑次。最终选择 `HIP_VISIBLE_DEVICES=3`，只暴露
PCI `0000:23:00.0`/AMD SMI GPU 1。它通过了完整 guard，且与 05 accepted
controller 证据使用同一 BDF。从该卡的 standalone 复验开始，所有可比数据
都固定在该卡上。

guard 最初以 `H3_CONTENTION_MIN_VRAM_BYTES=1048576`（1 MiB）过滤仅由 HIP
设备枚举产生的几十 KiB 瞬时上下文；阈值和实际值均写入 `run.meta`。Phase B
进一步捕获到在别卡计算的外部 HIP 程序会在目标卡保留稳定的 2 MiB 枚举上下文，
因此根据实测将默认 material-allocation 下限校准为 4 MiB。超过阈值的存活外部
进程仍立即使跑次失效，启动前的 GPU activity/总显存空闲检查保持不变。

### 11.3 不可变实验契约

- prompt：`A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind.`
- seed：`42`
- geometry：`640×384`, 124 frames, 24 fps
- schedule：50 NFE, 50 DiT blocks, reuse 1
- DiT attention：`H3_BF16_SDPA=sage-e33`
- SSD streaming：开启
- Phase A resident blocks：20
- Phase A 唯一变量：`H3_F32_SDPA_SPLIT_SCORES=0/1`
- 冻结 wave32 MP4 SHA-256：
  `8abdf79b11886af448b428f1350f9592c4a592f896b1006f33da85d21ad37bc9`

### 11.4 Phase A standalone 复验

使用当前主分支重新构建 `h3` 和 `h3_f32_sdpa_bench`，然后通过单卡 guard 执行：

```sh
H3_PHYSICAL_GPU=3 scripts/profile_vdn_gpu4.sh \
  /tmp/h3-kernel-integration-20260918/phase-a-preflight-pci23-threshold -- \
  ./h3_f32_sdpa_bench 1797 32 10
```

结果：

| Path | 平均/call | Hash | 相对 scalar |
|---|---:|---|---:|
| scalar | `251.657 ms` | `4b91e3e41f165b51` | `1.000×` |
| wave32 | `55.498 ms` | `4b91e3e41f165b51` | `4.535×` |
| split scores | `30.119 ms` | `4b91e3e41f165b51` | `8.355×` |
| auto | `57.195 ms` | `4b91e3e41f165b51` | `4.400×` |

- split 相对 wave32：`1.843×`；
- scalar/wave32/split/auto mismatch 均为 0；
- max abs/RMSE 均为 0；
- auto 时间与 wave32 同档，证明 S1797 默认路由尚未提前切换；production 验收仍
  使用显式 split 开关，门禁通过后才改 auto dispatch。

### 11.5 Phase C 源码整理

等待空闲 GPU 期间完成了不改变默认数学路径的集成：

1. `h3_dit.c`/`h3_dit.h` 已增加 block reads、source ranges、pread requests、
   bytes、read/wait seconds、resident blocks 和 stream slots 的结构化累计，并提供
   只读 `h3_dit_get_stream_stats()`。
2. 未带入 `asmevo03/`、本地 executable、绝对路径或硬编码 GPU 身份。
3. `tests/bench_f32_sdpa.c` 已增加 unset override 后的 `auto` 路由和
   `auto_mismatches`/hash 校验；此时 S1797 auto 仍应选 wave32。
4. 01 的 4 个被拒绝 E33 候选已追加到 SageAttention-AMD 优化 ledger；
   02/04 的 solution 负结果由本文第 7 节固定。
5. `scripts/profile_vdn_gpu4.sh` 的 contention guard 已增加 PID 存活性检查并
   忽略瞬时 `amd-smi` 自身采样上下文；真实存活的外部 Python/AsmEvo
   进程仍会 fail closed。

静态/主机验证：

- `git diff --check`：PASS；
- `bash -n scripts/profile_vdn_gpu4.sh`：PASS；
- `make -j16 BACKEND=hip HIP_ARCHS=gfx1201 h3 h3_f32_sdpa_bench h3_tests`：PASS，无新 warning；
- `./h3_tests`：PASS，`1774 checks`。

### 11.6 Phase A split production 跑次

#### 跑次 R1：计算有效，mux 无效

固定 PCI `0000:23:00.0`，显式 `H3_F32_SDPA_SPLIT_SCORES=1` 的第一次跑次
完成了 50 NFE、audio VAE 和完整 42-pass/124-frame Video VAE；在开始 FFmpeg
mux 时因本机 FFmpeg 仅位于仓库私有工具目录、未导出 `H3_FFMPEG`/
`LD_LIBRARY_PATH` 而退出。该问题不涉及 GPU 计算，但本跑次不计入三轮完整 E2E。

计算结果：

| 指标 | R1 split | 冻结 wave32 | 变化 |
|---|---:|---:|---:|
| DiT Euler denoise wall | `494.996 s` | `502.222 s` | `-1.44%` |
| SSD unhidden wait | `147.652 s` | `150.043 s` | `-1.59%` |
| Video VAE wall | `202.179 s` | `241.430 s` | `-16.26%` |
| Video VAE SDPA | `48.876 s` | `89.713 s` | `-45.52%` / `1.836×` |
| Video VAE peak | `9.750 GiB` | `9.365 GiB` | `+0.385 GiB` |

语义门禁：124 个 RGB24 PPM 全部生成；逐文件 SHA-256 列表与冻结 wave32
跑次逐字节相同，列表 SHA-256 均为
`6ead4cf5a1c6db45ece95520446b23ca74d64f4a3af35c19d6a45ef5accfe9c3`。
因此 split-score 的 42-pass Video VAE 图像输出已通过 bitwise exact 门禁。

R1 遥测位于
`/tmp/h3-kernel-integration-20260918/phase-a-split-pci23-valid-r1-telemetry`；
`run.meta` 记录 `exit_status=1`，原因是 `cannot start FFmpeg: No such file or
directory`。后续跑次显式使用仓库 `.tools/ffmpeg`，并在启动前重新构建，使新增
stream 计数也进入可执行文件。

#### 跑次 R2：第 1 个完整 E2E PASS

R2 显式加载 `scripts/use_vdn_tools.sh` 后完整退出，`run.meta` 记录：

- BDF：`0000:23:00.0`；
- started/finished：`06:30:04Z` / `06:41:59Z`，端到端 `715 s`；
- exit status：`0`；
- contention/concurrency guard：均为空；
- 遥测峰值：VRAM `20282 MiB`，edge/hotspot/memory `76/98/92°C`，socket
  power `509 W`。

数值/媒体门禁全部通过：

| 产物 | R2 | 冻结 wave32 | 结论 |
|---|---|---|---|
| 124 帧 SHA-256 列表 | `6ead4cf5...fe9c3` | `6ead4cf5...fe9c3` | bitwise exact |
| 解码 F32 PCM SHA-256 | `dd0d377c...5597` | `dd0d377c...5597` | bitwise exact |
| MP4 SHA-256 | `8abdf79b...37bc9` | `8abdf79b...37bc9` | bitwise exact |

R2 profile：

| 指标 | R2 split | 冻结 wave32 | 变化 |
|---|---:|---:|---:|
| 端到端 wall | `715 s` | `760.826 s` | `-6.02%` |
| DiT Euler denoise wall | `493.890 s` | `502.222 s` | `-1.66%` |
| SSD unhidden wait | `145.120 s` | `150.043 s` | `-3.28%` |
| Video VAE wall | `202.322 s` | `241.430 s` | `-16.20%` |
| Video VAE SDPA | `48.756 s` | `89.713 s` | `-45.65%` / `1.840×` |
| Video VAE peak | `9.750 GiB` | `9.365 GiB` | `+0.385 GiB` |

新增 stream 统计为：`1501` 次 block read、`9006` 个 source range、
`144096` 次 pread request、`2` 个 stream slot。R2 是三轮门禁中的第 1 轮；
单轮结果尚不足以修改默认 auto predicate。

#### 跑次 R3：第 2 个完整 E2E PASS

R3 在完全相同的命令和 BDF 上再次完整退出：

| 指标 | R2 | R3 |
|---|---:|---:|
| 端到端 wall | `715 s` | `715 s` |
| DiT Euler denoise wall | `493.890 s` | `493.502 s` |
| SSD unhidden wait | `145.120 s` | `145.254 s` |
| Video VAE wall | `202.322 s` | `202.485 s` |
| Video VAE SDPA | `48.756 s` | `48.700 s` |
| telemetry peak VRAM | `20282 MiB` | `20282 MiB` |
| max edge/hotspot/memory | `76/98/92°C` | `76/97/90°C` |

R3 的 124 帧列表、解码 F32 PCM 和 MP4 SHA-256 分别仍为
`6ead4cf5...fe9c3`、`dd0d377c...5597` 和 `8abdf79b...37bc9`；均与冻结
wave32 和 R2 exact。两份 guard 日志为空，stream 请求数仍为
`1501/9006/144096`。因此 R3 为三轮门禁中的第 2 轮 PASS。

#### 跑次 R4：第 3 个完整 E2E PASS

R4 再次完整退出，端到端 `714 s`；DiT Euler denoise `492.622 s`、SSD
unhidden wait `144.436 s`、Video VAE wall `202.421 s`、Video VAE SDPA
`48.902 s`。stream 请求数第三次保持 `1501/9006/144096`。

R4 的 124 帧列表、解码 F32 PCM 和 MP4 SHA-256 分别仍为
`6ead4cf5...fe9c3`、`dd0d377c...5597` 和 `8abdf79b...37bc9`。遥测峰值
VRAM `20282 MiB`，最大 edge/hotspot/memory `75/98/90°C`，两份 guard
日志为空。至此 R2/R3/R4 三轮 production split E2E 全部通过 exact 门禁。

三轮完整跑次的端到端时间为 `715/715/714 s`，中位数 `715 s`；相对冻结
wave32 `760.826 s` 为 `-6.02%`。Video VAE SDPA 为
`48.756/48.700/48.902 s`，中位数 `48.756 s`，相对 `89.713 s` 为
`1.840×`；Video VAE wall 中位数 `202.421 s`，相对 `241.430 s` 降低
`16.16%`。

三轮门禁通过后，按既定准入规则将 gfx1201、batch 1、heads 32、D64、
non-causal 的 S1797 加入 split-score 默认 predicate；S2273 保持原登记，环境变量
仍可显式禁用或强制该路径。随后执行正/负 shape route 复验。

### 11.7 S1797 auto-route 登记与边界测试

重建后在同一 BDF 上执行四组 scalar/wave32/forced split/auto 对照，全部
`mismatches=0`、hash 一致：

| Shape | wave32 | split | auto | 期望/结果 |
|---|---:|---:|---:|---|
| S1797/H32 | `62.151 ms` | `30.098 ms` | `30.300 ms` | auto split，PASS |
| S2273/H32 | `129.111 ms` | `60.032 ms` | `60.072 ms` | auto split，PASS |
| S509/H3 | `0.242 ms` | `0.363 ms` | `0.306 ms` | 非登记 shape 保持 wave32，PASS |
| S1797/H31 | `47.531 ms` | `29.146 ms` | `47.372 ms` | 非登记 heads 保持 wave32，PASS |

`./h3_gpu_ops_tests` 同卡回归 PASS：
`HIP core operators, rocBLAS linear and LoRA merge parity`。

最初用 `bash -c` 批量执行 shape 时，contention guard 把顶层命令的 benchmark
子进程误判为外部占卡并安全终止。guard 已改为沿 `/proc/PID/status` 的 PPid 链
识别完整 workload 子孙树；同一批量命令随后 `exit_status=0`，两份 guard 日志
为空。外部、非子孙且超过 1 MiB 的进程仍 fail closed。

### 11.8 Phase B resident=20/32 交叉验收

resident=20 直接复用 Phase A 的 R2/R3/R4。resident=32 的第一次尝试在
`29/50` NFE 时检测到非 workload 子孙 PID `4126309` 在目标卡分配 2 MiB，guard
只终止本次 h3，`exit_status=143`。该外部进程随后退出，无法再读取 cmdline；本次
标记为 `INVALID_EXTERNAL_CONTENTION`，不纳入性能或稳定性样本。终止前 DiT load
峰值 `26.341 GiB`，证明配置可进入 31.9 GiB 卡，但不能替代完整显存门禁。

随后两个启动尝试又被同一类 2 MiB context 在 tokenizer 前拦截。第二次已保留到
PID `4149876` 的完整证据：另一用户在执行 `check_mjwarp_gpu.py cuda:2`，实际卡
为 SMI GPU 0（约 28 GiB），而目标 SMI GPU 1 仅有 2 MiB context、约 3% idle
activity 和约 61 MiB 总已用显存。因此将 guard 默认阈值校准到 4 MiB；这三个
被终止跑次仍全部作废，后续用新目录重跑。

#### resident=32 R1d：第 1 个完整 PASS

阈值校准后首个完整跑次在同一 BDF 上 `exit_status=0`，两份 guard 日志为空：

| 指标 | resident=20 中位数 | resident=32 R1d | 变化 |
|---|---:|---:|---:|
| E2E wall | `715 s` | `672 s` | `-6.01%` |
| DiT Euler denoise | `493.502 s` | `451.001 s` | `-8.61%` |
| SSD stream bytes | `1077.378 GiB` | `646.714 GiB` | `-39.97%` |
| SSD unhidden wait | `145.120 s` | `99.330 s` | `-31.55%` |
| block reads | `1501` | `901` | `-39.97%` |
| source ranges | `9006` | `5406` | `-39.97%` |
| pread requests | `144096` | `86496` | `-39.97%` |
| DiT profile peak | `18.474 GiB` | `27.087 GiB` | `+8.613 GiB` |
| telemetry peak VRAM | `20282 MiB` | `29138 MiB` | `+8856 MiB` |

相对最初冻结 wave32 `760.826 s`，R1d 已达到 `-11.68%`，超过近期 10% E2E
目标。Video VAE wall/SDPA 为 `202.222/48.655 s`，符合独立于 resident 配置的
预期。124 帧、解码音频、MP4 三类 hash 均继续与冻结基线 exact；峰值
edge/hotspot/memory 为 `75/97/90°C`。仍需另外两轮完整跑次确认稳定性。

#### resident=32 R2：第 2 个完整 PASS

R2 端到端 `663 s`，DiT Euler denoise `442.540 s`，SSD unhidden wait
`90.238 s`；Video VAE wall/SDPA 为 `202.823/48.984 s`。I/O 请求数与 R1d
完全一致，DiT profile peak `27.087 GiB`、telemetry peak `29138 MiB`。

124 帧、解码音频和 MP4 hash 第 2 次保持 exact；两份 guard 日志为空。
峰值 edge/hotspot/memory 为 `78/101/94°C`。热点温度比 R1d 高 4°C，因此将
温度差作为第 3 轮必须观察的稳定性指标，但本轮没有 OOM、错误或性能退化。

第三次完整尝试的 DiT 阶段正常完成（Euler `445.521 s`、SSD wait
`92.174 s`、请求数 `901/5406/86496`），但 Video VAE 期间 guard 检测到外部
PID `1871691` 新增约 `181 MiB`、随后增至约 `1.24 GiB` 的目标卡分配。该进程为
另一用户的 `train_rocm.py`，因此 guard 正确终止本次 h3，`exit_status=143`；
本跑次标记为 `INVALID_EXTERNAL_CONTENTION`，DiT 数据只用于故障记录，不计第 3
轮 PASS。等待目标卡重新空闲后使用新目录重跑。

原 BDF 随后被另一用户启动的 3000-iteration 训练长期占用。为避免等待约 5 小时，
选择另一张空闲的同型号 gfx1201（PCI `0000:43:00.0`），仍严格一次只暴露一张卡；
先执行 resident=20 配对控制，再执行 resident=32 候选，避免把卡间差异记为收益。

配对结果：

| PCI43 指标 | resident=20 control | resident=32 candidate | 变化 |
|---|---:|---:|---:|
| E2E wall | `721 s` | `661 s` | `-8.32%` |
| DiT Euler denoise | `502.228 s` | `442.180 s` | `-11.96%` |
| SSD unhidden wait | `150.322 s` | `89.260 s` | `-40.62%` |
| Video VAE wall | `201.133 s` | `201.230 s` | `+0.05%` |
| telemetry peak VRAM | `20282 MiB` | `29138 MiB` | `+8856 MiB` |
| max edge/hotspot/memory | `72/105/92°C` | `72/99/92°C` | 无 candidate 热退化 |

control/candidate 的帧列表、PCM 和 MP4 均与冻结基线 exact，两份 guard 日志均为空。
因此 resident=32 获得第 3 个完整 PASS，并在第二张同型号卡上复现。

三轮有效 resident=32 的 E2E 为 `672/663/661 s`，中位数 `663 s`；DiT Euler
为 `451.001/442.540/442.180 s`，中位数 `442.540 s`；SSD wait 为
`99.330/90.238/89.260 s`，中位数 `90.238 s`。相对 resident=20 三轮中位数：

- E2E：`715 -> 663 s`，`-7.27%`；
- DiT Euler：`493.502 -> 442.540 s`，`-10.33%`；
- SSD wait：`145.120 -> 90.238 s`，`-37.82%`；
- 相对最初 wave32 `760.826 s`，组合后的中位 E2E 为 `-12.86%`。

三轮峰值均为 DiT profile `27.087 GiB`、telemetry `29138 MiB`，在 31.9 GiB
设备上保留约 2.7 GiB 的 telemetry 余量。三轮 exact、性能、显存和持续负载门禁
均满足。

#### 容量准入实现与 smoke

已实现显式 `H3_DIT_RESIDENT_BLOCKS=auto`。策略在 DiT 设备初始化后查询实时
free/total VRAM；从 free VRAM 中先扣除 `6 GiB` 运行余量和两个完整 stream slot，
再按每个 resident block 的精确 BF16 字节数计算容量，并将结果限制在已通过生产
验收的 `32`。同时最多只允许 `active_blocks - 1` 个 block 常驻，保证仍会执行真实
SSD streaming 路径。unset、空字符串和 `0` 均保留旧行为，数字配置仍是显式
override。纯容量函数覆盖 `0/8/16/24/32 GiB`、active-block 上限与溢出边界测试。

在 PCI `0000:43:00.0` 上执行了单卡有效 smoke：`64×32`、22 frames、2 NFE、
50 blocks、SSD streaming，并设置 `H3_DIT_RESIDENT_BLOCKS=auto`。运行时报告
`free=31.428 GiB`、`total=31.859 GiB`，自动选择 `32`；DiT、Video VAE、Audio
VAE 和 mux 全部完成，`exit_status=0`，两份 contention guard 均为空。stream
汇总报告 `resident blocks=32`、`stream slots=2`、`37` 次 block read、`222` 个
source range 和 `3552` 个 8 MiB pread。该 smoke 只验证自动选择与完整生命周期；
性能和 exact 结论仍以此前三次完整 640×384/124-frame/50-NFE resident=32 跑次为准。

### 11.9 最终回归与落地结论

最终源码回归全部通过：

- `make -j16 BACKEND=hip HIP_ARCHS=gfx1201 h3 h3_tests h3_f32_sdpa_bench h3_gpu_ops_tests`；
- `make BACKEND=hip HIP_ARCHS=gfx1201 test`，`h3_tests` 为 `1781 checks`，已安装的
  HIP Audio primitive 和 tokenizer 测试通过，未安装 fixture 的目标按 Makefile 契约跳过；
- `h3_gpu_ops_tests` 通过 HIP core、rocBLAS linear 和 LoRA merge parity；
- auto-route 复验：S1797/H32 为 `30.430 ms`、S2273/H32 为 `60.427 ms`，均选择
  split-score；S509/H3 和 S1797/H31 分别为 `0.281 ms`、`47.579 ms`，均保持
  wave32。四个 shape 的 scalar/wave32/split/auto hash 全部一致，mismatch 为 `0`；
- `bash -n scripts/profile_vdn_gpu4.sh`、`git diff --check` 和 SageAttention-AMD
  `make metadata-check` 全部通过。

因此本轮按评估清单完成三项可合入工作：05 的 exact S1797 split-score 默认路由、
03 的 stream stats 与有界 resident auto admission、01/02/04 的负结果归档。未复制
kernel workspace 的二进制、controller state、模型、媒体、绝对路径或硬编码 GPU
adapter。当前 performance 默认仍保持数值安全的 rocBLAS DiT attention；Sage E33
仍为显式实验开关。本结论是这份集成计划完成，不等同于整个仓库的 stable-release
声明。
