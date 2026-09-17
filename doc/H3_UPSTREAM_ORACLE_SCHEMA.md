# Original MiniMax-H3 upstream oracle schema

> 状态：`SCHEMA_V1_IMPLEMENTED_TEXT_VISION_AUDIO_VIDEO_FIXTURES_PASS`
>
> Schema：`h3-upstream-oracle-v1`
>
> 官方模型：`MiniMaxAI/MiniMax-H3`
>
> 固定 revision：`42ed227ee7df40d41602854ae760620d6eb651fe`
> 唯一 AMD 导出/验证设备：物理 GPU 4 / `0000:e3:00.0`

## 1. 目的

该 schema 为原始 MiniMax-H3（FL2VA/Ref2VA）提供可重复、可审计的上游数值 oracle。
fixture 中的显式 tensor bytes 是权威输入和输出；C/HIP 端不得根据 seed 重新生成随机数，
也不得把当前 VDN fixture、Metal 输出或另一个 C 实现当成最高层级 oracle。

上游权威顺序为：固定 revision 的官方实现和 checkpoint → 同权重的 PyTorch eager 导出
→ 已验证 MLX fixture → Metal reference → HIP reference。交叉实现一致只能作为附加证据。

## 2. 文件与 manifest

每个 bundle 由两个同名文件组成：

```text
<case>.<scope>.safetensors
<case>.<scope>.safetensors.json
```

Safetensors 使用小端、连续、无 gap/overlap 的 payload。tensor 按名称排序后写入，header
JSON 使用稳定排序和紧凑编码并 padding 到 8 字节。大模型权重不复制进 fixture。

sidecar manifest 的顶层 contract：

```json
{
  "schema": "h3-upstream-oracle-v1",
  "scope": "input-conditioning",
  "fixture": {
    "path": "h3_fl2va_text_tiny.input-conditioning.safetensors",
    "sha256": "<whole-file-sha256>"
  },
  "provenance": {
    "upstream_repo": "MiniMaxAI/MiniMax-H3",
    "upstream_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "model_repo": "MiniMaxAI/MiniMax-H3",
    "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "task": "FL2VA",
    "case_id": "fl2va-text-tiny-v1",
    "seed": "42",
    "rng_policy": "fixture tensor bytes are authoritative",
    "exporter_sha256": "<export-script-sha256>"
  },
  "execution": {
    "python": "<version>",
    "torch": "<version>",
    "torch_hip": "<version-or-none>",
    "device": "cpu-or-hip:0",
    "physical_gpu": "4",
    "pci_bdf": "0000:e3:00.0"
  },
  "tensors": {
    "input.token_ids": {
      "dtype": "I32",
      "shape": [6],
      "sha256": "<raw-payload-sha256>"
    }
  }
}
```

`scripts/validate_h3_oracle_fixture.py` 对整个文件 hash、每个 tensor 的 dtype/shape/raw hash、
payload 完整覆盖和必需 provenance 字段执行硬校验。缺 manifest、字段或 tensor 均为失败。

## 3. Scope 与 tensor 命名

### 3.1 `input-conditioning`

所有离散数据要求逐字节一致：

| Tensor 前缀 | 内容 |
|---|---|
| `input.*` | prompt UTF-8 bytes、token IDs、attention mask、显式 video/audio noise |
| `text.*` | 最终 text embedding，以及按需保存的 layer 0/25/50 输出 |
| `media.*` | resize/crop/normalize 后像素、抽帧时间戳、解码 PCM、采样率与声道布局 |
| `vision.*` | Vision Encoder patch/merged embedding |
| `condition.*` | first/last frame、Ref2VA image/video/audio latent 与 mask |
| `layout.*` | packed row map、modality tag、position ID、condition row 和 presentation 顺序 |
| `rope.*` | C/HIP 实际消费的 cosine/sine 表 |
| `schedule.*` | sigma、timestep、modality timestep row map 和 Euler scale |

### 3.2 `dit-block0`

必须包含 block 输入和第一处分叉所需观察点：

- `dit.block_00.input`、`adaln`、`q`、`k`、`v`、`q_norm`、`k_norm`、`rope_q`、
  `rope_k`；
- `attention.output`、`attention.residual`；
- `mlp.fc1`、`mlp.activation`、`mlp.fc2`、`mlp.residual`；
- `dit.block_00.output`。

### 3.3 `dit-forward`

保存 `dit.block_00.output` 到 `dit.block_49.output`，并至少单独列出计划要求的
0、1、2、9、19、29、39、49 层。另存 final norm、video/audio output head 和单 NFE
velocity。所有观察点必须关闭 production 时零开销。

### 3.4 `denoise-trajectory`

对每个 NFE 保存：

- `nfe_N.video_velocity`、`nfe_N.audio_velocity`；
- `nfe_N.video_after_euler`、`nfe_N.audio_after_euler`；
- 对应 sigma/timestep 和输入 hash。

最终 latent 不得只保存 video/audio 拼接值；两个 modality 必须独立报告误差。

### 3.5 `vae-media`

| Tensor | 约束 |
|---|---|
| `video.normalized_latent` | decoder 实际输入，F32 |
| `video.rgb_f32` | 编码前线性 RGB，`[frames,height,width,3]` |
| `audio.normalized_latent` | decoder 实际输入，F32 |
| `audio.pcm_f32` | 显式声道布局和采样率 |
| `video.encoder_latent` | first/last/ref video encoder 输出 |
| `audio.encoder_latent` | reference audio encoder 输出 |

H.264/AAC/MP4 只进入独立 `media.json` 容器门禁，不能替代 RGB/PCM 数值 oracle。

## 4. 固定用例

命名必须包含任务、条件、规模和版本：

| Case ID | 用途 |
|---|---|
| `fl2va-text-tiny-v1` | 最小 prompt/block/NFE 定位 |
| `fl2va-text-dev-v1` | 256×256/22 帧提交门禁 |
| `fl2va-first-dev-v1` | first-frame 条件 |
| `fl2va-last-dev-v1` | last-frame 条件 |
| `fl2va-first-last-dev-v1` | 双 anchor 条件 |
| `ref2va-image-dev-v1` | 单图片参考 |
| `ref2va-video-dev-v1` | silent video 参考 |
| `ref2va-video-audio-dev-v1` | video + audio 参考 |
| `ref2va-ordered-image-audio-dev-v1` | 多引用原始顺序 |
| `ref2va-order-swap-dev-v1` | 引用交换敏感性 |

release fixture 另加至少一个 768p-class 画布和一个 56 帧 A/V 用例；tiny/dev fixture
不能被重新标记成 release 证据。

## 5. dtype 与比较规则

- token、mask、layout index、position ID、tag、shape、schedule 和媒体元数据 exact；
- BF16/F32 tensor 同时报 max-abs、relative RMSE、relative L2、cosine、non-finite；
- 阈值由两次相同上游导出和两个正确上游 backend 的自然差异标定，必须先于 HIP 候选冻结；
- RGB 增加每帧 PSNR/SSIM 与 temporal delta；PCM 增加 correlation、SI-SDR、peak 和 silence；
- 不允许只比较最终 MP4 hash，也不允许用“能播放”作为准确性证据。

## 6. 当前资产锁与磁盘约束

固定官方 snapshot 的逻辑大小：

| Scope | 大小 |
|---|---:|
| FL2VA 完整 | 134.158 GiB |
| Ref2VA 完整 | 134.158 GiB |
| 双任务 content-addressed 去重后 | 195.865 GiB |
| 单任务 text encoder | 62.144 GiB |
| 单任务 transformer | 61.729 GiB |
| 单任务 video VAE | 9.700 GiB |
| 单任务 audio VAE | 0.564 GiB |

2026-09-17 本机目标文件系统仅约 121 GiB 可用。因此，即使只下载完整 FL2VA，也无法
保留 16 GiB 安全余量；双任务至少需要约 212 GiB 可用空间。不得在当前容量下盲目启动
完整 snapshot 下载。用以下命令重新计算在线 pinned tree 和本地容量：

```sh
python3 scripts/inspect_h3_reference_snapshot.py --scope fl2va --require-fits
python3 scripts/inspect_h3_reference_snapshot.py --scope both --require-fits
```

在扩容前可以先安装非权重 source/config，并分阶段下载 VAE 做 P5；但它不能解除 P2/P4
的 text encoder/transformer 真实权重阻塞，也不能把 P7 标为完成。

## 7. 验证命令

```sh
python3 scripts/validate_h3_oracle_fixture.py \
  misc/fixtures/h3_upstream/fl2va-text-tiny-v1.input-conditioning.safetensors
```

正式 GPU fixture 导出统一经 `scripts/profile_vdn_gpu4.sh` 的同等 GPU 4/BDF guard 运行，
并把 telemetry/command/stdout/stderr 与 manifest 放在同一个证据目录。后续将该 guard
重命名为通用 H3 helper；重命名前不改变它已验证的设备隔离语义。

## 8. 已实现 fixture

`scripts/export_h3_text_oracle.py` 已实现官方 FL2VA Qwen3-VL 文本路径导出。它使用
Transformers 4.57 eager 语义执行发布 checkpoint 的前 50 层，并通过逐层迁移满足单张
31.9 GiB GPU 的显存约束：

```text
fixture: misc/fixtures/h3_upstream_fl2va_text_v1.safetensors
fixture SHA-256: 9ead2d367959091cc467f771f5ca957b8b883d5d16467a4469ed25e241bc1f83
model index SHA-256: 06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a
```

GPU 4 上官方输出与 C/HIP 前 50 层输出：token IDs 逐字节一致，absolute-max `2`、
relative-max `1.31857e-4`、relative-L2 `5.04021e-4`，通过冻结的 real-prompt 阈值。
fixture 保存 canonical prompt bytes/token IDs 及第 1/25/50 层 BF16 输出；`x.ids/x.output`
只作为现有测试迁移期 alias。

`scripts/export_h3_vision_oracle.py` 已实现固定 64×64 图像的官方 FL2VA Qwen3-VL vision
tower 导出：

```text
fixture: misc/fixtures/h3_upstream_fl2va_vision_64_v1.safetensors
fixture SHA-256: e1884187d5d51581bef025fe0a0f1f67978558b03dfc320f9a3b4cab537e98dd
```

GPU 4 上 C/HIP 的 merged output relative-L2 为 `4.999506e-2`，三组 deepstack 分别为
`1.013053e-2`、`1.920320e-2`、`3.669904e-2`，全部无 non-finite 并通过既有 vision
门禁。`scripts/export_h3_multimodal_oracle.py` 已按官方 Diffusers conditioner 的显式
`attention_mask=torch.ones_like(input_ids)` 语义导出 FL2VA `<Picture 1>` 的 token、
mRoPE、tags、vision/deepstack、真实 layer-1 Q/K/V/attention 和前 50 层快照：

```text
fixture: misc/fixtures/h3_upstream_fl2va_multimodal_64_v2.safetensors
fixture SHA-256: 795403794fae73a2a9b6fafdeeefe13c86e2f001fba1dc1b8925c9c75c848307
model revision: 42ed227ee7df40d41602854ae760620d6eb651fe
Diffusers source revision: ae2e4c7907ccc21e52ebe86b349d67e9b0b9f316
```

GPU 4 上真实 GQA attention 与 upstream eager 逐位一致；C/HIP 完整 layer-50 的
max-abs 为 `8`、relative-max 为 `5e-4`、relative-L2 为 `0.001003644`，non-finite 为
`0`。旧 v1 exporter 没有传入 2-D attention mask，Transformers 会把 vision mRoPE 中
重复的 temporal position ID 误识别成 packed-document 边界；旧 v1 fixture 和其
`0.3006364` 失败仅保留为根因分析记录，不是有效产品 oracle。

`scripts/export_h3_vae_oracle.py` 已实现官方 FL2VA Audio VAE 的 encoder/decoder 导出：

```text
fixture: misc/fixtures/h3_upstream_fl2va_audio_vae_v1.safetensors
fixture SHA-256: eadbad9b0b238fc55e75c93e13da7437e2fea4d93d6292b09cbb4c57947be599
weight SHA-256: 37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2
```

GPU 4 上官方 PyTorch F32 与 C/HIP 的结果：decoder max-abs `1.144409e-4`、relative L2
`2.028145e-5`；encoder max-abs `1.108646e-5`、relative L2 `2.846299e-6`。两者均通过
预先存在的 Audio VAE 阈值。该 fixture 同时携带 canonical tensor 名和两个 legacy alias，
使现有 focused tests 可以迁移而不复制大 payload。

`scripts/export_h3_video_vae_oracle.py` 已实现官方 FL2VA Video VAE 的最小
encoder/decoder 导出。两 token decoder 输入严格复现原生 C 的首块语义：末 token 重复
补齐至 `T=7`，调用官方低层 decoder，再选择输出帧索引 `3..7`。

```text
fixture: misc/fixtures/h3_upstream_fl2va_video_vae_v1.safetensors
fixture SHA-256: 9fa7b7f863b90be31a4818fc8448cb34d8048be7f6d8370a09e66136d676fa6c
weight SHA-256: 5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3
```

GPU 4 上官方 PyTorch F32 与 C/HIP 的结果：decoder max-abs `1.93715e-6`、relative L2
`5.40566e-7`；encoder max-abs `1.239777e-5`、relative L2 `8.839093e-7`。两者均通过既有
数值阈值。该对齐同时发现并修复了两个原始 H3 HIP correctness 问题：Video VAE latent
归一化配置应优先从官方 `source/../config.json` 加载，以及 F32 SwiGLU 必须按官方
`[gate, value]` 顺序计算 `SiLU(gate) * value`。
