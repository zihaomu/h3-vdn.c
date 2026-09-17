# Original MiniMax-H3 HIP capability matrix

> 状态：`P3_REFERENCE_OPS_COMPLETE_MODEL_PARITY_BLOCKED`
>
> 审计日期：2026-09-17
>
> 审计命令：`make BACKEND=hip h3-hip-capability-report`
> 审计工具：`scripts/audit_h3_hip_capabilities.py`

## 1. 摘要

| 项目 | 数量 |
|---|---:|
| `h3_gpu.h` 声明的 GPU API | 120 |
| 原始 H3 源码直接引用的 GPU API | 90 |
| 阻塞 BF16 reference 的 HIP stub | 0 |
| 不阻塞 reference 的可选加速 stub | 10 |

结论：当前源码静态审计中，原始 H3 的 Text/Vision Encoder、Video VAE encoder/decoder、
Audio VAE 和 BF16 DiT reference 路径均不再引用缺失或 unsupported HIP API。完整
FL2VA/Ref2VA 的模型级正确性仍未证明：本地缺少固定 revision 的原始模型权重和 19 个
上游 oracle fixture，静态能力就绪不能替代真实模型 parity。

本表由源码静态审计得到。每次修改 `h3_gpu.h`、`h3_gpu_hip.cpp` 或原始 H3 组件调用后，
都必须重新执行审计，不能手工假定能力状态。

## 2. BF16 reference 阻塞项

无。`make BACKEND=hip h3-hip-reference-ready` 必须保持通过；后续新增原始 H3 GPU API
调用时，审计工具会把新出现的 missing/unsupported reference API 重新列为硬失败。

## 3. 已完成的 BF16 reference 算子

| API | 验证结果 |
|---|---|
| `h3_gpu_embedding_bf16` | 小尺寸 CPU oracle 逐位一致，包含越界 token→0 |
| `h3_gpu_head_rms_norm_bf16` | 小尺寸 CPU oracle max-abs `0` |
| `h3_gpu_rope_text_bf16` | Q/K 小尺寸 CPU oracle 逐位一致 |
| `h3_gpu_gqa_causal_bf16` | GQA、causal mask、BF16-scaled-Q CPU oracle max-abs `0` |
| `h3_gpu_vision_qkv_rope_bf16` | Q/K/V 拆分和二维 RoPE CPU oracle 逐位一致 |
| `h3_gpu_audio_qkv_split_f32` | Q/K/V split CPU oracle max-abs `0` |
| `h3_gpu_audio_attention_pool_f32` | causal attention pooling CPU oracle max-abs `0` |
| `h3_gpu_geglu_f32` | GeGLU CPU oracle max-abs `0` |
| `h3_gpu_vae_encoder_pad_f32` | temporal zero + spatial reflection padding CPU oracle max-abs `0` |
| `h3_gpu_conv3d_f32` | NDHWC/OIDHW、stride 和 bias CPU oracle max-abs `0` |
| `h3_gpu_vae_encoder_group_norm_silu_f32` | GroupNorm + SiLU CPU oracle max-abs `1.1920929e-07` |

完整 GPU 证据：`outputs/h3-reference-ops-p3-complete-20260917/`，物理 GPU 4，BDF
`0000:e3:00.0`。新增 `make BACKEND=hip H3_PHYSICAL_GPU=4
h3-reference-ops-test` 作为快速门禁。现有 1774 项 host checks、backend BDF probe 和 HIP
DiT operator suite 同轮回归通过。

## 4. 可选加速缺口

以下接口只属于当前源码中的 INT8/NAX、head-major fast path 或 token reduction，不阻塞
BF16 reference bring-up：

| API | 类别 |
|---|---|
| `h3_gpu_gate_adaln_quantize_int8` | INT8 fusion |
| `h3_gpu_grouped_qkv_linear_rope_int8` | INT8 QKV |
| `h3_gpu_linear_int8_head_major_bf16` | INT8 projection |
| `h3_gpu_mlp_int8_bf16` | INT8 MLP |
| `h3_gpu_mlp_nax_bf16` | NAX MLP |
| `h3_gpu_sdpa_bf16_head_major_output` | head-major fast path |
| `h3_gpu_token_pool_bf16` | token reduction |
| `h3_gpu_token_pool_adaln_bf16` | token reduction fusion |
| `h3_gpu_token_expand_delta_bf16` | token reduction |
| `h3_gpu_token_expand_adaln_bf16` | token reduction fusion |

这些接口在 P7 reference baseline 通过前不进入实施优先级。

## 5. Reference bring-up 顺序

1. Text Encoder：真实 FL2VA prompt parity；
2. BF16 DiT：tiny block、单 NFE 和完整轨迹 oracle；
3. Video VAE decoder：纯文本生成的最终 video latent/RGB parity；
4. Audio VAE：audio latent/PCM parity；
5. Vision Encoder：图片条件 parity；
6. Video Encoder：first/last frame 和 Ref2VA video 条件 parity；
7. 可选加速接口：仅在完整 BF16 correctness baseline 冻结后处理。

上述顺序现在全部依赖固定 revision 的模型和上游 fixture，不再依赖 HIP API 补洞。

## 6. 门禁命令

```sh
# 只生成报告，发现缺口也返回成功。
make BACKEND=hip h3-hip-capability-report

# 静态 reference API 门禁；当前应通过。
make BACKEND=hip h3-hip-reference-ready

# GPU 4 上的 11 个 reference operator CPU-oracle 门禁。
make BACKEND=hip H3_PHYSICAL_GPU=4 h3-reference-ops-test
```

`h3-hip-reference-ready` 已达到 PASS 所需的静态条件。算子小尺寸 CPU oracle 通过后仍
必须继续通过固定原始权重的上游数值 oracle；静态审计和单算子 PASS 本身不等于模型准确。
