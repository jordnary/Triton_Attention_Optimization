# 第三方来源与使用范围

本仓库包含改编后的模型代码与 Attention 实验实现。以下说明用于明确来源，不替代上游许可证，也不为第三方代码重新授予许可。

| 范围 | 来源与说明 |
| --- | --- |
| `PixArt-alpha/` | 基于 [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) 模型代码；本项目加入后端切换、计时及推理流程适配 |
| `attention/sparse_int8.py` | 文件头声明改编自 [SpargeAttn](https://github.com/thu-ml/SpargeAttn) 的 `Triton_SpargeAttn/triton_kernel_example.py` 与 `spas_sage_attn/quant_per_block.py`，并注明 Apache-2.0 |
| 扩散工具与采样器 | 文件保留了 GLIDE、ADM、IDDPM、DiT、Diffusers 或 SA-Solver 等来源注释，应逐文件核对 |
| 模型权重 | 从 [PixArt-α 模型仓库](https://huggingface.co/PixArt-alpha/PixArt-alpha)另行获取，未随源码分发 |

仓库当前没有为自有代码指定统一许可证，第三方来源清单也不是完整许可证审计。部分源文件引用上游 LICENSE，而本地副本未包含相应完整文本；再分发前需依据实际来源版本核对并补齐所需许可证和声明。现有记录未固定所有上游提交版本，因此不推断整个仓库适用单一许可证。

## 方法参考

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [PixArt-α](https://arxiv.org/abs/2310.00426)
- [SpargeAttention](https://arxiv.org/abs/2502.18137)
- [SageAttention](https://arxiv.org/abs/2410.02367)

各后端功能以本仓库实现为准，不应将其视为上述项目的完整复现或官方实现。
