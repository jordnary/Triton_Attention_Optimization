# 实验 2：Attention 机制实现与 PixArt-Alpha 推理评估

日期：2026-06-06

## 1. 实验目标

本实验实现并评估四类 Attention 后端：

- `vanilla_attention`：纯 PyTorch 版本的 scaled dot-product attention。
- `flash_attention_2`：基于 Triton 的 Flash Attention 2 forward kernel。
- `sparse_attention`：基于 block selection 的 Triton block-sparse attention。
- `sparse_int8_attention`：对 Q/K 做 per-block int8 量化的 Triton sparse attention。

实验分为两部分：第一部分将不同 attention 后端接入 PixArt-Alpha Text-to-Image 推理，比较图像质量与采样时间；第二部分使用 `benchmark_attention.py` 和 `test_sparse_int8.py` 系统评估速度与数值误差。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| 终端 / 环境 | PowerShell, Miniconda `na` |
| GPU | CUDA-capable GPU |
| Python | 3.11.15 |
| PyTorch | 2.12.0+cu130 |
| Triton | 3.4.0 |
| CUDA runtime | 13.0 |
| T2I 模型 | PixArt-Alpha `PixArt-XL-2-1024-MS.pth` |
| VAE | `sd-vae-ft-ema` |
| T2I 设置 | image size 1024, DPM-Solver, 20 steps, batch size 1, fp16 |
| Prompt 设置 | 使用 `data/prompt_embeddings/` 中 20 个预提取 T5 embeddings |

原始结果文件：

- `doc/t2i_timing_summary.csv`
- `doc/benchmark_results_full.txt`
- `doc/benchmark_results_full.csv`
- `doc/benchmark_key_summary.csv`
- `doc/sparse_int8_results_full.txt`

## 3. 实现方法

### 3.1 Vanilla Attention

`attention/vanilla.py` 使用显式矩阵乘法实现：

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt{d})V
\]

实现中将 `q`、`k`、`v` 转为 `float32` 做 score 与 softmax，再将输出转回 `v.dtype`。该实现不调用 `F.scaled_dot_product_attention`，保留了 `attn_mask`、`dropout_p`、`training` 接口兼容性。

### 3.2 Triton Flash Attention 2

`attention/fa2.py` 使用 Triton JIT 实现 forward-only Flash Attention 2。核心设计如下：

- 使用二维 grid `(ceil(N / BLOCK_M), B * H)`。
- 每个 program 处理一个 query block。
- 按 K/V block 流式扫描，使用 online softmax 维护局部最大值 `m_i`、归一化因子 `l_i` 和累积输出 `acc`。
- 使用 `exp2`，缩放系数中合入 `1 / ln(2)`。
- `head_dim` 支持非 2 的幂，例如 PixArt 中常见的 `D=72`，通过 `BLOCK_D=next_power_of_2(D)` 和 mask 处理。

### 3.3 Block-Sparse Attention

`attention/sparse.py` 分为两步：

1. Block selection：
   - 将 Q/K 沿序列维划分为 block，默认 `block_size=64`。
   - 对每个 block 做 mean pooling。
   - 计算 block-level score。
   - 对每个 Q block 选取 top-k K blocks，返回 `[B, H, num_q_blocks, topk]` 的 int32 indices。

2. Triton sparse attention kernel：
   - 结构类似 FA2 kernel。
   - 每个 Q block 只访问被选中的 K/V blocks。
   - `topk_ratio=1.0` 时选择全部 block，数值上退化为 dense attention。

### 3.4 Sparse Int8 Attention

`attention/sparse_int8.py` 复用 block selection，并增加 Q/K per-block int8 量化：

- 每个 block 使用 symmetric int8 量化，`scale=max(abs(x))/127`。
- Q 的 scale 合入 `1.44269504/sqrt(d)`，方便 attention kernel 中使用 `exp2`。
- K 默认启用 `smooth_k`，即量化前减去 token 维度上的 per-channel mean；该偏移在 softmax 中会被抵消。
- V 保持 fp16，避免量化 V 带来更明显的输出误差。
- int8 attention kernel 中使用 `tl.dot(q_int8, k_int8)` 得到 int32 accum，再乘 `q_scale*k_scale` 反量化。

## 4. 任务 1：T2I 生成评估

### 4.1 生成效果对比

下图展示 prompt 000 在主要后端上的生成结果。SDPA、Vanilla、Triton FA2、Sparse topk=1.0、Int8 topk=1.0 在主体结构和语义上非常接近；Sparse/Int8 在 topk=0.5 时仍能保持 prompt 语义，但构图和细节会出现更明显变化。

![T2I attention comparison](../pic/t2i_attention_prompt000.jpg)

Sparse attention 的 topk 消融如下。topk 越小，速度越快，但局部细节和构图变化更明显；topk=0.8/0.9 与 topk=1.0 更接近，但速度收益降低。

![Sparse topk prompt 000](../pic/sparse_topk_prompt000.jpg)

![Sparse topk prompt 001](../pic/sparse_topk_prompt001.jpg)

Sparse Int8 在 5 个 topk 设置下均成功生成图像，无 crash、NaN、全黑或全白输出。

![Sparse int8 topk prompt 000](../pic/sparse_int8_topk_prompt000.jpg)

### 4.2 T2I 采样时间

| Attention | topk | 图片数 | 平均采样时间 s/image | 总采样时间 s |
|---|---:|---:|---:|---:|
| SDPA | - | 20 | 20.419 | 408.387 |
| Vanilla | - | 20 | 54.203 | 1084.065 |
| Triton FA2 | - | 20 | 22.001 | 440.023 |
| Sparse | 0.3 | 20 | 15.589 | 311.784 |
| Sparse | 0.5 | 20 | 17.792 | 355.836 |
| Sparse | 0.8 | 20 | 20.481 | 409.626 |
| Sparse | 0.9 | 20 | 22.092 | 441.839 |
| Sparse | 1.0 | 20 | 22.868 | 457.358 |
| Sparse Int8 | 0.3 | 20 | 15.582 | 311.632 |
| Sparse Int8 | 0.5 | 20 | 17.237 | 344.741 |
| Sparse Int8 | 0.8 | 20 | 19.719 | 394.376 |
| Sparse Int8 | 0.9 | 20 | 20.856 | 417.119 |
| Sparse Int8 | 1.0 | 20 | 21.309 | 426.179 |

![T2I backend timing](../pic/t2i_backend_timing_bar.png)

![T2I topk timing curve](../pic/t2i_topk_timing_curve.png)

主要观察：

- Vanilla 是最慢的，平均 `54.203s/image`，约为 SDPA 的 `2.65x` 耗时。
- Triton FA2 平均 `22.001s/image`，约为 SDPA 的 `1.08x` 耗时，满足“不超过 SDPA 2.5 倍”的要求。
- Sparse topk=1.0 平均 `22.868s/image`，约为 SDPA 的 `1.12x` 耗时，数值与生成效果接近 dense attention。
- Sparse topk=0.5 平均 `17.792s/image`，相比 topk=1.0 的 `22.868s/image` 快约 `22.2%`，满足 topk=0.5 至少加速 10% 的要求。
- Sparse Int8 topk=0.5 平均 `17.237s/image`，比 fp16 sparse topk=0.5 略快；在 PixArt 当前序列长度下，int8 的量化开销和 matmul 收益接近抵消。

## 5. 任务 2：Attention Benchmark

Benchmark 使用默认设置：

- Batch size: 2
- dtype: fp16
- `N ∈ {2048, 4096, 8192, 16384}`
- `H ∈ {8, 16}`
- `D ∈ {64, 128}`
- sparse / sparse_int8 topk ratios: `{0.3, 0.5, 0.8, 0.9, 1.0}`
- warmup 10, iterations 50

下表保留验收最关键的后端。单元格格式为：

`time_ms / speedup_vs_SDPA / CosSim / RelL1`

| H | N | D | SDPA ms | FA2 ms/speed/Cos/RelL1 | Sparse1.0 ms/speed/Cos/RelL1 | Sparse0.8 ms/speed/Cos/RelL1 | Int8-1.0 ms/speed/Cos/RelL1 | Int8-0.8 ms/speed/Cos/RelL1 |
|---:|---:|---:|---:|---|---|---|---|---|
| 8 | 2048 | 64 | 1.019 | 0.878/1.16x/1.000/1.22e-07 | 0.908/1.12x/1.000/1.22e-07 | 0.790/1.29x/0.906/4.66e-01 | 0.643/1.59x/1.000/1.22e-02 | 0.629/1.62x/0.906/4.66e-01 |
| 8 | 4096 | 64 | 3.672 | 3.090/1.19x/1.000/1.73e-07 | 3.269/1.12x/1.000/1.73e-07 | 2.608/1.41x/0.904/4.74e-01 | 2.468/1.49x/1.000/1.25e-02 | 2.049/1.79x/0.904/4.74e-01 |
| 8 | 8192 | 64 | 16.430 | 12.612/1.30x/1.000/2.35e-07 | 13.813/1.19x/1.000/2.35e-07 | 11.006/1.49x/0.901/4.82e-01 | 8.714/1.89x/1.000/1.23e-02 | 7.575/2.17x/0.901/4.82e-01 |
| 8 | 16384 | 64 | 65.579 | 52.798/1.24x/1.000/3.24e-07 | 54.093/1.21x/1.000/3.24e-07 | 44.831/1.46x/0.900/4.88e-01 | 36.930/1.78x/1.000/1.23e-02 | 30.968/2.12x/0.899/4.88e-01 |
| 16 | 2048 | 64 | 2.103 | 1.732/1.21x/1.000/1.16e-07 | 1.793/1.17x/1.000/1.16e-07 | 1.462/1.44x/0.905/4.69e-01 | 1.352/1.56x/1.000/1.23e-02 | 1.177/1.79x/0.905/4.69e-01 |
| 16 | 4096 | 64 | 8.527 | 6.582/1.30x/1.000/1.74e-07 | 6.934/1.23x/1.000/1.74e-07 | 5.737/1.49x/0.905/4.71e-01 | 4.879/1.75x/1.000/1.23e-02 | 4.364/1.95x/0.905/4.71e-01 |
| 16 | 8192 | 64 | 33.669 | 26.974/1.25x/1.000/2.27e-07 | 27.566/1.22x/1.000/2.27e-07 | 22.492/1.50x/0.902/4.81e-01 | 19.128/1.76x/1.000/1.23e-02 | 15.857/2.12x/0.901/4.82e-01 |
| 16 | 16384 | 64 | 131.887 | 104.949/1.26x/1.000/3.20e-07 | 107.601/1.23x/1.000/3.20e-07 | 88.602/1.49x/0.901/4.83e-01 | 74.374/1.77x/1.000/1.22e-02 | 61.788/2.13x/0.901/4.83e-01 |
| 8 | 2048 | 128 | 2.401 | 1.908/1.26x/1.000/5.22e-05 | 1.943/1.24x/1.000/5.22e-05 | 2.001/1.20x/0.906/4.69e-01 | 1.849/1.30x/1.000/1.28e-02 | 2.624/0.91x/0.905/4.69e-01 |
| 8 | 4096 | 128 | 8.756 | 8.307/1.05x/1.000/4.25e-05 | 8.622/1.02x/1.000/4.25e-05 | 7.051/1.24x/0.905/4.70e-01 | 7.335/1.19x/1.000/1.29e-02 | 6.022/1.45x/0.905/4.71e-01 |
| 8 | 8192 | 128 | 34.516 | 35.417/0.97x/1.000/3.37e-05 | 36.192/0.95x/1.000/3.37e-05 | 29.142/1.18x/0.902/4.79e-01 | 30.620/1.13x/1.000/1.27e-02 | 25.194/1.37x/0.902/4.79e-01 |
| 8 | 16384 | 128 | 138.473 | 143.640/0.96x/1.000/2.70e-05 | 145.003/0.95x/1.000/2.70e-05 | 117.034/1.18x/0.900/4.86e-01 | 120.660/1.15x/1.000/1.28e-02 | 96.886/1.43x/0.900/4.86e-01 |
| 16 | 2048 | 128 | 4.476 | 4.563/0.98x/1.000/5.21e-05 | 4.578/0.98x/1.000/5.21e-05 | 3.913/1.14x/0.905/4.69e-01 | 4.293/1.04x/1.000/1.28e-02 | 3.785/1.18x/0.905/4.70e-01 |
| 16 | 4096 | 128 | 17.619 | 17.885/0.99x/1.000/4.20e-05 | 18.114/0.97x/1.000/4.20e-05 | 15.364/1.15x/0.906/4.68e-01 | 15.947/1.10x/1.000/1.27e-02 | 13.384/1.32x/0.905/4.69e-01 |
| 16 | 8192 | 128 | 69.587 | 70.526/0.99x/1.000/3.39e-05 | 71.799/0.97x/1.000/3.39e-05 | 58.524/1.19x/0.902/4.80e-01 | 60.330/1.15x/1.000/1.28e-02 | 50.378/1.38x/0.902/4.80e-01 |
| 16 | 16384 | 128 | 277.092 | 279.068/0.99x/1.000/2.70e-05 | 291.577/0.95x/1.000/2.70e-05 | 229.690/1.21x/0.900/4.86e-01 | 245.049/1.13x/1.000/1.28e-02 | 191.269/1.45x/0.900/4.86e-01 |

![Benchmark speedup H16 D64](../pic/benchmark_speedup_h16_d64.png)

### 5.1 达标情况

| 后端 | 关键指标 | 结果 |
|---|---|---|
| Triton FA2 | speedup ≥ 0.4, CosSim > 0.99, RelL1 < 1e-3 | 16 个配置全部满足；最小 speedup `0.964x`，最大 RelL1 `5.22e-05` |
| Sparse topk=1.0 | speedup ≥ 0.4, CosSim > 0.99, RelL1 < 1e-3 | 16 个配置全部满足；最小 speedup `0.950x` |
| Sparse topk=0.8 | CosSim > 0.8, RelL1 < 1.0 | 16 个配置全部满足；最小 CosSim `0.8995`，最大 RelL1 `0.488` |
| Sparse Int8 topk=1.0 | speedup ≥ 0.4, CosSim > 0.99, RelL1 < 2e-2 | 16 个配置全部满足；最小 CosSim `0.999916`，最大 RelL1 `0.01286` |
| Sparse Int8 topk=0.8 | CosSim > 0.8, RelL1 < 1.0 | 16 个配置全部满足；最小 CosSim `0.899439`，最大 RelL1 `0.488` |

注：在少数短序列配置中，`topk=0.8` 相比 `topk=1.0` 的 kernel 计时会受到固定开销和测量抖动影响，不一定严格更快。例如 `H=8,N=2048,D=128` 的 sparse / sparse_int8 结果中，`topk=0.8` 没有明显快于 `topk=1.0`。但在 T2I 主实验与 `N>=4096` 的 benchmark 中，稀疏度降低带来的速度收益是稳定的。

## 6. Sparse Int8 长序列测试

默认配置：`B=2,H=16,D=64,topk=0.8,dtype=fp16,warmup=10,iters=30`。

| N | Sparse fp16 ms | Sparse Int8 ms | Speedup | CosSim | RelL1 | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.631 | 0.608 | 1.04x | 0.904601 | 4.69e-01 | 2.43e-02 |
| 2048 | 1.823 | 1.253 | 1.46x | 0.904990 | 4.69e-01 | 1.73e-02 |
| 4096 | 5.056 | 3.749 | 1.35x | 0.904740 | 4.71e-01 | 1.22e-02 |
| 8192 | 22.100 | 15.509 | 1.42x | 0.901482 | 4.82e-01 | 8.85e-03 |
| 16384 | 85.740 | 60.166 | 1.43x | 0.901004 | 4.83e-01 | 6.33e-03 |
| 32768 | 350.490 | 237.585 | 1.48x | 0.901479 | 4.82e-01 | 4.48e-03 |

![Sparse int8 long sequence speedup](../pic/sparse_int8_long_seq_speedup.png)

实验中 `N=16384` 时 speedup 为 `1.43x`，满足作业要求的 `>=1.20x`。本机上 int8 从 `N=1024` 起已经略快于 fp16 sparse，但 `N=2048` 之后优势更稳定，长序列下 speedup 维持在 `1.35x` 到 `1.48x`。

## 7. 结果分析

### 7.1 为什么 Flash Attention 2 比 Vanilla 更快

Vanilla attention 显式构造完整的 `N x N` score 矩阵，再做 softmax 和矩阵乘法。这会带来较高的显存占用和大量 HBM 读写。Flash Attention 2 使用 tiling 和 online softmax，只在 SRAM 中保留当前 Q/K/V block 和累积状态，避免 materialize 完整 attention matrix。这样既减少访存，也提升 kernel 内计算密度，所以在 benchmark 中 FA2 相比 SDPA 也能达到接近或略快的速度；在 T2I 推理中，FA2 平均 `22.001s/image`，与 SDPA 的 `20.419s/image` 接近。

### 7.2 Block-Sparse 的速度与质量平衡

Sparse attention 的速度收益来自减少参与 attention 的 K/V blocks。topk 越低，kernel 扫描的 K/V block 越少，因此速度越快。但这会丢失部分全局上下文，导致生成图像的构图、局部纹理和主体位置发生变化。

从 T2I 结果看：

- `topk=0.3` 最快，平均 `15.589s/image`，但图像构图变化较明显。
- `topk=0.5` 平均 `17.792s/image`，相比 `topk=1.0` 快约 `22.2%`，仍能保持 prompt 语义，是较好的速度-质量折中。
- `topk=0.8/0.9` 质量更接近 dense attention，但速度收益明显下降。
- `topk=1.0` 用于验证 dense 等价性，数值和生成效果接近 SDPA，但由于 block selection 和 index 访问开销，T2I 中比 SDPA 稍慢。

### 7.3 Int8 短序列慢、长序列快的原因

Int8 sparse attention 的耗时由三部分组成：

- Q/K per-block 量化开销。
- block selection 开销。
- int8 attention matmul 和 V 加权求和开销。

短序列时，attention 计算量较小，量化和 selection 的固定成本占比高，因此 int8 不一定显著快于 fp16 sparse。PixArt 的 T2I 序列长度相对不大，所以 sparse_int8 与 fp16 sparse 的差距不大。

长序列时，attention matmul 成为主导成本，int8 tensor core 吞吐与更低访存带宽需求开始体现优势。`test_sparse_int8.py` 中，`N=16384` 时 int8 达到 `1.43x`，`N=32768` 达到 `1.48x`。因此本机实验中 int8 加速拐点约在 `N=1024-2048`，长序列收益更稳定。

## 8. 总结

本实验完成了四种 attention 后端的实现，并在 PixArt-Alpha 与独立 benchmark 上进行了评估：

- Vanilla 实现正确但速度最慢，适合作为纯 PyTorch 正确性参考。
- Triton FA2 与 SDPA 数值非常接近，速度满足验收要求。
- Sparse attention 在降低 topk 后能明显加速，`topk=0.5` 在本次 T2I 中是较好的速度-质量折中点。
- Sparse Int8 在 PixArt 短序列上收益有限，但在长序列 benchmark 中加速明显，`N>=16384` 满足 1.2x 以上要求。

总体上，Flash Attention 主要优化 dense attention 的访存模式；Sparse attention 通过减少参与计算的 block 提供近似加速；Int8 attention 则更适合长序列、高计算量场景。
