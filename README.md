# Attention 算子实现与 PixArt-α 推理评估

本项目研究注意力计算中的访存开销、块稀疏近似与低精度量化，基于 PyTorch 和 Triton 实现四种 Attention 后端，并在 PixArt-α 文生图推理与合成张量基准中评估其计算效率和数值误差。

仓库提供实现代码、复现说明、测试提示词、实验报告与配套图表。模型权重、文本嵌入和逐次运行输出需在本地准备或生成。

## 方法与实现范围

| 后端 | 实现 | 计算特征 |
| --- | --- | --- |
| `sdpa` | PyTorch `scaled_dot_product_attention` | 参考后端；具体内核由 PyTorch 调度 |
| `vanilla` | PyTorch 显式矩阵运算 | 以 FP32 计算 score、softmax 与加权和，输出转换回输入精度 |
| `triton_fa2` | Triton 分块稠密注意力 | 使用在线 softmax，避免显式存储完整注意力矩阵 |
| `sparse` | 块均值评分、top-k 选择与 Triton 稀疏计算 | 通过 `topk_ratio` 控制每个查询块保留的 K/V 块数 |
| `sparse_int8` | 块稀疏注意力与 Q/K 分块 Int8 量化 | QK 使用 Int8 乘法和 Int32 累加，softmax 与 PV 保持浮点计算 |

张量接口采用 `[B, H, N, D]`。Triton 后端面向 CUDA 前向推理，支持包括 `D=72` 在内的非二次幂头维度，当前上限为 `D=128`；未实现反向传播、因果注意力及自定义 attention mask。默认块大小为 64。

PixArt-α 中仅替换 self-attention，cross-attention 继续使用 PyTorch SDPA。本仓库的 Triton FA2 是基于 FlashAttention 分块与在线 softmax 思路的实验实现，不等同于官方 `flash-attn` 软件包。

## 实验结果概览

下列数值摘自[实验报告](doc/report.md)，并非跨硬件性能保证。T2I 表中每种配置使用 20 个提示词，时间单位为秒/图；计时区间为扩散采样，不含模型加载、文本编码、VAE 解码、模型搬移及文件保存。

| 后端 | `topk_ratio` | 平均采样时间 | 相对 SDPA 加速比 |
| --- | ---: | ---: | ---: |
| SDPA | — | 20.419 | 1.00× |
| Vanilla | — | 54.203 | 0.38× |
| Triton FA2 | — | 22.001 | 0.93× |
| Sparse | 1.0 | 22.868 | 0.89× |
| Sparse | 0.5 | 17.792 | 1.15× |
| Sparse Int8 | 0.5 | 17.237 | 1.18× |

加速比定义为 `T_SDPA / T_backend`，大于 1 表示加速。在单独的长序列实验中，`B=2, H=16, D=64, N=32768, topk_ratio=0.8` 时，Sparse Int8 相对 FP16 Sparse 的加速比为 1.48×；该比较的基准与上表不同。

![各后端的平均采样时间](pic/t2i_backend_timing_bar.png)

降低块保留比例能够缩短采样时间，同时会改变生成图像的构图与细节；Int8 的收益依赖序列长度和头维度。图像质量评价基于固定提示词的视觉比较，未测量 FID、CLIP score 或人类偏好指标。完整数据与局限见[实验报告](doc/report.md)。

## 复现入口

### 1. 配置环境

使用支持 NVIDIA CUDA 的 Linux 环境，或 Windows 上已配置 GPU 支持的 WSL2 Linux 环境。Triton 内核不以原生 Windows 或 CPU 为验证目标。依赖版本和检查步骤见[环境配置](data/env_install.md)。

### 2. 运行算子基准

算子基准不需要模型权重。以下小规模配置用于检查导入、编译和计算：

```bash
python benchmark_attention.py --seq-lens 1024 --num-heads 8 --head-dims 64 --topk-ratios 0.8 1.0 --warmup 3 --iters 10
```

完整基准与长序列比较：

```bash
python benchmark_attention.py --txt output/benchmark_results.txt --csv output/benchmark_results.csv
python test_sparse_int8.py --txt output/sparse_int8_results.txt
```

默认配置、指标定义和计时方法见[算子评估方法](data/task2.md)。

### 3. 运行文生图实验

先按[文生图复现指南](data/task1.md)准备 PixArt 权重、VAE 和 20 组 T5 嵌入，再从仓库根目录运行：

```bash
python test_t2i.py --attention_mode sdpa
python test_t2i.py --attention_mode vanilla
python test_t2i.py --attention_mode triton_fa2
python test_t2i.py --attention_mode sparse --topk_ratio 0.5
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.5
```

图像及 `timing.txt` 写入 `output/<date>_attention_<mode>/`；稀疏后端目录名追加 `_topk<ratio>`。同一天重复运行相同配置会复用目录，独立实验应先保存已有输出。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `attention/` | Attention 后端及共享的块选择逻辑 |
| `PixArt-alpha/` | 模型、采样器及适配后的推理脚本 |
| `benchmark_attention.py` | 多形状、多后端的延迟与数值误差评估 |
| `test_sparse_int8.py` | Sparse Int8 与浮点 Sparse 的长序列比较 |
| `test_t2i.py` | 文生图推理入口 |
| `data/test.txt` | 20 个公开测试提示词 |
| `data/env_install.md` | 环境配置 |
| `data/task1.md`、`data/task2.md` | 文生图与算子评估方法 |
| `doc/report.md`、`pic/` | 实验报告及配套图表 |

## 复现边界

- 报告记录的是既有结果；依赖版本固定不等同于完整、跨平台验证过的环境锁定。
- 报告的 CUDA 记录与依赖安装使用的 CUDA 构建版本口径不同，现有记录不足以确定两者的对应关系，详见环境说明。
- 随机 Q/K/V 的近似误差不直接等价于文生图质量变化。
- T2I 脚本没有独立排除首图的 Triton 编译开销；采样时间不等同于严格稳态或端到端性能。
- `sparse` 在 `topk_ratio=1.0` 时保留全部块；`sparse_int8` 在同一设置下仍包含量化误差。

## 来源与参考

模型代码及改编实现来源见[第三方来源说明](THIRD_PARTY.md)。模型权重应从上游获取，并遵循各自使用条款。

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [PixArt-α](https://arxiv.org/abs/2310.00426)
- [SpargeAttention](https://arxiv.org/abs/2502.18137)
- [SageAttention](https://arxiv.org/abs/2410.02367)
- [Triton 文档](https://triton-lang.org/)
