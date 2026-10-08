# 运行环境与依赖

本项目使用 PyTorch、Triton 和 CUDA 执行 Attention 前向计算。以下步骤面向 Linux；Windows 用户可在已配置 NVIDIA GPU 支持的 WSL2 Linux 环境中执行。命令均以仓库根目录为工作位置，使用 Bash 语法。

## 版本记录

| 项目 | 记录来源 | 版本或说明 |
| --- | --- | --- |
| Python | 原依赖文件说明 | 3.12 |
| PyTorch / torchvision | `requirements.txt` | 2.8.0 / 0.23.0 |
| Triton | 依赖文件及实验报告 | 3.4.0 |
| PyTorch CUDA 构建 | 原安装说明 | CUDA 12.6 对应的 `cu126` wheel |
| CUDA | 既有实验报告 | 13.0，未注明是驱动支持版本还是 Toolkit 版本 |

驱动报告的 CUDA 支持版本、CUDA Toolkit 版本和 `torch.version.cuda` 含义不同。现有记录不足以证明上述 CUDA 数值来自同一软件层，复现时应分别记录，不应据此认定实验使用了 CUDA 13.0 构建的 PyTorch。

`requirements.txt` 记录实验依赖的版本约束，尚未经完整的跨平台兼容性验证。不同系统、驱动及依赖组合可能需要适配。

## 安装依赖

使用独立 Python 环境。以下 `.venv` 仅为通用相对目录示例，已由 `.gitignore` 排除：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

Triton JIT 需要可用的 C 编译器及 CUDA 驱动库。脚本会尝试查找 CUDA Toolkit 的常见 stub 目录以支持链接；stub 不能替代运行时驱动。编译失败时应检查编译器、驱动与 CUDA 安装，不要将 stub 目录当作运行时驱动库加入 `LD_LIBRARY_PATH`。

## 验证环境

```bash
python -c "import torch, triton; print('PyTorch:', torch.__version__); print('Triton:', triton.__version__); print('CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is required'; x = torch.randn(256, 256, device='cuda'); y = x @ x; torch.cuda.synchronize(); print('CUDA tensor operation: OK')"
python benchmark_attention.py --seq-lens 1024 --num-heads 8 --head-dims 64 --topk-ratios 0.8 1.0 --warmup 3 --iters 10
```

首轮包含 JIT 编译等待；算子脚本在正式计时前预热。结果写入 `output/`，默认仅在本地保存。

## 资源需求

算子基准的内存需求随批量、序列长度、头数与头维度变化，首次运行宜采用小规模配置。

文生图推理在采样和解码之间切换 DiT 与 VAE 的驻留位置，以降低同时占用的显存。峰值显存仍依赖模型、分辨率、后端和文本编码方式，现有记录不足以给出通用最低显存保证。

T5-XXL 嵌入生成使用 FP32，脚本会在 CUDA 可用时选择 GPU。若显存不足，可在独立进程中禁用可见 GPU 后使用 CPU；这会增加内存需求和执行时间。具体命令见[文生图复现指南](task1.md)。
