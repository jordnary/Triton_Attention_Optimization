# 文生图复现指南

本实验在 PixArt-α 中切换 self-attention 后端，比较固定提示词与随机种子条件下的采样时间和生成结果。cross-attention 保持 PyTorch SDPA。运行环境见[环境配置](env_install.md)。

## 1. 准备模型与提示词

提示词位于 `data/test.txt`，共 20 行。模型文件从 [PixArt-α 官方模型仓库](https://huggingface.co/PixArt-alpha/PixArt-alpha)获取，目标布局如下：

```text
pretrained_models/
├── PixArt-XL-2-1024-MS.pth
├── sd-vae-ft-ema/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── t5-v1_1-xxl/
    ├── config.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── spiece.model
    ├── pytorch_model.bin.index.json
    └── pytorch_model-*.bin
```

安装依赖后，可使用 Hugging Face CLI 按相对目录下载：

```bash
hf download PixArt-alpha/PixArt-alpha PixArt-XL-2-1024-MS.pth --local-dir pretrained_models
hf download PixArt-alpha/PixArt-alpha --include "sd-vae-ft-ema/*" "t5-v1_1-xxl/*" --local-dir pretrained_models
```

命令依赖上游文件结构与访问权限，下载后应核对上述布局。当前 VAE 加载调用指定 `use_safetensors=False`，因此需保留对应的 `.bin` 权重。

`PixArt-alpha/tools/download.py` 使用另一套输出目录约定，与推理脚本默认路径不同；本指南直接准备到 `pretrained_models/`，避免目录歧义。

## 2. 生成 T5 嵌入

```bash
python PixArt-alpha/scripts/save_t5_embeddings.py --t5_path pretrained_models --txt_file data/test.txt --output_dir data/prompt_embeddings
```

必须显式传入 `--txt_file` 和 `--output_dir`：该脚本默认值与本项目的提示词和推理目录不一致。输出应包含 `prompt_000.pt` 至 `prompt_019.pt`，每项保存提示词、`caption_emb` 与 `emb_mask`。

该脚本默认以 FP32 加载 T5-XXL，并在 CUDA 可用时使用 GPU。显存不足且主存足够时，可在 Bash 中使用 CPU 生成：

```bash
CUDA_VISIBLE_DEVICES="" python PixArt-alpha/scripts/save_t5_embeddings.py --t5_path pretrained_models --txt_file data/test.txt --output_dir data/prompt_embeddings
```

嵌入可供所有后端重复使用。修改提示词后应重新生成。推理脚本只检查是否存在 `prompt_*.pt`，不会验证数量是否为 20 或内容是否匹配。

## 3. 运行后端比较

从仓库根目录执行：

```bash
python test_t2i.py --attention_mode sdpa
python test_t2i.py --attention_mode vanilla
python test_t2i.py --attention_mode triton_fa2
python test_t2i.py --attention_mode sparse --topk_ratio 1.0
python test_t2i.py --attention_mode sparse --topk_ratio 0.5
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 1.0
python test_t2i.py --attention_mode sparse_int8 --topk_ratio 0.5
```

默认配置为图像基准尺寸 1024、批量 1、`dpm-solver`、20 步、CFG scale 4.5、随机种子 0、模型权重 FP16。默认优先读取预计算嵌入；缺少嵌入或指定 `--online_t5` 时加载 T5 在线编码。

顶层 `test_t2i.py` 仅接受 `--attention_mode`、`--topk_ratio` 和 `--online_t5`。控制采样参数需直接调用底层脚本：

```bash
python PixArt-alpha/scripts/inference.py --attention_mode triton_fa2 --image_size 1024 --bs 1 --sampling_algo dpm-solver --step 20 --cfg_scale 4.5 --seed 0
```

## 4. 稀疏比例消融

分别对两个稀疏后端评估 `topk_ratio ∈ {0.3, 0.5, 0.8, 0.9, 1.0}`。以下循环适用于 Bash：

```bash
for mode in sparse sparse_int8; do
  for ratio in 0.3 0.5 0.8 0.9 1.0; do
    python test_t2i.py --attention_mode "$mode" --topk_ratio "$ratio"
  done
done
```

`topk_ratio` 表示保留的 K/V 块比例，不是被删除的比例。块数向上取整，至少保留一个块。`sparse` 在比例为 1 时执行全块计算；`sparse_int8` 此时仍执行 Q/K 量化。

## 5. 输出与评价

输出目录为 `output/<date>_attention_<mode>/`，稀疏配置追加 `_topk<ratio>`，包括图像和记录逐图采样时间的 `timing.txt`。同日同配置会复用目录并覆盖同名结果，重复实验前应另行归档。

计时从采样分支开始，到 CUDA 同步完成为止，包括该分支中的噪声初始化、采样器构建及扩散迭代；不含模型加载、文本编码、VAE 解码、CPU/GPU 模型搬移和图像保存。脚本没有独立预热并剔除首图，首次 Triton 编译可能进入首图计时。

比较时应固定提示词及其顺序、嵌入、随机种子、采样器、步数、分辨率和 CFG 参数。不同后端的浮点运算顺序不同，不保证图像逐像素相同。视觉评价应观察主体语义、构图、纹理与局部伪影，并与计时结果分别报告。

既有图像和数据见[实验报告](../doc/report.md)。
