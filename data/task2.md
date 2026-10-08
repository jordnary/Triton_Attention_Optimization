# Attention 算子评估方法

实验使用合成 Q/K/V 张量测量前向延迟，以 PyTorch SDPA 输出作为数值参考，不需要模型权重。运行环境见[环境配置](env_install.md)。

## 1. 多后端基准

`benchmark_attention.py` 评估 SDPA、Triton FA2、Sparse 和 Sparse Int8。虽然脚本导入了 Vanilla，其注册项当前被注释，默认结果不包含该后端。

| 参数 | 默认值 |
| --- | --- |
| `--batch` | 2 |
| `--seq-lens` | 2048、4096、8192、16384 |
| `--num-heads` | 8、16 |
| `--head-dims` | 64、128 |
| `--topk-ratios` | 0.3、0.5、0.8、0.9、1.0 |
| `--dtype` | `fp16` |
| `--warmup` / `--iters` | 10 / 50 |

每个形状重新设置随机种子为 0，从标准正态分布生成 Q/K/V。默认遍历 16 种形状、每种 12 个后端配置。

```bash
python benchmark_attention.py --txt output/benchmark_results.txt --csv output/benchmark_results.csv
```

检查非二次幂头维度时，可单独运行：

```bash
python benchmark_attention.py --seq-lens 1024 --num-heads 8 --head-dims 72 --topk-ratios 0.8 1.0 --warmup 3 --iters 10
```

每个后端预热后，用 CUDA events 记录连续前向调用总时间并除以迭代次数。Sparse 的块选择、Sparse Int8 的块选择与量化均包含在被计时调用中；所得数据不是单个 Triton 内核的独立耗时。计时不含预热与首次 JIT 编译。

默认文本结果写入 `output/benchmark_results.txt`，CSV 需显式指定。脚本只自动创建 `output/`；其他输出父目录须事先存在。`FAILED` 配置不能作为有效性能记录。

## 2. 数值指标

将 SDPA 参考输出记为 `R`、被测输出记为 `O`，展平并转换为 FP32 后计算：

| 指标 | 定义 | 解释 |
| --- | --- | --- |
| CosSim | `sum(R * O) / (sqrt(sum(R²)) * sqrt(sum(O²)) + ε)` | 越接近 1，方向越一致 |
| RelL1 | `sum(abs(O - R)) / (sum(abs(R)) + ε)` | 相对绝对误差，越小越好 |
| RMSE | `sqrt(mean((O - R)²))` | 均方根误差，越小越好 |
| Speedup | `T_SDPA / T_backend` | 大于 1 表示比 SDPA 更快 |

脚本采用 `ε=1e-12`。SDPA 实际内核由 PyTorch 选择，代码未固定某一种实现；比较结果与框架版本、GPU 及形状有关。

Sparse 的块裁剪误差与 Sparse Int8 额外引入的量化误差应分别解释，不能仅凭速度评价优劣。

## 3. Sparse Int8 长序列比较

`test_sparse_int8.py` 在相同块保留比例下比较浮点 Sparse 与 Sparse Int8。

| 参数 | 默认值 |
| --- | --- |
| `--batch` / `--num-heads` / `--head-dim` | 2 / 16 / 64 |
| `--seq-lens` | 1024、2048、4096、8192、16384、32768 |
| `--topk-ratio` | 0.8 |
| `--dtype` | `fp16` |
| `--warmup` / `--iters` | 10 / 30 |

```bash
python test_sparse_int8.py --seq-lens 1024 2048 4096 8192 16384 32768 --topk-ratio 0.8 --txt output/sparse_int8_results.txt
```

脚本在预热后逐次计时，对排序后的样本裁剪尾部并求平均。默认 30 次迭代时，两端各剔除 3 个样本；建议保持默认值，或采用不小于 10 的 10 的整数倍，以避免现有切片实现产生非对称裁剪。

此处 Speedup 为 `T_sparse / T_sparse_int8`；精度指标仍是 Sparse Int8 相对 SDPA 的误差。因此速度基准与数值参考不同，误差也不只是量化误差。

默认输出为 `output/sparse_int8_results.txt`。`--dtype bf16` 可选择 BF16 输入，但表头仍写作 `sparse(fp16)`；解释结果应以配置行中的 dtype 为准。

## 4. 结果解释与局限

- 固定形状、精度、随机种子、预热和迭代次数，避免其他 GPU 工作负载。
- 当前脚本报告单次实验均值，没有独立重复实验的置信区间。小幅时间差异不足以证明稳定优势。
- 随机张量与扩散模型真实激活分布不同，算子误差不能直接推导图像质量。
- 短序列中的块选择、量化和 kernel launch 开销可能抵消收益，Int8 不保证在所有形状上加速。
- `D=72` 补充命令用于检查实现覆盖，不属于既有报告中 `D∈{64,128}` 的基准统计。

完整结果见[实验报告](../doc/report.md)。
