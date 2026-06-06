# Formal Experiment Summary

GPU: NVIDIA GeForce RTX 4060 Laptop GPU

## T2I Timing

| mode | topk | images | avg s/image | total s | output |
|---|---:|---:|---:|---:|---|
| sdpa | - | 20 | 20.419 | 408.387 | `output\2026-06-06_attention_sdpa` |
| sparse_int8 | 0.3 | 20 | 15.582 | 311.632 | `output\2026-06-06_attention_sparse_int8_topk0.3` |
| sparse_int8 | 0.5 | 20 | 17.237 | 344.741 | `output\2026-06-06_attention_sparse_int8_topk0.5` |
| sparse_int8 | 0.8 | 20 | 19.719 | 394.376 | `output\2026-06-06_attention_sparse_int8_topk0.8` |
| sparse_int8 | 0.9 | 20 | 20.856 | 417.119 | `output\2026-06-06_attention_sparse_int8_topk0.9` |
| sparse_int8 | 1.0 | 20 | 21.309 | 426.179 | `output\2026-06-06_attention_sparse_int8_topk1.0` |
| sparse | 0.3 | 20 | 15.589 | 311.784 | `output\2026-06-06_attention_sparse_topk0.3` |
| sparse | 0.5 | 20 | 17.792 | 355.836 | `output\2026-06-06_attention_sparse_topk0.5` |
| sparse | 0.8 | 20 | 20.481 | 409.626 | `output\2026-06-06_attention_sparse_topk0.8` |
| sparse | 0.9 | 20 | 22.092 | 441.839 | `output\2026-06-06_attention_sparse_topk0.9` |
| sparse | 1.0 | 20 | 22.868 | 457.358 | `output\2026-06-06_attention_sparse_topk1.0` |
| triton_fa2 | - | 20 | 22.001 | 440.023 | `output\2026-06-06_attention_triton_fa2` |
| vanilla | - | 20 | 54.203 | 1084.065 | `output\2026-06-06_attention_vanilla` |

## Benchmark Key Summary

| backend | cases | min speedup vs SDPA / ratio | min CosSim | max RelL1 | max RMSE |
|---|---:|---:|---:|---:|---:|
| triton_fa2 | 16 | 0.964025 | 1 | 5.22056e-05 | 5.60227e-06 |
| sparse(topk=1.0) | 16 | 0.950323 | 1 | 5.22056e-05 | 5.60227e-06 |
| sparse(topk=0.8) | 16 | 1.14376 | 0.899504 | 0.48818 | 0.0172614 |
| sparse_int8(topk=1.0) | 16 | 1.04265 | 0.999916 | 0.0128646 | 0.000469533 |
| sparse_int8(topk=0.8) | 16 | 0.914834 | 0.899439 | 0.488374 | 0.0172688 |
| sparse: topk1.0_time/topk0.8_time | 16 | 0.970769 | - | - | - |
| sparse_int8: topk1.0_time/topk0.8_time | 16 | 0.704683 | - | - | - |

## Sparse Int8 Long Sequence

```text
  1024 |      0.631 ms |      0.608 ms |    1.04x OK |   0.904601 | 4.69e-01 | 2.43e-02
  2048 |      1.823 ms |      1.253 ms |    1.46x OK |   0.904990 | 4.69e-01 | 1.73e-02
  4096 |      5.056 ms |      3.749 ms |    1.35x OK |   0.904740 | 4.71e-01 | 1.22e-02
  8192 |     22.100 ms |     15.509 ms |    1.42x OK |   0.901482 | 4.82e-01 | 8.85e-03
 16384 |     85.740 ms |     60.166 ms |    1.43x OK |   0.901004 | 4.83e-01 | 6.33e-03
 32768 |    350.490 ms |    237.585 ms |    1.48x OK |   0.901479 | 4.82e-01 | 4.48e-03
```
