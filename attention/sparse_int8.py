"""
Block-sparse attention with int8 Q/K quantization (Triton).

Adapted from thu-ml/SpargeAttn (Apache-2.0):
  - Triton_SpargeAttn/triton_kernel_example.py
  - spas_sage_attn/quant_per_block.py

Compared to attention/sparse.py, this variant:
  - Quantizes Q and K to int8 per-block (V stays fp16).
  - Bakes 1.44269504/sqrt(d) into Q's scale so the inner softmax uses exp2.
  - Uses int8 tensor-core matmul (tl.dot of int8 inputs -> int32, dequantized
    by `* q_scale * k_scale`).
  - Optional smooth_k (subtract per-channel K mean) — mathematically a no-op
    for softmax output but improves K's int8 dynamic range.
"""

import math

import torch
import triton
import triton.language as tl

from .sparse import _ceil_div, _is_power_of_2, _next_power_of_2, select_topk_blocks


@triton.jit
def _quantize_per_block_kernel(
    x_ptr,
    mean_ptr,
    x_i8_ptr,
    scale_ptr,
    scale_factor: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_MEAN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)

    base = pid_bh * N_CTX * HEAD_DIM
    x = tl.load(
        x_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    if HAS_MEAN:
        mean = tl.load(
            mean_ptr + pid_bh * HEAD_DIM + offs_d,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        x = x - mean[None, :]
        x = tl.where(valid, x, 0.0)

    abs_x = tl.abs(x)
    max_per_d = tl.max(abs_x, axis=0)
    max_abs = tl.max(max_per_d, axis=0)
    max_abs = tl.maximum(max_abs, 1.0e-6)

    scaled = x * (127.0 / max_abs)
    rounded = scaled + tl.where(scaled >= 0.0, 0.5, -0.5)
    clipped = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    x_i8 = clipped.to(tl.int8)

    tl.store(
        x_i8_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        x_i8,
        mask=valid,
    )
    tl.store(scale_ptr + pid_bh * NUM_BLOCKS + pid_n, (max_abs / 127.0) * scale_factor)


@triton.jit
def _sparse_int8_attention_fwd_kernel(
    q_i8_ptr,
    k_i8_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    block_indices_ptr,
    out_ptr,
    M_CTX: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_i8_ptr + pid_bh * M_CTX * HEAD_DIM
    k_base = k_i8_ptr + pid_bh * N_CTX * HEAD_DIM
    v_base = v_ptr + pid_bh * N_CTX * HEAD_DIM
    out_base = out_ptr + pid_bh * M_CTX * HEAD_DIM
    idx_base = block_indices_ptr + (pid_bh * NUM_Q_BLOCKS + pid_q) * TOPK

    q = tl.load(
        q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < M_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0,
    )
    q_scale = tl.load(q_scale_ptr + pid_bh * NUM_Q_BLOCKS + pid_q)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for sparse_j in tl.range(0, TOPK):
        block_n = tl.load(idx_base + sparse_j)
        n = block_n * BLOCK_N + offs_n
        k_scale = tl.load(k_scale_ptr + pid_bh * NUM_K_BLOCKS + block_n)

        k = tl.load(
            k_base + n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0,
        )
        v = tl.load(
            v_base + n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores_i32 = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        scores = scores_i32.to(tl.float32) * (q_scale * k_scale)
        valid_scores = (offs_m[:, None] < M_CTX) & (n[None, :] < N_CTX)
        scores = tl.where(valid_scores, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new = tl.where(offs_m < M_CTX, m_new, 0.0)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(valid_scores, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < M_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


def _quantize_qk(q, k, block_size, block_d, smooth_k):
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    num_q_blocks = _ceil_div(M, block_size)
    num_k_blocks = _ceil_div(N, block_size)

    q_i8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    k_i8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    q_scales = torch.empty((B, H, num_q_blocks), device=q.device, dtype=torch.float32)
    k_scales = torch.empty((B, H, num_k_blocks), device=k.device, dtype=torch.float32)

    k_mean = k.float().mean(dim=-2).contiguous() if smooth_k else k
    q_scale_factor = 1.4426950408889634 / math.sqrt(D)

    _quantize_per_block_kernel[(num_q_blocks, B * H)](
        q,
        q,
        q_i8,
        q_scales,
        q_scale_factor,
        M,
        D,
        num_q_blocks,
        HAS_MEAN=False,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    _quantize_per_block_kernel[(num_k_blocks, B * H)](
        k,
        k_mean,
        k_i8,
        k_scales,
        1.0,
        N,
        D,
        num_k_blocks,
        HAS_MEAN=smooth_k,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )

    return q_i8, k_i8, q_scales, k_scales


def sparse_int8_attention(q, k, v, attn_mask=None, topk_ratio=0.5,
                          block_size=64, smooth_k=True):
    """Block-sparse attention with int8 Q/K quantization.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        topk_ratio: Ratio of K blocks to select per Q block (default 0.5)
        block_size: Block size for selection and kernel tiles (default 64)
        smooth_k: Whether to subtract per-channel K mean before quantization (default True)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    if attn_mask is not None:
        raise NotImplementedError("sparse_int8_attention does not support attn_mask.")
    if q.device.type != "cuda":
        raise ValueError("sparse_int8_attention requires CUDA tensors.")
    if not _is_power_of_2(block_size):
        raise ValueError("block_size must be a power of two for the Triton kernel.")

    B, H, M, D = q.shape
    _, _, N, Dk = k.shape
    if v.shape != k.shape or Dk != D:
        raise ValueError("q, k, and v must have compatible [B, H, N, D] shapes.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    block_d = _next_power_of_2(D)
    if block_d > 128:
        raise ValueError(f"head_dim {D} is too large; supported maximum is 128.")

    block_indices = select_topk_blocks(q, k, topk_ratio=topk_ratio, block_size=block_size)
    topk = block_indices.shape[-1]
    num_q_blocks = block_indices.shape[-2]
    num_k_blocks = _ceil_div(N, block_size)

    q_i8, k_i8, q_scales, k_scales = _quantize_qk(q, k, block_size, block_d, smooth_k)

    out = torch.empty_like(q)
    grid = (num_q_blocks, B * H)
    num_warps = 4 if block_d <= 64 else 8

    _sparse_int8_attention_fwd_kernel[grid](
        q_i8,
        k_i8,
        v,
        q_scales,
        k_scales,
        block_indices,
        out,
        M,
        N,
        D,
        num_q_blocks,
        num_k_blocks,
        topk,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )

    return out
