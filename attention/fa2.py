"""
Triton-based Flash Attention 2 implementation (forward only).
Optimized for inference without backward pass.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attention_2_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    M_CTX: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + pid_bh * M_CTX * HEAD_DIM
    k_base = k_ptr + pid_bh * N_CTX * HEAD_DIM
    v_base = v_ptr + pid_bh * N_CTX * HEAD_DIM
    out_base = out_ptr + pid_bh * M_CTX * HEAD_DIM

    q = tl.load(
        q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < M_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        n = start_n + offs_n
        k = tl.load(
            k_base + n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_base + n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * sm_scale
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


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def flash_attention_2(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    Flash Attention 2 forward pass using Triton.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        dropout_p: Dropout probability (default 0.0, ignored for API compatibility)
        training: Training mode flag (default False, ignored for API compatibility)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    if attn_mask is not None:
        raise NotImplementedError("flash_attention_2 does not support attn_mask.")
    if dropout_p > 0.0 and training:
        raise NotImplementedError("flash_attention_2 does not support dropout.")
    if q.device.type != "cuda":
        raise ValueError("flash_attention_2 requires CUDA tensors.")

    B, H, M, D = q.shape
    _, _, N, Dk = k.shape
    if v.shape != k.shape or Dk != D:
        raise ValueError("q, k, and v must have compatible [B, H, N, D] shapes.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    block_d = _next_power_of_2(D)
    if block_d > 128:
        raise ValueError(f"head_dim {D} is too large; supported maximum is 128.")

    block_m = 64
    block_n = 64
    num_warps = 4 if block_d <= 64 else 8
    grid = (triton.cdiv(M, block_m), B * H)
    sm_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634

    _flash_attention_2_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        M,
        N,
        D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )

    return out
