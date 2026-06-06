"""
Block-sparse attention with Triton kernel.

Uses block indices instead of boolean mask for better efficiency.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _sparse_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    block_indices_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    M_CTX: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
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

    q_base = q_ptr + pid_bh * M_CTX * HEAD_DIM
    k_base = k_ptr + pid_bh * N_CTX * HEAD_DIM
    v_base = v_ptr + pid_bh * N_CTX * HEAD_DIM
    out_base = out_ptr + pid_bh * M_CTX * HEAD_DIM
    idx_base = block_indices_ptr + (pid_bh * NUM_Q_BLOCKS + pid_q) * TOPK

    q = tl.load(
        q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < M_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for sparse_j in tl.range(0, TOPK):
        block_n = tl.load(idx_base + sparse_j)
        n = block_n * BLOCK_N + offs_n

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


def _ceil_div(x, y):
    return (x + y - 1) // y


def _is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def _pool_blocks(x, block_size):
    B, H, N, D = x.shape
    num_blocks = _ceil_div(N, block_size)
    pad_n = num_blocks * block_size - N

    if pad_n == 0:
        return x.view(B, H, num_blocks, block_size, D).mean(dim=-2)

    x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_n))
    pooled = x_pad.view(B, H, num_blocks, block_size, D).sum(dim=-2)
    counts = torch.full((num_blocks,), block_size, device=x.device, dtype=pooled.dtype)
    counts[-1] = block_size - pad_n
    return pooled / counts.view(1, 1, num_blocks, 1)


def select_topk_blocks(q, k, topk_ratio=0.5, block_size=64):
    """Return K block indices selected for each Q block.

    Shape: [B, H, num_q_blocks, topk], dtype int32.
    """
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    num_q_blocks = _ceil_div(M, block_size)
    num_k_blocks = _ceil_div(N, block_size)
    topk = max(1, min(num_k_blocks, math.ceil(num_k_blocks * float(topk_ratio))))

    if topk >= num_k_blocks:
        indices = torch.arange(num_k_blocks, device=q.device, dtype=torch.int32)
        return indices.view(1, 1, 1, num_k_blocks).expand(B, H, num_q_blocks, num_k_blocks).contiguous()

    q_pool = _pool_blocks(q, block_size).float()
    k_pool = _pool_blocks(k, block_size).float()
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * (D ** -0.5)
    return torch.topk(scores, k=topk, dim=-1, sorted=False).indices.to(torch.int32).contiguous()


def sparse_attention(q, k, v, attn_mask=None, topk_ratio=0.5, block_size=64):
    """
    Block-sparse attention with Triton kernel using block indices.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        topk_ratio: Ratio of K blocks to select per Q block (default 0.5)
        block_size: Block size for selection (default 64)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    if attn_mask is not None:
        raise NotImplementedError("sparse_attention does not support attn_mask.")
    if q.device.type != "cuda":
        raise ValueError("sparse_attention requires CUDA tensors.")
    if not _is_power_of_2(block_size):
        raise ValueError("block_size must be a power of two for the Triton kernel.")

    B, H, M, D = q.shape
    _, _, N, Dk = k.shape
    if v.shape != k.shape or Dk != D:
        raise ValueError("q, k, and v must have compatible [B, H, N, D] shapes.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    block_indices = select_topk_blocks(q, k, topk_ratio=topk_ratio, block_size=block_size)
    topk = block_indices.shape[-1]
    num_q_blocks = block_indices.shape[-2]

    block_d = _next_power_of_2(D)
    if block_d > 128:
        raise ValueError(f"head_dim {D} is too large; supported maximum is 128.")

    out = torch.empty_like(q)
    grid = (num_q_blocks, B * H)
    num_warps = 4 if block_d <= 64 else 8
    sm_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634

    _sparse_attention_fwd_kernel[grid](
        q,
        k,
        v,
        block_indices,
        out,
        sm_scale,
        M,
        N,
        D,
        num_q_blocks,
        topk,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )

    return out
