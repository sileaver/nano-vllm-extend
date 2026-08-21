"""GPU-native batch preparation — Triton kernels that replace the CPU loop.

MRV2 core insight: keep request state on GPU, use Triton kernels to
*gather* per-step inputs.  The CPU only decides *which* requests are in
the batch; everything else happens on-device.

This eliminates:
- Python ``list.append`` / ``list.extend`` in a hot loop
- ``torch.tensor(..., pin_memory=True).cuda(non_blocking=True)`` H2D transfers
- Nested ``for i in range(num_scheduled_tokens)`` per sequence
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gather_tokens_kernel(
    # ── Per-token routing (pre-built on GPU) ──────────────────
    batch_idx_ptr,       # [total_tokens] int32 — which batch entry
    local_idx_ptr,       # [total_tokens] int32 — token index within seq

    # ── Per-sequence metadata (indexed by batch position) ─────
    seq_indices_ptr,     # [batch_size] int32 — row in GpuStateTable
    num_cached_ptr,      # [batch_size] int32
    num_computed_ptr,    # [batch_size] int32
    is_prefill_ptr,      # [batch_size] int32  (0/1 bool)

    # ── GPU state table ───────────────────────────────────────
    token_ids_ptr,       # [max_reqs, max_len] int64
    block_table_ptr,     # [max_reqs, max_blocks] int32

    # ── Output buffers ────────────────────────────────────────
    out_input_ids_ptr,   # [total_tokens] int64
    out_positions_ptr,   # [total_tokens] int64
    out_slot_mapping_ptr,# [total_tokens] int32

    # ── Constants ─────────────────────────────────────────────
    stride_tokens: tl.constexpr,
    stride_bt: tl.constexpr,
    block_size: tl.constexpr,
):
    """One program per *token* — no variable loop counts, fully parallel."""
    pid = tl.program_id(0)

    batch_idx = tl.load(batch_idx_ptr + pid)
    local_idx = tl.load(local_idx_ptr + pid)

    seq_idx = tl.load(seq_indices_ptr + batch_idx)
    num_cached = tl.load(num_cached_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    is_prefill = tl.load(is_prefill_ptr + batch_idx)

    # ── Compute input_id, position, and KV slot ───────────────
    # All arithmetic is done in int64 to avoid type mismatches.
    nc = num_cached.to(tl.int64)
    li = local_idx.to(tl.int64)
    nc_total = num_computed.to(tl.int64)

    if is_prefill:
        token_pos = nc + li
        kv_pos = token_pos
    else:
        token_pos = nc_total - 1
        kv_pos = nc

    token = tl.load(token_ids_ptr + seq_idx * stride_tokens + token_pos)
    tl.store(out_input_ids_ptr + pid, token)
    tl.store(out_positions_ptr + pid, token_pos)

    # ── slot_mapping ──────────────────────────────────────────
    block_idx = kv_pos // block_size
    off = kv_pos % block_size
    blk = tl.load(block_table_ptr + seq_idx * stride_bt + block_idx)
    tl.store(out_slot_mapping_ptr + pid, blk * block_size + off)


def gather_batch_inputs(
    seq_indices: torch.Tensor,       # [batch_size] int32, GPU
    num_cached: torch.Tensor,        # [batch_size] int32, GPU
    num_computed: torch.Tensor,      # [batch_size] int32, GPU
    is_prefill: torch.Tensor,        # [batch_size] int32, GPU
    sched_tokens: torch.Tensor,      # [batch_size] int32, GPU
    token_ids: torch.Tensor,         # [max_reqs, max_len] int64, GPU
    block_table: torch.Tensor,       # [max_reqs, max_blocks] int32, GPU
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """GPU-native replacement for the CPU ``prepare_batch`` loop.

    Returns
    -------
    input_ids:      [total_tokens] int64
    positions:      [total_tokens] int64
    slot_mapping:   [total_tokens] int32
    cu_seqlens_q:   [batch_size + 1] int32
    cu_seqlens_k:   [batch_size + 1] int32
    """
    total_tokens = int(sched_tokens.sum().item())
    if total_tokens == 0:
        raise ValueError("No tokens to prepare")

    # ── Build per-token routing (all on GPU, no sync) ────────
    # batch_idx[i] = which batch entry owns output token i
    batch_idx = torch.repeat_interleave(
        torch.arange(len(sched_tokens), device="cuda"),
        sched_tokens,
    )
    # local_idx[i] = token index within that sequence
    # Avoid .tolist() which syncs GPU→CPU.  Build via cumsum trick.
    offsets = torch.zeros(len(sched_tokens) + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(sched_tokens, dim=0, dtype=torch.int32, out=offsets[1:])
    total_tokens = int(offsets[-1].item())  # single int, cheap
    local_idx = torch.arange(total_tokens, dtype=torch.int32, device="cuda")
    local_idx -= offsets[:-1].repeat_interleave(sched_tokens)

    # ── Output buffers ────────────────────────────────────────
    input_ids = torch.empty(total_tokens, dtype=torch.int64, device="cuda")
    positions = torch.empty(total_tokens, dtype=torch.int64, device="cuda")
    slot_mapping = torch.empty(total_tokens, dtype=torch.int32, device="cuda")

    # ── Launch kernel ─────────────────────────────────────────
    grid = (total_tokens,)
    gather_tokens_kernel[grid](
        batch_idx,
        local_idx,
        seq_indices,
        num_cached,
        num_computed,
        is_prefill,
        token_ids,
        block_table,
        input_ids,
        positions,
        slot_mapping,
        stride_tokens=token_ids.stride(0),
        stride_bt=block_table.stride(0),
        block_size=block_size,
    )

    # ── Compute cumulative sequence lengths (on GPU) ──────────
    # seqlen_q = sched_tokens (already known)
    # seqlen_k = is_prefill ? num_cached + sched_tokens : num_computed
    seqlen_k = torch.where(
        is_prefill.bool(),
        num_cached + sched_tokens,
        num_computed,
    )

    cu_seqlens_q = torch.zeros(len(sched_tokens) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.zeros(len(sched_tokens) + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(sched_tokens, dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
    torch.cumsum(seqlen_k, dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

    return input_ids, positions, slot_mapping, cu_seqlens_q, cu_seqlens_k
