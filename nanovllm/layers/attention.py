import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    # No flash-attn wheel exists for this torch/CUDA combo (cu13 + sm120).
    # vLLM vendors the same FA2 varlen kernel with an identical API —
    # numerically verified against a matmul reference on this GPU.
    # Two differences: it exports no flash_attn_with_kvcache (unused here),
    # and with a block_table it requires seqused_k instead of cu_seqlens_k.
    from vllm.vllm_flash_attn import flash_attn_varlen_func


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # Draft-model layers (small-model speculative decoding) read their
        # own KV cache: draft_slot_mapping / draft_block_tables instead of
        # the target ones, and always use the flash_attn kernel (the
        # flashinfer wrappers are planned for the target's head counts).
        self.is_draft = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if self.is_draft and context.draft_slot_mapping is not None:
            slot_mapping = context.draft_slot_mapping
            block_tables = context.draft_block_tables
        else:
            slot_mapping = context.slot_mapping
            block_tables = context.block_tables

        # Step 1 — write new KV to paged cache at positions in slot_mapping.
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # Step 2 — unified varlen attention for both prefill AND decode tokens.
        # When block_tables is available the KV cache is used as the
        # key/value source; otherwise the raw k/v tensors are used (e.g.
        # warmup with no block table).
        if block_tables is not None:
            k, v = k_cache, v_cache

        # FlashInfer backend: the batch is split into a decode part (one
        # query token per seq) and a prefill part (variable-length chunks).
        # Wrappers are planned once per step in ModelRunner.prepare_batch
        # and shared by every layer; end_forward happens in reset_context.
        if not self.is_draft and (
                context.flashinfer_decode is not None
                or context.flashinfer_prefill is not None):
            nd = context.num_decode_tokens
            if nd == 0:
                return context.flashinfer_prefill.run(q, (k_cache, v_cache))
            if nd == q.size(0):
                return context.flashinfer_decode.run(q, (k_cache, v_cache))
            return torch.cat((
                context.flashinfer_decode.run(q[:nd], (k_cache, v_cache)),
                context.flashinfer_prefill.run(q[nd:], (k_cache, v_cache)),
            ))

        # Paged path (block_tables): the per-seq KV length is seqused_k =
        # diff(cu_seqlens_k); the vllm FA2 fork takes it INSTEAD of
        # cu_seqlens_k when a block_table is given. The ragged path (raw
        # k/v, no block table) keeps cu_seqlens_k.
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=(None if block_tables is not None
                          else context.cu_seqlens_k),
            seqused_k=(torch.diff(context.cu_seqlens_k)
                       if block_tables is not None else None),
            softmax_scale=self.scale,
            causal=True,
            block_table=block_tables,
        )
        return o
