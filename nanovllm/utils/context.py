from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # FlashInfer backend: planned wrappers shared by all layers in this step.
    # num_decode_tokens splits the batch: decode tokens come first (q_len = 1
    # per seq), prefill chunks follow.
    num_decode_tokens: int = 0
    flashinfer_decode: object | None = None
    flashinfer_prefill: object | None = None
    # Draft-model KV metadata (speculative decoding with a small draft
    # model).  Draft attention layers read these instead of the target
    # slot_mapping / block_tables.
    draft_slot_mapping: torch.Tensor | None = None
    draft_block_tables: torch.Tensor | None = None
    # Hybrid models (linear attention layers): per-seq recurrent-state slot
    # ids into the GatedDeltaNet state pools ([bs] int64).
    linear_state_ids: torch.Tensor | None = None
    # CPU-side per-seq query lengths (same values cu_seqlens_q encodes) —
    # lets the GDN layer split mixed decode/prefill batches without a
    # GPU→CPU sync.  None when the caller didn't provide them (falls back
    # to a .tolist() on the tensor).
    lens_q: list[int] | None = None
    # Multimodal prefill: per-seq (row_start, row_end, image_embeds) triples
    # — vision-tower embeddings (already sliced to this chunk's pending
    # image tokens) replacing image_token_id rows after embedding.  Built
    # by ModelRunner.prepare_*; consumed by Qwen3_5Model._scatter_vision_embeds
    # on the first pipeline stage.  Decode steps leave it empty.
    vision_embeds: tuple = ()


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, block_tables=None, num_decode_tokens=0,
                flashinfer_decode=None, flashinfer_prefill=None,
                draft_slot_mapping=None, draft_block_tables=None,
                linear_state_ids=None, lens_q=None, vision_embeds=()):
    global _CONTEXT
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, block_tables, num_decode_tokens,
                       flashinfer_decode, flashinfer_prefill,
                       draft_slot_mapping, draft_block_tables,
                       linear_state_ids, lens_q, vision_embeds)


def reset_context():
    global _CONTEXT
    # Release FlashInfer plan buffers once per step, after all layers ran.
    for wrapper in (_CONTEXT.flashinfer_decode, _CONTEXT.flashinfer_prefill):
        if wrapper is not None:
            wrapper.end_forward()
    _CONTEXT = Context()
