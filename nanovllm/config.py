import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    async_scheduling: bool = False
    continuous_batching: bool = False
    gpu_prepare: bool = False
    attention_backend: str = "flash_attn"
    sampling_backend: str = "torch"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Speculative decoding.  num_spec_tokens: 0 = disabled, K > 0 = number
    # of draft tokens per step.  spec_draft_model: optional path to a small
    # draft model (must share the tokenizer/vocab of the target).  When
    # empty, a Jacobi-style parallel draft of the target itself is used;
    # when set, K sequential autoregressive drafts run on the draft model.
    num_spec_tokens: int = 0
    spec_draft_model: str = ""
    num_draft_kvcache_blocks: int = -1
    draft_hf_config: AutoConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.attention_backend in ("flash_attn", "flashinfer")
        assert self.sampling_backend in ("torch", "flashinfer")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.num_spec_tokens > 0:
            assert 1 <= self.num_spec_tokens <= 8, "num_spec_tokens must be in [1, 8]"
            assert self.max_num_batched_tokens >= self.num_spec_tokens + 1
        if self.spec_draft_model:
            assert self.num_spec_tokens > 0, "spec_draft_model requires num_spec_tokens > 0"
            assert os.path.isdir(self.spec_draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.spec_draft_model)
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, \
                "draft and target vocab sizes must match"
            if "DFlashDraftModel" in self.draft_hf_config.architectures:
                # DFlash block diffusion draft: one forward proposes
                # block_size-1 tokens; K is fixed by the block size.
                self.num_spec_tokens = self.draft_hf_config.block_size - 1
        else:
            self.draft_hf_config = None
