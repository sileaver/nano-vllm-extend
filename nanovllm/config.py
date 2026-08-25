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
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    # TCP-store port offset per DP replica (each replica needs its own
    # rendezvous); assigned by the engine, 0 for standalone groups.
    dist_port: int = 2333
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
    # Hybrid models: number of linear-attention recurrent-state slots
    # (computed by ModelRunner alongside the KV-cache block count).
    num_linear_state_slots: int = 0
    # Speculative decoding.  num_spec_tokens: 0 = disabled, K > 0 = number
    # of draft tokens per step.  spec_draft_model: optional path to a small
    # draft model (must share the tokenizer/vocab of the target).  When
    # empty, a Jacobi-style parallel draft of the target itself is used;
    # when set, K sequential autoregressive drafts run on the draft model.
    num_spec_tokens: int = 0
    spec_draft_model: str = ""
    num_draft_kvcache_blocks: int = -1
    draft_hf_config: AutoConfig | None = None
    # Multimodal (qwen3_5 shell checkpoints): outer vision half.  None for
    # text-only models or NANOVLLM_QWEN35_TEXTONLY=1.  image_token_id etc.
    # are set alongside (see __post_init__).
    vision_config: AutoConfig | None = None
    image_token_id: int = -1
    video_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.pipeline_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 8
        assert self.data_parallel_size * self.pipeline_parallel_size \
            * self.tensor_parallel_size <= 8, "at most 8 GPUs supported"
        assert self.attention_backend in ("flash_attn", "flashinfer")
        assert self.sampling_backend in ("torch", "flashinfer")
        hf_config = AutoConfig.from_pretrained(self.model)
        # Multimodal shells (e.g. Qwen3.5's Qwen3_5ForConditionalGeneration)
        # nest the language model inside text_config — unwrap it so every
        # downstream consumer sees a plain text config, but keep the outer
        # shell's vision half for multimodal builds (NANOVLLM_QWEN35_TEXTONLY=1
        # restores the old text-only behaviour).
        shell = hf_config if getattr(hf_config, "text_config", None) is not None else None
        if shell is not None:
            hf_config = hf_config.text_config
        self.hf_config = hf_config
        text_only = os.environ.get("NANOVLLM_QWEN35_TEXTONLY", "0") == "1"
        if (shell is not None and not text_only
                and getattr(shell, "vision_config", None) is not None):
            self.vision_config = shell.vision_config
            self.image_token_id = shell.image_token_id
            self.video_token_id = getattr(shell, "video_token_id", None)
            self.vision_start_token_id = getattr(shell, "vision_start_token_id", None)
            self.vision_end_token_id = getattr(shell, "vision_end_token_id", None)
        else:
            self.vision_config = None
        if "qwen3_5" in hf_config.model_type:
            # Hybrid architecture: TP shards the GDN heads (num_v_heads must
            # divide evenly).  Spec decode (state rollback), gpu_prepare and
            # the flashinfer wrappers remain unsupported.
            assert self.num_spec_tokens == 0, "qwen3_5: speculative decoding unsupported"
            assert not self.gpu_prepare, "qwen3_5: gpu_prepare unsupported"
            assert hf_config.linear_num_value_heads % self.tensor_parallel_size == 0, \
                "qwen3_5: linear_num_value_heads not divisible by tensor_parallel_size"
            assert hf_config.linear_num_key_heads % self.tensor_parallel_size == 0, \
                "qwen3_5: linear_num_key_heads not divisible by tensor_parallel_size"
            assert hf_config.num_key_value_heads % self.tensor_parallel_size == 0, \
                "qwen3_5: num_key_value_heads not divisible by tensor_parallel_size"
            assert self.attention_backend == "flash_attn", "qwen3_5: use the flash_attn backend"
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
