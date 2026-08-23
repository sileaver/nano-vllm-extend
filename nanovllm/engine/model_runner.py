import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.gpu_state import GpuStateTable
from nanovllm.engine.gpu_prepare import gather_batch_inputs
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.flashinfer_env import setup_flashinfer_env
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.use_gpu_prepare = config.gpu_prepare
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # ── GPU-native state table (MRV2-style persistent batch) ──
        if self.use_gpu_prepare:
            max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
            self.gpu_state = GpuStateTable(
                max_num_reqs=config.max_num_seqs,
                max_model_len=config.max_model_len,
                max_num_blocks=max_num_blocks,
                block_size=self.block_size,
            )
        else:
            self.gpu_state = None

        # ── FlashInfer attention backend (optional) ───────────────
        self.use_flashinfer = config.attention_backend == "flashinfer"
        if self.use_flashinfer:
            setup_flashinfer_env()
            import flashinfer
            self.flashinfer = flashinfer
            # One reusable workspace for plan/run aux buffers (split-K etc.).
            # 本环境 (SM120 AOT 内核) 实测 16k chunk prefill 的 workspace 需求
            # 为 0 (workspace_size() 返回 0); 128 MB 是安全下限, 再大会
            # 白占 KV cache 块 (每 29 MB ≈ 1 块), 加剧长队列的抢占.
            self._fi_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        else:
            self.flashinfer = None
            self._fi_workspace = None

        if "qwen3_5" in hf_config.model_type:
            self.model = Qwen3_5ForCausalLM(hf_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # ── Hybrid models: linear-attention recurrent-state pools ──
        # Allocated before warmup (the warmup forward consumes slots) and
        # before the KV cache budget computation (the pools are resident).
        self.linear_s_pool: torch.Tensor | None = None
        self.linear_conv_pool: torch.Tensor | None = None
        if any(hasattr(m, "s_cache") for m in self.model.modules()):
            self._allocate_linear_state_pool()
        # ── Draft model (classic small-model or DFlash block diffusion) ──
        self.use_draft_model = bool(config.spec_draft_model)
        self.use_dflash_draft = False
        if self.use_draft_model:
            if "DFlashDraftModel" in config.draft_hf_config.architectures:
                from nanovllm.models.dflash import DFlashDraftModel
                self.draft_model = DFlashDraftModel(config.draft_hf_config)
                load_model(self.draft_model, config.spec_draft_model)
                self.use_dflash_draft = True
                # Per-sequence draft context: the accepted tokens' target
                # hidden features ([C, n_layers*H] GPU tensors keyed by
                # seq_id).  Initialised at prefill, updated every step.
                self._dflash_ctx: dict[int, torch.Tensor] = {}
            else:
                self.draft_model = Qwen3ForCausalLM(config.draft_hf_config)
                load_model(self.draft_model, config.spec_draft_model)
        else:
            self.draft_model = None
        self.sampling_backend = config.sampling_backend
        self.sampler = Sampler(config.sampling_backend)
        # ── Speculative decoding (draft model, or Jacobi draft) ──
        self.num_spec_tokens = config.num_spec_tokens
        if self.num_spec_tokens > 0:
            assert config.tensor_parallel_size == 1, "spec v1: tensor_parallel_size must be 1"
            assert config.sampling_backend == "torch", "spec v1: sampling_backend must be 'torch'"
            assert not config.gpu_prepare, "spec v1: gpu_prepare unsupported"
        self.warmup_model()
        self.allocate_kv_cache()
        # CUDA graphs cover uniform-decode only.  With the FlashInfer
        # backend the graph embeds the frozen batched decode wrapper
        # (plan outside, run inside) instead of flash_attn.
        if not self.enforce_eager:
            self.graphs, self.graph_vars = self.capture_cudagraph()
            if self.draft_model is not None and not self.use_dflash_draft:
                # Draft decode graphs (single-token steps of the small
                # model replay with dynamic cu_seqlens_k / draft slots).
                # DFlash drafts run one block_size-row non-causal forward
                # per step and stay eager.
                self.draft_graphs, self.draft_graph_vars = self.capture_cudagraph(
                    model=self.draft_model, is_draft=True)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def _allocate_linear_state_pool(self):
        """GPU pools for the gated-delta-net recurrent/conv states.

        Layout: s_pool [slots, L, H, K, V] fp32 + conv_pool [slots, L,
        conv_dim, k-1]; per-layer views are bound to the GatedDeltaNet
        modules (same pattern as the paged-KV k/v_cache binding).  Slots
        cost ~19.5 MB each on Qwen3.5-2B, so max_num_seqs is clamped to a
        ~4 GB budget.
        """
        gdns = [m for m in self.model.modules() if hasattr(m, "s_cache")]
        L = len(gdns)
        m0 = gdns[0]
        H, K, V = m0.num_v_heads, m0.head_k_dim, m0.head_v_dim
        conv_dim, conv_states = m0.conv_dim, m0.conv_kernel - 1
        per_slot = L * (H * K * V * 4 + conv_dim * conv_states * 2)
        free, total = torch.cuda.mem_get_info()
        budget = min(4 * 1024**3, int(total * 0.25))
        num_slots = min(self.config.max_num_seqs, max(8, budget // per_slot))
        self.config.max_num_seqs = min(self.config.max_num_seqs, num_slots)
        self.config.num_linear_state_slots = num_slots
        self.linear_s_pool = torch.zeros(num_slots, L, H, K, V,
                                         dtype=torch.float32, device="cuda")
        self.linear_conv_pool = torch.zeros(num_slots, L, conv_dim, conv_states,
                                            device="cuda")
        for layer_id, module in enumerate(gdns):
            module.s_cache = self.linear_s_pool[:, layer_id]
            module.conv_cache = self.linear_conv_pool[:, layer_id]

    def reset_linear_states(self, slot_ids: torch.Tensor):
        """Zero a slot's recurrent + conv state (fresh-prefill semantics)."""
        self.linear_s_pool[slot_ids] = 0
        self.linear_conv_pool[slot_ids] = 0

    def _linear_state_context(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Per-seq state slot ids for the context; fresh prefills (the
        first chunk, or a recomputed prefill after preemption) get their
        slots zeroed — a zero state IS the initial state."""
        if self.linear_s_pool is None:
            return None
        fresh = [seq.linear_state_id for seq in seqs
                 if seq.is_prefill and seq.num_cached_tokens == 0]
        if fresh:
            self.reset_linear_states(
                torch.tensor(fresh, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True))
        return torch.tensor([seq.linear_state_id for seq in seqs],
                            dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        if self.linear_s_pool is not None:
            # 纯 torch chunk kernel 跑超长 warmup 太慢 (18 层 fp32 chunk 循环);
            # warmup 只为预热分配器与 compile 缓存, 2048 token 足够.
            # 单个 seq 即可: 多 seq 并发 chunk 的 fp32 中间张量峰值会被
            # allocate_kv_cache 的 peak-memory 扣减吃掉数 GB 的 KV 预算.
            seq_len = min(seq_len, 2048)
            num_seqs = 1
        else:
            num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
            seq.is_prefill = True
        if self.linear_s_pool is not None:
            # Warmup forwards need valid slots; borrow 0..num_seqs-1 (they
            # are re-zeroed whenever a real sequence starts on them).
            for i, seq in enumerate(seqs):
                seq.linear_state_id = i
        self.run(seqs)
        if self.num_spec_tokens > 0:
            # Trigger inductor compilation of sample_with_probs so the
            # first speculative step doesn't pay a multi-second cold start.
            dummy_logits = torch.zeros(2, self.config.hf_config.vocab_size,
                                       dtype=torch.float32, device="cuda")
            dummy_temps = torch.ones(2, dtype=torch.float32, device="cuda")
            self.sampler.sample_with_probs(dummy_logits, dummy_temps)
        if self.draft_model is not None and not self.use_dflash_draft:
            # Warm up the draft model (torch.compile caches + allocator).
            with torch.inference_mode():
                cu = torch.arange(17, dtype=torch.int32, device="cuda")
                set_context(cu, cu, 16, 16, None, None)
                self.draft_model(torch.zeros(16, dtype=torch.int64, device="cuda"),
                                 torch.arange(16, dtype=torch.int64, device="cuda"))
                reset_context()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # Hybrid models: only the full-attention layers hold paged KV.
        layer_types = getattr(hf_config, "layer_types", None)
        num_kv_layers = (sum(1 for t in layer_types if t != "linear_attention")
                         if layer_types else hf_config.num_hidden_layers)
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if self.draft_model is not None and not self.use_dflash_draft:
            # Split the remaining budget between target and draft KV by
            # equal block COUNT (both caches hold the same tokens per seq).
            # (DFlash drafts keep no KV cache — only the context feature.)
            dcfg = config.draft_hf_config
            dkv_heads = dcfg.num_key_value_heads
            d_head_dim = getattr(dcfg, "head_dim", dcfg.hidden_size // dcfg.num_attention_heads)
            d_block_bytes = 2 * dcfg.num_hidden_layers * self.block_size * dkv_heads * d_head_dim * dcfg.dtype.itemsize
            budget = int(total * config.gpu_memory_utilization - used - peak + current)
            num_blocks = budget // (block_bytes + d_block_bytes)
            assert num_blocks > 0, "not enough memory for KV caches"
            config.num_kvcache_blocks = num_blocks
            config.num_draft_kvcache_blocks = num_blocks
        else:
            # Activation headroom: the warmup forward (a single small batch)
            # does not capture the transient peak of a full
            # max_num_batched_tokens prefill step (~128KB/token across the
            # layers); without reserving it the first big prefill OOMs into
            # an allocator retry livelock.
            headroom = config.max_num_batched_tokens * 128 * 1024 if self.linear_s_pool is not None else 0
            config.num_kvcache_blocks = int(
                total * config.gpu_memory_utilization - used - peak + current
                - headroom) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_kv_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if self.draft_model is not None and not self.use_dflash_draft:
            self.draft_kv_cache = torch.empty(
                2, dcfg.num_hidden_layers, config.num_draft_kvcache_blocks,
                self.block_size, dkv_heads, d_head_dim)
            layer_id = 0
            for module in self.draft_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.draft_kv_cache[0, layer_id]
                    module.v_cache = self.draft_kv_cache[1, layer_id]
                    module.is_draft = True
                    layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence],
                             block_table_attr: str = "block_table"):
        tables = [getattr(seq, block_table_attr) for seq in seqs]
        max_len = max(len(t) for t in tables)
        block_tables = [t + [-1] * (max_len - len(t)) for t in tables]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # ------------------------------------------------------------------
    # FlashInfer backend: plan decode/prefill wrappers once per step
    # ------------------------------------------------------------------

    def plan_flashinfer(self, seqs: list[Sequence]):
        """Build paged-KV metadata and plan FlashInfer wrappers for one step.

        The scheduler emits decode seqs (one query token each) before
        prefill chunks, so the batch splits at the first seq with more than
        one scheduled token.  Returns ``(num_decode_tokens, decode_wrapper,
        prefill_wrapper)``; wrappers are ``None`` when their part is empty
        or the backend is disabled / the batch has no block tables (then
        attention falls back to the flash_attn path, which also covers
        uniform-decode steps replayed from CUDA graphs).

        The wrappers are shared by every layer via the attention context;
        ``end_forward`` runs once per step in ``reset_context``.
        """
        if not self.use_flashinfer:
            return 0, None, None
        if not seqs or any(not seq.block_table for seq in seqs):
            return 0, None, None

        num_decode = 0
        for seq in seqs:
            if seq.num_scheduled_tokens == 1:
                num_decode += 1
            else:
                break

        # 纯 decode 批走 flash_attn 的 CUDA graph 回放 (或 enforce_eager 下
        # 的 varlen 路径), flashinfer wrapper 不会被用到 — 跳过元数据构建
        # 与 plan, 省下每个 decode 步骤的固定开销.
        if num_decode == len(seqs):
            return 0, None, None

        # Paged-KV metadata: pages come from the block table; the KV length
        # visible to attention = cached tokens + this step's tokens (the
        # just-written KV from store_kvcache is included).  Only pages
        # covering the actual KV length are listed — the block table may
        # hold pre-allocated pages beyond it (chunked prefill), and
        # flashinfer would treat those as valid KV.
        kv_indptr = [0]
        kv_indices = []
        last_page_len = []
        for seq in seqs:
            kv_len = seq.num_cached_tokens + seq.num_scheduled_tokens
            num_kv_pages = (kv_len + self.block_size - 1) // self.block_size
            kv_indices.extend(seq.block_table[:num_kv_pages])
            kv_indptr.append(kv_indptr[-1] + num_kv_pages)
            last_page_len.append(kv_len % self.block_size or self.block_size)

        indptr = torch.tensor(kv_indptr, dtype=torch.int32, device="cuda")
        indices = torch.tensor(kv_indices, dtype=torch.int32, device="cuda")
        last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device="cuda")

        # TP-local head counts from the first attention layer.
        attn = self.model.model.layers[0].self_attn.attn
        dtype = next(self.model.parameters()).dtype

        fi_decode = fi_prefill = None
        if num_decode > 0:
            fi_decode = self.flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._fi_workspace, "NHD", use_tensor_cores=True)
            fi_decode.plan(
                indptr[:num_decode + 1],
                indices[:indptr[num_decode]],
                last_page_len[:num_decode],
                attn.num_heads, attn.num_kv_heads, attn.head_dim, self.block_size,
                pos_encoding_mode="NONE", q_data_type=dtype,
            )
        if num_decode < len(seqs):
            start = num_decode
            qo_indptr = [0]
            for seq in seqs[start:]:
                qo_indptr.append(qo_indptr[-1] + seq.num_scheduled_tokens)
            qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device="cuda")
            fi_prefill = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self._fi_workspace, "NHD")
            fi_prefill.plan(
                qo_indptr,
                indptr[start:] - indptr[start],
                indices[indptr[start]:],
                last_page_len[start:],
                attn.num_heads, attn.num_kv_heads, attn.head_dim, self.block_size,
                causal=True, pos_encoding_mode="NONE", q_data_type=dtype,
            )
        return num_decode, fi_decode, fi_prefill

    # ------------------------------------------------------------------
    # Unified batch preparation (continuous batching)
    #
    # Handles prefill chunks (variable query length) and decode tokens
    # (query length = 1) in a single forward pass via varlen attention.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Batch preparation — dispatches between CPU (legacy) and GPU-native
    # ------------------------------------------------------------------

    def prepare_batch(self, seqs: list[Sequence]):
        if self.use_gpu_prepare and self.gpu_state is not None:
            return self._prepare_batch_gpu(seqs)
        else:
            return self._prepare_batch_cpu(seqs)

    def _sync_seq_to_gpu(self, seq: Sequence):
        """Ensure GPU state table is up-to-date for *seq*."""
        if not hasattr(seq, '_gpu_row'):
            row = self.gpu_state.alloc_row()
            seq._gpu_row = row
            self.gpu_state.add_request(seq, row)
        else:
            self.gpu_state.update(seq)
            # Token ids change during prefill (chunk by chunk), so sync
            # the full array when in prefill mode.
            if seq.is_prefill:
                self.gpu_state.update_token_ids(seq)

    def _prepare_batch_gpu(self, seqs: list[Sequence]):
        """GPU-native: Triton kernel gathers inputs from persistent table."""
        # ── Sync CPU → GPU state (metadata only, O(batch_size)) ──
        for seq in seqs:
            self._sync_seq_to_gpu(seq)

        batch_size = len(seqs)

        # ── Build batch-metadata tensors (small, on GPU) ────────
        seq_indices = torch.tensor(
            [seq._gpu_row for seq in seqs], dtype=torch.int32, device="cuda"
        )
        num_cached = torch.tensor(
            [seq.num_cached_tokens for seq in seqs], dtype=torch.int32, device="cuda"
        )
        num_computed = torch.tensor(
            [seq.num_computed_tokens for seq in seqs], dtype=torch.int32, device="cuda"
        )
        is_prefill = torch.tensor(
            [seq.is_prefill for seq in seqs], dtype=torch.int32, device="cuda"
        )
        sched_tokens = torch.tensor(
            [seq.num_scheduled_tokens for seq in seqs], dtype=torch.int32, device="cuda"
        )

        # ── Launch Triton gather kernel ─────────────────────────
        (input_ids, positions, slot_mapping,
         cu_seqlens_q, cu_seqlens_k) = gather_batch_inputs(
            seq_indices=seq_indices,
            num_cached=num_cached,
            num_computed=num_computed,
            is_prefill=is_prefill,
            sched_tokens=sched_tokens,
            token_ids=self.gpu_state.token_ids,
            block_table=self.gpu_state.block_table,
            block_size=self.block_size,
        )

        # ── Compute max sequence lengths ────────────────────────
        max_seqlen_q = int(sched_tokens.max().item()) if batch_size > 0 else 0
        seqlen_k = torch.where(
            is_prefill.bool(), num_cached + sched_tokens, num_computed
        )
        max_seqlen_k = int(seqlen_k.max().item()) if batch_size > 0 else 0
        is_uniform_decode = (max_seqlen_q == 1)

        # ── Block tables for flash-attn ─────────────────────────
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        else:
            block_tables = None

        num_decode, fi_decode, fi_prefill = self.plan_flashinfer(seqs)
        linear_state_ids = self._linear_state_context(seqs)
        set_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, block_tables, num_decode, fi_decode, fi_prefill,
                    linear_state_ids=linear_state_ids)
        return input_ids, positions, is_uniform_decode

    def _prepare_batch_cpu(self, seqs: list[Sequence]):
        """Original CPU loop (kept for reference / fallback)."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            if seq.is_prefill:
                start = seq.num_cached_tokens
                seqlen_q = seq.num_scheduled_tokens
                end = start + seqlen_q
                seqlen_k = end
                input_ids.extend(seq[start:end])
                positions.extend(range(start, end))
            else:
                seqlen_q = 1
                # +1 包含 store_kvcache 刚写入的当前 token (self), 与 async
                # 路径的 placeholder 语义 (num_computed = num_cached + 1)
                # 保持一致; 旧代码用 num_computed 在 sync 模式下少算一个
                # token 且 RoPE 位置落后 1。
                seqlen_k = seq.num_cached_tokens + 1
                input_ids.append(seq.last_token)
                positions.append(seq.num_cached_tokens)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue

            for i in range(seq.num_scheduled_tokens):
                pos = seq.num_cached_tokens + i
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                slot_mapping.append(
                    seq.block_table[block_idx] * self.block_size + offset
                )

        is_uniform_decode = (max_seqlen_q == 1)

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        num_decode, fi_decode, fi_prefill = self.plan_flashinfer(seqs)
        linear_state_ids = self._linear_state_context(seqs)
        set_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, block_tables, num_decode, fi_decode, fi_prefill,
                    linear_state_ids=linear_state_ids)
        return input_ids, positions, is_uniform_decode

    def prepare_sample(self, seqs: list[Sequence]):
        """采样参数张量; 统一返回元组供 ``sampler(logits, *params)`` 解包
        (torch 后端单元素元组, flashinfer 后端 (temperatures, top_k, top_p))."""
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float32,
            pin_memory=True).cuda(non_blocking=True)
        if self.sampling_backend != "flashinfer":
            return (temperatures,)
        vocab_size = self.config.hf_config.vocab_size
        top_k = torch.tensor(
            [vocab_size if seq.top_k < 0 else seq.top_k for seq in seqs],
            dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_p = torch.tensor(
            [seq.top_p for seq in seqs], dtype=torch.float32,
            pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_k, top_p

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  is_uniform_decode: bool):
        context = get_context()
        # A single-token prompt's prefill has cu_k == cu_q, so no block
        # tables are built (raw k/v is exactly the full KV there); the
        # graph-replay path needs them — fall back to eager for that step.
        if (not is_uniform_decode or self.enforce_eager or input_ids.size(0) > 512
                or context.block_tables is None):
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["cu_seqlens_q"].zero_()
            graph_vars["cu_seqlens_q"][:bs + 1] = torch.arange(bs + 1, dtype=torch.int32)
            graph_vars["cu_seqlens_k"].zero_()
            graph_vars["cu_seqlens_k"][:bs + 1] = context.cu_seqlens_k
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if "linear_state_ids" in graph_vars:
                graph_vars["linear_state_ids"][:bs] = context.linear_state_ids
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def run(self, seqs: list[Sequence]) -> list[int]:
        if not seqs:
            # Starved schedule (e.g. no sequence fits the KV pool): nothing
            # to run this step.  An empty batch would break the hybrid
            # model's conv window (max_seqlen_q == 0) and is pointless GPU
            # work anyway.
            return None
        input_ids, positions, is_uniform = self.prepare_batch(seqs)
        sample_params = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_uniform)
        if self.draft_model is not None and any(s.is_prefill and s.block_table for s in seqs):
            if self.use_dflash_draft:
                # Initialise the DFlash draft context per sequence: the
                # hidden features of ALL rows of this prefill chunk (the
                # official impl passes the full prompt hidden_states —
                # logits_to_keep only trims logits).
                with torch.inference_mode():
                    _, ctx_cat = self.model(
                        input_ids, positions,
                        output_layer_hidden=self.draft_model.target_layer_ids)
                context = get_context()
                for seq, r0, r1 in zip(
                        seqs, context.cu_seqlens_q[:-1].tolist(),
                        context.cu_seqlens_q[1:].tolist()):
                    if r1 > r0:
                        self._dflash_ctx[seq.seq_id] = ctx_cat[r0:r1].clone()
            else:
                # Keep the draft KV cache in sync with the target: any
                # token whose target KV was written this step also needs
                # draft KV (the draft model reads its own cache during
                # speculation).
                self._run_draft_mirror(seqs, input_ids, positions)
        token_ids = self.sampler(logits, *sample_params).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def _run_draft_mirror(self, seqs: list[Sequence], input_ids: torch.Tensor,
                          positions: torch.Tensor):
        """Run the draft model over the same batch (same rows/positions as
        the target forward) writing into the draft KV cache."""
        context = get_context()
        draft_slots = []
        for seq in seqs:
            for i in range(seq.num_scheduled_tokens):
                pos = seq.num_cached_tokens + i
                block_idx = pos // self.block_size
                offset = pos % self.block_size
                draft_slots.append(
                    seq.draft_block_table[block_idx] * self.block_size + offset)
        draft_slots = torch.tensor(draft_slots, dtype=torch.int32,
                                   pin_memory=True).cuda(non_blocking=True)
        draft_bt = self.prepare_block_tables(seqs, "draft_block_table")
        set_context(context.cu_seqlens_q, context.cu_seqlens_k,
                    context.max_seqlen_q, context.max_seqlen_k,
                    None, None, 0, None, None,
                    draft_slot_mapping=draft_slots,
                    draft_block_tables=draft_bt)
        self.draft_model(input_ids, positions)

    # ------------------------------------------------------------------
    # Speculative decode step (Jacobi-style parallel draft)
    #
    # Draft phase: the FULL model runs once over K rows per seq (all rows
    # fed the last accepted token, causally visible to each other — pure
    # Jacobi parallel draft), writing KV into slots C..C+K-1
    # (C = num_cached_tokens).  Note: the layer-skipped draft (self-spec)
    # was measured and abandoned — Qwen3-0.6B has no layer redundancy
    # (skipping 1 layer drops the draft match rate from ~0.5 to ~0.14).
    #
    # Verify phase: the full model runs once over K+1 rows per seq
    # ([last, d_1..d_K] at positions C..C+K), then per-position
    # rejection sampling accepts a variable-length prefix.
    #
    # The scheduler pre-allocated the block table for K+1 positions and
    # set num_scheduled_tokens = K+1.  FlashInfer wrappers are never
    # passed here — the speculative step always uses the flash_attn
    # kernel path regardless of attention_backend.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_speculative(self, seqs: list[Sequence]) -> list[list[int]]:
        if self.use_dflash_draft:
            return self.run_speculative_dflash(seqs)
        K = self.num_spec_tokens
        bs = len(seqs)
        block_size = self.block_size

        temps = torch.tensor([s.temperature for s in seqs], dtype=torch.float32,
                             pin_memory=True).cuda(non_blocking=True)
        nc = torch.tensor([s.num_cached_tokens for s in seqs], dtype=torch.int64,
                          pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Slot table [bs, K+1] for positions C .. C+K, built once on CPU:
        # the first K columns cover the draft rows, the full table covers
        # the verify rows.  Matches the standard decode accounting — the
        # input token of a decode step lives at position num_cached (the
        # next free KV slot) and reads seqlen_k = num_cached + 1 rows.
        slot_rows = []
        for seq in seqs:
            c = seq.num_cached_tokens
            slot_rows.append([
                seq.block_table[p // block_size] * block_size + p % block_size
                for p in range(c, c + K + 1)
            ])
        slot_table = torch.tensor(slot_rows, dtype=torch.int32,
                                  pin_memory=True).cuda(non_blocking=True)

        # ── Draft phase ──
        last_tok = torch.tensor([s.last_token for s in seqs], dtype=torch.int64,
                                pin_memory=True).cuda(non_blocking=True)
        if self.draft_model is not None:
            # Classic speculative decoding: K sequential autoregressive
            # drafts on the small model (each with its own KV written to
            # the draft cache), sampling d_i and saving the exact draft
            # distribution p_i for the rejection step.
            draft_slot_rows = []
            for seq in seqs:
                c = seq.num_cached_tokens
                draft_slot_rows.append([
                    seq.draft_block_table[p // block_size] * block_size + p % block_size
                    for p in range(c, c + K)
                ])
            draft_slot_table = torch.tensor(draft_slot_rows, dtype=torch.int32,
                                            pin_memory=True).cuda(non_blocking=True)
            draft_bt = self.prepare_block_tables(seqs, "draft_block_table")
            draft_tokens = torch.empty(bs, K, dtype=torch.int64, device="cuda")
            vocab_size = self.config.hf_config.vocab_size
            draft_probs = torch.empty(bs, K, vocab_size, dtype=torch.float32, device="cuda")
            cu_q_d = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
            use_graph = not self.enforce_eager and bs <= 512
            d_prev = last_tok
            for i in range(1, K + 1):
                positions = nc + i - 1                   # draft input position C+i-1
                cu_k_d = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
                cu_k_d[1:] = torch.cumsum((nc + i).to(torch.int32), 0)
                # Context is set in both branches: the eager forward reads
                # it inside the layers, and the out-of-graph compute_logits
                # (last-row gather) needs cu_seqlens_q either way.
                set_context(cu_q_d, cu_k_d, 1, int(nc.max().item()) + i,
                            None, None, 0, None, None,
                            draft_slot_mapping=draft_slot_table[:, i - 1],
                            draft_block_tables=draft_bt)
                if use_graph:
                    graph = self.draft_graphs[next(x for x in self.graph_bs if x >= bs)]
                    gv = self.draft_graph_vars
                    gv["input_ids"][:bs] = d_prev
                    gv["positions"][:bs] = positions
                    gv["slot_mapping"].fill_(-1)
                    gv["slot_mapping"][:bs] = draft_slot_table[:, i - 1]
                    gv["cu_seqlens_q"].zero_()
                    gv["cu_seqlens_q"][:bs + 1] = cu_q_d
                    gv["cu_seqlens_k"].zero_()
                    gv["cu_seqlens_k"][:bs + 1] = cu_k_d
                    gv["block_tables"][:bs, :draft_bt.size(1)] = draft_bt
                    graph.replay()
                    hidden = gv["outputs"][:bs]
                else:
                    hidden = self.draft_model(d_prev, positions)
                d_i, p_i = self.sampler.sample_with_probs(
                    self.draft_model.compute_logits(hidden), temps)
                draft_tokens[:, i - 1] = d_i
                draft_probs[:, i - 1] = p_i
                d_prev = d_i
        else:
            # Jacobi-style parallel draft: one K-row forward of the full
            # model (all rows fed the last accepted token).
            draft_ids = last_tok.unsqueeze(1).expand(bs, K).reshape(-1)  # [bs*K]
            draft_pos = nc.unsqueeze(1) + torch.arange(
                K, dtype=torch.int64, device="cuda").unsqueeze(0)
            draft_pos = draft_pos.reshape(-1)
            cu_q_d = torch.arange(0, bs * K + 1, K, dtype=torch.int32, device="cuda")
            cu_k_d = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            cu_k_d[1:] = torch.cumsum((nc + K).to(torch.int32), 0)
            set_context(cu_q_d, cu_k_d, K, int(nc.max().item()) + K,
                        slot_table[:, :K].reshape(-1), block_tables)
            hidden = self.model(draft_ids, draft_pos)
            d_tokens, d_probs = self.sampler.sample_with_probs(
                self.model.compute_all_logits(hidden),
                temps.repeat_interleave(K))   # [bs*K]: one temp per draft row
            draft_tokens = d_tokens.view(bs, K)
            draft_probs = d_probs.view(bs, K, -1)

        # ── Verify phase: one (K+1)-row forward of the full model ──
        verify_ids = torch.cat([last_tok.unsqueeze(1), draft_tokens], dim=1).reshape(-1)
        verify_pos = nc.unsqueeze(1) + torch.arange(
            K + 1, dtype=torch.int64, device="cuda").unsqueeze(0)
        verify_pos = verify_pos.reshape(-1)
        cu_q_v = torch.arange(0, bs * (K + 1) + 1, K + 1,
                              dtype=torch.int32, device="cuda")
        cu_k_v = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        cu_k_v[1:] = torch.cumsum((nc + K + 1).to(torch.int32), 0)
        set_context(cu_q_v, cu_k_v, K + 1, int(nc.max().item()) + K + 1,
                    slot_table.reshape(-1), block_tables)
        hidden = self.model(verify_ids, verify_pos)
        logits_all = self.model.compute_all_logits(hidden)   # [bs*(K+1), V]
        reset_context()

        return self._accept_speculative(logits_all, draft_tokens, draft_probs, temps)

    @torch.inference_mode()
    def run_speculative_dflash(self, seqs: list[Sequence]) -> list[list[int]]:
        """DFlash block-diffusion draft: one block_size-row non-causal
        forward of the 5-layer draft proposes block_size-1 tokens, verified
        by one block_size-row target forward (standard rejection sampling,
        K = block_size - 1)."""
        draft = self.draft_model
        T = draft.block_size
        K = T - 1
        bs = len(seqs)
        block_size = self.block_size
        ctx_dim = len(draft.target_layer_ids) * self.config.hf_config.hidden_size

        temps = torch.tensor([s.temperature for s in seqs], dtype=torch.float32,
                             pin_memory=True).cuda(non_blocking=True)
        nc = torch.tensor([s.num_cached_tokens for s in seqs], dtype=torch.int64,
                          pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Verify slot table [bs, T] for positions C..C+T-1 (standard
        # decode accounting: the anchor lives at the next free slot C).
        slot_rows = []
        for seq in seqs:
            c = seq.num_cached_tokens
            slot_rows.append([
                seq.block_table[p // block_size] * block_size + p % block_size
                for p in range(c, c + T)
            ])
        slot_table = torch.tensor(slot_rows, dtype=torch.int32,
                                  pin_memory=True).cuda(non_blocking=True)

        # ── Block: anchor + K mask tokens per seq ──
        last_tok = torch.tensor([s.last_token for s in seqs], dtype=torch.int64,
                                pin_memory=True).cuda(non_blocking=True)
        block_ids = torch.cat([
            last_tok.unsqueeze(1),
            torch.full((bs, K), draft.mask_token_id, dtype=torch.int64, device="cuda"),
        ], dim=1)                                          # [bs, T]

        # ── Draft context (padded to C_max) ──
        ctx_list = [self._dflash_ctx.get(s.seq_id) for s in seqs]
        ctx_lens = torch.tensor([0 if c is None else c.shape[0] for c in ctx_list],
                                dtype=torch.int64, device="cuda")
        C_max = int(ctx_lens.max().item())
        ctx_pad = torch.zeros(bs, C_max, ctx_dim, dtype=torch.bfloat16, device="cuda")
        for b, c in enumerate(ctx_list):
            if c is not None:
                ctx_pad[b, :c.shape[0]] = c

        # ── Draft forward: one parallel block per seq ──
        noise_emb = self.model.model.embed_tokens(block_ids.reshape(-1))  # [bs*T, H]
        q_pos = nc.unsqueeze(1) + torch.arange(T, dtype=torch.int64, device="cuda")
        # Key positions: context rows keep their true history positions
        # [start-C, start), the block rows cover [start, start+T).  Pad
        # rows get arbitrary positions — masked out below.
        k_pos = (nc - ctx_lens).unsqueeze(1) + torch.arange(
            C_max + T, dtype=torch.int64, device="cuda").unsqueeze(0)
        attn_mask = torch.zeros(bs, 1, T, C_max + T, dtype=torch.bool, device="cuda")
        for b in range(bs):
            attn_mask[b, 0, :, :int(ctx_lens[b].item()) + T] = True
        hidden = draft(noise_emb, ctx_pad, ctx_lens, q_pos, k_pos, attn_mask)

        # ── Draft sampling: the last K rows propose the block's tokens ──
        d_logits = self.model.lm_head.forward_all(
            hidden[:, -K:, :].reshape(-1, hidden.shape[-1]))
        d_tokens, d_probs = self.sampler.sample_with_probs(
            d_logits, temps.repeat_interleave(K))
        draft_tokens = d_tokens.view(bs, K)
        draft_probs = d_probs.view(bs, K, -1)

        # ── Verify: one T-row target forward + context extraction ──
        verify_ids = torch.cat([last_tok.unsqueeze(1), draft_tokens], dim=1).reshape(-1)
        verify_pos = (nc.unsqueeze(1) + torch.arange(
            T, dtype=torch.int64, device="cuda").unsqueeze(0)).reshape(-1)
        cu_q_v = torch.arange(0, bs * T + 1, T, dtype=torch.int32, device="cuda")
        cu_k_v = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        cu_k_v[1:] = torch.cumsum((nc + T).to(torch.int32), 0)
        set_context(cu_q_v, cu_k_v, T, int(nc.max().item()) + T,
                    slot_table.reshape(-1), block_tables)
        hidden_v, ctx_cat = self.model(
            verify_ids, verify_pos,
            output_layer_hidden=draft.target_layer_ids)
        logits_all = self.model.compute_all_logits(hidden_v)   # [bs*T, V]
        reset_context()

        accepted = self._accept_speculative(logits_all, draft_tokens, draft_probs, temps)

        # ── Update the per-seq draft context: the accepted rows' hidden
        # features (n == T includes the bonus token; the context keeps the
        # T verify rows, matching the official hidden[:acc+1] semantics) ──
        ctx_view = ctx_cat.view(bs, T, -1)
        for b, s in enumerate(seqs):
            n = len(accepted[b])
            if n > 0:
                self._dflash_ctx[s.seq_id] = ctx_view[b, :n].clone()
        return accepted

    @torch.inference_mode()
    def _accept_speculative(self, logits_all, draft_tokens, draft_probs, temps):
        """Per-position rejection sampling (GPU-vectorized).

        Rows align as: draft row i (distribution p_i) and target row i-1
        (distribution q_{i-1}) describe the same position C+i-1; position 0
        (t_0) is always accepted.  On a match-with-reject, resample from
        (q-p)^+ so the marginal distribution of every accepted position
        stays exactly q (strict rejection sampling).
        """
        K = self.num_spec_tokens
        bs = draft_tokens.size(0)
        logits = logits_all.float().view(bs, K + 1, -1)
        logits.div_(temps.unsqueeze(1).unsqueeze(2))
        target_probs = torch.softmax(logits, dim=-1)             # [bs, K+1, V]
        target_tokens = target_probs.div(
            torch.empty_like(target_probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)                                          # [bs, K+1]

        # Accept d_i (i=1..K) iff target and draft agree and U < q/p.
        q_g = target_probs[:, :K].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        p_g = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        u = torch.rand(bs, K, device="cuda")
        eq = (target_tokens[:, :K] == draft_tokens)
        acc = eq & (u < (q_g / p_g.clamp_min(1e-10)).clamp_max(1.0))
        n = 1 + acc.cumprod(1).sum(1)                            # [bs], in [1, K+1]

        # First-reject position: mismatch -> take the target token; match
        # but rejected -> resample from (q - p)^+.
        i_star = (n - 1).clamp(max=K - 1)
        rows = torch.arange(bs, device="cuda")
        q_rej = target_probs[rows, i_star]                       # [bs, V]
        p_rej = draft_probs[rows, i_star]                        # [bs, V]
        adjusted = (q_rej - p_rej).clamp_min(0.0)
        f_adj = adjusted.div(
            torch.empty_like(adjusted).exponential_(1).clamp_min_(1e-10)
        ).argmax(-1)
        f = torch.where(eq[rows, i_star], f_adj,
                        target_tokens[rows, i_star])

        # Assemble per-seq accepted token lists (CPU).
        n_list = n.tolist()
        f_list = f.tolist()
        draft_list = draft_tokens.tolist()
        bonus = target_tokens[:, K].tolist()
        accepted = []
        for b in range(bs):
            nb = n_list[b]
            if nb == K + 1:
                accepted.append(draft_list[b] + [bonus[b]])
            else:
                accepted.append(draft_list[b][:nb - 1] + [f_list[b]])
        return accepted

    # ------------------------------------------------------------------
    # Async scheduling: three-phase execution
    # ------------------------------------------------------------------

    # Pending step state (stored between phases).
    _async_seqs: list[Sequence] | None = None
    _async_input_ids: torch.Tensor | None = None
    _async_positions: torch.Tensor | None = None
    _async_is_uniform: bool = False
    _async_sample_params: tuple | torch.Tensor | None = None
    _async_logits: torch.Tensor | None = None

    def prepare_step(self, seqs: list[Sequence]):
        """Phase 1: prepare metadata tensors and set global context."""
        if not seqs:
            return  # starved schedule — skip this step
        input_ids, positions, is_uniform = self.prepare_batch(seqs)
        self._async_input_ids = input_ids
        self._async_positions = positions
        self._async_is_uniform = is_uniform
        self._async_seqs = seqs
        self._async_sample_params = self.prepare_sample(seqs) if self.rank == 0 else None

    def execute_model(self):
        """Phase 2: launch forward pass (non-blocking on GPU)."""
        self._async_logits = self.run_model(
            self._async_input_ids, self._async_positions, self._async_is_uniform
        )

    def sample(self) -> list[int] | None:
        """Phase 3: synchronise GPU and sample tokens."""
        if self.rank == 0:
            token_ids = self.sampler(self._async_logits, *self._async_sample_params).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    def free_finished_gpu_rows(self, seqs: list[Sequence]):
        """Release GPU table rows for finished sequences."""
        for seq in seqs:
            if seq.is_finished:
                if self.use_dflash_draft:
                    self._dflash_ctx.pop(seq.seq_id, None)
                if self.gpu_state is not None and hasattr(seq, '_gpu_row'):
                    self.gpu_state.free_row(seq._gpu_row)
                    del seq._gpu_row

    # ------------------------------------------------------------------
    # CUDA graph capture (uniform-decode only)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture_cudagraph(self, model=None, skip_layers: tuple[int, ...] | None = None,
                          is_draft: bool = False):
        """Capture per-bs decode graphs for *model* (default: target model).

        ``is_draft=True`` captures the small draft model: its layers read
        the draft slot/block-table context fields, so those buffers are
        bound into the graph instead of the target ones.  Returns
        (graphs, graph_vars); the first (non-draft) capture creates the
        shared memory pool.
        """
        model = model or self.model
        config = self.config
        hf_config = config.draft_hf_config if is_draft else config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # Linear-attention state slot ids (hybrid models); the recurrent
        # update inside the graph reads/writes fixed pool addresses, only
        # this id buffer's contents change between replays.
        linear_state_ids = (torch.zeros(max_bs, dtype=torch.int64)
                            if self.linear_s_pool is not None else None)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # The 16-step sequence ends at the last multiple of 16 <= max_bs;
        # cover the remainder (e.g. max_num_seqs clamped to 218 by the
        # state pool) so lookups of bs in (208, 218] find a graph.
        if self.graph_bs[-1] < max_bs:
            self.graph_bs.append(max_bs)
        graphs = {}
        if not is_draft:
            # First graph set creates the memory pool; the draft set
            # (skip_layers) reuses it.
            self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            cu_seqlens_q[:bs + 1] = torch.arange(bs + 1, dtype=torch.int32)
            cu_seqlens_k[:bs + 1] = context_k = torch.arange(1, bs + 2, dtype=torch.int32)
            # 图内一律用 flash_attn 内核: FlashInfer 的 decode 内核在
            # cudagraph + 可变元数据组合下不可靠 (经典内核有数值误差,
            # tensor-core 内核长 KV 崩溃), 见 git log/commit 讨论。
            if is_draft:
                set_context(cu_seqlens_q[:bs + 1], cu_seqlens_k[:bs + 1],
                            1, bs + 1, None, None, 0, None, None,
                            draft_slot_mapping=slot_mapping[:bs],
                            draft_block_tables=block_tables[:bs])
            else:
                set_context(cu_seqlens_q[:bs + 1], cu_seqlens_k[:bs + 1],
                            1, bs + 1,
                            slot_mapping[:bs], block_tables[:bs],
                            linear_state_ids=(linear_state_ids[:bs]
                                              if linear_state_ids is not None else None))
            outputs[:bs] = model(input_ids[:bs], positions[:bs],
                                 skip_layers=skip_layers)  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = model(input_ids[:bs], positions[:bs],
                                     skip_layers=skip_layers)  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
        )
        if linear_state_ids is not None:
            graph_vars["linear_state_ids"] = linear_state_ids
        return graphs, graph_vars
