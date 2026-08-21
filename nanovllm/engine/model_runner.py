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
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
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

        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
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
        if not self.enforce_eager:
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

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
            seq.is_prefill = True
        self.run(seqs)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

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

        set_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, block_tables)
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
                seqlen_k = seq.num_computed_tokens
                input_ids.append(seq.last_token)
                positions.append(seq.num_computed_tokens - 1)

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
        set_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, block_tables)
        return input_ids, positions, is_uniform_decode

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  is_uniform_decode: bool):
        if not is_uniform_decode or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
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
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions, is_uniform = self.prepare_batch(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_uniform)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ------------------------------------------------------------------
    # Async scheduling: three-phase execution
    # ------------------------------------------------------------------

    # Pending step state (stored between phases).
    _async_seqs: list[Sequence] | None = None
    _async_input_ids: torch.Tensor | None = None
    _async_positions: torch.Tensor | None = None
    _async_is_uniform: bool = False
    _async_temperatures: torch.Tensor | None = None
    _async_logits: torch.Tensor | None = None

    def prepare_step(self, seqs: list[Sequence]):
        """Phase 1: prepare metadata tensors and set global context."""
        input_ids, positions, is_uniform = self.prepare_batch(seqs)
        self._async_input_ids = input_ids
        self._async_positions = positions
        self._async_is_uniform = is_uniform
        self._async_seqs = seqs
        self._async_temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

    def execute_model(self):
        """Phase 2: launch forward pass (non-blocking on GPU)."""
        self._async_logits = self.run_model(
            self._async_input_ids, self._async_positions, self._async_is_uniform
        )

    def sample(self) -> list[int] | None:
        """Phase 3: synchronise GPU and sample tokens."""
        if self.rank == 0:
            token_ids = self.sampler(self._async_logits, self._async_temperatures).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    def free_finished_gpu_rows(self, seqs: list[Sequence]):
        """Release GPU table rows for finished sequences."""
        if self.gpu_state is None:
            return
        for seq in seqs:
            if seq.is_finished and hasattr(seq, '_gpu_row'):
                self.gpu_state.free_row(seq._gpu_row)
                del seq._gpu_row

    # ------------------------------------------------------------------
    # CUDA graph capture (uniform-decode only)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            cu_seqlens_q[:bs + 1] = torch.arange(bs + 1, dtype=torch.int32)
            cu_seqlens_k[:bs + 1] = context_k = torch.arange(1, bs + 2, dtype=torch.int32)
            set_context(cu_seqlens_q[:bs + 1], cu_seqlens_k[:bs + 1],
                        1, bs + 1,
                        slot_mapping[:bs], block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
        )
