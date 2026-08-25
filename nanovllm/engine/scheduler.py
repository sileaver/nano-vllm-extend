from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.linear_state_pool import LinearStatePool


@dataclass
class SchedulerOutput:
    """Result of one scheduling step.

    ``scheduled_seqs`` may contain a mix of prefill chunks and decode
    tokens — this is the essence of *continuous batching*.
    """
    scheduled_seqs: list[Sequence]
    num_scheduled_tokens: int   # total tokens across all seqs this step
    is_prefill: bool            # True if ANY seq is a prefill chunk
    is_speculative: bool = False  # pure-decode step with speculative decoding


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # Second KV space for the small draft model (spec_draft_model).
        # num_draft_kvcache_blocks is computed by ModelRunner.__init__
        # (which runs before the scheduler is constructed).
        self.draft_block_manager = (
            BlockManager(config.num_draft_kvcache_blocks, config.kvcache_block_size,
                         block_table_attr="draft_block_table")
            if config.num_draft_kvcache_blocks > 0 else None)
        # Hybrid models: linear-attention recurrent-state slots.  Kept for
        # the whole sequence lifetime (preemption keeps the slot — the
        # recomputed prefill re-zeroes it via the fresh-prefill path).
        self.state_pool = (
            LinearStatePool(config.num_linear_state_slots)
            if config.num_linear_state_slots > 0 else None)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.continuous_batching = config.continuous_batching
        self.num_spec_tokens = config.num_spec_tokens
        # Speculative acceptance statistics.
        self.spec_accepted = 0
        self.spec_attempted = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # Length-biased insertion (longest-first, stable): the hybrid
        # model's linear-attention layers pad each batch to [bs, T=max
        # len], so grouping similar lengths keeps the padded fill ratio
        # near 1.0 (random mixed-length batches average ~0.6, wasting GEMM
        # + GDN compute on padding).  Equal lengths keep arrival order.
        n = seq.num_tokens
        for i in range(len(self.waiting)):
            if self.waiting[i].num_tokens < n:
                self.waiting.insert(i, seq)
                return
        self.waiting.append(seq)

    # ------------------------------------------------------------------
    # Dual-KV helpers (target + optional draft model)
    # ------------------------------------------------------------------

    def _can_append_all(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        if not self.block_manager.can_append(seq, num_new_tokens):
            return False
        if self.draft_block_manager is not None \
                and not self.draft_block_manager.can_append(seq, num_new_tokens):
            return False
        return True

    def _ensure_append_all(self, seq: Sequence, num_new_tokens: int = 1):
        self.block_manager.ensure_append(seq, num_new_tokens)
        if self.draft_block_manager is not None:
            self.draft_block_manager.ensure_append(seq, num_new_tokens)

    def _alloc_state_slot(self, seq: Sequence):
        if self.state_pool is not None and seq.linear_state_id == -1:
            seq.linear_state_id = self.state_pool.alloc()

    def _free_state_slot(self, seq: Sequence):
        if self.state_pool is not None and seq.linear_state_id != -1:
            self.state_pool.free(seq.linear_state_id)
            seq.linear_state_id = -1

    # ------------------------------------------------------------------
    # Continuous batching (V1 style)
    # ------------------------------------------------------------------

    def _schedule_continuous(self) -> tuple[list[Sequence], int, bool]:
        """Decode-first schedule with token-budget enforcement.

        1. Schedule *all* eligible decode requests (latency-sensitive).
        2. Fill the remaining token budget with prefill chunks from the
           waiting queue.

        Returns (seqs, num_tokens, has_prefill).
        """
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0
        # Speculative decode steps schedule K+1 verified positions per seq.
        ntok = self.num_spec_tokens + 1 if self.num_spec_tokens else 1

        # ── Phase 1: decode ──────────────────────────────────────
        for seq in list(self.running):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if num_batched_tokens + ntok > self.max_num_batched_tokens:
                break

            while not self._can_append_all(seq, ntok):
                if not self.running:
                    self.preempt(seq)
                    break
                self.preempt(self.running[-1])
                if self.running[-1] is seq:
                    break
            else:
                seq.num_scheduled_tokens = ntok
                seq.is_prefill = False
                self._ensure_append_all(seq, ntok)
                num_batched_tokens += ntok
                scheduled_seqs.append(seq)

        # ── Phase 2: prefill chunks ──────────────────────────────
        # Cap the total active (running) sequences at max_num_seqs: without
        # this the prefill loop admits every waiting request and running
        # outgrows the per-step scheduling cap, starving the tail.
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs \
                and len(self.running) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            seq = self.waiting[0]

            if not seq.block_table:
                # First chunk — check prefix-cache and allocate blocks.
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
                self._alloc_state_slot(seq)
                if self.draft_block_manager is not None:
                    # Draft KV gets a plain (non-hashed) allocation.
                    self.draft_block_manager.allocate(seq, 0)
                    seq.num_cached_tokens = num_cached_blocks * self.block_size
            else:
                # num_computed (placeholder-inclusive) covers chunks that
                # are still in flight under async scheduling, so a
                # continuation chunk never re-executes its predecessor.
                num_tokens = seq.num_tokens - seq.num_computed_tokens

            # Allow chunked prefill for *any* sequence in the continuous
            # path (not just the first).  This is what lets new short
            # prompts squeeze into the remaining budget.
            chunk = min(num_tokens, remaining)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            num_batched_tokens += chunk
            scheduled_seqs.append(seq)

            if seq.num_computed_tokens + chunk == seq.num_tokens:
                # Full prompt consumed → promote to running.
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

        has_prefill = any(s.is_prefill for s in scheduled_seqs)
        return scheduled_seqs, num_batched_tokens, has_prefill

    # ------------------------------------------------------------------
    # Legacy V0-style schedule (prefill *or* decode — never both)
    # ------------------------------------------------------------------

    def _schedule_legacy(self) -> tuple[list[Sequence], int, bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # (running-count cap: see the note in _schedule_continuous — running
        # beyond max_num_seqs starves the queue tail under this scheduler)
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs \
                and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                self._alloc_state_slot(seq)
                if self.draft_block_manager is not None:
                    self.draft_block_manager.allocate(seq, 0)
                    seq.num_cached_tokens = num_cached_blocks * self.block_size
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, num_batched_tokens, True

        # decode
        ntok = self.num_spec_tokens + 1 if self.num_spec_tokens else 1
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self._can_append_all(seq, ntok):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = ntok
                seq.is_prefill = False
                self._ensure_append_all(seq, ntok)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            # Fully drained mid-step: with async scheduling every running
            # sequence can finish in steps reaped at the top of _step_async
            # (after the engine's is_finished() check), leaving both queues
            # empty here.  Also covers a starved prefill loop (waiting head
            # cannot allocate).  Return an empty schedule — the engine
            # treats it as a no-op step.
            return scheduled_seqs, 0, False
        # Round-robin: re-queue at the TAIL.  extendleft(reversed(...)) puts
        # the scheduled batch back at the head, so whenever running exceeds
        # max_num_seqs the queue tail is never scheduled (starvation → the
        # engine never finishes).
        self.running.extend(scheduled_seqs)
        return scheduled_seqs, -len(scheduled_seqs), False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        if self.continuous_batching:
            seqs, num_tokens, has_prefill = self._schedule_continuous()
        else:
            seqs, num_tokens, has_prefill = self._schedule_legacy()
        is_speculative = bool(self.num_spec_tokens) and not has_prefill and bool(seqs)
        if self.num_spec_tokens and has_prefill:
            # Mixed batch: decode seqs were scheduled with K+1 tokens for
            # speculative execution, but a prefill chunk is present so the
            # standard path runs instead — reset to 1 or postprocess would
            # advance num_cached_tokens by K+1 while appending one token.
            for seq in seqs:
                if not seq.is_prefill:
                    seq.num_scheduled_tokens = 1
        return SchedulerOutput(seqs, num_tokens, has_prefill, is_speculative)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.draft_block_manager is not None:
            self.draft_block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Use per-sequence flag — in continuous batching the batch may
            # mix prefill chunks and decode tokens.
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.draft_block_manager is not None:
                    self.draft_block_manager.deallocate(seq)
                self._free_state_slot(seq)
                self.running.remove(seq)

    def postprocess_speculative(
        self,
        seqs: list[Sequence],
        accepted: list[list[int]],
    ) -> int:
        """Apply accepted tokens from a speculative step (variable length
        per seq).  Returns the total number of appended tokens.
        """
        total = 0
        for seq, tokens in zip(seqs, accepted):
            appended = 0
            finished = False
            for token_id in tokens:
                seq.append_token(token_id)
                appended += 1
                if (not seq.ignore_eos and token_id == self.eos) \
                        or seq.num_completion_tokens == seq.max_tokens:
                    finished = True
                    break
            # Hash and advance KV by the *actual* accepted count (may be
            # cut short by eos / max_tokens).  Must run after append_token
            # (hash reads token_ids) and before num_cached_tokens advances
            # (hash_blocks derives the block range from the old value).
            self.block_manager.hash_blocks(seq, appended)
            seq.num_cached_tokens += appended
            seq.num_scheduled_tokens = 0
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.draft_block_manager is not None:
                    self.draft_block_manager.deallocate(seq)
                self._free_state_slot(seq)
                self.running.remove(seq)
            # Acceptance stats: position 0 (t_0) is always accepted and not
            # a draft token — count draft acceptances only (appended - 1),
            # against the K draft tokens this step attempted.
            self.spec_accepted += appended - 1
            self.spec_attempted += self.num_spec_tokens
            total += appended
        return total

    def spec_stats(self) -> dict:
        """Acceptance statistics for the current speculative run."""
        return {
            "accepted": self.spec_accepted,
            "attempted": self.spec_attempted,
            "accept_rate": self.spec_accepted / self.spec_attempted
            if self.spec_attempted else 0.0,
        }
