from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass
class SchedulerOutput:
    """Result of one scheduling step.

    ``scheduled_seqs`` may contain a mix of prefill chunks and decode
    tokens — this is the essence of *continuous batching*.
    """
    scheduled_seqs: list[Sequence]
    num_scheduled_tokens: int   # total tokens across all seqs this step
    is_prefill: bool            # True if ANY seq is a prefill chunk


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.continuous_batching = config.continuous_batching

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

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

        # ── Phase 1: decode ──────────────────────────────────────
        for seq in list(self.running):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if num_batched_tokens + 1 > self.max_num_batched_tokens:
                break

            while not self.block_manager.can_append(seq):
                if not self.running:
                    self.preempt(seq)
                    break
                self.preempt(self.running[-1])
                if self.running[-1] is seq:
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                num_batched_tokens += 1
                scheduled_seqs.append(seq)

        # ── Phase 2: prefill chunks ──────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
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
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Allow chunked prefill for *any* sequence in the continuous
            # path (not just the first).  This is what lets new short
            # prompts squeeze into the remaining budget.
            chunk = min(num_tokens, remaining)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            num_batched_tokens += chunk
            scheduled_seqs.append(seq)

            if seq.num_cached_tokens + chunk == seq.num_tokens:
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
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
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
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, -len(scheduled_seqs), False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        if self.continuous_batching:
            seqs, num_tokens, has_prefill = self._schedule_continuous()
        else:
            seqs, num_tokens, has_prefill = self._schedule_legacy()
        return SchedulerOutput(seqs, num_tokens, has_prefill)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
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
                self.running.remove(seq)
