from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int,
                 block_table_attr: str = "block_table"):
        self.block_size = block_size
        self.block_table_attr = block_table_attr
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        table = getattr(seq, self.block_table_attr)
        assert not table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        table = getattr(seq, self.block_table_attr)
        for block_id in reversed(table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        table.clear()

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        # ``num_computed_tokens`` is the next free KV-cache position (0-indexed).
        # Speculative steps may need several new positions at once; the block
        # table is a minimal cover of num_computed_tokens, so the missing
        # block count is the cover size of the extended range minus its length.
        table = getattr(seq, self.block_table_attr)
        need = (seq.num_computed_tokens + num_new_tokens - 1) // self.block_size + 1 - len(table)
        return len(self.free_block_ids) >= max(0, need)

    def may_append(self, seq: Sequence):
        if seq.num_computed_tokens % self.block_size == 0:
            getattr(seq, self.block_table_attr).append(self._allocate_block())

    def ensure_append(self, seq: Sequence, num_new_tokens: int = 1):
        """Reserve blocks so the block table covers num_new_tokens new KV
        positions (speculative decode writes K+1 slots per step)."""
        table = getattr(seq, self.block_table_attr)
        need = (seq.num_computed_tokens + num_new_tokens - 1) // self.block_size + 1 - len(table)
        for _ in range(max(0, need)):
            table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_scheduled_tokens: int | None = None):
        if num_scheduled_tokens is None:
            num_scheduled_tokens = seq.num_scheduled_tokens
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + num_scheduled_tokens) // self.block_size
        if start == end: return
        table = getattr(seq, self.block_table_attr)
        h = self.blocks[table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
