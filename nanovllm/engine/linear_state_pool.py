"""CPU-side free list for the linear-attention recurrent-state slots.

The GPU pools live in ModelRunner (s_pool / conv_pool); slot ids are the
row indices of those pools, so this class only tracks which rows are in
use.  Slots are assigned on first scheduling and freed when a sequence
finishes.  Preemption keeps the slot: a preempted sequence restarts as a
fresh prefill, whose state is zeroed before the recomputed forward runs.
"""
from collections import deque


class LinearStatePool:

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.free_slots: deque[int] = deque(range(num_slots))

    def alloc(self) -> int:
        assert self.free_slots, "no free linear-state slots"
        return self.free_slots.popleft()

    def free(self, slot: int):
        self.free_slots.append(slot)

    @property
    def num_free(self) -> int:
        return len(self.free_slots)
