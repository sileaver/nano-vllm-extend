"""GPU-resident persistent state table.

Mirrors the CPU-side ``Sequence`` state on the GPU so that input
preparation can be done entirely on-device via Triton kernels,
removing Python loops and H2D transfers from the critical path.

This follows MRV2's decoupled-persistent-batch design: each request
gets a fixed row index, and per-step inputs are *gathered* from the
table rather than rebuilt from scratch on CPU.
"""

import torch


class GpuStateTable:
    """Fixed-size GPU table holding per-request state.

    Each active request occupies one row.  Rows are recycled when
    requests finish.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks: int,
        block_size: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size

        # ── Per-request state tensors (all on GPU) ────────────
        # token_ids:   [max_reqs, max_model_len]   full token sequence
        # num_tokens:  [max_reqs]                  len(token_ids)
        # num_cached:  [max_reqs]                  confirmed KV slots
        # num_computed:[max_reqs]                  cached + placeholders
        # block_table: [max_reqs, max_blocks]      -1 = unused slot
        # num_blocks:  [max_reqs]                  allocated blocks
        # is_prefill:  [max_reqs]   bool           prefill vs decode
        # sched_tokens:[max_reqs]                  tokens scheduled this step

        self.token_ids = torch.zeros(
            max_num_reqs, max_model_len, dtype=torch.int64, device="cuda"
        )
        self.num_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
        self.num_cached = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
        self.num_computed = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
        self.block_table = torch.full(
            (max_num_reqs, max_num_blocks), -1, dtype=torch.int32, device="cuda"
        )
        self.num_blocks = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")
        self.is_prefill = torch.zeros(max_num_reqs, dtype=torch.bool, device="cuda")
        self.sched_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device="cuda")

        # Free-row pool.
        self._free_rows = list(range(max_num_reqs))

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def alloc_row(self) -> int:
        """Reserve a row index for a new request."""
        if not self._free_rows:
            raise RuntimeError("GpuStateTable out of rows — increase max_num_reqs")
        return self._free_rows.pop()

    def free_row(self, row: int):
        """Release a row when a request finishes."""
        # Zero out the row so stale data doesn't leak.
        self.token_ids[row, :] = 0
        self.num_tokens[row] = 0
        self.num_cached[row] = 0
        self.num_computed[row] = 0
        self.block_table[row, :] = -1
        self.num_blocks[row] = 0
        self.is_prefill[row] = False
        self.sched_tokens[row] = 0
        self._free_rows.append(row)

    # ------------------------------------------------------------------
    # Sync from CPU → GPU
    # ------------------------------------------------------------------

    def add_request(self, seq, row: int):
        """Copy initial request state to GPU (called once per request)."""
        n = min(len(seq.token_ids), self.max_model_len)
        self.token_ids[row, :n] = torch.tensor(
            seq.token_ids[:n], dtype=torch.int64, device="cuda"
        )
        self.num_tokens[row] = n
        self.num_cached[row] = seq.num_cached_tokens
        self.num_computed[row] = seq.num_computed_tokens
        self.num_blocks[row] = len(seq.block_table)
        if seq.block_table:
            self.block_table[row, :len(seq.block_table)] = torch.tensor(
                seq.block_table, dtype=torch.int32, device="cuda"
            )
        self.is_prefill[row] = seq.is_prefill
        self.sched_tokens[row] = seq.num_scheduled_tokens

        # Store row index on the CPU sequence for later sync.
        seq._gpu_row = row

    def update(self, seq):
        """Incremental sync — only fields that changed since last step."""
        row = seq._gpu_row
        self.num_tokens[row] = seq.num_tokens
        self.num_cached[row] = seq.num_cached_tokens
        self.num_computed[row] = seq.num_computed_tokens
        self.is_prefill[row] = seq.is_prefill
        self.sched_tokens[row] = seq.num_scheduled_tokens

        # Block table changes less frequently — only sync when it changed.
        if len(seq.block_table) != int(self.num_blocks[row].item()):
            self.num_blocks[row] = len(seq.block_table)
            if seq.block_table:
                self.block_table[row, :len(seq.block_table)] = torch.tensor(
                    seq.block_table, dtype=torch.int32, device="cuda"
                )

    def update_token_ids(self, seq):
        """Sync full token_ids (only needed for prefill chunks)."""
        row = seq._gpu_row
        n = min(len(seq.token_ids), self.max_model_len)
        self.token_ids[row, :n] = torch.tensor(
            seq.token_ids[:n], dtype=torch.int64, device="cuda"
        )
        self.num_tokens[row] = n
