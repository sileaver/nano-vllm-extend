import time
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(),
                 pixel_values=None, image_grid_thw: list[list[int]] | None = None,
                 mrope_positions=None, rope_delta: int = 0):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.num_output_placeholders = 0
        self.is_prefill = True
        self.block_table = []
        self.draft_block_table: list[int] = []
        # Hybrid models: slot into the linear-attention recurrent-state pool
        # (-1 = unallocated; assigned by the scheduler on first scheduling).
        self.linear_state_id = -1
        # Multimodal payload (Qwen3.5-style): processor-emitted pixel rows
        # [n_patches, C*T*P*P] (CPU float) + per-image patch grids [t, h, w];
        # precomputed MRoPE positions [3, num_prompt_tokens] for prefill
        # chunks and the decode-time scalar offset (the port of
        # Qwen3_5Model.get_rope_index lives in nanovllm/utils/multimodal.py).
        # Text-only sequences keep everything at None/0 — positions stay a
        # plain arange exactly as before.
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.mrope_positions = mrope_positions
        self.rope_delta = rope_delta
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # Timing (opt-in, set by LLMEngine when collect_timing=True).
        self.timing = False
        self.arrival_time: float = 0.0
        self.first_token_time: float | None = None
        self.token_times: list[float] = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_computed_tokens(self):
        return self.num_cached_tokens + self.num_output_placeholders

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        if self.timing:
            now = time.time()
            self.token_times.append(now)
            if self.first_token_time is None:
                self.first_token_time = now

    def __getstate__(self):
        # Prefill seqs ship their full token list (workers re-run the
        # scheduled chunk from it); decode seqs only need the last token.
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # Multimodal: pixels and the [3, len] MRoPE table only travel with
        # prefill states (a decode step re-derives positions from
        # rope_delta); a preempted seq flips back to is_prefill and ships
        # them again.
        pixel_values = self.pixel_values if self.is_prefill else None
        mrope_positions = self.mrope_positions if self.is_prefill else None
        return (self.seq_id, self.status, self.is_prefill,
                self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.num_output_placeholders,
                self.block_table, self.draft_block_table, self.linear_state_id,
                self.temperature, self.top_k, self.top_p,
                self.max_tokens, self.ignore_eos, last_state,
                pixel_values, self.image_grid_thw, mrope_positions,
                self.rope_delta)

    def __setstate__(self, state):
        (self.seq_id, self.status, self.is_prefill,
         self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.num_output_placeholders,
         self.block_table, self.draft_block_table, self.linear_state_id,
         self.temperature, self.top_k, self.top_p,
         self.max_tokens, self.ignore_eos, last_state,
         pixel_values, image_grid_thw, mrope_positions, rope_delta) = state
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.mrope_positions = mrope_positions
        self.rope_delta = rope_delta
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
        # Attributes the engine touches but the wire format omits.
        self.timing = False
        self.arrival_time = 0.0
        self.first_token_time = None
        self.token_times = []
