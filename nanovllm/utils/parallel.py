"""Parallel-state singleton for TP / PP / DP.

Rank layout (one node): the world of one DP replica is ``pp_size * tp_size``
processes; the global rank of a process within its replica group is

    rank = pp_rank * tp_size + tp_rank

and its GPU is ``dp_idx * (pp_size * tp_size) + rank``.  Each DP replica
forms its own torch.distributed process group (its own TCP store port), so
"global" here means "within the replica group".

Layers and models must never call ``dist.get_rank()``/``get_world_size()``
directly — with PP (or when several engines coexist for DP) those return the
replica-group rank, not the TP position.  Use ``get_tp_rank()`` /
``get_tp_size()`` / ``get_tp_group()`` here instead.
"""
import torch.distributed as dist

# ── Process state (set once by ModelRunner before model construction) ──
_rank = 0            # rank within the replica group
_world_size = 1      # pp_size * tp_size
_tp_rank = 0
_tp_size = 1
_pp_rank = 0
_pp_size = 1
_dp_idx = 0
_initialized = False
_tp_group = None
# Ranks (within the replica group) that hold (last stage, tp rank 0) — the
# sampling rank — and (first stage, tp rank 0) — the driver-side rank 0.
_sampler_rank = 0


def init_parallel_state(rank: int, world_size: int, tp_size: int, pp_size: int,
                        dp_idx: int = 0):
    """Create the TP sub-groups and record this process's coordinates.

    Must be called by EVERY process of the replica group, in the same order
    (``dist.new_group`` is collective).  With pp_size == 1 the TP group is
    the full world — a new group is still created so collectives never fall
    back to the default group (harmless, and keeps one code path).
    """
    global _rank, _world_size, _tp_rank, _tp_size, _pp_rank, _pp_size, _dp_idx
    global _initialized, _tp_group, _sampler_rank
    assert world_size == tp_size * pp_size
    _rank, _world_size = rank, world_size
    _tp_size, _pp_size = tp_size, pp_size
    _pp_rank, _tp_rank = rank // tp_size, rank % tp_size
    _dp_idx = dp_idx
    _sampler_rank = (pp_size - 1) * tp_size
    for pp in range(pp_size):
        ranks = list(range(pp * tp_size, (pp + 1) * tp_size))
        group = dist.new_group(ranks) if world_size > 1 else None
        if _pp_rank == pp:
            _tp_group = group
    _initialized = True


def parallel_initialized() -> bool:
    return _initialized


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def get_tp_rank() -> int:
    return _tp_rank


def get_tp_size() -> int:
    return _tp_size


def get_tp_group():
    """Process group spanning one pipeline stage (None for single-process)."""
    return _tp_group


def get_pp_rank() -> int:
    return _pp_rank


def get_pp_size() -> int:
    return _pp_size


def is_first_pp_stage() -> bool:
    return _pp_rank == 0


def is_last_pp_stage() -> bool:
    return _pp_rank == _pp_size - 1


def get_sampler_rank() -> int:
    """Rank that samples tokens: last pipeline stage, tp rank 0."""
    return _sampler_rank


def get_dp_idx() -> int:
    return _dp_idx
