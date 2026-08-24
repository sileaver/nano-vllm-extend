"""Qwen3.5 (hybrid gated-delta-net + sparse full attention) for nano-vllm.

Text-only port of the language model inside transformers'
Qwen3_5ForConditionalGeneration (Qwen/Qwen3.5-*).  Of the 24 layers, 18 are
linear attention (gated delta net, Qwen3-Next style) carrying a recurrent
state S [H, K, V] plus a causal-conv prefix instead of KV cache, and 6
layers (every 4th) are standard GQA full attention with partial RoPE
(rotary_dim = 0.25 * head_dim) and an attention output gate (q_proj emits
query and gate packed per head: out = attn_out * sigmoid(gate)).  All
standalone norms are Gemma-style (1 + weight); only the delta-net output
RMSNormGated uses plain weight * x.  The gated delta rule kernels are
pure-torch ports of the transformers reference implementation (no
fla / causal_conv1d dependency).

Recurrent-state pooling: each Qwen3_5GatedDeltaNet gets ``s_cache``
[num_slots, H, K, V] fp32 and ``conv_cache`` [num_slots, conv_dim, 3]
views bound by ModelRunner (the same pattern as Attention.k_cache).  The
per-step slot ids come from ``context.linear_state_ids``.  A fresh prefill
is encoded by zeroing the slot: a zero conv_state is mathematically the
zero left-padding the causal conv would apply anyway, and a zero S is the
null initial recurrent state — so no branch is needed.

Tensor parallelism shards every projection, the conv channels and the
recurrent state at v-head granularity (see Qwen3_5GatedDeltaNet); pipeline
parallelism splits the 24 layers across stages (see Qwen3_5Model).
"""
import re
import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import PipelineLayerShell, split_layers
from nanovllm.utils.context import get_context
from nanovllm.utils.parallel import get_pp_rank, get_pp_size, get_tp_rank, get_tp_size

# Triton gated-delta kernels (fla / flash-linear-attention) when available —
# numerically equivalent to the torch reference below (~1e-3 bf16 diff) but
# fused: the torch chunk loop costs ~150us/tok and the torch decode step
# ~30 kernel launches per layer.  Falls back to the pure-torch ports when
# fla is not installed.
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk,
        fused_recurrent_gated_delta_rule as _fla_recurrent,
    )
    _HAS_FLA = True
except ImportError:
    _fla_chunk = _fla_recurrent = None
    _HAS_FLA = False

# Kernel selection (env-overridable A/B switches).  Measured on RTX 5090
# (fla 0.5.2 / torch 2.13 / triton 3.7): the triton chunk kernel is 8-30x
# faster than the torch port at engine-batch shapes (16k tokens: ~1.0ms vs
# 9-30ms per layer) with ~1e-3 bf16-level numerical agreement, and the
# triton recurrent is ~1.4x faster at decode bs≈218 — so both default on
# now that fla is installed.  FLA_CHUNK=0 / FLA_RECURRENT=0 fall back to
# the pure-torch ports.
import os as _os
_USE_FLA_RECURRENT = _HAS_FLA and _os.environ.get("FLA_RECURRENT", "1") == "1"
_USE_FLA_CHUNK = _HAS_FLA and _os.environ.get("FLA_CHUNK", "1") == "1"
# A/B switch for the in-place decode recurrent (NANOVLLM_GDN_INPLACE=0
# restores the gather → recurrent → scatter reference path).
_USE_GDN_INPLACE = _os.environ.get("NANOVLLM_GDN_INPLACE", "1") == "1"


# ── Shared dynamic-shape compiled kernels (see class docstrings: instance
# method compiles exhaust the dynamo recompile cache and fall back to
# eager; one dynamic kernel serves all 24 layers and every batch shape) ──

@torch.compile(dynamic=True)
def _gemma_rms_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = x * (1.0 + weight.float())
    return x.to(weight.dtype)


@torch.compile(dynamic=True)
def _gemma_add_rms_forward(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    residual = (x.float() + residual.float()).to(orig_dtype)
    x = residual.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = x * (1.0 + weight.float())
    return x.to(orig_dtype), residual


@torch.compile(dynamic=True)
def _rms_gated_forward(
    hidden_states: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = weight * x.to(input_dtype)
    x = x * F.silu(gate.to(torch.float32))
    return x.to(input_dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """Matches the l2norm used inside the fla kernels."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                           initial_state=None):
    """Chunked gated delta rule: fla triton kernel when installed, else the
    pure-torch port below.  Signature/semantics identical either way."""
    if _USE_FLA_CHUNK:
        return _fla_chunk(query, key, value, g=g, beta=beta,
                          initial_state=initial_state, output_final_state=True,
                          use_qk_l2norm_in_kernel=True)
    return _torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size,
                                         initial_state)


def _torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                  initial_state=None):
    """Chunked gated delta rule (port of transformers' torch version).

    query/key [B, T, HK, K], value [B, T, HV, V], beta/g [B, T, HV] in the
    model dtype; returns (out [B, T, HV, V], final_state [B, HV, K, V]).
    Rows past a sequence's true length must be zero (padded-batch
    convention): their k/v/beta are zero and their g adds nothing to the
    cumulative decay, so the recurrent state stays unpolluted.
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, _, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_len = sequence_length + pad_size

    scale = k_head_dim ** -0.5
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=0)
    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch_size, query.shape[1], k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value))
    core_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=1)
    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_out[:, :, i] = attn_inter + attn @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2) @ v_new
        )
    core_out = core_out.reshape(core_out.shape[0], core_out.shape[1],
                                -1, core_out.shape[-1])
    core_out = core_out[:, :, :sequence_length]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, last_state


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    """Single-step gated delta rule: fla triton kernel when installed, else
    the pure-torch port below."""
    if _USE_FLA_RECURRENT:
        return _fla_recurrent(query, key, value, g=g, beta=beta,
                              initial_state=initial_state, output_final_state=True,
                              use_qk_l2norm_in_kernel=True)
    return _torch_recurrent_gated_delta_rule(query, key, value, g, beta,
                                             initial_state)


# ── In-place decode recurrent ────────────────────────────────────────
# The decode step used to gather s_cache[ids] (452MB at bs=218), run the
# recurrent, then scatter back — ~3.6GB of pure state copying per step
# (~40% of decode GPU time).  This kernel indexes the pool rows directly
# and updates them in place: S ← S·exp(g) + k⊗β(v − Sᵀk); out = Sᵀq.
# Numerically mirrors _torch_recurrent_gated_delta_rule (fp32 math, q/k
# l2-normalised with eps 1e-6 then q scaled by DK^-0.5).

@triton.jit
def _recurrent_inplace_kernel(
    Q, K, V, G, B, S_POOL, IDS, OUT, scale,
    q_bstride, k_bstride, v_bstride, slot_stride,
    H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
):
    pid = tl.program_id(0)                       # one program per (seq, head)
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, DK)
    offs_v = tl.arange(0, DV)
    # q/k/v come from split_qkv: views into the [bs, conv_dim, 1] mixed
    # buffer, so the batch stride is conv_dim (6144), NOT H*D — index via
    # the passed strides or every b > 0 row reads garbage.
    q = tl.load(Q + b * q_bstride + h * DK + offs_k).to(tl.float32)
    k = tl.load(K + b * k_bstride + h * DK + offs_k).to(tl.float32)
    v = tl.load(V + b * v_bstride + h * DV + offs_v).to(tl.float32)
    beta = tl.load(B + b * H + h).to(tl.float32)
    g = tl.load(G + b * H + h)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    q = q * scale
    slot = tl.load(IDS + b).to(tl.int64)
    # s_pool is a per-layer VIEW of the state pool: consecutive slots are
    # L·H·K·V elements apart (slot_stride), heads are dense within a slot.
    sp = S_POOL + slot * slot_stride + h * (DK * DV)
    s = tl.load(sp + offs_k[:, None] * DV + offs_v[None, :])   # [DK, DV] fp32
    s = s * tl.exp(g)
    kv = tl.sum(s * k[:, None], 0)                             # [DV]
    delta = (v - kv) * beta
    s = s + k[:, None] * delta[None, :]
    tl.store(sp + offs_k[:, None] * DV + offs_v[None, :], s)
    out = tl.sum(s * q[:, None], 0)                            # [DV]
    tl.store(OUT + pid * DV + offs_v, out.to(OUT.dtype.element_ty))


def recurrent_gdn_inplace(q, k, v, g, beta, s_pool, ids):
    """q/k/v [bs, 1, H, D] bf16, g/beta [bs, 1, H]; updates s_pool rows
    `ids` in place, returns out [bs, 1, H, DV] bf16.  q/k/v may be strided
    views of the conv output; s_pool a per-layer view of the state pool."""
    bs, _, H, DK = k.shape
    DV = v.shape[-1]
    assert q.stride(2) == DK and k.stride(2) == DK and v.stride(2) == DV
    assert g.is_contiguous() and beta.is_contiguous()
    assert s_pool.stride(1) == DK * DV and s_pool.stride(2) == DV \
        and s_pool.stride(3) == 1
    out = torch.empty(bs, 1, H, DV, dtype=v.dtype, device=v.device)
    _recurrent_inplace_kernel[(bs * H,)](
        q, k, v, g, beta, s_pool, ids, out, DK ** -0.5,
        q.stride(0), k.stride(0), v.stride(0), s_pool.stride(0),
        H=H, DK=DK, DV=DV, num_warps=4,
    )
    return out


@torch.compile(dynamic=True)
def _torch_recurrent_gated_delta_rule(query, key, value, g, beta,
                                      initial_state=None):
    """Single-step gated delta rule (port of transformers' torch version).

    query/key [B, 1, HK, K], value [B, 1, HV, V], beta/g [B, 1, HV].
    S <- S·exp(g); delta = β(v − Sᵀk); S += k ⊗ delta; out = Sᵀq.
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, _, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = k_head_dim ** -0.5
    query = query * scale
    state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value))
    q_t = query[:, :, 0]
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, 0].unsqueeze(-1)
    state = state * g_t
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    # [B, H, V] -> [B, T=1, H, V] — same layout the chunk kernel returns
    out = out.unsqueeze(1).to(initial_dtype)
    return out, state


class Qwen3_5RMSNorm(nn.Module):
    """Gemma-style RMSNorm: zero-init weight applied as (1 + weight).

    Rounding follows transformers exactly (fp32 norm, fp32 (1+w) multiply,
    single bf16 rounding at the end; the residual sum is rounded to bf16
    before normming) — the real checkpoint's activation scale amplifies
    any extra intermediate rounding.

    The compiled kernels are module-level free functions with (weight, eps)
    passed explicitly: instance-method compiles guard on `self`, so the
    per-layer instances each recompile and exhaust the dynamo cache — every
    later call runs eager under guard overhead, and CUDA graphs captured
    after the 8th recompile bake the eager fallback.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _gemma_rms_forward(x, self.weight, self.eps)

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        return _gemma_add_rms_forward(x, residual, self.weight, self.eps)

    def forward(self, x, residual=None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)


class Qwen3_5RMSNormGated(nn.Module):
    """Delta-net output norm on the value head dim, gated by silu(z).

    Unlike Qwen3_5RMSNorm this is a plain w·x norm (weight ones-init).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor):
        return _rms_gated_forward(hidden_states, gate, self.weight, self.eps)


class Qwen3_5Attention(nn.Module):
    """Full-attention layer (every full_attention_interval-th layer).

    q_proj outputs 2x width: per head, the first head_dim channels are the
    query and the second head_dim the output gate (attn_output *=
    sigmoid(gate)).  RoPE is partial: only the first rotary_dim = 64 of the
    256 head channels are rotated.
    """

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_heads * self.head_dim * 2, bias=False)
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rope_params = getattr(config, "rope_parameters", None) or {}
        rotary_dim = int(self.head_dim * rope_params.get("partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_params.get("rope_theta", 10000000.0),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_out = self.q_proj(hidden_states)
        q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_out.chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1) * torch.sigmoid(gate.flatten(1, -1))
        return self.o_proj(o)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Linear-attention layer (gated delta net, Qwen3-Next style).

    Recurrent state pooling: s_cache [num_slots, H, K, V] fp32 and
    conv_cache [num_slots, conv_dim, kernel-1] are views bound by
    ModelRunner; the active slots come from context.linear_state_ids.
    """

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.hidden_size = config.hidden_size
        self.total_num_k_heads = config.linear_num_key_heads
        self.total_num_v_heads = config.linear_num_value_heads
        assert self.total_num_v_heads % self.total_num_k_heads == 0, \
            "qwen3_5: v/k head replication not supported (2B has 16/16)"
        assert self.total_num_k_heads % tp_size == 0
        assert self.total_num_v_heads % tp_size == 0
        # TP-local head counts / dims: every projection, the conv channels
        # and the recurrent state all shard at v-head granularity.
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads          # TP-local
        self.value_dim = self.head_v_dim * self.num_v_heads        # TP-local
        self.conv_kernel = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim          # TP-local
        self._total_conv_dim = self.head_k_dim * self.total_num_k_heads * 2 \
            + self.head_v_dim * self.total_num_v_heads

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size, self._total_conv_dim, bias=False)
        # The qkv checkpoint layout is [q(key), k(key), v(value)] — a plain
        # contiguous column shard would cut mid-head at the q/k boundary;
        # shard each part by heads instead.
        self.in_proj_qkv.weight.weight_loader = self._conv_layout_loader
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size, self.head_v_dim * self.total_num_v_heads, bias=False)
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size, self.total_num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size, self.total_num_v_heads, bias=False)
        # padding is applied per-call (0 or kernel-1 depending on state)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.conv_kernel,
            groups=self.conv_dim, padding=0, bias=False)
        self.conv1d.weight.weight_loader = self._conv_layout_loader
        A = torch.empty(self.total_num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))     # full; sliced per rank
        self.dt_bias = nn.Parameter(torch.ones(self.total_num_v_heads))
        self._v_head_slice = slice(
            get_tp_rank() * self.num_v_heads, (get_tp_rank() + 1) * self.num_v_heads)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.head_v_dim * self.total_num_v_heads, self.hidden_size, bias=False)
        # State-pool views bound by ModelRunner (index by slot id).
        self.s_cache: torch.Tensor | None = None      # [slots, H, K, V] fp32
        self.conv_cache: torch.Tensor | None = None   # [slots, conv_dim, kernel-1]

    def _conv_layout_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard a [q | k | v] concatenated dim at head granularity.

        Applies to both in_proj_qkv.weight [conv_dim, hidden] and
        conv1d.weight [conv_dim, 1, kernel]: dim 0 holds the q/k/v parts
        back to back, each part uniform in head size, so chunking each part
        separately keeps rank boundaries head-aligned."""
        kd = self.head_k_dim * self.total_num_k_heads
        vd = self.head_v_dim * self.total_num_v_heads
        q, k, v = loaded_weight.split([kd, kd, vd], dim=0)
        r, s = get_tp_rank(), get_tp_size()
        param.data.copy_(torch.cat([t.chunk(s, dim=0)[r] for t in (q, k, v)], dim=0))

    # Chunk-kernel call budget (padded tokens per call): the pure-torch
    # kernel's fp32 intermediates need ~150MB per 1k padded tokens, so one
    # call over a full 16k-token engine batch demands multi-GB transient
    # peaks — with a large KV pool that leaves no headroom and the
    # allocator livelocks in OOM-retry.  The fla triton chunk kernel has no
    # such blow-up, so it segments at the full engine batch instead (fewer
    # Python launcher calls).  Segments thread the recurrent/conv state,
    # so splitting is exactly equivalent to one call.
    PREFILL_SEGMENT_TOKENS_FLA = 16384
    PREFILL_SEGMENT_TOKENS_TORCH = 4096

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,   # [N, hidden] varlen concat
    ) -> torch.Tensor:
        context = get_context()
        cu_q = context.cu_seqlens_q
        lens = cu_q[1:] - cu_q[:-1]                       # [bs]
        bs = lens.size(0)
        ids = context.linear_state_ids                   # [bs]
        T = int(context.max_seqlen_q)
        H = self.hidden_size

        # varlen [N, H] -> padded [bs, T, H]; mask keeps the row order so the
        # output can be gathered back with the same mask.
        if T == 1:
            x = hidden_states.view(bs, 1, H)
            mask = None
        else:
            mask = torch.arange(T, device=hidden_states.device) < lens.unsqueeze(1)
            x = hidden_states.new_zeros(bs, T, H)
            x[mask] = hidden_states

        b = self.in_proj_b(x)                             # [bs, T, H]
        a = self.in_proj_a(x)                             # [bs, T, H]
        beta = b.sigmoid()
        A_log = self.A_log[self._v_head_slice]
        dt_bias = self.dt_bias[self._v_head_slice]
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        if mask is not None:
            # Padded rows: k/v are zero there (bias-free conv over zero
            # input) but g is NOT (-exp(A_log)·softplus(dt_bias) < 0) —
            # left unmasked it decays the recurrent state across the
            # padding tail, wiping short sequences' state in a
            # mixed-length batch.  g=0 makes trailing chunks exact no-ops
            # (beta is irrelevant: the state update is proportional to k).
            g = g.masked_fill(~mask.unsqueeze(-1), 0.0)
        conv_w = self.conv1d.weight                       # [conv_dim, 1, k]

        def split_qkv(mixed, n):
            q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=1)
            q = q.transpose(1, 2).reshape(bs, n, self.num_k_heads, self.head_k_dim)
            k = k.transpose(1, 2).reshape(bs, n, self.num_k_heads, self.head_k_dim)
            v = v.transpose(1, 2).reshape(bs, n, self.num_v_heads, self.head_v_dim)
            if self.num_v_heads > self.num_k_heads:
                rep = self.num_v_heads // self.num_k_heads
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            return q, k, v

        if T == 1:
            # Decode: roll the conv state and convolve the 4-token window.
            mixed = self.in_proj_qkv(x).transpose(1, 2)   # [bs, conv_dim, 1]
            window = torch.cat([self.conv_cache[ids], mixed], dim=-1)
            self.conv_cache[ids] = window[..., 1:]
            mixed = F.silu(F.conv1d(window, conv_w, groups=self.conv_dim))
            q, k, v = split_qkv(mixed, 1)
            z = self.in_proj_z(x)
            if _USE_GDN_INPLACE:
                # In-place pool update (no gather/scatter of the 452MB state).
                out = recurrent_gdn_inplace(q, k, v, g, beta, self.s_cache, ids)
            else:
                out, new_s = recurrent_gated_delta_rule(
                    q, k, v, g, beta, self.s_cache[ids])
                self.s_cache[ids] = new_s
        else:
            # Prefill / chunk continuation, in T-segments (see
            # PREFILL_SEGMENT_TOKENS).  The conv state threads across
            # segments; a zero state on the first segment of a fresh prefill
            # is identical to the causal conv's zero left-padding.
            z = self.in_proj_z(x)                         # [bs, T, value_dim]
            seg_tokens = (self.PREFILL_SEGMENT_TOKENS_FLA if _USE_FLA_CHUNK
                          else self.PREFILL_SEGMENT_TOKENS_TORCH)
            seg = max(1, seg_tokens // bs)
            conv_states = self.conv_cache[ids]            # [bs, conv_dim, k-1]
            state = self.s_cache[ids]
            outs = []
            for t0 in range(0, T, seg):
                t1 = min(t0 + seg, T)
                n = t1 - t0
                mixed = self.in_proj_qkv(x[:, t0:t1]).transpose(1, 2)
                window = torch.cat([conv_states, mixed], dim=-1)
                mixed = F.silu(F.conv1d(window, conv_w, groups=self.conv_dim))
                seg_mask = mask[:, t0:t1] if mask is not None else None
                if seg_mask is not None:
                    # Padded rows' conv output would see the real history
                    # through the sliding window (non-zero k/v) and pollute
                    # the recurrent state — zero them so padding is an
                    # exact no-op.
                    mixed = mixed * seg_mask.unsqueeze(1).to(mixed.dtype)
                # Per-seq conv-state writeback at the latest real position:
                # window layout is [state(k-1), x(n)], token t0+i sits at
                # column k-1+i, so the 3-wide state after a seq's last real
                # token in this segment starts at column min(r, n) (r =
                # remaining real tokens).  Batched gather replaces the old
                # per-seq loop (one GPU sync per row via int(lens[b_i])).
                if seg_mask is not None:
                    r = (lens - t0)
                    starts = r.clamp(min=0, max=n)        # [bs]
                    valid = r > 0                          # padding-only seqs keep their state
                    cols = starts.view(bs, 1, 1) + torch.arange(
                        3, device=window.device, dtype=torch.int64)
                    sel = window.gather(-1, cols.expand(bs, window.size(1), 3))
                    self.conv_cache[ids[valid]] = sel[valid]
                else:
                    self.conv_cache[ids] = window[..., -3:]
                conv_states = window[..., -3:]
                q, k, v = split_qkv(mixed, n)
                out_i, state = chunk_gated_delta_rule(
                    q, k, v, g[:, t0:t1], beta[:, t0:t1], initial_state=state)
                outs.append(out_i)
            self.s_cache[ids] = state
            out = torch.cat(outs, dim=1)

        # Norm on the value head dim: keep the head dim explicit (weight is
        # [head_v_dim], shared across heads — matches the reference's
        # reshape(-1, head_v_dim) 2D norm).
        core = out.reshape(bs, T, self.num_v_heads, self.head_v_dim)
        core = self.norm(core, z.view(bs, T, self.num_v_heads, self.head_v_dim))
        core = core.reshape(bs, T, self.value_dim)
        y = self.out_proj(core)                           # [bs, T, H]
        return y.view(-1, H) if mask is None else y[mask]


class Qwen3_5MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        self.mlp = Qwen3_5MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        mixer = self.linear_attn if self.block_type == "linear_attention" else self.self_attn
        hidden_states = mixer(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.pp_first = get_pp_rank() == 0
        self.pp_last = get_pp_rank() == get_pp_size() - 1
        self.start_layer, self.end_layer = split_layers(
            config.num_hidden_layers, get_pp_size(), get_pp_rank())
        if self.pp_first:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i)
            if self.start_layer <= i < self.end_layer else PipelineLayerShell()
            for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def checkpoint_skips(self, weight_name: str) -> bool:
        """True for checkpoint weights this pipeline stage does not own."""
        if self.embed_tokens is None and ".embed_tokens." in weight_name:
            return True
        m = re.search(r"\.layers\.(\d+)\.", weight_name)
        return bool(m) and not (self.start_layer <= int(m.group(1)) < self.end_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        skip_layers: tuple[int, ...] | None = None,
        output_layer_hidden: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ):
        # skip_layers / output_layer_hidden exist for signature parity with
        # Qwen3ForCausalLM (speculative-decoding hooks); the qwen3_5 config
        # asserts speculative decoding off, so they are ignored here.
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        for layer in self.layers:
            if isinstance(layer, PipelineLayerShell):
                continue
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not self.pp_last:
            # Intermediate pipeline stage: hand the fused-residual chain on.
            return hidden_states, residual
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    # q/k/v stay split (q_proj packs the output gate, so no qkv merge).
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # The checkpoint is a multimodal shell: strip the language-model prefix
    # and skip the vision tower / MTP head (text-only inference).
    weight_remapping = (("model.language_model.", "model."),)
    ignored_weight_prefixes = ("model.visual.", "mtp.")

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.pp_first, self.pp_last = self.model.pp_first, self.model.pp_last
        if self.pp_last:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if config.tie_word_embeddings and self.pp_first:
                self.lm_head.weight.data = self.model.embed_tokens.weight.data
        else:
            self.lm_head = None
        # Tied embeddings across pipeline stages: the last stage has no
        # embed_tokens to share storage with, so route the embed checkpoint
        # entry into lm_head directly (its vocab-sharded loader applies).
        self.checkpoint_aliases = (
            {"model.embed_tokens.weight": "lm_head.weight"}
            if config.tie_word_embeddings and self.lm_head is not None and not self.pp_first
            else {})

    def checkpoint_skips(self, weight_name: str) -> bool:
        if self.model.checkpoint_skips(weight_name):
            return True
        return self.lm_head is None and weight_name.startswith("lm_head.")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        skip_layers: tuple[int, ...] | None = None,
        output_layer_hidden: list[int] | None = None,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions, hidden_states=hidden_states,
                          residual=residual)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None, "compute_logits on a non-last pipeline stage"
        return self.lm_head(hidden_states)

    def compute_all_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Logits for every row (speculative verification needs K+1 rows
        per seq; the regular head gathers only the last row per seq)."""
        return self.lm_head.forward_all(hidden_states)
