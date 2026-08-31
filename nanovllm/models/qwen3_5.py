"""Qwen3.5 (hybrid gated-delta-net + sparse full attention) for nano-vllm.

Port of transformers' Qwen3_5ForConditionalGeneration (Qwen/Qwen3.5-*):
the language model below plus — for multimodal checkpoints — the vision
tower (see qwen3_5_vision.py), whose merged patch embeddings replace
image_token_id rows of the token embedding, and interleaved MRoPE
positions ([3, N] T/H/W tables over image regions; see
MRotaryEmbedding and utils/multimodal.py).

Of the 24 layers, 18 are
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
parallelism splits the 24 layers across stages (see Qwen3_5Model).  The
vision tower is replicated across TP ranks on the first pipeline stage.
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
from nanovllm.models.qwen3_5_vision import Qwen3_5VisionModel
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

# FlashInfer fused kernels for the norm / activation sites (the same
# kernels vLLM uses): gemma_rmsnorm / gemma_fused_add_rmsnorm cover the
# (1 + w) Gemma semantics natively, silu_and_mul replaces the compiled
# MLP activation.  NANOVLLM_FI_NORM=0 falls back to the torch.compile'd
# reference paths.
try:
    import flashinfer.norm as _fi_norm
    import flashinfer.activation as _fi_act
    _HAS_FI_NORM = True
except ImportError:
    _fi_norm = _fi_act = None
    _HAS_FI_NORM = False
_USE_FI_NORM = _HAS_FI_NORM and _os.environ.get("NANOVLLM_FI_NORM", "1") == "1"


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
def _causal_conv_silu_kernel(
    X, S, W, Y, ENDP,                       # in: x [n,C,T], state [n,C,K-1], w [C,K]; out: y [n,C,T]
    T, C,
    sXn, sXc, sXt, sSn, sSc, sYn, sYc, sYt,
    K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Depthwise causal conv + SiLU with in-place state roll.

    y[n,c,t] = silu(Σᵢ w[c,i]·win[t−K+1+i]) with win = [state | x]
    (state serves as the K−1 left-padding).  After the T loop, the row's
    new conv state (its last K−1 inputs at ENDP — the row's last real
    position in this call, not the padded tail) is written back INTO S
    in place; each program only reads S entries it wrote after reading
    them, and rows/channels partition the grid, so the in-place update
    is race-free.  ENDP < 0 skips the writeback (padding-only row).
    """
    n = tl.program_id(0)
    pc = tl.program_id(1)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    Xn = X + n * sXn
    Sn = S + n * sSn
    Yn = Y + n * sYn
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
        for i in tl.static_range(K):
            p = offs_t - (K - 1) + i
            xv = tl.load(Xn + offs_c[:, None] * sXc + p[None, :] * sXt,
                         mask=cm[:, None] & (p >= 0)[None, :] & (p < T)[None, :],
                         other=0.0)
            sv = tl.load(Sn + offs_c[:, None] * sSc + (p + (K - 1))[None, :],
                         mask=cm[:, None] & (p < 0)[None, :], other=0.0)
            w_i = tl.load(W + offs_c * K + i, mask=cm, other=0.0).to(tl.float32)
            acc += (xv.to(tl.float32) + sv.to(tl.float32)) * w_i[:, None]
        y = acc * tl.sigmoid(acc)
        tl.store(Yn + offs_c[:, None] * sYc + offs_t[None, :] * sYt,
                 y.to(Y.dtype.element_ty),
                 mask=cm[:, None] & (offs_t < T)[None, :])
    end = tl.load(ENDP + n)
    if end >= 0:
        for i in tl.static_range(K - 1):
            p = end - (K - 1) + i
            xv = tl.load(Xn + offs_c * sXc + p * sXt,
                         mask=cm & (p >= 0), other=0.0)
            sv = tl.load(Sn + offs_c * sSc + (p + (K - 1)),
                         mask=cm & (p < 0), other=0.0)
            tl.store(Sn + offs_c * sSc + i, (xv + sv), mask=cm)


def causal_conv_silu(x, state, weight, endp):
    """x [n, C, T] (arbitrary strides), state [n, C, K-1] (updated in
    place at each row's ENDP), weight [C, K].  Returns y [n, C, T]
    contiguous.  endp: int32 [n] (last real position per row, -1 = skip
    writeback)."""
    n, C, T = x.shape
    K = weight.shape[-1]
    y = torch.empty(n, C, T, dtype=x.dtype, device=x.device)
    BLOCK_C = 64
    BLOCK_T = 64
    _causal_conv_silu_kernel[(n, triton.cdiv(C, BLOCK_C))](
        x, state, weight, y, endp, T, C,
        x.stride(0), x.stride(1), x.stride(2),
        state.stride(0), state.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        K=K, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C, num_warps=4,
    )
    return y


@triton.jit
def _gdn_g_beta_kernel(
    Bp, Ap, AL, DT, MASK, Gp, BETA, total,
    VH: tl.constexpr, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr,
):
    """g = -exp(A_log)·softplus(a + dt_bias) (fp32, zero on padding),
    beta = sigmoid(b) (bf16) — one kernel for the whole [n, T, Vh] batch
    (was ~7 eager elementwise kernels per layer)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    h = offs % VH
    a = tl.load(Ap + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(Bp + offs, mask=m, other=0.0).to(tl.float32)
    al = tl.load(AL + h, mask=m, other=0.0).to(tl.float32)
    dt = tl.load(DT + h, mask=m, other=0.0).to(tl.float32)
    z = a + dt
    sp = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))
    g = -tl.exp(al) * sp
    if HAS_MASK:
        real = tl.load(MASK + offs // VH, mask=m, other=0)
        g = tl.where(real, g, 0.0)
    tl.store(Gp + offs, g, mask=m)
    tl.store(BETA + offs, tl.sigmoid(b).to(BETA.dtype.element_ty), mask=m)


def fused_g_beta(b, a, a_log, dt_bias, mask):
    """b/a [.., Vh] bf16 contiguous; a_log/dt_bias [Vh]; mask [n, T] bool
    or None (padding rows get g = 0).  Returns (g fp32, beta bf16)."""
    total = a.numel()
    vh = a.shape[-1]
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(a)
    flat_mask = (mask.reshape(-1).contiguous() if mask is not None
                 else b)  # dummy pointer when unused
    BLOCK = 256
    _gdn_g_beta_kernel[(triton.cdiv(total, BLOCK),)](
        b, a, a_log, dt_bias, flat_mask,
        g, beta, total, VH=vh, HAS_MASK=mask is not None,
        BLOCK=BLOCK, num_warps=4,
    )
    return g, beta


# A/B switch for the fused g/beta kernel (NANOVLLM_GDN_GBETA=0 restores
# the eager fp32 elementwise chain).
_USE_GDN_GBETA = _os.environ.get("NANOVLLM_GDN_GBETA", "1") == "1"


# ── Fused attention output gate (o = o · sigmoid(g)) ─────────────────
# The eager form ran two full-activation elementwise kernels per
# attention layer (sigmoid, then mul) plus a hidden contiguous copy —
# gate.flatten(1,-1) on the strided q_proj view materialises the gate.
# One row-wise kernel reads o and the strided gate directly and writes o
# in place.  Rounding follows torch exactly: sigmoid in fp32 rounded to
# bf16, then an fp32 multiply with a single bf16 round.
@triton.jit
def _sigmoid_mul_kernel(
    X, G, x_stride_n, x_stride_h, g_stride_n, g_stride_h,
    HEADS: tl.constexpr, DIM: tl.constexpr,
):
    pid = tl.program_id(0)            # one program per (token, head) row
    n = pid // HEADS
    h = pid % HEADS
    offs = tl.arange(0, DIM)
    x = tl.load(X + n * x_stride_n + h * x_stride_h + offs).to(tl.float32)
    g = tl.load(G + n * g_stride_n + h * g_stride_h + offs).to(tl.float32)
    sig = tl.sigmoid(g).to(X.dtype.element_ty).to(tl.float32)
    y = (x * sig).to(X.dtype.element_ty)
    tl.store(X + n * x_stride_n + h * x_stride_h + offs, y)


def sigmoid_mul_(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """x [N, H, D] (fresh, contiguous) mutated in place: x *= sigmoid(gate);
    gate may be a strided view (e.g. the second half of a packed q_proj
    output).  Returns x."""
    n, heads, dim = x.shape
    _sigmoid_mul_kernel[(n * heads,)](
        x, gate, x.stride(0), x.stride(1), gate.stride(0), gate.stride(1),
        HEADS=heads, DIM=dim, num_warps=2)
    return x


@triton.jit
def _causal_conv_silu_varlen_kernel(
    X, S, W, Y, LENS, CUSEQ, C, D3, sXn, sSn, sSc, sYp,
    K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Varlen depthwise causal conv + SiLU over flat x [N, C] rows.

    Program (row r of n_seqs, channel block) processes that sequence's
    [0, T_r) tokens at flat offset CUSEQ[r] — projections stay dense
    varlen GEMMs and nothing is padded.  The conv window never crosses a
    sequence boundary (p ∈ [0, T_r) by construction), and the state is
    rolled in place at the row's end (= T_r, all rows real).

    Output layout: Y is [3, N, C//3] — the q, k, v thirds of the packed
    conv channels land in three separately-contiguous slabs, so the
    downstream [N, H, D] views are contiguous and the fla chunk kernel's
    input_guard does not re-copy all three (that copy chain showed up as
    ~1.7% of prefill GPU time).  A channel block never straddles thirds
    (C//3 % BLOCK_C == 0).
    """
    r = tl.program_id(0)
    pc = tl.program_id(1)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    part = (pc * BLOCK_C) // D3
    local = offs_c - part * D3
    T = tl.load(LENS + r)
    row0 = tl.load(CUSEQ + r)          # global row of this sequence
    Xr = X + row0 * sXn
    Sn = S + r * sSn
    Yr = Y + part * sYp + row0 * D3 + local
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        acc = tl.zeros((BLOCK_C, BLOCK_T), dtype=tl.float32)
        for i in tl.static_range(K):
            p = offs_t - (K - 1) + i
            xv = tl.load(Xr + offs_c[:, None] + p[None, :] * sXn,
                         mask=cm[:, None] & (p >= 0)[None, :] & (p < T)[None, :],
                         other=0.0)
            sv = tl.load(Sn + offs_c[:, None] * sSc + (p + (K - 1))[None, :],
                         mask=cm[:, None] & (p < 0)[None, :],
                         other=0.0)
            w_i = tl.load(W + offs_c * K + i, mask=cm, other=0.0).to(tl.float32)
            acc += (xv.to(tl.float32) + sv.to(tl.float32)) * w_i[:, None]
        y = acc * tl.sigmoid(acc)
        tl.store(Yr[:, None] + offs_t[None, :] * D3,
                 y.to(Y.dtype.element_ty),
                 mask=cm[:, None] & (offs_t < T)[None, :])
    for i in tl.static_range(K - 1):
        p = T - (K - 1) + i
        xv = tl.load(Xr + offs_c + p * sXn,
                     mask=cm & (p >= 0) & (p < T), other=0.0)
        sv = tl.load(Sn + offs_c * sSc + (p + (K - 1)),
                     mask=cm & (p < 0), other=0.0)
        tl.store(Sn + offs_c * sSc + i, (xv + sv), mask=cm)


def causal_conv_silu_varlen(x, state, weight, lens, cuseq):
    """x [N, C] bf16 contiguous varlen; state [n, C, K-1] (updated in
    place); lens int32 [n] GPU; cuseq int32 [n] GPU (per-row starts).
    Returns [3, N, C//3]: the q/k/v thirds, each contiguous."""
    N, C = x.shape
    K = weight.shape[-1]
    d3 = C // 3
    assert d3 * 3 == C and d3 % 64 == 0
    y = torch.empty(3, N, d3, dtype=x.dtype, device=x.device)
    _causal_conv_silu_varlen_kernel[(lens.shape[0], triton.cdiv(C, 64))](
        x, state, weight, y, lens, cuseq, C, d3,
        x.stride(0), state.stride(0), state.stride(1), y.stride(0),
        K=K, BLOCK_T=64, BLOCK_C=64, num_warps=4,
    )
    return y


# A/B switch for the varlen prefill path (NANOVLLM_GDN_VARLEN=0 restores
# the padded [n, max_T] chunk path; requires fla).
_USE_GDN_VARLEN = (_USE_FLA_CHUNK
                   and _os.environ.get("NANOVLLM_GDN_VARLEN", "1") == "1")

# Per-step varlen metadata (cu/lens GPU tensors + pinned host copies),
# shared by every GDN layer: built once per new length-tuple, reused by
# the other 17 layers of the same step (building pinned tensors per
# layer swamped small mixed-batch steps in ~200µs×18 of pinned-allocator
# churn).
_VARLEN_META: tuple | None = None


def _varlen_meta(lens_q: list[int]):
    global _VARLEN_META
    key = tuple(lens_q)
    if _VARLEN_META is not None and _VARLEN_META[0] == key:
        return _VARLEN_META[1]
    cu = [0]
    for l in lens_q:
        cu.append(cu[-1] + l)
    cu_cpu = torch.tensor(cu, dtype=torch.int32, pin_memory=True)
    lens_cpu = torch.tensor(lens_q, dtype=torch.int32, pin_memory=True)
    meta = (cu_cpu.cuda(non_blocking=True), lens_cpu.cuda(non_blocking=True),
            cu_cpu.to(torch.int64), cu_cpu.to(torch.int64).cuda(non_blocking=True))
    _VARLEN_META = (key, meta)
    return meta


# A/B switch: NANOVLLM_CONV_TRITON=0 restores the F.conv1d reference path.
# The Triton kernel only covers T > 1 (prefill/mixed chunks): at T = 1 its
# 64-wide T tiles are 98% masked and the per-K scalar state stores add
# ~30µs/layer — the reference cat+conv1d there is 3 tiny kernels.
_USE_CONV_TRITON = _os.environ.get("NANOVLLM_CONV_TRITON", "1") == "1"


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
        if _USE_FI_NORM:
            return _fi_norm.gemma_rmsnorm(x, self.weight, eps=self.eps)
        return _gemma_rms_forward(x, self.weight, self.eps)

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        if _USE_FI_NORM:
            # In-place: x becomes the normed output, residual becomes
            # x + residual.  The decoder layer rebinds both immediately
            # and drops the originals, so the mutation is safe — and it
            # is exactly the buffer-reuse pattern the kernel is built
            # for (works unchanged under CUDA graphs).
            _fi_norm.gemma_fused_add_rmsnorm(x, residual, self.weight,
                                             eps=self.eps)
            return x, residual
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
        mrope_section = rope_params.get("mrope_section")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_params.get("rope_theta", 10000000.0),
            mrope_section=tuple(mrope_section) if mrope_section else None,
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
        o = sigmoid_mul_(o, gate)                      # in place: o · σ(gate)
        return self.o_proj(o.flatten(1, -1))


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

        if T == 1:
            return self._gdn_padded(
                hidden_states.view(bs, 1, H), lens, ids, 1).view(-1, H)

        lens_q = context.lens_q if context.lens_q is not None else lens.tolist()
        # Mixed batches (continuous scheduling emits decode rows before
        # prefill chunks): route the decode prefix through the T=1
        # recurrent path and only the prefill rows through the chunk path
        # at their own max length.  A single padded [bs, T=max_seqlen_q]
        # chunk pass over a mostly-decode batch computes up to ~150x the
        # real token count (the #1 loss vs vLLM on ragged workloads).
        n1 = 0
        while n1 < bs and lens_q[n1] == 1:
            n1 += 1
        pre_ok = n1 > 0 and n1 < bs and min(lens_q[n1:]) > 1
        if n1 == 0 or pre_ok:
            off = 0
            parts = []
            if n1:
                parts.append(self._gdn_padded(
                    hidden_states[:n1].view(n1, 1, H),
                    lens[:n1], ids[:n1], 1).view(-1, H))
                off = n1
            if _USE_GDN_VARLEN:
                # True varlen prefill group: dense [N, H] projections, no
                # padding anywhere (the fla chunk kernel natively takes
                # cu_seqlens; verified bit-equivalent to the padded call).
                parts.append(self._gdn_prefill_varlen(
                    hidden_states[off:], lens_q[off:], ids[off:]))
                return torch.cat(parts, dim=0)
            lens_p = lens[off:]
            Tp = T if off == 0 else max(lens_q[off:])
            mask_p = torch.arange(
                Tp, device=hidden_states.device) < lens_p.unsqueeze(1)
            x = hidden_states.new_zeros(bs - off, Tp, H)
            x[mask_p] = hidden_states[off:]
            parts.append(self._gdn_padded(x, lens_p, ids[off:], Tp)[mask_p])
            return torch.cat(parts, dim=0)
        # Defensive: len-1 rows are not a contiguous prefix (no scheduler
        # produces this today) — one padded pass over everything.
        mask = torch.arange(T, device=hidden_states.device) < lens.unsqueeze(1)
        x = hidden_states.new_zeros(bs, T, H)
        x[mask] = hidden_states
        return self._gdn_padded(x, lens, ids, T)[mask]

    def _gdn_prefill_varlen(self, hidden, lens_q, ids) -> torch.Tensor:
        """Prefill group in true varlen (requires fla): dense [N, H]
        projections over the flat token axis, a varlen causal-conv kernel,
        and the fla chunk kernel driven by cu_seqlens — no padding at any
        stage, so ragged prompt groups cost exactly their real tokens.
        Returns the varlen output [N, H]."""
        n = len(lens_q)
        b = self.in_proj_b(hidden)                      # [N, Vh]
        a = self.in_proj_a(hidden)
        A_log = self.A_log[self._v_head_slice]
        dt_bias = self.dt_bias[self._v_head_slice]
        if _USE_GDN_GBETA:
            g, beta = fused_g_beta(b, a, A_log, dt_bias, None)
        else:
            beta = b.sigmoid()
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        z = self.in_proj_z(hidden)                      # [N, value_dim]
        mixed = self.in_proj_qkv(hidden)                # [N, conv_dim]
        cu_gpu, lens_gpu, cu64_cpu, cu64_gpu = _varlen_meta(lens_q)
        conv_state = self.conv_cache[ids]
        y3 = causal_conv_silu_varlen(
            mixed, conv_state,
            self.conv1d.weight.view(self.conv_dim, self.conv_kernel),
            lens_gpu, cu_gpu[:-1])
        self.conv_cache[ids] = conv_state
        # y3 [3, N, part_dim]: q/k/v land contiguous, so the fla chunk
        # kernel's input_guard does not re-copy each 64MB slab.
        q = y3[0].view(-1, self.num_k_heads, self.head_k_dim)
        k = y3[1].view(-1, self.num_k_heads, self.head_k_dim)
        v = y3[2].view(-1, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        out, new_state = _fla_chunk(
            q[None], k[None], v[None], g=g[None], beta=beta[None],
            initial_state=self.s_cache[ids], output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu64_gpu, cu_seqlens_cpu=cu64_cpu)
        self.s_cache[ids] = new_state
        core = out[0]                                   # [N, H, DV]
        core = self.norm(core, z.view(-1, self.num_v_heads, self.head_v_dim))
        y = self.out_proj(core.reshape(-1, self.value_dim))
        return y

    def _gdn_padded(self, x, lens, ids, T) -> torch.Tensor:
        """GDN over an already-padded [n, T, H] batch — padding is an exact
        no-op there (zero qkv, g=0).  Returns the padded output [n, T, H].
        """
        n = x.size(0)
        mask = None if T == 1 else (
            torch.arange(T, device=x.device) < lens.unsqueeze(1))
        b = self.in_proj_b(x)                             # [n, T, H]
        a = self.in_proj_a(x)                             # [n, T, H]
        A_log = self.A_log[self._v_head_slice]
        dt_bias = self.dt_bias[self._v_head_slice]
        if _USE_GDN_GBETA:
            # Fused kernel handles the padding (g = 0 there) directly.
            g, beta = fused_g_beta(b, a, A_log, dt_bias, mask)
        else:
            beta = b.sigmoid()
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
            if mask is not None:
                # Padded rows: k/v are zero there (bias-free conv over zero
                # input) but g is NOT (-exp(A_log)·softplus(dt_bias) < 0) —
                # left unmasked it decays the recurrent state across the
                # padding tail, wiping short sequences' state in a
                # mixed-length batch.  g=0 makes trailing chunks exact
                # no-ops (beta is irrelevant: the state update is
                # proportional to k).
                g = g.masked_fill(~mask.unsqueeze(-1), 0.0)
        z = self.in_proj_z(x)                             # [n, T, value_dim]
        conv_w = self.conv1d.weight                       # [conv_dim, 1, K]
        K = self.conv_kernel

        def split_qkv(mixed, m):
            q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=1)
            q = q.transpose(1, 2).reshape(n, m, self.num_k_heads, self.head_k_dim)
            k = k.transpose(1, 2).reshape(n, m, self.num_k_heads, self.head_k_dim)
            v = v.transpose(1, 2).reshape(n, m, self.num_v_heads, self.head_v_dim)
            if self.num_v_heads > self.num_k_heads:
                rep = self.num_v_heads // self.num_k_heads
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            return q, k, v

        if T == 1:
            # Decode: roll the conv state and convolve the K-token window.
            mixed = self.in_proj_qkv(x).transpose(1, 2)   # [n, conv_dim, 1]
            window = torch.cat([self.conv_cache[ids], mixed], dim=-1)
            self.conv_cache[ids] = window[..., 1:]
            mixed = F.silu(F.conv1d(window, conv_w, groups=self.conv_dim))
            q, k, v = split_qkv(mixed, 1)
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
            seg_tokens = (self.PREFILL_SEGMENT_TOKENS_FLA if _USE_FLA_CHUNK
                          else self.PREFILL_SEGMENT_TOKENS_TORCH)
            seg = max(1, seg_tokens // n)
            conv_states = self.conv_cache[ids]            # [n, conv_dim, K-1]
            state = self.s_cache[ids]
            outs = []
            for t0 in range(0, T, seg):
                t1 = min(t0 + seg, T)
                m = t1 - t0
                mixed = self.in_proj_qkv(x[:, t0:t1]).transpose(1, 2)
                r = (lens - t0)
                valid = r > 0               # padding-only rows keep their state
                if _USE_CONV_TRITON:
                    # -1 skips the writeback for padding-only rows; the
                    # kernel rolls the state in place at each row's last
                    # real position (min(r, m)), replacing the cat + conv
                    # + window-gather of the reference path.
                    endp = r.clamp(min=-1, max=m).to(torch.int32)
                    mixed = causal_conv_silu(
                        mixed, conv_states, conv_w.view(self.conv_dim, K), endp)
                    self.conv_cache[ids[valid]] = conv_states[valid]
                else:
                    window = torch.cat([conv_states, mixed], dim=-1)
                    mixed = F.silu(F.conv1d(window, conv_w, groups=self.conv_dim))
                    starts = r.clamp(min=0, max=m)
                    cols = starts.view(n, 1, 1) + torch.arange(
                        K - 1, device=window.device, dtype=torch.int64)
                    sel = window.gather(-1, cols.expand(n, window.size(1), K - 1))
                    self.conv_cache[ids[valid]] = sel[valid]
                    conv_states = window[..., -(K - 1):]
                if mask is not None:
                    seg_mask = mask[:, t0:t1]
                    # Padded rows' conv output would see the real history
                    # through the sliding window (non-zero k/v) and pollute
                    # the recurrent state — zero them so padding is an
                    # exact no-op.
                    mixed = mixed * seg_mask.unsqueeze(1).to(mixed.dtype)
                q, k, v = split_qkv(mixed, m)
                out_i, state = chunk_gated_delta_rule(
                    q, k, v, g[:, t0:t1], beta[:, t0:t1], initial_state=state)
                outs.append(out_i)
            self.s_cache[ids] = state
            out = torch.cat(outs, dim=1)

        # Norm on the value head dim: keep the head dim explicit (weight is
        # [head_v_dim], shared across heads — matches the reference's
        # reshape(-1, head_v_dim) 2D norm).
        core = out.reshape(n, T, self.num_v_heads, self.head_v_dim)
        core = self.norm(core, z.view(n, T, self.num_v_heads, self.head_v_dim))
        core = core.reshape(n, T, self.value_dim)
        y = self.out_proj(core)                           # [n, T, H]
        return y


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
        if _USE_FI_NORM:
            x = _fi_act.silu_and_mul(gate_up)
        else:
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

    def __init__(self, config, vision_config=None, image_token_id=None) -> None:
        super().__init__()
        self.pp_first = get_pp_rank() == 0
        self.pp_last = get_pp_rank() == get_pp_size() - 1
        self.start_layer, self.end_layer = split_layers(
            config.num_hidden_layers, get_pp_size(), get_pp_rank())
        # Multimodal: the vision tower lives on the first pipeline stage
        # only, replicated over TP ranks (its output is scattered into the
        # all-reduced token embeddings, so every rank stays in agreement).
        self.visual = (Qwen3_5VisionModel(vision_config)
                       if self.pp_first and vision_config is not None else None)
        self.image_token_id = image_token_id
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
        if self.visual is None and ".visual." in weight_name:
            return True
        m = re.search(r"\.layers\.(\d+)\.", weight_name)
        return bool(m) and not (self.start_layer <= int(m.group(1)) < self.end_layer)

    def _scatter_vision_embeds(self, input_ids: torch.Tensor,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        """Replace image_token_id rows with vision-tower embeddings.

        context.vision_embeds (built by ModelRunner alongside the batch)
        holds per-seq (row range, embedding slice) pairs — the slice a
        chunk scatters is its prefix of pending image tokens, so chunked
        prefill crossing an image region stays row-aligned."""
        for row_start, row_end, embeds in get_context().vision_embeds:
            rows = hidden_states[row_start:row_end]
            mask = input_ids[row_start:row_end] == self.image_token_id
            rows[mask] = embeds
        return hidden_states

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
            if self.visual is not None:
                hidden_states = self._scatter_vision_embeds(input_ids, hidden_states)
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
    # and skip the MTP head.  ``visual`` (None for text-only builds, e.g.
    # NANOVLLM_QWEN35_TEXTONLY=1) keeps the vision tower weights loading.
    weight_remapping = (("model.language_model.", "model."),)
    ignored_weight_prefixes = ("mtp.",)

    def __init__(self, config, vision_config=None, image_token_id=None) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config, vision_config, image_token_id)
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
