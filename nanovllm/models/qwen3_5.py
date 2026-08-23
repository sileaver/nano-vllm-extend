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
"""
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context

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

# Kernel selection (env-overridable A/B switches).  Measured on RTX 5080
# (bench_qwen35, 64-seq mixed load, cudagraph): the triton recurrent wins
# inside the decode graph (no launch overhead, less GPU time), but the
# triton chunk's ~0.56ms/call Python launcher overhead makes eager prefill
# CPU-bound and 44% slower end-to-end despite 4.7x faster kernels — so the
# torch chunk stays the default.  FLA_CHUNK=1 / FLA_RECURRENT=0 override.
import os as _os
_USE_FLA_RECURRENT = _HAS_FLA and _os.environ.get("FLA_RECURRENT", "1") == "1"
_USE_FLA_CHUNK = _HAS_FLA and _os.environ.get("FLA_CHUNK", "0") == "1"


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


@torch.compile
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
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(self.weight.dtype)

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        orig_dtype = x.dtype
        residual = (x.float() + residual.float()).to(orig_dtype)
        x = residual.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(orig_dtype), residual

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

    @torch.compile
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor):
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


class Qwen3_5Attention(nn.Module):
    """Full-attention layer (every full_attention_interval-th layer).

    q_proj outputs 2x width: per head, the first head_dim channels are the
    query and the second head_dim the output gate (attn_output *=
    sigmoid(gate)).  RoPE is partial: only the first rotary_dim = 64 of the
    256 head channels are rotated.
    """

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
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
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        assert self.num_v_heads % self.num_k_heads == 0, \
            "qwen3_5: v/k head replication not supported (2B has 16/16)"
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = ColumnParallelLinear(
            self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size, self.num_v_heads, bias=False)
        # padding is applied per-call (0 or kernel-1 depending on state)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.conv_kernel,
            groups=self.conv_dim, padding=0, bias=False)
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)
        # State-pool views bound by ModelRunner (index by slot id).
        self.s_cache: torch.Tensor | None = None      # [slots, H, K, V] fp32
        self.conv_cache: torch.Tensor | None = None   # [slots, conv_dim, kernel-1]

    # Chunk-kernel call budget (padded tokens per call): the pure-torch
    # kernel's fp32 intermediates need ~150MB per 1k padded tokens, so one
    # call over a full 16k-token engine batch demands multi-GB transient
    # peaks — with a large KV pool that leaves no headroom and the
    # allocator livelocks in OOM-retry.  Segments thread the recurrent/
    # conv state, so splitting is exactly equivalent to one call.
    PREFILL_SEGMENT_TOKENS = 4096

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
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
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
            out, new_s = recurrent_gated_delta_rule(q, k, v, g, beta, self.s_cache[ids])
            self.s_cache[ids] = new_s
        else:
            # Prefill / chunk continuation, in T-segments (see
            # PREFILL_SEGMENT_TOKENS).  The conv state threads across
            # segments; a zero state on the first segment of a fresh prefill
            # is identical to the causal conv's zero left-padding.
            z = self.in_proj_z(x)                         # [bs, T, value_dim]
            seg = max(1, self.PREFILL_SEGMENT_TOKENS // bs)
            conv_states = self.conv_cache[ids]            # [bs, conv_dim, k-1]
            state = self.s_cache[ids]
            outs = []
            for t0 in range(0, T, seg):
                t1 = min(t0 + seg, T)
                n = t1 - t0
                mixed = self.in_proj_qkv(x[:, t0:t1]).transpose(1, 2)
                window = torch.cat([conv_states, mixed], dim=-1)
                mixed = F.silu(F.conv1d(window, conv_w, groups=self.conv_dim))
                # Per-seq conv-state writeback at the latest real position:
                # window layout is [state(k-1), x(n)].
                for b_i in range(bs):
                    r = int(lens[b_i]) - t0
                    if r >= n:
                        self.conv_cache[ids[b_i]] = window[b_i, :, -3:]
                    elif r > 0:
                        self.conv_cache[ids[b_i]] = window[b_i, :, r:r + 3]
                    # r <= 0: padding-only rows for this seq — keep its state
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
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        skip_layers: tuple[int, ...] | None = None,
        output_layer_hidden: list[int] | None = None,
    ):
        # skip_layers / output_layer_hidden exist for signature parity with
        # Qwen3ForCausalLM (speculative-decoding hooks); the qwen3_5 config
        # asserts speculative decoding off, so they are ignored here.
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
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
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        skip_layers: tuple[int, ...] | None = None,
        output_layer_hidden: list[int] | None = None,
    ):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def compute_all_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Logits for every row (speculative verification needs K+1 rows
        per seq; the regular head gathers only the last row per seq)."""
        return self.lm_head.forward_all(hidden_states)
