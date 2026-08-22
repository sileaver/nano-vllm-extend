"""DFlash draft model (z-lab/Qwen3-4B-DFlash-b16, block diffusion spec draft).

The draft is a 5-layer Qwen3-shaped transformer whose attention is
*non-causal*: the query comes from the current block (anchor + mask
tokens), while keys/values concatenate the context feature (target
hidden states extracted from 5 target layers, projected by ``fc``) with
the block's own hidden states.  One forward pass over block_size tokens
proposes block_size-1 draft tokens at once.

Parameter names follow the original safetensors layout (split q/k/v
projections, gate/up/down MLP) so ``utils.loader.load_model`` matches
them directly.
"""
import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear, RowParallelLinear
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import apply_rotary_emb, get_rope


class DFlashAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 rms_norm_eps, rope_theta, max_position):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.q_proj = ReplicatedLinear(hidden_size, num_heads * head_dim)
        self.k_proj = ReplicatedLinear(hidden_size, num_kv_heads * head_dim)
        self.v_proj = ReplicatedLinear(hidden_size, num_kv_heads * head_dim)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = get_rope(head_dim, head_dim, max_position, rope_theta)

    def forward(self, hidden_states, ctx_hidden, q_pos, k_pos, attn_mask):
        """hidden_states [bs, T, H]; ctx_hidden [bs, C, H] (padded to C_max).

        q_pos [bs, T], k_pos [bs, C+T]: absolute RoPE positions (the query
        uses the block positions, the keys use [start-C, start+T) so the
        context rows keep their true history positions, as in the official
        implementation).  attn_mask [bs, 1, T, C+T] bool, True = attend —
        masks the padding region of the context.
        """
        bs, T, H = hidden_states.shape
        C = ctx_hidden.shape[1]
        q = self.q_proj(hidden_states.reshape(-1, H)).view(bs, T, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k_ctx = self.k_proj(ctx_hidden.reshape(-1, H)).view(bs, C, self.num_kv_heads, self.head_dim)
        k_noise = self.k_proj(hidden_states.reshape(-1, H)).view(bs, T, self.num_kv_heads, self.head_dim)
        k = torch.cat([k_ctx, k_noise], dim=1)          # [bs, C+T, kv, dim]
        v_ctx = self.v_proj(ctx_hidden.reshape(-1, H)).view(bs, C, self.num_kv_heads, self.head_dim)
        v_noise = self.v_proj(hidden_states.reshape(-1, H)).view(bs, T, self.num_kv_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1)
        k = self.k_norm(k)

        cache = self.rotary_emb.cos_sin_cache            # [max_pos, 1, 2*dim]
        q_cos, q_sin = cache[q_pos].chunk(2, dim=-1)    # [bs, T, 1, dim]
        k_cos, k_sin = cache[k_pos].chunk(2, dim=-1)    # [bs, C+T, 1, dim]
        q = apply_rotary_emb(q, q_cos, q_sin)
        k = apply_rotary_emb(k, k_cos, k_sin)

        # Non-causal attention (causal=False in the official model).  The
        # GQA groups are expanded explicitly so the query, key and mask all
        # carry num_heads heads (SDPA's masked path insists on aligned
        # head counts).
        n_groups = self.num_heads // self.num_kv_heads
        k = k.transpose(1, 2).repeat_interleave(n_groups, dim=1)  # [bs, H, C+T, dim]
        v = v.transpose(1, 2).repeat_interleave(n_groups, dim=1)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k, v,
            attn_mask=attn_mask.expand(-1, self.num_heads, -1, -1),
            scale=self.scaling, is_causal=False)
        o = o.transpose(1, 2).reshape(bs, T, -1)
        return self.o_proj(o.reshape(-1, o.shape[-1])).view(bs, T, H)


class DFlashMLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x):
        g = F.silu(self.gate_proj(x))
        return self.down_proj(g * self.up_proj(x))


class DFlashDraftLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rms_norm_eps, rope_theta, max_position):
        super().__init__()
        self.self_attn = DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            rms_norm_eps, rope_theta, max_position)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, ctx_hidden, q_pos, k_pos, attn_mask):
        # Standard residual chain (not fused), as in the official model.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ctx_hidden, q_pos, k_pos, attn_mask)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class DFlashDraftModel(nn.Module):

    def __init__(self, hf_config):
        super().__init__()
        self.block_size = hf_config.block_size
        dflash_cfg = hf_config.dflash_config
        self.mask_token_id = dflash_cfg["mask_token_id"]
        self.target_layer_ids = list(dflash_cfg["target_layer_ids"])
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        # transformers >= 5.x moves rope_theta into rope_parameters.
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = hf_config.rope_parameters.get("rope_theta", 10000.0)
        self.layers = nn.ModuleList([
            DFlashDraftLayer(
                hf_config.hidden_size, hf_config.num_attention_heads,
                hf_config.num_key_value_heads, head_dim,
                hf_config.intermediate_size, hf_config.rms_norm_eps,
                rope_theta, hf_config.max_position_embeddings)
            for _ in range(hf_config.num_hidden_layers)
        ])
        self.fc = ReplicatedLinear(len(self.target_layer_ids) * hf_config.hidden_size,
                                   hf_config.hidden_size)
        self.hidden_norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)

    def forward(self, noise_embedding, ctx_hidden, ctx_lens, q_pos, k_pos, attn_mask):
        """One parallel draft over block_size rows per sequence.

        noise_embedding [bs*T, H]: embeddings of the block (anchor + masks).
        ctx_hidden [bs, C_max, n_layers*H]: padded target context features.
        ctx_lens [bs]: true context lengths.  Returns [bs, T, H].
        """
        bs, C_max, _ = ctx_hidden.shape
        T = self.block_size
        hidden_states = noise_embedding.view(bs, T, -1)
        ctx = self.hidden_norm(self.fc(ctx_hidden.reshape(-1, ctx_hidden.shape[-1])))
        ctx = ctx.view(bs, C_max, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, ctx, q_pos, k_pos, attn_mask)
        return self.norm(hidden_states.reshape(-1, hidden_states.shape[-1])).view(bs, T, -1)
