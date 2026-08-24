import re
import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.parallel import get_pp_rank, get_pp_size, get_tp_size


class PipelineLayerShell(nn.Module):
    """Placeholder for decoder layers owned by other pipeline stages.

    Keeps ``model.layers.<i>`` name slots aligned with the checkpoint so
    weight loading maps unchanged; holds no parameters and never runs."""
    def forward(self, *args, **kwargs):
        raise RuntimeError("pipeline layer shell must not be executed")


def split_layers(num_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    """Contiguous, balanced layer range for one pipeline stage."""
    per_stage = (num_layers + pp_size - 1) // pp_size
    start = pp_rank * per_stage
    return start, min(num_layers, start + per_stage)


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
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
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # transformers >= 5.x moves rope_theta into rope_parameters.
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = config.rope_parameters.get("rope_theta", 1000000.0)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
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
            Qwen3DecoderLayer(config) if self.start_layer <= i < self.end_layer
            else PipelineLayerShell()
            for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        layer_hidden = []
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, PipelineLayerShell):
                continue
            if skip_layers is not None and idx in skip_layers:
                # Layer-skipped draft (self-speculative).  The fused
                # residual chain (hidden_states/residual) is untouched, so
                # the skipped layer's input feeds the next layer unchanged.
                # NOTE: measured on Qwen3-0.6B, skipping even 1 layer drops
                # the draft-vs-target match rate from ~0.5 to ~0.14 — the
                # model has no layer redundancy.  Kept as a general model
                # capability; the speculative path uses Jacobi drafts.
                continue
            hidden_states, residual = layer(positions, hidden_states, residual)
            if output_layer_hidden is not None and idx in output_layer_hidden:
                # DFlash draft context: collect selected layer outputs
                # (0-indexed, matching the target_layer_ids semantics).
                # In the fused chain the residual accumulates attention AND
                # previous MLP outputs (both layernorms fold their input
                # into the residual), so residual + hidden_states (this
                # layer's raw MLP output) is the full residual sum — the
                # layer output as transformers reports it.
                layer_hidden.append(hidden_states + residual)
        if not self.pp_last:
            # Intermediate pipeline stage: hand the fused-residual chain on.
            return hidden_states, residual
        hidden_states, _ = self.norm(hidden_states, residual)
        if output_layer_hidden is not None:
            return hidden_states, torch.cat(layer_hidden, dim=-1)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
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
        return self.model(input_ids, positions, skip_layers, output_layer_hidden,
                          hidden_states, residual)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.lm_head is not None, "compute_logits on a non-last pipeline stage"
        return self.lm_head(hidden_states)

    def compute_all_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Logits for *every* row (speculative verification needs K+1 rows
        per seq; the regular head gathers only the last row per seq)."""
        return self.lm_head.forward_all(hidden_states)
