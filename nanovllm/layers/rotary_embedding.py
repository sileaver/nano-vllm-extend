from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    # rotary_dim < head_size = partial RoPE (only the leading rotary_dim
    # channels rotate; the rest pass through, e.g. Qwen3.5: 64 of 256).

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _rope_forward(
            self.cos_sin_cache, self.rotary_dim, self.head_size,
            positions, query, key,
        )


# Compiled free function with the cache/dims passed explicitly (see
# activation.py: instance-method compiles recompile per instance and per
# shape, exhausting the dynamo cache — every later CUDA graph bakes the
# eager fallback).
@torch.compile(dynamic=True)
def _rope_forward(
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    q_rot, k_rot = query[..., :rotary_dim], key[..., :rotary_dim]
    q_rot = apply_rotary_emb(q_rot, cos, sin)
    k_rot = apply_rotary_emb(k_rot, cos, sin)
    if rotary_dim < head_size:
        # Partial RoPE: the tail channels pass through unrotated.
        query = torch.cat((q_rot, query[..., rotary_dim:]), dim=-1)
        key = torch.cat((k_rot, key[..., rotary_dim:]), dim=-1)
    else:
        query, key = q_rot, k_rot
    return query, key


class MRotaryEmbedding(nn.Module):
    """Multi-section RoPE with the Qwen3.5 interleaved layout.

    Positions are interleaved across sections (T, H, W for multimodal
    position ids) as repeating [T, H, W] triplets: pair j keeps frequency
    inv_freq[j] but reads its position from section ``j % 3`` (the
    reference's ``apply_interleaved_mrope`` overwrites same slots across
    the section tables; with mrope_section [11, 11, 10] and 32 pairs the
    layout tiles exactly).

    ``positions`` may be [N] (all three sections share it — plain RoPE
    semantics, used by decode steps and CUDA-graph replay) or [3, N]
    (independent T/H/W positions, multimodal prefill chunks).
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        base_rope = RotaryEmbedding(head_size, rotary_dim,
                                    max_position_embeddings, base)
        self.register_buffer("cos_sin_cache", base_rope.cos_sin_cache,
                             persistent=False)
        # inv_freq drives the [3, N] path (freqs computed directly from the
        # per-pair section positions; same fp32 values as a cache lookup).
        self.register_buffer("inv_freq", base_rope.cos_sin_cache.new_empty(
            rotary_dim // 2), persistent=False)
        with torch.no_grad():
            self.inv_freq.copy_(1.0 / (base ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)))
        # Section id per rope pair: j % 3, validated against mrope_section.
        n = rotary_dim // 2
        src = torch.empty(n, dtype=torch.long)
        filled = torch.zeros(n, dtype=torch.bool)
        for s, count in enumerate(mrope_section):
            idx = torch.arange(s, count * 3, 3)
            idx = idx[idx < n]
            assert not filled[idx].any(), "overlapping mrope sections"
            src[idx] = s
            filled[idx] = True
        assert filled.all(), \
            f"mrope_section {mrope_section} does not tile {n} rope pairs"
        self.register_buffer("pair_src", src, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 1:
            # Uniform positions: identical to plain RoPE.
            return _rope_forward(
                self.cos_sin_cache, self.rotary_dim, self.head_size,
                positions, query, key,
            )
        return _mrope_forward(
            self.inv_freq, self.rotary_dim, self.head_size,
            self.pair_src,
            positions, query, key,
        )


@torch.compile(dynamic=True)
def _mrope_forward(
    inv_freq: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    pair_src: torch.Tensor,
    positions: torch.Tensor,          # [3, N]
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Per rope pair j: frequency inv_freq[j] with the POSITION of section
    # pair_src[j] (the reference interleaves positions, not frequencies —
    # apply_interleaved_mrope overwrites same slots across section tables).
    pp = positions[pair_src]                          # [pairs, N]
    freqs = pp * inv_freq.unsqueeze(1)                # [pairs, N]
    cos = freqs.cos().transpose(0, 1).unsqueeze(1)    # [N, 1, pairs]
    sin = freqs.sin().transpose(0, 1).unsqueeze(1)

    def apply(x):
        x_rot = x[..., :rotary_dim].float()
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        y = torch.cat((y1, y2), dim=-1).to(x.dtype)
        if rotary_dim < head_size:                        # partial RoPE
            y = torch.cat((y, x[..., rotary_dim:]), dim=-1)
        return y

    return apply(query), apply(key)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    mrope_section: tuple[int, ...] | None = None,
):
    if mrope_section is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    else:
        rotary_emb = MRotaryEmbedding(head_size, rotary_dim, max_position,
                                      base, mrope_section)
    return rotary_emb
