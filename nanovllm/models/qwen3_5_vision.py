"""Qwen3.5 vision tower (Qwen3-VL style ViT) for nano-vllm.

Port of transformers' Qwen3_5VisionModel — the ``model.visual.*`` half of
the Qwen3_5ForConditionalGeneration checkpoint: a Conv3d patch embed, a
learned 48x48 position-embedding table bilinearly resampled to each
image's patch grid (align_corners=True), 24 pre-LN blocks with fused-qkv
full (non-causal, per-frame) attention under 2D RoPE, and a patch merger
projecting 2x2 patch groups to the language model's hidden size.

Patch order: the Qwen3-VL processor emits pixel patches in spatial-merge
block order ((block_row, block_col, in_row, in_col)), and every position
table here (pos-embed interpolation indices, 2D rope ids, the merger's
4-patch grouping) is built in the same order — verified empirically
against the processor and against the reference tower end to end.

The tower is replicated across TP ranks (every rank holds full visual
weights; embedding scatter happens after VocabParallelEmbedding's
all-reduce, so all ranks agree) and lives only on the first pipeline
stage.  Attention runs through the same flash_attn varlen kernel the
text model uses, with cu_seqlens = one segment per video frame
(images: the whole image).
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.attention import flash_attn_varlen_func


# ── Grid-derived tables (ports of transformers.vision_utils helpers) ──
# All keyed by the grid tuple and cached: a prompt's chunks and the 24
# blocks of one forward reuse the same tables.

def _interp_axis_taps(index: torch.Tensor, size: torch.Tensor, side: int,
                      align_corners: bool = True):
    """Bilinear taps into a side-length table for target positions on an
    axis of per-element length ``size`` (closed form of
    torch.linspace(0, side-1, size)[index] when align_corners)."""
    index = index.to(torch.float32)
    if align_corners:
        src = index * (side - 1) / torch.clamp(size - 1, min=1)
    else:
        src = (index + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    taps = (floor.long()[:, None] + torch.arange(0, 2)).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - torch.arange(0, 2)).abs()
    weights = (1 - distance).clamp(min=0)
    return taps, weights


@lru_cache(8)
def _pos_embed_interp_indices(grids: tuple[tuple[int, int, int], ...],
                              side: int, merge: int):
    """Per-patch gather indices/weights resampling the square learned
    [side, side] pos-embed table to each image's (h, w) patch grid.
    Patches are decoded from their flat merge-block position into
    (row, col), matching the processor's patch order."""
    grid_thw = torch.tensor(grids)
    counts = grid_thw.prod(-1)
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    total = int(counts.sum())
    within = (torch.arange(total) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_w = _interp_axis_taps(row, heights, side)
    w_taps, w_w = _interp_axis_taps(col, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices, weights.to(torch.float32)


@lru_cache(8)
def _vision_position_ids(grids: tuple[tuple[int, int, int], ...], merge: int):
    """(h, w) index per patch for the 2D rope, laid out block-major over
    merge x merge blocks (get_vision_position_ids in transformers)."""
    position_ids = []
    for t, h, w in grids:
        hpos, wpos = torch.meshgrid(torch.arange(h), torch.arange(w),
                                    indexing="ij")
        block_shape = (h // merge, merge, w // merge, merge)
        hpos = hpos.reshape(block_shape).transpose(1, 2).flatten()
        wpos = wpos.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


@lru_cache(8)
def _vision_cu_seqlens(grids: tuple[tuple[int, int, int], ...]):
    """Cumulative patch counts, one segment per frame (images: the whole
    image; videos: frames attend independently)."""
    seqlens = []
    for t, h, w in grids:
        seqlens.extend([h * w] * t)
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return torch.tensor(cu, dtype=torch.int32), max(seqlens)


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """2D rope over the vision head: the head_dim//2-wide pair table takes
    its first half of frequencies from the h index and the second half
    from the w index (theta 10000)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids [N, 2] -> freqs [N, dim] (h pairs then w pairs);
        # cos/sin stay 32-wide — apply_rotary_emb pairs channel i with
        # i + dim/2, which is the reference's rotate_half over cat(f, f).
        freqs = (position_ids.unsqueeze(-1).float() * self.inv_freq).flatten(1)
        return freqs.cos(), freqs.sin()


class Qwen3_5VisionAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen: int, cos: torch.Tensor, sin: torch.Tensor):
        total = x.size(0)
        qkv = self.qkv(x).view(total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        # 2D rope (fp32 math, single bf16 rounding — reference semantics).
        q1, q2 = q[..., :cos.size(-1)].float(), q[..., cos.size(-1):].float()
        k1, k2 = k[..., :cos.size(-1)].float(), k[..., cos.size(-1):].float()
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        q = torch.cat((q1 * c - q2 * s, q2 * c + q1 * s), -1).to(x.dtype)
        k = torch.cat((k1 * c - k2 * s, k2 * c + k1 * s), -1).to(x.dtype)
        # Full (non-causal) attention per frame: flash_attn varlen over the
        # packed patch axis (3D [total, heads, dim] input, cu per frame).
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen, cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen, cu_seqlens_k=cu_seqlens,
            softmax_scale=self.scaling, causal=False,
        )
        o = o.reshape(total, -1)
        return self.proj(o)


class Qwen3_5VisionMLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3_5VisionBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(self, x, cu_seqlens, max_seqlen, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_5VisionPatchEmbed(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values [total, C*T*P*P] -> [total, hidden]
        x = pixel_values.view(-1, self.proj.in_channels, *self.proj.kernel_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.proj.out_channels)


class Qwen3_5VisionPatchMerger(nn.Module):
    """2x2 spatial merge: norm at patch resolution, then group 4 adjacent
    (merge-block ordered) patches and project to the LM hidden size."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3_5VisionModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings,
                                      config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3_5VisionBlock(config) for _ in range(config.depth))
        self.merger = Qwen3_5VisionPatchMerger(config)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """pixel_values [total_patches, C*T*P*P], grid_thw [n, 3] (t, h, w
        at patch resolution).  Returns merged embeddings
        [total_patches / merge**2, out_hidden_size] in merge-block order
        — the rows that replace image_token_id tokens in the LM input."""
        grids = tuple(tuple(g) for g in grid_thw.tolist())
        device = pixel_values.device
        interp_idx, interp_w = _pos_embed_interp_indices(
            grids, self.num_grid_per_side, self.spatial_merge_size)
        pos_ids = _vision_position_ids(grids, self.spatial_merge_size).to(device)
        cu_seqlens, max_seqlen = _vision_cu_seqlens(grids)
        cu_seqlens = cu_seqlens.to(device)

        x = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(interp_idx.to(device))
                      * interp_w.to(device)[:, :, None]).sum(1)
        x = x + pos_embeds.to(x.dtype)
        cos, sin = self.rotary_pos_emb(pos_ids)
        for blk in self.blocks:
            x = blk(x, cu_seqlens, max_seqlen, cos, sin)
        return self.merger(x)
