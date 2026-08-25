"""CPU-side multimodal preprocessing for Qwen3.5 (single sequence).

Port of transformers' Qwen3_5Model.get_rope_index: builds the [3, len]
T/H/W position table for one prompt from its modality runs, plus the
decode-time ``rope_delta`` offset (llm_positions.max() + 1 - len — MRoPE
positions advance by max(h, w) over an image region, not by its token
count, so text after an image sits ``delta`` below its token offset).
"""
import torch


def qwen35_mrope_positions(
    token_ids: list[int],
    mm_token_type_ids: list[int],
    image_grid_thw: list[list[int]],
    spatial_merge_size: int = 2,
) -> tuple[torch.Tensor, int]:
    """token_ids / mm_token_type_ids: one prompt (0 = text, 1 = image,
    2 = video); image_grid_thw: per-image patch grids [t, h, w].  Returns
    ([3, len] int64 positions, rope_delta).  Video entries consume one
    grid per contiguous run (the reference splits timestamped clips; this
    engine takes images only, so the simple mapping suffices)."""
    positions = []
    grid_iter = iter(image_grid_thw)
    current_pos = 0
    n = len(token_ids)
    i = 0
    while i < n:
        j = i
        while j < n and mm_token_type_ids[j] == mm_token_type_ids[i]:
            j += 1
        if mm_token_type_ids[i] == 0:
            seg = torch.arange(current_pos, current_pos + j - i)
            positions.append(seg.unsqueeze(0).expand(3, -1))
            current_pos += j - i
        else:
            grid_t, grid_h, grid_w = next(grid_iter)
            h, w = grid_h // spatial_merge_size, grid_w // spatial_merge_size
            pos_t = torch.arange(grid_t)
            pos_h = torch.arange(h) + current_pos
            pos_w = torch.arange(w) + current_pos
            T, H, W = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
            vis = torch.stack([T, H, W]).reshape(3, -1)
            vis[0] += current_pos          # after the time_interval product
            positions.append(vis)
            current_pos += max(grid_h, grid_w) // spatial_merge_size
        i = j
    pos = torch.cat(positions, dim=1)
    rope_delta = int(pos.max()) + 1 - n
    return pos, rope_delta


def mm_token_types_from_ids(token_ids: list[int], image_token_id: int,
                            video_token_id: int | None = None) -> list[int]:
    """Derive mm_token_type_ids from the prompt when the processor did not
    emit them: 1 over image-pad runs, 2 over video-pad runs, else 0."""
    types = []
    for t in token_ids:
        if t == image_token_id:
            types.append(1)
        elif video_token_id is not None and t == video_token_id:
            types.append(2)
        else:
            types.append(0)
    return types
