"""Stage-exact parity of the ported Qwen3.5 vision tower vs transformers.

Builds both towers in fp32 and compares patch embed, the interpolated
pos-embeds, the 2D rope tables, every block (identical exact-math
attention applied to both sides, isolating layout/weight correctness
from backend numerics) and the merger.  Every stage must be bit-exact;
the bf16 flash-attn path is covered end-to-end by mm_check_qwen35.py.

Usage: python tests/mm_vision_parity_qwen35.py
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("QWEN35_MODEL", os.path.expanduser("~/huggingface/Qwen3.5-2B"))
HERE = os.path.dirname(os.path.abspath(__file__))
from nanovllm.models.qwen3_5_vision import (
    Qwen3_5VisionModel, _pos_embed_interp_indices, _vision_position_ids,
    _vision_cu_seqlens)
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
from transformers.vision_utils import get_vision_interpolation_indices_and_weights
from nanovllm.utils.loader import load_model

torch.set_grad_enabled(False)
ref_data = torch.load(os.path.join(HERE, "ref_mm/img_prompt.pt"), weights_only=False)
vcfg = AutoConfig.from_pretrained(MODEL).vision_config
full = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float32).eval().cuda()
vis = full.model.visual
pv = ref_data["pixel_values"].cuda().float()
grid = ref_data["image_grid_thw"].cuda()
grids = tuple(tuple(g) for g in grid.tolist())

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
ours = Qwen3_5VisionModel(vcfg)
ours.weight_remapping = (("model.visual.", ""),)
ours.ignored_weight_prefixes = ("mtp.", "model.language_model.", "lm_head.")
load_model(ours, MODEL)

def math_attn(attn_mod, x, cu, cos, sin, num_heads, head_dim):
    total = x.size(0)
    qkv = attn_mod.qkv(x).view(total, 3, num_heads, head_dim)
    q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
    q1, q2 = q[..., :cos.size(-1)], q[..., cos.size(-1):]
    k1, k2 = k[..., :cos.size(-1)], k[..., cos.size(-1):]
    q = torch.cat((q1*cos.unsqueeze(1) - q2*sin.unsqueeze(1),
                   q2*cos.unsqueeze(1) + q1*sin.unsqueeze(1)), -1)
    k = torch.cat((k1*cos.unsqueeze(1) - k2*sin.unsqueeze(1),
                   k2*cos.unsqueeze(1) + k1*sin.unsqueeze(1)), -1)
    outs = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        o = F.scaled_dot_product_attention(
            q[a:b].transpose(0,1)[None], k[a:b].transpose(0,1)[None],
            v[a:b].transpose(0,1)[None])[0]
        outs.append(o.transpose(0, 1))
    return attn_mod.proj(torch.cat(outs).reshape(total, -1))

# ── stage 1: patch embed ──
hf_x = vis.patch_embed(pv)
my_x = ours.patch_embed(pv)
DIFFS = [(hf_x - my_x).abs().max().item()]

# ── stage 2: pos embeds ──
ii, iw = get_vision_interpolation_indices_and_weights(
    grid, num_grid_per_side=vis.num_grid_per_side, mode="bilinear",
    align_corners=True, spatial_merge_size=vis.spatial_merge_size)
hf_pos = (vis.pos_embed(ii.cuda()) * iw.cuda()[:, :, None]).sum(1)
mi, mw = _pos_embed_interp_indices(grids, ours.num_grid_per_side, ours.spatial_merge_size)
my_pos = (ours.pos_embed(mi.cuda()) * mw.cuda()[:, :, None]).sum(1)
DIFFS.append((hf_pos - my_pos).abs().max().item())

# ── stage 3: rope tables ──
pids = _vision_position_ids(grids, ours.spatial_merge_size).cuda()
my_cos, my_sin = ours.rotary_pos_emb(pids)
emb = torch.cat([vis.rotary_pos_emb(pids)] * 2, dim=-1)
hf_cos, hf_sin = emb.cos()[..., :32], emb.sin()[..., :32]
DIFFS += [(hf_cos - my_cos).abs().max().item(), (hf_sin - my_sin).abs().max().item()]

# ── stage 4: blocks with identical math attention on both towers ──
from transformers.vision_utils import get_vision_attention_seqlens
hf_cu, _ = get_vision_attention_seqlens(grid, vcfg)
my_cu, my_max = _vision_cu_seqlens(grids)
hf_cu = hf_cu.cuda()
my_cu2 = my_cu.cuda()
assert torch.equal(hf_cu.cpu(), my_cu.cpu()), "cu_seqlens mismatch"

hf_x2, my_x2 = hf_x + hf_pos.to(hf_x.dtype), my_x + my_pos.to(my_x.dtype)
for i, (hb, mb) in enumerate(zip(vis.blocks, ours.blocks)):
    h = hb.norm1(hf_x2)
    h = hf_x2 + math_attn(hb.attn, h, hf_cu, hf_cos, hf_sin, hb.attn.num_heads, hb.attn.head_dim)
    h = h + hb.mlp(hb.norm2(h))
    m = mb.norm1(my_x2)
    m = my_x2 + math_attn(mb.attn, m, my_cu2, my_cos, my_sin, mb.attn.num_heads, mb.attn.head_dim)
    m = m + mb.mlp(mb.norm2(m))
    DIFFS.append((h - m).abs().max().item())
    hf_x2, my_x2 = h, m

hf_m = vis.merger(hf_x2)
my_m = ours.merger(my_x2)
DIFFS.append((hf_m - my_m).abs().max().item())

assert max(DIFFS) == 0.0, f"vision tower parity failed: {DIFFS}"
print(f"VISION TOWER PARITY OK ({len(DIFFS)} stages bit-exact)")
