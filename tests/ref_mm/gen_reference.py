"""Regenerate the multimodal parity reference (tests/ref_mm/*).

Loads Qwen3_5ForConditionalGeneration from transformers, runs one image
prompt (prefill logits + 8 greedy tokens), and saves the processed inputs
plus the source image so tests/mm_check_qwen35.py can replay the exact
batch through nano-vllm.
"""
import sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL = "/root/autodl-tmp/huggingface/Qwen3.5-2B"
OUT = "tests/ref_mm"

rng = np.random.default_rng(42)
img = Image.fromarray(rng.integers(0, 256, (448, 672, 3), dtype=np.uint8))
img.save(f"{OUT}/image.png")

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16).eval().cuda()
proc = AutoProcessor.from_pretrained(MODEL)
text = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is in this image?<|im_end|>\n<|im_start|>assistant\n")
out = proc(text=[text], images=[img], return_tensors="pt")
out = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in out.items()}

with torch.inference_mode():
    pos = model.model.compute_3d_position_ids(
        out["input_ids"], inputs_embeds=None,
        image_grid_thw=out["image_grid_thw"],
        mm_token_type_ids=out["mm_token_type_ids"])
    logits = model(**out, position_ids=pos).logits
    gen = model.generate(**out, position_ids=pos, max_new_tokens=8,
                         do_sample=False)

torch.save({
    "logits_all": logits[0].float().cpu(),
    "position_ids": pos.cpu(),
    "rope_deltas": model.model.rope_deltas.cpu(),
    "greedy8": gen[0, -8:].cpu(),
    "input_ids": out["input_ids"].cpu(),
    "pixel_values": out["pixel_values"].cpu(),
    "image_grid_thw": out["image_grid_thw"].cpu(),
    "mm_token_type_ids": out["mm_token_type_ids"].cpu(),
}, f"{OUT}/img_prompt.pt")
print("reference saved:", {k: tuple(v.shape) for k, v in torch.load(f"{OUT}/img_prompt.pt").items() if hasattr(v, "shape")})
