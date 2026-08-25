"""Multimodal parity check: nano-vllm Qwen3.5 vs the transformers reference.

Replays the saved reference batch (tests/ref_mm/img_prompt.pt, produced by
tests/ref_mm/gen_reference.py against Qwen3_5ForConditionalGeneration)
through the engine as an image-bearing sequence and compares the greedy
continuation (8 tokens) with the reference generate().

max_num_batched_tokens=256 forces the 310-token prompt (294 image rows)
to prefill in two chunks with the boundary INSIDE the image region —
stressing the row-aligned vision-embed scatter and the MRoPE position
slices across chunks.  A second, mixed batch (image + text seq together)
checks batch invariance of the image path; the text prompt's own tokens
are only sanity-checked (a base model on an unterminated prompt has
near-tie argmax logits that legitimately flip with batch shape).
"""
import sys
import torch

sys.path.insert(0, ".")

from nanovllm import LLM, SamplingParams

MODEL = "/root/autodl-tmp/huggingface/Qwen3.5-2B"
TEXT = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is in this image?<|im_end|>\n<|im_start|>assistant\n")


def main():
    ref = torch.load("tests/ref_mm/img_prompt.pt", weights_only=False)
    ref_gen = ref["greedy8"].tolist()
    llm = LLM(model=MODEL, max_model_len=4096, max_num_batched_tokens=256,
              gpu_memory_utilization=0.85)
    # The sampler is gumbel-argmax over logits/T; 1e-5 makes it a
    # deterministic argmax (greedy) to compare with the reference.
    sp = SamplingParams(temperature=1e-5, max_tokens=8)

    out = llm.generate([{"prompt": TEXT, "images": ["tests/ref_mm/image.png"]}],
                       sampling_params=sp, use_tqdm=False)
    print(f"chunked-prefill image gen : {out[0]['token_ids']}")
    print(f"reference                 : {ref_gen}")
    assert out[0]["token_ids"] == ref_gen, "image generation mismatch"

    out2 = llm.generate(
        [{"prompt": TEXT, "images": ["tests/ref_mm/image.png"]},
         "Introduce yourself."],
        sampling_params=sp, use_tqdm=False)
    print(f"mixed-batch image gen     : {out2[0]['token_ids']}")
    assert out2[0]["token_ids"] == ref_gen, "batched image generation mismatch"
    assert len(out2[1]["token_ids"]) == 8, "text seq in mixed batch broken"
    print("text seq (sanity only)    :", out2[1]["token_ids"])

    print("MM CHECK PASSED")


if __name__ == "__main__":
    main()
