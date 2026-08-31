import os

from nanovllm import LLM, SamplingParams

# Any Qwen3.5 checkpoint works (multimodal shell or not); override with
# QWEN35_MODEL=/path if it does not live in ~/huggingface.
MODEL = os.environ.get("QWEN35_MODEL", os.path.expanduser("~/huggingface/Qwen3.5-2B"))


def main():
    # Any Qwen3.5 checkpoint ships the full multimodal shell; the engine
    # loads the vision tower alongside the language model (set
    # NANOVLLM_QWEN35_TEXTONLY=1 to skip it).
    llm = LLM(model=MODEL,
              max_model_len=4096, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # Multimodal prompts are dicts: the text carries one
    # <|vision_start|><|image_pad|><|vision_end|> placeholder per image
    # (the processor expands each to the image's grid tokens), and
    # "images" lists PIL images / paths / urls in order.
    prompts = [
        {
            "prompt": ("<|im_start|>user\n"
                       "<|vision_start|><|image_pad|><|vision_end|>"
                       "Describe this image in one sentence.<|im_end|>\n"
                       "<|im_start|>assistant\n"),
            "images": ["assets/example-image.jpg"],
        },
        # Plain text prompts work on the same engine.
        "The capital of France is",
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    for output in outputs:
        print(llm.tokenizer.decode(output["token_ids"]))


if __name__ == "__main__":
    main()
