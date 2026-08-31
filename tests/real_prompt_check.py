"""Real-prompt token-level equivalence: single vs a parallel layout.

Random-token prompts have flat logits (near-ties flip under any change of
reduction order); real language has wide argmax gaps, so this check
separates "numerically equivalent" from "broken sharding".

    OMP_NUM_THREADS=8 python tests/real_prompt_check.py [single|tp2|pp2|dp2] [model]
"""
import sys

from nanovllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1492, Columbus sailed",
    "The three primary colors are",
    "Photosynthesis converts sunlight into",
    "The largest planet in the solar system is",
    "A triangle has",
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/huggingface/Qwen3.5-2B")
    # T=1e-6 ~ true greedy: gumbel noise (O(1)) is negligible against
    # logits/T (O(1e6)), so only actual argmax changes flip tokens.
    sp = SamplingParams(temperature=1e-6, max_tokens=24)
    kw = dict(max_model_len=512, enforce_eager=True)
    if mode == "tp2":
        kw |= dict(tensor_parallel_size=2)
    elif mode == "pp2":
        kw |= dict(pipeline_parallel_size=2)
    elif mode == "dp2":
        kw |= dict(data_parallel_size=2)
    llm = LLM(model, **kw)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    for i, o in enumerate(outs):
        print(f"[{i}] {o['token_ids']}")


if __name__ == "__main__":
    main()
