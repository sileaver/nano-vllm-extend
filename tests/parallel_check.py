"""Parallel-strategy equivalence check for nano-vllm.

Runs the same prompt batch through the engine at a given parallel layout
and prints the sampled tokens, so layouts can be diffed against the
single-GPU reference:

    OMP_NUM_THREADS=8 python tests/parallel_check.py single
    OMP_NUM_THREADS=8 python tests/parallel_check.py tp2
    OMP_NUM_THREADS=8 python tests/parallel_check.py pp2
    OMP_NUM_THREADS=8 python tests/parallel_check.py dp2
    OMP_NUM_THREADS=8 python tests/parallel_check.py tp2eager   # TP + enforce_eager

Low temperature (0.01) makes sampling effectively argmax-greedy, so token
streams across layouts should match ~exactly (rare near-ties may flip).
"""
import sys
from random import randint, seed

from nanovllm import LLM, SamplingParams

MODEL = "/root/huggingface/Qwen3.5-2B"
NUM_SEQS = 8
MAX_INPUT = 256
MAX_OUTPUT = 32


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    kwargs = dict(max_model_len=2048, enforce_eager=True)
    if mode == "single":
        pass
    elif mode == "tp2":
        kwargs |= dict(tensor_parallel_size=2, enforce_eager=False)
    elif mode == "tp2eager":
        kwargs |= dict(tensor_parallel_size=2)
    elif mode == "pp2":
        kwargs |= dict(pipeline_parallel_size=2, enforce_eager=False)
    elif mode == "pp2eager":
        kwargs |= dict(pipeline_parallel_size=2)
    elif mode == "dp2":
        kwargs |= dict(data_parallel_size=2)
    else:
        raise SystemExit(f"unknown mode {mode}")

    seed(0)
    prompts = [[randint(0, 10000) for _ in range(randint(64, MAX_INPUT))]
               for _ in range(NUM_SEQS)]
    sps = [SamplingParams(temperature=0.01, ignore_eos=True, max_tokens=MAX_OUTPUT)
           for _ in range(NUM_SEQS)]

    llm = LLM(model, **kwargs)
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    for i, out in enumerate(outputs):
        print(f"[{i}] {out['token_ids']}")
    # Compact fingerprint for diffing runs
    import hashlib
    h = hashlib.sha1(
        repr([o["token_ids"] for o in outputs]).encode()).hexdigest()[:16]
    print(f"MODE={mode} FINGERPRINT={h}")


if __name__ == "__main__":
    main()
