"""Throughput bench across parallel layouts (same workload as bench1-style).

    OMP_NUM_THREADS=8 python tests/bench_parallel.py [single|tp2|pp2|dp2] [model] [num_seqs]
"""
import sys
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

MODES = {
    "single": {},
    "tp2": dict(tensor_parallel_size=2),
    "pp2": dict(pipeline_parallel_size=2),
    "dp2": dict(data_parallel_size=2),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/huggingface/Qwen3.5-2B")
    num_seqs = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    seed(0)
    max_input_len, max_ouput_len = 1024, 1024

    llm = LLM(model, enforce_eager=False, max_model_len=4096, **MODES[mode])
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, max_ouput_len))
                       for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())  # warmup
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"MODE={mode} Total: {total_tokens}tok, Time: {t:.2f}s, "
          f"Throughput: {total_tokens / t:.2f}tok/s")


if __name__ == "__main__":
    main()
