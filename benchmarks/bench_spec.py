"""bench1 风格的投机解码吞吐对比: K = 0 (基线) vs 4 vs 8."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
NUM_SEQS = 256
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024


def run_once(num_spec_tokens: int):
    seed(0)
    llm = LLM(
        model=PATH,
        enforce_eager=False,
        max_model_len=4096,
        collect_timing=True,
        num_spec_tokens=num_spec_tokens,
    )
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=randint(100, MAX_OUTPUT_LEN))
        for _ in range(NUM_SEQS)
    ]
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    llm.reset_timing()

    torch.cuda.synchronize()
    t0 = time.time()
    outputs, stats = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    num_prompt_tokens = sum(len(p) for p in prompt_token_ids)
    num_generated_tokens = sum(sp.max_tokens for sp in sampling_params)
    decode_tps = num_generated_tokens / elapsed
    total_tps = (num_prompt_tokens + num_generated_tokens) / elapsed
    spec = llm.scheduler.spec_stats() if num_spec_tokens else None
    llm.exit()
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    line = (f"K={num_spec_tokens}: Time {elapsed:.2f}s  "
            f"Decode {decode_tps:.0f} tok/s  Total {total_tps:.0f} tok/s  "
            f"TPOT {stats['tpot_mean']*1000:.1f}ms")
    if spec:
        line += f"  accept_rate {spec['accept_rate']:.3f}"
    print(line, flush=True)
    return decode_tps


if __name__ == "__main__":
    for k in [0, 4, 8]:
        run_once(k)
