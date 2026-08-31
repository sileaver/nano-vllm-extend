"""Fair nano-vllm vs vLLM benchmark on one identical workload.

Both engines see the same prompts (same seed), both run in their best
scheduling config (vLLM async_scheduling / nano-vllm continuous batching +
async scheduling), both get a shape-matched warmup before the timed reps
(cold-start compile/capture otherwise skews the first run by 20-30%).

    OMP_NUM_THREADS=8 python bench_vs_vllm.py                 # both engines
    OMP_NUM_THREADS=8 python bench_vs_vllm.py --engine nano   # one side
    OMP_NUM_THREADS=8 python bench_vs_vllm.py --engine vllm
    OMP_NUM_THREADS=8 python bench_vs_vllm.py --mode decode   # mixed|decode|prefill
    OMP_NUM_THREADS=8 python bench_vs_vllm.py --seqs 64 --model /path/to/Qwen3-0.6B

Each engine is a separate subprocess (own CUDA context, no teardown
interference); startup takes ~1-3 min per engine, the timed part is 3 reps.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import subprocess
import sys
import time
from random import randint, seed


def build_workload(mode, num_seqs):
    seed(0)
    if mode == "mixed":        # ragged arrivals, prefill+decode interleaved
        max_in, max_out = 1024, 1024
        prompts = [[randint(0, 10000) for _ in range(randint(100, max_in))]
                   for _ in range(num_seqs)]
        outs = [randint(100, max_out) for _ in range(num_seqs)]
    elif mode == "decode":     # decode-dominated: short prompt, long output
        prompts = [[100 + i] * 64 for i in range(num_seqs)]
        outs = [256] * num_seqs
    else:                      # prefill-dominated: long prompt, tiny output
        prompts = [[randint(0, 10000) for _ in range(2048)]
                   for _ in range(num_seqs)]
        outs = [2] * num_seqs
    return prompts, outs


def run(engine, model, mode, num_seqs, max_model_len):
    # Import lazily inside the child (via ENGINE env) so each subprocess
    # loads exactly one engine.
    if engine == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model, enforce_eager=False, max_model_len=max_model_len,
                  async_scheduling=True)
        wrap = lambda p: {"prompt_token_ids": p}
    else:
        from nanovllm import LLM, SamplingParams
        llm = LLM(model, enforce_eager=False, max_model_len=max_model_len,
                  continuous_batching=True, async_scheduling=True)
        wrap = lambda p: p

    prompts, out_lens = build_workload(mode, num_seqs)
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
           for n in out_lens]

    # Shape-matched warmup (same batch shapes as the measured run).
    if mode == "decode":
        llm.generate([wrap(p) for p in prompts[:8]],
                     [SamplingParams(temperature=0.6, ignore_eos=True,
                                     max_tokens=32)] * 8, use_tqdm=False)
    else:
        llm.generate([wrap(prompts[0])],
                     SamplingParams(temperature=0.6, ignore_eos=True,
                                    max_tokens=2), use_tqdm=False)

    n_in = sum(len(p) for p in prompts)
    for rep in range(3):
        sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
               for n in out_lens]
        t0 = time.perf_counter()
        llm.generate([wrap(p) for p in prompts], sps, use_tqdm=False)
        dt = time.perf_counter() - t0
        n_out = sum(out_lens)
        print(f"[{engine:4s}|{mode:7s}|rep{rep}] {dt:6.2f}s  "
              f"total {(n_in + n_out) / dt / 1000:7.1f}k tok/s  "
              f"out {n_out / dt / 1000:7.1f}k tok/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["nano", "vllm", "both"], default="both")
    ap.add_argument("--mode", choices=["mixed", "decode", "prefill"], default="mixed")
    ap.add_argument("--seqs", type=int, default=256)
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3.5-2B/"))
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    env = dict(os.environ, ENGINE="", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"))
    engines = ["nano", "vllm"] if args.engine == "both" else [args.engine]
    for eng in engines:
        if args.engine == "both":
            # Child runs a single engine via --engine; parent just chains them.
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--engine", eng, "--mode", args.mode, "--seqs", str(args.seqs),
                   "--model", args.model, "--max-model-len", str(args.max_model_len)]
            subprocess.run(cmd, check=True, env=env)
        else:
            run(eng, args.model, args.mode, args.seqs, args.max_model_len)
