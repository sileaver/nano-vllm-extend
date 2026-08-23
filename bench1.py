import os
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    # path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    path = os.path.expanduser("~/huggingface/Qwen3.5-2B/")
    llm = LLM(
        model=path,
        enforce_eager=False,
        max_model_len=4096,
        collect_timing=True
        # sampling_backend="flashinfer"
        # attention_backend="flashinfer"
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    # ── Warmup ────────────────────────────────────────────────
    print("Warming up...")
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    llm.reset_timing()

    # ── Benchmark ─────────────────────────────────────────────
    print(f"Benchmarking {num_seqs} requests...")

    torch.cuda.synchronize()
    t0 = time.time()

    outputs, stats = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # ── Throughput ────────────────────────────────────────────
    num_prompt_tokens = sum(len(p) for p in prompt_token_ids)
    num_generated_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_tokens = num_prompt_tokens + num_generated_tokens

    decode_throughput = num_generated_tokens / elapsed
    total_throughput = total_tokens / elapsed

    print(f"Time:              {elapsed:.3f} s")
    print(f"Total tokens:      {total_tokens}  "
          f"(prefill {num_prompt_tokens} + decode {num_generated_tokens})")
    print(f"Decode Throughput: {decode_throughput:.2f} tok/s")
    print(f"Total Throughput:  {total_throughput:.2f} tok/s")

    # ── Latency ───────────────────────────────────────────────
    print(f"\nLatency (n={stats['num_requests']} requests):")
    print(f"  TTFT mean:   {stats['ttft_mean']*1000:.1f} ms")
    print(f"  TTFT p50:    {stats['ttft_p50']*1000:.1f} ms")
    print(f"  TTFT p99:    {stats['ttft_p99']*1000:.1f} ms")
    print(f"  TPOT mean:   {stats['tpot_mean']*1000:.1f} ms")
    print(f"  TPOT p50:    {stats['tpot_p50']*1000:.1f} ms")
    print(f"  TPOT p99:    {stats['tpot_p99']*1000:.1f} ms")


if __name__ == "__main__":
    main()
