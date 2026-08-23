import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 64
    max_input_len = 512
    max_ouput_len = 256
    async_sched = os.environ.get("ASYNC") == "1"

    path = os.path.expanduser("~/huggingface/Qwen3.5-2B/")
    llm = LLM(path, enforce_eager=os.environ.get("EAGER") == "1",
              max_model_len=4096, continuous_batching=True,
              async_scheduling=async_sched)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, max_ouput_len))
                       for _ in range(num_seqs)]

    llm.generate(["Warmup: "], SamplingParams(max_tokens=4))
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"[qwen3.5-2B {'async' if async_sched else 'sync'}"
          f"{' eager' if os.environ.get('EAGER') == '1' else ' graph'}]"
          f" Total: {total_tokens}tok, Time: {t:.2f}s,"
          f" Throughput: {total_tokens / t:.2f}tok/s")


if __name__ == "__main__":
    main()
