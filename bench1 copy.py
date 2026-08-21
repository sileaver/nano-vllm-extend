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

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 【修正 1】必须彻底关闭日志、统计状态和进度条 (以 vLLM 兼容参数为例)
    llm = LLM(
        model=path, 
        enforce_eager=False, 
        max_model_len=4096, 
        async_scheduling=True,
        disable_log_stats=True,      # 关闭定期打印的资源监控日志
        # disable_log_requests=True, # 视引擎 API 是否支持
    )

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # 预热引擎，让 CUDA Graph 完成捕获
    print("Warming up...")
    llm.generate([[1, 2, 3]], [SamplingParams(max_tokens=10)], use_tqdm=False)

    print(f"Benchmarking with {num_seqs} requests...")
    
    # 【修正 2】严格的 CUDA 同步计时
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 假设此处是阻塞调用，直到所有生成完毕
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 【修正 3】计算包含 Prefill 的真实总吞吐量
    num_prompt_tokens = sum(len(p) for p in prompt_token_ids)
    num_generated_tokens = sum(sp.max_tokens for sp in sampling_params)
    total_tokens = num_prompt_tokens + num_generated_tokens
    
    decode_throughput = num_generated_tokens / elapsed_time
    total_throughput = total_tokens / elapsed_time

    print(f"Time: {elapsed_time:.3f} s")
    print(f"Decode Throughput: {decode_throughput:.2f} tok/s")
    print(f"Total Throughput (Prefill+Decode): {total_throughput:.2f} tok/s")

if __name__ == "__main__":
    main()