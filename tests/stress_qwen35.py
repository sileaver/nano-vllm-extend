"""Stage-3 stress test for the Qwen3.5 engine path.

Each case runs in its own subprocess (fresh CUDA context) and prints its
greedy tokens as JSON; the parent compares.

1. CUDA-graph decode (enforce_eager=False) vs eager: outputs must match.
2. Chunked prefill: re-chunked long prompt vs one-shot prefill.
3. Continuous batching: short+long interleaved vs solo run.
4. Preemption: starved KV pool forces preempt+recompute.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.path.expanduser("~/huggingface/Qwen3.5-2B")


def run_case(eager, prompt_ids, max_num_batched_tokens=16384, blocks=None):
    code = f"""
import json, sys
sys.path.insert(0, '/home/a/nano-vllm')
from nanovllm import LLM, SamplingParams
llm = LLM({MODEL!r}, enforce_eager={eager}, tensor_parallel_size=1,
          max_model_len=4096, max_num_batched_tokens={max_num_batched_tokens},
          continuous_batching=True)
if {blocks} is not None:
    from collections import deque
    bm = llm.scheduler.block_manager
    bm.free_block_ids = deque(list(bm.free_block_ids)[:{blocks}])
ids = json.loads({json.dumps(prompt_ids)!r})
sp = SamplingParams(temperature=1e-6, max_tokens=48)
outs = llm.generate(ids, sp, use_tqdm=False)
llm.exit()
print("JSON" + json.dumps([o["token_ids"] for o in outs]))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    raise RuntimeError(f"case failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    short = tok(chat("What is 2+2? Answer with the number only."),
                return_tensors="pt").input_ids[0].tolist()
    # ~2500-token prompt: repeats a paragraph so the model still answers well
    para = ("The quick brown fox jumps over the lazy dog. Pack my box with "
            "five dozen liquor jugs. How vexingly quick daft zebras jump! ")
    long_ids = tok(chat(f"Repeat-check this text, then say OK: {para * 40}"),
                   return_tensors="pt").input_ids[0].tolist()
    print(f"prompts: short={len(short)} long={len(long_ids)} tokens")

    a = run_case(False, [short])[0]
    b = run_case(True, [short])[0]
    print(f"[1] cudagraph == eager     : {a == b}  ({len(a)}/{len(b)} tok)")
    print(f"    text: {tok.decode(a)[:70]!r}")

    one = run_case(True, [long_ids])[0]
    two = run_case(True, [long_ids], max_num_batched_tokens=768)[0]
    print(f"[2] chunked(4x) == one-shot: {one == two}  ({len(one)}/{len(two)} tok)")

    pair = run_case(True, [short, long_ids])[0]
    print(f"[3] batched short == solo  : {pair == a}")

    # blocks=12 (3072 tokens of KV): seq1's prefill fits, the second
    # sequence then forces preemption (a 2548-token prompt needs 10 blocks)
    starved = run_case(True, [long_ids, long_ids], blocks=12)[0]
    print(f"[4] preempt == one-shot    : {starved == one}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
