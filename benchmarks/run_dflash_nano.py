"""nano-vllm DFlash acceptance rate on the official chat prompt.

Counterpart to run_dflash_official.py: same prompt (official README math
question, thinking disabled), same temperature, nano-vllm's ported DFlash
draft.  Compares against the official transformers numbers.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

TARGET_PATH = os.path.expanduser("~/huggingface/Qwen3-4B")
DRAFT_PATH = os.path.expanduser("~/huggingface/Qwen3-4B-DFlash-b16")

tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
chat_prompt = tokenizer.apply_chat_template(
    [{"role": "user",
      "content": "How many positive whole-number divisors does 196 have?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)

llm = LLM(
    model=TARGET_PATH,
    spec_draft_model=DRAFT_PATH,
    num_spec_tokens=1,  # overridden to block_size-1=15 by DFlash detection
    max_model_len=4096,
    collect_timing=True,
)

# nano's sampler forbids greedy (temperature=0); compare at 0.6 where the
# official implementation measured 0.525 (chat) / 0.203 (capital).
for temp in [0.6]:
    torch.manual_seed(0)
    llm.reset_timing()
    t0 = time.time()
    outputs, stats = llm.generate(
        [chat_prompt, "The capital of France is"],
        [SamplingParams(temperature=temp, max_tokens=128, ignore_eos=True)] * 2,
        use_tqdm=False)
    elapsed = time.time() - t0
    spec = llm.scheduler.spec_stats()
    print(f"\n=== temperature={temp} ===")
    for out in outputs:
        print(f"--- {out['text'][:120]!r}")
    print(f"elapsed {elapsed:.2f}s  tpot {stats['tpot_mean']*1000:.1f}ms  "
          f"accept_rate {spec['accept_rate']:.3f} "
          f"({spec['accepted']}/{spec['attempted']})")

llm.exit()
