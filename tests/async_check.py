"""Async-scheduling correctness check.

Part 1 — single sequence: identical batch shapes ([1]) every step in both
engines, so with a fixed seed the outputs must match EXACTLY (driver
diffs two runs of this script with ASYNC=0/1).

Part 2 — staggered multi-sequence load: structural asserts (all lengths
correct, no garbage token 0, KV block pool fully restored afterwards,
engine reusable for a second generate — regression for deferred frees).

    OMP_NUM_THREADS=8 ASYNC=1 python tests/async_check.py [single|tp2|pp2] [eager|graph]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm import LLM, SamplingParams

MODEL = os.environ.get("QWEN35_MODEL", os.path.expanduser("~/huggingface/Qwen3.5-2B"))

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1492, Columbus sailed",
    "The three primary colors are",
    "Photosynthesis converts sunlight into",
    "The largest planet in the solar system is",
    "A triangle has",
    "The speed of light in vacuum is approximately",
    "Newton's second law of motion says that",
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    path = sys.argv[2] if len(sys.argv) > 2 else "graph"
    async_ = os.environ.get("ASYNC", "1") == "1"

    kw = dict(max_model_len=768, enforce_eager=(path == "eager"),
              # Force chunked prefill: the longest prompt spans 2 chunks
              # and prefills interleave with running decode seqs.
              max_num_batched_tokens=512,
              continuous_batching=True)
    if mode == "tp2":
        kw |= dict(tensor_parallel_size=2)
    elif mode == "pp2":
        kw |= dict(pipeline_parallel_size=2)

    llm = LLM(MODEL, async_scheduling=async_, **kw)
    engine = llm  # LLM subclasses LLMEngine
    pool = engine.scheduler.block_manager
    n_blocks_before = len(pool.free_block_ids)

    # ── Part 1: single sequence, exact-match reference ──
    torch.manual_seed(0)
    ref = llm.generate([PROMPTS[0]], SamplingParams(
        temperature=1e-6, max_tokens=32), use_tqdm=False)
    print(f"[ref] {ref[0]['token_ids']}")
    assert len(ref[0]["token_ids"]) == 32

    # ── Part 2: staggered multi-sequence load ──
    # max_tokens staggered → sequences finish at different decode steps
    # (max_tokens finish → exact; EOS finish → one wasted row + deferred
    # release).  temperature 1e-6 ≈ greedy.
    sps = [
        SamplingParams(temperature=1e-6, max_tokens=24 + 7 * i)
        for i in range(len(PROMPTS))
    ]
    torch.manual_seed(0)
    outs = llm.generate(PROMPTS, sps, use_tqdm=False)

    for i, (o, sp) in enumerate(zip(outs, sps)):
        toks = o["token_ids"]
        assert len(toks) == sp.max_tokens, \
            f"seq {i}: len {len(toks)} != max_tokens {sp.max_tokens}"
        assert all(t != 0 for t in toks), f"seq {i}: garbage token 0: {toks}"
        print(f"[{i}] {toks}")

    # Every sequence released: the KV pool must be fully restored.
    n_after = len(pool.free_block_ids)
    assert n_after == n_blocks_before, \
        f"KV block leak: {n_blocks_before - n_after} blocks lost"
    assert not engine._inflight, "in-flight steps left after generate"
    print(f"[blocks] {n_after}/{n_blocks_before} free — OK")

    # Engine reusable (deferred frees really ran): a second generate works.
    torch.manual_seed(1)
    again = llm.generate([PROMPTS[1]], SamplingParams(
        temperature=1e-6, max_tokens=16), use_tqdm=False)
    assert len(again[0]["token_ids"]) == 16
    assert len(pool.free_block_ids) == n_blocks_before
    print("[reuse] OK")
    print("PASS")


if __name__ == "__main__":
    main()
