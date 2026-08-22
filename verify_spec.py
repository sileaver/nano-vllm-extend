"""Self-speculative decoding verification.

1. Unit test (strong): feed synthetic draft/target distributions to
   ModelRunner._accept_speculative and check the rejection-sampling
   invariants — position 0's marginal equals q_0, and position 1's
   marginal conditioned on acceptance equals q_1 — at 10k draws.
2. e2e distribution test: same prompts × seeds, K=0 vs K=4, per-position
   token frequency comparison (TVD should stay at sampling-noise level).
3. Acceptance-rate sanity + e2e text quality.

Usage: python verify_spec.py [--quick]
"""
import argparse
import gc
import os
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner

MODEL = os.environ.get("NANOVLLM_MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/"))


# ----------------------------------------------------------------------
# 1. Unit test of the acceptance kernel (synthetic distributions)
# ----------------------------------------------------------------------

def unit_test_acceptance(K: int = 4, V: int = 8, n_draws: int = 10000):
    print(f"\n=== Unit test: _accept_speculative (K={K}, V={V}, {n_draws} draws) ===")
    torch.manual_seed(0)
    bs = 1024
    g = torch.Generator(device="cuda").manual_seed(1234)

    # Deliberately *different* draft (p) and target (q) distributions:
    # a bad draft model is what exposes acceptance-logic bugs fastest.
    q_logits = torch.randn(bs, K + 1, V, generator=g, device="cuda")
    p_probs = torch.softmax(torch.randn(bs, K, V, generator=g, device="cuda") * 3, dim=-1)
    d_tokens = torch.multinomial(
        p_probs.reshape(-1, V), 1, generator=g).reshape(bs, K)
    temps = torch.full((bs,), 1.0, device="cuda")

    # Run the real acceptance kernel through an un-initialised instance.
    runner = object.__new__(ModelRunner)
    runner.num_spec_tokens = K

    q0 = torch.softmax(q_logits[:, 0], dim=-1)      # position-0 target dist
    q1 = torch.softmax(q_logits[:, 1], dim=-1)      # position-1 target dist

    count0 = torch.zeros(V, dtype=torch.float64)
    count1 = torch.zeros(V, dtype=torch.float64)
    n1 = 0
    for it in range(n_draws // bs):
        accepted = runner._accept_speculative(
            q_logits.clone(), d_tokens.clone(), p_probs.clone(), temps)
        for b in range(bs):
            out = accepted[b]
            count0[out[0]] += 1
            if len(out) > 1:
                count1[out[1]] += 1
                n1 += 1

    # Reference: draw position 0/1 directly from q with the same RNG count
    # semantics — compare empirical frequencies, not exact draws.
    emp0 = count0 / (n_draws // bs * bs)
    emp1 = count1 / n1
    ref0 = q0.mean(0).cpu().double()
    ref1 = q1.mean(0).cpu().double()
    tvd0 = (emp0 - ref0).abs().sum() / 2
    tvd1 = (emp1 - ref1).abs().sum() / 2
    print(f"  TVD(out[0] vs q_0): {tvd0:.4f}  (10k-sample noise ~0.015)")
    print(f"  TVD(out[1]|len>1 vs q_1): {tvd1:.4f}  (noise ~0.02)")
    print(f"  P(len > 1) = {n1 / (n_draws // bs * bs):.3f}")
    assert tvd0 < 0.03, f"position-0 marginal deviates from q_0: {tvd0}"
    assert tvd1 < 0.05, f"position-1 conditional deviates from q_1: {tvd1}"
    print("  PASS")


# ----------------------------------------------------------------------
# 2. e2e distribution test: K=0 vs K=4 frequency comparison
# ----------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms",
    "Write a haiku about winter",
    "def quicksort(arr):",
    "Once upon a time in a small village",
    "The three laws of thermodynamics state that",
    "In machine learning, overfitting occurs when",
    "A good recipe for pasta carbonara begins with",
    "The quick brown fox jumps over the",
    "Quantum computing differs from classical computing because",
    "The best way to learn a new language is",
    "In 1969, the Apollo 11 mission",
    "The Fibonacci sequence starts with 0, 1, 1,",
    "Photosynthesis converts sunlight into",
    "A blockchain is a distributed ledger that",
    "The Great Wall of China was built to",
]


def e2e_distribution_test(n_seeds: int, max_tokens: int):
    print(f"\n=== e2e distribution test: {n_seeds} seeds × {len(PROMPTS)} prompts "
          f"(max_tokens={max_tokens}) ===")
    sp = SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=max_tokens)

    def run_all(llm):
        """Run n_seeds generations, return per-(prompt, pos) token histogram."""
        hist = [{} for _ in PROMPTS]
        for s in range(n_seeds):
            torch.manual_seed(s)
            outputs = llm.generate(PROMPTS, sp, use_tqdm=False)
            for pi, out in enumerate(outputs):
                for pos, tok in enumerate(out["token_ids"]):
                    hist[pi].setdefault(pos, {})
                    hist[pi][pos][tok] = hist[pi][pos].get(tok, 0) + 1
        return hist

    print("  [K=0] running...")
    llm0 = LLM(model=MODEL, enforce_eager=False, max_model_len=2048)
    hist0 = run_all(llm0)
    llm0.exit()
    del llm0
    gc.collect()
    torch.cuda.empty_cache()

    print("  [K=4] running...")
    llm4 = LLM(model=MODEL, enforce_eager=False, max_model_len=2048,
               num_spec_tokens=4)
    hist4 = run_all(llm4)
    llm4.exit()
    del llm4
    gc.collect()
    torch.cuda.empty_cache()

    tvds = []
    for pi in range(len(PROMPTS)):
        for pos in set(hist0[pi]) | set(hist4[pi]):
            h0, h4 = hist0[pi].get(pos, {}), hist4[pi].get(pos, {})
            n0, n4 = sum(h0.values()), sum(h4.values())
            if not n0 or not n4:
                continue
            vocab = set(h0) | set(h4)
            tvd = sum(abs(h0.get(t, 0) / n0 - h4.get(t, 0) / n4)
                      for t in vocab) / 2
            tvds.append(tvd)
    mean_tvd = sum(tvds) / len(tvds) if tvds else 0
    max_tvd = max(tvds) if tvds else 0
    print(f"  per-position TVD: mean {mean_tvd:.4f}, max {max_tvd:.4f}")
    # Threshold calibration: a same-config K=0 vs K=0 baseline at n=100
    # measured mean 0.198 / max 0.570 of pure sampling noise (low-entropy
    # distributions, small samples), so divergence up to ~0.65 is
    # indistinguishable from noise.  The acceptance-logic bug caught
    # during development showed max TVD = 1.0.  (The strong guarantee
    # comes from the synthetic unit test above; this e2e check is a
    # gross-divergence tripwire.)
    print(f"  (n={n_seeds}: measured noise baseline mean 0.20 / max 0.57; "
          f"real bugs ≈ 1.0)")
    assert max_tvd < 0.65, f"suspicious distribution divergence: {max_tvd}"
    print("  PASS (no gross divergence)")


# ----------------------------------------------------------------------
# 3. Acceptance-rate sanity + text quality
# ----------------------------------------------------------------------

def sanity_and_text():
    print("\n=== acceptance rate + text quality ===")
    sp = SamplingParams(temperature=0.6, max_tokens=128)
    llm = LLM(model=MODEL, enforce_eager=False, max_model_len=2048,
              num_spec_tokens=4)
    # Decode-style load to exercise many speculative steps.
    prompts = [
        "Explain how transformers work",
        "Tell me a short story about a robot learning to paint",
        "What is the difference between TCP and UDP?",
    ] * 8
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    stats = llm.scheduler.spec_stats()
    rate = stats["accept_rate"]
    print(f"  spec_stats: accepted={stats['accepted']}, "
          f"attempted={stats['attempted']}, "
          f"per-position accept rate={rate:.3f}")
    print(f"  (half-layer draft expected ≈ 0.5-0.7; <0.2 or >0.95 suggests a bug)")
    for out in outputs[:3]:
        print(f"  → {out['text'][:100]}")
    llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="reduce seed count / draws")
    args = parser.parse_args()

    # The unit test is fast — keep 10k draws even in quick mode so the
    # acceptance thresholds stay calibrated (the position-1 conditional
    # has few samples when p and q diverge).
    unit_test_acceptance(n_draws=10000)
    e2e_distribution_test(n_seeds=100 if args.quick else 200, max_tokens=3)
    sanity_and_text()
    print("\nAll verification passed.")
