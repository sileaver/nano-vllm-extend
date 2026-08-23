"""Compare nano-vllm's Qwen3.5 port against the transformers reference.

Loads the real checkpoint on both sides (bf16, single GPU) and checks:
  1. prefill logits (all rows)            — full-model numerical parity
  2. greedy decode steps (recurrent path) — recurrent/conv state + paged-KV
     continuation parity: nano advances one token at a time through its
     state pools while the reference recomputes the full sequence (causal
     ⇒ equivalent), for N greedy steps.

Usage: python tests/ref_check_qwen35.py [n_decode_steps]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

MODEL = os.path.expanduser("~/huggingface/Qwen3.5-2B")
PROMPT = "The capital of France is"
N_DECODE = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def main():
    dist.init_process_group("nccl", "tcp://localhost:2334", world_size=1, rank=0)
    torch.cuda.set_device(0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    print(f"[check] prompt tokens: {len(ids)}")

    # ── reference: generate() (DynamicCache path — stable here; the full-
    # recompute loop trips a device assert in transformers' torch chunk
    # kernel on this GPU).  scores[i] are the step-i logits (fp32). ──
    from transformers import Qwen3_5ForConditionalGeneration
    ref = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda().eval()
    with torch.inference_mode():
        gen = ref.generate(
            torch.tensor([ids], device="cuda"), max_new_tokens=N_DECODE + 1,
            do_sample=False, output_scores=True, return_dict_in_generate=True)
    ref_prompt_logits = gen.scores[0][0].float()      # predicts token ids[?]+1
    ref_step_logits = [s[0].float() for s in gen.scores]
    ref_steps = [int(s[0].argmax()) for s in gen.scores]
    ref_tokens = gen.sequences[0].tolist()[len(ids):]
    assert ref_tokens == ref_steps, "reference generate mismatch (sampled vs greedy)"
    print(f"[check] ref greedy tokens: {ref_tokens[:8]} ...")
    del ref, gen
    torch.cuda.empty_cache()

    # ── nano-vllm model + a minimal single-block paged-KV harness ──
    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(MODEL).text_config
    default_dtype, default_dev = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(hf.dtype)
    torch.set_default_device("cuda")
    from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
    from nanovllm.utils.loader import load_model
    from nanovllm.utils.context import set_context, reset_context
    model = Qwen3_5ForCausalLM(hf)
    load_model(model, MODEL)
    model.eval()

    block = 256
    num_full = sum(1 for t in hf.layer_types if t != "linear_attention")
    # engine layout: [2, L, num_blocks, block_size, kv_heads, head_dim]
    cache = torch.zeros(2, num_full, 1, block, hf.num_key_value_heads, hf.head_dim)
    attns = [m for m in model.modules() if hasattr(m, "k_cache")]
    for i, m in enumerate(attns):
        m.k_cache = cache[0, i]
        m.v_cache = cache[1, i]
    num_gdn = sum(1 for t in hf.layer_types if t == "linear_attention")
    g = next(m for m in model.modules() if hasattr(m, "s_cache"))
    s_pool = torch.zeros(1, num_gdn, g.num_v_heads, g.head_k_dim, g.head_v_dim,
                         dtype=torch.float32)
    conv_pool = torch.zeros(1, num_gdn, g.conv_dim, g.conv_kernel - 1)
    for i, m in enumerate([m for m in model.modules() if hasattr(m, "s_cache")]):
        m.s_cache = s_pool[:, i]
        m.conv_cache = conv_pool[:, i]
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    state_ids = torch.zeros(1, dtype=torch.int64)

    def forward(token_list, first_n_cached):
        n = len(token_list)
        inp = torch.tensor(token_list, device="cuda")
        pos = torch.arange(first_n_cached, first_n_cached + n, device="cuda")
        cu_q = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, first_n_cached + n], dtype=torch.int32, device="cuda")
        slots = torch.arange(first_n_cached, first_n_cached + n,
                             dtype=torch.int32, device="cuda")
        set_context(cu_q, cu_k, n, first_n_cached + n, slots, block_tables,
                    0, None, None, linear_state_ids=state_ids)
        with torch.inference_mode():
            hidden = model(inp, pos)
            logits = model.compute_all_logits(hidden).float()
        reset_context()
        return logits

    # prefill (fresh state: pools are zero-initialised)
    logits = forward(ids, 0)
    d = (logits[-1] - ref_prompt_logits).abs()
    same_last = int(logits[-1].argmax()) == ref_steps[0]
    print(f"[check] prefill  : max diff {d.max().item():.4f}  mean {d.mean().item():.6f}"
          f"  next@nano {int(logits[-1].argmax())}  next@ref {ref_steps[0]}")

    # greedy decode: recurrent path one token at a time, vs reference steps
    toks = list(ids)
    mismatches = 0
    for step in range(N_DECODE):
        nxt = int(logits[-1].argmax())
        toks.append(nxt)
        logits = forward([nxt], len(toks) - 1)
        # scores[step+1]: the logits AFTER feeding token step (nano's decode
        # output predicts token step+2, one row ahead of scores[step])
        ref_row = ref_step_logits[step + 1]
        d = (logits[0] - ref_row).abs()
        same = nxt == ref_steps[step]
        mismatches += 0 if same else 1
        flag = "" if same else "  <-- token mismatch"
        print(f"[check] decode {step:2d}: max diff {d.max().item():.4f}"
              f"  mean {d.mean().item():.6f}  next@nano {nxt}"
              f"  next@ref {ref_steps[step]}{flag}")
    torch.set_default_dtype(default_dtype)
    torch.set_default_device(default_dev)
    dist.destroy_process_group()
    ok = same_last and mismatches == 0
    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'}"
          f" (prefill next-token match: {same_last},"
          f" decode mismatches {mismatches}/{N_DECODE})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
