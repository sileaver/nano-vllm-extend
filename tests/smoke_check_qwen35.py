"""Smoke-test nano-vllm's Qwen3.5 port against transformers with RANDOM
weights (no checkpoint needed) — catches shape/numerics bugs while the
real weights download.

Builds transformers' Qwen3_5ForCausalLM from the local config (random
init), copies every weight into the nano-vllm model (merging gate/up),
then compares prefill logits and greedy recurrent-decode steps through a
minimal paged-KV + state-pool harness.

Usage: python tests/smoke_check_qwen35.py [n_decode_steps]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

MODEL = os.path.expanduser("~/huggingface/Qwen3.5-2B")   # config/tokenizer only
PROMPT = "The capital of France is"
N_DECODE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
torch.manual_seed(0)


def main():
    dist.init_process_group("nccl", "tcp://localhost:2334", world_size=1, rank=0)
    torch.cuda.set_device(0)

    from transformers import AutoConfig, AutoTokenizer
    hf = AutoConfig.from_pretrained(MODEL).text_config
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    print(f"[smoke] prompt tokens: {len(ids)}")

    # ── reference (random weights, language model only) ──
    from transformers import Qwen3_5ForCausalLM as RefLM
    ref = RefLM(hf).to(torch.bfloat16).cuda().eval()

    # ── nano model with the reference weights copied in ──
    default_dtype, default_dev = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
    from nanovllm.utils.context import set_context, reset_context
    model = Qwen3_5ForCausalLM(hf)
    ref_sd = {k: v for k, v in ref.state_dict().items()}
    sd = {}
    for k, v in ref_sd.items():
        if k.endswith("mlp.gate_proj.weight"):
            base = k[: -len("gate_proj.weight")]
            sd[base + "gate_up_proj.weight"] = torch.cat(
                [v, ref_sd[base + "up_proj.weight"]], dim=0)
        elif k.endswith("mlp.up_proj.weight"):
            continue
        else:
            sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("lm_head.weight")]  # tied
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    # minimal paged-KV + recurrent-state harness (single 256 block, slot 0)
    block = 256
    num_full = sum(1 for t in hf.layer_types if t != "linear_attention")
    # engine layout: [2, L, num_blocks, block_size, kv_heads, head_dim]
    cache = torch.zeros(2, num_full, 1, block, hf.num_key_value_heads, hf.head_dim)
    for i, m in enumerate([m for m in model.modules() if hasattr(m, "k_cache")]):
        m.k_cache = cache[0, i]
        m.v_cache = cache[1, i]
    gdns = [m for m in model.modules() if hasattr(m, "s_cache")]
    g = gdns[0]
    s_pool = torch.zeros(1, len(gdns), g.num_v_heads, g.head_k_dim, g.head_v_dim,
                         dtype=torch.float32)
    conv_pool = torch.zeros(1, len(gdns), g.conv_dim, g.conv_kernel - 1)
    for i, m in enumerate(gdns):
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

    # ── prefill parity ──
    with torch.inference_mode():
        ref_logits = ref(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
    logits = forward(ids, 0)
    d = (logits - ref_logits).abs()
    agree = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    print(f"[smoke] prefill : max diff {d.max().item():.5f}  mean {d.mean().item():.6f}"
          f"  argmax agree {agree:.4f}")

    # ── greedy decode: nano recurrent steps vs reference full recompute ──
    toks = list(ids)
    max_diff = 0.0
    mismatch = 0
    for step in range(N_DECODE):
        nxt = int(logits[-1].argmax())
        toks.append(nxt)
        logits = forward([nxt], len(toks) - 1)
        with torch.inference_mode():
            ref_row = ref(input_ids=torch.tensor([toks], device="cuda")).logits[0, -1].float()
        d = (logits[0] - ref_row).abs().max().item()
        max_diff = max(max_diff, d)
        mismatch += int(logits[0].argmax() != ref_row.argmax())
        print(f"[smoke] decode {step:2d}: max diff {d:.5f}"
              f"  next@nano {int(logits[0].argmax())}  next@ref {int(ref_row.argmax())}")

    torch.set_default_dtype(default_dtype)
    torch.set_default_device(default_dev)
    dist.destroy_process_group()
    ok = agree > 0.99 and mismatch <= N_DECODE // 4 and max_diff < 0.05
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}"
          f" (argmax {agree:.4f}, decode max diff {max_diff:.5f},"
          f" mismatches {mismatch}/{N_DECODE})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
