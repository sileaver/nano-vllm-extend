"""Run DFlash speculative decoding with the OFFICIAL implementation.

Loads z-lab's ``modeling_dflash.py`` straight from the downloaded model
dir and calls the official ``DFlashDraftModel.spec_generate`` — nano-vllm
is not imported at all.  ``spec_generate_stats`` below is a line-by-line
copy of the official ``spec_generate`` with only stats added (per-block
acceptance lengths and wall time), so it doubles as the "purely official"
run plus a measurement of acceptance rate / throughput.

Usage:
    python run_dflash_official.py [--prompt "..."] [--max-new-tokens N] \
        [--temperatures 0.0 0.6] [--baseline]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/huggingface/Qwen3-4B-DFlash-b16"))
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from modeling_dflash import (DFlashDraftModel, extract_context_feature,
                             sample)

TARGET_PATH = os.path.expanduser("~/huggingface/Qwen3-4B")
DRAFT_PATH = os.path.expanduser("~/huggingface/Qwen3-4B-DFlash-b16")
EOS = 151645  # <|im_end|>


def spec_generate_stats(
    draft: DFlashDraftModel,
    target,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
):
    """Line-by-line copy of the official spec_generate, plus stats."""
    draft.eval()
    target.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    block_size = draft.block_size
    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    t_start = time.perf_counter()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

    # Decode stage
    acceptance_lengths = []
    start = input_ids.shape[1]
    while start < max_length:
        block_output_ids = output_ids[:, start: start + block_size].clone()
        block_position_ids = position_ids[:, start: start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -block_size + 1:, :])
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start: start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)[:, :acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break
    elapsed = time.perf_counter() - t_start

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    return output_ids, acceptance_lengths, elapsed


def run(target, draft, tokenizer, prompt, max_new_tokens, temperature,
        official=False, seed=0):
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    if official:
        # Pure official path, no instrumentation.
        t0 = time.perf_counter()
        out = draft.spec_generate(
            target, input_ids, draft.config.dflash_config["mask_token_id"],
            max_new_tokens, [EOS], temperature)
        elapsed = time.perf_counter() - t0
        acceptance_lengths = None
    else:
        out, acceptance_lengths, elapsed = spec_generate_stats(
            draft, target, input_ids, draft.config.dflash_config["mask_token_id"],
            max_new_tokens, [EOS], temperature)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - input_ids.shape[1]
    result = {
        "temperature": temperature,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tok_per_s": new_tokens / elapsed,
        "acceptance_lengths": acceptance_lengths,
    }
    return result, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.6])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true",
                        help="also run plain non-speculative generation")
    parser.add_argument("--chat", action="store_true",
                        help="use the official README chat-template prompt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"loading target  {TARGET_PATH}")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"loading draft   {DRAFT_PATH}")
    draft = DFlashDraftModel.from_pretrained(
        DRAFT_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
    print(f"GPU mem used: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    prompts = [args.prompt]
    if args.chat:
        # Official README example prompt (thinking disabled).
        text = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": "How many positive whole-number divisors does 196 have?"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(text)

    for prompt in prompts:
        for temp in args.temperatures:
            res, text = run(target, draft, tokenizer, prompt,
                            args.max_new_tokens, temp, official=False, seed=args.seed)
            acc = res["acceptance_lengths"]
            rate = sum(acc) / (len(acc) * (draft.block_size - 1))
            print(f"\n=== temperature={temp}  prompt={prompt[:60]!r} ===")
            print(f"generated ({res['new_tokens']} tokens, {res['elapsed_s']:.2f}s, "
                  f"{res['tok_per_s']:.1f} tok/s):")
            print(text[:500])
            print(f"blocks={len(acc)}  per-block accepted lengths={acc}")
            print(f"acceptance rate = {rate:.3f}  (mean len {sum(acc)/len(acc):.1f} / {draft.block_size-1})")

    # one pure-official run to prove the unmodified method itself works
    res, text = run(target, draft, tokenizer, args.prompt, 64, 0.6,
                    official=True, seed=args.seed)
    print(f"\n=== official spec_generate (unmodified) ===")
    print(f"generated {res['new_tokens']} tokens, {res['elapsed_s']:.2f}s, "
          f"{res['tok_per_s']:.1f} tok/s")
    print(text[:300])

    if args.baseline:
        # Plain non-speculative generation on the same target.
        for temp in args.temperatures:
            torch.manual_seed(args.seed)
            input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
            t0 = time.perf_counter()
            out = target.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=temp > 1e-5, temperature=temp if temp > 1e-5 else None,
                eos_token_id=EOS, pad_token_id=tokenizer.pad_token_id)
            elapsed = time.perf_counter() - t0
            new_tokens = out.shape[1] - input_ids.shape[1]
            print(f"\n=== baseline (no speculation) temperature={temp} ===")
            print(f"generated {new_tokens} tokens in {elapsed:.2f}s = "
                  f"{new_tokens / elapsed:.1f} tok/s")
            print(tokenizer.decode(out[0], skip_special_tokens=True)[:300])


if __name__ == "__main__":
    main()
