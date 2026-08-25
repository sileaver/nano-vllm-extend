<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor/Pipeline/Data Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Parallelism (TP / PP / DP)

The three strategies can be combined freely (`dp * pp * tp` ranks, one GPU
per rank, single node).  Correctness was validated on 2x RTX 5090 against
single-GPU runs: prefill argmax is token-identical across layouts
(32/32 real prompts), and generation streams stay within the bf16
implementation-noise floor (small models have near-tied logits; see
`tests/real_prompt_check.py`, `tests/parallel_check.py`):

```python
llm = LLM(path, tensor_parallel_size=2)    # TP: shard heads/neurons per rank
llm = LLM(path, pipeline_parallel_size=2)  # PP: split layers across stages
llm = LLM(path, data_parallel_size=2)      # DP: replicas, each own scheduler
```

* **TP** shards attention/MLP projections, the vocab embedding, the hybrid
  Qwen3.5 gated-delta-net heads (conv channels + recurrent state) and the
  KV cache.  Communication: one all-reduce per layer.
* **PP** splits decoder layers across stages (`dist.send/recv` of the fused
  hidden+residual chain between stage peers).  Each stage owns only its
  layers' KV cache / recurrent-state pools — roughly halves per-GPU memory
  at pp=2.  KV-block and state-slot counts are min-synced across stages so
  the rank-0 scheduler stays consistent.  Decode CUDA graphs are captured
  per stage.  v1 runs one batch through the pipeline synchronously (no
  micro-batch overlap) — use it for capacity, not latency.
* **DP** replicates the whole engine (scheduler + runners) per GPU group;
  the driver LPT-bin-packs requests and merges results in order.  Best for
  offline throughput when the load exceeds one replica's concurrency —
  measured 1.94x on 512 ragged requests over 2x RTX 5090 (vs 1.38x at 256
  requests where each replica is batch-starved).

Speculative decoding requires `tp = pp = 1` (it can run under DP — each
replica speculates independently).  `collect_timing` and step-level
`add_request`/`step` streaming are single-group only.

## Async Scheduling (vLLM V1 style)

`async_scheduling=True` pipelines CPU scheduling under GPU execution the
way vLLM V1's async scheduler does — sampled tokens never round-trip
through the CPU:

* **GPU token ring** — each step's sampled ids stay on the GPU (a small
  ring of slots).  The next decode step gathers its `input_ids`
  on-device from the previous step's slot (row map known from scheduler
  metadata); TP/PP groups move the slot with one NCCL broadcast.
* **Lagging output processing** — results return via an async D2H into
  pinned memory guarded by a CUDA event.  The engine polls events at the
  start of each iteration and applies outputs one to two steps late,
  entirely off the critical path.  Steady-state decode never blocks the
  CPU on the GPU (pipeline depth capped at 2).
* **Optimistic scheduling** — KV slots for in-flight tokens are reserved
  via output placeholders (vLLM's "future token ids"), and a sequence
  that hits EOS runs one extra wasted row before the CPU notices; its
  blocks/state are released when the last step referencing it completes.
* **Synchronous fallback** — prefill-containing batches (and preemption,
  which accompanies them) drain the pipeline and run synchronously, so
  chunk accounting always sees a caught-up scheduler.

Measured on RTX 5090 (decode-bound load): Qwen3-0.6B @ bs=64
20.1k → 22.6k tok/s (+11.6%), Qwen3.5-2B @ bs=64 10.2k → 11.1k tok/s,
and the gain grows with parallelism (more CPU + NCCL launch overhead to
hide): Qwen3-0.6B @ tp2 11.9k → 19.0k tok/s (+59%), Qwen3.5-2B @ tp2
10.1k → 11.4k tok/s (+11.7%).  After this, decode is GPU-bound: the
remaining per-step time is the CUDA graph replay plus the memory-bound
lm-head GEMM.

The hybrid-model (Qwen3.5) optimizations bring nano-vllm to parity with
vLLM 0.27 on ragged serving workloads (mixed 9.5k → 22.4k tok/s vs
23.3k, prefill 33.6k → 62.8k vs 62.5k, decode within 3-5%):

* **Varlen GDN** — the gated-delta-net layer used to pad batches to
  `[bs, max_query_len]`, wasting up to ~150× compute on mixed
  decode+prefill steps.  Decode rows take the O(1) recurrent path;
  prefill groups run fully varlen (dense `[N, H]` projections, a varlen
  causal-conv Triton kernel, and the fla chunk kernel driven by
  `cu_seqlens` — verified bit-equivalent to the padded call).
* **Fused GDN plumbing** — a fused g/beta kernel replaces the eager
  fp32 elementwise chain; the Gemma-style norms and MLP activation run
  on FlashInfer's fused kernels (`gemma_rmsnorm` /
  `gemma_fused_add_rmsnorm` / `silu_and_mul`); the decode CUDA graphs
  capture lm_head + a two-stage fused gumbel-max sampling kernel.
* **Unified async scheduling** — every scheduled row (prefill chunks
  included) reserves output placeholders, so prefill batches flow
  through the same non-draining pipeline as decode: chunk continuation,
  decode-on-top-of-unreaped-prefill and mixed-batch input gathering all
  fall out of one `num_computed_tokens` invariant.

## Speculative Decoding

Two modes, both gated by `num_spec_tokens=K` and verified by strict
per-position rejection sampling (the accepted sequence distribution is
guaranteed to match ordinary autoregressive sampling; see `verify_spec.py`):

```python
# 1. Jacobi-style parallel draft (no draft model): the target itself runs
#    one K-row forward to propose K candidates.
llm = LLM("/YOUR/MODEL/PATH", num_spec_tokens=4)

# 2. Classic draft-model speculation: a small model (same tokenizer/vocab)
#    runs K sequential autoregressive drafts, verified by the target.
llm = LLM("/YOUR/TARGET/PATH", num_spec_tokens=4,
          spec_draft_model="/YOUR/DRAFT/PATH")
```

Notes on the models tested locally (16GB RTX 5080):

* **Correctness** is verified for all modes: synthetic-distribution unit
  tests of the acceptance kernel, draft-vs-standalone logits are bitwise
  identical, and per-position frequency comparison against non-speculative
  sampling stays at the measured noise floor.
* **Layer-skipped self-speculation does not work on 0.6B/4B** — skipping
  even one layer drops the draft/target match rate from ~0.5 to ~0.14 (no
  layer redundancy). The layer-skip path is kept as a model capability
  (`Qwen3Model.forward(skip_layers=...)`) for larger models.
* **No throughput win on 0.6B/4B**: Jacobi draft is 2-4x slower than the
  CUDA-graph baseline; 0.6B→4B draft-model speculation reaches only
  0.05-0.13 acceptance (the draft is too small — the ~10x size ratio used
  in the literature needs a 7B+ target and a properly sized draft). Each
  speculative step pays two multi-row forwards plus 151k-vocab softmaxes
  while a 4B decode token is still cheap — the pipeline pays off on larger
  models, where it is plug-and-play via `spec_draft_model`.
* **DFlash (z-lab block-diffusion draft) is ported and integrated**
  (`nanovllm/models/dflash.py`, auto-detected via the draft config), and
  the official transformers implementation was also benchmarked on this
  machine for reference (`run_dflash_official.py`).  Official numbers:
  acceptance 0.53-0.58 on the README chat prompt (~140 tok/s, ~6x the
  non-speculative baseline) but only 0.18-0.20 on free-form continuation
  — DFlash's acceptance is strongly prompt-dependent.  nano-vllm's
  ported draft matches the official implementation bit-for-bit on
  identical inputs and generates correct text, but its acceptance rate
  collapses to 0.02-0.04 (10-13x below the official rate on the same
  prompts).  Root cause: the target's hidden states (which the draft
  uses as KV context) diverge from transformers' bf16 hidden states in
  deep layers — a 0.125 divergence at layer 5 amplifies 128x in layer 6's
  attention and explodes in the residual sum.  This is an inherent limit
  of a bf16 reimplementation that does not replicate transformers' exact
  rounding order, not a correctness bug (final logits match HF to 0.19
  max diff).

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)