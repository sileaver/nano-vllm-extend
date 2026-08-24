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