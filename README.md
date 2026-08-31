<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center"><b>English</b> | <a href="README.zh-CN.md">简体中文</a></p>

# Nano-vLLM — my extensions

A lightweight vLLM-style inference engine, **forked from
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** (MIT,
© Xingkai Yu). The upstream snapshot (~1,200 lines: continuous batching, paged
KV, CUDA graphs, TP, prefix caching) is the `Initial commit` of this repo;
**everything after it is my work** — ~4,900 added lines implementing a hybrid
linear-attention model, multimodal inference, pipeline/data parallelism,
vLLM-V1-style async scheduling, speculative decoding, and a set of
kernel-level optimizations, each validated against reference implementations.

## What I built on top of upstream

| Area | What | Where | How it's validated |
|---|---|---|---|
| **Hybrid model (Qwen3.5-2B)** | Port of the gated-delta-net + sparse-attention hybrid: recurrent-state pooling, causal-conv prefix, partial RoPE with output gates | `nanovllm/models/qwen3_5.py` | `tests/ref_check_qwen35.py` — prefill logits + greedy decode vs transformers |
| **Multimodal (Qwen3.5 vision)** | Full `Qwen3_5ForConditionalGeneration`: vision tower, embedding scatter, interleaved MRoPE, chunked-prefill-safe staging | `qwen3_5_vision.py`, `utils/multimodal.py`, `MRotaryEmbedding` | `tests/mm_vision_parity_qwen35.py` (29 stages **bit-exact**), `tests/mm_check_qwen35.py` (token-exact vs HF under chunked prefill / mixed batch / TP / PP / async) |
| **GDN performance** | Varlen prefill (fla chunk kernel + custom varlen causal-conv Triton kernel), fused g/β kernel, in-place recurrent decode (no 450MB state gather/scatter), FlashInfer fused norms | `qwen3_5.py` | bit-equivalence A/B flags (`NANOVLLM_GDN_*`), 8–30× per-layer prefill, ~1.4× decode |
| **Async scheduling (vLLM V1 style)** | GPU token ring, lagged output processing, optimistic scheduling with output placeholders | `engine/async_scheduler.py`, `model_runner.py` | `tests/async_check.py`; +6–59% decode throughput |
| **Parallelism** | PP (layer-split stages, fused-residual handoff) and DP (replicated engines, LPT sharding) on top of upstream TP, incl. hybrid-state sharding | `utils/parallel.py`, `engine/*` | `tests/parallel_check.py` — token-identical prefill across tp/pp/dp layouts |
| **Speculative decoding** | Jacobi parallel draft + classic draft-model + DFlash block-diffusion draft, strict rejection-sampling verification | `model_runner.py`, `models/dflash.py`, `benchmarks/verify_spec.py` | distributional unit tests; honest negative results documented below |
| **Bug hunts** | Two latent hybrid-engine correctness bugs found & fixed via the multimodal tests (details below) | `block_manager.py`, `model_runner.py` | both reproduced before the fix, green after |

## Quick start

```bash
pip install -e .          # or: pip install git+https://github.com/<you>/nano-vllm.git
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python example.py
```

The API mirrors vLLM's (`LLM.generate` returns token ids + text):

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello, Nano-vLLM."],
                       SamplingParams(temperature=0.6, max_tokens=256))
outputs[0]["text"]
```

Model paths in examples/tests default to `~/huggingface/...`; override with
`QWEN35_MODEL=/path` for Qwen3.5 scripts.

## Multimodal (Qwen3.5 vision)

Qwen3.5 checkpoints ship as multimodal shells; the engine loads the whole
thing — vision tower, MRoPE and all. Image prompts follow vLLM's dict form
(one `<|vision_start|><|image_pad|><|vision_end|>` placeholder per image; the
checkpoint's own processor expands it to the image's grid tokens):

```python
prompts = [
    {"prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
               "Describe this image.<|im_end|>\n<|im_start|>assistant\n",
     "images": ["path/or/PIL.Image"]},
    "plain text prompts work on the same engine",
]
outputs = llm.generate(prompts, sampling_params)   # example_qwen35_mm.py
```

Under the hood:

* The **vision tower** (Qwen3-VL style ViT: Conv3d patch embed, a learned
  position table bilinearly resampled per image grid, per-frame non-causal
  flash attention under 2D rope, 2×2 patch merger) is ported in
  `nanovllm/models/qwen3_5_vision.py` — **bit-exact against the transformers
  reference at every one of 29 stages** in fp32. It is replicated across TP
  ranks and lives on the first pipeline stage; merged patch embeddings
  replace `image_token_id` rows after the (all-reduced) token embedding —
  one vision-tower forward per sequence, however many chunks its prefill
  splits into.
* **Interleaved MRoPE** (`MRotaryEmbedding`) — prefill carries [3, N] T/H/W
  positions over image regions (the port of `get_rope_index` lives in
  `utils/multimodal.py`); decode collapses back to 1D via a per-sequence
  `rope_delta`, so CUDA-graph decode replays are untouched.
* Pixels travel bf16 (lossless — the reference casts to the tower dtype
  anyway) and only with prefill states; the TP shm command segment is sized
  accordingly.
* `NANOVLLM_QWEN35_TEXTONLY=1` skips the tower (upstream behaviour).

### Two latent bugs the multimodal tests uncovered

Both are inherent to *any* hybrid (linear-attention) engine, existed in the
text path before, and are the kind of thing only cross-implementation parity
testing surfaces:

1. **Prefix-cache reuse corrupts hybrid state.** Hash-matched KV blocks let a
   repeated prompt resume prefill mid-sequence, but a GDN layer's recurrent
   state is not reconstructible from cached KV — the continuation silently
   diverged. Fix: disable hash-based prefix reuse for hybrid models
   (`BlockManager(enable_prefix_cache=False)`), the same choice vLLM makes
   for mamba-style layers.
2. **CUDA-graph pad rows poisoned a live slot.** Replaying a graph at
   `bs <` captured size leaves stale rows carrying capture-time
   `linear_state_ids = 0` — a *real* slot — so the in-place recurrent kernel
   corrupted that sequence's state every replayed step (only the slot-0
   sequence diverged, only with graphs on). Fix: a dedicated dummy slot that
   pad/capture rows point at (the paged-KV write already had the analogous
   `-1` guard).

## Parallelism (TP / PP / DP)

Freely combinable (`dp * pp * tp` ranks, one GPU per rank, single node).
Correctness validated on 2× RTX 5090 against single-GPU runs — prefill argmax
is token-identical across layouts (32/32 real prompts; see
`tests/parallel_check.py`, `tests/real_prompt_check.py`):

```python
llm = LLM(path, tensor_parallel_size=2)    # TP: shard heads/neurons per rank
llm = LLM(path, pipeline_parallel_size=2)  # PP: split layers across stages
llm = LLM(path, data_parallel_size=2)      # DP: replicas, each own scheduler
```

* **TP** shards attention/MLP projections, the vocab embedding, the GDN
  heads (conv channels + recurrent state at v-head granularity) and the KV
  cache; one all-reduce per layer.
* **PP** splits decoder layers across stages (`dist.send/recv` of the fused
  hidden+residual chain between stage peers). Each stage owns only its
  layers' KV/recurrent-state pools — roughly halves per-GPU memory at pp=2;
  block/slot counts are min-synced across stages; decode CUDA graphs are
  captured per stage. v1 runs one batch through synchronously (no
  micro-batch overlap) — use it for capacity, not latency.
* **DP** replicates the whole engine per GPU group; the driver LPT-bin-packs
  requests and merges results in order — 1.94× on 512 ragged requests over
  2× RTX 5090 (vs 1.38× at 256, where replicas are batch-starved).

Speculative decoding requires `tp = pp = 1` (it runs under DP). Step-level
`add_request`/`step` streaming and `collect_timing` are single-group only.

## Async scheduling (vLLM V1 style)

`async_scheduling=True` pipelines CPU scheduling under GPU execution —
sampled tokens never round-trip through the CPU:

* **GPU token ring** — sampled ids stay on the GPU; the next decode step
  gathers its input ids on-device from the previous step's slot, moved
  across TP/PP with one NCCL broadcast.
* **Lagging output processing** — results return via async D2H into pinned
  memory guarded by a CUDA event; the engine applies them 1–2 steps late,
  off the critical path (pipeline depth capped at 2).
* **Optimistic scheduling** — KV slots for in-flight tokens are reserved via
  output placeholders (vLLM's "future token ids"); a sequence hitting EOS
  runs one wasted row before the CPU notices.
* **Unified pipeline** — prefill chunks reserve placeholders too, so they
  flow through the same non-draining pipeline: chunk continuation,
  decode-on-top-of-unreaped-prefill and mixed-batch input gathering all
  fall out of one `num_computed_tokens` invariant.

Measured on RTX 5090 (decode-bound): Qwen3-0.6B @ bs=64 20.1k → 22.6k tok/s
(+11.6%), Qwen3.5-2B @ bs=64 10.2k → 11.1k tok/s; gains grow with
parallelism (more CPU/NCCL overhead to hide): Qwen3-0.6B @ tp2 11.9k → 19.0k
(+59%), Qwen3.5-2B @ tp2 10.1k → 11.4k (+11.7%).

Together with the GDN optimizations below this reaches parity with vLLM
0.27 on ragged serving workloads — mixed throughput went **9.5k → 22.7k
tok/s (~41% → 98% of vLLM)** in the course of that rework (full
head-to-head table in the Benchmark section below).

## Hybrid-model (GDN) optimizations

* **Varlen GDN** — the layer used to pad batches to `[bs, max_query_len]`,
  wasting up to ~150× compute on mixed decode+prefill steps. Decode rows
  take the O(1) recurrent path; prefill groups run fully varlen (dense
  `[N, H]` projections, a varlen causal-conv Triton kernel, and the fla
  chunk kernel driven by `cu_seqlens` — verified bit-equivalent to the
  padded call).
* **Fused plumbing** — a fused g/β Triton kernel replaces the eager fp32
  elementwise chain; Gemma-style norms and the MLP activation run on
  FlashInfer fused kernels; decode CUDA graphs capture lm_head + a
  two-stage fused gumbel-max sampling kernel.
* **In-place recurrent decode** — the decode step used to gather a 450MB
  state batch, run the recurrent, and scatter back; a custom kernel now
  indexes pool rows directly (~40% of decode time saved at bs≈218).
* **Copy-free GDN prefill plumbing** — profiling against vLLM found the
  fla chunk kernel's input guard re-copying the q/k/v slabs (3×64MB per
  layer, ~1.7% of prefill time): the varlen causal-conv kernel now writes
  its output as three separately-contiguous `[N, H, D]` slabs so nothing
  downstream copies.  RoPE and the attention output gate got the same
  treatment — one in-place Triton kernel each instead of a ~16-kernel
  eager chain plus full-head copies (bit-exact vs the eager forms;
  `NANOVLLM_FUSED_ROPE=0` A/Bs it).  Net: prefill 61.0k → 63.0k tok/s,
  taking the hybrid model from −1.8% to **+1.5%** vs vLLM.
* **Single-token conv kernel** — decode steps used to roll the causal-conv
  state with an eager gather → cat → conv1d → scatter chain (four
  full-state memory passes, ~18k kernels each of cat and scatter per
  mixed run — visible only in mixed serving, where decode rows share
  batches with prefill chunks and fall outside CUDA graphs).  A dedicated
  Triton kernel does window-conv + SiLU + state roll in one pass
  (state roll bit-exact, 2.6× per layer at bs≈218): mixed 22.8k → 24.1k
  tok/s (**104% of vLLM**), pure decode 20.4k → 21.9k (**109%**).

## Speculative decoding

Two modes, gated by `num_spec_tokens=K`, verified by strict per-position
rejection sampling (accepted-sequence distribution provably matches plain
autoregressive sampling; `benchmarks/verify_spec.py`):

```python
llm = LLM(path, num_spec_tokens=4)                      # Jacobi parallel draft
llm = LLM(target, num_spec_tokens=4, spec_draft_model=draft)  # draft model
```

Honest negative results (documented, not hidden):

* **No win at 0.6B–4B scale**: Jacobi drafting is 2–4× slower than the
  CUDA-graph baseline; 0.6B→4B draft speculation only reaches 0.05–0.13
  acceptance. The pipeline pays two multi-row forwards + 151k-vocab
  softmaxes per step — it needs a 7B+ target to pay off.
* **Layer-skipped self-speculation fails**: skipping one layer drops draft
  match from ~0.5 to ~0.14 — these models have no layer redundancy.
* **DFlash block-diffusion draft ported and integrated**
  (`models/dflash.py`) — bit-identical to the official implementation on
  identical inputs, but acceptance collapses 10–13× below official on the
  same prompts: the draft consumes the *target's* hidden states as context,
  and bf16 reimplementation rounding differences (invisible in final
  logits) amplify through its attention. Documented as an inherent limit of
  non-bit-identical reimplementations, with the amplification chain
  measured layer by layer.

## Project structure

```
nanovllm/
├── engine/            # scheduler (legacy/continuous/async), model runner,
│                      #   block manager, shm-broadcast worker loop
├── layers/            # attention (paged KV + flash varlen), rope/MRoPE,
│                      #   parallel linear, sampler (+ gumbel triton kernel)
├── models/            # qwen3 (dense), qwen3_5 (hybrid GDN + attention),
│                      #   qwen3_5_vision (ViT tower), dflash (draft)
└── utils/             # loader, parallel state, multimodal preprocessing
benchmarks/            # throughput benches (nano vs vLLM), spec verification
tests/                 # parity checks vs transformers / across parallel
                      #   layouts / async; ref_mm/ regenerates the
                      #   multimodal golden reference
example*.py            # text, hybrid-model and multimodal usage
```

## Validation

Every subsystem has a runnable check against a reference implementation
(all green on 2× RTX 5090, torch 2.13 / cu130):

```bash
python tests/ref_check_qwen35.py          # hybrid LM: prefill logits + greedy decode vs HF
python tests/ref_mm/gen_reference.py      # regenerate multimodal golden reference (~310MB, gitignored)
python tests/mm_vision_parity_qwen35.py   # vision tower: 29 stages bit-exact vs HF (fp32)
python tests/mm_check_qwen35.py           # multimodal e2e: token-exact vs HF under chunked
                                          #   prefill with the boundary inside the image region
python tests/parallel_check.py            # tp/pp/dp layouts vs single GPU
python tests/async_check.py               # async scheduler output equivalence
```

`tests/smoke_check_qwen35.py` (random weights, loose thresholds) also runs;
its pass rate is unchanged by this work.

## Benchmark: head-to-head vs vLLM

Same workload, same GPU, both engines in their best scheduling config
(vLLM `async_scheduling`; nano-vllm `continuous_batching +
async_scheduling`), shape-matched warmup, 3 timed reps — median shown
(reps vary <2%). Reproduce with `benchmarks/bench_vs_vllm.py`.

**Hardware/software**: 1× RTX 5090, torch 2.13 / cu130, vLLM 0.27.1,
bf16, temperature 0.6, `ignore_eos`, 256 requests, `max_model_len` 4096.
Raw per-rep output: `benchmarks/results/vs-vllm-rtx5090.log`.

**Qwen3.5-2B** (the hybrid GDN model ported in this fork — vLLM runs its
hand-tuned kernels, nano-vllm the clean-room port):

| Workload | nano-vllm | vLLM 0.27.1 | nano / vLLM |
|---|---|---|---|
| prefill-bound — 2048-token prompts, 2 out | **62.5k** total tok/s | 62.1k | **100.6%** |
| decode-bound — 64-token prompts, 256 out | **21.9k** total / 17.5k out tok/s | 20.1k / 16.0k | **108.9%** |
| mixed serving — in/out ~ U(100, 1024) | **24.1k** total / 11.7k out tok/s | 23.1k / 11.2k | **104.3%** |

**Qwen3-0.6B** (dense model, upstream code path + this fork's async
scheduling):

| Workload | nano-vllm | vLLM 0.27.1 | nano / vLLM |
|---|---|---|---|
| mixed serving — in/out ~ U(100, 1024) | **23.6k** total / 11.4k out tok/s | 23.5k / 11.4k | **100.4%** |
| decode-bound — 64-token prompts, 256 out | **45.2k** total / 36.2k out tok/s | 43.5k / 34.8k | **103.9%** |

Reading: nano-vllm now beats vLLM on every regime of both models.  This
was not always so: mixed throughput was **9.5k tok/s (~41% of vLLM)**
before the varlen-GDN and unified-async rework; prefill then sat 2%
behind until a profile pass found fla's input-guard copy chain; and the
last mixed-mode deficit (98.7%) traced to the eager decode conv chain
that mixed batches run outside CUDA graphs — all three stories are in the
optimization notes below.

```bash
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py            # both engines, mixed
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py --mode decode
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py --mode prefill
# per-mode workloads are defined in the script header
```

Further benches: `benchmarks/bench1.py` (nano standalone),
`benchmarks/bench.py` (vLLM baseline), `benchmarks/bench_parallel.py`
(tp/pp/dp scaling).

## License

MIT — upstream © 2025 Xingkai Yu ([GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm));
modifications in this fork © 2026 sileaver.
