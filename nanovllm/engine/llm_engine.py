import atexit
from collections import deque
from dataclasses import dataclass, field, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler, SchedulerOutput
from nanovllm.engine.async_scheduler import AsyncScheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.multimodal import (
    mm_token_types_from_ids, qwen35_mrope_positions)


@dataclass
class _InflightStep:
    """One GPU step launched but whose sampled tokens are not yet
    processed on the CPU (reaped when its CUDA event fires)."""
    ring_idx: int
    seqs: list[Sequence]
    # Launch-time snapshots: a later schedule may overwrite
    # seq.num_scheduled_tokens / seq.is_prefill before this step reaps.
    sched_counts: list[int] = field(default_factory=list)
    pf_flags: list[bool] = field(default_factory=list)
    # seq_id -> row in this step's batch: the gather source map for the
    # next step's on-device input assembly.
    row_of: dict[int, int] = field(default_factory=dict)
    # Finished sequences whose KV/slot/GPU-row release must wait for this
    # step (the last one still referencing them) to complete.
    deferred_free: list[Sequence] = field(default_factory=list)


def dp_engine_worker(config: Config, dp_idx: int, req_queue, result_queue):
    """One DP replica: a full engine (own scheduler + tp*pp runners) fed
    from its request shard.  Lives for the LLM's lifetime, rounds demarcated
    by "round" markers; results stream back tagged with the driver's
    request ids."""
    config.data_parallel_size = 1
    config.dist_port = 2333 + dp_idx
    engine = LLMEngine(config=config, dp_idx=dp_idx)
    seq_to_req: dict[int, int] = {}
    try:
        while True:
            msg = req_queue.get()
            kind = msg[0]
            if kind == "exit":
                return
            elif kind == "req":
                _, req_id, prompt, sp = msg
                seq = engine.add_request(prompt, sp)
                seq_to_req[seq.seq_id] = req_id
            elif kind == "round":
                while not engine.is_finished():
                    finished, _ = engine.step()
                    for seq_id, tokens in finished:
                        result_queue.put(("finished", seq_to_req.pop(seq_id), tokens))
                for seq_id, tokens in engine.drain():
                    result_queue.put(("finished", seq_to_req.pop(seq_id), tokens))
                result_queue.put(("round_done", dp_idx))
    finally:
        engine.exit()


class LLMEngine:

    def __init__(self, model=None, *, config: Config | None = None,
                 dp_idx: int = 0, **kwargs):
        if config is None:
            config_fields = {field.name for field in fields(Config)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
            config = Config(model, **config_kwargs)
        self.config = config
        self.dp_idx = dp_idx
        Sequence.block_size = config.kvcache_block_size
        self.async_scheduling = config.async_scheduling
        self.continuous_batching = config.continuous_batching
        self.collect_timing = kwargs.get("collect_timing", False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # Timing collection (single-group only: DP replicas collect their
        # own stats internally; the list stays empty on the driver).
        self._finished_seqs: list[Sequence] = []
        atexit.register(self.exit)
        if config.data_parallel_size > 1:
            self._init_data_parallel(config)
        else:
            self._init_single_group(config)

    # ------------------------------------------------------------------
    # Process layout
    # ------------------------------------------------------------------

    def _init_single_group(self, config: Config):
        """One replica group: tp*pp - 1 worker processes plus this process
        as group rank 0."""
        world = config.tensor_parallel_size * config.pipeline_parallel_size
        assert torch.cuda.device_count() >= world, f"{world} ranks need {world} GPUs"
        self.ps = []
        self.events = []
        self.acks = []
        ctx = mp.get_context("spawn")
        for i in range(1, world):
            event = ctx.Event()
            ack = ctx.Event()
            ack.set()  # phantom ack: no command has been written yet
            process = ctx.Process(target=ModelRunner,
                                  args=(config, i, event, self.dp_idx, ack))
            process.start()
            self.ps.append(process)
            self.events.append(event)
            self.acks.append(ack)
        self.model_runner = ModelRunner(config, 0, self.events, self.dp_idx,
                                        self.acks)
        if config.num_spec_tokens > 0:
            assert not config.async_scheduling, \
                "spec v1: async_scheduling unsupported with speculative decoding"
        if self.async_scheduling:
            self.scheduler = AsyncScheduler(config)
        else:
            self.scheduler = Scheduler(config)

        # Async scheduling state: in-flight steps awaiting CPU-side
        # output processing (bounded pipeline depth of 2).
        self._inflight: deque[_InflightStep] = deque()

    def _init_data_parallel(self, config: Config):
        """Data parallelism: dp replica processes, each a full engine on its
        own GPU group with its own scheduler.  The driver only shards
        requests (LPT) and merges results."""
        dp = config.data_parallel_size
        gpus_needed = dp * config.pipeline_parallel_size * config.tensor_parallel_size
        assert torch.cuda.device_count() >= gpus_needed, \
            f"data_parallel={dp} needs {gpus_needed} GPUs"
        assert not self.collect_timing, "collect_timing unsupported with data parallelism"
        self.dp_req_queues = [mp.get_context("spawn").Queue() for _ in range(dp)]
        self.dp_result_queue = mp.get_context("spawn").Queue()
        self.dp_ps = []
        ctx = mp.get_context("spawn")
        for d in range(dp):
            process = ctx.Process(
                target=dp_engine_worker,
                args=(config, d, self.dp_req_queues[d], self.dp_result_queue))
            process.start()
            self.dp_ps.append(process)

    def exit(self):
        # Idempotent: atexit fires again after a manual exit().
        if getattr(self, "config", None) is not None and \
                self.config.data_parallel_size > 1:
            if not hasattr(self, "dp_ps"):
                return
            for q in self.dp_req_queues:
                q.put(("exit",))
            for p in self.dp_ps:
                p.join()
            del self.dp_ps
            return
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int] | dict,
                    sampling_params: SamplingParams, images=None):
        """``prompt`` may be a string, token list, or — for multimodal
        models — a dict ``{"prompt": ..., "images": [...]}`` (``images``
        also accepted as a separate argument).  Images are processed by
        the checkpoint's own processor (Qwen3VL style); the prompt string
        must contain the ``<|vision_start|><|image_pad|><|vision_end|>``
        placeholder the processor expands per image."""
        assert self.config.data_parallel_size == 1, \
            "data parallelism supports generate() only (requests are sharded by the driver)"
        if isinstance(prompt, dict):
            images = prompt.get("images", images)
            prompt = prompt["prompt"]
        pixel_values = image_grid_thw = mrope_positions = None
        rope_delta = 0
        if images:
            assert self.config.vision_config is not None, \
                "images passed to a text-only engine (or NANOVLLM_QWEN35_TEXTONLY=1)"
            if not hasattr(self, "processor"):
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.config.model)
            assert isinstance(prompt, str), "image prompts must be text"
            inputs = self.processor(text=[prompt], images=images,
                                    return_tensors="pt", padding=True)
            assert getattr(inputs, "pixel_values_videos", None) is None, \
                "video inputs are not supported"
            prompt = inputs.input_ids[0].tolist()
            # bf16 is lossless here — the tower casts to its dtype on
            # entry (the reference's pixel_values.type(visual.dtype)), and
            # halved payloads keep the TP shm command segment small.
            pixel_values = inputs.pixel_values.to(torch.bfloat16)
            image_grid_thw = inputs.image_grid_thw.tolist()
            mm_types = (inputs.mm_token_type_ids[0].tolist()
                        if getattr(inputs, "mm_token_type_ids", None) is not None
                        else mm_token_types_from_ids(
                            prompt, self.config.image_token_id,
                            self.config.video_token_id))
            merge = self.config.vision_config.spatial_merge_size
            n_tokens = sum(t * h * w // merge ** 2
                           for t, h, w in image_grid_thw)
            assert prompt.count(self.config.image_token_id) == n_tokens, \
                "image placeholder / grid token count mismatch"
            mrope_positions, rope_delta = qwen35_mrope_positions(
                prompt, mm_types, image_grid_thw, merge)
        elif isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params,
                       pixel_values=pixel_values,
                       image_grid_thw=image_grid_thw,
                       mrope_positions=mrope_positions,
                       rope_delta=rope_delta)
        if self.collect_timing:
            import time
            seq.timing = True
            seq.arrival_time = time.time()
            self._finished_seqs.append(seq)
        self.scheduler.add(seq)
        return seq

    def reset_timing(self):
        """Clear accumulated timing data (call after warmup)."""
        self._finished_seqs.clear()

    # ------------------------------------------------------------------
    # Synchronous step
    # ------------------------------------------------------------------

    def _step_sync(self):
        """Schedule → execute → postprocess (continuous or legacy)."""
        output: SchedulerOutput = self.scheduler.schedule()
        seqs = output.scheduled_seqs
        if output.is_speculative:
            accepted = self.model_runner.call("run_speculative", seqs)
            num_tokens = self.scheduler.postprocess_speculative(seqs, accepted)
            # Report the *accepted* token count (decode semantics: negative),
            # not the K+1 verified positions — otherwise Decode tok/s would
            # be inflated by the draft width.
            num_tokens = -num_tokens
        else:
            token_ids = self.model_runner.call("run", seqs)
            self.scheduler.postprocess(seqs, token_ids, output.is_prefill)
            num_tokens = output.num_scheduled_tokens
        self.model_runner.call("free_finished_gpu_rows", seqs)
        finished = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs if seq.is_finished
        ]
        return finished, num_tokens

    # ------------------------------------------------------------------
    # Asynchronous step (vLLM V1 style)
    #
    # Steady-state decode never blocks the CPU on the GPU: while step N
    # executes, the CPU schedules step N+1, builds its metadata, gathers
    # its input ids ON DEVICE from step N's sampled-token ring slot, and
    # enqueues execute + sample.  Outputs come back through pinned memory
    # guarded by CUDA events and are processed (token append, EOS/finish
    # detection) one to two steps late, off the critical path.  Batches
    # containing prefill chunks (and preemption, which accompanies them)
    # fall out of the async pipeline entirely: the pipeline is drained
    # and the step runs synchronously with a caught-up scheduler.
    # ------------------------------------------------------------------

    def _step_async(self):
        finished_all = []
        mr = self.model_runner

        # 1) Reap completed steps, oldest first.
        while self._inflight:
            step = self._inflight[0]
            tokens = mr.poll_sampled(step.ring_idx, len(step.seqs))
            if tokens is None:
                break
            self._inflight.popleft()
            self._process_async_output(step, tokens, finished_all)

        # 2) Pipeline depth cap: schedule() below reserves KV placeholders
        #    per step, so never queue more than 2 steps ahead.
        if len(self._inflight) >= 2:
            return finished_all, 0

        # 3) Schedule and launch the next step.
        output: SchedulerOutput = self.scheduler.schedule()
        seqs = output.scheduled_seqs
        if not seqs:
            return finished_all, output.num_scheduled_tokens

        if output.is_prefill:
            # Prefill batches now enter the async pipeline too (no drain):
            # unified placeholder accounting keeps num_computed_tokens
            # exact under lag, so chunk continuation and decode rows
            # sharing the batch are both correct.  Prompt tokens come from
            # the CPU, decode-prefix rows gather theirs from the previous
            # step's ring slot.
            seqs = [s for s in seqs if not s.is_finished]
            if not seqs:
                return finished_all, output.num_scheduled_tokens
            n1 = 0
            for s in seqs:
                if s.is_prefill:
                    break
                n1 += 1
            if self._inflight:
                src = self._inflight[-1]
                if all(s.seq_id in src.row_of for s in seqs[:n1]):
                    prev_rows = [src.row_of[s.seq_id] for s in seqs[:n1]]
                    mr.call("prepare_mixed_async", seqs, src.ring_idx,
                            prev_rows, n1)
                else:
                    # A decode row missing from the source batch (budget
                    # exclusion) — rare; fall back to a drained step.
                    self._drain_inflight(finished_all)
                    mr.call("prepare_step", seqs)
            else:
                # Pipeline empty: everything is reaped, CPU tokens exact.
                mr.call("prepare_step", seqs)
        else:
            seqs = [s for s in seqs if not s.is_finished]
            if not seqs:
                return finished_all, output.num_scheduled_tokens
            if self._inflight:
                src = self._inflight[-1]
                if all(s.seq_id in src.row_of for s in seqs):
                    # Steady state: gather input ids from the previous
                    # step's ring slot — the CPU never sees the tokens.
                    prev_rows = [src.row_of[s.seq_id] for s in seqs]
                    mr.call("prepare_step_async", seqs, src.ring_idx, prev_rows)
                else:
                    # A decode seq absent from the source batch (budget
                    # exclusion): fall back to a drained synchronous step.
                    self._drain_inflight(finished_all)
                    mr.call("prepare_step", seqs)
            else:
                # First decode step after a fully-reaped state — tokens are
                # already on the CPU, plain prepare is exact.
                mr.call("prepare_step", seqs)

        mr.call("execute_model")
        ring_idx = mr.begin_ring_slot()
        mr.call("sample_async", ring_idx)
        self._inflight.append(_InflightStep(
            ring_idx, seqs,
            [s.num_scheduled_tokens for s in seqs],
            [s.is_prefill for s in seqs],
            {s.seq_id: i for i, s in enumerate(seqs)}))
        return finished_all, output.num_scheduled_tokens

    def _process_async_output(self, step: _InflightStep, tokens: list[int],
                              finished_all: list):
        """Apply a reaped step's tokens (with its launch-time snapshots);
        defer KV-block releases of sequences that newer in-flight steps
        still reference (the wasted extra row)."""
        finished = self.scheduler.update_from_output(
            step.seqs, tokens, step.sched_counts, step.pf_flags)
        finished_all.extend(
            (s.seq_id, s.completion_token_ids) for s in finished)
        if finished:
            self.model_runner.call("free_finished_gpu_rows", finished)
            if self._inflight:
                self._inflight[-1].deferred_free.extend(finished)
            else:
                self._release_blocks(finished)
        if step.deferred_free:
            self._release_blocks(step.deferred_free)

    def _release_blocks(self, seqs: list[Sequence]):
        for seq in seqs:
            self.scheduler.release(seq)

    def _drain_inflight(self, finished_all: list):
        """Wait for every in-flight step and process its outputs."""
        while self._inflight:
            step = self._inflight.popleft()
            tokens = self.model_runner.wait_sampled(
                step.ring_idx, len(step.seqs))
            self._process_async_output(step, tokens, finished_all)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self):
        if self.config.data_parallel_size > 1:
            raise NotImplementedError(
                "data parallelism supports generate() only — stepping happens inside the replica workers")
        if self.async_scheduling:
            return self._step_async()
        else:
            return self._step_sync()

    def is_finished(self):
        if self.config.data_parallel_size > 1:
            raise NotImplementedError(
                "data parallelism supports generate() only — state lives inside the replica workers")
        return self.scheduler.is_finished()

    def drain(self) -> list[tuple[int, list[int]]]:
        """Process the trailing in-flight steps after the last step():
        each finished sequence runs one wasted extra step, so its KV
        blocks / GPU rows are only released here."""
        tail: list[tuple[int, list[int]]] = []
        if self.async_scheduling:
            self._drain_inflight(tail)
        return tail

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """``prompts`` entries may be dicts (multimodal): ``{"prompt": ...,
        "images": [...]}`` — see add_request."""
        if self.config.data_parallel_size > 1:
            assert not any(isinstance(p, dict) for p in prompts), \
                "data parallelism supports text prompts only"
            return self._generate_dp(prompts, sampling_params, use_tqdm)
        return self._generate_single(prompts, sampling_params, use_tqdm)

    # ------------------------------------------------------------------
    # Data-parallel generate: LPT-shard across replicas, merge in order
    # ------------------------------------------------------------------

    def _generate_dp(self, prompts, sampling_params, use_tqdm):
        dp = self.config.data_parallel_size
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Longest-processing-time bin packing on prompt+output tokens: for
        # ragged request mixes this balances replica completion times far
        # better than round-robin.
        costs = []
        for prompt, sp in zip(prompts, sampling_params):
            n = len(self.tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)
            costs.append(n + sp.max_tokens)
        order = sorted(range(len(prompts)), key=lambda i: -costs[i])
        loads = [0] * dp
        for i in order:
            d = loads.index(min(loads))
            loads[d] += costs[i]
            self.dp_req_queues[d].put(("req", i, prompts[i], sampling_params[i]))
        for q in self.dp_req_queues:
            q.put(("round",))

        pbar = tqdm(total=len(prompts), desc="Generating (DP)",
                    dynamic_ncols=True, disable=not use_tqdm)
        outputs: dict[int, list[int]] = {}
        rounds_done = 0
        while rounds_done < dp:
            msg = self.dp_result_queue.get()
            if msg[0] == "finished":
                _, req_id, token_ids = msg
                outputs[req_id] = token_ids
                pbar.update(1)
            else:
                rounds_done += 1
        pbar.close()
        return [{"text": self.tokenizer.decode(outputs[i]),
                 "token_ids": outputs[i]} for i in range(len(prompts))]

    # ------------------------------------------------------------------
    # Single-group generate
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            finished, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
                pbar.update(1)
        # Drain the trailing in-flight steps: releases their KV blocks /
        # GPU rows and reports finishes detected in the very last steps.
        for seq_id, token_ids in self.drain():
            outputs[seq_id] = token_ids
            pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        if self.collect_timing:
            ttfts, tpots = [], []
            for seq in self._finished_seqs:
                if seq.first_token_time is not None and seq.arrival_time > 0:
                    ttft = seq.first_token_time - seq.arrival_time
                    ttfts.append(ttft)
                if len(seq.token_times) >= 2:
                    times = seq.token_times
                    total_decode_time = times[-1] - times[0]
                    tpot = total_decode_time / (len(times) - 1)
                    tpots.append(tpot)
            stats = {
                "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else 0,
                "ttft_p50": sorted(ttfts)[len(ttfts)//2] if ttfts else 0,
                "ttft_p99": sorted(ttfts)[int(len(ttfts)*0.99)] if ttfts else 0,
                "tpot_mean": sum(tpots) / len(tpots) if tpots else 0,
                "tpot_p50": sorted(tpots)[len(tpots)//2] if tpots else 0,
                "tpot_p99": sorted(tpots)[int(len(tpots)*0.99)] if tpots else 0,
                "num_requests": len(self._finished_seqs),
            }
            self._finished_seqs.clear()
            return outputs, stats

        return outputs
