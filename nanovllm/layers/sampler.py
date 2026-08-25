import torch
import triton
import triton.language as tl
from torch import nn

from nanovllm.utils.flashinfer_env import setup_flashinfer_env


@triton.jit
def _gumbel_argmax_stage1(
    LOGITS, TEMPS, SEED, PMAX, PIDX, V, NB,
    sL, BLOCK_V: tl.constexpr,
):
    """Stage 1: each program scores one [BLOCK_V] vocab slice of one row.

    score = logits[row, v] / temp[row] + g, g = -ln(-ln(u)), u ~ U(0,1)
    via philox — the same estimator as Sampler._forward_torch
    (argmax(l/T − ln e), e ~ Exp(1)) fused over the bf16 logits with no
    fp32 [bs, V] materialisation.  Writes per-slice (max, argmax) for the
    tiny stage-2 reduction.  The seed lives in GPU memory and is bumped
    by the caller between CUDA-graph replays (fresh noise every step).
    """
    row = tl.program_id(0)
    pv = tl.program_id(1)
    offs = pv * BLOCK_V + tl.arange(0, BLOCK_V)
    m = offs < V
    l = tl.load(LOGITS + row * sL + offs, mask=m, other=-float("inf")).to(tl.float32)
    temp = tl.load(TEMPS + row)
    seed = tl.load(SEED)
    u = tl.rand(seed, (offs + row * V).to(tl.int32))
    u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-6)
    g = -tl.log(-tl.log(u))
    score = l / temp + g
    score = tl.where(m, score, -float("inf"))
    tl.store(PMAX + row * NB + pv, tl.max(score, axis=0))
    tl.store(PIDX + row * NB + pv, tl.argmax(score, axis=0) + pv * BLOCK_V)


@triton.jit
def _gumbel_argmax_stage2(PMAX, PIDX, OUT, NB, NB_PAD: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, NB_PAD)
    m = offs < NB
    mx = tl.load(PMAX + row * NB + offs, mask=m, other=-float("inf"))
    ix = tl.load(PIDX + row * NB + offs, mask=m, other=0)
    loc = tl.argmax(mx, axis=0)
    tl.store(OUT + row, tl.sum(tl.where(offs == loc, ix, 0)))


def gumbel_argmax(logits: torch.Tensor, temps: torch.Tensor,
                  out: torch.Tensor, seed: torch.Tensor,
                  pmax: torch.Tensor, pidx: torch.Tensor):
    """logits [bs, V] bf16 (row-contiguous), temps [≥bs] fp32, seed int32
    scalar tensor (value must differ between CUDA-graph replays);
    pmax/pidx [≥bs, NB] fp32/int32 scratch (NB = cdiv(V, 2048))."""
    assert logits.stride(1) == 1
    bs, vocab = logits.shape
    nb = pmax.shape[1]
    nb_pad = triton.next_power_of_2(nb)
    _gumbel_argmax_stage1[(bs, nb)](
        logits, temps, seed, pmax, pidx, vocab, nb,
        logits.stride(0), BLOCK_V=2048, num_warps=8,
    )
    _gumbel_argmax_stage2[(bs,)](pmax, pidx, out, nb, NB_PAD=nb_pad)


class Sampler(nn.Module):

    def __init__(self, backend: str = "torch"):
        super().__init__()
        self.backend = backend
        if backend == "flashinfer":
            setup_flashinfer_env()
            import flashinfer.sampling
            self._fi = flashinfer.sampling

    # dynamic=True: batch size changes every decode step as sequences
    # finish; static compiles exhaust the dynamo cache and fall to eager.
    @torch.compile(dynamic=True)
    def _forward_torch(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Gumbel-max 温度采样 (与类别采样等价): argmax(softmax(l/T)/e) ≡
        # argmax(l/T − log e) (softmax 归一化常数在 argmax 中消去), 所以
        # 无需物化 [bs, vocab] 的 softmax 概率. 注意 top_k/top_p 仅
        # flashinfer 后端支持, 此处忽略.
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        sample_tokens = logits.sub_(
            torch.empty_like(logits).exponential_(1).clamp_min_(1e-10).log_()
        ).argmax(dim=-1)
        return sample_tokens

    @torch.compile(dynamic=True)
    def _sample_with_probs_torch(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 同 Gumbel-max 温度采样, 但非 in-place 且额外返回 softmax 分布 —
        # 投机解码的拒绝采样验证需要草稿模型生成 d_i 所用的精确分布 p_i.
        # (不能用上面的 in-place 版本: div_ 会把 probs 破坏成噪声比.)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens, probs

    def _forward_flashinfer(self, logits: torch.Tensor, temperatures: torch.Tensor,
                            top_k: torch.Tensor, top_p: torch.Tensor):
        # FlashInfer 融合 top-k/top-p 拒绝采样内核 (无显式排序).
        # 温度缩放沿用 torch 路径的语义: 先对 logits 除温度.
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        return self._fi.top_k_top_p_sampling_from_logits(logits, top_k, top_p)

    def sample_with_probs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        assert self.backend == "torch", "spec v1: sampling_backend must be 'torch'"
        return self._sample_with_probs_torch(logits, temperatures)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_k: torch.Tensor | None = None, top_p: torch.Tensor | None = None):
        if self.backend == "flashinfer":
            return self._forward_flashinfer(logits, temperatures, top_k, top_p)
        return self._forward_torch(logits, temperatures)
