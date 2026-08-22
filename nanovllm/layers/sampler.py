import torch
from torch import nn

from nanovllm.utils.flashinfer_env import setup_flashinfer_env


class Sampler(nn.Module):

    def __init__(self, backend: str = "torch"):
        super().__init__()
        self.backend = backend
        if backend == "flashinfer":
            setup_flashinfer_env()
            import flashinfer.sampling
            self._fi = flashinfer.sampling

    @torch.compile
    def _forward_torch(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Gumbel-max 温度采样 (与类别采样等价). 注意 top_k/top_p 仅
        # flashinfer 后端支持, 此处忽略.
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    @torch.compile
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
