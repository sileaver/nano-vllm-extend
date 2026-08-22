from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1          # -1 = 关闭 (torch 后端暂不支持 top_k/top_p)
    top_p: float = 1.0       # 1.0 = 关闭
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or positive"
        assert 0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
