import torch
from torch import nn
import torch.nn.functional as F

# Module-level compiled free function: instance-method @torch.compile guards
# on `self`, so the 24 per-layer instances each recompile and exhaust the
# dynamo cache (8 entries) — everything falls back to eager.  A shared
# dynamic-shape kernel compiles once and serves every instance/shape.
@torch.compile(dynamic=True)
def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _silu_and_mul(x)
