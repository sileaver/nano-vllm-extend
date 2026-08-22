"""FlashInfer 环境准备 (import flashinfer 之前必须调用)."""

import os

import torch


def setup_flashinfer_env():
    """Work around SM 12.x capability detection on older CUDA runtimes.

    SM 12.x needs CUDA >= 12.9 for ``get_device_capability``; with older
    torch builds the detected arch list ends up empty and flashinfer
    raises the misleading "requires GPUs with sm75 or higher".  Forcing
    the arch lets the pre-built cubin / jit-cache kernels load without
    nvcc.  Override via FLASHINFER_CUDA_ARCH_LIST if needed.
    """
    major, _ = torch.cuda.get_device_capability(0)
    if major == 12 and float(torch.version.cuda) < 12.9:
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
