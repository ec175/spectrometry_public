# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""gpu.py - optional CuPy backend. GPU by default, silent-but-loud CPU fallback.

Set EK_CUPY=0 to force CPU. EK_VRAM_MB caps the pool (default 1400) so several render
lanes can share one card.

GOTCHA: a system CUDA 11 toolkit on PATH makes cupy-cuda12x fail to find CUDA 12 headers.
We scrub CUDA_PATH and the toolkit's PATH entries FOR THIS PROCESS ONLY, and rely on the
pip `cupy-cuda12x[ctk]` wheels. Never edit the system environment - other tools on this
machine depend on the v11 pin.
"""
from __future__ import annotations

import os

GPU = False
cp = None
_gf = None

if os.environ.get("EK_CUPY", "1") != "0":
    try:
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as _cp
            from cupyx.scipy.ndimage import gaussian_filter as _gf_gpu
        (_cp.zeros(8, dtype=_cp.float32) + 1).sum()      # force the JIT now, not mid-render
        _cp.get_default_memory_pool().set_limit(
            size=int(os.environ.get("EK_VRAM_MB", "1400")) * 1024 * 1024)
        cp, _gf, GPU = _cp, _gf_gpu, True
    except Exception as e:
        print("[gpu] CuPy unavailable (%s: %s) - using CPU. Slower, not worse."
              % (type(e).__name__, str(e)[:120]))
        cp = None

import numpy as np  # noqa: E402


def to_gpu(a):
    return cp.asarray(a) if GPU else a


def to_cpu(a):
    return cp.asnumpy(a) if (GPU and isinstance(a, cp.ndarray)) else a


def zeros(shape, dtype=np.float32):
    return cp.zeros(shape, dtype=dtype) if GPU else np.zeros(shape, dtype=dtype)


def gaussian(a, sigma):
    """Gaussian blur on (H, W, 3), sigma in pixels."""
    if GPU and isinstance(a, cp.ndarray):
        return _gf(a, (sigma, sigma, 0), mode="nearest")
    from .canvas import _blur_direct
    return _blur_direct(a, sigma)


def info():
    if not GPU:
        return "CPU (numpy)"
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        name = "CUDA device"
    lim = cp.get_default_memory_pool().get_limit() / 1048576
    return "GPU: %s, pool cap %.0f MB" % (name, lim)
