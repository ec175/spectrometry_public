"""gpu.py — CuPy backend for the hot per-frame math (bloom, film filter). GPU BY DEFAULT.

The GPU (RTX 2060) is the default backend for all rendering — Ethan's standing policy
(2026-07-18): GPU use is allowed and preferred. Set OSC_CUPY=0 to force the CPU path.
Everything degrades to numpy/scipy transparently (with a loud warning) if cupy or the CUDA
runtime is unavailable, so a broken driver can never stop a render — only slow it.

GOTCHA (this machine): the global ML tooling exports CUDA_PATH=...\\CUDA\\v11.3 and puts its
bin on PATH, so cupy-cuda12x would JIT-compile against the CUDA **11** NVRTC and die with
"CUDA versions below 12 are not supported". We therefore ship NVRTC 12 via the
`nvidia-cuda-nvrtc-cu12` pip package and, for THIS PROCESS ONLY, point CUDA_PATH at it and
prepend its bin dir before importing cupy. The system env is never modified.

Exports: `GPU` (bool), `xp` (cupy or numpy), `gaussian_filter`, `map_coordinates`,
`asnumpy(a)`, `asarray(a)`, `randn(rng_seed, shape)` (normal(0,1) float32 on either backend).
"""
from __future__ import annotations

import os

GPU = False

if os.environ.get("OSC_CUPY", "1") != "0":
    try:
        # Hide the system CUDA 11.3 toolkit from THIS PROCESS so cupy discovers the pip-
        # installed CUDA 12 pieces (cupy-cuda12x[ctk]: nvrtc/runtime/cublas/... wheels)
        # instead of JIT-compiling against 11.3 headers. System env is never modified.
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import cupy as xp
        from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
        (xp.zeros(8, dtype=xp.float32) + 1).sum()               # force a tiny JIT compile now
        GPU = True
    except Exception as e:                                      # any failure -> CPU fallback
        print(f"[gpu] WARNING: GPU backend unavailable ({type(e).__name__}: {e}) — "
              "falling back to CPU (slow). Fix CUDA or set OSC_CUPY=0 to silence.")
        import numpy as xp
        from scipy.ndimage import gaussian_filter, map_coordinates
else:
    import numpy as xp
    from scipy.ndimage import gaussian_filter, map_coordinates


def asnumpy(a):
    return xp.asnumpy(a) if GPU else a


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


def randn(seed, shape):
    """Deterministic N(0,1) float32 of `shape` on the active backend."""
    if GPU:
        return xp.random.default_rng(seed).standard_normal(shape, dtype=xp.float32)
    import numpy as np
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
