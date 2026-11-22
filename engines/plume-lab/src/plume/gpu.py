# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""gpu.py - CuPy backend for the solver and the frame math. GPU BY DEFAULT.

Same contract (and the same CUDA_PATH gotcha) as ``Wind_Tunnel\\wt\\gpu.py`` and
``Oscilloscope\\osc\\gpu.py``: Clancy's standing policy 2026-07-18 is that the GPU is the DEFAULT
for render work, opt-OUT not opt-in. Set ``PL_CUPY=0`` (or ``WT_CUPY=0`` / ``OSC_CUPY=0``) to
force the CPU path.

GOTCHA (this machine): the global ML tooling exports ``CUDA_PATH=...\\CUDA\\v11.3`` and puts its
bin on PATH, so cupy-cuda12x would JIT against the CUDA **11** NVRTC and die with "CUDA versions
below 12 are not supported". We scrub both FOR THIS PROCESS ONLY before importing cupy and rely
on the pip-installed ``cupy-cuda12x[ctk]`` wheels. Never edit the system env - the global ML
stack needs 11.3 (see the ``cuda-path-v11-gotcha`` memory). The "CUDA path could not be
detected" warning on startup is expected and harmless.

Exports: ``GPU`` (bool), ``xp``, ``asnumpy``, ``asarray``, ``ElementwiseKernel`` (None on CPU).
"""
from __future__ import annotations

import os

GPU = False
ElementwiseKernel = None

_want = os.environ.get("PL_CUPY", os.environ.get("WT_CUPY", os.environ.get("OSC_CUPY", "1"))) != "0"

if _want:
    try:
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import cupy as xp
        from cupy import ElementwiseKernel        # noqa: F401
        (xp.zeros(8, dtype=xp.float32) + 1).sum()  # force a tiny JIT compile now
        GPU = True
    except Exception as e:                         # any failure -> CPU fallback
        print(f"[gpu] WARNING: GPU backend unavailable ({type(e).__name__}: {e}) - "
              "falling back to CPU (slow). Fix CUDA or set PL_CUPY=0 to silence.")
        import numpy as xp
        ElementwiseKernel = None
else:
    import numpy as xp


def asnumpy(a):
    return xp.asnumpy(a) if GPU else a


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


def sync():
    """Block until the device is idle - only for honest timing, never in the hot path."""
    if GPU:
        xp.cuda.runtime.deviceSynchronize()
