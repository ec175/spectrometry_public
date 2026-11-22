# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""gpu.py - CuPy backend, GPU BY DEFAULT (library standing policy, 2026-07-18).

Ported verbatim in spirit from ..\Oscilloscope\osc\gpu.py - same per-process CUDA scrub,
same loud-warning fallback. Set GL_CUPY=0 to force numpy.

GOTCHA (this machine): the global ML tooling exports CUDA_PATH=...\CUDA\v11.3 and puts its
bin on PATH, so cupy-cuda12x would JIT against the CUDA 11 NVRTC and die. We hide it FROM
THIS PROCESS ONLY before importing cupy. The system variable is never modified.

Exports: GPU (bool), xp, gaussian_filter, asnumpy, asarray, Gen (a seeded RNG on the
active backend with .normal/.random/.permutation).
"""
from __future__ import annotations

import os

GPU = False

if os.environ.get("GL_CUPY", "1") != "0":
    try:
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import cupy as xp
        from cupyx.scipy.ndimage import gaussian_filter
        (xp.zeros(8, dtype=xp.float32) + 1).sum()          # force a tiny JIT compile now
        GPU = True
    except Exception as e:
        print(f"[gpu] WARNING: GPU backend unavailable ({type(e).__name__}: {e}) - "
              "falling back to CPU (slow). Fix CUDA or set GL_CUPY=0 to silence.")
        import numpy as xp
        from scipy.ndimage import gaussian_filter
else:
    import numpy as xp
    from scipy.ndimage import gaussian_filter


def asnumpy(a):
    return xp.asnumpy(a) if GPU else a


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


class _Rng:
    """The three draws the solver makes, spelled the same on both backends.

    cupy's Generator has `standard_normal` but NOT `normal`, and its `random` does not take a
    `dtype=` the way numpy's does on every version. Papering over that here keeps world.py
    free of `if GPU:` branches - which matters, because a backend branch inside the solver is
    a branch where the CPU preview and the GPU final can silently diverge."""

    __slots__ = ("g",)

    def __init__(self, seed: int):
        self.g = xp.random.default_rng(seed)

    def random(self, n, dtype=None):
        a = self.g.random(n)
        return a.astype(xp.float32) if dtype is not None else a

    def normal(self, loc, scale, n):
        return (self.g.standard_normal(n, dtype=xp.float32) * xp.float32(scale)
                + xp.float32(loc))

    def permutation(self, n):
        # cupy's Generator has no `permutation` either; argsort of uniform keys is the same
        # thing and is what a GPU wants anyway (one sort, no gather-scatter shuffle).
        if hasattr(self.g, "permutation"):
            return self.g.permutation(n)
        return xp.argsort(self.g.random(int(n)))


def Gen(seed: int):
    """Seeded generator on the active backend."""
    return _Rng(seed)
