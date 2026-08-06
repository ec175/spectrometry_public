"""gpu.py — CuPy backend for the hot per-frame math (LBM lattice, splats, blurs). GPU BY DEFAULT.

Same contract as ``Oscilloscope\\osc\\gpu.py`` (Ethan's standing policy 2026-07-18: the GPU is the
DEFAULT for render work, opt-OUT not opt-in). Set ``WT_CUPY=0`` (or ``OSC_CUPY=0``) to force the
CPU path. Everything degrades to numpy/scipy transparently with a loud warning, so a broken
driver can only slow a render, never stop it.

GOTCHA (this machine): the global ML tooling exports ``CUDA_PATH=...\\CUDA\\v11.3`` and puts its
bin on PATH, so cupy-cuda12x would JIT against the CUDA **11** NVRTC and die with "CUDA versions
below 12 are not supported". We therefore scrub both, FOR THIS PROCESS ONLY, before importing
cupy and rely on the pip-installed ``cupy-cuda12x[ctk]`` wheels. Never edit the system env —
the global ML stack needs 11.3 (see the `cuda-path-v11-gotcha` memory).

Exports: ``GPU`` (bool), ``xp`` (cupy or numpy), ``gaussian_filter``, ``map_coordinates``,
``asnumpy(a)``, ``asarray(a)``, ``scatter_add(a, idx, v)``.
"""
from __future__ import annotations

import os

GPU = False

_want = os.environ.get("WT_CUPY", os.environ.get("OSC_CUPY", "1")) != "0"

if _want:
    try:
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import cupy as xp
        from cupyx import scatter_add as _scatter_add
        from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
        (xp.zeros(8, dtype=xp.float32) + 1).sum()          # force a tiny JIT compile now
        GPU = True
    except Exception as e:                                  # any failure -> CPU fallback
        print(f"[gpu] WARNING: GPU backend unavailable ({type(e).__name__}: {e}) — "
              "falling back to CPU (slow). Fix CUDA or set WT_CUPY=0 to silence.")
        import numpy as xp
        from scipy.ndimage import gaussian_filter, map_coordinates
        _scatter_add = None
else:
    import numpy as xp
    from scipy.ndimage import gaussian_filter, map_coordinates
    _scatter_add = None


def asnumpy(a):
    return xp.asnumpy(a) if GPU else a


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


def scatter_add(a, idx, v):
    """``a[idx] += v`` with duplicate indices accumulating (np.add.at semantics)."""
    if GPU:
        _scatter_add(a, idx, v)
    else:
        xp.add.at(a, idx, v)
