# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""gpu.py — CuPy backend for the hot per-frame math. GPU BY DEFAULT.

The GPU (RTX 2060) is the default backend for all rendering — Clancy's standing policy
(2026-07-18). Set `PF_CUPY=0` to force the CPU path. Everything degrades to numpy/scipy
transparently (with a loud warning) if cupy or the CUDA runtime is unavailable, so a broken
driver can never stop a render — only slow it. Ported from `..\\Oscilloscope\\osc\\gpu.py`.

⚠ GOTCHA (this machine): the global ML tooling exports `CUDA_PATH=...\\CUDA\\v11.3` and puts
its bin on PATH, so cupy-cuda12x would JIT-compile against the CUDA **11** NVRTC and die with
"CUDA versions below 12 are not supported". We therefore ship NVRTC 12 via pip
(`cupy-cuda12x[ctk]`) and, for THIS PROCESS ONLY, hide the system toolkit before importing
cupy. **The system env is never modified** — `CUDA_PATH` stays pinned to v11.3 for the global
numpy<2 ML stack, and breaking that is the failure this scrub exists to prevent.

⚠ This module must be imported BEFORE cupy anywhere else. `geo.py` imports it at the top for
that reason, and every render lane is a fresh process, so the scrub runs once per lane.

DISPATCH, NOT A SWITCH
----------------------
The shared primitives (`contour.py`, `draw.py`, the `geo.py` samplers) do NOT read a global
backend flag. They call `xp_of(a)`, which returns numpy or cupy according to the array they
were handed. A scene therefore opts in by putting its DATA on the GPU, and everything
downstream follows automatically.

That is what keeps the twelve shipped static scenes bit-identical: they still hand in numpy,
so they still run the numpy path, instruction for instruction. Mixing a global switch into
those code paths would have forced a re-gate of every approved final.
"""
from __future__ import annotations

import os

GPU = False
cp = None

if os.environ.get("PF_CUPY", "1") != "0":
    try:
        # Hide the system CUDA 11.3 toolkit from THIS PROCESS so cupy discovers the pip
        # CUDA 12 pieces instead of JIT-compiling against 11.3 headers.
        os.environ.pop("CUDA_PATH", None)
        os.environ["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if "NVIDIA GPU Computing Toolkit" not in p)
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as _gf_gpu
        (cp.zeros(8, dtype=cp.float32) + 1).sum()          # force a tiny JIT compile now
        # ⚠ CAP THE POOL, or one lane eats the card. CuPy's default allocator never returns
        # memory to the driver and grows opportunistically: a single 3840x2160 lane was
        # measured holding 6256 of 6442 MB, against a live set nearer 1.2 GB. Uncapped, the
        # second lane simply fails to allocate. The cap costs nothing when it is not binding -
        # the pool just recycles its own blocks instead of asking for more.
        _limit = int(os.environ.get("PF_VRAM_MB", "2600")) * 1024 * 1024
        cp.get_default_memory_pool().set_limit(size=_limit)
        GPU = True
    except Exception as e:                                  # any failure -> CPU fallback
        print(f"[gpu] WARNING: GPU backend unavailable ({type(e).__name__}: {e}) — "
              "falling back to CPU (slow). Fix CUDA or set PF_CUPY=0 to silence.")
        cp = None

import numpy as np
from scipy.ndimage import gaussian_filter as _gf_cpu


def xp_of(*arrays):
    """The array module that owns these arrays: cupy if ANY of them is on the GPU.

    `cupy.get_array_module` only looks at one array; the primitives here routinely mix a
    GPU field with a CPU scalar-turned-array, and a single stray numpy operand would
    otherwise silently pull the whole expression back to the host.
    """
    if GPU:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def asarray(a, dtype=None):
    """Move to the GPU if it is available, else a plain numpy asarray."""
    if GPU:
        return cp.asarray(a, dtype=dtype)
    return np.asarray(a, dtype=dtype)


def asnumpy(a):
    """Back to the host. Cheap no-op for arrays that are already numpy."""
    if GPU and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def gaussian_filter(a, sigma, mode="nearest"):
    """scipy or cupyx gaussian_filter, chosen by where the array already lives."""
    if GPU and isinstance(a, cp.ndarray):
        return _gf_gpu(a, sigma, mode=mode)
    return _gf_cpu(a, sigma, mode=mode)


def sync():
    """Wait for the GPU. Only needed when TIMING - the pipe write syncs on its own."""
    if GPU:
        cp.cuda.runtime.deviceSynchronize()


def mem_mb():
    """(used, total) VRAM in MB, or (0, 0) on CPU. Used by the render loop's first-frame log."""
    if not GPU:
        return 0.0, 0.0
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 1e6, total / 1e6


def scatter_add(dst, idx, vals):
    """`dst.flat[idx] += vals` with REPEATED indices accumulating.

    This is the hottest call in the project — every particle deposits into four pixels, three
    channels, every substep — and it is the one operation where numpy and cupy disagree about
    the fast path:

    * cupy has `cupyx.scatter_add`, a hardware-atomic scatter. It is the whole reason a
      250k-particle system renders at 4K in real budget.
    * numpy's `np.add.at` is a documented slow path (an unbuffered ufunc loop, ~50x slower
      than the alternative). `np.bincount` with `weights` and `minlength` does the same
      accumulation vectorised, and is what `../../../Spectrometry_MP4/Attractors/at/draw.py` uses.

    Both are exact — this is integer-indexed float accumulation, not a reduction whose order
    changes the answer beyond float rounding.
    """
    if GPU and isinstance(dst, cp.ndarray):
        _scatter_add_gpu(dst, idx, vals)
    else:
        flat = dst.reshape(-1)
        flat += np.bincount(idx, weights=vals, minlength=flat.size).astype(flat.dtype)


if True:                                     # bind the cupy symbol once, not per call
    try:
        from cupyx import scatter_add as _scatter_add_gpu   # noqa: E402
    except Exception:
        _scatter_add_gpu = None


def scatter_max(dst, idx, vals):
    """`dst[idx] = maximum(dst[idx], vals)` with REPEATED indices resolving to the max.

    Needed by `branching`, where many wanderers can freeze into the same column in one substep
    and the column's new height is the highest of them. `np.maximum.at` is the unbuffered
    numpy path; cupyx has a real atomic version.
    """
    if GPU and isinstance(dst, cp.ndarray) and _scatter_max_gpu is not None:
        _scatter_max_gpu(dst, idx, vals)
    else:
        np.maximum.at(dst, idx, vals)


try:
    from cupyx import scatter_max as _scatter_max_gpu    # noqa: E402
except Exception:
    _scatter_max_gpu = None
