# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""splats.py - many radial splats in ONE kernel launch.

`Canvas.radial` is a slice-add against a cached profile, which is the right shape for a
handful of splats. Bio draws one per nucleus plus one per granule plus one per
tri-cellular junction: measured on a 531-cell frame, 1974 calls costing 0.34 s, of which
0.13 s was nothing but `cp.asarray` on a 3-element colour array, once per call.

Batching turns that into one launch: one block per splat, threads striding its bounding
box. Same arithmetic, same cached profile, ~200x less overhead.
"""
from __future__ import annotations

import numpy as np

from . import gpu as G

_SRC = r'''
extern "C" __global__
void splats(const float* cx, const float* cy, const float* rr,
            const float* col, const float* gain,
            const float* lut, int nlut, int nsp,
            float* out, int W, int H)
{
    int k = blockIdx.x;
    if (k >= nsp) return;
    float r = rr[k];
    if (r <= 0.f) return;
    float g = gain[k];
    if (g == 0.f) return;
    int x0 = (int)floorf(cx[k] - r), x1 = (int)ceilf(cx[k] + r);
    int y0 = (int)floorf(cy[k] - r), y1 = (int)ceilf(cy[k] + r);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > W) x1 = W;
    if (y1 > H) y1 = H;
    int bw = x1 - x0, bh = y1 - y0;
    if (bw <= 0 || bh <= 0) return;
    float cr = col[k*3] * g, cg = col[k*3+1] * g, cb = col[k*3+2] * g;
    int n = bw * bh;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int iy = t / bw, ix = t - iy * bw;
        int px = x0 + ix, py = y0 + iy;
        float dx = (float)px + 0.5f - cx[k];
        float dy = (float)py + 0.5f - cy[k];
        float u = sqrtf(dx*dx + dy*dy) / r;
        if (u > 1.f) continue;
        float v = lut[(int)(u * (nlut - 1) + 0.5f)];
        if (v == 0.f) continue;
        int o = (py * W + px) * 3;
        atomicAdd(&out[o],     v * cr);
        atomicAdd(&out[o + 1], v * cg);
        atomicAdd(&out[o + 2], v * cb);
    }
}
'''

_KERNEL = None


def warm():
    global _KERNEL
    if _KERNEL is None and G.GPU:
        k = G.cp.RawKernel(_SRC, "splats")
        k.compile()
        _KERNEL = k
    return _KERNEL


def draw(target, cx, cy, r, cols, gains, prof_name, prof_fn, W, H):
    """Add `len(cx)` radial splats into `target` (H, W, 3)."""
    n = len(cx)
    if n == 0:
        return
    from .cellfield import profile_lut
    if G.GPU and isinstance(target, G.cp.ndarray):
        def f(a):
            return G.to_gpu(np.ascontiguousarray(a, np.float32))
        warm()((n,), (128,),
               (f(cx), f(cy), f(r), f(np.asarray(cols).reshape(-1)), f(gains),
                profile_lut(prof_name, prof_fn, 1024, True), np.int32(1024),
                np.int32(n), target, np.int32(W), np.int32(H)))
        return
    lut = profile_lut(prof_name, prof_fn, 1024, False)
    cols = np.asarray(cols, np.float32).reshape(-1, 3)
    for k in range(n):
        rk = float(r[k])
        if rk <= 0 or gains[k] == 0:
            continue
        x0, x1 = max(0, int(cx[k] - rk)), min(W, int(np.ceil(cx[k] + rk)) + 1)
        y0, y1 = max(0, int(cy[k] - rk)), min(H, int(np.ceil(cy[k] + rk)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = (np.arange(x0, x1) + 0.5) - cx[k]
        dy = (np.arange(y0, y1) + 0.5) - cy[k]
        u = np.hypot(dx[None, :], dy[:, None]) / rk
        v = np.where(u <= 1.0, lut[np.clip(u * 1023, 0, 1023).astype(np.int32)], 0.0)
        target[y0:y1, x0:x1] += (v.astype(np.float32)[..., None]
                                 * (cols[k] * float(gains[k])))
