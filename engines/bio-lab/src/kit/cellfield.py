# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cellfield.py - the cell FILM as an analytic per-pixel field.

WHY THIS REPLACED CONCENTRIC POLYGON FILLS
------------------------------------------
The film used to be `rings` shrunk copies of the cell polygon, each adding one step of a
radial profile. Three things were wrong with that, and all three shipped:

1. BANDING. Rings cost a polygon fill each, so the count was adaptive - 4-6 at wide
   framing. Four steps is not a gradient, it is a stepped pyramid.
2. JUNCTION VOIDS. Corner rounding (Chaikin) cut each polygon inward at every corner.
   Three neighbours each cut their own, so a black triangle opened at every tri-cellular
   junction - reported as "visual bug at intersections of lots of cells".
3. BLOCKY WALLS. Without rounding the wall is the raw Voronoi facet: straight and sharp.

A clipped Voronoi cell is CONVEX, so about its site it is an intersection of half-planes
{(n_e, d_e)}. For a point q, with m_e = d_e - (q-s).n_e:

    min_e m_e  is EXACTLY the distance to the wall, and its sign is the inside test.

One pass over the edges gives clipping, wall distance and antialiasing at once. Using a
SMOOTH minimum rounds the corners by ~0.7k while leaving the walls themselves untouched,
so neighbours still share their wall exactly and no void can open. No rings, so no
banding at any zoom.

Profile argument is r = 1 - m/R with R the cell equivalent radius: r=1 at the wall, 0 at
the deepest point, the convention `canvas.PROFILES` already used. `wall_gain` adds a band
of ABSOLUTE width on top, which is what a membrane is - a thickness that does not grow
with the cell.
"""
from __future__ import annotations

import numpy as np

from . import gpu as G

_KERNEL_SRC = r'''
extern "C" __global__
void cellfield(const float* nx, const float* ny, const float* dd,
               const int* eoff, const int* bx, const int* by,
               const int* bw, const int* bh,
               const float* sx, const float* sy,
               const float* Rk, const float* Kk,
               const float* col, const float* fg,
               const float* wg, const float* ww, const float* fill,
               const float* lut, int nlut,
               float* out, int W, int H)
{
    int k = blockIdx.x;
    int n = bw[k] * bh[k];
    int e0 = eoff[k], e1 = eoff[k + 1];
    float kk = Kk[k], R = Rk[k];
    float cr = col[k*3], cg = col[k*3+1], cb = col[k*3+2];
    float g = fg[k], wgk = wg[k], wwk = ww[k], fl = fill[k];
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int iy = t / bw[k], ix = t - iy * bw[k];
        int px = bx[k] + ix, py = by[k] + iy;
        if (px < 0 || py < 0 || px >= W || py >= H) continue;
        float qx = (float)px + 0.5f - sx[k];
        float qy = (float)py + 0.5f - sy[k];
        float mmin = 1e30f;
        for (int e = e0; e < e1; ++e) {
            float m = dd[e] - (qx * nx[e] + qy * ny[e]);
            if (m < mmin) mmin = m;
        }
        float ms = mmin;
        if (kk > 1e-4f && mmin > -4.0f * kk) {
            float s = 0.f;
            for (int e = e0; e < e1; ++e) {
                float m = dd[e] - (qx * nx[e] + qy * ny[e]);
                s += __expf(-(m - mmin) / kk);
            }
            ms = mmin - kk * __logf(s);
        }
        if (ms <= 0.f) continue;
        float a = ms < 1.f ? ms : 1.f;
        float r = 1.f - ms / R;
        if (r < 0.f) r = 0.f;
        if (r > 1.f) r = 1.f;
        float v = lut[(int)(r * (nlut - 1) + 0.5f)] * g + fl;
        if (wgk > 0.f && wwk > 0.f) {
            float u = ms / wwk;
            v += wgk * __expf(-u * u);
        }
        v *= a;
        if (v == 0.f) continue;
        int o = (py * W + px) * 3;
        atomicAdd(&out[o],     v * cr);
        atomicAdd(&out[o + 1], v * cg);
        atomicAdd(&out[o + 2], v * cb);
    }
}
'''

_KERNEL = None
_LUT_CACHE = {}


def warm():
    """Compile now. NVRTC's first compile takes seconds; mid-render it stalls a lane."""
    global _KERNEL
    if _KERNEL is None and G.GPU:
        k = G.cp.RawKernel(_KERNEL_SRC, "cellfield")
        k.compile()
        _KERNEL = k
    return _KERNEL


def profile_lut(name, fn, n=1024, on_gpu=False):
    key = (name, n, on_gpu)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = np.asarray(fn(np.linspace(0.0, 1.0, n)), np.float32)
        if on_gpu:
            lut = G.to_gpu(np.ascontiguousarray(lut))
        if len(_LUT_CACHE) > 64:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


def halfplanes(polys, sites):
    """CSR (nx, ny, d, offsets) of each polygon's outward half-planes about its site.

    The sign of each normal is fixed from the site, so winding does not matter - which it
    otherwise would, because world_to_px flips y and therefore reverses it.
    """
    NX, NY, D = [], [], []
    off = np.zeros(len(polys) + 1, np.int32)
    for k, q in enumerate(polys):
        q = np.asarray(q, np.float64)
        if len(q) < 3:
            off[k + 1] = off[k]
            continue
        a = q
        t = np.roll(q, -1, axis=0) - a
        L = np.hypot(t[:, 0], t[:, 1])
        keep = L > 1e-9
        a, t, L = a[keep], t[keep], L[keep]
        if len(L) < 3:
            off[k + 1] = off[k]
            continue
        n = np.stack([t[:, 1], -t[:, 0]], 1) / L[:, None]
        d = ((a - sites[k]) * n).sum(1)
        n[d < 0] *= -1.0
        NX.append(n[:, 0])
        NY.append(n[:, 1])
        D.append(np.abs(d))
        off[k + 1] = off[k] + len(d)
    if not D:
        z = np.zeros(0, np.float32)
        return z, z, z, off
    return (np.concatenate(NX).astype(np.float32),
            np.concatenate(NY).astype(np.float32),
            np.concatenate(D).astype(np.float32), off)


def draw(target, polys, sites, cols, R, ksoft, fgain, wgain, wwidth, fill,
         prof_name, prof_fn, W, H, pad=2.0):
    """Add every cell film into `target` (H, W, 3), CuPy or numpy.

    All geometry is in RENDER PIXELS. `R`, `ksoft`, `wwidth` are per-cell pixel scalars,
    `fgain`/`wgain` per-cell multipliers, `fill` a flat interior term, `cols` an (N, 3)
    colour array.

    `fill` is the near-black cell interior. It used to be one PIL polygon per cell into
    the opaque base layer; folding it in here costs nothing (the field already knows
    inside from outside) and removed ~530 rasterised polygons a frame.
    """
    n = len(polys)
    if n == 0:
        return
    sites = np.ascontiguousarray(sites, np.float64)
    nx, ny, dd, off = halfplanes(polys, sites)
    bx = np.zeros(n, np.int32)
    by = np.zeros(n, np.int32)
    bw = np.zeros(n, np.int32)
    bh = np.zeros(n, np.int32)
    for k, q in enumerate(polys):
        if off[k + 1] == off[k]:
            continue
        q = np.asarray(q, np.float64)
        x0 = max(0, int(np.floor(q[:, 0].min() - pad)))
        y0 = max(0, int(np.floor(q[:, 1].min() - pad)))
        x1 = min(W, int(np.ceil(q[:, 0].max() + pad)))
        y1 = min(H, int(np.ceil(q[:, 1].max() + pad)))
        if x1 <= x0 or y1 <= y0:
            continue
        bx[k], by[k], bw[k], bh[k] = x0, y0, x1 - x0, y1 - y0

    live = np.nonzero(bw * bh)[0]
    if len(live) == 0:
        return

    if G.GPU and isinstance(target, G.cp.ndarray):
        def f(a, t=np.float32):
            return G.to_gpu(np.ascontiguousarray(a, t))
        warm()((n,), (128,),
               (f(nx), f(ny), f(dd), f(off, np.int32), f(bx, np.int32), f(by, np.int32),
                f(bw, np.int32), f(bh, np.int32), f(sites[:, 0]), f(sites[:, 1]),
                f(R), f(ksoft), f(np.asarray(cols).reshape(-1)), f(fgain), f(wgain),
                f(wwidth), f(fill),
                profile_lut(prof_name, prof_fn, 1024, True), np.int32(1024),
                target, np.int32(W), np.int32(H)))
        return

    lut = profile_lut(prof_name, prof_fn, 1024, False)
    for k in live:
        e0, e1 = int(off[k]), int(off[k + 1])
        x0, y0, w, h = int(bx[k]), int(by[k]), int(bw[k]), int(bh[k])
        qx = (np.arange(x0, x0 + w, dtype=np.float64) + 0.5) - sites[k, 0]
        qy = (np.arange(y0, y0 + h, dtype=np.float64) + 0.5) - sites[k, 1]
        m = (dd[e0:e1][:, None, None]
             - (qx[None, None, :] * nx[e0:e1][:, None, None]
                + qy[None, :, None] * ny[e0:e1][:, None, None]))
        mmin = m.min(axis=0)
        kk = float(ksoft[k])
        ms = mmin - kk * np.log(np.exp(-(m - mmin) / kk).sum(0)) if kk > 1e-4 else mmin
        inside = ms > 0
        if not inside.any():
            continue
        r = np.clip(1.0 - ms / float(R[k]), 0.0, 1.0)
        v = lut[(r * 1023).astype(np.int32)] * float(fgain[k]) + float(fill[k])
        if wgain[k] > 0 and wwidth[k] > 0:
            v = v + float(wgain[k]) * np.exp(-(ms / float(wwidth[k])) ** 2)
        v = np.where(inside, v * np.clip(ms, 0.0, 1.0), 0.0).astype(np.float32)
        target[y0:y0 + h, x0:x0 + w] += v[..., None] * np.asarray(cols[k], np.float32)
