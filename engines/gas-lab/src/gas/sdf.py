# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""sdf.py - a SOLID is a signed-distance grid. The gas bounces off it; the picture is shaded from it.

The 2026-09-20 engine could only express axis-aligned boxes and pistons that are box floors, so
anything that turns - a rotor, a vane, a turbine wheel - was out of reach. The 2026-09-25 parts
set needs exactly those. A Solid is built by CSG from polygon rings (shapes.py) onto a grid in
the part's own frame, and that ONE grid is used twice:

  PHYSICS   solids.py samples the signed distance d and its gradient per particle per substep.
            d < 0 is metal: the particle is mirrored back out along the normal and its velocity
            reflected in the frame of the moving surface (v' = v - 2((v - u).n) n, with u the
            surface velocity V + omega x r). That is the piston rule generalised: compression
            heating off a rotor lobe is the same reflection off a wall that is moving.
  PICTURE   metal.py samples the same d for the edge (anti-aliased alpha), and a HEIGHT field
            built on top of it - a rounded bevel along every edge, recessed pockets, raised
            bosses - for the normals it lights with a fixed lamp. So what you see is the
            outline the gas meets, to the grid resolution, and the detail is RELIEF on the part
            rather than more parts in the frame (Clancy, 2026-09-25: "models with better detail,
            but lets avoid just adding more parts into each frame").

The distance is EXACT to the polygon (a CUDA kernel over every segment), not an Euclidean
distance transform of a raster: a raster's half-pixel steps would show as jaggies on the metal
edge and as a rough wall to the gas.
"""
from __future__ import annotations

import math

import numpy as np

from .gpu import GPU, xp

_POLY_SRC = r"""
extern "C" __global__ void polysdf(const float* px, const float* py, const int n,
                                   const float* seg, const int ns, float* dout) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float x = px[i], y = py[i];
    float best = 1e30f;
    int inside = 0;
    for (int k = 0; k < ns; k++) {
        float ax = seg[4*k], ay = seg[4*k+1], bx = seg[4*k+2], by = seg[4*k+3];
        float ex = bx - ax, ey = by - ay, wx = x - ax, wy = y - ay;
        float L = ex*ex + ey*ey;
        float t = L > 0.f ? (wx*ex + wy*ey) / L : 0.f;
        t = fminf(fmaxf(t, 0.f), 1.f);
        float dx = wx - ex*t, dy = wy - ey*t;
        float d2 = dx*dx + dy*dy;
        if (d2 < best) best = d2;
        if ((ay > y) != (by > y)) {
            float xi = ax + (y - ay) * ex / ey;
            if (x < xi) inside ^= 1;
        }
    }
    float d = sqrtf(best);
    dout[i] = inside ? -d : d;
}
"""
_poly_kernel = None


def _kernel():
    global _poly_kernel
    if _poly_kernel is None:
        _poly_kernel = xp.RawKernel(_POLY_SRC, "polysdf")
    return _poly_kernel


def ring_sdf(X, Y, ring):
    """Exact signed distance from grid points (flat float32 arrays) to one closed ring.
    Negative inside."""
    P = np.asarray(ring, dtype=np.float32)
    Q = np.roll(P, -1, axis=0)
    seg = np.concatenate([P, Q], 1).astype(np.float32).ravel()
    n = X.size
    out = xp.empty(n, dtype=xp.float32)
    if GPU:
        k = _kernel()
        segd = xp.asarray(seg)
        blk = 256
        k(((n + blk - 1) // blk,), (blk,), (X, Y, np.int32(n), segd, np.int32(len(P)), out))
        return out
    # CPU fallback (slow, only for smoke tests without CUDA)
    best = np.full(n, 1e30, dtype=np.float32)
    inside = np.zeros(n, dtype=bool)
    for (ax, ay), (bx, by) in zip(P, Q):
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        L = ex * ex + ey * ey
        t = np.clip((wx * ex + wy * ey) / L if L > 0 else 0.0, 0.0, 1.0)
        dx, dy = wx - ex * t, wy - ey * t
        best = np.minimum(best, dx * dx + dy * dy)
        cross = (ay > Y) != (by > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = ax + (Y - ay) * ex / ey
        inside ^= cross & (X < xi)
    return np.where(inside, -np.sqrt(best), np.sqrt(best)).astype(np.float32)


def _smooth01(u):
    u = xp.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Solid:
    """A part in its own frame, on a grid of spacing h over [x0, x1] x [y0, y1].

        s = Solid((-300, -300, 300, 300), h=0.6)
        s.add(shapes.roots_rotor(180)).sub(shapes.circle(0, 0, 40))
        s.recess(shapes.circle(0, 0, 90), depth=6)       # a machined face, relief only
        s.paint(shapes.circle(0, 0, 40), (60, 60, 64))   # a colour zone, relief only
        s.done(material)

    `add`/`sub` change the OUTLINE (what the gas meets). Everything after is relief and colour
    on top of it and never touches the physics."""

    def __init__(self, bbox, h=0.75, bevel=7.0, name=""):
        x0, y0, x1, y1 = bbox
        self.h = float(h)
        self.gw = int(math.ceil((x1 - x0) / h)) + 2
        self.gh = int(math.ceil((y1 - y0) / h)) + 2
        self.x0, self.y0 = float(x0) - h, float(y0) - h
        self.bevel = float(bevel)
        self.name = name
        gx = self.x0 + self.h * xp.arange(self.gw, dtype=xp.float32)
        gy = self.y0 + self.h * xp.arange(self.gh, dtype=xp.float32)
        self.X = xp.broadcast_to(gx[None, :], (self.gh, self.gw)).ravel().copy()
        self.Y = xp.broadcast_to(gy[:, None], (self.gh, self.gw)).ravel().copy()
        self.d = xp.full(self.gw * self.gh, 1e6, dtype=xp.float32)
        self.relief = []           # (kind, sdf, amount, bevel)
        self.paints = []           # (sdf, rgb)
        self.lines = []            # (sdf of a thin band, rgb): scribed detail
        self.done_ = False

    # ---- the outline ----
    def add(self, ring):
        self.d = xp.minimum(self.d, ring_sdf(self.X, self.Y, ring))
        return self

    def sub(self, ring):
        self.d = xp.maximum(self.d, -ring_sdf(self.X, self.Y, ring))
        return self

    def intersect(self, ring):
        self.d = xp.maximum(self.d, ring_sdf(self.X, self.Y, ring))
        return self

    def grow(self, delta):
        """Offset the whole outline outward by delta px (negative shrinks) - a clearance."""
        self.d = self.d - float(delta)
        return self

    # ---- relief and colour (the picture only) ----
    def recess(self, ring, depth=6.0, bevel=4.0):
        self.relief.append(("recess", ring_sdf(self.X, self.Y, ring), float(depth), float(bevel)))
        return self

    def boss(self, ring, height=6.0, bevel=4.0):
        self.relief.append(("boss", ring_sdf(self.X, self.Y, ring), float(height), float(bevel)))
        return self

    def paint(self, ring, rgb, feather=1.2):
        self.paints.append((ring_sdf(self.X, self.Y, ring), rgb, float(feather)))
        return self

    def done(self, base_rgb, rough=0.35, brush="radial", brush_c=(0.0, 0.0), sectioned=False,
             spec=0.55, seed=0):
        """Freeze: the physics grids (d, unit gradient) and the picture grids (albedo, normal,
        a gloss scalar). `brush` is the machining texture: 'radial' = turned (circumferential
        lines about brush_c), 'linear' = ground along x, 'cast' = sand-cast mottle."""
        gh, gw, h = self.gh, self.gw, self.h
        D = self.d.reshape(gh, gw)
        # ---- physics: unit gradient of d (points OUT of the metal) ----
        gy, gx = xp.gradient(D, h)
        m = xp.sqrt(gx * gx + gy * gy) + 1e-6
        self.nx_p = (gx / m).astype(xp.float32).ravel()
        self.ny_p = (gy / m).astype(xp.float32).ravel()
        self.d = self.d.astype(xp.float32)
        # ---- picture: height = a rounded bevel on the outline, then relief ----
        u = xp.clip(-D / self.bevel, 0.0, 1.0)
        H = self.bevel * 0.8 * (1.0 - (1.0 - u) ** 2)        # quarter-round edge
        for kind, sd, amt, bv in self.relief:
            s = _smooth01(-sd.reshape(gh, gw) / bv)
            H = H - amt * s if kind == "recess" else H + amt * s
        hy, hx = xp.gradient(H, h)
        nz = xp.ones_like(hx)
        nn = xp.sqrt(hx * hx + hy * hy + nz * nz)
        self.nx_l = (-hx / nn).astype(xp.float32)
        self.ny_l = (-hy / nn).astype(xp.float32)
        # ---- albedo ----
        A = xp.empty((3, gh, gw), dtype=xp.float32)
        for c in range(3):
            A[c] = base_rgb[c] / 255.0
        for sd, rgb, fe in self.paints:
            w = _smooth01(0.5 - sd.reshape(gh, gw) / fe)
            for c in range(3):
                A[c] = A[c] * (1 - w) + (rgb[c] / 255.0) * w
        # machining texture: a few percent of multiplicative variation, never a pattern that
        # reads at viewing distance
        rng = np.random.default_rng(seed + 11)
        X = self.X.reshape(gh, gw)
        Y = self.Y.reshape(gh, gw)
        if brush == "radial":
            r = xp.sqrt((X - brush_c[0]) ** 2 + (Y - brush_c[1]) ** 2)
            ph = float(rng.uniform(0, 6.28))
            tex = (0.55 * xp.sin(r * 1.9 + ph) + 0.30 * xp.sin(r * 4.7 + 2 * ph)
                   + 0.15 * xp.sin(r * 0.43 + 3 * ph))
        elif brush == "linear":
            ph = float(rng.uniform(0, 6.28))
            tex = 0.6 * xp.sin(Y * 2.3 + ph) + 0.4 * xp.sin(Y * 5.1 + X * 0.07 + 2 * ph)
        else:
            tex = (xp.sin(X * 0.21 + Y * 0.13) * xp.sin(X * 0.07 - Y * 0.19)
                   + 0.5 * xp.sin(X * 0.55 + 1.3) * xp.sin(Y * 0.61 + 0.7))
        A = A * (1.0 + rough * 0.06 * tex)[None]
        self.albedo = xp.clip(A, 0.0, 1.0).astype(xp.float32)
        self.gloss = float(spec)
        self.sectioned = bool(sectioned)
        self.rad = float(xp.sqrt(xp.max(xp.where(self.d < 2.0, self.X * self.X + self.Y * self.Y, 0.0))))
        self.relief = self.paints = None          # free the per-feature grids
        self.done_ = True
        return self

    # ---- queries ----
    def area(self):
        return float((self.d < 0).sum()) * self.h * self.h

    def sample(self, lx, ly):
        """d at local points (numpy), bilinear - for the gate and for placing things."""
        D = self.d.reshape(self.gh, self.gw)
        fx = (np.asarray(lx) - self.x0) / self.h
        fy = (np.asarray(ly) - self.y0) / self.h
        ix = np.clip(np.floor(fx).astype(int), 0, self.gw - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, self.gh - 2)
        ux, uy = fx - ix, fy - iy
        Dn = D.get() if GPU else D
        d00, d10 = Dn[iy, ix], Dn[iy, ix + 1]
        d01, d11 = Dn[iy + 1, ix], Dn[iy + 1, ix + 1]
        v = (d00 * (1 - ux) + d10 * ux) * (1 - uy) + (d01 * (1 - ux) + d11 * ux) * uy
        out = (fx < 0) | (fy < 0) | (fx > self.gw - 1) | (fy > self.gh - 1)
        return np.where(out, 1e6, v)
