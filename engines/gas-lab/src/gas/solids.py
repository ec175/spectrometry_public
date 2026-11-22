# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""solids.py - rigid shaped parts the gas bounces off, and circular port reservoirs.

A BODY is a sdf.Solid plus a pose: fn(t) -> (x, y, angle, vx, vy, omega) in its group's local
frame (the angle about the body origin, clockwise on screen). Static housings have a constant
pose. Every physics body is handled by ONE fused CUDA kernel per substep: each particle loops
over the bodies of its group, skips any whose bounding circle it is outside, samples d
bilinearly, and if it is inside the metal it is MIRRORED back out along the normal and its
velocity reflected in the frame of the surface under it:

    u   = V + omega x r                    (the surface velocity at the particle)
    v' -= 2 ((v - u) . n) n   if (v - u) . n < 0

That is world.py's piston rule (v' = -v + 2u) for a surface of any shape moving any way. The
kernel runs the body loop twice, because a particle mirrored out of one rotor can land in its
mate across a mesh gap; the second pass puts it right.

Two tallies come back per body: the impulse the gas delivered (a pressure read on the metal),
and the force vector (for a FREE body - the shock's floating piston - that the physics moves
rather than a schedule). A mirror deeper than `deep` px counts as a RESCUE: a particle that got
further into metal than one substep can explain. Too many says the part is too thin for the
substep, exactly like the box engine's clamp count.

A HOLE is a circular reservoir inside a box: the eye of a turbocharger wheel, where gas arrives
or leaves ALONG the axis (out of the plane), or a scroll's discharge port. Particles that enter
it are removed; new ones are injected from its rim outward at the one-sided equilibrium flux of
its own density and temperature - the open-edge reservoir of world.py bent into a circle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .gpu import GPU, xp

_COLLIDE_SRC = r"""
__device__ __forceinline__ float urand(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int h = a * 747796405u + b * 2891336453u + c * 277803737u;
    h ^= h >> 17; h *= 0xed5ad4bbu; h ^= h >> 11; h *= 0xac4c1b51u; h ^= h >> 15;
    h *= 0x31848babu; h ^= h >> 14;
    return ((h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float bil(const float* a, int o, int gw, float ux, float uy) {
    float a00 = a[o], a10 = a[o + 1], a01 = a[o + gw], a11 = a[o + gw + 1];
    return (a00 * (1.f - ux) + a10 * ux) * (1.f - uy) + (a01 * (1.f - ux) + a11 * ux) * uy;
}

extern "C" __global__ void collide(float* x, float* y, float* vx, float* vy,
                                   const int* grp, const int n,
                                   const float* sdf, const float* gxs, const float* gys,
                                   const float* bp, const int* bi, const int nb,
                                   float* tally, int* rescue, const float deep,
                                   const int query, float* dmin,
                                   const float alpha, const unsigned int seed) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float X = x[i], Y = y[i], VX = vx[i], VY = vy[i];
    int g = grp[i];
    float dm = 1e30f;
    for (int it = 0; it < 2; it++) {
        int hitany = 0;
        for (int k = 0; k < nb; k++) {
            if (bi[5*k + 3] != g) continue;
            const float* p = bp + 12*k;   // cx cy cos sin vxb vyb om gx0 gy0 invh rad2 -
            float dx = X - p[0], dy = Y - p[1];
            if (dx*dx + dy*dy > p[10]) continue;
            float lx =  p[2]*dx + p[3]*dy;
            float ly = -p[3]*dx + p[2]*dy;
            float fx = (lx - p[7]) * p[9], fy = (ly - p[8]) * p[9];
            int gw = bi[5*k + 1], gh = bi[5*k + 2];
            if (fx < 0.f || fy < 0.f || fx >= (float)(gw - 1) || fy >= (float)(gh - 1)) continue;
            int ix = (int)fx, iy = (int)fy;
            float ux = fx - ix, uy = fy - iy;
            int o = bi[5*k] + iy*gw + ix;
            float d = bil(sdf, o, gw, ux, uy);
            if (d < dm) dm = d;
            if (query || d >= 0.f) continue;
            float nlx = bil(gxs, o, gw, ux, uy), nly = bil(gys, o, gw, ux, uy);
            float m = rsqrtf(nlx*nlx + nly*nly + 1e-12f);
            nlx *= m; nly *= m;
            float nx = p[2]*nlx - p[3]*nly, ny = p[3]*nlx + p[2]*nly;
            float pen = -d;
            X += 2.f * pen * nx;  Y += 2.f * pen * ny;
            float rx = X - p[0], ry = Y - p[1];
            float sux = p[4] - p[6]*ry, suy = p[5] + p[6]*rx;
            float vn = (VX - sux)*nx + (VY - suy)*ny;
            if (vn < 0.f) {
                float px0 = VX, py0 = VY;
                if (p[11] > 0.f && urand(i, seed, 2*k + it) < alpha) {
                    // DIFFUSE: re-emitted at the wall's temperature in the wall's frame
                    float u1 = urand(i, seed, 101 + 2*k + it), u2 = urand(i, seed, 211 + 2*k + it);
                    float u3 = urand(i, seed, 307 + 2*k + it);
                    float wn = p[11] * sqrtf(-2.f * logf(u1));
                    float wt = p[11] * sqrtf(-2.f * logf(u2)) * cosf(6.2831853f * u3);
                    VX = sux + wn * nx - wt * ny;
                    VY = suy + wn * ny + wt * nx;
                } else {
                    VX -= 2.f*vn*nx;  VY -= 2.f*vn*ny;
                }
                vn = 0.5f * ((px0 - VX) * nx + (py0 - VY) * ny);
                atomicAdd(&tally[4*k],     -2.f*vn);        // (vn recomputed so a diffuse
                                                                 //  bounce books its true impulse)
                atomicAdd(&tally[4*k + 1], 2.f*vn*nx);     // force ON THE BODY
                atomicAdd(&tally[4*k + 2], 2.f*vn*ny);
                atomicAdd(&tally[4*k + 3], 2.f*vn*(rx*ny - ry*nx));   // torque on the body
            }
            if (pen > deep) atomicAdd(rescue, 1);
            hitany = 1;
        }
        if (query || !hitany) break;
    }
    if (query) { dmin[i] = dm; return; }
    x[i] = X; y[i] = Y; vx[i] = VX; vy[i] = VY;
}
"""
_collide_kernel = None


def _kernel():
    global _collide_kernel
    if _collide_kernel is None:
        _collide_kernel = xp.RawKernel(_COLLIDE_SRC, "collide")
    return _collide_kernel


def const_pose(x=0.0, y=0.0, ang=0.0):
    p = (float(x), float(y), float(ang), 0.0, 0.0, 0.0)
    return lambda t: p


def spin_pose(x, y, drive, ratio=1.0, phase=0.0):
    """A body turning about a fixed centre at `ratio` x the drive's angle (negative = the
    other way). Angle and rate both come from the Drive, so the ramp is exact."""
    def f(t):
        return (x, y, ratio * drive.theta(t) + phase, 0.0, 0.0, ratio * drive.omega(t))
    return f


@dataclass
class Body:
    solid: object                       # sdf.Solid, done()
    pose: object                        # fn(t) -> (x, y, ang, vx, vy, omega), group-local
    group: int = 0
    physics: bool = True                # the gas meets it
    draw: bool = True                   # the picture shows it
    z: int = 0                          # paint order among bodies; static bodies first
    static: bool = False                # drawn once into the static layer
    name: str = ""
    free: object = None                 # optional callable(world, tally_row, dt) for FREE bodies
    wall_T: float = 0.0                 # > 0: a fraction ACCOM of bounces re-emit diffusely at
                                        # this temperature (the metal takes heat); 0 = specular


@dataclass
class Hole:
    """A circular reservoir inside a box (an axial port seen end-on)."""
    box: int
    cx: float
    cy: float
    r: float
    boost: float = 1.0                  # density, in ambients
    T: float = C.T0                     # reservoir temperature
    mix: dict = field(default_factory=lambda: {C.AIR: 1.0})
    role: str = "intake"                # 'intake' | 'exhaust' - which tally it books into
    inject: bool = True                 # False = a pure sink (the far side is at vacuum)
    label: str = ""


class SolidSet:
    """The physics bodies of one engine packed into one atlas for the kernel."""

    def __init__(self, bodies, group_of_box):
        self.bodies = [b for b in bodies if b.physics]
        self.all = bodies
        self.group_of_box = group_of_box
        nb = len(self.bodies)
        self.nb = nb
        if nb == 0:
            return
        offs, sd, gx, gy = [], [], [], []
        o = 0
        bi = np.zeros((nb, 5), dtype=np.int32)
        for k, b in enumerate(self.bodies):
            s = b.solid
            bi[k] = (o, s.gw, s.gh, b.group, 0)
            sd.append(s.d)
            gx.append(s.nx_p)
            gy.append(s.ny_p)
            o += s.gw * s.gh
        self.sdf = xp.concatenate(sd).astype(xp.float32)
        self.gxs = xp.concatenate(gx).astype(xp.float32)
        self.gys = xp.concatenate(gy).astype(xp.float32)
        self.bi = xp.asarray(bi.ravel())
        self.bp_np = np.zeros((nb, 12), dtype=np.float32)
        self.tally = xp.zeros(nb * 4, dtype=xp.float32)
        self.rescue = xp.zeros(1, dtype=xp.int32)
        self.dummy = xp.zeros(1, dtype=xp.float32)
        self.poses = [None] * nb
        self.accom = 0.2                  # accommodation fraction for bodies with wall_T > 0

    def set_poses(self, t):
        for k, b in enumerate(self.bodies):
            s = b.solid
            x, y, a, vx, vy, om = b.pose(t)
            self.poses[k] = (x, y, a, vx, vy, om)
            r = s.rad + 4.0
            sw = (C.V_TH / math.sqrt(2.0)) * math.sqrt(b.wall_T / C.T0) if b.wall_T > 0 else 0.0
            self.bp_np[k] = (x, y, math.cos(a), math.sin(a), vx, vy, om, s.x0, s.y0, 1.0 / s.h,
                             r * r, sw)
        self.bp = xp.asarray(self.bp_np.ravel())

    def collide(self, world, deep=10.0):
        if self.nb == 0:
            return
        n = world.x.size
        if n == 0:
            return
        grp = self.group_of_box[world.box]
        blk = 256
        _kernel()(((n + blk - 1) // blk,), (blk,),
                  (world.x, world.y, world.vx, world.vy, grp, np.int32(n),
                   self.sdf, self.gxs, self.gys, self.bp, self.bi, np.int32(self.nb),
                   self.tally, self.rescue, np.float32(deep), np.int32(0), self.dummy,
                   np.float32(self.accom), np.uint32(world.substep & 0x7FFFFFFF)))

    def dmin(self, x, y, box):
        """Signed distance to the nearest physics body, per particle (a query, nothing moves)."""
        n = x.size
        out = xp.full(n, 1e30, dtype=xp.float32)
        if self.nb == 0 or n == 0:
            return out
        grp = self.group_of_box[box]
        blk = 256
        xx, yy = x.astype(xp.float32).copy(), y.astype(xp.float32).copy()
        vx = xp.zeros(n, dtype=xp.float32)
        _kernel()(((n + blk - 1) // blk,), (blk,),
                  (xx, yy, vx, vx.copy(), grp, np.int32(n),
                   self.sdf, self.gxs, self.gys, self.bp, self.bi, np.int32(self.nb),
                   self.tally, self.rescue, np.float32(1e9), np.int32(1), out,
                   np.float32(0.0), np.uint32(0)))
        return out

    def take_tally(self):
        """(impulse, fx, fy, torque) per body since the last call, CPU."""
        if self.nb == 0:
            return np.zeros((0, 4))
        t = (self.tally.get() if GPU else self.tally).reshape(-1, 4).copy()
        self.tally[:] = 0.0
        return t

    def rescues(self):
        return int(self.rescue[0])
