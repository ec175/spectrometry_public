"""core.py - vectorised strange-attractor flows and a particle swarm that lives on them.

The difference from `..\\Oscilloscope\\osc\\attractors.py` is the whole point of this project.
There, ONE trajectory is traced by a beam and the picture is built by phosphor persistence. Here,
tens of thousands of INDEPENDENT particles are integrated through the same vector field at once,
and the picture is built by where they crowd.

That is not a restyling, it is a different measurement. A single orbit shows you the *shape* of an
attractor; a swarm shows you its **invariant measure** - which parts of the shape the system
actually spends its time in. The bright regions in these clips are not artistic emphasis, they are
where the dynamics dwell.

Everything is a whole-swarm numpy op. RK4 on an (N,3) array costs the same four function calls as
RK4 on one point, so 40 000 particles are essentially free next to the rasteriser - the same
lockstep argument as Field_Lines' field-line integrator.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# flows: f(P) -> dP/dt, all vectorised over an (N,3) array
# --------------------------------------------------------------------------------------
def lorenz(P, s=10.0, r=28.0, b=8.0 / 3.0):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([s * (y - x), x * (r - z) - y, x * y - b * z], axis=1)


def rossler(P, a=0.2, b=0.2, c=5.7):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=1)


def aizawa(P, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([(z - b) * x - d * y,
                     d * x + (z - b) * y,
                     c + a * z - z ** 3 / 3.0 - (x * x + y * y) * (1 + e * z)
                     + f * z * x ** 3], axis=1)


def halvorsen(P, a=1.4):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([-a * x - 4 * y - 4 * z - y * y,
                     -a * y - 4 * z - 4 * x - z * z,
                     -a * z - 4 * x - 4 * y - x * x], axis=1)


def thomas(P, b=0.19):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], axis=1)


def chen(P, a=5.0, b=-10.0, d=-0.38):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.stack([a * x - y * z, b * y + x * z, d * z + x * y / 3.0], axis=1)


FLOWS = {"lorenz": (lorenz, 0.0045, np.array([0.9, 0.0, 20.0]), 26.0),
         "rossler": (rossler, 0.0180, np.array([1.0, 1.0, 1.0]), 20.0),
         "aizawa": (aizawa, 0.0095, np.array([0.1, 0.0, 0.0]), 1.6),
         "halvorsen": (halvorsen, 0.0058, np.array([-1.48, -1.51, 2.04]), 9.0),
         "thomas": (thomas, 0.0420, np.array([1.1, 1.1, -0.01]), 4.5),
         "chen": (chen, 0.0040, np.array([5.0, 10.0, 10.0]), 32.0)}


def rk4(f, P, dt, **kw):
    k1 = f(P, **kw)
    k2 = f(P + 0.5 * dt * k1, **kw)
    k3 = f(P + 0.5 * dt * k2, **kw)
    k4 = f(P + dt * k3, **kw)
    return P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class Swarm:
    """N particles on one attractor.

    Seeding is the one subtle part. Particles started in a random ball are all OFF the attractor,
    and their approach transients are long, bright and identical-looking - the first second of the
    clip becomes a collapsing shell that has nothing to do with the dynamics. So the swarm is
    seeded ALONG a reference orbit instead (with a little scatter), which puts every particle on
    the attractor from frame zero. `respawn` uses the same trick.
    """

    def __init__(self, flow="lorenz", n=42000, seed=0, scatter=0.55, spin_up=4000):
        f, dt, x0, span = FLOWS[flow]
        self.f, self.dt, self.span = f, dt, span
        self.name = flow
        rng = np.random.default_rng(seed)
        self.rng = rng
        ref = np.empty((spin_up, 3))
        p = x0.reshape(1, 3).copy()
        for i in range(spin_up):
            p = rk4(f, p, dt)
            ref[i] = p[0]
        self.ref = ref[spin_up // 4:]                 # drop the approach transient
        idx = rng.integers(0, len(self.ref), n)
        self.P = self.ref[idx] + rng.normal(0.0, scatter, (n, 3))
        self.age = rng.random(n) * 6.0
        self.c = 0.5 * (self.ref.min(0) + self.ref.max(0))
        self.s = 1.0 / (0.5 * float(np.max(self.ref.max(0) - self.ref.min(0))) * 1.05)

    def step(self, dt, sub=2):
        h = dt / sub
        for _ in range(sub):
            prev = self.P
            self.P = rk4(self.f, self.P, h)
            self.V = (self.P - prev) / h
        self.age += dt
        return self

    def respawn(self, frac, max_age=None):
        """Recycle a fraction of the swarm back onto the reference orbit each second.

        Chaotic flows are volume-contracting, so an un-recycled swarm collapses onto a thin
        filament within a few seconds and the picture loses every part of the attractor the
        trajectory is not on RIGHT NOW. Recycling keeps the whole invariant measure lit.
        """
        n = len(self.P)
        k = int(n * frac)
        if k <= 0:
            return
        idx = self.rng.integers(0, n, k)
        src = self.rng.integers(0, len(self.ref), k)
        self.P[idx] = self.ref[src] + self.rng.normal(0.0, 0.35, (k, 3))
        self.age[idx] = 0.0

    def speed(self):
        return np.hypot(np.hypot(self.V[:, 0], self.V[:, 1]), self.V[:, 2])


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = (np.cos(a), np.sin(a), np.cos(b),
                              np.sin(b), np.cos(c), np.sin(c))
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def project(Q, c, s, ang, persp=0.55):
    """3-D -> screen, with a mild perspective divide and a returned DEPTH.

    Depth matters here in a way it does not on the scope: with 40 000 additive points, a purely
    orthographic view turns a 3-D object into a flat smear. A weak perspective plus depth-keyed
    size and brightness is enough to make the front and back lobes separate.
    """
    V = (Q - c) * s @ rot(*ang).T
    z = V[:, 2]
    w = 1.0 / (1.0 + persp * z)
    return np.stack([V[:, 0] * w, V[:, 1] * w], axis=1), z
