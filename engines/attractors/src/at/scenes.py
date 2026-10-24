# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""scenes.py - five ways to look at a Lorenz attractor.

The brief was "beautiful ways to demonstrate Lorenz attractors - particles attracted to the path,
different colours, get creative". Each scene here answers it by measuring something different, so
the set is five ANSWERS rather than five palettes:

  measure          what you actually see
  ---------------  ------------------------------------------------------------------
  swarm            invariant measure - where the system spends its time
  ribbon           a single orbit as a ribbon, with a halo of captured particles
  divergence       sensitive dependence, colour-keyed to how far a particle has drifted
  capture          particles pulled onto the manifold from a random cloud outside it
  tour             one swarm carried continuously between four different attractors

Colour always encodes a real quantity - speed, drift, capture distance, age - never decoration.
That is the rule that keeps this a visualisation and not a screensaver, and it is also what makes
the pictures legible: the bright regions mean something.
"""
from __future__ import annotations

import numpy as np

from . import core as C
from .draw import PALETTES


class Scene:
    duration = 15.0
    fps = 60
    bg = (0, 0, 0)
    trail = 0.86              # frame-to-frame persistence of the accumulation buffer
    point_gain = 1.0
    persp = 0.55
    exposure = 1.0

    def spin(self, t):
        return (0.4 + 0.13 * t, 0.30 + 0.085 * t, 0.05 * t)

    def spin_fast(self, t):
        return (0.4 + 0.23 * t, 0.30 + 0.145 * t, 0.09 * t)


class Swarm(Scene):
    """LORENZ SWARM - 46k particles, coloured by local speed.

    The plainest and most informative of the five. Nothing is traced; every bright filament is
    just where particles crowd, so the image IS the attractor's invariant measure. Speed colour
    makes the structure read three-dimensionally without any shading: the outer sweeps of each
    lobe are fast and hot, the slow crawl through the middle - where the trajectory decides which
    wing to take next - stays cool and dark.
    """
    name = "lorenz_swarm"
    palette = "ember"

    def __init__(self, flow="lorenz", n=46000, seed=1):
        self.s = C.Swarm(flow, n=n, seed=seed)
        self.recycle = 0.055

    def step(self, dt, t):
        self.s.step(dt)
        self.s.respawn(self.recycle * dt)

    def points(self, t):
        P, z = C.project(self.s.P, self.s.c, self.s.s, self.spin(t), self.persp)
        v = self.s.speed()
        u = np.clip(v / (np.percentile(v, 97) + 1e-9), 0, 1) ** 0.72
        return P, z, u


class Ribbon(Scene):
    """ORBIT + HALO - one true orbit drawn as a bright ribbon, with particles it has captured.

    This is the scene that most directly answers "particles attracted to the path". The ribbon is
    a genuine single trajectory; the halo is a cloud of particles under the SAME flow plus a weak
    spring toward the nearest point of that orbit. They are not stuck to it - the flow keeps
    tearing them off and the spring keeps gathering them back - so the halo breathes around the
    ribbon instead of coating it.
    """
    name = "lorenz_ribbon"
    palette = "ice"
    trail = 0.90

    def __init__(self, flow="lorenz", n=26000, seed=2, pull=2.4):
        self.s = C.Swarm(flow, n=n, seed=seed, scatter=2.6)
        self.pull = pull
        self.head = 0
        self.tail = 2600                       # ribbon length in reference-orbit samples

    def step(self, dt, t):
        # advance the swarm, then pull each particle toward the nearest ribbon sample. The
        # nearest-point search is over a strided subset - full nearest-neighbour on 26k x 2.6k
        # would dominate the frame for a correction this soft.
        self.s.step(dt)
        R = self.s.ref[:: 7]
        d = self.s.P[:, None, :] - R[None, :, :]
        j = np.argmin((d ** 2).sum(2), axis=1)
        self.s.P += self.pull * dt * (R[j] - self.s.P)
        self.head = int((t / self.duration) * (len(self.s.ref) - self.tail - 1))

    def points(self, t):
        P, z = C.project(self.s.P, self.s.c, self.s.s, self.spin(t), self.persp)
        return P, z, np.full(len(P), 0.34)

    def ribbon(self, t):
        seg = self.s.ref[self.head: self.head + self.tail]
        P, z = C.project(seg, self.s.c, self.s.s, self.spin(t), self.persp)
        u = np.linspace(0.15, 1.0, len(P)) ** 1.6      # brightest at the live end
        return P, z, u


class Divergence(Scene):
    """SENSITIVE DEPENDENCE - one cloud, coloured by how far it has drifted from its twin.

    Two swarms are launched from IDENTICAL positions and integrated with the same solver; the
    only difference is a 1e-7 offset. Colour is the distance between each particle and its twin,
    so the clip starts uniformly cool - the pairs are together - and then ignites in patches as
    chaos separates them. The pattern of ignition is itself meaningful: it lights up first along
    the stretching directions, which is where the Lyapunov exponent lives.
    """
    name = "lorenz_divergence"
    palette = "flare"
    trail = 0.83

    def __init__(self, flow="lorenz", n=34000, seed=3, eps=1e-7):
        self.a = C.Swarm(flow, n=n, seed=seed)
        self.b = C.Swarm(flow, n=n, seed=seed)
        self.b.P = self.a.P + eps
        self.scale = 0.0

    def step(self, dt, t):
        self.a.step(dt)
        self.b.step(dt)

    def points(self, t):
        P, z = C.project(self.a.P, self.a.c, self.a.s, self.spin(t), self.persp)
        d = np.linalg.norm(self.a.P - self.b.P, axis=1)
        # log distance: separation covers seven orders of magnitude in 15 s, and on a linear
        # scale the whole clip is black and then white in one frame
        u = np.clip((np.log10(d + 1e-12) + 7.0) / 8.0, 0, 1)
        return P, z, u


class Capture(Scene):
    """CAPTURE - a uniform cloud of particles falling onto the manifold.

    Every other scene starts on the attractor. This one deliberately does not: 40k particles are
    seeded in a big uniform box, and the flow does the rest. What you watch is an ATTRACTOR
    attracting - a structureless cloud collapsing onto a two-dimensional sheet in about four
    seconds, because the flow contracts volume everywhere.

    Colour is time-since-capture, so the sheet forms in cool colours while the stragglers still
    falling in stay hot, and the moment of collapse is visible rather than inferred.
    """
    name = "lorenz_capture"
    palette = "aurora"
    trail = 0.80
    # The sheet this scene forms is the densest thing in the whole set - 40k particles collapse
    # onto a 2-D surface, so the additive buffer saturates there and the core blows to flat
    # white, losing exactly the structure the collapse was meant to show. Lower exposure plus a
    # shorter memory keeps the roll-off working where it matters.
    exposure = 0.46
    point_gain = 0.72

    def __init__(self, flow="lorenz", n=40000, seed=5, box=2.1):
        self.s = C.Swarm(flow, n=n, seed=seed)
        rng = np.random.default_rng(seed)
        span = (self.s.ref.max(0) - self.s.ref.min(0)) * box
        self.s.P = self.s.c + (rng.random((n, 3)) - 0.5) * span
        self.d0 = None

    def step(self, dt, t):
        self.s.step(dt)

    def points(self, t):
        P, z = C.project(self.s.P, self.s.c, self.s.s, self.spin(t), self.persp)
        R = self.s.ref[:: 11]
        d = np.sqrt(((self.s.P[:, None, :] - R[None, :, :]) ** 2).sum(2)).min(1)
        u = np.clip(d / (0.30 / self.s.s), 0, 1) ** 0.6      # 0 = on the sheet
        return P, z, u


class Tour(Scene):
    """FOUR ATTRACTORS, ONE SWARM - continuous morph through the flow itself.

    The reference clip showed four attractors as four separate panels. This shows the same four
    as one object by interpolating the DERIVATIVE, not the picture:

        dP/dt = (1-u) * f_A(P) + u * f_B(P)

    Because the blend is of the vector fields, every intermediate is a real dynamical system with
    its own trajectories - the swarm is never teleported and never cross-faded. Watching a Lorenz
    butterfly stretch into a Halvorsen knot through a sequence of attractors nobody has named is
    the whole reason to do it this way.
    """
    name = "attractor_tour"
    palette = "spectrum"
    trail = 0.87

    ORDER = ["lorenz", "chen", "halvorsen", "thomas", "lorenz"]
    # Deliberately sped up: two things move: the swarm is integrated ~1.8x faster so the
    # filaments themselves flow, and the view spins harder - a morph that is slower than the
    # spin reads as a still object being turned, not as one system becoming another.

    def __init__(self, n=38000, seed=7):
        self.s = C.Swarm("lorenz", n=n, seed=seed)
        self.flows = [C.FLOWS[k] for k in self.ORDER]
        self.u = 0.0

    def _blend(self, k, u):
        fa, dta, _, _ = self.flows[k]
        fb, dtb, _, _ = self.flows[k + 1]
        # normalise each field to a common speed so the blend does not stall or explode
        def f(P):
            A, B = fa(P), fb(P)
            na = np.linalg.norm(A, axis=1, keepdims=True).mean() + 1e-9
            nb = np.linalg.norm(B, axis=1, keepdims=True).mean() + 1e-9
            return (1 - u) * A / na + u * B / nb
        return f, 0.5 * (dta / max(dta, 1e-9))

    def step(self, dt, t):
        segs = len(self.ORDER) - 1
        g = np.clip(t / self.duration, 0, 0.999) * segs
        k, u = int(g), g - int(g)
        u = u * u * (3 - 2 * u)
        f, _ = self._blend(k, u)
        self.s.f = f
        self.s.step(dt * 9.4)
        self.s.respawn(0.10 * dt)
        self.u = g

    def points(self, t):
        # the tour leaves the seed attractor's bounding box, so renormalise on the LIVE swarm
        lo, hi = np.percentile(self.s.P, [1.0, 99.0], axis=0)
        c = 0.5 * (lo + hi)
        s = 1.0 / (0.5 * float(np.max(hi - lo)) * 1.15)
        P, z = C.project(self.s.P, c, s, self.spin_fast(t), self.persp)
        v = self.s.speed()
        u = np.clip(v / (np.percentile(v, 96) + 1e-9), 0, 1) ** 0.7
        return P, z, u


SCENES = {s.name: s for s in (Swarm, Ribbon, Divergence, Capture, Tour)}
