"""motion.py - how sources move, and the small envelope helpers every scene needs.

Two motion backends, and the distinction between them is deliberate:

* `WanderPath` is a DECLARED path. A scene using it is saying "this source is on a rig" - the
  same class of input as Wind_Tunnel's "a body's POSITION at time t" or Shape_Physics' spin
  rate. It is honest because it is declared, not because it is physical.
* `NBody` is SOLVED. Nothing about where those charges go is authored; only their masses,
  charges, the well they sit in, and how they were launched.

Both are legitimate; what is not legitimate is dressing one up as the other.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------------------
def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def keyframes(t, ts, vs):
    """Smoothstep-interpolated piecewise curve through (ts, vs). Used for every animated
    scalar, so an envelope is written as the shape it should have rather than as a formula."""
    ts, vs = np.asarray(ts, float), np.asarray(vs, float)
    if t <= ts[0]:
        return float(vs[0])
    if t >= ts[-1]:
        return float(vs[-1])
    i = int(np.searchsorted(ts, t) - 1)
    u = (t - ts[i]) / (ts[i + 1] - ts[i])
    return float(vs[i] + (vs[i + 1] - vs[i]) * smoothstep(u))


def pulse_gate(t, windows, ramp=1.0, lo=1.0, hi=5.0):
    """A two-state square wave with smoothstep ramps - the reference clip's mobile-charge
    magnitude. `windows` is a list of (t_start, t_end) during which the value is `hi`."""
    v = lo
    for a, b in windows:
        up = smoothstep((t - a) / ramp)
        dn = 1.0 - smoothstep((t - b) / ramp)
        v = max(v, lo + (hi - lo) * min(up, dn))
    return v


# --------------------------------------------------------------------------------------
# a declared wandering path
# --------------------------------------------------------------------------------------
class WanderPath:
    """Sum of three incommensurate sinusoids per axis, seeded, then TIME-RESCALED so its
    measured mean speed matches `speed`.

    Measuring and rescaling rather than solving for amplitudes is the cheap honest move: the
    speed of a sinusoid sum has no tidy closed form once the ratios are irrational, and the
    reference clip gives a speed (median 170 px/s at 608 px wide == ~302 px/s in the 1080-wide
    reference frame) that is worth hitting exactly.

    Amplitudes deliberately exceed the half-frame so the source LEAVES and re-enters, which the
    reference does at least three times in 90 s.
    """
    RATIOS = (1.0, 1.7320508, 2.6180340)      # sqrt(3), golden-ish: never re-align

    def __init__(self, center, span, speed=302.0, over=1.18, seed=0, f0=0.10):
        rng = np.random.default_rng(seed)
        self.c = np.asarray(center, float)
        amp = np.array([0.56, 0.29, 0.17]) * (0.5 * np.asarray(span, float) * over)[:, None]
        self.A = amp                                   # (2,3)
        self.f = np.array(self.RATIOS) * f0
        self.p = rng.uniform(0, 2 * np.pi, (2, 3))
        self.k = 1.0
        self.k = speed / max(1e-6, self._mean_speed())

    def _raw(self, t):
        ph = 2.0 * np.pi * self.f[None, :] * np.atleast_1d(t)[:, None, None] + self.p[None]
        return self.c[None, :] + (self.A[None] * np.sin(ph)).sum(axis=2)

    def _mean_speed(self):
        t = np.linspace(0.0, 3.0 / self.f[0], 900)
        P = self._raw(t * self.k)
        d = np.diff(P, axis=0)
        return float(np.mean(np.hypot(d[:, 0], d[:, 1])) / (t[1] - t[0]))

    def at(self, t):
        return self._raw(np.asarray([t * self.k]))[0]


# --------------------------------------------------------------------------------------
# a solved N-body
# --------------------------------------------------------------------------------------
class NBody:
    """Charges free to move under their own mutual softened Coulomb forces, inside a declared
    anisotropic harmonic well.

    The well is the rig: without it a charge that gains energy in a close pass simply leaves and
    the frame empties, and no amount of tuning the interactions fixes that (Shape_Physics rev 4
    hit exactly this and solved it the same way). It is ANISOTROPIC - the restoring force is
    computed in a metric where ry = (y-cy)/aniso - so orbits come out as vertical ellipses that
    fill a 9:16 frame with nothing fighting anything. Bodies stay round; only the field is
    stretched. That is the Shape_Physics rev-6 lesson carried over verbatim.

    `softening` is not cosmetic. An unsoftened opposite-sign pair reaches unbounded speed in
    finite time, and no fixed substep can follow it.
    """

    def __init__(self, pos, q, mass=None, well=2.4, aniso=1.778, coupling=9.0e4,
                 softening=95.0, core_k=0.0, core_soft=42.0,
                 drag=0.0, v_max=1900.0, substeps_per_sec=600):
        self.p = np.asarray(pos, float).copy()
        self.q = np.asarray(q, float).copy()
        self.m = np.ones(len(self.p)) if mass is None else np.asarray(mass, float).copy()
        self.v = np.zeros_like(self.p)
        self.c = self.p.mean(axis=0).copy()
        self.well, self.aniso, self.C = well, aniso, coupling
        self.soft2 = softening * softening
        # A SOFT CORE: these sources are beads with a size, not mathematical points. That is a
        # MATERIAL choice - the same class of input as picking a body's density in Wind_Tunnel -
        # and it is what stops an opposite-sign pair from slamming together (measured 12 px on
        # the first build, i.e. two 15 px markers fully overlapped). Falling as 1/r^3 against
        # Coulomb's 1/r^2, it is negligible at working distances and dominant only on contact,
        # so pairs settle into loose orbiting dipoles instead of colliding.
        self.core_k, self.core_soft2 = core_k, core_soft * core_soft
        self.drag, self.v_max = drag, v_max
        self.sps = substeps_per_sec
        self._t = 0.0

    def launch_tangential(self, gain=1.0):
        """Circular-orbit speed for THIS well, in the stretched metric. Launching in a random
        direction instead drops everything straight down the well and the swarm reads as a
        clump, not as orbits (Shape_Physics learned this twice)."""
        d = self.p - self.c
        u = np.stack([d[:, 0], d[:, 1] / self.aniso], axis=1)
        r = np.hypot(u[:, 0], u[:, 1]) + 1e-9
        sp = np.sqrt(self.well) * r * gain
        self.v = np.stack([-u[:, 1] / r * sp, u[:, 0] / r * sp * self.aniso], axis=1)
        return self

    def _accel(self):
        d = self.p - self.c
        a = -self.well * np.stack([d[:, 0], d[:, 1] / (self.aniso ** 2)], axis=1)
        rel = self.p[:, None, :] - self.p[None, :, :]
        r2 = rel[:, :, 0] ** 2 + rel[:, :, 1] ** 2 + self.soft2
        w = self.C * (self.q[:, None] * self.q[None, :]) / (r2 * np.sqrt(r2))
        if self.core_k:
            # 1/r^5, against Coulomb's 1/r^2 - steep enough to be genuinely SHORT range. A
            # 1/r^3 core (the first try) was still comparable to Coulomb at 200 px, so any
            # strength that could actually stop a 900 px/s body also blew the swarm apart.
            # Size it by the ENERGY it must absorb: U(0) = core_k/(4*core_soft^4) has to exceed
            # v_typ^2/2, else a fast body simply passes through the wall.
            rc = (rel[:, :, 0] ** 2 + rel[:, :, 1] ** 2 + self.core_soft2)
            w += self.core_k / (rc * rc * rc)
        np.fill_diagonal(w, 0.0)
        a += np.einsum("ij,ijk->ik", w, rel) / self.m[:, None]
        if self.drag:
            a -= self.drag * self.v
        return a

    def advance_to(self, t):
        """Fixed-rate substeps, decoupled from fps - so a 30 fps preview and a 60 fps final are
        the SAME simulation, sampled differently. Same rule as the rest of the library."""
        n = int(round((t - self._t) * self.sps))
        if n <= 0:
            return self
        h = (t - self._t) / n
        for _ in range(n):
            self.v += h * self._accel()
            sp = np.hypot(self.v[:, 0], self.v[:, 1])
            over = sp > self.v_max
            if over.any():
                self.v[over] *= (self.v_max / sp[over])[:, None]
            self.p += h * self.v
        self._t = t
        return self
