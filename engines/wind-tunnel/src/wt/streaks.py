"""streaks.py — the advected streakline layer (the white dashes riding on the colour field).

This is the second half of what makes the source video read as *air* rather than as a heatmap:
a dense field of massless tracers, each drawn as a short comet-tail. The colour field says how
fast; the streaks say which way, and they are what makes a recirculating separation bubble
visibly recirculate.

Two design choices worth keeping:

1. **Tails are integrated BACKWARD from the current position each frame**, not accumulated from
   a position history. The velocity field is quasi-steady over one frame, so a backward walk is
   the true streakline tail to within the frame step — and it costs no per-particle history, it
   cannot desynchronise when a particle respawns, and the tail length becomes a free parameter
   (`span`) instead of being locked to however many frames of history we happened to store.

2. **Particles die of old age and respawn UNIFORMLY over the domain**, rather than being seeded
   at the inlet. Inlet seeding looks right for about a second and then starves: tracers pile up
   in the slow wake and evacuate the fast regions, so the dashes thin out exactly where the
   interesting flow is. A short randomised lifetime holds the density flat everywhere, which is
   what the source video does.

LOOPING MODE (`period > 0`)
--------------------------
The default respawn above draws a FRESH RANDOM position from an RNG stream every time a particle
dies, so the tracer layer can never repeat — which is fine for a clip that just ends, and fatal
for one that has to loop. In looping mode every slot instead gets a FIXED home position, a FIXED
birth phase, and a lifetime that is an exact integer division of the clip period:

    age_i(t) = (t - phase_i) mod cycle_i,   cycle_i = period / m_i,   m_i integer

Age is then a pure function of `t`, and a particle's position is the trajectory from `home_i`
started at `t - age_i(t)`. Both are T-periodic the instant the VELOCITY FIELD is, so the whole
ensemble is — with no randomness left anywhere in the update. `m_i` is drawn from the same
lifetime range as the default mode, so the dash-length distribution on screen is unchanged.

The one rule this imposes: a particle that leaves the domain (or enters a body) is **parked**,
invisible, until its next SCHEDULED birth. Respawning it early is what the default mode does and
it would break periodicity, because the extra birth is not on the schedule. Parked slots cost
some tracer density, so a looping scene wants a higher `n_streaks` (measure it, do not guess).

Note what looping mode does NOT need: any assumption about the flow before t=0. A particle alive
at t=0 was born at most `max(cycle)` earlier, so as long as the field is in the same quiet state
for that long before t=0 (the settle) and for that long before t=T (the clip's quiet tail), the
two ensembles match exactly. `warmup_frames()` is that window.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy, gaussian_filter, scatter_add, xp


def _bilinear(a, px, py):
    """Sample (ny, nx) array `a` at float lattice coords — the tracers live between cells."""
    ny, nx = a.shape
    x0 = xp.clip(xp.floor(px), 0, nx - 2).astype(xp.int32)
    y0 = xp.clip(xp.floor(py), 0, ny - 2).astype(xp.int32)
    fx = xp.clip(px - x0, 0.0, 1.0)
    fy = xp.clip(py - y0, 0.0, 1.0)
    x1, y1 = x0 + 1, y0 + 1
    return ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
            + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)


class Streaks:
    def __init__(self, nx, ny, n=16000, fps=60, steps=14, tail=14, span=1.05,
                 life=(0.55, 1.5), seed=7, period=0.0, step_dt=1.0):
        """`step_dt` is how much SOLVER TIME one step advances, and it is not always 1.

        Everything below advects with `dt = self.steps / substeps`, i.e. it assumes a tracer
        moves `velocity x steps` cells per frame. That is exactly right for the LBM, whose time
        unit IS one step - and wrong for `wt/cns.py`, which holds a fixed dt of ~0.135, so the
        same expression would carry every tracer 7.4x further per frame than the fluid moved.
        The dashes would have run away from the field they are drawn on, in a picture whose
        entire job is to show where the flow is going. Folding it in here once means both the
        advection and the backward-integrated tail inherit it, and `streak_span` keeps meaning
        FRAMES OF TRAVEL in either solver.
        """
        self.nx, self.ny, self.n = int(nx), int(ny), int(n)
        self.fps, self.steps = float(fps), float(steps) * float(step_dt)
        self.tail, self.span = int(tail), float(span)
        self.life_lo, self.life_hi = float(life[0]), float(life[1])
        self.period = float(period)
        self._rng = xp.random.default_rng(seed)
        self.px = self._rng.random(self.n, dtype=xp.float32) * (self.nx - 1)
        self.py = self._rng.random(self.n, dtype=xp.float32) * (self.ny - 1)
        self.life = self._new_life(self.n)
        # stagger initial ages across the lifetime so the whole field doesn't blink in unison
        self.age = self._rng.random(self.n, dtype=xp.float32) * self.life
        if self.period > 0:
            # Quantise each slot's lifetime to an exact integer division of the clip period. The
            # draw is the SAME lifetime distribution as above, just rounded to the nearest number
            # of whole lives per period, so nothing about the look changes - what changes is that
            # `age` becomes a closed-form function of t instead of an accumulator.
            m = xp.maximum(1.0, xp.round(self.period / self.life))
            self.cycle = (self.period / m).astype(xp.float32)
            self.life = self.cycle
            self.phase = self._rng.random(self.n, dtype=xp.float32) * self.cycle
            self.home_x = self._rng.random(self.n, dtype=xp.float32) * (self.nx - 1)
            self.home_y = self._rng.random(self.n, dtype=xp.float32) * (self.ny - 1)
            self.parked = xp.zeros(self.n, dtype=bool)
            self.reset_clock(0.0)

    def _new_life(self, k):
        return (self.life_lo + (self.life_hi - self.life_lo)
                * self._rng.random(k, dtype=xp.float32))

    # -- looping mode ---------------------------------------------------------------------
    def warmup_frames(self):
        """Frames of advection needed before t=0 for every slot to have been reborn at least once.

        Anything shorter and the ensemble at t=0 still remembers its arbitrary initial placement,
        which the ensemble at t=T does not - the one way this scheme can fail.
        """
        if self.period <= 0:
            return 6
        return int(np.ceil(float(asnumpy(self.cycle.max())) * self.fps)) + 1

    def reset_clock(self, t0):
        """Put the ensemble at time `t0` (used with a negative t0 to run the warm-up into t=0)."""
        self._t = float(t0)
        self.age = ((self._t - self.phase) % self.cycle).astype(xp.float32)
        self.px = self.home_x.copy()
        self.py = self.home_y.copy()
        self.parked = xp.zeros(self.n, dtype=bool)

    def _advance_periodic(self, ux, uy, solid, substeps):
        self._t += 1.0 / self.fps
        age = ((self._t - self.phase) % self.cycle).astype(xp.float32)
        born = age < self.age                    # the modulo wrapped -> this slot's next life
        self.px = xp.where(born, self.home_x, self.px)
        self.py = xp.where(born, self.home_y, self.py)
        self.parked = self.parked & ~born
        self.age = age

        dt = self.steps / float(substeps)
        for _ in range(substeps):
            vx = _bilinear(ux, self.px, self.py)
            vy = _bilinear(uy, self.px, self.py)
            mx = xp.clip(self.px + 0.5 * dt * vx, 0, self.nx - 1)
            my = xp.clip(self.py + 0.5 * dt * vy, 0, self.ny - 1)
            self.px = self.px + dt * _bilinear(ux, mx, my)
            self.py = self.py + dt * _bilinear(uy, mx, my)

        # PARK rather than respawn: an off-schedule birth is the one thing that would make the
        # ensemble non-periodic. A parked slot is simply invisible until its next scheduled birth.
        inside = _bilinear(solid.astype(xp.float32), xp.clip(self.px, 0, self.nx - 1),
                           xp.clip(self.py, 0, self.ny - 1)) > 0.5
        self.parked = (self.parked | inside
                       | (self.px < 0) | (self.px > self.nx - 1)
                       | (self.py < 0) | (self.py > self.ny - 1))

    # -- motion --------------------------------------------------------------------------
    def advance(self, ux, uy, solid, substeps=3):
        """Advect one frame's worth of flow (RK2 midpoint), then age / respawn."""
        if self.period > 0:
            return self._advance_periodic(ux, uy, solid, substeps)
        dt = self.steps / float(substeps)
        for _ in range(substeps):
            vx = _bilinear(ux, self.px, self.py)
            vy = _bilinear(uy, self.px, self.py)
            mx = self.px + 0.5 * dt * vx
            my = self.py + 0.5 * dt * vy
            mx = xp.clip(mx, 0, self.nx - 1)
            my = xp.clip(my, 0, self.ny - 1)
            self.px = self.px + dt * _bilinear(ux, mx, my)
            self.py = self.py + dt * _bilinear(uy, mx, my)

        self.age += 1.0 / self.fps
        inside = _bilinear(solid.astype(xp.float32), xp.clip(self.px, 0, self.nx - 1),
                           xp.clip(self.py, 0, self.ny - 1)) > 0.5
        dead = ((self.age > self.life) | inside
                | (self.px < 0) | (self.px > self.nx - 1)
                | (self.py < 0) | (self.py > self.ny - 1))
        k = int(dead.sum())
        if k:
            self.px = xp.where(dead, self._rng.random(self.n, dtype=xp.float32) * (self.nx - 1),
                               self.px)
            self.py = xp.where(dead, self._rng.random(self.n, dtype=xp.float32) * (self.ny - 1),
                               self.py)
            self.life = xp.where(dead, self._new_life(self.n), self.life)
            self.age = xp.where(dead, xp.zeros(self.n, dtype=xp.float32), self.age)

    # -- drawing -------------------------------------------------------------------------
    def _alpha(self):
        """Fade in over the first 15% of life and out over the last 30% — no popping."""
        u = xp.clip(self.age / xp.maximum(self.life, 1e-6), 0.0, 1.0)
        a = xp.clip(u / 0.15, 0, 1) * xp.clip((1.0 - u) / 0.30, 0, 1)
        if self.period > 0:
            a = xp.where(self.parked, 0.0, a)
        return a

    def draw(self, ux, uy, to_screen, w, h, sigma=0.75, speed_ref=0.10, speed_gain=0.55):
        """Backward-integrate each tracer's tail and splat it into an (h, w) float buffer.

        `to_screen(px, py) -> (sx, sy)` is supplied by the renderer, so the streaks inherit
        whatever lattice->screen orientation the format uses (see config.RenderConfig.flow).
        """
        buf = xp.zeros(h * w, dtype=xp.float32)
        a0 = self._alpha()
        # brighter where the flow is quick: a real streak photo exposes fast tracers harder
        sp = xp.sqrt(_bilinear(ux, self.px, self.py) ** 2
                     + _bilinear(uy, self.px, self.py) ** 2)
        a0 = a0 * (1.0 + speed_gain * (xp.clip(sp / max(speed_ref, 1e-6), 0.0, 1.8) - 1.0))

        bx, by = self.px.copy(), self.py.copy()
        dtb = -(self.steps * self.span) / max(self.tail, 1)
        for s in range(self.tail):
            wgt = a0 * (1.0 - s / float(self.tail)) ** 1.35
            self._splat(buf, to_screen(bx, by), wgt, w, h)
            vx = _bilinear(ux, xp.clip(bx, 0, self.nx - 1), xp.clip(by, 0, self.ny - 1))
            vy = _bilinear(uy, xp.clip(bx, 0, self.nx - 1), xp.clip(by, 0, self.ny - 1))
            bx = bx + dtb * vx
            by = by + dtb * vy

        img = buf.reshape(h, w)
        if sigma > 0:
            img = gaussian_filter(img, sigma=sigma)
        return img

    @staticmethod
    def _splat(buf, sxy, wgt, w, h):
        """Bilinear point splat into a FLAT buffer (one scatter_add per corner)."""
        sx, sy = sxy
        ok = (sx >= 0) & (sx <= w - 2) & (sy >= 0) & (sy <= h - 2)
        sx = xp.where(ok, sx, 0.0)
        sy = xp.where(ok, sy, 0.0)
        wgt = xp.where(ok, wgt, 0.0)
        x0 = sx.astype(xp.int32)
        y0 = sy.astype(xp.int32)
        fx, fy = sx - x0, sy - y0
        base = y0 * w + x0
        scatter_add(buf, base, wgt * (1 - fx) * (1 - fy))
        scatter_add(buf, base + 1, wgt * fx * (1 - fy))
        scatter_add(buf, base + w, wgt * (1 - fx) * fy)
        scatter_add(buf, base + w + 1, wgt * fx * fy)
