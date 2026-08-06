"""glitchfx.py — the "error-prone acquisition" layer for crtfilm: makes the trace look like
the SIGNAL CHAIN misbehaves, not just the camera. Two passes around FilmLook:

  pre(frame, i)  — SIGNAL/SCOPE faults, applied to the clean render so the film look then
                   "films" them (they inherit halation/CA/grain like everything real):
    - SYNC TEARS: a horizontal band slips sideways for a few frames (trigger/sync error),
      snapping hard then relaxing back.
    - STATIC BURSTS in vertical strips: a column section fills with additive noise while
      its rows jitter horizontally ±a few px — localized interference static.
    - HERRINGBONE: a faint fine diagonal RF-interference pattern that drifts through and
      fades away over a second or two.
    - DROPOUTS: 1–3 frames where the whole signal dims hard under a splash of noise.

  post(frame, i) — SENSOR defects, applied after the film look:
    - DEAD PIXELS: a dozen fixed near-black pixels.
    - HOT PIXELS: a few stuck bright green-white pixels, slowly flickering.

Events are Poisson-scheduled from a seeded RNG (deterministic per seed); one of each kind is
forced early so even a 3 s sample shows the vocabulary. `level` scales rate AND strength.
"""
from __future__ import annotations

import numpy as np


def _ramp(i, f0, f1, a=1.5):
    """Fast-attack / smooth-release envelope over frame span [f0, f1)."""
    up = np.clip((i - f0 + 1) / a, 0.0, 1.0)
    dn = np.clip((f1 - i) / max((f1 - f0) * 0.4, 1.0), 0.0, 1.0)
    return float(min(up, dn))


class GlitchFX:
    def __init__(self, w, h, fps, n_frames, seed=11, level=1.0):
        self.w, self.h, self.fps = w, h, fps
        rng = np.random.default_rng(seed)
        L = level

        # sensor defects (fixed positions for the whole clip)
        nd = int(9 + 6 * L)
        self._dead = (rng.integers(0, h, nd), rng.integers(0, w, nd))
        nh = max(2, int(1 + 2 * L))
        self._hot = (rng.integers(0, h, nh), rng.integers(0, w, nh))
        self._hotph = rng.uniform(0, 2 * np.pi, nh)

        def schedule(gap_s, dur_f, first_s, mk):
            evts, t = [], first_s
            while t * fps < n_frames:
                f0 = int(t * fps)
                f1 = f0 + int(rng.uniform(*dur_f))
                evts.append((f0, f1, mk()))
                t += rng.exponential(gap_s)
            return evts

        self.tears = schedule(6.5 / L, (2, 6), 0.7, lambda: dict(
            y0=int(rng.uniform(0.10, 0.82) * h), bh=int(rng.uniform(0.04, 0.15) * h),
            dx=int(rng.uniform(8, 26) * L) * int(rng.choice([-1, 1]))))
        self.strips = schedule(8.0 / L, (4, 10), 1.9, lambda: dict(
            x0=int(rng.uniform(0.05, 0.90) * w), wd=int(rng.uniform(0.015, 0.05) * w),
            amp=rng.uniform(0.11, 0.20) * L, jit=int(rng.uniform(1, 3.9))))
        self.herrs = schedule(11.0 / L, (int(0.7 * fps), int(1.8 * fps)), 0.3, lambda: dict(
            amp=rng.uniform(0.012, 0.026) * L, kx=rng.uniform(0.25, 0.5),
            ky=rng.uniform(0.05, 0.14), spd=rng.uniform(18, 40) * rng.choice([-1, 1])))
        self.drops = schedule(15.0 / L, (1, 3), 2.4, lambda: dict(
            dim=rng.uniform(0.30, 0.55), namp=rng.uniform(0.05, 0.11) * L))

        yy, xx = np.mgrid[0:h, 0:w]
        self._xg = xx.astype(np.float32)
        self._yg = yy.astype(np.float32)

    @staticmethod
    def _active(evts, i):
        return [(e, _ramp(i, f0, f1)) for f0, f1, e in evts if f0 <= i < f1]

    # ------------------------------------------------------------------ signal faults
    def pre(self, frame_u8, i):
        acts_t = self._active(self.tears, i)
        acts_s = self._active(self.strips, i)
        acts_h = self._active(self.herrs, i)
        acts_d = self._active(self.drops, i)
        if not (acts_t or acts_s or acts_h or acts_d):
            return frame_u8
        f = frame_u8.astype(np.float32)
        rng = np.random.default_rng((i + 1) * 104729 + 7)

        for e, g in acts_t:                               # sync tear: band slips sideways
            dx = int(round(e["dx"] * g))
            if dx:
                y0, y1 = e["y0"], min(e["y0"] + e["bh"], self.h)
                f[y0:y1] = np.roll(f[y0:y1], dx, axis=1)

        for e, g in acts_s:                               # vertical-strip static + row jitter
            x0, x1 = e["x0"], min(e["x0"] + e["wd"], self.w)
            jit = e["jit"]
            shifts = rng.integers(-jit, jit + 1, self.h)
            for s in range(-jit, jit + 1):                # roll rows in groups (vectorised-ish)
                rows = np.where(shifts == s)[0]
                if s and len(rows):
                    f[rows, x0:x1] = np.roll(f[rows, x0:x1], s, axis=1)
            noise = rng.normal(0, 255 * e["amp"] * g, (self.h, x1 - x0, 1)).astype(np.float32)
            f[:, x0:x1] += noise * np.array([0.65, 1.0, 0.75], np.float32)

        for e, g in acts_h:                               # drifting RF herringbone
            ph = e["spd"] * i / self.fps
            pat = np.sin(self._xg * e["kx"] + self._yg * e["ky"] + ph)
            f += (255 * e["amp"] * g) * pat[..., None] * np.array([0.5, 1.0, 0.6], np.float32)

        for e, g in acts_d:                               # dropout: hard dim + noise splash
            dim = 1.0 - g * (1.0 - e["dim"])
            f *= dim
            f += rng.normal(0, 255 * e["namp"] * g, f.shape).astype(np.float32) * \
                np.array([0.6, 1.0, 0.7], np.float32)

        return np.clip(f, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ sensor defects
    def post(self, frame_u8, i):
        f = frame_u8.copy()
        ys, xs = self._dead
        f[ys, xs] = (f[ys, xs] * 0.12).astype(np.uint8)
        ys, xs = self._hot
        flick = 0.75 + 0.25 * np.sin(0.31 * i + self._hotph)
        hot = np.stack([200 * flick, 255 * flick, 215 * flick], axis=1)
        f[ys, xs] = np.clip(hot, 0, 255).astype(np.uint8)
        return f
