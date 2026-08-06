"""holo.py - holofoil / iridescent colour.

Holofoil is NOT "a rainbow". Three properties separate it from a hue ramp, and all three are
what make the effect read:

1. **The hue cycles many times across the object.** A single sweep from red to violet is a
   gradient; foil repeats the spectrum several times over, so you see bands. `cycles`.

2. **The spectrum is not perceptually uniform - it is thin-film interference.** Real foil
   spends a long time in magenta/blue/cyan and passes quickly through green, with a strong
   gold band. An HSV sweep gives equal time to every hue and comes out looking like a
   test pattern. `_FILM` below is weighted, not uniform.

3. **Hue shifts with VIEWING DEPTH.** That is the physics - path length through the film sets
   the wavelength that constructively interferes. It is also the single cheapest depth cue
   available here: `eye.layer` is fed in as a hue offset, so the sclera, the iris and the
   pupil land on different bands of the same foil and read as separate sheets stacked in
   space rather than regions painted on one surface.

Plus a `sheen`: a narrow, higher-frequency band running across the hue field that spikes to
near-white. That is the specular glint off the foil's surface, and without it the colour is
saturated but flat - it looks like coloured plastic instead of metal.
"""
from __future__ import annotations

import numpy as np

# Thin-film-ish spectral ramp, as (position, rgb) stops on a 0..1 loop. Deliberately uneven:
# long dwell in magenta -> violet -> blue -> cyan, a quick pass through green, then a wide
# gold/amber shoulder back into magenta. Matches how real holographic film reads.
_FILM = [
    (0.000, (1.00, 0.18, 0.62)),   # magenta
    (0.085, (0.72, 0.20, 1.00)),   # violet
    (0.200, (0.22, 0.36, 1.00)),   # blue
    (0.330, (0.10, 0.86, 1.00)),   # cyan
    (0.430, (0.24, 1.00, 0.66)),   # spring green - passed through quickly
    (0.500, (0.72, 1.00, 0.25)),   # yellow-green
    (0.590, (1.00, 0.84, 0.16)),   # gold
    (0.720, (1.00, 0.50, 0.16)),   # amber
    (0.850, (1.00, 0.28, 0.34)),   # red-pink
    (1.000, (1.00, 0.18, 0.62)),   # back to magenta (closes the loop)
]


def film_lut(n=1024):
    pos = np.array([p for p, _ in _FILM], np.float32)
    col = np.array([c for _, c in _FILM], np.float32)
    x = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    i = np.clip(np.searchsorted(pos, x, side="right") - 1, 0, len(pos) - 2)
    f = ((x - pos[i]) / np.maximum(pos[i + 1] - pos[i], 1e-6))[:, None]
    return (col[i] * (1 - f) + col[i + 1] * f).astype(np.float32)


class Holofoil:
    """Iridescent colour as a function of (position, depth, time).

    `__call__` returns (N,3) linear RGB with mean luminance normalised, so swapping the
    palette does not silently re-expose the whole render.
    """

    def __init__(self, w, h, seed=0, cycles=3.4, axis_deg=118.0, warp=0.34, warp_scale=430.0,
                 drift=0.055, sheen=0.55, sheen_cycles=11.0, depth_shift=0.30, n=1024):
        self.lut = film_lut(n)
        self.w, self.h = float(w), float(h)
        self.cycles = float(cycles)
        self.depth_shift = float(depth_shift)
        self.sheen, self.sheen_cycles = float(sheen), float(sheen_cycles)
        self.drift = float(drift)
        rng = np.random.default_rng(seed + 4421)
        a = np.deg2rad(axis_deg)
        # base sweep direction, normalised so `cycles` means cycles across the frame diagonal
        d = np.hypot(w, h)
        self.kx, self.ky = np.cos(a) / d, np.sin(a) / d
        # warp: a low-frequency wobble so the bands are not dead-straight ruled lines
        self.warp = float(warp)
        na = rng.uniform(0, 2 * np.pi, 4)
        nk = 2 * np.pi / (warp_scale * rng.uniform(0.7, 1.6, 4))
        self.wkx, self.wky = np.cos(na) * nk, np.sin(na) * nk
        self.wph = rng.uniform(0, 2 * np.pi, 4)
        self.wsp = rng.uniform(-1, 1, 4) * 0.09
        self.lum = self.lut @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        self.mean_lum = float(self.lum.mean())

    def coord(self, P, t, layer=None):
        """The foil coordinate in [0,1) - hue phase at each point."""
        u = (P[:, 0] * self.kx + P[:, 1] * self.ky) * self.cycles
        wob = np.sin(P[:, 0:1] * self.wkx[None] + P[:, 1:2] * self.wky[None]
                     + self.wph[None] + self.wsp[None] * t).mean(1)
        u = u + self.warp * wob + self.drift * t
        if layer is not None:
            u = u + self.depth_shift * layer
        return np.mod(u, 1.0)

    def __call__(self, P, t, layer=None, jitter=None):
        u = self.coord(P, t, layer)
        if jitter is not None:
            u = np.mod(u + 0.018 * jitter, 1.0)
        idx = np.minimum((u * len(self.lut)).astype(np.int32), len(self.lut) - 1)
        rgb = self.lut[idx]
        if self.sheen > 1e-4:
            # narrow specular band; ** high power keeps it a glint, not a wash
            s = 0.5 + 0.5 * np.sin(2 * np.pi * (u * self.sheen_cycles))
            rgb = rgb + self.sheen * (s ** 9)[:, None] * (1.0 - rgb)
        return np.clip(rgb, 0.0, 1.0)
