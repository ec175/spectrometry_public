# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""draw.py - the look. Black ground, a luminous gas in two layers, opaque cutaway metal over it.

THE GAS (2026-09-20 pass). The first look splatted every molecule as a 1.4 px dot: at 0.055
molecules per px^2 that is a dot every 4 px, and it rendered as a uniform noise carpet - the
texture was the problem, not the hue. Now the gas is TWO LAYERS from the same particles:

  HAZE    every molecule, splatted soft and wide. This is the density x temperature field:
          where the gas is dense it glows, where it is hot it glows in that species' hot
          colour. It reads as a continuous luminous gas.
  MOTES   a fixed SUBSET of molecules (by a per-particle tag carried through injection and
          removal, so the subset never reshuffles and never flickers), drawn as bright soft
          points with a short motion streak. These are the molecules you can follow.

THE COLOUR RULE (OUTPUT_RULES, and the Plume_Lab flicker fix): hue comes from the SPECIES,
brightness and the shift toward the hot stop from the particle's own temperature on a FIXED
[T0, T_REF_HOT] ramp. Nothing is normalised to the frame's own peak. The exposure curve is
applied to LUMINANCE with chromaticity preserved, so dense hot gas saturates to deep colour,
never to a white slab (that was the full-frame flash the first pass had).

Blue in, orange bang, red out: the convention every engine diagram uses, and the one the
reference cutaways use for their flow arrows.

THE METAL is parts.Canvas: opaque RGBA, drawn once for the static block and per frame for
the moving parts, downsampled with a box filter, composited OVER the gas. Metal now occludes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import engine as E
from . import parts
from .gpu import GPU, asnumpy, gaussian_filter
from .gpu import xp as gxp

# cold / mid / hot stops per species. Air heats through pale, never through grey; CO2 is
# born hot and cools into a deep ember red so burnt gas stays legible against the charge.
SPECIES_COLD = {C.AIR: (48, 122, 255), C.FUEL: (255, 178, 40), C.CO2: (236, 58, 28), C.H2O: (232, 128, 104)}
SPECIES_MID = {C.AIR: (104, 174, 255), C.FUEL: (255, 214, 110), C.CO2: (255, 110, 40), C.H2O: (255, 186, 150)}
SPECIES_HOT = {C.AIR: (255, 222, 168), C.FUEL: (255, 244, 196), C.CO2: (255, 150, 60), C.H2O: (255, 214, 176)}


@dataclass
class Style:
    # The splat weight is 1 per molecule per world px^2 and the blurs are normalised, so the
    # haze at ambient density is ~N0 (0.055) x haze_gain x the cold brightness (0.3); the
    # gains below put ambient at ~0.2 after the exposure curve and a hot TDC charge near
    # saturation. Tuned on cached snapshots with scratch/relook.py, 2026-09-20.
    gas_gain: float = 5.5          # the FIXED exposure curve 1 - exp(-gain * L)
    mote_frac: float = 0.09        # fraction of molecules drawn as bright motes
    mote_sigma: float = 2.2        # world px
    mote_gain: float = 6.0
    streak: int = 5                # sub-samples along last frame's displacement
    streak_max: float = 10.0       # world px cap on the motion streak
    haze_sigma: float = 9.0        # world px
    haze_gain: float = 3.6
    bloom_sigma: float = 26.0
    bloom_gain: float = 0.5
    dim_floor: float = 0.48        # brightness of a 300 K particle
    hot_knee: float = 1900.0       # K at which the brightness ramp is full
    mid_K: float = 1300.0
    hot_K: float = 2400.0
    saturation: float = 0.92       # < 1 pulls every colour a little toward its luminance
    ss: int = 2


def _lut(st: Style):
    bins = 64
    lut = np.zeros((4, bins, 3), dtype=np.float32)
    for sp in (C.AIR, C.FUEL, C.CO2, C.H2O):
        cold, mid, hot = SPECIES_COLD[sp], SPECIES_MID[sp], SPECIES_HOT[sp]
        for j in range(bins):
            T = C.T0 + (C.T_REF_HOT - C.T0) * j / (bins - 1.0)
            b = st.dim_floor + (1.0 - st.dim_floor) * min(1.0, max(0.0, (T - C.T0) / (st.hot_knee - C.T0)))
            if T <= st.mid_K:
                u = (T - C.T0) / max(1.0, st.mid_K - C.T0)
                c = [cold[k] + (mid[k] - cold[k]) * u for k in range(3)]
            else:
                u = min(1.0, (T - st.mid_K) / max(1.0, st.hot_K - st.mid_K))
                c = [mid[k] + (hot[k] - mid[k]) * u for k in range(3)]
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            c = [lum + (v - lum) * st.saturation for v in c]
            lut[sp, j] = [v * b / 255.0 for v in c]
    return lut


class Frame:
    """Reusable buffers. One instance per render."""

    def __init__(self, cfg, style: Style, eng: E.Engine):
        self.cfg, self.st, self.eng = cfg, style, eng
        self.W, self.H = cfg.width, cfg.height
        self.k = cfg.scale
        self.lut = gxp.asarray(_lut(style).reshape(-1, 3))
        from . import looks
        self.canvas = parts.Canvas(self.W, self.H, self.k, cfg.ss)
        if getattr(self, "_zoom", 1.0) != 1.0 or getattr(self, "_vc", None):
            self.canvas.set_view(getattr(self, "_zoom", 1.0), *getattr(self, "_vc", (None, None)))
        looks.draw_static(self.canvas, eng)
        self.static = self.canvas.snapshot()
        self.spark_t = {}

    # ------------------------------------------------------------------ the gas
    def _colours(self, snap):
        v2 = gxp.asarray(snap["v2"])
        sp = gxp.asarray(snap["sp"])
        T = C.T0 * v2 / (C.V_TH ** 2)
        tb = gxp.clip((T - C.T0) / (C.T_REF_HOT - C.T0) * 63.0, 0, 63).astype(gxp.int32)
        return self.lut[gxp.clip(sp, 0, 3) * 64 + tb]          # (n, 3)

    def _splat(self, acc, x, y, col, w):
        ix = gxp.rint(x * self.k).astype(gxp.int64)
        iy = gxp.rint(y * self.k).astype(gxp.int64)
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        idx = gxp.where(ok, iy * self.W + ix, 0)
        wk = gxp.where(ok, w, 0.0).astype(gxp.float32)
        n = self.H * self.W
        # the channel count comes from the ACCUMULATOR, not a literal 3: field.py splats a
        # one-channel density alongside the three-channel colour, and a hard-coded 3 threw
        # IndexError 15 minutes into a physics run rather than in the first second.
        for ch in range(acc.shape[1]):
            acc[:, ch] += gxp.bincount(idx, weights=col[:, ch] * wk, minlength=n)[:n]

    def gas(self, pairs):
        """pairs: [(snapshot, weight)]. Returns the exposed gas as float32 (H, W, 3) on the
        active backend. Weights are for the loop seam - equal-power, see render.py."""
        st, k = self.st, self.k
        n = self.H * self.W
        acc_m = gxp.zeros((n, 3), gxp.float32)
        acc_h = gxp.zeros((n, 3), gxp.float32)
        for snap, wgt in pairs:
            x, y = gxp.asarray(snap["x"]), gxp.asarray(snap["y"])
            if x.size == 0:
                continue
            px, py = gxp.asarray(snap["px"]), gxp.asarray(snap["py"])
            tag = gxp.asarray(snap["tag"])
            col = self._colours(snap)
            # weight per WORLD px^2 (k^2), so a preview and a final expose the same
            self._splat(acc_h, x, y, col, wgt * k * k)
            mote = (tag < int(st.mote_frac * 65536)).astype(gxp.float32)
            dx, dy = x - px, y - py
            L = gxp.sqrt(dx * dx + dy * dy)
            f = gxp.minimum(1.0, st.streak_max / gxp.maximum(L, 1e-6))
            px, py = x - dx * f, y - dy * f
            S = max(1, st.streak)
            for s in range(S):
                u = (s + 0.5) / S
                self._splat(acc_m, px + (x - px) * u, py + (y - py) * u, col,
                            mote * (wgt * k * k * st.mote_gain / S))
        m = acc_m.reshape(self.H, self.W, 3)
        h = acc_h.reshape(self.H, self.W, 3)
        motes = gaussian_filter(m, (st.mote_sigma * k, st.mote_sigma * k, 0))
        haze = gaussian_filter(h, (st.haze_sigma * k, st.haze_sigma * k, 0)) * st.haze_gain
        bloom = gaussian_filter(h, (st.bloom_sigma * k, st.bloom_sigma * k, 0)) * st.bloom_gain
        g = motes + haze + bloom
        lum = g.max(axis=2, keepdims=True)
        g = g * (1.0 - gxp.exp(-st.gas_gain * lum)) / gxp.maximum(lum, 1e-6)
        return g

    # ------------------------------------------------------------------ the metal
    def solids(self, t):
        from . import looks
        cv = self.canvas
        cv.restore(self.static)
        looks.draw_moving(cv, self.eng, t)
        return cv.finish()

    # ------------------------------------------------------------------ composite
    def compose(self, gas, t):
        rgb, a = self.solids(t)
        rgb, a = gxp.asarray(rgb), gxp.asarray(a)
        out = gas * (1.0 - a) + rgb * a
        out = gxp.clip(out, 0.0, 1.0) * 255.0 + 0.5
        return asnumpy(out.astype(gxp.uint8))
