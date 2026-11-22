# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pframe.py - one frame of the 2026-09-25 parts set: the gas as a field, the metal from metal.py.

THE GAS is field.py's density/temperature field (Clancy, 2026-09-20: "way more 'solid'
gradients that show the pressure/temp of a gas rather than simulating all the individual gas
particles"), without any of that build's music machinery: no schedule, no flashes, no lock.
Every molecule deposits its weight and its weight x colour; both are blurred and divided, so
the picture is the local mean species x temperature at a brightness set by the local density
against a FIXED reference (ambient N0), never the frame's own peak (OUTPUT_RULES).

LOCAL GLOWS. A spark or an injector's flash is a small additive glow at a point, supplied by
the part (`eng.notes['glows'](t)` -> [(x, y, sigma, (r, g, b), amp)]). A local event, never a
frame-wide one.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .draw import _lut
from .field import field_style
from .gpu import asnumpy, gaussian_filter
from .gpu import xp as gxp
from .metal import Metal


# cold / mid / hot per species over the PART's own range. Air: deep blue -> ice -> warm white;
# fuel gold; burnt gas ember; water rose. The same conventions as draw.py, re-spaced.
P_COLD = {C.AIR: (40, 104, 236), C.FUEL: (236, 160, 30), C.CO2: (214, 60, 30), C.H2O: (210, 110, 96),
          C.N2: (62, 84, 140)}
P_MID = {C.AIR: (120, 196, 255), C.FUEL: (255, 206, 96), C.CO2: (255, 118, 44), C.H2O: (255, 170, 140),
         C.N2: (130, 152, 200)}
P_HOT = {C.AIR: (255, 226, 176), C.FUEL: (255, 242, 190), C.CO2: (255, 176, 84), C.H2O: (255, 214, 180),
         C.N2: (236, 226, 206)}


def part_lut(T_lo, T_hi, floor=0.62):
    bins = 64
    lut = np.zeros((5, bins, 3), dtype=np.float32)
    for sp in (C.AIR, C.FUEL, C.CO2, C.H2O, C.N2):
        for j in range(bins):
            u = j / (bins - 1.0)
            a, b, w = (P_COLD[sp], P_MID[sp], u / 0.5) if u < 0.5 else (P_MID[sp], P_HOT[sp], (u - 0.5) / 0.5)
            c = [a[k] + (b[k] - a[k]) * w for k in range(3)]
            br = floor + (1.0 - floor) * min(1.0, u / 0.7)
            lut[sp, j] = [v * br / 255.0 for v in c]
    return lut


class PartFrame:
    def __init__(self, cfg, eng, style=None, bands=10, band_gain=0.22):
        self.cfg, self.eng = cfg, eng
        self.st = style or field_style(gas_gain=1.05, bloom_gain=0.18, haze_sigma=11.0)
        self.W, self.H = cfg.width, cfg.height
        self.k = cfg.scale
        self.zoom = float(eng.notes.get("zoom", 1.0))
        self.vc = eng.notes.get("view", (C.REF_W * 0.5, C.REF_H * 0.5))
        # each part gets a FIXED temperature range of its own (a blower heats 300 -> 340 K,
        # a burn goes to 2600 K): fixed per part, never normalised to the frame (OUTPUT_RULES)
        self.T_lo, self.T_hi = eng.notes.get("T_range", (C.T0, C.T_REF_HOT))
        self.lut = gxp.asarray(part_lut(self.T_lo, self.T_hi).reshape(-1, 3))
        self.metal = Metal(cfg, eng, self.zoom, self.vc)
        self.bands, self.band_gain = int(bands), float(band_gain)
        self._rm = {}

    def _view(self, x, y):
        z, (cx, cy) = self.zoom, self.vc
        # the view centre lands at the middle of THIS frame (a split-screen half is not 9:16)
        return ((x - cx) * z + 0.5 * self.W / self.k, (y - cy) * z + 0.5 * self.H / self.k)

    def _colours(self, snap):
        v2 = gxp.asarray(snap["v2"])
        sp = gxp.asarray(snap["sp"])
        T = C.T0 * v2 / (C.V_TH ** 2)
        tb = gxp.clip((T - self.T_lo) / (self.T_hi - self.T_lo) * 63.0, 0, 63).astype(gxp.int32)
        return self.lut[gxp.clip(sp, 0, 4) * 64 + tb]

    def _splat(self, acc, x, y, col, w):
        ix = gxp.rint(x * self.k).astype(gxp.int64)
        iy = gxp.rint(y * self.k).astype(gxp.int64)
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        idx = gxp.where(ok, iy * self.W + ix, 0)
        wk = gxp.where(ok, w, 0.0).astype(gxp.float32)
        n = self.H * self.W
        for ch in range(acc.shape[1]):
            acc[:, ch] += gxp.bincount(idx, weights=col[:, ch] * wk, minlength=n)[:n]

    def gas(self, snap):
        st, k = self.st, self.k
        n = self.H * self.W
        acc_c = gxp.zeros((n, 3), gxp.float32)
        acc_n = gxp.zeros((n, 1), gxp.float32)
        x, y = gxp.asarray(snap["x"]), gxp.asarray(snap["y"])
        heat = bool(self.eng.notes.get("heat_look"))
        if x.size:
            col = self._colours(snap)
            x, y = self._view(x, y)
            w = k * k * self.zoom * self.zoom
            if heat:
                # COMBUSTION GRADE (four_stroke, wankel): brightness from PRESSURE (n*T), colour
                # weighted toward the hotter molecules. The plain field averaged a half-burnt
                # chamber's red with its charge's blue into purple, and dimmed hot gas as it
                # expanded - the burn was invisible in the first four_stroke final.
                Tn = gxp.asarray(snap["v2"]) / (C.V_TH ** 2)                   # T / T0
                wc = gxp.clip(Tn, 0.5, 12.0) ** 1.5
                self._splat(acc_c, x, y, col * wc[:, None], w)
                self._splat(acc_n, x, y, wc[:, None], w)
                acc_p = gxp.zeros((n, 1), gxp.float32)
                self._splat(acc_p, x, y, gxp.clip(Tn, 0.0, 12.0)[:, None], w)
            else:
                sw = self.eng.notes.get("species_weight")
                if sw:
                    # a TRACER weight: a few % of fuel in a runner vanishes into the mean colour;
                    # weighting it in the colour average (not the brightness) makes the plume read
                    sp = gxp.asarray(snap["sp"])
                    wc = gxp.ones(x.size, gxp.float32)
                    for k_, v_ in sw.items():
                        wc = gxp.where(sp == k_, gxp.float32(v_), wc)
                    self._splat(acc_c, x, y, col * wc[:, None], w)
                    self._splat(acc_n, x, y, wc[:, None], w)
                    acc_d = gxp.zeros((n, 1), gxp.float32)
                    self._splat(acc_d, x, y, gxp.ones((x.size, 1), gxp.float32), w)
                else:
                    self._splat(acc_c, x, y, col, w)
                    self._splat(acc_n, x, y, gxp.ones((x.size, 1), gxp.float32), w)
        s = st.haze_sigma * k * self.zoom
        Cf = gaussian_filter(acc_c.reshape(self.H, self.W, 3), (s, s, 0))
        D = gaussian_filter(acc_n.reshape(self.H, self.W, 1), (s, s, 0))
        mean_col = Cf / gxp.maximum(D, 1e-6)
        if heat and x.size:
            Pf = gaussian_filter(acc_p.reshape(self.H, self.W, 1), (s, s, 0))
            u = gxp.clip(Pf / gxp.float32(C.N0 * 1.15), 0.0, 12.0) * gxp.float32(1.6)
        elif self.eng.notes.get("species_weight") and x.size:
            Dd = gaussian_filter(acc_d.reshape(self.H, self.W, 1), (s, s, 0))
            u = gxp.clip(Dd / gxp.float32(C.N0 * 1.15), 0.0, 6.0)
        else:
            u = gxp.clip(D / gxp.float32(C.N0 * 1.15), 0.0, 6.0)
        amt = 1.0 - gxp.exp(-st.gas_gain * u)
        g = mean_col * amt
        if st.bloom_gain > 0:
            b = st.bloom_sigma * k * self.zoom
            g = g + gaussian_filter(g, (b, b, 0)) * st.bloom_gain
        if self.bands > 0 and self.band_gain > 0:
            lev = amt * self.bands
            fr = lev - gxp.floor(lev)
            edge = gxp.exp(-((fr - 0.5) ** 2) / (2.0 * 0.085 ** 2))
            major = gxp.exp(-((gxp.floor(lev) % 3) / 0.45) ** 2)
            g = g * (1.0 + self.band_gain * edge * (0.55 + 0.45 * major) * amt)
        lum = g.max(axis=2, keepdims=True)
        return g * (1.0 - gxp.exp(-1.35 * lum)) / gxp.maximum(lum, 1e-6)

    def _glow(self, x, y, sigma):
        X, Y = self._view(x, y)
        key = (round(X, 1), round(Y, 1), round(sigma, 1))
        if key not in self._rm:
            if len(self._rm) > 32:
                self._rm.clear()
            s = sigma * self.zoom
            ys = gxp.arange(self.H, dtype=gxp.float32).reshape(-1, 1) / self.k
            xs = gxp.arange(self.W, dtype=gxp.float32).reshape(1, -1) / self.k
            self._rm[key] = gxp.exp(-((xs - X) ** 2 + (ys - Y) ** 2) / (2.0 * s * s)).astype(gxp.float32)
        return self._rm[key]

    def compose(self, gas, t):
        rgb, a = self.metal.render(t)
        fx = self.eng.notes.get("glows")
        if fx is not None:
            for (x, y, sig, col, amp) in fx(t):
                if amp > 1e-3:
                    gas = gas + self._glow(x, y, sig)[..., None] * gxp.asarray(
                        np.asarray(col, np.float32)) * float(amp)
        out = gas * (1.0 - a) + rgb
        over = self.eng.notes.get("over")          # glows that sit ON the metal (a spark gap)
        if over is not None:
            for (x, y, sig, col, amp) in over(t):
                if amp > 1e-3:
                    out = out + self._glow(x, y, sig)[..., None] * gxp.asarray(
                        np.asarray(col, np.float32)) * float(amp)
        out = gxp.clip(out, 0.0, 1.0) * 255.0 + 0.5
        return asnumpy(out.astype(gxp.uint8))

    def layout(self, t=0.0):
        """Metal only, on black: the drawing check, and the rendered-extent check."""
        rgb, a = self.metal.render(t)
        out = gxp.clip(rgb, 0.0, 1.0) * 255.0 + 0.5
        return asnumpy(out.astype(gxp.uint8)), asnumpy(a[..., 0])
