# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""field.py - the gas as a SOLID FIELD, and the combustion flashes.

Clancy, 2026-09-20: *"the gasses 'move' way too quickly, it looks fuzzy all the time. This can
and should be solved with way more 'solid' gradients that show the pressure/temp of a gas
rather than simulating all the individual gas particles."*

He is right about the cause. At V_TH the molecules cross 150-200 output px in one 60 fps
frame. Drawing each one - even as a capped streak - means every pixel is re-rolled every
frame from a different handful of molecules, and that IS the fuzz: it is sampling noise, not
motion. The molecules keep being simulated (the physics is untouched and still gates on
PV = NkT); they stop being drawn.

WHAT IS DRAWN INSTEAD. Every molecule deposits into two accumulators at output resolution:
its weight, and its weight times its colour (which already carries species and temperature
from draw._lut). Blur both; divide. That gives, per pixel,

    D  = local number density   -> the PRESSURE read: how much gas is here
    C  = local mean colour      -> the SPECIES x TEMPERATURE read: what it is and how hot

`D` is the thing that stops the fuzz: a ratio of two blurred sums is stable frame to frame
even though the molecules under it have all moved, because it is an average over thousands of
them rather than a scatter of hundreds. The blur radius is the only knob that trades softness
against detail, and it is quoted in WORLD px so a preview and a final agree.

ISO-BANDS (his choice of the four looks). A thermal map reads its own gradient better with
level lines on it, so the exposed field is banded: thin bright contours at `bands` equal
levels of the exposure value, heavier on every third. They are drawn from the SAME scalar
the brightness comes from, so a contour is never in a place the gradient is not.

THE FLASHES. OUTPUT_RULES bans a full-frame luminance jump and Clancy has asked for that five
times - this module is the one place in the library that deliberately does it, on his explicit
2026-09-20 instruction, for the Flashing Lights clips only. Three treatments, one per final;
each is a pair, because the brief is two DIFFERENT flashes, one per word of the hook:

    strobe_flood   'a' three hard stutters inside the firing chamber | 'b' one sustained wash
    chamber_engine 'a' a white core in the chamber alone             | 'b' the metal itself lights
    core_rays      'a' a tight core burst                            | 'b' rays + a shock ring

Every flash is anchored on the cylinder that actually fired, at the time the schedule says it
fired, which is a vocal onset. Nothing here invents an event.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .draw import Frame, Style
from .gpu import asnumpy, gaussian_filter
from .gpu import xp as gxp


class FieldStyle(Style):
    pass


def field_style(**kw):
    st = Style()
    # the field wants density, not sparkle: no motes, a wide blur, a gentler curve
    st.mote_frac = 0.0
    st.gas_gain = 3.2
    st.haze_sigma = 13.0        # world px - THE softness knob
    st.haze_gain = 1.0
    st.bloom_sigma = 34.0
    st.bloom_gain = 0.22
    st.dim_floor = 0.30
    st.saturation = 1.0
    for k, v in kw.items():
        setattr(st, k, v)
    return st


class FieldFrame(Frame):
    """draw.Frame with the particle splat replaced by a density/temperature field, plus the
    flash treatments. Solids, the colour LUT and the composite are inherited unchanged."""

    def __init__(self, cfg, style, eng, sched=None, treatment="strobe_flood",
                 bands=12, band_gain=0.32, ema=0.0, zoom=1.0, view=None):
        self._zoom = float(zoom)
        self._vc = view or (C.REF_W * 0.5, C.REF_H * 0.5)
        super().__init__(cfg, style, eng)
        self.sched = sched
        self.treatment = treatment
        assert abs(self.canvas.zoom - self._zoom) < 1e-9, "view must be set before draw_static"

        self.bands = int(bands)
        self.band_gain = float(band_gain)
        self.ema = float(ema)
        self._prev = None
        self._centres = self._cyl_centres()

    def _view(self, x, y):
        """World px -> view px, the SAME map parts.Canvas applies to the metal."""
        z, (cx, cy) = self._zoom, self._vc
        return ((x - cx) * z + C.REF_W * 0.5, (y - cy) * z + C.REF_H * 0.5)

    # ------------------------------------------------------------------ geometry
    def _cyl_centres(self):
        """World-frame centre of each cylinder's combustion space, and its outward unit
        vector - what the flashes are anchored on."""
        eng = self.eng
        out = []
        for i in self.eng.cyl_boxes():
            b = eng.boxes[i]
            g = eng.groups[b.group]
            cx = 0.5 * (b.x0 + b.x1)
            cy = b.y0 + 0.18 * (b.y1 - b.y0)
            X, Y = g.to_world(cx, cy)
            ax, ay = g.to_world(cx, cy - 100.0)
            n = math.hypot(ax - X, ay - Y) or 1.0
            out.append(dict(box=i, group=b.group, x=X, y=Y,
                            ux=(ax - X) / n, uy=(ay - Y) / n))
        return out

    def _radial_map(self, cx, cy, sigma):
        """exp(-r^2/2s^2) about a world point, at output resolution. Cached per (cx,cy,s)."""
        cx, cy = self._view(cx, cy)
        sigma = sigma * self._zoom
        key = (round(cx, 1), round(cy, 1), round(sigma, 1))
        if not hasattr(self, "_rm"):
            self._rm = {}
        if key not in self._rm:
            if len(self._rm) > 24:
                self._rm.clear()
            ys = gxp.arange(self.H, dtype=gxp.float32).reshape(-1, 1) / self.k
            xs = gxp.arange(self.W, dtype=gxp.float32).reshape(1, -1) / self.k
            r2 = (xs - cx) ** 2 + (ys - cy) ** 2
            self._rm[key] = gxp.exp(-r2 / (2.0 * sigma * sigma)).astype(gxp.float32)
        return self._rm[key]

    # ------------------------------------------------------------------ the field
    def gas(self, pairs):
        """Returns the exposed, banded field as float32 (H, W, 3) on the active backend."""
        st, k = self.st, self.k
        n = self.H * self.W
        acc_c = gxp.zeros((n, 3), gxp.float32)
        acc_n = gxp.zeros((n, 1), gxp.float32)
        for snap, wgt in pairs:
            x, y = gxp.asarray(snap["x"]), gxp.asarray(snap["y"])
            if x.size == 0:
                continue
            col = self._colours(snap)
            x, y = self._view(x, y)
            w = wgt * k * k * self._zoom * self._zoom
            self._splat(acc_c, x, y, col, w)
            self._splat(acc_n, x, y, gxp.ones((x.size, 1), gxp.float32), w)
        s = st.haze_sigma * k * self._zoom
        Cf = gaussian_filter(acc_c.reshape(self.H, self.W, 3), (s, s, 0))
        D = gaussian_filter(acc_n.reshape(self.H, self.W, 1), (s, s, 0))
        # mean colour where there is gas; the ratio is what makes the picture stable
        mean_col = Cf / gxp.maximum(D, 1e-6)
        # density -> opacity, on a FIXED reference (ambient N0), never the frame's own peak
        u = gxp.clip(D / gxp.float32(C.N0 * 1.15), 0.0, 6.0)
        amt = 1.0 - gxp.exp(-st.gas_gain * u)
        g = mean_col * amt
        if st.bloom_gain > 0:
            b = st.bloom_sigma * k * self._zoom
            g = g + gaussian_filter(g, (b, b, 0)) * st.bloom_gain
        # ---- iso-bands, from the same scalar the brightness came from ----
        if self.bands > 0 and self.band_gain > 0:
            lev = amt * self.bands
            fr = lev - gxp.floor(lev)
            edge = gxp.exp(-((fr - 0.5) ** 2) / (2.0 * 0.085 ** 2))
            major = gxp.exp(-((gxp.floor(lev) % 3) / 0.45) ** 2)     # every 3rd line heavier
            g = g * (1.0 + self.band_gain * edge * (0.55 + 0.45 * major) * amt)
        if self.ema > 0:
            g = g if self._prev is None else (1 - self.ema) * g + self.ema * self._prev
            self._prev = g
        lum = g.max(axis=2, keepdims=True)
        return g * (1.0 - gxp.exp(-1.35 * lum)) / gxp.maximum(lum, 1e-6)

    # ------------------------------------------------------------------ the flashes
    def _fire_state(self, t):
        """(kind -> envelope, kind -> anchor cylinder).

        THE ANCHOR IS PER KIND, and it has to be. The first version took the most recent fire
        of ANY kind while the brightness came from the most recent fire of THAT kind - so
        while a word flash was still decaying, the next combustion moved the anchor and a
        chamber that had not fired lit up. Invisible in a still taken at the flash; obvious a
        quarter of a second later.
        """
        s = self.sched
        if s is None:
            return {}, {}
        # the schedule owns both the envelope and the cylinder, so a flash can belong to a
        # combustion it is not simultaneous with (an echo rides the burn already in progress)
        env, anchor = {}, {}
        for kind in ("a", "b"):
            env[kind] = s.flash(t, kind)
            cyl = s.flash_anchor(t, kind)
            if cyl is not None:
                for e in self._centres:
                    if e["group"] == cyl:
                        anchor[kind] = e
                        break
        return env, anchor

    def compose(self, gas, t):
        rgb, a = self.solids(t)
        return self.compose_with(gas, rgb, a, t)

    def compose_with(self, gas, rgb, a, t):
        """The composite, given a gas field and a solids layer that were computed ONCE. The
        three treatments differ only here, so a three-clip batch runs the physics, the field
        and the metal once per frame and pays three times only for this."""
        env, anchor = self._fire_state(t)
        rgb, a = gxp.asarray(rgb), gxp.asarray(a)
        ea, eb = env.get("a", 0.0), env.get("b", 0.0)
        ca, cb = anchor.get("a"), anchor.get("b")
        gas = gas.copy()
        metal_boost = gxp.float32(0.0)

        if ea > 0.002 or eb > 0.002:
            tr = self.treatment
            if tr == "strobe_flood":
                # 'a': three hard stutters inside the one chamber
                if ea > 0.002 and ca is not None:
                    st = 0.35 + 0.65 * abs(math.cos(2.0 * math.pi * 3.0 * self._since(t, "a")))
                    m = self._radial_map(ca["x"], ca["y"], 82.0)
                    gas = gas + m[..., None] * gxp.asarray(
                        np.array([1.0, 0.93, 0.80], np.float32)) * (3.3 * ea * st)
                # 'b': one sustained wash over the whole engine
                if eb > 0.002:
                    wash = self._engine_map(190.0)
                    gas = gas + wash[..., None] * gxp.asarray(
                        np.array([1.0, 0.86, 0.66], np.float32)) * (1.55 * eb)
                    metal_boost = gxp.float32(0.55 * eb)

            elif tr == "chamber_engine":
                # 'a': a white core in the chamber alone - the gas, not the metal
                if ea > 0.002 and ca is not None:
                    m = self._radial_map(ca["x"], ca["y"], 78.0)
                    gas = gas + m[..., None] * gxp.asarray(
                        np.array([1.0, 0.97, 0.90], np.float32)) * (3.6 * ea)
                # 'b': the light escapes and catches the machine
                if eb > 0.002:
                    # 1.7 drove every metal pixel to flat white and lost the machine inside
                    # the flash; 1.05 keeps the hatching and the edges readable while the
                    # whole engine still jumps. Measured on out\look\_fx_B.png.
                    metal_boost = gxp.float32(1.05 * eb)
                    gas = gas + self._engine_map(240.0)[..., None] * gxp.asarray(
                        np.array([1.0, 0.82, 0.58], np.float32)) * (0.50 * eb)

            elif tr == "core_rays":
                if ea > 0.002 and ca is not None:
                    # MEASURED on v3: at sigma 66 / gain 4.0 the "flashing" stab lifted the
                    # frame mean 0.72% against the rays' 6.38% - visible (243 pixels at full
                    # white) but nine times quieter than the other word, in a clip named after
                    # it. Widened and lifted to land, and NOT to the rays' level: the two
                    # words are meant to read differently, one local stab and one frame-wide
                    # burst, and matching them would collapse the whole point.
                    m = self._radial_map(ca["x"], ca["y"], 88.0)
                    gas = gas + m[..., None] * gxp.asarray(
                        np.array([1.0, 0.95, 0.86], np.float32)) * (5.2 * ea)
                if eb > 0.002 and cb is not None:
                    gas = gas + self._rays(cb, self._since(t, "b"))[..., None] * gxp.asarray(
                        np.array([1.0, 0.88, 0.70], np.float32)) * (1.75 * eb)
                    metal_boost = gxp.float32(0.35 * eb)

        if float(metal_boost) > 0.0:
            rgb = gxp.clip(rgb * (1.0 + metal_boost) + metal_boost * 0.22, 0.0, 1.0)
        out = gas * (1.0 - a) + rgb * a
        out = gxp.clip(out, 0.0, 1.0) * 255.0 + 0.5
        return asnumpy(out.astype(gxp.uint8))

    # ---- helpers ----
    def _since(self, t, kind):
        s, best = self.sched, 1e9
        for f in s.flashes:
            if f["kind"] == kind and 0.0 <= t - f["t"] < best:
                best = t - f["t"]
        return 0.0 if best > 1e8 else best

    def _engine_map(self, sigma):
        """One falloff centred on the crank, covering the whole radial."""
        cx, cy = self.eng.notes.get("crank", (540.0, 960.0))
        return self._radial_map(cx, cy, sigma)

    def _rays(self, c, age):
        """A starburst plus an expanding shock ring, from the barrel that fired."""
        vx, vy = self._view(c["x"], c["y"])
        key = ("rays", round(vx, 1), round(vy, 1))
        if not hasattr(self, "_ra"):
            self._ra = {}
        if key not in self._ra:
            ys = gxp.arange(self.H, dtype=gxp.float32).reshape(-1, 1) / self.k
            xs = gxp.arange(self.W, dtype=gxp.float32).reshape(1, -1) / self.k
            dx, dy = xs - vx, ys - vy
            r = gxp.sqrt(dx * dx + dy * dy) + 1e-3
            ang = gxp.arctan2(dy, dx)
            spokes = gxp.abs(gxp.cos(6.0 * ang)) ** 8
            self._ra[key] = (r.astype(gxp.float32), spokes.astype(gxp.float32))
        r, spokes = self._ra[key]
        z = self._zoom
        star = spokes * gxp.exp(-r / (330.0 * z))
        ring_r = (90.0 + 1750.0 * age) * z
        ring = gxp.exp(-((r - ring_r) ** 2) / (2.0 * (46.0 * z) ** 2)) * math.exp(-age / 0.22)
        return gxp.clip(star * 0.85 + ring * 0.9, 0.0, 4.0).astype(gxp.float32)


TREATMENTS = ("strobe_flood", "chamber_engine", "core_rays")
