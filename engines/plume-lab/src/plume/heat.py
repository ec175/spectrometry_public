# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""heat.py - the atmospheric-friction sheath. DRAWN, and it has to be said out loud.

There is no oncoming freestream anywhere in this project: the solve is an axisymmetric jet
firing into still ambient, so there is no bow shock on the body, no stagnation heating and
nothing that could produce this. It is a **drawn stand-in** for aerodynamic heating.
Wind_Tunnel could model the real thing (`wt\\cns.py` captures a detached bow shock and carries a
real temperature); this engine structurally cannot, because its whole domain is the plume. So
this is art, it is labelled as art, and nothing downstream should read a temperature off it.

THE THREE THINGS THE FIRST VERSION GOT WRONG, and what replaced them
--------------------------------------------------------------------
Clancy on v1: *"too concentrated right in the cow's face. It should be larger, somewhat related
to the shape of the cow's body, be much less dense, and much more detailed (it just looks like a
slightly changing image right now)."* Each of those is a different defect with a different fix:

1. **CONCENTRATED / NOT RELATED TO THE BODY.** v1 emitted everything from one circular cap on
   the axis, so it was a torch bolted to the nose. v2 emits from the **body's own leading
   profile**: `StageObject` scans the silhouette's alpha and reports, for every lateral offset,
   how far forward the surface is. Filaments start a fixed stand-off ahead of THAT, so the
   sheath is the shape of the animal - it bulges at the shoulder, pinches at the neck, and
   spans the whole body rather than the head. The stagnation glow is a **ridge that follows the
   same profile**, not a blob, which is also what a real bow shock does.
2. **TOO DENSE.** Fewer filaments, dimmer, and weighted so the flanks thin out - the reference's
   trail has black between its hairs and v1 had none.
3. **"A SLIGHTLY CHANGING IMAGE".** The real fault, and the interesting one. v1 baked the
   filament geometry once and modulated only its BRIGHTNESS with a travelling sine, so the
   shapes never moved - it was a still photograph with the lights flickering. v2 recomputes the
   lateral position every frame from **two incommensurate travelling waves at different
   wavelengths**, so the filaments genuinely writhe, cross and separate. That costs one more
   sine over the live set and is the whole difference between "lit" and "alive".

WHY IT IS DRAWN IN A LOCAL FRAME AND THEN WARPED
------------------------------------------------
Splatting filaments directly into the output frame needs samples spaced closer than a pixel or
the trail comes out as **speckle** - the identical under-sampling bug the star streaks had. Drawn
in its own axis-aligned buffer, with the trail along +x, every filament gets exactly ONE sample
per column by construction, at a cost independent of the trail's length. The finished buffer is
warped into the frame by the same affine gather the field uses, which is also what makes the
whole sheath rotate rigidly with the body for free.
"""
from __future__ import annotations

import numpy as np

# distance back along the trail -> colour: white-hot at the surface, straw, orange, then the
# brown that ablated material cools into before it fades out.
_RAMP = np.array([
    [0.00, 255, 250, 232],
    [0.05, 255, 236, 190],
    [0.16, 252, 198, 122],
    [0.36, 226, 148,  70],
    [0.58, 166,  99,  46],
    [0.80,  98,  58,  27],
    [1.00,  38,  22,  10],
], np.float32)


def _ramp(s):
    return np.stack([np.interp(s, _RAMP[:, 0], _RAMP[:, k]) for k in (1, 2, 3)], -1) / 255.0


class HeatShield:
    """A leading-edge sheath and its swept trail, drawn in the body's own frame.

    `profile` is (v, forward) - for each lateral offset `v` in pixels from the pivot, how far
    FORWARD along the heading the body's surface is. `StageObject.lead_profile` measures it off
    the subject's own alpha, which is what ties the effect to the shape of the animal.
    """

    def __init__(self, n, profile, standoff=26.0, length=620.0, spread=1.5, thick=34.0,
                 seed=5, rate=1.1, gain=1.0, writhe=1.0):
        self.n = int(n)
        self.length, self.spread = float(length), float(spread)
        self.rate, self.gain, self.writhe = float(rate), float(gain), float(writhe)
        v, fwd = profile
        rng = np.random.default_rng(int(seed))

        vmax = float(np.max(np.abs(v))) + 6.0
        self.half = int(round(vmax * spread + 46.0 + 24.0))
        self.hl = 2 * self.half
        self.fwd = int(round(float(np.max(fwd)) + standoff + 74.0))
        self.wl = self.fwd + int(round(length)) + 8

        # ---- filaments, seeded ACROSS THE BODY rather than on one circular cap
        # `cos` puts a few more at the middle than at the tips without leaving the flanks bare -
        # a uniform draw makes the sheath read as a rectangle, and a strongly peaked one puts it
        # back on the nose, which is the complaint this whole rewrite exists for.
        q = rng.uniform(-1.0, 1.0, self.n).astype(np.float32)
        self.v0 = (np.sign(q) * np.abs(q) ** 0.82 * vmax).astype(np.float32)
        d0 = np.interp(self.v0, v, fwd).astype(np.float32)
        self.flen = (rng.uniform(0.30, 1.0, self.n) ** 0.62).astype(np.float32)
        self.bri = (rng.uniform(0.18, 1.0, self.n) ** 1.7).astype(np.float32)
        # thin the flanks so the sheath has an edge instead of a wall
        self.bri *= (1.0 - (np.abs(self.v0) / vmax) ** 2.6) ** 0.5 + 0.06
        self.ph = rng.uniform(0, 2 * np.pi, self.n).astype(np.float32)
        self.ph2 = rng.uniform(0, 2 * np.pi, self.n).astype(np.float32)
        self.k1 = rng.uniform(1.4, 3.2, self.n).astype(np.float32)
        self.k2 = rng.uniform(4.5, 9.0, self.n).astype(np.float32)
        self.a1 = rng.uniform(0.5, 1.0, self.n).astype(np.float32) * thick
        self.a2 = rng.uniform(0.2, 0.6, self.n).astype(np.float32) * thick * 0.45
        self.sf = rng.uniform(2.0, 6.5, self.n).astype(np.float32)

        # ---- the live (filament, column) set, compacted once
        j = np.arange(self.wl, dtype=np.float32)[None, :]
        start = (self.fwd - (d0 + standoff))[:, None]          # column where the filament begins
        s = (j - start) / max(length, 1.0)
        live = (s >= 0.0) & (s <= self.flen[:, None])
        fi, ci = np.nonzero(live)
        self.fi = fi
        self.ic = ci.astype(np.int32)
        sv = s[fi, ci].astype(np.float32)
        self.sv = sv
        sc = np.clip(sv, 0.0, 1.0)
        # base lateral position: the cone opens with distance back
        self.y0 = (self.half + self.v0[fi] * (1.0 + (spread - 1.0) * sc)).astype(np.float32)
        onset = 0.25 + 0.75 * np.clip(sv / 0.04, 0.0, 1.0)
        self.amp = (((1.0 - sc) ** 1.30) * (0.20 + 0.80 * np.exp(-2.6 * sc))
                    * self.bri[fi] * onset).astype(np.float32)
        self.crgb = _ramp(sc).astype(np.float32)
        # per-element copies of the wave constants, so a frame is pure arithmetic
        for a in ("k1", "k2", "a1", "a2", "ph", "ph2", "sf"):
            setattr(self, a + "_e", getattr(self, a)[fi])

        # ---- the stagnation glow: a RIDGE following the leading profile, not a blob
        yy, xx = np.mgrid[0:self.hl, 0:self.wl].astype(np.float32)
        vv = yy - self.half
        ridge = self.fwd - (np.interp(vv[:, 0], v, fwd) + standoff)
        a = (xx - ridge[:, None]) / 30.0
        b = np.clip(np.abs(vv) / (vmax * 1.06), 0.0, 3.0)
        g = np.exp(-a * a * 1.15) * np.exp(-(b ** 3.2) * 1.5)
        grain = rng.uniform(0.0, 1.0, (self.hl, self.wl)).astype(np.float32)
        grain = 0.25 * (grain + np.roll(grain, 1, 0) + np.roll(grain, 1, 1)
                        + np.roll(grain, (1, 1), (0, 1)))
        self.sheath = (g * (0.55 + 0.70 * grain)).astype(np.float32)

    # ------------------------------------------------------------------ per frame
    def local(self, tc):
        """(hl, wl, 3) float32 - the sheath in its own frame, trail along +x."""
        t = float(tc)
        buf = np.zeros((self.hl, self.wl, 3), np.float32)
        # TWO travelling waves at different wavelengths, recomputed every frame. This is what
        # makes the filaments move rather than merely flicker - see the header.
        y = (self.y0
             + self.a1_e * np.sin(self.k1_e * self.sv * 6.0 - 2.1 * self.writhe * t + self.ph_e)
             + self.a2_e * np.sin(self.k2_e * self.sv * 6.0 + 3.7 * self.writhe * t + self.ph2_e))
        w = self.amp * (0.42 + 0.58 * np.sin(self.sf_e * self.sv * 6.0
                                             - self.rate * 6.0 * t + self.ph_e))
        ok = (y >= 0) & (y < self.hl - 1) & (w > 0)
        y = y[ok]
        iy = y.astype(np.int32)
        fy = (y - iy).astype(np.float32)
        ic = self.ic[ok]
        ww = w[ok]
        crgb = self.crgb[ok]
        for off, wt in ((0, 1.0 - fy), (1, fy)):
            v3 = (ww * wt)[:, None] * crgb
            idx = (iy + off, ic)
            for ch in range(3):
                np.add.at(buf[..., ch], idx, v3[:, ch])
        buf *= 0.85 * self.gain
        # Warm, not neutral. A near-white sheath colour is white only where it is BRIGHT; over
        # most of its area it is dim, and a dim near-white on black is grey - it read as smoke
        # in front of the animal rather than as compressed gas glowing.
        buf += self.sheath[..., None] * np.array([1.00, 0.80, 0.55], np.float32) \
            * (0.62 * self.gain * (0.90 + 0.10 * np.sin(9.1 * t)))
        return buf

    def place(self, deg, cx, cy):
        """(origin, back, across) in frame pixels - what the warp needs. Origin is the pivot."""
        th = np.radians(float(deg))
        f = (float(np.cos(th)), -float(np.sin(th)))     # forward (y is down on screen)
        return (cx, cy), (-f[0], -f[1]), (float(np.sin(th)), float(np.cos(th)))
