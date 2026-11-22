# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""stars.py - the parallax star field. DRAWN, not solved, and it must be described that way.

There is no exterior flow past the object in this project - the solve is the jet and only the
jet - so nothing here is coupled to the physics and nothing here should be read as a measurement.
What it IS: a scene schedule (a speed and a heading) drawn as motion-blurred points, which is the
same class of legitimate input as `Wind_Tunnel`'s ruler or its rolling road - the scene chooses
how fast the subject is going, and the picture shows that choice.

THREE THINGS MAKE IT READ AS SPEED RATHER THAN AS CONFETTI

  1. **Parallax.** Layers move at different multiples of one speed, so near stars sweep and far
     stars crawl. One speed for all of them reads as a flat texture sliding, i.e. as a background
     image being panned, which is exactly what it must not look like.
  2. **The streak is a real exposure.** Each star is splatted several times along the distance it
     travels in ONE FRAME, so the smear length is `speed / fps` and it grows when the scene
     accelerates without anything being tuned. A fixed-length dash is the giveaway.
  3. **Direction is the heading, and it turns with it.** The whole point of the clip is the
     attitude changing, so the streaks have to rotate with the object or the two elements
     disagree about which way the subject is going.

The field lives on a TORUS one margin larger than the frame on every side, so a star that leaves
by the left re-enters at the right with no popping, and a streak that straddles the seam is drawn
in both places. Wrapping rather than respawning also means a rotating heading cannot slowly
evacuate one corner of the frame.
"""
from __future__ import annotations

import numpy as np


def _splat(buf, x, y, w):
    """Bilinear scatter-add of `w` at float positions (x, y), wrapping on the torus."""
    H, W = buf.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    for oy in (0, 1):
        wy = fy if oy else (1.0 - fy)
        yy = (y0 + oy) % H
        for ox in (0, 1):
            wx = fx if ox else (1.0 - fx)
            np.add.at(buf, (yy, (x0 + ox) % W), w * wx * wy)


class StarField:
    """`layers` is a tuple of (speed multiplier, size, brightness), nearest layer first."""

    def __init__(self, w, h, n, layers, seed=7, margin=96):
        self.w, self.h, self.m = int(w), int(h), int(margin)
        self.W, self.H = self.w + 2 * self.m, self.h + 2 * self.m
        rng = np.random.default_rng(int(seed))
        lay = np.asarray(layers, np.float32)
        # Weighted toward the FAR layers: a sky with as many near stars as far ones reads as a
        # snowstorm. Weight goes as 1/mult, i.e. the slow layers are the populous ones.
        p = 1.0 / np.maximum(lay[:, 0], 1e-3)
        idx = rng.choice(len(lay), size=int(n), p=p / p.sum())
        self.mult = lay[idx, 0]
        size = lay[idx, 1]
        self.bri = (lay[idx, 2] * size * size
                    * rng.uniform(0.35, 1.0, int(n)).astype(np.float32) ** 2)
        self.x = rng.uniform(0, self.W, int(n)).astype(np.float32)
        self.y = rng.uniform(0, self.H, int(n)).astype(np.float32)

    def advance(self, dx, dy):
        """Move one frame's worth. `dx`/`dy` are the displacement of a multiplier-1.0 star."""
        self.x = (self.x + dx * self.mult) % self.W
        self.y = (self.y + dy * self.mult) % self.H

    # A single bilinear splat of weight 1 lands 0.25 on the centre pixel after the two tent
    # passes below, so this is what makes `bri = 1` mean "peaks at white".
    _UNIT = 4.0

    def render(self, dx, dy, streak=1.0, gain=1.0, rgb=(1.0, 1.0, 1.06)):
        """(h, w, 3) float32 of added light. The smear is this frame's OWN travel.

        THE STREAK IS NOT ENERGY-CONSERVING, ON PURPOSE. A true exposure spreads a star's light
        over its whole path, so at 45 px of travel each pixel gets a fortieth of it and the fast
        layer - the one carrying the sense of speed - goes invisible exactly when the shot speeds
        up. Dividing by `k**0.35` instead keeps a long streak visible while still dimming it
        relative to a short one, so acceleration still reads as "brighter, longer lines" rather
        than as the sky fading out. State it as the stylisation it is.
        """
        buf = np.zeros((self.H, self.W), np.float32)
        # ONE SAMPLE PER PIXEL OF THE LONGEST STREAK, or the trail comes out as a dashed line.
        # At the closing speed the fastest layer travels 72 px in a frame; at 28 samples that is
        # 2.7 px between splats, and with a 1 px tent on top the gaps are plainly visible - the
        # sky reads as morse code rather than as motion blur. The cap is cost, not physics.
        span = float(np.hypot(dx, dy) * streak * self.mult.max())
        k = int(np.clip(round(span) + 1, 1, 96))
        w = self.bri * (self._UNIT * float(gain)) / (k ** 0.35)
        for s in range(k):
            f = 0.0 if k == 1 else s / (k - 1)
            _splat(buf, (self.x - dx * streak * self.mult * f) % self.W,
                   (self.y - dy * streak * self.mult * f) % self.H, w)
        # A 3-tap separable tent, so a star is a soft point rather than a single hard pixel.
        # Cheaper than a gaussian and indistinguishable at this radius.
        buf = 0.25 * (np.roll(buf, 1, 0) + np.roll(buf, -1, 0)) + 0.5 * buf
        buf = 0.25 * (np.roll(buf, 1, 1) + np.roll(buf, -1, 1)) + 0.5 * buf
        core = buf[self.m:self.m + self.h, self.m:self.m + self.w]
        return np.clip(core[..., None] * np.asarray(rgb, np.float32) * 255.0, 0.0, 255.0)
