# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""sky.py - the ascent background: blue sky and cloud, thinning to a curved limb over black.

DRAWN, like `stars.py` and `heat.py`, and for the same reason: this engine's domain is the jet
and nothing else. What makes it worth building rather than crossfading two images is that every
element is a function of **one derived scalar, the altitude** - and the altitude is derived from
the subject's own motion, not from the clip's clock:

    dh/dt = v(t) * (sin(theta(t)) * (1 - floor) + floor)

i.e. the vertical component of the velocity the star field is already using, with `theta` the
heading the subject is already pitching through. So the vehicle climbs hard while it is vertical,
and its climb rate dies as it pitches over - a gravity turn. The background therefore changes
because of where the subject is and what it is doing, which is what a launch shot needs and what
a timed crossfade can never look like. The small `floor` is the residual climb a real vehicle
keeps after pitchover; without it the sky freezes for the last three seconds.

FOUR THINGS HAPPEN AT ONCE, AND THEY ARE ALL THE SAME EQUATION
---------------------------------------------------------------
The sky's brightness above the horizon is `A * exp(-d / L)`, where `d` is height above the
horizon line. Only `A` and `L` move with altitude, and between them they do the whole job:

  * at sea level `L` is longer than the frame, so the WHOLE picture is blue;
  * as `L` collapses, the blue retreats toward the horizon and the top of the frame goes black -
    which is exactly what a sky does as the air column above you thins;
  * `A_sky` decays fast (scale height ~14 km) so the diffuse sky dies, while a separate
    `A_limb` GROWS with altitude - because a limb is what is left when you are above the air
    looking through it edge-on, and it does not exist while you are inside it;
  * stars fade in on `1 - sky`, so they arrive because the sky left, not on a cue.

The horizon itself rises into frame from below and bows upward at the centre, its sagitta growing
with altitude - the curvature you can only see once you are high enough.

Clouds are the NEAREST PARALLAX LAYER, not a separate idea: they drift with the same velocity
vector the stars use, just far faster, and they shrink and fade with altitude on a cumulus scale
height. So early in the clip they rush past (mostly downward, because the vehicle is climbing)
and later they slide sideways (because it is going downrange), with no separate schedule.
"""
from __future__ import annotations

import numpy as np

from .gpu import xp

# Sky colour by height above the horizon, as a fraction of the local scale length.
_SKY = np.array([
    [0.00, 196, 224, 248],    # the pale band right on the horizon
    [0.12, 150, 196, 240],
    [0.30,  96, 156, 226],
    [0.55,  52, 112, 202],
    [0.80,  28,  74, 168],
    [1.10,  16,  42, 116],
    [1.60,   8,  18,  58],
    [2.40,   0,   0,   0],
], np.float32)

# The limb, once there is one: hotter and tighter than the diffuse sky, and it ends in violet.
_LIMB = np.array([
    [0.00, 214, 240, 255],
    [0.18, 150, 206, 252],
    [0.42,  70, 150, 240],
    [0.68,  34,  86, 206],
    [1.00,  26,  40, 140],
    [1.45,  16,  14,  62],
    [2.20,   0,   0,   0],
], np.float32)

_EARTH = np.array([14, 30, 58], np.float32)      # the ground, seen from above and unlit


def _lut(table, n=256, span=2.6):
    x = np.linspace(0.0, span, n)
    return np.stack([np.interp(x, table[:, 0], table[:, k]) for k in (1, 2, 3)], -1)


class Sky:
    """The ascent background. One call per frame; everything is a function of `alt_km`."""

    def __init__(self, w, h, seed=3, clouds=8, cloud_scale_km=6.5, sky_scale_km=14.0,
                 limb_from_km=30.0, horizon_km=30.0, lut_n=256, lut_span=2.6):
        self.w, self.h = int(w), int(h)
        self.cloud_km, self.sky_km = float(cloud_scale_km), float(sky_scale_km)
        self.limb_km, self.hor_km = float(limb_from_km), float(horizon_km)
        self.span = float(lut_span)
        self.sky_lut = xp.asarray(_lut(_SKY, lut_n, lut_span).astype(np.float32))
        self.limb_lut = xp.asarray(_lut(_LIMB, lut_n, lut_span).astype(np.float32))
        self.n = lut_n
        Y, X = np.meshgrid(np.arange(self.h, dtype=np.float32),
                           np.arange(self.w, dtype=np.float32), indexing="ij")
        self.X, self.Y = xp.asarray(X), xp.asarray(Y)
        # (x - cx)^2 normalised, for the horizon's bow
        self.bow = xp.asarray(((X - self.w / 2.0) / (self.w / 2.0)) ** 2)

        # ---- clouds: a handful of puffy silhouettes, each a smooth union of lobes.
        # They live on a torus larger than the frame so one can leave and another arrive without
        # anything popping, and they are the NEAREST parallax layer - same velocity vector as
        # the stars, roughly six times the rate.
        self.wc, self.hc = self.w * 2.4, self.h * 2.0
        rng = np.random.default_rng(int(seed))
        self.clouds = []
        for _ in range(int(clouds)):
            k = int(rng.integers(5, 9))
            lx = np.sort(rng.uniform(-1.0, 1.0, k)) * rng.uniform(0.85, 1.35)
            ly = -rng.uniform(0.0, 1.0, k) ** 1.4 * rng.uniform(0.35, 0.70)
            rx = rng.uniform(0.40, 0.70, k)
            ry = rx * rng.uniform(0.62, 0.95, k)
            # THE BOX IS DERIVED FROM THE LOBES, not assumed. The first cut used a fixed +-2.2
            # units while the lobe centres alone reached +-3.4, so every wide cloud was sliced
            # off square by its own bounding box - long hard vertical seams across the sky.
            hx = float(np.max(np.abs(lx) + 2.3 * rx))
            hy = float(np.max(np.abs(ly) + 2.3 * ry))
            self.clouds.append(dict(
                lx=lx.astype(np.float32), ly=ly.astype(np.float32),
                rx=rx.astype(np.float32), ry=ry.astype(np.float32), hx=hx, hy=hy,
                u0=float(rng.uniform(0.0, self.wc)), v0=float(rng.uniform(0.0, self.hc)),
                size=float(rng.uniform(85.0, 195.0)), bri=float(rng.uniform(0.80, 1.0))))

    # ------------------------------------------------------------------ altitude -> geometry
    def geom(self, alt_km):
        """Everything the frame needs, derived from altitude alone."""
        a = max(0.0, float(alt_km))
        f = 1.0 - np.exp(-a / self.hor_km)                 # 0 on the pad, ->1 in space
        y_h = self.h * (1.55 - 0.62 * f)                   # horizon, below frame at first
        sag = 215.0 * f                                    # how far it bows up at the centre
        l_sky = 2600.0 * np.exp(-a / 17.0) + 30.0          # blue reach above the horizon, px
        a_sky = np.exp(-a / self.sky_km)                   # the diffuse column
        # A limb cannot exist while you are inside the air it is made of.
        a_limb = 0.98 * (1.0 - np.exp(-a / self.limb_km)) ** 1.5
        l_limb = 30.0 + 46.0 * np.exp(-a / 60.0)
        cloud = np.exp(-a / self.cloud_km)
        return dict(f=f, y_h=y_h, sag=sag, l_sky=l_sky, a_sky=a_sky,
                    a_limb=a_limb, l_limb=l_limb, cloud=cloud)

    def star_fade(self, alt_km):
        """Stars are invisible in daylight and arrive as the sky leaves. Not a cue - a result."""
        g = self.geom(alt_km)
        return float(np.clip(1.0 - 2.4 * g["a_sky"], 0.0, 1.0)) ** 0.85

    # ------------------------------------------------------------------ per frame
    def render(self, alt_km, drift=(0.0, 0.0)):
        """(h, w, 3) float32 background in 0..255."""
        g = self.geom(alt_km)
        # the horizon line, bowed upward at the centre: the curvature you can only see from up
        hor = g["y_h"] - g["sag"] * (1.0 - self.bow)
        d = hor - self.Y                                   # px above the horizon
        above = d > 0
        du = xp.clip(d, 0.0, None)

        img = xp.zeros((self.h, self.w, 3), xp.float32)
        if g["a_sky"] > 1e-3:
            t = xp.clip(du / g["l_sky"] / self.span, 0.0, 1.0)
            i = (t * (self.n - 1)).astype(xp.int32)
            img = img + self.sky_lut[i] * (g["a_sky"] * above[..., None])
        if g["a_limb"] > 1e-3:
            t = xp.clip(du / g["l_limb"] / self.span, 0.0, 1.0)
            i = (t * (self.n - 1)).astype(xp.int32)
            img = img + self.limb_lut[i] * (g["a_limb"] * above[..., None])

        # the Earth below the horizon, with the atmosphere piled brightest right on the edge
        if bool((~above).any()):
            below = xp.clip(-d, 0.0, None)
            rim = xp.exp(-below / 26.0)
            earth = xp.asarray(_EARTH)[None, None, :] * (0.35 + 0.65 * g["f"])
            img = xp.where(above[..., None], img,
                           earth + xp.asarray(_LIMB[0, 1:]) * (rim[..., None] * 0.55))

        if g["cloud"] > 0.02:
            img = self._clouds(img, g, drift)
        img = xp.clip(img, 0.0, 255.0)
        # WHAT THE SKY HIDES. Stars are behind everything here - behind the planet, behind the
        # limb, behind a cloud - and adding them to the finished background put them on top of
        # the Earth, which is the one thing in the frame that is unmistakably solid. `block` is
        # 1 below the horizon and rises with the background's own brightness, so a star is also
        # washed out by the bright part of the limb and by daylight, without a second schedule.
        self.block = xp.clip(xp.maximum((~above).astype(xp.float32),
                                        img.max(axis=2) / 105.0), 0.0, 1.0)
        return img

    def _clouds(self, img, g, drift):
        """Puffy silhouettes, drawn only inside their own bounding boxes.

        Evaluating eight clouds of seven lobes over the whole frame is 56 full-frame gaussians a
        frame for something that covers a few percent of it. Each cloud gets a box instead.
        """
        # They shrink as the vehicle climbs past them, and settle toward the horizon.
        s = float(np.exp(-g["f"] * 1.25))
        sink = float(g["f"])
        for c in self.clouds:
            size = c["size"] * s
            if size < 8.0:
                continue
            cx = (c["u0"] + drift[0]) % self.wc - self.w * 0.7
            py = (c["v0"] + drift[1]) % self.hc - self.h * 0.5
            cy = py * (1.0 - 0.55 * sink) + self.h * (0.72 * sink)
            x0 = int(max(0, cx - c["hx"] * size))
            x1 = int(min(self.w, cx + c["hx"] * size))
            y0 = int(max(0, cy - c["hy"] * size))
            y1 = int(min(self.h, cy + 1.1 * size))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            xs = (self.X[y0:y1, x0:x1] - cx) / size
            ys = (self.Y[y0:y1, x0:x1] - cy) / size
            # A METABALL FIELD, thresholded - not a product of soft blobs. The soft version had
            # no silhouette at all: it read as fog. Summing the lobes and cutting the sum at a
            # level gives a cloud an EDGE, which is most of what makes it look like one, and
            # overlapping lobes still merge instead of doubling in brightness.
            fld = xp.zeros_like(xs)
            for lx, ly, rx, ry in zip(c["lx"], c["ly"], c["rx"], c["ry"]):
                q = ((xs - float(lx)) / float(rx)) ** 2 + ((ys - float(ly)) / float(ry)) ** 2
                fld = fld + xp.exp(-q * 1.6)
            u = xp.clip((fld - 0.44) / 0.34, 0.0, 1.0)
            m = u * u * (3.0 - 2.0 * u)
            m = m * xp.clip((0.62 - ys) / 0.30, 0.0, 1.0)      # cumulus sit on a flat base
            m = m * (c["bri"] * g["cloud"])
            # TWO shading cues. Lit on top and shadowed underneath is the one that says
            # "cumulus" - but it must be measured against the cloud's OWN height, not against
            # its bounding box: keying it to the box put almost the whole body on the clip floor
            # and every cloud came out a flat grey cut-out. The second is thickness: where more
            # lobes overlap the cloud is deeper, so it is brighter, and that is what gives it
            # internal form instead of a silhouette filled with one colour.
            shade = xp.clip(1.00 - (ys + 0.45) * 0.36, 0.56, 1.02)
            shade = shade * xp.clip(0.74 + 0.30 * fld, 0.74, 1.16)
            col = xp.stack([m * shade * 252.0, m * shade * 252.0, m * shade * 255.0], -1)
            sub = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = sub * (1.0 - m[..., None]) + col
        return img
