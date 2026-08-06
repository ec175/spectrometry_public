"""draw.py - additive point accumulation with a persistence buffer.

Forty thousand particles per frame is far past what PIL vector calls can draw, so nothing here is
a draw call: every point is bilinearly SPLATTED into a float accumulation buffer with one
`np.bincount`, exactly the device `Wind_Tunnel\\wt\\streaks.py` uses for its streak tails and
`Field_Lines` uses for its density channel. It is genuinely additive, which matters - the bright
filaments of an attractor are bright *because* many particles are there, and an overwriting
rasteriser would show a uniform sheet instead.

A PERSISTENCE buffer sits on top (`trail`), decayed a little each frame. This is the same idea as
the Oscilloscope's phosphor, and it does the same job for a different reason: a single frame of a
particle swarm is a dusting of dots, and only the accumulation over ~10 frames reads as a surface.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Colour ramps. Every one maps a MEASURED quantity, so they are built to be monotone in
# luminance - a viewer must be able to read "more" from "brighter" without a legend.
PALETTES = {
    "ember":    [(0.00, (6, 10, 34)), (0.30, (74, 30, 96)), (0.60, (214, 76, 62)),
                 (0.84, (255, 168, 66)), (1.00, (255, 246, 214))],
    "ice":      [(0.00, (4, 12, 30)), (0.35, (18, 78, 140)), (0.66, (66, 170, 224)),
                 (0.86, (150, 226, 246)), (1.00, (245, 253, 255))],
    "flare":    [(0.00, (8, 16, 46)), (0.42, (28, 96, 176)), (0.68, (150, 96, 210)),
                 (0.86, (255, 96, 128)), (1.00, (255, 236, 180))],
    "aurora":   [(0.00, (4, 20, 26)), (0.34, (16, 116, 108)), (0.62, (56, 208, 150)),
                 (0.84, (168, 245, 170)), (1.00, (238, 255, 236))],
    "spectrum": [(0.00, (30, 12, 74)), (0.26, (30, 96, 200)), (0.50, (36, 196, 176)),
                 (0.72, (214, 196, 62)), (1.00, (255, 128, 92))],
}


def ramp(name, n=256):
    stops = PALETTES[name]
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], float)
    u = np.linspace(0, 1, n)
    return np.stack([np.interp(u, ts, cs[:, c]) for c in range(3)], axis=1)


class Canvas:
    def __init__(self, w, h, palette="ember", trail=0.86, sigma=1.05,
                 glow=(0.42, 7.0), glow2=(0.20, 30.0), exposure=1.0):
        self.w, self.h = w, h
        self.lut = ramp(palette)
        self.acc = np.zeros((h, w, 3), np.float32)     # persistence buffer
        self.trail = trail
        self.sigma = sigma
        self.glow, self.glow2 = glow, glow2
        self.exposure = exposure

    def decay(self):
        self.acc *= self.trail

    def splat(self, P, u, z=None, gain=1.0, size=1.0):
        """P (N,2) in normalised [-1,1]-ish coords, u (N,) the colour key in [0,1]."""
        if len(P) == 0:
            return
        s = 0.5 * min(self.w, self.h) * 0.94
        x = P[:, 0] * s + self.w * 0.5
        y = P[:, 1] * s + self.h * 0.5
        m = (x > -2) & (x < self.w + 1) & (y > -2) & (y < self.h + 1)
        if not m.any():
            return
        x, y, u = x[m], y[m], np.clip(u[m], 0, 1)
        col = self.lut[(u * 255).astype(np.int32)]
        w = np.full(len(x), gain, np.float32)
        if z is not None:
            # nearer particles are brighter: the one depth cue that survives additive blending
            zz = z[m]
            w = w * (0.55 + 0.45 * np.clip(1.0 - (zz - zz.min()) /
                                           (np.ptp(zz) + 1e-9), 0, 1)).astype(np.float32)
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - ix, y - iy
        W, H = self.w, self.h
        flat = self.acc.reshape(-1, 3)
        for dx in (0, 1):
            for dy in (0, 1):
                gx, gy = ix + dx, iy + dy
                k = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
                if not k.any():
                    continue
                ww = (((1 - fx) if dx == 0 else fx) *
                      ((1 - fy) if dy == 0 else fy) * w)[k]
                idx = gy[k] * W + gx[k]
                for c in range(3):
                    flat[:, c] += np.bincount(idx, weights=ww * col[k, c],
                                              minlength=W * H).astype(np.float32)

    def to_rgb(self):
        a = self.acc * (self.exposure / 255.0)
        # Reinhard-ish roll-off: an additive swarm has a huge dynamic range and a hard clip
        # turns every dense filament into the same flat white blob
        out = 255.0 * (a / (1.0 + a))
        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        f = np.asarray(img, np.float32)
        if self.glow[0] > 0:
            f = f + self.glow[0] * np.asarray(
                img.filter(ImageFilter.GaussianBlur(self.glow[1])), np.float32)
        if self.glow2[0] > 0:
            f = f + self.glow2[0] * np.asarray(
                img.filter(ImageFilter.GaussianBlur(self.glow2[1])), np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)
