"""panel.py - the hexagon faces as a SURFACE, with the light source underneath.

The read Ethan asked for: the holofoil is not painted on the hexagons, it is a glowing sheet
UNDER them, and the hexagon faces are an opaque-ish surface laid on top. What you see is the
light escaping at the seams.

That is a compositing job, not a drawing job, and it needs two things the element splatter
cannot provide:

  transmission   a STATIC per-pixel map: ~1 along the seams, ~`face_floor` in the middle of a
                 face. Built once by rasterising every lattice edge and taking a Euclidean
                 distance transform, so the falloff away from a seam is a true distance and
                 the six seams around a cell meet correctly at the vertices.

  a SOURCE field the colour under the surface. Evaluated on a coarse grid and upsampled,
                 because it is diffuse by definition - it is behind frosted panels. At 1/8
                 resolution it costs ~32k points a frame instead of 2M, and nothing is lost
                 that the panels would have let through anyway.

The parallax that sells the two layers is free: the panels are fixed and the source drifts
underneath them. No offset or fake displacement is needed - the eye reads a moving colour
field behind a static grid as depth on its own.

`bevel` adds a thin brightening just INSIDE the seam. Real panels catch light on their chamfer,
and without it the seams read as glowing wires lying on top rather than as gaps between plates.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

_CACHE = {}


def transmission(lat, width, height, scale, seam_sigma=3.4, face_floor=0.035,
                 bevel=0.45, bevel_at=7.0, bevel_sigma=2.6, seam_power=1.0):
    """(H,W) float32 in [0,1]: how much of the underlying source escapes at each pixel.

    Cached per (lattice identity, output size, look parameters) - the distance transform over
    a 1080x1920 frame costs ~0.4 s and must not run per frame.
    """
    key = (id(lat), width, height, round(seam_sigma, 3), round(face_floor, 4),
           round(bevel, 3), round(bevel_at, 2), round(bevel_sigma, 2), round(seam_power, 3))
    if key in _CACHE:
        return _CACHE[key]

    img = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(img)
    a = lat.sites[lat.edges[:, 0]] * scale
    b = lat.sites[lat.edges[:, 1]] * scale
    for (x0, y0), (x1, y1) in zip(a, b):
        d.line([float(x0), float(y0), float(x1), float(y1)], fill=255, width=1)
    seam = np.asarray(img, np.uint8) > 0

    dist = distance_transform_edt(~seam).astype(np.float32)
    s = max(0.6, seam_sigma * scale)
    t = np.exp(-(dist ** 2) / (2.0 * s * s))
    if seam_power != 1.0:
        t = t ** seam_power
    if bevel > 1e-4:
        # a ring of extra transmission a few px inside the seam - the chamfer
        bs = max(0.6, bevel_sigma * scale)
        t = t + bevel * np.exp(-((dist - bevel_at * scale) ** 2) / (2.0 * bs * bs))
    t = face_floor + (1.0 - face_floor) * np.clip(t, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    _CACHE[key] = t
    return t


class SourceGrid:
    """The coarse grid the under-surface colour is evaluated on, plus its upsampler."""

    def __init__(self, width, height, div=8):
        # `div` MUST be a power of two: `expand` doubles its way up, so a div of 6 would
        # double twice (6 -> 3 -> 1) and land at 4x, leaving the source two-thirds size in
        # the top-left corner. That rendered as a second, smaller copy of the subject.
        d = 1
        while d * 2 <= div and width % (d * 2) == 0 and height % (d * 2) == 0:
            d *= 2
        self.div = div = d
        self.w, self.h = width // div, height // div
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        # sample at cell CENTRES, in world (reference-frame) coordinates
        self.P = np.stack([((xx.ravel() + 0.5) * div) * (1080.0 / width),
                           ((yy.ravel() + 0.5) * div) * (1920.0 / height)], 1)
        self.width, self.height = width, height

    def expand(self, lo, smooth=1.0):
        """(h,w,3) -> (H,W,3), smoothly.

        The blur has to come AFTER the replication. Blurring the low-res image first and then
        replicating leaves hard `div`-sized steps - the blur smooths the signal but the
        replication re-introduces the edges, and the source layer came out visibly blocky at
        8 px. Cascading 2x doublings with a small blur at each rung costs about a third of one
        full-resolution blur and lands smoother.
        """
        img = lo.reshape(self.h, self.w, -1)
        if smooth > 0.05:
            img = gaussian_filter(img, (smooth, smooth, 0), mode="nearest")
        d = self.div
        while d > 1:
            img = np.repeat(np.repeat(img, 2, 0), 2, 1)
            img = gaussian_filter(img, (1.0, 1.0, 0), mode="nearest")
            d //= 2
        dh, dw = self.height - img.shape[0], self.width - img.shape[1]
        if dh > 0 or dw > 0:
            img = np.pad(img, ((0, max(dh, 0)), (0, max(dw, 0)), (0, 0)), mode="edge")
        return img[:self.height, :self.width]
