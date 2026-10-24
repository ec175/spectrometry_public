# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""colormap.py — scalar field -> RGB, as a precomputed 256-entry LUT.

Matching the source video: its speed field is plain **jet** — deep navy in the separated wake,
cyan/green freestream, yellow/orange/red at the suction peak. That high-chroma banding is what
makes the flow structure legible at a glance on a phone, so `jet` is the default.

The LUT is generated once with numpy and uploaded to the active backend; colourising a frame is
then a single fancy-index, which is why we can afford to do it at full 1080x1920.
"""
from __future__ import annotations

import numpy as np

from .gpu import xp

# (stop, (r, g, b)) control points, linearly interpolated
_STOPS = {
    # classic MATLAB jet — the source video's palette
    "jet": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
            (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # Google turbo: same reading order as jet, perceptually far better behaved
    "turbo": [(0.00, (48, 18, 59)), (0.13, (28, 100, 220)), (0.28, (28, 200, 220)),
              (0.43, (70, 235, 135)), (0.58, (200, 240, 55)), (0.72, (255, 170, 40)),
              (0.86, (230, 70, 20)), (1.00, (122, 4, 3))],
    # ------------------------------------------------------------------ the ORCHID family
    # My note, 2026-08-17: the blue-to-red scale worked well, but I wanted the green and yellow
    # swapped out for purple and pink.
    #
    # So these are `jet` WITH ITS BOTTOM HALF UNTOUCHED. The first three stops (navy, blue, cyan)
    # and the last two (red, dark red) are jet's own values, byte for byte; only the 0.375-0.875
    # stretch - which is where jet interpolates through green and lands on yellow - is rewritten.
    # That is not tidiness for its own sake: on `tri_foil_air_fast` the three lane freestreams sit
    # at s = 0.50 / 0.30 / 0.23 of the ramp, so the two slower lanes ARE those first three stops
    # and keeping them identical is what makes this a recolour of the green rather than a new
    # palette. The GREEN in that clip is the fast lane's freestream at 0.50 and the YELLOW is its
    # suction peaks, which is exactly what the brief names.
    #
    # LUMINANCE IS THE CONSTRAINT, not hue. jet's green sits at luminance 202 against the mid
    # lane's cyan-blue at 135, so "faster = brighter" reads for free there. A deep violet at 0.50
    # would land near 97 and INVERT that - the fastest lane would read darker than the middle one.
    # Hence the 0.50 stop is a luminous orchid (143) and 0.625 a hot pink (172): the ramp still
    # climbs into the red peak, and it climbs more gently than jet did, which is the standing
    # "the green is very bright" complaint answered on the way past.
    "orchid": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
               (0.500, (185, 100, 255)), (0.625, (255, 120, 225)),
               (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # PLUM - the same swap with the two new stops in the OTHER order (pink first, then purple),
    # because I wasn't sure which ordering would read best. It puts the
    # brightest new colour on the fast freestream and darkens INTO the red peak, so the peaks
    # read as deepening rather than as brightening.
    "plum": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
             (0.500, (255, 120, 225)), (0.660, (150, 45, 235)),
             (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # IRIS - `orchid` with a deep saturated violet instead of the luminous one, i.e. the version
    # that accepts the luminance inversion above in exchange for a much stronger purple.
    "iris": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
             (0.500, (130, 10, 240)), (0.625, (255, 80, 210)),
             (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # VIOLET - the one member of the family that also gives up CYAN, so the whole ramp is a single
    # monotone hue sweep 240 -> 300 -> 360 deg. The most coherent of the four as a colour ramp and
    # the most different from what shipped: it moves the two SLOW lanes as well as the fast one.
    "violet": [(0.00, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (120, 0, 255)),
               (0.625, (255, 105, 235)), (0.875, (255, 0, 0)), (1.00, (128, 0, 0))],
    # cool "wind tunnel schlieren" alternative
    "ice": [(0.00, (4, 6, 26)), (0.25, (12, 46, 110)), (0.50, (30, 140, 190)),
            (0.75, (150, 225, 230)), (1.00, (255, 255, 255))],
    # WATER: shades of blue only - no green, no warm end at all. Built for a field that sits
    # near ZERO almost everywhere (the still_water scene): the bottom of the ramp is a deep ink
    # blue rather than black so calm water still reads as water, and the top stays a pale
    # blue-white instead of going to pure white, so a disturbance brightens without blowing out.
    "blues": [(0.00, (7, 17, 44)), (0.16, (11, 30, 72)), (0.36, (18, 60, 122)),
              (0.58, (36, 104, 172)), (0.78, (86, 156, 206)), (0.92, (152, 200, 228)),
              (1.00, (214, 236, 248))],
    # FIRE: reds only - the red counterpart of `blues`, built for the same near-zero field. Same
    # luminance ramp, hue rotated to red, but the TOP STOP STAYS RED instead of running out to a
    # pale tint the way `blues` does. That is deliberate: the fast fluid in these scenes is the
    # driven ring around each black hole, and the brief is for the red to *concentrate* there, so
    # a peak that washes to white would put the one thing the eye should read as red in the one
    # place the map has no red left. It ends on a hot ember instead.
    # FIRE, and note it is built on the OPPOSITE principle to `blues`. `blues` floors at an ink
    # blue rather than black because still water should still read as water. Here the brief is the
    # reverse (design note, 2026-07-29): outside a hole's sphere of influence the frame should be BLACK,
    # with the red concentrated at the rim. So this LUT floors at true black and stays near it for
    # the first fifth of the ramp, then climbs hard - all of the colour is spent on the top half,
    # which is where the driven ring lives. Pair it with a gamma ABOVE 1 (the scene uses 1.70) to
    # push the far field down into that black shelf; `blues` scenes want gamma below 1 instead.
    "reds": [(0.00, (0, 0, 0)), (0.10, (10, 1, 2)), (0.22, (40, 4, 6)),
             (0.38, (96, 11, 12)), (0.55, (158, 24, 18)), (0.72, (212, 52, 26)),
             (0.88, (242, 112, 56)), (1.00, (255, 178, 124))],
    "inferno": [(0.00, (0, 0, 4)), (0.25, (87, 16, 110)), (0.50, (188, 55, 84)),
                (0.75, (249, 142, 9)), (1.00, (252, 255, 164))],
    # EMBER: single-hue orange built for another project's TUNNEL skin (2026-07-31). Built like `reds`
    # (true-black floor, colour spent on the top half) but the hue band is 20-35 deg the whole
    # way so a frame passes the "one accent family" bar - inferno's purple lows fail it.
    "ember": [(0.00, (0, 0, 0)), (0.12, (18, 7, 3)), (0.30, (64, 24, 9)),
              (0.50, (140, 58, 18)), (0.70, (224, 110, 38)), (0.86, (255, 166, 88)),
              (1.00, (255, 232, 198))],
    # diverging, for vorticity (0.5 = zero vorticity -> near black)
    "vort": [(0.00, (40, 130, 255)), (0.30, (10, 30, 90)), (0.50, (3, 4, 8)),
             (0.70, (110, 25, 20)), (1.00, (255, 140, 40))],
}


def _build(name):
    stops = _STOPS[name]
    xs = np.array([s for s, _ in stops])
    cs = np.array([c for _, c in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _tone(lut8, sat, val):
    """Desaturate and/or dim a LUT. Identity at (1, 1), so nothing shipped moves by default.

    I've asked for this twice - that the colours were way too saturated on the original five, and
    that the green was too bright and needed reduced brightness against the normal wind-tunnel
    palette on `tri_shapes` (2026-08-04). Both complaints are about
    the SAME thing and neither is a filter: it is `jet` itself. Its middle is where the two
    brightest primaries overlap - green-yellow around (128, 255, 128) carries a luminance of
    ~203 against pure red's 76 - so the freestream, which is the largest area in the frame, is
    also the most luminous thing in it. A clip where every lane runs at the same speed is a
    frame that is almost entirely that colour, which is why this became unignorable here.

    TWO KNOBS, because they fix different halves of the complaint:
      - `sat` blends each entry toward its own luminance, which takes the neon out without
        moving anything up or down the ramp;
      - `val` scales luminance, which is the "reduced brightness" that was actually asked for.
    Both act on the LUT and therefore only on the FIELD. The white tracer dashes and the body
    are composited afterwards and keep their contrast, so taming the background makes the flow
    structure MORE legible rather than washing the picture out. That is also why this is done
    here and not as a post-pass over the finished frame: dimming the finished image would dim
    the streaks with it (which is the one inexactness in a saturation-preview tool not included
    here).
    """
    a = lut8.astype(np.float32)
    lum = (a * np.array([0.299, 0.587, 0.114], np.float32)).sum(1, keepdims=True)
    a = (lum + float(sat) * (a - lum)) * float(val)
    return np.clip(a, 0, 255).astype(np.uint8)


_CACHE = {}


def lut(name="jet", sat=1.0, val=1.0):
    """256x3 uint8 LUT on the active backend (cached per name AND tone)."""
    key = (name, round(float(sat), 4), round(float(val), 4))
    if key not in _CACHE:
        if name not in _STOPS:
            raise SystemExit(f"unknown cmap {name!r}; have: {', '.join(sorted(_STOPS))}")
        _CACHE[key] = xp.asarray(_tone(_build(name), sat, val))
    return _CACHE[key]


def apply(field01, name="jet", sat=1.0, val=1.0):
    """Normalised field in [0,1] (any shape) -> uint8 RGB (…, 3)."""
    idx = xp.clip(field01 * 255.0, 0, 255).astype(xp.int32)
    return lut(name, sat, val)[idx]


def names():
    return sorted(_STOPS)
