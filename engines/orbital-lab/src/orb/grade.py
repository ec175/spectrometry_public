# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""grade.py - a COLOUR-ONLY transform of the two ramps, and nothing else.

The viewer's sliders end up here. Six numbers, applied to `render.RAMP_MESH` and
`render.RAMP_CORE` and to nothing else in the engine.

⚠ COLOUR ONLY, AND THAT BOUNDARY IS THE POINT. Exposure in this engine is *settled* - `ink`,
`bloom_threshold`, `glow_gain`, `glow2_gain`, `vignette`, `trail_tau` and `point_sigma` were
each landed by measuring against the source reel or against a frame pulled out of a shipped
final, and `qm_negative`'s VOID block re-aimed all of them a second time for a frame-filling
field. A "colour" slider that quietly moved any of those would re-look a judged final while
claiming to be a hue control. So this module cannot reach them: it rewrites ramp STOPS, and a
ramp stop cannot change how much light lands on a pixel - only what colour that light is.

⚠ AND THE CHROMATIC SPLIT STAYS CONSISTENT FOR FREE. `render.split_luts` derives the
magenta/cyan pair from whatever the parent ramp is, per channel, so a graded parent still
splits into two LUTs that SUM to it exactly. Hand-authoring two graded split ramps would break
that identity at every point where the copies register - which is every node, every knot and
the polar axis. Grade the parent; the split follows.

THE SIX KNOBS

| knob | range | what it moves |
|---|---|---|
| `hue` | -180..180 deg | rotates the mesh ramp's hue |
| `sat` | 0..2 | mesh saturation, 1 = canon |
| `mid_bias` | -1..1 | **the one that actually matters** - green vs blue in the MIDS |
| `white` | 0..1 | how white the hot end goes; 1 = canon's (250,255,255) |
| `core_hue` | -180..180 | rotates the nucleus ramp (canon: magenta) |
| `core_sat` | 0..2 | nucleus saturation |

⚠ `mid_bias` IS NOT A SECOND SATURATION CONTROL, and it exists because of a documented
failure. `render.RAMP_MESH`'s comment records that Particle_Fields' `cyan` ramp is B-dominant
through 0.3-0.7, that a wireframe puts nearly all of its area in exactly that band, and that
the first pass of this engine therefore came back "an even electric blue with none of the
source's green". Every mid stop in canon leads on GREEN for that reason. So the mids are the
colour of the clip, they are a separate axis from hue and saturation, and they get their own
slider - weighted to the middle of the ramp and tapering to nothing at both ends, so the
indigo skin and the white knots are left where they are.
"""
from __future__ import annotations

import colorsys

#: Identity. `apply(stops, DEFAULT)` returns the canon ramp unchanged, entry for entry - which
#: is what makes "reset" exact rather than approximately right.
DEFAULT = dict(hue=0.0, sat=1.0, mid_bias=0.0, white=1.0, core_hue=0.0, core_sat=1.0)


def _clamp(v, a, b):
    return a if v < a else (b if v > b else v)


def _hsv(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)


def _rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp(s, 0.0, 1.0), _clamp(v, 0.0, 1.0))
    return (r * 255.0, g * 255.0, b * 255.0)


def _mid_weight(t: float) -> float:
    """A hump peaking at t=0.5 and zero at both ends - where a wireframe's area actually is.

    Tapering to zero at 0 and 1 is what keeps `mid_bias` from being a global tint: the dark
    end is the lobe SKIN (canon indigo) and the top is the KNOTS (canon white), and both were
    settled against the source. Only the body between them is up for grabs.
    """
    x = _clamp(t, 0.0, 1.0)
    return (4.0 * x * (1.0 - x)) ** 1.4


def apply(stops, g: dict, core: bool = False):
    """A new stop list. `stops` is untouched.

    Ordinary hue/saturation live in HSV because that is the space the words mean something in.
    `mid_bias` is applied in RGB afterwards, because "lead on green rather than blue" is a
    statement about two CHANNELS, and expressing it as a hue rotation would also swing the
    skin and the knots - the exact thing `_mid_weight` is there to prevent.
    """
    g = {**DEFAULT, **(g or {})}
    dh = (float(g["core_hue" if core else "hue"]) % 360.0) / 360.0
    ds = float(g["core_sat" if core else "sat"])
    out = []
    for t, rgb in stops:
        h, s, v = _hsv(rgb)
        r, gg, b = _rgb(h + dh, s * ds, v)
        if not core:
            w = _mid_weight(t) * float(g["mid_bias"])
            if w:
                # Trade between G and B at constant G+B, so the stop's total energy - and
                # therefore the frame's level - does not move with the slider.
                tot = gg + b
                k = _clamp(0.5 + 0.5 * w, 0.0, 1.0)
                gg, b = tot * k, tot * (1.0 - k)
            wl = _clamp(float(g["white"]), 0.0, 1.0)
            if wl != 1.0 and t > 0.6:
                # ⚠ DE-WHITEN AT CONSTANT VALUE, never by darkening. The hot end of the ramp
                # is where the knots live and they are the only thing that crosses
                # `bloom_threshold` - dimming them would silently remove the glow, which is
                # an exposure change wearing a colour slider's clothes. Instead the stop is
                # pushed AWAY from grey toward its own hue at the same brightness: `v` is
                # held and only `s` rises, so the knot stays exactly as hot and merely stops
                # being white.
                f = (t - 0.6) / 0.4
                h2, s2, v2 = _hsv((r, gg, b))
                r, gg, b = _rgb(h2, s2 + (1.0 - s2) * (1.0 - wl) * f, v2)
        out.append((t, (_clamp(r, 0, 255), _clamp(gg, 0, 255), _clamp(b, 0, 255))))
    return out


def lut(stops, g: dict, core: bool = False, n: int = 256):
    """The 256-entry LUT the renderer will actually build, as a list of [r,g,b] ints.

    ⚠ THE VIEWER IS SENT THIS, rather than reimplementing the grade in JavaScript. Two
    implementations of a colour transform drift, and the one the browser shows is the one
    Clancy judges - so the browser is handed the authoritative table and only ever indexes it.
    What is left approximate in the preview is "luma stands in for the ramp key", which is
    stated in the UI; the colour maths is not approximate at all.
    """
    import numpy as np

    from . import render as R
    return [[int(round(c)) for c in row]
            for row in np.asarray(R.ramp(apply(stops, g, core=core), n=n))]
