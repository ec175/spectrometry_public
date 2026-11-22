# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""colour.py - named colour ramps, and the one that was measured off real microscopy.

A ramp maps a scalar in [0, 1] to an RGB triple in 0-255. Everything that colours by a
value - depth, pressure, age, speed - goes through here, so changing a look is changing a
ramp name in a scene file rather than editing an engine.


WHY `depth_lut` IS NOT A RAINBOW
--------------------------------
The Bio engine's default ramp is measured, not invented. It comes from a real confocal
clip of intestinal epithelial cells (HHMI / Matthew Tyska, Vanderbilt - see
`reference/docs/CELL_REFERENCE.md`), in which the colours are DEPTH-CODED: each z-slice is
assigned a lookup table and the slices are max-projected together.

The measurement that matters: in three frames sampled across that clip, the hue weight
falling in the **210-270 degree band - the deep blues - was 0.4%, 1.0% and 1.3%**. It is
essentially absent. The ramp runs yellow -> green -> cyan -> magenta and JUMPS the blues.

That gap is not an accident of the sample, it is what fluorescence LUT compositing does,
and it is a large part of why the source looks expensive rather than like a heat map. A
naive `hsv` ramp sweeps straight through 240 degrees of saturated blue and immediately
reads as a physics-textbook colourbar. If you want the microscopy look, keep the hole.

Measured anchors, and their measured hue weights:

    stop   hue     what it is                     weight t=0.5 / 4.0 / 13.0 s
    0.00   ~50     warm yellow, the FAR depth     21.2% / 8.7% / 16.4%
    0.33   ~90     yellow-green                    8.9% / 24.2% / 13.8%
    0.66   ~180    cyan, the mid depth            17.3% / 15.9% / 23.9%
    1.00   ~315    magenta, the NEAR depth        31.4% / 5.1% / 4.7%

`ramp_report` re-measures any ramp against those constraints, and `verify_install.py`
asserts that `depth_lut` still satisfies them - so a well-meaning edit that "adds some
blue for balance" fails a test instead of quietly changing the engine's identity.
"""
from __future__ import annotations

import numpy as np

#: name -> list of (stop, (r, g, b)). Stops must start at 0.0 and end at 1.0.
RAMPS = {
    # --- the measured one --------------------------------------------------------------
    # Measures blue_pct = 1.95, sat_mean = 0.570 - both inside the reference envelope
    # (blue 0.4-1.3%, edge saturation 0.47-0.55). Do not "balance" this with a violet
    # stop: an earlier draft had one at 0.80 and measured 12.89% blue, three times the
    # reference, and it read as a physics colourbar rather than as microscopy.
    #
    # The cyan -> magenta hop is the delicate part. A straight RGB interpolation between
    # them passes through (159, 151, 222), which is squarely blue. It is bridged instead
    # through a PALE step at 0.79, so the ramp desaturates toward white and comes back out
    # the other side rather than travelling round the hue circle through 240 degrees.
    "depth_lut": [
        (0.00, (255, 214, 64)),      # warm yellow    hue  47   - the FAR depth
        (0.25, (214, 236, 72)),      # yellow-green   hue  71
        (0.48, (96, 236, 168)),      # green-cyan     hue 152
        (0.70, (64, 224, 236)),      # cyan           hue 185   - the MID depth
        (0.79, (214, 238, 244)),     # pale bridge - desaturate, do not go through blue
        (0.88, (255, 152, 228)),     # pale magenta
        (1.00, (255, 78, 208)),      # magenta        hue 316   - the NEAR depth
    ],
    #: The same family, warm end only - for when magenta reads as too synthetic.
    "depth_warm": [
        (0.00, (255, 196, 48)),
        (0.35, (255, 128, 64)),
        (0.70, (236, 72, 120)),
        (1.00, (255, 64, 196)),
    ],
    #: Cool end only. Good for a colony where you want the drama in the outline, not the fill.
    "depth_cool": [
        (0.00, (40, 120, 160)),
        (0.40, (64, 224, 236)),
        (0.75, (120, 244, 200)),
        (1.00, (232, 255, 236)),
    ],
    # --- RICH ramps: saturated all the way, and they still never enter the blues -------
    #
    # `depth_lut` bridges cyan to magenta through a PALE step, because a straight
    # interpolation between them runs through blue. That bridge costs saturation exactly
    # where the colour driver peaks, so a crowded colony centre washes out to near-white -
    # correct by the measurement, disappointing on screen.
    #
    # These ramps solve the same constraint the other way round: they travel from magenta
    # to cyan going through RED, ORANGE, GOLD and GREEN - the long way round the hue
    # circle, but the way that never passes 210-270 degrees. Every stop stays saturated,
    # nothing goes pale, and the blue gap is preserved for free because the path simply
    # never goes near it.
    #
    # Use these when the picture matters more than matching the reference clip exactly.
    "polar_rich": [
        (0.00, (255, 60, 190)),      # magenta      316
        (0.16, (255, 46, 96)),       # crimson      344
        (0.34, (255, 122, 30)),      # orange        24
        (0.52, (250, 224, 40)),      # gold          52
        (0.70, (126, 240, 92)),      # chartreuse   106
        (0.86, (44, 232, 200)),      # teal         172
        (1.00, (70, 216, 255)),      # cyan         193
    ],
    #: The same path, warmer and heavier - reads as stained glass.
    "polar_amber": [
        (0.00, (255, 40, 128)),
        (0.22, (255, 90, 40)),
        (0.46, (255, 176, 32)),
        (0.68, (236, 240, 70)),
        (0.86, (120, 235, 150)),
        (1.00, (46, 226, 214)),
    ],
    #: Cool half only, fully saturated - jade through cyan to a green-white.
    "polar_jade": [
        (0.00, (16, 120, 110)),
        (0.30, (28, 196, 150)),
        (0.60, (70, 238, 210)),
        (0.82, (150, 248, 226)),
        (1.00, (224, 255, 246)),
    ],
    #: Magenta through crimson to gold. No greens at all - a single warm sweep.
    "polar_bloom": [
        (0.00, (150, 20, 120)),
        (0.26, (238, 40, 150)),
        (0.54, (255, 70, 96)),
        (0.78, (255, 150, 46)),
        (1.00, (255, 226, 120)),
    ],
    #: Deep violet-magenta to hot pink to white-gold. The most "cross-polar" of the set.
    "polar_orchid": [
        (0.00, (92, 24, 132)),
        (0.24, (188, 42, 190)),
        (0.50, (255, 72, 168)),
        (0.74, (255, 140, 120)),
        (1.00, (255, 236, 190)),
    ],
    # --- general purpose ---------------------------------------------------------------
    "ember": [(0.0, (12, 4, 2)), (0.35, (168, 40, 12)), (0.7, (248, 152, 40)),
              (1.0, (255, 244, 208))],
    "ice": [(0.0, (4, 10, 28)), (0.4, (32, 108, 196)), (0.75, (120, 208, 248)),
            (1.0, (240, 252, 255))],
    "mono": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "viridis_ish": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                    (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
    #: Included ONLY as the counterexample the docstring warns about. It sweeps the blues.
    "hsv": [(0.0, (255, 0, 0)), (0.17, (255, 255, 0)), (0.33, (0, 255, 0)),
            (0.5, (0, 255, 255)), (0.67, (0, 0, 255)), (0.83, (255, 0, 255)),
            (1.0, (255, 0, 0))],
}

_CACHE = {}


def lut(name: str, n: int = 256) -> np.ndarray:
    """(n, 3) float32 lookup table for a named ramp, linearly interpolated between stops."""
    key = (name, n)
    if key in _CACHE:
        return _CACHE[key]
    if name not in RAMPS:
        raise KeyError("unknown ramp %r; have %s" % (name, sorted(RAMPS)))
    stops = RAMPS[name]
    xs = np.array([s for s, _ in stops], np.float32)
    cs = np.array([c for _, c in stops], np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], axis=1).astype(np.float32)
    _CACHE[key] = out
    return out


def sample(name: str, v, gamma: float = 1.0) -> np.ndarray:
    """Map scalar(s) in [0, 1] to RGB. `v` may be a float or an array.

    `gamma` biases WHERE along the ramp values land without changing the ramp itself:
    gamma > 1 pushes values toward the low (first) stop. Use it to make a colour driver
    sit in the interesting part of the ramp instead of rescaling the driver.
    """
    table = lut(name)
    a = np.clip(np.asarray(v, np.float32), 0.0, 1.0)
    if gamma != 1.0:
        a = np.power(a, gamma)
    idx = a * (len(table) - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, len(table) - 1)
    f = (idx - lo)[..., None]
    return table[lo] * (1.0 - f) + table[hi] * f


def hue_of(rgb) -> np.ndarray:
    """Hue in degrees for RGB triples in 0-255. Used by the gates, not by rendering."""
    a = np.asarray(rgb, np.float32) / 255.0
    a = a.reshape(-1, 3)
    mx = a.max(1)
    mn = a.min(1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros_like(mx)
    m0 = mx == r
    m1 = (mx == g) & ~m0
    m2 = (mx == b) & ~m0 & ~m1
    h[m0] = ((g - b)[m0] / d[m0]) % 6
    h[m1] = ((b - r)[m1] / d[m1]) + 2
    h[m2] = ((r - g)[m2] / d[m2]) + 4
    return h * 60.0


def ramp_report(name: str) -> dict:
    """Measure a ramp the same way the reference clip was measured.

    `blue_pct` is the fraction of the ramp whose hue falls in 210-270 degrees. For
    `depth_lut` this must stay small - see the module docstring.
    """
    t = lut(name, 256)
    h = hue_of(t)
    sat = 1.0 - t.min(1) / np.maximum(t.max(1), 1e-6)
    return {
        "ramp": name,
        "blue_pct": round(float(((h >= 210) & (h <= 270)).mean() * 100), 2),
        "hue_span_deg": [round(float(h.min()), 1), round(float(h.max()), 1)],
        "sat_mean": round(float(sat.mean()), 3),
        "value_mean": round(float(t.max(1).mean()), 1),
    }


if __name__ == "__main__":
    for nm in sorted(RAMPS):
        print(ramp_report(nm))


#: YIQ basis. Rotating the (I, Q) plane is a hue rotation that leaves LUMA untouched,
#: which is what you want here: two strains should differ in colour without one of them
#: becoming darker than the other and reading as "further away".
_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.274, -0.322],
                     [0.211, -0.523, 0.312]], np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def rotate_hue(rgb, degrees):
    """Rotate hue by `degrees`, preserving luma. `rgb` is (..., 3) in 0-255.

    `degrees` may be a scalar or one value per row, so a whole population can be
    recoloured by strain in a single call.
    """
    a = np.asarray(rgb, np.float32)
    flat = a.reshape(-1, 3)
    yiq = flat @ _RGB2YIQ.T
    th = np.radians(np.asarray(degrees, np.float32)).reshape(-1)
    c, s_ = np.cos(th), np.sin(th)
    i, q = yiq[:, 1].copy(), yiq[:, 2].copy()
    yiq[:, 1] = c * i - s_ * q
    yiq[:, 2] = s_ * i + c * q
    return np.clip(yiq @ _YIQ2RGB.T, 0.0, 255.0).reshape(a.shape)
