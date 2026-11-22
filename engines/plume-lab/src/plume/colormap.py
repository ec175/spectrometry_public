# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""colormap.py - the LUTs, and the per-frame normalisation the reference clip advertises.

`plumefire` is MEASURED, not invented. The control points below came out of
``Video_Interpreter\\rocket_pressure.mp4`` by taking radial profiles through the jet (along which
the plotted scalar is monotone), projecting every sampled pixel onto the longest such profile and
binning by arc length in RGB. Its shape is turbo-like - but on a **violet floor** instead of
turbo's dark blue, and it tops out in **dark red**, which is the one thing that is easy to get
backwards: in the source the fastest gas is the deep maroon lens just downstream of the exit, NOT
the bright orange around it. Read the ramp end-to-end before "fixing" a frame that looks dark in
the middle of the jet.

MEASURED-DOWNSTREAM-OF-EVERYTHING WARNING (Video_Interpreter development notes, and it has bitten this
library before): these swatches are post-encode. They already contain whatever the source's
renderer did after mapping - so they are the right thing to map TO, and the wrong thing to map to
and then bloom again. `render.py` therefore applies no bloom at all by default.
"""
from __future__ import annotations

import numpy as np

from .gpu import asarray, xp

# (position, R, G, B) - position 0 = ambient / slowest, 1 = fastest.
PLUMEFIRE = [
    (0.000, 46, 21, 61),      # #2E153D  ambient violet floor
    (0.070, 49, 27, 68),      # #311B44
    (0.150, 51, 49, 100),     # #333164
    (0.250, 51, 71, 131),     # #334783
    (0.340, 52, 99, 171),     # #3463AB
    (0.430, 52, 125, 205),    # #347DCD
    (0.510, 53, 156, 236),    # #359CEB  peak blue
    (0.580, 47, 171, 209),    # #2FABD1
    (0.640, 44, 180, 191),    # #2CB4BF
    (0.700, 46, 190, 148),    # #2EBE94
    (0.750, 68, 194, 117),    # #44C275
    (0.795, 110, 199, 81),    # #6EC751
    (0.835, 166, 200, 49),    # #A6C831
    (0.870, 199, 190, 39),    # #C7BE27
    (0.900, 238, 162, 36),    # #EEA224
    (0.930, 248, 140, 30),    # #F88C1E
    (0.955, 222, 105, 25),    # #DE6919
    (0.980, 170, 58, 10),     # #AA3A0A
    (1.000, 110, 28, 2),      # #6E1C02  fastest - dark red, NOT bright
]

# BLACK-FLOOR PLUMEFIRE. Identical to the measured ramp from 0.150 upward - only the two
# lowest control points move, so the ambient reads as true black instead of the source's violet
# while every colour the JET occupies is the one measured off `rocket_pressure.mp4`.
#
# Why it is safe to swap under VERT_LOOK without re-tuning anything: those settings (clip_hi
# 0.94, vmax_mult 1.12, lut_gamma 0.80) were solved to put the core at ramp 0.86 (p50) and 0.92
# (p90). Nothing below 0.150 is involved in that, so the core colour is bit-for-bit unchanged.
# The one visible consequence is intended - the undisturbed field stops glowing, so the plume is
# the only lit thing in the frame.
PLUMENOIR = [
    (0.000, 0, 0, 0),         # #000000  true black floor (source: #2E153D violet)
    (0.070, 17, 9, 24),       # ~0.35x the measured #311B44 - fades violet toward black
] + PLUMEFIRE[2:]             # rejoins the MEASURED ramp at 0.150, untouched

# ROCKETFIRE - the PRODUCTION ramp, and the one table here that is NOT a measurement.
#
# Everything else in this file was recovered from `rocket_pressure.mp4` because the job was to
# match a CFD panel. This one exists because Clancy asked for the opposite: *"less of a physics
# demonstration, more of an intensely visual production done with our engines, derived from our
# physics engine"*. The FIELD underneath is untouched - the same solved axisymmetric jet, the
# same `norm="auto"` range pinned to the jet speed - only what a given speed LOOKS like changes.
#
# Why a flame ramp is also the more honest picture of an exhaust, not just the prettier one:
# `plumefire` spends most of its range on blues and greens, which is right for a scalar field
# with a colour key beside it and wrong for a self-luminous gas. Real exhaust runs black -> deep
# ember -> orange -> straw -> white, monotonically in BRIGHTNESS, so a shock-cell train reads as
# the bright diamonds it actually is instead of as a hue cycle. It also takes the entrained halo
# to near-black by construction, which is what stops a rocket plume glowing pale blue in space.
ROCKETFIRE = [
    (0.000, 0, 0, 0),         # ambient - nothing at all
    (0.070, 14, 3, 1),
    (0.150, 46, 9, 2),        # first ember
    (0.260, 104, 22, 3),
    (0.380, 168, 48, 6),
    (0.490, 214, 78, 11),
    (0.590, 240, 112, 20),
    (0.680, 252, 146, 33),
    (0.760, 255, 178, 58),
    (0.830, 255, 205, 95),
    (0.890, 255, 227, 145),
    (0.940, 255, 242, 197),
    (0.975, 255, 251, 234),
    (1.000, 255, 255, 255),   # the core
]

# A cool alternative for schlieren-style fields: near-black floor, white waves.
GRAPHITE = [
    (0.000, 10, 10, 14),
    (0.300, 32, 36, 52),
    (0.600, 96, 116, 150),
    (0.820, 186, 202, 224),
    (1.000, 255, 255, 255),
]

# Ice: the same violet discipline, cold end. Useful for temperature fields.
CRYO = [
    (0.000, 20, 14, 40),
    (0.250, 32, 52, 110),
    (0.500, 44, 122, 186),
    (0.720, 120, 196, 214),
    (0.880, 206, 232, 238),
    (1.000, 255, 252, 240),
]

TABLES = {"plumefire": PLUMEFIRE, "plumenoir": PLUMENOIR, "rocketfire": ROCKETFIRE,
          "graphite": GRAPHITE, "cryo": CRYO}


def build_lut(name="plumefire", n=256, gamma=1.0, clip_hi=1.0):
    """(n, 3) uint8 LUT, linearly interpolated between the control points.

    `gamma` != 1 bends the VALUE axis before lookup, which is the honest place to put a
    brightness taste knob - it does not move any measured colour, it only changes which scalar
    lands on it.

    `clip_hi` < 1 STOPS SHORT of the ramp's top, and that is a legibility fix with a measurement
    behind it. `plumefire` tops out in dark red (#6E1C02) because the source does - so the
    FASTEST gas is the DARKEST, and a shock-cell train drives the core back and forth across that
    peak. Measured on the first shipped `plume_ref`: **37.8% of jet-core pixels were darker than
    luma 110**, and a single core pixel swung R 41-247 / G 19-200 / B 3-229 over the clip - the
    whole ramp - which reads as strobing and buries the structure it is supposed to show.
    `clip_hi` ~0.88 ends the ramp at bright orange instead, so speed maps monotonically to
    brightness across the range the jet actually occupies. Nothing about the measured colours
    changes; the top of the ramp simply stops being reachable.
    """
    tab = TABLES[name]
    pos = np.array([c[0] for c in tab], np.float64)
    rgb = np.array([c[1:] for c in tab], np.float64)
    t = (np.linspace(0.0, 1.0, n) ** float(gamma)) * float(clip_hi)
    out = np.empty((n, 3), np.float64)
    for k in range(3):
        out[:, k] = np.interp(t, pos, rgb[:, k])
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Mapper:
    """Scalar field -> RGB, with the source clip's **per-frame colour range**.

    The reference burns `per-frame colour range` into its own footer, and that single choice is
    responsible for the clip's most distinctive frame: at t=0 the field is uniform, so min == max,
    the normalisation is 0/0 and everything lands on ONE colour - a flat green panel. That is
    reproduced here explicitly rather than by accident (`_UNIFORM`), because a NaN would give a
    black frame and look like a bug.

    modes:
      `minmax`  - exactly what the source says it does. Twitchy by construction: one hot cell
                  moves the whole picture. That twitch is part of the look.
      `p999`    - 99.9th percentile instead of the max. Steadier, and what to reach for when a
                  startup transient is washing out the rest of the clip.
      `fixed`   - a constant range supplied by the scene. The only mode in which two clips are
                  directly comparable to each other.
      `stable`  - `lo` pinned to the ambient value, `hi` an EMA-smoothed high percentile, so a
                  single transient cell cannot set the range for a whole frame.

                  ⚠️ **This is NOT the flicker fix, and an earlier version of this docstring
                  claimed it was.** Measured on one captured field, swapping minmax -> stable at
                  the same `clip_hi` moves the core frame-to-frame delta by about 4% (22.7 ->
                  21.8), and a first attempt with q=0.995/alpha=0.10 made it *worse* than minmax
                  (40.4 vs 34.8) - a lower percentile puts `hi` below the true peak, which
                  stretches the jet across MORE of the ramp rather than less. The flicker lever
                  is `build_lut(clip_hi=...)`. This mode is kept because it is mildly better and
                  because it cannot be driven by one hot cell, not because it fixed anything.
    """

    _UNIFORM = 0.5

    def __init__(self, lut, mode="minmax", lo=None, hi=None, floor=0.0,
                 q=0.9999, alpha=0.30, hi_min=1e-6):
        self.lut = asarray(np.ascontiguousarray(lut))
        self.n = int(lut.shape[0])
        self.mode = mode
        self.lo, self.hi = lo, hi
        self.floor = float(floor)
        self.q = float(q)
        self.alpha = float(alpha)
        self.hi_min = float(hi_min)
        self._ema = None
        self.last_range = (0.0, 0.0)

    def normalise(self, f):
        if self.mode == "fixed":
            lo = 0.0 if self.lo is None else float(self.lo)
            hi = 1.0 if self.hi is None else float(self.hi)
        elif self.mode == "stable":
            lo = float(self.floor)
            tgt = float(xp.quantile(f, self.q))
            self._ema = tgt if self._ema is None else self._ema + self.alpha * (tgt - self._ema)
            hi = max(self._ema, lo + self.hi_min)
        elif self.mode == "p999":
            lo = float(self.floor)
            hi = float(xp.quantile(f, 0.999))
        else:
            lo = float(f.min()) if self.lo is None else float(self.lo)
            hi = float(f.max())
        self.last_range = (lo, hi)
        span = hi - lo
        if not np.isfinite(span) or span <= 1e-12:
            return xp.full(f.shape, self._UNIFORM, dtype=xp.float32)
        return xp.clip((f - lo) / span, 0.0, 1.0)

    def rgb(self, f):
        """(H, W) scalar -> (H, W, 3) uint8, one gather. Cheap; do not micro-optimise."""
        t = self.normalise(f)
        idx = xp.clip((t * (self.n - 1) + 0.5).astype(xp.int32), 0, self.n - 1)
        return self.lut[idx]
