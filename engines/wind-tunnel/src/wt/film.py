# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""film.py — the optional "filmed off a screen" post-pass, borrowed from the oscilloscope engine.

We do NOT re-implement it. `engines/oscilloscope/src/osc/crtfilm.py` is a heavily tuned
three-stage physical model (screen halation/grain/dust -> lens barrel+CA+defocus+vignette ->
camera flicker/hum/highlight-blowout/noise) and a second copy would immediately drift out of
sync with it. This module just puts that package on sys.path and hands it a NEUTRAL WHITE
phosphor config — mandatory here, because the green-phosphor defaults are only correct for a
single-hue trace and would wreck a full-colour jet field (this is the same reasoning as the
oscilloscope engine's ascii_video.py).

If the oscilloscope engine is missing, `--film` fails with a clear message; everything else in
this project keeps working.
"""
from __future__ import annotations

import os
import sys

from .config import OSC_ROOT


def make_look(w, h, fps, n_frames, *, glow=0.55, ambient=0.16, bow=0.045, seed=7,
              exposure=0.72):
    """`exposure` is the one setting that MUST be lowered for this project. FilmLook defaults to
    1.75 because an oscilloscope screen is mostly black with a thin bright trace to expose FOR.
    A wind-tunnel frame is bright edge to edge, so 1.75 blows the whole field out to pastel and
    turns the body's black fill grey. ~0.7 (and roughly half the halation) puts a full-frame
    image back in range."""
    if not os.path.isdir(os.path.join(OSC_ROOT, "osc")):
        raise SystemExit(
            f"--film needs the oscilloscope engine's CRT filter, not found in {OSC_ROOT}. "
            "Keep engines/oscilloscope next to engines/wind-tunnel.")
    if OSC_ROOT not in sys.path:
        sys.path.insert(0, OSC_ROOT)
    from osc.crtfilm import FilmLook, film_config_for_phosphor    # noqa: E402

    cfg = film_config_for_phosphor((255, 255, 255), halation_boost=glow)
    look = FilmLook(w, h, fps, n_frames, seed=seed,
                    ambient_gain=ambient, barrel_k=bow, **cfg)
    look.exposure = float(exposure)
    return look
