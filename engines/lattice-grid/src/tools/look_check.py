# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""look_check.py - does the render actually LOOK like the source?

`cli.py --dry` answers "how many elements are lit", which is necessary and not sufficient: a
frame can sit dead-centre of the density band and still be twice too dark, have no white
cores, or have drifted off the palette. So measure the render exactly the way the source was
measured, and print the two columns side by side.

This is the only honest way to tune this format: every constant in `scenes.py` was set against
these numbers, not by eye.

SOURCE targets are measured over reel_51d5e4f3 in t = [3, 40] s (i.e. excluding its fade-in
and its 3 s fade to black) at 3 fps, 720x1280. The render is sampled over the equivalent
interior window so the two arcs are compared on like ground.

    python tools\\look_check.py circuit_square
    python tools\\look_check.py circuit_hex --frames 14

Tolerances are wide on purpose. This is a recreation of a format, not a frame-match of a
specific clip - `dark_frac` within ~0.08 and `hot_frac` within a factor of ~2.5 is the band
where the two read as the same thing.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lg.config import RenderConfig            # noqa: E402
from lg.draw import Frame                     # noqa: E402
from lg.scenes import SCENES                  # noqa: E402

# --- measured on reel_51d5e4f3, t in [3,40] s, 3 fps -------------------------------------
# reel_466a0f52 - the creator's own reference for the colour, measured the same way. A much
# harder glitch than reel_51d5e4f3: nearly twice the saturation, a third of the white, and a
# fifth of the blowout. Its lattice is also coarser (~28 px against 12.8).
GLITCH = {
    "mean_luma": 51.79, "dark_frac": 0.011, "bright_frac": 0.327, "hot_frac": 0.0107,
    "white_of_bright": 0.186, "mean_sat_bright": 0.531, "hue_warm": 0.242,
    "hue_cool": 0.514,
}

SOURCE = {
    "mean_luma":        51.9,
    "dark_frac":        0.408,     # luma < 20
    "bright_frac":      0.338,     # luma > 45
    "hot_frac":         0.0543,    # luma > 200   <- the white cores
    "white_of_bright":  0.321,     # saturation < 0.25 among bright pixels
    "mean_sat_bright":  0.389,
    "hue_warm":         0.346,
    "hue_cool":         0.511,
}
TOL = {
    "mean_luma": 14.0, "dark_frac": 0.09, "bright_frac": 0.09,
    "white_of_bright": 0.13, "mean_sat_bright": 0.13,
    "hue_warm": 0.15, "hue_cool": 0.15,
}
HOT_RATIO = 2.5                    # hot_frac is compared as a ratio, not a difference


def stats(rgb):
    f = rgb.astype(np.float32)
    lum = f.mean(2)
    mx, mn = f.max(2), f.min(2)
    sat = np.where(mx > 1, 1.0 - mn / np.maximum(mx, 1), 0.0)
    br = lum > 45
    out = {
        "mean_luma": float(lum.mean()),
        "dark_frac": float((lum < 20).mean()),
        "bright_frac": float(br.mean()),
        "hot_frac": float((lum > 200).mean()),
    }
    if br.sum() > 100:
        out["white_of_bright"] = float((sat[br] < 0.25).mean())
        out["mean_sat_bright"] = float(sat[br].mean())
        sel = f[br & (sat > 0.3)]
        if len(sel) > 100:
            mxv, mnv = sel.max(1), sel.min(1)
            d = np.maximum(mxv - mnv, 1e-6)
            r, g, b = sel[:, 0], sel[:, 1], sel[:, 2]
            h = np.where(mxv == r, ((g - b) / d) % 6,
                         np.where(mxv == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
            out["hue_warm"] = float(((h < 50) | (h > 340)).mean())
            out["hue_cool"] = float(((h >= 160) & (h <= 260)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--frames", type=int, default=11)
    # Default to FULL resolution. Grain, glyph size and halo sigma are all pixel-scale
    # quantities, so a downscaled measurement is only trustworthy for the scale-invariant
    # metrics - and the atlas-scale bug proved how badly that can go wrong.
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--ref", choices=("circuit", "glitch"), default="circuit",
                    help="circuit = reel_51d5e4f3; glitch = reel_466a0f52 (the creator's "
                         "reference for the RGB-delay colour)")
    a = ap.parse_args()

    scene = SCENES[a.scene]()
    cfg = RenderConfig(width=a.width, height=int(round(a.width * 1920 / 1080)), fps=30)
    fr = Frame(cfg, scene.atlas_for(cfg))

    # interior window, matching how SOURCE excludes its own fades
    t0, t1 = 0.18 * scene.duration, 0.92 * scene.duration
    acc = {}
    for k in range(a.frames):
        t = t0 + (t1 - t0) * k / max(a.frames - 1, 1)
        for key, v in stats(scene.render_frame(fr, t)).items():
            acc.setdefault(key, []).append(v)
    got = {k: float(np.mean(v)) for k, v in acc.items()}

    ref = GLITCH if a.ref == "glitch" else SOURCE
    print(f"\n  {a.scene}   {a.frames} frames, t={t0:.1f}..{t1:.1f}s, {cfg.width}px"
          f"   ref={a.ref}\n")
    print(f"  {'metric':<18}{'render':>10}{'source':>10}   verdict")
    bad = 0
    for k, want in ref.items():
        g = got.get(k)
        if g is None:
            continue
        if k == "hot_frac":
            ok = (max(g, 1e-6) / want) < HOT_RATIO and (want / max(g, 1e-6)) < HOT_RATIO
            note = f"x{g/want:.2f}" if want else ""
        else:
            ok = abs(g - want) <= TOL[k]
            note = f"{g-want:+.3f}" if want < 5 else f"{g-want:+.1f}"
        bad += not ok
        print(f"  {k:<18}{g:>10.4f}{want:>10.4f}   {'ok  ' if ok else 'OFF '}{note}")
    print(f"\n  {'MATCH' if not bad else str(bad) + ' metric(s) off'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
