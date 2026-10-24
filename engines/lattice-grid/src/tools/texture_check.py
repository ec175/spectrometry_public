# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""texture_check.py - does the render have the source's TEXTURE?

The third blind spot. `look_check` measures one frame's bulk statistics; `motion_check`
measures how it moves; both were fully passing on a render that still read as "quite
different" by eye. What they cannot see is how colour and brightness are distributed at the
scale of individual ELEMENTS.

SOURCE constants measured on reel_51d5e4f3, t=[12,20]s at 720x1280 (pitch 12.83 px there):

    hue_corr_1p        0.765   correlation between the hues of elements ONE pitch apart
    hue_corr_8p        0.484   ...and EIGHT pitches apart
    local_contrast     0.593   sd/mean of luma in a 3-pitch window: crispness
    coherence          0.326   structure-tensor anisotropy: 0 = blobs, 1 = filaments
    luma p50/p75/p90   33.3 / 78.0 / 173.3

The two hue numbers are the point, and they must be read TOGETHER. The source drops to 0.77
at one pitch (real per-element variety - an isolated green dot beside an orange ring) and
then PLATEAUS near 0.5 (a broad regional tint underneath). A single smooth hue field cannot
do both: it gives ~0.98 next door and ~0 by eight pitches, which is what the first build did.

    python tools\\texture_check.py circuit_hex
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lg.config import RenderConfig            # noqa: E402
from lg.draw import Frame                     # noqa: E402
from lg.scenes import SCENES                  # noqa: E402

SOURCE = {"hue_corr_1p": 0.765, "hue_corr_2p": 0.609, "hue_corr_4p": 0.539,
          "hue_corr_8p": 0.484, "local_contrast": 0.593, "coherence": 0.326,
          "luma_p50": 33.3, "luma_p75": 78.0, "luma_p90": 173.3}
TOL = {"hue_corr_1p": 0.11, "hue_corr_2p": 0.13, "hue_corr_4p": 0.15, "hue_corr_8p": 0.17,
       "local_contrast": 0.10, "coherence": 0.075,
       "luma_p50": 13.0, "luma_p75": 24.0, "luma_p90": 34.0}


def _hue(f):
    mx, mn = f.max(-1), f.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return np.where(mx == r, ((g - b) / d) % 6,
                    np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0


def measure(frames, pitch):
    lum = frames.mean(3)
    out = {}
    # -- hue correlation vs separation ----------------------------------------------------
    acc = {k: [] for k in (1, 2, 4, 8)}
    for f in frames:
        h = np.deg2rad(_hue(f))
        mx, mn = f.max(-1), f.min(-1)
        sat = np.where(mx > 1, 1 - mn / np.maximum(mx, 1), 0)
        m = (f.mean(-1) > 40) & (sat > 0.30)
        hx, hy = np.cos(h) * m, np.sin(h) * m
        for s in acc:
            k = max(1, int(round(pitch * s)))
            w = m[:, :-k] & m[:, k:]
            if w.sum() > 200:
                acc[s].append(float((hx[:, :-k] * hx[:, k:] +
                                     hy[:, :-k] * hy[:, k:])[w].mean()))
    for s in acc:
        out[f"hue_corr_{s}p"] = float(np.mean(acc[s])) if acc[s] else float("nan")

    # -- local contrast --------------------------------------------------------------------
    w = max(3, int(round(pitch * 3)) | 1)
    mu = uniform_filter(lum, size=(1, w, w), mode="nearest")
    sd = np.sqrt(np.maximum(uniform_filter(lum ** 2, size=(1, w, w), mode="nearest")
                            - mu ** 2, 0))
    sel = mu > 12
    out["local_contrast"] = float((sd[sel] / mu[sel]).mean())

    # -- structure coherence: filaments vs blobs -------------------------------------------
    co = []
    for f in lum:
        b = gaussian_filter(f, 2.0)
        gy, gx = np.gradient(b)
        Jxx = gaussian_filter(gx * gx, 9); Jyy = gaussian_filter(gy * gy, 9)
        Jxy = gaussian_filter(gx * gy, 9)
        tr, det = Jxx + Jyy, Jxx * Jyy - Jxy * Jxy
        disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0))
        c = np.where(tr > 1e-6, disc / np.maximum(tr, 1e-6), 0)
        if (b > 18).sum() > 500:
            co.append(float(c[b > 18].mean()))
    out["coherence"] = float(np.mean(co)) if co else float("nan")

    for p in (50, 75, 90):
        out[f"luma_p{p}"] = float(np.percentile(lum, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    scene = SCENES[a.scene]()
    # 720x1280 - the source's own pixel basis, so `pitch` and the window sizes mean the
    # same thing on both sides.
    cfg = RenderConfig(width=720, height=1280)
    for k, v in getattr(scene, "cfg_overrides", {}).items():
        setattr(cfg, k, v)
    fr = Frame(cfg, scene.atlas_for(cfg))
    t0, t1 = 0.28 * scene.duration, 0.80 * scene.duration
    F = np.stack([scene.render_frame(fr, t0 + (t1 - t0) * i / max(a.frames - 1, 1),
                                     frame_index=i).astype(np.float32)
                  for i in range(a.frames)])
    got = measure(F, scene.pitch * cfg.scale)

    print(f"\n  {a.scene}   {a.frames} frames @720x1280\n")
    print(f"  {'metric':<16}{'render':>10}{'source':>10}   verdict")
    bad = 0
    for k, want in SOURCE.items():
        g = got.get(k, float("nan"))
        ok = abs(g - want) <= TOL[k]
        bad += not ok
        print(f"  {k:<16}{g:>10.3f}{want:>10.3f}   {'ok  ' if ok else 'OFF '}{g-want:+.3f}")
    print(f"\n  {'MATCH' if not bad else str(bad) + ' metric(s) off'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
