# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""motion_check.py - does the render MOVE like the source, and carry its VFX layer?

`look_check.py` measures single frames: luminance, palette, contrast. It is blind to
everything this file measures, and the first build passed all eight of its metrics while
being 4.4x too still. So this is the second half of the acceptance criteria.

SOURCE targets measured on reel_51d5e4f3 over t=[12,22]s at FULL 30 fps (the first build
sampled at 6 fps, which aliased away exactly the fast motion this measures - that is the
mistake this tool exists to prevent):

    per-cell temporal std          67.3     how hard a cell swings over time
    HF/total (>~1.7 Hz)            0.128    share of that carried above ~1.7 Hz
    neighbour-HF correlation       0.20     LOW => each element twinkles independently
    global-flicker std             0.93     ~0 => it is not a global exposure wobble
    lit regions per frame          12.9     granularity of the light field
    halo anisotropy (max/min)      1.62     anamorphic horizontal streak
    halo peak angle                0/180    horizontal, not diagonal, not vertical
    grain: HF spatial residual     11.45
    grain: temporal noise in dark   7.14

    python tools\\motion_check.py circuit_square
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lg.config import RenderConfig            # noqa: E402
from lg.draw import Frame                     # noqa: E402
from lg.scenes import SCENES                  # noqa: E402

SOURCE = {
    "cell_std": 67.3, "hf_ratio": 0.128, "nbr_hf_corr": 0.20, "global_flicker": 0.93,
    "lit_regions": 12.9, "halo_aniso": 1.62, "grain_spatial": 11.45, "grain_dark": 7.14,
}
TOL = {
    "cell_std": 22.0, "hf_ratio": 0.06, "nbr_hf_corr": 0.22, "global_flicker": 2.5,
    "lit_regions": 6.0, "halo_aniso": 0.45, "grain_spatial": 6.0, "grain_dark": 4.5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=360)
    a = ap.parse_args()

    scene = SCENES[a.scene]()
    cfg = RenderConfig(width=a.width, height=int(round(a.width * 1920 / 1080)), fps=a.fps)
    fr = Frame(cfg, scene.atlas_for(cfg))

    t0 = 0.30 * scene.duration
    n = int(a.seconds * a.fps)
    vid = np.empty((n, cfg.height, cfg.width), np.float32)
    full = None
    for i in range(n):
        t = t0 + i / a.fps
        rgb = scene.render_frame(fr, t, frame_index=i)
        vid[i] = rgb.astype(np.float32).mean(2)
    # Halo and grain are measured at 720x1280 - the SOURCE's own pixel basis. Both are
    # absolute pixel-scale quantities (a sigma in px, a noise std per px), so measuring them
    # at any other resolution compares different things. Render at 720 rather than resampling
    # 1080, which would filter the grain being measured.
    hcfg = RenderConfig(width=720, height=1280)
    hfr = Frame(hcfg, scene.atlas_for(hcfg))
    fg = scene.render_frame(hfr, t0 + 1.0, frame_index=0).astype(np.float32).mean(2)
    fg2 = scene.render_frame(hfr, t0 + 1.0 + 1 / 30, frame_index=1).astype(np.float32).mean(2)

    got = {}
    # -- temporal ------------------------------------------------------------------------
    c = vid[:, 4::6, 4::6].reshape(n, -1)
    win = max(3, int(round(a.fps * 0.3)) | 1)
    hf = c - uniform_filter(c, size=(win, 1), mode="nearest")
    got["cell_std"] = float(c.std(0).mean())
    got["hf_ratio"] = float(hf.std(0).mean() / max(c.std(0).mean(), 1e-6))
    got["global_flicker"] = float(hf.mean(1).std())
    ny, nx = vid[:, 4::6, 4::6].shape[1:]
    hh = hf.reshape(n, ny, nx)
    got["nbr_hf_corr"] = float(np.mean([
        np.corrcoef(hh[:, :, :-1].ravel()[::29], hh[:, :, 1:].ravel()[::29])[0, 1],
        np.corrcoef(hh[:, :-1].ravel()[::29], hh[:, 1:].ravel()[::29])[0, 1]]))

    # -- light-field granularity ---------------------------------------------------------
    ns = []
    for f in vid[::max(1, n // 12)]:
        b = gaussian_filter(f, 7 * cfg.width / 360.0)
        _, k = label(b > np.percentile(b, 80))
        ns.append(k)
    got["lit_regions"] = float(np.mean(ns))

    # -- halo anisotropy: measured as the POINT SPREAD FUNCTION ---------------------------
    # The source pass sampled real cores that happened to sit in dark surroundings. A render
    # dense enough to match the source has almost none of those (measured: 1), so hunting for
    # them here measures nothing. Splat a single element into an empty frame and put it
    # through the identical to_rgb instead: that IS the optical layer, isolated exactly, and
    # it is the quantity the source measurement was approximating in the first place.
    pfr = Frame(hcfg, scene.atlas_for(hcfg))
    pfr.clear()
    pfr.splat(scene.atlas.node["dot"],
              np.array([[540.0, 960.0]], np.float32),
              np.ones((1, 3), np.float32), np.array([6.0], np.float32))
    grain0, hcfg.grain = hcfg.grain, 0.0          # grain would swamp the halo's tail
    psf = pfr.to_rgb(exposure=scene.exposure, glow_mul=scene.glow_mul,
                     glow2_mul=scene.glow2_mul,
                     streak_mul=scene.streak_mul).astype(np.float32).mean(2)
    hcfg.grain = grain0
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    R = 90
    pa = psf[cy - R:cy + R + 1, cx - R:cx + R + 1]
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(dy, dx)
    aa = ((np.degrees(np.arctan2(dy, dx)) % 180) / 2.5).astype(int) % 72
    ang = np.zeros(72); cnt = np.zeros(72)
    sel = (rr > 10) & (rr < 60)
    np.add.at(ang, aa[sel], pa[sel]); np.add.at(cnt, aa[sel], 1)
    ang /= np.maximum(cnt, 1)
    got["halo_aniso"] = float(ang.max() / max(ang.min(), 1e-6))
    peak = int(np.argmax(ang) * 2.5)
    print(f"  [halo] point-spread function: peak angle {peak} deg, trough "
          f"{int(np.argmin(ang) * 2.5)} deg   (source: peak 0/180 horizontal, trough 90)")

    # -- grain ----------------------------------------------------------------------------
    got["grain_spatial"] = float((fg - uniform_filter(fg, 3)).std())
    got["grain_dark"] = float((fg2 - fg)[fg < 12].std()) if (fg < 12).sum() > 500 else 0.0

    print(f"\n  {a.scene}   {n} frames @{a.fps}fps from t={t0:.1f}s\n")
    print(f"  {'metric':<18}{'render':>10}{'source':>10}   verdict")
    bad = 0
    for key, want in SOURCE.items():
        if key not in got:
            continue
        g = got[key]
        ok = abs(g - want) <= TOL[key]
        bad += not ok
        print(f"  {key:<18}{g:>10.3f}{want:>10.3f}   {'ok  ' if ok else 'OFF '}{g-want:+.2f}")
    print(f"\n  {'MATCH' if not bad else str(bad) + ' metric(s) off'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
