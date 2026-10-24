# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py - the entry point.

    python cli.py list
    python cli.py <scene> --dry                     density report, no drawing (~2 s)
    python cli.py <scene> --still 0,4,7.5,12        PNGs from the same scene clock
    python cli.py <scene> --preview                 540x960 @30
    python cli.py <scene>                           1080x1920 @60 -> out\\<scene>.mp4

--dry FIRST, ALWAYS. It answers the question a render cannot afford to answer twice: how many
elements are actually lit per frame. Both failure modes of this format - an empty grid and a
bloomed-out wash - are visible in that number long before they are visible in a render.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lg.config import OUT, RenderConfig                              # noqa: E402
from lg.render import dry_report, render_scene, render_stills        # noqa: E402
from lg.scenes import SCENES                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--still", default="")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.scene == "list":
        for k, v in SCENES.items():
            doc = (v.__doc__ or "").strip().splitlines()[0]
            print(f"  {k:16s} {v.duration:4.0f}s  {doc}")
        return

    if a.scene not in SCENES:
        raise SystemExit(f"unknown scene {a.scene!r}; try `cli.py list`")

    cfg = RenderConfig.preview() if a.preview else RenderConfig()
    if a.width:
        cfg.width = a.width
        cfg.height = int(round(a.width * 1920 / 1080))
    if a.fps:
        cfg.fps = a.fps

    scene = SCENES[a.scene]()
    # A scene may need output-stage settings that live on RenderConfig - grain in particular,
    # which is a look decision but is applied in to_rgb.
    for k, v in getattr(scene, "cfg_overrides", {}).items():
        setattr(cfg, k, v)
    if a.seconds:
        scene.duration = a.seconds

    if a.dry:
        print(f"[dry] {scene.name}  {scene.duration:.1f}s")
        dry_report(scene, cfg)
        return

    if a.still:
        times = [float(x) for x in a.still.split(",") if x.strip()]
        render_stills(scene, cfg, times,
                      prefix=f"{scene.name}{'_prev' if a.preview else ''}")
        return

    out = a.out or os.path.join(
        OUT, f"{scene.name}{'_preview' if a.preview else ''}.mp4")
    render_scene(scene, cfg, out=out)


if __name__ == "__main__":
    main()
