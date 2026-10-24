# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py — render an oscilloscope scene to a vertical mp4.

  python cli.py wind_up
  python cli.py wind_up --preview          # fast: 15 fps, half res
  python cli.py lissajous_tour --seconds 6
  python cli.py list
"""
from __future__ import annotations

import argparse
import os

from osc.config import OUT, RenderConfig
from osc.render import render_scene
from osc.signals import SCENES


def main():
    ap = argparse.ArgumentParser(description="Render an oscilloscope scene to a vertical mp4.")
    ap.add_argument("scene", help="scene name, or 'list'")
    ap.add_argument("--preview", action="store_true", help="fast draft: 15 fps, 540x960")
    ap.add_argument("--seconds", type=float, default=None, help="override duration")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.scene == "list":
        print("scenes:", ", ".join(SCENES))
        return
    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene '{args.scene}'. options: {', '.join(SCENES)}, list")

    cfg = RenderConfig()
    if args.preview:
        cfg.width, cfg.height, cfg.fps = 540, 960, 15
    if args.fps:
        cfg.fps = args.fps

    import inspect
    params = inspect.signature(SCENES[args.scene]).parameters   # works for classes and lambdas
    kw = {}
    if args.seconds is not None and "dur" in params:
        kw["dur"] = args.seconds
    scene = SCENES[args.scene](**kw)
    tag = "_preview" if args.preview else ""
    out_path = args.out or os.path.join(OUT, f"{args.scene}{tag}.mp4")

    print(f"rendering {args.scene} -> {out_path}  ({cfg.width}x{cfg.height} @ {cfg.fps}fps)")
    render_scene(scene, out_path, cfg, seconds=args.seconds)
    print("done:", out_path)


if __name__ == "__main__":
    main()
