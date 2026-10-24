# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py - entry point.

    python cli.py list
    python cli.py escape_rings --dry                    # physics only: does it do the brief?
    python cli.py escape_rings --preview --seconds 20   # 540x960 @30
    python cli.py escape_rings                          # 1080x1920 @60

No song ships with this engine - with no `--song`, every scene renders silently but for its
bounce/event voices. Pass `--song PATH` to a file of your own to hear the escape-triggered
progressive playback `escape_rings` was built to show off.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys

from sim import config, render, scenes


def main(argv=None):
    p = argparse.ArgumentParser(description="shape-physics - simple shape physics shorts")
    p.add_argument("scene", nargs="?", default="escape_rings")
    p.add_argument("--preview", action="store_true", help="540x960 @30 (same physics)")
    p.add_argument("--dry", action="store_true", help="simulate only, no render")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--ss", type=int, default=None, help="supersample factor (AA)")
    p.add_argument("--substeps", type=int, default=None, help="physics steps per second")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--song", default=None, help="a path to your own audio file (none = silent)")
    p.add_argument("--song-start", type=float, default=0.0,
                   help="where in the song the first slice starts")
    p.add_argument("--slice", type=float, default=0.5, help="seconds played per event")
    p.add_argument("--r0", type=float, default=26.0, help="first ball radius (world px)")
    p.add_argument("--shrink", type=float, default=0.78, help="radius factor per generation")
    p.add_argument("--max-balls", type=int, default=20, dest="max_balls")
    p.add_argument("--bounce-gain", type=float, default=0.13, dest="bounce_gain",
                   help="bounce blip level (0 = off)")
    p.add_argument("--bounce-duck", type=float, default=0.55, dest="bounce_duck",
                   help="blip level while the song is sounding")
    p.add_argument("--r-min", type=float, default=7.0, dest="r_min")
    p.add_argument("--split", type=int, default=2, help="balls spawned per ball that exits")
    p.add_argument("--split-min-r", type=float, default=9.5, dest="split_min_r",
                   help="below this CHILD radius an exit spawns 1, not `split` (bounds the "
                        "population - see sim/world.py)")
    p.add_argument("--silent", action="store_true", help="skip the audio pass")
    p.add_argument("--out", default=None, help="output name or path")
    a = p.parse_args(argv)

    if a.scene == "list":
        for k, f in scenes.SCENES.items():
            try:
                b = f(seconds=1).blurb
            except Exception:
                b = ""
            tag = "  (library)" if k in scenes.LIBRARY else ""
            print(f"{k:16s} {b}{tag}")
        return 0
    if a.scene not in scenes.SCENES:
        print(f"unknown scene {a.scene!r}; try: {', '.join(scenes.SCENES)}")
        return 2

    factory = scenes.SCENES[a.scene]
    kw = dict(seconds=a.seconds, seed=a.seed, song=a.song, song_start=a.song_start,
              slice_len=a.slice, r0=a.r0, shrink=a.shrink, max_balls=a.max_balls,
              r_min=a.r_min, split=a.split, split_min_r=a.split_min_r,
              bounce_gain=a.bounce_gain, bounce_duck=a.bounce_duck)
    sig = inspect.signature(factory).parameters
    scene = factory(**{k: v for k, v in kw.items() if k in sig})

    cfg = config.RenderConfig.preview() if a.preview else config.RenderConfig()
    if a.width:
        cfg.width = a.width
        cfg.height = int(round(a.width * config.REF_H / config.REF_W))
    if a.fps:
        cfg.fps = a.fps
    if a.ss:
        cfg.ss = a.ss
    if a.substeps:
        cfg.substeps_per_sec = a.substeps

    if a.dry:
        render.simulate(scene, cfg)
        return 0

    name = a.out or (f"{a.scene}_preview" if a.preview else a.scene)
    out = name if os.path.isabs(name) or name.endswith(".mp4") else os.path.join(
        config.OUT, name + ".mp4")
    render.render_scene(scene, cfg, out, silent=a.silent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
