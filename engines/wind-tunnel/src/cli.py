# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py - render a wind-tunnel scene. Run it from this folder (engines/wind-tunnel/src).

    python cli.py list
    python cli.py aoa_sweep --preview              # 540x960 @30, same physics -> out/aoa_sweep_preview.mp4
    python cli.py aoa_sweep                        # 1080x1920 @60 -> out/aoa_sweep.mp4
    python cli.py vortex_street --film --cmap turbo
    python cli.py stall --still 8 --preview        # a PNG at t = 8 s -> out/stall.png
    python cli.py tri_foil_air_fast --cmap orchid  # a colour variant is a flag, not a new scene

`--preview` halves the frame size and the fps but keeps the LATTICE identical, so a preview
shows the same flow at the same instants as the final — only smaller and choppier.
"""
from __future__ import annotations

import argparse
import os

from wt import colormap
from wt.config import OUT, RenderConfig
from wt.render import render_scene, render_stills
from wt.scenes import SCENES


def build_config(args, cls):
    cfg = RenderConfig()
    for k, v in cls.OVERRIDES.items():        # scene defaults ...
        setattr(cfg, k, v)
    for k in ("flow", "field", "cmap", "scale", "re", "u0", "steps", "settle", "solver",
              "vmax", "gamma", "saturation", "brightness",
              "streak_gain", "streak_span", "streak_sigma",
              "streak_speed_gain", "csm", "seed", "perturb", "glow", "ambient", "bow",
              "exposure"):
        v = getattr(args, k, None)            # ... overridden by anything explicitly passed
        if v is not None:
            setattr(cfg, k, v)
    if args.fps is not None:
        cfg.fps = args.fps
    if args.streaks is not None:
        cfg.n_streaks = args.streaks
    if args.no_streaks:
        cfg.n_streaks = 0
        cfg.streak_gain = 0.0
    cfg.film = bool(args.film)
    # COPY before mutating: a scene that puts `extras` in its OVERRIDES hands over the class-level
    # dict itself, so writing into it here would persist into every later construction of that
    # scene in the same process (dbg.py, the checks, a batch lane).
    cfg.extras = dict(cfg.extras)
    cfg.extras["hud"] = bool(args.hud)
    if args.hq:
        cfg.extras["quality"] = "high"
    if args.legend:                           # opt-in on top of whatever the scene asked for
        cfg.legend = True
    if args.no_legend:
        cfg.legend = False
    if args.fbd:
        cfg.fbd = True
    if args.no_fbd:
        cfg.fbd = False
    if args.axes:
        cfg.axes = True
    if args.no_axes:
        cfg.axes = False
    # Explicit frame size, for the 16:9 stock renders. Applied BEFORE the preview halving so
    # `--width 1920 --height 1080 --preview` gives a true half-size 16:9 frame rather than a
    # halved 9:16 one. The lattice follows from width/height/scale (config.nx/ny), and `--flow
    # right` is what puts the streamwise axis across the frame - the two have to be set
    # together or a 16:9 frame gets a stream running down its short side.
    if args.width:
        cfg.width = args.width
    if args.height:
        cfg.height = args.height
    if args.preview:
        cfg.width, cfg.height = cfg.width // 2, cfg.height // 2
        cfg.scale = cfg.scale / 2.0           # same lattice, half the pixels
        if args.fps is None:
            cfg.fps = 30
    return cfg


def main():
    ap = argparse.ArgumentParser(description="2-D wind tunnel (lattice-Boltzmann) video engine")
    ap.add_argument("scene", help="scene name, or 'list'")
    ap.add_argument("-o", "--out", default=None, help="output path (default out/<scene>.mp4)")
    ap.add_argument("--preview", action="store_true", help="half res / 30 fps, same physics")
    ap.add_argument("--width", type=int, default=None,
                    help="frame width in px (use with --height for 16:9 stock renders)")
    ap.add_argument("--height", type=int, default=None, help="frame height in px")
    ap.add_argument("--seconds", type=float, default=None, help="override scene duration")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--still", default=None, metavar="T[,T...]",
                    help="render PNGs at these times (one sim run) instead of a video")
    # format / look
    ap.add_argument("--flow", choices=["up", "down", "right", "left"], default=None,
                    help="streamwise direction on screen (default up: fills the 9:16 frame). "
                         "'left' is 'right' rotated 180 deg - for a body that faces RIGHT into "
                         "the oncoming stream")
    ap.add_argument("--field", choices=["speed", "vort", "pressure", "stag", "mach",
                                        "schlieren"],
                    default=None,
                    help="'stag' = stagnation pressure p + rho*u^2/2, the field to use for a "
                         "pressure-driven machine ('pressure' alone hides the plume, which is "
                         "where the chamber's pressure has GONE). 'mach'/'schlieren' need "
                         "--solver cns")
    ap.add_argument("--solver", choices=["lbm", "cns"], default=None,
                    help="lbm = D2Q9 lattice-Boltzmann (default); cns = compressible "
                         "Navier-Stokes (real gamma=1.4 gas, captures shocks). NOTE u0 means a "
                         "lattice velocity under lbm and a MACH NUMBER under cns")
    ap.add_argument("--cmap", choices=colormap.names(), default=None)
    ap.add_argument("--vmax", type=float, default=None, help="field normalisation (x u0)")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--sat", dest="saturation", type=float, default=None,
                    help="colormap chroma (1 = the shipped palettes; 0.7 takes the neon out)")
    ap.add_argument("--bright", dest="brightness", type=float, default=None,
                    help="colormap luminance scale (1 = untouched). Acts on the FIELD only, so "
                         "the white tracer dashes keep their contrast")
    ap.add_argument("--streaks", type=int, default=None, help="tracer count")
    ap.add_argument("--streak-gain", dest="streak_gain", type=float, default=None)
    ap.add_argument("--streak-span", dest="streak_span", type=float, default=None,
                    help="dash length, in frames of travel")
    ap.add_argument("--streak-sigma", dest="streak_sigma", type=float, default=None)
    ap.add_argument("--streak-speed-gain", dest="streak_speed_gain", type=float, default=None)
    ap.add_argument("--no-streaks", action="store_true")
    ap.add_argument("--hud", action="store_true", help="burn in the scene title / readout")
    ap.add_argument("--legend", action="store_true",
                    help="burn in the colour-bar speed key (top left); some scenes default it on")
    ap.add_argument("--no-legend", dest="no_legend", action="store_true",
                    help="suppress a scene's built-in colour-bar key")
    ap.add_argument("--fbd", action="store_true",
                    help="burn in the free-body diagram (labelled force arrows) for a scene that "
                         "supplies arrows(); surf_wave defaults it on")
    ap.add_argument("--no-fbd", dest="no_fbd", action="store_true",
                    help="suppress a scene's built-in free-body diagram")
    ap.add_argument("--axes", action="store_true",
                    help="burn in the instrument block (scrolling distance ruler + time bar) "
                         "for a scene that supplies axes()")
    ap.add_argument("--no-axes", dest="no_axes", action="store_true",
                    help="suppress a scene's built-in instrument block")
    # solver
    ap.add_argument("--scale", type=float, default=None, help="screen px per lattice cell")
    ap.add_argument("--re", type=float, default=None, help="Reynolds number")
    ap.add_argument("--u0", type=float, default=None, help="inlet speed (lattice units, <=0.12)")
    ap.add_argument("--steps", type=float, default=None, help="LBM steps per frame at 60 fps")
    ap.add_argument("--settle", type=float, default=None, help="seconds of flow before frame 0")
    ap.add_argument("--csm", type=float, default=None, help="Smagorinsky constant")
    ap.add_argument("--perturb", type=float, default=None,
                    help="symmetry-breaking noise during settling, as a fraction of u0")
    # film
    ap.add_argument("--film", action="store_true", help="CRT 'filmed off a screen' pass")
    ap.add_argument("--hq", action="store_true",
                    help="high-quality encode: NVENC p7/cq14/90M (or x264 crf12 slow). These "
                         "clips are all fine streak texture, so the default cq19 spends its "
                         "bits smoothing exactly what the picture is made of.")
    ap.add_argument("--glow", type=float, default=None, help="film halation boost")
    ap.add_argument("--exposure", type=float, default=None,
                    help="film pre-exposure (raise to blow the field out, lower to tame it)")
    ap.add_argument("--ambient", type=float, default=None)
    ap.add_argument("--bow", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.scene == "list":
        for name, cls in SCENES.items():
            print(f"  {name:<15} {cls.duration:>5.1f}s  {cls.__doc__.strip().splitlines()[0]}")
        print("\n  colormaps:", ", ".join(colormap.names()))
        return
    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene {args.scene!r}; try: {', '.join(SCENES)}")

    cls = SCENES[args.scene]
    cfg = build_config(args, cls)
    scene = cls(cfg)

    if args.still is not None:
        times = [float(x) for x in str(args.still).split(",") if x.strip()]
        out = args.out or os.path.join(OUT, f"{args.scene}.png")
        render_stills(scene, cfg, times, out)
        return

    suffix = "_preview" if args.preview else ""
    out = args.out or os.path.join(OUT, f"{args.scene}{suffix}.mp4")
    print(f"{args.scene}: {cfg.width}x{cfg.height} @{cfg.fps}fps, flow={cfg.flow}, "
          f"solver={cfg.solver}, Re={cfg.re:g}, field={cfg.field}/{cfg.cmap}, "
          f"film={'on' if cfg.film else 'off'}")
    render_scene(scene, out, cfg, seconds=args.seconds)


if __name__ == "__main__":
    main()

