# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cli.py - the entry point.

    python cli.py list
    python cli.py dry   <scene> [overrides]     # ALWAYS before a render
    python cli.py still <scene> [-o path]
    python cli.py render <scene> [overrides]

Overrides are Cfg field names: --seconds 3 --mach-e 2.4 --field mach --scale 0.5 ...
"""
from __future__ import annotations

import argparse
import os
import sys

from plume import scenes
from plume.render import Scene

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

_OVERRIDES = [
    ("--seconds", float), ("--fps", int), ("--width", int), ("--height", int),
    ("--scale", float), ("--t-end", float), ("--cfl", float),
    ("--mach-e", float), ("--pr-e", float), ("--tr-e", float), ("--gamma", float),
    ("--r-exit", float), ("--nozzle-px", int), ("--t-ramp", float),
    ("--field", str), ("--lut", str), ("--lut-gamma", float), ("--norm", str),
    ("--crf", int), ("--active-start", int),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cli.py")
    ap.add_argument("cmd", choices=["list", "dry", "still", "render"])
    ap.add_argument("scene", nargs="?")
    ap.add_argument("-o", "--out")
    # Several taps on ONE causal run, as Wind_Tunnel's `--still T,T` is - not one solve each.
    ap.add_argument("--at", default="1.0",
                    help="still only: comma-separated fractions of the clip, e.g. 0.002,0.5")
    ap.add_argument("--no-fused", action="store_true", help="vectorised HLLC instead of the kernel")
    ap.add_argument("--no-active", action="store_true", help="disable the growing active box")
    ap.add_argument("--no-nvenc", action="store_true")
    for flag, typ in _OVERRIDES:
        ap.add_argument(flag, type=typ, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for k in sorted(scenes.REGISTRY):
            doc = (scenes.REGISTRY[k].__doc__ or "").strip().splitlines()[0]
            print(f"  {k:18s} {doc}")
        return 0
    if not a.scene:
        ap.error("scene required")

    over = {f.lstrip("-").replace("-", "_"): getattr(a, f.lstrip("-").replace("-", "_"))
            for f, _ in _OVERRIDES}
    if a.no_fused:
        over["fused"] = False
    if a.no_active:
        over["active"] = False
    if a.no_nvenc:
        over["nvenc"] = False
    cfg = scenes.build(a.scene, **over)

    if a.cmd == "dry":
        return 0 if Scene(cfg).dry() else 1
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "still":
        fracs = [float(s) for s in str(a.at).split(",") if s.strip()]
        stem = a.out or os.path.join(OUT, f"{cfg.name}.png")
        base, ext = os.path.splitext(stem)
        at = {f: (stem if len(fracs) == 1 else f"{base}_{int(round(f * cfg.frames)):04d}{ext}")
              for f in fracs}
        Scene(cfg).stills(at)
        return 0
    Scene(cfg).render(a.out or os.path.join(OUT, f"{cfg.name}.mp4"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
