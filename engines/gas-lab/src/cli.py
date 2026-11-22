# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cli.py - entry point.

    python cli.py list
    python cli.py single_sohc --dry                    # physics only, warm-up + ramp: is it an ENGINE?
    python cli.py single_sohc --still 15.1 --out look  # one PNG (t is measured from the warm-up start)
    python cli.py single_sohc --preview                # 720x1280 @30, same physics
    python cli.py single_sohc                          # the final: 1440x2560 @60, 10 s
    python cli.py radial --layout                      # solids only at t=0: check the drawing

--dry FIRST, ALWAYS. It runs the whole schedule with no drawing and prints, per cylinder, the
pressure cross-check, the temperature swing, volumetric efficiency, the mass balance and the
reaction ledger - for the whole run and again for the HOLD phase at 3x. Tune against --dry
output; never against renders.

Every clip: a warm-up period at base speed that is not shown, then 10 s in which the crank
accelerates to `--mult` x (default 3) by the midpoint and holds (Clancy, 2026-09-20).
"""
from __future__ import annotations

import argparse
import os
import sys

from gas import catalog, config, render
from gas.draw import Style


def main(argv=None):
    p = argparse.ArgumentParser(description="Gas_Lab - kinetic-theory gas engine (vertical)")
    p.add_argument("engine", nargs="?", default="single_sohc")
    p.add_argument("--dry", action="store_true", help="simulate and measure, no drawing")
    p.add_argument("--still", type=float, default=None, help="one PNG at this time (s)")
    p.add_argument("--layout", action="store_true", help="solids-only PNG at t=0 (no physics)")
    p.add_argument("--preview", action="store_true", help="720x1280 @30 (same physics)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--warm", type=float, default=None, help="warm-up seconds (default: max(period, 10))")
    p.add_argument("--mult", type=float, default=3.0, help="speed multiplier reached at the midpoint")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--ss", type=int, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gain", type=float, default=None, help="gas exposure gain")
    p.add_argument("--out", default=None, help="output name or path")
    a = p.parse_args(argv)

    from gas import pcatalog, pgate
    if a.engine == "list":
        print("Engine scenes")
        print("\n".join(catalog.ENGINES))
        print("\nParts and ledger scenes")
        print("\n".join((*pcatalog.PARTS, *pcatalog.VARIANTS)))
        return 0
    if a.engine in pcatalog.PARTS or a.engine in pcatalog.VARIANTS:
        # THE PARTS SET (2026-09-25): 13 s clips, a field-drawn gas, metal shaded from the
        # solids the gas meets. --still t is CLIP time (0 = first frame, after the warm-up).
        eng = pcatalog.build(a.engine)
        secs = a.seconds if a.seconds != 10.0 else 13.0
        cfg = config.RenderConfig.preview() if a.preview else config.RenderConfig.final()
        if a.substeps:
            cfg.substeps_per_sec = a.substeps
        if a.dry:
            pgate.simulate(eng, cfg, seconds=secs, warm=a.warm)
            return 0
        if a.layout:
            pgate.layout(eng, cfg, a.out or os.path.join(config.OUT, "look", "parts",
                                                         f"layout_{a.engine}.png"))
            return 0
        if a.still is not None:
            out = a.out or os.path.join(config.OUT, "look", "parts", f"{a.engine}_t{a.still:g}.png")
            pgate.render_still(eng, cfg, a.still, out, seconds=secs, warm=a.warm)
            return 0
        out = a.out or os.path.join(config.OUT, "finals", "2026-09-25_parts",
                                    f"{a.engine}{'_preview' if a.preview else ''}.mp4")
        pgate.render_clip(eng, cfg, out, seconds=secs, warm=a.warm)
        return 0
    if a.engine not in catalog.ENGINES:
        print(f"unknown engine {a.engine!r}; try: {', '.join(catalog.ENGINES)}")
        return 2

    eng = catalog.build(a.engine)
    cfg = config.RenderConfig.preview() if a.preview else config.RenderConfig.final()
    if a.width:
        cfg.width = a.width
        cfg.height = int(round(a.width * config.REF_H / config.REF_W))
    if a.fps:
        cfg.fps = a.fps
    if a.ss:
        cfg.ss = a.ss
    if a.substeps:
        cfg.substeps_per_sec = a.substeps
    style = Style()
    if a.gain is not None:
        style.gas_gain = a.gain

    if a.dry:
        render.simulate(eng, cfg, seconds=a.seconds, warm=a.warm, mult=a.mult, seed=a.seed)
        return 0

    if a.layout:
        from gas.draw import Frame
        from PIL import Image
        import numpy as np
        fr = Frame(cfg, style, eng)
        rgb, al = fr.solids(0.0)
        img = (np.clip(rgb * al, 0, 1) * 255 + 0.5).astype(np.uint8)
        out = a.out or os.path.join(config.OUT, "look", f"layout_{a.engine}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(img).save(out)
        print(f"[layout] -> {out}")
        return 0

    if a.still is not None:
        name = a.out or f"{a.engine}_t{a.still:g}"
        out = name if name.lower().endswith(".png") else os.path.join(config.OUT, "look", name + ".png")
        render.render_still(eng, cfg, style, a.still, out, seed=a.seed, seconds=a.seconds,
                            warm=a.warm, mult=a.mult)
        return 0

    name = a.out or (f"{a.engine}_preview" if a.preview else a.engine)
    out = name if os.path.isabs(name) or name.endswith(".mp4") else os.path.join(
        config.OUT, "finals" if not a.preview else "", name + ".mp4")
    render.render_clip(eng, cfg, style, out, seconds=a.seconds, warm=a.warm, mult=a.mult,
                       seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
