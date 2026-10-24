# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""Run a scene to completion at preview resolution and report the FIRST thing to break: the
lattice (max|u| blowing up / NaN) or a body's own state. Nothing is drawn or encoded.

    python tools/dbg.py falling_discs            # run from engines/wind-tunnel/src
    python tools/dbg.py tri_foil re=3000         # any RenderConfig field as key=value

Run this before rendering any scene that moves: it must print COMPLETED. It only answers "did
the lattice survive?" - a stable solve can still be doing the wrong thing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wt.config import RenderConfig
from wt.scenes import SCENES
from wt.render import Tunnel

name = sys.argv[1] if len(sys.argv) > 1 else "falling_discs"
cfg = RenderConfig()
cls = SCENES[name]
for k, v in cls.OVERRIDES.items():
    setattr(cfg, k, v)
# Preview res: half the pixels AND half the scene's OWN `scale`, so the lattice -- and therefore
# the physics -- is identical to the final. Derived, not pinned at 1.5: that constant is the right
# half only for a scene that leaves `scale` at the 3.0 default, and it would silently hand any
# scene with its own resolution (tri_foil_air runs 2.0) a coarser lattice than it ships with,
# i.e. gate the wrong solve. Identical to the old behaviour for every scale=3.0 scene.
cfg.width, cfg.height, cfg.fps = 540, 960, 30
cfg.scale = cfg.scale / 2.0
for k, v in dict(a.split("=") for a in sys.argv[2:]).items():
    setattr(cfg, k, type(getattr(cfg, k))(v))
sc = cls(cfg)
# Only DYNAMIC (free-body) scenes carry a BodySystem. A static scene -- aoa_sweep,
# stall, flutter, vortex_street, tandem, shape_tour -- positions its bodies from
# bodies(t) and has no `.sys`, so every body readout below is conditional. Without
# this the tool died on line 1 for half the registry.
bodysys = getattr(sc, "sys", None)
kind = "dynamic" if bodysys is not None else "static"
print(f"{name} [{kind}]: nx={cfg.nx} ny={cfg.ny} u0={cfg.u0} re={cfg.re} "
      f"csm={cfg.csm} steps={cfg.steps}")
for b in (bodysys.bodies if bodysys is not None else []):
    print(f"  size={b.size:5.1f} rho={b.density:6.3f} m={b.mass:8.1f} r={b.radius:5.1f} "
          f"x={b.x:6.1f} y={b.y:6.1f} born={b.born:.2f}")
tun = Tunnel(sc, cfg)
tun.settle(cfg.settle, progress=False)
print(f"after settle max|u|={tun.lbm.health():.4f}")
n = int(cls.duration * cfg.fps)
peak = 0.0
for i in range(n):
    tun.advance(i / cfg.fps)
    h = tun.lbm.health()
    peak = max(peak, h if np.isfinite(h) else peak)
    # the abort threshold belongs to the SOLVER: 0.45 for the LBM (lattice sound speed 0.577),
    # but under CNS a velocity of 1 IS Mach 1 and a supersonic pocket is the physics, not a
    # failure. Sits a little above render_scene's guard so dbg reports slightly further in.
    lim = 1.2 * getattr(tun.lbm, "health_limit", 0.45)
    bad = (not np.isfinite(h)) or h > lim or (
        bodysys is not None and any(
            not np.isfinite(b.x + b.y + b.vx + b.vy) for b in bodysys.bodies))
    if i % 30 == 0 or bad:
        xs = (" ".join(f"{b.x / cfg.nx:.2f}" for b in bodysys.active(i / cfg.fps))
              if bodysys is not None else "")
        extra = f"  x/nx: {xs}" if xs else ""
        # flush: python block-buffers stdout when it is redirected to a file, so a run as
        # `dbg.py scene > log.txt` would otherwise stay empty until it finished, and a slow
        # gate would look exactly like a dead one.
        print(f"  f{i:4d} t={i/cfg.fps:5.2f} max|u|={h:.4f}{extra}"
              f"  [{100*i//n}%]", flush=True)
    if bad:
        print("  ^^ BROKE HERE")
        break
else:
    xs = (" ".join(f"{b.x / cfg.nx:.2f}" for b in bodysys.bodies)
          if bodysys is not None else "")
    extra = f", final x/nx: {xs}" if xs else ""
    print(f"COMPLETED {n} frames, peak max|u|={peak:.4f}{extra}")
