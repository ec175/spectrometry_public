# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cli.py - dry | still | contact | render.

⚠ THE GATE IS A SUBCOMMAND, not a `--dry` flag - the Plume_Lab / Fluid_Studio /
Mechanics_Board / Particle_Fields convention, not the Field_Lines / Shape_Physics one.

    py cli.py dry                      # the gate. run it before anything that writes an mp4
    py cli.py contact                  # one PNG contact sheet of the whole choreography
    py cli.py still 0 4.5 9 14 19      # PNGs at those seconds
    py cli.py render                   # 1080x1920 @60, 20 s - the deliverable
    py cli.py render --tier 720        # 720x1280 @30, the proof pass
    py cli.py render --tier preview    # 540x960 @30, for a fast look
    py cli.py render --seconds 3       # a short encode smoke test
    py cli.py dry --electrons lit      # the flash without the cloud deformation
    py cli.py flowtest                 # the zero-drift equality check. RUN AFTER TOUCHING flow.py
    py cli.py bondtest                 # THE BOND CHECK. RUN BEFORE RENDERING ANY bond_* SCENE
    py cli.py moltest                  # THE MOLECULE CHECK. RUN BEFORE ANY mol_* SCENE
    py cli.py render --flow tide       # a different treatment of the grid

⚠ `bondtest` IS NOT OPTIONAL FOR THE BOND SET, and it is the one place in this project where
the gate cannot stand in for a test. `bond_sigma_s` and `bond_sigma_star_s` are the same two
atoms at the same separation with ONE SIGN flipped: they have the same framing, the same
exposure, the same motion and the same field coverage, so `dry` passes both and cannot tell you
which is which. `bondtest` asserts the thing the names claim - constructive at the overlap point
for a bonding scene, a node there for an antibonding one. See `orb/bond.py`.

`--flow none|pulse|weave|spin|tide` overrides the scene's grid treatment (`orb/flow.py`), and
`--electrons off|lit|pull` overrides the scene. The module default is `off` as of 2026-09-01:
deforming the orbital around a random lit point makes the SHAPE - which is the subject - a
function of an event, so the flagship walk no longer does it. Every named scene in the registry
still asks for `pull` explicitly, so the 2026-08-29 stock catalogue is unchanged.

`py` here means `..\\Particle_Fields\\.venv\\Scripts\\python.exe`. This project has no venv of
its own on purpose; see `orb/_pf.py`.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orb import render as R                                        # noqa: E402
from orb import harmonics as _H                                    # noqa: E402
from orb import scene as S                                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog="orbital")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="the scene registry")
    sub.add_parser("flowtest", help="flow.py's zero-drift equality check")
    sub.add_parser("bondtest", help="assert every bond_* scene is the bond its name claims")
    sub.add_parser("moltest", help="assert the molecular scenes' chemistry, geometry and paths")

    d = sub.add_parser("dry", help="the pre-render gate")
    d.add_argument("scene", nargs="?", default=None)
    d.add_argument("--tier", default="hd")
    d.add_argument("--seconds", type=float, default=None)
    d.add_argument("--electrons", choices=S._MODES, default=None)
    d.add_argument("--flow", default=None)
    d.add_argument("-n", type=int, default=7)

    s = sub.add_parser("still", help="PNGs at the given seconds")
    s.add_argument("times", nargs="+", type=float)
    s.add_argument("--scene", default=None)
    s.add_argument("--tier", default="hd")
    s.add_argument("--seconds", type=float, default=None)
    s.add_argument("--electrons", choices=S._MODES, default=None)
    s.add_argument("--flow", default=None)

    c = sub.add_parser("contact", help="one contact sheet across the whole clip")
    c.add_argument("-n", type=int, default=12)
    c.add_argument("--scene", default=None)
    c.add_argument("--tier", default="hd")
    c.add_argument("--seconds", type=float, default=None)
    c.add_argument("--electrons", choices=S._MODES, default=None)
    c.add_argument("--flow", default=None)

    r = sub.add_parser("render", help="write the mp4")
    r.add_argument("scene", nargs="?", default=None)
    r.add_argument("--tier", default="hd")
    r.add_argument("--seconds", type=float, default=None)
    r.add_argument("--electrons", choices=S._MODES, default=None)
    r.add_argument("--flow", default=None)
    r.add_argument("--out", default=None)

    a = ap.parse_args()
    # ⚠ BEFORE ANYTHING IS BUILT. `build_mesh`, `plan` and `fit_scales` all read scene.py's
    # module globals at call time, so a scene that is applied after the first build silently
    # renders the previous one.
    if a.cmd == "flowtest":
        from orb import flow as _FL
        sys.exit(0 if _FL.selftest() else 1)
    if a.cmd == "bondtest":
        from orb import bond as _BD
        sys.exit(0 if _BD.selftest() else 1)
    if a.cmd == "moltest":
        from orb import molecule as _MD
        sys.exit(0 if _MD.selftest() else 1)
    if a.cmd == "list":
        for k, v in S.SCENES.items():
            n = len(v.get("states", [])) or len(_H.STATES)
            b = v.get("bond")
            tag = (f"  BOND (l,m)={b['state']} phase=({b.get('phase_a', 1):+d},"
                   f"{b.get('phase_b', 1):+d}) {b.get('kind', '?')}") if b else ""
            print(f"  {k:18s} {n:2d} states  hold={v.get('hold', S.HOLD):.2f}  "
                  f"electrons={v.get('electrons', S.ELECTRONS):4s}  "
                  f"flow={v.get('flow', S.FLOW)}{tag}")
        return
    # ⚠ CONFIGURE UNCONDITIONALLY, and this is a fix. The old test was `if scene or seconds`,
    # so `--electrons lit` with no scene named silently did nothing - the flag parsed, and the
    # clip rendered with the module default. `configure(None)` with no overrides is a no-op
    # that reassigns each global to the value it already holds, so there is nothing to guard.
    S.configure(getattr(a, "scene", None),
                seconds=getattr(a, "seconds", None),
                electrons=getattr(a, "electrons", None),
                flow=getattr(a, "flow", None))
    cfg = R.config(a.tier)

    if a.cmd == "dry":
        sys.exit(0 if R.dry(cfg, n=a.n) else 1)
    if a.cmd == "still":
        R.stills(cfg, a.times)
        return
    if a.cmd == "contact":
        times = [S.DURATION * i / float(a.n - 1) - (1e-3 if i == a.n - 1 else 0.0)
                 for i in range(a.n)]
        paths = R.stills(cfg, times, prefix="contact")
        _sheet(paths, os.path.join(R.OUT, "contact_sheet.png"))
        return
    if a.cmd == "render":
        R.render(cfg, out=a.out, seconds=a.seconds)


def _sheet(paths, out, cols=6, w=300):
    """Tile the stills. Particle_Fields S7 and Lattice_Grid S5 both say the same thing in
    different words - bulk statistics cannot see structure - so LOOKING has to be one command."""
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    h = int(round(w * ims[0].height / ims[0].width))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h), Image.LANCZOS), ((i % cols) * w, (i // cols) * h))
    sheet.save(out)
    print("[contact]", out)


if __name__ == "__main__":
    main()
