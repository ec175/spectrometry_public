# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pcatalog.py - the 2026-09-25 PARTS set: eight car parts that operate, one per final.

Clancy, 2026-09-25: "create 8 finals ... of a mechanical car part that is operating. We have
done this most obviously with car cylinders, but there are a few other critical parts that
should be included ... We need models with better detail, but lets avoid just adding more
parts into each frame."

Each builder returns an engine.Engine with `notes['part'] = True`: boxes for the gas domain
and its reservoirs (world.py, unchanged in kind), plus shaped SOLIDS (solids.Body over
sdf.Solid) for everything that is not a box wall - rotors, housings, vanes, valves - and
circular HOLES for ports that run along an axis out of the page. Every body is drawn from the
grid the gas bounces off (metal.py), so the part you see is the part the gas meets.

All geometry is REFERENCE px (1080x1920, y down), group 0 is the page (no rotation).
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from . import shapes as S
from .engine import BOT, LEFT, RIGHT, TOP, Box, Drive, Engine, Group
from .sdf import Solid
from .solids import Body, Hole, const_pose, spin_pose

OFF = (False, False, False, False)
OPEN_TOP = (True, False, False, False)
OPEN_BOT = (False, True, False, False)

# ---------------------------------------------------------------- materials (sRGB 0-255)
STEEL = (150, 158, 170)
STEEL_DK = (104, 112, 124)
ALU = (92, 100, 112)           # a cast housing, sectioned
IRON = (78, 84, 94)
BRASS = (176, 150, 96)
BRONZE = (150, 112, 70)
COPPER = (184, 110, 70)
RUBBER = (46, 48, 54)
CERAMIC = (226, 222, 212)
DARK = (40, 44, 50)


def _page():
    return [Group("page", 0.0, (540.0, 960.0), (540.0, 960.0))]


def _bolt_ring(s, cx, cy, R, n, r=9.0, a0=0.0):
    """Socket-head bolts as relief on a solid: a boss, a hex recess."""
    for i in range(n):
        a = a0 + 2 * math.pi * i / n
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        s.boss(S.circle(x, y, r), height=3.0, bevel=2.0)
        s.recess(S.rot(S.circle(x, y, r * 0.5, 6), a, (x, y)), depth=3.0, bevel=1.2)
        s.paint(S.circle(x, y, r * 0.5, 6), DARK)


# ====================================================================== 4. ROOTS BLOWER
def roots_blower():
    """A three-lobe Roots supercharger, end-on. Two cycloidal rotors (exactly conjugate:
    epicycloid tips, hypocycloid roots) counter-rotate in a figure-eight housing and carry
    pockets of intake air round the OUTSIDE of each bore, from the inlet at the top to the
    boosted manifold at the bottom. There is no internal compression in a Roots blower: a
    pocket arrives at the outlet at inlet pressure and the manifold BACKFLOWS into it, which is
    why they heat the charge - the gate measures that (outlet T vs inlet T)."""
    rp, n = 175.0, 3
    cxA, cxB, cy = 540.0 - rp, 540.0 + rp, 980.0
    tip = rp * (1 + 1.0 / n)
    bore = tip + 1.0                  # 3 px tip clearance once the rotor is shrunk by 2
    drive = Drive(3.0)
    eng = Engine(name="roots_blower", rev_seconds=3.0, drive=drive, cycle_deg=360.0,
                 fuel_frac=0.0, groups=_page())
    eng.notes.update(part=True, rp=rp, centres=((cxA, cy), (cxB, cy)), tip=tip, bore=bore)
    # ---- the gas domain ----
    ix0, ix1 = 400.0, 680.0           # inlet / outlet throat
    hy0, hy1 = cy - bore - 30.0, cy + bore + 30.0
    hb = eng.add_box(Box(cxA - bore - 4, cxB + bore + 4, hy0, hy1, 0, "duct", "",
                         label="housing"))
    ib = eng.add_box(Box(ix0 - 70, ix1 + 70, -120.0, hy0, 0, "duct", "intake", open=OPEN_TOP,
                         label="inlet"))
    ob = eng.add_box(Box(ix0 - 70, ix1 + 70, hy1, 2060.0, 0, "duct", "exhaust", open=OPEN_BOT,
                         boost=1.45, label="manifold"))
    eng.connect(hb, ib, TOP, ix0, ix1)
    eng.connect(hb, ob, BOT, ix0, ix1)

    # ---- the rotors ----
    prof = S.roots_rotor(rp, n, samples=120)
    rot_sd = []
    for k, cx in enumerate((cxA, cxB)):
        s = Solid((-tip - 4, -tip - 4, tip + 4, tip + 4), h=0.6, bevel=9.0, name=f"rotor{k}")
        s.add(prof).grow(-2.0)
        # relief: a turned face, a raised hub, the shaft and its key, three cored pockets
        s.recess(S.circle(0, 0, rp * 0.80), depth=4.0, bevel=5.0)
        for j in range(n):
            a = 2 * math.pi * j / n + math.pi / n
            s.recess(S.circle(0.47 * rp * math.cos(a), 0.47 * rp * math.sin(a), 0.19 * rp),
                     depth=9.0, bevel=5.0)
            s.paint(S.circle(0.47 * rp * math.cos(a), 0.47 * rp * math.sin(a), 0.15 * rp),
                    (84, 90, 100), feather=6.0)
        s.boss(S.circle(0, 0, 58.0), height=7.0, bevel=4.0)
        s.recess(S.circle(0, 0, 30.0), depth=6.0, bevel=2.0)
        s.paint(S.circle(0, 0, 30.0), STEEL_DK)
        s.boss(S.rect(-7, -38, 7, -26), height=3.0, bevel=1.5)
        s.done(STEEL, brush="radial", spec=0.7, seed=k)
        rot_sd.append(s)
    ratio = (-1.0, 1.0)               # left rotor turns counter-clockwise: pockets go DOWN the outside
    for k, cx in enumerate((cxA, cxB)):
        eng.bodies.append(Body(rot_sd[k], spin_pose(cx, cy, drive, ratio[k], 0.0), z=10,
                               name=f"rotor {'AB'[k]}"))

    # ---- the housing: a sectioned casting with the figure-eight bore and the two throats ----
    H = Solid((20.0, hy0 - 70.0, 1060.0, hy1 + 70.0), h=0.8, bevel=6.0, name="housing")
    H.add(S.rrect(28.0, hy0 - 60.0, 1052.0, hy1 + 60.0, 70.0))
    H.sub(S.circle(cxA, cy, bore, 720)).sub(S.circle(cxB, cy, bore, 720))
    H.sub(S.rect(ix0, hy0 - 80.0, ix1, cy))
    H.sub(S.rect(ix0, cy, ix1, hy1 + 80.0))
    # relief: machined bore lips, the casting's ribs, bolts round each bore
    for cx in (cxA, cxB):
        H.recess(S.circle(cx, cy, bore + 12.0, 720), depth=2.5, bevel=2.0)
        _bolt_ring(H, cx, cy, bore + 34.0, 10, r=8.0, a0=0.3)
    H.done(ALU, brush="cast", sectioned=True, spec=0.25)
    eng.bodies.append(Body(H, const_pose(), z=0, static=True, name="housing"))

    # ---- the ducts: a bell-mouth inlet above, the manifold plenum below ----
    D = Solid((300.0, -130.0, 780.0, hy0 + 2.0), h=0.8, bevel=6.0, name="inlet duct")
    bell = np.concatenate([S.bezier((ix0 - 60, -130), (ix0 - 20, 60), (ix0 + 20, 150), (ix0 + 22, hy0 + 12), 40),
                           S.bezier((ix1 - 22, hy0 + 12), (ix1 - 20, 150), (ix1 + 20, 60), (ix1 + 60, -130), 40)])
    D.add(S.rect(300.0, -130.0, 780.0, hy0 + 2.0)).sub(bell)
    D.done(ALU, brush="cast", sectioned=True, spec=0.25)
    eng.bodies.append(Body(D, const_pose(), z=1, static=True, name="inlet"))
    M = Solid((300.0, hy1 - 2.0, 780.0, 2070.0), h=0.8, bevel=6.0, name="manifold")
    plen = np.concatenate([S.bezier((ix0 + 22, hy1 - 12), (ix0 + 10, hy1 + 160), (ix0 - 40, hy1 + 260), (ix0 - 50, 2070), 40),
                           S.bezier((ix1 + 50, 2070), (ix1 + 40, hy1 + 260), (ix1 - 10, hy1 + 160), (ix1 - 22, hy1 - 12), 40)])
    M.add(S.rect(300.0, hy1 - 2.0, 780.0, 2070.0)).sub(plen)
    M.done(ALU, brush="cast", sectioned=True, spec=0.25)
    eng.bodies.append(Body(M, const_pose(), z=1, static=True, name="manifold"))

    # ---- what the gate measures ----
    eng.probes = [
        ("inlet", ib, (ix0 + 60, 120.0, ix1 - 60, 360.0)),
        ("manifold", ob, (ix0 + 30, hy1 + 420.0, ix1 - 30, hy1 + 620.0)),
    ]
    eng.notes["pocket_probe"] = [(hb, cxA - tip + 10, cy - 60, cxA - rp * 0.7, cy + 60),
                                 (hb, cxB + rp * 0.7, cy - 60, cxB + tip - 10, cy + 60)]
    eng.notes["T_range"] = (260.0, 520.0)
    eng.blurb = "3-lobe Roots supercharger: pockets carried round the outside, manifold at 1.45 bar"
    return eng


def _lazy(mod, fn):
    def f(**kw):
        import importlib
        return getattr(importlib.import_module(f".{mod}", __package__), fn)(**kw)
    return f


PARTS = {
    "four_stroke": _lazy("pcat_engine", "four_stroke"),
    "wankel": _lazy("pcat_rotary", "wankel"),
    "turbocharger": _lazy("pcat_flow", "turbocharger"),
    "roots_blower": roots_blower,
    "scroll_ac": _lazy("pcat_rotary", "scroll_ac"),
    "vane_pump": _lazy("pcat_rotary", "vane_pump"),
    "shock_absorber": _lazy("pcat_flow", "shock_absorber"),
    "fuel_injector": _lazy("pcat_flow", "fuel_injector"),
}
VARIANTS = {"four_stroke_ledger": _lazy("pcat_engine", "four_stroke_ledger"),
            "ledger_revs": _lazy("ledger_set", "ledger_revs"),
            "ledger_mixture": _lazy("ledger_set", "ledger_mixture"),
            "ledger_boost": _lazy("ledger_set", "ledger_boost"),
            "ledger_wankel": _lazy("ledger_set", "ledger_wankel")}
FINALS = list(PARTS)


# N0 is raised for the FIELD's statistics, not the physics: the solver is kernel-launch bound,
# so 0.30 (245k molecules in the blower) costs the same 35 s per simulated second as 0.055
# (measured 2026-09-25), and the density field's noise falls by sqrt(5.5). Calibration-neutral:
# pressure is reported against N0*T0, DSMC acceptance is n_cell/n_ref with n_ref ~ N0, and
# seeding and injection both scale with it.
PARTS_N0 = 0.30
CAP = 900_000


def build(name, **kw):
    C.N0 = PARTS_N0
    eng = (PARTS.get(name) or VARIANTS[name])(**kw)
    C.N0 = float(eng.notes.get("N0", PARTS_N0))
    eng.check()
    return eng
