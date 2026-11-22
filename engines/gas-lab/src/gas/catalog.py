# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""catalog.py - the engines, as box graphs. Physics only: what each one looks like is looks.py.

Every builder returns an `engine.Engine`. All geometry is in REFERENCE px (1080x1920, y down),
in each cylinder GROUP's local frame - the cylinder pointing up with its crank below it - and
the group carries the rotation into the world (engine.Group).

THE EIGHT (2026-09-20), each a different MECHANISM, not a palette swap:

| engine            | what is different                                                   |
|-------------------|---------------------------------------------------------------------|
| single_sohc       | the reference four-stroke: one overhead cam, finger followers        |
| dohc              | oversquare, two cams over bucket tappets, short stroke               |
| diesel            | no plug: air only, fuel INJECTED near TDC, lights by compression     |
| two_stroke        | no valves: wall ports the piston uncovers, the CRANKCASE is the pump |
| opposed_piston    | two pistons, two cranks, one bore; uniflow ports at each end         |
| flathead          | side valves in the block, an L-shaped chamber beside the bore        |
| v_twin            | two cylinders 50 deg apart on one crankpin: uneven firing            |
| radial            | five cylinders round one crank; master rod + articulated rods        |

LOOPING: every engine here runs at rev = 5 s so that 10 s is exactly one four-stroke cycle
(or two two-stroke ones). The clip's period IS the engine's period; render.py warms up a full
period first and blends the gas across the seam.

VALVE TIMING (the 2026-09-20 correction). The first batch opened the exhaust 12 deg before BDC
and shut the intake 16 deg after it - the sequence was right but the events were ~40 deg too
close to the dead centres, and engine-literate viewers said so. Real engines: exhaust opens
40-60 deg BBDC (blowdown while the piston is still descending), intake closes 40-60 deg ABDC.
"""
from __future__ import annotations

import math

from . import config as C
from .engine import (BOT, LEFT, RIGHT, TOP, Box, Drive, Engine, Group, Igniter, Injector,
                     Piston, RadialKinematics, Valve, clock, slider_crank)

REV = 5.0                       # seconds per revolution: 10 s = one four-stroke cycle
IVO, IVC = -12.0, 230.0         # intake  opens 12 BTDC, closes 50 ABDC
EVO, EVC = 490.0, 732.0         # exhaust opens 50 BBDC, closes 12 ATDC
SPARK = 340.0
OFF = (False, False, False, False)
OPEN_TOP = (True, False, False, False)
OPEN_BOT = (False, True, False, False)


def _flame(bore, rev=None):
    """A front that crosses the bore in 27 crank degrees, whatever the speed: px per degree."""
    return bore / 27.0


# ====================================================================== singles
def _four_stroke_single(name, bore, crank, R, L, clearance, pocket_w, pocket_h, duct_h,
                        rev=REV, fuel_frac=0.30, q_rxn=7.5e7, plug=True, risers="up",
                        timing=(IVO, IVC, EVO, EVC), angle=0.0):
    """One cylinder, two valves in the head, ducts leaving the valve pockets sideways and
    rising off the top of the frame. `risers` 'up' puts both reservoir ends off-frame."""
    x0, x1 = bore
    cx, cy = crank
    drive = Drive(rev)
    tdc = cy - (R + L)
    deck = tdc - clearance
    eng = Engine(name=name, rev_seconds=rev, drive=drive, cycle_deg=720.0, fuel_frac=fuel_frac,
                 q_rxn=q_rxn, groups=[Group("main", angle, (540.0, 960.0), (540.0, 960.0))])
    clk = clock(drive, 720.0, 0.0)
    eng.pistons.append(Piston(0, slider_crank(cy, R, L, drive), True, 190.0, "piston"))
    mid = 0.5 * (x0 + x1)
    gap = 0.5 * (x1 - x0) - pocket_w      # pocket edge distance from the bore centre
    # the two valve pockets sit symmetric about the bore centre, `gap`/2 apart at the middle
    inner = 60.0
    ip = (mid - inner - pocket_w, mid - inner)
    ep = (mid + inner, mid + inner + pocket_w)
    assert ip[0] > x0 + 20 and ep[1] < x1 - 20, "pockets must sit inside the bore"
    cyl = eng.add_box(Box(x0, x1, deck, cy, 0, "cyl", "", 0, -1, "crown", OFF, 1.0, [], "cylinder"))
    pin = eng.add_box(Box(ip[0], ip[1], deck - pocket_h, deck, 0, "duct", "intake", label="intake pocket"))
    pex = eng.add_box(Box(ep[0], ep[1], deck - pocket_h, deck, 0, "duct", "exhaust", label="exhaust pocket"))
    eng.connect(cyl, pin, TOP, ip[0], ip[1], gate=0)
    eng.connect(cyl, pex, TOP, ep[0], ep[1], gate=1)
    # ducts out to the sides, then risers off the top edge
    dw = pocket_h                               # duct height = pocket height (no choke)
    din = eng.add_box(Box(40.0, ip[0], deck - dw, deck, 0, "duct", "intake", label="intake duct"))
    dex = eng.add_box(Box(ep[1], 1040.0, deck - dw, deck, 0, "duct", "exhaust", label="exhaust duct"))
    eng.connect(pin, din, LEFT, deck - dw, deck)
    eng.connect(pex, dex, RIGHT, deck - dw, deck)
    rw = pocket_h + 10
    rin = eng.add_box(Box(40.0, 40.0 + rw, -200.0, deck - dw, 0, "duct", "intake", open=OPEN_TOP,
                          label="intake riser"))
    rex = eng.add_box(Box(1040.0 - rw, 1040.0, -200.0, deck - dw, 0, "duct", "exhaust",
                          open=OPEN_TOP, label="exhaust riser"))
    eng.connect(din, rin, TOP, 40.0, 40.0 + rw)
    eng.connect(dex, rex, TOP, 1040.0 - rw, 1040.0)
    ivo, ivc, evo, evc = timing
    eng.valves.append(Valve(clk, ivo, ivc, "cam", None, 720.0, "intake"))
    eng.valves.append(Valve(clk, evo, evc, "cam", None, 720.0, "exhaust"))
    if plug:
        eng.igniters.append(Igniter((cyl,), (mid, deck + 12.0), clk, SPARK, 0.0, False,
                                    _flame(x1 - x0, rev), label="plug"))
    eng.notes.update(dict(bore=bore, crank=crank, R=R, L=L, deck=deck, tdc=tdc, bdc=cy - (L - R),
                          pockets=(ip, ep), pocket_h=pocket_h, duct_h=dw, riser_w=rw, mid=mid,
                          cyl=cyl, clearance=clearance, plug=(mid, deck + 12.0)))
    return eng


def single_sohc():
    """The reference. 560 x 520, 5.0:1 by area (what 10:1 does in a real engine: gamma = 2).
    One overhead cam between the valves, finger followers - the lobe you see is drawn FROM the
    lift curve the physics gates on."""
    e = _four_stroke_single("single_sohc", (260.0, 820.0), (540.0, 1520.0), 260.0, 560.0,
                            130.0, 190.0, 190.0, 190.0)
    e.blurb = "the reference four-stroke: one overhead cam, 5.0:1, spark 20 deg BTDC"
    e.notes["head"] = "sohc"
    return e


def dohc():
    """Oversquare: 620 bore, 400 stroke. Two cams directly over bucket tappets, the modern
    high-revving layout. Same 5:1 area ratio, so the difference you see is the SHAPE of the
    charge - wide and shallow - and the two lobes."""
    e = _four_stroke_single("dohc", (230.0, 850.0), (540.0, 1560.0), 200.0, 460.0,
                            100.0, 200.0, 180.0, 180.0)
    e.blurb = "oversquare 620 x 400, twin overhead cams over bucket tappets"
    e.notes["head"] = "dohc"
    return e


def diesel():
    """Compression ignition. The intake is AIR ONLY; fuel is sprayed from an injector in the
    head over ~28 deg around TDC and lights itself on the compressed air (need_fuel: the front
    starts when the first fuel arrives, not when the air is merely hot). Higher ratio, no
    plug, and the burn is mixing-limited - it lives where the spray is."""
    # FEWER fuel particles, MORE energy each: every injected particle dilutes the hot charge
    # (2600 of them took the mean from 1869 K to ~1620 K before any burn), and a burn that
    # trails past TDC is fighting the expansion. 1500 particles over 346-366 deg at 1.2e8.
    e = _four_stroke_single("diesel", (260.0, 820.0), (540.0, 1520.0), 260.0, 560.0,
                            100.0, 190.0, 190.0, 190.0, fuel_frac=0.0, q_rxn=1.2e8, plug=False)
    n = e.notes
    tip = (n["mid"], n["deck"] + 10.0)
    clk = e.valves[0].clock
    e.injectors.append(Injector(n["cyl"], tip, 90.0, 55.0, 1.3, clk, 346.0, 366.0, 1500,
                                C.FUEL, "injector"))
    e.igniters.append(Igniter((n["cyl"],), tip, clk, 0.0, 1400.0, True,
                              2.2 * _flame(560.0, REV), (300.0, 440.0), 0.5, "injector"))
    e.blurb = "compression ignition: air only, fuel injected at TDC, no plug"
    n["head"] = "diesel"
    n["injector"] = tip
    return e


# ====================================================================== two-stroke
def two_stroke():
    """Piston-ported, crankcase-scavenged. No valves at all: the piston skirt covers and
    uncovers an exhaust port on one wall and a transfer port on the other, and the CRANKCASE
    under the piston is the pump - its ceiling is the piston's underside, so the up-stroke
    draws charge in through a reed and the down-stroke pushes it up the transfer passage.
    Fires every revolution."""
    rev = REV
    drive = Drive(rev)
    x0, x1 = 300.0, 780.0
    cx, cy = 540.0, 1360.0
    R, L = 200.0, 480.0
    tdc, bdc = cy - (R + L), cy - (L - R)            # 680, 1080
    deck = tdc - 100.0                                # 580
    H = 200.0                                         # piston height
    eng = Engine("two_stroke", rev, drive, 360.0, 0.25, [Group()], q_rxn=7.5e7)
    pfn = slider_crank(cy, R, L, drive)
    eng.pistons.append(Piston(0, pfn, True, H, "piston"))
    clk = clock(drive, 360.0, 0.0)
    cyl = eng.add_box(Box(x0, x1, deck, bdc, 0, "cyl", "", 0, -1, "crown", OFF, 1.0, [], "cylinder"))
    case = eng.add_box(Box(x0, x1, tdc + H, 1600.0, 0, "case", "intake", -1, 0, "under", OFF, 1.0,
                           [], "crankcase"))
    # ports in the walls: exhaust higher (opens first), transfer lower. THE BLOWDOWN GAP IS
    # THE WHOLE TWO-STROKE: with this scaled gas the cylinder takes ~40 deg of crank to blow
    # down, and with the ports only 13 deg apart the first gate showed burnt gas at several
    # bar flooding DOWN the transfer into the crankcase, the reed then backflowing, and a net
    # throughput of zero. Exhaust opens at 77 deg ATDC, transfer at 117 - a 40 deg lead.
    ex_port = (bdc - 240.0, bdc)                      # 840..1080: opens 64 deg ATDC
    tr_port = (bdc - 60.0, bdc)                       # 1020..1080: opens 124, shuts 236
    tr = eng.add_box(Box(180.0, x0, tr_port[0], 1560.0, 0, "duct", "intake", label="transfer"))
    eng.connect(cyl, tr, LEFT, tr_port[0], tr_port[1], cover=(0, "crown"))
    eng.connect(case, tr, LEFT, 1440.0, 1560.0, cover=(0, "under"))
    dex = eng.add_box(Box(x1, 980.0, ex_port[0], ex_port[1], 0, "duct", "exhaust", label="exhaust duct"))
    eng.connect(cyl, dex, RIGHT, ex_port[0], ex_port[1], cover=(0, "crown"))
    rex = eng.add_box(Box(820.0, 980.0, -200.0, ex_port[0], 0, "duct", "exhaust", open=OPEN_TOP,
                          label="exhaust riser"))
    eng.connect(dex, rex, TOP, 820.0, 980.0)
    din = eng.add_box(Box(x1, 980.0, 1400.0, 1600.0, 0, "duct", "intake", label="intake duct"))
    eng.connect(case, din, RIGHT, 1400.0, 1600.0, gate=0)
    rin = eng.add_box(Box(820.0, 980.0, 1600.0, 2100.0, 0, "duct", "intake", open=OPEN_BOT,
                          label="intake riser"))
    eng.connect(din, rin, BOT, 820.0, 980.0)
    # THE REED IS PRESSURE-DRIVEN. A kinematic reed ("open while the piston rises") opened at
    # BDC with the crankcase still at 2 bar from the down-stroke, and scratch/reed_probe.py
    # watched the charge blow back OUT the intake: net throughput zero. It opens when the
    # crankcase drops below the intake, which is what a reed is.
    eng.valves.append(Valve(clk, 0.0, 0.0, "reed", None, 360.0, "reed", box=case, p_ref=1.0))
    plug = (540.0, deck + 12.0)
    eng.igniters.append(Igniter((cyl,), plug, clk, SPARK, 0.0, False, _flame(x1 - x0, rev),
                                rearm=0.5, label="plug"))
    eng.notes.update(dict(bore=(x0, x1), crank=(cx, cy), R=R, L=L, deck=deck, tdc=tdc, bdc=bdc,
                          H=H, ex_port=ex_port, tr_port=tr_port, case_bot=1600.0, cyl=cyl,
                          plug=plug, head="two_stroke"))
    eng.blurb = "no valves: wall ports the piston uncovers, the crankcase is the pump"
    return eng


# ====================================================================== opposed piston
def opposed_piston():
    """Two pistons in one bore, cranks top and bottom, meeting in the middle. Uniflow
    scavenging: intake ports in the wall at the lower piston's end, exhaust at the upper's,
    each uncovered by its own piston. The upper crank LEADS by 12 deg so the exhaust opens
    first (the Junkers trick). A two-stroke, fed by a blower (intake at 1.5 ambients)."""
    rev = REV
    drive = Drive(rev)
    x0, x1 = 340.0, 740.0
    R, L = 150.0, 380.0
    cy_lo, cy_hi = 1560.0, 360.0
    lo = slider_crank(cy_lo, R, L, drive, 0.0, True)          # crown 1030 (TDC) .. 1330
    hi = slider_crank(cy_hi, R, L, drive, 12.0, False)        # crown 890 (TDC) .. 590
    eng = Engine("opposed_piston", rev, drive, 360.0, 0.30, [Group()], q_rxn=7.5e7)
    eng.pistons.append(Piston(0, lo, True, 170.0, "lower piston"))
    eng.pistons.append(Piston(1, hi, False, 170.0, "upper piston"))
    clk = clock(drive, 360.0, 0.0)
    cyl = eng.add_box(Box(x0, x1, 590.0 - 10.0, 1330.0 + 10.0, 0, "cyl", "", 0, 1, "crown", OFF,
                          1.0, [], "cylinder"))
    ex_port = (600.0, 730.0)
    in_port = (1190.0, 1320.0)
    dex = eng.add_box(Box(x1, 940.0, ex_port[0], ex_port[1], 0, "duct", "exhaust", label="exhaust duct"))
    eng.connect(cyl, dex, RIGHT, ex_port[0], ex_port[1], cover=(1, "crown"))
    rex = eng.add_box(Box(820.0, 940.0, -200.0, ex_port[0], 0, "duct", "exhaust", open=OPEN_TOP,
                          label="exhaust riser"))
    eng.connect(dex, rex, TOP, 820.0, 940.0)
    din = eng.add_box(Box(140.0, x0, in_port[0], in_port[1], 0, "duct", "intake", boost=1.5,
                          label="intake duct"))
    eng.connect(cyl, din, LEFT, in_port[0], in_port[1], cover=(0, "crown"))
    rin = eng.add_box(Box(140.0, 260.0, in_port[1], 2100.0, 0, "duct", "intake", open=OPEN_BOT,
                          boost=1.5, label="intake riser"))
    eng.connect(din, rin, BOT, 140.0, 260.0)
    plug = (x0 + 14.0, 960.0)
    eng.igniters.append(Igniter((cyl,), plug, clk, SPARK, 0.0, False, _flame(x1 - x0, rev),
                                rearm=0.5, label="plug"))
    eng.notes.update(dict(bore=(x0, x1), R=R, L=L, cy_lo=cy_lo, cy_hi=cy_hi, ex_port=ex_port,
                          in_port=in_port, cyl=cyl, plug=plug, head="opposed"))
    eng.blurb = "two pistons, two cranks, one bore - uniflow two-stroke, exhaust leads 12 deg"
    return eng


# ====================================================================== flathead
def flathead():
    """Side valves. The valves stand in the block BESIDE the bore, opening upward into a
    chamber that lies over them, and the chamber joins the bore at the top: an L-shaped
    combustion space burnt by one front. The camshaft sits under the valves in the block."""
    rev = REV
    drive = Drive(rev)
    x0, x1 = 140.0, 540.0
    cx, cy = 340.0, 1300.0
    R, L = 150.0, 380.0
    tdc, bdc = cy - (R + L), cy - (L - R)      # 770, 1070
    # THE CHAMBER IS CLEARANCE TOO. Bore clearance 400 x 60 plus a 360 x 60 side chamber
    # against a 400 x 300 sweep is 3.6:1 by area - a flathead really is a low-compression
    # engine - so the reaction energy is raised to get a burn you can see.
    deck = tdc - 60.0                            # 710
    eng = Engine("flathead", rev, drive, 720.0, 0.30, [Group()], q_rxn=1.0e8)
    eng.pistons.append(Piston(0, slider_crank(cy, R, L, drive), True, 170.0, "piston"))
    clk = clock(drive, 720.0, 0.0)
    cyl = eng.add_box(Box(x0, x1, deck, bdc, 0, "cyl", "", 0, -1, "crown", OFF, 1.0, [], "cylinder"))
    ch = eng.add_box(Box(x1, 900.0, deck, tdc, 0, "cyl", "", label="chamber"))
    eng.connect(cyl, ch, RIGHT, deck, tdc)
    ip, ep = (560.0, 710.0), (730.0, 880.0)
    rin = eng.add_box(Box(ip[0], ip[1], tdc, 1070.0, 0, "duct", "intake", label="intake port"))
    rex = eng.add_box(Box(ep[0], ep[1], tdc, 890.0, 0, "duct", "exhaust", label="exhaust port"))
    eng.connect(ch, rin, BOT, ip[0], ip[1], gate=0)
    eng.connect(ch, rex, BOT, ep[0], ep[1], gate=1)
    dex = eng.add_box(Box(ep[1], 1040.0, tdc, 890.0, 0, "duct", "exhaust", label="exhaust duct"))
    eng.connect(rex, dex, RIGHT, tdc, 890.0)
    xr = eng.add_box(Box(920.0, 1040.0, -200.0, tdc, 0, "duct", "exhaust", open=OPEN_TOP,
                         label="exhaust riser"))
    eng.connect(dex, xr, TOP, 920.0, 1040.0)
    din = eng.add_box(Box(ip[1], 1040.0, 940.0, 1070.0, 0, "duct", "intake", label="intake duct"))
    eng.connect(rin, din, RIGHT, 940.0, 1070.0)
    ir = eng.add_box(Box(920.0, 1040.0, 1070.0, 2100.0, 0, "duct", "intake", open=OPEN_BOT,
                         label="intake riser"))
    eng.connect(din, ir, BOT, 920.0, 1040.0)
    eng.valves.append(Valve(clk, IVO, IVC, "cam", None, 720.0, "intake"))
    eng.valves.append(Valve(clk, EVO, EVC, "cam", None, 720.0, "exhaust"))
    plug = (720.0, deck + 10.0)
    eng.igniters.append(Igniter((cyl, ch), plug, clk, SPARK, 0.0, False, _flame(760.0, rev),
                                label="plug"))
    eng.notes.update(dict(bore=(x0, x1), crank=(cx, cy), R=R, L=L, deck=deck, tdc=tdc, bdc=bdc,
                          chamber=(x1, 900.0, deck, tdc), pockets=(ip, ep), cyl=cyl, plug=plug,
                          head="flathead"))
    eng.blurb = "side valves in the block, an L-shaped chamber - the 1930s layout"
    return eng


# ====================================================================== V-twin
def v_twin():
    """Two cylinders 50 deg apart sharing one crankpin. Each is a full four-stroke with its
    own overhead cam; the intakes rise through the V and the exhaust headers run down the
    outside of each barrel. Because the pin is shared the two TDCs are 50 deg of crank apart,
    so the firing is UNEVEN: 310 deg, then 410 deg. That thump is the clip."""
    # THE FRAME SETS THE SIZE. At the head level a bank leaning V/2 off vertical has only
    # 540 - s_head*sin(V/2) px of width to the frame edge on its outer side, and the exhaust
    # has to live there (the intake takes the inside of the V, as on the reference). 50 deg
    # and a 300 x 220 cylinder is what fits with an exhaust duct outside each head.
    rev = REV
    drive = Drive(rev)
    HALF = 25.0
    C0 = (540.0, 1300.0)
    x0, x1 = 390.0, 690.0
    R, L = 110.0, 280.0
    tdc, bdc = C0[1] - (R + L), C0[1] - (L - R)       # 910, 1130
    deck = tdc - 60.0                                 # 850  -> 4.7:1
    eng = Engine("v_twin", rev, drive, 720.0, 0.30,
                 [Group("left", -HALF, C0, C0), Group("right", HALF, C0, C0)], q_rxn=7.5e7)
    ph, ph_ = deck - 120.0, deck                      # pocket top / deck
    for g, (phase, inside_right) in enumerate(((HALF, True), (-HALF + 360.0, False))):
        # the pin angle in a bank's LOCAL frame is the world angle MINUS the bank's angle, so
        # the left bank (at -HALF) sees +HALF. Getting this sign wrong put its spark 70 deg
        # before the real TDC and the gate showed it as a 3204 K peak.
        eng.pistons.append(Piston(g, slider_crank(C0[1], R, L, drive, HALF if g == 0 else -HALF),
                                  True, 130.0, f"piston {g}"))
        clk = clock(drive, 720.0, phase)
        cyl = eng.add_box(Box(x0, x1, deck, bdc, g, "cyl", "", g, -1, "crown", OFF, 1.0, [],
                              f"cylinder {g}"))
        pa, pb = (400.0, 520.0), (560.0, 680.0)
        ip, ep = (pb, pa) if inside_right else (pa, pb)
        pin = eng.add_box(Box(ip[0], ip[1], ph, ph_, g, "duct", "intake", label=f"intake pocket {g}"))
        pex = eng.add_box(Box(ep[0], ep[1], ph, ph_, g, "duct", "exhaust", label=f"exhaust pocket {g}"))
        nv = len(eng.valves)
        eng.connect(cyl, pin, TOP, ip[0], ip[1], gate=nv)
        eng.connect(cyl, pex, TOP, ep[0], ep[1], gate=nv + 1)
        eng.valves.append(Valve(clk, IVO, IVC, "cam", None, 720.0, f"intake {g}"))
        eng.valves.append(Valve(clk, EVO, EVC, "cam", None, 720.0, f"exhaust {g}"))
        # intake: a duct to the INSIDE of the V, then a riser up through the V and off the
        # top edge. exhaust: a duct to the OUTSIDE, then a header running DOWN the outside of
        # the barrel to a flange near the crankcase (the open end hides under the flange).
        if inside_right:
            din = eng.add_box(Box(ip[1], 790.0, ph, ph_, g, "duct", "intake", label=f"intake duct {g}"))
            eng.connect(pin, din, RIGHT, ph, ph_)
            rin = eng.add_box(Box(690.0, 790.0, -250.0, ph, g, "duct", "intake", open=OPEN_TOP,
                                  label=f"intake riser {g}"))
            eng.connect(din, rin, TOP, 690.0, 790.0)
            dex = eng.add_box(Box(240.0, ep[0], ph, ph_, g, "duct", "exhaust", label=f"exhaust duct {g}"))
            eng.connect(pex, dex, LEFT, ph, ph_)
            rex = eng.add_box(Box(240.0, 340.0, ph_, ph_ + 300.0, g, "duct", "exhaust", open=OPEN_BOT,
                                  label=f"exhaust riser {g}"))
            eng.connect(dex, rex, BOT, 240.0, 340.0)
        else:
            din = eng.add_box(Box(290.0, ip[0], ph, ph_, g, "duct", "intake", label=f"intake duct {g}"))
            eng.connect(pin, din, LEFT, ph, ph_)
            rin = eng.add_box(Box(290.0, 390.0, -250.0, ph, g, "duct", "intake", open=OPEN_TOP,
                                  label=f"intake riser {g}"))
            eng.connect(din, rin, TOP, 290.0, 390.0)
            dex = eng.add_box(Box(ep[1], 840.0, ph, ph_, g, "duct", "exhaust", label=f"exhaust duct {g}"))
            eng.connect(pex, dex, RIGHT, ph, ph_)
            rex = eng.add_box(Box(740.0, 840.0, ph_, ph_ + 300.0, g, "duct", "exhaust", open=OPEN_BOT,
                                  label=f"exhaust riser {g}"))
            eng.connect(dex, rex, BOT, 740.0, 840.0)
        plug = (540.0, deck + 10.0)
        eng.igniters.append(Igniter((cyl,), plug, clk, SPARK, 0.0, False, _flame(x1 - x0, rev),
                                    label=f"plug {g}"))
    eng.notes.update(dict(bore=(x0, x1), crank=C0, R=R, L=L, deck=deck, tdc=tdc, bdc=bdc,
                          pocket=(ph, ph_), half=HALF, head="vtwin"))
    eng.blurb = "50 deg V-twin on one crankpin: uneven firing, 310 then 410 deg"
    return eng


# ====================================================================== radial
def radial(n=5, drive=None, head="radial"):
    """Five cylinders round one crank. Cylinder 0 (top) carries the MASTER rod on the
    crankpin; the other four hang off knuckle pins on its big end, so their pistons are
    solved from those pins, not from a pure slider-crank. Firing order 1-3-5-2-4, one bang
    every 144 deg. Everything is small because the frame is 1080 wide, so the ports are
    narrow and the gate's volumetric efficiency is the number to read."""
    drive = drive or Drive(REV)
    rev = getattr(drive, "rev_seconds", REV)
    C0 = (540.0, 960.0)
    R, Lm, rk, La = 85.0, 240.0, 42.0, 205.0
    kin = RadialKinematics(n, R, Lm, rk, La, drive, C0, pin_to_crown=50.0)
    x0, x1 = 465.0, 615.0
    clear = 45.0
    eng = Engine("radial", rev, drive, 720.0, 0.30,
                 [Group(f"cyl{j}", 360.0 * j / n, C0, C0) for j in range(n)], q_rxn=7.5e7)
    period = rev * 2.0
    for j in range(n):
        fn = kin.piston_fn(j)
        crown_min = min(fn(period * k / 720)[0] for k in range(720))
        crown_max = max(fn(period * k / 720)[0] for k in range(720))
        deck = crown_min - clear
        eng.pistons.append(Piston(j, fn, True, 70.0, f"piston {j}"))
        # 1-3-5-2-4: even cylinders lead by a revolution
        off = 360.0 if j % 2 == 0 else 0.0
        clk = clock(drive, 720.0, -360.0 * j / n + off)
        cyl = eng.add_box(Box(x0, x1, deck, crown_max + 1.0, j, "cyl", "", j, -1, "crown", OFF,
                              1.0, [], f"cylinder {j}"))
        ph = 70.0
        ip, ep = (478.0, 528.0), (552.0, 602.0)
        pin = eng.add_box(Box(ip[0], ip[1], deck - ph, deck, j, "duct", "intake", label=f"in pocket {j}"))
        pex = eng.add_box(Box(ep[0], ep[1], deck - ph, deck, j, "duct", "exhaust", label=f"ex pocket {j}"))
        nv = len(eng.valves)
        eng.connect(cyl, pin, TOP, ip[0], ip[1], gate=nv)
        eng.connect(cyl, pex, TOP, ep[0], ep[1], gate=nv + 1)
        eng.valves.append(Valve(clk, IVO, IVC, "cam", None, 720.0, f"intake {j}"))
        eng.valves.append(Valve(clk, EVO, EVC, "cam", None, 720.0, f"exhaust {j}"))
        din = eng.add_box(Box(378.0, ip[0], deck - ph, deck, j, "duct", "intake", label=f"in duct {j}"))
        dex = eng.add_box(Box(ep[1], 702.0, deck - ph, deck, j, "duct", "exhaust", label=f"ex duct {j}"))
        eng.connect(pin, din, LEFT, deck - ph, deck)
        eng.connect(pex, dex, RIGHT, deck - ph, deck)
        # risers run back along the cylinder to the ring, open ends hidden under it
        rin = eng.add_box(Box(388.0, 440.0, deck, deck + 130.0, j, "duct", "intake", open=OPEN_BOT,
                              label=f"in riser {j}"))
        rex = eng.add_box(Box(640.0, 692.0, deck, deck + 130.0, j, "duct", "exhaust", open=OPEN_BOT,
                              label=f"ex riser {j}"))
        eng.connect(din, rin, BOT, 388.0, 440.0)
        eng.connect(dex, rex, BOT, 640.0, 692.0)
        plug = (540.0, deck + 8.0)
        eng.igniters.append(Igniter((cyl,), plug, clk, SPARK, 0.0, False, _flame(x1 - x0),
                                    label=f"plug {j}"))
        eng.notes.setdefault("decks", []).append(deck)
        eng.notes.setdefault("crown_range", []).append((crown_min, crown_max))
    eng.notes.update(dict(bore=(x0, x1), crank=C0, R=R, Lm=Lm, rk=rk, La=La, kin=kin, n=n,
                          pockets=((478.0, 528.0), (552.0, 602.0)), pocket_h=70.0,
                          riser_end=eng.notes["decks"][0] + 130.0, head=head))
    eng.blurb = "five-cylinder radial: master rod + articulated rods, fires 1-3-5-2-4"
    return eng


# ====================================================================== the old graph
def otto_legacy():
    """The 2026-09-15 five-box graph and its timing, verbatim - the regression case for the
    generalised core. Gated then at PV/NkT 0.953, VE 0.77, 2361 K, 30.4 bar (one 12 s cycle at
    rev 6). Not a final."""
    rev = 6.0
    drive = Drive(rev)
    eng = Engine("otto_legacy", rev, drive, 720.0, 0.30, [Group()], q_rxn=7.5e7)
    eng.pistons.append(Piston(0, slider_crank(1520.0, 260.0, 560.0, drive), True, 190.0))
    clk = clock(drive, 720.0, 0.0)
    ipl = eng.add_box(Box(150.0, 510.0, -200.0, 300.0, 0, "duct", "intake", open=OPEN_TOP, label="in plenum"))
    epl = eng.add_box(Box(570.0, 930.0, -200.0, 300.0, 0, "duct", "exhaust", open=OPEN_TOP, label="ex plenum"))
    irn = eng.add_box(Box(290.0, 480.0, 300.0, 570.0, 0, "duct", "intake", label="in runner"))
    ern = eng.add_box(Box(600.0, 790.0, 300.0, 570.0, 0, "duct", "exhaust", label="ex runner"))
    cyl = eng.add_box(Box(260.0, 820.0, 570.0, 1220.0, 0, "cyl", "", 0, -1, "crown", OFF, 1.0, [], "cylinder"))
    eng.connect(ipl, irn, BOT, 290.0, 480.0)
    eng.connect(epl, ern, BOT, 600.0, 790.0)
    eng.connect(irn, cyl, BOT, 290.0, 480.0, gate=0)
    eng.connect(ern, cyl, BOT, 600.0, 790.0, gate=1)
    eng.valves.append(Valve(clk, -12.0, 196.0, "cam", None, 720.0, "intake"))
    eng.valves.append(Valve(clk, 528.0, 732.0, "cam", None, 720.0, "exhaust"))
    eng.igniters.append(Igniter((cyl,), (540.0, 590.0), clk, 340.0, 0.0, False,
                                _flame(560.0, rev), label="plug"))
    eng.notes.update(dict(bore=(260.0, 820.0), crank=(540.0, 1520.0), R=260.0, L=560.0, deck=570.0,
                          tdc=700.0, bdc=1220.0, pockets=((290.0, 480.0), (600.0, 790.0)),
                          pocket_h=270.0, duct_h=270.0, riser_w=0, mid=540.0, cyl=cyl,
                          clearance=130.0, plug=(540.0, 590.0), head="legacy"))
    eng.blurb = "the 2026-09-15 graph, for regression only"
    return eng



# ====================================================================== the music cut
def radial_flash(slug="flashing_lights", warm=4.0, v_th=13000.0, n0=0.20):
    """The radial, locked to the song and drawn as a field. ONE PROCESS PER RENDER.

    Three things are raised from the batch radial, and each has a reason that is not taste:

    V_TH 9000 -> 13000.  The lock (sync.BeatDrive) turns the crank at 0.5525 s/rev, 9.05x the
      radial's 5 s base, because that is what puts a combustion on every grid line of the
      hook. Piston speed rises with it, and below a thermal/piston ratio of about 8 the ports
      choke and the cylinder fills with a partial vacuum. 13000 puts the ratio at 11.2.
      Substeps rise with it to keep a molecule at ~3 px per substep.

    N0 0.055 -> 0.20.  The field render (field.py) averages molecules per cell; its noise -
      the fuzz Clancy asked to be rid of - falls as 1/sqrt(N). This is calibration-NEUTRAL:
      pressure is reported against N0*T0, the DSMC acceptance is n_cell/n_ref with n_ref
      proportional to N0, and seeding and injection both scale with it. Only the statistics
      improve.

    q_rxn is quoted at Q_REF_VTH and rescaled by the world, so raising V_TH does not change
    the temperature a burn buys.

    warm 4.0 s, not the batch's 10: at the locked 0.5525 s/rev that is 7.2 four-stroke cycles,
    where the batch's 10 s bought one. The warm-up is unshown physics and it is the single
    biggest cost in the run, so it is sized by cycles, not by seconds.
    """
    from . import config as C
    from . import sync as SY
    C.V_TH = float(v_th)                 # module constants: one scene per process
    C.N0 = float(n0)
    sch = SY.Schedule(slug, warm)
    drv = sch.drive(rev0=REV)
    eng = radial(5, drive=drv, head="radial_flash")
    eng.name = "radial_flash"
    eng.blurb = "five-cylinder radial locked to the song: a combustion on every vocal onset"
    eng.notes["sched"] = sch
    eng.notes["drive_check"] = drv.check() if hasattr(drv, "check") else {}
    eng.notes["slug"] = slug
    eng.notes["warm"] = warm
    eng.notes["substeps"] = int(round(v_th / 3.0 / 100.0) * 100)
    # THE VIEW. Measured 2026-09-20: the drawn rocker gear reaches 625 world px from the crank
    # and the frame's half-width is 540, so the radial - INCLUDING the one already shipped -
    # ran off both edges. Geometry is gated, so the fix is a view zoom (parts.Canvas.set_view),
    # not a smaller engine: 0.79 leaves 46 px of margin each side at 1080 wide.
    eng.notes["zoom"] = 0.79
    from . import looks as L
    eng.notes["cam_phase"] = 0.0
    eng.notes["cam_prof"] = [L._cam_profile(eng, 0), L._cam_profile(eng, 1)]
    return eng

ENGINES = {
    "single_sohc": single_sohc,
    "dohc": dohc,
    "diesel": diesel,
    "two_stroke": two_stroke,
    "opposed_piston": opposed_piston,
    "flathead": flathead,
    "v_twin": v_twin,
    "radial": radial,
    "radial_flash": radial_flash,
    "otto_legacy": otto_legacy,
}

FINALS = ["single_sohc", "dohc", "diesel", "two_stroke", "opposed_piston", "flathead",
          "v_twin", "radial"]


def build(name: str, **kw) -> Engine:
    eng = ENGINES[name](**kw)
    eng.check()
    bad = eng.frame_fit()
    assert not bad, f"{name}: boxes leave the frame: {bad}"
    return eng
