# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""looks.py - what each engine LOOKS like: the cutaway, drawn from the same numbers the
physics runs on. A gas passage is a box cut out of hatched metal, so what you see IS the
box graph; a cam lobe is generated from the valve's lift curve, so what pushes the valve is
the lift the port is gated by; the rods you see end on the pins the pistons are solved from.

Two callables per engine, keyed by `eng.notes['head']`:
    static(cv, eng)      the block, head, ducts, ring - drawn once
    moving(cv, eng, t)   pistons, rods, crank, valves, cams, followers, spark - every frame
"""
from __future__ import annotations

import math

from . import engine as E
from . import parts as P

LIFT_PX = 34.0            # full valve lift, world px (the port gate is the lift FRACTION)
WALL = 40.0
COOLANT = (24, 44, 74)
IDENT = E.Group()


def _cut_group(cv, eng, g, kinds=("duct",), ycap=None, hide=0.0):
    """Cut every duct box of group g out of the metal and edge it. `hide`: leave that many px
    of an in-frame OPEN end uncut, so the reservoir boundary stays under the metal drawn over
    it (a flange, the radial's ring) - nothing pops into existence (OUTPUT_RULES)."""
    for b in eng.boxes:
        if b.group != g or b.kind not in kinds:
            continue
        y0, y1 = b.y0, (b.y1 if ycap is None else min(b.y1, ycap))
        if hide > 0 and b.open[E.BOT] and b.y1 < 1900:
            y1 = b.y1 - hide
        if hide > 0 and b.open[E.TOP] and b.y0 > 0:
            y0 = b.y0 + hide
        cv.cut(b.x0, y0, b.x1, y1)
        cv.edge(b.x0, y0, b.x1, y1)


def _flash(clock, spark_deg, cycle, t):
    d = (clock(t) - spark_deg) % cycle
    return max(0.0, 1.0 - d / 7.0) if d < 7.0 else 0.0


# ====================================================================== singles
def _single_static(cv, eng, head):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    deck, bdc, R = n["deck"], n["bdc"], n["R"]
    ph, dw, rw = n["pocket_h"], n["duct_h"], n["riser_w"]
    cv.set_group(eng.groups[0])
    # risers (off the top of the frame) and the head casting
    for b in eng.boxes:
        if b.label.endswith("riser"):
            cv.hatched_rect(b.x0 - 14, b.y0, b.x1 + 14, b.y1 + 2)
    head_top = deck - ph - 46
    cv.hatched_rect(26, head_top, 1054, deck)
    # cam box over the head: an open cover, the cam and followers visible inside
    cb0, cb1 = 250.0, 830.0
    cv.hatched_rect(cb0 - 16, head_top - 210, cb1 + 16, head_top)
    cv.cut(cb0, head_top - 194, cb1, head_top)
    cv.edge(cb0, head_top - 194, cb1, head_top)
    # block with fins, crankcase below
    wall_bot = bdc + 120
    cv.hatched_rect(x0 - WALL, deck, x1 + WALL, wall_bot)
    if head in ("sohc", "diesel", "legacy"):
        cv.fins(x0 - WALL, x0 - WALL - 52, deck + 40, wall_bot - 40, 9)
        cv.fins(x1 + WALL, x1 + WALL + 52, deck + 40, wall_bot - 40, 9)
    else:
        for sx in (x0 - WALL + 10, x1 + WALL - 24):           # a water jacket
            cv.rect(sx, deck + 40, sx + 14, wall_bot - 60, COOLANT, P.BLOCK_EDGE, 1.0)
    cv.hatched_rect(x0 - 90, wall_bot, x1 + 90, cy + R + 90)
    cv.cut(x0 - 50, wall_bot, x1 + 50, cy + R + 50)
    cv.edge(x0 - 50, wall_bot, x1 + 50, cy + R + 50)
    # passages
    _cut_group(cv, eng, 0)
    cv.cut(x0, deck, x1, wall_bot)
    cv.edge(x0, deck, x1, wall_bot)
    # head bolts and gasket line
    for bx in (x0 - 20, x1 + 20, 60, 1020):
        cv.bolt(bx, deck - 22, 10.0)
    cv.line((26, deck), (1054, deck), P.BLOCK_EDGE, 2.4)
    # plug or injector, static body
    if head == "diesel":
        cv.injector(n["mid"], deck)
    else:
        cv.plug(n["mid"], deck)


def _single_moving(cv, eng, t, head):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    deck, R, L = n["deck"], n["R"], n["L"]
    ph = n["pocket_h"]
    cv.set_group(eng.groups[0])
    crown, _ = eng.pistons[0].fn(t)
    H = eng.pistons[0].height
    pin = (0.5 * (x0 + x1), crown + 0.5 * H)
    th = eng.drive.theta(t)
    big = (cx + R * math.sin(th), cy - R * math.cos(th))
    cv.crank((cx, cy), R, th)
    cv.rod(pin, big)
    cv.piston(x0, x1, crown, H, pin[1])
    # valves, springs, cams
    head_top = deck - ph - 46
    stem_top = head_top - 120
    for vi, (pk, valve) in enumerate(zip(n["pockets"], eng.valves)):
        cxv = 0.5 * (pk[0] + pk[1])
        lift = valve.lift(t)
        lp = lift * LIFT_PX
        cv.valve(cxv, deck, 0.5 * (pk[1] - pk[0]) + 8, lp, stem_top)
        cv.rect(cxv - 26, head_top - 16, cxv + 26, head_top, P.CAM, P.CAM_HI, 1.0)   # guide/seat
        cv.spring((cxv, head_top - 16), (cxv, stem_top + 12 + lp), 6, 24.0)
        cv.rect(cxv - 20, stem_top + 4 + lp, cxv + 20, stem_top + 16 + lp, P.VALVE, P.VALVE_LO, 1.0)
        if head == "dohc":
            # bucket tappet under its own cam
            cv.rect(cxv - 34, stem_top - 26 + lp, cxv + 34, stem_top + 4 + lp, P.CAM_HI, P.CAM, 1.2)
            cv.cam_lobe((cxv, stem_top - 26 - 44), 44.0, LIFT_PX, valve, t, 90.0)
    if head in ("sohc", "diesel", "legacy"):
        # one cam in the middle, finger followers out to each stem
        mid = n["mid"]
        cam = (mid, stem_top - 60)
        for vi, (pk, valve) in enumerate(zip(n["pockets"], eng.valves)):
            cxv = 0.5 * (pk[0] + pk[1])
            lp = valve.lift(t) * LIFT_PX
            sgn = -1 if cxv < mid else 1
            pivot = (mid + sgn * 200, stem_top - 10)
            cv.finger(pivot, (cxv, stem_top + 4), lp, 18.0)
        cv.cam_lobe(cam, 46.0, LIFT_PX, eng.valves[0], t, 135.0)
        cv.cam_lobe(cam, 46.0, LIFT_PX, eng.valves[1], t, 45.0)
        cv.circle(cam[0], cam[1], 18, P.CAM_HI, P.CAM, 1.0)
    if head != "diesel":
        ig = eng.igniters[0]
        f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
        if f > 0:
            cv.plug(n["mid"], deck, True, f)


# ====================================================================== two-stroke
def _two_stroke_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    deck, bdc, cb = n["deck"], n["bdc"], n["case_bot"]
    cv.set_group(eng.groups[0])
    # exhaust and intake risers, the head, the block round bore + crankcase
    for b in eng.boxes:
        if b.label.endswith("riser"):
            cv.hatched_rect(b.x0 - 14, b.y0 - 2, b.x1 + 14, b.y1 + 2)
    cv.hatched_rect(x0 - WALL, deck - 90, x1 + WALL, deck)
    cv.hatched_rect(x0 - WALL - 90, deck, x1 + WALL + 90, cb + 40)
    cv.fins(x0 - WALL - 90, x0 - WALL - 150, deck + 20, bdc - 20, 9)
    cv.fins(x1 + WALL + 90, x1 + WALL + 150, deck + 20, bdc - 20, 9)
    _cut_group(cv, eng, 0, ("duct",))
    cv.cut(x0, deck, x1, cb)
    cv.edge(x0, deck, x1, cb)
    cv.line((x0 - WALL, deck), (x1 + WALL, deck), P.BLOCK_EDGE, 2.4)
    for bx in (x0 - 20, x1 + 20):
        cv.bolt(bx, deck - 22, 10.0)
    cv.plug(540.0, deck)


def _two_stroke_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    R = n["R"]
    cv.set_group(eng.groups[0])
    crown, _ = eng.pistons[0].fn(t)
    H = eng.pistons[0].height
    pin = (540.0, crown + 0.42 * H)
    th = eng.drive.theta(t)
    big = (cx + R * math.sin(th), cy - R * math.cos(th))
    cv.crank((cx, cy), R, th, 40.0, 70.0)
    cv.rod(pin, big, 34.0, 32.0)
    cv.piston(x0, x1, crown, H, pin[1], rings=2)
    # the reed: a flap on the crankcase wall at the intake port, opening inward with lift
    lift = eng.valves[0].lift(t)
    hinge = (x1, 1440.0)
    a = math.radians(90.0 + 55.0 * lift)      # closed: along the wall; open: swung into the case
    tip = (hinge[0] - 120.0 * math.cos(a), hinge[1] + 120.0 * math.sin(a))
    cv.line(hinge, tip, P.RUBBER, 9.0)
    cv.line(hinge, tip, P.VALVE_LO, 3.0)
    cv.circle(hinge[0], hinge[1], 7, P.PIN, P.PIN_RIM, 1.0)
    ig = eng.igniters[0]
    f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
    if f > 0:
        cv.plug(540.0, n["deck"], True, f)


# ====================================================================== opposed piston
def _opposed_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    R = n["R"]
    cy_lo, cy_hi = n["cy_lo"], n["cy_hi"]
    cv.set_group(eng.groups[0])
    for b in eng.boxes:
        if b.label.endswith("riser"):
            cv.hatched_rect(b.x0 - 14, b.y0 - 2, b.x1 + 14, b.y1 + 2)
    top, bot = 540.0, 1380.0
    cv.hatched_rect(x0 - WALL, top, x1 + WALL, bot)
    for sx in (x0 - WALL + 8, x1 + WALL - 22):
        cv.rect(sx, 760.0, sx + 14, 1160.0, COOLANT, P.BLOCK_EDGE, 1.0)
    # crankcases top and bottom
    for cyc, ya, yb in ((cy_hi, cy_hi - R - 90, top), (cy_lo, bot, cy_lo + R + 90)):
        cv.hatched_rect(x0 - 90, ya, x1 + 90, yb)
        cv.cut(x0 - 50, ya + (40 if ya < top else 0), x1 + 50, yb - (40 if yb > bot else 0))
        cv.edge(x0 - 50, ya + (40 if ya < top else 0), x1 + 50, yb - (40 if yb > bot else 0))
    _cut_group(cv, eng, 0, ("duct",))
    cv.cut(x0, top, x1, bot)
    cv.edge(x0, top, x1, bot)
    # a plug in the wall at mid-height, pointing left
    px, py = n["plug"]
    cv.set_group(E.Group("plug", 90.0, (x0, py), (x0, py)))
    cv.plug(x0, py)
    cv.set_group(eng.groups[0])


def _opposed_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    R = n["R"]
    cv.set_group(eng.groups[0])
    th = eng.drive.theta(t)
    lo, hi = eng.pistons
    c_lo, _ = lo.fn(t)
    c_hi, _ = hi.fn(t)
    # lower crank + rod + piston
    big = (540.0 + R * math.sin(th), n["cy_lo"] - R * math.cos(th))
    cv.crank((540.0, n["cy_lo"]), R, th, 34.0, 60.0)
    pin = (540.0, c_lo + 0.5 * lo.height)
    cv.rod(pin, big, 30.0, 28.0, 16.0)
    cv.piston(x0, x1, c_lo, lo.height, pin[1], rings=2)
    # upper crank leads by 12 deg and turns the other way on screen (geared)
    th2 = th + math.radians(12.0)
    big2 = (540.0 - R * math.sin(th2), n["cy_hi"] + R * math.cos(th2))
    cv.crank((540.0, n["cy_hi"]), R, th2 + math.pi, 34.0, 60.0)
    pin2 = (540.0, c_hi - 0.5 * hi.height)
    cv.rod(pin2, big2, 30.0, 28.0, 16.0)
    cv.piston(x0, x1, c_hi, hi.height, pin2[1], rings=2, up=False)
    ig = eng.igniters[0]
    f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
    if f > 0:
        px, py = n["plug"]
        cv.set_group(E.Group("plug", 90.0, (x0, py), (x0, py)))
        cv.plug(x0, py, True, f)
        cv.set_group(eng.groups[0])


# ====================================================================== flathead
def _flathead_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    deck, bdc, tdc, R = n["deck"], n["bdc"], n["tdc"], n["R"]
    cv.set_group(eng.groups[0])
    for b in eng.boxes:
        if b.label.endswith("riser"):
            cv.hatched_rect(b.x0 - 14, b.y0 - 2, b.x1 + 14, b.y1 + 2)
    # a flat head plate over bore and chamber, a wide block under it
    cv.hatched_rect(x0 - WALL, deck - 70, 1054, deck)
    wall_bot = bdc + 120
    cv.hatched_rect(x0 - WALL, deck, 1054, 1330.0)
    cv.rect(x0 - WALL + 8, deck + 60, x0 - WALL + 22, wall_bot - 40, COOLANT, P.BLOCK_EDGE, 1.0)
    cv.rect(560.0, 1120.0, 890.0, 1140.0, COOLANT, P.BLOCK_EDGE, 1.0)
    # crankcase
    cv.hatched_rect(x0 - 90, 1330.0, x1 + 100, cy + R + 90)
    cv.cut(x0 - 50, 1330.0, x1 + 60, cy + R + 50)
    cv.edge(x0 - 50, 1330.0, x1 + 60, cy + R + 50)
    _cut_group(cv, eng, 0, ("duct", "cyl"), ycap=wall_bot)
    cv.cut(x0, deck, x1, wall_bot)
    cv.edge(x0, deck, x1, wall_bot)
    cv.line((x0 - WALL, deck), (1054, deck), P.BLOCK_EDGE, 2.4)
    for bx in (x0 - 20, x1 + 10, 920, 1020):
        cv.bolt(bx, deck - 30, 10.0)
    # camshaft bar under the valves (seen along the block), tappet guides
    cv.hatched_rect(600.0, 1240.0, 840.0, 1262.0, spacing=8.0)
    cv.plug(n["plug"][0], deck)


def _flathead_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    cx, cy = n["crank"]
    R, tdc = n["R"], n["tdc"]
    cv.set_group(eng.groups[0])
    crown, _ = eng.pistons[0].fn(t)
    H = eng.pistons[0].height
    pin = (0.5 * (x0 + x1), crown + 0.5 * H)
    th = eng.drive.theta(t)
    big = (cx + R * math.sin(th), cy - R * math.cos(th))
    cv.crank((cx, cy), R, th, 34.0, 60.0)
    cv.rod(pin, big, 30.0, 28.0, 16.0)
    cv.piston(x0, x1, crown, H, pin[1])
    # side valves open UPWARD from the chamber floor; stems run down to tappets on the cam
    for pk, valve in zip(n["pockets"], eng.valves):
        cxv = 0.5 * (pk[0] + pk[1])
        lp = valve.lift(t) * LIFT_PX
        stem_bot = 1180.0
        cv.valve(cxv, tdc, 0.5 * (pk[1] - pk[0]) + 8, lp, stem_bot, up=True)
        cv.spring((cxv, 1110.0), (cxv, stem_bot - 14 - lp), 5, 22.0)
        cv.rect(cxv - 22, stem_bot - 14 - lp, cxv + 22, stem_bot - 2 - lp, P.VALVE, P.VALVE_LO, 1.0)
        cv.rect(cxv - 18, stem_bot - lp, cxv + 18, stem_bot + 22 - lp, P.CAM_HI, P.CAM, 1.2)
        cv.cam_lobe((cxv, stem_bot + 22 + 44), 44.0, LIFT_PX, valve, t, -90.0)
    ig = eng.igniters[0]
    f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
    if f > 0:
        cv.plug(n["plug"][0], n["deck"], True, f)


# ====================================================================== V-twin
def _vtwin_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    C0, R = n["crank"], n["R"]
    deck, bdc = n["deck"], n["bdc"]
    ph0, ph1 = n["pocket"]
    # crankcase in the world frame, first, so the cylinders sit on it
    cv.set_group(IDENT)
    cv.hatched_rect(C0[0] - 330, C0[1] - 300, C0[0] + 330, C0[1] + R + 110)
    cv.cut(C0[0] - 280, C0[1] - 260, C0[0] + 280, C0[1] + R + 60)
    cv.edge(C0[0] - 280, C0[1] - 260, C0[0] + 280, C0[1] + R + 60)
    for g in range(2):
        cv.set_group(eng.groups[g])
        inside_right = (g == 0)
        for b in eng.boxes:
            if b.group == g and b.label.startswith("intake riser"):
                cv.hatched_rect(b.x0 - 14, b.y0, b.x1 + 14, b.y1 + 2)
            if b.group == g and b.label.startswith("exhaust riser"):
                cv.hatched_rect(b.x0 - 14, b.y0 - 2, b.x1 + 14, b.y1)
        head_top = ph0 - 40
        cv.hatched_rect(206.0, head_top, 874.0, deck)
        # cam cover over the head
        cv.hatched_rect(392.0, head_top - 180, 688.0, head_top)
        cv.cut(406.0, head_top - 166, 674.0, head_top)
        cv.edge(406.0, head_top - 166, 674.0, head_top)
        wall_bot = bdc + 70
        cv.hatched_rect(x0 - WALL, deck, x1 + WALL, wall_bot)
        # fins on the inside of the V only: the exhaust header runs down the outside
        if inside_right:
            cv.fins(x1 + WALL, x1 + WALL + 46, deck + 30, wall_bot - 30, 8)
        else:
            cv.fins(x0 - WALL, x0 - WALL - 46, deck + 30, wall_bot - 30, 8)
        _cut_group(cv, eng, g, hide=36.0)
        cv.cut(x0, deck, x1, wall_bot + 60)
        cv.edge(x0, deck, x1, wall_bot + 60)
        cv.line((206.0, deck), (874.0, deck), P.BLOCK_EDGE, 2.2)
        for bx in (x0 - 18, x1 + 18):
            cv.bolt(bx, deck - 22, 9.0)
        # the exhaust flange that hides the header's open end
        for b in eng.boxes:
            if b.group == g and b.label.startswith("exhaust riser"):
                cv.flange(b.x0 - 22, b.y1 - 44, b.x1 + 22, b.y1 + 4, 2)
        cv.plug(540.0, deck)


def _vtwin_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    C0, R = n["crank"], n["R"]
    deck = n["deck"]
    ph0, ph1 = n["pocket"]
    th = eng.drive.theta(t)
    cv.set_group(IDENT)
    cv.crank(C0, R, th, 40.0, 70.0)
    for g in range(2):
        grp = eng.groups[g]
        cv.set_group(grp)
        crown, _ = eng.pistons[g].fn(t)
        H = eng.pistons[g].height
        pin = (540.0, crown + 0.5 * H)
        a = th - math.radians(grp.angle_deg)
        big = (C0[0] + R * math.sin(a), C0[1] - R * math.cos(a))
        cv.rod(pin, big, 28.0, 28.0, 15.0)
        cv.piston(x0, x1, crown, H, pin[1], rings=2)
        head_top = ph0 - 40
        stem_top = head_top - 96
        vals = [v for v in eng.valves if v.label.endswith(str(g))]
        pockets = [b for b in eng.boxes if b.group == g and b.label.endswith(f"pocket {g}")]
        cam = (540.0, stem_top - 50)
        for b, valve in zip(pockets, vals):
            cxv = 0.5 * (b.x0 + b.x1)
            lp = valve.lift(t) * LIFT_PX
            cv.valve(cxv, deck, 0.5 * (b.x1 - b.x0) + 6, lp, stem_top, disc_h=10.0, stem_w=8.0)
            cv.rect(cxv - 22, head_top - 14, cxv + 22, head_top, P.CAM, P.CAM_HI, 1.0)
            cv.spring((cxv, head_top - 14), (cxv, stem_top + 10 + lp), 5, 20.0)
            cv.rect(cxv - 17, stem_top + 3 + lp, cxv + 17, stem_top + 13 + lp, P.VALVE, P.VALVE_LO, 1.0)
            sgn = -1 if cxv < 540.0 else 1
            cv.finger((540.0 + sgn * 118, stem_top - 8), (cxv, stem_top + 3), lp, 14.0)
        cv.cam_lobe(cam, 40.0, LIFT_PX, vals[0], t, 135.0 if pockets[0].x0 < 540 else 45.0)
        cv.cam_lobe(cam, 40.0, LIFT_PX, vals[1], t, 45.0 if pockets[0].x0 < 540 else 135.0)
        cv.circle(cam[0], cam[1], 15, P.CAM_HI, P.CAM, 1.0)
        ig = eng.igniters[g]
        f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
        if f > 0:
            cv.plug(540.0, deck, True, f)


# ====================================================================== radial
def _radial_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    C0 = n["crank"]
    ph = n["pocket_h"]
    cv.set_group(IDENT)
    # the crankcase / induction ring: hatched disc with the middle cut
    ring_o, ring_i = 335.0, 252.0
    disc = [(C0[0] + ring_o * math.cos(2 * math.pi * i / 72), C0[1] + ring_o * math.sin(2 * math.pi * i / 72))
            for i in range(72)]
    cv.hatched(disc)
    cv.cut_poly([(C0[0] + ring_i * math.cos(2 * math.pi * i / 72), C0[1] + ring_i * math.sin(2 * math.pi * i / 72))
                 for i in range(72)])
    cv.polyline([(C0[0] + ring_i * math.cos(2 * math.pi * i / 72), C0[1] + ring_i * math.sin(2 * math.pi * i / 72))
                 for i in range(73)], P.BLOCK_EDGE, 2.0)
    for j in range(n["n"]):
        cv.set_group(eng.groups[j])
        deck = n["decks"][j]
        # risers along the cylinder, head over the pockets, the finned barrel
        for b in eng.boxes:
            if b.group == j and b.label.startswith(("in riser", "ex riser")):
                cv.hatched_rect(b.x0 - 10, b.y0, b.x1 + 10, b.y1 - 20)
        head_top = deck - ph - 34
        cv.hatched_rect(366.0, head_top, 714.0, deck)
        barrel_bot = C0[1] - 250.0
        cv.hatched_rect(x0 - 16, deck, x1 + 16, barrel_bot + 20)
        cv.fins(x0 - 16, x0 - 44, deck + 20, barrel_bot - 10, 7, th=6.0)
        cv.fins(x1 + 16, x1 + 44, deck + 20, barrel_bot - 10, 7, th=6.0)
        _cut_group(cv, eng, j, hide=34.0)          # riser ends stay under the ring
        cv.cut(x0, deck, x1, barrel_bot + 20)
        cv.edge(x0, deck, x1, barrel_bot + 20)
        cv.line((366.0, deck), (714.0, deck), P.BLOCK_EDGE, 1.8)
        # rocker posts
        for px in (503.0, 577.0):
            cv.rect(px - 7, head_top - 70, px + 7, head_top, P.CAM, P.CAM_HI, 1.0)
        cv.plug(540.0, deck)


def _radial_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    C0, R = n["crank"], n["R"]
    kin = n["kin"]
    ph = n["pocket_h"]
    th = eng.drive.theta(t)
    cv.set_group(IDENT)
    # crank pin and the master rod's big-end flange with the knuckle pins
    Pw = (C0[0] + R * math.sin(th), C0[1] - R * math.cos(th))
    cv.crank(C0, R, th, 30.0, 50.0)
    K = [(C0[0] + kx, C0[1] + ky) for kx, ky in kin.knuckles_local(t)]
    pins = []
    for j in range(n["n"]):
        crown, _ = eng.pistons[j].fn(t)
        pins.append(eng.groups[j].to_world(540.0, crown + kin.h))
    cv.circle(Pw[0], Pw[1], kin.rk + 16, P.ROD, P.ROD_LO, 1.6)
    for j in range(1, n["n"]):
        cv.rod(pins[j], K[j], 16.0, 12.0, 11.0, bolts=False)
    cv.rod(pins[0], Pw, 22.0, kin.rk + 16, 12.0, bolts=False)
    for j in range(1, n["n"]):
        cv.circle(K[j][0], K[j][1], 7.0, P.PIN, P.PIN_RIM, 1.0)
    cv.circle(Pw[0], Pw[1], 13.0, P.PIN, P.PIN_RIM, 1.2)
    for j in range(n["n"]):
        grp = eng.groups[j]
        cv.set_group(grp)
        deck = n["decks"][j]
        crown, _ = eng.pistons[j].fn(t)
        H = eng.pistons[j].height
        cv.piston(x0, x1, crown, H, crown + kin.h, rings=2)
        head_top = deck - ph - 34
        stem_top = head_top - 56
        vals = eng.valves[2 * j: 2 * j + 2]
        for pk, valve in zip(n["pockets"], vals):
            cxv = 0.5 * (pk[0] + pk[1])
            lp = valve.lift(t) * 16.0
            cv.valve(cxv, deck, 0.5 * (pk[1] - pk[0]) + 4, lp, stem_top, disc_h=6.0, stem_w=5.0)
            cv.spring((cxv, head_top - 4), (cxv, stem_top + 6 + lp), 4, 11.0, 2.0)
            sgn = -1 if cxv < 540 else 1
            cv.finger((cxv + sgn * 36, stem_top - 22), (cxv, stem_top + 2), lp, 9.0)
        ig = eng.igniters[j]
        f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
        if f > 0:
            cv.plug(540.0, deck, True, f)



# ====================================================================== radial, music cut
# The radial with its REAL valve gear in the space behind the barrels, which is the space
# Clancy asked to fill (2026-09-20: *"Fill out more space in the background of the engine ...
# the backdrop of the rest of the radial engine just needs to move/whir a little"* and
# *"there should be other working/moving parts, just like the valves ... lots more subtle
# mechanics"*).
#
# THE CAM RING IS DERIVED, NOT DRAWN BY EYE. A radial has one lobed ring, concentric with the
# crank, turning slowly the other way; each cylinder's tappet rides it and pushes a pushrod up
# the barrel to a rocker that opens the valve. So the ring's PROFILE is sampled from the lift
# curve the physics already gates the port with: at cam-relative angle b, the radius is
# base + lift(b) * throw, read off cylinder 0 over one whole cam revolution. Rotate that rigid
# profile and every other cylinder's tappet gets its own lift for free - `cam_check()` measures
# how well, and prints the worst disagreement rather than asserting it away.
#
# Ratio: 3 lobes turning at crank/6 the opposite way, so each tappet meets one lobe per 720
# deg of crank - which is exactly one event per four-stroke cycle. That is the real design.

CAM_LOBES = 3
# THE RATIO IS DERIVED, NOT LOOKED UP. For one rigid ring to serve all five tappets, every
# cylinder's event must land on the same point of the lobe pattern. With tappet j on axis
# 72j, a ring turning at c crank-revs and L lobes, the j-dependent part of the cam-relative
# angle at cylinder j's event is  72j(1-c) + c*eps_j  (eps_j = 360 on even j, the firing-order
# offset catalog.radial uses). Requiring that to be constant mod 360/L reduces to
#     144 (1 - c)  =  k * 360 / L
# and for L = 3 the first solution is c = +1/6: three lobes turning WITH the crank at a sixth
# of its speed, which is the real five-cylinder radial arrangement. It also puts exactly one
# lobe past each tappet per 720 deg, which is one event per four-stroke cycle.
# c = -1/6, the first thing tried, fails: it spreads the five cylinders 24 deg apart on the
# ring, and cam_check() reported the full 26 px of throw as error rather than letting it pass.
CAM_RATIO = 1.0 / 6.0           # cam revolutions per crank revolution, same sense
CAM_BASE = 196.0                # ring base radius, world px from the crank centre
CAM_THROW = 26.0                # lobe height
TAPPET_R = CAM_BASE + CAM_THROW + 12.0
PUSH_X = (451.0, 629.0)         # pushrod tube x in a cylinder's local frame (intake, exhaust)


def _cam_profile(eng, track, n_samp=360):
    """Sample the ring radius against cam-relative angle from cylinder 0's own lift."""
    n = eng.notes["n"]
    rev = eng.rev_seconds
    prof = [CAM_BASE] * n_samp
    valve = eng.valves[track]                     # cylinder 0: valves[0]=intake, [1]=exhaust
    cam_rev_s = rev / abs(CAM_RATIO)              # seconds for one cam revolution
    for i in range(int(n_samp * 3)):
        t = cam_rev_s * i / (n_samp * 3.0)
        lift = valve.lift(t)
        if lift <= 1e-3:
            continue
        b = (_cam_beta(eng, 0, t)) % 360.0
        k = int(round(b / 360.0 * n_samp)) % n_samp
        prof[k] = max(prof[k], CAM_BASE + CAM_THROW * lift)
    # close any single-sample gaps left by the sampling grid
    for _ in range(2):
        prof = [max(prof[k], 0.5 * (prof[k - 1] + prof[(k + 1) % n_samp]) - 0.6)
                for k in range(n_samp)]
    return prof


def _cam_angle(eng, t):
    """The ring's own rotation, world degrees."""
    return CAM_RATIO * eng.drive.deg(t) + eng.notes.get("cam_phase", 0.0)


def _cam_beta(eng, j, t):
    """Cam-relative angle of cylinder j's tappet: its axis direction minus the ring rotation."""
    axis = -90.0 + 360.0 * j / eng.notes["n"]
    return axis - _cam_angle(eng, t)


def _cam_radius(prof, beta):
    n = len(prof)
    x = (beta % 360.0) / 360.0 * n
    i = int(x) % n
    f = x - int(x)
    return prof[i] * (1 - f) + prof[(i + 1) % n] * f


def cam_check(eng, n_t=240):
    """How closely the rigid ring reproduces every cylinder's real lift. Printed by the gate;
    never asserted away - if the geometry cannot be made exact, the number says so."""
    worst = 0.0
    for track in (0, 1):
        prof = eng.notes["cam_prof"][track]
        for j in range(eng.notes["n"]):
            v = eng.valves[2 * j + track]
            for i in range(n_t):
                t = eng.period * i / n_t
                want = CAM_BASE + CAM_THROW * v.lift(t)
                got = _cam_radius(prof, _cam_beta(eng, j, t))
                worst = max(worst, abs(want - got))
    return worst


def _radialx_static(cv, eng):
    n = eng.notes
    x0, x1 = n["bore"]
    C0 = n["crank"]
    ph = n["pocket_h"]
    cv.set_group(IDENT)
    ring_o, ring_i = 352.0, 238.0
    circ = lambda r, k=96: [(C0[0] + r * math.cos(2 * math.pi * i / k),
                             C0[1] + r * math.sin(2 * math.pi * i / k)) for i in range(k)]
    # ---- the exhaust COLLECTOR RING, outboard of everything: where the exhausts go ----
    cv.hatched(circ(470.0), spacing=11.0)
    cv.cut_poly(circ(430.0))
    cv.polyline(circ(430.0) + [circ(430.0)[0]], P.DUCT_EDGE, 2.0)
    cv.polyline(circ(470.0) + [circ(470.0)[0]], P.BLOCK_EDGE, 2.4)
    # ---- crankcase / nose case ----
    cv.hatched(circ(ring_o))
    cv.cut_poly(circ(ring_i))
    cv.polyline(circ(ring_i) + [circ(ring_i)[0]], P.BLOCK_EDGE, 2.0)
    for i in range(20):                      # case bolts round the split line
        a = 2 * math.pi * i / 20
        cv.bolt(C0[0] + 318 * math.cos(a), C0[1] + 318 * math.sin(a), 7.0)
    # ---- the supercharger volute at the centre, which the intakes are fed from ----
    cv.hatched(circ(126.0), spacing=9.0)
    cv.cut_poly(circ(112.0))
    cv.polyline(circ(112.0) + [circ(112.0)[0]], P.DUCT_EDGE, 1.8)
    for j in range(n["n"]):
        cv.set_group(eng.groups[j])
        deck = n["decks"][j]
        for b in eng.boxes:
            if b.group == j and b.label.startswith(("in riser", "ex riser")):
                cv.hatched_rect(b.x0 - 10, b.y0, b.x1 + 10, b.y1 - 20)
        head_top = deck - ph - 34
        cv.hatched_rect(366.0, head_top, 714.0, deck)
        barrel_bot = C0[1] - 250.0
        cv.hatched_rect(x0 - 14, deck, x1 + 14, barrel_bot + 20)
        cv.fins(x0 - 14, x0 - 34, deck + 18, barrel_bot - 10, 8, th=5.0)
        cv.fins(x1 + 14, x1 + 34, deck + 18, barrel_bot - 10, 8, th=5.0)
        # pushrod TUBES: static housings from the case out to the head
        for pxr in PUSH_X:
            cv.rect(pxr - 11, head_top + 6, pxr + 11, C0[1] - TAPPET_R + 6,
                    (52, 58, 68), P.BLOCK_EDGE, 1.4)
        _cut_group(cv, eng, j, hide=34.0)
        cv.cut(x0, deck, x1, barrel_bot + 20)
        cv.edge(x0, deck, x1, barrel_bot + 20)
        cv.line((366.0, deck), (714.0, deck), P.BLOCK_EDGE, 1.8)
        for pxx in (503.0, 577.0):
            cv.rect(pxx - 7, head_top - 70, pxx + 7, head_top, P.CAM, P.CAM_HI, 1.0)
        cv.plug(540.0, deck)


def _radialx_moving(cv, eng, t):
    n = eng.notes
    x0, x1 = n["bore"]
    C0, R = n["crank"], n["R"]
    kin = n["kin"]
    ph = n["pocket_h"]
    th = eng.drive.theta(t)
    cv.set_group(IDENT)

    # ---- the accessory gear train, behind everything, whirring ----
    gdeg = eng.drive.deg(t)
    for gi, (ga, gr, gteeth, gspd) in enumerate(((18.0, 300.0, 22, -0.5), (90.0, 300.0, 18, 0.62),
                                                 (162.0, 300.0, 14, -0.8), (234.0, 300.0, 18, 0.62),
                                                 (306.0, 300.0, 22, -0.5))):
        a = math.radians(ga)
        gx, gy = C0[0] + gr * math.cos(a), C0[1] + gr * math.sin(a)
        _gear(cv, gx, gy, 34.0, gteeth, gdeg * gspd)
    # ---- THE CAM RING: two rigid lobed tracks, turning WITH the crank at a sixth of its
    # speed. Drawn as an ANNULUS - a cam ring is a ring, and the crank and the blower have to
    # be visible through it - so each track is filled to its lobed profile and then cut back
    # to its own inner bore. Cut before the impeller and the rods, or it erases them.
    cang = _cam_angle(eng, t)
    tracks = ((0, P.CAM, 1.00, 0.86), (1, P.CRANK_LO, 0.80, 0.62))
    for track, col, scale, bore in tracks:
        prof = n["cam_prof"][track]
        pts = [(C0[0] + _cam_radius(prof, 360.0 * i / 220.0) * scale
                * math.cos(math.radians(360.0 * i / 220.0 + cang)),
                C0[1] + _cam_radius(prof, 360.0 * i / 220.0) * scale
                * math.sin(math.radians(360.0 * i / 220.0 + cang))) for i in range(220)]
        cv.poly(pts, col, P.CAM_HI, 1.4)
        ri = CAM_BASE * scale * bore
        cv.cut_poly([(C0[0] + ri * math.cos(2 * math.pi * i / 96),
                      C0[1] + ri * math.sin(2 * math.pi * i / 96)) for i in range(96)])
        cv.polyline([(C0[0] + ri * math.cos(2 * math.pi * i / 96),
                      C0[1] + ri * math.sin(2 * math.pi * i / 96)) for i in range(97)],
                    P.CAM_HI, 1.2)

    # ---- the supercharger impeller, spinning fast at the centre, seen through the ring ----
    _impeller(cv, C0[0], C0[1], 96.0, 11, gdeg * 2.4)

    # ---- crank, master rod, articulated rods (unchanged physics, drawn over the cam) ----
    Pw = (C0[0] + R * math.sin(th), C0[1] - R * math.cos(th))
    cv.crank(C0, R, th, 30.0, 50.0)
    K = [(C0[0] + kx, C0[1] + ky) for kx, ky in kin.knuckles_local(t)]
    pins = []
    for j in range(n["n"]):
        crown, _ = eng.pistons[j].fn(t)
        pins.append(eng.groups[j].to_world(540.0, crown + kin.h))
    cv.circle(Pw[0], Pw[1], kin.rk + 16, P.ROD, P.ROD_LO, 1.6)
    for j in range(1, n["n"]):
        cv.rod(pins[j], K[j], 16.0, 12.0, 11.0, bolts=False)
    cv.rod(pins[0], Pw, 22.0, kin.rk + 16, 12.0, bolts=False)
    for j in range(1, n["n"]):
        cv.circle(K[j][0], K[j][1], 7.0, P.PIN, P.PIN_RIM, 1.0)
    cv.circle(Pw[0], Pw[1], 13.0, P.PIN, P.PIN_RIM, 1.2)

    # ---- per cylinder: piston, tappet + pushrod + rocker, valves, plug ----
    for j in range(n["n"]):
        cv.set_group(eng.groups[j])
        deck = n["decks"][j]
        crown, _ = eng.pistons[j].fn(t)
        H = eng.pistons[j].height
        cv.piston(x0, x1, crown, H, crown + kin.h, rings=2)
        head_top = deck - ph - 34
        stem_top = head_top - 56
        vals = eng.valves[2 * j: 2 * j + 2]
        for track, (pk, valve) in enumerate(zip(n["pockets"], vals)):
            cxv = 0.5 * (pk[0] + pk[1])
            lift = valve.lift(t)
            lp = lift * 16.0
            pxr = PUSH_X[track]
            # the tappet rides the ring; the pushrod carries that lift to the rocker
            tap_y = C0[1] - (TAPPET_R + CAM_THROW * lift)
            rk_y = stem_top - 22 - CAM_THROW * lift * 0.55
            cv.rect(pxr - 13, tap_y - 14, pxr + 13, tap_y + 12, P.CAM_HI, P.CAM, 1.2)
            cv.line((pxr, tap_y - 12), (pxr, rk_y), P.VALVE, 7.0)
            cv.circle(pxr, rk_y, 7.0, P.CAM_HI, P.CAM, 1.0)
            sgn = -1 if cxv < 540 else 1
            cv.finger((cxv + sgn * 36, rk_y), (cxv, stem_top + 2), lp, 9.0)
            cv.valve(cxv, deck, 0.5 * (pk[1] - pk[0]) + 4, lp, stem_top, disc_h=6.0, stem_w=5.0)
            cv.spring((cxv, head_top - 4), (cxv, stem_top + 6 + lp), 4, 11.0, 2.0)
        ig = eng.igniters[j]
        f = _flash(ig.clock, ig.spark_deg, eng.cycle_deg, t)
        if f > 0:
            cv.plug(540.0, deck, True, f)


def _gear(cv, cx, cy, r, teeth, deg):
    """A spur gear as a toothed disc - cheap, and it reads as driven because it turns."""
    pts = []
    for i in range(teeth * 4):
        a = math.radians(deg) + 2 * math.pi * i / (teeth * 4)
        rr = r if (i % 4) in (1, 2) else r * 0.86
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    cv.poly(pts, P.CRANK, P.CRANK_LO, 1.2)
    cv.circle(cx, cy, r * 0.42, P.CRANK_HI, P.CRANK_LO, 1.0)
    cv.circle(cx, cy, r * 0.16, P.PIN, P.PIN_RIM, 1.0)


def _impeller(cv, cx, cy, r, blades, deg):
    """The supercharger rotor the intake risers are fed from: curved blades, spinning fast."""
    for i in range(blades):
        a0 = math.radians(deg) + 2 * math.pi * i / blades
        pts = []
        for s in range(7):
            u = s / 6.0
            a = a0 + 0.85 * u
            rr = r * (0.30 + 0.70 * u)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        cv.polyline(pts, P.CAM_HI, 4.0)
    cv.circle(cx, cy, r * 0.30, P.CRANK, P.CAM_HI, 1.4)
    cv.circle(cx, cy, r * 0.11, P.PIN, P.PIN_RIM, 1.0)

LOOKS = {
    "sohc": (lambda cv, e: _single_static(cv, e, "sohc"), lambda cv, e, t: _single_moving(cv, e, t, "sohc")),
    "dohc": (lambda cv, e: _single_static(cv, e, "dohc"), lambda cv, e, t: _single_moving(cv, e, t, "dohc")),
    "diesel": (lambda cv, e: _single_static(cv, e, "diesel"), lambda cv, e, t: _single_moving(cv, e, t, "diesel")),
    "legacy": (lambda cv, e: _single_static(cv, e, "legacy"), lambda cv, e, t: _single_moving(cv, e, t, "legacy")),
    "two_stroke": (_two_stroke_static, _two_stroke_moving),
    "opposed": (_opposed_static, _opposed_moving),
    "flathead": (_flathead_static, _flathead_moving),
    "vtwin": (_vtwin_static, _vtwin_moving),
    "radial": (_radial_static, _radial_moving),
    "radial_flash": (_radialx_static, _radialx_moving),
}


def draw_static(cv, eng):
    LOOKS[eng.notes["head"]][0](cv, eng)


def draw_moving(cv, eng, t):
    LOOKS[eng.notes["head"]][1](cv, eng, t)
