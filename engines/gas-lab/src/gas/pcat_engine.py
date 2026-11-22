# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pcat_engine.py - the parts set's cylinder: a DOHC four-stroke with a pent-roof head.

What is new against the 2026-09-20 singles, all of it detail rather than more parts:

  VALVES ARE SOLIDS. The old engine gated a port's WIDTH by the lift (a box edge that opened
  from the middle out). Here each poppet valve is a shaped body that moves along its own stem
  axis, tilted 30 degrees, in a CURVED port cut through a sectioned head; the gas finds the gap
  between the valve face and its seat by itself. Shut, the head overlaps the seat ring by 10 px
  each side: a real seal.
  PENT ROOF. The combustion chamber is the roof's triangle plus the squish gap, sized so the
  area ratio is ~5:1 (gamma = 2 in 2D: what 10:1 does in a real engine).
  THE ROD NO LONGER STRETCHES. The kinematics solve the WRIST PIN, and the crown is the pin
  less its offset, so the drawn rod is exactly L between two pins every frame (the old engine
  put the pin at the crown and drew it 60 px lower: ~2 % stretch).
  CAMS FROM THE LIFT CURVE. Each lobe is generated from the lift function that moves its valve,
  so the lobe you see pushing the bucket is the lift the gas gets.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from . import shapes as S
from .engine import (BOT, LEFT, RIGHT, TOP, Box, Drive, Engine, Igniter, Piston, cam_lift,
                     clock, slider_crank)
from .pcatalog import (ALU, BRASS, CERAMIC, DARK, IRON, RUBBER, STEEL, STEEL_DK, _bolt_ring,
                       _page)
from .sdf import Solid
from .solids import Body, const_pose, spin_pose

OPEN_L = (False, False, True, False)
OPEN_R = (False, False, False, True)
IVO, IVC, EVO, EVC, SPARK = -12.0, 230.0, 490.0, 732.0, 340.0


def four_stroke(rev=4.0):
    """The hero cylinder. See the module docstring."""
    drive = Drive(rev)
    eng = Engine(name="four_stroke", rev_seconds=rev, drive=drive, cycle_deg=720.0,
                 fuel_frac=0.30, q_rxn=9.5e7, groups=_page())
    xm, x0, x1 = 540.0, 280.0, 800.0              # bore 520
    cx, cy = 540.0, 1540.0                         # crank centre
    R, L, h_pin = 230.0, 560.0, 70.0               # stroke 460, rod 560 pin-to-pin
    tdc_crown = cy - (R + L) - h_pin               # 680
    bdc_crown = cy - (L - R) - h_pin               # 1140
    deck = tdc_crown - 40.0                        # 640: a 40 px squish gap at TDC
    roof_h = 150.0
    apex = (xm, deck - roof_h)                     # 490
    eng.notes.update(part=True, bore=(x0, x1), crank=(cx, cy), R=R, L=L, deck=deck)
    # ---- the piston: the crown IS a box floor (exact), solved from the pin ----
    pfn = slider_crank(cy - h_pin, R, L, drive)
    eng.pistons.append(Piston(0, pfn, True, 230.0, "piston"))
    # ---- boxes: cylinder, head (chamber + both ports), two runners ----
    cyl = eng.add_box(Box(x0, x1, deck, 1200.0, 0, "cyl", "", 0, -1, "crown",
                          (False, False, False, False), 1.0, [], "cylinder"))
    head = eng.add_box(Box(40.0, 1040.0, 180.0, deck, 0, "duct", "", label="head"))
    eng.connect(cyl, head, TOP, x0, x1)
    run_in = eng.add_box(Box(-160.0, 40.0, 290.0, 450.0, 0, "duct", "intake", open=OPEN_L,
                             label="intake runner"))
    # the exhaust pipe is full of EXHAUST: an ambient-air reservoir there backflowed fresh blue
    # air into the port and the first regraded final showed "blue out"
    run_ex = eng.add_box(Box(1040.0, 1240.0, 290.0, 450.0, 0, "duct", "exhaust", open=OPEN_R,
                             label="exhaust runner", res_T=700.0,
                             res_mix={C.CO2: 1.0 / 3.0, C.H2O: 2.0 / 3.0}))
    eng.connect(head, run_in, LEFT, 292.0, 448.0)
    eng.connect(head, run_ex, RIGHT, 292.0, 448.0)

    # ---- geometry of the two valves (intake left, exhaust right; mirror images) ----
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    valves = []
    for side in (-1, 1):
        seat = (xm + side * 130.0, deck - roof_h / 2)                      # mid-roof-plane
        up = (side * s30, -c30)                                            # stem axis, outward
        valves.append(dict(side=side, seat=seat, up=up, ang=side * math.radians(30.0)))
    L_lift, stem_len = 55.0, 350.0

    # ---- the head casting: sectioned, with chamber, ports, guides, jackets ----
    Hd = Solid((0.0, 0.0, 1080.0, deck + 36.0), h=0.8, bevel=6.0, name="head")
    Hd.add(S.rect(10.0, 170.0, 1070.0, deck + 30.0))
    Hd.sub(np.array([(x0, deck + 40.0), (x0, deck), apex, (x1, deck), (x1, deck + 40.0)]))
    ports = []
    for v in valves:
        sd, (sx, sy) = v["side"], v["seat"]
        ux, uy = v["up"]
        p0 = (sx - 14.0 * ux, sy - 14.0 * uy)                          # just inside the chamber
        p1 = (sx + 150.0 * ux, sy + 150.0 * uy)
        p2 = (xm + sd * 330.0, 372.0)
        p3 = (xm + sd * 610.0, 370.0)      # past the head's outer wall: the first build
                                              # stopped 10 px short and sealed both ports
        cl = S.bezier(p0, p1, p2, p3, 60)
        band = S.polyline_band(cl, 80.0)
        Hd.sub(band)
        ports.append(cl)
    # the valve guides (bronze) and spring pockets: relief along each stem
    for v in valves:
        (sx, sy), (ux, uy), a = v["seat"], v["up"], v["ang"]
        g0 = (sx + 205.0 * ux, sy + 205.0 * uy)
        guide = S.shift(S.rot(S.rect(-17, -60, 17, 60), a), *g0)
        Hd.recess(guide, depth=3.0, bevel=1.0)
        Hd.paint(guide, (150, 116, 76), feather=1.0)
        pocket = S.shift(S.rot(S.rect(-58, -70, 58, 70), a), sx + 320.0 * ux, sy + 320.0 * uy)
        Hd.recess(pocket, depth=8.0, bevel=2.0)
        Hd.paint(pocket, (40, 44, 50), feather=2.0)
    for (wx, wy, w, h) in ((120.0, 520.0, 110.0, 70.0), (850.0, 520.0, 110.0, 70.0),
                           (540.0, 250.0, 150.0, 60.0)):
        jk = S.rrect(wx - w / 2, wy - h / 2, wx + w / 2, wy + h / 2, 22.0)
        Hd.recess(jk, depth=7.0, bevel=3.0)
        Hd.paint(jk, (34, 56, 78), feather=3.0)                          # coolant jacket
    for bx in (70.0, 1010.0):
        Hd.boss(S.circle(bx, deck - 30.0, 14.0), height=3.0, bevel=1.5)
        Hd.recess(S.circle(bx, deck - 30.0, 7.0, 6), depth=3.0, bevel=1.0)
    Hd.done(ALU, brush="cast", sectioned=True, spec=0.28)
    eng.bodies.append(Body(Hd, const_pose(), z=0, static=True, name="head"))

    # ---- the runner pipes, out to the frame edges ----
    for sd in (-1, 1):
        xa, xb = (-180.0, 60.0) if sd < 0 else (1020.0, 1260.0)
        Rp = Solid((xa, 240.0, xb, 500.0), h=0.8, bevel=5.0, name="runner")
        Rp.add(S.rect(xa, 252.0, xb, 488.0)).sub(S.rect(xa - 10, 290.0, xb + 10, 450.0))
        fx = 40.0 if sd < 0 else 1040.0
        Rp.boss(S.rect(fx - 22.0, 252.0, fx + 22.0, 488.0), height=4.0, bevel=2.0)
        for by in (268.0, 472.0):
            Rp.boss(S.circle(fx, by, 9.0), height=2.5, bevel=1.2)
        Rp.done(ALU if sd < 0 else (104, 88, 82), brush="cast", sectioned=True, spec=0.28)
        eng.bodies.append(Body(Rp, const_pose(), physics=False, static=True, z=1, name="runner"))

    # ---- the cam carrier and cover above the head (drawn, and fills the top of the frame) ----
    Cv = Solid((0.0, -40.0, 1080.0, 200.0), h=0.8, bevel=6.0, name="cam cover")
    Cv.add(S.rrect(30.0, 0.0, 1050.0, 190.0, 50.0))
    cams = []
    for v in valves:
        (sx, sy), (ux, uy) = v["seat"], v["up"]
        top = (sx + stem_len * ux, sy + stem_len * uy)
        cc = (top[0] + 128.0 * ux, top[1] + 128.0 * uy)
        cams.append(cc)
        Cv.sub(S.circle(cc[0], cc[1], 128.0))
    Cv.sub(S.rect(xm - 40.0, 100.0, xm + 40.0, 200.0))
    Cv.done(ALU, brush="cast", sectioned=True, spec=0.28)
    eng.bodies.append(Body(Cv, const_pose(), physics=False, z=1, static=True, name="cam cover"))

    # ---- the valves (physics) ----
    clk = clock(drive, 720.0, 0.0)
    timing = ((IVO, IVC), (EVO, EVC))
    Vs = []
    for k, v in enumerate(valves):
        Vv = Solid((-104.0, -stem_len - 6.0, 104.0, 6.0), h=0.4, bevel=4.0, name="valve")
        tulip = np.concatenate([
            np.array([(-100.0, 0.0), (100.0, 0.0), (100.0, -16.0)]),
            S.bezier((100.0, -16.0), (60.0, -26.0), (20.0, -50.0), (12.0, -96.0), 24),
            np.array([(12.0, -stem_len), (-12.0, -stem_len)]),
            S.bezier((-12.0, -96.0), (-20.0, -50.0), (-60.0, -26.0), (-100.0, -16.0), 24)])
        Vv.add(tulip)
        Vv.recess(S.rect(-12.0, -stem_len + 6.0, 12.0, -stem_len + 22.0), depth=3.0, bevel=1.0)
        Vv.done((208, 212, 220) if k == 0 else (190, 178, 168), brush="linear", spec=0.75)
        Vs.append(Vv)

    def lift_fn(k):
        o, c_ = timing[k]
        span = c_ - o
        rate = 360.0 / (2 * math.pi)

        def f(t):
            d = clk(t)
            l = cam_lift(d, o, c_, 720.0)
            # d(lift)/dt: the half-sine's slope times the crank rate
            dd = (d - o) % 720.0
            dl = (math.pi / span) * math.cos(math.pi * dd / span) if (l > 0 and dd <= span) else 0.0
            return l, dl * drive.omega(t) * rate
        return f
    lifts = [lift_fn(0), lift_fn(1)]

    def valve_pose(k):
        v = valves[k]
        (sx, sy), (ux, uy) = v["seat"], v["up"]

        def f(t):
            l, dl = lifts[k](t)
            return (sx - L_lift * l * ux, sy - L_lift * l * uy, v["ang"],
                    -L_lift * dl * ux, -L_lift * dl * uy, 0.0)
        return f
    for k in range(2):
        eng.bodies.append(Body(Vs[k], valve_pose(k), z=6, name=("intake valve", "exhaust valve")[k]))

    # ---- valve train (drawn): buckets, springs, cams ----
    Bk = Solid((-52.0, -66.0, 52.0, 4.0), h=0.4, bevel=5.0, name="bucket")
    Bk.add(S.rrect(-48.0, -62.0, 48.0, 0.0, 8.0))
    Bk.recess(S.rect(-40.0, -54.0, 40.0, -46.0), depth=2.0, bevel=1.0)
    Bk.done((170, 176, 186), brush="radial", brush_c=(0.0, -31.0), spec=0.7)

    def bucket_pose(k):
        v = valves[k]
        (ux, uy) = v["up"]

        def f(t):
            x, y, a, vx, vy, _ = valve_pose(k)(t)
            return (x + stem_len * ux, y + stem_len * uy, a, vx, vy, 0.0)
        return f
    Wi = Solid((-9.0, -9.0, 9.0, 9.0), h=0.3, bevel=6.0, name="wire")
    Wi.add(S.circle(0.0, 0.0, 7.5))
    Wi.done((200, 204, 212), brush="radial", spec=0.7)
    n_coil = 6

    def spring_pose(k, j, s2):
        v = valves[k]
        (sx, sy), (ux, uy) = v["seat"], v["up"]
        px_, py_ = -uy, ux                           # across the stem
        base = 255.0                                 # the spring seat on the head

        def f(t):
            x, y, a, vx, vy, _ = valve_pose(k)(t)
            top = stem_len - 30.0                    # the retainer, moving with the valve
            b_ = (sx + base * ux, sy + base * uy)
            r_ = (x + top * ux, y + top * uy)
            u = (j + 0.5 * s2 + 0.25) / (n_coil + 0.5)
            return (b_[0] + (r_[0] - b_[0]) * u + s2 * 0 + (1 if s2 else -1) * 42.0 * px_,
                    b_[1] + (r_[1] - b_[1]) * u + (1 if s2 else -1) * 42.0 * py_, 0.0,
                    vx * u, vy * u, 0.0)
        return f
    Rt = Solid((-60.0, -12.0, 60.0, 12.0), h=0.4, bevel=4.0, name="retainer")
    Rt.add(S.rrect(-56.0, -9.0, 56.0, 9.0, 4.0))
    Rt.done(STEEL_DK, brush="linear", spec=0.5)

    def retainer_pose(k):
        v = valves[k]
        (ux, uy) = v["up"]

        def f(t):
            x, y, a, vx, vy, _ = valve_pose(k)(t)
            return (x + (stem_len - 24.0) * ux, y + (stem_len - 24.0) * uy, a, vx, vy, 0.0)
        return f
    base_r = 62.0
    for k in range(2):
        eng.bodies.append(Body(Bk, bucket_pose(k), physics=False, z=8, name="bucket"))
        eng.bodies.append(Body(Rt, retainer_pose(k), physics=False, z=9, name="retainer"))
        for j in range(n_coil):
            for s2 in (0, 1):
                eng.bodies.append(Body(Wi, spring_pose(k, j, s2), physics=False, z=9,
                                       name="valve spring"))
        # the cam: lobe generated from the lift curve that moves this valve
        v = valves[k]
        ux, uy = v["up"]
        psi = math.atan2(-uy, -ux)                  # contact direction: toward the bucket
        o, c_ = timing[k]
        gam = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
        dgs = np.degrees(2.0 * (psi - gam)) % 720.0
        rr = np.array([base_r + L_lift * cam_lift(d, o, c_, 720.0) for d in dgs])
        prof = np.stack([rr * np.cos(gam), rr * np.sin(gam)], 1)
        Cm = Solid((-base_r - L_lift - 6, -base_r - L_lift - 6, base_r + L_lift + 6,
                    base_r + L_lift + 6), h=0.4, bevel=6.0, name="cam")
        Cm.add(prof)
        Cm.boss(S.circle(0.0, 0.0, 40.0), height=5.0, bevel=2.0)
        Cm.recess(S.circle(0.0, 0.0, 16.0, 6), depth=4.0, bevel=1.0)
        Cm.paint(S.circle(0.0, 0.0, 16.0, 6), DARK)
        Cm.done((164, 170, 182), brush="radial", spec=0.8)
        # check the lobe against the bucket: the profile point facing the bucket at t is the
        # lift the valve has at t (the cam turns at half crank speed, zero phase)
        eng.bodies.append(Body(Cm, spin_pose(cams[k][0], cams[k][1], drive, 0.5, 0.0),
                               physics=False, z=7, name="cam"))
    eng.notes["cams"] = cams

    # ---- the plug, between the valves at the roof apex ----
    Pl = Solid((xm - 30.0, 170.0, xm + 30.0, apex[1] + 20.0), h=0.4, bevel=3.0, name="plug")
    Pl.add(S.rect(xm - 13.0, apex[1] - 70.0, xm + 13.0, apex[1] + 2.0))
    Pl.add(S.rect(xm - 24.0, apex[1] - 104.0, xm + 24.0, apex[1] - 70.0))
    Pl.add(S.rect(xm - 14.0, 180.0, xm + 14.0, apex[1] - 104.0))
    Pl.add(S.rect(xm - 3.0, apex[1], xm + 3.0, apex[1] + 16.0))
    Pl.add(S.rect(xm + 5.0, apex[1], xm + 11.0, apex[1] + 20.0))
    Pl.paint(S.rect(xm - 15.0, 180.0, xm + 15.0, apex[1] - 104.0), CERAMIC)
    Pl.paint(S.rect(xm - 4.0, apex[1] - 2.0, xm + 12.0, apex[1] + 22.0), (220, 190, 120))
    Pl.done((176, 182, 192), brush="linear", spec=0.5)
    eng.bodies.append(Body(Pl, const_pose(), physics=False, static=True, z=2, name="plug"))

    # ---- block, liner, crankcase (static) ----
    Bl = Solid((0.0, deck - 8.0, 1080.0, 1900.0), h=0.9, bevel=6.0, name="block")
    Bl.add(S.rrect(60.0, deck - 4.0, 1020.0, 1250.0, 20.0))
    Bl.sub(S.rect(x0, deck - 20.0, x1, 1260.0))
    Bl.add(S.rrect(90.0, 1240.0, 990.0, 1880.0, 60.0))
    Bl.sub(S.rrect(150.0, 1236.0, 930.0, 1830.0, 50.0))
    Bl.add(S.rect(x0 - 32.0, 1180.0, x0, 1262.0)).add(S.rect(x1, 1180.0, x1 + 32.0, 1262.0))
    for s2 in (-1, 1):                               # water jacket around the liner
        jx = xm + s2 * 330.0
        jk = S.rrect(jx - 42.0, deck + 30.0, jx + 42.0, 1150.0, 30.0)
        Bl.recess(jk, depth=7.0, bevel=3.0)
        Bl.paint(jk, (34, 56, 78), feather=3.0)
    Bl.recess(S.rect(x0 - 32.0, deck, x0 - 20.0, 1180.0), depth=2.0, bevel=1.0)   # liner line
    Bl.recess(S.rect(x1 + 20.0, deck, x1 + 32.0, 1180.0), depth=2.0, bevel=1.0)
    for bx in (120.0, 960.0):
        for by in (720.0, 1180.0):
            Bl.boss(S.circle(bx, by, 14.0), height=3.0, bevel=1.5)
            Bl.recess(S.circle(bx, by, 7.0, 6), depth=3.0, bevel=1.0)
    Bl.done(IRON, brush="cast", sectioned=True, spec=0.22)
    eng.bodies.append(Body(Bl, const_pose(), physics=False, static=True, z=0, name="block"))
    # the main bearing (drawn behind the crank)
    Mb = Solid((cx - 90.0, cy - 90.0, cx + 90.0, cy + 90.0), h=0.5, bevel=4.0, name="main bearing")
    Mb.add(S.circle(cx, cy, 84.0)).sub(S.circle(cx, cy, 58.0))
    Mb.paint(S.circle(cx, cy, 64.0), BRASS, feather=2.0)
    Mb.done(STEEL_DK, brush="radial", brush_c=(cx, cy), spec=0.4)
    eng.bodies.append(Body(Mb, const_pose(), physics=False, static=True, z=3, name="main bearing"))

    # ---- piston, rod, crank (drawn from the same kinematics as the crown) ----
    pw = (x1 - x0) - 4.0
    Pn = Solid((-pw / 2 - 4, -4.0, pw / 2 + 4, 234.0), h=0.5, bevel=6.0, name="piston")
    Pn.add(np.concatenate([np.array([(-pw / 2, 0.0), (pw / 2, 0.0)]),
                           S.bezier((pw / 2, 0.0), (pw / 2, 120.0), (pw / 2 - 6, 200.0), (pw / 2 - 10, 230.0), 16),
                           S.bezier((-pw / 2 + 10, 230.0), (-pw / 2 + 6, 200.0), (-pw / 2, 120.0), (-pw / 2, 0.0), 16)]))
    for j, ry in enumerate((22.0, 42.0, 62.0)):
        Pn.recess(S.rect(-pw / 2, ry - 5.0, pw / 2, ry + 5.0), depth=4.0, bevel=0.8)
        Pn.paint(S.rect(-pw / 2, ry - 4.0, pw / 2, ry + 4.0), (58, 62, 70) if j < 2 else (84, 88, 96))
    Pn.recess(S.circle(0.0, h_pin, 62.0), depth=6.0, bevel=4.0)        # the pin boss
    Pn.boss(S.circle(0.0, h_pin, 38.0), height=4.0, bevel=2.0)
    Pn.paint(S.circle(0.0, h_pin, 30.0), (120, 126, 136))
    for s2 in (-1, 1):
        Pn.recess(S.rect(s2 * 100.0 - 50.0, 120.0, s2 * 100.0 + 50.0, 210.0), depth=4.0, bevel=3.0)
    Pn.done((168, 174, 184), brush="linear", spec=0.6)

    def piston_pose(t):
        y, v = pfn(t)
        return (xm, y, 0.0, 0.0, v, 0.0)
    eng.bodies.append(Body(Pn, piston_pose, physics=False, z=5, name="piston"))

    def pins(t):
        th = drive.theta(t)
        y, _ = pfn(t)
        return (xm, y + h_pin), (cx + R * math.sin(th), cy - R * math.cos(th))
    Rd = Solid((-86.0, -52.0, 86.0, L + 90.0), h=0.5, bevel=6.0, name="rod")
    shank = np.array([(-30.0, 30.0), (30.0, 30.0), (46.0, L - 70.0), (-46.0, L - 70.0)])
    Rd.add(shank).add(S.circle(0.0, 0.0, 46.0)).add(S.circle(0.0, L, 84.0))
    Rd.sub(S.circle(0.0, 0.0, 26.0)).sub(S.circle(0.0, L, 56.0))
    Rd.recess(np.array([(-16.0, 50.0), (16.0, 50.0), (26.0, L - 90.0), (-26.0, L - 90.0)]),
              depth=7.0, bevel=3.0)                                       # the I-beam web
    Rd.recess(S.rect(-84.0, L - 5.0, 84.0, L + 5.0), depth=2.0, bevel=0.8)   # cap split line
    for s2 in (-1, 1):
        Rd.boss(S.rect(s2 * 70.0 - 12.0, L - 30.0, s2 * 70.0 + 12.0, L + 40.0), height=4.0, bevel=2.0)
    Rd.paint(S.circle(0.0, L, 60.0), BRASS, feather=2.0)
    Rd.done((172, 160, 136), brush="linear", spec=0.55)                 # forged steel

    def rod_pose(t):
        (px_, py_), (qx, qy) = pins(t)
        ang = math.atan2(-(qx - px_), qy - py_)
        return (px_, py_, ang, 0.0, 0.0, 0.0)
    eng.bodies.append(Body(Rd, rod_pose, physics=False, z=6, name="rod"))
    Wp = Solid((-30.0, -30.0, 30.0, 30.0), h=0.4, bevel=4.0, name="wrist pin")
    Wp.add(S.circle(0.0, 0.0, 25.0)).sub(S.circle(0.0, 0.0, 12.0))
    Wp.done((196, 200, 208), brush="radial", spec=0.8)
    eng.bodies.append(Body(Wp, lambda t: (pins(t)[0][0], pins(t)[0][1], 0.0, 0.0, 0.0, 0.0),
                           physics=False, z=7, name="wrist pin"))
    Ck = Solid((-300.0, -R - 70.0, 300.0, 300.0), h=0.6, bevel=7.0, name="crank")
    cw = np.concatenate([S.arc(0.0, 0.0, 285.0, 35.0, 145.0), S.arc(0.0, 0.0, 70.0, 150.0, 30.0)])
    Ck.add(cw)
    Ck.add(np.array([(-78.0, 0.0), (-58.0, -R), (58.0, -R), (78.0, 0.0)]))
    Ck.add(S.circle(0.0, 0.0, 88.0)).add(S.circle(0.0, -R, 62.0))
    Ck.recess(S.circle(0.0, -R, 54.0), depth=3.0, bevel=1.5)
    Ck.paint(S.circle(0.0, -R, 54.0), (196, 200, 208))
    Ck.boss(S.circle(0.0, 0.0, 56.0), height=4.0, bevel=2.0)
    Ck.paint(S.circle(0.0, 0.0, 54.0), (196, 200, 208))
    for j in range(3):                              # drillings in the counterweight
        a = math.radians(70.0 + 20.0 * j)
        Ck.recess(S.circle(210.0 * math.cos(a), 210.0 * math.sin(a), 22.0), depth=8.0, bevel=3.0)
    Ck.done((128, 134, 146), brush="radial", spec=0.5)
    eng.bodies.append(Body(Ck, spin_pose(cx, cy, drive, 1.0, 0.0), physics=False, z=4, name="crank"))

    # ---- ignition: the front grows from the plug and burns only BELOW the roof ----
    def chamber(world):
        rx = xp_abs(world.x - xm)
        return (world.box == cyl) | ((world.box == head) & (world.y > apex[1] + roof_h * rx / 260.0 - 3.0))
    eng.igniters.append(Igniter((cyl, head), (xm, apex[1] + 18.0), clk, SPARK, 0.0, False,
                                520.0 / 27.0, label="plug", region=chamber))

    def over(t):
        d = (clk(t) - SPARK) % 720.0
        amp = math.exp(-d / 3.0) if d < 14.0 else 0.0
        return [(xm + 4.0, apex[1] + 14.0, 7.0, (0.85, 0.92, 1.0), 1.8 * amp)]
    eng.notes["over"] = over

    def gate(world):
        from .gpu import xp
        m = chamber(world)
        n = int(m.sum())
        T = C.T0 * float(xp.where(m, world.vx ** 2 + world.vy ** 2, 0.0).sum()) / max(n, 1) / C.V_TH ** 2
        y, _ = pfn(world.t)
        area = (x1 - x0) * (y - deck) + (x1 - x0) * roof_h / 2.0
        return {"T_chamber": T, "P_chamber": n * T / area / (C.N0 * C.T0),
                "lift_in": lifts[0](world.t)[0], "lift_ex": lifts[1](world.t)[0]}
    eng.notes["gate_probe"] = gate
    eng.notes.update(chamber_fn=chamber, clock=clk, cyl_box=cyl, head_box=head,
                     run_in=run_in, run_ex=run_ex, bore=(x0, x1), stroke=2 * R)
    eng.notes["warm"] = 8.0
    eng.notes["heat_look"] = True
    eng.notes["T_range"] = (280.0, 1800.0)       # burnt gas at 800-1600 K reads ember, not brick
    eng.blurb = "DOHC pent-roof four-stroke: valves that move, cams cut from their own lift"
    return eng


def xp_abs(a):
    from .gpu import xp
    return xp.abs(a)


def four_stroke_ledger():
    """2026-09-26 variation (Clancy: "keep the four stroke video as the top half ... on the
    bottom half ... the input of fuel, and oxygen on the left, and the gas, and water on the
    right with amounts ... have the four stroke cylinder speed up throughout the video").

    The same cylinder, but faster at base (2.5 s/rev) and accelerating for the WHOLE clip
    (ledger.py sets the drive). To keep the ports from choking at the top speed the gas is
    run hotter in thermal speed: V_TH 9000 -> 13000 with the substep rate raised in step
    (3 px per substep), exactly as the radial music cut did; q_rxn is quoted at Q_REF_VTH and
    rescaled by the world, so the temperature a burn buys is unchanged."""
    C.V_TH = 13000.0
    eng = four_stroke(rev=2.5)
    eng.name = "four_stroke_ledger"
    eng.notes["substeps"] = 4300
    eng.notes["zoom"] = 0.5
    eng.notes["view"] = (540.0, 930.0)
    eng.notes["warm"] = 6.0
    return eng
