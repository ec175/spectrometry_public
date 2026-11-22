# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pcat_rotary.py - the parts set's orbiting and sliding machines: scroll, vane pump, Wankel.

Split out of pcatalog.py only for length; the conventions are pcatalog's (reference px, y down,
every solid drawn from the grid the gas meets).
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from . import shapes as S
from .engine import BOT, TOP, Box, Drive, Engine, Group, Igniter, clock
from .pcatalog import (ALU, BRASS, BRONZE, CERAMIC, DARK, OPEN_BOT, OPEN_TOP, STEEL, STEEL_DK,
                       _bolt_ring, _page)
from .sdf import Solid
from .solids import Body, Hole, const_pose, spin_pose


def _mean_T(world, m):
    from .gpu import xp
    n = int(m.sum())
    if not n:
        return 0.0, 0
    return C.T0 * float(xp.where(m, world.vx ** 2 + world.vy ** 2, 0.0).sum()) / n / C.V_TH ** 2, n


# ====================================================================== SCROLL (A/C)
def scroll_ac():
    """An automotive A/C scroll compressor, end-on. The fixed scroll is an involute wrap; the
    orbiting scroll is the same wrap turned 180 degrees, ORBITING on radius ro = pi*a - t
    without turning (an Oldham ring stops it rotating). For involutes of one base circle that
    orbit keeps the flanks touching along seal lines at every angle - checked numerically,
    -0.05 px - so crescent pockets close at the rim, travel inward, shrink, and reach the
    discharge port in the middle compressed and hot. The heating is a reflection off a wall
    that is moving inward, nothing else."""
    a, t = 22.0, 18.0
    ro = math.pi * a - t
    cx, cy = 540.0, 1130.0
    turns, start = 3.15, 0.5
    drive = Drive(1.1)
    eng = Engine(name="scroll_ac", rev_seconds=1.1, drive=drive, cycle_deg=360.0,
                 fuel_frac=0.0, groups=_page())
    wrap = S.involute_wrap(a, t, turns, 0.0, start=start)
    r_out = float(np.hypot(*wrap.T).max())
    ring_in = r_out + ro + 22.0
    eng.notes.update(part=True, a=a, t=t, ro=ro, centre=(cx, cy), ring_in=ring_in)
    half = ring_in + 12.0
    sb = eng.add_box(Box(cx - half, cx + half, cy - half, cy + half, 0, "duct", "", label="scroll"))
    px0, px1 = cx - 70.0, cx + 70.0
    ib = eng.add_box(Box(px0 - 40, px1 + 40, -140.0, cy - half, 0, "duct", "intake",
                         open=OPEN_TOP, label="suction"))
    eng.connect(sb, ib, TOP, px0, px1)
    # the discharge port has a REED (a one-way valve): gas leaves, nothing comes back. With a
    # two-way 2.4 bar reservoir there the line leaked back through every seal faster than the
    # pockets carried gas forward, and the gate measured it as a net flow OUT of the suction.
    eng.holes.append(Hole(sb, cx, cy, 34.0, boost=0.0, T=470.0, role="exhaust", inject=False,
                          label="discharge port (reed)"))
    # ---- fixed scroll: the wrap plus its shell, one casting, sectioned ----
    Fx = Solid((cx - half - 60, cy - half - 60, cx + half + 60, cy + half + 60), h=0.7,
               bevel=5.0, name="fixed scroll")
    Fx.add(S.rrect(cx - half - 50, cy - half - 50, cx + half + 50, cy + half + 50, 90.0))
    Fx.sub(S.circle(cx, cy, ring_in, 900))
    Fx.sub(S.rect(px0, cy - half - 80, px1, cy - ring_in + 30))
    Fx.add(S.shift(wrap, cx, cy))
    _bolt_ring(Fx, cx, cy, ring_in + 38.0, 12, r=9.0, a0=0.26)
    Fx.done(ALU, brush="cast", sectioned=True, spec=0.3)
    # the scroll's flanks close on each other at every seal line; with perfectly specular walls
    # a molecule caught in that closing wedge gains speed on every bounce and the rim ran at
    # 900 K. Real metal takes heat: 20 % of bounces re-emit at the wall's temperature.
    eng.bodies.append(Body(Fx, const_pose(), z=0, static=True, name="fixed scroll", wall_T=320.0))
    # ---- the suction pipe, down to the shell ----
    Pp = Solid((px0 - 60, -150.0, px1 + 60, cy - half + 8), h=0.8, bevel=5.0, name="suction pipe")
    Pp.add(S.rect(px0 - 50, -150.0, px1 + 50, cy - half + 8))
    Pp.sub(S.rect(px0, -160.0, px1, cy - half + 20))
    Pp.boss(S.rect(px0 - 50, cy - half - 90, px1 + 50, cy - half - 50), height=4.0, bevel=2.5)
    Pp.done(STEEL_DK, brush="linear", spec=0.45)
    eng.bodies.append(Body(Pp, const_pose(), z=1, static=True, name="suction pipe"))
    # ---- orbiting scroll: the wrap turned 180 degrees ----
    Ob = Solid((-r_out - 10, -r_out - 10, r_out + 10, r_out + 10), h=0.6, bevel=4.0,
               name="orbiting")
    Ob.add(S.rot(wrap, math.pi))
    Ob.done(STEEL, brush="linear", spec=0.7)
    # a GAP, never an overlap: a molecule caught inside two overlapping solids is reflected off
    # the moving one every substep with no way out, and heats without limit (measured: the
    # first Wankel reached 4000 K with ignition OFF). A 1 px gap leaks a trace - real slip.
    seal = 0.0          # exact contact: the walls take the heat now (wall_T), so no trap

    def orbit(t_):
        # the orbit runs AGAINST the wrap's winding (the wrap winds clockwise outward), so the
        # seal points travel INWARD along the spiral. The first build orbited the other way and
        # the gate caught it as an expander: discharge gas driven out through the suction.
        ps = -drive.theta(t_)
        dps = -drive.omega(t_)
        rr = ro + seal
        return (cx + rr * math.cos(ps), cy + rr * math.sin(ps), math.pi,
                -rr * dps * math.sin(ps), rr * dps * math.cos(ps), 0.0)
    eng.bodies.append(Body(Ob, orbit, z=10, name="orbiting scroll", wall_T=320.0))
    eng.probes = [("suction", ib, (px0 + 10, 120.0, px1 - 10, 420.0))]

    def pockets(world):
        r2 = (world.x - cx) ** 2 + (world.y - cy) ** 2
        Tc, _ = _mean_T(world, (world.box == sb) & (r2 < 150.0 ** 2))
        Tr, _ = _mean_T(world, (world.box == sb) & (r2 > (r_out - 60) ** 2))
        return {"T_centre": Tc, "T_rim": Tr}
    eng.notes["gate_probe"] = pockets
    eng.notes["T_range"] = (260.0, 760.0)
    eng.notes["warm"] = 3.0
    eng.blurb = "A/C scroll compressor: pockets close at the rim and are squeezed to the centre"
    return eng


# ====================================================================== VANE PUMP
def vane_pump():
    """A sliding-vane pump - here the brake-booster VACUUM pump a diesel carries, pulling the
    booster down to ~0.5 bar and pushing to atmosphere. A slotted rotor turns off-centre in a
    round cam ring; seven vanes slide out of their slots so their tips ride the ring (each tip
    is solved from the ring every substep, so it never lifts off). Chambers grow across the
    inlet port at the top and shrink across the outlet at the bottom."""
    Rr, Rc, nv = 220.0, 270.0, 7
    e = Rc - Rr - 2.0
    cx, cy = 505.0, 980.0                       # ROTOR centre
    rcx, rcy = cx + e, cy                       # ring centre, offset to the right
    vw, vl = 20.0, 2 * e + 52.0
    drive = Drive(2.4)
    eng = Engine(name="vane_pump", rev_seconds=2.4, drive=drive, cycle_deg=360.0,
                 fuel_frac=0.0, groups=_page())
    eng.notes.update(part=True, rotor=(cx, cy), ring=(rcx, rcy), Rr=Rr, Rc=Rc, e=e)
    half = Rc + 70.0
    pb = eng.add_box(Box(rcx - half, rcx + half, rcy - half, rcy + half, 0, "duct", "",
                         label="pump"))
    ia = (-128.0, -62.0)
    oa = (62.0, 128.0)
    ix0 = rcx + Rc * math.cos(math.radians(ia[0])) + 6
    ix1 = rcx + Rc * math.cos(math.radians(ia[1])) - 6
    ox0 = rcx + Rc * math.cos(math.radians(oa[1])) + 6
    ox1 = rcx + Rc * math.cos(math.radians(oa[0])) - 6
    ib = eng.add_box(Box(ix0 - 30, ix1 + 30, -140.0, rcy - half, 0, "duct", "intake",
                         open=OPEN_TOP, boost=0.5, label="from booster"))
    ob = eng.add_box(Box(ox0 - 30, ox1 + 30, rcy + half, 2060.0, 0, "duct", "exhaust",
                         open=OPEN_BOT, boost=1.0, label="to atmosphere"))
    eng.connect(pb, ib, TOP, ix0, ix1)
    eng.connect(pb, ob, BOT, ox0, ox1)
    # ---- the cam ring + housing: one sectioned casting ----
    H = Solid((rcx - half - 60, rcy - half - 60, rcx + half + 60, rcy + half + 60), h=0.8,
              bevel=6.0, name="housing")
    H.add(S.rrect(rcx - half - 50, rcy - half - 50, rcx + half + 50, rcy + half + 50, 110.0))
    H.sub(S.circle(rcx, rcy, Rc, 900))
    H.sub(np.concatenate([S.arc(rcx, rcy, Rc + 3, ia[0], ia[1]),
                          [(ix1, rcy - half - 80), (ix0, rcy - half - 80)]]))
    H.sub(np.concatenate([S.arc(rcx, rcy, Rc + 3, oa[0], oa[1]),
                          [(ox0, rcy + half + 80), (ox1, rcy + half + 80)]]))
    H.recess(S.circle(rcx, rcy, Rc + 18.0, 900), depth=2.0, bevel=2.0)
    _bolt_ring(H, rcx, rcy, Rc + 62.0, 8, r=9.0, a0=0.0)
    H.done(ALU, brush="cast", sectioned=True, spec=0.25)
    eng.bodies.append(Body(H, const_pose(), z=0, static=True, name="housing"))
    for (x0, x1, y0, y1, flip) in ((ix0, ix1, -150.0, rcy - half + 6, False),
                                   (ox0, ox1, rcy + half - 6, 2070.0, True)):
        Dd = Solid((x0 - 70, y0, x1 + 70, y1), h=0.8, bevel=5.0, name="duct")
        Dd.add(S.rect(x0 - 60, y0, x1 + 60, y1)).sub(S.rect(x0, y0 - 20, x1, y1 + 20))
        yb = y1 - 60 if not flip else y0 + 20
        Dd.boss(S.rect(x0 - 60, yb, x1 + 60, yb + 40), height=4.0, bevel=2.5)
        Dd.done(ALU, brush="cast", sectioned=True, spec=0.25)
        eng.bodies.append(Body(Dd, const_pose(), z=1, static=True, name="duct"))
    # ---- the rotor ----
    R_ = Solid((-Rr - 4, -Rr - 4, Rr + 4, Rr + 4), h=0.6, bevel=8.0, name="rotor")
    R_.add(S.circle(0, 0, Rr, 900))
    for i in range(nv):
        ang = 2 * math.pi * i / nv
        slot = S.rot(S.rect(Rr - vl - 6, -vw / 2 - 1.5, Rr + 10, vw / 2 + 1.5), ang)
        R_.recess(slot, depth=10.0, bevel=1.5)
        R_.paint(slot, DARK, feather=1.0)
    R_.recess(S.circle(0, 0, Rr * 0.52), depth=3.0, bevel=3.0)
    R_.boss(S.circle(0, 0, 54.0), height=6.0, bevel=3.0)
    R_.recess(S.rect(-22, -22, 22, 22), depth=5.0, bevel=1.5)           # the drive tang
    R_.paint(S.rect(-22, -22, 22, 22), STEEL_DK)
    R_.done(STEEL, brush="radial", spec=0.65)
    eng.bodies.append(Body(R_, spin_pose(cx, cy, drive, 1.0, 0.0), z=5, name="rotor"))
    # ---- the vanes: each tip solved from the ring, every substep ----
    # a GAP, never an overlap: a molecule caught inside two overlapping solids is reflected off
    # the moving one every substep with no way out, and heats without limit (measured: the
    # first Wankel reached 4000 K with ignition OFF). A 1 px gap leaks a trace - real slip.
    seal = -1.0

    def tip_r(phi):
        s_, c_ = math.sin(phi), math.cos(phi)
        root = math.sqrt(Rc * Rc - (e * s_) ** 2)
        return e * c_ + root, -e * s_ - (e * e) * s_ * c_ / root

    def vane_pose(i):
        def f(t_):
            phi = drive.theta(t_) + 2 * math.pi * i / nv
            w = drive.omega(t_)
            r, dr = tip_r(phi)
            r += seal
            c_, s_ = math.cos(phi), math.sin(phi)
            vx = dr * w * c_ - r * w * s_
            vy = dr * w * s_ + r * w * c_
            return (cx + r * c_, cy + r * s_, phi, vx, vy, w)
        return f
    V = Solid((-vl - 4, -vw / 2 - 4, 4, vw / 2 + 4), h=0.5, bevel=4.0, name="vane")
    V.add(np.concatenate([S.arc(-vw / 2, 0, vw / 2, -90, 90), [(-vl, vw / 2), (-vl, -vw / 2)]]))
    V.boss(S.rect(-vl + 8, -3, -14, 3), height=1.5, bevel=1.0)
    V.done((196, 176, 132), brush="linear", spec=0.6)                   # carbon-graphite
    for i in range(nv):
        eng.bodies.append(Body(V, vane_pose(i), z=8, name=f"vane {i}"))
    eng.probes = [("booster", ib, (ix0 + 20, 140.0, ix1 - 20, 400.0)),
                  ("outlet", ob, (ox0 + 20, rcy + half + 260, ox1 - 20, rcy + half + 520))]
    eng.notes["T_range"] = (230.0, 560.0)
    eng.notes["warm"] = 3.0
    eng.blurb = "sliding-vane vacuum pump: seven vanes riding an offset ring, booster to air"
    return eng


# ====================================================================== WANKEL
def wankel():
    """A Wankel rotary engine. The housing bore is the epitrochoid its own apexes trace; the
    rotor is that curve's inner envelope (computed, not drawn), with a combustion recess in
    each flank and apex seals that ride the bore. The rotor turns at a THIRD of the eccentric
    shaft, its centre orbiting the shaft on radius e - the 3:2 internal/external gear pair that
    forces that ratio is drawn on the rotor face. Three chambers, each through intake,
    compression, power and exhaust once per rotor turn: one firing every shaft revolution.

    Built in a local frame with the major axis along x and turned 90 degrees onto the page, so
    the long axis runs up the frame, the ports leave to the right and the plugs sit left."""
    e, R = 52.0, 374.0
    cx, cy = 540.0, 960.0
    drive = Drive(2.0)
    eng = Engine(name="wankel", rev_seconds=2.0, drive=drive, cycle_deg=360.0, fuel_frac=0.30,
                 q_rxn=5.5e7, groups=[Group("rot", 90.0, (540.0, 960.0), (540.0, 960.0))])
    eng.notes.update(part=True, e=e, R=R, centre=(cx, cy))
    ymin = cy - (R - e)                         # the port-side waist (local top)
    ymax = cy + (R - e)                         # the plug-side waist
    wb = eng.add_box(Box(cx - R - e - 8, cx + R + e + 8, ymin - 12, ymax + 12, 0, "duct", "",
                         label="rotor housing"))
    ip = (cx + 40.0, cx + 250.0)
    ep = (cx - 250.0, cx - 40.0)
    ib = eng.add_box(Box(ip[0] - 20, ip[1] + 20, -150.0, ymin - 12, 0, "duct", "intake",
                         open=OPEN_TOP, label="intake"))
    xb = eng.add_box(Box(ep[0] - 20, ep[1] + 20, -150.0, ymin - 12, 0, "duct", "exhaust",
                         open=OPEN_TOP, label="exhaust", res_T=700.0,       # a pipe of exhaust
                         res_mix={C.CO2: 1.0 / 3.0, C.H2O: 2.0 / 3.0}))
    eng.connect(wb, ib, TOP, ip[0], ip[1])
    eng.connect(wb, xb, TOP, ep[0], ep[1])
    troch = S.shift(S.epitrochoid(e, R, 1440), cx, cy)
    # ---- housing: sectioned, with coolant passages as relief and the port runners cut ----
    H = Solid((cx - R - e - 110, -160.0, cx + R + e + 110, ymax + 130), h=0.9, bevel=6.0,
              name="housing")
    H.add(S.rrect(cx - R - e - 100, ymin - 110, cx + R + e + 100, ymax + 110, 120.0))
    for (a0, a1) in (ip, ep):
        H.add(S.rect(a0 - 60, -160.0, a1 + 60, ymin - 100))
    H.sub(troch)
    for (a0, a1) in (ip, ep):
        H.sub(S.rect(a0, -170.0, a1, ymin + 20))
    for k in range(10):
        a = 2 * math.pi * (k + 0.5) / 10
        x = cx + (R + e + 50) * math.cos(a)
        y = cy + (R - e + 56) * math.sin(a)
        H.recess(S.circle(x, y, 22.0), depth=6.0, bevel=3.0)          # coolant passages
        H.paint(S.circle(x, y, 18.0), (40, 60, 80), feather=4.0)
    H.done(ALU, brush="cast", sectioned=True, spec=0.25)
    eng.bodies.append(Body(H, const_pose(), z=0, static=True, name="housing"))
    # ---- the rotor ----
    env = S.wankel_rotor(e, R, 1080)
    Rt = Solid((-R - 6, -R - 6, R + 6, R + 6), h=0.7, bevel=9.0, name="rotor")
    # FLANK CLEARANCE 10 px. The first build shrank the envelope by only 3 px, so each flank
    # swept past the waist as a closing 3 px wedge: too thin for collisions, molecules caught
    # in it ping-pong between the closing walls and gain speed every bounce (Fermi
    # acceleration). With ignition OFF it ran away past 1.7x speed to 4000 K, at 3000 or 6000
    # substeps alike. Real rotors clear the housing on the flanks and seal only at the apexes.
    Rt.add(env).grow(-10.0)
    for k in range(3):
        am = math.radians(60.0 + 120.0 * k)
        rm = float(np.hypot(*env[int(round((60.0 + 120.0 * k) / 360.0 * 1080)) % 1080]))
        # the combustion recess: sized so the chamber's area ratio is ~5:1 (measured 6.6:1 with
        # a 30 x 120 recess, which with gamma = 2 took the charge to ~2000 K before the burn)
        Rt.sub(S.rot(S.ellipse(rm, 0.0, 45.0, 150.0, 140), am))   # measured 4.8:1 with the real solids
    for k in range(3):
        aa = math.radians(120.0 * k)
        # the apex seal: a radial blade reaching R - 1.5, a 1.5 px gap to the bore (a gap,
        # never an overlap - see the scroll's seal note)
        blade = S.rot(np.array([(R - 44.0, -6.0), (R - 1.5 - 5.0, -6.0), (R - 1.5, -2.0),
                                (R - 1.5, 2.0), (R - 1.5 - 5.0, 6.0), (R - 44.0, 6.0)]), aa)
        Rt.add(blade)
        Rt.paint(blade, BRONZE, feather=1.0)
    Rt.recess(S.involute_gear(3.0 * e, 36), depth=7.0, bevel=2.0)     # the internal gear
    Rt.paint(S.involute_gear(3.0 * e, 36), (112, 118, 130), feather=2.0)
    for k in range(3):
        am = math.radians(60.0 + 120.0 * k)
        Rt.recess(S.rot(S.ellipse(215.0, 0.0, 34.0, 70.0, 64), am), depth=8.0, bevel=5.0)
    Rt.done(STEEL, brush="radial", spec=0.6)

    def rotor_pose(t_):
        ph = drive.theta(t_)
        w = drive.omega(t_)
        return (cx + e * math.cos(ph), cy + e * math.sin(ph), ph / 3.0,
                -e * w * math.sin(ph), e * w * math.cos(ph), w / 3.0)
    eng.bodies.append(Body(Rt, rotor_pose, z=10, name="rotor"))
    # ---- the eccentric shaft: the stationary gear the rotor rolls on, and the journal ----
    G = Solid((-2 * e - 20, -2 * e - 20, 2 * e + 20, 2 * e + 20), h=0.5, bevel=4.0, name="gear")
    G.add(S.involute_gear(2.0 * e, 24)).sub(S.circle(0, 0, 40.0))
    G.done(STEEL_DK, brush="radial", spec=0.5)
    eng.bodies.append(Body(G, const_pose(cx, cy), physics=False, z=11, name="stationary gear"))
    J = Solid((-e - 100, -e - 100, e + 100, e + 100), h=0.5, bevel=5.0, name="eccentric")
    J.add(S.circle(e, 0.0, 88.0))
    J.boss(S.circle(0.0, 0.0, 40.0), height=6.0, bevel=3.0)
    J.recess(S.rect(-8, -52, 8, -36), depth=3.0, bevel=1.0)
    J.paint(S.circle(0.0, 0.0, 40.0), STEEL)
    J.done(BRASS, brush="radial", spec=0.55)                          # the rotor bearing
    eng.bodies.append(Body(J, spin_pose(cx, cy, drive, 1.0, 0.0), physics=False, z=12,
                           name="eccentric"))
    # ---- plugs: leading and trailing, at the plug-side waist ----
    for k, dx in enumerate((-48.0, 48.0)):
        Pl = Solid((cx + dx - 26, ymax - 4, cx + dx + 26, ymax + 150), h=0.5, bevel=3.0,
                   name="plug")
        Pl.add(S.rect(cx + dx - 11, ymax + 2, cx + dx + 11, ymax + 50))
        Pl.add(S.rect(cx + dx - 20, ymax + 50, cx + dx + 20, ymax + 76))
        Pl.add(S.rect(cx + dx - 11, ymax + 76, cx + dx + 11, ymax + 146))
        Pl.paint(S.rect(cx + dx - 12, ymax + 76, cx + dx + 12, ymax + 150), CERAMIC)
        Pl.done((176, 182, 192), brush="linear", spec=0.5)
        eng.bodies.append(Body(Pl, const_pose(), physics=False, static=True, z=3, name="plug"))
    clk = clock(drive, 360.0, 0.0)

    def chamber_at_plugs(world):
        """The chamber bounded by the two apexes either side of the plug-side waist."""
        from .gpu import xp
        ph = drive.theta(world.t)
        rx, ry = cx + e * math.cos(ph), cy + e * math.sin(ph)
        al = ph / 3.0
        j = int(((math.pi / 2 - al) % (2 * math.pi)) // (2 * math.pi / 3))
        a0 = al + 2 * math.pi * j / 3
        ang = xp.arctan2(world.y - ry, world.x - rx)
        return ((ang - a0) % (2 * math.pi)) < (2 * math.pi / 3)
    sparks = (72.0, 78.0)
    for k, dx in enumerate((-48.0, 48.0)):
        eng.igniters.append(Igniter((wb,), (cx + dx, ymax - 6), clk, sparks[k], 0.0, False,
                                    9.0, rearm=0.5, label=("leading", "trailing")[k],
                                    region=chamber_at_plugs))

    def glows(t_):
        out = []
        for k, dx in enumerate((-48.0, 48.0)):
            d = (clk(t_) - sparks[k]) % 360.0
            amp = math.exp(-d / 4.0) if d < 20 else 0.0
            X, Y = eng.groups[0].to_world(cx + dx, ymax - 8)
            out.append((X, Y, 7.0, (0.85, 0.92, 1.0), 1.6 * amp))
        return out
    eng.notes["over"] = glows

    def chamber_T(world):
        T, n = _mean_T(world, (world.box == wb) & chamber_at_plugs(world))
        return {"T_plug_chamber": T, "N_plug_chamber": float(n)}
    eng.notes["gate_probe"] = chamber_T
    eng.notes.update(chamber_fn=chamber_at_plugs, clock=clk, intake_box=ib, exhaust_box=xb,
                     spark_deg=sparks[0])
    eng.notes["warm"] = 6.0
    eng.notes["heat_look"] = True
    eng.notes["T_range"] = (280.0, 1800.0)       # burnt gas at 800-1600 K reads ember, not brick
    eng.notes["zoom"] = 1.1
    eng.blurb = "Wankel rotary: epitrochoid bore, rotor at 1/3 shaft speed, two plugs"
    return eng
