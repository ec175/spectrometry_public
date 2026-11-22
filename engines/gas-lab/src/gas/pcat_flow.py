# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pcat_flow.py - the parts set's flow machines: turbocharger, shock absorber, fuel injector.

Conventions are pcatalog's (reference px, y down, every solid drawn from the grid the gas meets).
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from . import shapes as S
from .engine import BOT, LEFT, RIGHT, TOP, Box, Drive, Engine, smoothstep
from .pcatalog import (ALU, BRASS, COPPER, DARK, IRON, RUBBER, STEEL, STEEL_DK, _bolt_ring,
                       _page)
from .sdf import Solid
from .solids import Body, Hole, const_pose, spin_pose

OPEN_L = (False, False, True, False)
OPEN_R = (False, False, False, True)
OPEN_LR = (False, False, True, True)
HOT_IRON = (104, 88, 82)            # a turbine housing that has been hot


def _mean(world, m):
    from .gpu import xp
    n = int(m.sum())
    if not n:
        return 0.0, 0
    return C.T0 * float(xp.where(m, world.vx ** 2 + world.vy ** 2, 0.0).sum()) / n / C.V_TH ** 2, n


def _volute(cx, cy, r_wheel, r0, r1, lip, x_far, mirror=False, n=720):
    """The gas passages of a volute housing: the wheel chamber, a spiral whose outer wall
    grows from r0 at the tongue to r1 after one turn (clockwise from the top), and the
    tangential duct that leaves from the top towards x_far. mirror=True flips it about cx
    (a counter-clockwise volute, duct leaving the other way)."""
    ph = np.linspace(0.0, 2 * math.pi, n)
    ang = -math.pi / 2 + ph
    ro = r0 + (r1 - r0) * ph / (2 * math.pi)
    outer = np.stack([cx + ro * np.cos(ang), cy + ro * np.sin(ang)], 1)
    inner = np.stack([cx + (r_wheel - 3) * np.cos(ang[::-1]), cy + (r_wheel - 3) * np.sin(ang[::-1])], 1)
    band = np.concatenate([outer, inner])
    duct = S.rect(cx - 2.0, cy - r1, x_far, cy - r0 - lip)
    rings = [S.circle(cx, cy, r_wheel, 900), band, duct]
    if mirror:
        rings = [np.stack([2 * cx - P[:, 0], P[:, 1]], 1) for P in rings]
    return rings


def _blade_wheel(r_hub, r0, r1, n, thick, sweep_deg=0.0, splitters=False, h=0.5):
    """A radial wheel's BLADES as a solid, centred on the origin. Blade centreline angle
    grows by sweep * u^2 from root to tip (a backswept compressor blade); straight if 0."""
    s = Solid((-r1 - 6, -r1 - 6, r1 + 6, r1 + 6), h=h, bevel=4.0, name="wheel")

    def blade(a0, ra):
        rs = np.linspace(ra, r1, 40)
        u = (rs - r0) / (r1 - r0)
        a = a0 + math.radians(sweep_deg) * np.clip(u, 0, 1) ** 2
        return S.polyline_band(np.stack([rs * np.cos(a), rs * np.sin(a)], 1), thick / 2)
    for k in range(n):
        a0 = 2 * math.pi * k / n
        s.add(blade(a0, r0))
        if splitters:
            s.add(blade(a0 + math.pi / n, r0 + 0.45 * (r1 - r0)))
    s.add(S.circle(0, 0, r_hub))
    return s


# ====================================================================== 3. TURBOCHARGER
def turbocharger():
    """A turbocharger's two wheels, each seen end-on: the TURBINE (top) in its hot volute and
    the COMPRESSOR (bottom) in its cold one. They are the two ends of one shaft, so they turn
    together, the same way, at the same speed. Exhaust enters the turbine volute tangentially,
    swirls in through the blades and leaves along the axis at the wheel's eye (a circular
    reservoir: out of the page). Air arrives along the compressor's axis at its eye, is flung
    outward by backswept blades with splitters, and is collected by the volute into the charge
    pipe. As the engine speeds up (the drive's ramp) the exhaust feed pressure rises with it,
    the shaft spools, and the boost pressure the compressor holds against builds: all three
    follow the same schedule. Measured: the exhaust COOLS crossing the turbine (it does work on
    receding blades) and the air HEATS crossing the compressor."""
    drive = Drive(0.9)
    eng = Engine(name="turbocharger", rev_seconds=0.9, drive=drive, cycle_deg=360.0,
                 fuel_frac=0.0, groups=_page())
    Tc, Kc = (540.0, 575.0), (540.0, 1375.0)
    rw, r0, r1, lip = 192.0, 212.0, 330.0, 18.0
    eng.notes.update(part=True, turbine=Tc, compressor=Kc)
    s = drive.s
    p_exh = lambda t: 1.30 + 0.48 * (s(t) - 1.0)                 # noqa: E731  exhaust feed
    p_boost = lambda t: 1.00 + 0.16 * (s(t) - 1.0)               # noqa: E731  charge pipe
    burnt = {C.CO2: 1.0 / 3.0, C.H2O: 2.0 / 3.0}
    half = 395.0
    tb = eng.add_box(Box(Tc[0] - half, 1200.0, Tc[1] - half, Tc[1] + half, 0, "duct", "intake",
                         open=OPEN_R, boost=p_exh, label="turbine", res_T=950.0, res_mix=burnt))
    kb = eng.add_box(Box(-120.0, Kc[0] + half, Kc[1] - half, Kc[1] + half, 0, "duct", "exhaust",
                         open=OPEN_L, boost=p_boost, label="compressor", res_T=345.0))
    eng.holes.append(Hole(tb, Tc[0], Tc[1], 60.0, boost=1.05, T=760.0, mix=burnt,
                          role="exhaust", label="turbine exducer"))
    eng.holes.append(Hole(kb, Kc[0], Kc[1], 60.0, boost=1.0, T=300.0, role="intake",
                          label="compressor inducer"))
    # ---- housings ----
    for (c, mirror, col, name, far) in ((Tc, False, HOT_IRON, "turbine housing", 1250.0),
                                        (Kc, True, ALU, "compressor housing", 1250.0)):
        H = Solid((-140.0, c[1] - half - 20, 1220.0, c[1] + half + 20), h=0.8, bevel=6.0, name=name)
        body = S.rrect(c[0] - half + 10, c[1] - half + 10, c[0] + half - 10, c[1] + half - 10, 150.0)
        flange = S.rect(c[0], c[1] - r1 - 60, far, c[1] - r0 - lip + 44)
        if mirror:
            flange = np.stack([2 * c[0] - flange[:, 0], flange[:, 1]], 1)
        H.add(body).add(flange)
        for ring in _volute(c[0], c[1], rw, r0, r1, lip, far + 10, mirror):
            H.sub(ring)
        _bolt_ring(H, c[0], c[1], half - 45.0, 10, r=9.0, a0=0.9)
        H.done(col, brush="cast", sectioned=True, spec=0.25 if not mirror else 0.3)
        eng.bodies.append(Body(H, const_pose(), z=0, static=True, name=name))
        # the gas box is a rectangle and the casting is rounded: an INVISIBLE filler takes the
        # corners, or stagnant gas glows outside the housing (the first still showed it)
        box = eng.boxes[tb if not mirror else kb]
        Fl = Solid((box.x0 - 4, box.y0 - 4, box.x1 + 4, box.y1 + 4), h=1.5, bevel=1.0, name="filler")
        Fl.add(S.rect(box.x0 - 3, box.y0 - 3, box.x1 + 3, box.y1 + 3)).sub(body).sub(flange)
        Fl.done(DARK, brush="cast")
        eng.bodies.append(Body(Fl, const_pose(), draw=False, static=True, name="filler"))
    # ---- wheels: one shaft, so ONE rotation for both (counter-clockwise on the page) ----
    Tw = _blade_wheel(34.0, 68.0, 180.0, 11, 15.0, sweep_deg=0.0)
    Tw.boss(S.circle(0, 0, 34.0), height=5.0, bevel=2.0)
    Tw.recess(S.rot(S.circle(0, 0, 18.0, 6), 0.0), depth=4.0, bevel=1.0)
    Tw.done((170, 160, 150), brush="radial", spec=0.6)          # Inconel: warmer than steel
    Kw = _blade_wheel(34.0, 68.0, 182.0, 7, 12.0, sweep_deg=34.0, splitters=True)
    Kw.boss(S.circle(0, 0, 34.0), height=5.0, bevel=2.0)
    Kw.recess(S.circle(0, 0, 18.0, 6), depth=4.0, bevel=1.0)
    Kw.done((178, 184, 194), brush="radial", spec=0.75)         # machined aluminium
    eng.bodies.append(Body(Tw, spin_pose(Tc[0], Tc[1], drive, -1.0, 0.0), z=10, name="turbine wheel"))
    eng.bodies.append(Body(Kw, spin_pose(Kc[0], Kc[1], drive, -1.0, 0.3), z=10, name="compressor wheel"))
    # ---- what the gate measures ----
    eng.probes = [("exhaust in", tb, (880.0, Tc[1] - r1 + 8, 1060.0, Tc[1] - r0 - lip - 8)),
                  ("charge out", kb, (20.0, Kc[1] - r1 + 8, 200.0, Kc[1] - r0 - lip - 8))]

    def wheels(world):
        r2t = (world.x - Tc[0]) ** 2 + (world.y - Tc[1]) ** 2
        r2k = (world.x - Kc[0]) ** 2 + (world.y - Kc[1]) ** 2
        Tt, _ = _mean(world, (world.box == tb) & (r2t < 110.0 ** 2))
        Tk, _ = _mean(world, (world.box == kb) & (r2k > 150.0 ** 2) & (r2k < 196.0 ** 2))
        return {"T_turb_exit": Tt, "T_comp_tip": Tk}
    eng.notes["gate_probe"] = wheels
    eng.notes["T_range"] = (260.0, 1100.0)
    eng.notes["warm"] = 2.5
    eng.blurb = "turbocharger: exhaust spins the turbine, the compressor on the same shaft builds boost"
    return eng


# ====================================================================== 7. SHOCK ABSORBER
def shock_absorber():
    """A gas-charged monotube damper inside its coil spring (a coilover). The road drives the
    rod: two superposed bumps whose rate follows the drive's ramp. The piston on the rod has
    two passages, each closed by a SHIM STACK on one face - compression shims on top, rebound
    shims underneath - that lift on the pressure difference measured across the piston every
    few substeps (a pressure-driven valve, like the two-stroke's reed). Below, a floating piston
    keeps the gas charge separate and moves by the rod's displaced volume. Damping IS
    throttling: the work the road does pushing gas through the passages ends up as heat, so the
    damper's gas warms through the clip - measured, not set."""
    drive = Drive(2.0)
    eng = Engine(name="shock_absorber", rev_seconds=2.0, drive=drive, cycle_deg=360.0,
                 fuel_frac=0.0, groups=_page())
    xl, xr = 330.0, 750.0
    cx = 0.5 * (xl + xr)
    ytop, ybot = 420.0, 1600.0
    yp0, A = 820.0, 90.0
    ph_, rod_w, pist_h = 0.9, 84.0, 84.0
    yf0, fl_h = 1300.0, 50.0
    eng.notes.update(part=True)

    def road(t):
        th = drive.theta(t)
        w = drive.omega(t)
        y = yp0 + A * (0.62 * math.sin(th) + 0.38 * math.sin(2.37 * th + ph_))
        v = A * (0.62 * math.cos(th) + 0.38 * 2.37 * math.cos(2.37 * th + ph_)) * w
        return y, v
    bb = eng.add_box(Box(xl, xr, ytop, ybot, 0, "duct", "", boost=2.0, label="damper"))
    # ---- the tube, caps, mounts: one static casting (sectioned) ----
    Tu = Solid((xl - 60, ytop - 80, xr + 60, ybot + 230), h=0.8, bevel=5.0, name="tube")
    Tu.add(S.rect(xl - 32, ytop - 60, xr + 32, ybot + 60))
    Tu.sub(S.rect(xl, ytop, xr, ybot))
    Tu.sub(S.rect(cx - rod_w / 2, ytop - 70, cx + rod_w / 2, ytop + 2))       # the rod bore
    Tu.add(S.rrect(cx - 48, ybot + 40, cx + 48, ybot + 120, 10.0))
    Tu.add(S.circle(cx, ybot + 165, 62.0)).sub(S.circle(cx, ybot + 165, 30.0))
    Tu.recess(S.rect(cx - rod_w / 2 - 18, ytop - 60, cx + rod_w / 2 + 18, ytop - 30), depth=4.0)
    Tu.paint(S.rect(cx - rod_w / 2 - 14, ytop - 58, cx - rod_w / 2 - 2, ytop - 32), RUBBER)
    Tu.paint(S.rect(cx + rod_w / 2 + 2, ytop - 58, cx + rod_w / 2 + 14, ytop - 32), RUBBER)
    Tu.done(STEEL_DK, brush="linear", sectioned=True, spec=0.35)
    eng.bodies.append(Body(Tu, const_pose(), z=0, static=True, name="tube"))
    Bu = Solid((cx - 40, ybot + 130, cx + 40, ybot + 200), h=0.5, bevel=3.0, name="bushing")
    Bu.add(S.circle(cx, ybot + 165, 30.0)).sub(S.circle(cx, ybot + 165, 14.0))
    Bu.done(RUBBER, brush="cast", spec=0.15)
    eng.bodies.append(Body(Bu, const_pose(), physics=False, static=True, z=1, name="bushing"))
    # lower spring seat, welded to the tube
    Ls = Solid((xl - 260, 1210, xr + 260, 1262), h=0.8, bevel=4.0, name="lower seat")
    Ls.add(S.rrect(xl - 250, 1222, xr + 250, 1252, 6.0)).sub(S.rect(xl - 33, 1200, xr + 33, 1270))
    Ls.done(STEEL_DK, brush="linear", spec=0.4)
    eng.bodies.append(Body(Ls, const_pose(), physics=False, static=True, z=2, name="lower seat"))
    # ---- piston + rod + upper seat + top eye: one moving part ----
    hole_x = ((-122.0, -102.0), (102.0, 122.0))
    P = Solid((-(xr - xl) / 2 - 4, -pist_h / 2 - 4, (xr - xl) / 2 + 4, pist_h / 2 + 4), h=0.5,
              bevel=4.0, name="piston")
    P.add(S.rect(-(xr - xl) / 2 - 2, -pist_h / 2, (xr - xl) / 2 + 2, pist_h / 2))
    for (a, b) in hole_x:
        P.sub(S.rect(a, -pist_h / 2 - 5, b, pist_h / 2 + 5))
    P.recess(S.rect(-(xr - xl) / 2 - 2, -8, (xr - xl) / 2 + 2, 8), depth=3.0, bevel=1.0)   # the band
    P.paint(S.rect(-(xr - xl) / 2 - 2, -7, (xr - xl) / 2 + 2, 7), (70, 60, 52))
    P.done(STEEL, brush="linear", spec=0.6)

    def piston_pose(t):
        y, v = road(t)
        return (cx, y, 0.0, 0.0, v, 0.0)
    eng.bodies.append(Body(P, piston_pose, z=10, name="piston"))
    rod_len = 600.0
    Rd = Solid((-rod_w / 2 - 70, -rod_len - 70, rod_w / 2 + 70, 2.0), h=0.5, bevel=5.0, name="rod")
    Rd.add(S.rect(-rod_w / 2, -rod_len, rod_w / 2, 2.0))
    Rd.add(S.circle(0.0, -rod_len - 4, 60.0)).sub(S.circle(0.0, -rod_len - 4, 30.0))
    Rd.boss(S.rect(-rod_w / 2 + 10, -rod_len + 60, rod_w / 2 - 10, -10), height=2.0, bevel=6.0)
    Rd.done((196, 200, 208), brush="linear", spec=0.9)            # hard chrome

    def rod_pose(t):
        y, v = road(t)
        return (cx, y - pist_h / 2, 0.0, 0.0, v, 0.0)
    eng.bodies.append(Body(Rd, rod_pose, z=9, name="rod"))
    Ub = Solid((-60, -rod_len - 40, 60, -rod_len + 30), h=0.5, bevel=3.0, name="eye bush")
    Ub.add(S.circle(0.0, -rod_len - 4, 30.0)).sub(S.circle(0.0, -rod_len - 4, 14.0))
    Ub.done(RUBBER, brush="cast", spec=0.15)
    eng.bodies.append(Body(Ub, rod_pose, physics=False, z=11, name="eye bushing"))
    us_y = 525.0                                   # upper seat: this far above the piston crown
    Us = Solid((-(xr - xl) / 2 - 260, -us_y - 30, (xr - xl) / 2 + 260, -us_y + 30), h=0.8,
               bevel=4.0, name="upper seat")
    Us.add(S.rrect(-(xr - xl) / 2 - 250, -us_y - 16, (xr - xl) / 2 + 250, -us_y + 14, 6.0))
    Us.sub(S.rect(-rod_w / 2 - 1, -us_y - 40, rod_w / 2 + 1, -us_y + 40))
    Us.done(STEEL_DK, brush="linear", spec=0.4)
    eng.bodies.append(Body(Us, rod_pose, physics=False, z=12, name="upper seat"))
    # ---- the shims: pressure-driven, measured across the piston every few substeps ----
    st = dict(comp=0.0, reb=0.0, dp=0.0, k=0)
    shim_w, shim_h, lift_max = 64.0, 8.0, 15.0
    Sh = Solid((-shim_w / 2 - 4, -shim_h / 2 - 4, shim_w / 2 + 4, shim_h / 2 + 4), h=0.5,
               bevel=2.0, name="shim")
    Sh.add(S.rect(-shim_w / 2, -shim_h / 2, shim_w / 2, shim_h / 2))
    Sh.done((210, 214, 220), brush="linear", spec=0.8)

    def shim_pose(side):
        def f(t):
            y, v = road(t)
            if side == 0:       # compression shim, on top of the left passage, lifts UP
                return (cx + 0.5 * sum(hole_x[0]), y - pist_h / 2 - shim_h / 2 + 1.5
                        - lift_max * st["comp"], 0.0, 0.0, v, 0.0)
            return (cx + 0.5 * sum(hole_x[1]), y + pist_h / 2 + shim_h / 2 - 1.5
                    + lift_max * st["reb"], 0.0, 0.0, v, 0.0)
        return f
    eng.bodies.append(Body(Sh, shim_pose(0), z=11, name="compression shim"))
    eng.bodies.append(Body(Sh, shim_pose(1), z=11, name="rebound shim"))

    def pre_step(world):
        st["k"] += 1
        if st["k"] % 6:
            return
        from .gpu import xp
        y, _ = road(world.t)
        slab = 60.0
        up = (world.y > y - pist_h / 2 - slab) & (world.y < y - pist_h / 2)
        dn = (world.y > y + pist_h / 2) & (world.y < y + pist_h / 2 + slab)
        e2 = world.vx ** 2 + world.vy ** 2
        # the rod takes part of the slab above the piston; the area must be the GAS area or
        # the shims read a false 0.4 bar and open at rest
        a_up = (xr - xl - rod_w) * slab
        a_dn = (xr - xl) * slab
        pu = float(xp.where(up, e2, 0.0).sum()) / C.V_TH ** 2 / a_up / C.N0
        pd = float(xp.where(dn, e2, 0.0).sum()) / C.V_TH ** 2 / a_dn / C.N0
        dp = pd - pu
        st["dp"] = dp
        # a shim stack is a stiff spring: a crack pressure, then lift ~ excess; first-order lag
        tc, cr, full = 0.9, 0.06, 0.40
        st["comp"] += (smoothstep((dp - cr) / full) - st["comp"]) * tc
        st["reb"] += (smoothstep((-dp - cr) / full) - st["reb"]) * tc
    eng.notes["pre_step"] = pre_step
    # ---- the floating piston: moved by the rod's displaced volume ----
    Fp = Solid((-(xr - xl) / 2 - 4, -fl_h / 2 - 4, (xr - xl) / 2 + 4, fl_h / 2 + 4), h=0.5,
               bevel=4.0, name="floating piston")
    Fp.add(S.rect(-(xr - xl) / 2 - 2, -fl_h / 2, (xr - xl) / 2 + 2, fl_h / 2))
    Fp.recess(S.rect(-(xr - xl) / 2 - 2, -6, (xr - xl) / 2 + 2, 6), depth=3.0, bevel=1.0)
    Fp.paint(S.rect(-(xr - xl) / 2 - 2, -5, (xr - xl) / 2 + 2, 5), RUBBER)
    Fp.done(STEEL_DK, brush="radial", spec=0.5)

    def float_pose(t):
        y, v = road(t)
        k = rod_w / (xr - xl)
        return (cx, yf0 + k * (y - yp0), 0.0, 0.0, k * v, 0.0)
    eng.bodies.append(Body(Fp, float_pose, z=10, name="floating piston"))
    # ---- the coil spring: wire sections either side, between the two seats ----
    n_c, R_sp, wire = 8, 330.0, 24.0
    Wr = Solid((-wire - 3, -wire - 3, wire + 3, wire + 3), h=0.4, bevel=10.0, name="wire")
    Wr.add(S.circle(0, 0, wire))
    Wr.done((150, 60, 52), brush="radial", brush_c=(0, 0), spec=0.55)    # painted spring steel

    def coil(k, side):
        def f(t):
            y, v = road(t)
            y_up = y - pist_h / 2 - us_y + 16 + wire
            y_lo = 1222.0 - wire
            u = (k + (0.5 if side else 0.0) + 0.25) / (n_c + 0.5)
            return (cx + (R_sp if side else -R_sp), y_up + (y_lo - y_up) * u, 0.0, 0.0,
                    v * (1 - u), 0.0)
        return f
    for k in range(n_c):
        for side in (0, 1):
            eng.bodies.append(Body(Wr, coil(k, side), physics=False, z=3 if side else 13,
                                   name=f"coil {k}{'rl'[side]}"))
    # the back half of each turn, from each wire section in to the tube, BEHIND the tube: the
    # front half is cut away, as in a section drawing. A stub's slope follows the compression.
    stub_len = R_sp - (xr - xl) / 2 - 28 + 40
    Sb = Solid((-stub_len / 2 - 4, -wire - 4, stub_len / 2 + 4, wire + 4), h=0.5, bevel=12.0,
               name="coil back")
    Sb.add(S.rrect(-stub_len / 2, -wire * 0.92, stub_len / 2, wire * 0.92, wire * 0.9))
    Sb.done((112, 46, 40), brush="linear", spec=0.35)

    def back(k, side):
        def f(t):
            xa, ya, _, _, va, _ = coil(k, 0)(t)
            xb, yb, _, _, vb, _ = coil(k, 1)(t)
            ang = math.atan2(yb - ya, xb - xa)
            u = (stub_len / 2 + 6) / (xb - xa)
            if side == 0:
                x, y, v = xa + (xb - xa) * u, ya + (yb - ya) * u, va + (vb - va) * u
            else:
                x, y, v = xb - (xb - xa) * u, yb - (yb - ya) * u, vb - (vb - va) * u
            return (x, y, ang, 0.0, v, 0.0)
        return f
    for k in range(n_c):
        for side in (0, 1):
            eng.bodies.append(Body(Sb, back(k, side), physics=False, z=-1, name="coil back"))
    # ---- what the gate measures ----

    def chambers(world):
        y, _ = road(world.t)
        Tu_, _ = _mean(world, world.y < y - pist_h / 2)
        Tl_, _ = _mean(world, (world.y > y + pist_h / 2) & (world.y < yf0 - 40))
        Tg_, _ = _mean(world, world.y > yf0 + 40)
        return {"T_rebound": Tu_, "T_compress": Tl_, "T_gas": Tg_, "dP_bar": st["dp"],
                "shim_c": st["comp"], "shim_r": st["reb"]}
    eng.notes["gate_probe"] = chambers
    eng.notes["T_range"] = (270.0, 470.0)
    eng.notes["warm"] = 2.0
    eng.blurb = "gas-charged coilover damper: shim-stack valves on the piston, floating piston below"
    return eng


# ====================================================================== 8. FUEL INJECTOR
def fuel_injector():
    """A port fuel injector in section, hanging from its fuel rail over the intake runner.
    Fuel at rail pressure fills the bore; when the solenoid is energised the ARMATURE is pulled
    up against its return spring and lifts the NEEDLE off its conical seat, and fuel sprays
    through the orifice disc into air flowing along the runner toward the intake valve. One
    pulse per engine cycle, a fixed crank window, so the pulses come faster as the drive ramps
    - more fuel per second, as an engine at higher revs needs. The needle's seat is a real
    seal in the physics (1.5 px of overlap when shut): nothing leaks between pulses except the
    sac's own contents.

    Honesty: fuel is a liquid; here it is a dense gas at rail pressure, so the spray expands
    as a gas jet rather than breaking into droplets."""
    drive = Drive(0.7)
    eng = Engine(name="fuel_injector", rev_seconds=0.7, drive=drive, cycle_deg=720.0,
                 fuel_frac=0.0, groups=_page())
    cx = 540.0
    eng.notes.update(part=True)
    fuel = {C.FUEL: 1.0}
    # ---- boxes: the rail across the top, the injector's bore, the runner across the bottom --
    y_rail0, y_rail1 = 90.0, 190.0
    y_run0, y_run1 = 1300.0, 1740.0
    rail = eng.add_box(Box(-120.0, 1200.0, y_rail0, y_rail1, 0, "duct", "intake", open=OPEN_LR,
                           boost=6.0, label="fuel rail", res_mix=fuel))
    ij = eng.add_box(Box(cx - 64, cx + 64, y_rail1, y_run0, 0, "duct", "intake", boost=5.0,
                         label="injector", res_mix=fuel))
    run = eng.add_box(Box(-120.0, 1200.0, y_run0, y_run1, 0, "duct", "", label="runner"))
    eng.connect(rail, ij, BOT, cx - 36, cx + 36)
    eng.connect(ij, run, BOT, cx - 64, cx + 64)
    # the runner's two ends: air in on the left, out to the valve on the right
    run_b = eng.boxes[run]
    run_b.open = OPEN_LR
    run_b.boost = 1.1
    # two reservoirs on one box: left at 1.0, right at 0.8 - a pressure DIFFERENCE is what
    # makes a runner flow. world.py injects each open edge at the box's boost, so the right
    # edge's lower pressure is modelled by a sink hole just inside it instead:
    run_b.open = OPEN_LR
    run_b.side_boost = {RIGHT: 0.70}
    # ---- the injector body: silhouette minus its passages (one static solid) ----
    seat_top, seat_bot = 1160.0, 1190.0
    Bd = Solid((cx - 170, y_rail1 - 30, cx + 240, y_run0 + 8), h=0.5, bevel=5.0, name="injector")
    Bd.add(S.rrect(cx - 70, y_rail1 - 20, cx + 70, 420.0, 6.0))           # inlet tube
    Bd.add(S.rrect(cx - 135, 410.0, cx + 135, 910.0, 22.0))               # solenoid housing
    Bd.add(S.rrect(cx - 95, 900.0, cx + 95, 1150.0, 14.0))                # valve body
    Bd.add(S.rrect(cx - 70, 1140.0, cx + 70, y_run0 + 8, 8.0))            # nozzle
    Bd.add(S.rrect(cx + 120, 520.0, cx + 230, 700.0, 16.0))               # connector shroud
    # passages
    bore = np.array([(cx - 36, y_rail1 - 40), (cx + 36, y_rail1 - 40), (cx + 36, 560.0),
                     (cx + 62, 568.0), (cx + 62, 770.0), (cx + 32, 780.0), (cx + 32, seat_top),
                     (cx + 12, seat_bot), (cx + 12, 1232.0), (cx - 12, 1232.0), (cx - 12, seat_bot),
                     (cx - 32, seat_top), (cx - 32, 780.0), (cx - 62, 770.0), (cx - 62, 568.0),
                     (cx - 36, 560.0)])
    Bd.sub(bore)
    for hx in (-9.0, 9.0):                         # the orifice disc's two holes
        Bd.sub(np.array([(cx + hx - 8, 1226.0), (cx + hx + 8, 1226.0),
                         (cx + hx * 2.6 + 8, y_run0 + 20), (cx + hx * 2.6 - 8, y_run0 + 20)]))
    # relief: coil windings (copper, in section), o-ring grooves, the filter basket, terminal
    for sgn in (-1, 1):
        x0, x1 = cx + sgn * 72, cx + sgn * 124
        Bd.recess(S.rect(min(x0, x1), 450.0, max(x0, x1), 870.0), depth=5.0, bevel=1.5)
        Bd.paint(S.rect(min(x0, x1), 450.0, max(x0, x1), 870.0), COPPER, feather=1.0)
        for j in range(18):
            for i in range(3):
                wx = min(x0, x1) + 9 + i * 17
                wy = 462.0 + j * 23.0
                Bd.boss(S.circle(wx, wy, 7.5), height=2.0, bevel=3.0)
    for (y0_, w_) in ((300.0, 62.0), (1200.0, 70.0)):
        for sgn in (-1, 1):
            Bd.paint(S.rect(cx + sgn * w_ - 9 if sgn > 0 else cx - w_, y0_,
                            cx + w_ if sgn > 0 else cx - w_ + 9, y0_ + 26), RUBBER)
    Bd.paint(S.rrect(cx + 150, 560.0, cx + 214, 660.0, 6.0), (54, 58, 64))
    Bd.boss(S.rect(cx + 170, 590.0, cx + 196, 630.0), height=3.0, bevel=1.0)
    Bd.paint(S.rect(cx + 170, 590.0, cx + 196, 630.0), BRASS)
    Bd.done(STEEL_DK, brush="linear", sectioned=True, spec=0.35)
    eng.bodies.append(Body(Bd, const_pose(), z=0, static=True, name="injector body"))
    # ---- rail and runner castings ----
    Rl = Solid((-140.0, y_rail0 - 60, 1220.0, y_rail1 + 30), h=0.9, bevel=6.0, name="rail")
    Rl.add(S.rect(-140.0, y_rail0 - 70, 1220.0, y_rail1 + 40)).sub(S.rect(-150.0, y_rail0, 1230.0, y_rail1))
    for bx_ in (140.0, 940.0):
        Rl.boss(S.rect(bx_ - 60, y_rail1 + 4, bx_ + 60, y_rail1 + 40), height=4.0, bevel=2.0)
        Rl.boss(S.circle(bx_, y_rail1 + 22, 11.0), height=3.0, bevel=1.5)
    Rl.sub(S.rect(cx - 36, y_rail1 - 5, cx + 36, y_rail1 + 40))
    Rl.done(ALU, brush="cast", sectioned=True, spec=0.3)
    eng.bodies.append(Body(Rl, const_pose(), z=1, static=True, name="rail"))
    Rn = Solid((-140.0, y_run0 - 60, 1220.0, y_run1 + 70), h=0.9, bevel=6.0, name="runner")
    Rn.add(S.rect(-140.0, y_run0 - 70, 1220.0, y_run1 + 110)).sub(S.rect(-150.0, y_run0, 1230.0, y_run1))
    for bx_ in (140.0, 940.0):
        Rn.boss(S.rect(bx_ - 70, y_run0 - 66, bx_ + 70, y_run0 - 30), height=4.0, bevel=2.0)
        Rn.boss(S.circle(bx_, y_run0 - 48, 11.0), height=3.0, bevel=1.5)
    Rn.sub(S.rect(cx - 72, y_run0 - 60, cx + 72, y_run0 + 5))
    Rn.done(ALU, brush="cast", sectioned=True, spec=0.3)
    eng.bodies.append(Body(Rn, const_pose(), z=1, static=True, name="runner"))
    # ---- the armature + needle: one moving part ----
    L_lift = 20.0
    win0, win = 240.0, 260.0

    def lift(t):
        d = drive.deg(t) % 720.0
        u = (d - win0) % 720.0
        rate = drive.omega(t) * 180.0 / math.pi          # crank deg per s
        ramp = 12.0
        if u < ramp:
            x = u / ramp
            return x * x * (3 - 2 * x), 6 * x * (1 - x) / ramp * rate
        if u < win - ramp:
            return 1.0, 0.0
        if u < win:
            x = (win - u) / ramp
            return x * x * (3 - 2 * x), -6 * x * (1 - x) / ramp * rate
        return 0.0, 0.0
    Nd = Solid((-60, 570.0, 60, 1205.0), h=0.4, bevel=4.0, name="needle")
    Nd.add(S.rrect(-50, 580.0, 50, 752.0, 6.0))                      # armature
    Nd.add(S.rect(-20, 740.0, 20, 1170.0))                            # needle
    # the tip cone, matched to the seat cone and 1.5 px proud of it: a seal when shut
    k_ = (32.0 - 12.0) / (seat_bot - seat_top)

    def seat_w(y):
        return 32.0 - (y - seat_top) * k_
    y_a = seat_top + (32.0 - 21.5) / k_
    y_b = seat_top + (32.0 - 7.5) / k_
    Nd.add(np.array([(-20, 1165.0), (20, 1165.0), (21.5, y_a), (7.5, y_b), (-7.5, y_b), (-21.5, y_a)]))
    for sgn in (-1, 1):
        Nd.recess(S.rect(sgn * 30 - 6, 596.0, sgn * 30 + 6, 736.0), depth=4.0, bevel=1.0)  # fuel flats
    Nd.recess(S.rect(-20, 580.0, 20, 600.0), depth=3.0, bevel=1.0)
    Nd.done(STEEL, brush="linear", spec=0.75)

    def needle_pose(t):
        l, dl = lift(t)
        return (cx, -L_lift * l, 0.0, 0.0, -L_lift * dl, 0.0)
    eng.bodies.append(Body(Nd, needle_pose, z=10, name="armature + needle"))
    # the return spring: wire sections in the bore above the armature
    Sw = Solid((-8, -8, 8, 8), h=0.3, bevel=5.0, name="spring wire")
    Sw.add(S.circle(0, 0, 6.0))
    Sw.done((200, 204, 212), brush="radial", spec=0.7)

    def sp(k, side):
        def f(t):
            l, dl = lift(t)
            top, bot = 470.0, 580.0 - L_lift * l
            u = (k + 0.5 * side + 0.25) / 6.5
            return (cx + (22.0 if side else -22.0), top + (bot - top) * u, 0.0, 0.0,
                    -L_lift * dl * u, 0.0)
        return f
    for k in range(6):
        for side in (0, 1):
            eng.bodies.append(Body(Sw, sp(k, side), physics=False, z=12, name="return spring"))
    eng.notes["seat"] = (seat_top, seat_bot, seat_w(seat_top))
    eng.probes = [("rail", rail, (20.0, y_rail0 + 20, 300.0, y_rail1 - 20)),
                  ("runner in", run, (40.0, y_run0 + 40, 300.0, y_run1 - 40)),
                  ("runner out", run, (760.0, y_run0 + 40, 980.0, y_run1 - 40))]

    def spray(world):
        from .gpu import xp
        m = (world.box == run) & (world.sp == C.FUEL)
        return {"fuel_in_runner": float(int(m.sum())), "lift": lift(world.t)[0]}
    eng.notes["gate_probe"] = spray
    eng.notes["T_range"] = (240.0, 460.0)
    eng.notes["species_weight"] = {C.FUEL: 9.0}
    eng.notes["warm"] = 2.0
    eng.notes["zoom"] = 1.1
    eng.notes["view"] = (540.0, 930.0)
    eng.blurb = "port fuel injector: solenoid lifts the needle, fuel sprays into the runner air"
    return eng
