# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""world.py - THE PHYSICS. Kinetic-theory gas: discrete particles, not a field.

Wind_Tunnel already owns the field approach (a fluid on an LBM lattice). This engine is the
other half of the library's fluid story: individual molecules, where the macroscopic
quantities Clancy asked for are MEASURED, never set.

    temperature  is the mean kinetic energy of the particles in a region
    pressure     is the momentum they deliver to the walls, per unit wall, per second
    compression  heating is a moving-wall reflection and nothing else
    combustion   is a species conversion inside an expanding flame front

Nothing on that list is a keyframe or a tuning curve. That is the point: --dry can therefore
check PV = NkT and the reaction ledger, which is a check on the PHYSICS rather than a check
that the process exited zero.

2026-09-20: the world is generic over an `engine.Engine`. Any box may have a piston for its
floor AND/OR its ceiling, ports on all four sides, open reservoir edges on any side, and a
piston-COVERED port (a wall port the skirt uncovers). Cylinders sit in rotated groups that
share one particle array. The physics is the same as the 2026-09-15 engine; what changed is
what it can be pointed at. One real defect came out in the rewrite: the old floor pass added
the piston's velocity to EVERY floor hit, including the plenum floors - static walls now carry
u = 0, as they must.

FIVE THINGS THAT WILL BITE THE NEXT PERSON
------------------------------------------
1. The moving wall IS the thermodynamics. A particle bouncing off a piston closing at speed u
   leaves with |v_n| + 2u. If you ever replace it with a plain |v| flip to "fix" an energy
   drift, you delete the Otto cycle.
2. Collisions are DSMC (Bird), not pairwise detection. Binned by (box, cell), paired at random
   within a cell, accepted with probability ~ |v_rel| x local density. No radius, no O(N^2).
3. Species mass is 1 for everything, so T = <v^2> is exact and the reaction ledger is clean.
4. The gas is SCALED and TWO-DIMENSIONAL: v_thermal/v_piston ~ 12 (a real engine ~100), and
   gamma = 2, so a 5:1 area ratio does what 10:1 does in a real engine. --dry prints both.
5. No `.any()` / `.sum()` / boolean-mask indexing in the substep unless the value is needed on
   the CPU: each one is a pipeline sync, and at 3000 substeps/s they add up to minutes. The
   wall pass is written with unconditional `where` and `bincount(weights=where(...))` for
   exactly that reason.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from . import engine as E
from .gpu import Gen, xp

TWO_PI = 2.0 * math.pi
SQRT_2PI = math.sqrt(TWO_PI)


def temperature_K(v2):
    return C.T0 * v2 / (C.V_TH ** 2)


def sigma_at(T_K: float) -> float:
    return (C.V_TH / math.sqrt(2.0)) * math.sqrt(T_K / C.T0)


def stroke_name(deg: float, cycle: float = 720.0) -> str:
    d = deg % cycle
    if cycle >= 720.0:
        return ("intake" if d < 180 else "compression" if d < 360
                else "power" if d < 540 else "exhaust")
    return "power" if d < 180 else "compression"


class World:
    """Particles, the box graph, the pistons, and the thermodynamic tallies."""

    def __init__(self, eng: E.Engine, seed=7, cap=220_000, coll_every=8, ignite=True):
        self.eng = eng
        self.rng = Gen(seed)
        self.mfp = float(eng.mfp)
        # a reaction's energy is in px^2/s^2; the catalogue quotes it at Q_REF_VTH so that the
        # temperature rise it buys is the same at any V_TH
        self.q_rxn = float(eng.q_rxn) * (C.V_TH / C.Q_REF_VTH) ** 2
        self.ignite = bool(ignite)
        self.coll_every = int(coll_every)
        self.cap = int(cap)
        self.t = 0.0
        self.substep = 0
        self.nbox = len(eng.boxes)
        self._pack()

        # ---- particle arrays ----
        z = xp.zeros(0, dtype=xp.float32)
        self.x, self.y, self.vx, self.vy = z, z.copy(), z.copy(), z.copy()
        self.sp = xp.zeros(0, dtype=xp.int32)
        self.box = xp.zeros(0, dtype=xp.int32)
        self.tag = xp.zeros(0, dtype=xp.int32)   # a carried per-particle id (draw subset)
        self.px, self.py = self.x, self.y

        # ---- tallies ----
        self.imp = xp.zeros(self.nbox, dtype=xp.float32)   # wall impulse per box
        self.impulse_window = 0.0
        self.n_in = 0            # injected at intake reservoirs
        self.n_in_ex = 0         # injected at exhaust reservoirs (back-flow)
        self.n_out_in = 0        # lost back out of intake reservoirs
        self.n_out_ex = 0        # left through exhaust reservoirs
        self.n_injected = 0      # from injectors
        self.n_created = 0       # molecules a reaction ADDED (gasoline: 27 -> 34 per unit)
        self.flow_in = {"intake": np.zeros(5, np.int64), "exhaust": np.zeros(5, np.int64)}
        self.flow_out = {"intake": np.zeros(5, np.int64), "exhaust": np.zeros(5, np.int64)}
        self.n_clamped = 0
        self.reactions = 0
        self.made_co2 = 0
        self.made_h2o = 0
        self.events = []
        self._inject_acc = [0.0] * len(self.open_edges)
        self._inj_acc = [0.0] * len(eng.injectors)
        self._inj_done = [False] * len(eng.injectors)
        self.ig = [dict(spark_t=None, spark_deg=0.0, flame_r=0.0) for _ in eng.igniters]
        self.reeds = [(k, v) for k, v in enumerate(eng.valves) if v.mode == "reed"]
        self._last_stroke = None
        self._ps = self._piston_state(0.0)
        self._lifts = [v.lift(0.0) for v in eng.valves]
        self.D = xp.asarray(self.D_np)
        # 2026-09-25: shaped solids and circular port reservoirs (solids.py)
        from . import solids as S
        self.solids = S.SolidSet(eng.bodies, self.group_of) if eng.bodies else None
        if self.solids is not None and self.solids.nb == 0:
            self.solids = None
        if self.solids is not None:
            self.solids.set_poses(0.0)
        self.holes = list(eng.holes)
        self._hole_acc = [0.0] * len(self.holes)
        self.n_hole_in = [0] * len(self.holes)
        self.n_hole_out = [0] * len(self.holes)
        self.seed_domain()

    # ------------------------------------------------------------------ setup
    def _pack(self):
        eng, nb = self.eng, self.nbox
        bx = eng.boxes
        f32 = xp.float32
        self.bx0 = xp.asarray([b.x0 for b in bx], dtype=f32)
        self.bx1 = xp.asarray([b.x1 for b in bx], dtype=f32)
        self.y0s = np.array([b.y0 for b in bx], dtype=np.float32)
        self.y1s = np.array([b.y1 for b in bx], dtype=np.float32)
        self.open = xp.asarray([[1.0 if b.open[s] else 0.0 for b in bx] for s in range(4)],
                               dtype=f32)                                # (4, nb)
        self.is_cyl = np.array([b.kind == "cyl" for b in bx])
        self.group_of = xp.asarray([b.group for b in bx], dtype=xp.int32)
        gs = eng.groups
        self.g_cos = xp.asarray([g.cs[0] for g in gs], dtype=f32)
        self.g_sin = xp.asarray([g.cs[1] for g in gs], dtype=f32)
        self.g_pvx = xp.asarray([g.pivot[0] for g in gs], dtype=f32)
        self.g_pvy = xp.asarray([g.pivot[1] for g in gs], dtype=f32)
        self.g_atx = xp.asarray([g.at[0] for g in gs], dtype=f32)
        self.g_aty = xp.asarray([g.at[1] for g in gs], dtype=f32)
        self.ex_boxes = xp.asarray([i for i, b in enumerate(bx)
                                    if b.role == "exhaust" and any(b.open)] or [-1],
                                   dtype=xp.int32)

        # ports: static dest, and the base interval; dynamic (gate/cover) rows are rewritten
        # every substep into the packed D array.
        dest = np.full((4, nb, 2), -1, dtype=np.int32)
        self.pa0 = np.full((4, nb, 2), -1e9, dtype=np.float32)
        self.pb0 = np.full((4, nb, 2), -1e9, dtype=np.float32)
        self.dyn_ports = []          # (side, box, k, gate, cover)
        for i, b in enumerate(bx):
            slot = [0, 0, 0, 0]
            for p in b.ports:
                k = slot[p.side]
                slot[p.side] += 1
                dest[p.side, i, k] = p.dest
                self.pa0[p.side, i, k] = p.a
                self.pb0[p.side, i, k] = p.b
                if p.gate >= 0 or p.cover is not None:
                    self.dyn_ports.append((p.side, i, k, p.gate, p.cover))
        self.pdest = xp.asarray(dest)
        # D rows: 0 y0, 1 y1, 2 u_top, 3 u_bot, then 4 + (side*2+k)*2 -> a, +1 -> b
        self.D_np = np.zeros((20, nb), dtype=np.float32)
        self.D_np[0], self.D_np[1] = self.y0s, self.y1s
        for s in range(4):
            for k in range(2):
                self.D_np[4 + (s * 2 + k) * 2] = self.pa0[s, :, k]
                self.D_np[5 + (s * 2 + k) * 2] = self.pb0[s, :, k]
        # pistons that move a box wall
        self.floor_of = [(i, b.floor) for i, b in enumerate(bx) if b.floor >= 0]
        self.ceil_of = [(i, b.ceil, b.ceil_face) for i, b in enumerate(bx) if b.ceil >= 0]
        # open edges: (box, side, length, boost, fuel, role)
        self.open_edges = []
        for i, b in enumerate(bx):
            for s in range(4):
                if b.open[s]:
                    L = (b.x1 - b.x0) if s in (E.TOP, E.BOT) else (b.y1 - b.y0)
                    fuel = eng.fuel_frac if b.role == "intake" else 0.0
                    bst = (b.side_boost or {}).get(s, b.boost)
                    self.open_edges.append((i, s, L, bst, fuel, b.role))

    def _mix(self, n, mix, fuel):
        """Species for n new particles: a {species: fraction} dict (or fn(t) -> dict, a
        mixture that sweeps), or air + `fuel`."""
        if callable(mix):
            mix = mix(self.t)
        r = self.rng.random(n, dtype=xp.float32)
        if mix:
            out = xp.full(n, C.AIR, dtype=xp.int32)
            acc = 0.0
            for sp, f in mix.items():
                out = xp.where((r >= acc) & (r < acc + f), sp, out)
                acc += f
            return out.astype(xp.int32)
        if fuel > 0:
            return xp.where(r < fuel, C.FUEL, C.AIR).astype(xp.int32)
        return xp.full(n, C.AIR, dtype=xp.int32)

    def _piston_state(self, t):
        """(crown, velocity) for every piston, CPU floats."""
        return [p.fn(t) for p in self.eng.pistons]

    def _dynamics(self, t):
        """Fill the packed D array for this substep: moving walls and effective ports."""
        eng, D = self.eng, self.D_np
        ps = self._piston_state(t)
        D[2] = 0.0
        D[3] = 0.0
        for i, pi in self.floor_of:
            y, u = ps[pi]
            D[1, i] = y
            D[3, i] = u
        for i, pi, face in self.ceil_of:
            y, u = ps[pi]
            if face == "under":
                y = y + eng.pistons[pi].height
            D[0, i] = y
            D[2, i] = u
        lifts = [v.lift(t) for v in eng.valves]
        for s, i, k, gate, cover in self.dyn_ports:
            a, b = self.pa0[s, i, k], self.pb0[s, i, k]
            if gate >= 0:
                lift = lifts[gate]
                if lift < 1e-3:
                    a, b = 1.0, -1.0                 # shut: an empty interval
                else:
                    mid, half = 0.5 * (a + b), 0.5 * (b - a)
                    a, b = mid - half * lift, mid + half * lift
            if cover is not None:
                pi, face = cover
                y, _ = ps[pi]
                p = eng.pistons[pi]
                if face == "crown":
                    if p.up:
                        b = min(b, y)               # gas is above the crown
                    else:
                        a = max(a, y)               # crown faces down: gas below it
                else:
                    a = max(a, y + p.height)        # under the skirt of an up piston
            D[4 + (s * 2 + k) * 2, i] = a
            D[5 + (s * 2 + k) * 2, i] = b
        return ps, lifts

    def seed_domain(self):
        """Fill every box with ambient gas at rest, at T0 (times the box's boost). The clip
        therefore starts with a live gas on frame one - nothing fades in (OUTPUT_RULES)."""
        eng = self.eng
        ps = self._piston_state(0.0)
        xs, ys, bs, sps, sig = [], [], [], [], []
        for i, b in enumerate(eng.boxes):
            y0, y1 = b.y0, b.y1
            if b.floor >= 0:
                y1 = ps[b.floor][0]
            if b.ceil >= 0:
                y0 = ps[b.ceil][0] + (eng.pistons[b.ceil].height if b.ceil_face == "under"
                                      else 0.0)
            area = max(0.0, (b.x1 - b.x0) * (y1 - y0))
            bst = b.boost(0.0) if callable(b.boost) else b.boost
            n = int(round(C.N0 * bst * area))
            if n <= 0:
                continue
            u = self.rng.random(n, dtype=xp.float32)
            v = self.rng.random(n, dtype=xp.float32)
            xs.append(b.x0 + u * (b.x1 - b.x0))
            ys.append(y0 + v * (y1 - y0))
            bs.append(xp.full(n, i, dtype=xp.int32))
            fuel = eng.fuel_frac if b.role == "intake" else 0.0
            sps.append(self._mix(n, b.res_mix, fuel))
            sig.append(xp.full(n, sigma_at(b.res_T or C.T0), dtype=xp.float32))
        self.x = xp.concatenate(xs).astype(xp.float32)
        self.y = xp.concatenate(ys).astype(xp.float32)
        self.box = xp.concatenate(bs)
        self.sp = xp.concatenate(sps)
        sg = xp.concatenate(sig)
        # a part's metal and its port holes sit inside boxes: seed only the gas
        keep = xp.ones(self.x.size, dtype=bool)
        if self.solids is not None:
            keep &= self.solids.dmin(self.x, self.y, self.box) > 0.5
        for h in self.holes:
            keep &= ~((self.box == h.box) & ((self.x - h.cx) ** 2 + (self.y - h.cy) ** 2 < h.r * h.r))
        self.x, self.y, self.box, self.sp, sg = (self.x[keep], self.y[keep], self.box[keep],
                                                 self.sp[keep], sg[keep])
        n = self.x.size
        self.vx = (self.rng.normal(0.0, 1.0, n) * sg).astype(xp.float32)
        self.vy = (self.rng.normal(0.0, 1.0, n) * sg).astype(xp.float32)
        self.tag = (self.rng.random(n, dtype=xp.float32) * 65536.0).astype(xp.int32)
        self.px, self.py = self.x.copy(), self.y.copy()
        self._seeded = int(n)

    # ------------------------------------------------------------------ the step
    def _reeds(self):
        """A reed opens on PRESSURE: when its crankcase is below the intake. Two syncs per
        reed per substep - the only ones in the step apart from the removal count."""
        for k, v in self.reeds:
            b = self.eng.boxes[v.box]
            mem = self.box == v.box
            n = int(mem.sum())
            y0 = b.y0
            if b.ceil >= 0:
                y0 = self._ps[b.ceil][0] + (self.eng.pistons[b.ceil].height
                                            if b.ceil_face == "under" else 0.0)
            area = (b.x1 - b.x0) * max(1.0, b.y1 - y0)
            if n:
                T = temperature_K(float(xp.where(mem, self.vx ** 2 + self.vy ** 2, 0.0).sum()) / n)
            else:
                T = C.T0
            p = (n * T / area) / (C.N0 * C.T0)
            v.last = E.smoothstep((v.p_ref - p) / (0.10 * v.p_ref))

    def step(self, dt: float):
        eng = self.eng
        if self.reeds:
            self._ps = self._piston_state(self.t)
            self._reeds()
        hook = eng.notes.get("pre_step")          # a part's own pressure-driven valves (shims)
        if hook is not None:
            hook(self)
        self._ddeg = eng.drive.deg(self.t + dt) - eng.drive.deg(self.t)
        ps, lifts = self._dynamics(self.t)
        self.D = xp.asarray(self.D_np)
        self._ps, self._lifts = ps, lifts

        if eng.igniters:
            st = stroke_name(eng.igniters[0].clock(self.t), eng.cycle_deg)
            if st != self._last_stroke:
                self.events.append((self.t, "stroke", st))
                self._last_stroke = st

        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt

        self._walls(self.D)
        if self.solids is not None:
            self.solids.set_poses(self.t + dt)
            self.solids.collide(self)
        self._boundaries(dt)
        if eng.injectors:
            self._inject(dt)
        if self.ignite and eng.igniters:
            self._combust(dt)
        self.substep += 1
        if self.coll_every and self.substep % self.coll_every == 0:
            self._collide(dt * self.coll_every)
        self.t += dt
        self.impulse_window += dt

    # .................................................................. walls
    def _port_pass(self, D, beyond, side, perp):
        """Move particles that are beyond an edge and inside one of that edge's (effective)
        port intervals into the destination box. Returns the mask of those that were moved."""
        b = self.box
        moved = xp.zeros_like(beyond)
        for k in (0, 1):
            dst = self.pdest[side][b, k]
            a = D[4 + (side * 2 + k) * 2][b]
            bb = D[5 + (side * 2 + k) * 2][b]
            thru = beyond & ~moved & (dst >= 0) & (perp >= a) & (perp <= bb)
            self.box = xp.where(thru, dst, self.box)
            moved = moved | thru
        return moved

    def _walls(self, D):
        """One gathered pass per side over every particle. Floors and ceilings may move
        (pistons) and reflect in the wall's frame: v' = -v + 2u. Static walls have u = 0."""
        nb = self.nbox
        # --- ceiling ---
        b = self.box
        y0 = D[0][b]
        beyond = self.y < y0
        moved = self._port_pass(D, beyond, E.TOP, self.x)
        b = self.box
        hit = beyond & ~moved & (self.open[E.TOP][b] < 0.5)
        u = D[2][b]
        self.imp += xp.bincount(b, weights=xp.where(hit, 2.0 * xp.abs(self.vy - u), 0.0),
                                minlength=nb)[:nb].astype(xp.float32)
        self.y = xp.where(hit, 2.0 * y0 - self.y, self.y)
        self.vy = xp.where(hit, -self.vy + 2.0 * u, self.vy)

        # --- floor: the piston, for a cylinder ---
        b = self.box
        y1 = D[1][b]
        beyond = self.y > y1
        moved = self._port_pass(D, beyond, E.BOT, self.x)
        b = self.box
        hit = beyond & ~moved & (self.open[E.BOT][b] < 0.5)
        u = D[3][b]
        self.imp += xp.bincount(b, weights=xp.where(hit, 2.0 * xp.abs(self.vy - u), 0.0),
                                minlength=nb)[:nb].astype(xp.float32)
        self.y = xp.where(hit, 2.0 * y1 - self.y, self.y)
        self.vy = xp.where(hit, -self.vy + 2.0 * u, self.vy)

        # --- left ---
        b = self.box
        x0 = self.bx0[b]
        beyond = self.x < x0
        moved = self._port_pass(D, beyond, E.LEFT, self.y)
        b = self.box
        hit = beyond & ~moved & (self.open[E.LEFT][b] < 0.5)
        self.imp += xp.bincount(b, weights=xp.where(hit, 2.0 * xp.abs(self.vx), 0.0),
                                minlength=nb)[:nb].astype(xp.float32)
        self.x = xp.where(hit, 2.0 * x0 - self.x, self.x)
        self.vx = xp.where(hit, -self.vx, self.vx)

        # --- right ---
        b = self.box
        x1 = self.bx1[b]
        beyond = self.x > x1
        moved = self._port_pass(D, beyond, E.RIGHT, self.y)
        b = self.box
        hit = beyond & ~moved & (self.open[E.RIGHT][b] < 0.5)
        self.imp += xp.bincount(b, weights=xp.where(hit, 2.0 * xp.abs(self.vx), 0.0),
                                minlength=nb)[:nb].astype(xp.float32)
        self.x = xp.where(hit, 2.0 * x1 - self.x, self.x)
        self.vx = xp.where(hit, -self.vx, self.vx)

        # --- last-resort clamp: a particle that crossed two walls in one substep ---
        # An open side is deliberately left violated for _boundaries(); a particle that has
        # just stepped through a port sits a hair outside its new box, which is rounding, not
        # a rescue. Only excursions past EPS count - that count says "raise --substeps".
        EPS = 0.75
        b = self.box
        lo_x, hi_x = self.bx0[b] + 0.02, self.bx1[b] - 0.02
        lo_y, hi_y = D[0][b] + 0.02, D[1][b] - 0.02
        o_t, o_b = self.open[E.TOP][b] > 0.5, self.open[E.BOT][b] > 0.5
        o_l, o_r = self.open[E.LEFT][b] > 0.5, self.open[E.RIGHT][b] > 0.5
        bad = (((self.x < lo_x - EPS) & ~o_l) | ((self.x > hi_x + EPS) & ~o_r)
               | ((self.y < lo_y - EPS) & ~o_t) | ((self.y > hi_y + EPS) & ~o_b))
        self.n_clamped += int(bad.sum())
        self.x = xp.where(o_l & (self.x < lo_x), self.x,
                          xp.where(o_r & (self.x > hi_x), self.x, xp.clip(self.x, lo_x, hi_x)))
        self.y = xp.where(o_t & (self.y < lo_y), self.y,
                          xp.where(o_b & (self.y > hi_y), self.y, xp.clip(self.y, lo_y, hi_y)))

    # .................................................. open reservoir boundaries
    def _boundaries(self, dt):
        """Open edges are ambient reservoirs: particles that cross are removed, and particles
        are injected at the one-sided equilibrium flux n*sigma/sqrt(2pi) per unit length (times
        the edge's boost). Self-regulating: a plenum denser than ambient loses more than it
        gains. That is what makes an intake an intake without a script."""
        b = self.box
        D = self.D
        gone = ((self.open[E.TOP][b] > 0.5) & (self.y < D[0][b])
                | (self.open[E.BOT][b] > 0.5) & (self.y > D[1][b])
                | (self.open[E.LEFT][b] > 0.5) & (self.x < self.bx0[b])
                | (self.open[E.RIGHT][b] > 0.5) & (self.x > self.bx1[b]))
        ng = int(gone.sum())
        if ng and self.eng.notes.get("species_flows"):
            gx = gone & xp.isin(b, self.ex_boxes)
            sp_np = self.sp
            fe = xp.bincount(xp.where(gx, sp_np, 5), minlength=6)[:5]
            fi = xp.bincount(xp.where(gone & ~gx, sp_np, 5), minlength=6)[:5]
            self.flow_out["exhaust"] += (fe.get() if hasattr(fe, "get") else fe).astype(np.int64)
            self.flow_out["intake"] += (fi.get() if hasattr(fi, "get") else fi).astype(np.int64)
        if ng:
            n_ex = int((gone & xp.isin(b, self.ex_boxes)).sum())
            self.n_out_ex += n_ex
            self.n_out_in += ng - n_ex
        for k, h in enumerate(self.holes):
            inh = (b == h.box) & ((self.x - h.cx) ** 2 + (self.y - h.cy) ** 2 < h.r * h.r) & ~gone
            nh = int(inh.sum())
            if nh:
                self.n_hole_out[k] += nh
                if h.role == "exhaust":
                    self.n_out_ex += nh
                else:
                    self.n_out_in += nh
                gone = gone | inh
                ng += nh
        if ng:
            keep = ~gone
            self.x, self.y = self.x[keep], self.y[keep]
            self.vx, self.vy = self.vx[keep], self.vy[keep]
            self.sp, self.box = self.sp[keep], self.box[keep]
            self.px, self.py = self.px[keep], self.py[keep]
            self.tag = self.tag[keep]

        new = []
        for slot, (bi, side, L, boost, fuel, role) in enumerate(self.open_edges):
            s = sigma_at(self.eng.boxes[bi].res_T or C.T0)
            bst = boost(self.t) if callable(boost) else boost      # a reservoir that follows the drive
            self._inject_acc[slot] += C.N0 * bst * L * s / SQRT_2PI * dt
            n = int(self._inject_acc[slot])
            if n <= 0:
                continue
            self._inject_acc[slot] -= n
            if self.x.size + n > self.cap:
                continue
            bx = self.eng.boxes[bi]
            u = self.rng.random(n, dtype=xp.float32)
            r = self.rng.random(n, dtype=xp.float32)
            # inward normal speed is FLUX-weighted, not Maxwellian: Rayleigh, not Gaussian
            vn = (s * xp.sqrt(-2.0 * xp.log(xp.maximum(r, 1e-7)))).astype(xp.float32)
            vt = self.rng.normal(0.0, s, n).astype(xp.float32)
            if side == E.TOP:
                x, y, vx, vy = bx.x0 + u * L, xp.full(n, bx.y0 + 0.5, dtype=xp.float32), vt, vn
            elif side == E.BOT:
                x, y, vx, vy = bx.x0 + u * L, xp.full(n, bx.y1 - 0.5, dtype=xp.float32), vt, -vn
            elif side == E.LEFT:
                x, y, vx, vy = xp.full(n, bx.x0 + 0.5, dtype=xp.float32), bx.y0 + u * L, vn, vt
            else:
                x, y, vx, vy = xp.full(n, bx.x1 - 0.5, dtype=xp.float32), bx.y0 + u * L, -vn, vt
            sp = self._mix(n, bx.res_mix, fuel)
            bb = xp.full(n, bi, dtype=xp.int32)
            if self.solids is not None:
                # an open edge may run under metal (a housing past a duct's mouth): only the
                # stretch that is gas injects, so the flux is the free length's, as it should be
                x = x.astype(xp.float32) if hasattr(x, "astype") else xp.full(n, x, dtype=xp.float32)
                y = y.astype(xp.float32) if hasattr(y, "astype") else xp.full(n, y, dtype=xp.float32)
                ok = self.solids.dmin(x, y, bb) > 0.5
                x, y, vx, vy, sp, bb = x[ok], y[ok], vx[ok], vy[ok], sp[ok], bb[ok]
                n = int(x.size)
                if n == 0:
                    continue
            new.append((x, y, vx, vy, sp, bb))
            if self.eng.notes.get("species_flows"):
                cnt = xp.bincount(sp, minlength=5)[:5]
                self.flow_in["exhaust" if role == "exhaust" else "intake"] += (
                    cnt.get() if hasattr(cnt, "get") else cnt).astype(np.int64)
            if role == "exhaust":
                self.n_in_ex += n
            else:
                self.n_in += n
        for k, h in enumerate(self.holes):
            if not h.inject:
                continue
            s = sigma_at(h.T)
            bst = h.boost(self.t) if callable(h.boost) else h.boost
            self._hole_acc[k] += C.N0 * bst * (2.0 * math.pi * h.r) * s / SQRT_2PI * dt
            n = int(self._hole_acc[k])
            if n <= 0:
                continue
            self._hole_acc[k] -= n
            if self.x.size + n > self.cap:
                continue
            a = self.rng.random(n, dtype=xp.float32) * xp.float32(2.0 * math.pi)
            r = self.rng.random(n, dtype=xp.float32)
            vn = (s * xp.sqrt(-2.0 * xp.log(xp.maximum(r, 1e-7)))).astype(xp.float32)
            vt = self.rng.normal(0.0, s, n).astype(xp.float32)
            ca, sa = xp.cos(a), xp.sin(a)
            x = (h.cx + (h.r + 0.5) * ca).astype(xp.float32)
            y = (h.cy + (h.r + 0.5) * sa).astype(xp.float32)
            vx = (vn * ca - vt * sa).astype(xp.float32)
            vy = (vn * sa + vt * ca).astype(xp.float32)
            sp = self._mix(n, h.mix, 0.0)
            bx_ = xp.full(n, h.box, dtype=xp.int32)
            if self.solids is not None:
                ok = self.solids.dmin(x, y, bx_) > 0.5
                x, y, vx, vy, sp, bx_ = x[ok], y[ok], vx[ok], vy[ok], sp[ok], bx_[ok]
                n = int(x.size)
                if n == 0:
                    continue
            new.append((x, y, vx, vy, sp, bx_))
            self.n_hole_in[k] += n
            if h.role == "exhaust":
                self.n_in_ex += n
            else:
                self.n_in += n
        if new:
            self._append(new)

    def _append(self, new):
        xs, ys, vxs, vys, sps, bxs = zip(*new)
        n0 = self.x.size
        self.x = xp.concatenate([self.x, *xs]).astype(xp.float32)
        self.y = xp.concatenate([self.y, *ys]).astype(xp.float32)
        self.vx = xp.concatenate([self.vx, *vxs]).astype(xp.float32)
        self.vy = xp.concatenate([self.vy, *vys]).astype(xp.float32)
        self.sp = xp.concatenate([self.sp, *sps])
        self.box = xp.concatenate([self.box, *bxs])
        pad = self.x.size - n0
        self.px = xp.concatenate([self.px, self.x[-pad:]])
        self.py = xp.concatenate([self.py, self.y[-pad:]])
        self.tag = xp.concatenate([
            self.tag, (self.rng.random(pad, dtype=xp.float32) * 65536.0).astype(xp.int32)])

    # ................................................................ injectors
    def _inject(self, dt):
        """A diesel's fuel: sprayed from the tip, over a crank window, as a cone."""
        for k, inj in enumerate(self.eng.injectors):
            d = inj.clock(self.t)
            span = (inj.end_deg - inj.start_deg)
            inside = ((d - inj.start_deg) % self.eng.cycle_deg) < span
            if not inside:
                self._inj_done[k] = False
                continue
            self._inj_acc[k] += inj.n_total * self._ddeg / span
            n = int(self._inj_acc[k])
            if n <= 0 or self.x.size + n > self.cap:
                continue
            self._inj_acc[k] -= n
            if not self._inj_done[k]:
                self._inj_done[k] = True
                self.events.append((self.t, "inject", f"{inj.label} {d:.0f} deg"))
            a0 = math.radians(inj.angle_deg)
            sp = math.radians(inj.spread_deg)
            ang = xp.float32(a0) + (self.rng.random(n, dtype=xp.float32) - 0.5) * xp.float32(sp)
            v = xp.float32(inj.speed * C.V_TH) * (0.7 + 0.6 * self.rng.random(n, dtype=xp.float32))
            vx = v * xp.cos(ang) + self.rng.normal(0.0, 0.3 * sigma_at(C.T0), n)
            vy = v * xp.sin(ang) + self.rng.normal(0.0, 0.3 * sigma_at(C.T0), n)
            x = xp.float32(inj.tip[0]) + self.rng.normal(0.0, 3.0, n)
            y = xp.float32(inj.tip[1]) + self.rng.normal(0.0, 2.0, n)
            self._append([(x.astype(xp.float32), y.astype(xp.float32), vx.astype(xp.float32),
                           vy.astype(xp.float32), xp.full(n, inj.species, dtype=xp.int32),
                           xp.full(n, inj.box, dtype=xp.int32))])
            self.n_injected += n

    # ........................................................... DSMC collisions
    def _collide(self, dtc):
        """Bird's no-time-counter scheme, sorted-cell flavour. Shuffle, STABLE-sort by
        (box, cell) so the within-cell order is random; pair adjacent entries; accept with
        probability ~ |v_rel| x local density; 2D equal-mass elastic kernel."""
        n = self.x.size
        if n < 4:
            return
        cs = 40.0
        ix = xp.clip((self.x / cs).astype(xp.int32), 0, 63)
        iy = xp.clip(((self.y + 256.0) / cs).astype(xp.int32), 0, 63)
        key = (self.box.astype(xp.int64) * 4096 + iy.astype(xp.int64) * 64
               + ix.astype(xp.int64))
        perm = self.rng.permutation(n)
        order = perm[xp.argsort(key[perm], kind="stable")]
        k = (n // 2) * 2
        a, b = order[0:k:2], order[1:k:2]
        same = key[a] == key[b]
        counts = xp.bincount(key, minlength=self.nbox * 4096)
        n_cell = counts[key[a]].astype(xp.float32)
        n_ref = C.N0 * cs * cs
        gx, gy = self.vx[a] - self.vx[b], self.vy[a] - self.vy[b]
        g = xp.sqrt(gx * gx + gy * gy)
        p = (n_cell / n_ref) * g * dtc / (self.mfp * math.sqrt(2.0))
        acc = same & (self.rng.random(a.size, dtype=xp.float32) < xp.minimum(p, 1.0))
        ia, ib = a[acc], b[acc]
        if ia.size == 0:
            return
        cx = 0.5 * (self.vx[ia] + self.vx[ib])
        cy = 0.5 * (self.vy[ia] + self.vy[ib])
        h = 0.5 * xp.sqrt((self.vx[ia] - self.vx[ib]) ** 2 + (self.vy[ia] - self.vy[ib]) ** 2)
        ang = self.rng.random(ia.size, dtype=xp.float32) * xp.float32(TWO_PI)
        dx, dy = h * xp.cos(ang), h * xp.sin(ang)
        self.vx[ia], self.vy[ia] = cx + dx, cy + dy
        self.vx[ib], self.vy[ib] = cx - dx, cy - dy

    # ................................................................ combustion
    def _member(self, boxes):
        m = self.box == boxes[0]
        for bi in boxes[1:]:
            m = m | (self.box == bi)
        return m

    def _combust(self, dt):
        """Per igniter: spark or autoignition, then a front expanding from `origin` at
        flame_speed. Inside the front:  1 FUEL + 2 AIR -> 1 CO2 + 2 H2O, plus q_rxn of kinetic
        energy shared over the three participants as a random-direction velocity increment.
        Particle COUNT is conserved exactly, which is what makes the --dry ledger a real check.

        The exotherm is deliberately a COLOUR and SPEED change confined to the chamber, never a
        luminance event: OUTPUT_RULES treats a full-frame flash as a defect."""
        eng = self.eng
        for k, ig in enumerate(eng.igniters):
            st = self.ig[k]
            d = ig.clock(self.t)
            if st["spark_t"] is None:
                if ig.auto_K > 0.0:
                    if not (ig.window[0] < d < ig.window[1]):
                        continue
                    mem = self._member(ig.boxes)
                    if ig.region is not None:
                        mem = mem & ig.region(self)
                    n_mem = int(mem.sum())
                    if n_mem == 0:
                        continue
                    if ig.need_fuel and int((mem & (self.sp == C.FUEL)).sum()) < 8:
                        continue
                    T = temperature_K(float(xp.where(mem, self.vx ** 2 + self.vy ** 2, 0.0)
                                            .sum()) / n_mem)
                    if T < ig.auto_K:
                        continue
                    st["spark_t"], st["spark_deg"], st["flame_r"] = self.t, eng.drive.deg(self.t), 0.0
                    self.events.append((self.t, "autoignite",
                                        f"{ig.label} {d:.0f} deg, {T:.0f} K"))
                elif 0.0 <= (d - ig.spark_deg) % eng.cycle_deg < 4.0:
                    st["spark_t"], st["spark_deg"], st["flame_r"] = self.t, eng.drive.deg(self.t), 0.0
                    self.events.append((self.t, "spark", f"{ig.label} {d:.0f} deg"))
                else:
                    continue
            since = eng.drive.deg(self.t) - st["spark_deg"]         # crank degrees since ignition
            if since > ig.rearm * eng.cycle_deg:
                st["spark_t"], st["flame_r"] = None, 0.0
                continue
            st["flame_r"] = ig.flame_deg * since
            r = st["flame_r"]
            dx = self.x - ig.origin[0]
            dy = self.y - ig.origin[1]
            inside = self._member(ig.boxes) & (dx * dx + dy * dy <= r * r)
            if ig.region is not None:
                inside = inside & ig.region(self)
            fuel_idx = xp.where(inside & (self.sp == C.FUEL))[0]
            if fuel_idx.size == 0:
                continue
            air_idx = xp.where(inside & (self.sp == C.AIR))[0]
            rx = eng.notes.get("rxn")
            if rx:
                self._react_units(rx, fuel_idx, air_idx)
                continue
            nf = int(min(int(fuel_idx.size), int(air_idx.size) // 2))
            if nf <= 0:
                continue
            fuel_idx = fuel_idx[:nf]
            air_idx = air_idx[self.rng.permutation(air_idx.size)[:2 * nf]]
            self.sp[fuel_idx] = C.CO2
            self.sp[air_idx] = C.H2O
            self.reactions += nf
            self.made_co2 += nf
            self.made_h2o += 2 * nf
            parts = xp.concatenate([fuel_idx, air_idx])
            kick = math.sqrt(2.0 * self.q_rxn / 3.0)
            ang = self.rng.random(parts.size, dtype=xp.float32) * xp.float32(TWO_PI)
            self.vx[parts] = self.vx[parts] + xp.float32(kick) * xp.cos(ang)
            self.vy[parts] = self.vy[parts] + xp.float32(kick) * xp.sin(ang)

    def _react_units(self, rx, fuel_idx, air_idx):
        """General stoichiometry, per reaction UNIT: rx['fuel'] FUEL + rx['ox'] AIR(=O2) ->
        rx['co2'] CO2 + rx['h2o'] H2O. Reactant molecules are re-labelled as products and any
        surplus products are CREATED at the positions (and with the velocities) of random
        reactants, so the gas really gains molecules - gasoline goes 27 -> 34, which is part of
        why its burn raises the pressure. q_rxn is per UNIT, shared over all its products."""
        f, o, c, h = int(rx["fuel"]), int(rx["ox"]), int(rx["co2"]), int(rx["h2o"])
        u = int(min(int(fuel_idx.size) // f, int(air_idx.size) // o))
        if u <= 0:
            return
        fi = fuel_idx[self.rng.permutation(fuel_idx.size)[:f * u]]
        ai = air_idx[self.rng.permutation(air_idx.size)[:o * u]]
        reac = xp.concatenate([fi, ai])                       # (f + o) u molecules
        n_r, n_p = int(reac.size), (c + h) * u
        self.sp[reac[:c * u]] = C.CO2
        self.sp[reac[c * u:]] = C.H2O
        parts = reac
        extra = n_p - n_r
        if extra > 0:
            src = reac[self.rng.permutation(n_r)[:extra] % n_r] if extra <= n_r else \
                reac[(self.rng.random(extra, dtype=xp.float32) * n_r).astype(xp.int64)]
            n0 = self.x.size
            jit = (self.rng.random(extra, dtype=xp.float32) - 0.5) * 1.0
            self._append([(self.x[src] + jit, self.y[src] - jit, self.vx[src].copy(),
                           self.vy[src].copy(), xp.full(extra, C.H2O, dtype=xp.int32),
                           self.box[src].copy())])
            parts = xp.concatenate([reac, xp.arange(n0, n0 + extra)])
            self.n_created += extra
        self.reactions += u
        self.made_co2 += c * u
        self.made_h2o += h * u
        kick = math.sqrt(2.0 * self.q_rxn / (c + h))
        ang = self.rng.random(parts.size, dtype=xp.float32) * xp.float32(TWO_PI)
        self.vx[parts] = self.vx[parts] + xp.float32(kick) * xp.cos(ang)
        self.vy[parts] = self.vy[parts] + xp.float32(kick) * xp.sin(ang)

    # ------------------------------------------------------------------ measurement
    def cyl_state(self, i):
        """Everything --dry reports for cylinder box i, measured from the particles."""
        eng = self.eng
        b = eng.boxes[i]
        ps = self._ps
        y1 = ps[b.floor][0] if b.floor >= 0 else b.y1
        y0 = b.y0
        if b.ceil >= 0:
            y0 = ps[b.ceil][0] + (eng.pistons[b.ceil].height if b.ceil_face == "under" else 0.0)
        mem = self.box == i
        n = int(mem.sum())
        area = (b.x1 - b.x0) * max(1.0, y1 - y0)
        if n:
            T = temperature_K(float(xp.where(mem, self.vx ** 2 + self.vy ** 2, 0.0).sum()) / n)
        else:
            T = C.T0
        p0 = C.N0 * C.T0
        p_ideal = (n * T / area) / p0
        perim = 2.0 * (b.x1 - b.x0) + 2.0 * max(1.0, y1 - y0)
        if self.impulse_window > 0:
            p_wall = ((float(self.imp[i]) / (perim * self.impulse_window))
                      / (C.N0 * C.V_TH ** 2 / 2.0))
        else:
            p_wall = 0.0
        counts = {k: int((mem & (self.sp == k)).sum()) for k in (C.AIR, C.FUEL, C.CO2, C.H2O)}
        return dict(box=i, label=b.label, n=n, T=T, area=area, p_ideal=p_ideal, p_wall=p_wall,
                    counts=counts, y0=y0, y1=y1)

    def frame_state(self):
        eng = self.eng
        cyls = [self.cyl_state(i) for i in eng.cyl_boxes()]
        deg = (eng.igniters[0].clock(self.t) if eng.igniters
               else eng.drive.deg(self.t) % eng.cycle_deg)
        c0 = cyls[0] if cyls else dict(T=C.T0, p_ideal=1.0, p_wall=0.0, n=0, area=1.0,
                                       counts={k: 0 for k in (0, 1, 2, 3)})
        return dict(t=self.t, deg=deg, stroke=stroke_name(deg, eng.cycle_deg),
                    n_total=int(self.x.size), cyl=cyls,
                    T=c0["T"], p_ideal=c0["p_ideal"], p_wall=c0["p_wall"],
                    n_cyl=c0["n"], area=c0["area"], counts=c0["counts"],
                    lifts=[v.lift(self.t) for v in eng.valves],
                    flame=[s["flame_r"] if s["spark_t"] is not None else 0.0 for s in self.ig])

    def reset_impulse(self):
        self.imp[:] = 0.0
        self.impulse_window = 0.0

    def mark_frame(self):
        """Remember this frame's positions so the next frame can draw the motion streak."""
        self.px, self.py = self.x.copy(), self.y.copy()

    # ------------------------------------------------------------------ draw feed
    def snapshot(self):
        """World-frame copies of what draw.py needs: positions now and a frame ago (rotated
        out of each group's local frame), speed squared, species, and the carried tag."""
        n = self.x.size
        px, py = (self.px, self.py) if self.px.size == n else (self.x, self.y)
        g = self.group_of[self.box]
        c, s = self.g_cos[g], self.g_sin[g]
        pvx, pvy, atx, aty = self.g_pvx[g], self.g_pvy[g], self.g_atx[g], self.g_aty[g]

        def tw(x, y):
            dx, dy = x - pvx, y - pvy
            return atx + c * dx - s * dy, aty + s * dx + c * dy
        xw, yw = tw(self.x, self.y)
        pxw, pyw = tw(px, py)
        return dict(x=xw, y=yw, px=pxw, py=pyw, v2=self.vx * self.vx + self.vy * self.vy,
                    sp=self.sp, tag=self.tag)
