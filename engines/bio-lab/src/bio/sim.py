# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""sim.py - the cell population: growth, division, and the forces between cells.

State is five parallel arrays indexed by cell, and nothing else:

    pos     (N, 2)  centre of the cell
    a0      (N,)    TARGET area - what the cell is trying to be. Growth raises this.
    age     (N,)    seconds since this cell last divided
    gen     (N,)    how many divisions deep this cell's lineage is
    line    (N,)    lineage tag, inherited on division - drives the lineage colouring

Everything else - polygons, neighbours, pressure - is derived per step from `tess.py`.
Keeping derived quantities OUT of the state is what makes division trivial: to divide a
cell you append a row to five arrays, and no other structure needs to hear about it.


THE FORCES, AND WHICH SLIDER IS WHICH
-------------------------------------
Overdamped: there is no inertia. ``v = F / eta``, position integrates directly. Cells in
a tissue are deep in the viscous regime - a cell that stops being pushed stops moving,
it does not coast - and using inertia here makes the colony wobble like jelly.

* **repulsion** - cells overlapping by more than their target radii push apart. This is
  incompressibility, and it is what makes the colony fill space rather than pile up.
* **adhesion** - cells slightly further apart than touching pull together. This is what
  holds the colony as ONE object rather than a cloud.
* **outside pressure** - acts along ``free_dir``: minus the mean unit vector to a cell's
  neighbours. For a cell in the bulk that sum cancels to nearly zero; for a cell on the
  rim it points outward. So a single expression applies pressure to the boundary and
  nowhere else, with no boundary detection and no special case.
* **surface tension** - Laplacian smoothing of the boundary, weighted by how exposed a
  cell is. Pressure COMPACTS the colony; tension SMOOTHS its outline. They are genuinely
  different: high pressure with low tension gives a dense colony with a ragged, fingered
  rim, and low pressure with high tension gives a loose but perfectly round one. If they
  had been implemented as one knob you could never get either of those.


CONTACT INHIBITION IS ALSO THE POPULATION CAP
---------------------------------------------
A cell only divides when its local pressure is below `inhibit`. That is real biology -
crowded epithelial cells stop dividing - and it doubles as the thing that stops the
population running away exponentially and taking the frame rate with it. `max_cells` is
a hard backstop on top, but in a well-set scene inhibition is what actually binds.
"""
from __future__ import annotations

import numpy as np

from .tess import disc_poly, polygon_area, polygons_of, voronoi_of


class Colony:
    def __init__(self, p, seed=7):
        self.p = p
        self.rng = np.random.default_rng(int(seed))
        self.t = 0.0
        self.reset()

    # --- setup -------------------------------------------------------------------------
    def reset(self):
        p = self.p
        n0 = int(p["seed_cells"])
        a0 = float(p["cell_area"])
        r0 = np.sqrt(a0 / np.pi)
        nclust = max(1, int(p.get("clusters", 1)))

        # SEED ONE OR MORE CLUSTERS.
        #
        # Several clusters seeded apart from each other, each its own STRAIN with its own
        # colour and its own growth and adhesion, is the whole "competing colonies" idea.
        # They are placed on a ring rather than at random so no two ever start on top of
        # each other - overlapping seeds resolve into one blob and you lose the premise.
        centres = [np.zeros(2)]
        if nclust > 1:
            spread = float(p.get("cluster_spread", 520.0))
            ang = np.linspace(0, 2 * np.pi, nclust, endpoint=False)
            ang = ang + self.rng.uniform(-0.18, 0.18, nclust)
            rad = spread * (0.72 + 0.42 * self.rng.random(nclust))
            asp0 = max(0.2, float(p.get("seed_aspect", 1.0)))
            centres = [np.array([np.cos(a) * r, np.sin(a) * r * asp0])
                       for a, r in zip(ang, rad)]

        # PER-STRAIN CELL SIZE. Different cultures are made of different-sized cells, which
        # is most of what makes them read as different ORGANISMS rather than as one
        # organism in several colours.
        smul = self.strain_sizes(nclust)

        if str(p.get("seed_mode", "blob")) == "filled":
            # ALREADY-DEVELOPED START. Pack a hexagonal lattice across a disc and hand each
            # site to its nearest culture centre, so the frame opens full of mature,
            # interlocking territories instead of a few specks that have to grow into one.
            R = float(p.get("seed_radius", 700.0))
            # SEED THE FRAME'S SHAPE, not a circle. A disc inscribed in a 9:16 frame
            # leaves black wedges at top and bottom that no amount of zoom removes -
            # measured 78% coverage where the brief asked for "the vast majority of the
            # frame". An ellipse at the frame's aspect covers the same frame with a third
            # fewer cells than a disc large enough to contain it.
            asp = max(0.2, float(p.get("seed_aspect", 1.0)))
            Ry = R * asp
            pitch = r0 * 1.92
            rows = int(np.ceil(2 * Ry / (pitch * 0.866))) + 2
            cols = int(np.ceil(2 * R / pitch)) + 2
            pts = []
            for iy in range(-rows // 2, rows // 2 + 1):
                y = iy * pitch * 0.866
                off = (pitch * 0.5) if (iy % 2) else 0.0
                for ix in range(-cols // 2, cols // 2 + 1):
                    x = ix * pitch + off
                    if (x / R) ** 2 + (y / Ry) ** 2 <= 1.0:
                        pts.append((x, y))
            P = np.array(pts, np.float64)
            if len(P) == 0:
                P = np.zeros((1, 2))
            P += self.rng.normal(0.0, r0 * 0.10, P.shape)
            C = np.array(centres)
            d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            S = np.argmin(d2, axis=1).astype(np.int32)
            # Territories are Voronoi of the culture centres, so boundaries are already
            # in contact at t=0 - the interaction starts immediately.
            cap = int(p.get("max_cells", 900))
            if len(P) > cap:
                # Keep the INNERMOST cells, not a random subset. A random subset of a
                # lattice is a lattice with holes in it, and the holes are permanent -
                # nothing in the force model refills a gap the size of a cell.
                q = np.hypot(P[:, 0] / R, P[:, 1] / Ry)
                keep = np.argsort(q)[:cap]
                P, S = P[keep], S[keep]
        else:
            per = max(1, n0 // nclust)
            Ps, Ss = [], []
            for si, cc in enumerate(centres):
                pts = [np.zeros(2)]
                ring = 1
                rs = r0 * np.sqrt(smul[si])
                while len(pts) < per:
                    k = 6 * ring
                    a = np.arange(k) * (2 * np.pi / k)
                    pts.extend((ring * rs * 1.9 * np.stack([np.cos(a), np.sin(a)], 1)))
                    ring += 1
                blob = np.array(pts[:per], np.float64) + cc
                Ps.append(blob)
                Ss.append(np.full(len(blob), si, np.int32))
            P = np.vstack(Ps)
            S = np.concatenate(Ss)
            P += self.rng.normal(0.0, r0 * 0.09, P.shape)

        n = len(P)
        self.pos = P
        self.strain = S if isinstance(S, np.ndarray) else np.concatenate(S)
        #: per-cell BASE area - the size this cell's culture wants to be. Growth and the
        #: division threshold are both measured against this, not against a global.
        self.abase = a0 * smul[np.clip(self.strain, 0, len(smul) - 1)] * self._pareto(n)
        # CELL-CYCLE PHASE. A developed culture is not a culture of newborns.
        #
        # Seeding every cell at its base area means every cell needs a full cycle before
        # anything divides: measured on the first `filled` build, vitreous_bloom sat at 91
        # cells with ZERO divisions for the first nine seconds of an eighteen second clip,
        # and the arc report called it out as "little happens". Spreading the initial areas
        # across the whole cycle instead puts cells at every phase, so divisions start on
        # the first frame and keep coming.
        if str(p.get("seed_mode", "blob")) == "filled":
            self.a0 = self.abase * self.rng.uniform(0.88, float(p["divide_at"]) * 0.99, n)
        else:
            self.a0 = self.abase * self.rng.uniform(0.85, 1.15, n)
        #: how long this cell's cytokinesis takes. Proportional to radius, so a big cell
        #: pinches slowly and visibly and a small one snaps.
        self.divT = self._divT(self.abase)
        #: the SMOOTHED colour driver. Colour lags the physics by `colour_tau`, and a
        #: daughter inherits its parent's value, so nothing changes hue the instant it is
        #: born. -1 means "not set yet".
        self.cval = np.full(n, -1.0)
        self.age = self.rng.uniform(0.0, max(1.0, float(p["cycle_min"])), n)
        self.gen = np.zeros(n, np.int32)
        self.line = np.arange(n, dtype=np.int32) % 8
        # PER-CELL GROWTH RATE MULTIPLIER, log-normal about 1 and INHERITED with mutation.
        #
        # Without this every cell grows at the same rate and reaches the division
        # threshold at nearly the same moment, so the colony divides in synchronised
        # waves and comes out with uniform cells in near-perfect radial symmetry. It
        # reads as a diagram. Real tissue has a spread of cycle times, and that spread is
        # most of what makes a colony look alive rather than generated.
        h = float(p.get("heterogeneity", 0.35))
        self.rate = self.rng.lognormal(0.0, h, n) if h > 0 else np.ones(n)

        # NUCLEUS IDENTITY. A hue offset per cell, inherited with a small drift, so
        # sisters are recognisably related and the field as a whole shimmers with
        # distinct points rather than one flat colour.
        self.nuc = self.rng.random(n).astype(np.float32)
        #: index of this cell's sister, or -1. Set at division, used to draw the
        #: cytokinetic bridge that makes a split legible.
        self.sister = np.full(n, -1, np.int32)
        #: time of this cell's most recent division, or -1e9 for "never".
        self.div_t = np.full(n, -1e9)

        #: 0 normal, 1 producer, 2 courier, 3 recipient. See `_emit`.
        self.kind = np.zeros(n, np.int8)
        #: a courier's destination cell, or -1. INDEX - remapped by `_remove`.
        self.target = np.full(n, -1, np.int32)
        #: a courier's destination attractor, or -1. Attractors are fixed points on the
        #: seed rim, so this is not an index into any cell array.
        self.attr = np.full(n, -1, np.int32)
        #: when this producer may next bud.
        self.emit_t = np.full(n, 1e9)
        #: organelle assortment seed. Fresh on BOTH daughters at division, which is what
        #: stops the pattern smearing across a cytokinesis.
        self.org = self.rng.integers(1, 2 ** 31, n).astype(np.int64)
        #: a producer's crawl heading.
        self.head = self.rng.uniform(0, 2 * np.pi, n)
        #: a stable identity. Removal shifts every index past it, so a cell cannot be
        #: followed by index through a clip - which makes every "is this cell jittering?"
        #: question unanswerable without one of these.
        self._next_uid = n
        self.uid = np.arange(n, dtype=np.int64)
        self._assign_kinds()

        self.polys = []
        self._vor = None
        self._nreal = n
        self._poly_t = None
        self.nb = np.zeros((0, 2), int)
        self.ncount = np.zeros(n, int)
        self.pressure = np.zeros(n)
        self.free = np.zeros_like(P)
        self._retess()

    #: every per-cell array, in one place. `_remove` and `check_invariants` both walk it,
    #: so adding a new one to this list is the whole job.
    ARRAYS = ("pos", "a0", "age", "gen", "line", "rate", "pressure", "free", "ncount",
              "strain", "nuc", "sister", "div_t", "abase", "divT", "cval",
              "kind", "target", "attr", "emit_t", "org", "head", "uid")

    def _assign_kinds(self):
        """Hand out producer and recipient roles.

        Recipients are taken from the INNERMOST cells, so the cell that visibly swells is
        one the frame is looking at. Producers are scattered everywhere, most of them
        outside the frame - that is the point of seeding a culture larger than the shot.
        """
        p = self.p
        n = len(self.pos)
        pf = float(p.get("producer_frac", 0.0))
        if pf > 0:
            self.kind[self.rng.random(n) < pf] = 1
            m = self.kind == 1
            self.emit_t[m] = self.rng.uniform(
                0.0, float(p.get("emit_period", 2.0)), int(m.sum()))
            self.head[m] = np.radians(float(p.get("flow_angle", 0.0)))                 + self.rng.normal(0.0, 0.9, int(m.sum()))
        nr = int(p.get("recipients", 0))
        if nr > 0 and n:
            R = float(p.get("seed_radius", 700.0))
            asp = max(0.2, float(p.get("seed_aspect", 1.0)))
            q = np.hypot(self.pos[:, 0] / R, self.pos[:, 1] / (R * asp))
            inner = np.nonzero(q < 0.45)[0]
            if len(inner):
                pick = inner[np.argsort(-self.abase[inner])[:nr * 3]]
                pick = self.rng.permutation(pick)[:nr]
                self.kind[pick] = 3

    def attractors(self):
        """Fixed destinations on the seed rim.

        On the SEED ellipse, not the frame: the sim never sees the render config, and the
        fit tools run at a different aspect than the final. Anything keyed to frame extent
        would make the gated sim and the rendered sim two different simulations.
        """
        na = int(self.p.get("attractors", 0))
        if na <= 0:
            return np.zeros((0, 2))
        R = float(self.p.get("seed_radius", 700.0))
        asp = max(0.2, float(self.p.get("seed_aspect", 1.0)))
        g = np.random.default_rng(int(self.p.get("_seed", 7)) * 104729 + 17)
        a = np.linspace(0, 2 * np.pi, na, endpoint=False) + g.uniform(0, 2 * np.pi)
        return np.stack([np.cos(a) * R * 1.05, np.sin(a) * R * asp * 1.05], 1)

    def _pareto(self, n):
        """Per-cell base-area multiplier, bounded Pareto, mean 1.

        A log-normal spread gives a population where every cell is roughly the same size.
        A Pareto tail gives what a real culture has: mostly small cells and a few much
        larger ones. Normalising by the REALISED mean keeps `cell_area` meaning "the
        average cell", so framing and the fit tools are unaffected by the spread.
        """
        al = float(self.p.get("size_pareto", 0.0))
        if al <= 0:
            return np.ones(n)
        x = np.minimum(self.rng.pareto(al, n) + 1.0, float(self.p.get("size_max", 6.0)))
        return x / max(x.mean(), 1e-6)

    def _divT(self, abase):
        """Cytokinesis duration per cell, proportional to radius."""
        a = np.atleast_1d(abase)
        t0 = float(self.p.get("division_time", 0.0))
        if t0 <= 0:
            return np.full(len(a), 1e-6)
        return t0 * np.sqrt(np.maximum(a, 1e-9) / max(float(self.p["cell_area"]), 1e-9))

    def drag(self):
        """Per-cell drag. Big cells are hard to move, small ones flow between them.

        This is the mass-transfer behaviour: at `size_drag` 1 the drag is Stokes-like, so
        in a jammed sheet the small cells are the ones that find a way through and the
        large ones sit and are flowed around.
        """
        eta = max(1e-3, float(self.p["viscosity"]))
        e = float(self.p.get("size_drag", 0.0))
        if e <= 0:
            return eta
        rref = np.sqrt(max(float(self.p["cell_area"]), 1e-9) / np.pi)
        # DRAG FLOOR. Without it the exponent runs away at the small end: a courier a
        # fifth the reference radius gets 0.19**2.6 = 0.014 of the drag, and the contact
        # forces on it are only about a quarter smaller - so it was flung at 1000 ru/s and
        # its 110 ru/s swim toward its target was noise. Measured: with no floor, 262 of
        # 299 couriers were still bouncing around at t=18 having delivered nothing.
        f = float(self.p.get("drag_floor", 0.12))
        return eta * np.clip((self.radii / rref) ** e, f, None)

    # --- derived -----------------------------------------------------------------------
    @property
    def radii(self):
        return np.sqrt(np.maximum(self.a0, 1e-6) / np.pi)

    def _retess(self):
        """Neighbour graph only - see `tess.polygons_of` for why polygons are not built
        here. `self.polys` is filled lazily by `polygons()`."""
        self._vor, self.nb, self.ncount, self._nreal = voronoi_of(self.pos)
        self._poly_t = None

    def polygons(self):
        """Clipped polygons for the CURRENT state, built at most once per simulation
        instant and cached, because a frame asks for them and nothing else does."""
        # TWO ways the cache goes stale, and BOTH have to be checked.
        #
        # A division appends to `pos` after the step's tessellation, so the graph can be a
        # site short - that is the count check.
        #
        # `_vor is None` is the other, and it is the one that shipped broken. `_remove` and
        # `_append` invalidate by hand because a step that removes k cells and adds k
        # leaves the COUNT unchanged while every index past the removal has moved. But the
        # guard here only compared counts, so on exactly those steps `_retess` was skipped
        # and `polygons_of` was handed a None Voronoi - whose documented fallback is to
        # return each cell as a FULL DISC of radius clip_scale. Every cell in the frame
        # overlapping every neighbour, the film covering the whole frame: one blinding
        # white frame, five to eleven times per clip. Measured: mean luma 90 -> 190 for
        # exactly one frame, polygon areas x3.5, minimum vertex count 4 -> 16 (the disc).
        if self._vor is None or self._nreal != len(self.pos):
            self._retess()
        if self._poly_t != self.t or not self.polys:
            self.polys = self._build_polys()
            self._poly_t = self.t
        return self.polys

    def _build_polys(self):
        """Polygons for drawing. COURIERS ARE NOT SITES IN THIS TESSELLATION.

        A courier crosses the sheet at 130 ru/s - two reference units per frame, five times
        anything else in the picture. As a Voronoi site it carves a moving notch through
        every cell it passes, so each of those cells' walls snap in and out frame by frame
        and a few dozen couriers keep the whole frame twitching. It also produces the
        occasional fan of sliver cells where several couriers bunch.

        A courier is not a member of the epithelium; it is a small cell moving through the
        space between. So it is tessellated OUT and drawn as its own disc. Forces are
        untouched - the neighbour graph the physics uses still includes couriers, so they
        still push, squeeze and deliver exactly as before. This changes only what is drawn.
        """
        r = self.radii * float(self.p["clip_scale"])
        keep = np.nonzero(self.kind != 2)[0]
        out = [None] * len(self.pos)
        if len(keep) >= 4 and len(keep) < len(self.pos):
            vor, _, _, nreal = voronoi_of(self.pos[keep])
            sub = polygons_of(vor, self.pos[keep], r[keep], nreal)
            for i, k in enumerate(keep):
                out[k] = sub[i] if i < len(sub) else np.zeros((0, 2))
            # A TIGHT disc, not the clipping radius the sheet cells use. Drawn at
            # clip_scale a courier is a wide translucent bubble lying across two cells;
            # at its own radius it is a small cell sitting in the gap between them.
            for k in np.nonzero(self.kind == 2)[0]:
                out[k] = disc_poly(self.pos[k], self.radii[k] * 1.05)
            return out
        return polygons_of(self._vor, self.pos, r, self._nreal)

    def _pair_terms(self):
        """Neighbour geometry, once per step. Returns (i, j, d, u) for real neighbours."""
        if len(self.nb) == 0:
            n = len(self.pos)
            return (np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                    np.zeros((0, 2)))
        i = self.nb[:, 0]
        j = self.nb[:, 1]
        dv = self.pos[j] - self.pos[i]
        d = np.hypot(dv[:, 0], dv[:, 1])
        u = dv / np.maximum(d, 1e-9)[:, None]
        return i, j, d, u

    # --- the step ----------------------------------------------------------------------
    def step(self, dt):
        p = self.p
        n = len(self.pos)
        if n == 0:
            return
        self._retess()
        i, j, d, u = self._pair_terms()
        r = self.radii

        F = np.zeros((n, 2))
        press = np.zeros(n)

        if len(i):
            rest = (r[i] + r[j]) * float(p["contact"])
            # COURIERS SQUEEZE. As plain hard discs they jam: measured, a courier parked
            # at a tri-cellular junction where three repulsions cancel and stayed there -
            # its 96 ru/s swim produced 4 ru of wall penetration and no travel at all.
            # Shrinking the contact distance for courier pairs lets it thread the
            # interstices, which is what a small motile cell actually does.
            cq = float(p.get("courier_squeeze", 0.35))
            cr = self.kind == 2
            if cq < 1.0 and cr.any():
                # XOR, not OR. With OR a courier pair's contact distance shrinks too and
                # couriers heading for the same destination stack into one blob instead of
                # queueing - measured as a solid bright clump that the exposure fitter then
                # read as a blown frame and cut the whole scene's gain to 23%.
                rest = rest * np.where(cr[i] ^ cr[j], cq, 1.0)
            over = rest - d                       # > 0 means overlapping

            # REPULSION - linear in overlap, only when overlapping.
            f_rep = np.maximum(over, 0.0) * float(p["repulsion"])
            # ADHESION - linear pull once separated, capped so a distant stray cell is
            # not yanked across the frame at absurd speed.
            gap = np.maximum(-over, 0.0)
            # DIFFERENTIAL ADHESION. A pair of same-culture cells pulls with that culture's
            # own affinity; a pair from different cultures pulls with `cross_affinity`.
            # When cross < homo the two cultures form a sharp, taut interface and refuse to
            # interpenetrate; when cross is comparable they interdigitate and mix.
            adh = np.full(len(i), float(p["adhesion"]))
            # A courier that sticks to the first membrane it meets never gets anywhere.
            if cr.any():
                adh *= np.where(cr[i] | cr[j], float(p.get("courier_adhesion", 0.1)), 1.0)
            aff = self.strain_affinity()
            if len(aff) > 1:
                same = self.strain[i] == self.strain[j]
                adh = np.where(same, adh * aff[self.strain[i]],
                               adh * float(p.get("cross_affinity", 1.0)))
            f_adh = -np.minimum(gap, rest * 0.6) * adh

            f = (f_rep + f_adh)[:, None] * u      # along i -> j; positive pushes j away
            np.add.at(F, i, -f)
            np.add.at(F, j, f)

            # LOCAL PRESSURE, as a DIMENSIONLESS fraction of the touching distance -
            # `overlap / rest`, not the raw overlap.
            #
            # This matters because pressure is compared against `inhibit` and fed to the
            # colour ramp, and raw overlap is in reference units. With cells ~29 ru
            # across, the raw mean pressure measured 46.3 while `inhibit` defaulted to
            # 0.75, so contact inhibition blocked EVERY division and the colony sat at
            # its seed count forever. Worse, it would have silently re-broken itself
            # every time someone changed `cell_area`, because the scale of the number
            # tracked the size of a cell.
            frac = np.maximum(over, 0.0) / np.maximum(rest, 1e-9)
            np.add.at(press, i, frac)
            np.add.at(press, j, frac)

            # FREE DIRECTION - see the module docstring. Near zero in the bulk.
            free = np.zeros((n, 2))
            np.add.at(free, i, -u)
            np.add.at(free, j, u)
            free /= np.maximum(self.ncount, 1)[:, None]
        else:
            free = np.zeros((n, 2))

        self.free = free
        expose = np.hypot(free[:, 0], free[:, 1])       # ~0 interior, ~1 on the rim

        # OUTSIDE PRESSURE - inward, on the rim only.
        #
        # SCALE NOTE. These multipliers are not free constants, they are set by what the
        # force has to BALANCE. Repulsion is `overlap * repulsion`, so with cells ~29 ru
        # across and a few ru of overlap it lands around 10-30. Pressure and tension have
        # to be the same order or they simply win.
        #
        # The first draft used 1000 here and 40 for tension, which produced rim
        # velocities of several hundred reference units per second: the colony imploded
        # within a few steps to 22% of its rest spacing (mean fractional overlap 0.78),
        # every cell was permanently contact-inhibited, and nothing ever divided. It
        # looked like a growth bug and was really a units bug.
        F -= free * (float(p["pressure"]) * 60.0)

        # SURFACE TENSION - Laplacian smoothing weighted by exposure.
        if float(p["tension"]) > 0 and len(i):
            mean_nb = np.zeros((n, 2))
            np.add.at(mean_nb, i, self.pos[j])
            np.add.at(mean_nb, j, self.pos[i])
            mean_nb /= np.maximum(self.ncount, 1)[:, None]
            F += (mean_nb - self.pos) * (float(p["tension"]) * expose)[:, None] * 2.0

        # MOTILITY - a little random walk, so a jammed colony still breathes.
        m = float(p["motility"])
        if m > 0:
            F += self.rng.normal(0.0, m * 60.0, (n, 2))

        self.pressure = press / np.maximum(self.ncount, 1)

        # overdamped integration
        eta = self.drag()
        v = F / (eta[:, None] if np.ndim(eta) else eta)
        # Clamp the step so one pathological force cannot teleport a cell. This is a
        # safety net, not physics: if it ever binds hard the scene is misconfigured, and
        # the dry gate reports the speed so you can see that it did.
        vmax = float(p["cell_area"]) ** 0.5 * 12.0
        sp = np.hypot(v[:, 0], v[:, 1])
        too = sp > vmax
        if too.any():
            v[too] *= (vmax / sp[too])[:, None]
        v = v + self._drive()
        self.pos = self.pos + v * dt

        self._cytokinesis(dt)
        self._grow(dt)
        self.t += dt
        # Everything that CHANGES THE INDEXING happens here, once, at the end of the step.
        self._emit(dt)
        self._expire()
        self._cull()

    def _remove(self, idx):
        """Delete cells. The only operation in this file that can invalidate an INDEX.

        `sister` and `target` hold indices, so both are remapped; anything pointing at a
        deleted row goes to -1. The tessellation is invalidated by hand rather than by the
        length check in `polygons()` - remove k and divide k in the same step and the
        counts match while every index past the removal has shifted, which would paint one
        frame with the wrong cell's colour and size.
        """
        if len(idx) == 0:
            return
        n = len(self.pos)
        keep = np.ones(n, bool)
        keep[idx] = False
        newid = np.full(n, -1, np.int32)
        newid[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        for nm in self.ARRAYS:
            a = getattr(self, nm, None)
            if a is not None and len(a) == n:
                setattr(self, nm, a[keep])
        for nm in ("sister", "target"):
            a = getattr(self, nm)
            ok = a >= 0
            a[ok] = newid[a[ok]]
            a[a < 0] = -1
        self._vor = None
        self._poly_t = None
        self.polys = []

    def _emit(self, dt):
        """Producers bud couriers; couriers deliver or leave."""
        p = self.p
        per = float(p.get("emit_period", 0.0))
        cf = float(p.get("courier_frac", 0.0))
        if per > 0 and cf > 0:
            self._replenish()
            self._bud(per, cf)
        self._deliver()

    def _replenish(self):
        """Keep the producer fraction up.

        Producers crawl off the edge and are culled, and nothing was replacing them:
        measured, vitreous_lanes ended an 18 s clip with 5 producers of the 30 it started
        with, and its delivery traffic died with them. Promoting ordinary cells at the rate
        the boundary takes them IS the larger culture off screen - it is the only place new
        producers could be coming from.
        """
        pf = float(self.p.get("producer_frac", 0.0))
        if pf <= 0:
            return
        body = self.kind != 2
        want = int(pf * body.sum())
        have = int((self.kind == 1).sum())
        if have >= want:
            return
        pool = np.nonzero(self.kind == 0)[0]
        if len(pool) == 0:
            return
        k = min(want - have, len(pool), 4)
        pick = self.rng.choice(pool, k, replace=False)
        self.kind[pick] = 1
        self.emit_t[pick] = self.t + float(self.p.get("emit_period", 2.0))             * self.rng.uniform(0.2, 1.0, k)
        # Headings biased along the bulk flow, so producers leave the way the sheet moves
        # rather than milling about.
        self.head[pick] = np.radians(float(self.p.get("flow_angle", 0.0)))             + self.rng.normal(0.0, 0.9, k)

    def _bud(self, per, cf):
        p = self.p
        due = np.nonzero((self.kind == 1) & (self.t >= self.emit_t))[0]
        if len(due) == 0:
            return
        self.emit_t[due] = self.t + per * self.rng.uniform(0.6, 1.5, len(due))
        room = int(p["max_cells"]) - len(self.pos)
        if room <= 0:
            return
        area = self.abase[due] * cf
        # A producer only buds what it can spare, and the area comes OUT of it, so one that
        # keeps emitting visibly shrinks instead of manufacturing mass.
        due = due[self.a0[due] > area * 4.0]
        if len(due) == 0:
            return
        due = due[:room]
        area = self.abase[due] * cf
        k = len(due)

        ang = self.rng.uniform(0, 2 * np.pi, k)
        u = np.stack([np.cos(ang), np.sin(ang)], 1)
        pos = self.pos[due] + u * (self.radii[due] * 1.05)[:, None]
        self.a0[due] -= area

        att = self.attractors()
        off = self.rng.random(k) < float(p.get("courier_offscreen", 0.5))
        tgt = np.full(k, -1, np.int32)
        atr = np.full(k, -1, np.int32)
        if len(att):
            atr[off] = self.rng.integers(0, len(att), int(off.sum()))
        recip = np.nonzero(self.kind == 3)[0]
        want = ~off if len(att) else np.ones(k, bool)
        if len(recip):
            tgt[want] = recip[self.rng.integers(0, len(recip), int(want.sum()))]
        elif len(att):
            atr[want] = self.rng.integers(0, len(att), int(want.sum()))

        self._append(k, pos=pos, abase=area, a0=area, kind=2, target=tgt, attr=atr,
                     strain=self.strain[due], line=self.line[due],
                     div_t=self.t, org=self.org[due], cval=self.cval[due])

    def _deliver(self):
        """Absorb couriers that reached their cell, and cull the ones that are done."""
        p = self.p
        cour = np.nonzero(self.kind == 2)[0]
        if len(cour) == 0:
            return
        drop = []
        tg = self.target[cour]
        has = tg >= 0
        if has.any():
            a, b = cour[has], tg[has]
            d = np.hypot(*(self.pos[b] - self.pos[a]).T)
            hit = d < (self.radii[a] + self.radii[b]) * 0.9
            if hit.any():
                # The delivered area goes into BOTH a0 and abase, so a fed cell grows
                # rather than being pushed straight over its division threshold.
                cap = float(p["cell_area"]) * float(p.get("size_max", 6.0))
                np.add.at(self.abase, b[hit], self.a0[a[hit]])
                np.add.at(self.a0, b[hit], self.a0[a[hit]])
                np.minimum(self.abase, cap, out=self.abase)
                np.minimum(self.a0, cap * 1.2, out=self.a0)
                self.div_t[b[hit]] = self.t          # flash the recipient
                self.divT[b[hit]] = np.maximum(self.divT[b[hit]], 0.4)
                drop.append(a[hit])
        att = self.attractors()
        if len(att):
            aa = self.attr[cour]
            hasa = aa >= 0
            if hasa.any():
                a = cour[hasa]
                d = np.hypot(*(att[aa[hasa]] - self.pos[a]).T)
                gone = d < self.radii[a] * 3.0
                if gone.any():
                    drop.append(a[gone])
        if drop:
            self._remove(np.unique(np.concatenate(drop)))

    def _expire(self):
        """Couriers have a lifetime. Without one the population is a race between the bud
        rate and the travel time: measured, 333 created and 55 removed over 18 s, which
        walks into `max_cells` and stops every division in the scene."""
        life = float(self.p.get("courier_life", 0.0))
        if life <= 0:
            return
        old_ = np.nonzero((self.kind == 2) & (self.t - self.div_t > life))[0]
        if len(old_):
            self._remove(old_)

    def _cull(self):
        """Bound the population: anything past the seeded region is off stage for good."""
        m = float(self.p.get("cull_radius", 0.0))
        if m <= 0 or len(self.pos) == 0:
            return
        R = float(self.p.get("seed_radius", 700.0))
        asp = max(0.2, float(self.p.get("seed_aspect", 1.0)))
        q = np.hypot(self.pos[:, 0] / R, self.pos[:, 1] / (R * asp))
        out = np.nonzero(q > m)[0]
        if len(out):
            self._remove(out)

    def _append(self, k, **cols):
        """Add k cells. Any array not named gets the neutral value for its column."""
        n = len(self.pos)
        d = {
            "pos": np.zeros((k, 2)), "a0": np.full(k, float(self.p["cell_area"])),
            "age": np.zeros(k), "gen": np.zeros(k, np.int32),
            "line": np.zeros(k, np.int32), "rate": np.ones(k),
            "pressure": np.zeros(k), "free": np.zeros((k, 2)),
            "ncount": np.zeros(k, int), "strain": np.zeros(k, np.int32),
            "nuc": self.rng.random(k).astype(np.float32),
            "sister": np.full(k, -1, np.int32), "div_t": np.full(k, -1e9),
            "abase": np.full(k, float(self.p["cell_area"])),
            "divT": np.full(k, 1e-6), "cval": np.full(k, -1.0),
            "kind": np.zeros(k, np.int8), "target": np.full(k, -1, np.int32),
            "attr": np.full(k, -1, np.int32), "emit_t": np.full(k, 1e9),
            "org": self.rng.integers(1, 2 ** 31, k).astype(np.int64),
            "head": self.rng.uniform(0, 2 * np.pi, k),
            "uid": np.arange(self._next_uid, self._next_uid + k, dtype=np.int64),
        }
        self._next_uid += k
        for key, v in cols.items():
            d[key] = np.broadcast_to(np.asarray(v), d[key].shape).copy() \
                if np.ndim(v) == 0 else np.asarray(v)
        for nm in self.ARRAYS:
            cur = getattr(self, nm)
            add = d[nm]
            if cur.ndim == 2:
                setattr(self, nm, np.vstack([cur, add.reshape(k, 2)]))
            else:
                setattr(self, nm, np.concatenate([cur, np.asarray(add).reshape(k)
                                                  .astype(cur.dtype)]))
        self._vor = None
        self._poly_t = None
        return np.arange(n, n + k)

    def _cytokinesis(self, dt):
        """Walk each dividing pair apart over `divT` seconds.

        Division used to place the daughters 0.28r apart in one step, which is a pop: one
        cell was there, then two were. Here they start almost coincident, so the Voronoi
        wall between them IS the cleavage furrow, and this relaxes their separation toward
        the touching distance over the cell's own cytokinesis time - the furrow pinches in
        and the pair rounds up.

        A POSITION relaxation, not a stiff spring. A spring's stability bound scales with
        1/drag, and `size_drag` gives small cells small drag, so the stiff version would be
        least stable exactly where it is least visible. err * (1 - exp(-dt/tau)) cannot
        overshoot at any dt.
        """
        if len(self.pos) == 0:
            return
        sis = self.sister
        k = np.nonzero((sis >= 0) & (sis < len(self.pos))
                       & (self.t - self.div_t < self.divT))[0]
        if len(k) == 0:
            return
        k = k[k < sis[k]]
        if len(k) == 0:
            return
        j = sis[k]
        u = np.clip((self.t - self.div_t[k]) / np.maximum(self.divT[k], 1e-9), 0.0, 1.0)
        r = self.radii
        want = (r[k] + r[j]) * float(self.p["contact"]) * (0.12 + 0.88 * u ** 0.8)
        d = self.pos[j] - self.pos[k]
        L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        rate = 1.0 - np.exp(-dt / np.maximum(self.divT[k] * 0.14, 1e-6))
        off = (d / L[:, None]) * ((want - L) * 0.5 * rate)[:, None]
        np.subtract.at(self.pos, k, off)
        np.add.at(self.pos, j, off)

    def _drive(self):
        """Velocities that are PRESCRIBED rather than solved: bulk flow, the producer
        crawl, and a courier's pull toward its destination.

        Added to the velocity, not to the force. A courier is tiny, so its drag is tiny,
        and a force large enough to move it against a wall of big cells is large enough to
        make the step size the thing deciding where it ends up.
        """
        n = len(self.pos)
        if n == 0:
            return np.zeros((0, 2))
        p = self.p
        out = np.zeros((n, 2))

        fs = float(p.get("flow_speed", 0.0))
        if fs != 0:
            a = np.radians(float(p.get("flow_angle", 0.0)))
            d = np.array([np.cos(a), np.sin(a)])
            sh = float(p.get("flow_shear", 0.0))
            per = np.array([-d[1], d[0]])
            across = self.pos @ per
            out += d * (fs + sh * across / max(float(p.get("seed_radius", 700.0)), 1e-6))[:, None]

        ps_ = float(p.get("producer_speed", 0.0))
        if ps_ > 0:
            m = self.kind == 1
            if m.any():
                h = self.head[m]
                out[m] += np.stack([np.cos(h), np.sin(h)], 1) * ps_

        cs = float(p.get("courier_speed", 0.0))
        if cs > 0:
            m = self.kind == 2
            if m.any():
                idx = np.nonzero(m)[0]
                dest = np.zeros((len(idx), 2))
                tg = self.target[idx]
                ok = tg >= 0
                if ok.any():
                    dest[ok] = self.pos[tg[ok]]
                att = self.attractors()
                aa = self.attr[idx]
                oka = (~ok) & (aa >= 0) & (len(att) > 0)
                if oka.any():
                    dest[oka] = att[aa[oka]]
                live = ok | oka
                if live.any():
                    # Aim slightly off. Every courier steering at the same point converges
                    # to a queue; a per-courier offset spreads the approach into a lane.
                    jit = float(p.get("courier_spread", 0.0))
                    if jit > 0:
                        a = (self.org[idx] % np.int64(6280)).astype(np.float64) / 1000.0
                        dest = dest + np.stack([np.cos(a), np.sin(a)], 1) * jit
                    dv = dest[live] - self.pos[idx[live]]
                    L = np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-9)
                    out[idx[live]] += dv / L[:, None] * cs
        return out

    # --- growth and division ---------------------------------------------------------
    def _grow(self, dt):
        p = self.p
        self.age += dt
        g = float(p["growth_rate"])
        if g > 0:
            # Exponential in the TARGET area. Crowded cells grow more slowly, which is
            # what makes the interior of a big colony go quiet while the rim keeps going.
            slow = 1.0 / (1.0 + self.pressure * float(p["inhibit_growth"]))
            # PER-STRAIN growth. This is what makes clusters COMPETE rather than merely
            # coexist: a faster strain claims territory, and its neighbours feel it as
            # pressure, which inhibits them further. The asymmetry compounds on its own.
            self.a0 = self.a0 * np.exp(g * self.rate * self.strain_mul() * slow * dt)
            # GROWTH ARREST. A cell that has reached its division size but is too crowded
            # to divide stops growing - it does not keep inflating forever.
            #
            # This is real (contact-inhibited cells arrest in G1) and it is also load
            # bearing here. Without it, cells blocked by contact inhibition kept growing
            # exponentially: measured a 90th-percentile cell area of 4.5x the base while
            # the 10th percentile sat at 1.0x, which is not a living tissue, it is a few
            # balloons among normal cells - and it dragged edge saturation down to 0.39
            # and blue up to 5.3%.
            self.a0 = np.minimum(self.a0, self.abase * float(p["divide_at"]) * 1.12)

        cap = self.abase * float(p["divide_at"])
        ready = ((self.a0 > cap)
                 & (self.kind != 2)
                 & (self.age > float(p["cycle_min"]))
                 & (self.t - self.div_t >= self.divT)
                 & (self.pressure < float(p["inhibit"])))
        if not ready.any() or len(self.pos) >= int(p["max_cells"]):
            return
        idx = np.nonzero(ready)[0]
        room = int(p["max_cells"]) - len(self.pos)
        if len(idx) > room:
            idx = idx[:room]
        self._divide(idx)

    def _divide(self, idx):
        """Split each named cell into two, along a random axis through its centre.

        The daughters are offset by a quarter of the parent's radius each way, which is
        enough for the tessellation to separate them immediately without the pair
        potential launching them apart.
        """
        p = self.p
        k = len(idx)
        ang = self.rng.uniform(0, 2 * np.pi, k)
        axis = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        r = self.radii[idx][:, None]
        # ALMOST COINCIDENT. `_cytokinesis` walks them apart over divT. A large offset
        # here is what made division instantaneous.
        off = axis * r * 0.05

        half = self.a0[idx] * 0.5
        jitter = self.rng.uniform(0.92, 1.08, k)

        new_pos = self.pos[idx] + off
        self.pos[idx] = self.pos[idx] - off
        self.a0[idx] = half * jitter
        self.age[idx] = 0.0
        self.gen[idx] += 1

        self.pos = np.vstack([self.pos, new_pos])
        self.a0 = np.concatenate([self.a0, half * (2.0 - jitter)])
        self.age = np.concatenate([self.age, np.zeros(k)])
        self.gen = np.concatenate([self.gen, self.gen[idx]])
        # Lineage: one daughter keeps the tag, the other mutates it. With 8 tags a big
        # colony shows lineage DOMAINS - contiguous patches of common descent - which is
        # the real and rather beautiful thing lineage tracing actually shows.
        mutate = self.rng.random(k) < float(p["lineage_split"])
        newline = np.where(mutate, self.rng.integers(0, 8, k), self.line[idx])
        self.line = np.concatenate([self.line, newline.astype(np.int32)])
        # Daughters inherit the parent's growth rate with a little drift, so lineages
        # stay recognisably fast or slow instead of re-randomising every generation.
        h = float(p.get("heterogeneity", 0.35))
        drift = self.rng.lognormal(0.0, h * 0.45, k) if h > 0 else np.ones(k)
        self.rate[idx] = np.clip(self.rate[idx] * drift, 0.15, 6.0)
        self.rate = np.concatenate([self.rate, np.clip(self.rate[idx] / drift, 0.15, 6.0)])

        # EXTEND THE DERIVED ARRAYS TOO. Daughters inherit the parent's row.
        #
        # `pressure`, `free` and `ncount` are computed in `step` BEFORE `_grow` runs, so
        # the moment a division appends a cell they are one row shorter than `pos`. The
        # next step's `_retess` would fix them - but `draw` runs in between, and it
        # indexes both, so the frame right after a division raised
        #
        #     ValueError: operands could not be broadcast together with shapes (27,) (28,)
        #
        # It is not a threading bug and it is not confined to the live view: a render
        # crashes the same way whenever a division lands on the last substep before a
        # frame, which is a coin toss you would lose on a long clip. Padding here keeps
        # every per-cell array the same length at all times, which is the invariant the
        # dry gate now also checks.
        self.pressure = np.concatenate([self.pressure, self.pressure[idx]])
        self.free = np.vstack([self.free, self.free[idx]])
        self.ncount = np.concatenate([self.ncount, self.ncount[idx]])

        # Strain is inherited exactly - a cell never changes species.
        self.strain = np.concatenate([self.strain, self.strain[idx]])
        # Daughters inherit their parent's SIZE with a little drift, so a Pareto tail
        # becomes lineages of large and small cells instead of a fresh lottery each
        # generation.
        sd = float(p.get("size_drift", 0.06))
        dr = self.rng.lognormal(0.0, sd, k) if sd > 0 else np.ones(k)
        cap_a = float(p["cell_area"]) * float(p.get("size_max", 6.0))
        self.abase[idx] = np.clip(self.abase[idx] * dr, 1e-6, cap_a)
        self.abase = np.concatenate([self.abase,
                                     np.clip(self.abase[idx] / dr, 1e-6, cap_a)])
        self.divT = np.concatenate([self.divT, self._divT(self.abase[len(self.abase) - k:])])
        self.divT[idx] = self._divT(self.abase[idx])
        self.cval = np.concatenate([self.cval, self.cval[idx]])
        # A recipient's daughters: one keeps the role, the other is an ordinary cell, or
        # the number of cells being fed doubles every generation.
        self.kind = np.concatenate([self.kind, np.where(self.kind[idx] == 3, 0,
                                                        self.kind[idx])]).astype(np.int8)
        self.target = np.concatenate([self.target, np.full(k, -1, np.int32)])
        self.attr = np.concatenate([self.attr, np.full(k, -1, np.int32)])
        et = np.where(self.kind[idx] == 1,
                      self.t + float(p.get("emit_period", 2.0))
                      * self.rng.uniform(0.4, 1.2, k), 1e9)
        self.emit_t = np.concatenate([self.emit_t, et])
        # FRESH organelle seeds on BOTH daughters - a shared one smears the pattern
        # across the cytokinesis, which is exactly what it must not do.
        self.org[idx] = self.rng.integers(1, 2 ** 31, k)
        self.org = np.concatenate([self.org, self.rng.integers(1, 2 ** 31, k)])
        self.head = np.concatenate([self.head, self.rng.uniform(0, 2 * np.pi, k)])
        self.uid = np.concatenate([self.uid, np.arange(self._next_uid, self._next_uid + k,
                                                       dtype=np.int64)])
        self._next_uid += k

        # NUCLEUS IDENTITY drifts a little between sisters, so a split is visible as one
        # point becoming two slightly different points rather than one point widening.
        drift_n = self.rng.normal(0.0, float(p.get("nucleus_drift", 0.06)), k)
        self.nuc = np.concatenate([self.nuc,
                                   np.mod(self.nuc[idx] + drift_n, 1.0)]).astype(np.float32)

        # SISTERHOOD, and when it happened. `_draw` uses these to put a bright bridge
        # between the two daughters for a moment after they separate, which is what makes
        # a division read as an EVENT instead of as a cell quietly appearing.
        n_before = len(self.sister)
        new_ids = np.arange(n_before, n_before + k, dtype=np.int32)
        self.sister = np.concatenate([self.sister, idx.astype(np.int32)])
        self.sister[idx] = new_ids
        self.div_t = np.concatenate([self.div_t, np.full(k, self.t)])
        self.div_t[idx] = self.t

    # --- readouts ----------------------------------------------------------------------
    def strain_sizes(self, ns=None):
        """Per-strain cell-size multiplier. Deterministic from the scene seed."""
        ns = ns if ns is not None else (int(self.strain.max()) + 1 if len(self.strain) else 1)
        sp = float(self.p.get("strain_size_spread", 0.0))
        if sp <= 0 or ns <= 1:
            return np.ones(max(1, ns))
        g = np.random.default_rng(int(self.p.get("_seed", 7)) * 5701 + 29)
        return np.exp(g.normal(0.0, sp, ns))

    def strain_affinity(self, ns=None):
        """Per-strain HOMOTYPIC adhesion multiplier - how strongly a culture holds on to
        its own kind. Cultures with high affinity round up and refuse to mix; low-affinity
        ones spread and interdigitate. This is differential adhesion, and it is what makes
        two cultures sort out rather than blend."""
        ns = ns if ns is not None else (int(self.strain.max()) + 1 if len(self.strain) else 1)
        sp = float(self.p.get("affinity_spread", 0.0))
        if sp <= 0 or ns <= 1:
            return np.ones(max(1, ns))
        g = np.random.default_rng(int(self.p.get("_seed", 7)) * 3301 + 71)
        return np.exp(g.normal(0.0, sp, ns))

    def strain_mul(self):
        """Per-cell growth multiplier from its strain. 1.0 everywhere for a single
        cluster, so a one-cluster scene is unaffected by any of this."""
        spread = float(self.p.get("strain_growth_spread", 0.0))
        if spread <= 0:
            return 1.0
        ns = int(self.strain.max()) + 1 if len(self.strain) else 1
        if ns <= 1:
            return 1.0
        # Deterministic per strain, symmetric about 1, from a seeded generator so the
        # same scene always produces the same winner.
        g = np.random.default_rng(int(self.p.get("_seed", 7)) * 7919 + 13)
        muls = np.exp(g.normal(0.0, spread, ns))
        return muls[self.strain]

    def areas(self):
        return np.array([polygon_area(q) for q in self.polygons()])

    def state_arrays(self):
        return {"pos": self.pos, "a0": self.a0, "age": self.age, "rate": self.rate,
                "pressure": self.pressure, "free": self.free, "nuc": self.nuc}

    def check_invariants(self):
        """Every per-cell array must have the same length. Returns a list of complaints.

        Called by the dry gate. This exists because a length mismatch here does not fail
        where it is caused - it fails later, in drawing, with a numpy broadcast error
        that names two numbers and no variable."""
        n = len(self.pos)
        bad = []
        for nm in self.ARRAYS[1:]:
            a = getattr(self, nm, None)
            if a is not None and len(a) != n:
                bad.append("colony.%s has %d rows, expected %d" % (nm, len(a), n))
        for nm in ("sister", "target"):
            a = getattr(self, nm, None)
            if a is not None and len(a) and int(a.max()) >= n:
                bad.append("colony.%s points at cell %d of %d" % (nm, int(a.max()), n))
        return bad

    def counts(self):
        return {"normal": int((self.kind == 0).sum()),
                "producers": int((self.kind == 1).sum()),
                "couriers": int((self.kind == 2).sum()),
                "recipients": int((self.kind == 3).sum())}

    def summary(self):
        strains = int(self.strain.max()) + 1 if len(self.strain) else 0
        return {"cells": int(len(self.pos)),
                "strains": strains,
                "mean_neighbours": round(float(self.ncount.mean()), 2) if len(self.ncount) else 0,
                "mean_pressure": round(float(self.pressure.mean()), 3) if len(self.pressure) else 0,
                "generation": int(self.gen.max()) if len(self.gen) else 0}
