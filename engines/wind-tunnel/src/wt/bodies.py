"""bodies.py â€” FREE bodies: objects the flow actually pushes around.

Everything in `shapes.py` is placed by the scene at a position the scene chose. A `FreeBody` is
placed by the *fluid*: each LBM step it reads the momentum the flow handed to its surface
(`LBM.force_field`, the same momentum-exchange sum that gives Cd/Cl), adds gravity and buoyancy,
and integrates. Nothing about the trajectory is keyframed â€” a falling disc genuinely slows as
drag builds, drifts when a shed vortex passes, and tumbles because the torque about its own
centre is non-zero.

Two coupling modes, because one integrator cannot cover both cases:

  `dynamic` â€” full Newtonâ€“Euler. Correct for anything appreciably denser than the fluid.
  `tracer`  â€” the body is carried at the local fluid velocity and spun at half the local
              vorticity (the fluid's solid-body rotation rate). This is what a *neutrally
              buoyant* object does, and it is used deliberately: explicit two-way coupling
              becomes unstable as the density ratio approaches 1, because the fluid the body
              accelerates (its added mass) is comparable to the body's own mass, and each side
              over-corrects the other. Advecting is both unconditionally stable and, at that
              density, physically right.

Sizing bodies by TARGET FALL SPEED
----------------------------------
`density_for_fall` picks a body's density from the terminal-velocity balance
(weight âˆ’ buoyancy = drag) so that a given shape at a given size settles at a chosen speed.
This is choosing the object's *material*, not scripting its path: once released, the motion is
entirely the solver's. Without it, sizing a set of discs by eye gives wildly different speeds â€”
drag grows with d but weight grows with dÂ², so big discs plummet while small ones hang.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import shapes
from .gpu import asnumpy, xp

# Drag coefficients used ONLY to choose a body's density up front â€” never in the solve.
# These are EFFECTIVE values for this tunnel, not textbook free-stream ones: a body occupying
# ~15-20% of the span in a slip-walled channel drags roughly twice its unconfined figure
# (measured Cd ~ 2.8 for a cylinder here against a textbook ~1.2). Calibrating against the real
# configuration is what makes the resulting fall speed land near the target.
CD_HINT = {"circle": 2.7, "ellipse": 1.8, "teardrop": 1.0, "square": 2.9,
           "plate": 3.0, "wedge": 2.2, "wedge_rev": 3.2, "naca": 1.9,
           # A surfboard PLANFORM spends most of a sweep at real incidence, so the figure that
           # matters is not its 0-deg one - it sits between an ellipse and a plate. This started
           # as an estimate of 2.2 and was CORRECTED from a measured run (AGENT_GUIDE 2a.21:
           # invert the measured ratio, do not bisect); see `SurfSweep.U_REF`.
           "surfboard": 2.4}


def _ring(mask):
    """The one-cell shell of FLUID cells hugging a body â€” where its wall force is deposited."""
    d = mask.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        d |= np.roll(mask, sh, axis=ax)
    for sy in (1, -1):
        for sx in (1, -1):
            d |= np.roll(np.roll(mask, sy, axis=0), sx, axis=1)
    return d & ~mask


def droplet(blobs, area, iters=3):
    """`shapes.metaball`, with the blob radii scaled so the outline encloses exactly `area`.

    The raw metaball union of two blobs that are merely NEAR each other is 27% fatter than the
    sum of their circles - all of it in the neck. Handing that to the solver in one step turns
    ~500 cells of fluid solid at once, which is the "nothing may materialise inside moving fluid"
    failure (AGENT_GUIDE 2a.7) in miniature, and it makes the droplet's mass balloon and then
    deflate as it fuses.

    Rescaling the blobs to hold the area fixed removes both problems, and it is what a real pair
    of coalescing droplets does anyway: the neck is not new liquid, it is drawn out of the two
    drops, which visibly shrink as it forms. Volume is then conserved at every instant of the
    merge rather than only at the ends. Area goes roughly as s^2 at fixed separation, so the
    fixed-point iteration converges in two or three passes.
    """
    s = 1.0
    for _ in range(int(iters)):
        p = shapes.metaball([(x, y, r * s) for x, y, r in blobs])
        a = polygon_area(p)
        if a <= 1e-9:
            break
        s *= float(np.sqrt(area / a))
    return shapes.metaball([(x, y, r * s) for x, y, r in blobs])


def smoothstep01(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def polygon_area(p):
    x, y = np.asarray(p)[:, 0], np.asarray(p)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def density_for_fall(profile_name, size, v_fall, u0, g, unit_area):
    """Density ratio giving terminal fall speed `v_fall` against an upward stream `u0`.

    (rho_b - 1) * A * g  =  1/2 * Cd * (u0 + v_fall)^2 * d      [weight-buoyancy = drag]
    """
    cd = CD_HINT.get(profile_name, 1.2)
    area = unit_area * size * size
    drag = 0.5 * cd * (u0 + v_fall) ** 2 * size
    return 1.0 + drag / max(area * g, 1e-12)


class FreeBody:
    def __init__(self, profile, size, x, y, *, name="circle", ang=0.0, vx=0.0, vy=0.0,
                 omega=0.0, density=2.0, mode="dynamic", pivot=0.5, born=0.0,
                 on_exit="clamp", spin=None):
        self.name, self.size, self.pivot, self.mode = name, float(size), pivot, mode
        self.x, self.y, self.ang = float(x), float(y), float(ang)
        self.vx, self.vy, self.omega = float(vx), float(vy), float(omega)
        self.density, self.born, self.alive = float(density), float(born), True
        # what happens when it reaches the top: "clamp" (rest against it), "kill" (leave the
        # tunnel for good), "recycle" (re-enter at the bottom — keeps a drift scene populated)
        self.on_exit = on_exit
        # PRESCRIBED rotation rate, or None for the usual fluid-driven one. A body with `spin`
        # set is an ACTUATOR, in the same sense as the foil whose incidence `AoASweep`
        # prescribes: its rotation is an INPUT to the problem, and everything the fluid does in
        # response is still entirely the solver's answer. `black_holes` uses it - the spin is
        # what drives each hole's disc, so it cannot itself be an output of the flow.
        self.spin = None if spin is None else float(spin)
        self.set_profile(profile)
        # low-passed fluid load - see `integrate`
        self._fx = self._fy = self._tq = 0.0
        self.smooth = 0.04
        # EXTERNAL force from an agency outside the fluid, in the same units as the fluid load.
        # This is the same class of input as a prescribed position or a prescribed spin (CLAUDE.md
        # section 1): a rig holding a foil, or a RIDER holding a surfboard on the wave. It is not
        # a fudge for containment - a scene that uses it owns saying so, and a free-body diagram
        # that draws the other forces must draw this one too or it does not add up.
        self.ext = (0.0, 0.0)

    def set_profile(self, profile):
        """(Re)derive every inertial property from the outline.

        Split out of `__init__` because a coalescing droplet's shape genuinely CHANGES: its mass
        must track its area or a merged body keeps weighing whatever its parents did before the
        neck filled in, and its inertia and circumradius feed the rotation cap and the contact
        tests. Density, size and pivot describe the material and the frame, so a reshape leaves
        them alone.
        """
        self.prof = np.asarray(profile, dtype=np.float64)
        self.unit_area = polygon_area(self.prof)
        self.area = self.unit_area * self.size ** 2
        self.mass = self.density * self.area
        # radius of gyration of the (unit) profile about its pivot, by direct sampling â€” the
        # polygon may be any shape, so an analytic formula per profile is not worth having
        pts = self.prof - np.array([self.pivot, 0.0])
        self.inertia = self.mass * float(np.mean((pts ** 2).sum(1))) * self.size ** 2 * 0.5
        # circumradius in lattice cells, for contact resolution
        self.radius = float(np.hypot(*pts.T).max()) * self.size

    # -- geometry ---------------------------------------------------------------------
    def polygon(self):
        return shapes.place(self.prof, self.size, self.x, self.y,
                            np.degrees(self.ang), pivot=self.pivot)

    def local_mask(self, nx, ny, supersample=2, pad=3):
        """Rasterise into the body's own BOUNDING BOX, not the whole lattice.

        With a handful of bodies re-rasterised every LBM step, a full-domain polygon fill per
        body dominates the frame time; a 60x60 fill does not. The box is padded so the ring of
        FLUID cells just outside the body â€” where the wall force actually lives â€” fits inside it.
        Returns (mask, x0, y0) or None if entirely off-lattice.
        """
        p = self.polygon()
        x0 = int(np.floor(p[:, 0].min())) - pad
        y0 = int(np.floor(p[:, 1].min())) - pad
        x1 = int(np.ceil(p[:, 0].max())) + pad + 1
        y1 = int(np.ceil(p[:, 1].max())) + pad + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, nx), min(y1, ny)
        if x1 <= x0 or y1 <= y0:
            return None
        s = max(1, int(supersample))
        w, h = (x1 - x0), (y1 - y0)
        img = Image.new("L", (w * s, h * s), 0)
        ImageDraw.Draw(img).polygon(
            [((px - x0) * s, (py - y0) * s) for px, py in p], fill=255)
        a = np.asarray(img, np.uint8)
        if s > 1:
            a = a.reshape(h, s, w, s).mean(axis=(1, 3))
        return a >= 128, x0, y0

    # -- dynamics ----------------------------------------------------------------------
    def integrate(self, fx, fy, torque, g_vec, dt, vmax=0.10):
        """One Newton-Euler step from the fluid force/torque already summed over this body."""
        # LOW-PASS the raw per-step load first. A sharp-interface moving boundary re-rasterises
        # onto whole cells, so as a body creeps forward its surface gains and loses links in
        # discrete jumps and the instantaneous force spikes by an order of magnitude - the
        # standard "staircase noise" of moving boundaries in LBM. The physical load varies over
        # many steps, so filtering over ~25 of them removes the artefact and nothing else;
        # without it a disc periodically gets flicked clear across the tunnel.
        a = self.smooth
        self._fx += a * (fx - self._fx)
        self._fy += a * (fy - self._fy)
        self._tq += a * (torque - self._tq)
        fx, fy, torque = self._fx, self._fy, self._tq

        gx, gy = g_vec
        # net gravity = (body - displaced fluid) * g  â€” buoyancy is not optional at these ratios
        wx = (self.density - 1.0) * self.area * gx
        wy = (self.density - 1.0) * self.area * gy
        self.vx += (fx + wx + self.ext[0]) / self.mass * dt
        self.vy += (fy + wy + self.ext[1]) / self.mass * dt
        sp = np.hypot(self.vx, self.vy)
        if sp > vmax:                       # a runaway body would tear the lattice apart
            self.vx *= vmax / sp
            self.vy *= vmax / sp
        # Cap the SURFACE speed, not omega. A fixed omega limit means nothing on its own: at a
        # 22-cell radius, omega = 0.05 puts the rim at over 1.0 lattice units, i.e. supersonic on
        # this lattice. Residual torque noise random-walks omega upward with little to damp it,
        # and the body ends up spinning fluid up at its own boundary until the solve dies - which
        # is exactly where every remaining blow-up was located.
        if self.spin is None:
            w_max = 0.02 / max(self.radius, 1.0)
            self.omega = float(np.clip((self.omega + torque / self.inertia * dt) * 0.999,
                                       -w_max, w_max))
        else:
            # Prescribed: the body is being turned, so the fluid torque does not get a vote.
            # The scene owns the surface-speed budget here instead of the w_max cap - it must
            # keep |spin| * radius under the moving-wall limit itself (see set_wall_velocity).
            self.omega = self.spin
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ang += self.omega * dt

    def advect(self, ux, uy, vort, dt):
        """Tracer mode: ride the flow, spin at half the local vorticity.

        The spin is capped by SURFACE speed for the same reason as in `integrate`: half the
        vorticity in a shear layer is a perfectly ordinary number, but multiplied by a 20-cell
        radius it puts the rim far past the lattice's speed of sound, and the drifter starts
        driving the fluid instead of following it.
        """
        w_max = 0.02 / max(self.radius, 1.0)
        self.vx, self.vy = float(ux), float(uy)
        self.omega = float(np.clip(0.5 * float(vort), -w_max, w_max))
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ang += self.omega * dt


class BodySystem:
    """Owns the free bodies, the per-step coupling, and the combined solid mask."""

    def __init__(self, nx, ny, g=5e-4, floor=0.055, ceiling=0.985, wall_gap=7.0,
                 ybounds=None):
        self.nx, self.ny = int(nx), int(ny)
        self.g = float(g)
        self.floor, self.ceiling = float(floor), float(ceiling)
        self.wall_gap = float(wall_gap)
        # Cross-stream limits in CELLS, (lo, hi). None = the lattice edges, less `wall_gap`.
        # A scene running OVERSCAN wants these set to the visible window: the real free-slip
        # walls are then off camera, and a body cannot wander out of shot and leave the frame
        # emptier than it was composed to be.
        self.ybounds = None if ybounds is None else (float(ybounds[0]), float(ybounds[1]))
        self._rng = np.random.default_rng(12345)      # respawn positions only
        # Optional CENTRAL gravity (cx, cy, G, softening) instead of a uniform field: acceleration
        # G*r/(r^2+soft^2)^1.5 toward the centre, i.e. Newtonian far out and finite at the middle.
        # Softening matters — an unsoftened point mass gives an infinite kick to anything that
        # wanders into the centre cell.
        self.central = None
        # MUTUAL attraction between the bodies themselves: a = pair_g * m_other / r^2, softened
        # by their radii so it stays finite as two of them close on each other. Unlike `central`
        # there is no fixed centre - where they end up is decided by the three-body problem plus
        # whatever the fluid is doing to them, which at these Reynolds numbers is a lot.
        self.pair_g = 0.0
        # LIQUID COALESCENCE: when two bodies come within `merge_gap`, they are replaced by one
        # droplet whose outline is their metaball union, and the neck closes over `merge_time`
        # seconds. Off by default - the other scenes want bodies that bounce, not fuse.
        self.merge = False
        self.merge_gap = 9.0
        self.merge_time = 1.2
        self.on_merge = None            # optional scene callback(new_body, a, b)
        self.merges = 0
        self.bodies = []
        self._masks = []
        self.wux = np.zeros((self.ny, self.nx), np.float32)
        self.wuy = np.zeros((self.ny, self.nx), np.float32)

    def add(self, body):
        self.bodies.append(body)
        return body

    def active(self, t):
        return [b for b in self.bodies if b.alive and t >= b.born]

    def build_masks(self, t):
        """Per-body local masks, the combined lattice mask, and the WALL VELOCITY field.

        The wall-velocity field is what lets the solver bounce off a *moving* surface: at each
        solid cell it is the body's rigid-body velocity there, v + omega x r. Without it the
        boundary behaves as if the body were nailed in place (see LBM.set_wall_velocity).
        """
        full = np.zeros((self.ny, self.nx), bool)
        self.wux = np.zeros((self.ny, self.nx), np.float32)
        self.wuy = np.zeros((self.ny, self.nx), np.float32)
        self._masks = []
        for b in self.active(t):
            m = b.local_mask(self.nx, self.ny)
            if m is None:
                self._masks.append(None)
                continue
            mask, x0, y0 = m
            sl = (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))
            full[sl] |= mask
            yy, xx = np.mgrid[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
            self.wux[sl] = np.where(mask, b.vx - b.omega * (yy - b.y), self.wux[sl])
            self.wuy[sl] = np.where(mask, b.vy + b.omega * (xx - b.x), self.wuy[sl])
            self._masks.append((b, mask, x0, y0))
        return full

    def step(self, lbm, t, dt=1.0, g_dir=(-1.0, 0.0), freeze=False):
        """Advance every free body one LBM step, then hand back the new solid mask.

        Called INSIDE the step loop rather than once per frame: at these speeds a body moves a
        few cells per frame, and moving the mask in one jump per frame injects an audible
        pressure slap into the field (and visibly stutters the wake).

        `freeze` holds the bodies still while the flow settles around them. Without it a body
        materialises inside an already-established stream, instantaneously stopping the fluid
        that occupied its volume, and the resulting pressure impulse fires it across the tunnel
        before drag has any say.
        """
        act = self.active(t)
        # Give a body that is appearing NOW the velocity of the fluid it appears in. A drifter
        # that materialises at rest inside a fast current is, to the solver, a wall suddenly
        # thrown across the stream: the boundary annihilates the momentum in its volume and the
        # pressure pulse takes the lattice out within a few steps. Sampling at the centre is
        # valid here precisely because the body is not solid yet. (Bodies present from t=0 are
        # handled instead by settling with them frozen in place.)
        for b in act:
            if not getattr(b, "_seeded", False):
                if b.mode == "tracer" or b.on_exit == "recycle":
                    b.vx, b.vy, _ = self._sample(lbm, b.x, b.y)
                b._seeded = True
        if not act or freeze:
            return self.build_masks(t)

        dyn = [b for b in act if b.mode == "dynamic"]
        if dyn and self._masks:
            fxf, fyf = lbm.force_field()
            # Accumulate every body's three reductions as DEVICE scalars and pull them back in
            # ONE transfer. Each asnumpy() is a blocking GPU sync; doing three per body per LBM
            # step drained the pipeline and left the card idle between kernels for most of the
            # frame. Same arithmetic, an order of magnitude fewer stalls.
            pend, vals = [], []
            for entry in self._masks:
                if entry is None:
                    continue
                b, mask, x0, y0 = entry
                if b.mode != "dynamic" or not b.alive:
                    continue
                sl = (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))
                # LBM.force_field deposits the wall force on the FLUID cells adjacent to the
                # body, so integrate over the surrounding RING â€” summing over the body's own
                # cells returns exactly zero. (This is why bodies first fell straight through
                # the stream as if the air were not there.)
                h, w = mask.shape
                gm = xp.asarray(_ring(mask).astype(np.float32))
                gfx, gfy = fxf[sl] * gm, fyf[sl] * gm
                rx = xp.arange(x0, x0 + w, dtype=xp.float32)[None, :] - b.x
                ry = xp.arange(y0, y0 + h, dtype=xp.float32)[:, None] - b.y
                vals += [gfx.sum(), gfy.sum(), (rx * gfy - ry * gfx).sum()]
                pend.append(b)
            if pend:
                got = asnumpy(xp.stack(vals)).reshape(-1, 3)
                for b, (fx, fy, tq) in zip(pend, got):
                    b.integrate(float(fx), float(fy), float(tq),
                                self.gravity_at(b, g_dir), dt)

        # A body still OFF the lattice has no mask, so it feels no fluid force — and with gravity
        # off (a drift scene) nothing would ever move it and it would wait off-screen forever.
        # Coast it in at the inlet flow speed until it has a boundary of its own.
        held = {id(e[0]) for e in (self._masks or []) if e is not None}
        for b in act:
            if id(b) not in held and b.on_exit == "recycle":
                b.vx, b.vy, _ = self._sample(lbm, b.x, b.y)
                b.x += b.vx * dt
                b.y += b.vy * dt

        tracers = [b for b in act if b.mode == "tracer"]
        if tracers:
            vort = lbm.vorticity()
            have = {id(e[0]): e for e in (self._masks or []) if e is not None}
            for b in tracers:
                entry = have.get(id(b))
                if entry is None:
                    # Still (partly) outside the lattice — it is drifting IN from below the frame.
                    # Sample at a clamped point so it keeps moving until it has a mask of its own;
                    # otherwise a body waiting off-screen would sit there forever.
                    b.advect(*self._sample(lbm, b.x, b.y), dt)
                    continue
                _, mask, x0, y0 = entry
                # Sample the flow on the body's surrounding RING, never at its centre: the centre
                # is inside the body's own solid, where the stored populations are bounce-back
                # leftovers, not fluid. Reading them hands the body a nonsense velocity, which it
                # then imposes back on the lattice as a moving wall - the solve dies in a
                # fraction of a second.
                sl = (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))
                gm = xp.asarray(_ring(mask).astype(np.float32))
                w = float(asnumpy(gm.sum())) or 1.0
                b.advect(float(asnumpy((lbm.ux[sl] * gm).sum())) / w,
                         float(asnumpy((lbm.uy[sl] * gm).sum())) / w,
                         float(asnumpy((vort[sl] * gm).sum())) / w, dt)
        if self.merge:
            self._coalesce(t)
            act = self._merge_pass(act, t)
        else:
            self._contacts(act)
        for b in act:
            self._contain(b)
        return self.build_masks(t)

    contact_k = 4.0e-3        # spring stiffness (acceleration per unit reduced mass, per gap)
    contact_damp = 0.06       # normal damping; keeps the spring from ringing

    def gravity_at(self, b, g_dir):
        """Acceleration on body `b`: uniform, central if `self.central` is set, plus the mutual
        pull of the other bodies if `self.pair_g` is set. The three compose."""
        if self.central is not None:
            cx, cy, G, soft = self.central
            dx, dy = cx - b.x, cy - b.y
            s = (dx * dx + dy * dy + soft * soft) ** 1.5
            ax, ay = G * dx / s, G * dy / s
        else:
            ax, ay = self.g * g_dir[0], self.g * g_dir[1]
        if self.pair_g > 0.0:
            for o in self.bodies:
                if o is b or not o.alive:
                    continue
                dx, dy = o.x - b.x, o.y - b.y
                # Softened by the pair's own size: two bodies in contact are not point masses,
                # and an unsoftened 1/r^2 would hand them an enormous kick at the moment they
                # touch - exactly when the boundary can least afford a velocity step.
                soft = 0.7 * (b.radius + o.radius)
                s = (dx * dx + dy * dy + soft * soft) ** 1.5
                ax += self.pair_g * o.mass * dx / s
                ay += self.pair_g * o.mass * dy / s
        return (ax, ay)

    # -- liquid coalescence ---------------------------------------------------------------
    def _merge_pass(self, act, t):
        """Fuse any two bodies that have closed to within `merge_gap`. Returns the new list.

        The trigger distance is the SAME stand-off the soft contacts use, and that is deliberate
        rather than conservative: the lattice cannot represent a gap thinner than a cell, so two
        bodies allowed to actually touch first squeeze the fluid between them into a huge
        spurious force (see `_contacts`). Merging at the stand-off means that sub-cell gap never
        exists - and at that separation the metaball union has *already* grown a thin neck, so
        what replaces the pair is the shape two droplets genuinely make as they start to fuse.
        """
        for i in range(len(act)):
            a = act[i]
            if not a.alive or getattr(a, "_coal", None) is not None:
                continue
            for j in range(i + 1, len(act)):
                b = act[j]
                if not b.alive or getattr(b, "_coal", None) is not None:
                    continue
                d = float(np.hypot(b.x - a.x, b.y - a.y))
                if d > a.radius + b.radius + self.merge_gap:
                    continue
                self._fuse(a, b, d, t)
                return self.active(t)          # one merge per step is plenty
        return act

    def _fuse(self, a, b, d, t):
        m = a.mass + b.mass
        x = (a.mass * a.x + b.mass * b.x) / m
        y = (a.mass * a.y + b.mass * b.y) / m
        vx = (a.mass * a.vx + b.mass * b.vx) / m
        vy = (a.mass * a.vy + b.mass * b.vy) / m
        ra, rb = a.radius, b.radius
        ux, uy = (b.x - a.x) / max(d, 1e-9), (b.y - a.y) / max(d, 1e-9)
        # Offsets from the centre of mass. With equal densities the r^2 weighting the metaball
        # uses IS the mass weighting, so the outline's centroid and the body's centre of mass
        # coincide exactly - which is what lets the merged droplet rotate about its own centre.
        fa, fb = b.mass / m, a.mass / m
        area = a.area + b.area
        new = FreeBody(
            droplet([(-fa * d * ux, -fa * d * uy, ra), (fb * d * ux, fb * d * uy, rb)], area),
            1.0, x, y, name="droplet", density=a.density, mode="dynamic",
            pivot=0.0, born=0.0, on_exit=a.on_exit, spin=a.spin)
        new.vx, new.vy = vx, vy
        new.omega = 0.0 if new.spin is None else new.spin
        new._seeded = True
        # animate the two centres together: the neck fills, and at zero separation the metaball
        # is exactly a circle of radius sqrt(ra^2 + rb^2) - the area-conserving one
        new._coal = dict(t0=float(t), d=float(d), ux=ux, uy=uy,
                         ra=ra, rb=rb, fa=fa, fb=fb, area=area)
        a.alive = b.alive = False
        self.bodies = [z for z in self.bodies if z is not a and z is not b]
        self.bodies.append(new)
        self.merges += 1
        if self.on_merge is not None:
            self.on_merge(new, a, b)
        return new

    def _coalesce(self, t):
        """Advance every in-progress merge one step: shrink the separation, reshape the body."""
        for b in self.bodies:
            c = getattr(b, "_coal", None)
            if c is None:
                continue
            u = (float(t) - c["t0"]) / max(self.merge_time, 1e-6)
            if u >= 1.0:
                # at zero separation the metaball IS the circle of radius sqrt(ra^2+rb^2),
                # so `droplet` finds s = 1 and this is the exact area-conserving disc
                b.set_profile(droplet([(0.0, 0.0, float(np.hypot(c["ra"], c["rb"])))], c["area"]))
                b._coal = None
                continue
            # ease OUT: a slow neck, then a surface-tension pinch - what a droplet pair does
            d = c["d"] * (1.0 - smoothstep01(u))
            b.set_profile(droplet(
                [(-c["fa"] * d * c["ux"], -c["fa"] * d * c["uy"], c["ra"]),
                 (c["fb"] * d * c["ux"], c["fb"] * d * c["uy"], c["rb"])], c["area"]))

    def _contacts(self, act, restitution=0.0, gap=9.0):
        """Keep bodies from overlapping, and let them bounce off each other inelastically.

        This is not polish — it is required. The lattice cannot resolve a gap thinner than a
        cell, so two approaching bodies first squeeze the fluid between them into an enormous
        spurious force and then simply merge into one solid blob, at which point the per-body
        force attribution is meaningless. The stand-off has to be several cells, not one or two:
        a 2-cell channel between two bodies is not resolvable either, and the speed in it runs
        away exactly as it does in a sub-cell wall gap. Physically this stands in for the
        lubrication layer a coarse lattice cannot represent; the collision response on top is
        what a pair of dropped discs does anyway.
        """
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                a, b = act[i], act[j]
                dx, dy = b.x - a.x, b.y - a.y
                dist = float(np.hypot(dx, dy))
                if dist < 1e-6:
                    dx, dist = 1e-3, 1e-3
                overlap = (a.radius + b.radius + gap) - dist
                if overlap <= 0:
                    continue
                nx_, ny_ = dx / dist, dy / dist
                ia, ib = 1.0 / a.mass, 1.0 / b.mass
                mu = 1.0 / (ia + ib)                        # reduced mass
                # SOFT spring-damper, spread over many steps. An instantaneous impulse (plus the
                # position projection that went with it) changes each body's wall velocity in a
                # single step, and the boundary radiates that discontinuity as a pressure wave -
                # the visible RIPPLE that crossed the frame whenever two shapes touched. A smooth
                # force does the same job over ~30 steps and the fluid never sees a step change.
                vn = (b.vx - a.vx) * nx_ + (b.vy - a.vy) * ny_
                f = self.contact_k * mu * (overlap / gap)
                if vn < 0.0:
                    f -= self.contact_damp * mu * vn        # resist closing, never suck together
                a.vx -= f * nx_ * ia
                a.vy -= f * ny_ * ia
                b.vx += f * nx_ * ib
                b.vy += f * ny_ * ib

    @staticmethod
    def _sample(lbm, x, y):
        i = int(np.clip(round(x), 1, lbm.nx - 2))
        j = int(np.clip(round(y), 1, lbm.ny - 2))
        ux = float(asnumpy(lbm.ux[j, i]))
        uy = float(asnumpy(lbm.uy[j, i]))
        vt = float(asnumpy((lbm.uy[j, i + 1] - lbm.uy[j, i - 1]
                            - lbm.ux[j + 1, i] + lbm.ux[j - 1, i]) * 0.5))
        return ux, uy, vt

    def _contain(self, b):
        """Keep bodies off the inlet/outlet rows; kill anything that leaves sideways or exits."""
        if b.on_exit == "escape":
            # Free flight: no walls at all. The body simply ceases to exist once it is clear of
            # the lattice — used by the orbit scene, where leaving is a legitimate outcome
            # (something injected above escape velocity is *supposed* to fly off and be gone).
            m = b.radius + 4.0
            if b.x < -m or b.x > self.nx + m or b.y < -m or b.y > self.ny + m:
                b.alive = False
            return
        if b.on_exit == "recycle":
            # Drifters are never held against a boundary: they ride in from below the frame and
            # leave over the top, then re-enter from below again. Re-entering off-screen (rather
            # than popping back in mid-tunnel) is what keeps the recycle from spiking the field.
            marg = b.radius + self.wall_gap
            b.y = float(np.clip(b.y, marg, self.ny - marg))
            if b.x > self.ceiling * self.nx + b.radius:
                b.x = -b.radius - 4.0
                b.y = float(self._rng.uniform(0.2, 0.8)) * self.ny
                b.vx = b.vy = b.omega = 0.0
                b._seeded = False
            return
        # size-aware, so a body never overlaps the inlet or outlet row (which would have it
        # bounce off a boundary condition rather than off the flow)
        lo = self.floor * self.nx + b.radius
        hi = self.ceiling * self.nx - b.radius - self.wall_gap
        band = self.wall_gap
        if b.x < lo:
            if b.on_exit == "kill":                   # fired downward: let it leave the frame
                b.alive = False
                return
        if b.x > hi:
            if b.on_exit == "kill":
                b.alive = False
                return
            if b.on_exit == "recycle":
                b.x = self.floor * self.nx + b.radius + 2.0
                b.y = float(self._rng.uniform(0.18, 0.82)) * self.ny
                b.omega = 0.0
                b._seeded = False           # re-seed its velocity from the flow it lands in
                return
        if b.x < lo + band:
            self._wall(b, (lo + band) - b.x, lo, 1.0, 0.0)
        if b.x > hi - band:
            self._wall(b, b.x - (hi - band), hi, -1.0, 0.0)
        # Hold a RESOLVABLE gap to the side walls. Let a body touch the wall and the passage
        # between them narrows below one cell, where the lattice cannot represent the flow at
        # all: the speed there runs away and takes the whole solve with it. (This was the actual
        # cause of every late-clip blow-up — the failure always sat in a sub-cell wall gap.)
        ylo = self.wall_gap if self.ybounds is None else self.ybounds[0]
        yhi = (self.ny - self.wall_gap) if self.ybounds is None else self.ybounds[1]
        yl, yh = ylo + b.radius, yhi - b.radius
        if b.y < yl + band:
            self._wall(b, (yl + band) - b.y, yl, 0.0, 1.0)
        if b.y > yh - band:
            self._wall(b, b.y - (yh - band), yh, 0.0, -1.0)

    soft_walls = True         # False restores the pre-2026-07-27 hard clamp (for A/B only)
    wall_k = 2.4e-3           # wall spring: acceleration at full penetration of the band
    wall_damp = 0.07          # normal damping, same role as contact_damp
    wall_stop = 0.35          # positional backstop, as a fraction of the band past the surface

    def _wall(self, b, pen, surf, nx_, ny_):
        """Push a body off a wall with a SOFT spring-damper instead of clamping it.

        A hard clamp (`b.y, b.vy = marg, -abs(b.vy) * 0.4`) steps the body's velocity - and
        therefore its WALL velocity, which the moving-boundary term feeds straight into the
        populations - inside a single tick. The boundary broadcasts that discontinuity as a
        pressure wave, and it crosses the whole picture as a visible RIPPLE every time anything
        touches a wall. It is the identical failure that impulse-based body-body contacts had
        (AGENT_GUIDE 2a.10), and it wants the identical fix: spread the same momentum change
        over ~30 steps so the fluid never sees a step change.

        The spring engages one `wall_gap` BEFORE the stand-off surface, so a body decelerates
        into the wall rather than being caught at it: with the stiffness below, a disc falling
        at terminal speed is stopped in ~1.5 cells and a resting one hovers about 5 cells clear.
        `pen` is the depth into that band, `surf` the stand-off coordinate, (nx_, ny_) the
        inward normal. The positional backstop only fires if the spring is badly outrun - the
        one thing that must never happen here is a sub-cell gap to the wall.
        """
        if not self.soft_walls:                       # the old behaviour, kept for comparison
            surf_pen = pen - self.wall_gap
            if surf_pen <= 0.0:
                return
            if ny_ == 0.0:
                b.x = surf
                if b.vx * nx_ < 0.0:
                    b.vx = 0.0
            else:
                b.y = surf
                b.vy = abs(b.vy) * 0.4 * ny_
            return
        a = self.wall_k * (pen / max(self.wall_gap, 1e-6))
        vn = b.vx * nx_ + b.vy * ny_                  # < 0 while it is still closing on the wall
        if vn < 0.0:
            a -= self.wall_damp * vn
        b.vx += a * nx_
        b.vy += a * ny_
        n = nx_ + ny_                                  # +1 or -1; only one axis is ever active
        stop = surf - self.wall_stop * self.wall_gap * n
        cur = b.x if ny_ == 0.0 else b.y
        if (cur - stop) * n < 0.0:
            if ny_ == 0.0:
                b.x = stop
            else:
                b.y = stop

    def polygons(self, t):
        return [b.polygon() for b in self.active(t)]


