"""world.py - the physics. Balls against a list of OBSTACLES.

Deliberately small and float-based (no numpy in the inner loop - numpy scalar ops cost ~10x a
plain float multiply, and this loop runs ~1e6+ times for a clip).

Three obstacle primitives cover everything the scene library needs:

    Ring     - a hollow shell of thickness h with N gaps, optionally spinning.
    Capsule  - a thick segment. Spin it about a pivot for a paddle/gear tooth, swing it for a
               pendulum, chain them for a polygon / funnel / spiral / chute, give it `hp` for a
               destructible brick.
    Peg      - a disc. Static for a plinko pin or a bumper; give it a pivot to orbit.

All three reduce to the same two resolvers (`_disc` for anything round, `_resolve` for the
impulse), which is why adding a shape is cheap and why they all interact correctly.

WHAT A SCENE MAY DO (the Wind_Tunnel discipline): choose geometry, materials and forcing -
positions, sizes, spin rates, restitution, friction, gravity, what spawns and when. It may NOT
place a ball on a path. Every bounce, escape and fall is solved.

Units are REFERENCE px (1080x1920, y DOWN) and seconds; see config.REF_W/REF_H. Angles are
radians from +x toward +y, matching both atan2(dy,dx) and PIL's arc convention - so a physics
angle is a drawing angle, no conversion anywhere.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

TAU = 2.0 * math.pi


def _wrap(a: float) -> float:
    """to [-pi, pi)"""
    return (a + math.pi) % TAU - math.pi


def _rot(x, y, px, py, a):
    """rotate (x,y) about (px,py) by a"""
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - px, y - py
    return px + dx * c - dy * s, py + dx * s + dy * c


# --------------------------------------------------------------------------- obstacles

@dataclass
class Ring:
    """A hollow circular shell of thickness `thickness`, with `n_gaps` evenly spaced holes,
    spinning at `omega` rad/s. Collision is against the band [R-h, R+h] where the arc is
    solid, PLUS a cap circle of radius h at each gap edge - the edge is a real object a ball
    can clip, and a sweeping edge is what shoves a half-escaped ball back inside."""
    radius: float
    thickness: float
    n_gaps: int
    gap_width: float                # radians
    omega: float = 0.0              # rad/s
    phase0: float = 0.0
    e: float = 0.92
    mu: float = 0.16
    color: tuple = (255, 255, 255)
    name: str = "ring"
    kind: str = "ring"
    alive: bool = True
    hp: int = -1                    # rings are indestructible; field exists so _resolve is
    hp0: int = -1                   # uniform across every obstacle kind

    @property
    def h(self) -> float:
        return 0.5 * self.thickness

    def phase(self, t: float) -> float:
        return self.phase0 + self.omega * t

    def gap_center(self, k: int, t: float) -> float:
        return self.phase(t) + TAU * k / max(1, self.n_gaps)

    def gap_index(self, ang: float, t: float) -> int:
        best, bd = 0, 1e9
        for k in range(max(1, self.n_gaps)):
            d = abs(_wrap(ang - self.gap_center(k, t)))
            if d < bd:
                best, bd = k, d
        return best

    def is_gap(self, ang: float, t: float) -> bool:
        if self.n_gaps <= 0:
            return False
        hw = 0.5 * self.gap_width
        p = self.phase(t)
        for k in range(self.n_gaps):
            if abs(_wrap(ang - (p + TAU * k / self.n_gaps))) < hw:
                return True
        return False

    def solid_arcs(self, t: float):
        """[(a0, a1), ...] radians - the drawable solid spans between consecutive gaps."""
        if self.n_gaps <= 0:
            return [(0.0, TAU)]
        hw = 0.5 * self.gap_width
        p = self.phase(t)
        step = TAU / self.n_gaps
        return [(p + step * k + hw, p + step * (k + 1) - hw) for k in range(self.n_gaps)]

    def edge_angles(self, t: float):
        if self.n_gaps <= 0:
            return []
        hw = 0.5 * self.gap_width
        p = self.phase(t)
        step = TAU / self.n_gaps
        out = []
        for k in range(self.n_gaps):
            c = p + step * k
            out.append(c - hw)
            out.append(c + hw)
        return out

    def collide(self, w: "World", b: "Ball", t: float) -> None:
        dx, dy = b.x - w.cx, b.y - w.cy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        reach = self.h + b.r
        if abs(dist - self.radius) > reach + 2.0:
            return                                  # nowhere near this shell - cheap out
        nx, ny = dx / dist, dy / dist
        ang = math.atan2(dy, dx)
        om = self.omega
        if not self.is_gap(ang, t):
            if dist <= self.radius:
                s = self.radius - reach             # inner face: normal points inward
                if dist > s:
                    px, py = w.cx + nx * (self.radius - self.h), w.cy + ny * (self.radius - self.h)
                    w._resolve(b, -nx, -ny, w.cx + nx * s, w.cy + ny * s,
                               -om * (py - w.cy), om * (px - w.cx), self.e, self.mu, self)
            else:
                s = self.radius + reach             # outer face: normal points outward
                if dist < s:
                    px, py = w.cx + nx * (self.radius + self.h), w.cy + ny * (self.radius + self.h)
                    w._resolve(b, nx, ny, w.cx + nx * s, w.cy + ny * s,
                               -om * (py - w.cy), om * (px - w.cx), self.e, self.mu, self)
        for ea in self.edge_angles(t):
            qx, qy = w.cx + self.radius * math.cos(ea), w.cy + self.radius * math.sin(ea)
            w._disc(b, qx, qy, self.h, -om * (qy - w.cy), om * (qx - w.cx),
                    self.e, self.mu, self)


@dataclass
class Capsule:
    """A thick segment: the workhorse. Endpoints are given at phase 0; `pivot` + `omega`
    spins it (paddle, gear tooth), `swing_amp`/`swing_freq` swings it (pendulum), `hp` makes
    it destructible (brick). Chain several for a polygon, funnel, chute or spiral."""
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float = 14.0
    pivot: tuple | None = None
    omega: float = 0.0
    phase0: float = 0.0
    swing_amp: float = 0.0          # radians; if non-zero, angle = amp*sin(2pi f t + ph)
    swing_freq: float = 0.0
    sway_amp: float = 0.0           # px of HORIZONTAL travel (a sliding, not turning, piece)
    sway_freq: float = 0.0
    sway_phase: float = 0.0
    e: float = 0.92
    mu: float = 0.10
    color: tuple = (255, 255, 255)
    name: str = "capsule"
    kind: str = "capsule"
    hp: int = -1                    # -1 = indestructible
    alive: bool = True
    hp0: int = -1                   # set from hp, so draw.py can fade a damaged brick

    def __post_init__(self):
        self.hp0 = self.hp

    @property
    def h(self) -> float:
        return 0.5 * self.thickness

    def angle(self, t: float) -> float:
        if self.swing_amp:
            return self.phase0 + self.swing_amp * math.sin(TAU * self.swing_freq * t)
        return self.phase0 + self.omega * t

    def omega_at(self, t: float) -> float:
        if self.swing_amp:
            return self.swing_amp * TAU * self.swing_freq * math.cos(TAU * self.swing_freq * t)
        return self.omega

    def sway(self, t: float) -> float:
        if not self.sway_amp:
            return 0.0
        return self.sway_amp * math.sin(TAU * self.sway_freq * t + self.sway_phase)

    def sway_v(self, t: float) -> float:
        if not self.sway_amp:
            return 0.0
        return self.sway_amp * TAU * self.sway_freq * math.cos(
            TAU * self.sway_freq * t + self.sway_phase)

    def ends(self, t: float):
        sx = self.sway(t)
        if self.pivot is None or (not self.omega and not self.swing_amp and not self.phase0):
            return self.x0 + sx, self.y0, self.x1 + sx, self.y1
        px, py = self.pivot
        a = self.angle(t)
        ax, ay = _rot(self.x0, self.y0, px, py, a)
        bx, by = _rot(self.x1, self.y1, px, py, a)
        return ax + sx, ay, bx + sx, by

    def collide(self, w: "World", b: "Ball", t: float) -> None:
        if not self.alive:
            return
        ax, ay, bx, by = self.ends(t)
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            u = 0.0
        else:
            u = ((b.x - ax) * dx + (b.y - ay) * dy) / L2
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        qx, qy = ax + u * dx, ay + u * dy
        om = self.omega_at(t)
        if om and self.pivot:
            px, py = self.pivot
            sx = self.sway(t)
            wvx, wvy = -om * (qy - py), om * (qx - px - sx)
        else:
            wvx = wvy = 0.0
        wvx += self.sway_v(t)       # a sliding piece carries the ball with it
        w._disc(b, qx, qy, self.h, wvx, wvy, self.e, self.mu, self)


@dataclass
class Peg:
    """A disc. Static pin/bumper, or give it a pivot to orbit. `e` above 1 makes it a kicker
    (a pinball bumper genuinely adds energy - that is what a solenoid does)."""
    x: float
    y: float
    radius: float = 12.0
    e: float = 0.92
    mu: float = 0.05
    color: tuple = (255, 255, 255)
    name: str = "peg"
    kind: str = "peg"
    pivot: tuple | None = None
    omega: float = 0.0
    hp: int = -1
    hp0: int = -1
    alive: bool = True
    ghost: bool = False             # drawn, but nothing collides with it

    def pos(self, t: float):
        if self.pivot and self.omega:
            return _rot(self.x, self.y, self.pivot[0], self.pivot[1], self.omega * t)
        return self.x, self.y

    def collide(self, w: "World", b: "Ball", t: float) -> None:
        if not self.alive or self.ghost:
            return
        qx, qy = self.pos(t)
        if self.pivot and self.omega:
            px, py = self.pivot
            wvx, wvy = -self.omega * (qy - py), self.omega * (qx - px)
        else:
            wvx = wvy = 0.0
        w._disc(b, qx, qy, self.radius, wvx, wvy, self.e, self.mu, self)


# --------------------------------------------------------------------------- balls

@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    gen: int = 0
    state: int = 0          # 0 = inside the inner shell, 1 = between shells, 2 = outside both
    alive: bool = True
    trail: list = field(default_factory=list)
    last_bounce: float = -1e9   # refractory clock for the bounce blips
    zone: int = -1              # last scoring zone entered (so entry fires once)
    born: float = 0.0           # spawn time, for age-gated rules like decay
    uid: int = 0                # stable identity, for tracking a contact episode
    contacts: set = field(default_factory=set)   # partners in an already-decided overlap

    @property
    def m(self) -> float:
        return self.r * self.r


class World:
    """Balls + obstacles. Owns the event log the renderer, the Score and the blips read."""

    def __init__(self, obstacles, cx, cy, width, height, *, gravity=1500.0, seed=7,
                 ball_e=0.90, r0=26.0, shrink=0.78, r_min=7.0, max_balls=28,
                 spawn_speed=260.0, spawn_jitter=30.0, v_max=1600.0, split=2,
                 split_min_r=0.0, bounce_vmin=70.0, bounce_gap=0.05,
                 escape_ring=None, inner_ring=None, central_g=0.0, drag=0.0,
                 spawn_mode="center", spawn_every=0.0, initial=1, respawn="split",
                 zones=None, bottom_triggers=False, wrap_sides=False, kill_below=True,
                 sensor_r=0.0, tier_shrink=0.0, n_tiers=6, g_size_exp=0.0,
                 scatter_chance=0.0, scatter_into=3, recombine_chance=0.0,
                 dmg_ref=0.0, size_speed=0.0, orbit_dir=0, orbit_boost=1.0, tan_kick=0.0,
                 swirl_a=0.0, decay_after=0.0, decay_rate=0.0, decay_tier=1,
                 guides=None, guide_a=0.0, guide_soft=150.0, aniso=1.0,
                 plane_z=None, plane_ramp=0.0, angle_rule=False, align_exp=1.0,
                 tier_dir=None):
        self.obstacles = list(obstacles)
        self.rings = [o for o in self.obstacles if o.kind == "ring"]
        self.cx, self.cy = float(cx), float(cy)
        self.width, self.height = float(width), float(height)
        self.g = float(gravity)
        self.central_g = float(central_g)
        self.drag = float(drag)
        self.ball_e = float(ball_e)
        self.r0, self.shrink, self.r_min = float(r0), float(shrink), float(r_min)
        self.max_balls, self.split = int(max_balls), int(split)
        self.split_min_r = float(split_min_r)
        # ---- TIERS: one size/colour ladder shared by every mechanic that makes a ball
        # smaller (a multiplier pit, a scatter). `gen` IS the tier index, and draw.py already
        # colours a ball by gen, so a scene only has to order its palette to name the tiers.
        self.tier_shrink = float(tier_shrink) or float(shrink)
        self.n_tiers = int(n_tiers)
        self.g_size_exp = float(g_size_exp)     # central pull scaled by (r/r0)**exp
        self.size_speed = float(size_speed)     # extra speed given to a smaller child
        self.scatter_chance = float(scatter_chance)
        self.scatter_into = int(scatter_into)
        self.recombine_chance = float(recombine_chance)
        self.dmg_ref = float(dmg_ref)           # px of radius worth 1 hp of brick damage
        # ---- INTRINSIC ANGULAR MOMENTUM. `orbit_dir` makes the whole swarm CO-ROTATE instead
        # of half the balls running each way, `orbit_boost` scales the launch above the circular
        # speed, and `tan_kick` biases a scatter's fragments along the circulation instead of
        # purely radially. Together they give the system a net angular momentum it keeps.
        # This is a declared FORCING (the same category as Wind_Tunnel's prescribed spin), not
        # an emergent result - a random-direction swarm cancels its own circulation to zero.
        self.orbit_dir = int(orbit_dir)
        # PER-TIER circulation direction. A tier that runs the other way is head-on to every
        # other tier, which under the angle rule means it never interacts with them - the two
        # rules compose rather than fighting.
        self.tier_dir = list(tier_dir or [])
        self.orbit_boost = float(orbit_boost)
        self.tan_kick = float(tan_kick)
        # `swirl_a` is the one that actually HOLDS the circulation. A launch bias alone does not:
        # collisions with the shell, the kickers and each other randomise headings, and after
        # 30 s only 6 of 13 spheres were still co-rotating (i.e. a coin flip - the swarm had
        # cancelled its own angular momentum). This is a tangential acceleration that drives
        # each sphere TOWARD the local circular speed in the shared direction - the same
        # drive-toward-a-target device as Wind_Tunnel's `set_drive`, and a declared forcing
        # rather than a fudge: a stirrer, not a keyframe.
        self.swirl_a = float(swirl_a)
        # ---- DECAY: an age-gated climb back UP the ladder. Together with scatter (down) and
        # recombine (up, smallest tier only) this closes the ladder into a cycle instead of a
        # slide, so a clip can run long without settling into one colour.
        self.decay_after = float(decay_after)   # seconds a ball must live before it may decay
        self.decay_rate = float(decay_rate)     # probability per second once eligible
        self.decay_tier = int(decay_tier)
        self.decays = 0
        # ---- ORBIT GUIDES: one ELLIPSE per tier, rotationally staggered. A weak radial
        # restoring pull toward the ellipse's own radius at the body's current angle - so a
        # sphere is "somewhat drawn" to its track while the central pull still does the real
        # work. Another declared forcing, like `swirl_a`.
        # Each entry is (semi_major, semi_minor, rotation). Three ellipses 60 deg apart read as
        # a SIX-lobed petal rosette, because an ellipse already has two-fold symmetry.
        self.guides = list(guides or [])
        self.guide_a = float(guide_a)
        self.guide_soft = float(guide_soft)     # px of error at which the pull saturates
        # ---- ANISOTROPY: stretch the whole orbital FIELD vertically so orbits fill a 9:16
        # frame. Everything radial (central pull, guides, swirl) is computed in a stretched
        # metric where ry = (y-cy)/aniso, i.e. under a potential V(u) with
        # u = hypot(x-cx, (y-cy)/aniso). A circle in that metric IS a vertical ellipse on
        # screen, so orbits come out stretched by `aniso` without anything fighting anything.
        #
        # It has to be the FIELD, not just the guide tracks: the central pull is what actually
        # shapes an orbit, and stretching only the weak guides would leave gravity pulling
        # every orbit back to round. Bodies stay circular - only the field is anisotropic.
        self.aniso = float(aniso)
        # ---- 2D -> 3D PLANE SEPARATION (Ethan 2026-07-30). Once every destructible segment is
        # gone, tiers stop intersecting: each tier is given a DEPTH offset and ball-ball contact
        # is tested in 3D. Two spheres separated by dz collide only when
        #     d_2d^2 < (ra+rb)^2 - dz^2
        # so the effective 2D contact radius shrinks as dz grows and reaches zero at dz = ra+rb.
        # Nothing is faked and nothing is switched off: it is the real 3D test, which is why the
        # transition can be RAMPED smoothly - cross-tier hits first become glancing, then rarer,
        # then impossible, and the eye reads it as the tracks separating into their own planes.
        # Same-tier pairs share a plane (dz = 0) and keep colliding exactly as before.
        self.plane_z = list(plane_z or [])
        self.plane_ramp = float(plane_ramp)
        self.t_break = None                     # when the last destructible segment fell
        self._destructibles = [o for o in self.obstacles if getattr(o, "hp0", -1) > 0]
        self.cross_passes = 0                   # cross-tier pairs that passed through
        # ---- ANGLE RULE (Ethan 2026-07-30). Whether two spheres interact at all depends on how
        # ALIGNED their headings are, not on whether they touch:
        #     head-on (opposite headings)  -> 0 %  : they never interact, they pass through
        #     rear-end (same heading)      -> 100 %: they always scatter, into `scatter_into`
        # p = ((cos(angle between velocities) + 1) / 2) ** align_exp, so p runs 0 -> 1 as the
        # headings swing from opposed to identical. An interaction at the SMALLEST tier
        # recombines instead of scattering (nothing smaller exists), which keeps the ladder a
        # cycle. There is no elastic bounce between spheres any more: the outcome is transmute
        # or pass straight through.
        self.angle_rule = bool(angle_rule)
        self.align_exp = float(align_exp)
        self.angle_passes = 0                   # contacts that passed through on the angle roll
        self.spawn_immunity = 0.12              # s of no-interaction after a ball is created
        self._uid = 0
        self.scatters = 0
        self.recombines = 0
        self.mult_hits = 0
        self.pair_hits = 0          # ball-ball impacts: the pool scatter/recombine draw from
        # bounce blips: a contact only counts if it is a real IMPACT, not a rest contact.
        # A ball rolling on a shell resolves against it on nearly every substep, so without
        # both an approach-speed floor and a per-ball refractory gap the track turns into a
        # buzz (900 contacts/s/ball at the default physics rate).
        self.bounce_vmin = float(bounce_vmin)
        self.bounce_gap = float(bounce_gap)
        self.bounces: list[dict] = []
        self._tnow = 0.0
        self.spawn_speed, self.spawn_jitter = float(spawn_speed), float(spawn_jitter)
        self.v_max = float(v_max)
        self.spawn_mode = spawn_mode
        # radius band the "orbit" spawn mode launches into: inside the rim if there is one
        self._orbit_span = (0.90 * max((r.radius for r in self.rings), default=0.0)
                            or 0.42 * float(width))
        self.spawn_every = float(spawn_every)
        self.respawn = respawn
        self.zones = list(zones or [])
        self.bottom_triggers = bool(bottom_triggers)
        self.wrap_sides = bool(wrap_sides)
        self.kill_below = bool(kill_below)
        self.t = 0.0
        self._last_spawn = 0.0
        self.balls: list[Ball] = []
        # event log (times are exact, at substep resolution)
        self.triggers: list[dict] = []      # the HEADLINE event (whatever the scene's is)
        self.inner_exits: list[dict] = []
        self.flashes: list[dict] = []       # visual only: {t, x, y, kind, r}
        self.exits_bottom = 0
        self.suppressed_spawns = 0
        self.destroyed = 0
        self.rng = random.Random(seed)
        # which ring (if any) defines "escaped" for the headline trigger
        self._outer = escape_ring if escape_ring is not None else (
            max(self.rings, key=lambda r: r.radius) if self.rings else None)
        self._inner = inner_ring if inner_ring is not None else (
            min(self.rings, key=lambda r: r.radius) if self.rings else None)
        self.escape_ring = self._outer
        # The escape test is RADIAL and independent of what the shell is made of, so a polygon
        # or brick shell gets the same headline event as a Ring does. Defaults to just outside
        # the outer ring when there is one.
        self.sensor_r = float(sensor_r) if sensor_r else (
            (self._outer.radius + self._outer.h) if self._outer is not None else 0.0)
        self.spawn(int(initial), gen=0, r=self.r0)

    # ---------------------------------------------------------------- spawning
    def _spawn_point(self, r: float, gen: int = 0):
        if self.spawn_mode == "top":
            x = self.rng.uniform(0.22 * self.width, 0.78 * self.width)
            return x, -r * 1.5, self.rng.uniform(-60.0, 60.0), 40.0
        if self.spawn_mode == "top_center":
            return (self.cx + self.rng.uniform(-24.0, 24.0), -r * 1.5,
                    self.rng.uniform(-30.0, 30.0), 40.0)
        if self.spawn_mode == "orbit":
            # Launch TANGENTIALLY at the circular-orbit speed. For this constant-magnitude
            # central pull that is v = sqrt(a*r) (from v^2/r = a), not the sqrt(GM/r) of an
            # inverse square. Spawning in a random direction instead just drops every ball
            # straight down the well, and they pile on the core rather than orbiting.
            rad = self.rng.uniform(0.42, 0.92) * self._orbit_span
            a = self.rng.uniform(0, TAU)
            v = (math.sqrt(max(self.central_g, 1.0) * rad) * self.orbit_boost
                 * self.rng.uniform(0.90, 1.08))
            d0 = self.dir_for(gen)
            sgn = float(d0) if d0 else (1.0 if self.rng.random() < 0.5 else -1.0)
            return (self.cx + rad * math.cos(a), self.cy + rad * math.sin(a),
                    -sgn * v * math.sin(a), sgn * v * math.cos(a))
        a = self.rng.uniform(0, TAU)
        d = self.rng.uniform(0.0, self.spawn_jitter)
        va = self.rng.uniform(0, TAU)
        sp = self.spawn_speed * self.rng.uniform(0.75, 1.25)
        return (self.cx + d * math.cos(a), self.cy + d * math.sin(a),
                sp * math.cos(va), sp * math.sin(va))

    def spawn_at(self, x, y, vx, vy, r, gen=0, flash=True) -> Ball | None:
        """Place ONE ball explicitly - used by mechanics that happen somewhere specific (a
        scatter at a contact point, a recombination at a midpoint) rather than at a spawner."""
        if self.n_live() >= self.max_balls:
            self.suppressed_spawns += 1
            return None
        self._uid += 1
        b = Ball(x, y, vx, vy, max(self.r_min, r), gen, born=self._tnow, uid=self._uid)
        self.balls.append(b)
        if flash:
            self.flashes.append({"t": self._tnow, "x": x, "y": y, "kind": "spawn", "r": b.r})
        return b

    def spawn(self, n: int, gen: int = 0, r: float | None = None) -> None:
        r = self.r0 if r is None else max(self.r_min, r)
        live = sum(1 for b in self.balls if b.alive)
        for _ in range(n):
            if live >= self.max_balls:
                self.suppressed_spawns += 1
                continue
            x, y, vx, vy = self._spawn_point(r, gen)
            self._uid += 1
            b = Ball(x, y, vx, vy, r, gen, born=self._tnow, uid=self._uid)
            self.balls.append(b)
            live += 1
            self.flashes.append({"t": self.t, "x": b.x, "y": b.y, "kind": "spawn",
                                 "r": b.r})

    # ---------------------------------------------------------------- collisions
    def _resolve(self, b: Ball, nx: float, ny: float, px: float, py: float,
                 wvx: float, wvy: float, e: float, mu: float, obs=None) -> None:
        """Snap the ball to touching and reflect its velocity RELATIVE TO THE MOVING WALL.
        (n = unit surface normal pointing at the ball.) Doing this in the wall frame is what
        makes a spinning shell fling a ball instead of merely stopping it."""
        b.x, b.y = px, py
        rvx, rvy = b.vx - wvx, b.vy - wvy
        vn = rvx * nx + rvy * ny
        if vn < 0.0:
            tvx, tvy = rvx - vn * nx, rvy - vn * ny
            b.vx = -e * vn * nx + (1.0 - mu) * tvx + wvx
            b.vy = -e * vn * ny + (1.0 - mu) * tvy + wvy
            self._log_bounce(b, -vn, obs)
            if obs is not None and obs.hp > 0 and -vn > self.bounce_vmin:
                # damage scales with the BALL, so a big sphere smashes a brick a small one
                # only chips (dmg_ref = px of radius worth one hp)
                dmg = 1 if not self.dmg_ref else max(1, int(b.r / self.dmg_ref))
                obs.hp -= dmg
                if obs.hp <= 0:
                    obs.alive = False
                    self.destroyed += 1
                    self.triggers.append({"t": self._tnow, "x": px, "y": py, "r": b.r,
                                          "gen": b.gen, "gap": 0, "kind": "destroy"})
                    self.flashes.append({"t": self._tnow, "x": px, "y": py,
                                         "kind": "trigger", "r": max(b.r, 14.0)})

    def _disc(self, b: Ball, cx: float, cy: float, rc: float, wvx: float, wvy: float,
              e: float, mu: float, obs=None) -> None:
        dx, dy = b.x - cx, b.y - cy
        d2 = dx * dx + dy * dy
        s = rc + b.r
        if d2 >= s * s or d2 < 1e-12:
            return
        d = math.sqrt(d2)
        nx, ny = dx / d, dy / d
        self._resolve(b, nx, ny, cx + nx * s, cy + ny * s, wvx, wvy, e, mu, obs)

    def _log_bounce(self, b: Ball, impact: float, obs=None) -> None:
        """One audible bounce. `impact` is the approach speed in the WALL's frame, which is the
        right quantity: a ball settling onto a shell that is sweeping along with it barely
        knocks, and should barely sound."""
        t = self._tnow
        if impact > self.bounce_vmin and (t - b.last_bounce) > self.bounce_gap:
            b.last_bounce = t
            self.bounces.append({"t": t, "r": b.r, "v": impact, "x": b.x,
                                 "obs": getattr(obs, "name", "")})

    def _collide_balls(self, live) -> None:
        e = self.ball_e
        n = len(live)
        sep = self.plane_sep(self._tnow)
        pz = self.plane_z
        for i in range(n - 1):
            a = live[i]
            if not a.alive:                 # a scatter/recombine may have consumed it
                continue
            for j in range(i + 1, n):
                b = live[j]
                if not (b.alive and a.alive):
                    continue
                dx, dy = b.x - a.x, b.y - a.y
                s = a.r + b.r
                d2 = dx * dx + dy * dy
                # 3D contact: depth separation shrinks the effective 2D contact radius
                thr2 = s * s
                if sep > 0.0 and pz and a.gen != b.gen:
                    dz = (pz[min(a.gen, len(pz) - 1)] - pz[min(b.gen, len(pz) - 1)]) * sep
                    thr2 -= dz * dz
                    if thr2 <= 1e-9:
                        if d2 < s * s:
                            self.cross_passes += 1      # overlaps on screen, misses in depth
                        continue
                if d2 >= thr2 or d2 < 1e-12:
                    # contact episode is over - re-arm this pair for a fresh decision
                    if a.contacts:
                        a.contacts.discard(b.uid)
                    if b.contacts:
                        b.contacts.discard(a.uid)
                    continue

                if self.angle_rule:
                    # ONE decision per contact episode. A touching pair stays overlapped for
                    # many substeps, and re-rolling every step would make even a 1%-per-step
                    # chance a near-certainty - the roll has to happen on first contact only.
                    if b.uid in a.contacts:
                        continue
                    a.contacts.add(b.uid)
                    b.contacts.add(a.uid)
                    # A freshly created ball is still separating from its siblings; letting it
                    # interact immediately cascades (fragments are born overlapping and moving
                    # in a fan, so adjacent ones re-fragment each other without limit - 32 525
                    # interactions in a 30 s clip before this guard existed).
                    imm = self.spawn_immunity
                    if imm and (self._tnow - a.born < imm or self._tnow - b.born < imm):
                        continue
                    sa = math.hypot(a.vx, a.vy)
                    sb = math.hypot(b.vx, b.vy)
                    if sa < 1e-6 or sb < 1e-6:
                        continue
                    c = (a.vx * b.vx + a.vy * b.vy) / (sa * sb)     # +1 same, -1 opposed
                    p = 0.5 * (c + 1.0)
                    if self.align_exp != 1.0:
                        p = p ** self.align_exp
                    if self.rng.random() >= p:
                        self.angle_passes += 1          # glanced off-heading: pass through
                        continue
                    self.pair_hits += 1
                    self._log_bounce(a if a.r <= b.r else b, max(sa, sb))
                    self._maybe_transmute(a, b, dx, dy)
                    continue

                d = math.sqrt(d2)
                nx, ny = dx / d, dy / d
                ma, mb = a.m, b.m
                inv = 1.0 / (ma + mb)
                pen = math.sqrt(thr2) - d
                a.x -= nx * pen * (mb * inv)
                a.y -= ny * pen * (mb * inv)
                b.x += nx * pen * (ma * inv)
                b.y += ny * pen * (ma * inv)
                vn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                if vn < 0.0:
                    jimp = -(1.0 + e) * vn * (ma * mb) * inv
                    a.vx -= jimp / ma * nx
                    a.vy -= jimp / ma * ny
                    b.vx += jimp / mb * nx
                    b.vy += jimp / mb * ny
                    # one blip per pair, pitched off the SMALLER ball (it is the one that
                    # rings) - logging both would double every ball-ball hit
                    self._log_bounce(a if a.r <= b.r else b, -vn)
                    self.pair_hits += 1
                    if self.scatter_chance or self.recombine_chance:
                        self._maybe_transmute(a, b, nx, ny)

    def dir_for(self, gen: int) -> int:
        """Circulation direction for a tier (falls back to the global `orbit_dir`)."""
        if self.tier_dir:
            return int(self.tier_dir[min(gen, len(self.tier_dir) - 1)])
        return self.orbit_dir

    def plane_sep(self, t: float | None = None) -> float:
        """0 while the shell stands, ramping to 1 over `plane_ramp` seconds after the last
        destructible segment falls. Smoothstepped, so depth eases in rather than snapping."""
        if not self.plane_z or self.t_break is None:
            return 0.0
        if self.plane_ramp <= 0.0:
            return 1.0
        u = ((self.t if t is None else t) - self.t_break) / self.plane_ramp
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return u * u * (3.0 - 2.0 * u)

    def ball_z(self, b: Ball, t: float | None = None) -> float:
        """Current depth of a ball's plane (0 until the separation starts)."""
        if not self.plane_z:
            return 0.0
        return self.plane_z[min(b.gen, len(self.plane_z) - 1)] * self.plane_sep(t)

    def _maybe_transmute(self, a: Ball, b: Ball, nx: float, ny: float) -> None:
        """SCATTER a colliding pair into `scatter_into` smaller balls, or RECOMBINE two of the
        smallest into one of the tier above. Both are chance-based, and both are applied to the
        PAIR (two in, three out / two in, one out) rather than per ball.

        Momentum is carried across: the children start from the pair's momentum and are then
        given a symmetric outward kick, whose vector sum is ~zero, so nothing is injected into
        the net motion of the system - only into its internal energy, which is what a real
        fragmentation does.
        """
        if not (a.alive and b.alive):
            return
        tier = max(a.gen, b.gen)
        px, py = 0.5 * (a.x + b.x), 0.5 * (a.y + b.y)
        ma, mb = a.m, b.m
        mt = ma + mb
        vx = (a.vx * ma + b.vx * mb) / mt
        vy = (a.vy * ma + b.vy * mb) / mt

        # Under the angle rule the interaction has ALREADY been decided by the heading roll, so
        # the per-outcome chance gates are bypassed: an interaction at the smallest tier fuses,
        # anything else fragments.
        forced = self.angle_rule
        if (tier >= self.n_tiers - 1 and a.gen == b.gen
                and (forced or (self.recombine_chance
                                and self.rng.random() < self.recombine_chance))):
            r_new = math.hypot(a.r, b.r)            # conserve AREA: r = sqrt(ra^2 + rb^2)
            a.alive = b.alive = False
            if self.spawn_at(px, py, vx, vy, r_new, max(0, tier - 1)) is not None:
                self.recombines += 1
                self.flashes.append({"t": self._tnow, "x": px, "y": py,
                                     "kind": "trigger", "r": r_new})
            return

        if (tier + 1 < self.n_tiers
                and (forced or (self.scatter_chance
                                and self.rng.random() < self.scatter_chance))):
            n = max(2, self.scatter_into)
            r_new = max(self.r_min, min(a.r, b.r) * self.tier_shrink)
            a.alive = b.alive = False
            base = math.atan2(ny, nx)
            kick = self.size_speed or 120.0
            made = 0
            kids: list[Ball] = []
            for k in range(n):
                ang = base + TAU * k / n
                ca, sa = math.cos(ang), math.sin(ang)
                sx, sy = px + ca * (r_new + 1.5), py + sa * (r_new + 1.5)
                cvx, cvy = vx + ca * kick, vy + sa * kick
                kdir = self.dir_for(tier + 1)
                if self.tan_kick and kdir:
                    # bias fragments into the circulation of the tier they are BECOMING, not the
                    # parent's - a white breaking into blues hands them the blue direction
                    rx, ry = sx - self.cx, sy - self.cy
                    rr = math.hypot(rx, ry)
                    if rr > 1e-6:
                        cvx += kdir * (-ry / rr) * self.tan_kick
                        cvy += kdir * (rx / rr) * self.tan_kick
                nb = self.spawn_at(sx, sy, cvx, cvy, r_new, tier + 1)
                if nb is not None:
                    kids.append(nb)
                    made += 1
            # siblings are born overlapping: mark them as already in contact so their shared
            # birth is not mistaken for a fresh collision
            for i2, k1 in enumerate(kids):
                for k2 in kids[i2 + 1:]:
                    k1.contacts.add(k2.uid)
                    k2.contacts.add(k1.uid)
            if made:
                self.scatters += 1
                self.triggers.append({"t": self._tnow, "x": px, "y": py, "r": r_new,
                                      "gen": tier + 1, "gap": 0, "kind": "scatter"})

    # ---------------------------------------------------------------- the step
    def step(self, dt: float) -> None:
        t = self.t
        self._tnow = t
        g = self.g * dt
        vmax = self.v_max
        live = [b for b in self.balls if b.alive]
        for b in live:
            if g:
                b.vy += g
            an = self.aniso
            cg_eff = self.central_g
            if self.central_g:
                # radial in the STRETCHED metric; the y-component of the resulting force picks
                # up a 1/aniso factor, which is exactly what lets an orbit run taller than wide
                dx, dy = self.cx - b.x, (self.cy - b.y) / an
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    cg = self.central_g
                    if self.g_size_exp:
                        # A SMALLER ball is held less tightly, so it runs faster and straighter
                        # while the big ones sit in slow close orbits. Scaling the pull is the
                        # honest way to get that: it is a property of the body, not a speed
                        # multiplier bolted onto the integrator.
                        cg *= (b.r / self.r0) ** self.g_size_exp
                    cg_eff = cg
                    a = cg * dt / d
                    b.vx += dx * a
                    b.vy += dy * a / an
            track_r = 0.0
            if self.guide_a and self.guides:
                A, B, phi = self.guides[min(b.gen, len(self.guides) - 1)]
                rx, ry = b.x - self.cx, (b.y - self.cy) / an
                rr = math.hypot(rx, ry)
                if rr > 1e-6:
                    c, s = math.cos(phi), math.sin(phi)
                    u, v = rx * c + ry * s, -rx * s + ry * c      # into the ellipse's frame
                    th = math.atan2(v, u)
                    ct, st = math.cos(th), math.sin(th)
                    r_ell = (A * B) / math.hypot(B * ct, A * st)  # ellipse radius at this angle
                    track_r = r_ell
                    err = r_ell - rr
                    k = err / self.guide_soft
                    k = -1.0 if k < -1.0 else (1.0 if k > 1.0 else k)
                    ag = self.guide_a * k * dt
                    b.vx += (rx / rr) * ag
                    b.vy += (ry / rr) * ag / an
            bdir = self.dir_for(b.gen)
            if self.swirl_a and bdir:
                rx, ry = b.x - self.cx, (b.y - self.cy) / an
                rr = math.hypot(rx, ry)
                if rr > 1e-6:
                    tx = bdir * (-ry / rr)
                    ty = bdir * (rx / rr)
                    # measure and drive the tangential speed in the stretched metric too, or
                    # the drive pushes across the intended track instead of along it
                    vt = b.vx * tx + (b.vy / an) * ty
                    # Target the circular speed of the ball's OWN TRACK, not of wherever it
                    # currently happens to be, and for the pull IT actually feels.
                    # Two failures this avoids, both measured:
                    #  - using the unscaled central_g over-drives small spheres (g_size_exp=0.85
                    #    means a red feels ~52% of the pull), which put reds at mean radius 815
                    #    on a 560 track with only 73% of the swarm on screen;
                    #  - using the CURRENT radius feeds energy to anything already flung wide,
                    #    so it is driven faster the further out it gets - a runaway that took
                    #    reds to 1348 px from centre on a 490 px track.
                    want = math.sqrt(max(cg_eff, 1.0) * (track_r or rr)) * self.orbit_boost
                    if vt < want:
                        k = self.swirl_a * dt
                        b.vx += tx * k
                        b.vy += ty * k * an
            if self.drag:
                k = 1.0 - self.drag * dt
                b.vx *= k
                b.vy *= k
            sp2 = b.vx * b.vx + b.vy * b.vy
            if sp2 > vmax * vmax:
                k = vmax / math.sqrt(sp2)
                b.vx *= k
                b.vy *= k
            b.x += b.vx * dt
            b.y += b.vy * dt
        for b in live:
            for obs in self.obstacles:
                obs.collide(self, b, t)
        if len(live) > 1:
            self._collide_balls(live)

        # the moment the shell is fully down, start easing the tiers apart in depth
        if (self.t_break is None and self._destructibles
                and not any(o.alive for o in self._destructibles)):
            self.t_break = t
            self.triggers.append({"t": t, "x": self.cx, "y": self.cy, "r": self.r0,
                                  "gen": 0, "gap": 0, "kind": "planes"})

        # ---- regions, the headline trigger, scoring zones, and leaving the frame
        outer, inner = self._outer, self._inner
        replace: list[Ball] = []
        for b in live:
            if self.sensor_r > 0.0:
                dx, dy = b.x - self.cx, b.y - self.cy
                dist = math.hypot(dx, dy)
                if b.state != 2 and dist > self.sensor_r + 0.5 * b.r:
                    b.state = 2
                    ang = math.atan2(dy, dx)
                    self.triggers.append({"t": t, "x": b.x, "y": b.y, "r": b.r, "gen": b.gen,
                                          "gap": outer.gap_index(ang, t) if outer else 0,
                                          "ang": ang, "kind": "escape"})
                    self.flashes.append({"t": t, "x": b.x, "y": b.y, "kind": "trigger",
                                         "r": b.r})
                elif b.state == 2 and dist < self.sensor_r - 0.5 * b.r - 8.0:
                    b.state = 1
                elif inner is not None and b.state == 0 and dist > inner.radius + inner.h + 0.5 * b.r:
                    b.state = 1
                    self.inner_exits.append({"t": t, "x": b.x, "y": b.y, "gen": b.gen})
                elif inner is not None and b.state == 1 and dist < inner.radius - inner.h - 0.5 * b.r:
                    b.state = 0

            hit_zone = -1
            for zi, z in enumerate(self.zones):
                if math.hypot(b.x - z["x"], b.y - z["y"]) < z["r"] + b.r:
                    hit_zone = zi
                    if b.zone != zi:
                        mult = int(z.get("mult", 0))
                        self.triggers.append({"t": t, "x": b.x, "y": b.y, "r": b.r,
                                              "gen": b.gen, "gap": zi, "kind": "zone",
                                              "zone": zi, "mult": mult})
                        if not z.get("hidden"):
                            # A hidden zone fires its EVENT but draws nothing at all - the
                            # expanding shockwave is what put a ring back around the core after
                            # the outline was removed. With ~80 core grazes in a 30 s clip there
                            # was almost always one or two hoops on screen, which read exactly
                            # like the ring that was supposed to be gone.
                            self.flashes.append({"t": t, "x": z["x"], "y": z["y"],
                                                 "kind": "trigger", "r": z["r"] * 0.6})
                        if mult:
                            # A MULTIPLIER PIT: the ball is consumed and comes back as `mult`
                            # balls one tier down (smaller, next colour). The tier ladder is
                            # what keeps an Nx pit from flooding the frame - N balls at
                            # tier_shrink radius carry less total area than the parent whenever
                            # N * shrink^2 < 1, and the r_min floor stops it regardless.
                            b.alive = False
                            self.mult_hits += 1
                            child = max(self.r_min, b.r * self.tier_shrink)
                            nxt = min(b.gen + 1, self.n_tiers - 1)
                            self.spawn(mult, gen=nxt, r=child)
                    break
            # clearing every zone re-arms the ball, so a second visit sounds again
            b.zone = hit_zone
            if not b.alive:
                continue

            if self.wrap_sides:
                if b.x < -b.r:
                    b.x += self.width + 2 * b.r
                elif b.x > self.width + b.r:
                    b.x -= self.width + 2 * b.r

            if self.kill_below and b.y - b.r > self.height:
                b.alive = False
                self.exits_bottom += 1
                if self.bottom_triggers:
                    self.triggers.append({"t": t, "x": b.x, "y": self.height, "r": b.r,
                                          "gen": b.gen, "gap": 0, "kind": "bottom"})
                    self.flashes.append({"t": t, "x": b.x, "y": self.height,
                                         "kind": "trigger", "r": b.r})
                replace.append(b)

        # Every ball that leaves the FRAME is replaced according to `respawn`.
        #
        # THE SIZE GATE (`split_min_r`) is what makes "split" terminate. Splitting 1 -> 2 on
        # every exit is EXPONENTIAL in the number of escape cycles, and shrinking the children
        # does not slow it down - it speeds it up, because a smaller ball clears a gap more
        # easily. Measured over 45 s: 116 exits, and no value of `max_balls` fixes it (raising
        # the cap to 60 gave 189 exits and 130 spawns silently dropped). So the population is
        # bounded by GEOMETRY instead: a ball splits in two only while its children are still
        # big enough to matter, and below that it is replaced one-for-one. The count then
        # plateaus on its own and the cap never has to bite.
        for b in replace:
            if self.respawn == "none":
                continue
            if self.respawn == "same":
                self.spawn(1, gen=b.gen, r=b.r)
            elif self.respawn == "fresh":
                self.spawn(1, gen=0, r=self.r0)
            else:                                   # "split"
                child = max(self.r_min, b.r * self.shrink)
                n = self.split if child >= self.split_min_r else 1
                self.spawn(n, gen=b.gen + 1, r=child)
        # compact whenever ANYTHING died - a multiplier pit and a scatter both consume balls
        # without going through `replace`, and dead entries would otherwise accumulate forever
        if replace or not all(b.alive for b in self.balls):
            self.balls = [b for b in self.balls if b.alive]

        # ---- DECAY: an eligible ball climbs one tier UP, in place. Mutating rather than
        # respawning is deliberate: position and velocity carry over, so the sphere visibly
        # grows and changes colour without teleporting out of its orbit.
        if self.decay_rate and self.decay_after:
            p = self.decay_rate * dt
            for b in self.balls:
                if (b.alive and b.gen == self.decay_tier and b.gen > 0
                        and (t - b.born) >= self.decay_after
                        and self.rng.random() < p):
                    b.gen -= 1
                    b.r = min(self.r0, b.r / self.tier_shrink) if self.tier_shrink else b.r
                    b.born = t
                    self.decays += 1
                    self.triggers.append({"t": t, "x": b.x, "y": b.y, "r": b.r,
                                          "gen": b.gen, "gap": 0, "kind": "decay"})
                    self.flashes.append({"t": t, "x": b.x, "y": b.y, "kind": "spawn",
                                         "r": b.r})

        # a steady drip, for scenes that rain rather than split
        if self.spawn_every > 0.0 and (t - self._last_spawn) >= self.spawn_every:
            self._last_spawn = t
            if self.n_live() < self.max_balls:
                self.spawn(1, gen=0, r=self.r0)

        self.t = t + dt

    def n_live(self) -> int:
        return sum(1 for b in self.balls if b.alive)
