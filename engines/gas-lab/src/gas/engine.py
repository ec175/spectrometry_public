# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""engine.py - an ENGINE is data. Boxes in local frames, pistons, valves, igniters, injectors.

The 2026-09-15 engine hard-wired one cylinder, one piston and five boxes into module constants.
The 2026-09-20 pass (V-twin, radial, two-stroke, opposed piston ...) needs cylinders at angles,
walls that move at the top as well as the bottom, ports uncovered by a piston rather than a
valve, and fuel that arrives from an injector. All of that is still ONE box graph and ONE
particle array - the physics in world.py is unchanged in kind - so the generalisation is a
description layer, and this is it.

THE FRAME TRICK. Every cylinder GROUP simulates in its own axis-aligned LOCAL frame (the
cylinder pointing up, the crank below it), and is rotated into the world only when drawn. A
radial's five cylinders are five groups sharing one particle array, so the kernel-launch cost -
the thing that bounds this solver - is the same as for one big cylinder. Groups never exchange
gas, so a port never crosses a frame.

Local coordinates must stay inside x in [0, 2560), y in [-256, 2304): the DSMC cell key in
world.py clips there, and a box outside that range would dump its particles into one edge cell.

VOCABULARY
  Box       an axis-aligned gas region. Up to two ports per side. A side may be OPEN (a
            reservoir at ambient), and the floor and/or ceiling may be a PISTON.
  Port      an interval on one side shared with the neighbouring box, gated by a VALVE
            (lift, from the middle out), or clipped by a piston COVER (a wall port that the
            piston skirt uncovers), or always open.
  Piston    a callable of time -> (crown y, crown velocity) in the group's local frame. `up`
            says which way the crown faces. `height` is crown-to-skirt for a crankcase ceiling.
  Valve     opens/closes on a per-cylinder CLOCK (degrees of a 720 or 360 cycle), or a reed
            that opens on its crankcase's own measured pressure (world.py sets `last`).
  Igniter   spark at an angle, or autoignition above a temperature (optionally only once fuel
            is present - a diesel). The flame front is kinematic from `origin` and covers a
            tuple of boxes, so a flathead's L-shaped chamber burns as one.
  Injector  fuel sprayed from a tip over a crank window: a diesel.
  Drive     the crank speed schedule every clock and piston reads: base speed, then a ramp
            to `mult` x and a hold. All timing is in crank DEGREES, so it survives the ramp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config as C

TOP, BOT, LEFT, RIGHT = 0, 1, 2, 3
SIDES = ("top", "bot", "left", "right")


# ------------------------------------------------------------------ frames
@dataclass
class Group:
    """A local frame. world = at + R(angle) . (local - pivot). Angle is clockwise on screen
    (y down), so +30 leans the cylinder's head to the RIGHT."""
    name: str = "main"
    angle_deg: float = 0.0
    pivot: tuple = (540.0, 960.0)
    at: tuple = (540.0, 960.0)

    @property
    def cs(self):
        a = math.radians(self.angle_deg)
        return math.cos(a), math.sin(a)

    def to_world(self, x, y):
        """Works on floats and on numpy/cupy arrays alike."""
        c, s = self.cs
        dx, dy = x - self.pivot[0], y - self.pivot[1]
        return (self.at[0] + c * dx - s * dy, self.at[1] + s * dx + c * dy)

    def to_local(self, X, Y):
        c, s = self.cs
        dx, dy = X - self.at[0], Y - self.at[1]
        return (self.pivot[0] + c * dx + s * dy, self.pivot[1] - s * dx + c * dy)


# ------------------------------------------------------------------ parts of the graph
@dataclass
class Port:
    side: int                     # TOP/BOT/LEFT/RIGHT
    a: float
    b: float                      # interval along the edge (x for top/bot, y for left/right)
    dest: int
    gate: int = -1                # valve index, -1 = always open
    cover: tuple | None = None    # (piston index, 'crown'|'under'): clipped to the gas side


@dataclass
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    group: int = 0
    kind: str = "duct"            # 'cyl' | 'case' | 'duct'
    role: str = ""                # 'intake' | 'exhaust' | '' - for the tallies and the seed
    floor: int = -1               # piston index whose crown is this box's floor (gas above)
    ceil: int = -1                # piston index: ceiling. `ceil_face` says crown or underside
    ceil_face: str = "crown"
    open: tuple = (False, False, False, False)
    boost: float = 1.0            # reservoir density, in ambients
    ports: list = field(default_factory=list)
    label: str = ""
    res_T: float = 0.0            # reservoir / seed temperature, K (0 = T0)
    res_mix: dict = None          # reservoir / seed species fractions (None = air + fuel_frac)
    side_boost: dict = None       # per-side reservoir density {side: boost}: a runner whose two
                                  # ends sit at different pressures is what makes it flow

    @property
    def w(self):
        return self.x1 - self.x0


@dataclass
class Piston:
    group: int
    fn: object                    # fn(t) -> (crown_y, dcrown/dt), local frame
    up: bool = True               # crown faces up (gas above it)
    height: float = 190.0         # crown -> skirt bottom
    label: str = ""


@dataclass
class Valve:
    clock: object                 # fn(t) -> cycle degrees
    opens: float = 0.0
    closes: float = 0.0
    mode: str = "cam"             # 'cam' | 'fn' | 'reed'
    lift_fn: object = None        # mode 'fn': lift(t) in 0..1
    cycle: float = 720.0
    label: str = ""
    box: int = -1                 # mode 'reed': the box whose pressure opens it
    p_ref: float = 1.0            # mode 'reed': opens when that box is below this, in ambients
    last: float = 0.0             # mode 'reed': the lift world.py measured this substep

    def lift(self, t):
        if self.mode == "reed":
            return self.last          # a reed is PRESSURE driven; world.py sets this
        if self.mode == "fn":
            return float(self.lift_fn(t))
        return cam_lift(self.clock(t), self.opens, self.closes, self.cycle)


@dataclass
class Igniter:
    boxes: tuple                  # boxes the front may burn in (the cylinder, plus a chamber)
    origin: tuple                 # local (x, y) the front grows from
    clock: object
    spark_deg: float = 340.0
    auto_K: float = 0.0           # > 0: compression ignition at this temperature
    need_fuel: bool = False       # autoignite only once fuel is present (diesel)
    flame_deg: float = 20.0       # front radius growth, local px per CRANK DEGREE, so the
                                  # burn takes the same crank angle at any speed - roughly
                                  # what a real turbulent flame does
    window: tuple = (120.0, 420.0)   # cycle degrees in which autoignition may trigger
    rearm: float = 0.5            # fraction of a cycle after which the front is dropped
    label: str = ""
    region: object = None         # optional fn(world) -> particle mask the front may burn in
                                  # (a Wankel chamber is bounded by two apexes, not a box)


@dataclass
class Injector:
    box: int
    tip: tuple
    angle_deg: float = 90.0       # direction of the spray, degrees clockwise from +x (90 = down)
    spread_deg: float = 40.0
    speed: float = 1.4            # in units of V_TH
    clock: object = None
    start_deg: float = 350.0
    end_deg: float = 378.0
    n_total: int = 2000           # particles per injection event
    species: int = C.FUEL
    label: str = ""


@dataclass
class Engine:
    name: str
    rev_seconds: float            # the BASE speed, s per revolution
    drive: object = None          # the Drive every clock and piston reads (set by builders)
    cycle_deg: float = 720.0      # the period of the whole engine, in crank degrees
    fuel_frac: float = 0.30       # fuel fraction of the INTAKE reservoir gas
    groups: list = field(default_factory=lambda: [Group()])
    boxes: list = field(default_factory=list)
    pistons: list = field(default_factory=list)
    valves: list = field(default_factory=list)
    igniters: list = field(default_factory=list)
    injectors: list = field(default_factory=list)
    mfp: float = 90.0
    q_rxn: float = 7.5e7
    draw_static: object = None    # fn(canvas, engine)
    draw_moving: object = None    # fn(canvas, engine, t)
    blurb: str = ""
    notes: dict = field(default_factory=dict)   # anything the drawing wants to remember
    # 2026-09-25 parts set: shaped solids (solids.Body), circular port reservoirs (solids.Hole)
    # and named measurement regions for the gate (see pgate.py)
    bodies: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    probes: list = field(default_factory=list)

    # ---- helpers used by world.py and the gate ----
    @property
    def period(self) -> float:
        return self.rev_seconds * self.cycle_deg / 360.0

    def cyl_boxes(self):
        return [i for i, b in enumerate(self.boxes) if b.kind == "cyl"]

    def add_box(self, box: Box) -> int:
        self.boxes.append(box)
        return len(self.boxes) - 1

    def connect(self, i, j, side_i, a, b, gate=-1, cover=None):
        """Add the port on box i's `side_i` into box j, and the matching one back."""
        back = {TOP: BOT, BOT: TOP, LEFT: RIGHT, RIGHT: LEFT}[side_i]
        self.boxes[i].ports.append(Port(side_i, a, b, j, gate, cover))
        self.boxes[j].ports.append(Port(back, a, b, i, gate, cover))

    def check(self):
        """Fail loudly on a graph that cannot work. Every one of these was a real way to lose
        a batch: a port not shared by both boxes teleports particles out of the domain; a box
        out of the DSMC key range dumps its gas into one edge cell; a piston past the head
        inverts the cylinder."""
        bx = self.boxes
        for i, b in enumerate(bx):
            assert b.x0 < b.x1 and b.y0 < b.y1, f"box {i} {b.label!r} is inside out"
            # a part's reservoir strip may run a little off the LEFT of the page: its particles
            # share the edge collision cells there, off-screen, which is harmless
            xmin = -200.0 if self.notes.get("part") else 0.0
            assert xmin <= b.x0 and b.x1 < 2560.0 and -256.0 <= b.y0 and b.y1 < 2304.0, \
                f"box {i} {b.label!r} leaves the DSMC key range"
            assert len([p for p in b.ports if p.side == TOP]) <= 2, f"box {i}: >2 top ports"
            assert len([p for p in b.ports if p.side == BOT]) <= 2, f"box {i}: >2 bottom ports"
            assert len([p for p in b.ports if p.side == LEFT]) <= 2, f"box {i}: >2 left ports"
            assert len([p for p in b.ports if p.side == RIGHT]) <= 2, f"box {i}: >2 right ports"
            for p in b.ports:
                d = bx[p.dest]
                assert d.group == b.group, f"port {i}->{p.dest} crosses a frame"
                if p.side in (TOP, BOT):
                    edge = b.y0 if p.side == TOP else b.y1
                    other = d.y1 if p.side == TOP else d.y0
                    assert abs(edge - other) < 1e-6, \
                        f"port {i}->{p.dest} ({b.label}->{d.label}): edges not shared"
                    assert max(b.x0, d.x0) - 1e-6 <= p.a < p.b <= min(b.x1, d.x1) + 1e-6, \
                        f"port {i}->{p.dest}: interval outside both boxes"
                else:
                    edge = b.x0 if p.side == LEFT else b.x1
                    other = d.x1 if p.side == LEFT else d.x0
                    assert abs(edge - other) < 1e-6, \
                        f"port {i}->{p.dest} ({b.label}->{d.label}): edges not shared"
                    assert max(b.y0, d.y0) - 1e-6 <= p.a < p.b <= min(b.y1, d.y1) + 1e-6, \
                        f"port {i}->{p.dest}: interval outside both boxes"
            # a box with a moving wall must not also be open on that side
            if b.floor >= 0:
                assert not b.open[BOT], f"box {i}: piston floor cannot be open"
            if b.ceil >= 0:
                assert not b.open[TOP], f"box {i}: piston ceiling cannot be open"
        # boxes in one group must not overlap
        for i in range(len(bx)):
            for j in range(i + 1, len(bx)):
                a, b = bx[i], bx[j]
                if a.group != b.group:
                    continue
                # a cylinder over a crankcase: the same piston is one's floor and the
                # other's ceiling, so their STATIC extents overlap while the gas never does
                if (a.floor >= 0 and a.floor == b.ceil) or (b.floor >= 0 and b.floor == a.ceil):
                    continue
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                assert not (ox > 1e-6 and oy > 1e-6), \
                    f"boxes {i} {a.label!r} and {j} {b.label!r} overlap"
        # pistons: the crown must stay inside its cylinder over a whole period
        for i, b in enumerate(bx):
            if b.floor >= 0 and b.kind == "cyl":
                p = self.pistons[b.floor]
                lo = min(p.fn(t)[0] for t in _samples(self.period))
                top = b.y0
                if b.ceil >= 0:
                    q = self.pistons[b.ceil]
                    top = max(q.fn(t)[0] for t in _samples(self.period))
                assert lo - top > 30.0, \
                    f"box {i} {b.label!r}: clearance {lo - top:.0f} px too thin to see gas in"
        if not self.notes.get("part"):
            assert self.pistons, "an engine needs a piston"
            assert self.cyl_boxes(), "an engine needs a cylinder"
        return True

    # ---- geometry the draw and the frame-fit check need ----
    def world_corners(self, i):
        b = self.boxes[i]
        g = self.groups[b.group]
        return [g.to_world(x, y) for x, y in ((b.x0, b.y0), (b.x1, b.y0),
                                              (b.x1, b.y1), (b.x0, b.y1))]

    def frame_fit(self, margin=0.0):
        """Which boxes reach outside the reference frame, and by how much. A box is ALLOWED
        out only through an open side (a reservoir leaving the frame). Returns a list of
        (box index, label, overhang px) for the violations."""
        bad = []
        for i, b in enumerate(self.boxes):
            if any(b.open):
                continue
            for x, y in self.world_corners(i):
                over = max(-x + margin, x - C.REF_W + margin, -y + margin, y - C.REF_H + margin)
                if over > 0:
                    bad.append((i, b.label, over))
                    break
        return bad


def _samples(period, n=720):
    return [period * k / n for k in range(n)]


# ------------------------------------------------------------------ the drive
class Drive:
    """THE CRANK SPEED SCHEDULE - one per engine, shared by every clock, piston and cam.

    Clancy, 2026-09-20: "the pistons should start to accelerate at the start of the video up
    until halfway through, and maintain a high speed ... at least three times the base speed".
    So the crank turns at `rev0` s/rev through the warm-up, then from t = warm ramps
    (smoothstep) to `mult` x over `ramp` seconds and holds. Everything that needs an angle
    asks for theta(t) here, so the valves, the spark, the injector window, the cam lobes and
    the drawn crank all stay in step through the ramp. The integral is closed-form."""

    def __init__(self, rev0, warm=0.0, ramp=0.0, mult=1.0):
        self.rev0 = float(rev0)
        self.configure(warm, ramp, mult)

    def configure(self, warm=0.0, ramp=0.0, mult=1.0):
        self.warm, self.ramp, self.mult = float(warm), float(ramp), float(mult)

    def s(self, t):
        """Speed multiplier at time t."""
        u = t - self.warm
        if u <= 0.0 or self.mult == 1.0:
            return 1.0
        if self.ramp <= 0.0 or u >= self.ramp:
            return self.mult
        x = u / self.ramp
        return 1.0 + (self.mult - 1.0) * (x * x * (3.0 - 2.0 * x))

    def _I(self, u):
        """Integral of s over [0, u] measured from the start of the ramp."""
        if u <= 0.0 or self.mult == 1.0:
            return max(u, 0.0)
        if self.ramp <= 0.0:
            return self.mult * u
        ur = min(u, self.ramp)
        x = ur / self.ramp
        out = ur + (self.mult - 1.0) * self.ramp * (x ** 3 - 0.5 * x ** 4)
        if u > self.ramp:
            out += self.mult * (u - self.ramp)
        return out

    def revs(self, t):
        if t <= self.warm:
            return t / self.rev0
        return (self.warm + self._I(t - self.warm)) / self.rev0

    def theta(self, t):
        return 2.0 * math.pi * self.revs(t)

    def deg(self, t):
        return 360.0 * self.revs(t)

    def omega(self, t):
        return 2.0 * math.pi / self.rev0 * self.s(t)


# ------------------------------------------------------------------ clocks and lifts
def clock(drive, cycle=720.0, phase=0.0):
    """Crank-degree clock for one cylinder: fn(t) -> degrees in [0, cycle). `phase` is added,
    so a V-twin's rear cylinder or a radial's third gets its own zero."""
    def f(t):
        return (drive.deg(t) + phase) % cycle
    return f


def cam_lift(deg, opens, closes, cycle=720.0):
    """Poppet lift 0..1 - half a sine across the open window, zero slope at both seats so the
    port never snaps open (a step in the gate is a pressure pop, and a pressure pop reads as a
    flash - OUTPUT_RULES). The cam lobe in draw.py is generated FROM this curve, so the lobe
    you see is the lift the physics uses."""
    d = float(deg)
    while d < opens:
        d += cycle
    while d - cycle >= opens:
        d -= cycle
    if d < opens or d > closes:
        return 0.0
    return math.sin(math.pi * (d - opens) / (closes - opens))


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# ------------------------------------------------------------------ kinematics
def slider_crank(crank_y, R, L, drive, phase_deg=0.0, up=True):
    """Exact slider-crank in the local frame. `up`: the crank is BELOW the piston (normal).
    Returns fn(t) -> (crown y, crown velocity), analytic - a finite difference here puts a step
    in the moving-wall reflection every substep and the gas picks that up as noise. The angle
    and its rate come from the Drive, so the ramp is exact too."""
    ph = math.radians(phase_deg)
    sgn = 1.0 if up else -1.0

    def f(t):
        th = drive.theta(t) + ph
        s, c = math.sin(th), math.cos(th)
        root = math.sqrt(max(1e-9, L * L - (R * s) ** 2))
        pos = R * c + root
        vel = -drive.omega(t) * (R * s + (R * R) * s * c / root)       # d(pos)/dt
        return crank_y - sgn * pos, -sgn * vel
    return f


def peak_speed(fn, period, n=720):
    return max(abs(fn(period * k / n)[1]) for k in range(n))


class RadialKinematics:
    """Master rod + articulated rods. Cylinder 0 carries the master rod on the crankpin; the
    others hang off knuckle pins on the master rod's big end, so their pistons do NOT follow a
    pure slider-crank - each is solved exactly from its knuckle pin, and the rods you see are
    the rods the pistons follow. Local frame per cylinder: the cylinder points up with the
    crank centre at `centre_local`; group j is rotated by 360 j / n about it."""

    def __init__(self, n, R, L_master, r_knuckle, L_art, drive, centre_local=(540.0, 960.0),
                 pin_to_crown=60.0):
        self.n, self.R, self.Lm, self.rk, self.La = n, R, L_master, r_knuckle, L_art
        self.drive = drive
        self.c = centre_local
        self.h = pin_to_crown
        self.phi = [360.0 * j / n for j in range(n)]

    def crank_pin_world(self, t, C):
        th = self.drive.theta(t)
        return (C[0] + self.R * math.sin(th), C[1] - self.R * math.cos(th))

    def master_pin_s(self, t):
        th = self.drive.theta(t)
        s, c = math.sin(th), math.cos(th)
        return self.R * c + math.sqrt(max(1e-9, self.Lm ** 2 - (self.R * s) ** 2))

    def knuckles_local(self, t):
        """Knuckle pin positions in the MASTER cylinder's local frame (up = -y), relative to
        the crank centre."""
        th = self.drive.theta(t)
        P = (self.R * math.sin(th), -self.R * math.cos(th))
        M = (0.0, -self.master_pin_s(t))
        d = (M[0] - P[0], M[1] - P[1])
        L = math.hypot(*d)
        d = (d[0] / L, d[1] / L)
        out = []
        for j in range(self.n):
            b = math.radians(self.phi[j])
            # rotate the rod direction by the knuckle's angular offset (clockwise on screen)
            kx = d[0] * math.cos(b) - d[1] * math.sin(b)
            ky = d[0] * math.sin(b) + d[1] * math.cos(b)
            out.append((P[0] + self.rk * kx, P[1] + self.rk * ky))
        return out

    def pin_s(self, j, t):
        """Piston-pin distance from the crank centre along cylinder j's axis."""
        if j == 0:
            return self.master_pin_s(t)
        K = self.knuckles_local(t)[j]
        # into cylinder j's frame: rotate by -phi_j
        b = math.radians(-self.phi[j])
        kx = K[0] * math.cos(b) - K[1] * math.sin(b)
        ky = K[0] * math.sin(b) + K[1] * math.cos(b)
        return -ky + math.sqrt(max(1e-9, self.La ** 2 - kx * kx))

    def piston_fn(self, j):
        h = 1e-4

        def f(t):
            s = self.pin_s(j, t)
            v = (self.pin_s(j, t + h) - self.pin_s(j, t - h)) / (2 * h)
            return self.c[1] - s - self.h, -v
        return f

    def stroke(self, j, period):
        ss = [self.pin_s(j, period * k / 720) for k in range(720)]
        return min(ss), max(ss)
