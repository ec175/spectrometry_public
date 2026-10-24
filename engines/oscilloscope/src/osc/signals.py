# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""signals.py — the scenes. Each scene yields, for a given time t, one or more beam paths.

A "beam" is (pts, gain, closed): pts = Nx2 in normalised scope space [-1,1], gain = brightness
(fade in/out), closed = join last->first. A scene has `.duration` and `.beams(t) -> list`.

Scenes here
-----------
  wind_up        the requested one: dot -> growing precessing "changing ellipses" that speed
                 up and grow harmonics (wind UP) -> collapse back down -> settle to the normal
                 calibration pulse (wind DOWN to the standard scope trace).
  lissajous_tour eye-candy: morph continuously through the classic Lissajous frequency ratios.
  pulse_to_shape the end-goal demo: the standard pulse trace morphing into circle/square/
                 triangle/star geometries and back.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# --- small 3D helpers (for the electron-orbital wireframes) -----------------
def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def _orient(pts3, axis):
    """Rotate a +z-pointing shape so it points along `axis` (Rodrigues)."""
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, a)
    c = np.dot(z, a)
    if np.linalg.norm(v) < 1e-9:
        return pts3 * (1 if c > 0 else -1)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return pts3 @ R.T


# --- easing / envelope helpers ---------------------------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ramp_hold(t, t0, t1, t2, t3):
    """0 before t0, smooth 0->1 over [t0,t1], hold 1 over [t1,t2], smooth 1->0 over [t2,t3]."""
    up = smoothstep(t0, t1, t)
    down = 1.0 - smoothstep(t2, t3, t)
    return float(min(up, down))


def _cal_wave(n, cycles=3.0, amp=0.55, span=0.92):
    """A rounded square-wave calibration pulse: the iconic default scope trace. Returns Nx2."""
    x = np.linspace(-span, span, n)
    phase = (x / span + 1) * 0.5 * cycles * TWO_PI
    # tanh-rounded square wave (finite edge slope, like a real bandwidth-limited pulse)
    y = amp * np.tanh(2.5 * np.sin(phase))
    return np.stack([x, y], axis=1)


def _sample_polygon(verts, n, closed=True):
    """Evenly sample n points along the perimeter of a polygon (list of (x,y))."""
    v = np.asarray(verts, dtype=np.float32)
    if closed:
        v = np.vstack([v, v[:1]])
    seg = np.diff(v, axis=0)
    d = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    s = np.linspace(0, total, n, endpoint=False)
    out = np.empty((n, 2), np.float32)
    for i, si in enumerate(s):
        k = np.searchsorted(cum, si, side="right") - 1
        k = min(k, len(seg) - 1)
        f = (si - cum[k]) / (d[k] + 1e-9)
        out[i] = v[k] + f * seg[k]
    return out

# ===========================================================================
class WindUp:
    """Dot -> growing, precessing, breathing ellipses that speed up and grow harmonics
    (WIND UP) -> collapse -> settle into the standard calibration pulse (WIND DOWN)."""

    N = 1500

    def __init__(self, duration=14.0):
        self.duration = duration
        # timeline knots
        self.t0, self.t1 = 0.3, 4.5      # wind up
        self.t2 = 6.2                    # peak hold end
        self.t3 = 9.8                    # collapsed to baseline
        self.t4 = 10.8                   # crossfaded to sweep
        # precompute cumulative spin angle rho(t): omega grows with energy, so the figure
        # visibly accelerates as it winds up and decelerates as it winds down.
        tt = np.linspace(0, duration, 6000)
        e = np.array([ramp_hold(t, self.t0, self.t1, self.t2, self.t3) for t in tt])
        omega = TWO_PI * (0.15 + 1.35 * e)      # rev/s -> rad/s spin of the whole figure
        dt = tt[1] - tt[0]
        self._tt = tt
        self._rho = np.concatenate([[0], np.cumsum(omega[:-1] * dt)])
        self._e = e

    def _energy(self, t):
        return float(np.interp(t, self._tt, self._e))

    def _rho_at(self, t):
        return float(np.interp(t, self._tt, self._rho))

    def _figure(self, t):
        e = self._energy(t)
        theta = np.linspace(0, TWO_PI, self.N)
        # breathing ellipse: openness oscillates so the ellipse opens/closes ("changing ellipses")
        openness = 0.5 + 0.5 * np.sin(TWO_PI * 0.35 * t) * e
        delta = (np.pi / 2) * openness
        A = 0.82 * e                       # amplitude grows as it winds up
        ry = 0.85 * e                      # y-scale -> 0 collapses figure to a flat line
        # harmonics grow with energy -> clean ellipse at low e, busy multi-lobe at peak
        hx = 0.35 * e * np.cos(2 * theta + 0.7 * t)
        hy = 0.30 * e * np.sin(3 * theta + 1.1 * t)
        x = A * (np.cos(theta) + hx)
        y = ry * (np.sin(theta + delta) + hy)
        # rotate (precession); spin fades out as energy dies so it settles horizontal
        rho = self._rho_at(t) * (0.3 + 0.7 * e)
        c, s = np.cos(rho), np.sin(rho)
        X = x * c - y * s
        Y = x * s + y * c
        return np.stack([X, Y], axis=1)

    def beams(self, t):
        out = []
        fig_gain = 1.0 - smoothstep(self.t3, self.t4, t)
        if fig_gain > 0.01:
            out.append((self._figure(t), 1.0 * fig_gain, False))
        sweep_gain = smoothstep(self.t3, self.t4, t)
        if sweep_gain > 0.01:
            out.append((_cal_wave(self.N), 1.0 * sweep_gain, False))
        return out
# ===========================================================================
class LissajousTour:
    """Morph continuously through the classic Lissajous ratios; the phase slowly rotates so
    each figure spins/opens as the frequencies drift into the next ratio."""

    N = 1600
    RATIOS = [(1, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 4), (1, 1)]

    def __init__(self, duration=14.0):
        self.duration = duration

    def beams(self, t):
        seg = self.duration / (len(self.RATIOS) - 1)
        k = min(int(t / seg), len(self.RATIOS) - 2)
        f = smoothstep(0, 1, (t - k * seg) / seg)
        fx = (1 - f) * self.RATIOS[k][0] + f * self.RATIOS[k + 1][0]
        fy = (1 - f) * self.RATIOS[k][1] + f * self.RATIOS[k + 1][1]
        theta = np.linspace(0, TWO_PI, self.N)
        delta = 0.5 * np.pi * np.sin(0.25 * t)      # slow phase sweep -> figures breathe
        x = 0.82 * np.sin(fx * theta + delta)
        y = 0.82 * np.sin(fy * theta)
        return [(np.stack([x, y], axis=1), 1.0, False)]
# ===========================================================================
class PulseToShape:
    """The end-goal demo: the standard calibration pulse morphs into circle/square/triangle/
    star geometries and back to the pulse."""

    N = 1400

    def __init__(self, duration=16.0):
        self.duration = duration
        self.targets = [
            ("pulse", lambda: _cal_wave(self.N)),
            ("circle", lambda: self._circle()),
            ("square", lambda: _sample_polygon(
                [(-.7, -.7), (.7, -.7), (.7, .7), (-.7, .7)], self.N)),
            ("triangle", lambda: _sample_polygon(
                [(0, .8), (.75, -.55), (-.75, -.55)], self.N)),
            ("star", lambda: self._star()),
            ("pulse", lambda: _cal_wave(self.N)),
        ]
        self._cache = [f() for _, f in self.targets]

    def _circle(self):
        a = np.linspace(0, TWO_PI, self.N, endpoint=False)
        return np.stack([0.78 * np.cos(a), 0.78 * np.sin(a)], axis=1)

    def _star(self):
        pts = []
        for i in range(5):
            outer = i * TWO_PI / 5 - np.pi / 2
            inner = outer + TWO_PI / 10
            pts.append((0.8 * np.cos(outer), 0.8 * np.sin(outer)))
            pts.append((0.32 * np.cos(inner), 0.32 * np.sin(inner)))
        return _sample_polygon(pts, self.N)

    def beams(self, t):
        n = len(self.targets)
        seg = self.duration / (n - 1)
        k = min(int(t / seg), n - 2)
        f = smoothstep(0, 1, (t - k * seg) / seg)
        # hold each shape for the first ~45% of its slot, morph over the rest
        hold = 0.45
        f = 0.0 if f < hold else (f - hold) / (1 - hold)
        pts = (1 - f) * self._cache[k] + f * self._cache[k + 1]
        return [(pts, 1.0, False)]
# ===========================================================================
class OrbitTrace:
    """The intended mechanic: ONE bright pulse point orbits the centre at a steady rate; the
    ONLY thing that changes is how hard it's pulled in / pushed out (the radial "attraction/
    repulsion" intensity) as a function of angle. Its phosphor trail is what draws the shapes.

    Renderer treats this as a TRAIL scene (it exposes pos(ts), not beams(t)): the head is sub-
    frame stepped so the fast dot leaves a continuous, decaying comet trail. `decay` is high so
    roughly one full lap of trail stays lit — you read the whole shape with the bright head
    running around it. Radius as a function of angle is blended between shape profiles over time,
    so the pull deepens from a circle into a triangle, square, star, and back.
    """

    # trail-scene knobs (read by render.render_scene)
    decay = 0.935          # afterglow ≈ one orbit of trail
    substeps = 120         # sub-frame samples of the head path (continuous fast trail)
    trail_gain = 0.7       # the trail is a *light* trail; the head is the bright thing
    trail_width = 3
    head_gain = 1.0

    def __init__(self, duration=16.0, omega=1.15, R0=0.72):
        self.duration = duration
        self.omega = omega          # orbits per second (the point moves fast)
        self.R0 = R0
        # radial "shape profiles" r(theta) in ~[.., 1]; the attraction pulls the orbit onto these.
        self.stages = [
            self._circle,                                   # relaxed: pure orbit
            lambda th: self._poly(th, 3, phase=np.pi / 2),  # triangle (point up)
            lambda th: self._poly(th, 4, phase=np.pi / 4),  # square
            lambda th: self._star(th, 5, inner=0.42, phase=np.pi / 2),  # star
            self._circle,                                   # release back to a circle
        ]

    # --- radial shape profiles (attraction targets) ---
    @staticmethod
    def _circle(th):
        return np.ones_like(th)

    @staticmethod
    def _poly(th, n, phase=0.0):
        # regular n-gon circumradius-normalised: vertices touch r=1, edge midpoints pulled in.
        a = np.mod(th + phase, TWO_PI / n) - np.pi / n
        return np.cos(np.pi / n) / np.cos(a)

    @staticmethod
    def _star(th, p, inner=0.42, phase=0.0, sharp=0.55):
        spike = (0.5 + 0.5 * np.cos(p * (th + phase))) ** sharp
        return inner + (1.0 - inner) * spike

    def pos(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        theta = TWO_PI * self.omega * ts
        prof = np.stack([f(theta) for f in self.stages])    # (S, M)
        n = len(self.stages)
        seg = self.duration / (n - 1)
        k = np.clip((ts // seg).astype(int), 0, n - 2)
        f = smoothstep(0, 1, (ts - k * seg) / seg)
        idx = np.arange(len(ts))
        r = self.R0 * ((1 - f) * prof[k, idx] + f * prof[k + 1, idx])
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
# ===========================================================================
class Dimensional:
    """Two phases, drawn by the moving beam + its fading trace (no visible dot):

      2. NODES    opens by WINDING UP from the origin into a 2-lobe figure, then a demonstration of
                  how angular nodes reflect through the origin: 2 -> 3 -> 4 -> 5 lobes, each morphing
                  continuously into the next (the trace never disconnects), rotating and dynamically
                  stretching, accelerating. After the 5-lobe rotates a while, everything COLLAPSES
                  back to the origin point.
      3. ORBITALS out of that origin point the electron ORBITALS emerge (scale up from the centre)
                  and tumble in 3D: s sphere -> p dumbbell -> d cloverleaf (4 lobes) -> d_z2
                  (two lobes + ring). All one green colour.

    Phase 2 is the single-head trail model; phase 3 draws several wireframe traces per frame.
    """

    decay = 0.90
    substeps = 700
    trail_width = 3
    trail_gain = 0.55      # keep the node trace from over-accumulating into a white blob
    head_gain = 0.0        # no visible dot anywhere

    EMERGE = 0.7           # per-orbital: seconds to scale up out of the origin
    COLLAPSE = 0.7         # per-orbital: seconds to collapse back to the origin

    def __init__(self):
        # node-count transition windows (2->3 is quick), then a fast final collapse to the origin
        self._node_tx = [(2.6, 3.0), (4.6, 5.2), (7.0, 7.6)]
        self.t_collapse = (10.0, 11.0)
        self.t_end2 = 11.0       # phase 2 (nodes) ends / phase 3 (orbitals) begins
        self._orbitals = ["s", "p", "d4", "dz2"]
        self.scale = 0.8
        self.orbital_span = 6.0                          # each orbital: emerge -> hold -> collapse
        self.tail = 2.2                                  # let the trail decay to black -> loops
        orb_end = self.t_end2 + len(self._orbitals) * self.orbital_span
        self.duration = orb_end + self.tail
        # ω(t) rev/s: winds up from slow, accelerating through the node progression.
        tt = np.linspace(0, self.duration, 9000)
        w = 0.3 + 12.0 * smoothstep(1.0, 9.5, tt)
        self._tt = tt
        self._phi = np.concatenate([[0], np.cumsum(TWO_PI * w[:-1] * (tt[1] - tt[0]))])

    def _PHI(self, ts):
        return np.interp(ts, self._tt, self._phi)

    # --- grid schedule: cartesian(0) -> axes-only(1) -> back to cartesian(0) at the end ------
    def grid_state(self, t):
        up = smoothstep(1.5, 3.0, t)                       # cart -> axes
        down = smoothstep(self.duration - 2.0, self.duration - 0.5, t)  # axes -> cart (loop back)
        return up - down

    # --- phase 2 node shapes: n-lobe patterns reflecting through the origin ---
    @staticmethod
    def _node_xy(phi):
        """(x,y) for 2,3,4,5-lobe patterns as arrays (4, M). Signed-r roses pass THROUGH the
        origin (negative r reflects across it) and stay continuous — no disconnects."""
        x2, y2 = np.cos(phi), np.sin(phi) * np.cos(phi)          # 2 lobes (figure-8, stretches)
        r3 = np.cos(3 * phi); x3, y3 = r3 * np.cos(phi), r3 * np.sin(phi)   # 3 lobes
        r4 = np.cos(2 * phi); x4, y4 = r4 * np.cos(phi), r4 * np.sin(phi)   # 4 lobes
        r5 = np.cos(5 * phi); x5, y5 = r5 * np.cos(phi), r5 * np.sin(phi)   # 5 lobes
        return np.stack([x2, x3, x4, x5]), np.stack([y2, y3, y4, y5])

    def _node_path(self, ts):
        phi = self._PHI(ts)
        X, Y = self._node_xy(phi)
        # continuous node index 0->1->2->3 (n = 2->3->4->5), holding between transitions
        q = sum(smoothstep(a, b, ts) for a, b in self._node_tx)
        lo = np.clip(np.floor(q).astype(int), 0, 3)
        hi = np.clip(lo + 1, 0, 3)
        f = q - lo
        idx = np.arange(len(ts))
        x = (1 - f) * X[lo, idx] + f * X[hi, idx]
        y = (1 - f) * Y[lo, idx] + f * Y[hi, idx]
        st = 0.22 * np.sin(TWO_PI * 0.28 * ts)                  # dynamic anisotropic stretch
        x, y = x * (1 + st), y * (1 - st)
        # wind up from the origin, then collapse back to it at the end of phase 2
        A = 0.82 * smoothstep(0.0, 1.5, ts) * (1 - smoothstep(*self.t_collapse, ts))
        x, y = x * A, y * A
        rot = 0.5 * ts
        c, s = np.cos(rot), np.sin(rot)
        return np.stack([x * c - y * s, x * s + y * c], axis=1)

    # --- phase 3 orbital wireframe primitives (return Nx3, +z pointing) ------
    @staticmethod
    def _lobe(N=1200, W=13, ph=0.0):
        # a p-type lobe: sphere tangent at origin, pointing +z (z in [0,1]) — a ball of coils
        u = np.linspace(0, 1, N)
        th = (np.pi / 2) * u
        phi = TWO_PI * W * u + ph
        R = np.cos(th)
        return np.stack([R * np.sin(th) * np.cos(phi),
                         R * np.sin(th) * np.sin(phi), R * np.cos(th)], axis=1)

    @staticmethod
    def _sphere(N=2200, W=18, ph=0.0, puff=1.0):
        # puff 0 -> a single equator ring; puff 1 -> full sphere. The ring INFLATES into the
        # sphere, so phase 3 pops out of the phase-2 circle instead of cross-fading in.
        u = np.linspace(0, 1, N)
        half = (np.pi / 2) * puff
        th = (np.pi / 2 - half) + 2 * half * u
        phi = TWO_PI * (1 + (W - 1) * puff) * u + ph
        return np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], axis=1)

    @staticmethod
    def _torus(N=1700, Rr=0.62, rt=0.16, Wt=24):
        u = np.linspace(0, 1, N)
        b = TWO_PI * u
        g = TWO_PI * Wt * u
        return np.stack([(Rr + rt * np.cos(g)) * np.cos(b),
                         (Rr + rt * np.cos(g)) * np.sin(b), rt * np.sin(g)], axis=1)

    def _orbital_parts(self, name, puff=1.0):
        """List of pts3 (Nx3) lobes making up the orbital. All ONE colour (green, via None)."""
        if name == "s":
            return [self._sphere(puff=puff)]
        if name == "p":                                                  # p_z dumbbell
            return [_orient(self._lobe(), (0, 0, 1)), _orient(self._lobe(ph=0.7), (0, 0, -1))]
        if name == "d4":                                                 # cloverleaf (d_xy)
            L = self._lobe
            return [_orient(L(), (1, 1, 0)), _orient(L(ph=.5), (-1, -1, 0)),
                    _orient(L(ph=1.), (1, -1, 0)), _orient(L(ph=1.5), (-1, 1, 0))]
        # dz2: two lobes on z + a ring around the waist
        return [_orient(self._lobe(), (0, 0, 1)), _orient(self._lobe(ph=.7), (0, 0, -1)),
                self._torus()]

    def _project(self, pts3, rot_t, scale):
        # tumble grows from 0 at the moment of the pop (rot_t = t - t_2d) so the sphere inflates
        # face-on (matching the flat phase-2 circle) then rotates into 3D. No aspect correction ->
        # same anisotropic screen map as every other phase, so the transition rings line up.
        M = _rot_x(0.5 * rot_t) @ _rot_y(0.33 * rot_t)
        p = pts3 @ M.T
        return np.stack([p[:, 0] * scale, p[:, 1] * scale], axis=1)

    # --- assembled per-frame traces -----------------------------------------
    def frame_traces(self, t, ts):
        out = []
        te = t - self.t_end2                                            # time into phase 3
        # phase 2 node trace: fades out right as it collapses to the origin
        gnode = self.trail_gain * (1 - smoothstep(self.t_end2 - 0.3, self.t_end2 + 0.3, t))
        if gnode > 0.01 and t < self.t_end2 + 0.3:
            out.append((self._node_path(ts), None, gnode, False))
        if te >= -0.05:
            n = len(self._orbitals)
            span = self.orbital_span
            k = int(min(max(te / span, 0.0), n - 1e-6))
            sl = te - k * span                                          # time within this orbital's slot
            # each orbital EMERGES from the origin (scale 0->1) and later COLLAPSES back to it,
            # so between orbitals everything passes through the origin point
            env = smoothstep(0.0, self.EMERGE, sl) * (1 - smoothstep(span - self.COLLAPSE, span, sl))
            sc = self.scale * env
            if sc > 0.005:
                rot_t = te                                              # continuous 3D tumble
                # low per-frame gain: wireframe redrawn full each frame -> persistence would
                # otherwise accumulate it to a saturated blob
                for pts3 in self._orbital_parts(self._orbitals[k]):
                    out.append((self._project(pts3, rot_t, sc), None, 0.075, False))
        return out
# ===========================================================================
class FourierEpicycles:
    """The 'draw anything with rotating circles' demo: a chain of epicycles (from the DFT of a
    target outline) whose vector sum sweeps out the shape. Each frame lights, in green:
      - the faint epicycle circles,
      - the radial arm chain connecting their centres to the pen,
      - the bright curve traced so far (0 -> current pen angle).
    Curve scene (beams(t)); the traced path accumulates because we redraw 0..tau every frame."""

    M = 48            # number of epicycles (DFT terms kept, largest-amplitude first)
    SAMPLES = 900     # samples of the target outline / reconstructed curve

    def __init__(self, duration=12.0):
        self.duration = duration
        self.draw_frac = 0.82                     # fraction of duration spent drawing; rest holds
        self._build(self._heart())

    @staticmethod
    def _heart():
        t = np.linspace(0, TWO_PI, 900, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        z = (x + 1j * y)
        z = z - z.mean()
        return z / np.max(np.abs(z)) * 0.86       # centre + normalise into the scope face

    def _build(self, z):
        N = len(z)
        C = np.fft.fft(z) / N
        freqs = np.fft.fftfreq(N, d=1.0 / N).astype(int)
        order = np.argsort(-np.abs(C))[: self.M]  # keep the M biggest circles
        self._amp = C[order]
        self._freq = freqs[order]

    def _chain(self, tau):
        """Centres of the epicycle arm at parameter tau in [0,1] -> complex array (M+1,)."""
        steps = self._amp * np.exp(2j * np.pi * self._freq * tau)
        return np.concatenate([[0], np.cumsum(steps)])

    def beams(self, t):
        tau = min(t / (self.duration * self.draw_frac), 1.0)
        out = []
        chain = self._chain(tau)

        # 1) faint epicycle circles (each centred on the previous arm point)
        ang = np.linspace(0, TWO_PI, 64)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        for j in range(self.M):
            c, r = chain[j], np.abs(self._amp[j])
            if r < 0.006:
                continue
            circ = ring * r + [c.real, c.imag]
            out.append((circ, 0.22, True))

        # 2) the radial arm chain (centres -> pen)
        arm = np.stack([chain.real, chain.imag], axis=1)
        out.append((arm, 0.5, False))

        # 3) the curve traced so far (0 .. tau), bright
        s = np.linspace(0, tau, max(2, int(self.SAMPLES * tau)))
        if len(s) >= 2:
            waves = self._amp[:, None] * np.exp(2j * np.pi * self._freq[:, None] * s[None, :])
            path = waves.sum(axis=0)
            out.append((np.stack([path.real, path.imag], axis=1), 1.0, False))
        return out
# ===========================================================================
class Superformula:
    """Gielis superformula sweep: one polar equation morphs continuously through starfish,
    flowers, gears and polygons. Whole curve lit each frame (closed), slowly rotating. Pure
    eye-candy — the 2D analogue of lissajous_tour but organic instead of harmonic."""

    N = 1400
    # (m, n1, n2, n3) presets; the curve lerps between consecutive ones and loops back.
    PRESETS = [
        (3,  5.0, 18.0, 18.0),   # rounded triangle
        (5,  2.0,  7.0,  7.0),   # 5-point starfish
        (6,  1.0,  1.0,  1.0),   # soft hexagon flower
        (12, 15.0, 10.0, 10.0),  # 12-tooth gear
        (7,  3.0,  4.0, 17.0),   # spiky 7-star
        (4,  0.4,  0.4,  4.0),   # pillow square
        (3,  5.0, 18.0, 18.0),   # back to start (loop)
    ]

    def __init__(self, duration=14.0):
        self.duration = duration

    @staticmethod
    def _superformula(theta, m, n1, n2, n3, a=1.0, b=1.0):
        t1 = np.abs(np.cos(m * theta / 4.0) / a) ** n2
        t2 = np.abs(np.sin(m * theta / 4.0) / b) ** n3
        return (t1 + t2) ** (-1.0 / n1)

    def beams(self, t):
        seg = self.duration / (len(self.PRESETS) - 1)
        k = min(int(t / seg), len(self.PRESETS) - 2)
        f = smoothstep(0, 1, (t - k * seg) / seg)
        p = [(1 - f) * a + f * b for a, b in zip(self.PRESETS[k], self.PRESETS[k + 1])]
        theta = np.linspace(0, TWO_PI, self.N)
        r = self._superformula(theta, *p)
        r = r / (np.max(r) + 1e-9) * 0.85          # normalise so it always fills the face
        rot = 0.35 * t                             # slow spin for life
        x = r * np.cos(theta + rot)
        y = r * np.sin(theta + rot)
        return [(np.stack([x, y], axis=1), 1.0, True)]
# ===========================================================================
class Dimensional2D:
    """The 2D EXTENDED CUT of the final video's node section, built to LOOP. Same trail convention
    (one fast winding beam, phosphor persistence, no dot; signed-r figures passing THROUGH the
    origin so the trace never disconnects; rotating + dynamically stretching) — but no wind-up and
    no collapse: it opens already on the THREE-NODE (3-rose) figure, journeys through progressively
    more complex through-origin forms, and morphs back to the 3-rose so the clip loops seamlessly.

      3-rose -> 4-rose -> 5-rose -> 7-rose -> harmonic superposition -> 3:2 Lissajous knot ->
      5/2 rational rose (10 interlaced petals) -> back to 3-rose.

    Every time-varying quantity is PERIODIC over `duration` (winding = integer revs, precession =
    integer turns, stretch = integer sin cycles, shape order starts & ends on the 3-rose), and a
    short `warmup` pre-rolls the phosphor so the first frame's afterglow matches the last."""

    decay = 0.92           # richer afterglow so the dense figures read as they wind
    substeps = 800
    trail_width = 3
    trail_gain = 0.40      # dense roses over-accumulate; keep the per-frame light low
    head_gain = 0.0        # no visible dot (matches the final)
    warmup = 1.6           # seconds of phosphor pre-roll (not output) -> seamless loop seam

    A = 0.82               # constant amplitude (no wind-up / collapse)
    _Wrev = 190            # winding revolutions over the loop (~7 rev/s) — integer -> periodic
    _Rturn = 3             # whole-figure precession turns over the loop — integer -> periodic
    _stc = 4               # stretch sin cycles over the loop — integer -> periodic
    _st_amp = 0.24

    # brief "67" flash (variant only): a seven-segment "67" drawn into the TRANSIENT buffer, so it
    # appears only while drawn and vanishes cleanly (no persistence burn-in). Times = RENDER time.
    flash_win = (0.60, 0.75, 0.95, 1.10)  # fade-in start, full, hold-end, gone (~0.5s, clean)
    flash_gain = 1.1                       # digit brightness
    flash_color = (200, 255, 210)          # hotter near-white green so it pops

    def __init__(self, flash=False):
        self.duration = 27.0
        self.flash = flash
        # the through-origin node forms (phi -> (x,y), un-normalised)
        self._shapes = [
            self._rose(3),                                               # 0: 3-rose (the start/end)
            self._rose(2),                                               # 1: 4-rose
            self._rose(5),                                               # 2: 5-rose
            self._rose(7),                                               # 3: 7-rose
            lambda p: ((np.cos(2 * p) + 0.45 * np.cos(5 * p)) * np.cos(p),  # 4: harmonic superposition
                       (np.cos(2 * p) + 0.45 * np.cos(5 * p)) * np.sin(p)),
            lambda p: (np.sin(3 * p), np.sin(2 * p + np.pi / 4)),        # 5: 3:2 Lissajous knot
            self._rose(2.5),                                            # 6: 5/2 rational rose
        ]
        # loop keyframe order: start & end on the 3-rose (index 0), climbing through complexity
        self._order = np.array([0, 1, 2, 3, 4, 5, 6, 0])
        self.M = len(self._shapes)
        # per-shape normalisation: divide by max radius over a long sweep so all fill equally
        g = np.linspace(0, 12 * np.pi, 24000)
        self._norm = np.array([1.0 / max(float(np.max(np.hypot(*f(g)))), 1e-6) for f in self._shapes])
        self._digits = self._make_67() if flash else None

    @staticmethod
    def _rose(k):
        return lambda p: (np.cos(k * p) * np.cos(p), np.cos(k * p) * np.sin(p))

    # --- seven-segment "67" as beam polylines (bright flash) --------------------
    @staticmethod
    def _seg_digit(which):
        L, R, T, M, B = 0.12, 0.88, 1.9, 1.0, 0.1
        seg = {"a": [(L, T), (R, T)], "b": [(R, T), (R, M)], "c": [(R, M), (R, B)],
               "d": [(L, B), (R, B)], "e": [(L, M), (L, B)], "f": [(L, T), (L, M)],
               "g": [(L, M), (R, M)]}
        on = {"6": "afgecd", "7": "abc"}[which]
        return [np.asarray(seg[s], np.float64) for s in on]

    def _make_67(self):
        """Place a seven-segment '6' and '7' side by side, centred, in normalised scope coords.
        y is compressed (scope face is v-stretched) so the digits read roughly square."""
        polys = []
        for which, xoff in [("6", -0.17), ("7", 0.17)]:
            for seg in self._seg_digit(which):
                nx = xoff + (seg[:, 0] - 0.5) * 0.26
                ny = (seg[:, 1] - 1.0) * 0.16
                polys.append(np.stack([nx, ny], axis=1))
        return polys

    def _flash_g(self, t):
        f0, f1, f2, f3 = self.flash_win
        return min(smoothstep(f0, f1, t), 1.0 - smoothstep(f2, f3, t))

    def grid_state(self, t):
        return 1.0                                         # constant axes-only cross (loops trivially)

    def _node_path(self, ts):
        u = ts / self.duration                             # loop phase in [0,1] (may run a hair over)
        phi = TWO_PI * self._Wrev * u                      # winding: integer revs -> periodic
        X = np.stack([f(phi)[0] for f in self._shapes]) * self._norm[:, None]  # (M, len)
        Y = np.stack([f(phi)[1] for f in self._shapes]) * self._norm[:, None]
        # morph along the loop keyframe order (start & end on the 3-rose), easing each segment
        nseg = len(self._order) - 1
        s = np.clip(u * nseg, 0, nseg)
        k = np.clip(np.floor(s).astype(int), 0, nseg - 1)
        fr = smoothstep(0.0, 1.0, s - k)
        ai, bi = self._order[k], self._order[k + 1]
        idx = np.arange(len(ts))
        x = (1 - fr) * X[ai, idx] + fr * X[bi, idx]
        y = (1 - fr) * Y[ai, idx] + fr * Y[bi, idx]
        st = self._st_amp * np.sin(TWO_PI * self._stc * u)  # stretch: integer cycles -> periodic
        x, y = x * (1 + st), y * (1 - st)
        x, y = x * self.A, y * self.A
        rot = TWO_PI * self._Rturn * u                      # precession: integer turns -> periodic
        c, s2 = np.cos(rot), np.sin(rot)
        return np.stack([x * c - y * s2, x * s2 + y * c], axis=1)

    def frame_traces(self, t, ts):
        # during the flash, dim the node trace so "67" dominates; it returns to full after
        fg = self._flash_g(t) if self.flash else 0.0
        return [(self._node_path(ts), None, self.trail_gain * (1 - 0.7 * fg), False)]

    def flash_traces(self, t):
        """Transient (non-persistent) "67" digits, so the flash vanishes cleanly afterward."""
        if not self.flash:
            return []
        fg = self._flash_g(t)
        if fg <= 0.01:
            return []
        return [(poly, self.flash_color, self.flash_gain * fg) for poly in self._digits]
# ===========================================================================
def _epi(xt, yt):
    """Epicycle curve factory: phi -> (x, y) with x = Σ a·cos(k·phi + ph) over xt and
    y = Σ a·sin(k·phi + ph) over yt. Integer k (may be negative = retrograde) keeps every
    shape 2π-periodic in phi — the requirement for the seamless-loop winding trick."""
    def f(p):
        x = sum(a * np.cos(k * p + ph) for a, k, ph in xt)
        y = sum(a * np.sin(k * p + ph) for a, k, ph in yt)
        return x, y
    return f


def _z(terms):
    """Symmetric (complex-exponential) epicycle: z = Σ a·e^{i(k·phi+ph)} -> same terms both axes."""
    return _epi(terms, terms)


class TraceLoop:
    """The `Dimensional2D` seamless-loop convention generalised to ANY journey of 2π-periodic
    epicycle shapes: one fast winding beam + phosphor persistence, no dot, morphing through a
    keyframe order that starts & ends on shape 0, whole-figure precession + dynamic stretch as
    INTEGER cycles over `duration`, `warmup` tail pre-roll -> frame 0's afterglow == last frame.
    Formatting (decay/gain/width/grid/amplitude) matches the shipped dimensional2d loop; only
    the shape content differs (no roses / node flowers)."""

    decay = 0.92
    substeps = 800
    trail_width = 3
    trail_gain = 0.55      # tuned for 60 fps finals (60 fps reads fainter than 15 fps previews)
    head_gain = 0.0
    warmup = 1.6

    A = 0.82
    _st_amp = 0.24

    def __init__(self, shapes, order, duration=20.0, Wrev=140, Rturn=2, stc=3):
        self.duration = duration
        self._shapes = shapes
        self._order = np.asarray(order)          # must start & end on the same index -> loop
        self._Wrev = Wrev                        # winding revolutions (integer -> periodic)
        self._Rturn = Rturn                      # precession turns (integer -> periodic)
        self._stc = stc                          # stretch sin cycles (integer -> periodic)
        g = np.linspace(0, 12 * np.pi, 24000)    # per-shape normalisation: all fill the face
        self._norm = np.array([1.0 / max(float(np.max(np.hypot(*f(g)))), 1e-6) for f in shapes])

    def grid_state(self, t):
        return 1.0                               # constant axes-only cross (loops trivially)

    def _path(self, ts):
        u = ts / self.duration
        phi = TWO_PI * self._Wrev * u
        X = np.stack([f(phi)[0] for f in self._shapes]) * self._norm[:, None]
        Y = np.stack([f(phi)[1] for f in self._shapes]) * self._norm[:, None]
        nseg = len(self._order) - 1
        s = np.clip(u * nseg, 0, nseg)
        k = np.clip(np.floor(s).astype(int), 0, nseg - 1)
        fr = smoothstep(0.0, 1.0, s - k)
        ai, bi = self._order[k], self._order[k + 1]
        idx = np.arange(len(ts))
        x = (1 - fr) * X[ai, idx] + fr * X[bi, idx]
        y = (1 - fr) * Y[ai, idx] + fr * Y[bi, idx]
        st = self._st_amp * np.sin(TWO_PI * self._stc * u)
        x, y = x * (1 + st) * self.A, y * (1 - st) * self.A
        rot = TWO_PI * self._Rturn * u
        c, s2 = np.cos(rot), np.sin(rot)
        return np.stack([x * c - y * s2, x * s2 + y * c], axis=1)

    def frame_traces(self, t, ts):
        return [(self._path(ts), None, self.trail_gain, False)]


def _loop_gears():
    """Spirograph / guilloche journey: looped rings, star gears and lace — machine-engraved
    curves that never disconnect, rising in tooth count then releasing back to the deltoid."""
    shapes = [
        _z([(1, 1, 0), (.35, -2, 0)]),                    # 0: deltoid ring (3 soft cusps)
        _z([(1, 1, 0), (.45, 6, 0)]),                     # 1: 5-loop epitrochoid
        _z([(1, 1, 0), (.50, -4, 0)]),                    # 2: 5-point star gear
        _z([(1, 1, 0), (.20, 11, 0), (.12, -9, 0)]),      # 3: 10-tooth lace ring
        _z([(1, 1, 0), (.40, -6, 0), (.22, 8, 0)]),       # 4: dense guilloche band
        _z([(1, 1, 0), (.60, 4, 0)]),                     # 5: heavy 3-loop clover ring
    ]
    return TraceLoop(shapes, [0, 1, 2, 3, 4, 5, 0])


def _loop_weave():
    """Lissajous-lattice journey: through-centre woven knots climbing in ratio complexity,
    growing second harmonics at the peak, then relaxing back to the 3:2 knot."""
    HPI = np.pi / 2
    shapes = [
        _epi([(1, 3, -HPI)], [(1, 2, 0)]),                          # 0: 3:2 knot
        _epi([(1, 5, -HPI)], [(1, 4, 0)]),                          # 1: 5:4 lattice
        _epi([(1, 5, -HPI), (.30, 9, 0)], [(1, 4, 0), (.30, 7, -1.0)]),   # 2: 5:4 + harmonics
        _epi([(1, 7, -HPI)], [(1, 3, 0)]),                          # 3: 7:3 web
        _epi([(1, 7, -HPI), (.25, 11, .6)], [(1, 5, 0), (.25, 13, 0)]),   # 4: 7:5 dense weave
        _epi([(1, 4, -HPI)], [(1, 3, 0)]),                          # 5: 4:3 release
    ]
    return TraceLoop(shapes, [0, 1, 2, 3, 4, 5, 0])


def _loop_braid():
    """Counter-rotating braid journey: three-frequency epicycles whose retrograde terms plait
    the trace into orbital braids — the most intricate of the three, planetary-gear energy."""
    shapes = [
        _z([(1, 1, 0), (.55, -5, 0), (.25, 7, 0)]),       # 0: 6-strand braid ring
        _z([(1, 1, 0), (.50, 7, 0), (.30, -11, 0)]),      # 1: fine 6/18 lacework
        _z([(1, 2, 0), (.60, -7, 0)]),                    # 2: 9-fold through-centre braid
        _z([(1, 1, 0), (.45, -8, 0), (.30, 10, 0)]),      # 3: 9-strand crown
        _z([(1, 3, 0), (.50, -6, 0), (.25, 12, 0)]),      # 4: triple-orbit plait
        _z([(1, 1, 0), (.70, -2, 0), (.30, 5, 0)]),       # 5: heavy 3-lobe release
    ]
    return TraceLoop(shapes, [0, 1, 2, 3, 4, 5, 0])


# ===========================================================================
SCENES = {
    "dimensional": Dimensional,
    "dimensional2d": Dimensional2D,                       # looping base (starts on the 3-node)
    "dimensional2d_67": lambda: Dimensional2D(flash=True),  # same, with a brief "67" flash
    "wind_up": WindUp,
    "orbit_trace": OrbitTrace,
    "lissajous_tour": LissajousTour,
    "pulse_to_shape": PulseToShape,
    "fourier_epicycles": FourierEpicycles,
    "superformula": Superformula,
    "loop_gears": _loop_gears,                            # 20s seamless-loop trace journeys
    "loop_weave": _loop_weave,                            # (dimensional2d formatting, new shapes)
    "loop_braid": _loop_braid,
}
