# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""scenes.py - the catalogue.

A scene may only choose GEOMETRY, CHARGE and DECLARED FORCING - where a source is at time t,
how strong it is, what field kernel is in play, how many lines are lit. It may never place a
field line. Every curve on screen is the integrator's answer, which is the whole reason the
output is trustworthy as well as pretty.

The five scenes are a deliberate spread over the properties the reverse-engineering pass
established - one property each, no reskins:

  dipole_flower   seed count as an animation channel + capture topology events (the reference)
  rosette_waltz   continuous topology reconnection under moving sinks
  pair_dance      nothing keyframed - the sources themselves are solved (N-body)
  charge_wave     charge MAGNITUDE and SIGN as animation channels, read out by marker radius
  stream_glass    kernel generality - same integrator, potential flow instead of electrostatics
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dfield

import numpy as np

from . import draw as D
from .field import ChargeSet, PotentialFlow
from .lines import fade_order, seed_rake, seed_ring
from .motion import NBody, WanderPath, keyframes, pulse_gate
from .pulses import PulseTrain

W, H = 1080, 1920
CX, CY = 540.0, 960.0
TAU = 2.0 * math.pi


@dataclass
class FrameSpec:
    """Everything render.py needs for one frame. The scene fills it; nothing else knows about
    the scene."""
    fieldfn: object
    seeds: np.ndarray
    alphas: np.ndarray
    colors: object                       # (N,3) array, or a callable of `ended`
    sinks: np.ndarray
    markers: list
    ds: float = 6.0
    n_steps: int = 440
    capture_r: float = 20.0
    bounds: tuple = (-280.0, -280.0, 1360.0, 2200.0)
    # Stroke widths are chosen so PIL's INTEGER rounding lands on the same width RELATIVE to
    # the frame at both resolutions. A preview has k = 1.0 and a final k = 2.0, so 1.7 rounds
    # to 2 px then 3 px - i.e. 0.00185 of frame width at preview but 0.00139 at final, and the
    # final comes out 25% finer and darker than the preview promised. Even values keep them
    # identical, which is the whole point of having a preview.
    line_width: float = 2.0
    pulse: object = None
    pulse_width: float = 3.0
    dens_tint: tuple = (90, 170, 255)
    dens_gain: float = 1.0
    glow_mul: float = 1.0
    glow2_mul: float = 1.0


class SeedBank:
    """A fixed set of seed slots around one source.

    Angles NEVER move. `alphas(n_live)` lights the first `n_live` slots in BIT-REVERSED order,
    so a growing count fills the fan in evenly rather than sweeping a wedge round from one side,
    and a new line fades in exactly between two lines that did not budge. Re-spacing the seeds
    when the count changes is the single most likely visual bug in this format - the whole fan
    snaps on every integer crossing.
    """

    def __init__(self, n_slots, phase=0.0):
        self.n = n_slots
        self.phase = phase
        self.rank = fade_order(n_slots)

    def ring(self, center, r0):
        return seed_ring(center, r0, self.n, self.phase)

    def alphas(self, n_live, soft=2.5):
        return np.clip((n_live - self.rank) / soft, 0.0, 1.0)


def _ell(cx, cy, rx, ry, th):
    return np.stack([cx + rx * np.cos(th), cy + ry * np.sin(th)], axis=1)


def _mark(x, y, r, kind="ring", color=D.WHITE, sign=0, width=3.0, glyph=None):
    return dict(x=float(x), y=float(y), r=float(r), kind=kind, color=color,
                sign=sign, width=width, glyph=glyph)


# ======================================================================================
# 1. dipole_flower - the reference format, rebuilt
# ======================================================================================
class DipoleFlower:
    """+Q at frame centre, four -q on an anisotropic cross, one mobile -q on a declared path.

    Two independent scalars are animated on a musical envelope, exactly as measured off the
    source clip: the number of lit field lines, and the magnitude of the mobile charge. The
    second one is what produces the format's headline event - when |q_mobile| goes high its
    capacity exceeds the centre's emission and it CAPTURES THE ENTIRE FLUX, so the picture
    snaps from a five-pole flower to a single teardrop in about a second. Nothing switches; the
    topology genuinely changes.

    Cross geometry is the source's, scaled from 608 px wide to 1080: +-170 -> +-302 px
    horizontally, +-451.5 -> +-802 px vertically. The 1:2.66 anisotropy is what fills 9:16
    instead of leaving the top and bottom of the frame empty.
    """
    name, duration = "dipole_flower", 15.0

    def __init__(self):
        self.fixed = np.array([[CX, CY - 802.0], [CX + 302.0, CY],
                               [CX, CY + 802.0], [CX - 302.0, CY]])
        self.path = WanderPath((CX, CY), (W, H), speed=302.0, over=1.16, seed=5, f0=0.118)
        self.bank = SeedBank(220)
        self.cs = ChargeSet(soft=10.0)
        self.pulse = PulseTrain(period=2.5, speed=1155.0, dash=26.0, lead_in=0.35)
        self.pal = D.palette_alternate(220, D.CYAN, D.BLUE)

    def spec(self, t):
        qm = pulse_gate(t, [(5.8, 7.8), (11.2, 12.6)], ramp=1.0, lo=1.0, hi=5.0)
        n = keyframes(t, [0, 2.5, 5.5, 7.5, 11.0, 12.5, 15.0],
                      [16, 42, 112, 52, 190, 14, 84])
        pm = self.path.at(t)
        pos = np.vstack([[[CX, CY]], self.fixed, [pm]])
        q = np.array([6.0, -1.0, -1.0, -1.0, -1.0, -qm])
        self.cs.set(pos, q)

        rm = 12.4 * math.sqrt(qm)
        marks = [_mark(CX, CY, 16.0, "ring", D.WHITE, +1)]
        marks += [_mark(p[0], p[1], 16.0, "ring", D.WHITE, -1) for p in self.fixed]
        marks += [_mark(pm[0], pm[1], rm, "filled", D.MAGENTA, -1,
                        glyph=(40, 0, 40))]
        return FrameSpec(
            fieldfn=self.cs, seeds=self.bank.ring((CX, CY), 18.0),
            alphas=self.bank.alphas(n), colors=self.pal,
            sinks=self.cs.sinks(), markers=marks,
            capture_r=20.0, pulse=self.pulse)


# ======================================================================================
# 2. rosette_waltz - moving sinks, continuous reconnection
# ======================================================================================
class RosetteWaltz:
    """One strong +Q at centre; five -q on an inner ellipse that ACCELERATES clockwise while
    breathing, and three -q on an outer ellipse that counter-rotates.

    The point is that the rotating rosette is not drawn - the SINKS move, and the flux bundles
    follow. Two counter-rotating rings at incommensurate rates means the pattern never repeats
    and never sits still: bundles are handed from an inner sink to an outer one and back as the
    rings slide past each other, and each hand-off is a real topological reconnection through a
    null point.

    Both rings are ELLIPSES with the frame's own aspect, so every source stays on camera and the
    rosette fills 9:16 - stretching the FIELD's geometry rather than the bodies in it, so every
    source stays legitimately on camera. Total negative capacity (8.0) is deliberately below the
    centre's emission (11.0) so roughly a quarter of the flux always escapes to the frame edge:
    the long lines are what give the turning rosette something to sweep against.
    """
    name, duration = "rosette_waltz", 15.0

    def __init__(self):
        self.bank = SeedBank(280)
        self.cs = ChargeSet(soft=10.0)
        self.pulse = PulseTrain(period=1.875, speed=1250.0, dash=24.0, lead_in=0.4)
        self.pal = D.palette_angle(280, (D.CYAN, D.BLUE, D.MINT, D.BLUE))

    def rings(self, t):
        b1 = math.sin(TAU * t / 7.5)
        b2 = math.sin(TAU * t / 5.5 + 1.9)
        th1 = 0.50 * t + 0.030 * t * t + 0.3
        th2 = -0.36 * t - 0.012 * t * t + 1.1
        inner = _ell(CX, CY, 318 + 46 * b1, 566 + 82 * b1,
                     th1 + TAU * np.arange(5) / 5.0)
        outer = _ell(CX, CY, 452 + 50 * b2, 804 + 89 * b2,
                     th2 + TAU * np.arange(3) / 3.0)
        return inner, outer

    def spec(self, t):
        inner, outer = self.rings(t)
        pos = np.vstack([[[CX, CY]], inner, outer])
        q = np.concatenate([[11.0], np.full(5, -0.92), np.full(3, -1.62)])
        self.cs.set(pos, q)
        n = keyframes(t, [0, 3, 6, 9, 12, 15], [24, 120, 60, 250, 150, 240])

        marks = [_mark(CX, CY, 17.0, "ring", D.WHITE, +1, width=3.4)]
        marks += [_mark(p[0], p[1], 15.0, "ring", D.WHITE, -1) for p in inner]
        marks += [_mark(p[0], p[1], 18.0, "ring", D.AMBER, -1, width=3.6) for p in outer]
        return FrameSpec(
            fieldfn=self.cs, seeds=self.bank.ring((CX, CY), 19.0),
            alphas=self.bank.alphas(n), colors=self.pal,
            sinks=self.cs.sinks(), markers=marks,
            capture_r=21.0, pulse=self.pulse,
            dens_tint=(80, 190, 245))


# ======================================================================================
# 3. pair_dance - the sources themselves are solved
# ======================================================================================
class PairDance:
    """Three +q and three -q, free to move under their own mutual Coulomb forces inside a
    declared anisotropic harmonic well, launched tangentially. NOTHING about where they go is
    authored.

    Lines are coloured BY HOW THEY END - blue if a sink caught them, cyan if they reached the
    frame edge - which is only possible because the integrator reports each line's fate. It
    makes the topology directly readable: as a + and a - swing past each other the bundle
    between them flips from cyan to blue in the instant the pair becomes close enough to
    capture, so the colour change IS the physics, not a highlight.

    The well is the rig, and it is declared: without it, a charge that gains energy in a close
    pass leaves and never comes back, and no tuning of the interactions fixes that. Softening is
    not cosmetic either - an unsoftened opposite-sign pair reaches unbounded speed in finite
    time and no fixed substep can follow it.
    """
    name, duration = "pair_dance", 15.0

    def __init__(self, seed=3):
        th = TAU * np.arange(6) / 6.0 + 0.22
        p0 = _ell(CX, CY, 372.0, 662.0, th)
        self.q = np.array([2.4, -2.4, 2.4, -2.4, 2.4, -2.4])
        self.nb = NBody(p0, self.q, well=2.35, aniso=1.778, coupling=1.15e6,
                        softening=105.0, core_k=9.0e12, core_soft=40.0,
                        drag=0.0, v_max=1900.0,
                        substeps_per_sec=600).launch_tangential(1.0)
        self.banks = [SeedBank(74, phase=0.9 * k) for k in range(3)]
        self.cs = ChargeSet(soft=11.0)
        self.pulse = PulseTrain(period=2.5, speed=1100.0, dash=28.0, lead_in=0.6)

    def spec(self, t):
        self.nb.advance_to(t)
        pos = self.nb.p
        self.cs.set(pos, self.q)
        n = keyframes(t, [0, 2.5, 7.0, 10.0, 12.5, 15.0], [16, 58, 40, 72, 32, 66])

        seeds, alphas = [], []
        for k, bi in enumerate([0, 2, 4]):
            b = self.banks[k]
            seeds.append(b.ring(pos[bi], 20.0))
            alphas.append(b.alphas(n))
        marks = []
        for i in range(6):
            warm = self.q[i] > 0
            marks.append(_mark(pos[i, 0], pos[i, 1], 15.5, "filled",
                               (255, 214, 150) if warm else (122, 192, 255),
                               1 if warm else -1, glyph=(20, 12, 6)))
        return FrameSpec(
            fieldfn=self.cs, seeds=np.vstack(seeds),
            alphas=np.concatenate(alphas),
            colors=lambda ended: D.palette_fate(ended, D.BLUE, D.CYAN, D.ROSE),
            sinks=self.cs.sinks(), markers=marks,
            capture_r=22.0, pulse=self.pulse,
            dens_tint=(120, 165, 245))


# ======================================================================================
# 4. charge_wave - magnitude and SIGN as animation channels
# ======================================================================================
class ChargeWave:
    """Seven sources in a swaying vertical column, each carrying

        q_i(t) = A sin(2 pi (i/lam - t/T))

    so a travelling wave of charge runs down the column. Every source cycles + -> 0 -> - -> 0
    and back, and its marker radius is r ~ sqrt(|q|), so the picture reads out the state of the
    simulation exactly and cannot lie about it.

    What this demonstrates that no static field-line plot can: field lines are BORN AND DIE as
    sources cross zero. Seeds are allocated per source and lit in proportion to that source's
    charge, so a bundle grows out of a marker as it charges up, arcs to its neighbour, then
    shrinks back into nothing as the source neutralises - and the whole pattern of loops
    migrates down the frame at the wave speed. The column also sways, so no two frames present
    the same geometry.

    The wavelength is chosen so the sum over the column is ~0 at every instant: a near-neutral
    system keeps almost all flux local and legible instead of flooding the frame with escapes.
    """
    name, duration = "charge_wave", 15.0
    N, A, LAM, T = 7, 3.4, 3.5, 5.0

    def __init__(self):
        self.y = 240.0 + 240.0 * np.arange(self.N)
        self.banks = [SeedBank(40, phase=0.7 * k) for k in range(self.N)]
        self.cs = ChargeSet(soft=11.0)
        self.pulse = PulseTrain(period=1.25, speed=1000.0, dash=22.0, lead_in=0.3)
        self.stops = np.array([D.CYAN, (155, 238, 255), D.BLUE, (96, 118, 242)],
                              dtype=float)

    def state(self, t):
        i = np.arange(self.N)
        ph = i / self.LAM - t / self.T
        q = self.A * np.sin(TAU * ph)
        x = CX + 120.0 * np.sin(TAU * i / 3.5 + 0.42 * t)
        return np.stack([x, self.y], axis=1), q, ph

    def _hue(self, ph):
        u = (ph % 1.0) * len(self.stops)
        i0 = int(u) % len(self.stops)
        i1 = (i0 + 1) % len(self.stops)
        f = u - int(u)
        return self.stops[i0] * (1 - f) + self.stops[i1] * f

    def spec(self, t):
        pos, q, ph = self.state(t)
        self.cs.set(pos, q)
        gain = keyframes(t, [0, 2.0, 15.0], [4.5, 13.5, 13.5])

        seeds, alphas, cols = [], [], []
        for i in range(self.N):
            b = self.banks[i]
            seeds.append(b.ring(pos[i], 8.0 + 14.0 * math.sqrt(max(q[i], 0.0) / self.A)))
            alphas.append(b.alphas(gain * max(q[i], 0.0)))
            cols.append(np.tile(self._hue(ph[i]), (b.n, 1)))

        marks = []
        for i in range(self.N):
            a = abs(q[i])
            r = 5.0 + 13.0 * math.sqrt(a / self.A)
            warm = q[i] > 0
            marks.append(_mark(pos[i, 0], pos[i, 1], r, "filled",
                               (255, 206, 140) if warm else (126, 196, 255),
                               1 if warm else -1, glyph=(16, 10, 4)))
        return FrameSpec(
            fieldfn=self.cs, seeds=np.vstack(seeds),
            alphas=np.concatenate(alphas), colors=np.vstack(cols),
            sinks=self.cs.sinks(), markers=marks,
            capture_r=20.0, n_steps=380, pulse=self.pulse,
            dens_tint=(100, 200, 240))


# ======================================================================================
# 5. stream_glass - the same integrator, a different kernel
# ======================================================================================
class StreamGlass:
    """Potential flow: a tilting uniform stream past ONE cylinder whose bound circulation
    swings, with two free vortices orbiting it - every singularity imaged in the body by the
    Milne-Thomson circle theorem, so the surface is a streamline EXACTLY.

    This scene exists to prove the reuse claim. Not one line of `lines.trace`, `pulses.py` or
    `draw.py` changes - only the callable handed in. Seeds become an inlet rake instead of a
    ring, and the pulse packets stop reading as flux markers and start reading as TRACER
    PARTICLES, which is what makes it look like flow rather than like a diagram.

    Three things move, and each is a declared forcing a real rig would have: the tunnel's
    incidence, the cylinder's spin (its bound circulation), and two vortices convecting past.
    Nothing about the streamlines is authored.

    What it shows honestly: the two stagnation points RACING around the body as circulation
    swings - the genuine Magnus mechanism - and streamline bundles being wound into spirals by
    the passing vortices. What it cannot show: a wake, or separation. This flow is irrotational
    by construction, so d'Alembert holds and there is no drag. For a real wake the tool is a
    lattice-Boltzmann solver; this is the cheap exact one.

    ONE cylinder, not two, and that is a measured decision - see `PotentialFlow.cylinder`.
    """
    name, duration = "stream_glass", 15.0
    A_R = 208.0
    BODY = (540.0, 812.0)

    def __init__(self):
        self.pf = PotentialFlow()
        self.bank = SeedBank(190)
        # stagger=1.0 -> tracers scattered over a full period. On a parallel bundle the
        # synchronous ring that makes dipole_flower work degenerates into a solid bar.
        self.pulse = PulseTrain(period=1.05, speed=950.0, dash=34.0, lead_in=0.2,
                                stagger=1.0)
        self.pal = D.palette_alternate(190, D.CYAN, D.BLUE)

    def free_vortices(self, t):
        """Two vortices orbiting below the body on staggered ellipses. They are imaged, so they
        may pass as close as they like without leaking flow through the cylinder."""
        a1 = TAU * t / 8.0 + 0.4
        a2 = -TAU * t / 6.5 + 2.4
        return [(np.array([540.0 + 330.0 * math.cos(a1), 1420.0 + 300.0 * math.sin(a1)]),
                 165.0),
                (np.array([540.0 + 250.0 * math.cos(a2), 1560.0 + 235.0 * math.sin(a2)]),
                 -140.0)]

    def spec(self, t):
        tilt = 0.16 * math.sin(TAU * t / 8.5)
        self.pf.clear_singularities()
        self.pf.U = np.array([tilt, 1.0]) / math.hypot(tilt, 1.0)
        for c, g in self.free_vortices(t):
            self.pf.vortices.append((c, g))
        # the cylinder LAST: it reads U to orient its doublet, and it is what every free
        # singularity above gets imaged in
        self.pf.cylinder(np.array(self.BODY), self.A_R,
                         gamma=340.0 * math.sin(TAU * t / 6.0))
        n = keyframes(t, [0, 3.5, 8.5, 10.0, 15.0], [58, 190, 190, 112, 190])
        # The rake must be wide enough that TILT never empties a side of the frame. A 2000 px
        # descent at tilt 0.16 drifts a streamline ~320 px sideways, so a rake that only just
        # covers the frame leaves a third of it black downstream (measured at t=7.6 on the
        # first build). Overhang each edge by more than the worst drift and accept that ~45%
        # of the lines are off camera.
        seeds = seed_rake((-430.0, -60.0), (1510.0, -60.0), self.bank.n)

        marks = [_mark(self.BODY[0], self.BODY[1], self.A_R, "body",
                       (150, 232, 255), width=4.5)]
        return FrameSpec(
            fieldfn=self.pf, seeds=seeds, alphas=self.bank.alphas(n),
            colors=self.pal, sinks=np.array([self.BODY]),
            # s_max = ds*n_steps must clear the bounds traverse WITH SLACK: a straight run
            # from the rake to the floor is 2130 px, and the first build allowed 2795, so any
            # streamline that meandered ran out of arclength and reported END_BUDGET (34% of
            # them at t=10). 4320 px gives a meandering line room to finish.
            markers=marks, ds=8.0, n_steps=540, capture_r=self.A_R - 3.0,
            bounds=(-390.0, -130.0, 1470.0, 2070.0),
            pulse=self.pulse, pulse_width=3.0, line_width=2.0,
            dens_tint=(95, 185, 250), dens_gain=0.30,
            glow_mul=0.50, glow2_mul=0.22)


SCENES = {
    "dipole_flower": DipoleFlower,
    "rosette_waltz": RosetteWaltz,
    "pair_dance": PairDance,
    "charge_wave": ChargeWave,
    "stream_glass": StreamGlass,
}
