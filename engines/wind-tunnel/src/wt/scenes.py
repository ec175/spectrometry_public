# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""scenes.py - what is in the tunnel and what it does over time.

A scene is deliberately thin: it owns a duration and a `bodies(t)` method returning closed
polygons in lattice coordinates. Everything else - the flow, the separation, the shedding - is
the solver's business. That split is what keeps the library honest: a scene can NOT fake an
aerodynamic effect, it can only put a shape somewhere and let the physics answer.

Each class carries `OVERRIDES`, a dict of RenderConfig fields that the scene needs (mostly the
Reynolds number and the flow speed), applied by cli.py unless the user passes the flag
explicitly. Add a scene = add a class + one line in SCENES.
"""
from __future__ import annotations

import os

import numpy as np

from . import rig, shapes
from .bodies import BodySystem, FreeBody, density_for_fall, polygon_area
from .config import ROOT
from .gpu import asnumpy, xp


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def need_asset(path, what):
    """Photo subjects are not shipped with the repo: drop your own cut-out PNG in `src/assets/`."""
    if not os.path.isfile(path):
        raise SystemExit(
            f"This scene needs {what}: a cut-out PNG with a transparent background at\n"
            f"  {path}\n"
            "It is not included in the repo. Any side-on subject with transparency works; "
            "see src/assets/README.md.")
    return path


def keyframes(t, knots):
    """Piecewise-smoothstep interpolation through (time, value) knots."""
    ts = [k[0] for k in knots]
    vs = [k[1] for k in knots]
    if t <= ts[0]:
        return vs[0]
    for i in range(len(ts) - 1):
        if t <= ts[i + 1]:
            return vs[i] + (vs[i + 1] - vs[i]) * smoothstep(ts[i], ts[i + 1], t)
    return vs[-1]


class Scene:
    """Base: holds the lattice size and the placement constants derived from it.

    A STATIC scene implements `bodies(t)` only. A DYNAMIC scene sets `dynamic = True` and
    implements `substep(lbm, t, dt) -> solid mask`, which `render.Tunnel` calls once per LBM
    step so that free bodies move continuously with the solve rather than jumping once a frame.
    """
    OVERRIDES: dict = {}
    duration = 15.0
    title = ""
    dynamic = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        # Bodies are sized against the SCREEN WIDTH, not against a lattice axis: which lattice
        # axis is "across the picture" flips with cfg.flow, and a chord quoted in nx would come
        # out three times too big the moment the format changed.
        self._vert = cfg.flow in ("up", "down")     # streamwise runs along the screen's LONG axis
        self.wspan = self.ny if self._vert else self.nx
        # VISIBLE window (== the whole lattice unless cfg.overscan > 0). Place things with
        # sx()/sy() and size them against `vspan` to compose against what the viewer actually
        # sees; the raw nx/ny include the off-camera overscan margin.
        self.vx0, self.vnx = cfg.vis_x0, cfg.vis_nx
        self.vy0, self.vny = cfg.vis_y0, cfg.vis_ny
        self.vspan = self.vny if self._vert else self.vnx

    def sx(self, frac):
        """Fraction along the VISIBLE streamwise axis (0 = the INLET end) -> lattice x.

        Which screen edge that is depends on `flow`: the bottom for "up", the TOP for "down",
        the left for "right". Always downstream-positive, so a scene composes in the same terms
        whichever way the tunnel is pointed.
        """
        return self.vx0 + frac * (self.vnx - 1)

    def sy(self, frac):
        """Fraction across the VISIBLE cross-stream axis -> lattice y."""
        return self.vy0 + frac * (self.vny - 1)

    def ref_len(self):
        """Length scale the Reynolds number is built on (lattice cells)."""
        return 0.40 * self.wspan

    def bodies(self, t):
        raise NotImplementedError

    def solid_polys(self, t):
        """The polygons handed to the SOLVER, which need not be the ones that are DRAWN.

        Default: the drawn bodies, so every scene written before this hook existed is unchanged
        by construction - the two lists are the same list.

        A scene overrides this when part of what the viewer sees must not be a wall. The case it
        was built for is an animal's LEGS. A 2-D planar section through a cow turns each leg into
        a plate of infinite spanwise extent, which is why a photographic cow reads as a brick
        wall on stilts; in three dimensions the flow at leg height mostly goes around and between
        them. Returning the torso alone as the solid set, while `bodies()` still returns the whole
        animal, is the 2-D stand-in for that.

        BE HONEST ABOUT WHAT THIS IS. The flow is not approximated - it is exactly the flow around
        whatever `solid_polys` returns, with no fudge anywhere. What the picture then does is paint
        a region the solver never saw. That is a DRAWING decision, taken deliberately - my call,
        2026-08-10, to keep a cow's legs from acting as an actual barrier so the picture reads as
        3-D flow moving through them - and a scene that uses it should say so rather than let
        the frame imply the legs were resolved.

        Everything downstream follows for free and correctly: `Streaks.advance` kills tracers on
        `lbm.solid`, so a tracer crosses a phantom leg and is then hidden behind the drawn one;
        the body is composited after the streaks, so the leg still occludes; and because the hook
        only changes which polygons reach `rasterize`, it works identically on both solvers - it
        needs nothing from `LBM` or `CNS` at all. (A PARTIALLY porous leg would not: that is a
        Brinkman drag term in the collision step, and `CNS.set_drive` raises on purpose.)

        ANY TOOL THAT BUILDS A MASK MUST CALL THIS, not `bodies()` - otherwise a gate measures a
        solve with the legs solid while the render runs one without, and passes the wrong physics.
        """
        return self.bodies(t)

    def readout(self, t):
        """Optional (label, value) pairs for the --hud overlay."""
        return []

    def legend(self):
        """Optional colour-bar key for the --legend overlay, or None for no key.

        Returns dict(title=str, marks=[(frac, label), ...], note=str). `frac` is a position on
        the colour ramp in units of the FULL SCALE the renderer normalises by (u0 * vmax), i.e.
        a speed of `v` sits at frac = v / (u0 * vmax). The renderer applies the tone curve; a
        scene only ever has to say where on the linear speed scale a mark belongs.
        """
        return None


# --------------------------------------------------------------------------------------
class AoASweep(Scene):
    """A NACA section swept BOTH ways through stall: +stall, cross over, -stall, back to neutral.

    settle at 0 deg -> suction peak builds -> separates near 14-16 deg into a rolling wake at
    +24 -> crosses through 0 (wake flips) -> the mirror-image stall at -24 -> ease back to neutral.
    """
    OVERRIDES = dict(re=2600.0, u0=0.10, steps=14)
    duration = 15.0
    title = "NACA 2412"

    def __init__(self, cfg, code="2412"):
        super().__init__(cfg)
        self.code = code
        self.prof = shapes.naca4(code)
        self.chord = 0.36 * self.wspan
        self.cx = 0.33 * self.nx
        self.cy = 0.50 * self.ny

    def ref_len(self):
        return self.chord

    def aoa(self, t):
        # Bidirectional, expanded sweep: settle -> +stall -> hold -> cross to -stall ->
        # hold -> ease back to neutral. Peak widened 20 -> 24 deg each side. Fastest segment
        # (the crossover) is ~10 deg/s -> ~0.02 tip cells/step, well under the moving-boundary
        # comfort limit, so the solve stays as clean as the one-sided sweep.
        d = self.duration
        return keyframes(t, [(0.0, 0.0), (0.07 * d, 0.0), (0.30 * d, 24.0),
                             (0.40 * d, 24.0), (0.72 * d, -24.0),
                             (0.82 * d, -24.0), (1.00 * d, 0.0)])

    def bodies(self, t):
        return [shapes.place(self.prof, self.chord, self.cx, self.cy, self.aoa(t))]

    def readout(self, t):
        return [("AoA", f"{self.aoa(t):.1f} deg")]


class AoASweepFast(AoASweep):
    """`aoa_sweep`, same section and the SAME sweep schedule, in a tunnel run twice as fast.

    The only change is `steps`. Apparent speed on screen is (lattice speed x LBM steps per
    frame), and every lattice velocity has to stay small for the boundary terms, so wind is
    always bought with `steps` and never with `u0` - which also means
    `re`, `tau` and the per-step physics here are bit-for-bit the ones that were already
    verified. The sweep is written in SECONDS, so the foil turns at exactly the same rate through
    twice the wind: the point of the comparison. `streak_span` is quoted in FRAMES of travel and
    comes down so the dashes stay dashes instead of doubling into a smear.
    """
    OVERRIDES = dict(AoASweep.OVERRIDES, steps=28, streak_span=2.4)
    title = "NACA 2412 - FAST"


class Stall(AoASweep):
    """Held deep in stall - a continuous shed of leading-edge vortices down the wake."""
    OVERRIDES = dict(re=2600.0, u0=0.10, steps=15)
    duration = 15.0
    title = "STALLED"

    def aoa(self, t):
        return keyframes(t, [(0.0, 4.0), (2.5, 17.0)])


class TriFoil(Scene):
    """Three SYMMETRIC sections stacked across a left-to-right tunnel, sweeping as a wave.

    The horizontal format is the point of this one, and it is the only scene that asks for it.
    With `flow="right"` the stream runs across the 1080 px width and the full 1920 px of the
    frame becomes CROSS-STREAM span - room to stack three foils with real distance between them,
    which the vertical format has nowhere to put. The trade is a real one: only
    ~3 chords of wake fit downstream. That is deliberately what you are looking at here - three
    near wakes interfering, not one long one.

    A symmetric section (NACA 0015) has no camber, so at 0 deg it makes no lift and sheds
    nothing; everything on screen is bought by incidence alone, and the picture is exactly
    mirror-symmetric between the two halves of each cycle. The sweep is +-PEAK deg, a wider
    throw than `aoa_sweep`'s +-24, and the three lanes are phased a third of a period apart so
    the stall runs down the frame as a travelling wave rather than all three stalling at once.

    Slew rate is the thing to watch, not the peak angle: at PEAK=34 over PERIOD=9 s the fastest
    the foil turns is ~24 deg/s, which puts the trailing edge at ~0.033 cells per LBM step -
    between `aoa_sweep` (0.020) and the already-verified `Flutter` (~0.07). The angle itself
    costs nothing; how fast the mask moves is what a lattice minds.
    """
    # Overscan is not cosmetic here. Three foils at incidence block ~24% of the span, and that
    # blockage pushes a pressure field UPSTREAM; with the foils a chord from a hard inlet velocity
    # BC the two fight and the inlet region spikes. The margin buys 1.2 chords of run-in and puts
    # the tunnel walls off camera as well. It costs 1.44x the cells - the only real price of the
    # horizontal format, and worth it.
    # u0 is 0.075, not the 0.10 of `aoa_sweep`, and `steps` carries the difference. Three foils at
    # +-34 deg block ~21% of the span, and with FREE-SLIP walls blockage is real, so the
    # stream squeezing past them reaches ~3.6x u0. At u0=0.10 that is
    # max|u| ~0.36 and climbing, and the first build of this scene DIVERGED at t=7.8 s. Scaling
    # every velocity by 0.75 puts the same flow pattern at ~0.27, and `re` comes down with it so
    # that nu - and therefore tau - is unchanged; `steps` 14 -> 20 restores the apparent speed.
    # That is the standard move here: the limits are on absolute lattice velocity, the picture
    # only cares about (velocity x steps). csm is up as well, which adds eddy viscosity exactly
    # where the strain is highest - between the foils.
    OVERRIDES = dict(re=1600.0, u0=0.075, steps=20, csm=0.20, settle=1.2, flow="right",
                     vmax=2.60, perturb=0.02, overscan=0.10)
    duration = 15.0
    title = "TRIPLE FOIL"

    CODE = "0015"                  # symmetric: no camber at all
    CHORD = 0.22                   # of the VISIBLE streamwise span (the screen width, in this
                                   # format) - never of `wspan`, which includes the overscan
    LANES = (0.20, 0.50, 0.80)     # cross-stream placement = down the screen
    CX = (0.27, 0.34, 0.27)        # streamwise placement, staggered so the wakes do not line up
    PEAK = 34.0                    # +-deg. "even greater range" than the +-24 expanded sweep
    PERIOD = 9.0                   # seconds per full sweep cycle
    LAG = (0.0, 3.0, 6.0)          # a third of a period between lanes -> a travelling wave
    RAMP = 1.8                     # ease the sweep in from 0 deg, and back out at the end

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prof = shapes.naca4(self.CODE)
        self.chord = self.CHORD * self.vspan

    def ref_len(self):
        return self.chord

    def envelope(self, t):
        """Start and finish at exactly 0 deg, so the clip opens and closes on clean flow."""
        return (smoothstep(0.0, self.RAMP, t)
                * (1.0 - smoothstep(self.duration - self.RAMP, self.duration, t)))

    def aoa(self, k, t):
        return self.PEAK * self.envelope(t) * float(
            np.sin(2 * np.pi * (t - self.LAG[k]) / self.PERIOD))

    def bodies(self, t):
        return [shapes.place(self.prof, self.chord, self.sx(self.CX[k]), self.sy(fy),
                             self.aoa(k, t))
                for k, fy in enumerate(self.LANES)]

    def readout(self, t):
        return [("AoA", " ".join(f"{self.aoa(k, t):+.0f}" for k in range(len(self.LANES))))]


class TriFoilFast(TriFoil):
    """`tri_foil` with the tunnel run twice as fast and the SAME choreography.

    The flow rate is bought entirely with `steps` - the apparent speed of everything on screen is
    (lattice speed x LBM steps per frame), and every lattice velocity has to stay small for the
    boundary terms. Doubling `steps` doubles the wind and leaves `u0`,
    `re` and the per-step physics untouched. The sweep schedule is written in SECONDS, so it is
    unchanged - the foils turn at the same rate through twice the wind, which is the whole point
    of the comparison. `streak_span` is quoted in FRAMES of travel, so it comes down to hold the
    same ~30 px dash instead of doubling into a smear.
    """
    OVERRIDES = dict(TriFoil.OVERRIDES, steps=40, streak_span=2.4)
    title = "TRIPLE FOIL - FAST"


class TriFoilRates(TriFoil):
    """Three IDENTICAL symmetric sections in ONE tunnel, each in a stream of a different speed.

    `tri_foil` and `tri_foil_fast` answer "what does the same choreography look like in a faster
    tunnel?" across two clips. This one asks it INSIDE one clip, which is a stronger comparison:
    the three foils are the same section at the same chord, at the same streamwise station, at
    exactly the same angle at every instant (`LAG` is all zeros and `CX` is no longer staggered),
    so the ONLY variable left in the frame is how fast the air is. Everything that differs between
    the three lanes - how far the wake carries, when the suction peak collapses, whether the stall
    is a violent shed or a slow thickening - is the flow rate and nothing else.

    HOW THE THREE RATES ARE MADE. There is one lattice and one inlet, so the speeds come from a
    SPANWISE INLET PROFILE (`LBM.set_inlet`, the same mechanism `eddies` uses to make currents):
    the inlet is fast across the top third of the span, intermediate across the middle and slow
    across the bottom, with a smoothed transition between bands. That is a legitimate scene
    input - it is a forcing of the fluid at a boundary, exactly what a tunnel
    with three independently-throttled sections would do - and nothing about what the flow then
    DOES is scripted. The bands hold their identity by momentum, not by a wall: viscous spreading
    over one domain transit is only ~7 cells against a ~213-cell band, so the interfaces stay
    sharp for the length of the clip and read as visible shear layers, which is honest and useful
    - they are the seam between two sections and they belong in the picture.

    Two supporting pieces, both load-bearing:
      - `prepare()` calls `init_velocity` with the profile everywhere, not just at the inlet.
        `settle` is 1440 LBM steps and one domain transit is ~5800, so a domain started uniform
        would open the clip with the bands only a quarter of the way across it.
      - `prepare()` also calls `LBM.set_farfield` with the SAME profile. The outlet sponge relaxes
        toward the freestream, and with a uniform target that is a pump on the exit plane pulling
        the slow band back up to `u0` (see the note on that method).

    STABILITY. Cut 1 anchored the top band at exactly the `u0` `tri_foil` was verified at so that
    no lane could be asked to do more than the verified scene already does. My first pass
    moved the bands to 1.25 / 0.75 / 0.55, which gives that up: the top lane now runs 25% ABOVE
    `u0` (0.0938 absolute) while the foils peak together rather than a third of a period apart,
    so peak blockage is simultaneous AND the fastest lane is faster than anything measured here
    before. `dbg.py` re-answered it - peak max|u| 0.2566 against the 0.45 guard, and against
    `aoa_sweep_fast`'s shipped 0.297 - so it stands on its own measurement rather than on
    `tri_foil`'s. Anything faster than this should go back to the standard move instead of just
    raising the numbers: scale every velocity down, buy the speed back with `steps`, and bring
    `re` down with u0 so `nu` (and therefore `tau`) is unchanged.

    Note the lanes also differ in Reynolds number, not just in speed - chord and viscosity are
    fixed, so the lower lanes sit at 0.55 and 0.28 of the top lane's Re. That is not a side
    effect to apologise for, it is what a slower tunnel section physically IS, and it is most of
    why the bottom wake is a thick laminar smear where the top one is a shed street.
    """
    # Same solver settings as `tri_foil` - u0 is the TOP band and every other band is a fraction
    # of it, so the verified velocity ceiling is unchanged. `legend` is on because in this scene
    # the colour mapping is the subject rather than the decoration: without the key, "the top is
    # green and the bottom is navy" is a look, and with it, it is a measurement.
    OVERRIDES = dict(TriFoil.OVERRIDES, streak_span=5.0, legend=True)
    duration = 15.0
    title = "THREE FLOW RATES"

    LAG = (0.0, 0.0, 0.0)          # IDENTICAL movement - the whole point of this variation
    CX = (0.32, 0.32, 0.32)        # ...and identical stations, so the lanes are truly comparable.
                                   # 0.32 rather than 0.30 buys ~50 px of clearance between the
                                   # legend's right edge and the TOP lane's leading edge; the
                                   # legend sits at the same screen height as that lane.
    BANDS = (1.25, 0.75, 0.55)     # inlet speed per lane as a fraction of u0: fast / mid / slow.
                                   # my call on the first pass (2026-07-30), up from the
                                   # 1.00 / 0.55 / 0.28 of cut 1 - faster everywhere and a
                                   # narrower spread (top/low 3.57x -> 2.27x). Note this puts the
                                   # TOP lane ABOVE `u0`, so it is no longer covered by
                                   # `tri_foil`'s verification by construction: 1.25 * 0.075 =
                                   # 0.0938 absolute, and `dbg.py` had to re-answer for it.
    SPEEDUP = 1.0                  # how many times the base wind this clip runs at. Bought with
                                   # `steps`, so it is a LEGEND-LABEL scale factor and nothing
                                   # else - see TriFoilRatesFast.
    BLEND = 44.0                   # cells over which one band eases into the next. Wide on
                                   # purpose: a step change is a vortex sheet the lattice cannot
                                   # resolve, and a thin shear layer rolls up into Kelvin-
                                   # Helmholtz billows that would out-shout the foils. At 44 cells
                                   # the layer e-folds about once per transit - visible as a seam,
                                   # never as the subject.

    def profile(self):
        """Inlet speed for every lattice row j - BANDS[0] first, BANDS[-1] last.

        Which screen edge j = 0 is depends on `flow`, and the ORDER of the bands rides along with
        it: the TOP of the screen in this scene's horizontal format, and the LEFT of it in the
        vertical (`flow="up"`) one the duel set runs in. Nothing here has to change for that -
        `sy()` already answers in whichever axis is cross-stream - but a subclass naming its lanes
        "top" and "bottom" in a vertical format would be naming the wrong edge.
        """
        j = np.arange(self.ny, dtype=np.float32)
        m = np.full(self.ny, self.BANDS[0], dtype=np.float32)
        for k in range(len(self.BANDS) - 1):
            # band boundary sits midway between two lanes, and the ramps never overlap
            e = 0.5 * (self.sy(self.LANES[k]) + self.sy(self.LANES[k + 1]))
            m = m + (self.BANDS[k + 1] - self.BANDS[k]) * smoothstep(
                e - 0.5 * self.BLEND, e + 0.5 * self.BLEND, j)
        return (self.cfg.u0 * m).astype(np.float32)

    def prepare(self, lbm):
        u = self.profile()
        lbm.set_inlet(u, None)
        lbm.set_farfield(u)
        lbm.init_velocity(np.tile(u[:, None], (1, self.nx)),
                          np.zeros((self.ny, self.nx), dtype=np.float32))

    def legend(self):
        # Tick POSITIONS are b / vmax and never see SPEEDUP; only the LABELS scale. That split is
        # the whole reason a doubled clip can carry a doubled legend without a single pixel of the
        # colour field moving - the ramp maps LATTICE speed, which `steps` does not touch.
        k = self.SPEEDUP
        full = self.cfg.vmax                      # full scale, in units of u0 (= the top band)
        marks = [(0.0, "0")]
        for lab, b in zip(("TOP", "MID", "LOW"), self.BANDS):
            marks.append((b / full, f"{b * k:.2f} {lab}"))
        marks.append((1.0, f"{full * k:.1f}"))
        # No footnote: I cut the "x top inlet" line on the first pass. The numbers are
        # multiples of u0 and the lane tags carry the meaning, so nothing on screen is now
        # claiming anything the ticks do not already say.
        return dict(title="FLOW SPEED", marks=marks)

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (all three)")]


class TriFoilRatesFast(TriFoilRates):
    """`tri_foil_rates` at DOUBLE the flow rate, 10 s. Nothing else in the picture may move.

    My brief, 2026-07-30: make a copy of it with double the flow rate, without altering the
    colours - the only thing that should change is the flow rate (and the legend along with it,
    in response).

    Those two constraints turn out to select the same implementation, which is worth seeing.
    Apparent speed is (lattice speed x LBM steps per frame), so wind is always bought with
    `steps` and never with `u0` - and the COLOUR FIELD is a map of
    lattice speed, which `steps` does not touch. So the one legal way to double the wind is also
    the only way to double it without recolouring anything: every frame's colour field here is
    bit-for-bit what the 15 s clip would have shown at the same *simulated* instant. Doubling
    `u0` instead would have done both forbidden things at once - shifted every lane up the ramp,
    and put the top lane at 0.1875 lattice units where blockage alone would have killed the solve.

    Consequences of that, all of them intended:
      - `u0`, `re`, `csm`, `tau`, `vmax`, `gamma`, the palette and the inlet profile are the
        already-verified ones, untouched.
      - `SPEEDUP = 2.0` scales the legend's LABELS only. Tick positions are unchanged, because
        the colour a given speed maps to is unchanged; what changed is what that colour means.
      - The sweep is written in SECONDS, so the foils turn at the same rate through twice the
        wind. Per LBM step they turn HALF as fast (trailing edge 0.0263 -> 0.0132 cells/step),
        which is the easy direction for a moving mask.
      - `streak_span` is quoted in FRAMES OF TRAVEL, so leaving it at 5.0 would double every dash
        length in pixels and smear the field. Halving it to 2.5 is what KEEPS the picture the one
        I approved. Same reasoning, same arithmetic, as `tri_foil_fast` and `aoa_sweep_fast`.

    The one thing that genuinely is NOT inherited: at 10 s and `steps=40` this runs **24000 LBM
    steps against the 15 s clip's 18000**, so it goes a third further into the solve than anything
    measured here. `dbg.py` had to cover that, and did - COMPLETED, peak max|u| 0.2346 against the
    shipped clip's 0.2566.
    """
    OVERRIDES = dict(TriFoilRates.OVERRIDES, steps=40, streak_span=2.5)
    duration = 10.0
    title = "THREE FLOW RATES - FAST"

    SPEEDUP = 2.0


class TriFoilAir(TriFoilRatesFast):
    """`tri_foil_rates_fast` with the FLUID changed from a viscous liquid to air.

    My note, 2026-08-03: the set looked amazing visually, but read as more representative of an
    incompressible fluid than actual air - I wanted the fluid profile altered to be as close to
    a 2-D airstream as possible.

    Nothing in the PICTURE is meant to change: same three lanes, same section, same chord, same
    sweep, same palette, same legend, same apparent wind. What changes is what the fluid IS.

    WHAT WAS ACTUALLY WRONG, MEASURED. Against a real subsonic tunnel the shipped clip is off on
    two axes, and only one of them is the one the brief names:

    | quantity            | tri_foil_rates_fast | real 2-D airstream | verdict            |
    |---------------------|---------------------|--------------------|--------------------|
    | Re on the chord     | 2000 / 1200 / 880   | 1e5 - 1e6          | 100x too VISCOUS   |
    | freestream Mach     | 0.162               | 0.05 - 0.15        | slightly too HIGH  |
    | peak local Mach     | 0.41                | ~0.2               | too HIGH           |

    So the honest answer to "it looks incompressible rather than like air" is that the *look*
    complained about is not compressibility at all - it is VISCOSITY. At Re 2000 the boundary
    layer is ~9 cells thick (5c/sqrt(Re)), shear layers are fat, separation is a slow laminar
    thickening and the wake is a smooth regular street. That is the syrup. Real air at a tunnel's
    Re carries a boundary layer under 1% of chord and a wake full of small scales. And the sim is
    if anything *more* compressible than the air it is imitating, not less: real air below M 0.3
    is treated as incompressible by every aerodynamicist, while this lattice was running local
    M 0.41, where D2Q9's O(M^2) error is a genuine spurious compressibility.

    Hence three changes, in order of how much they matter:

    1. **Re up ~10x** (chord Re 20000 / 12000 / 8800 against 2000 / 1200 / 880) - the dominant
       lever, and the one that turns syrup into air. Predicted laminar BL thickness goes 8.9 ->
       2.7 cells, i.e. from a visible band to a line on the surface.
    2. **`scale` 3.0 -> 2.0, so the chord is 119 cells instead of 79.** Not cosmetic and not
       optional: raising Re without resolution just moves the smallest real structure below the
       grid, where it aliases into speckle and the LES closure has to eat it. Costs 2.25x the
       cells. The Reynolds number is defined on the chord, so `re` itself is resolution-free.
    3. **The Mach number brought DOWN to a real tunnel's**, `u0` 0.075 -> 0.055, with `steps`
       40 -> 55 buying the apparent wind straight back (the same move as
       every other `_fast` clip, run in reverse). Top lane M 0.162 -> 0.119, peak local ~0.41 ->
       ~0.30. This reads backwards against the brief and is not: it is what makes the solve a
       faithful model of subsonic air rather than of a weakly compressible fluid nobody ordered.
       It also pays for (1), because the stability budget the higher Re spends is exactly the
       budget a lower lattice velocity frees up.

    ...and a fourth that is about air's CHARACTER rather than its numbers:

    4. **Continuous freestream turbulence** (`LBM.set_inlet_turbulence`, ~2% of each band's own
       speed at an integral scale of 0.15 chord). `perturb` is a single kick during settling and
       is switched OFF here, because it is the wrong model of the thing: a tunnel's air is noisy
       for the whole run, not once. A silent solver sheds on a metronome - evenly spaced, equal
       vortices - and that regularity is a large part of what makes the shipped clips read as a
       simulation of a liquid. Fed real background noise, the separating shear layer breaks down
       at an irregular time and place, which is what air does.

    WHAT THIS STILL IS NOT, and cannot be in 2-D. Worth stating plainly rather than implying the
    clip is a wind-tunnel measurement:
      - **2-D turbulence cascades the wrong way.** With no vortex stretching, energy runs to
        LARGE scales instead of down to dissipation, so a 2-D wake organises into big persistent
        vortices where a real one shreds into fine 3-D structure. No Reynolds number fixes that;
        it is the dimensionality. Every 2-D tunnel animation ever made has this.
      - **Re 20000 is still 10x short of a real tunnel**, and is bounded here by tau: at
        chord Re 20000 the molecular tau is 0.5012, and 0.5 is the wall. Going further needs a
        finer lattice, not a bigger number - `re` and `scale` have to move together.
      - **The boundary layer is not resolved** (~2.7 cells) and there is no transition model, so
        the stall angle is the lattice's, not a real 0015's.
      - **D2Q9 is isothermal (gamma = 1) and caps out near M 0.3**, so genuine compressible
        aerodynamics - a supersonic pocket, a shock on the upper surface, the one thing that
        would be unmistakably a gas - is out of reach for this solver at any setting. It needs a
        higher-order lattice (D2Q37) or a different scheme, which is a rewrite, not a parameter.

    Everything the viewer sees is held fixed on purpose, and each one took arithmetic rather than
    taste (all three are the same trap in a new costume - a quantity quoted in
    lattice cells or frames silently changes meaning when the lattice or the clock does):
      - the COLOUR FIELD is `speed / (u0 * vmax)` and every band is a fixed multiple of `u0`, so
        dropping `u0` leaves the normalised field - and therefore the legend's tick positions -
        bit-for-bit unchanged. `vmax`, `gamma` and the palette are untouched.
      - `SPEEDUP` stays 2.0 because the apparent wind is unchanged (0.055 x 55 = 3.03 cells/frame
        against 0.075 x 40 = 3.00).
      - `streak_span` is in FRAMES of travel and the dash is drawn in PIXELS, so halving the px
        per cell halves the dash unless the span is raised: 2.5 -> 3.7 holds it at 28 px.
      - `BLEND` is in CELLS, so 44 -> 66 keeps the band seam at the same 0.56 chord and the same
        width on screen. Leaving it would have sharpened the shear layer into exactly the
        Kelvin-Helmholtz roll-up the constant exists to prevent - and at 10x the Re.
      - `sponge_out` 14 -> 21 for the same reason: it is quoted in cells and has to stay the same
        fraction of a chord to absorb the same structures.
    """
    # `re` here is the config value; the TOP lane runs 1.25x it (BANDS[0]), so chord Re is
    # 20000 / 12000 / 8800. Set by measurement, not by choice, following the same Reynolds-number
    # ladder used throughout this file. csm carries more of the dissipation now that molecular
    # tau is 0.5012, but is NOT pushed to 0.35: measurement showed that damping the sweep's own
    # vortices, and
    # the content is what this clip is for.
    OVERRIDES = dict(TriFoilRatesFast.OVERRIDES,
                     scale=2.0, u0=0.055, steps=55, re=16000.0, csm=0.24,
                     streak_span=3.7, perturb=0.0, settle=2.5, sponge_out=21)
    duration = 10.0
    title = "THREE FLOW RATES - AIR"

    SPEEDUP = 2.0                  # apparent wind is unchanged, so the legend is too
    BLEND = 66.0                   # cells; 0.56 chord, as at scale 3.0
    TU = 0.020                     # |u'|_rms as a fraction of each band's OWN speed. Real tunnels
                                   # run 0.001-0.01; 2-D needs the top of that range and a little
                                   # more, because with no vortex stretching a disturbance has
                                   # only one route to break a shear layer down instead of three.
    TU_LEN = 0.15                  # integral length scale, in CHORDS - quoted against the body
                                   # rather than the lattice so it survives a `scale` change

    def turb_span(self):
        """Frozen-patch length in cells, DERIVED from how far the run actually convects it.

        The patch recycles once the flow has carried it past its own length, and a recycled patch
        means the same gust arrives twice - which would read as a periodicity the scene never
        asked for. Computing it from (settle + duration) x steps x u0 rather than pinning a
        constant is what keeps that true for a variant: `tri_foil_air_fast` doubles `steps` and
        so convects 4537 cells against the base clip's 2269, which a hardcoded 3072 would have
        silently wrapped. 15% headroom, rounded up to a power of two.
        """
        travel = (self.cfg.settle + self.duration) * 60.0 * self.cfg.steps * self.cfg.u0
        return int(2 ** np.ceil(np.log2(max(1024.0, travel * 1.15))))

    def prepare(self, lbm):
        super().prepare(lbm)       # inlet profile + matching far field + seeded interior
        # Tu is a property of a tunnel SECTION, so each band gets the same fraction of its own
        # speed rather than the same absolute wobble - which is why this is a profile and not a
        # scalar.
        lbm.set_inlet_turbulence(self.TU * (self.profile() / self.cfg.u0),
                                 length=self.TU_LEN * self.chord,
                                 span=self.turb_span(), seed=self.cfg.seed + 3)

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (all three)"),
                ("Re", " / ".join(f"{b * self.cfg.re:.0f}" for b in self.BANDS))]


class JetEngine(Scene):
    """A ducted jet engine in a DOWNWARD tunnel: idle, spool up over 6 s, cut, then run on.

    My brief, 2026-08-03: aim the flow downward and put the engine in the centre of the top half
    of the frame, exhaust pointing down. It sits idle for a second while the tunnel flows
    downward around it, then starts, and the distinct cells should form as it pushes flow
    downward and ramps in intensity - along with the surrounding tunnel changing around it -
    over 6 seconds, before cutting off and continuing to run for 3 more seconds.

    The frame is split by design: the engine lives entirely in the TOP half and everything it
    does happens in the BOTTOM half, which is left clear to watch. `flow="down"` (new - see
    render.py) is what makes that possible at all; every other scene here runs the stream
    upward or across.

    THE ENGINE IS A DUCT, NOT A BLOCK, and that is the whole design decision. Two mirrored
    `shapes.turbofan` pod halves face each other across the axis with a real duct between them,
    and the "fan" is a `set_drive` patch INSIDE that duct. So the engine does not have a jet
    attached to it - it has a pump in a pipe, and the jet, the intake suction that feeds it, and
    the entrainment that drags the surrounding tunnel inward are all the solver's answer to that
    one forcing. That keeps to the rule for a legitimate scene input: a drive patch is "a fan",
    the one kind of fluid forcing a scene is allowed, and nothing about what the flow then does
    is scripted.
    A solid block with a velocity patch bolted behind it would have looked similar for the first
    second and been a lie for the other nine - it could not draw air in, so it could not
    entrain, and the surrounding tunnel would not have changed at all.

    IT MUST ALSO LOOK LIKE ONE. The first cut of this scene built the two halves from the NACA
    thickness law, and my review was that the two-airfoil look was confusing when the goal was a
    simple jet-engine block with a funnel at the end - no airfoil should be visible in the video,
    the point was to see the flow around a stationary jet engine. Correct: a duct whose walls are
    airfoil sections IS two airfoils. `shapes.turbofan` is the replacement - a thick-walled pod,
    a flat barrel, a funnel down to a narrow exhaust tube, and a spinner cone in the mouth - and
    its docstring records what each feature is for and which alternatives were tried and looked
    wrong. The physics argument above is untouched by it; only the outline changed.

    WHAT THERE IS TO SEE, and why each of it happens rather than being animated:
      - **the cells.** The jet leaves the nozzle several times faster than the surrounding
        stream, and a shear layer between two speeds is Kelvin-Helmholtz unstable: it rolls up
        into a train of discrete vortices. Those are the cells. As `intensity` climbs the shear
        ratio climbs with it, so the roll-up starts closer to the nozzle and the vortices get
        stronger and tighter-spaced - the ramp is legible in the SPACING, not just the colour.
      - **the tunnel changing around it.** A jet entrains: it drags ambient fluid in sideways to
        feed itself, so the whole surrounding column bends toward the axis as thrust rises. The
        intake does the same thing upstream, above the engine.
      - **the cut.** At `T_CUT_AT` the drive stops. The jet does not vanish - the momentum
        already in the fluid keeps convecting, so the last cells travel on down the frame and
        the picture empties from the top. That tail is the reason the clip runs 3 s past the cut.

    THE CUT IS 0.25 s, NOT INSTANT, and that is a solver constraint rather than a stylistic
    choice: dropping a strong body force to zero in a single step is a pressure discontinuity,
    and this project has fixed the same artefact three times over (an impulse anywhere radiates
    a visible ripple across the whole frame). At 60 fps
    0.25 s is 15 frames, which still reads as a cut.
    """
    # `vmax` is 4.0, not the usual 2.2-2.6, because this scene's dynamic range is the point: the
    # freestream has to sit low on the ramp so the jet has somewhere to go. At 4.0 the tunnel
    # lands at a quarter of the ramp (deep blue) and the jet at full power near the top (red), so
    # the spool-up reads as the frame filling with colour. `perturb` is off - the shear layer is
    # convectively unstable and needs no help, and the continuous inlet turbulence below is the
    # honest version of that forcing anyway (see TriFoilAir).
    # `streak_span` is 4.2, well above the 2.5-3.0 the recent clips use, for the reason
    # `tri_foil_rates` needed 5.0: dash length is proportional to LOCAL speed, and this scene has
    # the widest speed range in the project - the idle tunnel at 1x u0 next to a jet at 3.5x. At
    # 3.0 the tunnel's dashes degenerate into dots and the calm half of the clip reads as noise.
    # The cost is that the jet's own streaks run long, which is what a streak photograph of a jet
    # looks like anyway. It is quoted in FRAMES OF TRAVEL, and the screen speed per frame is
    # unchanged by the `scale`/`steps` pair below, so it did not have to move with them.
    # `scale` 1.6 rather than the usual 2.0: the pod silhouette needs a thick wall AND a duct
    # wide enough to carry a shear layer inside a body capped at a quarter of the screen width,
    # and at 2.0 cells/px there were not enough cells to spend on both (see `shapes.turbofan`).
    # `steps` 75 then holds the backdrop's SCREEN speed exactly where it was - 60*0.05*2.0 =
    # 75*0.05*1.6 = 6.0 px/frame - so the tunnel still reads as moving rather than as a still
    # image with a jet painted on it, and the streak dashes keep their reviewed length.
    OVERRIDES = dict(flow="down", scale=1.6, u0=0.05, re=2141.0, csm=0.20, steps=75,
                     settle=3.0, vmax=5.0, gamma=0.95, overscan=0.08, perturb=0.0,
                     streak_span=4.2, streak_speed_gain=0.55, sponge_out=26, legend=True)
    duration = 10.0
    title = "JET ENGINE"

    # --- geometry, quoted against the SCREEN so the brief's limits are checkable -----------
    # The pod's own proportions live in `shapes.turbofan`; these two place and size it. The
    # brief caps the body at 25% of screen WIDTH and 50% of the top half's HEIGHT, and
    # `report()` measures both off the placed polygons rather than trusting the arithmetic.
    LEN = 0.211       # of screen HEIGHT -> 42.2% of the top half, and 24.7% of screen width
    CX = 0.25         # streamwise station of the engine's MID-LENGTH = centre of the top half

    # --- the fan --------------------------------------------------------------------------
    FAN_X = 0.62      # station of the drive patch along the engine, 0 = lip. Downstream of the
                      # spinner's tail (0.48) so the patch sees the full duct, not an annulus.
    FAN_HX = 0.13     # its half-extent streamwise, in engine lengths
    FAN_MARGIN = 3.0  # cells of clearance kept between the patch and the duct wall, so the drive
                      # never sits on a bounce-back cell where it would fight the boundary
    JET = 2.75        # peak FAN target speed, x u0 - the speed inside the duct at the patch, NOT
                      # at the nozzle. The duct then contracts 1.50x from the fan station to the
                      # exit, and a rendered cut measured the fan-to-exit gain at 1.50x its own
                      # contraction, so the jet leaves at roughly 2.75 * 1.89 = 5.2x u0. That
                      # exit ratio against the freestream is what sets how hard the shear layer
                      # rolls up; the number here is only what the drive is told to do.
    BETA = 0.030      # drive strength at full power. Residence time in the patch is ~200 steps
                      # against a relaxation time of 1/BETA = 33, so the fluid reaches the target
                      # comfortably; more would only sharpen the patch edge into a discontinuity.
    TU = 0.010        # continuous freestream turbulence, as a fraction of u0

    # --- the schedule ---------------------------------------------------------------------
    T_START = 1.0     # idle until here
    T_CUT_AT = 7.0    # 6 s of ramp, then cut; 3 s of run-on to t=10
    T_CUT = 0.25      # seconds to spool down (see the class docstring - not instant on purpose)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.length = self.LEN * self.vspan_stream()
        self.tf = shapes.turbofan()
        cowl = self.tf["cowl"]
        mirror = np.stack([cowl[:, 0], -cowl[:, 1]], 1)
        cx, cy = self.sx(self.CX), self.sy(0.5)
        # pivot=0.5 so `cx` is the engine's mid-length: the brief places the engine by its
        # CENTRE in the top half, and the pod runs exactly 0..1, intake to nozzle exit.
        self._polys = [shapes.place(p, self.length, cx, cy, 0.0, pivot=0.5)
                       for p in [cowl, mirror, self.tf["spinner"]] + self.tf["blades"]]
        self._nose = cx - 0.5 * self.length
        self._exit = cx + 0.5 * self.length
        self._axis = cy
        self._blob = self._fan_patch()

    def vspan_stream(self):
        """Visible extent along the STREAM, in cells (the screen's long axis in this format)."""
        return self.vnx if self._vert else self.vny

    def duct_r(self, s):
        """Duct half-radius at fraction `s` along the engine, in CELLS.

        Read straight off the shape's own radius law rather than restated here, so the patch
        clearance below cannot drift away from the wall that was actually rasterised.
        """
        return float(self.tf["duct_r"](s)) * self.length

    def _fan_patch(self):
        """The (ny, nx) weight field for the fan - a flat-topped, smooth-edged blob in the duct.

        Smooth edges are not cosmetic. A hard-edged body force is a step in the momentum source,
        and the lattice answers a step with an acoustic pulse - the same failure mode as a hard
        contact or a hard wall clamp. The 40% taper costs nothing and
        removes it.
        """
        x = np.arange(self.nx, dtype=np.float32)[None, :]
        y = np.arange(self.ny, dtype=np.float32)[:, None]
        cx = self._nose + self.FAN_X * self.length
        hx = self.FAN_HX * self.length
        hy = max(2.0, self.duct_r(self.FAN_X) - self.FAN_MARGIN)
        fx = 1.0 - smoothstep(0.60 * hx, hx, np.abs(x - cx))
        fy = 1.0 - smoothstep(0.60 * hy, hy, np.abs(y - self._axis))
        # `FAN_MARGIN` already holds the patch off the duct wall; this is the belt-and-braces
        # version and it costs one rasterisation at construction. A drive cell that lands on a
        # bounce-back cell is not merely wasted - it pushes against the boundary condition.
        solid = shapes.rasterize(self._polys, self.nx, self.ny)
        return (fx * fy * ~solid).astype(np.float32)

    def ref_len(self):
        """Re is built on the NOZZLE EXIT DIAMETER - the jet is the subject, and the shear layer
        that makes the cells scales with it, not with the engine's overall length.

        `re` 2500 against this length is chosen to land nu at 0.00124 / tau at 0.5037, which is
        the pair the first cut of this scene was verified at. The exit is wider in cells than it
        was, so holding nu fixed means the JET's Reynolds number rises with the resolution -
        which is the point of spending the cells - while the lattice stays exactly as far from
        the tau -> 0.5 stability wall as it was measured to be.
        """
        return 2.0 * float(self.tf["duct_r"](1.0)) * self.length

    def intensity(self, t):
        """Throttle, 0..1. Linear over the 6 s ramp with the first ~0.6 s eased in, then a cut.

        Linear rather than a full smoothstep on purpose: the brief is that the intensity should
        be seen to increase across the whole six seconds, and a smoothstep spends most of its
        change in the middle third and visibly stops moving near the end.
        """
        if t < self.T_START:
            return 0.0
        if t < self.T_CUT_AT:
            u = (t - self.T_START) / (self.T_CUT_AT - self.T_START)
            return float(u * smoothstep(0.0, 0.10, u))
        return float(1.0 - smoothstep(self.T_CUT_AT, self.T_CUT_AT + self.T_CUT, t))

    def bodies(self, t):
        return self._polys

    def prepare(self, lbm):
        # Real tunnel air is never quiet, and a shear layer needs something to roll up FROM (the
        # same argument as --perturb, made continuous - see TriFoilAir).
        lbm.set_inlet_turbulence(self.TU, length=0.5 * self.ref_len(), span=4096,
                                 seed=self.cfg.seed + 5)

    def forcing(self, lbm, t):
        """The fan. BOTH the target speed and the drive strength scale with the throttle.

        Scaling only the target would leave a fully-engaged fan clamping the duct flow to zero at
        idle - a brake, not an engine. Scaling only beta would have an idle engine weakly pulling
        the duct toward full jet speed. Scaling both gives the right limits at both ends: at
        throttle 0 there is NO force at all and the duct simply passes the tunnel flow, and at
        throttle 1 the fluid is driven hard toward `JET * u0`.
        """
        a = self.intensity(t)
        if a <= 0.0:
            lbm.set_drive(None, None, 0.0)
            return
        u_target = self.cfg.u0 * (1.0 + (self.JET - 1.0) * a)
        lbm.set_drive(u_target, 0.0, self._blob * (self.BETA * a))

    def legend(self):
        full = self.cfg.vmax
        # The middle mark is labelled FAN, not JET: it is the speed the drive patch targets
        # inside the duct, and the nozzle contraction means the plume leaves appreciably faster
        # than that. Calling it JET would put a number on the bar that the plume does not have.
        return dict(title="FLOW SPEED",
                    marks=[(0.0, "0"),
                           (1.0 / full, "1.0 TUNNEL"),
                           (self.JET / full, f"{self.JET:.2f} FAN"),
                           (1.0, f"{full:.1f}")])

    def readout(self, t):
        return [("THROTTLE", f"{100 * self.intensity(t):3.0f} %")]

    def report(self):
        """The brief's limits, in screen percent - measured off the PLACED polygons.

        Measured, not asserted, and measured from the polygons rather than recomputed from the
        constants: that is what makes this a check. An arithmetic restatement of LEN and the
        pod's radii would agree with itself no matter what shape was actually built.
        """
        sc = self.cfg.scale
        half = 0.5 * self.cfg.height
        outer = max(float(np.abs(p[:, 1] - self._axis).max()) for p in self._polys)
        x0 = min(float(p[:, 0].min()) for p in self._polys)
        x1 = max(float(p[:, 0].max()) for p in self._polys)
        return dict(
            width_pct=100.0 * (2.0 * outer * sc) / self.cfg.width,   # limit 25
            len_pct_of_top_half=100.0 * ((x1 - x0) * sc) / half,     # limit 50
            top_px=(x0 - self.vx0) * sc,                             # both must stay < half
            bottom_px=(x1 - self.vx0) * sc,
            duct_exit_cells=2.0 * self.duct_r(1.0),
            fan_patch_cells=2.0 * max(2.0, self.duct_r(self.FAN_X) - self.FAN_MARGIN),
            wall_cells=(self.tf["r_out"] - float(self.tf["duct_r"](0.40))) * self.length)


class TriFoilAirFast(TriFoilAir):
    """`tri_foil_air` at DOUBLE the wind, bought the standard way - and deliberately NOT with Mach.

    My call, 2026-08-03, after seeing what a compressible run costs: not to push toward the sound
    limit. So the doubling that `tri_foil_gas` bought by raising the Mach number is bought here
    by `steps` instead, which is the move every other `_fast` clip in this project uses
    and the one the earlier measurement actually argued for: at
    wind-tunnel speeds real air IS effectively incompressible, so a faster clip has no physical
    need to go anywhere near M 1.

    What that buys, beyond honesty:
      - the per-step physics is **bit-for-bit the verified `tri_foil_air`** - `u0`, `re`, `csm`,
        `tau`, the palette, `vmax`, `gamma` and the inlet profile are all untouched, so the colour
        field is identical at the same *simulated* instant and the legend's tick positions cannot
        have moved. Only what it means changes, which is what `SPEEDUP` is for.
      - it runs on the LBM at ~14 ms/step instead of the compressible solver's ~50, and needs 110
        steps/frame instead of 128, so it is a ~20 minute render against ~70.

    `streak_span` 3.7 -> 1.85 is the usual correction, not a change: the span is quoted in FRAMES
    OF TRAVEL, so leaving it alone would double every dash in pixels and smear the field. Measured
    it holds the top-lane dash at 28.0 px.

    THE ONE THING NOT INHERITED, and the reason this needs its own verification run rather than
    resting on `tri_foil_air`'s: total lattice time. 110 steps/frame over 10 s is **66000 LBM
    steps against 33000** - twice as far into the solve as anything measured on this lattice.
    That is the same units-change trap exactly, and it is also why `turb_span()` is derived: at this step
    count the frozen turbulence patch convects 4537 cells, and the 3072 that was ample for the
    base clip would have recycled and delivered the same gust twice.
    """
    OVERRIDES = dict(TriFoilAir.OVERRIDES, steps=110, streak_span=1.85)
    duration = 10.0
    title = "THREE FLOW RATES - AIR, FAST"

    SPEEDUP = 4.0                  # legend LABELS only; 2x tri_foil_air, 4x tri_foil_rates


class TriShapes(TriFoilAirFast):
    """`tri_foil_air_fast` with the VARIABLE moved from the air to the BODY: one speed, three shapes.

    My brief, 2026-08-04: rather than three identical foils at different rates, keep the top rate
    on all three lanes and vary the foil design instead - the symmetric foil on top, the standard
    cambered airfoil in the middle, and a 2-D side-profile cow image for the third, with the
    airfoils moving as in earlier clips but the cow held stationary.

    This is the exact inverse of `tri_foil_rates` and its descendants, and that is the whole
    reason it is worth a clip. Those held the body fixed and varied the stream, so everything that
    differed down the frame was caused by flow rate. Here the stream is IDENTICAL in all three
    lanes - `BANDS` is flat, so the inlet, the far field and the seeded interior are one uniform
    1.25x`u0` and there are no bands, no shear seams and no `BLEND` - and the only variable left
    is what is standing in it. Same chord, same station, same sweep, same air.

    THE THREE BODIES, and why each one earns its lane:

      - **NACA 0015, symmetric** (top). No camber, so at 0 deg it makes no lift and sheds nothing:
        every feature in that lane is bought by incidence alone, and the picture is exactly
        mirror-symmetric between the two halves of each sweep. The control.
      - **the COW** (middle), stationary. My ordering, 2026-08-04: a foil on the top, a foil on
        the bottom, with the cow in the middle. That puts the one bluff body
        between the two streamlined ones, which is the comparison the clip is actually making. It
        holds still on purpose: with nothing moving in that lane, everything its wake does is the
        SHAPE's, so the frame carries a fixed reference for how a non-streamlined body behaves in
        exactly the air the two foils are working in. It is also just funny, which is allowed.
      - **NACA 2412, cambered** (bottom) - my "standard airfoil"; the section `aoa_sweep`
        runs and the one on half the light aircraft ever built. 2% camber at 40% chord over a 12%
        thickness. Against the symmetric section at the same angle in the same air the difference
        IS camber: it carries a suction peak at zero incidence, its two halves of the cycle are no
        longer mirror images, and it stalls at a different angle on each side.

    THE COW IS A PHOTOGRAPH, NOT A DRAWING, and that is a change to the ENGINE rather than to this
    scene. I supplied a cut-out and asked for a real cow shape and colour instead of a black
    shape (not for the airfoils). `shapes.image_body` turns the PNG's alpha into an
    ordinary closed polygon - so `place`, `rasterize` and the solver need no special case at all -
    and `render._draw_texture` paints that body from the image instead of `body_fill`, sampled
    through the same lattice-to-screen map the field is warped by. The two foils stay black on
    purpose, per the brief: a NACA section is a line drawing and a photograph of one would be a
    photograph of a wing. The split that makes this work is worth keeping in mind if it is reused:
    the polygon is traced at 3x the lattice so the PICTURE keeps the tail and the legs, and welded
    shut at 1.5 CELLS so the SOLVER is never handed a passage it cannot resolve.

    THE FIELD IS DIMMED, and that is the other thing this clip changed. My note: the green looked
    very bright, and I wanted the colours associated with the normal wind tunnel toned down in
    brightness. With every lane at the same speed the freestream covers
    most of the frame, and `jet`'s freestream colour is the single most luminous entry in the
    palette (see `colormap._tone`). `brightness` 0.72 with a light `saturation` 0.92 acts on the
    COLORMAP only - the white tracer dashes and the bodies composite afterwards at full contrast,
    so the flow structure actually reads BETTER against the tamed background rather than being
    washed out with it. Everything about the physics, the normalisation and the legend's meaning
    is untouched; `--sat 1 --bright 1` restores the parent clip's palette exactly.

    STABILITY IS ITS OWN QUESTION AGAIN, not an inheritance from `tri_foil_air_fast`. That clip
    ran the middle and bottom lanes at 0.75 and 0.55 of the top band; here they run at 1.25 like
    the top one, so two thirds of the frame is now faster than it has ever been measured at on
    this lattice. Blockage is what sets the peak here (the flat Re ladder used throughout this set), and it
    goes DOWN - the cow at rest blocks less than a foil at 34 deg - but the fast lanes multiply
    what is left, so `dbg.py` re-answers it rather than the docstring asserting it.
    """
    OVERRIDES = dict(TriFoilAirFast.OVERRIDES, saturation=0.92, brightness=0.72)
    duration = 9.0
    title = "THREE SHAPES - ONE AIRSTREAM"

    BANDS = (1.25, 1.25, 1.25)     # one speed everywhere: the TOP rate of the parent clip, kept
                                   # for all three lanes. Flat, so `profile()` is a constant and
                                   # BLEND has nothing to blend - by construction, not by luck.
    CODES = ("0015", None, "2412")  # None = the cow, which is not a NACA anything
    STATIC = (False, True, False)   # the cow does not move
    COW = os.path.join(ROOT, "assets", "cow.png")   # bring your own; the subject faces RIGHT
    COW_LEN = 0.86                  # of the chord. A cow of the same LENGTH as the sections
                                    # reads as an outsized animal next to two thin lines, and it
                                    # is the only body whose height is set by its own aspect
                                    # ratio rather than by a thickness law, so it is quoted
                                    # separately instead of inheriting `chord`.

    def __init__(self, cfg):
        super().__init__(cfg)
        # The two SECTIONS are mirrored; the cow is NOT, and the difference is where each one's
        # y sign came from. This is the horizontal format, where lattice +y is screen-DOWN, so a
        # profile drawn the natural way up (camber up, flat side down) arrives inverted and needs
        # `flip_y` - which does not touch the AoA convention, and leaves the symmetric 0015
        # unchanged either way. The cow's points are traced from IMAGE ROWS, which already run
        # downward, so it is in lattice convention to begin with; flipping it would stand it on
        # its spine.
        self.cow = shapes.image_body(need_asset(self.COW, "a cow"), cells=self.COW_LEN * self.chord,
                                     flip_x=True)
        self.profs = [shapes.flip_y(shapes.naca4(c)) if c else self.cow["points"]
                      for c in self.CODES]
        self.prof = self.profs[0]
        self._chords = [self.chord if c else self.COW_LEN * self.chord for c in self.CODES]

    def aoa(self, k, t):
        return 0.0 if self.STATIC[k] else super().aoa(k, t)

    def bodies(self, t):
        return [shapes.place(self.profs[k], self._chords[k], self.sx(self.CX[k]), self.sy(fy),
                             self.aoa(k, t))
                for k, fy in enumerate(self.LANES)]

    def body_textures(self):
        """Only the cow is painted; the two sections stay flat black (my call)."""
        return [None if c else self.cow["rgb"] for c in self.CODES]

    def legend(self):
        """One speed, so one mark. The parent's TOP/MID/LOW key would be three labels for three
        identical numbers - which is worse than no key, because it would claim a comparison the
        frame no longer contains."""
        full = self.cfg.vmax
        b = self.BANDS[0]
        return dict(title="FLOW SPEED",
                    marks=[(0.0, "0"), (b / full, f"{b * self.SPEEDUP:.2f} FLOW"),
                           (1.0, f"{full * self.SPEEDUP:.1f}")])

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (both foils)"),
                ("Re", f"{self.BANDS[0] * self.cfg.re:.0f} (all lanes)")]


class TriFoilGas(TriFoilAir):
    """`tri_foil_air` on the COMPRESSIBLE solver, at DOUBLE the flow speed.

    My call, 2026-08-03: do a solver rewrite, and double the speed of all three flows -
    answering the closing note of `tri_foil_air`, that a shock is a solver rewrite away rather
    than a parameter. `wt\\cns.py` is that rewrite: a real ideal gas, gamma = 1.4, with genuine
    shock capture. Read its module docstring for why finite-volume HLLC and not D2Q37.

    THE UNIT CHANGE IS THE WHOLE TRAP HERE. Under the LBM, `u0` is a lattice velocity and the
    Mach number is `u0 / (1/sqrt(3))`. Under CNS the freestream sound speed is 1 by construction,
    so **`u0` IS the Mach number**. `tri_foil_air` ran u0 = 0.055 lattice units = M 0.0953 on the
    reference, i.e. M 0.119 in the top lane. Doubling THE PHYSICS, not the number, means

        M_ref 0.0953 -> 0.1906, so u0 = 0.19 here

    and the bands become **M 0.2375 / 0.1425 / 0.1045** against 0.119 / 0.0715 / 0.0524. Copying
    `u0 = 0.11` across would have been a factor-of-two error dressed up as a doubling.

    WHAT "DOUBLE THE SPEED" NOW MEANS, AND WHY IT NEEDS NO SECOND DECISION. In every earlier
    `_fast` clip the wind had to be bought with `steps`, because `u0` was pinned by the LBM's
    Mach ceiling. That constraint is gone: this solver is *made* for
    Mach numbers. So the velocities themselves double, and because apparent speed is
    (distance per step x steps per frame), the on-screen wind doubles with them - both readings
    of the brief are satisfied by the same change. Chord Reynolds number doubles too, to
    **40000 / 24000 / 17600**, which is simply what a real tunnel does when you turn the fan up.

    WHAT THE COMPRESSIBILITY ACTUALLY BUYS AT THIS SPEED - stated in advance so the clip is not
    oversold. M_inf 0.238 is still firmly subsonic, and a freestream shock is not on the table.
    What IS on the table is that the suction peak multiplies the freestream: `tri_foil_air`
    measured peak local |u| at 3.08x the top band, and compressibility amplifies that further
    (Prandtl-Glauert goes as 1/sqrt(1 - M^2)), so peak LOCAL Mach should land near 0.75-0.9 - the
    transonic threshold, where density varies by tens of percent over the upper surface and the
    flow stops being describable as incompressible at all. Whether a genuine supersonic pocket
    appears at peak incidence is a MEASUREMENT, not a claim: a Mach/CFL check tool (not included
    here) reports the peak local Mach and the supersonic area every second, and whatever it says
    is what goes in the docs.

    `re` ALSO NEEDS CONVERTING, and in the opposite direction to the obvious one. `re` is defined
    as `u0 * ref_len / nu`, so carrying 16000 across while doubling `u0` doubles `nu` with it and
    leaves the chord Reynolds number exactly where it was - the fan turned up and the air got
    twice as thick to compensate, which is not what a tunnel does. Holding the FLUID fixed and
    doubling only the speed means `re` -> **32000**, giving chord Re 40000 / 24000 / 17600.
    Worth noting that this is free here in a way it never was on the lattice: the LBM paid for Re
    in stability margin (tau -> 0.5) whereas an explicit compressible solver
    is convection-limited, and at nu = 7.1e-4 the viscous step limit is dt < 355 against a dt of
    0.25. **The rewrite lifts the Reynolds ceiling as well as the Mach ceiling**, and the binding
    constraint on Re is now grid resolution alone.

    Everything the viewer sees is held to `tri_foil_air`'s composition again, and again by
    arithmetic rather than by eye:
      - `scale` unchanged, and `steps` 55 -> 128. The CNS time step is FIXED (see its docstring)
        at dt = cfl / (blockage*u0 + c_max) = 0.2496, so the top lane covers
        0.2375 * 0.2496 = 0.0593 cells/step against the LBM's 0.0688. 128 steps/frame therefore
        puts the apparent wind at 7.59 cells/frame against `tri_foil_air`'s 3.78 - the doubling
        asked for, to 0.4%. (It is more steps per frame than any clip in this project; a
        sound-speed-limited explicit scheme simply costs that, and the render time follows.)
      - `streak_span` 3.7 -> 1.85, because the span is in FRAMES OF TRAVEL and the wind doubled;
        this is what HOLDS the dash at 28 px rather than letting it smear to 56.
      - `vmax` 2.60 is unchanged and still means "2.6 x u0", so the colour field and every legend
        tick position are in the same places as before. `SPEEDUP` becomes 4.0: the legend's
        LABELS are relative to `tri_foil_rates`' original wind, and this clip is 4x that.
    """
    # `blockage` feeds the fixed-dt estimate and is the one number that can kill the run: it has
    # to bound peak |u| / u0 for the whole clip or the CFL is violated. `tri_foil_air` MEASURED
    # 3.85x, and compressibility steepens the suction peak further, so 4.6 carries real margin -
    # and the only cost of being generous is render time. A CFL check tool (not included here)
    # reports the CFL actually achieved, so this is verified rather than hoped.
    OVERRIDES = dict(TriFoilAir.OVERRIDES,
                     solver="cns", u0=0.19, steps=128, re=32000.0, csm=0.18,
                     streak_span=1.85, settle=2.0,
                     extras=dict(blockage=4.6, cfl=0.50, vmax_mach=1.4,
                                 schlieren_gain=260.0))
    duration = 10.0
    title = "THREE FLOW RATES - COMPRESSIBLE"

    SPEEDUP = 4.0                  # legend labels only; 2x tri_foil_air, 4x tri_foil_rates
    TU = 0.020
    TU_LEN = 0.15


class MachSweep(Scene):
    """ONE symmetric section in a tunnel that is throttled up from fast subsonic to SONIC.

    My brief, 2026-08-04: one symmetric airfoil, flow pointing upward, starting at the fast flow
    rate and then quickly jumping to much higher speeds to visualize real high-speed behaviour -
    fast rate for 0-2 s, accelerate to high speed 1 over 1 s, hold for 2 s, accelerate to high
    speed 2 over 1 s, hold - with high speed 2 at Mach 1.

    THIS IS THE CLIP `tri_foil_air` SAID WAS OUT OF REACH, so it runs on the other solver.
    D2Q9 is isothermal and its error is O(Ma^2); it degrades continuously and then fails above
    M ~ 0.3, which is why every LBM clip here buys speed with `steps` and keeps the lattice
    velocity small. A schedule that ENDS at M 1 cannot be faked that way - the whole subject is
    what happens when a real gas is pushed to its own sound speed. So `solver="cns"`
    (`wt\\cns.py`, MUSCL + HLLC, gamma = 1.4, validated against Sod and the theta-beta-M
    relation) - the rewrite that was built for `tri_foil_gas` and then cut unrendered because
    that scene had no compressible SUBJECT. This one does.

    AND THE UNITS CHANGE WITH THE SOLVER (the silent trap). Under CNS the
    freestream sound speed is 1 by construction, so **`u0` IS a Mach number**. `u0 = 1.00` here
    is not a lattice velocity a hair under the stability wall, it is the DESIGN POINT: Mach 1,
    the top of the schedule. Everything else is quoted against it - `mu` from `re` at that speed,
    the fixed `dt` from `blockage * u0 + c_max`, and the colour normalisation.

    THE SCHEDULE IS A BOUNDARY FORCING, which is a legitimate scene input - a
    tunnel whose fan is being turned up. `forcing()` re-points the inlet AND the outlet sponge at
    the same Mach every frame (never one without the other: a sponge left targeting the old
    freestream is a pump on the exit plane). Nothing about what the gas then
    does is scripted: the compression wave that runs downstream when the throttle moves, the
    supersonic pocket that opens on the suction side, the shock that terminates it and marches
    aft as M rises, and whatever the tunnel does when it can no longer swallow the flow, are all
    the solver's answers.

    WHY ONLY 5% BLOCKAGE, AND WHY A 12% SECTION. Sonic freestream plus ANY blockage is a choked
    duct - at M = 1 the tunnel is already at the minimum area that will pass its own mass flow,
    so there is no steady state with a body in it, and the side walls here are free-slip, which
    makes blockage REAL. Real transonic tunnels answer this with slotted or perforated
    walls; this one cannot, so it answers with the only other lever: keep the body small. NACA
    0012 rather than the 0015 the tri-foil family uses (0012 is also the canonical transonic
    test section), a chord of 0.20 of the span rather than `aoa_sweep`'s 0.36, and the incidence
    schedule below eases off as the speed climbs. Peak blockage lands at ~9% during the subsonic
    stall sweep and ~5% for the sonic hold. A check tool (not included here) measures the
    freestream at a station upstream of the body, which is the number that says whether the
    frame is honestly at the Mach the inlet is being held at.

    THE INCIDENCE SCHEDULE IS A RIG SCHEDULE, NOT A FUDGE. Rule 1 lets a scene put a body at a
    position at time t, and nobody holds a foil at 16 deg while taking a tunnel to Mach 1. So the
    stall sweep happens where it belongs - in the fast SUBSONIC opening, at M 0.30 - and the
    transonic and sonic phases are flown at a small fixed incidence with a gentle oscillation on
    top, which is what gives the terminating shock something to march against instead of a frozen
    picture. Slew rate is a non-issue on this solver: the fastest segment puts the trailing edge
    at 0.005 cells/step against the LBM scenes' 0.007-0.043.

    THE COLOUR IS `field="mach"`, not speed, and that is the honest choice for this clip.
    I asked for the base fluid to read deep blue instead of green, and there are two ways to
    get there - a blue palette, or the same `jet` ramp renormalised so the opening freestream
    sits at the bottom of it. The second is strictly better here, because the clip's entire
    content is the frame FILLING with colour as the gas speeds up, which a blue-only map would
    throw away. `vmax_mach = 1.80` puts M 0.30 in deep blue, M 0.70 in cyan, the SONIC LINE at
    green and a supersonic pocket in red - and unlike a speed field, a Mach field puts M = 1 at a
    fixed, quotable place on the bar, which is the one question a compressible picture has to be
    able to answer. `gamma` 1.15 rather than the usual 0.95 is a gamma lever, not a
    LUT edit: above 1 it crushes the slow field down the ramp, which is what makes the opening
    read as deep blue and the separated wake as navy.

    ONE THING TO EXPECT AND NOT MISREAD: the dashes get LONGER as well as faster. `streak_span`
    is quoted in frames of travel, so at a fixed span the dash length in pixels is proportional to
    the local speed - 1.40 holds the opening dash near 18 px and lets the sonic hold run at ~60.
    That is the speed reading, deliberately left in; every `_fast` clip in this project had to
    CANCEL it because those doubled the wind and had to keep the approved picture, and this one
    is the opposite case.
    """
    OVERRIDES = dict(
        solver="cns", flow="up", scale=2.4, overscan=0.12,
        u0=1.00,            # a MACH NUMBER here - the design point, i.e. the top of the schedule
        re=40000.0,         # on the chord, at the design Mach; 0.30x that in the opening phase
        csm=0.20, steps=134, settle=2.0, sponge_out=30, perturb=0.0,
        field="mach", cmap="jet", vmax=1.90, gamma=1.15,
        streak_span=1.40, streak_speed_gain=0.42, legend=True,
        extras=dict(cfl=0.50, blockage=1.80, vmax_mach=1.80),
    )
    duration = 9.0
    title = "SUBSONIC TO SONIC"

    CODE = "0012"                  # symmetric, and the canonical transonic section
    CHORD = 0.20                   # of the span. `aoa_sweep` runs 0.36; see the blockage note.
    CX = 0.32                      # streamwise station: low in the frame, so the wake gets the
                                   # upper two thirds of 1920 px
    CY = 0.50

    # The throttle, in absolute Mach. Read straight off the brief: hold, 1 s ramp, hold, 1 s
    # ramp, hold. `keyframes` is smoothstep between knots, so each ramp starts and ends with zero
    # slope - a step change in a hard velocity BC would be a shock tube fired at the inlet plane.
    # My note, 2026-08-05, on the first cut: compress the time before the first acceleration
    # from 2 seconds down to 1 - otherwise it was working well. So the opening
    # hold halves and the extra second goes to the SONIC hold at the end - the ramps and the M 0.70
    # plateau keep their lengths, because those are what the eye reads as "it accelerated" and
    # "it held", and shortening either would change the clip I approved rather than tighten it.
    M_START, M_HIGH1, M_HIGH2 = 0.30, 0.70, 1.00
    SCHEDULE = ((0.0, "M_START"), (1.0, "M_START"), (2.0, "M_HIGH1"),
                (4.0, "M_HIGH1"), (5.0, "M_HIGH2"), (9.0, "M_HIGH2"))

    TU = 0.003                     # |u'|_rms as a fraction of the DESIGN Mach, so ~1% of the
                                   # opening freestream and ~0.3% at the sonic hold - a real
                                   # tunnel's 0.1-1%, and continuous rather than a settling kick
                                   # (the air recipe, point 5)
    TU_LEN = 0.15                  # integral scale in CHORDS, so it survives a `scale` change
    TU_CONV = 0.60                 # the frozen patch is read at a FIXED convection speed and the
                                   # real one triples over the clip; the mean is the honest
                                   # single value, and the span below is sized on it

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prof = shapes.naca4(self.CODE)
        self.chord = self.CHORD * self.wspan
        self.cx = self.sx(self.CX)
        self.cy = self.sy(self.CY)

    def ref_len(self):
        return self.chord

    def mach(self, t):
        return keyframes(t, [(tt, getattr(self, k)) for tt, k in self.SCHEDULE])

    def aoa(self, t):
        """Stall sweep while it is subsonic, then a small held incidence with a slow wobble.

        The wobble is not decoration: a transonic shock sits where the local supersonic pocket
        closes, so a foil frozen at one angle gives a frozen shock. +-2.5 deg at 0.5 Hz walks it
        along the chord, which is the thing the clip is for.
        """
        base = keyframes(t, [(0.0, 0.0), (0.25, 0.0), (0.85, 16.0), (1.15, 16.0),
                             (1.85, -16.0), (2.15, -10.0), (2.70, 5.0),
                             (9.00, 5.0)])
        wob = 2.5 * smoothstep(2.7, 3.5, t) * float(np.sin(2 * np.pi * 0.5 * (t - 2.7)))
        return base + wob

    def bodies(self, t):
        return [shapes.place(self.prof, self.chord, self.cx, self.cy, self.aoa(t))]

    def turb_span(self):
        """Frozen-patch length in cells, derived from how far the patch is actually READ.

        `cns._bcs` indexes it at `steps_done * u_conv * dt`, so what matters is TU_CONV and the
        total step count, not the real (changing) freestream speed. Same derivation as
        `TriFoilAir.turb_span` and for the same reason: a patch that recycles delivers the same
        gust twice.
        """
        n = (self.cfg.settle + self.duration) * 60.0 * self.cfg.steps
        travel = n * self.TU_CONV * self._dt_estimate()
        return int(2 ** np.ceil(np.log2(max(1024.0, travel * 1.15))))

    def _dt_estimate(self):
        """The CNS fixed time step, reproduced from the solver's own formula.

        Duplicated rather than read off the instance because `turb_span()` is wanted at
        parameter-sanity time, before a solver exists. It is asserted against the real `dt` in
        `prepare()`, so the two cannot drift apart silently.
        """
        e = self.cfg.extras
        u0 = self.cfg.u0
        cmax = 1.0 + 0.5 * (1.4 - 1.0) * (2.0 * u0) ** 2 + 0.10
        return e.get("cfl", 0.50) / (e.get("blockage", 4.2) * u0 + cmax)

    def prepare(self, sim):
        assert abs(sim.dt - self._dt_estimate()) < 1e-9, "dt estimate drifted from wt/cns.py"
        u = np.full(self.ny, self.M_START, dtype=np.float32)
        sim.set_inlet(u, None)
        sim.set_farfield(u)
        sim.init_velocity(np.tile(u[:, None], (1, self.nx)),
                          np.zeros((self.ny, self.nx), dtype=np.float32))
        sim.set_inlet_turbulence(self.TU, length=self.TU_LEN * self.chord,
                                 span=self.turb_span(), seed=self.cfg.seed + 3,
                                 u_conv=self.TU_CONV)

    def forcing(self, sim, t):
        """Turn the fan up. Inlet AND far field together - see the class docstring."""
        u = np.full(self.ny, self.mach(t), dtype=np.float32)
        sim.set_inlet(u, None)
        sim.set_farfield(u)

    def legend(self):
        full = self.cfg.extras.get("vmax_mach", 1.80)
        return dict(title="MACH", marks=[
            (0.0, "0"),
            (self.M_START / full, f"{self.M_START:.2f}"),
            (self.M_HIGH1 / full, f"{self.M_HIGH1:.2f}"),
            (1.0 / full, "1.00 SONIC"),
            (1.0, f"{full:.2f}")])

    def readout(self, t):
        return [("M", f"{self.mach(t):.2f}"), ("AoA", f"{self.aoa(t):+.1f} deg")]


class _MachTrio(MachSweep):
    """Shared base for the THREE-LANE compressible clips: `mach_sweep`'s throttle, `tri_foil`'s
    frame.

    My note, 2026-08-05, after `mach_sweep`: use the setup of `tri_foil_rates_fast` with the Mach
    choreography - just borrow the three-lane, three-foil arrangement.
    So everything about TIME comes from `MachSweep` - the M 0.30 -> 0.70 -> 1.00 throttle, the
    stall sweep that runs while it is still subsonic, the held incidence with a wobble under it,
    the Mach colour ramp with the sonic line at green - and everything about SPACE comes from
    `TriFoil`: `flow="right"`, so the stream crosses the 1080 px width and the full 1920 px
    becomes cross-stream span, which is the only way to stack three sections with real distance
    between them.

    `BANDS` multiplies the throttle PER LANE, exactly as it does on the lattice side, so a
    subclass sets either three rates and one body (`mach_rates`) or one rate and three bodies
    (`mach_foils`, `mach_shapes`) - the same "only one variable in frame" discipline the whole
    tri-foil family is built on.

    TWO THINGS THAT ARE NOT INHERITED and had to be re-derived, both because this is the
    compressible solver rather than the lattice:

      - **`blockage` is the number that can kill the run.** `wt\\cns.py` fixes dt once from
        `blockage * u0 + c_max`, and an explicit Godunov scheme does not degrade past CFL 1, it
        fails. Three bodies in a duct block several times what one does, and a banded scene runs
        its top lane ABOVE the nominal Mach on top of that. Every subclass therefore carries its
        own `blockage`, and a Mach/CFL check tool (not included here) is run at the SONIC hold -
        the worst case - before any render.
      - **the sections are thinner and the chord is smaller than the lattice clips use.** A sonic
        freestream plus blockage is a choked duct (see `mach_sweep`'s measured choking), and with
        three bodies instead of one there is three times as much of it. Chord drops 0.22 -> 0.17
        of the span and `tri_foil`'s 15%-thick section gives way to 12% or less.
    """
    OVERRIDES = dict(MachSweep.OVERRIDES, flow="right", scale=2.4, overscan=0.10,
                     steps=140, streak_span=1.30,
                     extras=dict(cfl=0.50, blockage=2.60, vmax_mach=1.80))
    duration = 9.0

    LANES = (0.20, 0.50, 0.80)     # cross-stream placement = down the screen
    CX = (0.34, 0.34, 0.34)        # identical streamwise stations, so the lanes are comparable
    BANDS = (1.0, 1.0, 1.0)        # per-lane multiple of the throttle
    CODES = ("0012", "0012", "0012")
    CHORD = 0.20                   # of the visible streamwise span. Down from `tri_foil`'s 0.22:
                                   # three bodies in a tunnel that has to stay unchoked. Measured
                                   # rather than guessed - at 0.20 the SONIC-hold blockage is
                                   # ~4.6% against `mach_sweep`'s 3.3%, which is the number the
                                   # choking scales with.
    BLEND = 60.0                   # cells of smoothstep between bands (only used when BANDS vary)

    def __init__(self, cfg):
        Scene.__init__(self, cfg)
        self.chord = self.CHORD * self.vspan
        # `None` means "this lane is not a NACA section" - a subclass fills it in (the cow).
        self.profs = [shapes.flip_y(shapes.naca4(c)) if c else None for c in self.CODES]
        self.prof = self.profs[0]
        self._chords = [self.chord] * len(self.CODES)

    def ref_len(self):
        return self.CHORD * self.vspan

    def profile(self, t):
        """Inlet Mach for every lattice row j: the throttle, times each lane's band."""
        m = np.full(self.ny, self.BANDS[0], dtype=np.float32)
        for k in range(len(self.BANDS) - 1):
            e = 0.5 * (self.sy(self.LANES[k]) + self.sy(self.LANES[k + 1]))
            m = m + (self.BANDS[k + 1] - self.BANDS[k]) * smoothstep(
                e - 0.5 * self.BLEND, e + 0.5 * self.BLEND,
                np.arange(self.ny, dtype=np.float32))
        return (m * self.mach(t)).astype(np.float32)

    def prepare(self, sim):
        u = self.profile(0.0)
        sim.set_inlet(u, None)
        sim.set_farfield(u)
        sim.init_velocity(np.tile(u[:, None], (1, self.nx)),
                          np.zeros((self.ny, self.nx), dtype=np.float32))
        # Tu is a property of a tunnel SECTION, so each lane gets the same fraction of its own
        # speed - a profile, not a scalar (same reasoning as TriFoilAir.prepare).
        sim.set_inlet_turbulence(self.TU * np.asarray(self.BANDS).max() * np.ones(self.ny,
                                                                                  np.float32),
                                 length=self.TU_LEN * self.chord, span=self.turb_span(),
                                 seed=self.cfg.seed + 3, u_conv=self.TU_CONV)

    def forcing(self, sim, t):
        u = self.profile(t)
        sim.set_inlet(u, None)
        sim.set_farfield(u)

    def bodies(self, t):
        return [shapes.place(self.profs[k], self._chords[k], self.sx(self.CX[k]), self.sy(fy),
                             self.aoa(t))
                for k, fy in enumerate(self.LANES)]

    def legend(self):
        full = self.cfg.extras.get("vmax_mach", 1.80)
        marks = [(0.0, "0")]
        for lab, b in zip(("TOP", "MID", "LOW"), self.BANDS):
            marks.append((min(b * self.M_HIGH2, full) / full, f"{b * self.M_HIGH2:.2f} {lab}"))
        marks.append((1.0 / full, "1.00 SONIC"))
        # collapse duplicate ticks when every lane runs the same rate
        seen, out = set(), []
        for f, lab in marks:
            key = round(f, 3)
            if key not in seen:
                seen.add(key)
                out.append((f, lab))
        return dict(title="MACH", marks=out + [(1.0, f"{full:.2f}")])

    def readout(self, t):
        return [("M", " / ".join(f"{b * self.mach(t):.2f}" for b in self.BANDS)),
                ("AoA", f"{self.aoa(t):+.1f} deg")]


class MachRates(_MachTrio):
    """THREE FLOW RATES, one section, throttled to sonic - `tri_foil_rates_fast` on the gas solver.

    The bands are `tri_foil_rates_fast`'s exact 1.25 / 0.75 / 0.55, and on this solver they mean
    something they could not mean on the lattice: `u0` IS a Mach number here, so at the sonic hold
    the three lanes sit at **M 1.25 / 0.75 / 0.55** - genuinely SUPERSONIC, transonic and firmly
    subsonic, in one frame, with the same section at the same angle in each. That is the whole
    argument for the clip: the only variable is speed, and the three regimes it produces do not
    look alike.

    **The supersonic lane is also the WELL-POSED one, which is worth knowing.** `mach_sweep`
    measured its tunnel choking at exactly M 1 - the flow ahead of the model settles near M 0.78
    because a solid-wall tunnel cannot swallow a model at sonic conditions. Above M 1 that problem
    disappears: nothing can propagate upstream through a supersonic inflow, so the top lane's
    freestream really is at the Mach the throttle says, and it pays for it with a detached bow
    shock standing off the leading edge - which is the picture a transonic clip is for.
    """
    OVERRIDES = dict(_MachTrio.OVERRIDES,
                     extras=dict(cfl=0.50, blockage=3.20, vmax_mach=1.80))
    title = "THREE MACH NUMBERS"
    BANDS = (1.25, 0.75, 0.55)
    CODES = ("0012", "0012", "0012")


class MachFoils(_MachTrio):
    """ONE flow rate, THREE SECTIONS, throttled to sonic - `tri_foil_fast` on the gas solver.

    The complement of `mach_rates`: there the air differed and the body did not, here the air is
    one uniform stream and the bodies differ. My call: make all three foils different.

    **They differ in THICKNESS, and that is the choice the physics makes for you.** At subsonic
    speed thickness is a detail; approaching Mach 1 it is the dominant parameter, because the
    critical Mach number - the freestream at which the flow first goes sonic somewhere on the
    surface - falls as the section gets thicker. So a 8% / 12% / 18% stack, at one incidence in
    one stream, shows the thick section growing a supersonic pocket and a terminating shock while
    the thin one is still entirely subsonic, and then the shock marching aft on each in turn as
    the throttle climbs. Picking three sections that differed in camber or in family instead would
    have changed several things at once and shown nothing in particular - the same discipline
    `tri_foil_rates` applies to the air.
    """
    title = "THREE SECTIONS - ONE STREAM"
    BANDS = (1.0, 1.0, 1.0)
    CODES = ("0008", "0012", "0018")


class MachShapes(_MachTrio):
    """`tri_shapes` - foil / COW / foil - taken to Mach 1. Normal lighting.

    My call: make a copy of `tri_shapes` with the Mach choreography, and revert this copy's
    lighting to normal - the brightness adjustment in `tri_shapes` may not have read as intended.
    So `saturation` and `brightness` go back to 1.0 - the untouched palette - and
    that is the right call here for a reason beyond taste: `tri_shapes` needed dimming because
    every lane ran at one speed and `jet`'s freestream colour is its most luminous entry, so the
    frame was almost entirely that colour. This clip is a MACH field that starts deep blue and
    only reaches green at the sonic hold, so the problem the dimming solved does not arise.

    The cow is the point. It is a bluff body with a nearly flat face, so where the two sections
    grow a modest supersonic pocket, the cow gets a **detached bow shock** standing off it - the
    single most recognisable thing in compressible aerodynamics, in front of a dairy cow. Its
    blockage is what sets this scene's `blockage` factor, and it is the reason the chord here is
    the smallest in the family.
    """
    OVERRIDES = dict(_MachTrio.OVERRIDES, saturation=1.0, brightness=1.0,
                     extras=dict(cfl=0.50, blockage=3.00, vmax_mach=1.80))
    title = "THREE SHAPES AT MACH 1"
    BANDS = (1.0, 1.0, 1.0)
    CODES = ("0015", None, "2412")          # None = the cow
    COW = os.path.join(ROOT, "assets", "cow.png")
    COW_LEN = 0.86                          # of the chord, as in `tri_shapes`

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cow = shapes.image_body(need_asset(self.COW, "a cow"), cells=self.COW_LEN * self.chord,
                                     flip_x=True)
        self.profs = [shapes.flip_y(shapes.naca4(c)) if c else self.cow["points"]
                      for c in self.CODES]
        self._chords = [self.chord if c else self.COW_LEN * self.chord for c in self.CODES]

    def aoa_for(self, k, t):
        """The cow does not pitch - it is an animal in a tunnel, not a section in a rig."""
        return 0.0 if self.CODES[k] is None else self.aoa(t)

    def bodies(self, t):
        return [shapes.place(self.profs[k], self._chords[k], self.sx(self.CX[k]), self.sy(fy),
                             self.aoa_for(k, t))
                for k, fy in enumerate(self.LANES)]

    def body_textures(self):
        return [None if c else self.cow["rgb"] for c in self.CODES]


# ======================================================================================
# THE 2026-08-10 SET - four variations on `tri_foil_air_fast`, from my brief of that date.
# All four keep the parent's air: the fluid is not the variable in any of them.
# ======================================================================================
class TriFoilAirSweep2(TriFoilAirFast):
    """`tri_foil_air_fast` with the sweep run at DOUBLE rate - two full loops instead of one.

    My note, 2026-08-10: almost the exact same as `tri_foil_air_fast`, but with the range of
    motion twice as fast - completing an extra motion loop in the same video length.

    `PERIOD` 9.0 -> 4.5 is the ENTIRE change - one class attribute, no `OVERRIDES` at all, so
    `u0`, `re`, `csm`, `tau`, `steps`, `scale`, the palette, `vmax`, `gamma`, the inlet profile
    and the turbulence are the verified parent's, bit for bit. The clip stays 10 s and the foils
    now turn through 2.22 cycles against 1.11.

    WHY THIS IS CHEAP RATHER THAN RISKY, and it is the same argument every time in this project:
    the thing a lattice minds about a moving body is not the angle, it is how far the mask moves
    PER LBM STEP. The sweep is written in
    SECONDS, and `tri_foil_air_fast` doubled `steps` to 110 without touching it - which had
    already HALVED the per-step slew to 0.0056 cells/step, the slowest in the tri-foil family.
    Doubling the sweep rate spends exactly that back: **0.0112 cells/step, which is
    `tri_foil_air`'s own verified number**, against `tri_foil`'s 0.043 and `Flutter`'s ~0.07.
    So this is not new territory - it is the parent clip's slew at the parent clip's Reynolds
    number, and the total lattice time (66000 steps) is the parent's too.

    `RAMP` stays 1.8 s, deliberately, and it is the one thing a reviewer should look at. The
    envelope holds the angle AND its rate at zero at both ends so the clip still opens and closes
    on clean flow, but 1.8 s is now 40% of a cycle rather than 20%, so the FIRST peak reaches
    ~23 deg where the second reaches the full 34. That is visible. It is left alone because
    "almost the exact same" argues for the minimal diff, and because shortening the ramp is a
    one-number change if the attenuated opening is disliked.
    """
    duration = 10.0
    title = "THREE FLOW RATES - AIR, DOUBLE SWEEP"

    PERIOD = 4.5                   # 9.0 -> 4.5: two full sweeps in the same 10 s clip


class TriCylinders(TriFoilAirFast):
    """THREE CYLINDERS, THREE FLOW RATES - the canonical vortex-shedding experiment.

    My note, 2026-08-10: use the template, and replace the foils with unmoving circles -
    cylinders, to see vortex shedding.

    A circular cylinder in cross-flow is the most-measured object in fluid dynamics, and what it
    does is shed a Karman vortex street at a Strouhal number `St = f*D/U` that sits near **0.20**
    across four decades of Reynolds number. That constant is what makes this clip worth more than
    a picture: the three lanes carry the same cylinder at 1.25 / 0.75 / 0.55 of `u0`, so if the
    shedding is real the three lanes must shed at frequencies in that same ratio, and the number
    is checkable against a textbook rather than against this project. A Strouhal-measuring tool
    (not included here) measures it off the velocity field.

    NOTHING MOVES, and unlike the foils there is nothing to move: a cylinder has no incidence.
    Every feature in the frame is therefore the flow rate acting on one shape, which is the same
    single-variable discipline `tri_cows` runs.

    THE SIZE IS SET BY BLOCKAGE PARITY, not by taste. Free-slip side walls make blockage REAL,
    and the parent's three foils present 3 x chord x sin(34 deg) = ~21% of
    the cross-stream span at peak incidence. A cylinder blocks its own diameter at every instant,
    so `DIA = 0.56` of the chord puts three of them at the same ~21% - i.e. this clip asks no more
    of the lattice at any moment than the parent asks at its worst.

    `ref_len()` IS DELIBERATELY LEFT AS THE PARENT'S CHORD rather than moved to the diameter, and
    that is the point worth keeping. `nu = u0 * ref_len / re`, so re-basing `re` on a body 0.56x
    the size would have cut `nu` by the same factor and pushed tau from the verified 0.50123 down
    toward the 0.5 wall - a stability change nobody asked for, arriving through a definition. The
    fluid stays bit-identical to `tri_foil_air_fast`; the cylinder's own Reynolds number is
    REPORTED (Re_D = re * band * DIA), and at 11200 / 6720 / 4930 all three lanes sit in the
    sub-critical range where St ~ 0.2 is exactly what a textbook promises.

    SHEDDING HAS TO BE MEASURED, NOT ASSUMED, and this is the scene in the set most able to look
    fine and be wrong. A perfectly symmetric cylinder in perfectly symmetric flow sits in a stable
    twin-bubble wake essentially forever - a numerical artefact, and one that would post excellent
    stability numbers while showing nothing at all. `--perturb` is the
    usual answer and this clip does NOT use it: the parent turned it off in favour of continuous
    inlet turbulence, which is the better mechanism for exactly this (a one-shot kick decays, a
    real tunnel is never quiet - the air recipe, point 5). Whether that is enough to trip the
    instability, how long it takes, and what Strouhal number comes out are that check tool's
    three questions.
    """
    duration = 10.0
    title = "THREE FLOW RATES - CYLINDERS"

    DIA = 0.56                     # of the parent's chord: blockage parity with three foils at
                                   # 34 deg. See the docstring - this is a measured target, and
                                   # a check tool not included here re-measures it off the placed
                                   # polygons.

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prof = shapes.circle(192)
        self.dia = self.DIA * self.chord

    def aoa(self, k, t):
        """A circle has no incidence. Returning 0 rather than deleting the sweep keeps the
        parent's `readout`/`legend` machinery working unchanged."""
        return 0.0

    def bodies(self, t):
        # pivot=0.5 puts the circle's CENTRE on the station, where `place`'s default quarter-chord
        # pivot would offset it by a quarter of a diameter.
        return [shapes.place(self.prof, self.dia, self.sx(self.CX[k]), self.sy(fy), 0.0,
                             pivot=0.5)
                for k, fy in enumerate(self.LANES)]

    def readout(self, t):
        return [("Re_D", " / ".join(f"{b * self.cfg.re * self.DIA:.0f}" for b in self.BANDS))]


class TriFoilRatesLoop(TriFoilRatesFast):
    """`tri_foil_rates_fast` rebuilt as a SEAMLESS LOOP: frame 600 is frame 0, measurably.

    My note, 2026-07-30: shorten the airfoils' animation cycle slightly so it finishes moving
    around 9 seconds, and make sure the field returns to the same image as the starting frame by
    the end of the 10 seconds, so the video loops perfectly.

    A CFD clip does not loop by wanting to. Three separate things have to be true, and each one
    was a real obstacle:

    1. **The FIELD has to come home.** The flow has memory - every vortex the sweep sheds is still
       in the domain until it convects out of the visible window. So the foils must return to
       0 deg early enough that the tail flushes. That flush time is set by the SLOWEST lane, and
       it is the reason `MOTION_END` is not the ~9 s I asked for: at 0.55 u0 the last
       structures need ~2.5-3 s to clear the frame. A loop-timing tool (not included here)
       measured the return curve and `MOTION_END` is read off it - see the class constant.
    2. **Frame 0 has to BE the quiet state**, not a state still settling toward it. If the clip
       ends converged and starts half-converged they cannot match, however long the tail is.
       Hence `settle` is raised to cover a slow-lane transit, and `perturb` is 0: a symmetric
       foil at 0 deg sheds nothing, so the symmetry-breaking kick buys
       nothing here and would only inject a transient that has to flush back out.
    3. **The TRACERS have to loop too**, and by default they cannot - they respawn from an RNG
       stream. `cfg.loop` puts `Streaks` into its periodic mode (fixed home, fixed phase,
       lifetime an exact division of the period), which makes the ensemble a pure function of t.
       Without it the field loops perfectly and every white dash still jumps at the seam.
       Measured: with the period misaligned the seam sits at ~9 mean |dRGB|; aligned, ~1.

    4. **AND `re` HAD TO COME DOWN, 1600 -> 1100.** This is the one that was not foreseeable and
       it is the whole reason this scene exists separately rather than as a flag on
       `tri_foil_rates_fast`. At `re=1600` the fastest lane's wake is CONVECTIVELY UNSTABLE at
       0 deg: a straight wake grows a visible sinuous waviness over several seconds, never
       returns to straight, and never locks to the forcing period - so `field(T) = field(0)` is
       not merely hard, it is unavailable. Three separate experiments say so, and each ruled out
       an approach that looked promising:
         - hold the foils at 0 deg for the WHOLE clip and the residual still grows monotonically
           (0.19 -> 1.06 mean |dRGB|), so the quiet state is not a fixed point and no tail length
           can work;
         - scan the tail frame by frame for a phase-matched cut and there is no dip, only a slow
           decay - so ending on a matched wake phase is not available either (and could not be,
           with three lanes shedding at three different frequencies);
         - force periodically for 3 periods hoping for a T-periodic limit cycle: period 2 vs 1
           gives 1.66, period 3 vs 2 gives 1.43. Not converging. The wake is chaotic.
       Lowering `re` makes the straight wake linearly STABLE, so it is attracting and the flow
       genuinely returns to it. Measured at the seam, top lane's downstream third: mean |dRGB|
       **5.07 -> 0.65**, max 85 -> 21. Raising `csm` to 0.35 instead was tried and is not enough
       (5.07 -> 3.95) because it damps the sweep's vortices - the content - just as hard.
       The visual cost was checked side by side at the same instant and is very small: identical
       colours, identical suction peak, a marginally smoother separation bubble.

    The sweep itself is one full cycle over `MOTION_END` - up to +PEAK, through 0, down to -PEAK,
    back to 0 - and then exactly zero for the rest of the clip. The `RAMP` envelope is what makes
    the seam clean at the START as well: without it `sin` has its maximum slope at t=0, so the
    foils would be stationary at t=T and instantly at full angular rate at t=0. With it both the
    angle AND its rate are zero at each end, so nothing jerks across the join.

    Verification is not an argument, it is `tools\\loop_check.py`: it renders one frame PAST the
    end and diffs the fully composed RGB against frame 0 - field, tracers, bodies and legend, the
    actual thing a viewer sees at the seam.
    """
    OVERRIDES = dict(TriFoilRatesFast.OVERRIDES, loop=True, perturb=0.0, settle=4.0, re=1100.0)
    duration = 10.0
    title = "THREE FLOW RATES - LOOP"

    MOTION_END = 5.0               # seconds; the foils are at exactly 0 deg from here to the end.
                                   # MEASURED, not chosen: it is 10 s minus the SLOW lane's flush
                                   # time. At 0.55 u0 the last shed vortex needs ~5 s to carry
                                   # from the trailing edge out of the right-hand edge of frame,
                                   # and until it has, frame 600 cannot equal frame 0. The
                                   # measured residual against the tail length is
                                   # 3 s -> 1.28, 4 s -> 0.80, 5 s -> 0.36 mean |dRGB|.
                                   # This is why "around 9 s" was not reachable.
    RAMP = 1.2                     # short enough that the +-PEAK at t = MOTION_END/4 and 3/4
                                   # still reach full amplitude with the envelope applied

    def envelope(self, t):
        """Zero angle AND zero angular rate at both t=0 and t=MOTION_END."""
        e = self.MOTION_END
        return (smoothstep(0.0, self.RAMP, t)
                * (1.0 - smoothstep(e - self.RAMP, e, t)))

    def aoa(self, k, t):
        if t >= self.MOTION_END:
            return 0.0
        return self.PEAK * self.envelope(t) * float(
            np.sin(2 * np.pi * t / self.MOTION_END))


class Flutter(Scene):
    """A flat plate pitching sinusoidally: dynamic stall, with the wake reversing every cycle."""
    OVERRIDES = dict(re=5000.0, u0=0.10, steps=14)
    duration = 16.0
    title = "PITCHING PLATE"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prof = shapes.plate(0.05)
        self.chord = 0.42 * self.wspan
        self.cx, self.cy = 0.34 * self.nx, 0.50 * self.ny
        self.period = 4.0

    def ref_len(self):
        return self.chord

    def aoa(self, t):
        return 32.0 * np.sin(2 * np.pi * t / self.period) * smoothstep(0.0, 2.0, t)

    def bodies(self, t):
        return [shapes.place(self.prof, self.chord, self.cx, self.cy, self.aoa(t), pivot=0.35)]

    def readout(self, t):
        return [("AoA", f"{self.aoa(t):+.1f} deg")]


class VortexStreet(Scene):
    """A circular cylinder at a Reynolds number where the wake goes unsteady: the alternating
    shed vortices of a von Karman street, the textbook validation case for any 2-D solver."""
    OVERRIDES = dict(re=800.0, u0=0.10, steps=16)
    duration = 16.0
    title = "VON KARMAN STREET"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.d = 0.17 * self.wspan
        self.cx, self.cy = 0.26 * self.nx, 0.50 * self.ny
        self.prof = shapes.circle()

    def ref_len(self):
        return self.d

    def bodies(self, t):
        return [shapes.place(self.prof, self.d, self.cx, self.cy, 0.0, pivot=0.5)]


class Tandem(VortexStreet):
    """Three staggered cylinders - the downstream pair sit in the upstream wake and lock on."""
    OVERRIDES = dict(re=900.0, u0=0.10, steps=16)
    duration = 16.0
    title = "TANDEM CYLINDERS"

    def bodies(self, t):
        d = 0.12 * self.wspan
        out = []
        for fx, fy in ((0.20, 0.50), (0.42, 0.31), (0.42, 0.69)):
            out.append(shapes.place(self.prof, d, fx * self.nx, fy * self.ny, 0.0, pivot=0.5))
        return out


class ShapeTour(Scene):
    """Swap the body every few seconds - streamlined to bluff - so the wake tells you the drag.

    The teardrop closes its wake almost completely; the flat plate normal to the flow drags a
    wake wider than itself. The swap is deliberately instantaneous: the starting-vortex
    transient that follows each change is a real, and good-looking, piece of physics.
    """
    OVERRIDES = dict(re=2500.0, u0=0.10, steps=15)
    duration = 20.0
    title = "DRAG BY SHAPE"

    ORDER = [("teardrop", 0.0, 0.56), ("ellipse", 0.0, 0.46), ("circle", 0.0, 0.20),
             ("square", 0.0, 0.20), ("plate", 90.0, 0.26)]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cx, self.cy = 0.30 * self.nx, 0.50 * self.ny
        self.slot = self.duration / len(self.ORDER)

    def ref_len(self):
        return 0.20 * self.wspan

    def _current(self, t):
        return self.ORDER[min(int(t / self.slot), len(self.ORDER) - 1)]

    def bodies(self, t):
        name, aoa, frac = self._current(t)
        prof = shapes.PROFILES[name]()
        return [shapes.place(prof, frac * self.wspan, self.cx, self.cy, aoa, pivot=0.5)]

    def readout(self, t):
        return [("BODY", self._current(t)[0].upper())]


# =======================================================================================
# DYNAMIC scenes - the bodies are moved by the fluid, not by the scene.
# =======================================================================================
class _FreeScene(Scene):
    """Shared plumbing: a BodySystem, and the per-LBM-step hook the renderer drives."""
    dynamic = True
    g = 5e-4                      # lattice gravity; see bodies.density_for_fall
    FLOOR = 0.13                  # bodies rest this far above the inlet, as a fraction of nx.
                                  # Too low and a settled body throttles the inlet region, which
                                  # spikes the local speed and takes the lattice with it.
    CEIL = 0.985

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.sys = BodySystem(self.nx, self.ny, g=self.g,
                              floor=self.FLOOR, ceiling=self.CEIL)
        self.populate()

    def populate(self):
        raise NotImplementedError

    def ref_len(self):
        return 0.18 * self.wspan

    def substep(self, lbm, t, dt=1.0, freeze=False):
        # gravity points along lattice -x, i.e. DOWN the screen (streamwise +x is up)
        return self.sys.step(lbm, t, dt, g_dir=(-1.0, 0.0), freeze=freeze)

    def bodies(self, t):
        return self.sys.polygons(t)


class FallingDiscs(_FreeScene):
    """Discs dropped into a rising stream: they sink slowly, wobble, and shove each other's wakes.

    Each disc's DENSITY is solved from the terminal-velocity balance so that it takes about the
    length of the clip to reach the floor. That is choosing what the discs are made of - after
    release nothing is prescribed, so they drift sideways when a neighbour's shed vortex arrives
    and visibly slow as they enter each other's slipstreams.
    """
    # Re is deliberately modest here. With free bodies the lattice has to stay well-behaved while
    # the boundary MOVES, and tau near 0.5 (Re in the thousands) leaves no margin for that; at
    # Re ~700 tau sits around 0.52 and the shedding off a disc is still perfectly legible.
    OVERRIDES = dict(re=550.0, u0=0.075, csm=0.18, steps=14, settle=1.0)
    duration = 15.0
    title = "FALLING DISCS"
    PROFILE = "circle"
    SIZES = (0.135, 0.115, 0.098, 0.125)
    # Size the density against the speed the bodies ACTUALLY see, not the inlet speed. Several
    # bodies in a narrow tunnel block a large fraction of it, so the stream between and around
    # them runs well above u0 (measured ~2.5x with this set), and drag goes as u^2 - designing
    # against u0 leaves them ~6x too light and the stream simply carries them up the screen.
    # Calibrated by bisection against the real scene, not derived: what a disc actually feels is
    # neither the inlet speed nor the domain peak, and the terminal speed is sensitive to it
    # (drag ~ u^2). Too low and the stream carries the discs UP the screen; too high and they
    # sink fast enough that the moving boundary itself destabilises the lattice.
    U_REF = 1.10

    def populate(self):
        prof = shapes.PROFILES[self.PROFILE]()
        ua = polygon_area(prof)
        cfg = self.cfg
        # target: cross ~76% of the tunnel height in one clip
        span = 0.76 * self.nx
        v_fall = span / (self.duration * 60.0 * cfg.steps)
        # Release in SEPARATE LANES, staggered in height. Dropped in a tight cluster they line up
        # into a wall across the tunnel: all the mass flux is forced through the gaps, the local
        # speed hits 3-4x u0, and since drag goes as u^2 the whole group gets blown UP the screen
        # instead of falling (and the lattice goes unstable doing it). Spread out, each disc sees
        # something close to the free stream and still interacts when a wake drifts across.
        lanes = [-0.20, -0.07, 0.07, 0.20]
        self.rng.shuffle(lanes)
        for k, frac in enumerate(self.SIZES):
            d = frac * self.wspan
            v = v_fall * float(self.rng.uniform(0.82, 1.18))
            self.sys.add(FreeBody(
                prof, d,
                x=(0.84 - 0.055 * k) * self.nx,
                y=(0.5 + lanes[k]) * self.ny,
                name=self.PROFILE,
                ang=float(self.rng.uniform(0, np.pi)),
                density=density_for_fall(self.PROFILE, d, v, cfg.u0 * self.U_REF, self.g, ua),
                mode="dynamic"))


class FallingSquares(FallingDiscs):
    """The same drop with square plates - which, unlike discs, feel a torque and tumble.

    A square's centre of pressure wanders off its centroid as it turns, so the fluid keeps
    handing it angular momentum; the flutter-and-tumble is the torque term of the same
    momentum-exchange sum, not an animation.
    """
    # A square is bluffer than a disc and its corners sweep, so it crowds the tunnel harder:
    # lower Re for margin, a higher resting floor, and a slower fall so the clip ends about when
    # they arrive rather than leaving them parked on the inlet throttling it for five seconds.
    OVERRIDES = dict(re=450.0, u0=0.075, csm=0.18, steps=14, settle=1.0)
    duration = 15.0
    title = "FALLING PLATES"
    PROFILE = "square"
    SIZES = (0.125, 0.098, 0.112, 0.085)
    U_REF = 1.05
    FLOOR = 0.14


class Projectiles(_FreeScene):
    """Dense pellets fired DOWN into a fast rising stream, at angles.

    A ballistic entry decelerated by real drag: each pellet punches in, sheds a vortex pair,
    slows, and - if the stream wins - stalls and is carried back up. Nothing decides which;
    it depends on the mass it was fired with and what the flow is doing when it arrives.
    """
    # "Fast" is bought with STEPS, not with lattice velocity. Everything on screen moves at
    # (lattice speed x steps-per-frame), so doubling the steps doubles the apparent speed of both
    # the stream and the pellets while leaving every velocity small in lattice units - which is
    # what the moving-wall boundary needs. Launching at 0.2 lattice units instead (the obvious
    # reading of "high velocity") drives the boundary populations negative and kills the solve.
    OVERRIDES = dict(re=600.0, u0=0.10, csm=0.18, steps=30, settle=0.6)
    duration = 15.0
    title = "PROJECTILES"
    g = 6e-4

    def populate(self):
        prof = shapes.PROFILES["circle"]()
        n = 9
        for k in range(n):
            d = float(self.rng.uniform(0.050, 0.078)) * self.wspan
            ang = np.deg2rad(float(self.rng.uniform(-35, 35)))     # off straight-down
            spd = float(self.rng.uniform(0.050, 0.072))
            self.sys.add(FreeBody(
                prof, d,
                x=0.93 * self.nx,
                y=(0.5 + float(self.rng.uniform(-0.26, 0.26))) * self.ny,
                name="circle",
                vx=-spd * np.cos(ang), vy=spd * np.sin(ang),
                density=float(self.rng.uniform(3.2, 6.5)),
                mode="dynamic", on_exit="kill",
                born=0.3 + k * (self.duration - 2.5) / n))


class Eddies(_FreeScene):
    """Currents instead of a uniform stream - and light shapes set adrift in them.

    The inlet is a slowly changing set of jets (their strengths, positions and cross-flow all
    drift on incommensurate periods, so the pattern never repeats within the clip). Neighbouring
    jets shear against each other and roll up into eddies with no obstacle involved; two small
    fixed vanes seed a few more. The drifting shapes are NEUTRALLY BUOYANT and therefore ride
    the flow (tracer coupling - see bodies.py on why that is the stable and correct model at a
    density ratio of 1), so they trace out the current pattern as it changes.
    """
    # u0 is deliberately SMALL and `steps` large. The drifters ride at the local current speed,
    # and the moving-wall term 6 w_k (c_k.u_w) exceeds the diagonal populations themselves once
    # that speed passes ~0.12 lattice units - so a fast current cannot carry a solid body without
    # wrecking the boundary. Slow lattice, many steps per frame: identical motion on screen, all
    # velocities safely small. (Same trick as `Projectiles`.)
    # `steps` is set so the stream takes most of the clip to cross the frame ONCE: the drifters
    # then ride the whole way up without needing to be respawned (which cannot be done safely
    # here - see `populate`), and a slow mean flow lets you actually watch a shape work its way
    # around an eddy. The scene stays lively regardless, because the motion comes from the inlet
    # pattern changing on its own 4-7 s timescales, not from the convection speed.
    OVERRIDES = dict(re=600.0, u0=0.055, csm=0.18, steps=16, settle=1.4, perturb=0.03,
                     wall_clamp=0.11, overscan=0.12)
    duration = 15.0
    title = "CURRENTS"
    NJET = 3
    JET_MAX = 1.70          # cap the profile at this multiple of u0 (see `inlet`)
    CROSS = 0.08            # wandering transverse component, as a fraction of u0 - what stops
                            # the mean stream from running dead straight up the frame
    FLOOR = 0.05            # drifters are never meant to rest; keep the floor out of their way

    def populate(self):
        self._yc = np.arange(self.ny, dtype=np.float64)
        # Every drifter is ALREADY IN the tunnel at t=0, spread up its length, so they are all
        # present during the frozen settle and the flow establishes around them. Two other
        # arrangements were tried and both diverge within a second:
        #   - born mid-clip: a solid materialising in moving fluid annihilates the momentum in
        #     its volume, and the pressure pulse takes the lattice out;
        #   - drifting IN from below frame: entering bodies straddle the INLET plane, and
        #     blocking part of a hard velocity boundary spikes the pressure right against it.
        # They ride up, leave over the (soft, zero-gradient) outlet, and are gone; the stagger
        # is what keeps shapes on screen for the length of the clip.
        # Each shape is ballasted to HOVER: its density is solved so that weight exactly balances
        # the drag of a u0 stream. It therefore rises where a jet runs fast, sinks where the flow
        # is slack, and gets carried sideways by whatever eddy reaches it - so the shapes stay in
        # frame for the whole clip and read the current pattern for you. Simply letting them ride
        # the flow instead flushes every one of them out of the top within about ten seconds.
        names = ["teardrop", "square", "circle", "ellipse", "wedge",
                 "circle", "teardrop", "square", "ellipse", "wedge"]
        # ten shapes, staggered up the VISIBLE frame in two interleaved columns of five, with a
        # wide size spread (0.045 - 0.145 of the frame width, i.e. ~3x smallest to largest)
        for k, nm in enumerate(names):
            prof = shapes.PROFILES[nm]()
            size = float(self.rng.uniform(0.045, 0.145)) * self.vspan
            self.sys.add(FreeBody(
                prof, size,
                x=self.sx(0.06 + 0.093 * k),
                y=self.sy(0.5 + float(self.rng.uniform(-0.33, 0.33))),
                name=nm, ang=float(self.rng.uniform(0, np.pi)),
                # Balanced to HOVER exactly (target fall speed 0). Ballasting them heavier to
                # keep more of them in frame for longer was tried and DIVERGES: the shapes then
                # sink into the oncoming jets, and the head-on closing speed is more than the
                # moving boundary can carry. Hover is the verified setting - leave it alone.
                density=density_for_fall(nm, size, 0.0, self.cfg.u0, self.g,
                                         polygon_area(prof)),
                mode="dynamic", on_exit="kill"))

    def inlet(self, t):
        """Shaped, slowly evolving inlet: NJET gaussian jets + a wandering cross-flow.

        Renormalised to mean u0 so the tunnel's mass flow stays constant - otherwise the whole
        field would breathe brighter and darker as the jets came and went.
        """
        u0, ny = self.cfg.u0, self.ny
        prof = np.full(ny, 1.0, dtype=np.float64)
        for k in range(self.NJET):
            ph = 2 * np.pi * (t / (7.0 + 2.3 * k) + k / self.NJET)
            centre = (0.5 + 0.30 * np.sin(ph)) * ny
            width = (0.085 + 0.030 * np.sin(2 * np.pi * t / (5.0 + 1.7 * k))) * ny
            amp = 0.42 + 0.26 * np.sin(2 * np.pi * t / (4.0 + 1.3 * k) + k)
            prof += amp * np.exp(-0.5 * ((self._yc - centre) / width) ** 2)
        prof *= u0 / prof.mean()
        # Cap the peaks. Jets several times u0 shear hard enough to break the lattice within a
        # fraction of a second; this still leaves plenty of contrast to roll up into eddies.
        prof = np.clip(prof, 0.35 * u0, self.JET_MAX * u0)
        cross = self.CROSS * u0 * np.sin(2 * np.pi * t / 6.5 + 1.4 * self._yc / ny)
        return prof, cross

    def substep(self, lbm, t, dt=1.0, freeze=False):
        lbm.set_inlet(*self.inlet(t))
        # No fixed vanes: I cut them 2026-07-26. The eddies come from the jets shearing
        # against each other, so nothing anchored is needed to seed them.
        return self.sys.step(lbm, t, dt, g_dir=(-1.0, 0.0), freeze=freeze)   # ballasted to hover

    def bodies(self, t):
        return self.sys.polygons(t)


class Currents(Eddies):
    """Complex, curling, non-straight currents in BLUE - with small spheres sinking through them.

    The current pattern is `eddies`' mechanism: the inlet is a set of gaussian jets whose
    strengths, positions and widths all drift on incommensurate periods, so neighbouring jets
    shear against each other and roll up with no obstacle involved and nothing keyframed. Two
    changes make the flow read as CURRENTS rather than as a stream: a fourth jet, and twice the
    wandering transverse component, which makes the whole stream meander instead of running
    straight up the frame. Everything the water then does is the solver's.

    What is new here is what is IN the water. `eddies` ballasts its drifters to hover exactly, so
    they read the pattern and stay. These are five small spheres, each given the DENSITY that
    settles it at a chosen fall speed against the stream (`density_for_fall` - choosing what they
    are made of, not their path), released off the top of the frame at t=0. So they sink from the
    top of the stream to the bottom over the clip - but not in a straight line and not steadily:
    a sphere that drifts into a fast jet is slowed or briefly pushed back up, one that finds a
    slack patch drops away, and an eddy arriving side-on walks it across the frame. That is the
    demonstration - the descent is a *probe* of the current pattern, drawn by the flow.

    They are deliberately SMALL (19-28 cells across, ~5-8% of the frame width). `eddies` records
    a divergence from ballasting its ten drifters - some of them 0.145 of the frame - heavy
    enough to sink into the oncoming jets; a fifth of the blockage and half the fall speed is
    what keeps this the same scene without that failure.
    """
    # blues, not jet: the palette from `still_water` / `black_holes`. Its bottom stop is a deep
    # ink blue rather than black, so slack water still reads as water instead of as background.
    # vmax is lower than the tunnel default because the whole field is moving here - the mean
    # stream should land in the middle of the ramp so the jets have somewhere bright to go.
    OVERRIDES = dict(re=600.0, u0=0.055, csm=0.18, steps=16, settle=1.4, perturb=0.03,
                     wall_clamp=0.11, overscan=0.12,
                     cmap="blues", vmax=1.90, gamma=0.92,
                     n_streaks=30000, streak_span=8.0, streak_gain=0.44,
                     streak_speed_gain=0.42, streak_life=(0.8, 2.0),
                     body_fill=(9, 13, 24), body_edge=(226, 238, 246))
    duration = 15.0
    title = "CURRENTS"
    NJET = 4                # one more jet than `eddies` -> a busier, less repetitive pattern
    JET_MAX = 1.70          # leave this alone: jets much past ~2x u0 shear the lattice apart
    CROSS = 0.16            # twice `eddies`: the stream visibly snakes instead of running straight
    FLOOR = 0.13            # they are MEANT to arrive - rest them just inside the bottom of frame

    # five spheres: diameters as a fraction of the visible frame WIDTH, and where each starts.
    # x > 1.0 is ABOVE the top of the visible frame, in the overscan margin - a body may be
    # present-and-frozen through the settle or enter from off-lattice, never born mid-clip in
    # moving fluid. These do both: present at t=0, most of them off camera.
    SIZES = (0.052, 0.062, 0.070, 0.058, 0.078)
    POS = ((0.84, 0.22), (0.93, 0.46), (1.02, 0.68), (0.88, 0.34), (0.98, 0.82))
    # A sphere sees rather more than the inlet speed (the jets peak at 1.70 u0, and a sphere is
    # slowed most exactly where the flow is fastest, so it spends its time in the fast parts),
    # and drag goes as u^2, so the density is sized against u0 * U_REF. Same correction as
    # `falling_discs`, and like that one it is MEASURED, not derived: at 1.12 the spheres crossed
    # only half the frame in the clip. `dbg.py` prints x/nx per body per second, so one run gives
    # the real fall speed (27.5 cells/s against a designed 43.6); inverting the terminal balance
    # (u_seen + v)^2 ~ (rho - 1) on that puts what they actually see at 1.43 u0.
    U_REF = 1.45
    FALL = 0.92             # fraction of the visible height to cross ...
    FALL_T = 0.90           # ... in this fraction of the clip

    def populate(self):
        self._yc = np.arange(self.ny, dtype=np.float64)
        prof = shapes.PROFILES["circle"]()
        ua = polygon_area(prof)
        cfg = self.cfg
        v_fall = (self.FALL * self.vnx) / (self.FALL_T * self.duration * 60.0 * cfg.steps)
        for k, (frac, (fx, fy)) in enumerate(zip(self.SIZES, self.POS)):
            d = frac * self.vspan
            self.sys.add(FreeBody(
                prof, d, x=self.sx(fx), y=self.sy(fy), name="circle",
                ang=float(self.rng.uniform(0, np.pi)),
                density=density_for_fall("circle", d, v_fall, cfg.u0 * self.U_REF, self.g, ua),
                mode="dynamic"))

    def readout(self, t):
        return [("SPHERES", str(len(self.sys.active(t))))]


class _Patch:
    """One localised forcing event: a target velocity field, a weight mask, a time envelope.

    The mask is what makes the pool SEMI-still: `StillWater` hands the solver a per-cell beta,
    so outside the active patches the fluid is driven at essentially nothing and just carries on
    doing whatever it was already doing. A patch only ever injects momentum into its own little
    region - what that turns into (a rolled-up eddy, a shear braid, a wake off a sphere) is the
    solver's answer, exactly as everywhere else in this project.
    """

    def __init__(self, t0, rise, hold, fall, ux, uy, mask):
        self.t0 = float(t0)
        self.rise, self.hold, self.fall = float(rise), float(hold), float(fall)
        self.t1 = self.t0 + self.rise + self.hold + self.fall
        m = mask.astype(np.float32)
        # pre-multiplied: the per-frame combine is then two multiply-adds per patch, not four
        self.mask = xp.asarray(m)
        self.mux = xp.asarray((ux * m).astype(np.float32))
        self.muy = xp.asarray((uy * m).astype(np.float32))

    def amp(self, t):
        if t <= self.t0 or t >= self.t1:
            return 0.0
        return float(smoothstep(self.t0, self.t0 + self.rise, t)
                     * (1.0 - smoothstep(self.t1 - self.fall, self.t1, t)))


class StillWater(_FreeScene):
    """Five spheres adrift on water that is nearly still - until something disturbs it.

    The variation on the wind tunnel: there is NO tunnel. The domain is a sealed box (`closed`),
    the lattice is seeded at rest rather than at a freestream, and nothing drives the fluid
    globally - so the default state of the picture is calm, deep blue, and almost motionless.
    What happens instead is EVENTS: a swirl started somewhere in the middle, a counter-rotating
    pair that twists the water between them and walks itself across the frame, an outside stream
    pushing in from off camera and ploughing through whatever is in its way. Each one ramps up,
    holds for a second or two, and lets go; what it leaves behind decays on its own.

    Why it can be still at all. Two mechanisms, both of them solver-side rather than cosmetic:

    * A SPATIALLY VARYING drive (`LBM.set_drive` with a beta FIELD). Each event owns a mask, and
      beta is essentially zero outside every mask, so undisturbed water is genuinely undriven -
      not a slow uniform flow dressed up as calm.
    * A small friction floor (`DAMP`, target velocity zero). Without it the box would never
      settle: at these Reynolds numbers viscosity alone needs ~1e6 steps to kill a 50-cell eddy,
      i.e. hundreds of clips, so every event would accumulate and the pool would end as soup.
      A linear drag is also the physically right term for a shallow layer over a bed.

    The spheres are neutrally-ish buoyant beads in a PLAN view - we are looking down at the
    water, so gravity is out of the plane and `g = 0`. They are dense enough to be integrated as
    proper Newton-Euler bodies (a density ratio near 1 is where explicit coupling goes unstable)
    but light enough that drag governs them, so they ride a passing current, lag it slightly, and
    coast to a stop when it passes - which is what a floating ball does.

    Sizes are to brief: four spheres within 25% of each other end to end (0.89 - 1.11 of BASE,
    a ratio of 1.25) and a fifth 50% larger, which is big enough to shed a visible wake of its
    own when a stream hits it while the others mostly get pushed around.
    """
    # Everything on screen moves at (lattice speed x steps/frame), and lattice speeds have to
    # stay small for the moving-boundary term, so the currents are slow
    # in lattice units and `steps` is large. `re` is modest for free-body margin; the eddies read
    # as big soft rolls rather than fragmenting, which is what water at this scale looks like.
    OVERRIDES = dict(re=400.0, u0=0.045, csm=0.18, steps=26, settle=0.6, perturb=0.0,
                     closed=True, overscan=0.08, wall_clamp=0.07,
                     cmap="blues", vmax=1.25, gamma=0.85,
                     n_streaks=26000, streak_span=9.0, streak_gain=0.42,
                     # near 1.0, so a tracer sitting in still water is drawn at almost zero
                     # brightness and the dashes appear only where the water is actually moving
                     streak_speed_gain=0.98, streak_life=(0.9, 2.4),
                     body_fill=(9, 13, 24), body_edge=(226, 238, 246))
    duration = 16.0
    title = "STILL WATER"
    g = 0.0                       # plan view: gravity is out of the plane

    # --- the spheres ---
    BASE = 0.090                  # diameter as a fraction of the visible frame WIDTH
    SPREAD = (0.89, 0.96, 1.04, 1.11)     # 1.11/0.89 = 1.25 exactly: the "no more than 25%" four
    BIG = 1.50                    # the one exception, 50% larger
    BIG_AT = 2                    # where it sits in the placement list
    POS = ((0.30, 0.32), (0.62, 0.22), (0.44, 0.55), (0.72, 0.72), (0.26, 0.78))
    RHO = (3.4, 4.2, 5.0, 3.8, 4.6)
    INSET = 0.035                 # keep the bodies this far inside the VISIBLE frame

    # --- the disturbances ---
    DAMP = 2.0e-4                 # friction floor: ~1/DAMP steps to still, i.e. a few seconds
    BETA_E = 5.0e-3               # forcing rate inside an active patch (~0.16 s to establish)
    GAP = (2.1, 3.8)              # seconds between events. Wide enough that the envelopes only
                                  # overlap in their tails, so the pool visibly comes back to
                                  # calm between disturbances instead of never settling.

    def __init__(self, cfg):
        Scene.__init__(self, cfg)
        self.rng = np.random.default_rng(cfg.seed)
        ins = self.INSET
        self.sys = BodySystem(
            self.nx, self.ny, g=0.0,
            floor=(self.vx0 + ins * self.vnx) / self.nx,
            ceiling=(self.vx0 + (1.0 - ins) * self.vnx) / self.nx,
            ybounds=(self.vy0 + ins * self.vny, self.vy0 + (1.0 - ins) * self.vny))
        self._jj, self._ii = np.meshgrid(np.arange(self.ny, dtype=np.float32),
                                         np.arange(self.nx, dtype=np.float32), indexing="ij")
        self.patches = []
        self._build_events()
        self._key = None
        self._cache = None
        self.populate()

    def sizes(self):
        fr = [self.BASE * s for s in self.SPREAD]
        fr.insert(self.BIG_AT, self.BASE * self.BIG)
        return [f * self.vspan for f in fr]

    def ref_len(self):
        return self.BASE * self.BIG * self.vspan

    def populate(self):
        prof = shapes.PROFILES["circle"]()
        for k, (d, (fx, fy)) in enumerate(zip(self.sizes(), self.POS)):
            self.sys.add(FreeBody(
                prof, d, x=self.sx(fx), y=self.sy(fy), name="circle",
                ang=float(self.rng.uniform(0, np.pi)),
                # Dense enough for `dynamic` coupling to be stable (the added mass of the fluid a
                # body accelerates must stay well under its own), light enough that drag - not
                # inertia - decides where it ends up, so a sphere stops when the current does.
                density=self.RHO[k], mode="dynamic"))

    # -- the pool starts at rest ---------------------------------------------------------
    def prepare(self, lbm):
        """Seed the lattice STILL. `LBM.__init__` fills it with a uniform u0 freestream, which
        for a sealed box would be a slab of water sliding bodily sideways for the whole clip."""
        z = np.zeros((self.ny, self.nx), np.float32)
        lbm.init_velocity(z, z)

    # -- events ---------------------------------------------------------------------------
    def _vortex(self, cx, cy, R, V, sign):
        """A single swirl. Pure tangential flow, so it is divergence-free by construction and
        the lattice does not ring with acoustic waves while it establishes."""
        dx, dy = self._ii - cx, self._jj - cy
        r = np.hypot(dx, dy)
        rs = np.maximum(r, 1e-6)
        g = (r / R) * np.exp(0.5 * (1.0 - (r / R) ** 2))     # solid-body core, peak at r=R
        ux = -sign * V * g * (dy / rs)
        uy = sign * V * g * (dx / rs)
        return ux, uy, np.exp(-0.5 * (r / (1.5 * R)) ** 2)

    def _stream(self, p0x, p0y, dx, dy, V, width, reach):
        """An outside stream pushing in from off camera: a band of forcing at the lattice edge
        that fades out over `reach` cells. It is only pushed near where it enters - past that the
        jet carries itself, curls over and does whatever the water it hits makes it do."""
        sx_, sy_ = self._ii - p0x, self._jj - p0y
        s = sx_ * dx + sy_ * dy                  # along the stream
        q = -sx_ * dy + sy_ * dx                 # across it
        m = (np.exp(-0.5 * (q / width) ** 2) * np.exp(-np.maximum(s, 0.0) / reach)
             * (s > -3.0))
        return np.full_like(m, V * dx), np.full_like(m, V * dy), m

    def _add(self, *args):
        self.patches.append(_Patch(*args))

    def _build_events(self):
        """Schedule the disturbances. Their times, places, sizes and senses are drawn once from
        the seeded RNG - what each one DOES is not scripted at all."""
        rng = self.rng
        t = 0.5
        while t < self.duration - 2.2:
            kind = rng.choice(("vortex", "stream", "twist"), p=(0.40, 0.36, 0.24))
            if kind == "vortex":
                R = float(rng.uniform(0.10, 0.17)) * self.vny
                self._add(t, 0.9, float(rng.uniform(0.8, 2.0)), 1.7,
                          *self._vortex(self.sx(float(rng.uniform(0.20, 0.80))),
                                        self.sy(float(rng.uniform(0.22, 0.78))),
                                        R, float(rng.uniform(0.030, 0.052)),
                                        float(rng.choice((-1.0, 1.0)))))
            elif kind == "twist":
                # A counter-rotating PAIR. The two swirls shear the water between them into an S
                # and the dipole propels itself along its own axis - a twist that travels.
                R = float(rng.uniform(0.085, 0.13)) * self.vny
                V = float(rng.uniform(0.030, 0.048))
                cx = self.sx(float(rng.uniform(0.25, 0.75)))
                cy = self.sy(float(rng.uniform(0.28, 0.72)))
                th = float(rng.uniform(0.0, np.pi))
                ox, oy = 1.15 * R * np.cos(th), 1.15 * R * np.sin(th)
                hold = float(rng.uniform(0.7, 1.6))
                self._add(t, 0.8, hold, 1.5, *self._vortex(cx + ox, cy + oy, R, V, +1.0))
                self._add(t, 0.8, hold, 1.5, *self._vortex(cx - ox, cy - oy, R, V, -1.0))
            else:
                edge = int(rng.integers(0, 4))
                tilt = float(rng.uniform(-0.42, 0.42))
                if edge < 2:                       # in from the bottom / top of frame
                    sgn = 1.0 if edge == 0 else -1.0
                    p0x = 0.0 if edge == 0 else float(self.nx - 1)
                    p0y = self.sy(float(rng.uniform(0.15, 0.85)))
                    dx, dy = sgn * np.cos(tilt), np.sin(tilt)
                    reach = float(rng.uniform(0.22, 0.45)) * self.vnx
                else:                              # in from the left / right edge
                    sgn = 1.0 if edge == 2 else -1.0
                    p0x = self.sx(float(rng.uniform(0.15, 0.85)))
                    p0y = 0.0 if edge == 2 else float(self.ny - 1)
                    dx, dy = np.sin(tilt), sgn * np.cos(tilt)
                    reach = float(rng.uniform(0.40, 0.85)) * self.vny
                self._add(t, 0.7, float(rng.uniform(1.0, 2.2)), 1.4,
                          *self._stream(p0x, p0y, dx, dy,
                                        float(rng.uniform(0.034, 0.056)),
                                        float(rng.uniform(0.075, 0.155)) * self.vny, reach))
            t += float(rng.uniform(*self.GAP))

    def drive(self, t):
        """Combine the live patches into ONE (target field, per-cell beta) pair.

        The weights compose as an average, not a sum: with W = DAMP + sum(a_e * BETA_E * m_e) and
        U = sum(a_e * BETA_E * m_e * v_e) / W, the force W*(U - u) is exactly
        sum(a_e * BETA_E * m_e * (v_e - u)) - DAMP*u. So each event pushes toward its own velocity
        with its own weight, and the friction floor acts only where nothing else is asking.

        Rebuilt at 1/120 s of scene time (every ~13 LBM steps) rather than every step: the
        envelopes vary on ~1 s timescales, so the quantisation is invisible, and this is inside
        the per-step loop where an extra pass over the lattice would be felt.
        """
        key = int(t * 120.0)
        if key != self._key:
            wsum = xp.full((self.ny, self.nx), self.DAMP, dtype=xp.float32)
            tx = xp.zeros((self.ny, self.nx), dtype=xp.float32)
            ty = xp.zeros((self.ny, self.nx), dtype=xp.float32)
            for p in self.patches:
                a = p.amp(t)
                if a <= 1e-4:
                    continue
                c = np.float32(a * self.BETA_E)
                wsum = wsum + c * p.mask
                tx = tx + c * p.mux
                ty = ty + c * p.muy
            inv = 1.0 / wsum
            self._cache = (tx * inv, ty * inv, wsum)
            self._key = key
        return self._cache

    def substep(self, lbm, t, dt=1.0, freeze=False):
        lbm.set_drive(*self.drive(t))
        return self.sys.step(lbm, t, dt, freeze=freeze)      # g = 0, so g_dir is moot

    def readout(self, t):
        return [("EVENTS", str(sum(1 for p in self.patches if p.amp(t) > 0.02)))]


class BlackHoles(_FreeScene):
    """Three spinning black holes, each dragging a disc of water around itself, falling together
    over twenty seconds and fusing like droplets when they meet.

    Same sealed, seeded-at-rest pool as `still_water` - but instead of scheduled disturbances the
    only thing stirring it is the holes themselves, and instead of drifting they PULL on each
    other. Three things are inputs; everything you actually watch is the solver's answer:

    1. **Spin.** Each hole's rotation rate is prescribed (`FreeBody.spin`), the way `AoASweep`
       prescribes the foil's incidence. It is an actuator, not an output - a hole cannot be spun
       up by the flow it is itself driving.
    2. **The disc.** A swirl patch centred on each hole and carried with it, using the same
       per-cell `beta` drive as `still_water`: solid-body core, peak at DISC x the hole's radius,
       gaussian falloff. Its speed at the hole's own surface is exactly the hole's surface speed,
       so the boundary and the disc agree instead of shearing against each other. Where two discs
       overlap the drive weights average, and what comes out of that is not prescribed anywhere.
    3. **Mutual gravity** (`BodySystem.pair_g`), softened by the pair's radii.

    The fall is DRAG-LIMITED, not ballistic - and that is deliberate, because it is the one
    regime this engine handles well. `orbit` was cut for demanding the opposite: a pellet that
    holds its momentum against the fluid, which at these Reynolds numbers it simply will not do.
    Here a hole reaches the terminal speed where gravity balances drag
    within a fraction of a second and then creeps inward at it, accelerating as 1/r^2 pulls
    harder. Slow, smooth, and completely robust to the drag law being off by a factor of two.

    COALESCENCE. When two holes close to the contact stand-off they are replaced by a single
    droplet whose outline is their metaball union (`shapes.metaball`), and the neck closes over
    ~1.2 s. Merging at the stand-off rather than at true contact is not a shortcut: the lattice
    cannot represent a sub-cell gap, so two bodies allowed to touch would first squeeze the fluid
    between them into a huge spurious force. And because the metaball of two coincident blobs is
    a circle of radius sqrt(r1^2 + r2^2), the merge conserves area - the droplet's volume - with
    no correction factor. Nothing but another hole can ever get inside one: to the solver a body
    is bounce-back, so the fluid and every tracer are turned away at its surface.
    """
    OVERRIDES = dict(re=400.0, u0=0.045, csm=0.18, steps=26, settle=0.8, perturb=0.0,
                     closed=True, overscan=0.08, wall_clamp=0.07,
                     cmap="blues", vmax=1.25, gamma=0.85,
                     n_streaks=30000, streak_span=9.0, streak_gain=0.44,
                     streak_speed_gain=0.98, streak_life=(0.9, 2.4),
                     # a black hole has no rim: it is a hole punched in the field, not an object
                     # drawn on top of one
                     body_fill=(0, 0, 0), body_edge_px=0)
    duration = 20.0
    title = "BLACK HOLES"
    g = 0.0                       # plan view; the only gravity here is between the holes

    SIZES = (0.085, 0.100, 0.115)         # diameters as fractions of the visible frame width
    POS = ((0.28, 0.28), (0.36, 0.74), (0.66, 0.46))
    RHO = 5.0

    DISC = 2.6                    # disc radius, in hole radii (the swirl peaks here)
    V_DISC = 0.042                # peak tangential speed of a disc, lattice units
    BETA_D = 6.0e-3               # how hard the disc is driven (~0.11 s to establish)
    # Mutual gravity: a = PULL * m_other / r^2, softened by the pair's radii. NOT derived - the
    # collapse is drag-limited and the drag law at this Reynolds number is exactly what the
    # `orbit` post-mortem proves you cannot assume. Calibrated instead from two measured LBM runs
    # (measurement tooling not included here, at PULL 0 and 1.2e-5) fitted with a small ODE fit
    # (also not included here), which then predicts the merges at 12.0 s and 14.6 s of a 20 s clip.
    PULL = 5.85e-5
    DAMP = 2.0e-4                 # friction floor, as in still_water
    ABSORB = 4.0e-3               # extra damping in a band at the box walls - see `_absorber`
    ABSORB_BAND = 40.0
    INSET = 0.035
    # False restores the pre-2026-07-29 behaviour: the merged disc replaces the two parent discs
    # in a single step, and both it and the spin are sized off the circumradius. Kept for A/B
    # only (a check tool not included here, CROSSFADE=0) - it is what made the merge flash the
    # frame.
    CROSSFADE = True
    MERGE_T = 1.2                 # seconds for the neck to close (and for the disc cross-fade)
    # --- the merge RIPPLE absorber (my note, 2026-07-29: when the holes merge, the whole screen
    # ripples - remove the ripple effect) ---
    # A fuse necessarily changes the solid mask in ONE step, and that radiates an acoustic pulse
    # which crosses the picture as a travelling ripple. It is not reducible by tuning: measured
    # over trigger gaps from 3 to 18 cells, the number of lattice cells that change state at the
    # fuse is FLAT at 250-300 (~a quarter of a body), because the metaball union is ~27% fatter
    # than the two circles almost independently of separation, and a single closed polygon simply
    # cannot equal two disjoint circles. Some jump is unavoidable at any merge.
    # So absorb the wave instead, at the source. This is the SAME device as `_absorber` and for
    # the same documented reason (a sealed box has nowhere to put a pressure wave) - a band of
    # beta driving toward the existing target, which out here is rest. It is
    # keyed off `1 - mmax`, so it only ever acts where NO hole's disc reaches: the swirls, the
    # wakes and everything the clip is actually showing are untouched, and what it eats is the
    # numerical transient crossing the still water around them. It removes an artefact of the
    # discretisation, not a fluid effect. RIPPLE_A = 0 disables it for A/B.
    # Strength is set by the crossing time, not by taste: the pulse travels at the lattice sound
    # speed (0.577 cells/step) and has ~209 cells to cover to reach the frame edge, i.e. ~360
    # steps, so beta must be well above 1/360 = 2.8e-3 to kill it on the way out. Measured on the
    # rig (far-field luminance trace, quiet level 18.1): none +2.0 and ringing for over a second;
    # 3.0e-3 +1.4 and monotone; 1.4e-2 +0.4 and monotone. The ring dies at the low value, the
    # WAVEFRONT needs the high one.
    RIPPLE_A = 1.4e-2             # absorber strength in the clear field
    RIPPLE_T = 1.8                # seconds it stays up after a fuse
    RIPPLE_RISE = 0.05            # its own ramp-in, so the absorber does not step either - short,
                                  # because the pulse is already moving when the fuse fires

    def __init__(self, cfg):
        Scene.__init__(self, cfg)
        self.rng = np.random.default_rng(cfg.seed)
        ins = self.INSET
        self.sys = BodySystem(
            self.nx, self.ny, g=0.0,
            floor=(self.vx0 + ins * self.vnx) / self.nx,
            ceiling=(self.vx0 + (1.0 - ins) * self.vnx) / self.nx,
            ybounds=(self.vy0 + ins * self.vny, self.vy0 + (1.0 - ins) * self.vny))
        self.sys.pair_g = self.PULL
        self.sys.merge = True
        self.sys.merge_time = self.MERGE_T
        self.sys.on_merge = self._on_merge
        jj, ii = np.meshgrid(np.arange(self.ny, dtype=np.float32),
                             np.arange(self.nx, dtype=np.float32), indexing="ij")
        self._II, self._JJ = xp.asarray(ii), xp.asarray(jj)
        self._absorb = xp.asarray(self._absorber())
        # tangential speed of the disc AT the hole's own surface, as a fraction of V_DISC
        self._gs = float((1.0 / self.DISC) * np.exp(0.5 * (1.0 - 1.0 / self.DISC ** 2)))
        self._ripple = []             # fuse times, for the ripple absorber
        self._key = None
        self._cache = None
        self.populate()

    def _absorber(self):
        """A graded damping band along the box walls, target velocity zero.

        The sealed box has nowhere for a pressure wave to go: every acoustic transient a spinning
        hole or a merge emits reflects off the free-slip walls and comes back across the picture
        as a faint travelling RIPPLE, over and over, because nothing removes it. This is the
        second half of the de-rippling job (the first being soft wall contacts in bodies.py) - a
        band that absorbs an outgoing wave before it can turn round. Overscan keeps the band off
        camera, and the holes never go near it, so it only ever eats what is leaving.
        """
        i = np.arange(self.nx, dtype=np.float32)
        j = np.arange(self.ny, dtype=np.float32)
        di = np.minimum(i, self.nx - 1 - i)[None, :]
        dj = np.minimum(j, self.ny - 1 - j)[:, None]
        d = np.minimum(di, dj)
        return (self.ABSORB * np.clip(1.0 - d / self.ABSORB_BAND, 0.0, 1.0) ** 2).astype(
            np.float32)

    def ref_len(self):
        return max(self.SIZES) * self.vspan

    def populate(self):
        for d_frac, (fx, fy) in zip(self.SIZES, self.POS):
            r = 0.5 * d_frac * self.vspan
            self.sys.add(FreeBody(
                shapes.metaball([(0.0, 0.0, r)]), 1.0, self.sx(fx), self.sy(fy),
                name="hole", density=self.RHO, mode="dynamic", pivot=0.0,
                spin=self.V_DISC * self._gs / r))

    def prepare(self, lbm):
        """Start the pool at rest, not at a uniform freestream (see StillWater.prepare)."""
        z = np.zeros((self.ny, self.nx), np.float32)
        lbm.init_velocity(z, z)

    # -- the discs -------------------------------------------------------------------------
    @staticmethod
    def _r_eff(b):
        """A body's AREA-equivalent radius - NOT its circumradius. This sizes the disc.

        `FreeBody.radius` is the circumradius, which for a droplet mid-merge is half the length
        of the whole dumbbell: about 2.5x either parent's radius at the moment of the fuse,
        shrinking back over the next 1.2 s as the neck closes. Driving the disc off that made the
        disc BALLOON at the merge and deflate again, spinning several hundred cells of previously
        still pool up and back down - which is the white FLASH across the frame that I asked
        to have removed.

        The area-equivalent radius is the right scale for the disc anyway (the disc belongs to
        how much hole there is, not to how elongated it is at one instant), and `bodies.droplet`
        holds the area fixed through the entire coalescence, so it is CONSTANT across a merge.
        For a circle it equals the circumradius exactly, so nothing before a merge changes.
        """
        return float(np.sqrt(max(b.area, 1e-9) / np.pi))

    def _size_of(self, b):
        return self._r_eff(b) if self.CROSSFADE else b.radius

    def _on_merge(self, new, a, b):
        """Re-derive the merged droplet's prescribed spin at the instant of the fuse.

        `BodySystem._fuse` hands the new body the FIRST parent's `spin`, which on an outline
        2.5x that parent's circumradius puts the tip at ~0.083 lattice units - past `wall_clamp`
        - for the one step before `substep` recomputes it. Small, but it is a step change in a
        wall velocity, and that is the exact mechanism that radiates a ripple across the frame.
        """
        if not self.CROSSFADE:
            return
        new.spin = self.V_DISC * self._gs / max(self._r_eff(new), 1.0)
        new.omega = new.spin
        c = getattr(new, "_coal", None)
        if c is not None:
            self._ripple.append(float(c["t0"]))

    def ripple_amp(self, t):
        """Envelope of the merge ripple absorber: 0 normally, up for `RIPPLE_T` after each fuse.

        Rises over `RIPPLE_RISE` rather than switching on, because a step in beta is a step in a
        body force and there is no sense curing one transient by injecting another. Overlapping
        merges take the strongest envelope, not the sum.
        """
        a = 0.0
        for t0 in self._ripple:
            u = (float(t) - t0) / max(self.RIPPLE_T, 1e-6)
            if 0.0 <= u < 1.0:
                a = max(a, float(smoothstep(0.0, self.RIPPLE_RISE / self.RIPPLE_T, u)
                                 * (1.0 - smoothstep(0.35, 1.0, u))))
        return a

    def _disc_patches(self, t):
        """(cx, cy, R, amp) for every driven disc this step - merges CROSS-FADED, not switched.

        Outside a merge this is one patch per hole and nothing else. During one, the pair is not
        swapped for the merged disc in a single step: for `merge_time` the two PARENT discs are
        still driven, carried on the two lobe centres `BodySystem._coalesce` is animating and
        fading out, while the merged disc fades in on the same schedule. At u=0 that reproduces
        the pre-merge target field exactly, and at u=1 the merged one exactly, so the field the
        fluid is driven toward never steps - and since velocity fields superpose, the
        cross-fade in between is a legitimate field, not a blend artefact.
        """
        out = []
        for b in self.sys.active(t):
            c = getattr(b, "_coal", None)
            if c is None or not self.CROSSFADE:
                out.append((b.x, b.y, self.DISC * self._size_of(b), 1.0))
                continue
            # same easing `_coalesce` uses, so the ghost lobes sit exactly on the droplet's blobs
            w = smoothstep(0.0, 1.0, (float(t) - c["t0"]) / max(self.sys.merge_time, 1e-6))
            d = c["d"] * (1.0 - w)
            out.append((b.x - c["fa"] * d * c["ux"], b.y - c["fa"] * d * c["uy"],
                        self.DISC * c["ra"], 1.0 - w))
            out.append((b.x + c["fb"] * d * c["ux"], b.y + c["fb"] * d * c["uy"],
                        self.DISC * c["rb"], 1.0 - w))
            out.append((b.x, b.y, self.DISC * self._r_eff(b), w))
        return out

    def drive(self, t):
        """One swirl patch per live hole, carried with it, plus the friction floor and the
        wall absorber.

        The discs SUPERPOSE - their velocity fields add - rather than being weight-averaged the
        way `StillWater` blends its disturbances. That is not a stylistic choice. Two vortex
        fields add (Biot-Savart); averaging them halves the target speed wherever two discs
        overlap, and since the beta weight there is the SUM of both masks, the fluid between two
        neighbouring holes gets driven hard toward a target that is too slow. The result is a
        braked, stagnant patch between every pair, and its pressure pushes the holes APART - the
        first build of this scene had them separating steadily instead of falling together.
        Summing the velocities and taking the strongest mask for beta removes the artefact.
        """
        key = int(t * 60.0)
        if key != self._key:
            tx = xp.zeros((self.ny, self.nx), dtype=xp.float32)
            ty = xp.zeros((self.ny, self.nx), dtype=xp.float32)
            mmax = xp.zeros((self.ny, self.nx), dtype=xp.float32)
            for cx, cy, Rp, amp in self._disc_patches(t):
                if amp <= 1e-4:
                    continue
                R = np.float32(max(Rp, 1.0))
                dx, dy = self._II - np.float32(cx), self._JJ - np.float32(cy)
                r = xp.sqrt(dx * dx + dy * dy)
                rs = xp.maximum(r, np.float32(1e-6))
                q = r / R
                g = q * xp.exp(0.5 * (1.0 - q * q))          # solid-body core, peak at r = R
                m = xp.exp(-0.5 * (q / 1.5) ** 2)            # 0..1 reach of this disc
                v = np.float32(self.V_DISC * amp) * g * m
                tx = tx - v * (dy / rs)
                ty = ty + v * (dx / rs)
                # beta cross-fades with the patch too, or the forced REGION would still step
                mmax = xp.maximum(mmax, np.float32(amp) * m)
            beta = (self._absorb + np.float32(self.DAMP)
                    + np.float32(self.BETA_D) * mmax)
            ra = self.ripple_amp(t)
            if ra > 1e-4 and self.RIPPLE_A > 0.0:
                # `1 - mmax` to the fourth: essentially zero anywhere a disc reaches (0.02 at the
                # swirl's own radius, 0.56 at twice it), ~1 in the clear. So this only ever damps
                # water that no hole is driving - which is exactly where the ripple is visible and
                # exactly where the picture is supposed to be black.
                beta = beta + np.float32(ra * self.RIPPLE_A) * (1.0 - mmax) ** 4
            self._cache = (tx, ty, beta)
            self._key = key
        return self._cache

    def substep(self, lbm, t, dt=1.0, freeze=False):
        # Re-derive each hole's spin from its CURRENT size every step: after a merge the droplet
        # is bigger, so the same disc speed means a slower rotation. Setting omega too (not just
        # spin) matters during --settle, when `integrate` is not being called at all and the flow
        # is establishing around a hole that should already be turning.
        #
        # Off the AREA-equivalent radius, for the same reason the disc is (`_r_eff`): the
        # circumradius jumps 2.5x at a fuse and creeps back down over the coalescence, so keying
        # omega to it stepped the rotation rate down and then wound it back up right where the
        # picture should be settling. Area is conserved through the merge, so this is continuous
        # - and once the neck has closed the droplet is a circle again and the two agree exactly.
        for b in self.sys.active(t):
            b.spin = self.V_DISC * self._gs / max(self._size_of(b), 1.0)
            b.omega = b.spin
        lbm.set_drive(*self.drive(t))
        return self.sys.step(lbm, t, dt, freeze=freeze)

    def readout(self, t):
        return [("HOLES", str(len(self.sys.active(t))))]


class BlackHolesRed(BlackHoles):
    """`black_holes` in the RED palette - my variation, 2026-07-29.

    Colour only: same three holes, same discs, same gravity, same merges, same everything the
    solver does. `reds` is built to the same luminance ramp as `blues` (see colormap.py), which
    is why `vmax` and `gamma` carry over untouched - the field normalisation was tuned against
    that ramp, not against the hue. The base is a deep ember and the red concentrates where the
    fluid is fastest, which in this scene is exactly the driven ring around each hole.

    The merge no longer flashes the frame: that is not a colour change, it is `_disc_patches` /
    `_r_eff` on the parent class, and it applies to `black_holes` too.
    """
    # gamma 2.10, against the parent's 0.85 - the single most important number here. The tone
    # curve is applied to the NORMALISED field before the LUT (`s ** gamma`), so a gamma above 1
    # crushes everything slow toward the LUT's black floor and spends the whole colour range on
    # the fast fluid, which in this scene is only the driven ring at each hole's rim. At the
    # parent's 0.85 the far field was LIFTED instead and the frame read as mostly red.
    # My note, 2026-07-29: keep the areas outside each black hole's sphere of influence black -
    # red should mostly be concentrated closer to the edge of the holes.
    OVERRIDES = dict(BlackHoles.OVERRIDES, cmap="reds", gamma=2.10)
    title = "BLACK HOLES"


# ======================================================================================
# THE SURFING SET (2026-08-05) - three clips, one fluid, one body
# ======================================================================================
class _Water:
    """The WATER fluid profile: what it is, what it costs, and what it cannot be.

    My brief, 2026-08-05: make the fluid profile as close to water as possible, to attempt
    simulating 2-D surfing dynamics.

    RESTATE THE COMPLAINT AS A NUMBER FIRST (the AIR RECIPE's lesson 1).
    Water and air obey the SAME equations - Navier-Stokes does not know which one it is running -
    so "make it water" is not a change of physics, it is a change of DIMENSIONLESS NUMBERS, and
    the honest question is which ones and by how much. For a 2.1 m board at a surfing 6 m/s:

    | group                | water (a real board)  | `tri_foil_air_fast` | verdict           |
    |----------------------|-----------------------|---------------------|-------------------|
    | Reynolds (on length) | 1.3e7                 | 20000               | 630x too LOW      |
    | Mach                 | 0.0041                | 0.119               | 29x too HIGH      |
    | Froude               | 1.3                   | n/a - no free surface                   |

    THE TWO REACHABLE ONES ARE COUPLED, AND THAT IS THE WHOLE PROBLEM. On a D2Q9 lattice
    `nu = u0 * L / re` and `tau = 0.5 + 3 nu`, with `tau -> 0.5` the hard wall. Holding tau
    therefore holds `nu`, and holding `nu` means

        chord Reynolds number  is proportional to  u0 / scale

    so at a FIXED lattice, every reduction in Mach is paid for one-for-one in Reynolds number.
    Water wants Mach DOWN and Reynolds UP, i.e. the two things this lattice cannot do at once.
    The only currency that buys both is RESOLUTION, and it is expensive: work goes as
    cells x steps, i.e. as `1 / (u0 * scale^3)` once the apparent wind is held fixed.

    WHAT WAS ACTUALLY BOUGHT, and why this split and not another:

      - **Mach down 24%** (u0 0.055 -> 0.042, M 0.0953 -> 0.0727), which is the defining
        difference between water and air at these speeds and the direction the lattice likes.
      - **`scale` 2.0 -> 1.6 on the tri-lane clip**, which is what stops that Mach reduction from
        costing Reynolds number: chord Re lands at 19100 against `tri_foil_air_fast`'s 20000, i.e.
        HELD, at 1.9x the work.
      - **`steps` buys the apparent wind straight back**, so the on-screen
        speed is `tri_foil_rates_fast`'s to 0.1% and the legend does not move.
      - **tau is 0.5012 on all three clips** - deliberately the same number `tri_foil_air` was
        verified at, so the one quantity most likely to fail silently (grid-scale speckle) sits
        exactly where it has already been measured rather than somewhere new.

    ...and one change that is about water's CHARACTER rather than its numbers:

      - **the turbulence is re-scaled, not just inherited.** `tri_foil_air` injects Tu = 2% at an
        integral scale of 0.15 chord, which is a WIND TUNNEL's noise - fan and screen wake, small
        and busy. Open water on a wave face is not that: the face is comparatively smooth and what
        disturbance there is arrives at the scale of the wave's own motion, i.e. far larger and
        weaker. So Tu 2.0% -> 1.2% and the integral scale 0.15 -> 0.55 chord. Long, lazy
        disturbances instead of fine hash, which is what the surface of moving water looks like.

    WHAT THIS IS NOT, and cannot be. State it rather than implying the clip is a measurement -
    the same paragraph `tri_foil_air` carries, with one extra item that matters more here than
    anywhere else in this project:

      - **THERE IS NO FREE SURFACE.** A wave is a gravity wave on an interface, Froude number 1.3,
        and this solver has one phase and no interface at all. `surf_wave` does not simulate a
        breaking wave; it simulates the flow IN THE PLANE OF a wave's face, with the curl supplied
        as a forcing. That distinction is the honest description of the clip and it should be made
        whenever it is described.
      - **Re 19000 against water's 1.3e7.** 700x short, and bounded by tau, not by nerve. Going
        further needs a finer lattice, not a bigger number.
      - **M 0.073 against water's 0.004.** Still 18x high, so the lattice retains a spurious
        compressibility water does not have. It is O(M^2), i.e. ~0.5% density variation - invisible
        in the picture, real in the equations.
      - **2-D turbulence cascades the wrong way** (no vortex stretching), so wakes organise into
        large persistent vortices instead of shredding. No Reynolds number fixes dimensionality.
      - **no fins, no rocker, no rail bite.** The board is a planform in the plane of the flow, so
        everything a real board does out of plane - which is most of how it turns - is absent.
    """
    # One fluid for all three clips: the same lattice viscosity and the same Mach number, so the
    # only reason chord Reynolds number differs between them is that the boards differ in size.
    U0 = 0.042              # lattice velocity. M = u0*sqrt(3) = 0.0727
    TAU_NU = 4.0765e-4      # target kinematic viscosity -> tau = 0.50123, tri_foil_air's verified
    TU = 0.012              # turbulence intensity, fraction of the local band speed
    TU_LEN = 0.55           # integral scale in CHORDS - long and lazy, not tunnel hash

    @staticmethod
    def re_for(chord_cells, u0=U0, nu=TAU_NU):
        """The `re` that puts this chord at the water profile's viscosity. `re` is defined on the
        chord (`Scene.ref_len`), so this is the one line that keeps tau identical across clips
        whose bodies are different sizes - rather than three hand-tuned numbers that drift apart.
        """
        return float(u0) * float(chord_cells) / float(nu)


class SurfRates(_Water, TriFoilAirFast):
    """`tri_foil_rates_fast` with the FLUID changed to water and the FOILS changed to surfboards.

    My brief, 2026-08-05: starting from `tri_foil_rates_fast`, the major difference should be the
    foils changing to surfboards - keep the fixed objects with rotation from the original video,
    with only the fluid profile and the object shapes changing.

    So this clip is defined by what it must NOT change, and the constraint is sharper than it
    looks. Three things had to be held to the pixel and each took arithmetic rather than care:

      - **the legend is bit-identical.** Tick POSITIONS are `b / vmax` and `vmax` is untouched, so
        they cannot move; the LABELS are `b * SPEEDUP` and `SPEEDUP` stays 2.0 because the
        on-screen wind is held (below). The key reads 2.50 / 1.50 / 1.10 over a 5.2 bar exactly as
        the clip being copied does.
      - **the on-screen wind is held at 9.0 px/frame**, `tri_foil_rates_fast`'s. Note this is
        px/frame and not cells/frame, and the distinction is load-bearing HERE in a way it was not
        for `tri_foil_air`: apparent speed is `u0 * steps * scale`, and this clip changes `scale`,
        so holding cells/frame would have quietly changed the speed a viewer sees by the ratio of
        the two lattices. Measured: 0.042 * 134 * 1.6 = 9.005 against 0.075 * 40 * 3.0 = 9.000.
      - **the choreography is untouched.** `LAG` all zeros, `CX` all 0.32, +-34 deg on a 9 s
        period, 10 s. The boards are held and rotated by the rig exactly as the sections were -
        my "keep the fixed objects with rotation" - so `surf_sweep` and `surf_wave` are where
        the free motion lives and this clip stays a clean A/B against the one it copies.

    WHAT THE BOARD COSTS IN STABILITY, measured rather than assumed (a check tool not included
    here). Swapping a NACA 0015 for a planform is a blockage change and blockage is what sets
    peak `max|u|` here. The cos/sin estimate said +16%; the placed polygons say
    **+7% at 34 deg** (0.614 chords of cross-stream projection against 0.572), because a board's
    width is concentrated near mid-chord rather than spread along it. It is +87% at 0 deg, where
    there is margin to spare. The body's AREA is 2.09x the section's, which matters not at all to
    a scene whose bodies are held by a rig - and a great deal to `surf_sweep`, which lets go.

    Everything quoted in DERIVED units had to be re-derived, which is the same units-change trap
    in its fourth costume - a `scale` change silently rescales anything measured in cells or in frames:
      - `streak_span` 1.85 -> 2.49 holds the top-lane dash at 28.0 px (it is quoted in FRAMES of
        travel and drawn in PIXELS);
      - `BLEND` 66 -> 83 and `sponge_out` 21 -> 26 hold 0.56 chord and the same absorber depth
        (both quoted in CELLS);
      - `turb_span()` is already derived from the run rather than pinned, so it follows by itself.
    """
    CHORD_CELLS = 148.5             # 0.22 * (1080 / 1.6); see `_Water.re_for`
    OVERRIDES = dict(TriFoilAirFast.OVERRIDES,
                     scale=1.6, u0=_Water.U0, steps=134,
                     re=_Water.re_for(148.5), csm=0.24,
                     streak_span=2.49, sponge_out=26,
                     saturation=1.0, brightness=1.0)
    duration = 10.0
    title = "THREE FLOW RATES - WATER"

    SPEEDUP = 2.0                   # anchored on SCREEN speed; see the docstring
    BLEND = 83.0                    # cells; 0.56 chord, as at every earlier scale
    TU = _Water.TU
    TU_LEN = _Water.TU_LEN
    BOARD_W = 0.28                  # board width as a fraction of its length

    def __init__(self, cfg):
        super().__init__(cfg)
        # The planform replaces the section wholesale. It is exactly symmetric about its stringer,
        # so like the NACA 0015 it makes no lift at zero incidence and everything in the frame is
        # still bought by incidence alone - which is what keeps this a fair copy rather than a
        # different experiment. `flip_y` is a no-op on a symmetric outline and is not applied.
        self.prof = shapes.surfboard(width=self.BOARD_W)
        self._ylim = (float(self.prof[:, 1].min()), float(self.prof[:, 1].max()))
        self._tex = shapes.board_texture(width=self.BOARD_W)
        self._t = 0.0        # the instant `body_textures` builds its maps for; `bodies()` sets it

    def body_textures(self):
        """All three boards are PAINTED, and each gets its own exact inverse-placement map.

        The renderer's default texture map is a bounding box, which shears the deck art off a
        rotated body; these boards sweep +-34 deg, so every one of them needs `shapes.place_uv`.
        Painting them at all is the `turbofan` lesson applied up front: a
        black planform reads as a leaf, and a white deck with a stringer and a tail pad reads as a
        surfboard in the first half second. The two NACA sections this replaces stayed black
        because a section IS a line drawing; a surfboard is an object.
        """
        return [(self._tex,
                 shapes.place_uv(self.chord, self.sx(self.CX[k]), self.sy(fy),
                                 self.aoa(k, self._t), self._ylim))
                for k, fy in enumerate(self.LANES)]

    def bodies(self, t):
        self._t = t          # the texture map is built per frame and needs the same instant
        return super().bodies(t)

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (all three)"),
                ("Re", " / ".join(f"{b * self.cfg.re:.0f}" for b in self.BANDS))]


class _SurfPlane(_Water, _FreeScene):
    """Shared base for the two single-board clips: ONE surfboard planing on a TILTED PLANE.

    THE FRAME IS THE FACE OF THE WAVE, seen normal to it, and that one decision is what makes a
    2-D lattice able to say anything true about surfing. In an ordinary plan view of flat water
    (the blue-pool format) gravity is out of plane and set to zero, so a free body is moved only
    by drag and is simply washed downstream and out of shot. On a wave face gravity has a REAL
    in-plane component `g sin(theta)` pointing down the face, the water runs UP the face as the
    wave's orbital motion carries it, and a board planing at incidence is held between the two.
    That balance is the whole of surfing, it lives entirely in this plane, and it is also - not
    coincidentally - what keeps the board on screen without anything being keyframed.

    So the scene's inputs are exactly the ones a legitimate scene input allows:
      - **the plane is tilted**, i.e. there is an in-plane gravity. `flow="up"` already points
        streamwise +x up the screen, so `_FreeScene`'s existing `g_dir=(-1, 0)` is down the face
        with no new machinery at all.
      - **the board's MATERIAL**: `density_for_fall(..., v_fall=0)` ballasts it so the down-face
        pull exactly balances the drag of the oncoming face flow. That is choosing what it is made
        of, not where it goes - and `Eddies` records that ballasting heavier to hold bodies in
        frame longer DIVERGES, so hover is the setting, not a dial.
      - **the rider STEERS**: the board's attitude is prescribed and its POSITION is free. This is
        strictly less prescriptive than the clips being copied, which nailed the position too, and
        it is the honest model - a surfer sets the board's angle and the water decides where that
        takes them. It is also what makes the clip legible: a fully free planform is stable
        BROADSIDE to the stream (a flat plate's equilibrium), so an unsteered board turns sideways
        within a second or two and spends the clip being dragged, which is correct physics and a
        useless picture. `RIDER_FREE = True` restores fluid-driven rotation for an A/B.

    Prescribed attitude is fed in as `FreeBody.spin`, i.e. as an angular VELOCITY, so the moving
    wall the boundary sees is consistent with the rotation the body is actually undergoing (the
    `set_wall_velocity` term). Setting `ang` directly would have left the
    boundary believing the board was not turning.
    """
    OVERRIDES = dict(flow="up", scale=2.0, overscan=0.10,
                     u0=_Water.U0, steps=100, csm=0.24, settle=2.0, perturb=0.0,
                     cmap="blues", vmax=2.80, gamma=1.05,
                     streak_span=2.4, streak_speed_gain=0.55,
                     sponge_out=19, wall_clamp=0.11)
    duration = 15.0
    dynamic = True

    CHORD = 0.32            # of the VISIBLE cross-stream span. NOT `aoa_sweep`'s 0.36, and the
                            # difference is a stability decision rather than a taste one: what
                            # sets peak max|u| is BLOCKAGE, not chord, and a board is a far
                            # bluffer body than a section. At 0.36 the planform would span 22% of
                            # the tunnel at peak incidence against the foil's 15%; at 0.32 it
                            # spans 15.3%, i.e. the blockage `aoa_sweep_fast` was verified at.
    BOARD_W = 0.28
    G = 5.0e-5              # in-plane gravity = g sin(theta). Sized so the hover ballast lands at
                            # a density ratio near 2 - heavy enough that drag governs the board
                            # rather than added mass (the explicit-coupling limit, bodies.py), and
                            # light enough to stay far from the moving-wall speed limit.
    U_REF = 1.00            # what the board actually SEES, as a multiple of u0. A single body in
                            # a wide tunnel is not the blockage case `falling_discs` had to
                            # correct for, so this starts at 1.0 and is corrected by INVERTING a
                            # measured drift rather than bisected.
    RIDER_FREE = False      # True = fluid-driven rotation (A/B only; see the docstring)
    PEAK = 24.0             # +-deg, `aoa_sweep`'s expanded sweep
    START = (0.38, 0.50)    # (streamwise, cross-stream) fractions of the VISIBLE window

    # THE RIDER'S STATION-KEEPING, and why it exists at all. `surf_sweep` cut 1 shipped with the
    # board holding position by nothing but its ballast, and it spent four seconds of fifteen
    # PRESSED AGAINST THE SOFT CONTAINMENT SPRING - which sits well inside the frame, so on screen
    # it reads exactly as I described it: the board collides with an invisible wall.
    # Containment is a backstop against a bug; it was never meant to be the thing that holds a body
    # in shot, and using it that way is visible.
    #
    # A surfer is not ballast, though - they are an active agency that holds station on a wave, so
    # modelling one is not a fudge, it is the missing physics. `RIDER_T` is the period of the
    # station-keeping spring in SECONDS and `RIDER_Z` its damping ratio, so the behaviour is
    # quoted as "the rider corrects a displacement over about this long, without overshooting"
    # rather than as two opaque gains. Deliberately SLOW (seconds, not tenths): fast enough that
    # the board never reaches a wall, slow enough that every gust still visibly moves it.
    # 0 disables it, which is what `surf_wave` uses - see that class for why it does not need one.
    RIDER_T = 0.0           # seconds; 0 = no station-keeping
    RIDER_Z = 0.85          # damping ratio; ~1 is critically damped, so no oscillation is added

    FLOOR = 0.14            # soft containment, as fractions of nx. The visible window spans
    CEIL = 0.86             # 0.083-0.917 with overscan 0.10, so these sit comfortably inside it
                            # and the board is decelerated by a spring-damper (BodySystem._wall)
                            # rather than caught - a hard clamp steps the wall
                            # velocity and broadcasts a ripple across the whole frame.
    Y_MARGIN = 0.03         # Cross-stream containment inset, as a fraction of the visible span,
                            # ON TOP OF the board's own circumradius (which `_contain` already
                            # adds, so the WHOLE board is always in frame at any angle). 0.06 was
                            # costing 0.12 of the span for nothing; the radius is what does the
                            # work, and the extra inset only narrows the room a free body has to
                            # react in before it meets a wall.

    def __init__(self, cfg):
        # `populate` needs these, and `_FreeScene.__init__` calls it, so they are set first.
        Scene.__init__(self, cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.chord = self.CHORD * self.vspan
        self.prof = shapes.surfboard(width=self.BOARD_W)
        self._ylim = (float(self.prof[:, 1].min()), float(self.prof[:, 1].max()))
        self._tex = shapes.board_texture(width=self.BOARD_W)
        self.g = self.G
        # Containment is to the VISIBLE window, not to the lattice edge. With overscan the real
        # free-slip walls are off camera, so without this the board could wander out of shot and
        # leave the frame emptier than it was composed (the same reason `_BluePool` sets it).
        m = self.Y_MARGIN * self.vny
        self.sys = BodySystem(self.nx, self.ny, g=self.g,
                              floor=self.FLOOR, ceiling=self.CEIL,
                              ybounds=(self.vy0 + m, self.vy0 + self.vny - m))
        self.populate()
        # where the rider is trying to stay, and the grid any drive field is built on
        self._home = (self.board.x, self.board.y)
        self._gy, self._gx = np.mgrid[0:self.ny, 0:self.nx].astype(np.float32)
        self._drive_key = None

    def ref_len(self):
        return self.chord

    def turb_span(self):
        """Frozen-patch length in cells, DERIVED from how far this run actually convects it.

        Same reasoning as `TriFoilAir.turb_span` - a patch shorter than the travel recycles and
        delivers the same disturbance twice, which reads as a periodicity nobody asked for. It is
        re-derived rather than inherited because these clips run a different `steps` and a
        different duration.
        """
        travel = (self.cfg.settle + self.duration) * 60.0 * self.cfg.steps * self.cfg.u0
        return int(2 ** np.ceil(np.log2(max(1024.0, travel * 1.15))))

    def prepare(self, lbm):
        """Long, weak inflow disturbance - the water equivalent of `tri_foil_air`'s tunnel noise.

        The face of a wave is not glassy, but neither is it a wind tunnel: what disturbance it
        carries arrives at the scale of the wave's own motion rather than as fan-and-screen hash.
        So the intensity comes DOWN and the integral scale goes UP by nearly 4x (see `_Water`).
        `perturb` stays off for the same reason it does there - a one-shot kick during settling is
        the wrong model of a fluid that is disturbed for the whole run.
        """
        lbm.set_inlet_turbulence(self.TU, length=self.TU_LEN * self.chord,
                                 span=self.turb_span(), seed=self.cfg.seed + 5)

    def populate(self):
        ua = polygon_area(self.prof)
        self.board = self.sys.add(FreeBody(
            self.prof, self.chord,
            x=self.sx(self.START[0]), y=self.sy(self.START[1]),
            name="surfboard", ang=0.0,
            # Ballasted to HOVER: weight down the face exactly balances the drag of the face
            # flow, so the board planes in equilibrium and every departure from that position is
            # something the flow did.
            density=density_for_fall("surfboard", self.chord, 0.0,
                                     self.cfg.u0 * self.U_REF, self.g, ua),
            mode="dynamic", on_exit="clamp",
            spin=None if self.RIDER_FREE else 0.0))

    # -- the rider's input -------------------------------------------------------------
    def steer(self, t):
        """Prescribed board attitude in DEGREES - `aoa_sweep`'s schedule, in seconds."""
        d = self.duration
        return keyframes(t, [(0.0, 0.0), (0.07 * d, 0.0), (0.30 * d, self.PEAK),
                             (0.40 * d, self.PEAK), (0.72 * d, -self.PEAK),
                             (0.82 * d, -self.PEAK), (1.00 * d, 0.0)])

    def _rider(self, t):
        """Feed the attitude schedule in as an angular VELOCITY, in rad per LBM STEP.

        Per step rather than per second because `BodySystem.step` is called once per LBM step with
        dt = 1, and as a velocity rather than as an angle because the moving-wall boundary term is
        built from `omega` - assigning `ang` directly would turn the board while telling the
        solver it was stationary.
        """
        if self.RIDER_FREE:
            return
        h = 1.0 / (self.cfg.fps * max(self.cfg.steps * 60.0 / self.cfg.fps, 1e-9))
        rate = (self.steer(t + h) - self.steer(t - h)) / (2.0 * h)      # deg per scene-second
        self.board.spin = np.deg2rad(rate) * h

    def rider_force(self):
        """The surfer holding station: a spring-damper toward `self._home`, in FORCE units.

        Quoted through a period and a damping ratio and converted here, so the constants mean
        something physical. In LBM-step time (dt = 1 per step) the natural frequency of a
        `RIDER_T`-second oscillation is `2*pi / (RIDER_T * steps_per_second)`, and the force is
        `m * (-w^2 * dx - 2*z*w * v)`. Multiplying by the body's own mass is what makes the
        RESPONSE TIME the thing being specified rather than the stiffness - re-ballast the board
        and the rider still corrects over the same couple of seconds.

        It is a spring-damper for the same reason every contact and wall in this project is:
        anything that changes a body's velocity in one tick is fed
        straight into the moving-wall term and the boundary broadcasts it across the frame.
        """
        if self.RIDER_T <= 0.0:
            return (0.0, 0.0)
        b = self.board
        per_sec = 60.0 * self.cfg.steps                  # LBM steps per scene-second
        w = 2.0 * np.pi / (self.RIDER_T * per_sec)
        k, c = w * w, 2.0 * self.RIDER_Z * w
        return (b.mass * (-k * (b.x - self._home[0]) - c * b.vx),
                b.mass * (-k * (b.y - self._home[1]) - c * b.vy))

    def substep(self, lbm, t, dt=1.0, freeze=False):
        self._rider(t)
        # Rebuild any drive field 4x per frame rather than every LBM step - a full (ny, nx) rebuild
        # per step costs more than the solve, and nothing in these scenes moves far enough between
        # rebuilds to matter (see SurfWave for the full argument).
        if hasattr(self, "drive"):
            key = int(t * self.cfg.fps * 4.0)
            if key != self._drive_key:
                lbm.set_drive(*self.drive(max(t, 0.0)))
                self._drive_key = key
        self.board.ext = (0.0, 0.0) if freeze else self.rider_force()
        return self.sys.step(lbm, t, dt, g_dir=(-1.0, 0.0), freeze=freeze)

    def bodies(self, t):
        return self.sys.polygons(t)

    def body_textures(self):
        b = self.board
        return [(self._tex, shapes.place_uv(b.size, b.x, b.y, np.degrees(b.ang),
                                            self._ylim, pivot=b.pivot))]

    def legend(self):
        """One stream, so one mark, plus the top of the ramp. Quoted in multiples of the face
        flow, which is the only speed this clip has a name for."""
        full = self.cfg.vmax
        return dict(title="FACE FLOW",
                    marks=[(0.0, "0"), (1.0 / full, "1.00 FLOW"), (1.0, f"{full:.1f}")])

    def readout(self, t):
        b = self.board
        return [("angle", f"{np.degrees(b.ang):+.1f} deg"),
                ("drift", f"{b.x / self.nx:.3f} x / {b.y / self.ny:.3f} y")]


class SurfSweep(_SurfPlane):
    """A surfboard planing on a STEADY wave face - `aoa_sweep_fast`'s clip, in water, in blue.

    My brief, 2026-08-05: copy `aoa_sweep_fast`, change the base colour to a deep blue, make the
    fluid profile more like water, change the object to a surfboard, and rerender - against the
    standing note that the boards should now be free to move rather than pinned.

    Same framing as `aoa_sweep_fast` - `flow="up"`, 1080x1920, one body, 15 s, the same +-24 deg
    schedule written in the same seconds - so it stays a direct comparison. What changes is the
    fluid (see `_Water`), the body, the palette, and that the board's POSITION is now solved
    rather than fixed.

    THE DEEP BLUE IS THE `blues` LUT PLUS A TONE CURVE, and the tone curve is the part that does
    the work. `blues` floors at an ink blue rather than black precisely so still water reads as
    water (colormap.py), but a freestream sitting mid-ramp comes out a mid blue, not a deep one.
    The lever is `gamma`, exactly as it was for `black_holes_red`: the curve
    is applied to the NORMALISED field before the lookup, so `gamma` 1.05 with `vmax` 2.80 puts
    the undisturbed face at 0.339 of the ramp - a deep navy - and leaves the whole pale end of the
    palette for the wake and the suction peak. Editing the LUT's bottom stop instead would have
    darkened the fast fluid along with the slow.

    The apparent wind is `aoa_sweep_fast`'s: 0.042 * 100 * 2.0 = 8.400 px/frame against
    0.10 * 28 * 3.0 = 8.400. `streak_span` therefore does not move (2.4 in both) - the dash is
    `span x apparent wind`, and holding the second holds the first with no correction, which is
    the first time in this series that constant has been able to stay put.

    2026-08-05 REVIEW CUT. My review of cut 1: the board very clearly collided with an invisible
    wall; rather than flowing in one direction, the flow needed to curl so it would keep the
    board in the same spot, reading like the top view of a curling wave pushing and stabilizing
    the board; and the board needed an invisible force acting as a surfer, keeping it stable.
    Three changes, and the first is a bug:

    **THE INVISIBLE WALL WAS REAL AND I MIS-MEASURED IT.** Cut 1's board sat on the soft
    containment spring from t=5.5 to t=8.5 - four seconds of fifteen. I checked it and reported it
    clear, because I measured the board's distance to the FRAME EDGE (never closer than 9%) when
    the thing stopping it was the containment stand-off at 0.221 of the frame, i.e. deep inside the
    picture. That is why it reads as an *invisible* wall rather than as the board leaving shot.
    `tools\\surf_check.py` has had a "frames clear of the containment band" column the whole time
    and I never ran it on this scene. **Containment is a backstop against a bug; the moment it is
    load-bearing for composition, it is visible.**

    **THE FLOW IS NOW A CURLING CREST, and it is periodic ALONG the crest on purpose.** The forcing
    is the curl of a streamfunction

        psi = A * exp(-((x - x_crest)/w)^2) * cos(2*pi*(y - c*t)/L)

    which is a travelling row of counter-rotating cat's eyes confined to a narrow band at
    `x_crest` - exactly the pattern a shear layer rolls up into, and exactly what a breaking crest
    looks like from above. Three properties earn it:
      - **it is divergence-free by construction** (it is a curl), so it can be imposed without the
        acoustic transient that seeding a non-solenoidal field costs;
      - **it needs no net mass transport**, unlike the along-crest JET tried first. Side walls here
        are free-slip, so a sustained one-way cross-stream flow simply piles fluid against one wall
        and stalls;
      - **and it is PERIODIC in y, so a pattern travelling past a body exerts ZERO MEAN FORCE.**
        That is the property `surf_wave` cut 1 and cut 2 both lacked, and it is why this one does
        not need the alignment trick that scene needed: successive billows push the board opposite
        ways, so nothing accumulates no matter where the board sits.

    `w` is thin (0.22 chord) rather than fat, and that is arithmetic rather than taste: the two
    velocity components of the pattern scale as `A*2*pi/L` and `~0.86*A/w`, so they are comparable
    only when `w ~ 0.14*L`. A wide band gives a crest that undulates up and down the face instead
    of one that rolls along itself.

    **THE RIDER HOLDS STATION** (`RIDER_T` / `RIDER_Z`, see `_SurfPlane`) - a spring-damper on the
    board with a ~2.5 s response, which is the honest model of what a surfer does and removes the
    containment wall from the picture entirely. The board still visibly reacts to every billow;
    what it no longer does is wander into a boundary and stop.
    """
    duration = 15.0
    title = "SURFBOARD - CURLING CREST"
    OVERRIDES = dict(_SurfPlane.OVERRIDES, re=_Water.re_for(0.32 * 540), legend=True)

    # THE BOARD WAS UNDER-BALLASTED FOR ITS OWN STEERING, which is what actually drove cuts 3 and
    # 4 into the walls - not the crest. `density_for_fall` sizes the ballast from a drag
    # coefficient at ZERO incidence, and the rider sweeps this board to +-24 deg, where a planform
    # drags several times harder. Measured in the pocket, |F|/W ran 4-12 with the crest barely
    # reaching the board at all. A body carrying a permanent 10x imbalance rides up-face whatever
    # is above it, and no station-keeping spring tuned for a 2x imbalance will hold it.
    # `U_REF` 2.0 ballasts for the drag of the flow the board ACTUALLY works in (2a.21's rule:
    # invert a measured ratio rather than bisect), which drops the residual imbalance to ~1.5 W and
    # leaves the rider doing what a rider should - trimming, not hauling.
    U_REF = 2.0
    START = (0.38, 0.50)    # IN THE POCKET, ~2.9 band-widths below the crest - NOT on it.
                            # Putting the board on the crest line looked right (the shear there is
                            # exactly zero) and is an UNSTABLE equilibrium: the crossflow rises to
                            # 1.36 u0 within half a band-width, and side-on the board presents its
                            # LENGTH rather than its width - 3.6x broader - so that lands as ~6x
                            # the load its ballast was sized for. Measured, cut 3 was flung into
                            # the wall by t=2 at |F|/W = 17.9. At 2.5 band-widths the steady
                            # crossflow is 0.014 u0, i.e. nothing, while the SHED BILLOWS still
                            # reach the board at a few tenths of u0 because a vortex's induced
                            # velocity falls off as 1/r where the imposed profile falls off as a
                            # gaussian. That difference is the whole design: the board is worked by
                            # what the crest THROWS, not pinned by the crest itself.
    RIDER_T = 1.2           # seconds for the rider to correct a displacement. Firmer than
                            # `surf_wave`'s 2.5 because this scene asks the rider to hold station
                            # against a crest as well as a sweep, and my brief for it was
                            # "keeping the board stable". Note the rider's restoring force scales
                            # with the body's MASS, so re-ballasting above did most of the work
                            # here: a heavier board needs less displacement for the same hold.
                            # 0.6 s was measured and was TOO firm - combined with the 4x ballast it
                            # pinned the board to 0.07 chords of travel over the whole clip, which
                            # is stable in the sense of a bolted-down model rather than of a rider.
                            # Both knobs suppress motion, so they have to be set together: this is
                            # the value that leaves 0.2-0.5 chords of visible response while
                            # keeping the peak excursion inside the room the crest allows.
    RIDER_Z = 0.85

    CREST_X = 0.62          # crest station, fraction of the VISIBLE streamwise extent - ABOVE the
                            # board, which rides in the pocket below it (see START). The gap is set
                            # by how far the board is ALLOWED to travel: softening the rider enough
                            # to see the board react also lets it drift up-face, and it must not
                            # drift inside ~2 band-widths of the crest, where the crossflow rises
                            # past 1.2 u0 and grabs it. 3.8 band-widths of standoff buys 110 cells
                            # of travel before that becomes a risk.
    CREST_W = 0.35          # band half-width in CHORDS. Sets the roll-up wavelength too: the most
                            # unstable KH mode is ~7x the layer thickness, which is why this and
                            # CREST_L are chosen together rather than independently.
    CREST_L = 1.70          # wavelength ALONG the crest, in chords (~1.8 billows across a frame)
    CREST_U = 0.40          # amplitude of the cat's-eye SEED, as a multiple of u0
    SHEAR = 1.20            # peak along-crest shear, as a multiple of u0 - the thing that makes
                            # the crest ROLL. See the docstring.
    PEEL_T = 6.0            # seconds for the pattern to travel one frame width along the crest
    BETA = 0.025            # drive relaxation rate, applied only in the crest band
    RAMP = 1.2              # ease the forcing in and out; a body force switched on in one step is
                            # a pressure discontinuity

    def envelope(self, t):
        return (smoothstep(0.0, self.RAMP, t)
                * (1.0 - smoothstep(self.duration - self.RAMP, self.duration, t)))

    def drive(self, t):
        """(target ux, target uy, beta) for the curling crest - the analytic curl of `psi`."""
        u0 = self.cfg.u0
        xc = self.sx(self.CREST_X)
        w = self.CREST_W * self.chord
        L = self.CREST_L * self.chord
        # Phase speed of the PATTERN along the crest, in cells per SCENE-SECOND - quoted per
        # second rather than per LBM step because `t` is in seconds, and mixing the two is how the
        # first draft of this line ended up a factor of 60 out. This moves where the forcing is
        # applied; it is NOT added to the target velocity (doing that is what exported the board in
        # `surf_wave` cut 2 - a travelling vortex does not impose its own translation on the fluid).
        c = (self.vny - 1) / self.PEEL_T
        s = (self._gx - xc) / w
        env = np.exp(-s * s)
        ph = 2.0 * np.pi * (self._gy - c * t) / L
        # u = (d psi/dy, -d psi/dx), so this is exactly solenoidal
        A = self.CREST_U * u0 * L / (2.0 * np.pi)
        ux = -A * (2.0 * np.pi / L) * env * np.sin(ph)
        uy = A * (2.0 * s / w) * env * np.cos(ph)
        # THE MEAN SHEAR is what makes the crest roll rather than sit there. A pure cat's-eye
        # pattern is a picture OF a roll-up; a shear layer IS one, and the billows that grow on it
        # are the solver's, with the pattern above acting only as a seed that selects a wavelength.
        # `s*exp(-s^2)` is ODD in s, so it carries EXACTLY ZERO net cross-stream mass transport -
        # which the along-crest jet tried first did not, and free-slip side walls have nowhere to
        # put it. It also depends on x alone, so it adds nothing to the divergence. Normalised by
        # 1/max(s*exp(-s^2)) = 2.332 so `SHEAR` means the peak, not an amplitude nobody can picture.
        uy = uy + self.SHEAR * u0 * 2.332 * s * env
        e = self.envelope(t)
        # the crest rides ON the face flow, so the target keeps the tunnel's own stream
        # Wider than the pattern so the band's edge is not a step, then TRIMMED smoothly to
        # exactly zero: a gaussian never reaches zero, and an everywhere-nonzero beta is a weak
        # drive over the whole domain that quietly stiffens the freestream against its own tunnel.
        bw = np.exp(-0.25 * s * s)
        beta = self.BETA * e * np.clip((bw - 0.05) / 0.95, 0.0, 1.0)
        return ((u0 + ux).astype(np.float32), uy.astype(np.float32), beta.astype(np.float32))

    def readout(self, t):
        b = self.board
        return [("angle", f"{np.degrees(b.ang):+.1f} deg"),
                ("off-home", f"{np.hypot(b.x - self._home[0], b.y - self._home[1]) / self.chord:.2f} chord")]


class SurfWave(_SurfPlane):
    """The same board when the face starts CURLING - a wave peeling past, with a free-body diagram.

    My brief, 2026-08-05: a free body diagram on the surfboard, a single surfboard in a tunnel
    with a water fluid profile, the fluid 'curling' as if it were a true 3-D wave while the frame
    stays like `aoa_sweep_fast` - showing the full reactive motion of a surfboard responding to a
    wave without rendering a full 3-D simulation, which would take hours.

    THE HONEST DESCRIPTION OF WHAT THIS IS. It is not a breaking wave: a wave is a gravity wave on
    a free surface at Froude ~1.3, and this solver has one phase and no interface (see `_Water`).
    It is the flow IN THE PLANE OF THE WAVE'S FACE, with the curl put in as a FORCING - which is
    a legitimate scene input ("a fan, a jet, a stirrer") and is the same device `still_water` and
    `black_holes` use. What is forced is a pocket; what the
    pocket then does to the water, and what the water then does to the board, is entirely the
    solver's answer. That is the trade the brief asks for and it is worth stating plainly rather
    than letting the clip imply a free-surface simulation.

    THE POCKET, and why it is built this way. Seen normal to the face, the curling part of a
    peeling wave is a compact region that (a) rotates, (b) throws water up the face harder than
    the shoulder does, and (c) TRAVELS along the crest as the wave peels. So it is one elliptical
    patch, elongated along the crest, whose target velocity is the superposition of exactly those
    three: a solid-body rotation, an up-face jet, and the patch's own translation. Three notes:

      - **it carries its own translation velocity**, so the patch is a travelling vortex rather
        than a stationary stirrer that the mean flow has to fight past. Omitting that term makes
        the drive fight the stream it sits in and the pocket smears instead of holding together.
      - **the pieces SUPERPOSE - sum the velocities, take the max of the masks.** Weight-averaging
        them is what separated `black_holes`' three holes
        while reporting perfect stability.
      - **the drive field is rebuilt 4x per frame, not every LBM step.** A full (ny, nx) rebuild
        per step would cost more than the solve; the pocket moves ~0.25 cells in that time, so
        this is far finer than the once-per-frame `jet_engine` gets away with, and the field is a
        smooth body force rather than a mask - there is no discontinuity to slap the pressure.

    THE FREE-BODY DIAGRAM IS MEASURED, NOT DRAWN. Every arrow is a quantity the solve already
    computes: **F** is the momentum the fluid handed the board's surface this step
    (`LBM.force_field` summed over its ring - the same sum that gives Cd, validated against a
    textbook 1.2 by `tools/calib.py`), low-passed exactly as the integrator sees it; **W** is the
    in-plane weight `(rho - 1) A g` down the face; **NET** is their sum, i.e. the `m a` that
    actually moves the board. All three share ONE force-to-length scale, so their relative lengths
    are the content. **V** is the board's velocity and is drawn in a different colour and style
    because it is in different units and CANNOT share that scale - saying so is cheaper than
    letting a viewer assume it does.

    Note this clip deliberately reverses a standing preference. The source video this project was
    built from had force arrows on its foil and I cut them because I disliked them; here they are
    the brief, so `fbd` is on for this scene and this scene only, and `--no-fbd` turns them off.
    """
    duration = 15.0
    title = "SURFBOARD - CURLING FACE"
    OVERRIDES = dict(_SurfPlane.OVERRIDES, re=_Water.re_for(0.32 * 540),
                     legend=True, fbd=True, settle=2.4)

    # Ballasted against the flow the board ACTUALLY sees, which on a curling face is not `u0`.
    # Drag goes as u^2, so a zero-mean velocity perturbation of rms u' still raises the mean drag
    # by <u'^2>, and the right ballast is u0*sqrt(1 + <u'^2>/u0^2) rather than u0. At the pocket
    # strength above that puts U_REF near 1.15. `surf_sweep` keeps 1.00 and measured a board that
    # sits at 0.40-0.50 of the frame all clip, so the single-body hover balance itself is right -
    # this correction is the pocket's, and it belongs to this scene alone.
    # THE RIDER HOLDS STATION HERE TOO (cut 4). Cut 3's alignment trick took the pinning from 60%
    # of the clip down to 12%, which is a much better clip and still not a clean one: measured,
    # the board drifted x/vis 0.440 -> 0.801 and grazed the containment ceiling for ~1.8 s of 15.
    # Aligning `dx` removes the term that accumulates FASTEST, but `U_JET` and the <u'^2> drag rise
    # are still one-way, and over 15 s a small one-way force is enough. A restoring force is the
    # right answer to a residual drift, and a surfer IS one - so this scene gets the same
    # station-keeping `surf_sweep` does rather than a fourth round of forcing tweaks.
    RIDER_T = 2.5
    RIDER_Z = 0.85

    U_REF = 1.05
    START = (0.44, 0.50)    # ON the pocket's own station, and on the board's measured hover point
                            # (`surf_sweep` settled at x/nx 0.40-0.50, i.e. x/vis ~ 0.44). Starting
                            # a free body at its equilibrium is what stops the clip opening with a
                            # transient that then has to be absorbed by a wall.

    # THE FIRST CUT PINNED THE BOARD, and the cause is analytic rather than a tuning accident.
    # Cut 1 (U_JET 1.00, OMEGA_U 1.00, radii 1.05 x 1.75 chords) passed `dbg.py` cleanly at peak
    # max|u| 0.1851 - and its x/nx trace read 0.40, 0.51, 0.65, 0.77, then **0.77 for the last
    # nine seconds of fifteen**, i.e. parked against the soft containment ceiling at 0.778. Stable,
    # on screen, and not reacting to anything: the exact failure a health check exists to catch,
    # and `dbg.py` called it a pass. (The diagnostic that proved it stable
    # already contained the measurement that proved it wrong, one column to the right.)
    #
    # WHY, in one line: `U_JET` is a term with a systematic +x component, so every pass of the
    # pocket hands the board another up-face impulse and none of them cancel. The ROTATION term is
    # antisymmetric in the sweep direction - as the pocket comes past, `u_x = -om*(y - cy)` is
    # first positive and then negative - so its net displacement over a pass is zero by
    # construction. The fix is therefore not "less drive", it is **move the drive out of the term
    # that accumulates and into the term that does not**, which is also the term that means
    # "curling" in the first place.
    #
    # One second-order effect still has to be paid for: drag goes as u^2, so even a zero-MEAN
    # velocity perturbation raises the MEAN drag by <u'^2>. That is a ballast question, not a
    # forcing question, and it is answered by `U_REF` below rather than by weakening the pocket.
    # CUT 2 STILL PINNED, in the OTHER axis, and the first analysis was half right. Writing the
    # rotation out at the board's position, with `d = board - pocket`:
    #
    #     u_x = -om * dy * m(t)      dy CHANGES SIGN as the pocket sweeps past  -> cancels
    #     u_y = +om * dx * m(t)      dx is FIXED during a pass, m(t) > 0 throughout -> INTEGRATES
    #
    # So a rotating patch that sweeps past a body at a DIFFERENT streamwise station always shoves
    # it sideways, in one direction, every pass - which is real physics (a vortex passing below you
    # pushes you along) and fatal to a 15 s clip with walls. Worse, the two failures were coupled:
    # `U_JET` lifted the board UP the face, which put it ABOVE the pocket, which flipped `dx`
    # positive and drove it into the right-hand wall. Measured dx at the failure: +0.200 of the
    # visible span.
    #
    # So the cure is BOTH halves at once: kill the systematic lift so the board stays at its
    # ballast equilibrium, and put the pocket AT that equilibrium station so `dx ~ 0` and the only
    # term left is the one that cancels. What survives is the board being lifted up the face as the
    # curl arrives and set back down as it leaves - a pump, not a one-way shove, and much closer to
    # what riding a passing section actually looks like.
    # -- the pocket --------------------------------------------------------------------
    POCKET_X = 0.44         # station up the face, as a fraction of the VISIBLE streamwise extent.
                            # Set EQUAL to the board's hover station (see START) so that `dx ~ 0`
                            # and the accumulating term above is multiplied by nothing. This is the
                            # single most load-bearing constant in the scene.
    POCKET_RX = 0.70        # patch radii, in CHORDS. Cut 1 ran 1.05 x 1.75, which at 181 x 302
    POCKET_RY = 1.10        # cells in a 1152 x 648 lattice was not a pocket - it was most of the
                            # frame, so "the wave arrives" read as "everything speeds up". A curl
                            # is a compact feature and has to be drawn as one.
    U_JET = 0.10            # extra up-face push inside the pocket, as a multiple of u0. Nearly
                            # nothing: this is the term that lifts the board off its equilibrium
                            # and so sets `dx` for the term above. Cut 1 ran 1.00, cut 2 ran 0.30.
    OMEGA_U = 1.20          # rim speed of the rotation, as a multiple of u0. The DOMINANT term,
                            # which is what makes the pocket read as a curl rather than as a gust,
                            # and - at dx ~ 0 - what lifts the board and sets it back down again.
    BETA = 0.020            # drive relaxation rate. Weak on purpose - strong enough to establish
                            # the pocket, light enough that the board's own wake survives instead
                            # of being erased back to the target field (see LBM.set_drive).
    PEEL_T = 9.0            # seconds for the pocket to cross the frame once. 1.7 passes in the
                            # clip rather than 2.1: any residual systematic drift accumulates PER
                            # PASS, so fewer passes is margin, and one wave peeling past over most
                            # of the clip is the more honest picture anyway.
    PEEL_0 = -0.30          # it starts and ends OFF the visible span, so the clip opens on a
    PEEL_1 = 1.30           # clean face and the pocket arrives rather than appearing
    RAMP = 1.2              # seconds to ease the drive in and out - a body force switched on in
                            # one step is a pressure discontinuity, which this project has fixed
                            # three times already

    # -- the diagram -------------------------------------------------------------------
    ARROW_W = 0.40          # length of the WEIGHT arrow, in chords. Every force arrow is scaled
                            # by this one number, so the diagram is internally consistent - which
                            # also means the LONGEST arrow is set by peak |F|/W, not by taste. The
                            # measured distribution over a clip is mean 1.13 W and peak 9.4 W (a
                            # billow arriving on the board is genuinely that), so this is set from
                            # the MEAN - it puts the typical arrow near half a chord and lets rare
                            # transients overshoot. Deliberately NOT a per-arrow clamp: capping one
                            # arrow and not another is exactly the lie a free-body diagram must
                            # not tell.
    ARROW_V = 55.0          # SEPARATE scale for the velocity arrow, in chords per lattice unit.
                            # It has different units and must not appear to share the force scale.
    C_W = (255, 196, 72)
    C_F = (120, 232, 255)
    C_R = (214, 160, 255)
    C_N = (255, 96, 128)
    C_V = (150, 250, 186)

    # -- forcing -----------------------------------------------------------------------
    def envelope(self, t):
        return (smoothstep(0.0, self.RAMP, t)
                * (1.0 - smoothstep(self.duration - self.RAMP, self.duration, t)))

    def pocket_centre(self, t):
        """Where the curl is, in LATTICE coords, and how fast it is travelling (cells/step).

        The peel runs along the crest, i.e. across the frame, which in `flow="up"` is lattice +y.
        """
        u = (t % self.PEEL_T) / self.PEEL_T
        fy = self.PEEL_0 + (self.PEEL_1 - self.PEEL_0) * u
        y = self.sy(fy)
        x = self.sx(self.POCKET_X)
        # cells per LBM step: (visible span crossed) / (steps in one peel)
        c = ((self.PEEL_1 - self.PEEL_0) * (self.vny - 1)
             / (self.PEEL_T * self.cfg.fps * self.cfg.steps * 60.0 / self.cfg.fps))
        return x, y, c

    def drive(self, t):
        """(target ux, target uy, beta) for the curling pocket - superposed, never averaged."""
        u0 = self.cfg.u0
        cx, cy, c_peel = self.pocket_centre(t)
        rx, ry = self.POCKET_RX * self.chord, self.POCKET_RY * self.chord
        dx = (self._gx - cx) / rx
        dy = (self._gy - cy) / ry
        m = np.exp(-0.5 * (dx * dx + dy * dy) * 2.2)
        # Trim the gaussian's tail to a COMPACT patch - but by rescaling, not by thresholding.
        # `where(m < eps, 0, m)` leaves a 2%-of-peak step all the way round the patch, and a step
        # in a body force is a pressure discontinuity, which is the artefact this project has
        # chased three times. This reaches zero smoothly.
        m = np.clip((m - 0.02) / 0.98, 0.0, 1.0).astype(np.float32)
        # solid-body rotation about the patch centre, normalised so the rim runs at OMEGA_U * u0
        om = self.OMEGA_U * u0 / max(ry, 1.0)
        # SUPERPOSE the three pieces - jet + rotation + the patch's own translation along the
        # crest - by summing the velocities. Weight-averaging them is the same documented
        # failure as before, and the translation term is what makes this a travelling vortex
        # rather than a stirrer the mean flow has to shove past.
        # NOTE what is NOT here: a `+ c_peel` translation term. Cut 1 added one, on the argument
        # that a travelling vortex should carry its own translation - and that is wrong twice.
        # Physically, a vortex moving through fluid does not impose its translation on the fluid;
        # the rotation IS the field. And practically it is the y-axis version of the `U_JET`
        # mistake above: a sustained +y target inside the patch hands the board a cross-stream
        # impulse on every pass, none of which cancel, and the coarse preview showed it driven to
        # 78% of the way across the frame. The patch still TRAVELS - its centre moves, which is
        # the kinematics of a peel - but the along-crest flow that a real barrel has is now the
        # solver's answer to a moving stirrer rather than something imposed. `c_peel` is still
        # returned by `pocket_centre` because the checks and the readout quote it.
        utx = m * (self.U_JET * u0 - om * (self._gy - cy))
        uty = m * (om * (self._gx - cx))
        e = self.envelope(t)
        return (utx.astype(np.float32), uty.astype(np.float32),
                (m * (self.BETA * e)).astype(np.float32))

    # -- the free-body diagram ---------------------------------------------------------
    def arrows(self, t):
        """W, F, NET and V on the board, in LATTICE coordinates.

        `_fx/_fy` rather than the raw per-step sum: a sharp-interface moving boundary gains and
        loses links in whole cells, so the instantaneous load spikes by an order of magnitude
        (bodies.py `integrate`). The low-passed value is both what the body actually responds to
        and the only one that is legible as an arrow.
        """
        b = self.board
        wx = (b.density - 1.0) * b.area * self.g * (-1.0)       # down the face; g_dir = (-1, 0)
        fx, fy = b._fx, b._fy
        # The RIDER's force is drawn because it ACTS. Once station-keeping is switched on it enters
        # the same sum the integrator uses, so a diagram showing only W and F would not add up to
        # NET - it would be a picture of a different problem. A free-body diagram that omits a
        # force it is subject to is simply wrong, however tidy it looks.
        rx, ry = b.ext
        nx_, ny_ = fx + wx + rx, fy + ry
        s = self.ARROW_W * self.chord / max(abs(wx), 1e-12)     # ONE scale for every force
        out = []
        for (vx, vy, col, lab) in ((wx, 0.0, self.C_W, "W  down-face"),
                                   (fx, fy, self.C_F, "F  water"),
                                   (rx, ry, self.C_R, "S  rider"),
                                   (nx_, ny_, self.C_N, "NET")):
            out.append((b.x, b.y, b.x + vx * s, b.y + vy * s, col, lab))
        # The velocity arrow is deliberately THINNER and short-labelled. It is the only vector
        # here that is not a force, so it cannot share the scale above, and the reader has to be
        # told that without a caption long enough to be clamped against the frame edge - which is
        # what "V (own scale)" did, colliding with the legend every time the board moved left.
        # A distinct colour, a distinct weight and a label that names the quantity do the job.
        v = self.ARROW_V * self.chord
        out.append((b.x, b.y, b.x + b.vx * v, b.y + b.vy * v, self.C_V, "V (board)", 3))
        return out

    def readout(self, t):
        b = self.board
        cx, cy, _ = self.pocket_centre(t)
        return [("angle", f"{np.degrees(b.ang):+.1f} deg"),
                ("pocket", f"{(cy - self.vy0) / max(self.vny - 1, 1):.2f} across"),
                ("|F|", f"{np.hypot(b._fx, b._fy):.3e}")]


# ======================================================================================
# THE DUEL SET (10 s each, 2026-08-16) - `tri_foil_air_fast` STOOD UPRIGHT AND CUT TO TWO LANES
#
# My brief: queue up a series of renders that are mainly variations in format - flow upward, the
# objects reoriented, keeping two of the three trifoil lanes (the middle and bottom, and their
# colour scale), with the legends removed and the now-upward airfoils placed near the bottom of
# the screen - a video of "duel" lanes facing upward, for comparison.
#
# ONE FORMAT CLASS, THREE CLIPS. `_DuoFoilUp` owns the format and nothing else; each clip below
# changes exactly one thing about what is IN it. The fluid is never the variable in any of them -
# `u0` 0.055, `re` 16000, `csm` 0.24, tau 0.501225, `steps` 110, the `jet` palette, `vmax` 2.60
# and the continuous freestream turbulence are `tri_foil_air_fast`'s, field for field.
# ======================================================================================
class _DuoFoilUp(TriFoilAirFast):
    """THE FORMAT: `flow="up"`, two lanes side by side, bodies low in the frame, no legend.

    A format class, not a clip - it is never registered on its own. Everything here is about
    where the picture is and nothing is about what the fluid is, which is what makes the three
    clips below honest copies of a verified parent rather than three new solves.

    -----------------------------------------------------------------------------------------
    WHY THE ROTATION IS FREE, AND WHAT IT IS NOT
    -----------------------------------------------------------------------------------------

    **The bodies do not need reorienting, and writing code to reorient them would have been the
    bug.** The solver always works in wind coordinates - lattice +x is streamwise, always - and
    only `render.py` maps that onto the screen. `shapes.naca4` runs from a
    leading edge at unit x = 0 to a trailing edge at x = 1, so the section already faces INTO the
    stream by construction. Turning `flow` from "right" to "up" turns the whole picture, foil
    included: the nose that pointed left into a left-to-right stream now points DOWN into an
    upward one, and the wake that ran off the right edge now runs up the frame. Nothing in this
    class rotates a polygon, and nothing should - a hand-applied 90 degrees would be a rotation
    applied twice, and it would put the trailing edge into the wind.

    **`LANES` now runs ACROSS the screen, and `CX` runs UP it.** `sy()` is the cross-stream axis
    and `sx()` the streamwise one whichever way the tunnel points, so the constants keep their
    names and change their meaning - which is exactly the same units-change trap, in yet
    another costume. Under `flow="up"` lattice +y maps to screen RIGHT, so `LANES` reads LEFT to
    RIGHT and `BANDS[0]` is the LEFT lane. The parent's fast-to-slow ordering is preserved as a
    90-degree rotation of its frame: its MIDDLE lane (0.75) goes left, its BOTTOM lane (0.55)
    goes right.

    -----------------------------------------------------------------------------------------
    THE FOUR NUMBERS THAT ARE MEASURED RATHER THAN CARRIED OVER
    -----------------------------------------------------------------------------------------

    1. **`settle` 2.5 -> 4.5 s, and this one would have shipped as a visible artefact.** Settling
       is quoted in SECONDS but what it has to buy is DOMAIN TRANSITS, and the streamwise domain
       is what this format doubles: 648 cells becomes 1152. At 60 x 110 x 0.055 = 363 cells/s the
       parent's 2.5 s is 1.40 transits and would be **0.79** here - so the frozen inlet turbulence
       would have reached only ~79% of the way up the lattice at frame 0 and the top fifth of the
       picture would have opened on unperturbed laminar freestream, catching up during the first
       second. 4.5 s restores the family's 1.42 transits. (`prepare()` seeds the whole domain with
       the band profile, so the BANDS themselves are established at step 0 either way - it is the
       turbulence, which is injected at the inlet and convected, that has to make the crossing.)
    2. **`turb_span()` re-derives itself and lands on the parent's 8192 anyway.** It is quoted in
       CELLS OF CONVECTION and computed from `(settle + duration) x 60 x steps x u0`, so the
       longer settle raises the travel 4537 -> 5263 cells; 15% headroom rounded up to a power of
       two is 8192 in both. The patch still does not recycle, which is the only thing that matters
       (a recycled patch delivers the same gust twice).
    3. **`CX` 0.32 -> 0.16 is the brief, and half of it is the legend leaving.** The parent moved
       its bodies to 0.32 to buy ~50 px of clearance between the legend panel and the top lane's
       leading edge; with `legend=False` that constraint is gone, so the station is free to answer
       "near the bottom of the screen" instead. Measured on the placed polygons: the leading edge
       lands **248 px above the bottom edge** and the trailing edge at 486, over **1.85 chords**
       of run-in from the inlet plane (the parent has 1.66) and with **6.0 chords of visible
       wake** downstream against the parent's ~2.4. That last number is the whole reason this
       format exists.
    4. **Blockage goes UP and the peak goes DOWN, which is worth stating so nobody re-derives it
       the fast way.** Two foils at 34 deg block 2 x 81.2 = 162.4 of 648 cross-stream cells
       (**25.1%**) against the parent's three-in-1152 (21.1%) - a ~5% larger constriction factor.
       But the lane this set DROPS is the parent's fastest, and the peak here has been measured to
       be blockage-set on a flat Reynolds ladder at a velocity that scales
       with the band. Running 0.75 where the parent ran 1.25 is the larger effect by a wide
       margin, and `dbg.py` answers it per clip rather than this docstring asserting it.

    `chord` is deliberately NOT re-derived. It is `0.22 x vspan` and `vspan` is the screen's
    cross-stream extent in both formats - 540 cells either way at `scale` 2.0 - so the section is
    **118.8 cells / 238 px in this format and in the parent**, and `ref_len()` (and therefore
    `nu`, and therefore tau) is untouched. Shrinking it to buy blockage parity was considered and
    rejected: `nu = u0 * ref_len / re`, so a smaller chord at fixed `re` walks tau toward the 0.5
    wall through a DEFINITION - the trap `tri_cylinders` records - and would have needed `re`
    derived down with it to fix something that measurement says is not broken.

    THE BAND SEAM IS NOW THE DIVIDING LINE, and it is free. `BLEND` stays 66 cells (0.56 chord,
    the parent's), and the midpoint between two lanes at 0.25 and 0.75 is the frame's centre - so
    the shear layer between the two speeds runs straight down the middle of the picture at
    **540 px**, 132 px wide. In the parent that seam was a horizontal detail between lanes; here
    it is the line the two lanes duel across. Nothing was added to draw it: it is the same
    momentum-held interface `tri_foil_rates` built, seen edge on.
    """
    # `flow` belongs in OVERRIDES and NEVER on the command line - an explicit --flow silently wins
    # over a scene's own choice. `legend=False` has to be
    # explicit because `TriFoilRates` turns it ON - the colour key is that scene's SUBJECT, and it
    # is not this one's; I asked for it gone. `settle` is item 1 above.
    OVERRIDES = dict(TriFoilAirFast.OVERRIDES, flow="up", legend=False, settle=4.5)
    duration = 10.0

    LANES = (0.25, 0.75)           # ACROSS the screen now: lane centres at 270 and 809 px of
                                   # 1080, i.e. symmetric, 270 px of margin outside each and
                                   # 377 px of clear span between the two foils at full incidence
    CX = (0.16, 0.16)              # UP the screen: 0 is the inlet, which is the BOTTOM edge here.
                                   # One station for both lanes - the comparison needs one start
                                   # line, the same reason `_FarmSet` puts both noses on `NOSE`
    LAG = (0.0, 0.0)               # identical movement, so the only variable is the one the
                                   # clip is about. Inherited from `TriFoilRates`, restated
                                   # because a leftover 3-tuple would work silently and wrongly
    BANDS = (0.75, 0.55)           # the parent's MIDDLE and BOTTOM lanes, unchanged, so those two
                                   # lanes sit at exactly the colours I asked to keep. The
                                   # colour field is `speed / (u0 * vmax)` and every band is a
                                   # fixed multiple of `u0`, so this is bit-for-bit the parent's
                                   # two lower lanes rather than a match by eye
    SPEEDUP = 4.0                  # inherited; with no legend it now labels nothing, and is kept
                                   # only so `--legend` produces honest numbers if ever passed

    def legend(self):
        """No colour key: I cut it from every clip in this set.

        `None` rather than a two-mark key, and rather than leaving the parent's: `TriFoilRates`
        labels its marks TOP / MID / LOW, and with two bands in a VERTICAL format those words
        name the wrong edge of the frame entirely. A key that has to be read backwards is worse
        than no key. `render.Tunnel` treats None as "this scene has no legend", so `--legend`
        degrades to nothing instead of drawing something misleading.
        """
        return None

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (both lanes)"),
                ("Re", " / ".join(f"{b * self.cfg.re:.0f}" for b in self.BANDS))]


class DuoFoilAirFast(_DuoFoilUp):
    """TWO IDENTICAL SECTIONS, TWO FLOW RATES, ONE UPWARD TUNNEL. The set's base clip.

    This is `tri_foil_air_fast` with the format changed and the top lane removed, and that is the
    complete list. Same NACA 0015, same chord, same 9 s sweep, same air, same palette - two lanes
    at 0.75 and 0.55 of `u0`, so everything that differs between the left of the frame and the
    right of it is how fast that lane's air is, and nothing else.

    My brief listed this twice - once as `duo_foil_air_fast` (the comparison of two identical
    airfoils, with two lanes that differ in flow rate) and once as a
    rerender of `tri_foil_air_fast` in the new format with two foils. Those are the same clip - I
    confirmed it - and it is rendered once rather than shipped twice under two names.

    Nothing here needs its own physics argument that `_DuoFoilUp` has not already made: the class
    body is a title, because the format class carries the format and the parent carries the fluid.
    """
    title = "TWO FLOW RATES - AIR, UPRIGHT"


class DuoFoilAirSweep2(_DuoFoilUp):
    """`duo_foil_air_fast` with the sweep run at DOUBLE rate - two full loops in the same 10 s.

    The upright two-lane cut of `tri_foil_air_sweep2`, and `PERIOD` 9.0 -> 4.5 is the entire
    difference from `duo_foil_air_fast` - one class attribute, no `OVERRIDES` at all, so the
    fluid, the lattice, the palette and the placement are its sibling's bit for bit.

    The slew argument is `tri_foil_air_sweep2`'s and survives the format change untouched, because
    a lattice minds how far the MASK moves per LBM step and the mask does not know which way the
    screen is pointing: **0.0112 cells/step**, which is `tri_foil_air`'s own verified number,
    against `tri_foil`'s 0.043 and `Flutter`'s ~0.07. The parent bought that headroom by doubling
    `steps` to 110 and doubling the sweep rate spends exactly it back.

    `RAMP` stays 1.8 s, as it does on `tri_foil_air_sweep2`, and carries the same known cost: it
    is 40% of a cycle now rather than 20%, so the FIRST peak reaches ~23 deg where the second
    reaches the full 34. Left alone for the same reason - "the only change being the new format"
    argues for the minimal diff, and shortening the ramp is a one-number change if the attenuated
    opening is disliked.
    """
    title = "TWO FLOW RATES - AIR, DOUBLE SWEEP"

    PERIOD = 4.5                   # 9.0 -> 4.5: two full sweeps in the same 10 s clip


class DuoFoilAirShape(_DuoFoilUp):
    """TWO SECTIONS, ONE AIRSTREAM - the variable moved from the air to the BODY.

    My call: two airfoils on the bottom, same flow rate, different foil shapes. So this is the
    exact inverse of its two siblings, and that is what earns it a clip: they hold the body fixed
    and vary the stream, this holds the stream fixed and varies the body. Same chord, same
    station, same sweep, same air, one speed - and the only thing left different between the left
    lane and the right one is the section standing in it.

    **`BANDS` is flat at 1.25, which is the parent's TOP rate**, on my pick when asked. That
    is `tri_shapes` / `duo_shapes`' choice and it means `profile()` is a constant, `BLEND` has
    nothing to blend and there is no shear seam down the middle of this one - by construction,
    not by luck. It also means this clip's freestream sits at a different place on the ramp
    (0.481) than its two siblings' lanes do (0.288 and 0.212), which is a deliberate, stated
    departure from "keep the colour scale of the middle and bottom lanes": with one speed in both
    lanes there is no middle lane to keep.

    **`vmax` 2.60 -> 4.30, AND IT IS THE THIRD DIFFERENT ANSWER TO THE SAME OLD NOTE.** At the
    parent's 2.60 this clip's freestream sits at s = 0.481, which is `jet`'s single most luminous
    region (`colormap._tone`), and with one speed in both lanes it covers most of the frame -
    exactly what produced *"the green color is very bright"* on `tri_shapes`. Measured on the
    first cut: mean frame luminance **200-220 against its two siblings' 130-138** and
    `tri_foil_air_fast`'s 157, with 0.16% of pixels clipped flat at the top of the ramp at peak
    incidence (the parent: 0.0023%).

    The two answers this project already had were both unavailable. `tri_shapes` used
    `brightness` 0.72 and I rejected the dimming outright - it wasn't acceptable that
    `tri_shapes`' brightness had been turned down; `duo_shapes` left `jet` for `blues`, which would
    put this clip in a different palette from the two it ships beside. **`vmax` is the third
    lever and the right one here, because it moves the FREESTREAM down the ramp instead of
    dimming the ramp**: 1.25 / 4.30 = **0.291**, which is within 0.003 of where the rate clips'
    fast lane sits (0.75 / 2.60 = 0.288). So all three clips read as one blue set, the vortex
    cores keep the top of the map, and the clipping goes to zero. Chosen on a saturation-preview
    sheet (not included here) - three candidate `vmax` values re-mapped onto an ALREADY-RENDERED
    frame, no second simulation.

    ⚠️ **STATE THE COST RATHER THAN BURYING IT: the ramp no longer means the same speed in this
    clip as in its two siblings.** A given colour is 1.67x faster air here. Nothing on screen
    claims otherwise - no clip in this set carries a legend - and within this clip the comparison
    is untouched, since both lanes share one ramp and one speed. But the three files are not a
    common colour SCALE any more, only a common colour FAMILY, and a viewer told they were would
    be told something false. `vmax=2.60` restores the strict-inheritance version.

    THE TWO SECTIONS, and why this pair:

      - **NACA 0015, symmetric** (left). No camber, so at 0 deg it makes no lift and sheds
        nothing: every feature in that lane is bought by incidence alone and the picture is exactly
        mirror-symmetric between the two halves of each sweep. The control, and the section all
        three clips in this set otherwise use.
      - **NACA 2412, cambered** (right). The section `aoa_sweep` runs and the one on half the
        light aircraft ever built. What it buys against the symmetric section at the same angle in
        the same air is CAMBER: a suction peak at zero incidence, two halves of the cycle that are
        no longer mirror images, and a different stall angle on each side.

    ⚠️ **STRICTLY, THE PAIR DIFFERS IN TWO THINGS, AND THE CLIP SHOULD NOT CLAIM OTHERWISE.**
    2412 is also **12% thick against the 0015's 15%**, so the right lane is 3 points thinner as
    well as cambered. This is `tri_shapes`' pair, chosen for continuity with a clip I passed
    with no notes, and camber is by far the larger effect at these angles - but a frame that
    differs in two properties cannot attribute what it shows to one of them, and 0012 against 2412
    would have been the clean single-variable comparison at the cost of dropping the 0015 the
    other two clips in this set use as their control.

    Thickness alone (0008 against 0018, `mach_foils`' variable) was the other candidate and is the
    weaker comparison at these Reynolds numbers - it selects critical Mach, which an incompressible
    lattice at M 0.119 has nothing to say about.

    **NO `flip_y`, and both sections are built identically.** `shapes.flip_y` exists because which
    way up a profile appears is a property of the FORMAT: lattice +y is screen-DOWN in the
    horizontal format, so a profile authored camber-up arrives inverted there and `tri_shapes`
    mirrors it. In THIS format lattice +y is screen-RIGHT, so the raw profile is already in
    convention and the camber faces screen-right. Mirroring one lane and not the other would have
    flipped the sign of its lift and confounded the one comparison the clip is making, so both are
    constructed the same way and the section is genuinely the only difference.

    STABILITY IS ITS OWN QUESTION, not an inheritance. Both lanes run 1.25 where its siblings run
    0.75 and 0.55, so this clip's blockage (25.1%, `_DuoFoilUp` item 4) multiplies the fastest
    band in the family. `tri_shapes` measured 0.2504 at three lanes of 1.25 in the horizontal
    format; `dbg.py` re-answers it here rather than that number being carried across a format
    change.
    """
    # `vmax` is the ONLY entry here, and it is a normalisation rather than a fluid parameter - the
    # solve is bit-for-bit `_DuoFoilUp`'s, and `duo_check.py` still checks nu/tau against the
    # parent and still passes. See the docstring for why this clip alone needs it.
    OVERRIDES = dict(_DuoFoilUp.OVERRIDES, vmax=4.30)
    title = "TWO SHAPES - ONE AIRSTREAM, UPRIGHT"

    BANDS = (1.25, 1.25)           # one speed in both lanes: the parent clip's TOP rate. Flat, so
                                   # `profile()` is a constant and there is no band seam at all
    CODES = ("0015", "2412")       # symmetric | cambered - the section is the only variable

    def __init__(self, cfg):
        super().__init__(cfg)
        # Raw `naca4`, no `flip_y`: lattice +y is screen-RIGHT in this format, so a profile
        # authored camber-up is already in convention (see `shapes.flip_y`). Both lanes get the
        # same treatment, which is what keeps the section the only difference between them.
        self.profs = [shapes.naca4(c) for c in self.CODES]
        self.prof = self.profs[0]

    def bodies(self, t):
        return [shapes.place(self.profs[k], self.chord, self.sx(self.CX[k]), self.sy(fy),
                             self.aoa(k, t))
                for k, fy in enumerate(self.LANES)]

    def readout(self, t):
        return [("AoA", f"{self.aoa(0, t):+.0f} deg (both lanes)"),
                ("SECTION", " / ".join(f"NACA {c}" for c in self.CODES)),
                ("Re", f"{self.BANDS[0] * self.cfg.re:.0f} (both lanes)")]


SCENES = {
    "duo_foil_air_fast": DuoFoilAirFast,
    "duo_foil_air_sweep2": DuoFoilAirSweep2,
    "duo_foil_air_shape": DuoFoilAirShape,
    "surf_rates": SurfRates,
    "surf_sweep": SurfSweep,
    "surf_wave": SurfWave,
    "aoa_sweep": AoASweep,
    "aoa_sweep_fast": AoASweepFast,
    "tri_foil": TriFoil,
    "tri_foil_fast": TriFoilFast,
    "tri_foil_rates": TriFoilRates,
    "tri_foil_rates_fast": TriFoilRatesFast,
    "tri_foil_rates_loop": TriFoilRatesLoop,
    "jet_engine": JetEngine,
    "tri_foil_air": TriFoilAir,
    "tri_foil_air_fast": TriFoilAirFast,
    # the 2026-08-10 set - four variations on tri_foil_air_fast
    "tri_foil_air_sweep2": TriFoilAirSweep2,
    "tri_cylinders": TriCylinders,
    "tri_shapes": TriShapes,
    "tri_foil_gas": TriFoilGas,
    "mach_sweep": MachSweep,
    "mach_rates": MachRates,
    "mach_foils": MachFoils,
    "mach_shapes": MachShapes,
    "currents": Currents,
    "still_water": StillWater,
    "black_holes": BlackHoles,
    "black_holes_red": BlackHolesRed,
    "falling_discs": FallingDiscs,
    "falling_squares": FallingSquares,
    "projectiles": Projectiles,
    "eddies": Eddies,
    "stall": Stall,
    "flutter": Flutter,
    "vortex_street": VortexStreet,
    "tandem": Tandem,
    "shape_tour": ShapeTour,
}
