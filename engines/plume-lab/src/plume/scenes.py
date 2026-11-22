# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""scenes.py - the registry. One function per clip, returning a Cfg.

The five here are a DIAGNOSTIC SPREAD, not a shipping set: each one deliberately pushes a
different part of the engine, so a failure tells you which part.

| scene            | what it stresses                                                          |
|------------------|---------------------------------------------------------------------------|
| `plume_ref`      | the baseline look match against the reference clip's measured numbers      |
| `plume_under`    | a strong barrel shock + Mach disk - the carbuncle and positivity risk      |
| `plume_over`     | overexpanded: lip oblique shock instead of a lip expansion, inverted cells |
| `plume_vacuum`   | ambient / 20 - extreme density ratio, far-field sponge, fast box growth    |
| `plume_schlieren`| the alternate field + tone map, and the waves the velocity field hides     |

`t_end` is in cells / ambient sound speed. It is what decides how far the front gets, and it is
the number to change first when `dry` reports the front covering under a third of the panel.
"""
from __future__ import annotations

from .config import Cfg

REGISTRY = {}


def scene(fn):
    REGISTRY[fn.__name__] = fn
    return fn


# =============================================================================================
# THE TWO VERTICAL TEMPLATES. Every new clip starts from one of these.
# =============================================================================================
#
# CLEAN-LOOK DEFAULTS, and each one is a measured fix rather than a taste:
#
#   `lut_clip_hi = 0.85` - **the flicker fix.** Stops the ramp before its dark-red top. `plumefire` ends at #6E1C02
#       because the SOURCE does, which makes the FASTEST gas the DARKEST; a shock-cell train then
#       drives the core back and forth across that peak. Measured on the first shipped clip:
#       **37.8% of jet-core pixels were darker than luma 110**. Clipping ends the ramp at bright
#       orange, so within the range the jet occupies, faster now always means brighter.
#
#       Sweeping clip_hi alone on one captured field (core RGB delta per frame / chromaticity
#       delta / % of core darker than luma 110):
#           1.00 -> 32.5 / 20.9 / 3.0%      0.88 -> 22.7 / 14.8 / 0.0%
#           0.92 -> 27.0 / 17.5 / 0.0%      0.84 -> 17.9 / 11.7 / 0.0%
#
#   `norm = "auto"`, `vmax_mult = 1.12` - the range is DERIVED from the jet speed, not from the
#       frame's own peak, so a given physical speed is the same colour in every frame and in
#       every clip. Measured on the vertical template: the core sits at **1.004 x u_jet** (p90
#       1.085) while the field PEAK is **1.316 x u_jet** - that gap is the exit-expansion
#       overshoot, and pinning the range to the peak is what put the first vertical still's core
#       in the greens when the source's core is orange. Two failed settings, both measured:
#           vmax 1.02, clip 0.92 -> core ramp 0.91 but **42.7% of the core SATURATED** (flat
#               orange, shock cells erased);
#           vmax 1.32, clip 0.95 -> 0% saturated but core ramp 0.82, i.e. yellow-green again.
#       1.12 / 0.94 / gamma 0.80 puts the core p50 at ramp 0.86 (yellow) and p90 at 0.92
#       (orange), with ~1% saturated and 0% dark - a yellow core with orange in the fast cells,
#       which is what the source shows.
#
#   `norm = "stable"` is NOT the flicker fix and an earlier note here said it was: at fixed
#       clip_hi it is worth ~4% against plain minmax, and a first attempt at q=0.995/alpha=0.10
#       was WORSE than minmax (40.4 vs 34.8).
#
#   `lut_gamma = 0.90`  - slightly less lift than the horizontal clips used, because clipping the
#       top already brightens the range the jet lives in.

VERT_LOOK = dict(lut_clip_hi=0.94, norm="auto", vmax_mult=1.12, lut_gamma=0.80)


def _vertical(name, **kw):
    """SINGLE DOWNWARD STREAM - 1080x1920, nozzle at the TOP, flow DOWN. The default template.

    The solver is untouched: it always works in (z, r) with z along the stream, and `render.py`
    transposes for the vertical framings. Gravity is not modelled and would be invisible if it
    were - a Mach-2 plume crosses this frame in milliseconds, during which g moves it by microns.

    `scale = 1.5` is the cost lever. At 1:1 a 1080x1920 frame is a 1800x540 half-domain, ~1.0M
    cells and ~3.3x the horizontal panel; at 1.5 it is 1200x360 = 432k, and the exit radius is
    still ~34 cells - comfortably past the ~20 the Mach disk needs to read as a disk.
    """
    cfg = dict(frame="vertical", width=1080, height=1920, seconds=8.0, scale=1.5,
               nozzle_px=120, r_exit=51.0, **VERT_LOOK)
    cfg.update(kw)
    return Cfg(name=name, **cfg)


@scene
def plume_down():
    """THE SINGLE-STREAM TEMPLATE. Start every new vertical clip from this.

    Four things turn it into a clip: `mach_e` / `pr_e` / `tr_e` (the gas), and `t_end` (the
    pacing - set so the front reaches the bottom at ~67% of the clip, which is where the source
    puts it, leaving the last third as the steady shock-cell train).
    """
    return _vertical("plume_down", mach_e=1.80, pr_e=1.30, tr_e=2.00, t_end=2050.0)


# =============================================================================================
# THE OBJECT TEMPLATE - a PNG with an invisible nozzle under it
# =============================================================================================

# 1920 = 408 head + 72 cone + 1440 of solved stream, i.e. the exit plane at EXACTLY 25% and
# 75% of the frame left under it. Every one of those is a multiple of `scale` = 1.5, so the
# drawn exit line and the cone's last cell land on the same pixel instead of half a pixel apart.
OBJ_HEAD_PX = 408
OBJ_NOZZLE_PX = 72

# BLACK, not the source's violet. `plumenoir` only moves the two lowest control points of the
# measured ramp, so every colour the jet occupies is untouched and VERT_LOOK's solved numbers
# (core at ramp 0.86 / 0.92) still hold - see colormap.PLUMENOIR.
OBJ_LOOK = dict(VERT_LOOK, lut="plumenoir")


def _object(name, **kw):
    """THE OBJECT TEMPLATE - 1080x1920, a cut-out PNG at the top, a jet fired out of it.

    THE NOZZLE IS OCCLUDED, NOT REMOVED. The painted cone and the solved Dirichlet patch are
    exactly what `plume_down` has; the object is composited over the finished frame and hides
    them. That matters because the cone is what keeps the drawing honest - it is generated from
    the SAME exit state the solver is given, and `plume_check` test A gates that agreement at
    0.124%. An "invisible nozzle" built by deleting the cone would be a different, unchecked
    thing. `Scene.cover_report()` MEASURES the occlusion (worst alpha over the cone's footprint)
    instead of assuming it, and `dry` and `stills` both print it.

    THREE THINGS THIS RIG ADDS TO `plume_down`, and each is one config field:

      `head_px`     undisturbed ambient ABOVE the cone, so the exit plane can sit anywhere down
                    the frame without lengthening the drawn nozzle. The solve loses exactly
                    these cells (`Cfg.nz` subtracts them), so the panel length never changes.
      `obj_png`     the subject, cropped to its own alpha, rotated, and fitted to `obj_h_px`.
      `obj_exit_frac`  where the exit plane falls on the object's OWN height. This is the knob
                    that decides how much solid body sits over the cone AND how far the jet
                    travels hidden before it clears the silhouette - which is what makes the
                    exhaust look like it leaves the object rather than starting at a line.

    A NEW PAIRING IS FOUR EDITS AND NO NEW CODE:

      1. drop a background-removed PNG somewhere readable and set `obj_png` (relative paths
         resolve from the Plume_Lab root, so `..\\Wind_Tunnel\\assets\\...` works);
      2. set `obj_rotate` / `obj_h_px` / `obj_exit_frac`, then run `cli.py dry <scene>` and read
         the `cone cover` line - it must say HIDDEN. If it does not, raise `obj_h_px` or move
         `obj_exit_frac` deeper into the body; `r_exit` is the third lever and the cheapest,
         since 40 px is 26.7 cells at this scale against the ~20 a Mach disk needs to read;
      3. pick the gas (`mach_e` / `pr_e` / `tr_e`);
      4. set `t_end` from what `dry` reports, and NOTE the target here is not the source's.
    """
    cfg = dict(frame="vertical", width=1080, height=1920, seconds=10.0, scale=1.5,
               head_px=OBJ_HEAD_PX, nozzle_px=OBJ_NOZZLE_PX, r_exit=40.0, **OBJ_LOOK)
    cfg.update(kw)
    return Cfg(name=name, **cfg)


@scene
def plume_object():
    """THE OBJECT TEMPLATE, EMPTY - the rig with no subject loaded. Start new pairings here.

    Same rig as `cow_rocket` with `obj_png` unset, so the painted cone is on show and you can
    see what an object has to cover. Renders the black-floor look and the 25% exit plane and
    nothing else. Same purpose as Wind_Tunnel's `road_template`: inspect the rig before you
    put anything in it.
    """
    return _object("plume_object", mach_e=2.40, pr_e=2.00, tr_e=3.50,
                   t_ramp=0.15, t_end=1350.0)


@scene
def plume_triple():
    """THE THREE-STREAM TEMPLATE - three INDEPENDENT plumes, two divider bars, one clock.

    The three panels are separate `AxiEuler` instances with their own arrays, ghosts and
    absorbers, so they cannot interact and the bars are pure composition. Contrast Wind_Tunnel's
    road set, where the divider had to be a real bounce-back plate because its two halves were
    regions of ONE lattice. The layout is still solved (`3*D + 2*B = 1920` -> D = 632, B = 12)
    for the road set's other reason: unequal ducts would confound the comparison.

    Each panel is 1080x632 - wide and short - so the stream runs ACROSS it, as in the source's
    own framing. The default triple is a PRESSURE-RATIO ladder, which is the comparison this
    format is actually for: the same nozzle at three back pressures, so overexpanded /
    near-perfect / underexpanded read side by side and the shock-cell structure inverts down the
    frame. Swap `variants` for any other three-way comparison.

    PACING NOTE: `dry` reports arrival PER PANEL here, and the single-stream 55-80% band does not
    apply. Three pressure ratios mean three jet speeds, so the panels arrive at different times
    by construction - `t_end` 1450 puts the fastest (underexpanded) at ~55%, which leaves the
    slower two the rest of the clip to develop. Tuning t_end to put the fastest at 67% would mean
    the slowest never arrives.
    """
    return Cfg(name="plume_triple", frame="triple", width=1080, height=1920, seconds=8.0,
               scale=1.0, nozzle_px=60, r_exit=28.5,
               mach_e=2.20, tr_e=2.20, t_end=1450.0, **VERT_LOOK,
               variants=(
                   dict(pr_e=0.45, label="overexpanded"),
                   dict(pr_e=1.00, label="perfectly expanded"),
                   dict(pr_e=6.00, label="underexpanded"),
               ))


def _triple(name, variants, t_end, **kw):
    """A three-stream comparison on the standard 1080x1920 ladder rig.

    ONE COLOUR RANGE ACROSS THE THREE PANELS (`share_range`, on by default). This is the whole
    reason a ladder is readable: `norm="auto"` pins each panel to ITS OWN jet speed, which is
    right for a single clip and wrong for a comparison - a panel twice as fast as its neighbour
    would be scaled by twice the number and the two would render identically. The range is taken
    from the FASTEST panel, so a colour means one speed down the whole frame.
    """
    cfg = dict(frame="triple", width=1080, height=1920, seconds=8.0, scale=1.0,
               nozzle_px=60, r_exit=28.5, t_end=t_end, variants=variants, **VERT_LOOK)
    cfg.update(kw)
    return Cfg(name=name, **cfg)


@scene
def triple_mach():
    """EXIT MACH LADDER - M_e 1.6 / 2.4 / 3.2, same gas, same pressure ratio.

    What it shows: **shock-cell spacing grows with Mach number**, which is the one relation this
    engine is gated against (Pack, `L = 1.306 D sqrt(M^2 - 1)`). Top panel packs cells tightly;
    the bottom stretches them out. The plume also narrows and lengthens as M_e rises, because a
    faster jet spreads through the same angle over a longer distance.

    Physically this is a NOZZLE EXPANSION-RATIO ladder: A_e/A* of 1.25 / 2.40 / 5.12 for the three
    exit Machs, i.e. the same chamber with three bells on it.
    """
    return _triple("triple_mach", (
        dict(mach_e=1.60, label="M_e 1.6"),
        dict(mach_e=2.40, label="M_e 2.4"),
        dict(mach_e=3.20, label="M_e 3.2"),
    ), t_end=1250.0, pr_e=1.60, tr_e=2.20)


@scene
def triple_temp():
    """EXHAUST TEMPERATURE LADDER - T_e/T_inf 1 / 3 / 6, same nozzle, same pressure ratio.

    What combustion actually buys you. Exit velocity is `M_e * sqrt(T_e/T_inf)` in these units,
    so temperature - not pressure, not nozzle shape - is the lever on exhaust speed, which is the
    lever on specific impulse. Predicted fully-expanded jet speeds **2.58 / 4.47 / 6.32 c_inf**,
    a 2.4x spread; under the shared colour range the cold panel reads blue and the hot one orange,
    which IS the measurement.

      1  cold gas    - a nitrogen or argon attitude thruster, no combustion at all
      3  warm        - a monopropellant / resistojet
      6  combustion  - a real bipropellant chamber

    ⚠️ This slot originally held a **gamma** ladder (1.67 / 1.40 / 1.20). It was cut after the
    smoke test measured its three jet speeds at **3.95 / 4.00 / 4.05** - a 2.5% spread, i.e.
    three visually identical panels. Gamma changes the plume's shape and cell structure, but at
    any pressure ratio this format can afford it barely touches the speed the eye reads. A
    comparison whose variable does not move the picture is not a comparison.
    """
    return _triple("triple_temp", (
        dict(tr_e=1.0, label="cold gas  T_e/T_inf 1"),
        dict(tr_e=3.0, label="warm  T_e/T_inf 3"),
        dict(tr_e=6.0, label="combustion  T_e/T_inf 6"),
    ), t_end=1150.0, mach_e=2.40, pr_e=2.00)


@scene
def triple_altitude():
    """THE ALTITUDE SWEEP - one fixed engine, three ambient pressures. Sea level to vacuum.

    The most recognisable thing a rocket nozzle does, and the reason engines are altitude-
    compensated at all. `p_e/p_inf` is the SAME nozzle seen at three back pressures:

      0.30  over-expanded  - sea level. The lip turns the flow INWARD; an oblique shock springs
            off it and converges on the axis. This is the regime where a real first stage runs,
            and where flow separation inside the bell becomes a design problem.
      1.00  perfectly expanded - the design altitude. No lip wave at all; the shear layer leaves
            the lip straight and the cell train is at its weakest.
      18.0  highly under-expanded - near vacuum. A huge barrel shock and a flat Mach disk.

    Note this is the same VARIABLE as `plume_triple`'s default ladder but a far wider span
    (0.30-18 against 0.45-6), chosen so the three regimes are unmistakable rather than adjacent.

    Pacing: the three panels differ by 1.43x in jet speed (3.50 / 4.19 / 4.99), so they cannot
    all arrive at 67%. `t_end` is set by the SLOWEST - at 1150 the sea-level panel reached only
    84% of its duct and read as cut off, so it is raised until that one just lands, which puts
    the vacuum panel's arrival near 38%. For a ladder the slowest panel sets the clock; the fast
    ones simply get longer to develop.
    """
    return _triple("triple_altitude", (
        dict(pr_e=0.30, label="sea level  over-expanded"),
        dict(pr_e=1.00, label="design altitude"),
        dict(pr_e=18.0, label="vacuum  under-expanded"),
    ), t_end=1360.0, mach_e=2.60, tr_e=2.60, sponge_r=40)


@scene
def triple_field():
    """SAME FLOW, THREE INSTRUMENTS - speed, Mach number, schlieren. ONE solve.

    The only variant here that is not about the physics: all three panels are the SAME plume,
    rendered as three different measurements of it. It answers a question the other ladders
    cannot - which structures are an artefact of what you chose to plot?

      speed      - the shipped look. Shock cells read as a smooth bright/dark train.
      mach       - speed over the LOCAL sound speed, so it divides out the temperature drop
                   through the expansion; the sonic line lands in a fixed place on the ramp.
      schlieren  - |grad rho|/rho, blind to velocity. Draws the shocks and the starting blast as
                   LINES, including waves the speed field renders as a smooth ramp.

    `share_sim=True`: one `AxiEuler` feeds all three panels, so this costs ONE solve instead of
    three. `Scene` refuses the flag if any variant touches a physics parameter, because a shared
    solve that claimed to show three different gases would be a silent lie.

    `share_range` does nothing across these three - a speed, a Mach number and a density gradient
    have no common scale, so the grouping is per-field and each is alone in its group.
    """
    return _triple("triple_field", (
        dict(field="speed", label="velocity"),
        dict(field="mach", label="Mach number"),
        dict(field="schlieren", lut="graphite", label="schlieren"),
    ), t_end=1450.0, mach_e=2.20, pr_e=2.50, tr_e=2.20, share_sim=True, axis_line=False)


# =============================================================================================
# THE DIAGNOSTIC SET - the same five, on the single-stream vertical template.
# =============================================================================================
@scene
def plume_ref():
    """The reference recreation.

    Every number traces to a measurement of ``Video_Interpreter\\rocket_pressure.mp4``:

      - panel 1080x604 is its 720x402 (aspect 1.7910 vs 1.7881) at library width;
      - `nozzle_px` 60 and `r_exit` 28.5 are its exit plane at x=40 px and R_e=19 px, scaled;
      - `mach_e` / `pr_e` / `tr_e` are chosen so the FULLY EXPANDED jet Mach lands near 1.97,
        because the measured shock-cell spacing is 75-85 px against D_e = 38 px, i.e.
        L/D_e ~ 2.0-2.3, and Pack's relation L = 1.306 D sqrt(M^2 - 1) inverts that to M ~ 1.9.
        NOT real kerolox (M_e ~ 3.5-4, T_e/T_inf ~ 5): that would put u_e near 8 c_inf, cost ~2x
        the steps, and change nothing anyone can see in a 3 s clip.

    Predicted fully-expanded state: u_jet 2.68 c_inf, M_jet 1.97, L_cell/D_j 2.22.
    `t_end` is set so the front reaches the right edge at ~68% of the clip, which is where the
    reference puts it (frame 148 of 221) - the last third is the steady shock-cell train, and
    that train is the most recognisable thing in the whole look.

    ⚠️ The first cut got this wrong by a factor of 1.5 and `look_check` caught it. `dry` reports
    "front reached 98% of the PANEL", which is a distance; the target is a fraction of the CLIP.
    Reading one as the other put the front at the right edge on the LAST FRAME, so the steady
    train never appeared - and both of look_check's failures (front rate, cell spacing) were
    downstream of that single mistake.

    `lut_gamma` 0.85 bends the VALUE axis only. It exists because this solve is finer than the
    source's (~616k cells full-domain equivalent against its stated 289,380) and therefore
    carries a THINNER shear layer; the source's broad blue halo is partly its own numerical
    diffusion. Lifting the low end reads it back without moving a single measured colour.
    """
    return _vertical("plume_ref", mach_e=1.80, pr_e=1.30, tr_e=2.00, t_end=2120.0)


@scene
def plume_under():
    """Highly underexpanded: p_e = 6 p_inf. The textbook barrel shock and Mach disk.

    This is the scene most likely to find a numerical problem, and that is the point. A strong
    normal shock sitting square on the grid is where the CARBUNCLE instability lives, and the
    exit expansion fan here is the strongest rarefaction the engine will ever run - i.e. the one
    that finds out whether the positivity floor on the reconstructed state is doing its job.
    """
    return _vertical("plume_under", mach_e=2.20, pr_e=6.00, tr_e=2.40, t_end=1400.0)


@scene
def plume_over():
    """Overexpanded: p_e = 0.45 p_inf. The lip turns the flow INWARD.

    Topologically the opposite of `plume_under` - an oblique shock springs from the lip and
    converges on the axis instead of an expansion fan opening away from it, so the first cell is
    a compression. Worth having because it is the regime a real engine runs in at sea level, and
    because it exercises the base-plate slip wall much harder (the recirculation behind the lip
    is real here rather than a detail).
    """
    return _vertical("plume_over", mach_e=2.60, pr_e=0.45, tr_e=2.20, t_end=2410.0)


@scene
def plume_vacuum():
    """Near-vacuum ambient. An in-space burn: the plume balloons instead of forming cells.

    Implemented as a very large exit/ambient pressure ratio rather than by changing p_inf, since
    p_inf = 1/gamma is what fixes c_inf = 1 and therefore what makes a velocity a Mach number.
    Stresses the far-field sponge (the plume reaches the radial boundary early), the density
    floor (rho drops by orders of magnitude), and the active box, which has to grow fast.
    """
    return _vertical("plume_vacuum", mach_e=2.80, pr_e=22.0, tr_e=3.00, t_end=805.0,
                     sponge_r=40, active_start=256)


@scene
def plume_schlieren():
    """`plume_ref`'s flow, rendered as |grad rho|/rho on a near-black stage.

    Same solve, different question: schlieren is blind to velocity and sensitive only to density
    gradient, so it draws the acoustic waves, the starting blast and the cell shocks as lines -
    including the ones the speed field renders as a smooth ramp. If the two clips disagree about
    where a shock is, one of them is lying.
    """
    return _vertical("plume_schlieren", mach_e=1.80, pr_e=1.30, tr_e=2.00, t_end=2120.0,
                     field="schlieren", lut="graphite", axis_line=False, lut_clip_hi=1.0)


# --------------------------------------------------------------------------------------------
def build(name, **over):
    if name not in REGISTRY:
        raise SystemExit(f"unknown scene {name!r}; have: {', '.join(sorted(REGISTRY))}")
    cfg = REGISTRY[name]()
    for k, v in over.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown override {k!r}")
        setattr(cfg, k, v)
    return cfg


# The 16:9 stock catalogue (20 scenes: 10 single-stream, 10 three-stream). Imported at the
# BOTTOM because scenes_h reuses `scene`, `VERT_LOOK` and the Cfg factories from this module -
# importing it at the top would be circular. Without this line the @scene decorators in that
# file never run and none of its scenes exist, which is exactly how the first launch attempt
# found an empty registry.
from . import scenes_h as _scenes_h            # noqa: E402,F401

# The 30-clip PROPULSION & PLUME theme (2026-08-29 stock batch), at 3840x2160 and 30 s.
# Imported here for the same reason as `scenes_h`: its @scene decorators have to RUN or the
# registry simply does not contain them, and the failure is an empty listing rather than an
# error. It imports the horizontal templates, so it must come after that line.
from . import scenes_stock as _scenes_stock   # noqa: E402,F401
