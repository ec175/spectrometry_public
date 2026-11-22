# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""scenes_h.py - the HORIZONTAL (1920x1080) stock catalogue. 20 clips, 15 s each.

Brief (Clancy 2026-08-16): 20 clips at 15 s - **10 single-stream** and **10 three-stream
parameter comparisons** - every one of them **nozzle LEFT, stream pointing RIGHT**, and the
background **BLACK, not purple**. No text.

Three things this module does NOT have to invent, because the engine already had them:

  * **Left-to-right is the native orientation.** `frame="panel"` is the source clip's own shape,
    one horizontal stream flowing left -> right, and `frame="triple"` stacks three of those
    (`nz` runs along `width` for both). The vertical framings are the special case, not this.
  * **Black is a LUT swap.** The purple is `plumefire`'s `#2E153D` ambient floor. `plumenoir` is
    the SAME measured ramp with a `#000000` floor, rejoining the measurement at 0.150, so the
    plume core is bit-for-bit unchanged and only the undisturbed field stops glowing. This is
    why the cow clips and the schlieren clips already look black.
  * **No text.** Nothing in this package ever draws a glyph - no ImageDraw, no font, anywhere.
    The `label=` on a variant is CONSOLE-ONLY: the dry report prints it to say which panel is
    which. It cannot reach a pixel, so the labels are kept for the gate's benefit.

WHY 15 s IS NOT A RETUNE. `t_end` is a PHYSICS pacing parameter: the solve runs t in [0, t_end]
across the clip's frames, so a front arriving at physical time T shows up at fraction T/t_end -
which does not contain `seconds` at all. Doubling the clip length plays the same evolution over
twice as many frames (slower, smoother), it does not truncate it. The inherited `t_end` values
therefore carry over, and the dry gate's "ARRIVES at X% of the clip" line is what confirms it.
"""
from __future__ import annotations

from .config import Cfg
from .scenes import VERT_LOOK, scene

# The stock look: the measured ramp on a true black floor.
STOCK_LOOK = dict(VERT_LOOK, lut="plumenoir")

W, H = 1920, 1080


def _panel_h(name, **kw):
    """ONE horizontal stream - nozzle LEFT, flow RIGHT. The single-stream template.

    `scale = 1.5` is inherited from the vertical template and lands in the same place: the
    half-domain is ~1200 x 360 = 432k cells, and the exit radius is 51/1.5 = 34 cells, well past
    the ~20 the Mach disk needs to read as a disk rather than a smudge.
    """
    cfg = dict(frame="panel", width=W, height=H, seconds=15.0, scale=1.5,
               nozzle_px=120, r_exit=51.0, **STOCK_LOOK)
    cfg.update(kw)
    return Cfg(name=name, **cfg)


def _triple_h(name, variants, t_end, **kw):
    """THREE horizontal streams stacked, each nozzle LEFT / flow RIGHT, two divider bars.

    The layout is SOLVED, not placed - `3D + 2B = height` - so at 1080 with 12 px bars each panel
    is 352 px. That is a shorter panel than the 632 px the 9:16 ladder gets, so the plume has
    less room to spread before it meets the panel edge; `r_exit` is pulled in to 24 to keep a
    comparable spread ratio while staying clear of the Mach-disk resolution floor. The dry gate's
    fits/fill numbers are what decide whether that is enough - not this comment.

    ONE COLOUR RANGE ACROSS THE THREE (`share_range`, inherited on): `norm="auto"` pins a panel
    to its OWN jet speed, which is right for one clip and wrong for a ladder - two panels at
    different speeds would otherwise render identically.
    """
    cfg = dict(frame="triple", width=W, height=H, seconds=15.0, scale=1.0,
               nozzle_px=60, r_exit=24.0, t_end=t_end, variants=variants, **STOCK_LOOK)
    cfg.update(kw)
    return Cfg(name=name, **cfg)


# =============================================================================================
# TEN SINGLE STREAMS - one parameter each
# =============================================================================================

@scene
def pl_h_design():
    """Perfectly expanded: p_e/p_inf = 1. The clean shock-cell train, no barrel shock."""
    return _panel_h("pl_h_design", mach_e=2.40, pr_e=1.00, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_over():
    """Overexpanded: p_e < p_inf, so the ambient squeezes the jet and an oblique shock forms."""
    return _panel_h("pl_h_over", mach_e=2.40, pr_e=0.45, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_under():
    """Underexpanded: p_e > p_inf. The barrel shock and Mach disk this engine exists to show."""
    return _panel_h("pl_h_under", mach_e=2.40, pr_e=6.00, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_vacuum():
    """Vacuum-like: p_e/p_inf = 18. The plume balloons straight out of the nozzle."""
    return _panel_h("pl_h_vacuum", mach_e=2.40, pr_e=18.0, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_mach_low():
    """Low exit Mach 1.6 - tightly packed shock cells (Pack: L ~ sqrt(M^2 - 1))."""
    return _panel_h("pl_h_mach_low", mach_e=1.60, pr_e=1.60, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_mach_high():
    """High exit Mach 3.2 - the cells stretch out and the plume narrows and lengthens."""
    return _panel_h("pl_h_mach_high", mach_e=3.20, pr_e=1.60, tr_e=2.20, t_end=2050.0)


@scene
def pl_h_cold():
    """Cold gas, T_e/T_inf = 1. A cold-gas thruster rather than a combustion product."""
    return _panel_h("pl_h_cold", mach_e=2.40, pr_e=1.30, tr_e=1.00, t_end=2050.0)


@scene
def pl_h_hot():
    """Hot exhaust, T_e/T_inf = 6 - combustion temperatures, a faster and wider plume."""
    return _panel_h("pl_h_hot", mach_e=2.40, pr_e=1.30, tr_e=6.00, t_end=2050.0)


@scene
def pl_h_schlieren():
    """The same solve read as a SCHLIEREN image - density gradient, the wind-tunnel view."""
    return _panel_h("pl_h_schlieren", mach_e=2.40, pr_e=1.60, tr_e=2.20, t_end=2050.0,
                    field="schlieren", lut="graphite", lut_clip_hi=1.0, lut_gamma=1.0)


@scene
def pl_h_machfield():
    """The same solve coloured by MACH NUMBER instead of speed - the sonic line is legible."""
    return _panel_h("pl_h_machfield", mach_e=2.40, pr_e=1.60, tr_e=2.20, t_end=2050.0,
                    field="mach")


# =============================================================================================
# TEN THREE-STREAM COMPARISONS
# =============================================================================================

@scene
def pl_h_tri_mach():
    """EXIT MACH LADDER 1.6 / 2.4 / 3.2 - shock-cell spacing grows with Mach number."""
    return _triple_h("pl_h_tri_mach", (
        dict(mach_e=1.60, label="M_e 1.6"),
        dict(mach_e=2.40, label="M_e 2.4"),
        dict(mach_e=3.20, label="M_e 3.2"),
    ), t_end=1250.0, pr_e=1.60, tr_e=2.20)


@scene
def pl_h_tri_pressure():
    """PRESSURE-RATIO LADDER - overexpanded / perfectly expanded / underexpanded."""
    return _triple_h("pl_h_tri_pressure", (
        dict(pr_e=0.45, label="overexpanded"),
        dict(pr_e=1.00, label="perfectly expanded"),
        dict(pr_e=6.00, label="underexpanded"),
    ), t_end=1250.0, mach_e=2.40, tr_e=2.20)


@scene
def pl_h_tri_temp():
    """EXHAUST TEMPERATURE LADDER - T_e/T_inf 1 / 3 / 6, same nozzle and pressure."""
    return _triple_h("pl_h_tri_temp", (
        dict(tr_e=1.0, label="cold gas"),
        dict(tr_e=3.0, label="warm"),
        dict(tr_e=6.0, label="combustion"),
    ), t_end=1250.0, mach_e=2.40, pr_e=1.30)


@scene
def pl_h_tri_altitude():
    """ALTITUDE LADDER - the same engine at sea level, design altitude and vacuum."""
    return _triple_h("pl_h_tri_altitude", (
        dict(pr_e=0.30, label="sea level"),
        dict(pr_e=1.00, label="design altitude"),
        dict(pr_e=18.0, label="vacuum"),
    ), t_end=1250.0, mach_e=2.40, tr_e=2.20)


@scene
def pl_h_tri_field():
    """READOUT LADDER - one solve, three fields: velocity, Mach number, schlieren."""
    return _triple_h("pl_h_tri_field", (
        dict(field="speed", label="velocity"),
        dict(field="mach", label="Mach number"),
        dict(field="schlieren", lut="graphite", label="schlieren"),
    ), t_end=1250.0, mach_e=2.40, pr_e=1.60, tr_e=2.20)


@scene
def pl_h_tri_gamma():
    """SPECIFIC-HEAT-RATIO LADDER - gamma 1.20 / 1.30 / 1.40, i.e. three propellants."""
    return _triple_h("pl_h_tri_gamma", (
        dict(gamma=1.20, label="gamma 1.20"),
        dict(gamma=1.30, label="gamma 1.30"),
        dict(gamma=1.40, label="gamma 1.40"),
    ), t_end=1250.0, mach_e=2.40, pr_e=1.60, tr_e=2.20)


@scene
def pl_h_tri_mach_hi():
    """HYPERSONIC MACH LADDER 3.0 / 4.0 / 5.0 - the long-cell end of the Pack relation."""
    return _triple_h("pl_h_tri_mach_hi", (
        dict(mach_e=3.00, label="M_e 3.0"),
        dict(mach_e=4.00, label="M_e 4.0"),
        dict(mach_e=5.00, label="M_e 5.0"),
    ), t_end=1250.0, pr_e=1.60, tr_e=2.20)


@scene
def pl_h_tri_underexpanded():
    """UNDEREXPANSION LADDER 2 / 6 / 14 - the Mach disk grows and moves downstream."""
    return _triple_h("pl_h_tri_underexpanded", (
        dict(pr_e=2.00, label="p_e/p_inf 2"),
        dict(pr_e=6.00, label="p_e/p_inf 6"),
        dict(pr_e=14.0, label="p_e/p_inf 14"),
    ), t_end=1250.0, mach_e=2.40, tr_e=2.20)


@scene
def pl_h_tri_overexpanded():
    """OVEREXPANSION LADDER 0.25 / 0.45 / 0.75 - separation shock strengthening as p_e falls."""
    return _triple_h("pl_h_tri_overexpanded", (
        dict(pr_e=0.25, label="p_e/p_inf 0.25"),
        dict(pr_e=0.45, label="p_e/p_inf 0.45"),
        dict(pr_e=0.75, label="p_e/p_inf 0.75"),
    ), t_end=1250.0, mach_e=2.40, tr_e=2.20)


@scene
def pl_h_tri_temp_hi():
    """HOT LADDER - T_e/T_inf 4 / 7 / 10, the high-energy end of the temperature sweep."""
    return _triple_h("pl_h_tri_temp_hi", (
        dict(tr_e=4.0, label="T_e/T_inf 4"),
        dict(tr_e=7.0, label="T_e/T_inf 7"),
        dict(tr_e=10.0, label="T_e/T_inf 10"),
    ), t_end=1250.0, mach_e=2.40, pr_e=1.30)
