# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""scenes_stock.py — the PROPULSION & PLUME theme, 30 clips at 3840x2160 and 30 s.

The second aerodynamics theme in the 2026-08-29 stock batch (the first is `Wind_Tunnel`'s
external flow). Built on `scenes_h`'s horizontal templates, which already solve the two things
this brief would otherwise have to invent: left-to-right IS the native orientation of a panel
frame, and black is a LUT swap rather than a grade.

⚠ `t_end` IS PHYSICS PACING, NOT CLIP LENGTH, and this is the whole reason a 15 s catalogue
carries over to 30 s unchanged. The solve runs `t` in `[0, t_end]` across the clip's frames, so
a front arriving at physical time T lands at fraction `T/t_end` of the clip whatever `seconds`
is. Doubling `seconds` plays the same evolution over twice as many frames — slower and
smoother, not truncated. The values here are `scenes_h`'s, and the dry gate's "ARRIVES at X% of
the clip" line is what confirms it rather than this paragraph.

⚠ WHAT VARIES, AND WHY IT IS NOT THIRTY OF THE SAME CLIP. A rocket plume has exactly four
physical knobs — exit Mach, pressure ratio, temperature ratio, and gamma — and a catalogue that
only turns those is a catalogue of one picture at thirty brightnesses. So half of these move
something else instead: the FIELD being read (velocity, Mach, schlieren), the number of streams,
and above all the SCHEDULE, since `t_end` decides which part of the plume's own history the
clip is showing. An ignition, a startup blast and a steady train are the same solver at three
places in its evolution, and they do not look alike.

⚠ NOTHING HERE IS A NAMED ENGINE. These are axisymmetric Euler solutions of a converging-
diverging nozzle at stated exit conditions. No clip may be described as a particular vehicle's
engine, and the descriptions that ship say "a rocket-style nozzle".
"""
from __future__ import annotations

from .scenes import scene
from .scenes_h import _panel_h, _triple_h

#: 3840x2160, the batch's tier. The templates take width/height, so this is one override.
W4, H4 = 3840, 2160
SEC = 30.0


def _p(name, **kw):
    """One horizontal stream at the stock tier."""
    kw.setdefault("t_end", 2050.0)
    kw.setdefault("tr_e", 2.20)
    return _panel_h(name, width=W4, height=H4, seconds=SEC, **kw)


def _t3(name, variants, t_end=2050.0, **kw):
    """Three stacked streams at the stock tier."""
    return _triple_h(name, variants, t_end, width=W4, height=H4, seconds=SEC, **kw)


# =============================================================================================
# THE SCHEDULE FAMILY — the same nozzle at different points in its own history.
# `t_end` short = the clip is the STARTUP; long = the clip is the steady train.
# =============================================================================================
@scene
def aero_plume_ignite():
    """Ignition: the clip ends while the starting vortex is still leaving the nozzle."""
    return _p("aero_plume_ignite", mach_e=2.40, pr_e=3.00, t_end=420.0)


@scene
def aero_startup_blast():
    """The impulsive start — the blast wave ahead of the jet is the whole subject."""
    return _p("aero_startup_blast", mach_e=2.80, pr_e=8.00, tr_e=4.00, t_end=260.0)


@scene
def aero_start_vortex():
    """The starting vortex ring, rolling up and convecting away from the lip."""
    return _p("aero_start_vortex", mach_e=2.00, pr_e=2.20, t_end=700.0)


@scene
def aero_plume_steady():
    """Long after the transient: a settled shock-cell train, barely changing."""
    return _p("aero_plume_steady", mach_e=2.40, pr_e=1.30, t_end=3600.0)


@scene
def aero_throttle_up():
    """Rising chamber pressure — the plume grows and the cells lengthen as it runs."""
    return _p("aero_throttle_up", mach_e=2.40, pr_e=5.00, t_ramp=1400.0, t_end=2600.0)


@scene
def aero_throttle_down():
    """The opposite ramp: a long tail-off with the train collapsing back toward the lip."""
    return _p("aero_throttle_down", mach_e=2.40, pr_e=0.55, t_ramp=200.0, t_end=2600.0)


@scene
def aero_plume_pulse():
    """A short, hard pulse — enough time for one blast and one relaxation."""
    return _p("aero_plume_pulse", mach_e=2.60, pr_e=9.00, tr_e=3.00, t_ramp=90.0,
              t_end=900.0)


@scene
def aero_retro_burn():
    """A retro burn: cold, low-ratio and short, so the plume never develops a long train."""
    return _p("aero_retro_burn", mach_e=1.80, pr_e=2.40, tr_e=1.40, t_end=1100.0)


# =============================================================================================
# THE PRESSURE FAMILY — the barrel shock and Mach disk this engine exists to show.
# =============================================================================================
@scene
def aero_mach_disk():
    """Strongly underexpanded: one large Mach disk standing in the first cell."""
    return _p("aero_mach_disk", mach_e=2.40, pr_e=10.0, t_end=2050.0)


@scene
def aero_shock_cells():
    """Perfectly expanded — the clean repeating cell train with no barrel shock."""
    return _p("aero_shock_cells", mach_e=2.40, pr_e=1.00, t_end=2050.0)


@scene
def aero_underexpanded():
    """p_e > p_inf: the jet expands out of the lip before the first shock turns it back."""
    return _p("aero_underexpanded", mach_e=2.40, pr_e=6.00, t_end=2050.0)


@scene
def aero_overexpanded():
    """p_e < p_inf: the ambient squeezes the jet and an oblique shock forms at the lip."""
    return _p("aero_overexpanded", mach_e=2.40, pr_e=0.45, t_end=2050.0)


@scene
def aero_altitude_climb():
    """A vacuum-like ratio — the plume balloons straight out of the nozzle."""
    return _p("aero_altitude_climb", mach_e=2.40, pr_e=18.0, t_end=2050.0)


@scene
def aero_choked_flow():
    """Just past the throat condition: exit Mach 1.2, the shortest cells the relation allows."""
    return _p("aero_choked_flow", mach_e=1.20, pr_e=1.80, t_end=2050.0)


@scene
def aero_expansion_fan():
    """A high exit Mach with a strong ratio — the lip fan is the dominant feature."""
    return _p("aero_expansion_fan", mach_e=3.60, pr_e=4.00, t_end=2400.0)


# =============================================================================================
# THE NOZZLE FAMILY — geometry, not chamber conditions.
# =============================================================================================
@scene
def aero_nozzle_bell():
    """A large exit radius: a bell nozzle, wide and low-divergence."""
    return _p("aero_nozzle_bell", mach_e=2.60, pr_e=2.00, r_exit=74.0, nozzle_px=150)


@scene
def aero_nozzle_cone():
    """A narrow conical nozzle — a tighter, faster, more sharply defined plume."""
    return _p("aero_nozzle_cone", mach_e=2.60, pr_e=2.00, r_exit=34.0, nozzle_px=95)


@scene
def aero_ground_plate():
    """A very large exit against a short domain: the plume fills the frame at the lip."""
    return _p("aero_ground_plate", mach_e=2.00, pr_e=3.20, r_exit=92.0, nozzle_px=180,
              scale=1.2)


@scene
def aero_flame_trench():
    """A long, narrow high-ratio jet reading down a deep channel."""
    return _p("aero_flame_trench", mach_e=3.00, pr_e=7.00, tr_e=5.00, r_exit=30.0,
              nozzle_px=90, t_end=2600.0)


# =============================================================================================
# THE PROPELLANT FAMILY — temperature and gamma, i.e. what is burning.
# =============================================================================================
@scene
def aero_afterburner():
    """Very hot exhaust, T_e/T_inf = 9 — a faster, wider, brighter plume."""
    return _p("aero_afterburner", mach_e=2.20, pr_e=2.40, tr_e=9.00, t_end=1800.0)


@scene
def aero_exhaust_cool():
    """Cold gas, T_e/T_inf = 1 — a thruster rather than a combustion product."""
    return _p("aero_exhaust_cool", mach_e=2.40, pr_e=1.30, tr_e=1.00, t_end=2050.0)


@scene
def aero_ramjet():
    """A low-gamma, hot, moderate-Mach stream — the ducted-combustion end of the range."""
    return _p("aero_ramjet", mach_e=2.00, pr_e=1.80, tr_e=6.00, gamma=1.20, t_end=2200.0)


@scene
def aero_scramjet():
    """High Mach, low ratio, low gamma — a long thin supersonic-combustion plume."""
    return _p("aero_scramjet", mach_e=4.00, pr_e=1.40, tr_e=7.00, gamma=1.22, t_end=2800.0)


# =============================================================================================
# THE READOUT FAMILY — the same physics, a different measurement.
# The cheapest genuine variety in the theme, and the least like a recolour: a schlieren image
# and a velocity map of one solve do not resemble each other at all.
# =============================================================================================
@scene
def aero_plume_shear():
    """The shear layer read as SCHLIEREN — density gradient, the wind-tunnel view."""
    return _p("aero_plume_shear", mach_e=2.40, pr_e=1.60, t_end=2050.0,
              field="schlieren", lut="graphite", lut_clip_hi=1.0, lut_gamma=1.0)


@scene
def aero_plume_flare():
    """Coloured by MACH NUMBER, so the sonic line and every cell boundary are legible."""
    return _p("aero_plume_flare", mach_e=2.60, pr_e=5.00, t_end=2050.0, field="mach")


@scene
def aero_thrust_vector():
    """A schlieren read of a strongly underexpanded jet — the barrel shock in gradient."""
    return _p("aero_thrust_vector", mach_e=2.40, pr_e=8.00, t_end=2050.0,
              field="schlieren", lut="graphite", lut_clip_hi=1.0, lut_gamma=1.0)


@scene
def aero_gimbal_sweep():
    """Mach field on a low-Mach, tightly packed train — many cells, all readable."""
    return _p("aero_gimbal_sweep", mach_e=1.60, pr_e=2.20, t_end=2050.0, field="mach")


# =============================================================================================
# THE MULTI-STREAM FAMILY — three solves in one frame.
# =============================================================================================
@scene
def aero_twin_plume():
    """Two conditions against each other, run as the three-panel rig with a matched pair."""
    return _t3("aero_twin_plume",
               [dict(pr_e=6.0, label="under"), dict(pr_e=1.0, label="design"),
                dict(pr_e=6.0, label="under")], t_end=2050.0)


@scene
def aero_triple_plume():
    """The exit-Mach ladder: 1.6 / 2.4 / 3.2. Cell spacing grows as sqrt(M^2 - 1)."""
    return _t3("aero_triple_plume",
               [dict(mach_e=1.6, label="M 1.6"), dict(mach_e=2.4, label="M 2.4"),
                dict(mach_e=3.2, label="M 3.2")], t_end=2050.0)


@scene
def aero_stage_sep():
    """The altitude ladder: sea level, design, vacuum — the plume opening out as p_inf falls."""
    return _t3("aero_stage_sep",
               [dict(pr_e=0.45, label="sea level"), dict(pr_e=1.0, label="design"),
                dict(pr_e=14.0, label="vacuum")], t_end=2050.0)
