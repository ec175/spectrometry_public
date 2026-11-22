# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""nozzle.py - the isentropic relations, and the divergent section as an EXACT overlay.

The reference clip draws the nozzle interior: a cone at the left of the panel with a visible
exit-plane line at x = 40 px, R_e = 19 px, and a smoothly accelerating core inside it. This
module reproduces that WITHOUT putting the nozzle in the solve.

WHY THE INTERIOR IS NOT SOLVED, AND WHY THAT IS NOT A CHEAT
-----------------------------------------------------------
Inside a steady supersonic divergent section the flow is, to the accuracy anything in a 9:16
video could show, quasi-1-D isentropic - the area-Mach relation is the closed-form answer and
solving it numerically would reproduce that answer more slowly and less accurately. More
importantly the exit state is an INPUT to the plume solve, not an outcome of it: supersonic
inflow means every characteristic points downstream, so the nozzle cannot hear the plume. Paying
for cells that cannot influence the picture would be paying for nothing.

What IS honest about it: the painted interior is generated from the SAME exit state the solver
is given, so the cone and the jet always agree. `plume_check.py` asserts that agreement rather
than trusting it.

THE ONE THING TO REMEMBER
-------------------------
The area-Mach relation has TWO roots for any area ratio - one subsonic, one supersonic. This
module always takes the supersonic branch downstream of the throat. A solver fed the subsonic
root would produce a jet that immediately stagnates and looks like a broken boundary condition.
"""
from __future__ import annotations

import math

import numpy as np


def area_ratio(mach, gamma=1.4):
    """A/A* for a given Mach number (both branches; the formula is single-valued this way)."""
    m = np.asarray(mach, np.float64)
    g = gamma
    return (1.0 / m) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * m * m)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area(ar, gamma=1.4, supersonic=True):
    """Invert A/A* -> M by bisection on the requested branch. `ar` may be an array."""
    ar = np.asarray(ar, np.float64)
    lo = np.full(ar.shape, 1.0 + 1e-9) if supersonic else np.full(ar.shape, 1e-6)
    hi = np.full(ar.shape, 60.0) if supersonic else np.full(ar.shape, 1.0 - 1e-9)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = area_ratio(mid, gamma)
        # A/A* increases with M above 1 and decreases with M below 1
        if supersonic:
            lo = np.where(f < ar, mid, lo)
            hi = np.where(f < ar, hi, mid)
        else:
            lo = np.where(f > ar, mid, lo)
            hi = np.where(f > ar, hi, mid)
    return 0.5 * (lo + hi)


def isentropic_ratios(mach, gamma=1.4):
    """(T/T0, p/p0, rho/rho0) at a given Mach number."""
    m2 = np.asarray(mach, np.float64) ** 2
    g = gamma
    t = 1.0 / (1.0 + 0.5 * (g - 1.0) * m2)
    return t, t ** (g / (g - 1.0)), t ** (1.0 / (g - 1.0))


def jet_speed(mach_e, pr_e=1.0, tr_e=1.0, gamma=1.4):
    """Speed of the FULLY EXPANDED jet (p = p_ambient), in ambient sound speeds.

    Total enthalpy along a steady streamline: ``u^2/2 + c^2/(g-1)`` is conserved from the exit to
    the plane where the jet has expanded to ambient pressure. This is the physical speed the jet
    core actually runs at, and it is what the colour range should be pinned to - see
    `render.Panel`, and Wind_Tunnel's rule that a palette is DERIVED, not picked.
    """
    import math
    ce = math.sqrt(tr_e)
    ue = mach_e * ce
    E = 0.5 * ue * ue + ce * ce / (gamma - 1.0)
    c2 = tr_e * (1.0 / pr_e) ** ((gamma - 1.0) / gamma)
    return math.sqrt(max(2.0 * (E - c2 / (gamma - 1.0)), 1e-12))


def exit_from_geometry(area_ratio_e, gamma=1.4):
    """Exit Mach for a given expansion ratio A_e/A*. The supersonic root, always."""
    return float(mach_from_area(np.array(float(area_ratio_e)), gamma, supersonic=True))


def paint(nz_cone, nr, r_exit, mach_e, gamma=1.4, r_throat=None, field="speed",
          tr_e=1.0, pr_e=1.0):
    """The divergent section as a (nr, nz_cone) field, in the SOLVER's units.

    Returns the field on the same half-domain grid as the solve, so it can be concatenated onto
    the left of the solved region and colour-mapped by exactly the same LUT and the same
    per-frame range. Cells outside the wall are returned as NaN and the caller paints the base
    plate there.

    `r_throat` defaults to **the throat that actually produces `mach_e`**, i.e.
    ``r_exit / sqrt(A_e/A*)``, so the drawn silhouette IS the nozzle the solver is being given.

    THIS WAS A REAL BUG, AND plume_check FOUND IT
    ---------------------------------------------
    The first cut copied the reference's drawn taper (throat/exit = 0.29) and then read Mach off
    the cone's own area ratio. 0.29 implies A_e/A* = 11.9, hence M_e ~ 3.95 - so the painted cone
    ended at u = 3.56 while the solver's exit state was u = 2.55, a **40% velocity discontinuity
    straight across the exit-plane line**, visible in the render as a colour jump.

    Note what that says about the SOURCE: its drawn cone implies M_e ~ 3.95, while its measured
    shock-cell spacing implies M_jet ~ 1.97. **The reference's nozzle drawing is not consistent
    with the reference's own plume** - it is decorative. This engine chooses consistency, so its
    cone is a shallower taper than the source's.

    The local area ratio is taken RELATIVE TO THE EXIT, so `m` terminates at exactly `mach_e`
    whatever silhouette is asked for. An explicit `r_throat` override is honoured for art
    reasons, and A/A* is clamped at 1 (the throat) so the supersonic branch stays defined.
    """
    g = gamma
    ar_exit = float(area_ratio(np.array(float(mach_e)), g))       # A_e / A*
    if r_throat is None:
        r_throat = float(r_exit) / math.sqrt(ar_exit)
    r_throat = min(float(r_throat), float(r_exit) * 0.999)
    z = (np.arange(nz_cone, dtype=np.float64) + 0.5) / max(nz_cone, 1)
    rw = r_throat + (float(r_exit) - r_throat) * z                # straight-walled cone
    ar = np.maximum(ar_exit * (rw / float(r_exit)) ** 2, 1.0)     # A/A*, anchored at the EXIT
    m = mach_from_area(ar, g, supersonic=True)

    # Stagnation state consistent with the exit state the SOLVER is given, so the two agree by
    # construction rather than by luck.
    t_e, p_e_ratio, _ = isentropic_ratios(mach_e, g)
    p_amb = 1.0 / g
    p0 = pr_e * p_amb / p_e_ratio
    t0 = tr_e / t_e                                               # T0/T_ambient

    t_r, p_r, rho_r = isentropic_ratios(m, g)
    T = t0 * t_r                                                  # T/T_ambient
    c = np.sqrt(T)                                                # c/c_ambient
    u = m * c
    p = p0 * p_r
    rho = g * p / np.maximum(c * c, 1e-12)

    if field == "speed":
        line = u
    elif field == "mach":
        line = m
    elif field == "pressure":
        line = p
    elif field == "density":
        line = rho
    elif field == "temperature":
        line = T
    else:                                                          # schlieren: smooth interior
        line = np.zeros_like(u)

    rr = (np.arange(nr, dtype=np.float64) + 0.5)
    out = np.broadcast_to(line.reshape(1, -1), (nr, nz_cone)).copy()
    out[rr.reshape(-1, 1) > rw.reshape(1, -1)] = np.nan
    return out.astype(np.float32), rw.astype(np.float32)
