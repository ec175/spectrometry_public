# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""ledger_set.py - the four GASOLINE ledger films (2026-10-02).

Clancy: "lets make more of these ... Render four finals, 14 seconds each. they should all be
distinctly different. dont use methane, use gasoline." Each film varies ONE thing, so the
ledger has one story to tell:

    ledger_revs      the four-stroke speeds up x1 -> x3          (rate: same mass, faster)
    ledger_mixture   lean -> rich at a steady speed              (leftover O2, then unburnt fuel)
    ledger_boost     throttled -> boosted intake, steady speed   (more charge, bigger P-V loop)
    ledger_wankel    the rotary, speeding up                     (three faces, one firing per rev)

GASOLINE, AS THE MODEL BURNS IT. Gasoline is modelled as iso-octane in AIR (21 % O2, 79 % N2):

    2 C8H18 + 25 O2  ->  16 CO2 + 18 H2O          (N2 passes through)

one reaction UNIT in world.py (notes['rxn']). Products outnumber reactants, 34 vs 27, so the burn
CREATES molecules - a real source of gasoline's pressure rise the old count-conserving reaction
could not show. At stoichiometric (phi = 1) a charge is 1.65 % fuel, 20.66 % O2, 77.69 % N2 by
molecule count.
"""
from __future__ import annotations

from . import config as C

RXN = dict(fuel=2, ox=25, co2=16, h2o=18)
O2_PER_FUEL = 12.5
N2_PER_O2 = 79.0 / 21.0
# q per reaction UNIT, quoted at Q_REF_VTH: ~2000 K over the ~128 molecules a unit's share of
# a stoichiometric charge holds (gated; see S-LEDGER2 in development notes)
Q_UNIT = 4.6e9


def charge(phi):
    """Molecule fractions of a gasoline-air charge at equivalence ratio phi."""
    o2 = O2_PER_FUEL / phi
    n2 = o2 * N2_PER_O2
    tot = 1.0 + o2 + n2
    return {C.FUEL: 1.0 / tot, C.AIR: o2 / tot, C.N2: n2 / tot}


AIR_ONLY = {C.AIR: 0.21, C.N2: 0.79}
# an exhaust pipe full of stoichiometric exhaust: per unit 16 CO2 + 18 H2O + 94 N2
_EX = 16.0 + 18.0 + 25.0 * N2_PER_O2
EXHAUST = {C.CO2: 16.0 / _EX, C.H2O: 18.0 / _EX, C.N2: 25.0 * N2_PER_O2 / _EX}


def gasolinize(eng, phi=1.0, boost=1.0):
    """Turn an engine's charge into gasoline in air. phi and boost may be fn(t)."""
    eng.notes["rxn"] = dict(RXN)
    eng.notes["species_flows"] = True
    eng.fuel_frac = 0.0
    eng.q_rxn = Q_UNIT
    mix = (lambda t: charge(phi(t))) if callable(phi) else charge(phi)
    for b in eng.boxes:
        if b.role == "intake" and any(b.open):
            b.res_mix = mix
            b.boost = boost
        elif b.role == "exhaust" and any(b.open):
            b.res_mix = dict(EXHAUST)
            b.res_T = 700.0
        else:
            b.res_mix = dict(AIR_ONLY)
    return eng


def _clip_time(eng, t):
    return t - getattr(eng.drive, "warm", 0.0)


def _sweep(eng, a, b, seconds=14.0):
    def f(t):
        u = min(1.0, max(0.0, _clip_time(eng, t) / seconds))
        return a + (b - a) * u
    return f


def _fast_gas():
    # the radial cut's calibration: thermal speed 13000 with substeps raised in step, so the
    # ports do not choke at the top speeds; q is quoted at Q_REF_VTH and rescaled
    C.V_TH = 13000.0


def _four_stroke_hooks(eng):
    """What ledger2 measures on the four-stroke: chamber volume, the exhaust-valve-opening
    angle (leftovers are counted then), the port regions, and the swept volume for the scale."""
    from .gpu import xp
    x0, x1 = eng.notes["bore"]
    xm, deck, hb = 0.5 * (x0 + x1), eng.notes["deck"], eng.notes["head_box"]
    pfn = eng.pistons[0].fn

    def volume(t):
        return (pfn(t)[0] - deck) * (x1 - x0) + (x1 - x0) * 150.0 / 2.0
    eng.notes.update(volume_fn=volume, evo_deg=490.0, swept_area=(x1 - x0) * eng.notes["stroke"],
                     intake_region=lambda w: (w.box == hb) & (w.x < xm - 250.0),
                     exhaust_region=lambda w: (w.box == hb) & (w.x > xm + 250.0))
    return eng


def ledger_revs():
    from .pcat_engine import four_stroke
    _fast_gas()
    eng = _four_stroke_hooks(gasolinize(four_stroke(rev=2.5), phi=1.0, boost=1.0))
    eng.name = "ledger_revs"
    eng.notes.update(substeps=4300, zoom=0.5, view=(540.0, 930.0), warm=6.0, mult=3.2,
                     ramp_frac=1.3, film="revs", engine_kind="four_stroke")
    return eng


def ledger_mixture():
    from .pcat_engine import four_stroke
    _fast_gas()
    eng = four_stroke(rev=1.1)
    # intake phi 0.5 -> 2.2: the CYLINDER's mixture lags the intake (lean residual carries spare
    # O2), so the intake has to overshoot for the charge itself to cross into rich
    phi = _sweep(eng, 0.5, 2.2)
    _four_stroke_hooks(gasolinize(eng, phi=phi, boost=1.0))
    eng.name = "ledger_mixture"
    eng.notes.update(substeps=4300, zoom=0.5, view=(540.0, 930.0), warm=4.4, mult=1.0,
                     ramp_frac=1.0, film="mixture", engine_kind="four_stroke", phi_fn=phi)
    return eng


def ledger_boost():
    from .pcat_engine import four_stroke
    _fast_gas()
    eng = four_stroke(rev=1.1)
    bst = _sweep(eng, 0.5, 1.8)
    _four_stroke_hooks(gasolinize(eng, phi=1.0, boost=bst))
    eng.name = "ledger_boost"
    eng.notes.update(substeps=4300, zoom=0.5, view=(540.0, 930.0), warm=4.4, mult=1.0,
                     ramp_frac=1.0, film="boost", engine_kind="four_stroke", boost_fn=bst)
    return eng


def ledger_wankel():
    from .pcat_rotary import wankel
    _fast_gas()
    eng = gasolinize(wankel(), phi=1.0, boost=1.0)
    # NOT re-derived: halving q moved the rotary's measured chamber peak only 4247 -> 4047 K. That
    # peak is compression in the closing waist (the 2D-gas wedge heating of S-PARTS trap 2), not
    # the burn, so the ledger shows the rotary's EXHAUST temperature instead of a chamber peak
    eng.notes["show_peak"] = False
    eng.name = "ledger_wankel"
    ib, xb, clk, sp = (eng.notes[k] for k in ("intake_box", "exhaust_box", "clock", "spark_deg"))

    def phase(t):
        d = (clk(t) - sp) % 360.0
        return "power" if d < 150.0 else ("exhaust" if d < 240.0 else "compression")
    eng.notes.update(swept_area=74750.0, dual_plug=True, phase_fn=phase,
                     intake_region=lambda w: w.box == ib, exhaust_region=lambda w: w.box == xb)
    # the rotary drawn smaller so it sits in the top half (the half-frame is 1080 x 960 world)
    eng.notes.update(substeps=4300, zoom=0.62, view=(540.0, 960.0), warm=6.0, mult=3.0,
                     ramp_frac=1.3, film="wankel", engine_kind="wankel")
    return eng


LEDGERS = {"ledger_revs": ledger_revs, "ledger_mixture": ledger_mixture,
           "ledger_boost": ledger_boost, "ledger_wankel": ledger_wankel}
