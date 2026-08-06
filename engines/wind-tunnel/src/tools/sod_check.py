"""Does `wt\\cns.py` actually capture a shock, at the RIGHT place and the RIGHT strength?

THE GATE FOR THE COMPRESSIBLE SOLVER. Everything the rewrite was for rests on the claim that this
scheme resolves discontinuities correctly; that claim is checkable against an exact solution, so
it gets checked before any of it is believed. This is the same principle as `calib.py` measuring
Cd against a textbook 1.2 - and the reason it exists is AGENT_GUIDE 2a.1, where a force that
looked completely plausible was wrong by a factor of 20 with an arbitrary sign, for a whole
session, because nothing compared it to a known answer.

The Sod shock tube is the standard case: a diaphragm at x=0.5 separating
(rho, u, p) = (1, 0, 1) from (0.125, 0, 0.1), gamma = 1.4. Its exact solution is analytic and
contains, left to right, ALL THREE wave families a compressible solver has to get right -

    an expansion FAN | a CONTACT discontinuity | a SHOCK

- which is exactly why one test covers the scheme. HLLC's whole advantage over HLL is the middle
  one, so a run that nails the shock and smears the contact means the Riemann solver has silently
  degraded to HLL and every shear layer in a real scene will be over-diffused.

    tools\\sod_check.py [nx] [t]

Reports L1 error against the exact solution, the measured shock POSITION and PRESSURE JUMP
against theory, and the width in cells of each discontinuity. Runs in seconds - inviscid, 1-D,
no bodies.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wt import cns as C                       # noqa: E402
from wt.gpu import asnumpy, xp                # noqa: E402

G = C.GAMMA
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 400
TEND = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20


def exact_sod(x, t, rl=1.0, ul=0.0, pl=1.0, rr=0.125, ur=0.0, pr=0.1):
    """Exact Riemann solution, by Newton iteration on the star pressure (Toro chapter 4)."""
    cl, cr = np.sqrt(G * pl / rl), np.sqrt(G * pr / rr)

    def f(p, rk, pk, ck):
        if p > pk:                                   # shock branch
            a = 2.0 / ((G + 1.0) * rk)
            b = pk * (G - 1.0) / (G + 1.0)
            return (p - pk) * np.sqrt(a / (p + b)), np.sqrt(a / (p + b)) * (
                1.0 - 0.5 * (p - pk) / (p + b))
        return (2.0 * ck / (G - 1.0)) * ((p / pk) ** ((G - 1.0) / (2 * G)) - 1.0), \
               (1.0 / (rk * ck)) * (p / pk) ** (-(G + 1.0) / (2 * G))

    p = 0.5 * (pl + pr)
    for _ in range(60):
        fl, dfl = f(p, rl, pl, cl)
        fr, dfr = f(p, rr, pr, cr)
        pn = p - (fl + fr + (ur - ul)) / (dfl + dfr)
        p = max(pn, 1e-10)
    ps = p
    us = 0.5 * (ul + ur) + 0.5 * (f(ps, rr, pr, cr)[0] - f(ps, rl, pl, cl)[0])

    s = (x - 0.5) / max(t, 1e-12)
    r = np.zeros_like(x)
    u = np.zeros_like(x)
    pp = np.zeros_like(x)
    # left of the contact
    if ps > pl:                                                  # left shock (not in Sod)
        rsl = rl * ((ps / pl + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * ps / pl + 1))
        sl = ul - cl * np.sqrt((G + 1) / (2 * G) * ps / pl + (G - 1) / (2 * G))
        left, midl = s < sl, (s >= sl) & (s < us)
    else:                                                        # left expansion fan
        rsl = rl * (ps / pl) ** (1.0 / G)
        csl = cl * (ps / pl) ** ((G - 1) / (2 * G))
        sh, st = ul - cl, us - csl
        left, fan, midl = s < sh, (s >= sh) & (s < st), (s >= st) & (s < us)
        uf = 2.0 / (G + 1.0) * (cl + (G - 1) / 2.0 * ul + s)
        cf = 2.0 / (G + 1.0) * (cl + (G - 1) / 2.0 * (ul - s))
        r[fan] = (rl * (cf / cl) ** (2.0 / (G - 1)))[fan]
        u[fan] = uf[fan]
        pp[fan] = (pl * (cf / cl) ** (2 * G / (G - 1)))[fan]
    r[left], u[left], pp[left] = rl, ul, pl
    r[midl], u[midl], pp[midl] = rsl, us, ps
    # right of the contact: Sod's right wave is a shock
    rsr = rr * ((ps / pr + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * ps / pr + 1))
    sr = ur + cr * np.sqrt((G + 1) / (2 * G) * ps / pr + (G - 1) / (2 * G))
    midr, right = (s >= us) & (s < sr), s >= sr
    r[midr], u[midr], pp[midr] = rsr, us, ps
    r[right], u[right], pp[right] = rr, ur, pr
    return r, u, pp, ps, us, sr


# --- run the solver on a 1-D tube (a thin 2-D domain, uniform in y) ----------------------
# UNITS. The solver's length unit is one CELL and its time unit is cell / unit-speed; the
# analytic solution is written for a tube of length 1. Mapping the tube onto NX cells makes
# 1 cell = 1/NX tube-lengths and 1 solver-time = 1/NX tube-times, so the two scalings cancel and
# a VELOCITY is numerically the same in both. Only the elapsed time has to be converted:
# t_solver = t_tube * NX. Getting this backwards is a 400x error in the step count.
ny = 8
sim = C.CNS(NX, ny, u0=0.0, re=1e12, ref_len=1.0, csm=0.0, cfl=0.40, sponge_out=0)
sim.mu = 0.0                                    # INVISCID: the exact solution has no viscosity
sim.dt = 0.40 / 2.6                             # CFL 0.4 against max(|u| + c) ~ 2.6
Ng = C.NG
x = (np.arange(NX) + 0.5) / NX
Q = np.zeros((4, sim.Ny, sim.Nx), np.float32)
Q[0, :, Ng:-Ng] = np.where(x < 0.5, 1.0, 0.125).astype(np.float32)
Q[3, :, Ng:-Ng] = np.where(x < 0.5, 1.0, 0.1).astype(np.float32) / (G - 1.0)
Q[0, :, :Ng], Q[3, :, :Ng] = 1.0, 1.0 / (G - 1.0)
Q[0, :, -Ng:], Q[3, :, -Ng:] = 0.125, 0.1 / (G - 1.0)
sim.Q = xp.asarray(Q)


def _sod_bcs(q):
    """Hold both ends at their initial constant states, and stay uniform across the tube.

    The waves never reach the ends in the time we run, so this is exact. It is NOT optional
    though: `_flux_dir` uses `roll`, which wraps, so leaving the ghost cells to evolve would let
    the left end read the right end and vice versa, and that contamination spreads two cells per
    stage - about 2000 cells over this run.
    """
    q[0, :, :Ng], q[1, :, :Ng], q[2, :, :Ng], q[3, :, :Ng] = 1.0, 0.0, 0.0, 1.0 / (G - 1.0)
    q[0, :, -Ng:], q[1, :, -Ng:], q[2, :, -Ng:], q[3, :, -Ng:] = 0.125, 0.0, 0.0, 0.1 / (G - 1.0)
    q[:, :Ng, :] = q[:, Ng:Ng + 1, :]
    q[:, -Ng:, :] = q[:, -Ng - 1:-Ng, :]
    return q


sim._bcs = _sod_bcs
sim._q_far = xp.zeros((4, sim.Ny, 1), xp.float32)
sim._sponge = xp.zeros((1, 1, sim.Nx), xp.float32)          # no absorber in a validation case

dx = 1.0 / NX
nsteps = max(1, int(round(TEND * NX / sim.dt)))
print(f"Sod shock tube: nx={NX}, t={TEND}, dt={sim.dt:.4f} (cell units), {nsteps} steps")
for _ in range(nsteps):
    sim.step()

r_, u_, v_, p_ = sim._prim(sim.Q)
r = asnumpy(r_)[ny // 2 + Ng, Ng:-Ng].astype(np.float64)
u = asnumpy(u_)[ny // 2 + Ng, Ng:-Ng].astype(np.float64)
p = asnumpy(p_)[ny // 2 + Ng, Ng:-Ng].astype(np.float64)

re_, ue_, pe_, ps, us, sshock = exact_sod(x, TEND)
l1r = np.abs(r - re_).mean()
l1p = np.abs(p - pe_).mean()
l1u = np.abs(u - ue_).mean()
print(f"  L1 error   rho {l1r:.4f}   u {l1u:.4f}   p {l1p:.4f}    "
      f"(2nd-order Godunov on a discontinuity: ~1e-2 is right, ~1e-1 is broken)")

# --- the two things that must be EXACT, not merely small ---------------------------------
i = int(np.argmax(np.abs(np.diff(p))))           # steepest pressure jump = the shock
x_shock = x[i]
x_exact = 0.5 + sshock * TEND
print(f"  SHOCK position  {x_shock:.4f}  vs exact {x_exact:.4f}   "
      f"(err {abs(x_shock - x_exact) / dx:.2f} cells)")
print(f"  star pressure   {p[i - 6]:.4f}  vs exact {ps:.4f}   "
      f"(err {100 * abs(p[i - 6] - ps) / ps:.2f}%)   <- Rankine-Hugoniot")


def width(a, lo, hi, frac=(0.1, 0.9)):
    """How many cells a jump is smeared over, between `frac` of its total height."""
    span = hi - lo
    if abs(span) < 1e-9:
        return np.nan
    t = (a - lo) / span
    k = np.flatnonzero((t > frac[0]) & (t < frac[1]))
    return float(len(k))


j = int(np.argmax(np.abs(np.diff(r[:i - 4]))))   # the contact sits left of the shock
print(f"  SHOCK   smeared over {width(p[i - 8:i + 8], p[i + 7], p[i - 8]):.0f} cells   "
      f"(a TVD scheme should hold this at 2-3)")
print(f"  CONTACT smeared over {width(r[j - 10:j + 10], r[j + 9], r[j - 10]):.0f} cells   "
      f"(HLLC ~4-8; if this is 15+, the contact wave is being lost and it is really HLL)")
print(f"  min rho {r.min():.5f}, min p {p.min():.5f}   "
      f"(both must be > 0 - a negative one means the positivity floor was doing real work)")
