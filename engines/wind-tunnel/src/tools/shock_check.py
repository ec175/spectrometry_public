"""Oblique shock off a wedge: does the 2-D scheme AND the immersed boundary get it right?

The second gate for `wt\\cns.py`, and the one that matters for a real scene. `sod_check.py` is
1-D and body-free, so it validates the Riemann solver and the limiter and nothing else. This case
adds the two things an actual render depends on:

  - a genuinely 2-D shock, oblique to the grid, so grid-alignment cannot flatter it;
  - a SOLID, imposed by the ghost-cell immersed boundary - so if the mirroring, the normals or
    the extrapolation factor are wrong, the wall turns the flow by the wrong angle and the shock
    lands somewhere theory does not predict.

Supersonic flow over a wedge of half-angle theta has an exact closed-form answer, the
theta-beta-M relation:

    tan(theta) = 2 cot(beta) (M1^2 sin^2(beta) - 1) / (M1^2 (gamma + cos 2*beta) + 2)

and the pressure jump across it follows from the normal-component Rankine-Hugoniot relation with
M1n = M1 sin(beta). Both are checked, because the ANGLE alone can come out right for the wrong
reason - a shock in the right place with the wrong strength means the scheme is not conservative.

    tools\\shock_check.py [mach] [theta_deg] [nx] [re]

`re` is quoted on the RAMP LENGTH IN CELLS, so it is the dial that isolates the viscous part of
the error from the discretisation part: refining `nx` alone leaves the boundary layer the same
fraction of the ramp and therefore cannot converge that piece away. Sweep `re` to find out which
one you are looking at.

Runs in a minute or two. Inviscid-dominated (high Re), so the wall boundary layer is thin enough
not to shift the shock; the angle is measured away from it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wt import cns as C                       # noqa: E402
from wt.gpu import asnumpy, xp                # noqa: E402

G = C.GAMMA
M1 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 420
RE = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0e5
NY = NX * 3 // 4


def beta_exact(m, th_deg):
    """Weak-solution root of the theta-beta-M relation, by bisection."""
    th = np.deg2rad(th_deg)
    mu = np.arcsin(1.0 / m)                                    # Mach angle: the weak-root floor
    lo, hi = mu + 1e-9, np.pi / 2 - 1e-9

    def f(b):
        return (2.0 / np.tan(b) * (m * m * np.sin(b) ** 2 - 1.0)
                / (m * m * (G + np.cos(2 * b)) + 2.0) - np.tan(th))
    # the weak root is the SMALLER beta; bracket between the Mach angle and the detachment peak
    bs = np.linspace(lo, hi, 20000)
    vals = f(bs)
    k = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
    if len(k) == 0:
        raise SystemExit(f"no attached solution: theta={th_deg} exceeds the max for M={m}")
    a, b = bs[k[0]], bs[k[0] + 1]
    for _ in range(80):
        c = 0.5 * (a + b)
        if np.sign(f(a)) == np.sign(f(c)):
            a = c
        else:
            b = c
    return 0.5 * (a + b)


be = beta_exact(M1, TH)
m1n = M1 * np.sin(be)
pr_exact = 1.0 + 2.0 * G / (G + 1.0) * (m1n * m1n - 1.0)
print(f"wedge: M1={M1}, theta={TH} deg, grid {NX}x{NY}")
print(f"  THEORY  beta={np.rad2deg(be):.3f} deg,  M1n={m1n:.4f},  p2/p1={pr_exact:.4f}")

# --- geometry: a compression ramp on the bottom wall -------------------------------------
i0 = int(0.22 * NX)                                   # apex
jj, ii = np.mgrid[0:NY, 0:NX]
ramp = (NY - 1) - np.maximum(ii - i0, 0) * np.tan(np.deg2rad(TH))
solid = jj >= ramp                                    # row index grows downward

sim = C.CNS(NX, NY, u0=M1, re=RE, ref_len=float(NX), csm=0.0,
            cfl=0.40, sponge_out=0, blockage=1.0)
print(f"  Re={RE:g} on the ramp length -> mu={sim.mu:.3e}")
# dt set explicitly: __init__'s estimate is built for a subsonic tunnel with heavy blockage and
# would be needlessly small at M=2 with a mild ramp.
sim.dt = 0.40 / (M1 * 1.25 + 1.35)
sim.set_solid(xp.asarray(solid))
sim._sponge = xp.zeros((1, 1, sim.Nx), xp.float32)    # supersonic outflow needs no absorber

Ng = C.NG
cross = NX / (M1 * sim.dt)
nsteps = int(round(3.0 * cross))                      # 3 domain transits -> steady
print(f"  dt={sim.dt:.4f}, {nsteps} steps (3 transits)")
for k in range(nsteps):
    sim.step()
    if not np.isfinite(sim.health()) or sim.health() > 10:
        raise SystemExit(f"DIVERGED at step {k}, max|u|={sim.health()}")

r, u, v, p = (asnumpy(a) for a in sim._prim(sim.Q))
p = p[Ng:-Ng, Ng:-Ng]
sol = asnumpy(sim.solid)
print(f"  final max|u|={sim.health():.4f} (Mach), CFL={sim.cfl_now():.3f}")

# --- measure the shock angle -------------------------------------------------------------
# Locate the shock in each row by the CENTROID of the pressure rise, not by argmax: the jump is
# spread over ~3 cells, so an argmax quantises the position to whole cells and the fit inherits
# that as noise. Rows are restricted to a band that is clear of the wall boundary layer below and
# of the corner where the shock meets the top of the domain above - including either would fit a
# straight line through something that is legitimately not straight.
rows, cols = [], []
for j in range(int(0.28 * NY), int(0.66 * NY)):
    seg = p[j].astype(np.float64).copy()
    seg[sol[j]] = np.nan
    d = np.diff(seg)
    lo, hi = i0 - 4, NX - 6
    d = d[lo:hi]
    if np.all(~np.isfinite(d)):
        continue
    k = int(np.nanargmax(d))
    if not np.isfinite(d[k]) or d[k] < 0.01 * sim.p_inf:
        continue
    w = np.clip(np.nan_to_num(d[max(0, k - 4):k + 5]), 0, None)   # sub-cell centroid
    if w.sum() <= 0:
        continue
    off = float((w * np.arange(len(w))).sum() / w.sum())
    rows.append(j)
    cols.append(lo + max(0, k - 4) + off + 0.5)
rows = np.asarray(rows, float)
cols = np.asarray(cols, float)
if len(rows) < 20:
    raise SystemExit("could not locate the shock - no coherent pressure jump found")
a, b = np.polyfit(rows, cols, 1)          # i = a*j + b; j grows downward so a < 0
beta = np.rad2deg(np.arctan2(1.0, abs(a)))
resid = float(np.std(cols - (a * rows + b)))
print(f"  MEASURED beta={beta:.3f} deg   (err {beta - np.rad2deg(be):+.3f} deg, "
      f"straightness {resid:.2f} cells rms over {len(rows)} rows)")

# --- and the strength, which the angle alone does not prove -------------------------------
# Sample well clear of the jump on both sides: within ~8 cells the state is still inside the
# captured profile, and downstream the ramp's own compression is approached near the wall.
jm = int(0.45 * NY)
im = a * jm + b
i1a, i1b = max(4, int(im) - 55), max(8, int(im) - 15)
i2a, i2b = min(NX - 6, int(im) + 15), min(NX - 4, int(im) + 55)
p1 = float(np.nanmean(np.where(sol[jm, i1a:i1b], np.nan, p[jm, i1a:i1b])))
p2 = float(np.nanmean(np.where(sol[jm, i2a:i2b], np.nan, p[jm, i2a:i2b])))
# invert the measured jump back to an angle - an independent estimate of beta that does not use
# the fit at all, so agreement between the two is a real cross-check rather than a restatement
m1n_m = np.sqrt(1.0 + (p2 / p1 - 1.0) * (G + 1.0) / (2.0 * G))
print(f"  MEASURED p2/p1={p2 / p1:.4f}   (theory {pr_exact:.4f}, "
      f"err {100 * (p2 / p1 - pr_exact) / pr_exact:+.2f}%)   <- Rankine-Hugoniot in 2-D")
print(f"  beta implied by that jump = {np.rad2deg(np.arcsin(min(1.0, m1n_m / M1))):.3f} deg "
      f"(independent of the line fit)")
print(f"  upstream p={p1:.5f} vs p_inf={sim.p_inf:.5f} "
      f"(err {100 * (p1 - sim.p_inf) / sim.p_inf:+.2f}%)   <- the freestream must be undisturbed")

# --- WHICH of the two is wrong: the geometry, or the shock relation? -----------------------
# beta being high is consistent with two completely different faults, and they need different
# fixes, so measure the flow DEFLECTION behind the shock directly and compare it to the wedge.
#   theta_measured ~= theta_wedge  -> the body turns the flow correctly and the shock relation is
#                                     off, which would be a solver bug
#   theta_measured >  theta_wedge  -> the shock is exactly right FOR THE FLOW IT IS GIVEN, and it
#                                     is the immersed boundary that over-turns; a geometry error
flds = [asnumpy(z)[Ng:-Ng, Ng:-Ng] for z in sim._prim(sim.Q)]
u2, v2 = flds[1], flds[2]
# Sample along a line PARALLEL to the shock, offset 30 cells downstream of it. Sampling a fixed
# COLUMN across a band of rows does not work: the shock is slanted, so at rows above the middle
# that column still sits upstream of it, and the average silently mixes undisturbed freestream
# (deflection 0) with post-shock flow. That reads LOW and looks like a physics result.
uu, vv = [], []
for j in range(int(0.30 * NY), int(0.60 * NY)):
    c_ = int(a * j + b) + 30
    if c_ >= NX - 6 or sol[j, c_]:
        continue
    uu.append(u2[j, c_])
    vv.append(v2[j, c_])
th_m = np.rad2deg(np.arctan2(-np.mean(vv), np.mean(uu)))          # -v: row index grows downward
bb = np.deg2rad(beta)
th_from_beta = np.rad2deg(np.arctan(
    2.0 / np.tan(bb) * (M1 ** 2 * np.sin(bb) ** 2 - 1.0)
    / (M1 ** 2 * (G + np.cos(2 * bb)) + 2.0)))
print(f"  flow DEFLECTION behind the shock = {th_m:.3f} deg   (the wedge is {TH:.1f} deg)")
print(f"  theta implied by the measured beta = {th_from_beta:.3f} deg")
if abs(th_m - th_from_beta) < 0.6:
    print(f"  -> GEOMETRY: the shock is CORRECT for the flow it is actually given; the immersed "
          f"boundary over-turns by {th_m - TH:+.2f} deg")
else:
    print("  -> SHOCK RELATION is off - a solver bug, not a geometry one. Do not ship.")
