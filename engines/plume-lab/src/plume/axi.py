# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""axi.py - the solver: 2-D AXISYMMETRIC compressible Euler, finite volume, HLLC + MUSCL.

WHAT THIS IS, AND WHERE IT CAME FROM
------------------------------------
The numerics are ported from ``Wind_Tunnel\\wt\\cns.py``, which is validated (`tools/sod_check.py`
reproduces the exact Sod solution to 0.08% on the star pressure, and an oblique shock to 0.5 deg).
Nothing about the RIEMANN SOLVER or the LIMITER is new here and nothing about them should be
"improved" without re-running that gate. What is new is the geometry.

WHY NOT JUST USE `wt/cns.py`
----------------------------
Three structural reasons, all of which are about a plume having no freestream:

  - every scale in `cns.py` hangs off `u0`: ``nu = u0*ref_len/re``, ``dt`` from ``blockage*u0``,
    and the outlet sponge relaxes toward a ``u0`` state. A plume's ambient is **u0 = 0** and each
    of those degenerates rather than merely being wrong;
  - its boundary family is a TUNNEL - inlet plane, outlet, free-slip side walls. A plume needs a
    quiescent non-reflecting far field on two sides, a symmetry axis on the third and a nozzle
    patch in an otherwise solid base plate on the fourth;
  - it is planar. This one is axisymmetric, and that is not a rendering trick: the source clip is
    mirror-symmetric top-to-bottom to a measured **MAE of 0.60-1.49 out of 255**, i.e. exactly.
    The half domain IS the simulation.

THE EQUATIONS
-------------
Conservative axisymmetric Euler in (z, r), no swirl, U = (rho, rho*u, rho*v, E) with u axial and
v radial::

    dU/dt + dF/dz + dG/dr = -(1/r) * H

    F = (rho*u,  rho*u^2 + p,  rho*u*v,      (E+p)*u )
    G = (rho*v,  rho*u*v,      rho*v^2 + p,  (E+p)*v )
    H = (rho*v,  rho*u*v,      rho*v^2,      (E+p)*v )

**H is G with the pressure taken out of the radial-momentum component, and only that one.** That
is the single most common way to get an axisymmetric code wrong: leave `p` in H and a uniform gas
at rest spontaneously accelerates toward the axis, which looks like a plausible "confinement"
effect and is a bug. `tools/quiet_check.py` exists precisely to catch it - a still, uniform gas
must stay still to float32 epsilon, and with H written correctly every flux difference and the
source are identically zero, so it does.

THE AXIS
--------
Cell centres sit at ``r_j = (j + 1/2) * dr``, so no cell ever sits ON r = 0 and 1/r is finite
everywhere. The axis is imposed by MIRROR ghosts with the radial velocity sign-flipped, which is
the correct symmetry condition and costs nothing.

UNITS
-----
Non-dimensionalised on the AMBIENT: ``rho_inf = 1``, ``p_inf = 1/gamma``, hence ``c_inf = 1``.
So **a velocity in this solver is a Mach number referred to the ambient sound speed** - the same
convention as `wt/cns.py`, and the same trap if a number is carried across from `wt/lbm.py`,
where a velocity is a lattice velocity. Length unit is one cell, time unit is one cell per
ambient sound speed.

THE TIME STEP IS ADAPTIVE - and here that is the CORRECT choice
---------------------------------------------------------------
`wt/cns.py` deliberately fixes `dt`, because in that project apparent wind speed is
(distance per step x steps per frame) and a wandering `dt` would make on-screen speed a function
of the flow. **That argument does not transfer.** Here the physical clock IS the deliverable (the
source clip burns `t = ... ms of ... simulated` into the frame), so the honest thing is to
integrate the real time and emit a frame when `t` crosses the frame's target - which is what
`advance_to` does, clamping the final sub-step so it lands exactly on the target. The pay-off is
large and it is at the start: a quiescent ambient has ``max(|u|+c) = 1`` while a running plume
core has ~6, so the first frames take ~6x larger steps for free.
"""
from __future__ import annotations

import numpy as np

from .gpu import ElementwiseKernel, GPU, asnumpy, xp

NG = 2                      # ghost layers; MUSCL reaches 2 cells
_TINY = 1e-12
R_AX, Z_AX = 1, 2           # array axes of the (4, NR, NZ) state


def _minmod(a, b):
    """The limiter. Smaller slope in magnitude, or 0 where the two one-sided slopes disagree.

    The zero branch is the whole point - at an extremum the reconstruction drops to 1st order and
    the scheme stays TVD, which is what stops a captured shock ringing. Do not swap this for a
    smoother limiter without re-running tools/sod_check.py.
    """
    return xp.where(a * b > 0.0, xp.where(xp.abs(a) < xp.abs(b), a, b), 0.0)


_HLLC_SRC = r"""
    const float gm1 = gam - 1.0f;
    float cL = sqrtf(gam * pL / rL);
    float cR = sqrtf(gam * pR / rR);
    float SL = fminf(uL - cL, uR - cR);
    float SR = fmaxf(uL + cL, uR + cR);
    float EL = pL / gm1 + 0.5f * rL * (uL*uL + vL*vL);
    float ER = pR / gm1 + 0.5f * rR * (uR*uR + vR*vR);
    float mL = rL * (SL - uL);
    float mR = rR * (SR - uR);
    float den = mL - mR;
    if (fabsf(den) < 1e-12f) den = (den < 0.0f) ? -1e-12f : 1e-12f;
    float Ss = (pR - pL + mL*uL - mR*uR) / den;
    if (SL >= 0.0f) {
        f0 = rL*uL;  f1 = rL*uL*uL + pL;  f2 = rL*uL*vL;  f3 = (EL + pL)*uL;
    } else if (SR <= 0.0f) {
        f0 = rR*uR;  f1 = rR*uR*uR + pR;  f2 = rR*uR*vR;  f3 = (ER + pR)*uR;
    } else if (Ss >= 0.0f) {
        float d = SL - Ss;  if (fabsf(d) < 1e-12f) d = 1e-12f;
        float fk = mL / d;
        float mm = mL;      if (fabsf(mm) < 1e-12f) mm = 1e-12f;
        float e  = EL/rL + (Ss - uL) * (Ss + pL/mm);
        f0 = rL*uL         + SL * (fk        - rL);
        f1 = rL*uL*uL + pL + SL * (fk*Ss     - rL*uL);
        f2 = rL*uL*vL      + SL * (fk*vL     - rL*vL);
        f3 = (EL + pL)*uL  + SL * (fk*e      - EL);
    } else {
        float d = SR - Ss;  if (fabsf(d) < 1e-12f) d = 1e-12f;
        float fk = mR / d;
        float mm = mR;      if (fabsf(mm) < 1e-12f) mm = 1e-12f;
        float e  = ER/rR + (Ss - uR) * (Ss + pR/mm);
        f0 = rR*uR         + SR * (fk        - rR);
        f1 = rR*uR*uR + pR + SR * (fk*Ss     - rR*uR);
        f2 = rR*uR*vR      + SR * (fk*vR     - rR*vR);
        f3 = (ER + pR)*uR  + SR * (fk*e      - ER);
    }
"""

# The fused kernel is AGENT_GUIDE backlog item 8 from Wind_Tunnel, which was never started there.
# The vectorised form below evaluates every branch of the wave-pattern selection over the whole
# field and throws three of the four away - ~40 full-field temporaries per face sweep, on a scheme
# that is memory-bandwidth-bound. One ElementwiseKernel does the branch per cell instead.
# `tools/sod_check.py --both` runs the gate through BOTH paths and diffs them, so this is a
# measured equivalence and not an assumed one.
_HLLC_KERNEL = None
if ElementwiseKernel is not None:
    _HLLC_KERNEL = ElementwiseKernel(
        "float32 rL, float32 uL, float32 vL, float32 pL, "
        "float32 rR, float32 uR, float32 vR, float32 pR, float32 gam",
        "float32 f0, float32 f1, float32 f2, float32 f3",
        _HLLC_SRC, "plume_hllc")


def hllc_vec(rL, uL, vL, pL, rR, uR, vR, pR, gamma):
    """Reference HLLC, portable. `u` is normal to the face, `v` tangential.

    Kept as the definition of correct even when the fused kernel is what runs: it is what
    sod_check compares against, and it is the only version that runs on the CPU fallback.
    """
    g = gamma
    cL = xp.sqrt(g * pL / rL)
    cR = xp.sqrt(g * pR / rR)
    SL = xp.minimum(uL - cL, uR - cR)
    SR = xp.maximum(uL + cL, uR + cR)
    EL = pL / (g - 1.0) + 0.5 * rL * (uL * uL + vL * vL)
    ER = pR / (g - 1.0) + 0.5 * rR * (uR * uR + vR * vR)
    mL = rL * (SL - uL)
    mR = rR * (SR - uR)
    den = mL - mR
    Ss = (pR - pL + mL * uL - mR * uR) / xp.where(xp.abs(den) < _TINY, _TINY, den)

    FL = xp.stack([rL * uL, rL * uL * uL + pL, rL * uL * vL, (EL + pL) * uL])
    FR = xp.stack([rR * uR, rR * uR * uR + pR, rR * uR * vR, (ER + pR) * uR])
    UL = xp.stack([rL, rL * uL, rL * vL, EL])
    UR = xp.stack([rR, rR * uR, rR * vR, ER])

    def star(rK, uK, vK, pK, EK, SK, mK):
        f = mK / xp.where(xp.abs(SK - Ss) < _TINY, _TINY, SK - Ss)
        e = EK / rK + (Ss - uK) * (Ss + pK / xp.where(xp.abs(mK) < _TINY, _TINY, mK))
        return xp.stack([f, f * Ss, f * vK, f * e])

    FsL = FL + SL * (star(rL, uL, vL, pL, EL, SL, mL) - UL)
    FsR = FR + SR * (star(rR, uR, vR, pR, ER, SR, mR) - UR)
    return xp.where(SL >= 0.0, FL,
                    xp.where(Ss >= 0.0, FsL, xp.where(SR >= 0.0, FsR, FR)))


class AxiEuler:
    """Axisymmetric Euler on a uniform (z, r) grid. Half domain; the render mirrors it.

    `nz` / `nr` are INTERIOR sizes. Every public field accessor returns the interior view, so
    nothing outside this file has to know about the ghost padding.
    """

    def __init__(self, nz, nr, gamma=1.4, cfl=0.40, dz=1.0, dr=1.0,
                 sponge_r=28, sponge_z=32, sponge_k=0.22,
                 active=True, active_start=192, active_chunk=128,
                 active_probe=24, active_every=8, active_tol=2e-4,
                 dt_every=4, dt_margin=1.15, fused=True):
        self.nz, self.nr = int(nz), int(nr)
        self.NZ, self.NR = self.nz + 2 * NG, self.nr + 2 * NG
        self.gamma = float(gamma)
        self.cfl = float(cfl)
        self.dz, self.dr = float(dz), float(dr)
        self.t = 0.0
        self.steps_done = 0
        self.dt = 0.0
        self._dt_every = int(dt_every)
        self._dt_margin = float(dt_margin)
        self._wave = 1.0
        self.fused = bool(fused) and _HLLC_KERNEL is not None

        f32 = xp.float32
        self.rho_inf = 1.0
        self.p_inf = 1.0 / self.gamma                   # -> c_inf = 1 exactly
        self.e_inf = self.p_inf / (self.gamma - 1.0)

        self.Q = xp.zeros((4, self.NR, self.NZ), dtype=f32)
        self.Q[0] = self.rho_inf
        self.Q[3] = self.e_inf
        self._q_amb = xp.asarray(
            np.array([self.rho_inf, 0.0, 0.0, self.e_inf], np.float32).reshape(4, 1, 1))

        # cell-centre radius, INCLUDING the ghost rows, so the source term needs no special case
        # (NR, 1) so it broadcasts against a single (NR, NZ) component, not the (4, NR, NZ) stack
        j = np.arange(self.NR, dtype=np.float32) - NG + 0.5
        self._r = xp.asarray((j * self.dr).reshape(-1, 1))
        self._inv_r = 1.0 / self._r                     # ghost rows carry r<0; never read there

        # --- graded absorbers. A hard zero-gradient edge reflects, and what reflects in a
        # compressible solve is an acoustic wave that crosses the entire picture. Same device and
        # same justification as Wind_Tunnel's.
        wr = np.zeros((1, self.NR, 1), np.float32)
        if sponge_r > 0:
            d = (self.NR - 1 - NG - np.arange(self.NR, dtype=np.float32)) / float(sponge_r)
            wr[0, :, 0] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        wz = np.zeros((1, 1, self.NZ), np.float32)
        if sponge_z > 0:
            d = (self.NZ - 1 - NG - np.arange(self.NZ, dtype=np.float32)) / float(sponge_z)
            wz[0, 0, :] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        self._sponge_r = xp.asarray(wr * float(sponge_k))
        self._sponge_z_full = xp.asarray(wz * float(sponge_k))
        self._sponge_z = int(sponge_z)

        # --- nozzle patch (set by set_nozzle)
        self._patch = None          # bool (NR,) over ghost-padded rows
        self._q_exh = None          # (4, npatch, 1) full-strength exhaust state
        self._ramp = 1.0
        self.t_ramp = 0.0           # >0 -> the ramp is driven by self.t inside _bcs
        # Wave speed carried by the INFLOW, which is not in the interior field at t = 0 and is
        # the whole reason `plume_under` and `plume_vacuum` NaN'd on their first frames. See
        # `_pick_dt`.
        self._inflow_wave = 0.0

        # --- ACTIVE BOX. Solve only the columns that are not still exactly ambient, and GROW it
        # on measured evidence rather than on a wave-speed bound. The tempting bound - "the front
        # cannot outrun c_inf" - is FALSE for a starting jet: the lead blast shock outruns the
        # ambient sound speed by definition, by 1.5-2x early on. So instead the last
        # `active_probe` columns of the box are checked against ambient every `active_every`
        # steps and the box grows when they stop being ambient. One reduction, and it cannot be
        # wrong about physics it does not assume.
        self.active = bool(active)
        self.nz_act = min(self.nz, int(active_start)) if self.active else self.nz
        self._chunk = int(active_chunk)
        self._probe = int(active_probe)
        self._act_every = int(active_every)
        self._act_tol = float(active_tol)
        self.grew = 0

    # ------------------------------------------------------------------ geometry / forcing
    def set_nozzle(self, r_exit, mach_e, pr_e=1.0, tr_e=1.0):
        """Supersonic Dirichlet inflow over ``r < r_exit`` in the base plate at z = 0.

        Supersonic inflow means every characteristic points into the domain, so prescribing the
        full state is well-posed - no extrapolated variable, no reflection. That is the reason
        the nozzle INTERIOR is not solved: at the exit plane the state is an input, not an
        outcome (see `nozzle.py`, which computes it from the closed-form isentropic relations
        and paints the divergent section as an exact overlay).

        `pr_e` = p_exit/p_ambient (>1 underexpanded, <1 overexpanded, =1 perfectly expanded),
        `tr_e` = T_exit/T_ambient. In these units c_exit = sqrt(tr_e), so the exit velocity is
        ``mach_e * sqrt(tr_e)`` and THAT is the number the time step ends up paying for.
        """
        g = self.gamma
        rows = np.arange(self.NR, dtype=np.float32) - NG + 0.5
        patch = (rows * self.dr) < float(r_exit)
        patch &= rows >= 0.0                              # never the axis ghosts
        p = float(pr_e) * self.p_inf
        rho = p / (self.p_inf * float(tr_e)) * self.rho_inf    # p/(R T), with R folded in
        c = np.sqrt(g * p / rho)
        u = float(mach_e) * c
        E = p / (g - 1.0) + 0.5 * rho * u * u
        n = int(patch.sum())
        self._patch = xp.asarray(patch)
        self._patch_np = patch
        self._q_exh = xp.asarray(
            np.array([rho, rho * u, 0.0, E], np.float32).reshape(4, 1, 1) * np.ones((1, n, 1), np.float32))
        self.exit_state = dict(rho=float(rho), u=float(u), p=float(p), c=float(c),
                               mach=float(mach_e), r_exit=float(r_exit))
        self._inflow_wave = float(u + c)

    def set_ramp(self, frac):
        """Blend the inflow from ambient (0) to full exhaust (1). Manual override.

        Normally driven by `t_ramp` from the solver's own clock inside `_bcs`, because a ramp
        advanced once per FRAME is not a ramp: at 180 frames over t_end 1230 a frame is 6.8 time
        units and a 1.0-unit ramp collapses to a single instantaneous jump at a frame boundary.

        A real cold start IS impulsive - that is where the blast wave in the reference clip comes
        from - so this stays only a few steps wide. It exists to keep the very first face solve
        out of a state no limiter can help with, not to soften the start.
        """
        self._ramp = float(np.clip(frac, 0.0, 1.0))

    # ------------------------------------------------------------------ scheme
    @staticmethod
    def _sl(axis, a, b):
        idx = [slice(None), slice(None), slice(None)]
        idx[axis] = slice(a, b)
        return tuple(idx)

    def _prim(self, Q):
        r = xp.maximum(Q[0], 1e-7)
        u = Q[1] / r
        v = Q[2] / r
        p = xp.maximum((self.gamma - 1.0) * (Q[3] - 0.5 * r * (u * u + v * v)), 1e-9)
        return r, u, v, p

    def _hllc(self, WL, WR, i_n, i_t):
        """Face flux from reconstructed primitive stacks. Returns (mass, normal, tangential, E)."""
        rL, pL = WL[0], WL[3]
        rR, pR = WR[0], WR[3]
        if self.fused:
            shape = rL.shape
            f = [xp.empty(shape, dtype=xp.float32) for _ in range(4)]
            _HLLC_KERNEL(rL, WL[i_n], WL[i_t], pL, rR, WR[i_n], WR[i_t], pR,
                         xp.float32(self.gamma), f[0], f[1], f[2], f[3])
            return f
        F = hllc_vec(rL, WL[i_n], WL[i_t], pL, rR, WR[i_n], WR[i_t], pR, self.gamma)
        return [F[0], F[1], F[2], F[3]]

    def _sweep(self, W, axis, dx, i_n, i_t, out=None, accum=False):
        """MUSCL reconstruct -> HLLC -> flux difference along one axis.

        Index bookkeeping, for the next person: slopes are defined on cells 1..N-2, faces on
        1..N-3, and the interior [NG:-NG] therefore differences F[1:] against F[:-1].
        """
        sl = self._sl
        Wm = W[sl(axis, 0, -2)]
        Wc = W[sl(axis, 1, -1)]
        Wp = W[sl(axis, 2, None)]
        s = _minmod(Wc - Wm, Wp - Wc)
        WL = Wc[sl(axis, 0, -1)] + 0.5 * s[sl(axis, 0, -1)]
        WR = Wc[sl(axis, 1, None)] - 0.5 * s[sl(axis, 1, None)]
        # Floor the RECONSTRUCTED rho and p, not just the cell averages. minmod keeps the
        # reconstruction monotone but not positive, and one negative pressure inside a strong
        # expansion makes sqrt(gamma p/rho) NaN and takes the whole field with it in one step.
        # A plume is exactly the flow that finds this: the exit expansion fan is the strongest
        # rarefaction in the library. The floor only engages where the limiter already gave up.
        # Written in place on the freshly-built arrays - a concatenate here would allocate two
        # more full-size temporaries per sweep on a scheme that is memory-bandwidth-bound.
        WL[0] = xp.maximum(WL[0], 1e-7)
        WL[3] = xp.maximum(WL[3], 1e-9)
        WR[0] = xp.maximum(WR[0], 1e-7)
        WR[3] = xp.maximum(WR[3], 1e-9)
        F = self._hllc(WL, WR, i_n, i_t)

        if out is None:
            out = xp.zeros_like(W)
        hi = sl(axis, NG, -NG)[1:]
        a = sl(axis, 1, None)[1:]
        b = sl(axis, 0, -1)[1:]
        inv = 1.0 / dx
        for k, c in enumerate((0, i_n, i_t, 3)):
            d = -(F[k][a] - F[k][b]) * inv
            if accum:
                out[c][hi] += d
            else:
                out[c][hi] = d
        return out

    def _bcs(self, Q):
        """Ghost cells: axis, far field, base plate + nozzle patch. In place."""
        # AXIS (low r): mirror with the RADIAL momentum sign-flipped. The exact symmetry
        # condition; anything else puts a source or a sink on the centreline.
        for g, src in ((NG - 1, NG), (NG - 2, NG + 1)):
            Q[0, g] = Q[0, src]
            Q[1, g] = Q[1, src]
            Q[2, g] = -Q[2, src]
            Q[3, g] = Q[3, src]
        # FAR FIELD (high r) and OUTFLOW (high z): zero-gradient; the sponges do the absorbing.
        Q[:, -NG:] = Q[:, -NG - 1:-NG]
        Q[:, :, -NG:] = Q[:, :, -NG - 1:-NG]
        # BASE PLATE (z = 0): slip wall by mirroring with the AXIAL momentum reversed.
        for g, src in ((NG - 1, NG), (NG - 2, NG + 1)):
            Q[0, :, g] = Q[0, :, src]
            Q[1, :, g] = -Q[1, :, src]
            Q[2, :, g] = Q[2, :, src]
            Q[3, :, g] = Q[3, :, src]
        # NOZZLE PATCH: full Dirichlet in both ghost columns (supersonic inflow).
        if self._patch is not None:
            f = self._ramp_now()
            q = self._q_amb + f * (self._q_exh - self._q_amb)
            Q[:, self._patch, :NG] = q
        return Q

    def _ramp_now(self):
        if self.t_ramp <= 0.0:
            return self._ramp
        x = min(1.0, self.t / self.t_ramp)
        return x * x * (3.0 - 2.0 * x)                 # smoothstep, on the SOLVER's clock

    def _rhs(self, Q):
        Q = self._bcs(Q)
        r, u, v, p = self._prim(Q)
        W = xp.stack([r, u, v, p])
        d = self._sweep(W, Z_AX, self.dz, 1, 2)
        self._sweep(W, R_AX, self.dr, 2, 1, out=d, accum=True)
        # GEOMETRIC SOURCE  -(1/r) * (rho v, rho u v, rho v^2, (E+p) v).
        # Note the third component has NO pressure in it - see the module docstring.
        E = Q[3]
        rv = r * v
        s = self._inv_r
        d[0] -= s * rv
        d[1] -= s * rv * u
        d[2] -= s * rv * v
        d[3] -= s * (E + p) * v
        return d

    def _wave_speed(self, Q):
        """Fastest wave anywhere the next step will touch - INCLUDING the inflow ghosts.

        The inflow term is not a belt-and-braces margin, it is the fix for a real failure. At
        t = 0 the interior is quiescent, so a field-only estimate returns c_inf = 1 and hands
        back dt = CFL/1.15. The nozzle ghosts meanwhile carry u_e + c_e, which for
        `plume_vacuum` is 6.58 - so the first steps ran at CFL ~2.3 and both high-pressure-ratio
        scenes went NaN inside 60 steps while the two mild ones survived on the 4-step dt
        refresh. A bound taken over a field that does not yet contain the forcing is not a bound.
        """
        r, u, v, p = self._prim(Q)
        c = xp.sqrt(self.gamma * p / r)
        w = float(asnumpy(xp.maximum(xp.abs(u), xp.abs(v)).max() + c.max()))
        return max(w, self._inflow_wave)

    def step(self, dt=None):
        """One SSP-RK2 (Heun) step. Strong-stability-preserving, so the TVD limiter still holds."""
        ncol = self._ncol()
        Qa = self.Q[:, :, :ncol]
        if dt is None:
            dt = self._pick_dt(Qa)
        self.dt = dt
        Q0 = Qa.copy()
        Q1 = Q0 + dt * self._rhs(Qa)
        Qa[...] = 0.5 * Q0 + 0.5 * (Q1 + dt * self._rhs(Q1))
        # graded absorbers toward the quiescent ambient
        Qa += self._sponge_r * (self._q_amb - Qa)
        Qa += self._sponge_z_full[:, :, -ncol:] * (self._q_amb - Qa)
        self.t += dt
        self.steps_done += 1
        if self.active and self.nz_act < self.nz and self.steps_done % self._act_every == 0:
            self._grow()

    def _ncol(self):
        return self.nz_act + 2 * NG

    def _pick_dt(self, Qa):
        """CFL step, recomputed every `dt_every` steps with a margin.

        The reduction is a device sync, so doing it every step costs ~1% of a step for nothing:
        at CFL 0.4 the fastest wave in the domain cannot change by the 15% margin in four steps.
        `cfl_now()` reports what was actually achieved, so this is checkable rather than trusted.
        """
        if self.steps_done % self._dt_every == 0 or self.dt == 0.0:
            self._wave = self._wave_speed(Qa) * self._dt_margin
        return self.cfl * min(self.dz, self.dr) / max(self._wave, 1e-6)

    def _grow(self):
        """Extend the active box when the band just UPSTREAM of its sponge stops being ambient.

        Upstream of the sponge, not inside it: the absorber sits on the last `sponge_z` columns
        of the window and is strong enough to eat an arriving front before a probe placed there
        could see it - which would have hidden the growth trigger and clipped the plume at a
        fixed radius, while every stability number stayed perfect.
        """
        hi = NG + max(0, self.nz_act - self._sponge_z)
        lo = max(NG, hi - self._probe)
        if hi <= lo:
            self.nz_act = min(self.nz, self.nz_act + self._chunk)
            self.grew += 1
            return
        band = self.Q[:, NG:-NG, lo:hi]
        dev = float(asnumpy(xp.abs(band - self._q_amb).max()))
        if dev > self._act_tol:
            self.nz_act = min(self.nz, self.nz_act + self._chunk)
            self.grew += 1

    def advance_to(self, t_target):
        """Integrate to exactly `t_target`, clamping the last sub-step onto it.

        This is what makes the frame times exact: the scene asks for a wall clock, the solver
        lands on it, and nothing about the on-screen speed depends on how many steps it took.
        """
        guard = 0
        while self.t < t_target - 1e-9:
            dt = self._pick_dt(self.Q[:, :, :self._ncol()])
            self.step(min(dt, t_target - self.t))
            guard += 1
            if guard > 2_000_000:
                raise RuntimeError("advance_to: step count ran away; check dt / wave speed")

    # ------------------------------------------------------------------ fields (interior views)
    def _iv(self, a):
        return a[NG:-NG, NG:-NG]

    def prim(self):
        r, u, v, p = self._prim(self._bcs(self.Q))
        return self._iv(r), self._iv(u), self._iv(v), self._iv(p)

    def speed(self):
        _, u, v, _ = self.prim()
        return xp.sqrt(u * u + v * v)

    def mach(self):
        r, u, v, p = self.prim()
        c = xp.sqrt(self.gamma * p / xp.maximum(r, 1e-7))
        return xp.sqrt(u * u + v * v) / xp.maximum(c, 1e-6)

    def density(self):
        return self.prim()[0]

    def pressure(self):
        return self.prim()[3]

    def temperature(self):
        r, _, _, p = self.prim()
        return self.gamma * p / xp.maximum(r, 1e-7)

    def schlieren(self, k=8.0):
        """|grad rho| / rho, tone-mapped 1 - exp(-k x).

        What a real test cell shows you OF AIR: blind to velocity, sensitive only to density
        gradient. The tone map is not decoration - a shock's gradient is orders of magnitude above
        a shear layer's, and without it the frame is one white line on black.
        """
        r = self.prim()[0]
        gz = 0.5 * (xp.roll(r, -1, axis=1) - xp.roll(r, 1, axis=1)) / self.dz
        gr = 0.5 * (xp.roll(r, -1, axis=0) - xp.roll(r, 1, axis=0)) / self.dr
        g = xp.sqrt(gz * gz + gr * gr) / xp.maximum(r, 1e-7)
        return 1.0 - xp.exp(-float(k) * g)

    FIELDS = ("speed", "mach", "density", "pressure", "temperature", "schlieren")

    def field(self, name):
        if name not in self.FIELDS:
            raise ValueError(f"unknown field {name!r}; have {self.FIELDS}")
        return getattr(self, name)()

    # ------------------------------------------------------------------ diagnostics
    def health(self):
        """Peak |u| - which in these units IS the peak Mach number referred to ambient sound."""
        _, u, v, _ = self.prim()
        return float(asnumpy(xp.sqrt(u * u + v * v).max()))

    @property
    def steady_limit(self):
        """Speed reached expanding to vacuum along a STEADY streamline: sqrt(u_e^2 + 2c_e^2/(g-1)).

        Reported, never enforced. A plume that exceeds it is doing something interesting, not
        something wrong - see `health_limit`.
        """
        if self._patch is None:
            return float(np.sqrt(2.0 / (self.gamma - 1.0)))
        e = self.exit_state
        return float(np.sqrt(e["u"] ** 2 + 2.0 * e["c"] ** 2 / (self.gamma - 1.0)))

    @property
    def health_limit(self):
        """The UNSTEADY ceiling: ``u_e + 2 c_e/(gamma-1)``, the escape speed of a centred
        rarefaction. Derived from the scene's own exit state, never hardcoded.

        This started life as the steady bound above, and that was WRONG - it killed a correct
        `plume_under` render at frame 165 of 180 for a 0.6% overshoot. Total enthalpy is only
        conserved along a streamline in STEADY flow; the transport equation carries a ``dp/dt``
        source, and a starting jet is nothing but ``dp/dt``. The peak that tripped it was a
        smooth 7-cell plateau on the axis terminated by a clean Mach disk at z = 10.3 D_e, with
        0.14% of cells above the steady bound in one coherent region - i.e. real gas accelerated
        by an unsteady expansion, not a carbuncle (which is a single-cell spike with ringing).

        The unsteady bound is looser, and that is the point: `isfinite` already catches a true
        CFL runaway within a few steps, so this exists to catch large-but-finite garbage, not to
        adjudicate physics it was never given the standing to adjudicate. Same lesson as
        Wind_Tunnel AGENT_GUIDE 2a.28 - a guard that does not scale with what the scene is doing
        aborts correct work - one level deeper: a guard can also be wrong about the PHYSICS while
        scaling perfectly.
        """
        if self._patch is None:
            return 1.05 * float(2.0 / (self.gamma - 1.0))
        e = self.exit_state
        return 1.05 * float(e["u"] + 2.0 * e["c"] / (self.gamma - 1.0))

    def cfl_now(self):
        return self._wave_speed(self.Q[:, :, :self._ncol()]) * self.dt / min(self.dz, self.dr)

    def describe(self):
        e = getattr(self, "exit_state", None)
        s = (f"AxiEuler {self.nz}x{self.nr} (half domain), gamma={self.gamma}, CFL={self.cfl}, "
             f"HLLC={'fused' if self.fused else 'vectorised'}, GPU={GPU}")
        if e:
            s += (f"\n  nozzle: R_e={e['r_exit']:.1f} cells, M_e={e['mach']:.2f}, "
                  f"u_e={e['u']:.3f} c_inf, p_e/p_inf={e['p'] / self.p_inf:.2f}")
        return s
