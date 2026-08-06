"""cns.py - the COMPRESSIBLE solver: 2-D Navier-Stokes, finite volume, HLLC + MUSCL.

`lbm.py` is the engine this project was built on and is still the right one for almost every
scene here. This module exists for the one thing it structurally cannot do.

WHY A SECOND SOLVER AT ALL
--------------------------
D2Q9 lattice-Boltzmann is a WEAKLY COMPRESSIBLE, ISOTHERMAL model. Three consequences, and each
one is a wall rather than a setting:

  - its equation of state is p = rho*cs^2, i.e. an isothermal gas with **gamma = 1**. There is no
    temperature, no internal energy, no thermodynamics;
  - its error is O(Ma^2) and it has no shock-capturing mechanism, so it degrades continuously and
    then fails above **M ~ 0.3**;
  - the lattice sound speed is fixed at 1/sqrt(3) in grid units, so "faster" always means "closer
    to the model's own ceiling".

So a supersonic pocket, a shock, a real stagnation temperature - the phenomena that make air
unmistakably a GAS rather than a liquid - are unreachable in `lbm.py` at any parameter setting.
That was the open item at the end of `tri_foil_air`, and this module closes it.

WHY THIS SCHEME AND NOT D2Q37
-----------------------------
The in-keeping answer would be a higher-order lattice (D2Q37), which restores thermodynamics and
raises the Mach ceiling. It is also research-grade: an order-9 Gauss-Hermite quadrature, a 4th
order Hermite equilibrium, streaming with velocity components of +-3 (so no longer a single-cell
shift), and it still needs a separate shock-capturing filter bolted on to stop Gibbs oscillations
ringing off a discontinuity. Getting one stable is a project.

A GODUNOV FINITE-VOLUME scheme gets the same physics by a route that is textbook, robustly
stable, and just as vectorisable:

  - **conservative form** in (rho, rho*u, rho*v, E) with a real ideal-gas EOS, gamma = 1.4, so
    energy and thermodynamics are carried explicitly rather than emerging in a limit;
  - **MUSCL reconstruction** with a minmod limiter -> 2nd order in smooth flow, TVD across a
    discontinuity, which is what stops a shock from ringing;
  - an **HLLC approximate Riemann solver** at every face. HLLC restores the contact wave that
    HLL smears, which matters here specifically: a shear layer and a wake ARE contact
    discontinuities, and an HLL solve would diffuse the vortex sheet this whole project exists to
    show;
  - **SSP-RK2** in time, which preserves the TVD property of the spatial operator;
  - explicit **Navier-Stokes viscous terms** (full stress tensor + Fourier conduction, Pr = 0.71)
    plus a Smagorinsky eddy viscosity - so the boundary layer and the wake are real viscous
    features, not numerical dissipation.

Shocks are then captured by the scheme itself, at the correct jump conditions, with no filter and
no tuning: that is the entire point of a conservative Godunov method (Lax-Wendroff theorem - a
conservative, consistent, convergent scheme converges to the right weak solution, so the
Rankine-Hugoniot relations hold automatically).

UNITS - and they are NOT the lattice's
--------------------------------------
Non-dimensionalised on the freestream: **rho_inf = 1, p_inf = 1/gamma**, hence

    c_inf = sqrt(gamma * p_inf / rho_inf) = 1

so **a velocity in this solver IS a Mach number**, exactly, with no conversion. `u0 = 0.19` means
M 0.19. Contrast `lbm.py`, where `u0 = 0.055` is a lattice velocity and the Mach number is
`u0 / (1/sqrt(3))`. This is the single most likely thing to get wrong when porting a scene
between the two - a scene's `u0` means a different physical thing in each. Length unit is one
cell, time unit is one cell per freestream sound speed.

THE TIME STEP IS FIXED, ON PURPOSE
----------------------------------
The natural thing for a compressible code is an adaptive dt from the instantaneous CFL. This one
computes dt ONCE and holds it, because the whole project's timebase rests on `cfg.steps` being
LBM/solver steps per frame - apparent speed is (distance per step x steps per frame), and a dt
that wandered with the solution would make the on-screen wind speed a function of the flow, which
is precisely the kind of unfalsifiable animation this library refuses to produce. `dt` is set
from a conservative estimate of max(|u| + c) with a margin, and `cfl_now()` reports the achieved
CFL so a run can be checked rather than assumed - `tools\\cns_check.py` does exactly that.

INTERFACE
---------
Deliberately the same public surface as `LBM` (`set_solid`, `run`, `speed`, `vorticity`,
`pressure`, `health`, `set_inlet`, `set_farfield`, `init_velocity`, `set_inlet_turbulence`,
`perturb`, `.ux`, `.uy`, `.rho`, `.solid`) so `render.py` picks one with a config flag and
everything downstream - the streaks, the colour mapping, the legend, the body draw - is unchanged.
What is NOT implemented raises rather than guessing: `force_field` (free bodies would need a
different force integral here, and this project has already been bitten once by a plausible-
looking wrong drag - AGENT_GUIDE 2a.1) and `set_drive` (a blue-pool device with no meaning in a
compressible tunnel).
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy, map_coordinates, xp

GAMMA = 1.4
PR = 0.71                  # Prandtl number for air
NG = 2                     # ghost layers; MUSCL reaches 2 cells
_TINY = 1e-9


def _minmod(a, b):
    """The limiter. Returns whichever slope is smaller in magnitude, or 0 if they disagree.

    The zero branch is the whole point: at an extremum the two one-sided slopes have opposite
    signs, the reconstruction collapses to 1st order there, and the scheme stays TVD - which is
    what stops a captured shock from ringing. Do not "improve" this to a smoother limiter without
    re-running tools\\sod_check.py.
    """
    return xp.where(a * b > 0.0, xp.where(xp.abs(a) < xp.abs(b), a, b), 0.0)


class CNS:
    """A compressible Navier-Stokes lattice. `solid` is a (ny, nx) mask that may change per frame.

    Arrays are stored with `NG` ghost cells on every side; `nx`/`ny` are the INTERIOR size, and
    every public field property returns the interior view only, so nothing outside this file has
    to know the padding exists.
    """

    def __init__(self, nx, ny, u0=0.19, re=16000.0, ref_len=100.0, csm=0.20,
                 cfl=0.50, sponge_out=21, side_bc="slip", wall_clamp=0.30,
                 closed=False, blockage=4.2):
        if closed:
            raise NotImplementedError(
                "CNS has no sealed-box mode; the blue-pool scenes are LBM scenes (cfg.solver)")
        self.nx, self.ny = int(nx), int(ny)
        self.Nx, self.Ny = self.nx + 2 * NG, self.ny + 2 * NG
        self.u0 = float(u0)                      # freestream MACH number (c_inf = 1)
        self.re = float(re)
        # mu from the Reynolds number on the scene's reference length, at rho_inf = 1
        self.nu = max(1e-9, self.u0 * float(ref_len) / max(self.re, 1.0))
        self.mu = self.nu                        # rho_inf = 1, so mu and nu coincide
        self.csm2 = float(csm) ** 2
        self.side_bc = side_bc
        self.wall_clamp = float(wall_clamp)
        self.steps_done = 0

        # FIXED dt (see the module docstring). The estimate has to bound max(|u| + c) over the
        # whole clip, and the term that actually bites is BLOCKAGE, not the freestream: with
        # free-slip walls the stream squeezing past three foils at incidence reaches ~3-4x the
        # band speed (AGENT_GUIDE 2a.20), and a compressible stagnation region raises c by ~10%
        # on top. Under-estimate this and the run violates CFL and dies; over-estimate and the
        # only cost is render time. `cfl_now()` reports what was actually achieved.
        self.cfl = float(cfl)
        self.umax_design = float(blockage) * self.u0
        self.cmax_design = 1.0 + 0.5 * (GAMMA - 1.0) * (2.0 * self.u0) ** 2 + 0.10
        self.dt = self.cfl / (self.umax_design + self.cmax_design)

        f32 = xp.float32
        self.Q = xp.zeros((4, self.Ny, self.Nx), dtype=f32)
        self._solid_g = xp.zeros((self.Ny, self.Nx), dtype=bool)
        self.p_inf = 1.0 / GAMMA
        self._set_uniform(self.u0)

        self._in_ux = self._in_uy = None          # shaped inlet, INTERIOR rows (ny,)
        self._ff_ux = None                        # sponge target profile
        self._wux = self._wuy = None              # moving-wall velocity field (ny, nx)
        # inlet turbulence (see set_inlet_turbulence)
        self._turb_on = False
        self._turb_du = self._turb_dv = None
        self._turb_amp = 0.0
        self._turb_n = 0
        self._turb_c = 0.0
        # immersed-boundary ghost cache, rebuilt whenever the mask changes
        self._gy = self._gx = self._gcoord = None
        self._gnx = self._gny = self._gfac = None

        # graded outlet sponge, same device and same justification as the LBM one: a hard
        # zero-gradient exit reflects, and in a COMPRESSIBLE solve what reflects is an acoustic
        # wave that crosses the whole picture. Interior-shaped; broadcast over the 4 components.
        w = np.zeros((1, 1, self.Nx), np.float32)
        if sponge_out > 0:
            d = (self.Nx - 1 - NG - np.arange(self.Nx, dtype=np.float32)) / float(sponge_out)
            w[0, 0, :] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        self._sponge = xp.asarray(w * 0.25)
        self._q_far = None
        self.set_farfield(None)

        self.rho = self._interior(self.Q[0]).copy()
        self.ux = xp.full((self.ny, self.nx), self.u0, dtype=f32)
        self.uy = xp.zeros((self.ny, self.nx), dtype=f32)
        self._p = xp.full((self.ny, self.nx), self.p_inf, dtype=f32)

    # -- small helpers ------------------------------------------------------------------
    @staticmethod
    def _interior(a):
        return a[..., NG:-NG, NG:-NG]

    def _set_uniform(self, u):
        r = xp.ones((self.Ny, self.Nx), dtype=xp.float32)
        self.Q[0] = r
        self.Q[1] = r * u
        self.Q[2] = 0.0
        self.Q[3] = self.p_inf / (GAMMA - 1.0) + 0.5 * r * u * u

    @staticmethod
    def _prim(Q):
        """(rho, u, v, p) from the conservative state, floored so a transient cannot make NaN."""
        r = xp.maximum(Q[0], 1e-6)
        u = Q[1] / r
        v = Q[2] / r
        p = xp.maximum((GAMMA - 1.0) * (Q[3] - 0.5 * r * (u * u + v * v)), 1e-8)
        return r, u, v, p

    @staticmethod
    def _cons(r, u, v, p):
        return xp.stack([r, r * u, r * v,
                         p / (GAMMA - 1.0) + 0.5 * r * (u * u + v * v)])

    @property
    def solid(self):
        return self._solid_g[NG:-NG, NG:-NG]

    # -- geometry -----------------------------------------------------------------------
    def set_wall_velocity(self, wux, wuy):
        """Velocity of the moving wall at each solid cell (None = everything stationary)."""
        if wux is None:
            self._wux = self._wuy = None
            return
        lim = self.wall_clamp
        self._wux = xp.clip(xp.asarray(wux, dtype=xp.float32), -lim, lim)
        self._wuy = xp.clip(xp.asarray(wuy, dtype=xp.float32), -lim, lim)

    def set_solid(self, mask):
        """Swap in a new body mask and REBUILD the immersed-boundary ghost cache.

        The body is imposed by GHOST CELLS, not by bounce-back: for every solid cell within reach
        of the reconstruction stencil we mirror the fluid state across the surface, so the cells
        the scheme reads outside the fluid carry the values that make the wall condition hold.
        Everything needed to do that - which cells, where their image points are, the surface
        normal there, and the extrapolation factor - depends only on the MASK, so it is computed
        once here rather than at every one of the ~120 steps in a frame.

        The signed distance comes from two Euclidean distance transforms (`edt(solid)` is the
        distance from a solid cell out to the fluid, `edt(~solid)` the reverse), and the normal is
        its gradient. Both are exact for any polygon, which is what preserves the property that
        made this project's LBM worth using: an arbitrary shape, re-rasterised as it rotates,
        needs no mesh.
        """
        from scipy.ndimage import distance_transform_edt as _edt
        m = np.zeros((self.Ny, self.Nx), bool)
        m[NG:-NG, NG:-NG] = np.asarray(asnumpy(mask), bool)
        self._solid_g = xp.asarray(m)
        if not m.any():
            self._gy = self._gx = self._gcoord = None
            return
        d_in = _edt(m).astype(np.float32)               # solid cell -> nearest fluid cell
        d_out = _edt(~m).astype(np.float32)             # fluid cell -> nearest solid cell
        phi = d_out - d_in                              # >0 in fluid, <0 in solid
        gy, gx = np.gradient(phi)                       # points from solid toward fluid
        nrm = np.sqrt(gy * gy + gx * gx) + 1e-6
        gy, gx = gy / nrm, gx / nrm

        band = m & (d_in <= NG + 0.5)                   # only cells the stencil can reach
        jj, ii = np.nonzero(band)
        # HALF-CELL CORRECTION. `d_in` is the distance from a solid cell centre to the nearest
        # FLUID CELL CENTRE; what the wall condition needs is the distance to the SURFACE, which
        # lies about half a cell short of that. Kept because it is the correct definition - but
        # it is NOT the fix for the over-turn documented below, and the measurement says so:
        # tools\\shock_check.py (M=2, 15 deg wedge, inviscid limit) gives +1.30 deg over-turn
        # without it and +1.29 deg with it. What remains is the STAIRCASE itself - a 15 deg ramp
        # steps one row every 3.7 columns, and the EDT normals along it oscillate between the
        # tread and riser orientations. Quantified rather than removed: the immersed boundary
        # turns the flow ~8.6% more than the geometry asks, essentially independent of Reynolds
        # number (checked over Re 2e4-2e7) and only weakly of resolution. Fixing it properly
        # means a cut-cell or level-set surface representation, not a tweak here.
        d = np.maximum(d_in[jj, ii] - 0.5, 0.05)
        ny_, nx_ = gy[jj, ii], gx[jj, ii]
        # Image point at a FIXED stand-off h from the surface rather than at the mirror distance.
        # The mirror form (h = d) is the textbook one and it breaks on a thin body: near the
        # trailing edge this foil is only a few cells thick, and a deep ghost cell's mirror lands
        # on the far side, in the fluid of the OTHER surface. Pinning h >= 1.75 keeps the image in
        # the fluid this cell actually belongs to, and taking h = max(1.75, d) also caps the
        # extrapolation factor d/h at 1, so a deep ghost can never amplify the image state.
        h = np.maximum(1.75, d)
        self._gy = xp.asarray(jj.astype(np.int32))
        self._gx = xp.asarray(ii.astype(np.int32))
        self._gnx = xp.asarray(nx_)
        self._gny = xp.asarray(ny_)
        self._gfac = xp.asarray((d / h).astype(np.float32))
        yi = (jj + (d + h) * ny_).astype(np.float32)
        xi = (ii + (d + h) * nx_).astype(np.float32)
        n = len(jj)
        # one map_coordinates call over the stacked (4, Ny, Nx) primitive array: the leading
        # coordinate selects the field, so all four are sampled in a single kernel launch
        fi = np.repeat(np.arange(4, dtype=np.float32), n)
        self._gcoord = xp.asarray(np.stack([fi, np.tile(yi, 4), np.tile(xi, 4)]))
        self._gn = n

    def perturb(self, amp=0.02, seed=7, scale=6.0):
        """One-shot smooth transverse velocity - the symmetry breaker (see LBM.perturb)."""
        if amp <= 0:
            return
        from scipy.ndimage import gaussian_filter as _gf
        n = np.random.default_rng(seed).standard_normal((self.Ny, self.Nx)).astype(np.float32)
        n = _gf(n, scale)
        n = xp.asarray(n / (float(np.abs(n).max()) + 1e-9))
        r, u, v, p = self._prim(self.Q)
        self.Q = self._cons(r, u, v + amp * self.u0 * n, p)

    # -- forcing / boundaries ------------------------------------------------------------
    def set_inlet(self, ux_profile=None, uy_profile=None):
        """Override the inlet velocity across the span - arrays of length ny (interior rows)."""
        self._in_ux = None if ux_profile is None else xp.asarray(ux_profile, dtype=xp.float32)
        self._in_uy = None if uy_profile is None else xp.asarray(uy_profile, dtype=xp.float32)

    def set_farfield(self, ux_profile=None):
        """Retarget the outlet sponge to a spanwise profile (see LBM.set_farfield - same trap).

        A sponge relaxing toward one uniform `u0` is a pump bolted onto the exit plane of any
        scene whose inlet is banded, and it erodes the bands over the clip while every stability
        number stays perfect. Feed it the same profile that went into `set_inlet`.
        """
        f32 = xp.float32
        u = xp.full((self.Ny, 1), self.u0, dtype=f32)
        if ux_profile is not None:
            col = xp.asarray(ux_profile, dtype=f32).reshape(-1, 1)
            u = xp.concatenate([xp.repeat(col[:1], NG, axis=0), col,
                                xp.repeat(col[-1:], NG, axis=0)], axis=0)
            self._ff_ux = xp.asarray(ux_profile, dtype=f32)
        r = xp.ones((self.Ny, 1), dtype=f32)
        z = xp.zeros((self.Ny, 1), dtype=f32)
        p = xp.full((self.Ny, 1), self.p_inf, dtype=f32)
        self._q_far = self._cons(r, u, z, p)[:, :, :]     # (4, Ny, 1), broadcasts over x

    def set_inlet_turbulence(self, intensity=0.0, length=12.0, span=3072, seed=11, u_conv=None):
        """Continuous freestream turbulence at the inlet - see LBM.set_inlet_turbulence.

        Identical construction (a frozen divergence-free patch convected past the inlet plane),
        and divergence-free matters MORE here than it did there: this solver resolves acoustics
        properly, so the compressive part of a careless inflow perturbation would not merely add
        noise, it would radiate a pressure wave the scheme is good enough to keep.
        """
        arr = np.asarray(intensity, dtype=np.float32)
        if float(np.max(np.abs(arr))) <= 0.0:
            self._turb_on = False
            return
        from scipy.ndimage import gaussian_filter as _gf
        n = int(span)
        rng = np.random.default_rng(seed)
        psi = _gf(rng.standard_normal((self.ny, n)).astype(np.float32), float(length), mode="wrap")
        du = np.gradient(psi, axis=0)
        dv = -np.gradient(psi, axis=1)
        rms = float(np.sqrt(np.mean(du * du + dv * dv))) + 1e-12
        self._turb_du = xp.asarray(du / rms)
        self._turb_dv = xp.asarray(dv / rms)
        self._turb_n = n
        amp = arr * self.u0
        self._turb_amp = float(amp) if amp.ndim == 0 else xp.asarray(amp.reshape(-1))
        self._turb_c = float(self.u0 if u_conv is None else u_conv)
        self._turb_on = True

    def init_velocity(self, ux, uy):
        """Start from a given velocity field, at freestream density and pressure."""
        u = xp.zeros((self.Ny, self.Nx), dtype=xp.float32)
        v = xp.zeros((self.Ny, self.Nx), dtype=xp.float32)
        u[NG:-NG, NG:-NG] = xp.asarray(ux, dtype=xp.float32)
        v[NG:-NG, NG:-NG] = xp.asarray(uy, dtype=xp.float32)
        u[:NG] = u[NG:NG + 1]
        u[-NG:] = u[-NG - 1:-NG]
        u[:, :NG] = u[:, NG:NG + 1]
        u[:, -NG:] = u[:, -NG - 1:-NG]
        r = xp.ones((self.Ny, self.Nx), dtype=xp.float32)
        p = xp.full((self.Ny, self.Nx), self.p_inf, dtype=xp.float32)
        self.Q = self._cons(r, u, v, p)

    def set_drive(self, *a, **k):
        raise NotImplementedError("set_drive is an LBM blue-pool device; use cfg.solver='lbm'")

    def force_field(self):
        raise NotImplementedError(
            "CNS has no verified body-force integral yet, so free-body scenes must stay on the "
            "LBM (cfg.solver='lbm'). Implementing it means integrating pressure + viscous "
            "traction over the immersed surface and VALIDATING it against a known Cd, the way "
            "tools\\calib.py did for the LBM - a plausible-looking wrong force is exactly the "
            "bug that measured Cd~22 in this project once already (AGENT_GUIDE 2a.1).")

    # -- the scheme ----------------------------------------------------------------------
    def _bcs(self, Q):
        """Ghost cells for inlet / outlet / tunnel walls, in place."""
        # INLET: fixed primitive state, held for both ghost columns
        u_in = self.u0 if self._in_ux is None else self._in_ux.reshape(-1, 1)
        v_in = 0.0 if self._in_uy is None else self._in_uy.reshape(-1, 1)
        if self._turb_on:
            p = (self.steps_done * self._turb_c * self.dt) % self._turb_n
            i0 = int(p)
            fr = p - i0
            i1 = (i0 + 1) % self._turb_n
            a = self._turb_amp
            du = (1.0 - fr) * self._turb_du[:, i0] + fr * self._turb_du[:, i1]
            dv = (1.0 - fr) * self._turb_dv[:, i0] + fr * self._turb_dv[:, i1]
            u_in = u_in + (a * du).reshape(-1, 1)
            v_in = v_in + (a * dv).reshape(-1, 1)
        r1 = xp.ones((self.ny, 1), dtype=xp.float32)
        qin = self._cons(r1, xp.broadcast_to(xp.asarray(u_in, dtype=xp.float32), (self.ny, 1)),
                         xp.broadcast_to(xp.asarray(v_in, dtype=xp.float32), (self.ny, 1)),
                         xp.full((self.ny, 1), self.p_inf, dtype=xp.float32))
        Q[:, NG:-NG, :NG] = qin
        # OUTLET: zero-gradient (the sponge does the absorbing)
        Q[:, :, -NG:] = Q[:, :, -NG - 1:-NG]
        # SIDES: free-slip tunnel walls by mirroring, with the wall-normal momentum reversed.
        # Same choice and same reason as the LBM (AGENT_GUIDE 2a.13): imposing a velocity on the
        # edge rows draws a hard stripe wherever a body blocks the tunnel. Blockage is therefore
        # REAL here too - which is why `blockage` feeds the dt estimate above.
        for g, src in ((NG - 1, NG), (NG - 2, NG + 1)):
            Q[0, g] = Q[0, src]
            Q[1, g] = Q[1, src]
            Q[2, g] = -Q[2, src]
            Q[3, g] = Q[3, src]
        for g, src in ((-NG, -NG - 1), (-NG + 1, -NG - 2)):
            Q[0, g] = Q[0, src]
            Q[1, g] = Q[1, src]
            Q[2, g] = -Q[2, src]
            Q[3, g] = Q[3, src]
        return Q

    def _ibm(self, Q):
        """Impose the body by writing mirrored states into its near-surface cells, in place."""
        if self._gcoord is None:
            return Q
        r, u, v, p = self._prim(Q)
        W = xp.stack([r, u, v, p])
        s = map_coordinates(W, self._gcoord, order=1, mode="nearest").reshape(4, self._gn)
        ri, ui, vi, pi = s[0], s[1], s[2], s[3]
        if self._wux is not None:
            uw = self._wux[self._gy - NG, self._gx - NG]
            vw = self._wuy[self._gy - NG, self._gx - NG]
        else:
            uw = vw = 0.0
        # NO-SLIP by linear extrapolation through the wall: the ghost value is chosen so that a
        # straight line from the image point to the ghost cell passes through the wall velocity
        # AT the surface. `_gfac` = d/h <= 1 is the lever arm.
        ug = uw - self._gfac * (ui - uw)
        vg = vw - self._gfac * (vi - vw)
        # ADIABATIC wall, zero normal pressure gradient: carry density and pressure across
        # unchanged. Imposing a temperature instead would be a heat source the scene never asked
        # for - and at these Mach numbers the stagnation heating IS part of what we are showing.
        qg = self._cons(ri, ug, vg, pi)
        Q[:, self._gy, self._gx] = qg
        return Q

    def _visc_mu(self, r, u, v):
        """Molecular + Smagorinsky eddy viscosity at cell centres."""
        ux = 0.5 * (xp.roll(u, -1, axis=1) - xp.roll(u, 1, axis=1))
        uy = 0.5 * (xp.roll(u, -1, axis=0) - xp.roll(u, 1, axis=0))
        vx = 0.5 * (xp.roll(v, -1, axis=1) - xp.roll(v, 1, axis=1))
        vy = 0.5 * (xp.roll(v, -1, axis=0) - xp.roll(v, 1, axis=0))
        sxy = 0.5 * (uy + vx)
        smag = xp.sqrt(2.0 * (ux * ux + vy * vy + 2.0 * sxy * sxy))
        return self.mu + r * self.csm2 * smag

    def _flux_dir(self, r, u, v, p, mu, axis):
        """Net inviscid + viscous flux difference along one axis. `u` is the NORMAL velocity."""
        W = xp.stack([r, u, v, p])
        s = _minmod(W - xp.roll(W, 1, axis=axis + 1), xp.roll(W, -1, axis=axis + 1) - W)
        WL = W + 0.5 * s
        WR = xp.roll(W - 0.5 * s, -1, axis=axis + 1)
        # Floor the RECONSTRUCTED density and pressure, not just the cell averages. minmod keeps
        # the reconstruction monotone but not positive, and a single negative pressure inside a
        # strong expansion makes `sqrt(gamma p / rho)` NaN and takes the whole field with it in
        # one step. The floor only ever engages where the limiter has already given up.
        WL = xp.concatenate([xp.maximum(WL[:1], 1e-6), WL[1:3], xp.maximum(WL[3:], 1e-8)])
        WR = xp.concatenate([xp.maximum(WR[:1], 1e-6), WR[1:3], xp.maximum(WR[3:], 1e-8)])
        F = self._hllc(WL[0], WL[1], WL[2], WL[3], WR[0], WR[1], WR[2], WR[3])

        # --- viscous flux at the same faces (central, so it stays 2nd order and conservative)
        ax = axis + 0            # 0 = rows (y), 1 = cols (x) for the (Ny, Nx) scalars
        other = 1 - ax
        rf = xp.roll(u, -1, axis=ax) - u                     # d(un)/dn across the face
        vn = xp.roll(v, -1, axis=ax) - v                     # d(ut)/dn
        # tangential derivatives: average the two cells straddling the face
        def dt_(a):
            d = 0.5 * (xp.roll(a, -1, axis=other) - xp.roll(a, 1, axis=other))
            return 0.5 * (d + xp.roll(d, -1, axis=ax))
        ut = dt_(u)                                          # d(un)/dt
        vt = dt_(v)                                          # d(ut)/dt
        muf = 0.5 * (mu + xp.roll(mu, -1, axis=ax))
        div = rf + vt
        tnn = muf * (2.0 * rf - (2.0 / 3.0) * div)
        tnt = muf * (vn + ut)
        uf = 0.5 * (u + xp.roll(u, -1, axis=ax))
        vf = 0.5 * (v + xp.roll(v, -1, axis=ax))
        # Fourier conduction on T = p/rho (so T_inf = p_inf/rho_inf); cp/(gamma-1) folded in
        tt = p / r
        kf = muf * GAMMA / ((GAMMA - 1.0) * PR)
        q = kf * (xp.roll(tt, -1, axis=ax) - tt)
        G = xp.stack([xp.zeros_like(tnn), tnn, tnt, uf * tnn + vf * tnt + q])

        net = (F - G) - xp.roll(F - G, 1, axis=axis + 1)
        return net

    @staticmethod
    def _hllc(rL, uL, vL, pL, rR, uR, vR, pR):
        """HLLC flux for the 1-D Euler equations, `u` normal and `v` tangential to the face.

        Three waves: the two acoustic waves bounding the fan (SL, SR) and the CONTACT (S*) in
        between. HLL drops the middle one and smears every contact discontinuity over several
        cells - which in this scene would mean smearing the shear layers and the wake, i.e. the
        subject. Wave speeds use the Davis estimate, which is cheap and provably bounding for an
        ideal gas.
        """
        cL = xp.sqrt(GAMMA * pL / rL)
        cR = xp.sqrt(GAMMA * pR / rR)
        SL = xp.minimum(uL - cL, uR - cR)
        SR = xp.maximum(uL + cL, uR + cR)
        EL = pL / (GAMMA - 1.0) + 0.5 * rL * (uL * uL + vL * vL)
        ER = pR / (GAMMA - 1.0) + 0.5 * rR * (uR * uR + vR * vR)
        mL = rL * (SL - uL)
        mR = rR * (SR - uR)
        den = mL - mR
        Ss = (pR - pL + mL * uL - mR * uR) / xp.where(xp.abs(den) < _TINY, _TINY, den)

        FL = xp.stack([rL * uL, rL * uL * uL + pL, rL * uL * vL, (EL + pL) * uL])
        FR = xp.stack([rR * uR, rR * uR * uR + pR, rR * uR * vR, (ER + pR) * uR])
        UL = xp.stack([rL, rL * uL, rL * vL, EL])
        UR = xp.stack([rR, rR * uR, rR * vR, ER])

        def star(rK, uK, vK, pK, EK, SK, mK):
            f = rK * (SK - uK) / xp.where(xp.abs(SK - Ss) < _TINY, _TINY, SK - Ss)
            e = EK / rK + (Ss - uK) * (Ss + pK / xp.where(xp.abs(mK) < _TINY, _TINY, mK))
            return xp.stack([f, f * Ss, f * vK, f * e])

        FsL = FL + SL * (star(rL, uL, vL, pL, EL, SL, mL) - UL)
        FsR = FR + SR * (star(rR, uR, vR, pR, ER, SR, mR) - UR)
        return xp.where(SL >= 0.0, FL,
                        xp.where(Ss >= 0.0, FsL, xp.where(SR >= 0.0, FsR, FR)))

    def _rhs(self, Q):
        """dQ/dt for the whole domain, with boundaries and the body already imposed."""
        Q = self._ibm(self._bcs(Q))
        r, u, v, p = self._prim(Q)
        mu = self._visc_mu(r, u, v)
        dQ = -self._flux_dir(r, u, v, p, mu, 1)                 # x sweep
        # y sweep: hand the solver the NORMAL velocity first, then swap the momenta back
        dy = self._flux_dir(r, v, u, p, mu, 0)
        dQ = dQ - xp.stack([dy[0], dy[2], dy[1], dy[3]])
        return dQ

    def step(self):
        """One SSP-RK2 (Heun) step. Strong-stability-preserving, so the TVD limiter still holds."""
        Q0 = self.Q
        Q1 = Q0 + self.dt * self._rhs(Q0)
        self.Q = 0.5 * Q0 + 0.5 * (Q1 + self.dt * self._rhs(Q1))
        # graded outlet absorber, toward the (possibly banded) freestream state
        self.Q = self.Q + self._sponge * (self._q_far - self.Q)
        self.steps_done += 1
        r, u, v, p = self._prim(self._bcs(self.Q))
        self.rho = self._interior(r)
        self.ux = self._interior(u)
        self.uy = self._interior(v)
        self._p = self._interior(p)

    def run(self, n):
        for _ in range(int(n)):
            self.step()

    # -- derived fields ------------------------------------------------------------------
    def speed(self):
        u = xp.sqrt(self.ux * self.ux + self.uy * self.uy)
        return xp.where(self.solid, 0.0, u)

    def mach(self):
        """Local Mach number - the field this solver exists to be able to compute."""
        c = xp.sqrt(GAMMA * self._p / xp.maximum(self.rho, 1e-6))
        return xp.where(self.solid, 0.0, self.speed() / xp.maximum(c, 1e-6))

    def schlieren(self):
        """|grad rho| / rho - the classic compressible-flow visualisation.

        A schlieren image is what a wind tunnel actually shows you of AIR: it is blind to
        velocity and sensitive only to density gradient, so it renders the things that only a
        compressible fluid has - shocks as hairline discontinuities, expansion fans, and the
        density wake behind a body. Normalised by rho so a weak wave in low-density fluid reads
        as strongly as the same wave in dense fluid.
        """
        gy = 0.5 * (xp.roll(self.rho, -1, axis=0) - xp.roll(self.rho, 1, axis=0))
        gx = 0.5 * (xp.roll(self.rho, -1, axis=1) - xp.roll(self.rho, 1, axis=1))
        g = xp.sqrt(gx * gx + gy * gy) / xp.maximum(self.rho, 1e-6)
        return xp.where(self.solid, 0.0, g)

    def vorticity(self):
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        return xp.where(self.solid, 0.0, dvdx - dudy)

    def pressure(self):
        return xp.where(self.solid, 0.0, self._p - self.p_inf)

    def temperature(self):
        return xp.where(self.solid, 0.0, GAMMA * self._p / xp.maximum(self.rho, 1e-6))

    # -- diagnostics ----------------------------------------------------------------------
    def health(self):
        """Max |u| - which in these units IS the peak local Mach number."""
        return float(asnumpy(xp.sqrt(self.ux ** 2 + self.uy ** 2).max()))

    @property
    def health_limit(self):
        """Abort threshold. NOT the LBM's 0.45: here a velocity of 1 is Mach 1, and a local
        supersonic pocket is the physics this solver was written for, not a failure. What is
        genuinely wrong is a runaway, so the guard sits well above anything this tunnel can
        legitimately produce."""
        return 3.0

    def cfl_now(self):
        """Achieved CFL this instant. The fixed dt is only safe while this stays under ~1."""
        c = xp.sqrt(GAMMA * self._p / xp.maximum(self.rho, 1e-6))
        return float(asnumpy((xp.abs(self.ux) + xp.abs(self.uy) + c).max())) * self.dt

    def describe(self):
        return (f"CNS compressible (gamma={GAMMA}), M_inf={self.u0:.4f}, "
                f"mu={self.mu:.6f} (Re={self.re:g}), dt={self.dt:.4f}, CFL_design={self.cfl:g}")
