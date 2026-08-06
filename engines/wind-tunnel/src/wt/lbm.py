"""lbm.py — the physics: D2Q9 lattice-Boltzmann incompressible flow with an LES closure.

This is the engine that reproduces the source video's behaviour: a real, unsteady, separating
flow — attached flow at low incidence, a suction peak over the upper surface, a boundary layer
that separates and rolls into shed vortices at stall, and a von Karman street behind a bluff
body. None of it is scripted; it falls out of the solver.

WHY LBM (and not stable-fluids / a pressure-projection solver):
  - the whole update is local + a shift, so it is a handful of vectorised array ops -> CuPy
    turns it into a ~2 ms/step kernel chain on the RTX 2060 (GPU-default policy);
  - solid walls are HALF-WAY BOUNCE-BACK on a boolean mask, so an arbitrary shape (any polygon,
    changing every frame as the foil rotates) needs no meshing at all;
  - vortex shedding is captured naturally at the low viscosities we want.

Model
-----
D2Q9, weights w, velocity set c (index 0 = rest). BGK collision with a **Smagorinsky** subgrid
eddy viscosity: the target Reynolds numbers put the molecular tau at ~0.505, where plain BGK
goes unstable and speckles. The LES term raises tau locally wherever the strain rate is large
(exactly where the instability starts), which is both physically right and what keeps a 6000-Re
render clean. `Pi` below is the non-equilibrium momentum-flux tensor: it is available for free
from `f - feq`, which is why this closure costs almost nothing.

Boundaries
----------
  inlet  (i=0)      full equilibrium at (rho=1, u=(u0,0))       — clean uniform freestream
                    set_inlet() shapes it across the span; set_inlet_turbulence() adds a
                    continuously-convecting divergence-free fluctuation on top (real tunnel air)
  outlet (i=nx-1)   zero-gradient on the inward-facing populations — non-reflecting enough
  sides  (j=0,-1)   FREE-SLIP walls by specular reflection (side_bc="slip", default): inviscid,
                    so no wall boundary layer grows, and — unlike forcing the edge rows to the
                    freestream — no velocity is imposed, so nothing stripes the frame edge.
                    side_bc="free" instead relaxes a graded band toward the freestream.
  body              half-way bounce-back on `solid`

Units are lattice units throughout: dx = dt = 1, cs^2 = 1/3, nu = (tau - 1/2)/3.
"""
from __future__ import annotations

import numpy as np

from .gpu import asnumpy, xp

# --- D2Q9 lattice ------------------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
              1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
LEFTWARD = np.flatnonzero(CX < 0)      # populations entering the domain through the outlet


def equilibrium(rho, ux, uy, cx, cy, w):
    """f_k^eq = w_k rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u^2), arrays shaped (9, ny, nx)."""
    cu = 3.0 * (cx * ux + cy * uy)
    usq = 1.5 * (ux * ux + uy * uy)
    return w * rho * (1.0 + cu + 0.5 * cu * cu - usq)


class LBM:
    """A D2Q9 lattice. `solid` is a (ny, nx) boolean mask that may change every frame."""

    def __init__(self, nx, ny, u0=0.10, re=6000.0, ref_len=100.0, csm=0.16,
                 sponge_side=0, sponge_out=14, side_bc="slip", wall_clamp=0.08,
                 closed=False):
        self.nx, self.ny = int(nx), int(ny)
        self.u0 = float(u0)
        nu = max(1e-5, u0 * ref_len / max(re, 1.0))
        self.tau = 3.0 * nu + 0.5
        self.csm2 = float(csm) ** 2
        self.re, self.nu = float(re), nu

        f32 = xp.float32
        self.cx = xp.asarray(CX.reshape(9, 1, 1), dtype=f32)
        self.cy = xp.asarray(CY.reshape(9, 1, 1), dtype=f32)
        self.w = xp.asarray(W.reshape(9, 1, 1), dtype=f32)
        self._opp = xp.asarray(OPP)

        # uniform freestream state, reused for the inlet column and the open side rows
        rho1 = xp.ones((1, 1), dtype=f32)
        ux1 = xp.full((1, 1), self.u0, dtype=f32)
        uy1 = xp.zeros((1, 1), dtype=f32)
        # (9, 1, 1) by default; set_farfield() may replace it with a (9, ny, 1) spanwise profile,
        # so nothing may assume this is one value per population.
        self._feq_free = equilibrium(rho1, ux1, uy1, self.cx, self.cy, self.w).reshape(9, 1, 1)

        self.side_bc = side_bc
        self.wall_clamp = float(wall_clamp)
        # CLOSED mode: no inlet, no outlet, no sponge — free-slip on all four walls, so the
        # domain holds a self-contained recirculation instead of a stream passing through.
        self.closed = bool(closed)
        self._utx = self._uty = None       # optional drive target (see set_drive)
        self._beta = 0.0                   # scalar OR an (ny, nx) field
        self._msrc = None                  # optional (ny, nx) raw mass source rate
        self._ptgt = None                  # optional (rho_target, rate*mask) regulated source
        self._beta_on = 0.0                # max(beta), so the guard works for both
        # SPONGE: relax f toward the freestream over a BAND (used for the OUTLET, and optionally
        # for the sides when side_bc="free"). A one-row clamp is a discontinuity — the flow
        # accelerated around a blocking body slams into it and leaves a visible stripe along the
        # frame edge. A graded band absorbs the same disturbance far less visibly.
        wj = np.zeros((self.ny, 1), np.float32)
        if sponge_side > 0:
            r = np.arange(self.ny, dtype=np.float32)
            d = np.minimum(r, self.ny - 1 - r) / float(sponge_side)
            wj[:, 0] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        wi = np.zeros((1, self.nx), np.float32)
        if sponge_out > 0:
            d = (self.nx - 1 - np.arange(self.nx, dtype=np.float32)) / float(sponge_out)
            wi[0, :] = np.clip(1.0 - d, 0.0, 1.0) ** 2
        self._sponge = xp.asarray(np.maximum(wj, wi) * 0.35)[None]      # gentle: 35% pull/step

        self.f = xp.broadcast_to(self._feq_free, (9, self.ny, self.nx)).copy()
        self.solid = xp.zeros((self.ny, self.nx), dtype=bool)
        self.rho = xp.ones((self.ny, self.nx), dtype=f32)
        self.ux = xp.full((self.ny, self.nx), self.u0, dtype=f32)
        self.uy = xp.zeros((self.ny, self.nx), dtype=f32)
        self._in_ux = self._in_uy = None          # optional shaped inlet (see set_inlet)
        self._wux = self._wuy = None              # moving-wall velocity (see set_wall_velocity)
        # continuous freestream turbulence carried in through the inlet (see
        # set_inlet_turbulence); off unless a scene asks for it
        self._turb_on = False
        self._turb_du = self._turb_dv = None
        self._turb_amp = 0.0
        self._turb_n = 0
        self._turb_c = 0.0
        self.steps_done = 0

    # -- geometry ---------------------------------------------------------------------
    def set_wall_velocity(self, wux, wuy):
        """Velocity of the moving wall at each solid cell (None = everything stationary).

        Plain bounce-back reflects off a wall that is assumed to be AT REST. If the mask is being
        moved every step — a falling disc, a fired pellet — that assumption is wrong twice over:
        the boundary keeps injecting the momentum needed to hold fluid still against a wall that
        is not still (which pumps energy in until the lattice diverges), and the drag the body
        feels is computed against the wrong relative velocity. `step()` adds the standard
        moving-wall term 6 w_k (c_k . u_wall) to the reflected populations.
        """
        if wux is None:
            self._wux = self._wuy = None
            return
        # Clamp the wall speed the BOUNDARY sees, as a guard against violent transients (a
        # contact impulse). Set it too LOW and it does harm rather than good: a body drifting
        # along with a fast current then has its wall pinned to a slower speed, so the boundary
        # brakes the very fluid it should be riding and shears it hard enough to diverge. Scenes
        # with fast currents raise `wall_clamp` accordingly; the positivity floor in `step()` is
        # what actually prevents the negative-population failure.
        lim = self.wall_clamp
        self._wux = xp.clip(xp.asarray(wux, dtype=xp.float32), -lim, lim)
        self._wuy = xp.clip(xp.asarray(wuy, dtype=xp.float32), -lim, lim)

    def set_solid(self, mask):
        """Swap in a new body mask, REFILLING cells the wall just vacated.

        Refilling is not optional (stale populations inside a former solid inject a momentum
        spike), but refilling at REST is only right for a stationary body. Behind a body that is
        actually moving — a falling disc, a fired pellet — it leaves a trail of dead fluid
        against a moving stream, and that velocity discontinuity launches a pressure wave every
        step until the lattice goes unstable. Instead we EXTRAPOLATE: the new fluid takes the
        average velocity of its fluid neighbours, which is both smooth and what the flow closing
        in behind the body is actually doing.
        """
        new = xp.asarray(mask)
        freed = self.solid & ~new
        if bool(freed.any()):
            # Only neighbours that were ALREADY fluid last step carry a valid velocity: a cell
            # the body just vacated holds whatever bounce-back left behind, and averaging that
            # garbage back in is how a moving body poisons its own wake.
            fluid = ((~new) & (~self.solid)).astype(xp.float32)
            wsum = xp.zeros_like(self.rho)
            usum = xp.zeros_like(self.rho)
            vsum = xp.zeros_like(self.rho)
            for k in range(1, 9):
                sh = (-int(CY[k]), -int(CX[k]))
                wk = xp.roll(fluid, sh, axis=(0, 1)) * float(W[k])
                wsum += wk
                usum += wk * xp.roll(self.ux, sh, axis=(0, 1))
                vsum += wk * xp.roll(self.uy, sh, axis=(0, 1))
            inv = 1.0 / xp.maximum(wsum, 1e-6)
            feq = equilibrium(xp.ones_like(self.rho), usum * inv, vsum * inv,
                              self.cx, self.cy, self.w)
            self.f = xp.where(freed[None], feq, self.f)
        self.solid = new

    def perturb(self, amp=0.02, seed=7, scale=6.0):
        """One-shot smooth random transverse velocity — the symmetry breaker.

        A numerically PERFECT symmetric setup (a centred cylinder in a uniform stream) has
        nothing to trigger the von Karman instability: the wake sits as a stable symmetric twin
        bubble essentially forever, which is a numerical artefact, not physics. Real flow always
        carries freestream disturbance. So we inject a little once, during settling, and let the
        instability select and amplify its own mode from there — after which the shedding is
        entirely self-sustaining and the perturbation is long gone.
        """
        from scipy.ndimage import gaussian_filter as _gf
        n = np.random.default_rng(seed).standard_normal((self.ny, self.nx)).astype(np.float32)
        n = _gf(n, scale)
        n = xp.asarray(n / (float(np.abs(n).max()) + 1e-9))
        duy = amp * self.u0 * n * (1.0 - self._sponge[0] / 0.35)     # never fight the sponge
        rho = xp.maximum(self.f.sum(axis=0), 1e-6)
        ux = (self.cx * self.f).sum(axis=0) / rho
        uy = (self.cy * self.f).sum(axis=0) / rho + duy
        self.f = equilibrium(rho, ux, uy, self.cx, self.cy, self.w)

    def set_inlet_turbulence(self, intensity=0.0, length=12.0, span=3072, seed=11, u_conv=None):
        """Carry FREESTREAM TURBULENCE in through the inlet, continuously, for the whole run.

        `perturb()` is a single smooth kick during settling: enough to break a symmetry, and
        deliberately gone by frame 0. Real tunnel air is not like that. It carries a broadband
        fluctuation of ~0.1-2% of the freestream the entire time the fan is on, and that
        background is what trips a separating shear layer into breaking down at an irregular
        time and place instead of rolling up into the clean, evenly-spaced billows a silent
        solver produces. A noise-free 2-D solve reads as syrup no matter how high `re` goes,
        because nothing is ever there to disturb it - so this is as much a part of "make the
        fluid behave like air" as the Reynolds number is.

        METHOD - synthetic inflow by Taylor's frozen-turbulence hypothesis. A (ny, span) patch of
        turbulence is generated ONCE and convected past the inlet plane at `u_conv` cells per
        step, so the inlet sees a time series with the right length scale and the right
        correlation time for free: one interpolated column read per step, no RNG and no filter in
        the hot loop. `span` must exceed (total steps x u_conv) or the patch recycles - visible
        as the same gust arriving twice.

        The patch is the curl of a smoothed random STREAMFUNCTION (u' = dpsi/dy, v' = -dpsi/dx),
        so what gets injected is divergence-free by construction. That is not fastidiousness: the
        inlet is a hard equilibrium BC at rho=1, so any compressive part of an injected
        fluctuation leaves as an acoustic wave that then rattles around the domain - the artefact
        the wall absorber (AGENT_GUIDE 2a.17) exists to mop up. Take the curl and it is never
        created.

        intensity : |u'|_rms as a fraction of the local freestream. A SCALAR (a fraction of `u0`)
                    or an (ny,) ARRAY, which is what a scene with a banded inlet wants - Tu is a
                    property of a tunnel section, so each band should carry the same *fraction*
                    of its own speed, not the same absolute wobble. 0 disables.
        length    : integral length scale in cells (the Gaussian filter width). Quote it against
                    the CHORD, not against the lattice, so it survives a resolution change.
        """
        arr = np.asarray(intensity, dtype=np.float32)
        if float(np.max(np.abs(arr))) <= 0.0:
            self._turb_on = False
            return
        from scipy.ndimage import gaussian_filter as _gf
        n = int(span)
        rng = np.random.default_rng(seed)
        # mode="wrap" makes the patch periodic on both axes: seamless if it ever does recycle,
        # and no filter roll-off against the spanwise edges (which would otherwise leave the
        # rows nearest the tunnel walls quieter than the rest of the inlet).
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

    # -- one collision-streaming step ------------------------------------------------
    def step(self):
        f = self.f
        if self.side_bc == "slip":
            # FREE-SLIP side walls by specular reflection: a particle heading into the wall comes
            # back with its y-component mirrored. This is a real, inviscid tunnel wall — it grows
            # no boundary layer, and unlike forcing the edge rows to freestream it imposes no
            # velocity, so the flow may legitimately accelerate along it and no edge stripe
            # appears in the picture. Reflect the populations arriving FROM outside the domain.
            f[2, 0, :] = f[4, 0, :]                 # bottom wall: up-going set = down-going set
            f[5, 0, :] = f[8, 0, :]
            f[6, 0, :] = f[7, 0, :]
            f[4, -1, :] = f[2, -1, :]               # top wall: mirror image of the above
            f[7, -1, :] = f[6, -1, :]
            f[8, -1, :] = f[5, -1, :]
        if self.closed:
            # free-slip on the streamwise ends too (mirror the x-component), so the box is sealed
            f[1, :, 0] = f[3, :, 0]
            f[5, :, 0] = f[6, :, 0]
            f[8, :, 0] = f[7, :, 0]
            f[3, :, -1] = f[1, :, -1]
            f[6, :, -1] = f[5, :, -1]
            f[7, :, -1] = f[8, :, -1]
        else:
            # outlet: zero-gradient for populations that would otherwise stream in from outside
            for k in LEFTWARD:
                f[int(k), :, -1] = f[int(k), :, -2]

        rho = f.sum(axis=0)
        rho = xp.maximum(rho, 1e-6)
        ux = (self.cx * f).sum(axis=0) / rho
        uy = (self.cy * f).sum(axis=0) / rho
        # inlet column: clamp to the freestream before the collision sees it
        if not self.closed:
            rho[:, 0] = 1.0
            ux[:, 0] = self.u0 if self._in_ux is None else self._in_ux
            uy[:, 0] = 0.0 if self._in_uy is None else self._in_uy
            if self._turb_on:
                # Read the frozen patch at the fractional column the flow has convected to, and
                # interpolate between the two neighbours: without that the gust would step a
                # whole cell every ~18 steps, and a staircase in a velocity BC is exactly the
                # kind of discontinuity that radiates. All the index arithmetic is host-side
                # Python on `steps_done`, so it costs no reduction and no GPU sync.
                p = (self.steps_done * self._turb_c) % self._turb_n
                i0 = int(p)
                fr = p - i0
                i1 = (i0 + 1) % self._turb_n
                a = self._turb_amp
                ux[:, 0] += a * ((1.0 - fr) * self._turb_du[:, i0] + fr * self._turb_du[:, i1])
                uy[:, 0] += a * ((1.0 - fr) * self._turb_dv[:, i0] + fr * self._turb_dv[:, i1])

        feq = equilibrium(rho, ux, uy, self.cx, self.cy, self.w)
        fneq = f - feq

        # --- Smagorinsky LES: raise tau where the strain rate is high -------------------
        pxx = (self.cx * self.cx * fneq).sum(axis=0)
        pyy = (self.cy * self.cy * fneq).sum(axis=0)
        pxy = (self.cx * self.cy * fneq).sum(axis=0)
        pi = xp.sqrt(2.0 * (pxx * pxx + pyy * pyy + 2.0 * pxy * pxy))
        tau_e = 0.5 * (self.tau + xp.sqrt(self.tau * self.tau
                                          + 18.0 * 1.4142135 * self.csm2 * pi / rho))
        omega = 1.0 / tau_e

        fout = f - omega * fneq
        if self._utx is not None and self._beta_on > 0.0:
            # body force F = beta*rho*(u_target - u), injected as momentum: f_k += 3 w_k (c_k.F)
            fbx = self._beta * rho * (self._utx - ux)
            fby = self._beta * rho * (self._uty - uy)
            fout = fout + 3.0 * self.w * (self.cx * fbx + self.cy * fby)
        if self._msrc is not None:
            # isotropic mass injection: adds rho, adds no momentum. Applied BEFORE the sponge so
            # that a source placed (illegally) inside the sponge band is damped rather than
            # fighting it, and before bounce-back so it never writes into solid cells.
            fout = fout + self.w * self._msrc
        if self._ptgt is not None:
            # regulated form: the source is proportional to how far the cell is BELOW target, so
            # it shuts itself off on arrival and reverses on overshoot
            tgt, krate = self._ptgt
            fout = fout + self.w * (krate * (tgt - rho))
        if not self.closed:
            fout += self._sponge * (self._feq_free - fout)  # graded open far field / outlet
            fout[:, :, 0] = feq[:, :, 0]                    # inlet = pure equilibrium

        # --- body: half-way bounce-back (pre-collision populations, reversed) -----------
        if bool(self.solid.any()):
            back = f[self._opp]
            if self._wux is not None:
                # moving-wall bounce-back: the reflected population gains the wall's momentum
                back = xp.maximum(
                    back + 6.0 * self.w * (self.cx * self._wux + self.cy * self._wuy), 0.0)
            fout = xp.where(self.solid[None], back, fout)

        # --- stream ---------------------------------------------------------------------
        for k in range(9):
            if CX[k] or CY[k]:
                fout[k] = xp.roll(fout[k], (int(CY[k]), int(CX[k])), axis=(0, 1))
        self.f = fout
        self.rho, self.ux, self.uy = rho, ux, uy
        self.steps_done += 1

    def run(self, n):
        for _ in range(int(n)):
            self.step()

    # -- derived fields ---------------------------------------------------------------
    def speed(self):
        u = xp.sqrt(self.ux * self.ux + self.uy * self.uy)
        return xp.where(self.solid, 0.0, u)

    def vorticity(self):
        """dv/dx - du/dy, central differences (lattice units)."""
        dvdx = (xp.roll(self.uy, -1, axis=1) - xp.roll(self.uy, 1, axis=1)) * 0.5
        dudy = (xp.roll(self.ux, -1, axis=0) - xp.roll(self.ux, 1, axis=0)) * 0.5
        return xp.where(self.solid, 0.0, dvdx - dudy)

    def pressure(self):
        return xp.where(self.solid, 0.0, (self.rho - 1.0) / 3.0)

    def stagnation(self):
        """STAGNATION pressure, p + rho*u^2/2 - the pressure a parcel would reach if brought to
        rest, i.e. the total mechanical energy it is carrying.

        This is the field to draw for a pressure-driven machine, and static pressure is not.
        A rocket chamber is high STATIC pressure, but the nozzle exists precisely to convert that
        into velocity, so by the time the flow is a plume its static pressure is back near
        ambient - draw `pressure()` and the chamber blazes while the exhaust disappears, which is
        backwards. Stagnation pressure is high in BOTH: static in the chamber, dynamic in the
        plume, and near zero in the ambient tunnel that has neither. It is the high-energy stream,
        drawn as one quantity.

        THE STATIC TERM IS GAUGED AND ONE-SIDED, and both parts of that are there to kill an
        artefact Ethan reported as *"the screen appears like a bouncing jelly"*. A weakly
        compressible lattice carries acoustic waves: rho rings domain-wide whenever anything
        changes, and a mass source correcting a sealed chamber rings it constantly. Plotting
        `(rho - 1)/3` puts every one of those swings straight into the colour of EVERY cell, so
        the whole frame breathes.

        Two changes remove it. Gauging against the instantaneous FLUID MEAN rather than against
        1.0 cancels the uniform component - the part that makes the entire picture pulse together
        - and also absorbs the slow drift as injected mass accumulates. Clipping the result at
        zero then drops the rarefaction half of what is left: an acoustic wave swings both ways,
        and only the compression side carries information here. The ambient tunnel sits at the
        mean, so it lands flat at zero instead of shimmering, while the chamber - genuinely far
        above the mean - is untouched. The dynamic term is not gauged: it is a true local
        quantity with no acoustic component to remove.
        """
        m = ~self.solid
        ref = float((self.rho * m).sum() / xp.maximum(m.sum(), 1))
        p = xp.clip((self.rho - ref) / 3.0, 0.0, None)
        return xp.where(self.solid, 0.0,
                        p + 0.5 * self.rho * (self.ux * self.ux + self.uy * self.uy))

    def force_field(self):
        """PER-CELL momentum handed to the wall, as two (ny, nx) arrays.

        Same momentum-exchange sum as `force()`, but left un-summed so that a scene with several
        independent bodies can integrate it over each body's own mask. Computing it once for the
        whole lattice (8 rolls) and then taking N masked sums is far cheaper than running the
        link loop once per body, which matters when this is called every LBM step.
        """
        fx = xp.zeros_like(self.rho)
        fy = xp.zeros_like(self.rho)
        solid = self.solid
        for k in range(1, 9):
            ahead = xp.roll(solid, (-int(CY[k]), -int(CX[k])), axis=(0, 1))
            # x_b is FLUID and its neighbour along +c_k is SOLID. Evaluating at the fluid node is
            # not a detail: populations stored ON solid cells are whatever bounce-back last left
            # there, not physical, and summing those gives a drag ~20x too large.
            link = ((~solid) & ahead).astype(xp.float32)
            amt = (self.f[k] + self.f[OPP[k]]) * link
            fx += float(CX[k]) * amt
            fy += float(CY[k]) * amt
        return fx, fy

    def set_drive(self, utx, uty, beta):
        """Nudge the fluid toward a target velocity field with relaxation rate `beta`.

        Applied as a body force F = beta*rho*(u_target - u), added to the populations after
        collision. Deliberately WEAK: a closed vortex this size decays viscously over millions of
        steps, i.e. never within a clip, so the drive is not needed to sustain the circulation —
        only to establish it and, at the end, to wind it up for the flush. Keeping beta small is
        what lets the pellets' wakes survive instead of being erased back to the target field.

        `beta` may be a SCALAR (the whole domain relaxes at one rate — what `orbit` uses) or an
        (ny, nx) FIELD. The field form is what lets a scene force only part of the domain: with
        beta ~ 0 outside a patch the rest of the fluid is left completely alone, so a localised
        jet or vortex can be injected into water that stays otherwise undriven. A small non-zero
        floor with u_target = 0 there is then a linear (Rayleigh) friction — the term that lets a
        disturbance decay back to stillness in seconds instead of the ~1e6 steps viscosity alone
        would take at these Reynolds numbers.
        """
        self._utx = None if utx is None else xp.asarray(utx, dtype=xp.float32)
        self._uty = None if uty is None else xp.asarray(uty, dtype=xp.float32)
        if np.isscalar(beta):
            self._beta = float(beta)
            self._beta_on = float(beta)
        else:
            # No max() here on purpose: this is called every LBM step, and a reduction back to
            # the host is a blocking GPU sync. An all-zero field simply contributes no force.
            self._beta = xp.asarray(beta, dtype=xp.float32)
            self._beta_on = 1.0

    def set_mass_source(self, field):
        """Inject MASS (i.e. pressure) at a per-cell rate, or None to stop.

        `field` is an (ny, nx) density-per-step rate. Mass is added isotropically - spread over
        the populations by their weights - so it carries NO net momentum: it raises rho, and
        since this is a weakly compressible model with p = rho/3, raising rho IS raising
        pressure. What the fluid then does with that pressure is the solver's business.

        This is what lets a SEALED chamber drive a jet. A drive patch pumping fluid out of a
        closed cavity is a mass sink: rho falls away, and an LBM whose density is heading toward
        zero comes apart. A rocket, though, genuinely does carry its own propellant, so mass
        appearing inside a closed tank is the honest model of a tank being emptied - we simply
        do not model the tank running out. Pressurise the chamber and let the nozzle convert it:
        for a chamber at rho_c the ideal exit speed is ~sqrt(2*(rho_c - 1)/(3*rho_c)), which is
        the rocket's own energy equation and not a number anyone has to tune into place.

        It is also SELF-LIMITING, which is why it is safe to leave running. Pressure rises until
        the mass leaving through the throat equals the mass being injected, so the steady state
        is set by the injection rate against the nozzle area rather than by a velocity anyone
        picked. Overshoot the rate and the chamber simply sits at a higher pressure.
        """
        self._msrc = None if field is None else xp.asarray(field, dtype=xp.float32)

    def set_pressure_target(self, rho_target, mask, rate=0.02):
        """REGULATED pressure: relax rho toward `rho_target` inside `mask` at `rate` per step.

        This is `set_mass_source` with the loop closed, and the difference is not a refinement -
        it is the difference between a scene that runs and one that does not. Injecting mass at
        a fixed rate into a SEALED vessel is open-loop: nothing anywhere in the system pushes
        back until the throat chokes near the lattice sound speed, and past that point the
        injected mass has nowhere to go, so pressure climbs without bound and the solve dies. A
        measured run did exactly that - 0.2965 at two seconds of full throttle, 0.3217 at two and
        a half, 0.3905 at three and a half, still accelerating, with no sign of a steady state.

        Relaxing toward a target instead makes the source its own governor: it injects hard into
        a cold chamber, tapers as the chamber comes up to pressure, and REMOVES mass if it
        overshoots. The steady state is `rho_target` by construction rather than by balance, so
        "run the chamber at this pressure" is a number you set instead of a number you discover,
        and the exhaust velocity that follows from it is bounded before the solve begins:
        u_ideal = sqrt(2*(rho_target - 1)/3).

        `mask` is (ny, nx) in 0..1; `rate` is the fraction of the remaining gap closed per step.
        """
        self._ptgt = None if mask is None else (float(rho_target),
                                                xp.asarray(mask, dtype=xp.float32) * float(rate))

    def init_velocity(self, ux, uy):
        """Start the lattice from a given velocity field (equilibrium at rho=1)."""
        self.ux = xp.asarray(ux, dtype=xp.float32)
        self.uy = xp.asarray(uy, dtype=xp.float32)
        self.rho = xp.ones((self.ny, self.nx), dtype=xp.float32)
        self.f = equilibrium(self.rho, self.ux, self.uy, self.cx, self.cy, self.w)

    def set_farfield(self, ux_profile=None):
        """Retarget the SPONGE's equilibrium to a spanwise profile instead of a uniform `u0`.

        The outlet sponge relaxes `f` toward the freestream so a vortex leaving the domain does
        not reflect back into it. That is correct only while "the freestream" is one number. A
        scene that shapes the inlet into bands of DIFFERENT speed (`tri_foil_rates`) otherwise
        gets a 14-cell pump bolted onto its exit plane: the sponge accelerates the slow band back
        toward `u0` and that pressure gradient reaches upstream at the lattice sound speed, so the
        bands the scene asked for quietly wash out over the clip. Feed the same profile here that
        went into `set_inlet` and the absorber becomes what it was meant to be - transparent to
        the flow the scene established, opaque to what reflects off the boundary.

        `None` restores the uniform-`u0` target. Note this does NOT touch the current `f`; a scene
        wanting the whole domain to START on the profile calls `init_velocity` as well.
        """
        f32 = xp.float32
        if ux_profile is None:
            rho1 = xp.ones((1, 1), dtype=f32)
            ux1 = xp.full((1, 1), self.u0, dtype=f32)
            uy1 = xp.zeros((1, 1), dtype=f32)
            self._feq_free = equilibrium(rho1, ux1, uy1, self.cx, self.cy, self.w).reshape(9, 1, 1)
            return
        u = xp.asarray(ux_profile, dtype=f32).reshape(1, self.ny, 1)
        rho = xp.ones((1, self.ny, 1), dtype=f32)
        # (9, ny, 1): broadcasts against the (1, ny, nx) sponge weight exactly as the scalar did
        self._feq_free = equilibrium(rho, u, xp.zeros_like(u), self.cx, self.cy, self.w)

    def set_inlet(self, ux_profile=None, uy_profile=None):
        """Override the inlet velocity across the span — arrays of length ny (or None to reset).

        This is what lets a scene make *currents* instead of a uniform stream: a shaped, slowly
        changing inlet profile shears against itself and rolls up into eddies downstream, with no
        obstacle needed and nothing keyframed in the interior.
        """
        self._in_ux = None if ux_profile is None else xp.asarray(ux_profile, dtype=xp.float32)
        self._in_uy = None if uy_profile is None else xp.asarray(uy_profile, dtype=xp.float32)

    def force(self):
        """Net force on the body by the momentum-exchange method -> (drag, lift) in lattice units.

        For every bounce-back link crossing the wall, the momentum handed to the body is
        2 c_k f_k. Summing over the body's links gives the total force directly, with no surface
        integral or pressure reconstruction — the standard, and cheapest, LBM way to get Cl/Cd.
        """
        fx, fy = self.force_field()
        return float(asnumpy(fx.sum())), float(asnumpy(fy.sum()))

    def coefficients(self, ref_len):
        """(Cd, Cl) = F / (0.5 rho u0^2 L) — the usual 2-D non-dimensionalisation."""
        fx, fy = self.force()
        q = 0.5 * self.u0 * self.u0 * max(ref_len, 1e-9)
        return fx / q, fy / q

    def health(self):
        """Max |u| — a cheap divergence tripwire (lattice Mach must stay well under 1/sqrt(3))."""
        return float(asnumpy(xp.sqrt(self.ux ** 2 + self.uy ** 2).max()))

    @property
    def health_limit(self):
        """Abort threshold for `render_scene`. The lattice sound speed is 1/sqrt(3) = 0.577 and
        past ~0.45 the solve is already gone, so every remaining frame would be noise."""
        return 0.45

    def describe(self):
        return f"tau={self.tau:.4f} (nu={self.nu:.5f}), D2Q9 LBM + Smagorinsky"
