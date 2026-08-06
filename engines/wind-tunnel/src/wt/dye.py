"""dye.py — passive coloured species advected by the flow (D2Q5 BGK, one lattice per species).

WHY THIS EXISTS. Every field this project has drawn so far is a function of the velocity field:
speed, vorticity, pressure. That is enough to say *how fast* fluid is moving and never enough to
say *which fluid it is*. Ethan, 2026-08-03, asking for the rocket: *"noticibly different colored
fluids in tanks that expand and push thrust downward."* Two propellants that stay distinct down
their own tanks, meet, and mix is a statement about material identity, so it needs a transported
scalar. Painting the tanks two colours would have been a lie the moment the fluid left them.

THE SCHEME. Each species rides its own D2Q5 lattice relaxing to a linear-in-u equilibrium:

    g_i^eq = w_i * C * (1 + 3 c_i . u)          w = [1/3, 1/6, 1/6, 1/6, 1/6],  c_s^2 = 1/3

which is the standard lattice model for advection-diffusion; summing the populations gives the
concentration C and the scheme transports it at the fluid's own velocity with diffusivity
D = c_s^2 (tau - 1/2). Two reasons to use a lattice rather than, say, a semi-Lagrangian
backtrace: it reuses exactly the streaming primitive the fluid solver already has (one `roll`
per direction), and it shares the fluid's boundary treatment, so "no dye through a wall" is the
same bounce-back that means "no fluid through a wall" and the two can never disagree about where
the solid is.

`tau` is deliberately close to 1/2. D is what smears one propellant into the other, and the
mixing that should show on screen is the mixing the FLOW does — stirring by the shear layer
between the two streams — not numerical smearing. At the default tau = 0.515 the diffusivity is
0.005 lattice units, so over the ~2000 steps a parcel spends in the engine a dye front spreads
about sqrt(2 D t) ~ 4.5 cells. That is a soft edge, not a blur.

Species are stored as one (K, 5, ny, nx) array and stepped together, so K species cost K times
the memory but not K times the kernel launches — which is what actually matters on this GPU
(see AGENT_GUIDE section 4: this workload is launch-bound, not bandwidth-bound).
"""
from __future__ import annotations

import numpy as np

from .gpu import xp

# D2Q5: rest, +x, -x, +y, -y
CX = np.array([0, 1, -1, 0, 0], dtype=np.int32)
CY = np.array([0, 0, 0, 1, -1], dtype=np.int32)
OPP = np.array([0, 2, 1, 4, 3], dtype=np.int32)
W = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float32)


class Dye:
    """K passive species on a shared grid, advected by a velocity field supplied each step."""

    def __init__(self, nx, ny, species=2, tau=0.515, inlet_clean=True):
        self.nx, self.ny, self.k = int(nx), int(ny), int(species)
        self.tau = float(tau)
        self.omega = 1.0 / self.tau
        self.inlet_clean = bool(inlet_clean)
        self.w = xp.asarray(W).reshape(1, 5, 1, 1)
        self.cx = xp.asarray(CX.astype(np.float32)).reshape(1, 5, 1, 1)
        self.cy = xp.asarray(CY.astype(np.float32)).reshape(1, 5, 1, 1)
        self._opp = xp.asarray(OPP)
        self.g = xp.zeros((self.k, 5, self.ny, self.nx), dtype=xp.float32)
        self.solid = xp.zeros((self.ny, self.nx), dtype=bool)
        self._src = None                      # (K, ny, nx) injection weight, or None

    @property
    def diffusivity(self):
        return (self.tau - 0.5) / 3.0

    def set_solid(self, mask):
        self.solid = xp.asarray(mask, dtype=bool)

    def set_source(self, weights):
        """(K, ny, nx) injection weights in 0..1, or None to stop injecting.

        A weight of w relaxes that cell's concentration toward 1 by a fraction w each step, so
        a patch held at w = 0.05 saturates over ~20 steps. Injection is a RELAXATION rather than
        an addition on purpose: adding a fixed amount per step has no fixed point, so a patch
        left on would drive C past 1 without bound and the colour would clip to a flat slab.
        """
        self._src = None if weights is None else xp.asarray(weights, dtype=xp.float32)

    def concentration(self):
        """(K, ny, nx) concentrations."""
        return self.g.sum(axis=1)

    def step(self, ux, uy):
        c = self.g.sum(axis=1)                                    # (K, ny, nx)
        if self._src is not None:
            # source term, distributed over the populations by weight so it adds exactly dC
            dc = self._src * (1.0 - c)
            self.g = self.g + self.w * dc[:, None]
            c = c + dc

        # SECOND-ORDER equilibrium, and it has to be second order here. The linear form
        # w_i C (1 + 3 c_i.u) is the textbook advection-diffusion model and it is fine at the
        # velocities the rest of this project runs at - but it goes NEGATIVE for the upstream
        # direction as soon as |u| > 1/3, and a rocket plume is faster than that. Clipping the
        # negative away (which the final clip does) then MANUFACTURES concentration: a measured
        # run peaked at C = 1.625 against a source that can only ever relax toward 1.0. Carrying
        # the (c.u)^2 and u^2 terms keeps every population positive through the plume.
        eu = self.cx * ux[None, None] + self.cy * uy[None, None]
        usq = (ux * ux + uy * uy)[None, None]
        geq = self.w * c[:, None] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
        gout = self.g - self.omega * (self.g - geq)

        # no-flux wall: the same half-way bounce-back the fluid uses, so dye and fluid can never
        # disagree about where the boundary is
        if bool(self.solid.any()):
            gout = xp.where(self.solid[None, None], self.g[:, self._opp], gout)

        for i in range(1, 5):
            gout[:, i] = xp.roll(gout[:, i], (int(CY[i]), int(CX[i])), axis=(1, 2))

        if self.inlet_clean:
            gout[:, :, :, 0] = 0.0                                # tunnel fluid arrives undyed
            gout[:, :, :, -1] = gout[:, :, :, -2]                 # outlet: zero gradient
        self.g = xp.clip(gout, 0.0, None)
