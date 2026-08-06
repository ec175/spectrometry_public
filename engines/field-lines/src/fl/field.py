"""field.py - the vector fields. No PDE is ever solved in this project.

Every field here is a CLOSED FORM evaluated at an array of query points. That is the whole
reason this engine is cheap and can never diverge: there is no lattice, no timestep stability
condition, no CFL number. The only numerical care needed is (a) softening at a singular source
and (b) a small enough arclength step in `lines.trace`.

A field is any callable `f(P) -> V` with P and V both (N, 2) float64 arrays, in WORLD px
(1080x1920 reference, y DOWN). Everything downstream only needs that contract, which is what
makes the same integrator serve electrostatics, magnetostatics and potential flow.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# Point sources (electrostatics)
# --------------------------------------------------------------------------------------
def coulomb(P, pos, q, soft=9.0, kernel="inv_sq"):
    """E(p) = sum_i q_i (p - x_i) / |p - x_i|^n, Plummer-softened.

    P    (N,2) query points
    pos  (M,2) source positions
    q    (M,)  signed magnitudes
    soft scalar - Plummer eps in px. WITHOUT THIS an RK4 step that lands near a source takes a
         NaN and the line flies off the frame. eps ~ half a marker radius is invisible outside
         the glyph and completely removes the failure mode.
    kernel 'inv_sq' -> 3-D point charges viewed in a plane (the default, and what a "+/-" glyph
           implies); 'inv_r' -> true 2-D line charges. The two give nearly identical LINE
           SHAPES (a line's direction only depends on the direction of the sum, and both
           kernels are radial) but visibly different magnitude falloff.
    """
    d = P[:, None, :] - pos[None, :, :]            # (N,M,2)
    r2 = np.einsum("nmk,nmk->nm", d, d) + soft * soft
    if kernel == "inv_r":
        w = q[None, :] / r2                        # 1/r * unit  ==  d / r^2
    else:
        w = q[None, :] / (r2 * np.sqrt(r2))        # 1/r^2 * unit == d / r^3
    return np.einsum("nm,nmk->nk", w, d)


class ChargeSet:
    """A mutable bag of point sources. Scenes rebuild `pos`/`q` every frame; `trace` only ever
    sees the arrays, so a scene may move charges, change their magnitude, or flip their SIGN
    between frames with no bookkeeping anywhere else."""

    def __init__(self, soft=9.0, kernel="inv_sq"):
        self.pos = np.zeros((0, 2))
        self.q = np.zeros((0,))
        self.soft = soft
        self.kernel = kernel

    def set(self, pos, q):
        self.pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
        self.q = np.asarray(q, dtype=np.float64).reshape(-1)
        return self

    def __call__(self, P):
        return coulomb(P, self.pos, self.q, self.soft, self.kernel)

    # --- the two things `lines.trace` needs to know about terminating a line
    def sinks(self, sign=-1.0):
        """Positions a field line can END on: negatives for a forward trace. A source whose
        magnitude has decayed to nothing is not a sink - excluded, else lines snap onto an
        invisible marker."""
        m = (np.sign(self.q) == sign) & (np.abs(self.q) > 1e-3)
        return self.pos[m]


# --------------------------------------------------------------------------------------
# Potential flow  (same integrator, different kernel - the reuse proof)
# --------------------------------------------------------------------------------------
class PotentialFlow:
    """u = uniform stream + sum(doublets) + sum(point vortices) + sum(sources/sinks).

    Superposition of harmonic solutions, so the result is an exact solution of the Laplace
    equation everywhere outside the singularities - i.e. this is REAL potential flow, not a
    stylisation. It is irrotational, so there is no wake and no separation; what it does give
    (and what the clip shows) is the stagnation points sliding around a cylinder as its bound
    circulation changes, which is the genuine Magnus mechanism.

    A doublet of strength mu = |U| * a^2 reproduces flow past a cylinder of radius a EXACTLY -
    but only when its axis is aligned with the stream. `cylinder()` derives that angle from U
    rather than taking it as a parameter, because getting it wrong is silent: the body stops
    being a streamline surface and the flow pours straight through it. That is exactly what the
    first build did (stream along +y, doublet built for +x), and the only thing that caught it
    was `flux_report` showing 52% of streamlines "captured" inside bodies that a potential flow
    can never let them enter.
    """

    def __init__(self):
        self.U = np.array([0.0, 0.0])
        self.doublets = []      # (pos(2,), mu, axis_angle)
        self.vortices = []      # (pos(2,), gamma)   +gamma = clockwise on screen (y DOWN)
        self.sources = []       # (pos(2,), m)
        self.circle = None      # (center(2,), a) - the body whose surface must be a streamline

    def clear_singularities(self):
        self.doublets, self.vortices, self.sources = [], [], []
        self.circle = None
        return self

    def cylinder(self, center, a, gamma=0.0):
        """Add a cylinder of radius `a` with bound circulation `gamma`.

        The doublet axis is derived from U, never passed in: getting it wrong is silent, and it
        is what the first build did. The bound vortex sits at the centre, where it contributes
        no normal velocity at the surface - that is why circulation is free to swing without
        breaking the body.

        ONE cylinder only. A single doublet satisfies the boundary condition for an ISOLATED
        body; put two in the same flow and each one's field crosses the other's surface. That is
        not a tuning problem and it does not get better with distance fast enough to matter: the
        first `stream_glass` build measured max|v.n| = 0.81|U| on a two-body layout whose
        doublets alone contribute only 0.05, because the BOUND VORTICES dominate the cross-talk
        (gamma/r at the other body). Exactness for several bodies needs an iterated image system
        or a panel solve - out of scope here, and unnecessary, because one cylinder plus imaged
        free vortices is both exact and more legible.
        """
        speed = float(np.hypot(*self.U))
        c = np.asarray(center, float)
        self.doublets.append((c, speed * a * a, float(np.arctan2(self.U[1], self.U[0]))))
        self.circle = (c, float(a))
        if gamma:
            self.vortices.append((c, float(gamma)))
        return self

    def _images(self):
        """Milne-Thomson circle theorem, applied to every singularity OUTSIDE the body.

        For a circle of radius a centred at c, a vortex of strength G at z0 outside it is imaged
        by a vortex -G at the inverse point c + a^2 (z0-c)/|z0-c|^2 and +G at the centre; a
        source +m is imaged by +m at the inverse point and -m at the centre. With those present
        the surface is a streamline EXACTLY, for arbitrary external flow - which is what lets
        free vortices orbit right past the body without the flow leaking through it.

        Singularities already inside the body (the doublet, the bound vortex) are skipped: they
        are what the theorem is correcting *for*, not corrections themselves.
        """
        if self.circle is None:
            return [], []
        c, a = self.circle
        a2 = a * a
        iv, isr = [], []
        for z0, g in self.vortices:
            d = z0 - c
            r2 = float(d @ d)
            if r2 <= a2 * 1.0001:
                continue
            iv.append((c + a2 * d / r2, -g))
            iv.append((c, g))
        for z0, m in self.sources:
            d = z0 - c
            r2 = float(d @ d)
            if r2 <= a2 * 1.0001:
                continue
            isr.append((c + a2 * d / r2, m))
            isr.append((c, -m))
        return iv, isr

    def __call__(self, P):
        V = np.repeat(self.U[None, :], len(P), axis=0).astype(np.float64)
        iv, isr = self._images()
        for c, mu, ang in self.doublets:
            d = P - c
            r2 = d[:, 0] ** 2 + d[:, 1] ** 2 + 1.0
            ca, sa = np.cos(ang), np.sin(ang)
            # doublet aligned with (ca,sa): u = mu/r^2 * (2 n (n.d_hat) d_hat - n) form,
            # written directly in components
            dx, dy = d[:, 0], d[:, 1]
            k = mu / (r2 * r2)
            V[:, 0] += k * (ca * (dy * dy - dx * dx) - 2.0 * sa * dx * dy)
            V[:, 1] += k * (sa * (dx * dx - dy * dy) - 2.0 * ca * dx * dy)
        for c, g in list(self.vortices) + iv:
            d = P - c
            r2 = d[:, 0] ** 2 + d[:, 1] ** 2 + 4.0
            V[:, 0] += g * d[:, 1] / r2
            V[:, 1] += -g * d[:, 0] / r2
        for c, m in list(self.sources) + isr:
            d = P - c
            r2 = d[:, 0] ** 2 + d[:, 1] ** 2 + 4.0
            V[:, 0] += m * d[:, 0] / r2
            V[:, 1] += m * d[:, 1] / r2
        return V


# --------------------------------------------------------------------------------------
# Magnetostatics - here for the same reason `inv_r` is: the integrator is subject-agnostic and
# this is the cheapest possible proof of it. A current-carrying wire perpendicular to the plane
# has a purely azimuthal B, so its field lines are closed circles - lines that NEVER terminate,
# which is the one topological case electrostatics cannot produce.
# --------------------------------------------------------------------------------------
def wires(P, pos, I, soft=9.0):
    """B(p) = sum_i I_i * z_hat x (p - x_i) / |p - x_i|^2."""
    d = P[:, None, :] - pos[None, :, :]
    r2 = np.einsum("nmk,nmk->nm", d, d) + soft * soft
    w = I[None, :] / r2
    out = np.empty_like(d[:, 0, :])
    out[:, 0] = np.einsum("nm,nm->n", w, -d[:, :, 1])
    out[:, 1] = np.einsum("nm,nm->n", w, d[:, :, 0])
    return out
