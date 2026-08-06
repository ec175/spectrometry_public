"""rig.py - articulated deformation of a body, for the outline AND its texture.

Every body in this project has been RIGID: a scene may put a polygon somewhere at time t, and
`shapes.place` scales, rotates and translates the whole thing. That is enough for a foil in a rig
and for a free body being pushed around, and it is not enough for an animal, which was the point
of Ethan's note on the animated cow (2026-08-05): *"DO not just rotate the head"*. Rotating the
head polygon rigidly detaches it from the neck; what a neck does is BEND.

THE MODEL: a BEND FIELD - each point is rotated about the bone's pivot by its own share of the
angle, `w(p) * theta`.

A `Bone` is a pivot, an angle, and a WEIGHT FIELD - "how much of this rotation does a point at p
feel". A point deep in the head feels the head bone fully, a point at the shoulder feels none of
it, and the neck between them feels a smoothly increasing fraction, which is exactly what makes
the neck curve instead of hinging. Bones compose in sequence, so an ear flick rides on top of a
head that is already down.

NOT LINEAR BLEND SKINNING, and the difference is load-bearing rather than academic. LBS is the
textbook choice and was written first: it displaces by `w * (R(p-c) - (p-c))`, i.e. along the
CHORD of the rotation rather than the arc. Two things follow, and the second one killed it:

  - it loses volume at a bend (the "candy wrapper" pinch), which on a neck is exactly where the
    eye is looking;
  - **its inverse does not converge.** `render._draw_texture` walks SCREEN pixels, so for every
    one it has to ask "which part of the photograph is here now" - it needs the deformed -> undeformed
    map, which has no closed form. Fixed-point iteration on the LBS displacement was **measured
    at a residual of 0.43 chords, i.e. 44 lattice cells**, for the 45 degree head bend: the weight
    is a function of x, the deformation moves points a long way in x, and the iteration chases
    its own tail.

Rotating by `w * theta` fixes both, because it is an isometry about the pivot: `|q - c| = |p - c|`
EXACTLY. So the inverse only has to recover an ANGLE, with the radius already known for free, and
that iteration is well conditioned - measured residual **2e-4 chords, a fiftieth of a cell**, in
three passes. `max_residual` is kept as a gate so a future angle change has to re-answer it
rather than inherit this.

WHY THE TEXTURE MUST FOLLOW AT ALL. If the outline bends and the photograph does not, the cow's
head becomes a window that a picture of a head slides behind - worse than not animating it.
Deforming both with the same rig is what keeps them one object, the same reason
`shapes.image_body` takes the drawn alpha from the polygon rather than from the source cut-out.

WHY THE TEXTURE MUST FOLLOW. If the outline bends and the photograph does not, the cow's head
turns into a window that the picture of a head slides behind - which is worse than not animating
it at all. Deforming both with the same rig is what keeps them one object, the same reason
`shapes.image_body` takes the drawn alpha from the polygon rather than from the source cut-out.
"""
from __future__ import annotations

import numpy as np


def _sstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a if b != a else 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Bone:
    """One rotation about `pivot` by `angle` degrees, felt to the extent `weight(p)` says.

    Two weight shapes cover everything here and both are smooth (a discontinuous weight is a
    tear in the surface):

      - `axis=(lo, hi)` - a band along x. Full effect at or before `lo`, none at or after `hi`.
        This is a LIMB: the head bone runs 1 at the muzzle to 0 at the shoulder, so the neck is
        the ramp and it bends along its whole length.
      - `disc=(cx, cy, r0, r1)` - full effect inside r0 of the centre, fading to none by r1.
        This is a FEATURE: the ear, which has to move without dragging the skull with it.
    """

    def __init__(self, angle=0.0, pivot=(0.0, 0.0), axis=None, disc=None):
        self.angle = float(angle)
        self.pivot = np.asarray(pivot, dtype=np.float64)
        self.axis = axis
        self.disc = disc

    def weight(self, p):
        if self.axis is not None:
            lo, hi = self.axis
            return 1.0 - _sstep(lo, hi, p[:, 0])
        cx, cy, r0, r1 = self.disc
        r = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        return 1.0 - _sstep(r0, r1, r)

    def _spin(self, d, ang):
        ca, sa = np.cos(ang), np.sin(ang)
        return np.stack([d[:, 0] * ca - d[:, 1] * sa, d[:, 0] * sa + d[:, 1] * ca], 1)

    def apply(self, p):
        """Undeformed -> deformed: rotate each point about the pivot by its own `w * angle`."""
        if abs(self.angle) < 1e-12:
            return np.asarray(p, dtype=np.float64)
        d = p - self.pivot
        return self.pivot + self._spin(d, self.weight(p) * np.deg2rad(self.angle))

    def invert(self, q, iters=3):
        """Deformed -> undeformed.

        The radius from the pivot is preserved exactly, so only the angle has to be recovered:
        guess the weight at `q`, rotate back by it, re-read the weight there, repeat.
        """
        if abs(self.angle) < 1e-12:
            return np.asarray(q, dtype=np.float64)
        a = np.deg2rad(self.angle)
        d = q - self.pivot
        p = q
        for _ in range(int(iters)):
            p = self.pivot + self._spin(d, -self.weight(p) * a)
        return p


def warp(points, bones):
    """Undeformed -> deformed. What the polygon (and therefore the solver) sees."""
    p = np.asarray(points, dtype=np.float64)
    for b in bones:
        p = b.apply(p)
    return p


def unwarp(points, bones, iters=3):
    """Deformed -> undeformed. What the texture lookup needs. Bones undone in reverse order."""
    q = np.asarray(points, dtype=np.float64)
    for b in reversed(list(bones)):
        q = b.invert(q, iters)
    return q


def max_residual(bones, box, n=48):
    """Round-trip error of `unwarp(warp(.))` over a box, in the same units as the points.

    A verification hook rather than a utility: fixed-point inversion is only as good as its
    convergence, and "it looked fine" is not a measurement. `box` is (x0, x1, y0, y1).
    """
    x0, x1, y0, y1 = box
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    return float(np.abs(unwarp(warp(p, bones), bones) - p).max())
