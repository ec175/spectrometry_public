# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""camera.py — one pinhole camera, and the small vocabulary of moves the brief asks for.

The brief wants "slow macro pushes, lateral tracking, controlled orbital cameras, overhead
views, perspective fly-throughs, and very subtle rotations", and then says *avoid excessive
camera movement*. So this is deliberately not a rig: it is `eye`, `target`, `up`, `fov`, a
projection, and a handful of named paths that produce those five things as a function of `t`.

Everything it returns is in the **reference frame** (1920x1080 px), like every other length in
this project. `canvas.py` multiplies by `cfg.scale` on the way to the device.

⚠ There is NO occlusion and there is not going to be. An additive particle field has no
surfaces — a nearer particle does not hide a farther one, it adds to it — so a depth buffer
would be not just wasted but *wrong*. The only depth cue that survives additive blending is
brightness, which is what `depth_gain` supplies, and it is the same cue
`..\\Attractors\\at\\draw.py` settled on for the same reason.
"""
from __future__ import annotations

import math

import numpy as np

from .gpu import xp_of


def _unit(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / (n if n > 1e-9 else 1.0)


def ease(u: float) -> float:
    """Smoothstep. Every camera move in this project starts and stops at zero velocity.

    Neon_Trace's gate hard-fails a "velocity jump at join", and the fix there was always
    easing. Getting it right once, here, means no scene can reintroduce it.
    """
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def ease_io(u: float, hold_in=0.0, hold_out=0.0) -> float:
    """Smoothstep with optional dead time at each end, in normalised units."""
    lo, hi = hold_in, 1.0 - hold_out
    if hi <= lo:
        return ease(u)
    return ease((u - lo) / (hi - lo))


class Camera:
    """A pinhole camera. `project` returns reference px + depth in world units."""

    def __init__(self, eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=38.0,
                 ref=(1920, 1080), roll=0.0):
        self.eye = np.asarray(eye, np.float32)
        self.target = np.asarray(target, np.float32)
        self.fov = float(fov)
        self.ref_w, self.ref_h = ref
        f = _unit(self.target - self.eye)
        upv = _unit(up)
        # A camera looking straight down has `forward` parallel to the default up vector and
        # the cross product degenerates to zero — the overhead scenes (12, 07) live exactly
        # there. Swing the reference up onto +Z when that happens; the choice of replacement
        # is arbitrary (any vector off the axis works) but it must be DETERMINISTIC, or the
        # frame rolls at random the first time a path passes through the pole.
        if abs(float(np.dot(f, upv))) > 0.999:
            upv = _unit((0.0, 0.0, 1.0) if abs(f[2]) < 0.9 else (0.0, 0.0, 1.0))
            if abs(float(np.dot(f, upv))) > 0.999:
                upv = _unit((1.0, 0.0, 0.0))
        r = _unit(np.cross(f, upv))
        u = np.cross(r, f)
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            r, u = r * c + u * s, u * c - r * s
        self.fwd, self.right, self.upv = f, _unit(r), _unit(u)
        #: focal length in reference px, from the VERTICAL fov
        self.flen = (self.ref_h * 0.5) / math.tan(math.radians(self.fov) * 0.5)

    def project(self, P):
        """P: (N,3) world -> (x, y, z) with x,y in reference px and z the camera depth.

        Particles at or behind the pupil come back with `z <= 0`; the caller culls on that.
        Dividing by a near-zero depth would fling a particle to +/-1e9 px and, because the
        splat clamps into the frame, smear a single bad particle across an edge.
        """
        xp = xp_of(P)
        rel = P - xp.asarray(self.eye, dtype=P.dtype)
        z = rel @ xp.asarray(self.fwd, dtype=P.dtype)
        x = rel @ xp.asarray(self.right, dtype=P.dtype)
        y = rel @ xp.asarray(self.upv, dtype=P.dtype)
        iz = self.flen / xp.maximum(z, np.float32(1e-3))
        return (self.ref_w * 0.5 + x * iz,
                self.ref_h * 0.5 - y * iz,
                z)


# --------------------------------------------------------------------------------------
# the moves — each returns a Camera for a given t
# --------------------------------------------------------------------------------------
def push(t, dur, dist0, dist1, target=(0, 0, 0), elev=8.0, azim=0.0, fov=38.0, **kw):
    """Slow macro push (or pull, if dist1 > dist0) along a fixed bearing."""
    d = dist0 + (dist1 - dist0) * ease(t / dur)
    e, a = math.radians(elev), math.radians(azim)
    tg = np.asarray(target, np.float32)
    eye = tg + np.float32(d) * np.array(
        [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)], np.float32)
    return Camera(eye, tg, fov=fov, **kw)


def orbit(t, dur, dist, azim0=0.0, sweep=40.0, elev0=12.0, elev1=None,
          target=(0, 0, 0), fov=38.0, **kw):
    """Controlled orbital camera: one eased sweep, never a constant spin.

    A constant-rate orbit is the single most obvious tell of procedural footage — it starts
    and ends mid-motion and the eye reads a turntable. An eased partial sweep reads as a shot.
    """
    u = ease(t / dur)
    a = math.radians(azim0 + sweep * u)
    el = math.radians(elev0 + ((elev1 - elev0) * u if elev1 is not None else 0.0))
    tg = np.asarray(target, np.float32)
    eye = tg + np.float32(dist) * np.array(
        [math.cos(el) * math.sin(a), math.sin(el), math.cos(el) * math.cos(a)], np.float32)
    return Camera(eye, tg, fov=fov, **kw)


def track(t, dur, dist, span, elev=6.0, target=(0, 0, 0), fov=38.0, axis=0, **kw):
    """Lateral tracking shot: eye and target slide together, so the parallax is real."""
    u = ease(t / dur)
    off = np.zeros(3, np.float32)
    off[axis] = np.float32(span * (u - 0.5))
    tg = np.asarray(target, np.float32) + off
    e = math.radians(elev)
    eye = tg + np.float32(dist) * np.array([0.0, math.sin(e), math.cos(e)], np.float32)
    return Camera(eye, tg, fov=fov, **kw)


def overhead(t, dur, height0, height1=None, tilt0=0.0, tilt1=0.0, target=(0, 0, 0),
             fov=42.0, spin=0.0, **kw):
    """Plan view, optionally easing into a slight tilt and/or a sub-degree roll.

    `spin` is a ROLL of the camera about its own axis, not an orbit — at plan view they look
    the same and cost very differently, and a roll cannot introduce parallax error.
    """
    u = ease(t / dur)
    h = height0 + ((height1 - height0) * u if height1 is not None else 0.0)
    tilt = math.radians(tilt0 + (tilt1 - tilt0) * u)
    tg = np.asarray(target, np.float32)
    eye = tg + np.float32(h) * np.array(
        [0.0, math.cos(tilt), math.sin(tilt)], np.float32)
    return Camera(eye, tg, fov=fov, roll=math.radians(spin * u), **kw)


def flythrough(t, dur, z0, z1, target_ahead=900.0, drift=0.0, fov=52.0, **kw):
    """Perspective fly-through: the eye travels along -Z and looks further along it.

    A wider default fov than the other moves on purpose — a fly-through with a long lens has
    almost no perspective flow and reads as a zoom.
    """
    u = ease(t / dur)
    z = z0 + (z1 - z0) * u
    x = drift * (u - 0.5)
    eye = np.array([x, 0.0, z], np.float32)
    tg = np.array([x * 0.3, 0.0, z - target_ahead], np.float32)
    return Camera(eye, tg, fov=fov, **kw)


def still(dist=1400.0, elev=8.0, azim=0.0, drift=0.0, t=0.0, dur=1.0,
          target=(0, 0, 0), fov=38.0, **kw):
    """Nominally static, with an optional very slow drift — "very subtle rotations".

    Nothing here is ever perfectly locked off. A truly static camera over a moving field looks
    like a screensaver; a fraction of a degree of drift is what makes it read as a camera.
    """
    a = math.radians(azim + drift * (t / max(dur, 1e-6)))
    e = math.radians(elev)
    tg = np.asarray(target, np.float32)
    eye = tg + np.float32(dist) * np.array(
        [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)], np.float32)
    return Camera(eye, tg, fov=fov, **kw)
