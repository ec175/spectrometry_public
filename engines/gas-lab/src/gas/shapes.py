# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""shapes.py - the outlines of the 2026-09-25 car parts, as polygons. Pure numpy, no drawing.

Every function returns an (N, 2) float array, a closed ring (the last point is NOT repeated),
in the part's own local frame, reference px, y DOWN. A solid is built from rings by CSG in
sdf.py (add / subtract), so a rotor with a bore is `add(lobes) - add(circle)`.

The profiles here are the real ones, derived rather than eyeballed, because the gas meets
them: a Roots rotor is the cycloidal pair (epicycloid tips, hypocycloid roots on one rolling
circle, which is exactly conjugate), a Wankel housing is the epitrochoid its own apexes trace,
a scroll wrap is the involute of a base circle, a vane pump ring is an offset circle the vane
tips actually ride. If the outline were approximate, the seal lines would leak or collide.
"""
from __future__ import annotations

import math

import numpy as np

TAU = 2.0 * math.pi


def rot(P, ang, c=(0.0, 0.0)):
    """Rotate points by `ang` radians about c (clockwise on screen, y down)."""
    P = np.asarray(P, dtype=np.float64)
    cs, sn = math.cos(ang), math.sin(ang)
    d = P - np.asarray(c)
    return np.stack([c[0] + cs * d[:, 0] - sn * d[:, 1], c[1] + sn * d[:, 0] + cs * d[:, 1]], 1)


def shift(P, dx, dy):
    return np.asarray(P, dtype=np.float64) + np.array([dx, dy])


def circle(cx, cy, r, n=None):
    n = n or max(48, int(TAU * r / 3.0))
    a = np.linspace(0.0, TAU, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def ellipse(cx, cy, rx, ry, n=None):
    n = n or max(48, int(TAU * max(rx, ry) / 3.0))
    a = np.linspace(0.0, TAU, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def rect(x0, y0, x1, y1):
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)


def rrect(x0, y0, x1, y1, r, n=10):
    """A rectangle with rounded corners (a casting, not a sheet)."""
    r = max(0.0, min(r, 0.5 * (x1 - x0), 0.5 * (y1 - y0)))
    if r <= 0.0:
        return rect(x0, y0, x1, y1)
    pts = []
    for (cx, cy, a0) in ((x1 - r, y0 + r, -90.0), (x1 - r, y1 - r, 0.0),
                         (x0 + r, y1 - r, 90.0), (x0 + r, y0 + r, 180.0)):
        for i in range(n + 1):
            a = math.radians(a0 + 90.0 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def arc(cx, cy, r, a0, a1, n=None):
    """Points on an arc from a0 to a1 (degrees, screen-clockwise), inclusive."""
    n = n or max(8, int(abs(math.radians(a1 - a0)) * r / 3.0))
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def annulus_sector(cx, cy, r0, r1, a0, a1):
    return np.concatenate([arc(cx, cy, r1, a0, a1), arc(cx, cy, r0, a1, a0)])


def capsule(p, q, r, n=16):
    """A stadium from p to q with radius r (a slot, a rod, a pushrod)."""
    (x0, y0), (x1, y1) = p, q
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return np.concatenate([arc(x1, y1, r, ang - 90, ang + 90, n), arc(x0, y0, r, ang + 90, ang + 270, n)])


def polyline_band(pts, half, n_cap=6):
    """A band of half-width `half` along an open polyline (a duct centreline -> its outline).
    Offsets by the averaged normal at each vertex; fine for gentle bends."""
    P = np.asarray(pts, dtype=np.float64)
    d = np.gradient(P, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    left = P + nrm * half
    right = P - nrm * half
    return np.concatenate([left, right[::-1]])


def bezier(p0, p1, p2, p3, n=48):
    t = np.linspace(0.0, 1.0, n)[:, None]
    P = [np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3)]
    return ((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1] + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]


# ---------------------------------------------------------------- gears and rotors
def roots_rotor(rp, n=3, samples=90):
    """A cycloidal Roots rotor with `n` lobes on pitch radius rp, centred on the origin, one
    lobe TIP on the +x axis. The addendum is an epicycloid and the dedendum a hypocycloid of the
    SAME rolling circle a = rp/(2n), so two of these on centres 2*rp apart, counter-rotating at
    equal speed with one turned by pi/n, are exactly conjugate: the profiles roll on each other.
    Tip radius rp(1 + 1/n), root radius rp(1 - 1/n)."""
    a = rp / (2.0 * n)
    span = math.pi / n                         # each arch spans pi/n of the pitch circle
    pts = []
    for k in range(2 * n):
        t = np.linspace(0.0, span, samples, endpoint=False)
        if k % 2 == 0:      # epicycloid arch (addendum), starting on the pitch circle
            x = (rp + a) * np.cos(t) - a * np.cos((rp + a) / a * t)
            y = (rp + a) * np.sin(t) - a * np.sin((rp + a) / a * t)
        else:               # hypocycloid arch (dedendum)
            x = (rp - a) * np.cos(t) + a * np.cos((rp - a) / a * t)
            y = (rp - a) * np.sin(t) - a * np.sin((rp - a) / a * t)
        P = rot(np.stack([x, y], 1), k * span)
        pts.append(P)
    P = np.concatenate(pts)
    # the first epicycloid arch is centred on angle span/2: turn so a TIP sits on +x
    return rot(P, -span / 2.0)


def involute_gear(rp, z, pa_deg=20.0, samples=12, add=1.0, ded=1.25):
    """A spur gear outline: involute flanks, module m = 2 rp / z. Tooth centred on +x."""
    m = 2.0 * rp / z
    rb = rp * math.cos(math.radians(pa_deg))
    ra, rf = rp + add * m, max(rp - ded * m, 0.2 * rp)
    # half tooth thickness angle at the pitch circle, plus the involute function there
    inv = lambda a: math.tan(a) - a                                        # noqa: E731
    a_p = math.acos(rb / rp)
    half = math.pi / (2 * z) + inv(a_p)
    rs = np.linspace(max(rb, rf), ra, samples)
    flank = []
    for r in rs:
        al = math.acos(min(1.0, rb / r))
        flank.append((r, half - inv(al)))
    pts = []
    for k in range(z):
        c = TAU * k / z
        # root arc before the tooth
        for a in np.linspace(c - math.pi / z, c - flank[0][1], 4, endpoint=False):
            pts.append((rf * math.cos(a), rf * math.sin(a)))
        for r, th in flank:
            pts.append((r * math.cos(c - th), r * math.sin(c - th)))
        for r, th in flank[::-1]:
            pts.append((r * math.cos(c + th), r * math.sin(c + th)))
        for a in np.linspace(c + flank[0][1], c + math.pi / z, 4, endpoint=False):
            pts.append((rf * math.cos(a), rf * math.sin(a)))
    return np.array(pts)


def epitrochoid(e, R, samples=720):
    """The Wankel housing bore: the curve an apex traces. x = e cos 3a + R cos a,
    y = e sin 3a + R sin a. Major axis along x."""
    a = np.linspace(0.0, TAU, samples, endpoint=False)
    return np.stack([e * np.cos(3 * a) + R * np.cos(a), e * np.sin(3 * a) + R * np.sin(a)], 1)


def wankel_rotor(e, R, samples=720, shrink=0.0):
    """The rotor flank: the inner envelope of the epitrochoid in the rotor's own frame. For a
    rotor at angle phi about its centre, which sits at e(cos 3phi, sin 3phi), the housing
    point for rotor angle a is H(a); in rotor coords that is R(-phi)(H(a) - c). The rotor is
    the region every such housing picture leaves free - computed by taking, for each ray from
    the rotor centre, the minimum radius the housing reaches over all phi. The apexes land at
    radius R exactly (they ARE the generating points)."""
    n_ray = samples
    ray = np.linspace(0.0, TAU, n_ray, endpoint=False)
    best = np.full(n_ray, np.inf)
    H = epitrochoid(e, R, 1440)
    for phi in np.linspace(0.0, TAU / 3.0, 240, endpoint=False):
        c = np.array([e * math.cos(3 * phi), e * math.sin(3 * phi)])
        P = rot(H - c, -phi)
        ang = np.mod(np.arctan2(P[:, 1], P[:, 0]), TAU)
        rad = np.hypot(P[:, 0], P[:, 1])
        idx = np.round(ang / TAU * n_ray).astype(int) % n_ray
        np.minimum.at(best, idx, rad)
    # fill any ray bins the sampling missed
    m = np.isfinite(best)
    best = np.interp(np.arange(n_ray), np.nonzero(m)[0], best[m], period=n_ray)
    best = best - shrink
    return np.stack([best * np.cos(ray), best * np.sin(ray)], 1)


def involute_wrap(a, t, turns, phi0=0.0, samples_per_turn=160, start=0.35):
    """A scroll wrap: the band between the involutes of a base circle (radius a) with
    starting angles phi0 and phi0 + t/a, i.e. a wall of thickness t. Both involutes run from
    involute angle `start` turns to `turns` turns. Returned as one closed ring."""
    n = int(samples_per_turn * (turns - start)) + 2
    s = np.linspace(start * TAU, turns * TAU, n)

    def inv(s, ph):
        return np.stack([a * (np.cos(s + ph) + s * np.sin(s + ph)),
                         a * (np.sin(s + ph) - s * np.cos(s + ph))], 1)
    outer = inv(s, phi0)
    # the inner flank is the involute of phase +t/a taken at parameter s - t/a: the point with
    # the SAME normal as outer(s), so the two flanks are parallel at distance t and each end
    # cap is a short square end of length t (not a chord across the spiral)
    inner = inv(s - t / a, phi0 + t / a)
    return np.concatenate([outer, inner[::-1]])


def ring_band(outer, inner):
    """Concatenate two rings for callers that want both (sdf.py handles holes by CSG)."""
    return outer, inner
