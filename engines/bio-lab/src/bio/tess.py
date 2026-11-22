# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""tess.py - Voronoi tessellation, clipped, plus the neighbour graph.

WHY VORONOI AND NOT A VERTEX MODEL
----------------------------------
The classical way to simulate an epithelium is a VERTEX model: cells are polygons that
literally share vertices, and the state you integrate is the vertex positions. It is the
physically standard choice and it is what most of the literature uses.

It is not what this engine does, and the reason is worth writing down.

A vertex model has to maintain its own TOPOLOGY. As cells grow, divide and slide past one
another you need T1 transitions (two cells that were neighbours stop being neighbours and
two others start), T2 transitions (a shrinking cell disappears), and division has to
splice new vertices into two polygons at once while keeping every vertex list ordered and
every polygon simple. Every one of those operations can produce a degenerate cell -
self-intersecting, zero-area, two vertices in the same place - and when it does, the
failure shows up several hundred steps later as a NaN, far from its cause.

Here the tessellation is DERIVED, every frame, from the cell centres. Rearrangement is
automatic and always valid: if two cells slide past each other, next frame's Voronoi
diagram simply reflects it. There is no topology to corrupt, so there is no class of bug
here at all. Division is "insert a second point near the first" - two lines.

The trade is honest: forces come from a pair potential between centres rather than from
differentiating the tissue's energy with respect to vertex positions, so this is not a
quantitative epithelium. For an engine whose brief is to look right and respond to
growth, tension and pressure, that is the correct trade - and it is the one the brief
explicitly allows.


THE CLIP IS WHAT MAKES A COLONY LOOK LIKE A COLONY
--------------------------------------------------
An unbounded Voronoi diagram gives the outermost cells infinite regions. Rather than
special-casing them, every cell's polygon is intersected with a disc centred on its own
site, of a radius set by its target area. Interior cells are unaffected - their Voronoi
polygon is well inside their disc - while boundary cells get a smooth ROUNDED outer edge.

That single rule is what makes the outline read as a living colony bulging outward,
instead of as a diagram with a ragged convex edge.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import Voronoi, cKDTree
    HAVE_SCIPY = True
except Exception:                                     # pragma: no cover
    HAVE_SCIPY = False
    Voronoi = None


#: Number of half-planes used to approximate the clipping disc. 14 is visually round at
#: the sizes cells are drawn and costs 14 Sutherland-Hodgman passes per cell.
CLIP_SIDES = 14
_UNIT = np.stack([np.cos(np.linspace(0, 2 * np.pi, CLIP_SIDES, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, CLIP_SIDES, endpoint=False))], axis=1)


def _clip_halfplane(poly, n, d):
    """Sutherland-Hodgman: keep the part of `poly` where dot(p, n) <= d."""
    if len(poly) == 0:
        return poly
    out = []
    m = len(poly)
    for k in range(m):
        a = poly[k]
        b = poly[(k + 1) % m]
        da = float(a @ n) - d
        db = float(b @ n) - d
        if da <= 0:
            out.append(a)
        if (da > 0) != (db > 0):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return np.array(out) if out else np.zeros((0, 2))


def clip_to_disc(poly, centre, radius):
    """Intersect a polygon with a disc, approximated by CLIP_SIDES half-planes."""
    p = np.asarray(poly, np.float64)
    for n in _UNIT:
        p = _clip_halfplane(p, n, float(centre @ n) + radius)
        if len(p) < 3:
            return np.zeros((0, 2))
    return p


def voronoi_of(pos, pad=6.0):
    """The raw Voronoi diagram plus the real-site neighbour graph. NO POLYGON CLIPPING.

    Returns ``(vor, neighbours, n_neighbours, n_real)``.

    A ring of GHOST sites is added far outside the colony so every real site has a bounded
    region and scipy never reports an infinite ridge. They are dropped from the neighbour
    graph and are far enough out never to clip a real cell.
    """
    P = np.asarray(pos, np.float64)
    n = len(P)
    if n < 4 or not HAVE_SCIPY:
        return None, np.zeros((0, 2), int), np.zeros(n, int), n

    c = P.mean(axis=0)
    R = float(np.max(np.hypot(*(P - c).T))) * 1.6 + 600.0
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ghosts = c + R * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    vor = Voronoi(np.vstack([P, ghosts]))

    ridge = np.asarray(vor.ridge_points)
    keep = (ridge[:, 0] < n) & (ridge[:, 1] < n)
    nb = ridge[keep]
    cnt = np.zeros(n, int)
    if len(nb):
        np.add.at(cnt, nb[:, 0], 1)
        np.add.at(cnt, nb[:, 1], 1)
    return vor, nb, cnt, n


def polygons_of(vor, pos, radii, n_real):
    """Clipped cell polygons. CALL THIS ONLY WHEN DRAWING.

    THE SIMULATION DOES NOT NEED POLYGONS - it needs the neighbour graph, and that comes
    out of the Voronoi almost free. Clipping is a different matter entirely.
    Measured, per call:

        n=200   full tessellation 69.9 ms   Voronoi alone  2.0 ms   clipping 67.9 ms (97%)
        n=700   full tessellation 190.1 ms  Voronoi alone  4.2 ms   clipping 185.9 ms (98%)

    So clipping every substep - 120 of them per second - spent about 98% of the whole
    simulation budget producing polygons that were thrown away 119 times out of 120. At 60
    fps the polygons are wanted twice per twelve substeps; the rest were pure waste.

    The second saving is the `inside` fast path below. An interior cell's Voronoi polygon
    is entirely within its own clipping disc - the disc only ever bites on cells at the
    colony rim - so the expensive Sutherland-Hodgman pass is skipped for the large
    majority of a packed colony.
    """
    if vor is None:
        P = np.asarray(pos, np.float64)
        return [clip_to_disc(disc_poly(P[i], radii[i]), P[i], radii[i] * 1.0001)
                for i in range(len(P))]
    P = np.asarray(pos, np.float64)
    out = []
    for i in range(n_real):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            out.append(np.zeros((0, 2)))
            continue
        v = vor.vertices[reg]
        r = float(radii[i])
        d = v - P[i]
        if (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).max() <= r * r:
            out.append(v)                      # already inside the disc - nothing to do
        else:
            out.append(clip_to_disc(v, P[i], r))
    return out


def tessellate(pos, radii, pad=6.0):
    """Neighbours AND clipped polygons in one call. Convenience only - prefer the two
    halves above so the expensive half runs at frame rate, not at substep rate."""
    vor, nb, cnt, n = voronoi_of(pos, pad)
    return polygons_of(vor, pos, radii, n), nb, cnt


def disc_poly(centre, r, sides=20):
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return centre + r * np.stack([np.cos(a), np.sin(a)], axis=1)


def polygon_area(p):
    if len(p) < 3:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.abs((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum()))


def polygon_perimeter(p):
    if len(p) < 2:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return float(np.hypot(*(q - p).T).sum())


def chaikin(poly, iters=2):
    """Corner-cutting subdivision. Turns a 6-gon into a smooth ~24-gon.

    A Voronoi cell is a hard polygon, and at close range its straight facets and sharp
    corners read as blocky - real cell walls are not faceted. Chaikin replaces each corner
    with two points a quarter in from it, so EDGE MIDPOINTS STAY PUT and only the corners
    round off. Neighbouring cells therefore keep sharing their wall, and the small gaps
    that open at tri-cellular junctions are exactly the Plateau borders a real foam has.
    """
    p = np.asarray(poly, np.float64)
    for _ in range(int(iters)):
        if len(p) < 3:
            return p
        nxt = np.roll(p, -1, axis=0)
        q = np.empty((len(p) * 2, 2))
        q[0::2] = p * 0.75 + nxt * 0.25
        q[1::2] = p * 0.25 + nxt * 0.75
        p = q
    return p
