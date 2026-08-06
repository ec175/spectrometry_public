"""lines.py - the field-line integrator. This is the core of the project.

    dp/ds = E(p) / |E(p)|      RK4, CONSTANT arclength step ds

THREE decisions here are load-bearing. Each is a bug if it is "simplified" away:

1. THE STEP IS IN ARCLENGTH, NOT TIME, AND IT IS CONSTANT.
   Normalising E to a unit direction costs one divide and buys two things everything downstream
   depends on: (a) drawn line density is uniform instead of piling up wherever the field is
   weak, and (b) a marker advected along a line at speed v is simply the array index
   round(v*t/ds) - no second integrator, no particle state, and no possibility of the marker
   drifting off the line it belongs to. `pulses.py` is ~40 lines because of this.

2. EVERY LINE IS INTEGRATED IN LOCKSTEP, NOT ONE AT A TIME.
   The state is an (N,2) array and the loop is over the arclength index. A per-line Python loop
   turns ~450 vector ops into ~90k scalar-ish ones and costs about 50x. Same launch-bound
   lesson as Wind_Tunnel AGENT_GUIDE section 4 and Shape_Physics rule 5 - the work is tiny, the
   per-call overhead is not.

3. LINES ARE COMPACTED AS THEY DIE.
   Most lines terminate on a sink long before s_max, and in the dense scenes most rows are dead
   a third of the way in. Carrying them through the remaining steps is pure waste.

Termination, in priority order: captured by a sink, stalled at a null point, left the bounds,
or ran out of arclength. A line ending on `s_max` (code 3) is a BUG, not a taste matter - it
means capture_r is smaller than ds (the line stepped straight over its sink) or the bounds are
wrong. `tools/flux_check.py` counts these; do not raise s_max to make the number go away.
"""
from __future__ import annotations

import numpy as np

# `ended` codes
END_SINK, END_NULL, END_BOUNDS, END_BUDGET, END_LOOP = 0, 1, 2, 3, 4
END_NAMES = ("sink", "null", "bounds", "BUDGET", "loop")


def seed_ring(center, r0, n_max, phase=0.0):
    """n_max seed points on a circle of radius r0 about `center`.

    Angles are FIXED at 2*pi*i/n_max for a given index i and are never re-spaced. That matters:
    if seeds were re-spaced whenever the live count changed, every line would shift and the
    whole fan would visibly SNAP on each integer crossing of an animated N(t). The caller
    allocates n_max once and controls how many are *visible* with per-seed alpha (see
    `fade_order`), so a line fades in exactly between two lines that never moved.
    """
    i = np.arange(n_max)
    th = 2.0 * np.pi * i / n_max + phase
    return np.stack([center[0] + r0 * np.cos(th), center[1] + r0 * np.sin(th)], axis=1)


def seed_rake(p0, p1, n_max, jitter=0.0, seed=0):
    """n_max seeds evenly along the segment p0->p1. The inlet rake for a flow scene."""
    u = (np.arange(n_max) + 0.5) / n_max
    P = np.stack([p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u], axis=1)
    if jitter:
        P += np.random.default_rng(seed).normal(0.0, jitter, P.shape)
    return P


def fade_order(n_max):
    """`rank[i]` = the position at which seed i switches on, as a bit-reversal permutation.

    Bit reversal means the first k seeds to appear, for ANY k, are quasi-uniformly spread around
    the circle - so a growing N(t) fills the fan in evenly instead of sweeping a wedge round
    from one side. The angles themselves never change; only which ones are lit.
    """
    bits = max(1, int(np.ceil(np.log2(max(2, n_max)))))
    idx = np.arange(1 << bits)
    rev = np.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    order = rev[rev < n_max]
    rank = np.empty(n_max, dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def trace(fieldfn, seeds, ds=6.0, n_steps=440, sinks=None, capture_r=16.0,
          bounds=(-280.0, -280.0, 1360.0, 2200.0), backward=False, e_floor=1e-9):
    """Integrate every seed. Returns (pts, length, ended).

    pts    (N, n_steps+1, 2) float32 - polylines, padded past `length` with the final point
    length (N,) int32                - valid sample count, so pts[i, :length[i]] is line i
    ended  (N,) int8                 - END_SINK / END_NULL / END_BOUNDS / END_BUDGET

    `ended` is not decoration: it is what lets a scene colour a line by its FATE (escaped vs
    captured) and it is what `flux_check` measures.
    """
    P = np.asarray(seeds, dtype=np.float64).copy()
    n = len(P)
    pts = np.empty((n, n_steps + 1, 2), dtype=np.float32)
    length = np.full(n, n_steps + 1, dtype=np.int32)
    ended = np.full(n, END_BUDGET, dtype=np.int8)
    pts[:, 0, :] = P
    sgn = -1.0 if backward else 1.0

    live = np.arange(n)
    x0, y0, x1, y1 = bounds
    S = None if sinks is None or len(sinks) == 0 else np.asarray(sinks, dtype=np.float64)
    cr2 = capture_r * capture_r
    stalled = np.zeros(n, dtype=bool)

    def u(Q, note=False):
        V = sgn * fieldfn(Q)
        m = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        if note:
            # a line running into a saddle/null asymptotically stops advancing; without this it
            # would burn the whole arclength budget standing still and report END_BUDGET
            np.copyto(stall, m < e_floor)
        np.maximum(m, e_floor, out=m)
        return V / m[:, None]

    for k in range(1, n_steps + 1):
        if len(live) == 0:
            break
        stall = np.zeros(len(P), dtype=bool)
        k1 = u(P, note=True)
        k2 = u(P + (0.5 * ds) * k1)
        k3 = u(P + (0.5 * ds) * k2)
        k4 = u(P + ds * k3)
        P = P + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        pts[live, k, :] = P

        dead = stall.copy()
        ended[live[stall]] = END_NULL
        if S is not None:
            d = P[:, None, :] - S[None, :, :]
            hit = (d[:, :, 0] ** 2 + d[:, :, 1] ** 2 < cr2).any(axis=1) & ~dead
            ended[live[hit]] = END_SINK
            dead |= hit
        out = ((P[:, 0] < x0) | (P[:, 0] > x1) |
               (P[:, 1] < y0) | (P[:, 1] > y1)) & ~dead
        ended[live[out]] = END_BOUNDS
        dead |= out

        if dead.any():
            length[live[dead]] = k + 1
            keep = ~dead
            live, P = live[keep], P[keep]

    # Pad each line's tail with its own final point (rows past death are np.empty garbage), in
    # ONE gather rather than a per-line slice assignment.
    kk = np.arange(n_steps + 1)[None, :]
    src = np.minimum(kk, (length - 1)[:, None]).astype(np.intp)
    pts = np.take_along_axis(pts, src[:, :, None], axis=1)

    # A line that used its whole arclength budget is normally a BUG - but not always. A closed
    # streamline (a point vortex, or a magnetostatic wire) genuinely never terminates, and that
    # is a topological case electrostatics cannot produce at all. Distinguish the two by asking
    # whether the line re-crossed its own earlier path: if it did, it is circulating, not stuck.
    # Costs one distance sweep over the handful of rows that hit the budget.
    over = np.nonzero(ended == END_BUDGET)[0]
    if len(over):
        # Test COILEDNESS, not exact re-crossing. A closed orbit re-crosses itself, but a
        # streamline trapped in a vortex sitting in a cross-stream spirals instead and never
        # returns to a previous point - the first detector looked for re-crossing and caught
        # none of them. End-to-end displacement against arclength catches both: a line that
        # travelled 2800 px of arc and finished 300 px from where it started is circulating,
        # whatever the shape of the coil.
        d = pts[over, -1] - pts[over, 0]
        ended[over[np.hypot(d[:, 0], d[:, 1]) < 0.25 * ds * n_steps]] = END_LOOP
    del stalled
    return pts, length, ended
