"""build.py - structure builders. Turn a few numbers into a list of obstacles.

Everything here returns plain `Capsule`/`Peg`/`Ring` lists, so a scene is mostly a handful of
these calls. Adding a new STRUCTURE belongs here; adding a new PRIMITIVE belongs in world.py.
"""
from __future__ import annotations

import math

from .world import TAU, Capsule, Peg

DEG = math.pi / 180.0


def polygon_shell(cx, cy, radius, n_sides, thickness=14.0, missing=(), rot=0.0,
                  omega=0.0, color=(255, 255, 255), name="poly", hp=-1, inset=0.0):
    """A regular polygon made of capsules, optionally spinning about its centre.

    `missing` = indices of sides to leave out (the gaps a ball can escape through). `inset`
    shortens each side at both ends, which WIDENS the corner openings - a cheap way to make a
    leaky cage without removing a whole side.
    """
    out = []
    for k in range(n_sides):
        if k in missing:
            continue
        a0 = rot + TAU * k / n_sides
        a1 = rot + TAU * (k + 1) / n_sides
        x0, y0 = cx + radius * math.cos(a0), cy + radius * math.sin(a0)
        x1, y1 = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
        if inset:
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            x0 += ux * inset
            y0 += uy * inset
            x1 -= ux * inset
            y1 -= uy * inset
        out.append(Capsule(x0, y0, x1, y1, thickness, pivot=(cx, cy), omega=omega,
                           color=color, name=f"{name}{k}", hp=hp))
    return out


def arc_bricks(cx, cy, radius, n, thickness=16.0, gap_deg=2.0, hp=1, span=TAU,
               rot=0.0, color=(255, 255, 255), name="brick", omega=0.0, aniso=1.0):
    """A shell built from n short capsule chords - a destructible ring. Each brick takes `hp`
    real impacts before it vanishes, so the ball opens its own exit.

    Give it `omega` to ROTATE the shell. A static brick ring is quickly defeated: the ball
    settles into one spot and chews a single hole while the rest of the shell is never touched.
    Turning the shell presents fresh bricks to the same impact point.

    `aniso` stretches the shell vertically into an ELLIPSE, to match an anisotropic orbital
    field (see World.aniso). Note that a rotating elliptical shell sweeps - its radius at a
    given angle changes as it turns - which is physical and fine, but do not also expect it to
    stay concentric with a stretched orbit.
    """
    out = []
    step = span / n
    for k in range(n):
        a0 = rot + step * k + gap_deg * DEG * 0.5
        a1 = rot + step * (k + 1) - gap_deg * DEG * 0.5
        out.append(Capsule(cx + radius * math.cos(a0), cy + radius * aniso * math.sin(a0),
                           cx + radius * math.cos(a1), cy + radius * aniso * math.sin(a1),
                           thickness, pivot=(cx, cy) if omega else None, omega=omega,
                           color=color, name=f"{name}{k}", hp=hp))
    return out


def spokes(cx, cy, r_in, r_out, n, thickness=13.0, omega=0.0, rot=0.0,
           color=(255, 255, 255), name="spoke"):
    """Radial paddles/gear teeth about a hub - a paddle wheel or a stirrer."""
    out = []
    for k in range(n):
        a = rot + TAU * k / n
        ca, sa = math.cos(a), math.sin(a)
        out.append(Capsule(cx + r_in * ca, cy + r_in * sa, cx + r_out * ca, cy + r_out * sa,
                           thickness, pivot=(cx, cy), omega=omega,
                           color=color, name=f"{name}{k}"))
    return out


def peg_grid(x0, x1, y0, dy, rows, cols, radius=11.0, stagger=True, e=0.92,
             color=(255, 255, 255), name="pin"):
    """A staggered pin field - the Galton board. Odd rows are offset half a pitch, which is
    what makes the walk actually branch instead of funnelling down fixed lanes."""
    out = []
    for r in range(rows):
        n = cols if (not stagger or r % 2 == 0) else cols - 1
        if n <= 0:
            continue
        span = x1 - x0
        pitch = span / max(1, cols - 1)
        off = 0.0 if (not stagger or r % 2 == 0) else pitch * 0.5
        for c in range(n):
            out.append(Peg(x0 + off + c * pitch, y0 + r * dy, radius, e=e,
                           color=color, name=f"{name}{r}_{c}"))
    return out


def funnel(cx, y, half_gap, half_width, drop, thickness=13.0, color=(255, 255, 255),
           name="fun"):
    """A V: two capsules sloping inward to a central opening of 2*half_gap."""
    return [
        Capsule(cx - half_width, y, cx - half_gap, y + drop, thickness,
                color=color, name=name + "L"),
        Capsule(cx + half_width, y, cx + half_gap, y + drop, thickness,
                color=color, name=name + "R"),
    ]


def spiral(cx, cy, r_start, r_end, turns, n_seg, thickness=12.0, rot=0.0,
           color=(255, 255, 255), name="spi", omega=0.0):
    """A chute: a polyline of capsules along an Archimedean spiral.

    Give it `omega`. A STATIC spiral in a downward gravity field does not work: every coil has
    a local low point, a ball rolls to it and stays, and the clip dies after two seconds
    (measured - 0 balls reached the floor in 15 s). Rotating the whole spiral moves that low
    point continuously, so the coil acts as a conveyor and balls cascade outward under gravity
    instead of parking. The scene still only chooses the geometry and its spin rate.
    """
    out = []
    pts = []
    for i in range(n_seg + 1):
        u = i / n_seg
        a = rot + TAU * turns * u
        r = r_start + (r_end - r_start) * u
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for i in range(n_seg):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        out.append(Capsule(x0, y0, x1, y1, thickness,
                           pivot=(cx, cy) if omega else None, omega=omega,
                           color=color, name=f"{name}{i}"))
    return out


def walls(x0, x1, y0, y1, thickness=16.0, color=(255, 255, 255), sides="lr", name="wall"):
    """Straight frame pieces. `sides` picks from l/r/t/b."""
    out = []
    if "l" in sides:
        out.append(Capsule(x0, y0, x0, y1, thickness, color=color, name=name + "L"))
    if "r" in sides:
        out.append(Capsule(x1, y0, x1, y1, thickness, color=color, name=name + "R"))
    if "t" in sides:
        out.append(Capsule(x0, y0, x1, y0, thickness, color=color, name=name + "T"))
    if "b" in sides:
        out.append(Capsule(x0, y1, x1, y1, thickness, color=color, name=name + "B"))
    return out


def pendulum_row(y, n, x0, x1, length, thickness=13.0, amp=50 * DEG, freq=0.35,
                 phase_step=0.5, color=(255, 255, 255), name="pend"):
    """A row of swinging bars hanging from evenly spaced pivots. Neighbouring bars are phased
    apart so the row reads as a travelling wave rather than a rigid comb."""
    out = []
    for k in range(n):
        u = k / max(1, n - 1)
        px = x0 + (x1 - x0) * u
        c = Capsule(px, y, px, y + length, thickness, pivot=(px, y),
                    swing_amp=amp, swing_freq=freq, color=color, name=f"{name}{k}")
        # phase offset via phase0 is not usable with swing (it biases the angle), so stagger
        # the FREQUENCY minutely instead - incommensurate rates never re-align into a comb
        c.swing_freq = freq * (1.0 + 0.08 * k)
        out.append(c)
    return out
