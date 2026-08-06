# Attractors

**Tens of thousands of particles integrated through a strange-attractor flow at once.** Nothing is
traced — the picture is built entirely by where particles crowd.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Reference frames](frames/)** · 💾 **[Source](src/)**

---

## Why a swarm and not an orbit

A single trajectory shows you the **shape** of an attractor. A swarm shows you its **invariant
measure** — which parts of that shape the system actually spends its time in. The bright regions
in these clips are not artistic emphasis; they are where the dynamics dwell.

That is a different measurement, not a restyling, and it is cheap: RK4 on an `(N,3)` array costs
the same four function evaluations as RK4 on one point, so 40,000 particles are essentially free
next to the rasteriser.

**Colour always encodes a real quantity** — speed, drift, distance from the manifold, age. That is
the rule that keeps this a visualisation rather than a screensaver, and it is also what makes the
pictures legible: every palette is built monotone in luminance, so a viewer can read "more" from
"brighter" without a legend.

## Two things that are easy to get wrong

**Seeding.** Particles started in a random ball are all *off* the attractor, and their approach
transients are long, bright and identical-looking — the first second of the clip becomes a
collapsing shell that has nothing to do with the dynamics. Seed **along a reference orbit** with a
little scatter instead, which puts every particle on the attractor from frame zero.

(The one composition that deliberately does the opposite is the point of that composition: seed
40,000 particles in a big uniform box and watch a structureless cloud collapse onto a
two-dimensional sheet in about four seconds, because the flow contracts volume everywhere.)

**Recycling.** Chaotic flows are volume-*contracting*, so an un-recycled swarm collapses onto a
thin filament within a few seconds and the picture loses every part of the attractor the
trajectory is not on right now. Respawning a few percent per second back onto the reference orbit
keeps the whole invariant measure lit.

## Quick start

```bash
pip install numpy scipy pillow
```

```python
import sys; sys.path.insert(0, "src")
from at import core, draw

s = core.Swarm("lorenz", n=42000, seed=1)
canvas = draw.Canvas(1080, 1920, palette="ember", trail=0.86)

for i in range(900):                       # 15 s at 60 fps
    s.step(1 / 60.)
    s.respawn(0.055 / 60.)
    P, z, u = ...                          # project, then splat
```

`core.project` returns a 2-D position **and a depth**. Use the depth. With 40,000 additive points
a purely orthographic view turns a 3-D object into a flat smear; a weak perspective divide plus
depth-keyed size and brightness is enough to separate the front and back lobes.

## Six flows, and how to add one

`FLOWS` maps a name to a function, a suggested timestep, a seed point and a display span. Adding
one is a vectorised `f(P) -> dP/dt` plus a row in that dict — nothing else changes.

| flow | dt | what it looks like |
|---|---|---|
| `lorenz` | 0.0045 | the butterfly — two lobes with a slow crawl through the middle |
| `rossler` | 0.0180 | a single spiral band with a periodic fold out of plane |
| `aizawa` | 0.0095 | a torus-like shell with a spike through its axis — the most three-dimensional |
| `halvorsen` | 0.0058 | three-fold symmetric, knotted |
| `thomas` | 0.0420 | cyclically symmetric, built from sines — loops rather than lobes |
| `chen` | 0.0040 | a double-scroll relative of Lorenz with a broader sweep |

## The morph, which is the one non-obvious trick

To show four attractors as **one object**, interpolate the **derivative**, not the picture:

```
dP/dt = (1-u)·f_A(P) + u·f_B(P)
```

Because the blend is of the vector *fields*, every intermediate is a real dynamical system with
its own trajectories. The swarm is never teleported and never cross-faded — watching a Lorenz
butterfly stretch into a Halvorsen knot through a sequence of attractors nobody has named is the
whole reason to do it that way.

Two details it needs: **normalise each field to a common speed** before blending, or the transit
either stalls or explodes as one system's characteristic rate dominates; and **renormalise the
view on the LIVE swarm**, because the tour leaves the seed attractor's bounding box entirely.

Also: **the spin must be faster than the morph.** A morph slower than the view rotation reads as a
still object being turned, not as one system becoming another.

## Exposure

The one place this format blows out is where the swarm *concentrates*. A collapse onto a 2-D sheet
saturates the additive buffer and the core goes flat white — losing exactly the structure the
collapse was meant to show. Lower both exposure and per-point gain for those compositions rather
than trying to fix it afterwards.

## Orientation

`project` returns normalised coordinates and `Canvas` decides the frame, so nothing in the
dynamics knows the aspect ratio. A landscape plate is a different `Canvas` size and nothing else.

## What is not here

- **`at/scenes.py`** — the five compositions. See [CATALOG.md](CATALOG.md); each one is a
  different *measurement* rather than a different palette, and that distinction is the reusable
  part.
- One composition, a single true orbit drawn as a ribbon with a halo of particles it has captured,
  was built but never rendered — so it has no frames.

The oscilloscope engine has its own, unrelated attractor treatment: **one** beam tracing a single
orbit, with the figure emerging from phosphor persistence, which is exactly what a real X-Y scope
does with a pair of voltages. Same systems, opposite measurement.
