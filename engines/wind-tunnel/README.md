# Wind Tunnel

**A real fluid solver driving a colour-field animation.** Incompressible 2-D flow past bodies,
rendered as a colour-mapped speed field with advected white streaklines. It started as a wind
tunnel and is now more general than its name: the same solver runs a sealed pool of still water
seen from above, and three spinning holes that fall together and merge like droplets. What is
fixed is the *method*, not the subject.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Reference frames](frames/)** · 💾 **[Source](src/)**

---

## The one thing to understand

**Nothing here is animated by hand.** A composition may only put a shape somewhere at time *t*,
choose what a free body is made of and let go, or push the fluid somewhere. Every visible effect —
the suction peak, the separation bubble, the shed vortices, a tumbling plate, an eddy rolling up,
two droplets necking together — is the solver's answer, not a keyframe.

The legitimate inputs are exactly four, and all of them are things a real rig has:

| input | example | why it is not cheating |
|---|---|---|
| a body's POSITION at time *t* | sweeping incidence through stall | the rig holds the foil at an angle |
| a body's MATERIAL | `density_for_fall` | choosing what it is made of, not its path |
| a body's PRESCRIBED SPIN | `FreeBody.spin` | a motor turning a cylinder |
| a FORCING of the fluid | `set_inlet`, `set_drive` | a fan, a jet, a stirrer |

If a render looks wrong, the fix is a solver or geometry parameter — never a fudge in the
renderer. A composition must not be *able* to fake a fluid effect. That constraint is the whole
value of the project, and it is what makes the output trustworthy as well as good-looking.

## Two solvers, and the unit trap between them

| | `wt/lbm.py` | `wt/cns.py` |
|---|---|---|
| method | D2Q9 lattice-Boltzmann, BGK + Smagorinsky LES | finite-volume Navier-Stokes, MUSCL + minmod, HLLC, SSP-RK2 |
| fluid | effectively incompressible | real ideal gas, γ = 1.4 |
| gives you | free bodies, sealed boxes, dye transport, cheap | temperature, internal energy, genuine shock capture |
| body | half-way bounce-back on a boolean mask, with a moving-wall term | ghost-cell immersed boundary on the same mask |
| **`u0` means** | **a lattice velocity** | **a Mach number** (c∞ = 1 by construction) |

That last row is the single most dangerous thing in the project. The same field name means two
different physical quantities depending on which solver is selected, and a value that is a
comfortable lattice speed is a catastrophic Mach number.

Neither solver meshes anything. A body is a boolean mask re-rasterised from polygons every step,
which is why a body can move, rotate or change shape with nothing downstream noticing.

### The solvers are validated, not asserted

`src/tools/sod_check.py` runs the compressible solver against the **exact Sod shock-tube
solution**; `src/tools/shock_check.py` measures an oblique shock's angle and post-shock state
against the **θ–β–M relation**. Both ship. If you change anything in `cns.py`, run them.

## Quick start

```bash
pip install -r src/requirements.txt
```

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from wt import lbm, shapes

sim = lbm.LBM(nx=240, ny=420, u0=0.06, re=800)
foil = shapes.place(shapes.naca4("2412"), chord=90, cx=120, cy=140, aoa_deg=8)
sim.set_solid(shapes.rasterize([foil], 240, 420))
sim.run(400)

print(sim.health(), np.abs(sim.ux).max())
```

`wt/render.py` takes it from there — `Tunnel` owns the lattice-to-screen mapping, the frame clock,
the ffmpeg pipe and the divergence guard.

## Orientation

The solver always works in wind coordinates (lattice +x is streamwise); only the renderer maps
that to a screen. The same simulation shows three ways:

- **`up`** — streamwise is the screen's long axis, so the wake gets the full height. The vertical
  default.
- **`down`** — same lattice, same physics, running top to bottom. Use it when the subject belongs
  at the top and its effect belongs below it.
- **`right`** — the classic landscape tunnel. Correct, but a wake leaves frame after about two
  chords, so give it a wider `RenderConfig` than you think.

All three mappings are orientation-**preserving** on purpose. The obvious "+x down, +y right"
would be a reflection: every vortex on screen would spin the wrong way, vorticity would report the
wrong sign, and a cambered section would appear to lift backwards.

**Size bodies against the SCREEN, never against a lattice axis** — which axis is "across the
picture" flips with the flow direction, and a chord quoted in `nx` comes out about three times too
big.

## Nine rules for moving bodies

Each of these is a bug that was diagnosed and fixed. They are constraints, not suggestions.

1. **The wall force lives on the fluid cells *beside* the body**, never on its own cells. Getting
   this wrong measured a drag coefficient of about 22 against a textbook 1.2, with an arbitrary
   sign.
2. **Bounce-back must know the wall is moving** — and that term breaks above `|u_wall| ≈ 0.12`.
   Keep body speeds small and buy apparent speed with `steps`.
3. **Cap rotation by SURFACE SPEED** (`0.02/radius`), not by angular rate, in both integration and
   advection. A body with a prescribed spin opts out, and the composition then owns that budget.
4. **Never let a gap go sub-cell** — wall gaps, contact gaps, merge gaps.
5. **Nothing may materialise inside moving fluid.** Be present and frozen through the settling
   period, or enter from off-lattice — but never across the inlet plane. A shape that *changes*
   must change smoothly and at constant area.
6. **Low-pass the fluid load, and make contacts soft — including WALLS.** A hard clamp steps a
   body's wall velocity in one tick and the boundary broadcasts it as a visible ripple across the
   whole frame.
7. **Never sample the flow at a body's own centre** — it is inside its own solid. Sample the ring.
8. **Free-body scenes want modest Reynolds numbers** (450–900). A relaxation time near the
   stability limit leaves no margin for a moving boundary.
9. **Bodies are impenetrable to everything except a merge**, and this is free — bounce-back turns
   the fluid away and the streak advection kills any tracer that lands inside a solid.

## Two things not to "simplify" away

**Apparent speed = lattice speed × steps per frame.** Always buy speed with `steps`, never by
raising `u0`. Raising the lattice velocity is what breaks solves; running more steps per frame
costs time and nothing else.

**Superpose velocity fields — sum the components and take `max()` of the masks.** Weight-averaging
overlapping drive patches halves the target speed wherever two overlap while the drive weight
there is the *sum* of both, so the fluid between two neighbouring patches is driven hard toward a
target that is too slow. The resulting stagnation pressure pushes bodies apart: measured 2 cells/s
of spurious separation, cut to 0.19 by superposing correctly.

## A stable solve and a correct one are different questions

`LBM.health()` reports one number, and it answers only the first. The scar behind that sentence:
one composition printed COMPLETED with the project's best stability numbers while quietly drifting
its subjects apart for the whole clip. Anything with an *intent* needs a check that measures THAT
intent, not just that the solver did not blow up.

Several such checks exist upstream but import the withheld composition modules, so they are not
shipped. What they measured is worth reimplementing against your own work: separation drift over
the clip, apparent wind at a fixed probe against its nominal value, loop seam difference between
first and last frame, and mask coverage against the intended silhouette.

## Performance

Measured on an RTX 2060 / i7-9700K at 1080×1920. The LBM path parallelises to **three concurrent
lanes**, which is the NVENC session ceiling, not a CPU limit.

⚠️ **The three-lane rule does not transfer to `cns.py`.** The compressible solver is
bandwidth-bound, so concurrent lanes contend for memory rather than for encoder sessions and you
lose more than you gain.

`--overscan` pads the simulated domain beyond the visible frame, so bodies enter and leave
off-camera and an appearing body's pressure transient has room to decay. It costs
`(1 + 2·overscan)²` cells.

## What is not here

- **`wt/scenes.py`** — the compositions. See [CATALOG.md](CATALOG.md) for what each one
  demonstrates and frames from most of them.
- **`shapes.rocket()` and the `rocket_engine` composition** — withheld at the author's request.
  `wt/dye.py`, the D2Q5 species transport it used, *is* here: it is general, and `render.py`
  depends on it.
- **The composition-dependent check tools.** See the section above.

`wt/film.py` bridges to the CRT filter in the [oscilloscope engine](../oscilloscope/) rather than
duplicating a 300-line filter that would drift out of sync. It resolves that path relative to this
directory, so keep the two engines side by side or point `OSC_ROOT` elsewhere. Note that it must
be handed a **neutral white phosphor** config — the green defaults are only correct for a
single-hue trace and would wreck a full-colour field.
