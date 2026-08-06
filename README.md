# spectrometry_public

**Rendering profiles, and simulated object libraries used in my videos**

[![YouTube](https://img.shields.io/badge/YouTube-spectrometry.mp4-red)](https://www.youtube.com/channel/UChtdNI2BC1SmkmHEERA4dzg)
[![Instagram](https://img.shields.io/badge/Instagram-@spectrometry.mp4-E4405F)](https://instagram.com/spectrometry.mp4)
[![X](https://img.shields.io/badge/X-@spectrometrymp4-black)](https://x.com/spectrometrymp4)

---

## Getting started

Grab the repo:

```bash
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public
```

You need **Python 3.10 or newer** and **ffmpeg** on your PATH. That is genuinely it for most of
what is here. A CUDA GPU makes some things about ten times faster but nothing requires one, and
every GPU path falls back to CPU on its own if the card is missing or busy.

Checking ffmpeg is there:

```bash
ffmpeg -version
```

If that fails, grab a build from [ffmpeg.org](https://ffmpeg.org/download.html) and put it on your
PATH. On Windows the gyan.dev full builds work fine. You can also drop `ffmpeg.exe` into a `bin/`
folder next to whichever engine you are running and it will be found there.

### Then pick a direction

There are two halves to this repo and they are independent.

**Want to render something right now?** Go to [`chemical_profiles/`](chemical_profiles/). It is a
complete, working manim scene with a worked example molecule, and it will produce a finished video
in about ten minutes from a cold clone. Instructions are a few paragraphs down.

**Want the pieces to build your own thing?** Go to [`engines/`](engines/). Eight libraries covering
fluid dynamics, field lines, particle swarms, rigid-body physics, a simulated CRT, and publication
figure generation. Each one is standalone, each has a catalogue of everything it can make, and each
comes with reference stills so you can see what you are getting before you install anything.

The [full catalogue](CATALOG.md) lists all 193 objects and 83 compositions in one place if you want
to browse.

### Repo layout

```
chemical_profiles/     the manim scene, ready to render
engines/               eight libraries, each with src/ frames/ and a catalogue
  academia/            publication figures for compound classes
  attractors/          strange-attractor particle swarms
  chemical-scope/      29 molecules on a simulated CRT
  field-lines/         electrostatics, potential flow, magnetostatics
  lattice-grid/        illuminated circuit lattices
  oscilloscope/        the CRT itself, plus its film and glitch filters
  shape-physics/       discs, shells, destructible structures
  wind-tunnel/         two fluid solvers
tools/                 catalogue generator
CATALOG.md             everything, indexed
```

---

## Render a chemical profile

A vertical short showing one molecule and the spectra that identify it. The structure spins over
its ¹³C and ¹H NMR spectra with the peaks numbered to match the atoms, cross-fades to an FTIR
trace, then plays each infrared vibration mode while a cursor tracks the band it produces.

```bash
cd chemical_profiles

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

manim -qh -r 1080,1920 chemical_profile.py VerticalProfile_Aspirin
```

That is the full-quality render and it takes a while. For a fast look, swap `-qh` for `-ql`. Your
video lands in `chemical_profiles/renders/videos/chemical_profile/1920p30/`.

**You do not need LaTeX.** This trips people up because most manim projects require it. These
scenes use Pango text throughout instead of `MathTex`, so you can skip the entire MiKTeX or TeX
Live install that the manim docs walk you through.

### Adding your own molecule

One subclass. Every molecule-specific value is a class attribute, and `VerticalProfile_Aspirin` is
the template. Read it top to bottom and you have seen the whole data contract.

[**GUIDE.md**](chemical_profiles/GUIDE.md) covers installing from scratch, what every quality flag
actually does, how to read the output tree, authoring a molecule attribute by attribute, and what
to do when something breaks.

### About the spectra

They are representative, not measured. Peak positions come from literature values and
group-contribution estimates, hand-corrected so the diagnostic bands land where they should. Good
enough to teach with. Not a substitute for running the instrument.

If you author a molecule, check every shift and every band against a reference before you render.
Nothing in the code validates them, and a wrong assignment is a factual error sitting on screen in
front of whoever watches it.

---

# The engines

Seven of these compute something real, in numpy, and pipe raw frames straight to ffmpeg. No scene
graph, no timeline, no manim. The eighth makes the publication figures the whole thing grew out of.

They are published as **parts** rather than as finished videos. You get the solvers, the shape
generators, the field kernels, the physics primitives, the colour maps, the post filters, and the
render pipeline that drives all of it. What you do not get is my `scenes.py` files, which is where
the actual compositions live. That is on purpose: I would rather hand over a toolbox than a way to
re-emit my back catalogue.

Every composition is still catalogued, with a description of what it demonstrates and stills from
it, because the ideas transfer even when the code does not.

| engine | in one line |
|---|---|
| [Wind Tunnel](#wind-tunnel) | real 2-D fluid dynamics, two solvers |
| [Field Lines](#field-lines) | field lines of any closed-form vector field |
| [Lattice Grid](#lattice-grid) | illuminated circuit lattices with a strange colour model |
| [Shape Physics](#shape-physics) | bouncing shapes, and audio the simulation triggers |
| [Attractors](#attractors) | chaotic flows drawn by particle density |
| [Oscilloscope](#oscilloscope) | a simulated CRT and a very good film filter |
| [Chemical Scope](#chemical-scope) | 29 molecules with FTIR and Raman, on that CRT |
| [Academia](#academia) | stacked spectra figures for whole compound classes |

---

## Wind Tunnel

[Catalogue](engines/wind-tunnel/CATALOG.md) · [Stills](engines/wind-tunnel/frames/) ·
[Source](engines/wind-tunnel/src/) · [Full notes](engines/wind-tunnel/README.md)

Actual computational fluid dynamics driving a colour-field animation. Two solvers share one
interface:

`lbm.py` is a D2Q9 lattice-Boltzmann scheme with a Smagorinsky turbulence model. It gives you free
bodies the flow genuinely pushes around, sealed containers, and coloured dye transport, and it is
fast.

`cns.py` is a compressible Navier-Stokes solver with MUSCL reconstruction and an HLLC Riemann
solver. Real ideal gas at γ = 1.4, so you get temperature, internal energy, and honest shock
capture. It is validated against the exact Sod shock-tube solution and against oblique-shock
theory, and both of those checks ship with it.

Nothing on screen is keyframed. A scene can put a shape somewhere, choose what a free body is made
of, or push the fluid, and then it has to let go. Every vortex, every separation bubble, every
tumbling plate is the solver's answer.

### Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r engines/wind-tunnel/src/requirements.txt
```

```python
import sys; sys.path.insert(0, "engines/wind-tunnel/src")
import numpy as np
from wt import lbm, shapes

sim  = lbm.LBM(nx=240, ny=420, u0=0.06, re=800)
foil = shapes.place(shapes.naca4("2412"), chord=90, cx=120, cy=140, aoa_deg=8)
sim.set_solid(shapes.rasterize([foil], 240, 420))
sim.run(400)

print(f"stability {sim.health():.3f}, peak speed {np.abs(sim.ux).max():.4f}")
```

Bodies are boolean masks rasterised from polygons every step, so there is no meshing anywhere. You
can move them, rotate them, or change their shape mid-clip and nothing downstream notices.

There are 18 shape generators including NACA sections, cylinders, plates, wedges, a jet-engine
cross-section, a surfboard, and a cow. `image_body()` traces a polygon straight out of a PNG alpha
channel, so anything you can draw becomes a solid the fluid flows around.

### Watch out for

**`u0` means two different things.** Under the lattice-Boltzmann solver it is a lattice velocity
and lives around 0.02 to 0.10. Under the compressible solver it is a Mach number. A comfortable
value in one is catastrophic in the other, and this is the single easiest way to waste an hour
here.

**Buy speed with `steps`, never with `u0`.** Apparent speed is lattice speed multiplied by steps
per frame. Raising the lattice velocity is what breaks solves; running more steps costs wall clock
and nothing else.

**Size bodies against the screen, not against a lattice axis.** Which axis is "across the picture"
flips with the flow direction, and a chord quoted in the wrong one comes out about three times too
big.

The [engine README](engines/wind-tunnel/README.md) has nine hard rules for moving bodies, each of
which is a bug I already hit and fixed.

---

## Field Lines

[Catalogue](engines/field-lines/CATALOG.md) · [Stills](engines/field-lines/frames/) ·
[Source](engines/field-lines/src/) · [Full notes](engines/field-lines/README.md)

Field lines and streamlines of any closed-form vector field, drawn as neon polylines with packets
riding along them. Electrostatics, potential flow, and magnetostatics all come out of the same
integrator, because a field here is just a callable.

Nothing solves a PDE. Every field is a closed form evaluated at an array of points, so there is no
lattice, no timestep, no CFL condition, and nothing can blow up.

The step is in **arclength**, not time, and it is constant. That normalisation costs one divide and
buys two things. Line density comes out uniform instead of piling up wherever the field is weak.
And a packet moving at speed *v* becomes the array index `round(v*t/ds)`, which is why the entire
pulse module is a hundred lines with no integration in it at all.

### Getting started

```bash
pip install -r engines/field-lines/src/requirements.txt
```

```python
import sys; sys.path.insert(0, "engines/field-lines/src")
import numpy as np
from fl import field, lines

charges = field.ChargeSet().set([(0., 0.), (0., 300.)], [6.0, -2.0])
seeds   = lines.seed_ring((0., 0.), r0=40., n_max=48)

pts, length, ended = lines.trace(
    charges, seeds, ds=6.0, n_steps=440,
    sinks=charges.sinks(), capture_r=16.0,
    bounds=(-540., 540., -960., 960.))

for code, name in enumerate(lines.END_NAMES):
    print(f"{name:8s} {np.count_nonzero(ended == code)}")
```

Swap `charges` for a `PotentialFlow` and every line above still runs unchanged. That is the whole
design.

### Reading the output

Every line records how it ended, and those counts are your diagnostic:

| code | meaning |
|---|---|
| `sink` | captured by a negative source, the normal fate |
| `null` | stalled at a saddle point, legitimate |
| `bounds` | left the frame, legitimate |
| `BUDGET` | ran out of arclength. **This is a bug.** |
| `loop` | closed or spiralling, only possible for vortex fields |

Do not raise `n_steps` to make `BUDGET` go away. It means your capture radius is smaller than your
step size, so lines are stepping straight over their own sinks, or your bounds are wrong.

One result worth knowing before you build a potential flow: one doublet per body is only exact for
an isolated body. Two cylinders in one flow each violate the other's boundary condition, badly. Use
one cylinder plus Milne-Thomson images of everything else, which is exact for arbitrary external
flow. That took the worst surface-normal velocity from 0.81 down to 0.0005.

---

## Lattice Grid

[Catalogue](engines/lattice-grid/CATALOG.md) · [Stills](engines/lattice-grid/frames/) ·
[Source](engines/lattice-grid/src/) · [Full notes](engines/lattice-grid/README.md)

A fixed lattice of nodes and connectors lit by a moving field. The lattice never moves. What
changes is the content sitting on it, which re-rolls on the musical beat, and the illumination
sweeping across.

The interesting part is the colour, which is not a palette at all. The render is black and white,
and every colour you see comes from **delaying the RGB channels against each other**: blue
undelayed, green about 0.1 s behind, red about 0.2 s. An element appearing shows blue first and
reads cyan. One fading still has its red after the newer channels have gone and reads orange.
Anything steady has all three channels equal and comes out white.

I spent three sessions building an elaborate two-scale hue field with a calibrated bimodal palette
before working this out. All of it was fitting the shadow. If you ever reverse-engineer footage,
cross-correlate the R, G and B channels as time series first and look for a peak away from lag
zero. It is twenty lines and it would have saved me a week.

### Getting started

```bash
pip install -r engines/lattice-grid/src/requirements.txt   # numpy, Pillow, scipy
```

```python
import sys; sys.path.insert(0, "engines/lattice-grid/src")
from lg import lattice, config

for kind in ("square", "honeycomb", "triangular"):
    L = lattice.build(kind, 1080, 1920, config.SOURCE_PITCH)
    print(f"{kind:12s} {len(L.sites):5d} sites  {len(L.edges):6d} edges")
```

```
square        5858 sites   22957 edges
honeycomb     5252 sites    7771 edges
triangular    7140 sites   21063 edges
```

Swapping geometry is one class attribute. Adding a new one means a `Lattice` subclass providing
sites, edges, edge kinds and a box sampler, and nothing downstream changes.

`tools/lattice_check.py` verifies the substrate: coordination numbers, stamp lengths against real
edge lengths, and that a box is actually a closed ring.

### Watch out for

**Bloom is the enemy of this look.** A halo averages many elements, so it changes slowly even when
the ink underneath is switching hard. Slow-changing means white, and white is the one thing a
colour-delay effect cannot afford.

**Density controls brightness, not exposure.** A clipped pixel is white by definition. Keep
exposure low and get brightness from coverage instead.

**Grain sets your bitrate.** Per-frame noise is incompressible, so encode cost here is almost
entirely down to the grain setting. The usual 40 Mbit ceiling destroys about half of it, and what
h.264 leaves behind is correlated blocking rather than independent noise. Use cq 17 at 70 Mbit.

---

## Shape Physics

[Catalogue](engines/shape-physics/CATALOG.md) · [Stills](engines/shape-physics/frames/) ·
[Source](engines/shape-physics/src/) · [Full notes](engines/shape-physics/README.md)

Discs, spinning shells with gaps, destructible brick rings, plinko pins, pendulums, gears. The
satisfying-shapes format.

The audio relationship here runs backwards compared to everything else I have built. Normally a
song gets analysed into a band track and the render reads it. Here **the simulation fires events
and the events decide when the song plays**.

The rule that makes it work: each event plays the next half-second of the song, and an event
arriving while a segment is already sounding *extends* that segment rather than retriggering it.
Most videos of this kind retrigger, so two events a few frames apart either stack two copies of the
sample or hard-cut back to its start. Extending means an extended segment is one continuous read of
the song. No overlap, no abrupt stop while events keep arriving.

### Getting started

```bash
pip install numpy pillow   # that is the whole dependency list
```

```python
import sys; sys.path.insert(0, "engines/shape-physics/src")
from sim import world, build, audio

obstacles  = build.polygon_shell(540, 960, 210, n_sides=6,
                                 thickness=14., missing=(0,), omega=1.2)
obstacles += build.walls(60, 1020, 100, 1820)

w = world.World(obstacles, cx=540, cy=960, width=1080, height=1920,
                gravity=1500., initial=1)
for _ in range(2 * 900):          # 2 seconds at the 900 Hz physics rate
    w.step(1 / 900.)

print(f"{len(w.bounces)} bounces, {len(w.triggers)} headline events")

score = audio.Score(slice_len=0.5)
for t in (1.0, 1.2, 3.0):
    score.fire(t)
print(f"{len(score.segments)} segments from 3 events")   # 2, the first two merged
```

**Run the simulation with no drawing before you render anything.** It takes about a third of a
second for a twenty-second clip and it prints your escape counts, every trigger time, and the
resulting audio segments. Tune gap width, spin, friction and gravity against that. Never against
renders.

Three primitives build everything: a `Ring` (hollow shell with gaps), a `Capsule` (a segment swept
by a disc, which becomes polygons, funnels, chutes, spirals, paddles, pendulums and bricks), and a
`Peg` (a disc, optionally orbiting, optionally with restitution above 1 so it is a real energising
bumper). They all resolve through the same two functions, so they interact correctly with each
other for free.

### Watch out for

**Bounces resolve in the wall's frame.** A spinning shell has to *fling* a ball, not just stop it.
Set friction to zero, or do the reflection in the world frame, and balls rattle to the bottom and
sit there waiting for a gap.

**A gap must clear the ball, not a point.** A ball of radius *r* at shell radius *R* needs
`2·asin(r/R)` of arc just to fit through.

**Read spin rates against your clip length.** I once shipped a hexagon whose one missing side
passed the bottom of frame every 11 seconds, so in a 15-second clip the ball just sat and waited.
Seven escapes became zero.

**A rotating spiral is an Archimedes screw and only one sign conveys outward.** Measured over 15
seconds: one direction gave 16 escapes, the other gave zero.

---

## Attractors

[Catalogue](engines/attractors/CATALOG.md) · [Stills](engines/attractors/frames/) ·
[Source](engines/attractors/src/) · [Full notes](engines/attractors/README.md)

Forty thousand particles integrated through a chaotic flow at once. Nothing is traced. The picture
is built entirely by where particles crowd.

A single orbit shows you the *shape* of an attractor. A swarm shows you its **invariant measure**,
meaning which parts of that shape the system actually spends its time in. The bright regions are
not artistic emphasis, they are where the dynamics dwell.

It is also cheap. RK4 on an (N,3) array costs the same four function calls as RK4 on one point, so
the particles are essentially free next to the rasteriser.

### Getting started

```bash
pip install numpy scipy pillow
```

```python
import sys; sys.path.insert(0, "engines/attractors/src")
from at import core, draw

print("flows:   ", sorted(core.FLOWS))
print("palettes:", sorted(draw.PALETTES))

s = core.Swarm("lorenz", n=42000, seed=1)
s.step(1 / 60.)
print(f"{s.P.shape[0]} particles, mean speed {s.speed().mean():.1f}")
```

Six flows ship: lorenz, rossler, aizawa, halvorsen, thomas, chen. Adding one is a vectorised
`f(P) -> dP/dt` plus a row in a dict, and nothing else changes.

Every palette is built monotone in luminance, because colour here always encodes a real quantity
(speed, drift, distance from the manifold, age). You should be able to read "more" from "brighter"
without a legend.

### Two things that are easy to get wrong

**Seeding.** Particles started in a random ball are all off the attractor, and their approach
transients are long, bright, and identical-looking. Your first second becomes a collapsing shell
that has nothing to do with the dynamics. Seed along a reference orbit with a little scatter
instead.

**Recycling.** Chaotic flows contract volume, so an un-recycled swarm collapses onto a thin
filament within seconds and you lose every part of the attractor the trajectory is not on right
now. Respawn a few percent per second and the whole measure stays lit.

There is a nice trick in here for morphing between attractors: interpolate the **derivative**, not
the picture. `dP/dt = (1-u)·f_A(P) + u·f_B(P)`. Because you are blending the vector fields, every
intermediate is a real dynamical system with its own trajectories, so the swarm is never teleported
and never cross-faded. You get to watch a Lorenz butterfly stretch into a Halvorsen knot through a
sequence of attractors nobody has ever named.

---

## Oscilloscope

[Catalogue](engines/oscilloscope/CATALOG.md) · [Stills](engines/oscilloscope/frames/) ·
[Source](engines/oscilloscope/src/) · [Full notes](engines/oscilloscope/README.md)

A simulated CRT screen. The electron beam is stroked into an intensity layer each frame, added to a
float buffer that decays a little every frame, exactly like real P31 green phosphor. That decay is
what produces the orbiting ghost trails. Faster beam travel means a dimmer trace, so Lissajous
corners glow the way they do on actual hardware.

Three layers you can use independently.

**The screen** is the persistence buffer and the graticule. It is RGB rather than grayscale so
traces can be any colour, and the graticule crossfades between cartesian, radial, and a faint axis
cross.

**The film filter** models filming that screen with a camera, in three stages, in the order light
actually goes through the system. Screen: glass halation, phosphor grain in lit areas only, dust
and smudges that light up under the trace. Lens: barrel distortion, per-channel chromatic
aberration, corner defocus, vignette, sub-pixel hand-shake. Camera: exposure flicker, a drifting
hum band, highlight blowout, tinted black lift, sensor noise.

**The glitch layer** models a bad acquisition, split so signal faults land before the film pass
(and get filmed) while sensor defects land after it.

### The filter works on anything

This is the most immediately useful thing in the repo. It needs nothing else from here:

```bash
pip install -r engines/oscilloscope/src/requirements.txt

python engines/oscilloscope/src/filter_cli.py in.mp4 out.mp4 --glitch 0.8
```

Point it at any video. It probes size and frame rate, streams decode to process to encode, and
copies the source audio window across. About 0.8 is subtle, 1.8 is heavy. The
[`test_filmtest_*` stills](engines/oscilloscope/frames/) are a before/after set at three strengths
if you want to see the range before committing.

### Using the screen directly

```python
import sys; sys.path.insert(0, "engines/oscilloscope/src")
import numpy as np
from osc import scope, config, crtfilm

cfg = config.RenderConfig(width=1080, height=1920)
sc  = scope.Scope(cfg)

sc.new_frame()                                    # decay the phosphor
t = np.linspace(0, 2 * np.pi, 900)
sc.beam(np.stack([np.sin(3 * t), np.cos(2 * t)], 1) * 0.8, gain=0.5)
rgb = sc.render(grid_state=2)                     # 2 = faint axis cross, -1 = none

look = crtfilm.FilmLook(1080, 1920, 60, 900, seed=7,
                        **crtfilm.film_config_for_phosphor((0, 255, 80)))
frame = look.process(rgb, 0)
```

`film_config_for_phosphor()` derives the filter's colour behaviour from your trace colour. Blowout
keys off the lit channels and bleeds into the deficient ones, so any hue blows to white and the
blacks tint toward it. It is cheap enough to call every frame, which means a colour-changing scene
needs no filter tuning at all.

### Watch out for

**Filter exposure defaults to 1.75**, which is right for a thin bright trace on black. A
full-frame bright image needs about 0.72 or the whole thing goes pastel and your blacks turn grey.
Full-colour content also wants a neutral white phosphor config, not the green defaults.

**Flashed overlays must use the transient buffer.** Persistence burns a static bright shape in for
most of a second of decay. There is a separate `beam_transient()` for anything that appears and
vanishes.

**60 fps traces read fainter than a 30 fps preview**, because a shorter arc is drawn per frame so
less light accumulates per lap. Judge brightness at the frame rate you are shipping.

There is also a video-to-ASCII front end in here, which maps luminance to a glyph while keeping
the source pixel colour, and feeds the result through the same film filter.

---

## Chemical Scope

[Catalogue](engines/chemical-scope/CATALOG.md) · [Stills](engines/chemical-scope/frames/) ·
[Source](engines/chemical-scope/src/) · [Full notes](engines/chemical-scope/README.md)

Twenty-nine molecules, each with an FTIR and a Raman spectrum, drawn on that CRT. Molecule above,
spectrum below. This is the chemical profile format remade after I dropped manim.

The two methods are genuinely different, which took real work. Same band positions, reweighted by
selection rules: IR intensity goes with change in dipole moment, so carbonyls and C-O are strong,
while Raman goes with change in polarisability, so C=C and ring breathing are strong. The Raman
legend is the same diagnostic modes reordered by Raman strength, so the handover visibly re-ranks
what matters instead of just redrawing.

### Getting started

```bash
pip install -r engines/chemical-scope/src/requirements.txt
```

```python
import sys; sys.path.insert(0, "engines/chemical-scope/src")
from chemical_data import MOLECULES

print(len(MOLECULES), "molecules")
m = MOLECULES["Morphine"]
print(f"{len(m['AT'])} atoms, {len(m['BONDS'])} bonds, {len(m['ir_lines'])} IR lines")
```

```bash
cd engines/chemical-scope/src
python chemical_profile.py Morphine --preview     # one molecule, fast
python render_optimal.py                          # the whole set, three GPU lanes
```

`render_optimal.py` splits every molecule into frame chunks and work-steals them across three GPU
lanes, so idle lanes pick up the next chunk and the tail stays balanced. Concatenation comes out
seamless because each chunk warms the phosphor first and the film filter is indexed on absolute
frame number with a fixed seed.

### Adding a molecule

One entry in the `MOLECULES` dict. Geometry can be typed by hand or generated from SMILES; either
way you hand-author the IR lines and diagnostic bands from the functional groups.

Nine of the twenty-nine are 3-D and rotate to show depth. If one of yours moves badly, the problem
is almost always orientation rather than motion: something with a small ring or a non-flat face
swings edge-on and the labels collide. Orient by the whole-molecule principal plane instead of the
aromatic ring and it settles down. That fix took one molecule's depth spread from 6.2 to 3.1.

Validate the formula. The generator checks each molecule's computed molecular formula against a
known value, which is cheap and catches a mistyped bond immediately.

### About the spectra

Illustrative. Group-contribution positions with a heuristic Raman reweight. Positions are sound,
relative intensities are approximate, and only about 600 to 1900 cm⁻¹ is shown. Do not cite them.
For the same chemistry done properly, [Academia](#academia) has the real parsers and the DFT paths.

---

## Academia

[Catalogue](engines/academia/CATALOG.md) · [Figure previews](engines/academia/figures/) ·
[Source](engines/academia/src/) · [Full notes](engines/academia/README.md)

The figure code behind the PDF deliverables. Give it a compound class as a set of SMILES and it
produces a publication figure: every molecule in the class stacked as a simulated spectrum, with
its structure drawn beside it, as a vector PDF.

```
SMILES  ->  RDKit structure + functional groups
        ->  group-contribution line list
        ->  broadened to a curve, per technique
        ->  stacked with structures inset  ->  vector PDF
```

The estimator is what makes this feasible at all. A DFT job per molecule per technique would be
weeks of compute for one figure. A group-contribution prediction takes milliseconds and gets the
band positions right, which is what a class-comparison figure is actually asking about.

### Getting started

```bash
pip install -r engines/academia/src/requirements.txt   # RDKit, matplotlib, numpy<2, scipy

cd engines/academia/src
python amino_acid_sim_test.py
```

```
4 stacked PDFs in .../figures/amino_acids
```

Seven scripts produce 84 PDFs across six compound classes plus the BCS drug set. Read
`amino_acid_sim_test.py` first, it is the reference implementation and it is under 170 lines. Copy
`extra_classes_sim_test.py` if you are adding a class of your own.

Every class has its own natural pairing, and choosing it is the interesting design decision:

| class | paired as |
|---|---|
| amino acids, lipids, neurotransmitters, peptides, steroids | crystalline against amorphous |
| cannabinoids | neutral against acid |
| lipids | free acid against salt |
| neurotransmitters | freebase against protonated |
| peptides | reduced against oxidised |
| steroids | free against ester |

### Watch out for

**Axis direction.** FTIR runs 4000 to 400 cm⁻¹ and NMR runs high to low ppm. Raman, UV and XRD all
run low to high. It is a single boolean argument and getting it wrong gives you a figure that is
subtly, embarrassingly backwards.

**Line shape is physics, not preference.** Vibrational bands are Lorentzian. Electronic bands are
Gaussian *in energy*, which is why UV broadening happens in eV and gets mapped back to nm rather
than broadened in wavelength. Powder reflections are pseudo-Voigt.

**DFT harmonic frequencies run high** and need a scaling factor: about 0.967 for
B3LYP/6-311++G(2d,3p), about 0.95 for wB97XD/6-31G*.

Measured data is not part of this repo. `registry.py` fetches from wherever `SPECTRA_DATA_URL`
points, and the simulated figure scripts need no network at all.

### The manim subproject

[`engines/academia/manim/`](engines/academia/manim/) holds the older animated work: crystal
structures, morph sequences, the isoxazole series, and the render drivers. It is where the chemical
profile format came from originally. The library itself lives in
[`chemical_profiles/`](chemical_profiles/) rather than here, since that copy ships with a worked
scene and a guide.

---

# Things that apply everywhere

**Previews are trustworthy.** Every engine drives its frames from real seconds (`t = i/fps`), so a
half-resolution preview at 30 fps and a full-resolution final at 60 fps are the same animation,
sampled differently. Iterate on previews.

**Encodes are atomic.** Everything writes to `<name>.part.mp4` and renames only on success, so a
file bearing its final name is always finished and playable. A stray `.part.mp4` is safe to delete
on sight.

**GPU encoding is on by default** behind a one-time probe with an automatic libx264 fallback, so a
missing or busy encoder slows a render instead of killing it. Each engine has an environment
variable to force CPU if you want it.

**Three concurrent renders is the ceiling** on consumer hardware. GeForce NVENC only allows three
to five simultaneous sessions, and going wider gets you nothing.

**Orientation is your choice.** All of this was written for vertical 1080×1920 because that is what
I publish, but none of it is limited to that. The vertical framing lives in a `RenderConfig` and in
the geometry a composition picks, never in a solver or a field or a physics module. Wind Tunnel
solves in wind coordinates and only the renderer decides which axis is the long one, so
`--flow right` at a wide config gives you the classic landscape tunnel view. Field Lines,
Attractors and Lattice Grid compute in normalised coordinates and do not know what an aspect ratio
is. Shape Physics works in reference pixels with a scale factor. The two CRT engines size the
screen face as a fraction of the frame. Academia is matplotlib, so page size is a figure argument.

Every number measured in this repo came off one machine, an RTX 2060 with an i7-9700K. Treat the
timings as ratios rather than promises.

## Maintaining the catalogues

`engines/<name>/catalog.json` is the source of truth. The markdown beside it and the root index are
both generated:

```bash
python tools/build_catalogs.py
```

Edit the JSON, never the generated `CATALOG.md`.

---

## Legal

Privacy Policy and Terms of Service for the publishing automation, served via GitHub Pages and
referenced by the social-platform developer apps:

- <https://ec175.github.io/spectrometry_public/privacy.html>
- <https://ec175.github.io/spectrometry_public/terms.html>

## License and use

Personal project, shared so the method is reproducible. The rendered videos are not part of this
repository. If you build on it, a credit is appreciated.
