# Attractors — object catalogue

> Tens of thousands of particles integrated through a strange-attractor flow at once — the picture is where they crowd.

RK4 on an (N,3) array costs the same four function evaluations as RK4 on one point, so 40,000 particles are essentially free next to the rasteriser. That is the whole design: instead of tracing ONE orbit and drawing its shape, integrate a swarm and let the image be built by density. The difference is not a restyling — a single orbit shows you the SHAPE of an attractor, a swarm shows you its INVARIANT MEASURE, which parts of that shape the system actually spends its time in. The bright regions are not artistic emphasis; they are where the dynamics dwell.

**Output.** 1080x1920 @60. The projection returns normalised coordinates and a depth; `Canvas` decides the frame. Nothing in `core.py` knows the aspect ratio, so a landscape plate is a different `Canvas` size and nothing else.


## Modules

What each shipped file is. Compositions (`scenes.py` and friends) are deliberately not part of this repository — see the engine README.

| module | kind | what it gives you |
|---|---|---|
| `at/core.py` | dynamics | The six flows, vectorised over an (N,3) array; `rk4`; `Swarm` (seeding, stepping, respawn, speed); `rot` and `project` — 3-D to screen with a mild perspective divide and a returned depth. |
| `at/draw.py` | render | `Canvas` — an additive persistence buffer with per-frame decay, bilinear point splatting, a two-scale glow and an exposure tonemap — plus the five colour ramps. |


## Flows

Each is `f(P) -> dP/dt` vectorised over an (N,3) array, registered in `FLOWS` alongside a suggested timestep, a seed point and a display span. Adding one is a function plus a row.

| object | what it is | notes |
|---|---|---|
| `lorenz(P, s=10.0, r=28.0, b=8/3)` | The butterfly. Two lobes, with a slow crawl through the middle where the trajectory decides which wing to take next. | dt 0.0045, span 26.0. The default for every scene. |
| `rossler(P, a=0.2, b=0.2, c=5.7)` | A single spiral band with a periodic fold out of plane. | dt 0.0180, span 20.0. Slow in x/y and spiky in z, so it needs a larger step to read as a live trace. |
| `aizawa(P, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1)` | A torus-like shell with a spike through its axis. | dt 0.0095, span 1.6. The most three-dimensional of the six — the depth cue earns its keep here. |
| `halvorsen(P, a=1.4)` | Three-fold symmetric, knotted. | dt 0.0058, span 9.0. |
| `thomas(P, b=0.19)` | Cyclically symmetric, built from sines — a lattice of loops rather than lobes. | dt 0.0420, span 4.5. The largest stable step of the six. |
| `chen(P, a=5.0, b=-10.0, d=-0.38)` | A double-scroll relative of Lorenz with a broader sweep. | dt 0.0040, span 32.0. |


## Swarm

| object | what it is | notes |
|---|---|---|
| `Swarm(flow='lorenz', n=42000, seed=0, scatter=0.55, spin_up=4000)` | N particles on one attractor, with a reference orbit computed at construction. | Seeding is the subtle part. Particles started in a random ball are all OFF the attractor and their approach transients are long, bright and identical-looking — the first second becomes a collapsing shell that has nothing to do with the dynamics. So the swarm is seeded ALONG the reference orbit with a little scatter, which puts every particle on the attractor from frame zero. |
| `step(dt, sub=2)` | Advance the whole swarm; stores the velocity used, so `speed()` is free. |  |
| `respawn(frac)` | Recycle a fraction of the swarm back onto the reference orbit each second. | Not cosmetic. Chaotic flows are volume-CONTRACTING, so an un-recycled swarm collapses onto a thin filament within a few seconds and the picture loses every part of the attractor the trajectory is not on right now. Recycling keeps the whole invariant measure lit. |
| `project(Q, c, s, ang, persp=0.55)` | 3-D to screen with a weak perspective divide, returning a DEPTH alongside the 2-D position. | Depth matters more here than it would elsewhere: with 40,000 additive points a purely orthographic view turns a 3-D object into a flat smear. Weak perspective plus depth-keyed size and brightness is enough to separate the front and back lobes. |
| `rk4(f, P, dt, **kw)` | Classical RK4, whole-array. | Same lockstep argument as the field-line integrator — a per-particle loop costs orders of magnitude more for identical results. |


## Colour ramps

Every one maps a MEASURED quantity, so all five are built monotone in luminance — a viewer must be able to read 'more' from 'brighter' without a legend. Colour always encodes something real: speed, drift, capture distance, age. That is the rule that keeps this a visualisation rather than a screensaver, and it is also what makes the pictures legible.

| object | what it is | notes |
|---|---|---|
| **ember** | Deep indigo through violet and red to pale gold. | Paired with speed. |
| **ice** | Near-black blue through cyan to white. | Paired with a single bright orbit and its halo. |
| **flare** | Navy through blue and violet to pink and cream. | Paired with logarithmic separation. |
| **aurora** | Dark teal through green to pale mint. | Paired with time-since-capture. |
| **spectrum** | Purple, blue, teal, gold, coral — the widest hue travel of the five. | Paired with speed across a morph between different systems. |
| `ramp(name, n=256)` | Builds a 256-entry LUT from any of the stop lists. | Add a palette by adding stops; keep it monotone in luminance. |


## Canvas

| object | what it is | notes |
|---|---|---|
| `Canvas(w, h, palette='ember', trail=0.86, sigma=1.05, glow=(0.42, 7.0), glow2=(0.20, 30.0), exposure=1.0)` | Additive persistence buffer with frame-to-frame decay, two-scale glow and an exposure tonemap. | `trail` is the analogue of phosphor decay: higher holds the swarm's recent history longer, which reads as motion blur along the filaments. |


## Compositions

Five compositions, and they are five different MEASUREMENTS rather than five palettes — each one shows something the others cannot. The compositions are not shipped; the list is here because the measurement each one makes is the reusable idea.

| composition | what it does | status | frames |
|---|---|---|---|
| `lorenz_swarm` | The invariant measure, plainly. 46,000 particles coloured by local speed. Nothing is traced — every bright filament is just where particles crowd, so the image IS the measure. Speed colour makes the structure read three-dimensionally with no shading: the outer sweeps of each lobe are fast and hot, the slow crawl through the middle stays cool and dark. | shipped 15 s | [1](frames/lorenz_swarm_t020.jpg) [2](frames/lorenz_swarm_t055.jpg) [3](frames/lorenz_swarm_t085.jpg) |
| `lorenz_ribbon` | One true orbit drawn as a bright ribbon, with a halo of particles it has captured. The halo runs under the SAME flow plus a weak spring toward the nearest point of the orbit — they are not stuck to it, the flow keeps tearing them off and the spring keeps gathering them back, so the halo breathes around the ribbon instead of coating it. | built, not rendered | — |
| `lorenz_divergence` | Sensitive dependence. Two swarms launched from identical positions, same solver, differing by 1e-7. Colour is the distance between each particle and its twin, on a LOG scale — separation covers seven orders of magnitude in 15 s, and on a linear scale the whole clip is black and then white in one frame. It lights up first along the stretching directions, which is where the Lyapunov exponent lives. | shipped 15 s | [1](frames/lorenz_divergence_t020.jpg) [2](frames/lorenz_divergence_t055.jpg) [3](frames/lorenz_divergence_t085.jpg) |
| `lorenz_capture` | An attractor attracting. Every other scene starts on the manifold; this one deliberately does not. 40,000 particles are seeded in a big uniform box and the flow does the rest — a structureless cloud collapsing onto a two-dimensional sheet in about four seconds, because the flow contracts volume everywhere. Colour is time-since-capture, so the sheet forms cool while stragglers stay hot. | shipped 15 s | [1](frames/lorenz_capture_t020.jpg) [2](frames/lorenz_capture_t055.jpg) [3](frames/lorenz_capture_t085.jpg) |
| `attractor_tour` | Four attractors as ONE object, by interpolating the DERIVATIVE rather than the picture: dP/dt = (1-u)*f_A(P) + u*f_B(P). Because the blend is of the vector fields, every intermediate is a real dynamical system with its own trajectories — the swarm is never teleported and never cross-faded. Watching a Lorenz butterfly stretch into a Halvorsen knot through a sequence of attractors nobody has named is the whole reason to do it this way. | shipped 15 s | [1](frames/attractor_tour_t008.jpg) [2](frames/attractor_tour_t030.jpg) [3](frames/attractor_tour_t052.jpg) [4](frames/attractor_tour_t074.jpg) [5](frames/attractor_tour_t094.jpg) |


## Parameters that matter

Short list — the format has few knobs and most of them trade the same thing against itself.

| parameter | lives in | what it controls | usable range |
|---|---|---|---|
| `n` | Swarm | Particle count. Sets how completely the invariant measure is filled in. | 26,000–46,000. Cost is in the rasteriser, not the integrator. |
| `recycle` | Scene | Fraction of the swarm respawned per second onto the reference orbit. | 0.05–0.10. Zero collapses the picture onto a filament within seconds. |
| `trail` | Canvas | Frame-to-frame persistence of the accumulation buffer. | 0.80–0.90 |
| `exposure / point_gain` | Canvas | Tonemap gain and per-point brightness. | Lower both where the swarm CONCENTRATES. A collapse onto a 2-D sheet saturates the additive buffer and the core blows to flat white, losing exactly the structure the collapse was meant to show — 0.46 exposure with 0.72 point gain keeps the roll-off working. |
| `persp` | Scene | Perspective strength in the projection. | 0.55. Zero flattens a 3-D object into a smear at these point counts. |
| `spin` | Scene | View rotation rates about the three axes. | Use incommensurate rates. On a morphing scene the spin must be FASTER than the morph, or the result reads as a still object being turned rather than one system becoming another. |
| `sub` | Swarm.step | Integration substeps per frame. | 2 default. Raise for the stiffer flows rather than shrinking dt globally. |


---

*Generated from `catalog.json` by `tools/build_catalogs.py` — edit the JSON, not this file.*
