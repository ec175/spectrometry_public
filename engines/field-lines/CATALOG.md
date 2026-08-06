# Field Lines — object catalogue

> Field lines and streamlines of a closed-form vector field, integrated in arclength.

Nothing here solves a PDE. Every field is a closed form evaluated at an array of points, so there is no lattice, no timestep, no CFL condition and nothing can diverge. Lines are traced by RK4 on `dp/ds = E/|E|` — a CONSTANT step in ARCLENGTH, not in time — with every seed advanced in lockstep and dead lines compacted out of the array. Normalising to unit speed costs one divide and buys two things: drawn line density is uniform instead of piling up wherever the field is weak, and a packet advected at speed v becomes the array index `round(v*t/ds)`, which is why the pulse module has no integrator in it at all.

**Output.** 1080x1920 @60. Nothing in the integrator or the field kernels knows about aspect ratio — only `draw.Frame` and `RenderConfig` do. Change the resolution and the seed geometry and the same code draws a landscape plate.


## Modules

What each shipped file is. Compositions (`scenes.py` and friends) are deliberately not part of this repository — see the engine README.

| module | kind | what it gives you |
|---|---|---|
| `fl/field.py` | fields | The vector fields. `coulomb` with pluggable kernel and Plummer softening, `ChargeSet` (a mutable bag of sources), `PotentialFlow` (uniform stream + doublets + vortices + sources, with exact Milne-Thomson imaging), `wires` for magnetostatics. |
| `fl/lines.py` | integrator | `trace` — RK4 in arclength, all seeds in lockstep, compacting as lines die, returning a fate code per line. Plus the seed generators and the bit-reversed fade order. |
| `fl/pulses.py` | animation | `PulseTrain` — packets that ride the lines. 103 lines with no integration in it, because a packet is just an index into the already-traced polyline. |
| `fl/motion.py` | animation | How sources move. `WanderPath` (declared, speed-matched by measurement), `NBody` (solved — mutual Coulomb in an anisotropic well with a soft core), `keyframes`, `pulse_gate`, `smoothstep`. |
| `fl/draw.py` | render | `Frame` — supersampled Pillow strokes, box downsample, a bilinear density splat as a separate additive channel, then two-scale bloom. Palettes `palette_alternate`, `palette_angle`, `palette_fate`. |
| `fl/render.py` | render | `solve` / `compose` / `flux_report` / `render_stills` / `render_scene`. ffmpeg pipe with atomic output. |
| `fl/config.py` | config | `RenderConfig` (resolution, fps, supersample, bloom, density), the NVENC probe, ffmpeg resolution. |


## Field kernels

A field is any callable `f(P) -> V` on an (N,2) array. That is the entire contract — swap the callable and not one line of the integrator, the pulses or the drawing changes.

| object | what it is | notes |
|---|---|---|
| `coulomb(P, pos, q, soft=9.0, kernel='inv_sq')` | Point sources with `inv_sq` or `inv_r` falloff, Plummer-softened. | `soft` is not cosmetic: without it an RK4 step that lands near a charge takes a NaN. `inv_r` changes magnitude falloff, not line shape. |
| `ChargeSet(soft=9.0, kernel='inv_sq').set(pos, q)` | A mutable bag of point sources. Rebuild `pos`/`q` every frame to move charges, change their magnitude, or flip their SIGN. | `sinks(sign=-1)` returns the positions a line may legitimately terminate on, excluding any source whose magnitude has decayed to nothing — otherwise lines snap onto an invisible marker. |
| `PotentialFlow(); pf.cylinder(center, a, gamma=0.0)` | Uniform stream plus doublets, point vortices and sources. `cylinder()` places an exactly-imaged body with optional bound circulation. | One doublet per body is exact only for an ISOLATED body — two cylinders each violate the other's boundary condition (measured max\|v.n\| = 0.81\|U\|). The fix is ONE cylinder plus Milne-Thomson images of every external singularity, which is exact for arbitrary external flow: 0.81 -> 0.0005. The stream direction derives the doublet axis; it is never a free parameter. |
| `wires(P, pos, I, soft=9.0)` | Magnetostatic field of line currents. | Written and never used in a composition. It is the cheapest route to closed field lines that never terminate — the one topology electrostatics cannot make — and `END_LOOP` already reports it. |


## The integrator

| object | what it is | notes |
|---|---|---|
| `trace(fieldfn, seeds, ds=6.0, n_steps=440, sinks=None, capture_r=16.0, bounds=..., sgn=+1)` | Returns `pts (N,S+1,2)`, `length (N,)` and `ended (N,)`. RK4 in arclength, lockstep over the whole seed array. | `capture_r` must always be >= `ds`, or a line steps straight over its own sink. A per-line Python loop turns ~450 vector ops into ~90k scalar ones and costs about 50x. |
| `END_SINK, END_NULL, END_BOUNDS, END_BUDGET, END_LOOP = 0..4` | How each line ended: captured by a negative source, stalled at a saddle, reached the frame margin, ran out of arclength, or closed/spiralled. | `END_BUDGET` is a BUG, not a tuning target. It means `capture_r < ds` or the bounds are wrong. Do not raise `n_steps` to make it go away. |
| `seed_ring(center, r0, n_max, phase=0.0)` | Seeds on a circle — the radial fan. |  |
| `seed_rake(p0, p1, n_max, jitter=0.0, seed=0)` | Seeds on a line — the inlet rake for a stream. | A rake has to be WIDER than the frame: at a modest stream tilt a 2000 px descent drifts a streamline ~320 px sideways, so a rake that only just covers the frame leaves the downstream third black. |
| `fade_order(n_max)` | Bit-reversed lighting order, so the first k seeds for ANY k are quasi-uniformly spread. | This is what lets line count be an animation channel. Re-spacing seeds on every integer crossing of an animated N(t) makes the whole fan snap — the single most likely visual bug in this format. |


## Animation

| object | what it is | notes |
|---|---|---|
| `PulseTrain(period, speed, dash, stagger=0.0)` | Beat-locked packets riding the lines, as an index range into the traced polyline. | `stagger=0` for a radial fan — synchronous packets at equal arclength form the expanding ring that makes the format work. `stagger=1.0` for a parallel bundle, where the same packets stay collinear and sweep as a solid bar that reads as a scan artefact. |
| `WanderPath(...)` | A declared path for a source — the rig holding a charge somewhere. | Speed-matched by measurement, not by taste. |
| `NBody(..., well=..., core_k=..., core_soft=...)` | Sources solved under mutual Coulomb in an anisotropic well with a soft core. Nothing keyframed. | The core must be short-range against a 1/r^2 attraction: 1/r^3 is still comparable to Coulomb at 200 px. Use 1/r^5 and size it by the ENERGY it must absorb (U(0) = core_k/(4*core_soft^4) > v^2/2), not by a force at some distance. Substeps at a fixed rate independent of fps, so a 30 fps preview and a 60 fps final are the same animation. |
| `keyframes(t, ts, vs)` | Piecewise interpolation for any scalar channel. |  |
| `pulse_gate(t, windows, ramp=1.0, lo=1.0, hi=5.0)` | Windowed gate for turning a channel on and off. |  |


## Palettes

| object | what it is | notes |
|---|---|---|
| **palette_alternate** | Adjacent seeds alternate between two blues. | At high line counts these beat into a radial MOIRE. That is emergent, it is the cheapest visual interest in the format, and it is deliberate. |
| **palette_angle** | Colour by seed angle. |  |
| **palette_fate** | Colour by how the line ENDED — captured versus escaped. | The colour change IS the physics. Currently used by one composition; it suits any scene with capture events at least as well. |


## Compositions

Five, and they are a deliberate spread over the properties the format can carry — one property each, no reskins. The compositions are not shipped; the list is here because what each one PROVES is reusable even when the code is not.

| composition | what it does | status | frames |
|---|---|---|---|
| `dipole_flower` | Seed count as an animation channel, and capture as a topology EVENT. A mobile negative source toggles magnitude; above a threshold its capacity exceeds the centre's emission and it takes the entire flux — the picture snaps from a five-pole flower to a teardrop in about a second. | shipped 15 s | [1](frames/dipole_flower_t015.jpg) [2](frames/dipole_flower_t045.jpg) [3](frames/dipole_flower_t080.jpg) |
| `rosette_waltz` | Continuous reconnection under MOVING sinks. Bundles are handed between sinks through null points; the rosette is never drawn, only the sinks move. | shipped 15 s | [1](frames/rosette_waltz_t015.jpg) [2](frames/rosette_waltz_t045.jpg) [3](frames/rosette_waltz_t080.jpg) |
| `pair_dance` | Nothing keyframed — the SOURCES are solved. Six charges under mutual Coulomb in an anisotropic well, launched tangentially, with lines coloured by fate. | shipped 15 s | [1](frames/pair_dance_t015.jpg) [2](frames/pair_dance_t045.jpg) [3](frames/pair_dance_t080.jpg) |
| `charge_wave` | MAGNITUDE and SIGN as animation channels. Seven sources in a swaying column with a travelling-wave charge; lines are born and die as sources cross zero, and marker radius reads the state out honestly. | shipped 15 s | [1](frames/charge_wave_t015.jpg) [2](frames/charge_wave_t045.jpg) [3](frames/charge_wave_t080.jpg) |
| `stream_glass` | KERNEL GENERALITY — the reuse proof. Potential flow: a tilting stream, one exactly-imaged cylinder with swinging bound circulation, two free vortices orbiting it. Not one line of the integrator, the pulses or the drawing changes; only the callable. | shipped 15 s | [1](frames/stream_glass_t015.jpg) [2](frames/stream_glass_t045.jpg) [3](frames/stream_glass_t080.jpg) |


## Parameters that matter

Density is the content here — the biggest lever by a wide margin is how many lines are lit and how that number moves over the clip.

| parameter | lives in | what it controls | usable range |
|---|---|---|---|
| `n_slots / N(t)` | SeedBank | How many seeds exist, and how many are lit at time t. The whole arc of this format is a build and a drop in line count. | 60–240 slots |
| `ds` | trace | Arclength step. Sets drawn line resolution and the index scale that packets ride. | 4–8 px |
| `capture_r` | trace | How close a line must come to a sink to be captured. | MUST be >= ds, always |
| `s_max / n_steps` | trace | Total arclength budget per line. | Must clear a straight bounds traverse WITH SLACK. 2795 px allowed for a 2130 px straight run left 34% of lines out of budget as soon as they meandered. |
| `glow_mul / glow2_mul` | RenderConfig, per composition | Bloom exposure at two scales. | There is no single right exposure for a radial fan and a parallel bundle. 210 near-parallel lines cover a third of the frame, and a wide halo over that integrates to a flat field with no contrast anywhere. |
| `dens_gain / dens_ref` | RenderConfig | The additive density channel — what blows a converging core to white. |  |
| `line_width` | Style | Stroke width in world px. | Use EVEN world widths. Pillow rounds stroke widths to integers, so 1.7 becomes 2 px at preview and 3 px at final — the final comes out 25% finer and darker than the preview promised. |
| `period` | PulseTrain | Packet cadence. | 2.5 s is one 4/4 bar at 96 BPM |


---

*Generated from `catalog.json` by `tools/build_catalogs.py` — edit the JSON, not this file.*
