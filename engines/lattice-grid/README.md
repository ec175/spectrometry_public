# Lattice Grid

**A fixed lattice of nodes and connectors, lit by a moving field.** The lattice is generated once
and never moves; what changes is the content sitting on it, which re-rolls on the musical beat,
and the illumination sweeping across it. Everything composites additively and blooms, so dense
clusters run to white on their own.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Reference frames](frames/)** · 💾 **[Source](src/)**

---

## The format in one paragraph

Each site carries a glyph — empty, dot, ring or pad — and each adjacency may be lit as a
connector; boxes are closed rings of connectors around one cell. A smooth illumination field gates
which elements are visible and how hot they are. **The swappable part is the lattice**: the square
and honeycomb compositions differ by one class attribute.

Adding a geometry means a `Lattice` subclass providing sites, edges, edge kinds, link
specifications and a box sampler. Nothing downstream changes.

## The colour is an RGB temporal delay, not a palette

This is the single most important thing in the engine, and it took three sessions and a lot of
discarded machinery to arrive at.

The render is **black and white**. All the colour comes from delaying the channels against each
other — blue undelayed, green about 0.1 s behind, red about 0.2 s. Cross-correlating reference
footage as time series confirms it exactly: peaks at +6, +3 and +3 frames rather than at lag zero.

That one fact dissolves an enormous amount of apparatus:

- **A bimodal warm/cool palette is an OUTPUT of the delay.** An element appearing shows blue first
  and reads cyan; one fading still has its red after the newer channels have gone and reads
  orange; anything steady has all three channels equal and is white. Two clusters plus white —
  exactly the k-means result a hand-built LUT was constructed to reproduce.
- **Hue is rate-of-change**, so its spatial statistics follow the illumination field's for free.

Everything about the look then reduces to one question: **how much does the mono signal change
across the delay window, at the pixels that are bright?** That number *is* the saturation.

| lever | why |
|---|---|
| `twinkle` (high) | the dominant one — it is what makes bright pixels change inside the window |
| `twinkle_rate` (capped ~5 Hz) | faster and an element completes a whole cycle inside the window, so the channels sample unrelated phases and you get magenta/green instead of the blue-cyan-white-yellow-red progression |
| `exposure` LOW, density HIGH | a clipped pixel is white by definition. Brightness must come from coverage |
| `hot_gain` low | same reason — a blown core is white |
| `ambient` low | a high floor plus strong twinkle makes everything busy everywhere and the frame collapses into confetti with no composition |
| `xfade` snappy | a long crossfade moves all three channels together and stays white |

**Bloom is the enemy.** A halo averages many elements, so it changes slowly even when the ink
under it is switching hard — and slow-changing means white.

The general lesson, which cost the most: **modelling an artefact as a cause.** The measurement
that would have caught it is about twenty lines — cross-correlate the R, G and B channels of the
reference as time series and look for a peak away from lag zero. It was never run because the
colour was assumed to be per-element from the start. **On any reference footage, check for
channel-level temporal structure before modelling colour spatially.**

## The glitch is a LAYER, and the network is the subject

Two structural corrections worth carrying over.

**A glitch is ONE small block whose position jumps on the delay interval.** Keep its size and
content fixed and re-roll only its *position* every green-delay seconds, and the channel delay
produces a synced triple for free: blue where it is now, green where it was 0.1 s ago, red where
it was 0.2 s ago. Nothing draws three copies — there is only ever one block per event per frame.
Two constraints follow: the hold must **equal** the green delay, or the channels sample the same
position or skip positions and the triple collapses; and the jump must be **small relative to the
block**, a couple of block-widths, or the three copies land far apart and read as unrelated
rectangles.

**Order of operations is the whole thing:** mono render → glitch → RGB delay → background and
grain. A block comes out pure blue *because* blue is the undelayed channel. Apply the glitch after
the delay instead and it displaces all three channels together, giving grey rectangles.

**The CHAIN is the unit that lights, not the node.** Every node and edge in a chain shares one
schedule, so a circuit energises as a whole. Per-element independent flicker reads as noise
precisely because nothing is connected to anything.

## Quick start

```bash
pip install -r src/requirements.txt   # numpy, Pillow, scipy
```

```python
import sys; sys.path.insert(0, "src")
from lg import lattice, config

for kind in ("square", "honeycomb", "triangular"):
    L = lattice.build(kind, 1080, 1920, config.SOURCE_PITCH)
    print(f"{kind:12s} {len(L.sites):5d} sites  {len(L.edges):6d} edges")
```

```
square       5858 sites   22957 edges
honeycomb    5252 sites    7771 edges
triangular   7140 sites   21063 edges
```

`src/tools/lattice_check.py` verifies the substrate's invariants — coordination number, link-stamp
lengths against real edge lengths, and that a box is a closed ring.

## Every check is necessary and none is sufficient

Four separate check suites exist upstream, and each was added *after* a render passed all the
existing ones and still looked wrong. Two scars:

- One honeycomb composition lit 38% of edges — past percolation. Every cell joined up, the frame
  read as continuous chicken-wire with no distinguishable nodes, traces or empty cells, **and it
  passed all eight bulk-appearance metrics.** Bulk luminance and palette statistics cannot see
  structure. The right invariant across geometries is lit edges **per node**, not lit-edge
  fraction.
- A whole build was **4.4× too still** and passed the same suite perfectly, because every one of
  its metrics is computed on a single frame.

A third suite was added after both of those were green on a render still called "quite different".
It measured how colour and brightness are distributed **at the scale of individual elements** —
hue correlation at 1, 2, 4 and 8 pitches, local contrast, structure-tensor coherence. The render
had every element the same colour as its neighbour where the reference has an isolated green dot
beside an orange ring. **When a render passes everything and still reads wrong, the missing metric
is the deliverable, not more tuning.**

Only `lattice_check` ships, because the other three drive real compositions. Its own
scale-invariance section is removed for the same reason — but that section existed to catch the
worst bug in the project's history, so it is worth reimplementing:

> **The glyph atlas was baked in world pixels and never scaled to the output.** A half-width frame
> drew glyphs at twice their correct relative size, so a preview read a mean luminance of 58 where
> the final delivered 27 — and an entire tuning pass was calibrated against a picture the renderer
> could not produce. The finals were not corrupt; the *measurements* were. `atlas_for(cfg)` bakes
> per output scale, and the check renders the same frame at two resolutions and fails if the bulk
> statistics disagree.

The general lesson: **a measurement harness is code, and it can be the thing that is wrong.** Every
metric agreed with every other metric for a whole session because they all shared one broken
assumption. What caught it was measuring the encoded mp4 — the one artefact that could not share
the harness's bug. **Verify against the encoded file at least once per build.**

## Things that will bite

- **Every multi-scale upsample must CASCADE.** Double and blur repeatedly, never replicate in one
  jump. Expanding a 4-px mip level straight to full size painted 128 px blocks that no subsequent
  blurring removed. And an upsampler that doubles its way up needs its divisor to be a **power of
  two** — a divisor of 6 doubled twice lands at 4×, leaving the source two-thirds size in a
  corner. Blur *after* replicating; blurring first smooths the signal and then replication puts
  the hard steps straight back.
- **A `smoothstep` helper that cannot go downhill.** `clip((x-a)/max(b-a, eps))` is the reflex
  guard and it silently turns every *descending* ramp into a hard step in the wrong direction.
  Guard the magnitude, not the sign.
- **A "depth" parameter that cannot reach zero.** Multiplying by `(1-d) + d·v` leaves a permanent
  floor at any *d* below 1.0. Set to 0.62 for brightness, it meant not one tile ever went dark
  while the parameter looked entirely reasonable. If a control is supposed to reach an extreme,
  verify that it does.
- **"True black" is three settings, not one.** Setting ambient to zero still left the unlit frame
  at mean luminance 6.5 with a per-frame flicker of 6.55. The other two are the grain floor —
  per-frame noise in a dark region *is* low-level blinking — and the background colour itself.
  Check all three when something reads "too bright in the dark parts"; fixing one and re-rendering
  wastes an hour.
- **A default added for one composition silently regressing another.** A wide smooth glow added to
  fix one look was set as a config *default*; on compositions whose whole character is discrete
  elements on black it dropped local contrast from 0.596 to 0.359. Per-composition output settings
  belong on the composition, not in the shared config.
- **Grain and twinkle compete for the same budget.** Both are high-frequency temporal energy.
  Setting one without re-measuring the other moved the high-frequency ratio out of band twice.

## Bitrate — grain is the whole story

Per-frame noise is incompressible, so this format's encode cost is set almost entirely by the
grain setting. First-generation encodes of a 15 s square composition:

| setting | size | bitrate | measured grain in darks |
|---|---|---|---|
| cq 16 / 95M | 188 MB | 100 Mbit/s | 3.40 — overshoots |
| **cq 17 / 70M** | **140 MB** | 75 Mbit/s | **2.84 — on target** |
| cq 19 / 40M | 76 MB | 40 Mbit/s | 1.41 — grain half destroyed |

The usual 40 Mbit ceiling is **not enough here**: at that rate h.264 starts discarding the noise
it cannot compress, and what it leaves behind is correlated blocking rather than independent
grain — visible as a neighbour-correlation metric doubling.

> **Do not measure this with a transcode.** Re-encoding a master at cq 19 reports 2.67 against
> 1.41 for a real cq 19 render of the same frames. The master has already been denoised once, so a
> second-generation encode flatters low bitrates by nearly 2×.

## Performance

At 1080×1920, roughly 600 ms/frame with the full VFX layer — about 9–10 minutes for a 15 s final.
The bottleneck is the scatter-add and the halo/tonemap, both pure array maths and the natural
place for a GPU port, which has not been needed. Three lanes is the NVENC ceiling.

Two optimisations worth not undoing: the obvious spelling of decimation — a strided reduction over
a non-contiguous 5-D view — measured 117 ms at full resolution, four times the cost of the
gaussian it was feeding, where two contiguous slice-adds do the same thing. And a scatter-add via
bin-counting is correct and about 2.7× slower than the direct route, because it allocates a
whole-buffer accumulator per channel per call.

## Orientation

The lattice is generated to whatever frame size and pitch you give it, so a landscape frame is a
different `RenderConfig` and nothing else. **But** the atlas must be baked per output scale — see
the scale-invariance note above. That is the one thing that will silently produce a wrong picture
rather than an error.

## What is not here

- **`lg/scenes.py` and `lg/eyescene.py`** — the compositions. `lg/eye.py`, the eye *geometry*, is
  here: it returns three fields and is a reusable subject rather than a composition.
- **Three of the four check suites**, for the reason above.
- **Audio.** Content already re-rolls on a beat grid, so a real track's beat should drive it
  directly rather than a constant — but nothing is wired up.
