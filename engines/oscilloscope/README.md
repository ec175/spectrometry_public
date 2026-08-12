# Oscilloscope

**A simulated CRT screen.** A phosphor persistence buffer, a three-stage model of filming that
screen with a camera, an acquisition-fault layer, and a video→ASCII front end that feeds the same
filter.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Reference frames](frames/)** · 💾 **[Source](src/)**

---

## The core is one idea

The electron beam is stroked each frame into an intensity layer, which is added to a float buffer
that **decays a little every frame** — real P31 green afterglow:

$$P_{n+1} \;=\; \underbrace{\alpha\,P_{n}}_{\text{phosphor decay}} \;+\; I_{n+1},\qquad 0<\alpha<1$$

That one decay factor is what produces the orbiting ghost trails as a figure morphs or precesses,
and it is why the shape can emerge from persistence rather than being drawn. The screen remembers,
and how long it remembers is a single number.

Faster beam travel means a dimmer trace — intensity goes as the reciprocal of the segment length,

$$I \;\propto\; \left\lVert \frac{d\mathbf{r}}{dt} \right\rVert^{-1}$$

so the beam dwells at the turning points and Lissajous corners glow the way they do on real
hardware. Neither of those is a drawing trick; they are the two things a phosphor screen does. The
buffer is then colourised and given an additive gaussian bloom halo over a static graticule.

Two kinds of composition sit on that:

- **Trail** — the physically honest model. One bright beam head, sub-frame stepped so the fast dot
  leaves a *continuous* trail, and the shape exists only as phosphor. The head is drawn into a
  **separate transient buffer**, redrawn every frame, so it never smears into a chain of beads.
- **Curve** — the whole parametric curve lit at once each frame. Cheaper, and right when the
  figure is the subject rather than the beam.

## Three layers you can use independently

**`osc/scope.py`** is the screen. The persistence buffer is RGB rather than grayscale, so traces
can be any colour. The graticule has three states crossfaded by a float — cartesian, radial, and a
faint axis cross — with a sentinel for none at all.

**`osc/crtfilm.py`** is the filter, and it is the most reusable thing here. Three stages, in the
order light actually goes through the system:

| stage | what happens |
|---|---|
| SCREEN (linear light) | multi-scale glass halation, static phosphor grain in lit areas only, dust and smudges that light up under the trace, glass sheen |
| LENS | barrel distortion, transverse chromatic aberration per channel, corner defocus, vignette, sub-pixel hand-shake |
| CAMERA | exposure flicker, a drifting hum band, Reinhard-extended highlight blowout, tinted black lift, fine and coarse-chroma sensor noise |

All statics are seeded and precomputed per resolution, and the filter is **absolute-frame indexed
with a fixed seed**, which is what lets a clip be rendered in chunks and concatenated seamlessly.

`film_config_for_phosphor(color)` derives the filter's hue behaviour from a trace colour: blowout
keys off the **lit** channels and bleeds into the **deficient** ones, so any hue blows to white,
and blacks, dust and sheen tint toward the hue. It is pure and cheap enough to call per frame,
which is what makes a dynamic-colour composition possible with no hand-tuning.

**`osc/glitchfx.py`** is an acquisition-fault layer, split in two on purpose: signal faults apply
*before* the film pass so the camera films them, sensor defects apply *after* it.

### Use the filter on anything

`filter_cli.py` applies both to any existing mp4 and needs nothing else from this repository:

```bash
python src/filter_cli.py in.mp4 out.mp4 --glitch 0.8
```

It probes size and fps, streams decode→process→encode, and copies the source audio window. About
0.8 is subtle and 1.8 is heavy. The [`test_filmtest_*` frames](frames/) are the A/B set if you are
deciding how hard to push it.

## Quick start

```bash
pip install -r src/requirements.txt
```

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from osc import scope, config, crtfilm

cfg = config.RenderConfig(width=1080, height=1920)
sc = scope.Scope(cfg)

sc.new_frame()                                    # decay the phosphor
theta = np.linspace(0, 2*np.pi, 900)
sc.beam(np.stack([np.sin(3*theta), np.cos(2*theta)], 1) * 0.8, gain=0.5)
rgb = sc.render(grid_state=2)                     # 2 = faint axis cross, -1 = none

look = crtfilm.FilmLook(1080, 1920, 60, 900, seed=7,
                        **crtfilm.film_config_for_phosphor((0, 255, 80)))
look.exposure = 1.75                              # thin trace on black
frame = look.process(rgb, 0)
```

## Four things that cost time to learn

**Flashed overlays must use the transient buffer, never `beam()`.** Persistence burns a static
bright shape in for the better part of a second of decay. Anything that appears and vanishes goes
through `beam_transient`.

**60 fps traces read fainter than a 30 fps preview.** A shorter arc is drawn per frame, so less
light accumulates per lap. Judge trace brightness at the frame rate you will ship.

**`exposure` on the filter defaults to 1.75, and that is only right for a thin bright trace on
black.** A full-frame bright image — a colour field, an ASCII frame — needs about **0.72**, or the
whole picture blows out to pastel and black fills go grey. Full-colour content also needs a
**neutral white** phosphor config, not the green defaults, or hues halate wrongly.

**Seamless loops need a warmup pre-roll.** The persistence buffer starts empty, so a loop whose
first frame is not black cannot match its own last frame without pre-rolling the composition's
*tail* into the phosphor before frame 0. The rest of the recipe: express every rate as an
**integer** number of cycles, revolutions or turns over the duration, and have the keyframe order
return to its starting shape. About 1.6 s of warmup at 60 fps gets the seam difference to ~0.2%.
Judge seams only at the final frame rate — a preview seam is always worse.

## Audio

`osc/audio.py` decodes a window, computes per-band FFT magnitudes over **log-spaced** bins (pitch
is logarithmic; linear bands put almost every musical note in the bottom bin), and muxes the same
window back over a silent render.

The entire sync contract is one line: **build `start` must equal mux `-ss`.**

No music ships with this repository. Pass your own path.

## The ASCII front end

`osc/asciivid.py` maps luminance to a glyph, **keeps the source pixel colour**, and drops
near-black cells — assembling a whole frame at once through a precomputed glyph atlas and fancy
indexing rather than a per-cell blit loop, which is what makes it fast enough for video.

To fit a 16:9 source into a 9:16 screen it takes a slice of the source **width** and squeezes it
horizontally: the same pixels at a narrower display aspect, so the slice stands tall and fills the
frame. `osc/track.py` slides that slice to follow the action, and the way it does so is the
interesting part — it asks **which window holds the most energy**, not where the centroid is. The
centroid of a saliency map barely moves (measured 0.491 to 0.504 over a six-second test), because
energy is spread broadly and roughly symmetrically across a frame. Sliding the crop by centroid is
the fixed centre crop with extra steps.

`ascii_video.py` is the whole pipeline: decode, ASCII, film, encode, re-mux.

## GPU

`osc/gpu.py` is a CuPy backend with automatic numpy fallback, backing the scope buffers, bloom,
compose and the entire film filter. Measured about **10× on render** and **4× on the filter**.
Statics are generated from the seed, so the look is identical CPU or GPU — only per-frame sensor
noise differs by RNG.

Any CUDA failure degrades with a loud warning rather than stopping: a broken driver slows a render,
it never kills one. If your system pins `CUDA_PATH` to an older toolkit, the backend scrubs it for
the process only rather than asking you to change a system variable.

## Performance

The per-frame gaussian bloom is the bottleneck at full resolution. Two optimisations worth not
undoing: the bloom is computed at **half resolution** — the glow is low-frequency, so it is
visually identical — and the head bloom is **skipped entirely** when the head buffer is empty,
which is most compositions. Together about 4×.

## Orientation

The screen face is sized as a **fraction** of the frame, and the deflection map is deliberately
anisotropic so vertical deflection fills a tall frame. Set those two fractions and the frame size
for a landscape screen; the mapping follows. The film and glitch filters are resolution-agnostic
and re-seed their statics per size.

## What is not here

- **`osc/signals.py`** — the compositions, about 1,300 lines. [CATALOG.md](CATALOG.md) lists them
  grouped by output family, with frames.
- **`osc/attractors.py`** — the CRT attractor compositions. See the [attractors
  engine](../attractors/) for the swarm treatment of the same systems.
- **`osc/polarline.py` and `osc/modes.py`** — a separate polar-sonification format, outside the
  scope of this catalogue.
- **`build_circle_format.py`** — the builder for the split ASCII-cover-plus-scope format. It
  imports the withheld compositions, so it could not ship; the format itself is described in the
  catalogue and there are four frames from it.
- **The chemical-profile format**, which has its own engine: [chemical-scope](../chemical-scope/).
