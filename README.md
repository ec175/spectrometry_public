# spectrometry_public

Eight rendering engines built to make science visuals without a scene graph. Seven of them compute
something real — a fluid, a field, a swarm, a physics world — into numpy buffers, draw it
immediate-mode, and pipe raw frames to ffmpeg. The eighth produces the publication figures the
whole thing grew out of.

This repository publishes the **parts**, not the finished videos: the solvers, the shape
generators, the field kernels, the physics primitives, the colour maps, the post filters, the
spectroscopy core, and the render pipeline that drives them. Alongside each engine is a catalogue
of everything it can make and a set of reference frames showing what each of those things looks
like.

**[→ Full catalogue index](CATALOG.md)** — 193 objects and 83 compositions across eight engines.

| engine | what it makes |
|---|---|
| [Wind Tunnel](engines/wind-tunnel/) | A D2Q9 lattice-Boltzmann solver and a compressible Navier-Stokes solver, both driving a colour-field animation. Free bodies the flow genuinely moves. |
| [Field Lines](engines/field-lines/) | Field lines and streamlines of a closed-form vector field, integrated at constant arclength. Electrostatics, potential flow and magnetostatics from one integrator. |
| [Lattice Grid](engines/lattice-grid/) | An illuminated lattice of nodes and connectors whose content re-rolls on the beat, with a colour model built from RGB channel delay rather than a palette. |
| [Shape Physics](engines/shape-physics/) | Discs, spinning gapped shells and destructible structures — plus audio the simulation *triggers*, which is the inverse of how the rest of this works. |
| [Attractors](engines/attractors/) | Tens of thousands of particles integrated through a strange-attractor flow at once, so the image is the system's invariant measure rather than one orbit. |
| [Oscilloscope](engines/oscilloscope/) | A simulated CRT: a phosphor persistence buffer, a three-stage filmed-off-a-real-screen filter, an acquisition-fault layer, and a video→ASCII front end. |
| [Drug Scope](engines/drug-scope/) | Twenty-nine molecules, each with an FTIR and a Raman spectrum, drawn on that CRT. |
| [Academia](engines/academia/) | The figure code behind the PDF deliverables — stacked simulated spectra for whole compound classes, with structures drawn alongside. Carries the manim work as a subproject. |

---

## Orientation is yours to choose

Everything here was authored for **vertical 1080×1920**, because that is what it was built to
publish. **None of it is limited to that**, and the vertical framing is not baked into any solver,
field or physics module — it lives entirely in a `RenderConfig` and in the geometry a composition
chooses.

- **Wind Tunnel** always solves in wind coordinates, and only the renderer decides which lattice
  axis becomes the screen's long one. `--flow right` at a wide `RenderConfig` is the classic
  landscape tunnel view; `up` and `down` are the vertical ones.
- **Field Lines**, **Attractors** and **Lattice Grid** compute in normalised or world coordinates
  and know nothing about aspect ratio. Change the frame size and the seed geometry.
- **Shape Physics** works in reference pixels with a scale factor to output pixels — a landscape
  world is different centre coordinates and different structures, not different code.
- **Oscilloscope** and **Drug Scope** size the screen face as a *fraction* of the frame, with a
  deliberately anisotropic deflection map to fill a tall one. Set those fractions for a wide one.
- **Academia** is matplotlib throughout, so page size and aspect are figure arguments, and the
  stacked layout computes its spacing from the trace count rather than from constants.

Composition is the one place that choice becomes concrete, and composition is exactly what this
repository leaves to you.

## What is here and what is not

**Here.** Every engine's primitive layer and its full render pipeline — the solvers and
integrators, the geometry and structure builders, the fields, the drawing and bloom, the colour
maps and post filters, the ffmpeg piping with atomic output, the NVENC probe with automatic CPU
fallback, and the CuPy backends where they exist. Also the validation tools that do not depend on
a specific composition, including the compressible solver's Sod shock-tube and oblique-shock
checks.

Academia is the exception that proves the rule: there the figure scripts **are** the deliverable,
so all seven of them ship along with the spectroscopy core they sit on.

**Not here.** The `scenes.py` / `signals.py` modules — the compositions themselves. That is
deliberate: the aim is to hand over a toolbox rather than a way to re-emit somebody else's back
catalogue. Each engine's `CATALOG.md` still lists every composition, what it demonstrates, and
links to frames from it, because *what each one proves* is reusable even when the code is not.

Also not here: audio. No music is shipped. Where an engine reads a song it takes a path, and the
engine READMEs say where to put your own. Nor the rendered videos or the 63 MB of source PDFs —
those are represented by 336 reference frames instead.

A few check-tools import the withheld modules and so could not ship either. Where that happened,
the measurement they made and the bug they existed to catch are written up in the engine's README
instead — that is the part worth carrying over.

## Running any of it

Each engine is standalone. There is no shared package and no install step.

```bash
python -m venv .venv && .venv/bin/pip install -r engines/wind-tunnel/src/requirements.txt
```

Common ground across all seven:

- **numpy 2.x + Pillow** is the floor. Some engines add scipy; Wind Tunnel optionally adds CuPy.
- **ffmpeg** on `PATH`, or an engine-specific environment variable, or `bin/ffmpeg.exe` beside
  the source.
- **NVENC is on by default**, behind a one-time probe with an automatic libx264 fallback, so a
  missing or busy encoder slows a render rather than killing it. Each engine has an environment
  variable to force CPU encoding.
- **Every encode is atomic** — written to `<name>.part.mp4` and renamed only on success. A file
  bearing its final name is always finalized and playable; a stray `.part.mp4` is safe to delete.
- **Frames are driven by real seconds** (`t = i/fps`), so a half-resolution preview at 30 fps and
  a full-resolution final at 60 fps are the same animation, sampled differently. Previews are
  trustworthy — iterate on them.
- **Three concurrent renders is the ceiling** on consumer hardware: GeForce NVENC allows only
  three to five simultaneous sessions.

Anything measured in this repository was measured on one machine — an RTX 2060 with an i7-9700K.
The timings are there for ratios, not as promises.

## Also in this repository

### Pipeline — [`pipeline/`](pipeline/)

A sanitized copy of the automation stack that publishes the short-form videos: self-hosted
[n8n](https://n8n.io) with ffmpeg baked in, a small stdlib HTTP service bridging the container to
host-side rendering, and a generator that emits the batch workflow as importable JSON.

**[pipeline/USAGE.md](pipeline/USAGE.md)** is the operator reference — workflows, the form, every
platform, music and Drive integrations, limits, troubleshooting.

Nothing secret lives here. Copy `pipeline/.env.example` to `.env` and fill in your own values;
credential IDs and the render-service token are read from environment variables.

```bash
cp pipeline/.env.example pipeline/.env   # then edit
cd pipeline && docker compose up -d
```

### Legal pages

Privacy Policy and Terms of Service for the automation app, served on GitHub Pages. These are the
URLs the social-platform developer apps reference, so they stay where they are:

- Privacy: <https://ec175.github.io/spectrometry_public/privacy.html>
- Terms: <https://ec175.github.io/spectrometry_public/terms.html>

---

## Maintaining the catalogues

`engines/<name>/catalog.json` is the source of truth. The Markdown beside it and the root index
are both generated:

```bash
python tools/build_catalogs.py
```

Edit the JSON, never the generated `CATALOG.md`.

---

*Personal project. The published videos are not part of this repository.*
