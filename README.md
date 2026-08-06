# spectrometry_public

**Drug profiles built in [manim](https://www.manim.community/), and the eight engines behind the
rest of the work.**

A drug profile is a vertical short that shows one molecule and the spectra that identify it — the
structure spins over its ¹³C and ¹H NMR spectra with the peaks numbered to match the atoms,
cross-fades to an FTIR trace, and then plays each infrared vibration mode while a cursor tracks
the band it produces. That is in [`drug_profiles/`](drug_profiles/), and it is the thing to clone
if you want to render something today.

The rest of the repository is the toolbox everything else was built with: eight rendering and
figure engines, published as their **parts** rather than as finished videos, each with a catalogue
of what it can make and reference stills showing what those things look like.

### Where to watch them

- YouTube — [**spectrometry.mp4**](https://www.youtube.com/channel/UChtdNI2BC1SmkmHEERA4dzg)
- X — [**@spectrometrymp4**](https://x.com/spectrometrymp4)

---

## Render a drug profile

```bash
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public/drug_profiles

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

manim -qh -r 1080,1920 drug_profile.py VerticalProfile_Aspirin
```

You need Python 3.10+ and ffmpeg. **You do not need LaTeX** — these scenes use Pango text rather
than `MathTex`, so the usual manim LaTeX setup does not apply.

Output lands in `drug_profiles/renders/videos/drug_profile/1920p30/`.

> ### 📖 [**drug_profiles/GUIDE.md**](drug_profiles/GUIDE.md)
> The detailed walkthrough: installing from scratch, every quality flag and which to actually use,
> how to read the output tree, **how to author a new molecule**, the full data contract attribute
> by attribute, and troubleshooting.

A new molecule is **one subclass** — every molecule-specific value is a class attribute.
`VerticalProfile_Aspirin` is the template; read it top to bottom and you have the entire data
contract.

### A note on the data

The spectra are **representative, not measured**. Peak positions are seeded from literature values
and group-contribution estimates, then hand-corrected so the diagnostic bands are right. They are
good enough to teach with and are not a substitute for a real acquisition. If you author a
molecule, check every shift and every band against a reference before you render — nothing in the
code validates them, and a wrong assignment is a factual error displayed on screen.

---

## The engines

Eight of them. Seven compute something real — a fluid, a field, a swarm, a physics world — into
numpy buffers, draw it immediate-mode, and pipe raw frames to ffmpeg: no scene graph, no timeline,
no manim. The eighth produces the publication figures the whole thing grew out of.

**[→ Full catalogue index](CATALOG.md)** — 193 objects and 83 compositions across eight engines,
with 336 reference stills.

| engine | what it makes |
|---|---|
| [Wind Tunnel](engines/wind-tunnel/) | A D2Q9 lattice-Boltzmann solver and a compressible Navier-Stokes solver, both driving a colour-field animation. Free bodies the flow genuinely moves. |
| [Field Lines](engines/field-lines/) | Field lines and streamlines of a closed-form vector field, integrated at constant arclength. Electrostatics, potential flow and magnetostatics from one integrator. |
| [Lattice Grid](engines/lattice-grid/) | An illuminated lattice of nodes and connectors whose content re-rolls on the beat, with a colour model built from RGB channel delay rather than a palette. |
| [Shape Physics](engines/shape-physics/) | Discs, spinning gapped shells and destructible structures — plus audio the simulation *triggers*, which is the inverse of how the rest of this works. |
| [Attractors](engines/attractors/) | Tens of thousands of particles integrated through a strange-attractor flow at once, so the image is the system's invariant measure rather than one orbit. |
| [Oscilloscope](engines/oscilloscope/) | A simulated CRT: a phosphor persistence buffer, a three-stage filmed-off-a-real-screen filter, an acquisition-fault layer, and a video→ASCII front end. |
| [Drug Scope](engines/drug-scope/) | The drug profile above, remade on that CRT — 29 molecules, each with an FTIR and a Raman spectrum. |
| [Academia](engines/academia/) | The figure code behind the PDF deliverables — stacked simulated spectra for whole compound classes, with structures drawn alongside. |

## Orientation is yours to choose

The engines were authored for **vertical 1080×1920**, because that is what they were built to
publish. **None of them is limited to it.** Vertical framing is not baked into any solver, field
or physics module — it lives entirely in a `RenderConfig` and in the geometry a composition
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
maps and post filters, the spectroscopy core, the ffmpeg piping with atomic output, the NVENC
probe with automatic CPU fallback, and the CuPy backends where they exist. Also the validation
tools that do not depend on a specific composition, including the compressible solver's Sod
shock-tube and oblique-shock checks.

Academia and `drug_profiles/` are the exceptions that prove the rule: there the scripts **are**
the deliverable, so they ship whole.

**Not here.** The `scenes.py` / `signals.py` modules — the compositions themselves. That is
deliberate: the aim is to hand over a toolbox rather than a way to re-emit somebody else's back
catalogue. Each engine's `CATALOG.md` still lists every composition, what it demonstrates, and
links to stills from it, because *what each one proves* is reusable even when the code is not.

Also not here: audio, the rendered videos, and the source PDFs. 336 reference stills stand in for
them. A few check-tools import the withheld modules and so could not ship either; where that
happened, the measurement they made and the bug they existed to catch are written up in the
engine's README instead.

## Running any of it

Each engine is standalone. There is no shared package and no install step.

```bash
python -m venv .venv && .venv/bin/pip install -r engines/wind-tunnel/src/requirements.txt
```

Common ground across all eight:

- **numpy 2.x + Pillow** is the floor. Some engines add scipy; Wind Tunnel optionally adds CuPy;
  Academia needs RDKit and matplotlib and pins numpy below 2.
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

---

## Also here

Privacy Policy and Terms of Service for the publishing automation, served via GitHub Pages and
referenced by the social-platform developer apps:

- <https://ec175.github.io/spectrometry_public/privacy.html>
- <https://ec175.github.io/spectrometry_public/terms.html>

## Maintaining the catalogues

`engines/<name>/catalog.json` is the source of truth. The Markdown beside it and the root index
are both generated:

```bash
python tools/build_catalogs.py
```

Edit the JSON, never the generated `CATALOG.md`.

---

## License / use

Personal project, shared so the method is reproducible. The rendered videos are not part of this
repository. If you build on it, a credit is appreciated.
