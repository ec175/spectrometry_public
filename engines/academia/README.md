# Academia

**The figure code behind the PDF deliverables.** Give it a compound class as a set of SMILES and
it produces a publication figure: every molecule in the class stacked as a simulated spectrum,
with its structure drawn beside it, as a vector PDF.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Figure previews](figures/)** · 💾 **[Source](src/)** ·
🎬 **[manim subproject](manim/)**

---

## What it does

```
SMILES  ──►  RDKit structure + functional-group inventory
                    │
                    ▼
           group-contribution line list  (predict.py)
                    │
                    ▼
           broadened to a curve, per technique  (spectra.py)
                    │
                    ▼
           stacked with structures inset  (stacks.py)  ──►  vector PDF
```

The estimator is what makes this feasible. A DFT job per molecule per technique would be weeks of
compute for one figure; a group-contribution prediction from the functional-group inventory takes
milliseconds and gets the band *positions* right, which is what a class-comparison figure is
actually about. See the honesty note below.

The same core also reads real instrument scans. Every technique module accepts either a registry
sample id or an explicit `(label, colour, sources)` tuple, so a simulated trace and a measured one
overlay through one call — that is what `example_overlay.py` demonstrates in thirty lines.

## The figure sets

Seven scripts, 84 PDFs, six compound classes plus the BCS drug set.

| script | writes | what |
|---|---|---|
| `amino_acid_sim_test.py` | 4 PDFs | 20 amino acids × FTIR / Raman / UV / ¹H NMR |
| `amino_acid_crystal_vs_amorphous_test.py` | 4 PDFs | the same 20, crystalline **against** amorphous |
| `cannabinoid_sim_test.py` | 8 PDFs | 10 cannabinoids, plain and neutral-against-acid |
| `extra_classes_sim_test.py` | 48 PDFs | lipids, neurotransmitters, peptides, steroids — 20 each, plain and paired |
| `bcs_sim_test.py` | 16 PDFs | the BCS drug set, one figure set per class, all crystal-against-amorphous |
| `examples.py` | 9 PNGs | one end-to-end demo per technique; doubles as a network smoke test |
| `example_overlay.py` | 1 PNG | the minimal simulated-over-measured overlay |

**Every class has its own natural pairing**, and that is the interesting design decision — the
comparison axis is a chemical choice, not a fixed feature:

| class | paired as | why |
|---|---|---|
| amino acids, lipids, neurotransmitters, peptides, steroids, BCS drugs | crystalline / amorphous | the core question of the amorphous-solid-dispersion work |
| cannabinoids | neutral / acid | decarboxylation is the transformation that matters for this class |
| lipids | free acid / salt | what a counter-ion does to the carbonyl region |
| neurotransmitters | freebase / protonated | the amine is the reactive site |
| peptides | reduced / oxidised | the disulfide, which is why both glutathione forms are in the set |
| steroids | free / ester | the prodrug modification |

## Quick start

```bash
pip install -r src/requirements.txt   # RDKit, matplotlib, numpy<2, scipy
cd src && python amino_acid_sim_test.py
```

```
4 stacked PDFs in .../figures/amino_acids
```

Read `amino_acid_sim_test.py` first — it is the reference implementation of the format, and it is
under 170 lines. `extra_classes_sim_test.py` is the one to copy if you are adding a class of your
own.

## Two things to get right

**Axis direction.** FTIR runs 4000→400 cm⁻¹ and NMR runs high→low ppm; Raman, UV and XRD all run
low→high. This is a boolean argument to the stacking call, and getting it wrong produces a figure
that is subtly, embarrassingly backwards. It is the single most common mistake in this format.

**Line shape is physics, not preference.** Vibrational bands are Lorentzian. Electronic bands are
Gaussian **in energy**, which is why UV broadening is done in eV and mapped back to nm rather than
broadened in wavelength. Powder reflections are pseudo-Voigt. The defaults in the per-technique
table are right; changing one should be a deliberate decision.

Related: DFT harmonic frequencies are systematically high and need a scaling factor — about
**0.967** for B3LYP/6-311++G(2d,3p), about **0.95** for wB97XD/6-31G*.

## What these spectra are, and are not

The predicted spectra are **estimates**. Band positions come from a group-contribution model and
are sound; relative intensities are approximate. They are built for *comparison across a class* —
which bands move, which appear, which disappear when a crystal is disordered or an acid becomes a
salt — and they are good at exactly that.

They are **not** measured data and not DFT, and they should not be cited as either. Where the
substituent *is* the signal — a congeneric series where every member differs by one group — a
generic estimator erases precisely what you are looking for, and the right tool is an explicit
per-compound line list.

The `examples/` figures are the honest counterweight: those run the real loaders against real
scans.

## Data access

`registry.py` is the only module that knows where measured data lives, and it fetches over the
network rather than from a checkout. **Go through it; never hardcode a path in figure code.** If
you need offline or reproducible data, pin a commit of your data source and record it.

The simulated figure scripts need no network — they generate everything from SMILES. Only
`examples.py` and the overlay demo reach out.

## Orientation

Everything here is matplotlib, so page size, aspect and DPI are figure arguments. The stacked
layout computes its vertical offsets and label spacing from the trace count rather than from fixed
constants, so it scales from ten rows to twenty without re-tuning. The one thing to scale by hand
is the inset structure size — twenty rows needs smaller structures than ten, or they collide with
the trace above.

---

## manim — an unrelated subproject

[`manim/`](manim/) holds an older body of work: the animated counterpart to the same chemistry.
It is kept here because it is chemistry-adjacent and because it is the **origin of the CRT
chemical-profile format** in the [chemical-scope engine](../chemical-scope/) — those molecule geometries were
transcribed one-to-one from its vertical-profile scenes.

**The library itself lives at [`chemical_profiles/spectro_lib.py`](../../chemical_profiles/), not here** —
about 2,000 lines of axes, trace builders, molecule construction, peak callouts and the
transitions between techniques. That copy is the one to read: it ships with a worked scene, a
guide, pinned requirements, and a verified no-LaTeX render path. This directory holds only what
that one does not — the supporting data modules and build tools:

| file | what |
|---|---|
| `crystal_structures.py` | crystal lattice geometry for the structure animations |
| `morph_sequences.py` | which structure becomes which, and in what order |
| `isoxazole_video_data.py` | geometry and line lists for the arylidene-isoxazolone series |
| `build_montage.py`, `make_audio_bars.py`, `pick_color.py`, `calib.py` | build tools |
| `render_scenes.py`, `render_mp4.py` | the render drivers |
| `HORIZONTAL_FORMAT.md`, `ISOXAZOLE_MORPH_FORMAT.md` | the two format notes |

This subproject shares no code with the figure pipeline above and has entirely different
requirements — **manim** itself, and for some scenes a **LaTeX installation**. Nothing else in
this repository needs either. (The chemical profiles specifically do *not* need LaTeX; they use Pango
text throughout.)

The scene choreography is withheld, as everywhere else here — except for the one worked example in
`chemical_profiles/`.

⚠️ **Never run two manim renders that share a media directory concurrently.** They clobber each
other's partial-movie and text-SVG caches and produce crashes and truncated output. Render
sequentially, or give each render its own media directory.

⚠️ **Run the render drivers in a non-elevated shell.** LaTeX distributions commonly refuse to
compile from an elevated one, and the failure message does not say so.

---

## What is not here

- **The 63 MB of PDFs.** The code is what is being published; [`figures/`](figures/) holds JPEG
  previews of all 84 deliverables plus the example plots, so you can see what each script
  produces before running it.
- **`spectro_scenes.py`** — about 9,900 lines of manim choreography, withheld like every other
  scenes module in this repository, along with the three files that import it.
- **Measured data.** It lives in a separate repository and is fetched by `registry.py`.
- **The marimo UI and preset gallery**, the DFT job inputs, and the isoxazole deliverable
  sub-package — outside the figure pipeline this engine is scoped to.
