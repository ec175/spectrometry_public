# Rendering a chemical profile

Everything needed to go from a clean machine to a finished 1080×1920 MP4, and then
to a molecule of your own.

- [1. What you are building](#1-what-you-are-building)
- [2. Install](#2-install)
- [3. Render it](#3-render-it)
- [4. Reading the output](#4-reading-the-output)
- [5. Quality flags, and which to actually use](#5-quality-flags-and-which-to-actually-use)
- [6. Authoring a new molecule](#6-authoring-a-new-molecule)
- [7. The data contract, attribute by attribute](#7-the-data-contract-attribute-by-attribute)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. What you are building

A **chemical profile** is a vertical (9:16) short that shows one molecule and the
spectra that identify it. It runs in three phases, all in one continuous take:

| phase | top half | bottom half |
|-------|----------|-------------|
| **1 — NMR** | the molecule eases in and spins one full turn, atoms labelled with **assignment numbers** | ¹³C and ¹H NMR spectra, peak markers carrying the *same* numbers |
| **2 — FTIR** | the molecule reverts from numbers to **atom symbols** | the NMR spectra cross-fade to a single compact FTIR trace |
| **3 — vibrations** | each IR mode plays in turn: the driving atoms move, and the motion falls off sharply through the rest of the skeleton | a cursor and a numbered band marker track the matching FTIR peak |

The point of the numbering is that it ties together: peak 7 on the ¹³C spectrum is
atom 7 on the structure, and in phase 3 the bond that produces the 1750 cm⁻¹ band is
the bond you see moving.

Two files do all of it:

| file | what it is |
|------|-----------|
| **`spectro_lib.py`** | the object library — palette, axes, molecule builders, line-broadening maths, gradient fills. Nothing here is scene-specific. |
| **`chemical_profile.py`** | the `VerticalProfile` scene class (the choreography) and `VerticalProfile_Aspirin` (the worked example). |

---

## 2. Install

You need **Python 3.10 or newer** and **ffmpeg**. You do *not* need LaTeX — these
scenes use Pango text, not `MathTex`, so the usual manim LaTeX setup is skippable.

### Step 1 — ffmpeg

manim shells out to ffmpeg to combine frames into a video.

```bash
# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt install ffmpeg

# Windows (winget), or grab a build from https://ffmpeg.org/download.html
winget install Gyan.FFmpeg
```

Check it: `ffmpeg -version`. If that prints a version, you are set.

### Step 2 — a virtual environment

> **Do this in a fresh virtual environment.** manim requires **numpy 2.x**. If you
> pip-install it into an environment that already holds a numpy<2 scientific stack,
> you will silently break that stack. This is the single most common way to have a
> bad afternoon here.

```bash
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public/chemical_profiles

python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.venv\Scripts\activate            # Windows PowerShell

pip install -r requirements.txt
```

### Step 3 — confirm

```bash
manim --version          # v0.19.1
```

---

## 3. Render it

From inside `chemical_profiles/`:

```bash
manim -qh -r 1080,1920 chemical_profile.py VerticalProfile_Aspirin
```

That is the whole command. Broken down:

| part | meaning |
|------|---------|
| `-qh` | **q**uality **h**igh — 1080p. See §5 for the other levels. |
| `-r 1080,1920` | resolution, width first. **This is what makes it vertical.** Without it you get a 16:9 frame and the layout will be wrong — the format composes for portrait. |
| `chemical_profile.py` | the file |
| `VerticalProfile_Aspirin` | the scene class to render |

**Iterate at low quality first.** A `-qh` pass takes several minutes; `-ql` takes
well under one and is the right way to check that a change landed:

```bash
manim -ql -r 540,960 chemical_profile.py VerticalProfile_Aspirin
```

To render only the first couple of animations while you are working on the opening:

```bash
manim -ql -r 540,960 -n 0,3 chemical_profile.py VerticalProfile_Aspirin
```

---

## 4. Reading the output

`manim.cfg` in this folder sets `media_dir = ./renders`, so everything lands under:

```
chemical_profiles/
└── renders/
    └── videos/
        └── chemical_profile/
            ├── 1920p30/                      ← -qh -r 1080,1920
            │   └── VerticalProfile_Aspirin.mp4
            └── 480p15/                       ← -ql -r 540,960
                └── VerticalProfile_Aspirin.mp4
```

The folder is named for the **height** and frame rate, which is why a vertical
1080×1920 render appears under `1920p30` rather than `1080p30`.

`renders/` is gitignored — nothing you render will end up in a commit.

---

## 5. Quality flags, and which to actually use

| flag | resolution | fps | use it for |
|------|-----------|-----|-----------|
| `-ql` | 480p | 15 | every iteration. Fast enough to stay in flow. |
| `-qm` | 720p | 30 | checking that text and peak labels are legible |
| `-qh` | 1080p | 60 | the finished render |
| `-qk` | 4K | 60 | almost never worth it for a 9:16 short |

Two extras worth knowing:

- **`--fps 30`** — `-qh` defaults to 60 fps. These profiles look identical at 30 and
  render in half the time. Adding `--fps 30` to a final pass is close to free.
- **`--disable_caching`** — manim caches partial movie files. If a render starts
  producing stale or truncated output after you have been editing heavily, this is
  the fix. Do not reach for it first; it makes every render slower.

> **Never run two renders of the same file at the same time.** They share
> `renders/.../partial_movie_files` and the text-SVG cache, and will clobber each
> other — producing crashes and silently truncated MP4s. Render sequentially, or
> give each its own directory with `--media_dir renders_<name>`.

---

## 6. Authoring a new molecule

A new molecule is **one subclass**. Nothing else in the file changes. Copy
`VerticalProfile_Aspirin`, rename it, and replace every value.

```python
class VerticalProfile_Caffeine(VerticalProfile):
    """Caffeine — the methylxanthine stimulant."""
    title = "CAFFEINE"
    subtitle  = "C₈H₁₀N₄O₂  ·  194.19 g/mol  ·  Melting Point 235 °C"
    subtitle2 = "Methylxanthine  ·  Basic  ·  Stimulant"
    solvent   = "d6-DMSO"
    subtitle_ftir  = "logP −0.07  ·  TPSA 58.4 Å²  ·  H₂O sol. 21 g/L"
    subtitle2_ftir = "Amide  ·  Imidazole  ·  Aromatic"
    curve_shade = 'purple'

    AT    = {...}      # atom  -> (x, y) in build space
    BB    = [[...]]    # bounding box, fixes the scale
    BONDS = [...]      # (a, b, is_double)
    LBL_ATOM = [...]   # symbol labels, shown in the FTIR phase
    LBL_NUM  = [...]   # number labels, shown in the NMR phase

    nmrc = [...]       # ¹³C peaks: (ppm, intensity, assignment number)
    nmrh = [...]       # ¹H  peaks: (ppm, intensity, assignment number)
    ir_lines = [...]   # FTIR: (wavenumber, intensity)
    vib_modes = [...]  # the phase-3 sweep
```

Then:

```bash
manim -ql -r 540,960 chemical_profile.py VerticalProfile_Caffeine
```

### ⚠️ Things that are not auto-derived, and will be wrong if you copy them

This is the part worth slowing down on. Almost every value above is
molecule-specific text or molecule-specific data. Inheriting it from aspirin
produces a video that looks completely finished and states things that are false.

- **Every subtitle line.** `subtitle` / `subtitle2` / `subtitle_ftir` /
  `subtitle2_ftir` / `solvent` are all free text. Nothing validates them.
- **The NMR peak lists *and* their assignment numbers.** `nmrc` and `nmrh` must
  match this molecule's real spectrum, and the numbers must match your `LBL_NUM`
  atom numbering. A stale assignment is a factual error displayed on screen for
  fifteen seconds. Check each shift against a reference before rendering.
- **`ir_lines` and `vib_modes`.** The bands must be this molecule's IR, and each
  mode's `drivers` must name atoms that really do move in that mode.

### Getting the geometry

`AT` is a plain dict of 2-D coordinates in build space, so you can lay a molecule
out by hand for something small. For anything with a ring system it is much faster
to generate coordinates from a SMILES string with RDKit and paste the result in:

```python
from rdkit import Chem
from rdkit.Chem import AllChem

m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")      # aspirin
AllChem.Compute2DCoords(m)
conf = m.GetConformer()
for i, atom in enumerate(m.GetAtoms()):
    p = conf.GetAtomPosition(i)
    print(f"{atom.GetSymbol()}{i}: ({p.x:.2f}, {p.y:.2f})")
```

RDKit is **not** in `requirements.txt` — it is only needed if you want this
shortcut, and it is a heavy dependency. `pip install rdkit` in the same venv if you
want it.

An `AT` value may also be a **3-tuple** `(x, y, z)`. Flat 2-tuples are treated as
`z=0`. Use 3-D coordinates when a molecule's rings genuinely leave the plane and a
flat 2-D layout would collapse them on top of each other — the spin then reveals
real depth.

---

## 7. The data contract, attribute by attribute

Every one of these is a class attribute on `VerticalProfile`, with a default. A
subclass overrides what it needs.

### Identity and text

| attribute | type | what it is |
|-----------|------|-----------|
| `title` | `str` | the molecule name, shown in the bottom-left tag |
| `subtitle` | `str` | NMR phase, **quantitative** — formula · MW · mp |
| `subtitle2` | `str` | NMR phase, **qualities** — drawn in the accent colour |
| `solvent` | `str` | NMR acquisition solvent, e.g. `d6-DMSO` |
| `spin_rate` | `str` | sample spin rate, e.g. `20 Hz` |
| `subtitle_ftir` | `str` | FTIR phase, quantitative — logP · TPSA · solubility |
| `subtitle2_ftir` | `str` | FTIR phase, qualities |
| `curve_shade` | `str` | which `SHADE` family colours every curve: `blue` `red` `green` `purple` `pink` |

### Structure

| attribute | type | what it is |
|-----------|------|-----------|
| `AT` | `dict[str, tuple]` | atom key → `(x, y)` or `(x, y, z)` in build space |
| `BONDS` | `list[tuple]` | `(atom_a, atom_b, is_double)` |
| `BB` | `list[list]` | an invisible bounding box. **Fixes the scale** so the molecule does not "breathe" as labels change between phases. |
| `LBL_ATOM` | `list[tuple]` | `(atom, text, colour)` — the symbol labels used in the FTIR phase |
| `LBL_NUM` | `list[tuple]` | `(atom, text, colour)` — the numbered labels used in the NMR phase |
| `target_w_frac` | `float` | molecule width as a fraction of the frame. Raise it for large sprawling structures. |
| `label_font_size` | `int` | **does not auto-scale.** A wider molecule gets scaled down to fit, which shrinks its labels with it — raise this for big molecules or the symbols become unreadable. |

### Spectra

| attribute | type | what it is |
|-----------|------|-----------|
| `nmrc` | `list[tuple]` | ¹³C: `(ppm, intensity, assignment_number)` |
| `nmrh` | `list[tuple]` | ¹H: `(ppm, intensity, assignment_number)` |
| `nmrc_window` | `tuple` | ¹³C axis range, high→low. Default `(200.0, 0.0)`. |
| `nmrh_window` | `tuple` | ¹H axis range, high→low. Default `(10.0, 0.0)`. |
| `ir_lines` | `list[tuple]` | FTIR: `(wavenumber_cm⁻¹, intensity)` |
| `ir_window` | `tuple` | FTIR range. Default `(1900.0, 600.0)` — the fingerprint region. |
| `ir_fwhm` | `float` | line width used to broaden the stick list into a real trace |

### Vibrations (phase 3)

`vib_modes` is an ordered list of dicts. Each one is one beat of the sweep:

```python
{
    'name': "Ester C=O stretch",       # the label shown on screen
    'band': 1750,                      # which FTIR peak the cursor tracks
    'wn':   "1730–1770 cm⁻¹",          # the range, shown as text
    'drivers': [('O1', 'stretch', 'C2', +1)],
}
```

A driver is `(atom, kind, reference_atom, sign)`:

- **`'stretch'`** moves `atom` along the `atom → reference` axis.
- **`'perp'`** moves it perpendicular to that axis — bends and out-of-plane modes.
- `sign` flips the direction, which is how you make two halves of a ring move in
  opposition for an antisymmetric stretch.

Displacement spreads from each driver to every other atom with a sharp Gaussian
falloff, so the bulk of the molecule stays put and the mode reads as local. Two
knobs tune that: `vib_amp` (how far, default `0.35`) and `vib_sigma` (how far the
motion carries, default `0.65`).

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'manim'`**
The virtual environment is not active, or you installed into a different one. Run
`which manim` / `Get-Command manim` and check it points inside `.venv`.

**The layout is wrong — things overlap or run off the frame**
You almost certainly omitted `-r 1080,1920`. This format composes for a portrait
frame; in 16:9 the top and bottom halves collide.

**`FileNotFoundError` mentioning ffmpeg**
ffmpeg is not on `PATH`. See §2 step 1. manim finds it through `PATH` only.

**A numpy error, or an unrelated project breaks after installing**
manim needs numpy 2.x and you installed it into a shared environment. Make a fresh
venv and install there. See §2 step 2.

**Renders are corrupt or truncated after heavy editing**
The partial-movie cache is dirty. Add `--disable_caching` for one pass, or delete
the `renders/` folder.

**Labels are tiny on a large molecule**
Expected — label size scales with the molecule's fit-to-frame scale. Raise
`label_font_size`, and `target_w_frac` if it should be physically bigger.

**Double bonds look like two separate parallel lines**
The gap is too wide for this molecule's scale. Set `dbl_sep` to around `0.03`–`0.04`
on the subclass. When in doubt go tighter — the default reads loose on most
structures.
