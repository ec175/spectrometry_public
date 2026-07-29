# spectrometry_public

**Drug profiles, built in [manim](https://www.manim.community/).**

A drug profile is a vertical short that shows one molecule and the spectra that
identify it — the structure spins over its ¹³C and ¹H NMR spectra with the peaks
numbered to match the atoms, cross-fades to an FTIR trace, and then plays each
infrared vibration mode while a cursor tracks the band it produces.

This repo is the **source and the instructions**, so you can render one yourself.

### Where to watch them

- YouTube — [**spectrometry.mp4**](https://www.youtube.com/channel/UChtdNI2BC1SmkmHEERA4dzg)
- X — [**@spectrometrymp4**](https://x.com/spectrometrymp4)

<!-- The YouTube link uses the channel ID, which cannot go stale. Swap it for the
     vanity /@handle URL if you prefer. TikTok and Instagram are omitted on purpose
     -- add them here once those accounts are actually publishing. -->

---

## Render one

```bash
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public/drug_profiles

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

manim -qh -r 1080,1920 drug_profile.py VerticalProfile_Aspirin
```

You need Python 3.10+ and ffmpeg. **You do not need LaTeX** — these scenes use Pango
text rather than `MathTex`, so the usual manim LaTeX setup does not apply.

Output lands in `drug_profiles/renders/videos/drug_profile/1920p30/`.

> ### 📖 [**drug_profiles/GUIDE.md**](drug_profiles/GUIDE.md)
> The detailed walkthrough: installing from scratch, every quality flag and which to
> actually use, how to read the output tree, **how to author a new molecule**, the
> full data contract attribute by attribute, and troubleshooting.

---

## What is in here

| path | what |
|------|------|
| [`drug_profiles/drug_profile.py`](drug_profiles/drug_profile.py) | the `VerticalProfile` scene — all the choreography — plus `VerticalProfile_Aspirin` as a fully worked example |
| [`drug_profiles/spectro_lib.py`](drug_profiles/spectro_lib.py) | the object library: palette, axes, molecule builders, line-broadening maths, gradient fills |
| [`drug_profiles/GUIDE.md`](drug_profiles/GUIDE.md) | the detailed how-to |
| [`drug_profiles/requirements.txt`](drug_profiles/requirements.txt) | pinned dependencies |
| [`drug_profiles/manim.cfg`](drug_profiles/manim.cfg) | canonical dark background, 30 fps, output into `renders/` |

A new molecule is **one subclass** — every molecule-specific value is a class
attribute. `VerticalProfile_Aspirin` is the template; read it top to bottom and you
have the entire data contract. GUIDE.md §6 walks through it.

### A note on the data

The spectra are **representative, not measured**. Peak positions are seeded from
literature values and group-contribution estimates, then hand-corrected so the
diagnostic bands are right. They are good enough to teach with and are not a
substitute for a real acquisition. If you author a molecule, check every shift and
every band against a reference before you render — nothing in the code validates
them, and a wrong assignment is a factual error displayed on screen.

---

## Also here

Privacy Policy and Terms of Service for the publishing automation, served via
GitHub Pages and referenced by the social-platform developer apps:

- <https://ec175.github.io/spectrometry_public/privacy.html>
- <https://ec175.github.io/spectrometry_public/terms.html>

---

## License / use

Personal project, shared so the method is reproducible. The rendered videos are not
part of this repository. If you build on it, a credit is appreciated.
