# Spectroscopy — manim explainers

Animated, physically-grounded explainers for spectroscopy / spectrography, built
natively in [manim](https://www.manim.community/) (curves and transforms are
**computed**, not screenshotted). Self-contained: its own `.venv` (manim 0.19,
numpy 2.x, **scipy**) and bundled `bin/ffmpeg.exe`, kept separate from the finance
manim one folder up and from `Academia\.venv` (which pins numpy<2).

Scenes that draw equations/units use **`MathTex`**, so this project needs LaTeX
(MiKTeX is installed at `%LOCALAPPDATA%\Programs\MiKTeX`); `render_spectro.ps1`
puts it on PATH automatically.

## Code layout

- **`spectro_lib.py`** — the reusable **object library** and the single source of
  truth for the **CANONICAL FORMAT** (palette, frame/dual-orientation setup, axis &
  tick look, animation idioms — all in its module docstring). Importing it applies
  the canonical camera/background config. Reusable objects: `axis_line`,
  `tick_marks`, `frame_box`, `readout_pill`, `reveal_then_collapse_axis`,
  `lorentzian`/`broaden`/`build_aspirin`/`molecule_glow_updater`,
  `make_signal`/`_stft`/`_colormap`/`heatmap_image`/`HeatmapGeom`/`reveal_curtain`/
  `playhead_line`/`instant_spectrum`. Build new spectroscopy videos by composing
  these rather than copy-pasting from a scene.
- **`spectro_scenes.py`** — choreography only: each scene sequences library objects
  (`from spectro_lib import *`).

## Scenes (`spectro_scenes.py`)

| class | what it shows |
|-------|---------------|
| `Resolution_Temperature` | **★ TEMPLATE (landscape).** A discrete **line list** (stick spectrum, e.g. a DFT IR/Raman calc) grows in, each stick **blooms into a Lorentzian**, and the components **sum into the measured curve**. The Absorbance y-axis is shown then **dropped as the plot fits to full width** (canonical `reveal_then_collapse_axis` move). A live FWHM (γ, "resolution") sweep starts at a "perfect" γ=1 cm⁻¹, broadens to 16, then widens/narrows so bands merge and re-resolve; then a **temperature (T)** cycle redshifts + broadens (300→600→300→10→300 K). x-axis runs high→low cm⁻¹ (FTIR convention). Uses the lab's real `broaden()` math — swap `LINE_LIST` for `gaussian.parse_ir(...)` to make it exact. *Clean template — NO molecule model / scan line / read-out pill (those belong to the shorts variant only).* Final 16:9: `Resolution_Temperature_TEMPLATE.mp4`. |
| `PathLength_Temperature` | **★ TEMPLATE (landscape, UV-Vis).** Same format/choreography as `Resolution_Temperature`, but the **UV-Vis** spectrum: aspirin's electronic-transition sticks bloom into broad **Gaussian** bands (`gaussian`, `ASPIRIN_UV`), the Absorbance axis is shown then dropped to full width, then a **path-length (ℓ) sweep** scales the whole curve (Beer–Lambert A = εcℓ), then a temperature cycle to the solvent boiling point and a low temp. λ axis runs low→high (nm). Final 16:9: `PathLength_Temperature_TEMPLATE.mp4`. |
| `Raman_Spectrum` | **★ TEMPLATE (landscape, Raman).** Same format, five calculable knobs each with a fade-in ticker stacked above the plot: excitation wavelength (ν⁴ law), laser power (linear), temperature (Stokes/anti-Stokes via Boltzmann) — each swept baseline→high→baseline→low→baseline — plus two binary states that morph a copy of the trace into a shaded opposite-state region: polarization (∥→⊥ depolarization, y-zoomed so it reads) and resonance (ring modes enhanced). Double-sided x-axis −1000…2000 cm⁻¹ with the Intensity axis on the Rayleigh line. Band positions/ρ/resonance factors are illustrative. Final 16:9: `Raman_Spectrum_TEMPLATE.mp4`. |
| `NMR_Field_Temperature` | **★ TEMPLATE (landscape, NMR).** TWO stacked, non-overlapping spectra — ¹H (top, 0–12 ppm) and ¹³C (bottom, 0–200 ppm), each δ axis downfield-left, nucleus ID in its x-title; no y-axis. Two knobs: field strength B₀ (linewidth in ppm ∝ 1/B₀ → higher field sharpens + resolves) and temperature (broadens lines + drifts the exchangeable proton). Tickers top-left, clear of peaks; ¹³C peaks kept clear of the ¹H title. Illustrative shifts; multiplets omitted at full-range zoom. Final 16:9: `NMR_Field_Temperature_TEMPLATE.mp4`. |
| `PXRD_Wavelength_Temperature` | **★ TEMPLATE (landscape, PXRD).** Bragg-reflection sticks bloom into peaks; Intensity axis shows then drops to full width; knobs: X-ray wavelength (nλ=2d sinθ slides peaks in 2θ) and temperature (Debye–Waller damps high-angle peaks + thermal-expansion shift); Scherrer peak widths. Ends with a **predicted amorphous trace** (unshaded halo) that the crystalline pattern morphs into, then morphs back. Final 16:9: `PXRD_Wavelength_Temperature_TEMPLATE.mp4`. |
| `Spectrogram` | A genuine **STFT** (`scipy.signal.stft`) of a time-varying signal, drawn as a **time × frequency heatmap** with a moving playhead and a live instantaneous-spectrum panel — a spectrogram is a *stack of spectra over time*. Illustrative chirp + two gated tones; point `make_signal` at real time-resolved data (UV-Vis dissolution, DSC ramp) for lab data. |
| `SpectrogramBranched` | A branch of `Spectrogram` with a **scripted, non-constant red-line motion**: a damped settle to 1 s, holds, then slow back-and-forth "scrubbing" oscillations over the already-revealed heatmap. The reveal only ever advances; the spectrum window stays live throughout. |
| `Spectrogram3D` | **3D version.** The heatmap lies flat on the ground plane with the raw signal as a 3D wave along time; the heatmap then **extrudes into an energy surface** (height = magnitude) and the camera orbits the landscape. Then a **decomposition** phase: a ripple deconstructs a red high-frequency-component trace at the opposite edge. Screen-anchored t/f/E **gizmo** mirrors the orientation. `ThreeDScene` + `Surface`. Render-heavy. |

## Render

```powershell
.\render_spectro.ps1                       # every scene, 16:9 + 9:16, full quality
.\render_spectro.ps1 -Play                 # ...and open each video when it finishes
.\render_spectro.ps1 -Scene resolution_temperature   # one scene (resolution_temperature|spectrogram|branched|3d)
.\render_spectro.ps1 -Only short           # just the 9:16 of each
.\render_spectro.ps1 -Quality l            # fast low-res previews (use this for the 3D scene)
```

Each finished video is copied UP to the `Chemistry` folder. The method-template +
stacked scenes are grouped under `Methods_Manim/` (FTIR folder renamed); the
spectrogram scenes, UV-Vis, object tests and palette stay at the top level:

```
Chemistry/
  Methods_Manim/
    FTIR_Resolution_Temperature/  Resolution_Temperature_youtube.mp4  ..._short.mp4
    UV_PathLength_Temperature/    PathLength_Temperature_youtube.mp4  ...
    Raman_Spectrum/               Raman_Spectrum_youtube.mp4          ...
    NMR_Field_Temperature/        ...
    PXRD_Wavelength_Temperature/  ...
    Stacked_AllMethods/           Stacked_AllMethods_*.mp4 + per-molecule variations
  Spectrogram/  SpectrogramBranched/  Spectrogram3D/   (STFT scenes)
  Object_Tests/  (spectrography primitive demos)   Palette/  (colour swatches)
  manim/         (this project)
```

## Notes

- **One scene, two aspect ratios:** the camera frame is derived from the output
  pixel aspect, so the same file composes in 16:9 and 9:16 — just change `-r`
  (the render script does this for you).
- **No LaTeX for axis text:** numbers/labels use `Text` (Pango); only formulae use
  `MathTex`. MiKTeX auto-installs missing packages on first use. Run the script in a
  *normal* (non-admin) PowerShell — under elevation MiKTeX prints a harmless
  "security risk" warning but still compiles.
- **3D is slow** in the Cairo renderer (per-face). Keep `res_t`/`res_f` modest and
  preview with `-Quality l`.
- **Make it lab-specific:** `LINE_LIST` → a real DFT line list (`gaussian.parse_ir`);
  `make_signal` → real time-resolved data (dissolution kinetics, DSC).
