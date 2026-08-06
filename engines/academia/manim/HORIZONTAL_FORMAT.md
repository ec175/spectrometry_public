# Horizontal (16:9) spectroscopy format — the current standard

**Reference implementation:** [`pat_raman_scene.py`](pat_raman_scene.py) (class `PATRaman`).
**Render:** `.\render_pat_raman.ps1 -Quality h` → `Chemistry\Methods_Manim\PAT_Raman\`.

This supersedes the **old** landscape format (the method-template scenes in
`spectro_scenes.py`: `Resolution_Temperature`, `PathLength_Temperature`, `Raman_Spectrum`,
`NMR_Field_Temperature`, `PXRD_Wavelength_Temperature`, `Stacked_AllMethods`). Those are now
**legacy** — use this format for new horizontal videos. It reuses the same `spectro_lib`
palette/helpers and the excitation-colour spectrum look from `VerticalProfileUVRamanStrip`, but
is a clean standalone scene rather than a subclass, so it is easy to copy per molecule/technique.

---

## The layout (16:9, `fw≈14.222 × fh=8.0`)

```
┌───────────────────────────┬───────────────────────────────┐
│  MOLECULE NAME       (INK) │  <Method> Spectrum · <λ> nm    │  ← title bold, top-left of
│  subtitle           (MUTED)│  <mode name>   (laser colour)  │    EACH section
│                            │  <wavenumber>  (#b8c3d1)       │
│                            │                                │
│      2D structure          │        spectrum trace          │
│   (static, vibrating)      │   (red outline, no fill)       │
│                            │   1  2  3 …  numbered peaks     │
│                            │   ─────── Raman shift (cm⁻¹)    │
└───────────────────────────┴───────────────────────────────┘
        LEFT section                    RIGHT section
                    │ vertical divider (GRID) at DIVX≈-0.12
```

- **Two sections split by one vertical divider.** Molecule left, spectrum right.
- **Each section is titled top-left** — a bold `INK` title + a `MUTED`/coloured subtitle.
  The molecule title is the compound name; the spectrum title is `"<Method> Spectrum · <λ> nm"`.
- **No section is left unlabeled; there is no scene-level title.**

---

## Molecule (left) — the rules that matter

1. **One 2D skeletal structure**, built from `AT` (atom → (x,y)) + `BONDS` (a, b, double?) +
   `LBL_ATOM` (which atoms get a symbol). Coordinates come from
   `Academia\gen_vertical_profiles.py` `structure()` (SMILES → normalised 2D coords, consistent
   atom numbering). `structure3d()` exists if a genuinely 3D pose is ever wanted.
2. **Upright orientation via `BASE_ROT`** (degrees CCW applied to every coord once). Pick it so a
   defining bond is level/plumb — here `14.47°` makes the carbonyl **C=O exactly vertical**.
3. **STRUCTURE_PRESET look, drawn directly** (not via `styled_molecule`, so we control it):
   - **BG halos** behind each functional-group symbol (`Circle` fill = `BG`, `HALO_R×` symbol
     height), added UNDER the glyph so bonds don't cross the letters.
   - **Symbols pushed OUT along their own bond axis** (`LBL_PUSH` = per-symbol `(ref_atom, dist)`),
     so `O`/`S`/`CH₃` line up with their bonds instead of being shoved radially from the centre.
   - **Tight double bonds**, and the **ring C=C doubles slightly tighter** than the carbonyl
     (`DBL_GAP_RING < DBL_GAP`).
   - Fixed on-screen symbol size `LBL_ONSCREEN`, bond width `BOND_W`.
4. **FIXED screen transform `T(p)` — the anti-drift rule.** Compute `SCALE` + centre `CBASE` ONCE
   from the base geometry (fit `TARGET_W`, centre at `MCENTER`). Every frame maps atoms through the
   SAME `T`. **Never recentre on the perturbed centroid** — doing so makes asymmetric modes slide
   the whole molecule. With a fixed `T`, a vibrating atom moves only its own bonds.
5. **Static** — no wobble, no spin (both were tried and removed). The only motion is the vibration.

## Vibration engine — `perturb(drivers, ph, direct)`

A mode is a list of **drivers** `(atom, kind, ref, sign)`. `ph = sin(2π·freq·t)` oscillates.

- **kinds:** `'stretch'` = displace along `atom→ref`; `'perp'` = ⟂ to it (in-plane);
  `'axis'` = along the molecular **para axis** `PARA` (`ipso-N → ipso-S`). Add axes/kinds as needed.
- **`sign`** is a float (amplitude weight, e.g. `+1.6` for the atom that should move most).
- **spread modes:**
  - default (`direct=False`): each driver's displacement spreads to nearby atoms with a Gaussian
    (`SIGMA`) — soft, organic, good for delocalised modes (breathing, ring deformation).
  - **`direct=True`: rigid per-atom displacement, NO spread → untouched atoms are a perfectly
    frozen node.** Use when the real eigenvector has a hard node (e.g. an amide end that must not
    move). This is how the 1081 "ring-sensitive C–S stretch" was matched to the DFT/mol2 data.
- Global amplitude = `VA`; per-mode `ascale` scales it; per-mode `freq` sets the speed.
  `'RC'` is a virtual ring-centroid reference (never drawn).

**Matching a real mode:** if you have displaced geometries (e.g. `A` = equilibrium, `B`/`C` =
turning points, as `.mol2` with connectivity), read off which atoms move, in what direction, and
their relative amplitudes, then reproduce it with drivers — use `direct=True` if there's a node.
See `Downloads\Brian_Raman\mode_analysis.html` for the worked A/B/C read.

## Spectrum (right) — the rules that matter

- **Data = `MODES`** (one dict per band): `band` (cm⁻¹), `inten` (relative height), `name`, `wn`,
  `drivers`, `ascale`, `freq`, optional `direct`. `RAMAN_LINES` = `[(band, inten)]`.
- **Only include the peaks you've identified.** Don't draw unrelated experimental peaks.
- **Peak heights = experimental relative intensities** — read them off the real spectrum and set
  `inten` normalised to the tallest band. The curve is `broaden(RAMAN_LINES, FWHM)` normalised so
  the tallest peak = full height `H`.
- **Red trace OUTLINE only** — coloured by the **excitation-laser** rule
  `ramL = brighten(visible_color(λ))` (532→green, 633→orange-red, **785→deep red**, …).
  **No gradient fill, no baseline glow bar, no wavy/breathing motion** (all removed).
- **No y-axis** (intensity is relative). Keep the x-axis line + `"Raman shift (cm⁻¹)"` label only.
- **Numbered peak markers** `1..n`, **uniform**: constant `LIFT` → every leader line is the same
  length and every number sits the same height above its own peak.
- **Dashed white playhead** (`cursor`) that rides to each band.
- **Window** `[W_LO, W_HI]` wide enough to include every band (mind low bands like 390).

## Choreography

1. **Both panels fade/draw in together** in one `self.play` (molecule + both titles + axis + trace
   + peak markers). Keep the clock stopped here so the static `ref` matches the live molecule at
   `t=0` (no size/pose jump when it goes live).
2. Swap the static molecule for `always_redraw(build_mol)`, start the clock, reveal the cursor.
3. **Step through `MODES`**: set the active drivers/ascale/freq/direct, slide the playhead to the
   band, cross-fade the mode subtitle, dwell (~2 s) while the mode vibrates. Fade the last label.

---

## Adapting to a new molecule / technique — checklist

- [ ] `AT` / `BONDS` / `LBL_ATOM` from `gen_vertical_profiles.structure(SMILES)`; set `RING`.
- [ ] `BASE_ROT` so the structure sits upright (a defining bond level/plumb).
- [ ] `LBL_PUSH` per functional-group symbol; check halos/tight-doubles read cleanly at `-q l`.
- [ ] `MODES`: bands, **experimental `inten`**, names, wavenumbers, and **drivers** per mode
      (use `direct=True` + the right `axis`/`stretch`/`perp` to match the real eigenvector).
- [ ] `EXCITATION` (nm) → the trace colour; retitle `"<Method> Spectrum · <λ> nm"`; set molecule
      title/subtitle. Widen `[W_LO, W_HI]` to fit all bands.
- [ ] Iterate at `-q l`, extract frames with ffmpeg to check; only `-q h` for the final.
- [ ] Other techniques (UV/IR/NMR/PXRD): same skeleton — swap the x-axis label/units, the colour
      rule if not laser-based, and the peak-shape (`broaden` Lorentzian is fine for IR/Raman).

## Gotchas

- **`from manim import *` AND `from spectro_lib import *`** are both required (the lib doesn't
  re-export manim's `Scene`, `ValueTracker`, etc.).
- Some unicode subscripts aren't in the Pango font — `ₐ` renders as tofu. Avoid mode names like
  `ν₈ₐ`; write `8a` in normal text.
- Iterate landscape (default resolution); this format is landscape-only (no `-r 1080,1920`).
- Scene fps/length is set by the dwell timing, independent of `freq` — changing vibration speed
  doesn't change the video length.
