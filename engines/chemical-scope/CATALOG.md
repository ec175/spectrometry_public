# Chemical Scope — object catalogue

> Twenty-nine molecules, each with an FTIR and a Raman spectrum, drawn as a CRT oscilloscope screen.

The molecule sits above and its spectrum below, both stroked onto a simulated scope face. Three additive buffers composite over a faint graticule: a decaying persistence buffer carrying the spectrum as a scope trace, a second holding the skeletal BONDS, and a third holding the functional-group SYMBOLS and all text. Each gets a half-resolution bloom and then the CRT film filter. Two methods run in sequence, FTIR then Raman, cross-dissolving through the phosphor at the midpoint, with the transition signalled by a one-second ASCII scramble of the lower text rather than a fade.

The two methods are genuinely different, which took work: the same band POSITIONS are reweighted by selection rules, since IR intensity goes with dipole change (C=O, C-O and S=O strong) while Raman goes with polarisability change (C=C, ring breathing and C-C skeletal strong). The Raman legend is the same diagnostic modes REORDERED by Raman strength.

**Output.** 1080x1920 @60, 15 s, silent. The layout is driven by a handful of fractional y-coordinates on `ChemScope` (`sub1_y`, `sub2_y`, `header_bot`, `title_y`, `method_top`), so re-proportioning for a landscape frame is a matter of those numbers plus the molecule's `target_w_frac` — not of the drawing code.


## Modules

What each shipped file is. Compositions (`scenes.py` and friends) are deliberately not part of this repository — see the engine README.

| module | kind | what it gives you |
|---|---|---|
| `chemical_data.py` | data | The catalogue itself. Per molecule: atom coordinates (2-D or 3-D), bonds, labels in role tokens, a fixed bounding box for stable scale, IR line positions and intensities, the diagnostic vibrational bands, titles and metadata. Raman lines and bands are synthesised at import by reweighting the IR set through the selection rules. |
| `osc/chemscope.py` | compositor | The screen. Three additive buffers, the power-on intro, the FTIR-to-Raman handover, the red line playhead, the symbol knockout, the text flare, and the 3-D projection with its two motion modes. |
| `chemical_profile.py` | driver | Render one molecule, or one frame chunk of one. `--preview` for a fast look, `--frames LO:HI --out seg` for a chunk. |
| `render_optimal.py` | driver | The high-quality path. Splits every molecule into frame chunks and work-steals them across three GPU lanes, so idle lanes pick up the next chunk and the tail stays balanced, then concatenates per molecule. Seamless because each chunk warms the phosphor first and the film filter is absolute-frame indexed with a fixed seed. |
| `osc/scope.py` | core | The CRT persistence buffer and graticule — shared with the oscilloscope engine. |
| `osc/crtfilm.py` | post | The film filter, here with a neutral white-phosphor config because the content is multi-coloured. |
| `osc/config.py` | config | `RenderConfig`, ffmpeg resolution, the NVENC probe. |
| `osc/gpu.py` | backend | CuPy backend with numpy fallback. |
| `osc/audio.py` | audio | Decode and mux helpers. Unused by this format, which ships silent, but present because the render path shares them. |


## Format machinery

The parts that are reusable independently of which molecule is on screen.

| object | what it is | notes |
|---|---|---|
| `ChemScope(cfg, molecule, duration=15.0, intro_t=1.5)` | The compositor. Owns the three buffers, the layout, and the timeline. |  |
| `intro_style='speedup'` | The screen flicks on, then an INVISIBLE vertical line sweeps left to right; only where it crosses an object is that slice lit into the phosphor, so the accelerating slice SMEARS through persistence into a solid line. The traces appear to speed up until solid. | Alternatives are a gentler ramp and a single wipe. Setting the intro length to zero disables it; the handover leaves the phosphor already primed either way. |
| `t_switch = duration/2` | FTIR cross-dissolves to Raman through the phosphor, announced by a one-second ASCII scramble of all lower text. | A scramble rather than a fade — a fade reads as an accident, a scramble reads as an instrument changing mode. |
| **line playhead** | A short bright red leading segment of the trace sweeps the curve with a phosphor comet-tail. | A LINE, not a dot. A dot on a spectrum reads as a cursor; a lit leading segment reads as acquisition. |
| `label_push=1.0` | Each functional-group symbol carries a glyph-sized knockout ellipse that clears BONDS ONLY — never the graticule — so letters never sit on a bond while the grid still shows through behind them. | Symbols sit ON their atom. Pushing them further out moves N and O off their own bonds. |
| **text flare** | Subtitles and legend lines occasionally scramble a couple of glyphs and return, on a 1.8–3.6 s gap. | Small enough to read as signal noise rather than as an effect. |
| `motion='rock'` | A bounded left-right rotation, never edge-on, that reveals depth on a molecule with a big flat face. | Global, the same for every molecule. A 3-D molecule 'moves badly' only when its geometry presents a small ring or non-flat face to the camera. |
| `motion='tumble'` | Continuous rotation on three axes at incommensurate frequencies, plus a fixed in-plane spin. | Scale and centre come from the SAMPLED tumble envelope — the union bounding box over 96 sampled rotations — so a tumbling molecule never clips or drifts. |
| `structure3d(smiles, plane_ring='pca')` | Orients by the whole-molecule principal plane, so the flattest face points at the camera and z-variance is minimised. | The fix for anything that sprawls off its aromatic ring. The default aromatic-ring rule is right for scaffolds that HAVE a big flat ring. |
| `chemical_data._raman_factor` | Same positions, different intensity envelope, by selection rule. | Spectra are ILLUSTRATIVE — group-contribution positions with a heuristic Raman reweight, not measured or DFT. Positions are sound, relative intensities approximate, and only about 600–1900 cm-1 is shown. Do not cite them. |


## Molecules — Original set

Frame links are FTIR (first) and Raman (second) — the same molecule under the two methods.

| object | what it is | notes | frames |
|---|---|---|---|
| `MOLECULES['Proline']` | C5H9NO2 · 115.13 g/mol · 8 atoms / 8 bonds · 11 IR lines |  | [1](frames/proline_t030.jpg) [2](frames/proline_t082.jpg) |
| `MOLECULES['Ibuprofen']` | C13H18O2 · 206.28 g/mol · 15 atoms / 15 bonds · 16 IR lines |  | [1](frames/ibuprofen_t030.jpg) [2](frames/ibuprofen_t082.jpg) |
| `MOLECULES['Baclofen']` | C10H12ClNO2 · 213.66 g/mol · 14 atoms / 14 bonds · 12 IR lines |  | [1](frames/baclofen_t030.jpg) [2](frames/baclofen_t082.jpg) |
| `MOLECULES['Indigo']` | C16H10N2O2 · 262.27 g/mol · 20 atoms / 23 bonds · 11 IR lines | Sublimes rather than melts — the metadata line reflects that. | [1](frames/indigo_t030.jpg) [2](frames/indigo_t082.jpg) |
| `MOLECULES['MethyleneBlue']` | C16H18N3S+ · 284.40 g/mol · 20 atoms / 22 bonds · 10 IR lines | A cation; the formula carries the charge. | [1](frames/methyleneblue_t030.jpg) [2](frames/methyleneblue_t082.jpg) |
| `MOLECULES['Sertraline']` | C17H17Cl2N · 306.23 g/mol · 20 atoms / 22 bonds · 10 IR lines |  | [1](frames/sertraline_t030.jpg) [2](frames/sertraline_t082.jpg) |
| `MOLECULES['Morphine']` | C17H19NO3 · 285.34 g/mol · 21 atoms / 25 bonds · 12 IR lines | 3-D. The first molecule to use the billboard rock — depth read through a big flat aromatic ring. | [1](frames/morphine_t030.jpg) [2](frames/morphine_t082.jpg) |
| `MOLECULES['Omeprazole']` | C17H19N3O3S · 345.42 g/mol · 24 atoms / 26 bonds · 11 IR lines |  | [1](frames/omeprazole_t030.jpg) [2](frames/omeprazole_t082.jpg) |


## Molecules — Neurotransmitters, hormones and opioids

Frame links are FTIR (first) and Raman (second) — the same molecule under the two methods.

| object | what it is | notes | frames |
|---|---|---|---|
| `MOLECULES['Dopamine']` | C8H11NO2 · 153.18 g/mol · 11 atoms / 11 bonds · 10 IR lines |  | [1](frames/dopamine_t030.jpg) [2](frames/dopamine_t082.jpg) |
| `MOLECULES['Serotonin']` | C10H12N2O · 176.22 g/mol · 13 atoms / 14 bonds · 10 IR lines |  | [1](frames/serotonin_t030.jpg) [2](frames/serotonin_t082.jpg) |
| `MOLECULES['Adrenaline']` | C9H13NO3 · 183.21 g/mol · 13 atoms / 13 bonds · 10 IR lines |  | [1](frames/adrenaline_t030.jpg) [2](frames/adrenaline_t082.jpg) |
| `MOLECULES['Estradiol']` | C18H24O2 · 272.39 g/mol · 20 atoms / 23 bonds · 13 IR lines |  | [1](frames/estradiol_t030.jpg) [2](frames/estradiol_t082.jpg) |
| `MOLECULES['Testosterone']` | C19H28O2 · 288.43 g/mol · 21 atoms / 24 bonds · 11 IR lines |  | [1](frames/testosterone_t030.jpg) [2](frames/testosterone_t082.jpg) |
| `MOLECULES['Oxycodone']` | C18H21NO4 · 315.37 g/mol · 23 atoms / 27 bonds · 10 IR lines | 3-D, oriented on its aromatic ring like Morphine. | [1](frames/oxycodone_t030.jpg) [2](frames/oxycodone_t082.jpg) |
| `MOLECULES['Glutathione']` | C10H17N3O6S · 307.32 g/mol · 20 atoms / 19 bonds · 11 IR lines | The gamma-Glu-Cys-Gly tripeptide. Generated from SMILES. | [1](frames/glutathione_t030.jpg) [2](frames/glutathione_t082.jpg) |
| `MOLECULES['Mitragynine']` | C23H30N2O4 · 398.50 g/mol · 29 atoms / 32 bonds · 13 IR lines | 3-D. Chiral ring pucker and side chain read through depth. Generated from SMILES. | [1](frames/mitragynine_t030.jpg) [2](frames/mitragynine_t082.jpg) |


## Molecules — Eicosanoids (arachidonic-acid cascade)

Frame links are FTIR (first) and Raman (second) — the same molecule under the two methods.

| object | what it is | notes | frames |
|---|---|---|---|
| `MOLECULES['ProstaglandinE2']` | C20H32O5 · 352.47 g/mol · 25 atoms / 25 bonds · 12 IR lines | C/H/O only, like the rest of the eicosanoid batch. | [1](frames/prostaglandine2_t030.jpg) [2](frames/prostaglandine2_t082.jpg) |
| `MOLECULES['ThromboxaneA2']` | C20H32O5 · 352.47 g/mol · 25 atoms / 26 bonds · 11 IR lines | 3-D. A strained bicyclic acetal cage — the case that forced principal-plane orientation, because the default aromatic-ring rule swung it edge-on and the labels collided. | [1](frames/thromboxanea2_t030.jpg) [2](frames/thromboxanea2_t082.jpg) |
| `MOLECULES['LeukotrieneB4']` | C20H32O4 · 336.47 g/mol · 24 atoms / 23 bonds · 11 IR lines | A linear conjugated-triene fatty acid, 4:1 aspect — needs a wide target width. | [1](frames/leukotrieneb4_t030.jpg) [2](frames/leukotrieneb4_t082.jpg) |
| `MOLECULES['ThromboxaneB2']` | C20H34O6 · 370.49 g/mol · 26 atoms / 26 bonds · 12 IR lines | The cyclic hemiketal, TXA2's hydrolysis product. | [1](frames/thromboxaneb2_t030.jpg) [2](frames/thromboxaneb2_t082.jpg) |


## Molecules — Cannabinoids and alkaloids

Frame links are FTIR (first) and Raman (second) — the same molecule under the two methods.

| object | what it is | notes | frames |
|---|---|---|---|
| `MOLECULES['THC']` | C21H30O2 · 314.47 g/mol · 23 atoms / 25 bonds · 13 IR lines |  | [1](frames/thc_t030.jpg) [2](frames/thc_t082.jpg) |
| `MOLECULES['CBD']` | C21H30O2 · 314.47 g/mol · 23 atoms / 24 bonds · 13 IR lines | A constitutional isomer of THC — open resorcinol plus terpene, against THC's fused pyran. Side by side they are the clearest demonstration in the set that the same formula is not the same spectrum. | [1](frames/cbd_t030.jpg) [2](frames/cbd_t082.jpg) |
| `MOLECULES['THCA']` | C22H30O4 · 358.48 g/mol · 26 atoms / 28 bonds · 12 IR lines | THC plus a carboxylic acid; decarboxylates to THC. | [1](frames/thca_t030.jpg) [2](frames/thca_t082.jpg) |
| `MOLECULES['Cocaine']` | C17H21NO4 · 303.35 g/mol · 22 atoms / 24 bonds · 13 IR lines | 3-D with the multi-axis tumble rather than the single-axis rock. | [1](frames/cocaine_t030.jpg) [2](frames/cocaine_t082.jpg) |
| `MOLECULES['Nicotine']` | C10H14N2 · 162.23 g/mol · 12 atoms / 13 bonds · 12 IR lines |  | [1](frames/nicotine_t030.jpg) [2](frames/nicotine_t082.jpg) |


## Molecules — Controlled substances

Frame links are FTIR (first) and Raman (second) — the same molecule under the two methods.

| object | what it is | notes | frames |
|---|---|---|---|
| `MOLECULES['MDMA']` | C11H15NO2 · 193.25 g/mol · 14 atoms / 15 bonds · 16 IR lines | 3-D tumble, oriented on its benzodioxole ring. | [1](frames/mdma_t030.jpg) [2](frames/mdma_t082.jpg) |
| `MOLECULES['Fentanyl']` | C22H28N2O · 336.47 g/mol · 25 atoms / 27 bonds · 14 IR lines | 3-D tumble on the PRINCIPAL PLANE — it sprawls off its aromatic ring, so the ring rule swung it edge-on (z-spread 6.2 down to 3.1 after the fix). | [1](frames/fentanyl_t030.jpg) [2](frames/fentanyl_t082.jpg) |
| `MOLECULES['Methamphetamine']` | C10H15N · 149.23 g/mol · 11 atoms / 11 bonds · 14 IR lines | 3-D tumble, principal-plane oriented. | [1](frames/methamphetamine_t030.jpg) [2](frames/methamphetamine_t082.jpg) |
| `MOLECULES['Alprazolam']` | C17H13ClN4 · 308.77 g/mol · 22 atoms / 25 bonds · 18 IR lines | 3-D tumble, principal-plane oriented. Displayed under its trade name; the data key stays the chemical name. | [1](frames/alprazolam_t030.jpg) [2](frames/alprazolam_t082.jpg) |


## Parameters that matter

Per-molecule tuning is a small dict; the rest is global.

| parameter | lives in | what it controls | usable range |
|---|---|---|---|
| `target_w_frac` | MOLECULES[k]['size'] | How wide the molecule draws as a fraction of the frame. | 0.62–0.74 for steroids, tripeptides and alkaloids; 0.82–0.88 for long fatty-acid chains |
| `sym_px` | MOLECULES[k]['size'] | Functional-group symbol size. | Scaled to the molecule by default |
| `motion` | MOLECULES[k] | `rock` or `tumble`. | Only meaningful when the atom coordinates carry a z |
| `plane_ring` | structure3d() | `aromatic` keys orientation on the aromatic ring; `pca` on the whole-molecule principal plane. | Use `pca` for anything that sprawls off its ring |
| `intro_t` | ChemScope | Power-on sweep length. | 1.5 s; 0 disables |
| `t_switch` | ChemScope | FTIR-to-Raman handover time. | duration/2, so it follows the clip length automatically |
| `title_y / method_top / header_bot / sub1_y / sub2_y` | ChemScope | The layout. The molecule zone is the gap between `header_bot` and `method_top`. | Both the normal and intro text paths read these — keep them in sync |
| `rock_deg / rock_period` | ChemScope | Billboard rock amplitude and period. | Global: 32 deg over 13 s |
| `breath` | _curve_points | Trace liveliness. | Keep it SMALL — plus or minus 0.5%. At 5% the curve bounced too much to read. |


---

*Generated from `catalog.json` by `tools/build_catalogs.py` — edit the JSON, not this file.*
