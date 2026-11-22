# spectrometry.mp4

Simulation engines and video scenes by Clancy. Explore fluid flow, electric fields, orbitals, molecules, and cell colonies through code.

<img src="docs/media/hero.webp" width="100%" alt="A montage of the simulation engines">

## Find the engine behind a video

Select a clip to open its engine guide. The [video recipes](docs/render-recipes.md) list scene names, render commands, and any media you need to supply.

<p align="center">
<a href="engines/wind-tunnel/README.md"><img src="docs/media/wind-tunnel.webp" width="32%" alt="Wind Tunnel: three airfoils shedding vortices, coloured by flow speed"></a>
<a href="engines/field-lines/README.md"><img src="docs/media/field-lines.webp" width="32%" alt="Field Lines: glowing blue electric field lines around moving charges"></a>
<a href="engines/lattice-grid/README.md"><img src="docs/media/lattice-grid.webp" width="32%" alt="Lattice Grid: a honeycomb of tiny coloured circuit symbols flickering to a beat"></a>
<a href="engines/shape-physics/README.md"><img src="docs/media/shape-physics.webp" width="32%" alt="Shape Physics: balls breaking out of a spinning pink ring"></a>
<a href="engines/attractors/README.md"><img src="docs/media/attractors.webp" width="32%" alt="Attractors: a Lorenz butterfly drawn by thousands of glowing particles"></a>
<a href="engines/oscilloscope/README.md"><img src="docs/media/oscilloscope.webp" width="32%" alt="Oscilloscope: a green phosphor rose pattern on a simulated CRT"></a>
<a href="engines/chemical-scope/README.md"><img src="docs/media/chemical-scope.webp" width="32%" alt="Chemical Scope: serotonin and its FTIR spectrum drawn on a green CRT"></a>
<a href="chemical_profiles/README.md"><img src="docs/media/chemical-profiles.webp" width="32%" alt="Chemical Profiles: caffeine with its FTIR spectrum, animated in manim"></a>
<a href="engines/academia/README.md"><img src="docs/media/academia.webp" width="32%" alt="Academia: stacked simulated spectra for a whole class of molecules"></a>
<a href="engines/gas-lab/README.md"><img src="docs/media/gas-lab.webp" width="32%" alt="Gas Lab: gas molecules compressed and ignited inside an engine cylinder"></a>
<a href="engines/orbital-lab/README.md"><img src="docs/media/orbital-lab.webp" width="32%" alt="Orbital Lab: a glowing electron orbital morphing between shapes"></a>
<a href="engines/plume-lab/README.md"><img src="docs/media/plume-lab.webp" width="32%" alt="Plume Lab: three rocket exhaust plumes with shock diamonds"></a>
<a href="engines/bio-lab/README.md"><img src="docs/media/bio-lab.webp" width="32%" alt="Bio Lab: a colony of glowing green and yellow cells"></a>
</p>

## Render your first scene

Install Python 3.10 or newer and add ffmpeg to your PATH. Clone the repository and create a virtual environment:

```sh
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS or Linux. Then render a Wind Tunnel preview:

```sh
cd engines/wind-tunnel/src
python -m pip install -r requirements.txt
python cli.py tri_foil_air_fast --preview
```

The MP4 goes in the engine's `out/` folder. Use a separate environment for each engine. The [engine index](engines/README.md) links to the other guides.

## Reuse and credit

The code is available under the [MIT license](LICENSE). Keep the copyright and license notice with copies of the code. If you share a video or project made with these engines, please credit **spectrometry.mp4 by Clancy** and link to this repository. See [CITATION.cff](CITATION.cff) for a formal citation.

See [repository checks](docs/repository-checks.md) if you're contributing. If a render fails, [open an issue](https://github.com/ec175/spectrometry_public/issues/new?template=bug.yml) with the command and error.

[Website](https://ec175.github.io/spectrometry_public/) · [Instagram](https://instagram.com/spectrometry.mp4) · [YouTube](https://www.youtube.com/channel/UChtdNI2BC1SmkmHEERA4dzg)
