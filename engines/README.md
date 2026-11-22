<div align="center">

# All engines

[← Front page](../README.md) · [Full catalogue](CATALOG.md) · [Chemical Profiles](../chemical_profiles/README.md)

</div>

Choose a guide below for setup and a first render. For commands tied to specific clips, see the [video recipes](../docs/render-recipes.md).

| engine | what it makes |
|---|---|
| **[Wind Tunnel](wind-tunnel/README.md)** | real 2-D fluid dynamics with two solvers: flow past wings, falling plates, merging droplets, shock waves |
| **[Field Lines](field-lines/README.md)** | field lines of electric, magnetic and potential-flow fields, with glowing packets riding along them |
| **[Lattice Grid](lattice-grid/README.md)** | a lattice of circuit symbols that re-rolls on the beat, coloured by an RGB time delay |
| **[Shape Physics](shape-physics/README.md)** | balls, spinning shells, plinko and gears, where the simulation decides when the music plays |
| **[Attractors](attractors/README.md)** | tens of thousands of particles flowing through chaotic systems like the Lorenz butterfly |
| **[Oscilloscope](oscilloscope/README.md)** | a simulated CRT screen, plus a film filter you can put on any video |
| **[Chemical Scope](chemical-scope/README.md)** | 29 molecules with their FTIR and Raman spectra, drawn on that CRT |
| **[Academia](academia/README.md)** | stacked spectra figures for whole classes of molecules, as vector PDFs |
| **[Orbital Lab](orbital-lab/README.md)** | orbitals, bonding patterns, and molecular orbital animations |
| **[Plume Lab](plume-lab/README.md)** | animated exhaust plumes under different ambient conditions |
| **[Gas Lab](gas-lab/README.md)** | animated piston engines and the flow inside them |
| **[Bio Lab](bio-lab/README.md)** | stylized cell colonies and other microscopy inspired scenes |

## Setup

Install Python 3.10 or newer and ffmpeg. Create a virtual environment for each engine and install its requirements there. The engine guide gives the commands for that engine. Academia has separate NumPy requirements. A compatible GPU can speed up rendering in supported engines.

To update the generated catalogues, edit the corresponding `catalog.json` and run `python tools/build_catalogs.py` from the repository root.
