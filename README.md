<div align="center">

<img src="docs/media/hero.webp" width="100%" alt="spectrometry.mp4: looping clips from six of the engines - a wind tunnel, electric field lines, a strange attractor, an oscilloscope, a circuit lattice and a molecule on a CRT">

### The code behind the [spectrometry.mp4](https://instagram.com/spectrometry.mp4) videos

Fluid sims, field lines, particle swarms, bouncing shapes and molecules.<br>
Written in Python, turned into video with ffmpeg. Free to use.

[![Instagram](https://img.shields.io/badge/Instagram-@spectrometry.mp4-E4405F?logo=instagram&logoColor=white)](https://instagram.com/spectrometry.mp4)
[![YouTube](https://img.shields.io/badge/YouTube-spectrometry.mp4-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/channel/UChtdNI2BC1SmkmHEERA4dzg)
[![X](https://img.shields.io/badge/X-@spectrometrymp4-000000?logo=x&logoColor=white)](https://x.com/spectrometrymp4)<br>
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#try-one-in-five-minutes)
[![GPU optional](https://img.shields.io/badge/GPU-optional-76B900?logo=nvidia&logoColor=white)](#questions)

</div>

Hey! If you got here from one of my videos, you're in the right place. Nothing on the channel is
animated by hand: a real simulation runs, and every frame it produces goes straight into the video.
This repo is where I share the engines that do that, along with the scene scripts behind most of
my videos, so you can render them yourself, pull them apart, and make your own things with them.

## Find the engine behind a video

Tap the one that looks like what you saw.

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
</p>

### Not public yet

These made some of my newer videos, but their code isn't cleaned up for sharing yet. They'll
show up here when it is. Follow along on
[Instagram](https://instagram.com/spectrometry.mp4) in the meantime.

<p align="center">
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/gas-lab.webp" width="32%" alt="Gas Lab (not public yet): gas molecules compressed and ignited inside an engine cylinder"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/orbital-lab.webp" width="32%" alt="Orbital Lab (not public yet): a glowing electron orbital morphing between shapes"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/plume-lab.webp" width="32%" alt="Plume Lab (not public yet): three rocket exhaust plumes with shock diamonds"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/mechanics-board.webp" width="32%" alt="Mechanics Board (not public yet): a physics problem drawn and solved on white paper"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/cinematic.webp" width="32%" alt="Cinematic (not public yet): a title card over a swirling blue field"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/elements.webp" width="32%" alt="Elements (not public yet): the carbon tile from the 118 element series"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/field-notes.webp" width="32%" alt="Field Notes (not public yet): creatures swimming through a living blue ecosystem"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/bio-lab.webp" width="32%" alt="Bio Lab (not public yet): a colony of glowing green and yellow cells"></a>
<a href="https://instagram.com/spectrometry.mp4"><img src="docs/media/cow-launch.webp" width="32%" alt="Cow Launch (not public yet): a cow riding a rocket plume into the sky"></a>
</p>

Seen a video that isn't on either list?
[Ask which engine made it](https://github.com/ec175/spectrometry_public/issues/new?template=which-engine.yml).

## Try one in five minutes

You need **Python 3.10 or newer** and **[ffmpeg](https://ffmpeg.org/download.html)**. A GPU
makes things faster, but nothing needs one.

```bash
git clone https://github.com/ec175/spectrometry_public.git
cd spectrometry_public
python -m venv .venv
```

Turn the environment on. On Windows that's `.venv\Scripts\activate`; on macOS or Linux it's
`source .venv/bin/activate`. Then pick one:

**🌪️ Render one of my scenes.** Every engine has a `cli.py`. This is the wind tunnel clip from
the top of this page, at preview size:

```bash
pip install -r engines/wind-tunnel/src/requirements.txt
cd engines/wind-tunnel/src
python cli.py list
python cli.py tri_foil_air_fast --preview
```

Your video lands in `engines/wind-tunnel/src/out/`. Drop `--preview` for the full 1080×1920
version. Each [engine page](engines/README.md) has a **Render a video** section with its own
commands.

**🎬 Make a molecule video.** [Chemical Profiles](chemical_profiles/README.md) is a finished manim
scene: one molecule and its spectra.

```bash
pip install -r chemical_profiles/requirements.txt
cd chemical_profiles
manim -ql -r 540,960 chemical_profile.py VerticalProfile_Aspirin
```

**📺 Put the CRT look on your own clip.** The [Oscilloscope](engines/oscilloscope/README.md) film
filter works on any video you already have.

```bash
pip install -r engines/oscilloscope/src/requirements.txt
python engines/oscilloscope/src/filter_cli.py your_clip.mp4 filmed.mp4 --glitch 0.8
```

**🧪 Play with a simulation.** Every engine page starts with a short **Start here** section you
can paste straight into Python. [Wind Tunnel](engines/wind-tunnel/README.md#start-here) is a
good first one.

## Use it in your own stuff

**Anyone can use this code**, for videos, school projects, your own engine, whatever you like.
It's [MIT licensed](LICENSE). The only thing I ask is that you **leave a trail back here**:

1. **Keep the two-line header** at the top of any file you copy. The license needs it anyway.
2. **Credit it** in your project's README or your video's description:
   ```
   Made with spectrometry.mp4 engines by Clancy Earl - github.com/ec175/spectrometry_public
   ```
3. **Or add the badge:**
   [![made with spectrometry.mp4 engines](https://img.shields.io/badge/made%20with-spectrometry.mp4%20engines-ffc440?labelColor=0b0f14)](https://github.com/ec175/spectrometry_public)
   ```markdown
   [![made with spectrometry.mp4 engines](https://img.shields.io/badge/made%20with-spectrometry.mp4%20engines-ffc440?labelColor=0b0f14)](https://github.com/ec175/spectrometry_public)
   ```

If you post something you made, tag [@spectrometry.mp4](https://instagram.com/spectrometry.mp4).
I'd love to see it.

> **Using an AI coding agent?** Point it at [`AGENTS.md`](AGENTS.md). It explains how the repo
> fits together and how to credit it. For a formal citation, use
> [`CITATION.cff`](CITATION.cff) (on desktop, it's the **Cite this repository** button).

## Questions

<details>
<summary><b>Do I need a GPU?</b></summary>
<br>

No. Every engine runs on a normal CPU. If you have an NVIDIA card, the engines that support it
use it automatically for the maths (CuPy) and the video encoding (NVENC), and quietly fall back to
the CPU if anything is missing. All the timings in these docs came from one RTX 2060 with an
i7-9700K, so read them as rough ratios.

</details>

<details>
<summary><b>Do I need LaTeX?</b></summary>
<br>

No. Most manim projects do, but the chemical profile scenes use plain text rendering (Pango)
everywhere, so you can skip the whole LaTeX install.

</details>

<details>
<summary><b>Can I re-make your exact videos?</b></summary>
<br>

Most of them, yes. The scene scripts for 78 of my compositions are in the engines' `src/` folders,
and `python cli.py list` shows them. Some are held back: scenes built on photos or music I
don't own, the missile and ammunition studies, sets that aren't finished, and the "not public
yet" engines above. Each
engine's catalogue marks which scenes ship. My finished videos also have music and edits that
aren't part of the engines.

</details>

<details>
<summary><b>Can I make horizontal video?</b></summary>
<br>

Yes. Everything is written for vertical 1080×1920 because that's what I post, but the aspect
ratio lives in each engine's `RenderConfig`, never in the physics. Wind Tunnel even has a
`right` flow direction for the classic landscape tunnel.

</details>

<details>
<summary><b>Are the spectra real measurements?</b></summary>
<br>

No. They're representative simulations: peak positions come from literature values and
group-contribution estimates, so they're good for learning and for video, but not for citing.
Check against a real reference before you trust any band.

</details>

<details>
<summary><b>Something broke. What now?</b></summary>
<br>

[Open an issue](https://github.com/ec175/spectrometry_public/issues/new?template=bug.yml) with
the command you ran and the full error. Most problems are either ffmpeg missing from your PATH or
two engines sharing one virtual environment (use one per engine; Academia needs `numpy<2`).

</details>

<details>
<summary><b>What's in each folder?</b></summary>
<br>

```
engines/             one folder per engine: README, catalogue, stills, source
chemical_profiles/   a finished manim scene, the quickest way to a full video
docs/                the website, plus the preview clips on this page
tools/               rebuilds the catalogues and the stills pages
AGENTS.md            instructions for AI coding agents
```

</details>

## More

- **[All engines](engines/README.md)**: every engine on one page, plus the rules they all follow
- **[Full catalogue](engines/CATALOG.md)**: 193 objects and 83 compositions, indexed
- **[Chemical Profiles guide](chemical_profiles/GUIDE.md)**: from install to your own molecule
- **[Website](https://ec175.github.io/spectrometry_public/)**: the same gallery, as a page you can share

<div align="center">
<br>
<sub>Made by Clancy Earl · <a href="https://instagram.com/spectrometry.mp4">@spectrometry.mp4</a> · <a href="LICENSE">MIT</a></sub>
</div>
