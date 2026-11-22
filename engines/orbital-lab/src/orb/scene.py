# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""scene.py - the choreography, the electron events, the camera, and the SIZE GUARD.

Everything here is a closed-form function of `t`. There is no integrator and no persistent
state in the point cloud, which is the one structural difference from `..\\Particle_Fields\\`
(whose `sim.py` clock exists precisely because ITS points have history). The only stateful
thing in this engine is the canvas persistence buffer, and it is a pure decay.

THE THREE THINGS THE BRIEF ASKED FOR, and where each lives
----------------------------------------------------------
1. "reuse the orbital choreography"  -> `blend()`: a hold-and-morph walk over
   `harmonics.STATES`, interpolating the NORMALISED |Y_lm| magnitudes exactly the way the
   source's non-integer l/m readout says it does.
2. "orbitals get too large for the screen - avoid this" -> `fit_scales()`, a PRE-PASS over the
   whole timeline that solves, per frame, the world scale at which the projected cloud exactly
   fills the guard box, then takes a SOFT `min(nominal, fit)`. It is a clamp, not a
   framing: on most frames the nominal size is smaller and the guard does nothing, so shapes
   keep their honest relative sizes (a p_z really is twice as tall as it is wide) and only the
   sprawling high-l states are pulled back.
3. "random points light up inside the cloud, and the cloud converges to that point" ->
   `Electron`. The lit point is sampled on the mesh with probability proportional to |psi|^2,
   i.e. it lights up where the electron is actually LIKELY to be, and while it is lit the
   whole cloud is pulled toward it (a uniform term) with an extra local pucker (a gaussian
   term), so it reads as an attractor rather than a global shrink.

   ⚠ THAT THIRD ASK WAS WITHDRAWN (2026-09-01). Deforming the orbital around a lit electron
   makes the shape a function of a random event, and the shape IS the subject: a viewer cannot
   tell a d_xz from a d_xz being pulled sideways. `ELECTRONS` is now a three-way mode and the
   module default is `"off"`, so the flagship walk renders the harmonics and nothing else.
   The 15 registry scenes each carry `electrons="pull"` explicitly, which keeps the catalogue
   that was gated on 2026-08-29 byte-for-byte what it was.

   | mode | events planned | cloud deformed | lit point drawn |
   |---|---|---|---|
   | `"off"`  (default) | no  | no  | no  |
   | `"lit"`            | yes | **no** | yes — the flash without the pull |
   | `"pull"`           | yes | yes | yes — the original behaviour |

WARNING: the size guard is computed WITHOUT the electron displacement, on purpose. Convergence
only ever moves points toward a point that is already inside the cloud, so it can only shrink
the silhouette - measuring the undisplaced cloud keeps the guard conservative and, more
usefully, keeps the frame's size from twitching every time an electron fires. A corollary that
matters now: turning electrons off changes NOTHING about `fit_scales`, so the framing of the
electron-free clip is identical to the framing of the one that shipped.
"""
from __future__ import annotations

import math

import numpy as np

from . import flow as F
from . import harmonics as H
from . import sync as SY

DURATION = 20.0
FPS = 60

#: Fraction of a transition spent HELD on the state rather than morphing. The source holds
#: each named orbital visibly before moving on; without a hold the clip is one continuous
#: smear and no individual orbital ever reads.
HOLD = 0.34

#: World scale, in REFERENCE px, for a state whose normalised |Y| reaches 1. Sized so the `s`
#: sphere covers ~62% of the frame width at the nominal camera distance, which is what the
#: source's opening frame measures.
R_NOMINAL = 205.0

#: The guard's box, as a fraction of the half-frame.
#:
#: ⚠ These are a SAFETY NET, not the framing. Set too tight (0.46 on the first pass) the guard
#: binds on every frame, and it stops being a clamp and starts being an auto-zoom: the `s`
#: sphere came out 46% of the frame width against the source's measured 60%, and every shape
#: got the same width, which is exactly the honest size difference this engine is supposed to
#: keep. 0.86/0.88 leaves a real margin and binds only on the states that actually sprawl.
FIT_X = 0.86
FIT_Y = 0.88
#: Sharpness of the soft minimum in `fit_scales`. Higher = closer to a true min (and a
#: sharper elbow); 8 keeps the elbow inside a few frames while staying smooth.
SOFTMIN_P = 8.0

#: The FRAMING box - what a state would fill if it were fitted to the frame - and how much of
#: that fit is actually applied. See the long note in `fit_scales`: 0 is "every state at the
#: same world radius" (high-l states shrink away), 1 is "every state the same footprint on
#: screen" (an auto-zoom that flattens the real shape differences). 0.55 keeps both readable.
FRAME_X = 0.74
FRAME_Y = 0.76
FIT_BETA = 0.55

#: Camera. `depth_ref` is handed to the canvas so its additive depth cue is centred on the
#: object rather than on Particle_Fields' 1600 default.
CAM_DIST = 1700.0
CAM_FOV = 38.0


# ======================================================================================
# THE SCENE REGISTRY (2026-08-29)
#
# This engine shipped as a SINGLE CLIP: every constant above is a module global and
# `render.py` reads them directly. That is the right shape for one deliverable and the wrong
# one for a catalogue, so rather than rewrite the module into a class — which would move every
# line `render.py` depends on — a scene is a dict of OVERRIDES applied to these globals before
# anything is built.
#
# ⚠ APPLY BEFORE ANY BUILD. `build_mesh`, `plan_electrons` and `fit_scales` all read the
# globals at call time, and `render.py` calls them in that order, so `configure()` has to run
# first. `cli.py` does it immediately after parsing. Mutating module state is safe here for one
# specific reason: one process renders one clip. If that ever stops being true this needs to
# become a class.
#
# ⚠ `STATES` LIVES IN `harmonics`, not here, and the mesh is built from it — so a scene that
# changes the state list is changing what `build_mesh` returns, not just what `blend` walks
# through. That is why it is overridden through this function rather than by a caller poking
# `H.STATES` afterwards.
# ======================================================================================
_S = (0, 0, "s")
_P = [(1, 0, "p_z"), (1, 1, "p_x")]
_D = [(2, 0, "d_z2"), (2, 1, "d_xz"), (2, 2, "d_xy")]
_F = [(3, 0, "f_z3"), (3, 1, "f_xz2"), (3, 2, "f_xyz"), (3, 3, "f_x3")]
_G = [(4, 0, "g_z4"), (4, 2, "g_xz"), (4, 4, "g_xy")]

#: slug -> overrides. Only the keys a scene actually changes.
#:
#: ⚠ EVERY ENTRY SPELLS OUT `electrons=`, and none of them inherits the module default. These
#: 15 were gated at 3840x2160 on 2026-08-29 WITH the convergence on, and for six of them the
#: electron knobs (`rate`, `dur_lo/hi`) are half of what the scene IS - `qm_electron_jump` and
#: `qm_probability` are about the electron, not about the harmonic. Moving the module default
#: to "off" therefore had to leave the table alone, and the only way to be sure of that is for
#: the table to say so rather than to depend on a global it does not name.
#:
#: `qm_orbitals` is the 2026-09-01 addition and the odd one out: it is the flagship walk with
#: electrons OFF, which is the clip this project is actually for.
SCENES = {
 # THE FLAGSHIP: the plain harmonic walk, no electrons, with the grid flowing over it.
 "qm_orbitals":      dict(electrons="off", flow="weave"),
 # THE 2026-09-01 VARIATIONS - all three in DOT mode, all the same choreography and framing.
 "qm_dots":          dict(electrons="off", flow="porous_dots", mesh="dots",
                          seconds=15.0, loop=True),
 "qm_emerge":        dict(electrons="off", flow="emerge", mesh="dots",
                          seconds=15.0, loop=True),
 # ⚠ `void_mesh` 0.15 is a DIVISION OF LABOUR, not hedging: the field's ABSENCE reveals the
 # silhouette, and the residual wireframe says WHICH ORBITAL it is. Set it to 0 for the pure
 # inversion. `void` 20 goes with the FRAME-FILLING lattice at pitch 5.8: the gain and the
 # point count move together, so changing either without the other changes the frame level.
 # This one is deliberately bright - the ask was for the WHOLE screen to carry the effect.
 "qm_negative":      dict(electrons="off", flow="emerge", mesh="dots",
                          seconds=15.0, loop=True, void=20.0, void_mesh=0.15),
 # THE MUSIC SET - 2026-09-16. `qm_negative`'s look with the walk and a two-tone vibration
 # driven by a song. See MUSIC_PLAN.md and `orb/sync.py`. `seconds` and `states` come FROM the
 # session file. ⚠ Does NOT loop (a clip plays once; a song section is not periodic), and the
 # camera rate is stated: canon sweeps 1 turn per 15 s, so 24 s at the same speed is 1.6.
 "music_4morant":    dict(electrons="off", flow="emerge", mesh="dots", loop=False,
                          void=20.0, void_mesh=0.15, turns=1.6, el_cyc=1.6, size=1.10, react=False,
                          sync="audio/4morant.proposal.session.json"),
 # ⚠ THE MUSIC TEMPLATE (Clancy, 2026-09-16: "this will be the music template for now").
 # A new song starts by copying this entry - see development notes.
 # THE BLOOM VARIANT - 2026-09-16. Same clip, echo shells of the orbital filling the field.
 # See `void._bloom` / `sync.BLOOM_LAYERS`. The real orbital is 20% larger than canon.
 "music_4morant_bloom": dict(electrons="off", flow="emerge", mesh="dots", loop=False,
                          void=20.0, void_mesh=0.15, turns=1.6, el_cyc=1.6, size=1.20,
                          react=False, bloom=True, core=True, hq=True,
                          sync="audio/4morant.proposal.session.json"),
 # THE MOLECULE SET - 2026-09-02c. Variation C's format over a whole assembly; the hole is
 # TOTAL ELECTRON DENSITY rather than a signed MO. See `orb/molecule.py`, and run
 # `cli.py moltest` before either of these renders - `dry` cannot see a wrong reaction.
 "mol_water":        dict(electrons="off", flow="none", mesh="cloud", seconds=10.0,
                          mol="water_network", states=[_S, _S], fit_beta=0.0,
                          void=6.3, void_mesh=0.45, mesh_gain=0.5,
                          turns=0.10, el_cyc=0.5, az0=0.0, el_deg=(3.0, 2.2), nuc_gain=0.0),
 # ⚠ THE CRYSTALLIZATION FINAL. `mol_fill` INVERTS the framing rule - the sheet must exceed
 # the frame, not fit inside it - and `render.dry` inverts its overflow check to match.
 # `nuc_gain=0`: a sheet of fifty waters has no one nucleus. `mesh_gain` is down because 144
 # atoms deposit far more total flux than 21 did at the same per-dot gain.
 "mol_ice":          dict(electrons="off", flow="none", mesh="cloud", seconds=10.0,
                          mol="crystallization", states=[_S, _S], fit_beta=0.0,
                          void=5.2, void_mesh=0.45, mesh_gain=0.62, mol_fill=1.10,
                          turns=0.06, el_cyc=0.5, az0=0.0, el_deg=(2.0, 1.4),
                          nuc_gain=0.0),
 "mol_burn":         dict(electrons="off", flow="none", mesh="cloud", seconds=10.0,
                          mol="combustion", states=[_S, _S], fit_beta=0.0,
                          void=6.3, void_mesh=0.45, mesh_gain=0.5,
                          turns=0.10, el_cyc=0.5, az0=0.0, el_deg=(3.0, 2.2), nuc_gain=0.0),
 # THE BOND SET - 2026-09-02. Variation C's format with a TWO-CENTRE hole. See `orb/bond.py`.
 #
 # ⚠ EVERY ONE OF THESE IS THE SAME CLIP WITH ONE SIGN CHANGED, WHICH IS THE POINT AND ALSO
 # THE HAZARD. sigma and sigma* are the identical two atoms at the identical separation; only
 # `phase_b` differs, and NO framing, exposure, motion or coverage statistic can tell them
 # apart. `cli.py bondtest` is what does - run it before any of these render.
 #
 # ⚠ AND THE SIGN OF "BONDING" IS NOT A CONSTANT. For s+s (below) constructive is `+`; for two
 # p_z head-on it is `-`, because the lobes that face each other across the bond are that
 # atom's +z lobe and this atom's -z lobe. Spelled out per scene rather than inferred.
 #
 # `r_nominal` is down from 205 and `fit_beta` is 0: with the atoms 3.2 R apart the pair spans
 # 5.2 R, and at the shipped size the guard would bind on every frame - which per S5 stops
 # being a clamp and becomes an auto-zoom, so the atoms would appear to grow as they approach
 # and the viewer could not tell a size change from an approach. Sized to fit at full
 # separation instead, so R is CONSTANT and the only thing that moves is the separation.
 "bond_sigma_s":     dict(electrons="off", flow="none", mesh="cloud", seconds=5.0, mesh_gain=0.5,
                          void=7.0, void_mesh=0.45, r_nominal=170.0, fit_beta=0.0,
                          turns=0.18, el_cyc=0.35, states=[_S, _S],
                          bond=dict(state=(0, 0), phase_a=+1, phase_b=+1, kind="bonding")),
 "bond_sigma_star_s": dict(electrons="off", flow="none", mesh="cloud", seconds=5.0, mesh_gain=0.5,
                          void=7.0, void_mesh=0.45, r_nominal=170.0, fit_beta=0.0,
                          turns=0.18, el_cyc=0.35, states=[_S, _S],
                          bond=dict(state=(0, 0), phase_a=+1, phase_b=-1,
                                    kind="antibonding")),
 # HEAD-ON p_z. ⚠ phase_b = -1 IS THE BONDING ONE HERE. See the note above.
 "bond_sigma_p":     dict(electrons="off", flow="none", mesh="cloud", seconds=5.0, mesh_gain=0.5,
                          void=7.0, void_mesh=0.45, r_nominal=170.0, fit_beta=0.0,
                          turns=0.18, el_cyc=0.35, states=[_P[0], _P[0]], nuc_gain=0.13,
                          sep_near=1.7,
                          bond=dict(state=(1, 0), phase_a=+1, phase_b=-1, kind="bonding")),
 # SIDE-BY-SIDE p_x -> pi. ⚠ THE AZIMUTH IS CENTRED ON WORLD X and the sweep is short: the
 # void card contains the camera's right vector, and a p_x pair has nothing to show on the Z
 # axis. `az0 = -0.55` with `turns = 0.175` keeps |cos(az)| >= 0.85 for the whole clip.
 "bond_pi_p":        dict(electrons="off", flow="none", mesh="cloud", seconds=5.0, mesh_gain=0.5,
                          void=7.0, void_mesh=0.45, r_nominal=195.0, fit_beta=0.0,
                          turns=0.175, el_cyc=0.35, az0=-0.55, nuc_gain=0.13, states=[_P[1], _P[1]],
                          sep_far=2.6, sep_near=0.90,
                          bond=dict(state=(1, 1), phase_a=+1, phase_b=+1, kind="bonding")),
 "bond_pi_star_p":   dict(electrons="off", flow="none", mesh="cloud", seconds=5.0, mesh_gain=0.5,
                          void=7.0, void_mesh=0.45, r_nominal=195.0, fit_beta=0.0,
                          turns=0.175, el_cyc=0.35, az0=-0.55, nuc_gain=0.13, states=[_P[1], _P[1]],
                          sep_far=2.6, sep_near=0.90,
                          bond=dict(state=(1, 1), phase_a=+1, phase_b=-1,
                                    kind="antibonding")),
 "qm_s_to_p":        dict(states=[_S] + _P + [_S], hold=0.42, electrons="pull"),
 "qm_p_to_d":        dict(states=_P + _D + [_P[0]], hold=0.40, electrons="pull"),
 "qm_d_to_f":        dict(states=_D + _F + [_D[0]], hold=0.36, electrons="pull"),
 "qm_f_to_g":        dict(states=_F + _G + [_F[0]], hold=0.36, electrons="pull"),
 "qm_full_morph":    dict(electrons="pull"),       # the shipped 14-state walk, at 30 s
 "qm_cloud_condense": dict(states=[_G[2], _F[3], _D[2], _P[1], _S], hold=0.30,
                           rate=2.1, electrons="pull"),
 "qm_shells":        dict(states=[_S, _P[0], _D[0], _F[0], _G[0], _S], hold=0.44,
                          electrons="pull"),
 "qm_spin_flip":     dict(states=[_P[0], _P[1], _P[0], _P[1], _P[0]], hold=0.30,
                          rate=2.4, electrons="pull"),
 "qm_orbital_s":     dict(states=[_S, _P[0], _S], hold=0.62, rate=0.9, electrons="pull"),
 "qm_orbital_p":     dict(states=[_P[0], _P[1], _P[0]], hold=0.58, electrons="pull"),
 "qm_orbital_d":     dict(states=_D + [_D[0]], hold=0.52, electrons="pull"),
 "qm_orbital_f":     dict(states=_F + [_F[0]], hold=0.48, electrons="pull"),
 "qm_electron_jump": dict(states=[_S, _P[0], _D[0], _P[0], _S], hold=0.34, rate=3.2,
                          dur_lo=0.26, dur_hi=0.44, electrons="pull"),
 "qm_probability":   dict(states=[_D[1], _D[2], _F[2], _D[1]], hold=0.40, rate=4.0,
                          dur_lo=0.30, dur_hi=0.52, electrons="pull"),
 "qm_hybrid_sp3":    dict(states=[_S, _P[0], _P[1], _S, _P[0]], hold=0.26, rate=1.6,
                          electrons="pull"),
}

#: electron-planning knobs, overridable per scene; `render.py` reads them through `plan()`
RATE, DUR_LO, DUR_HI = 1.30, 0.42, 0.72

#: THE ELECTRON MODE - "off" | "lit" | "pull". See the table in the module docstring.
#:
#: ⚠ THE DEFAULT IS "off", and that is the 2026-09-01 change. `plan()` returns an empty event
#: list in that mode, so `render.points_at` has nothing to iterate and the per-frame cost drops
#: to the mesh build alone. Every entry in `SCENES` sets `electrons="pull"` explicitly rather
#: than inheriting, so the gated catalogue is unaffected by this default moving.
#:
#: ⚠ AND IT IS A MODULE GLOBAL RESOLVED AT CALL TIME, never a default argument - the same trap
#: `blend` documents. `render.Orbital.points_at` reads `S.ELECTRONS` on every frame.
ELECTRONS = "off"
_MODES = ("off", "lit", "pull")

#: The colour grade a sync scene carries from its session (`orb/grade.py`), or None = canon.
GRADE = None

#: THE FLOW PRESET - a name from `flow.FLOWS`, or a spec dict, or None. See `orb/flow.py`.
#:
#: ⚠ THE DEFAULT IS "none" AND THAT IS LOAD-BEARING: it means the 15 scenes gated on
#: 2026-08-29 inherit exactly the behaviour they were gated with, without any of them naming a
#: flow. Unlike `ELECTRONS`, no re-pinning pass was needed - the default IS their look. Only
#: `qm_orbitals` names one.
FLOW = "none"

#: THE MESH MODE - "lines" | "dots". Default "lines", so the 15 gated scenes are untouched.
#:
#: Clancy, 2026-09-01: *"i was kind of hoping that this would look almost like thousands of
#: 'dots' with lots concentrated at the shell edge (just so we can see the structure). the
#: connections between the nodes is really not needed ... so way more nodes, no connections.
#: the nodes will likely have to glow."*
#:
#: "Nodes without connections" is a LATTICE: `n_mer=0` collapses the meridian family, and the
#: latitude family alone carries `MESH_ROWS x MESH_COLS` discrete points - the grid's
#: intersections, with nothing drawn between them.
#:
#: ⚠ THE RIM CONCENTRATION IS FREE, and it is the whole reason this reads as a shell. At the
#: LIMB the surface turns away from the camera, so a band of rows projects into a few pixels
#: and the dot density there rises on its own. It is the same mechanism that made the solid
#: outline the porosity work had to break up - as DOTS it stops being an artefact and becomes
#: the structure.
#: ⚠ `"cloud"` IS THE THIRD MODE AND THE ONLY ONE THAT IS NOT A SURFACE PARAMETERISATION.
#: `lines` and `dots` are both `r = |Y(theta, phi)|` - one radius per direction about one
#: origin - and a MOLECULAR orbital has neither one origin nor one crossing per direction. See
#: `bond.ShellCloud`: a fixed 3-D dot set, with the surface lighting the dots it passes through.
#: Requires `bond`, because there is nothing else for the surface to be.
MESH = "lines"
MESH_ROWS, MESH_COLS = 200, 400
_MESHES = ("lines", "dots", "cloud")

#: Scalar handed to `Canvas.deposit` for the mesh passes.
#:
#: ⚠ THIS IS WHY DOT MODE NEEDS ITS OWN NUMBER AND WHY IT IS NOT `cfg.ink`. As LINES the points
#: are sub-pixel spaced, so every pixel of a curve receives many deposits; as separated DOTS a
#: pixel receives about ONE. Same per-point gain therefore renders dots roughly an order of
#: magnitude dimmer, and dim dots never cross `bloom_threshold`, so they do not glow - which is
#: the one thing Clancy asked for by name. Raising `ink` instead would also lift the nucleus and
#: the electrons, re-looking parts that are already settled; `deposit(gain=...)` moves only the
#: wireframe.
MESH_GAIN = 1.0

#: VARIATION C: fill the unoccupied space and carve the orbital out of it. See `orb/void.py`.
#: 0 = off (the default, so nothing else in the library changes). Otherwise the deposit gain
#: for the field, which is deliberately a SEPARATE knob from `cfg.ink` - the wireframe's
#: exposure is already settled and must not move because the void arrived.
VOID = 0.0
#: What is left of the wireframe when the void is on. The field's absence reveals the
#: SILHOUETTE; this says WHICH ORBITAL it is. 0 removes it entirely.
VOID_MESH = 0.30

#: Per-scene override for the magenta core's deposit gain. None = `render.NUC_GAIN`.
#:
#: ⚠ EXISTS BECAUSE OF WHERE A p ORBITAL PUTS ITS NUCLEUS. The core is a fixed-brightness blob,
#: and how hot it renders depends on WHAT IS ALREADY UNDER IT. In a bond scene the void field is
#: carved away inside the orbital, so an s nucleus sits in a dark hole and reads as the soft
#: magenta core variation C shipped. A p orbital has ZERO amplitude along its own axis - which
#: in these scenes is the bond axis - so its nucleus sits on UNCARVED, bright field, the two
#: deposits stack, and the core clips to a white star. Measured on the shipped frames: the two
#: s scenes blow 0.0000% of the frame, the three p scenes blow 0.13-0.28%.
#:
#: This is S7's tone-curve lesson again - the peak accumulator has to sit near a ~ 2, not above
#: it - and it is a PER-SCENE fact, not a bond-wide one, so it is a per-scene knob. Cutting it
#: globally in bond mode would dim the two nuclei that are already right.
NUC_GAIN = None

#: THE BOND - a spec dict (see `orb/bond.py`), or None. When set, the void field's carve is
#: made by a TWO-CENTRE molecular field instead of by one `r = |Y|` surface, and the wireframe
#: is deposited TWICE, once per atom.
#:
#: ⚠ DEFAULT None, so nothing else in the library changes. The three shipped finals and the 15
#: gated stock scenes never see any of this - same discipline as `VOID`, `MESH` and `FLOW`
#: before it: the default IS the old behaviour, and the new scenes name what they want.
BOND = None

#: THE MOLECULAR SCENE - a name from `molecule.SCENES`, or None. Mutually exclusive with
#: `BOND`: `bond.py` is a SIGNED two-centre wavefunction and `molecule.py` is TOTAL DENSITY over
#: many atoms, and they answer "is this point inside?" differently on purpose.
MOL = None
#: World pixels per Angstrom. None = solved from the trajectory, which is what every scene does.
MOL_SCALE = None
#: FILL MODE: the subject must EXCEED the frame by this factor, rather than fit inside it.
#: None = the ordinary behaviour. See `mol_scale` and the inverted check in `render.dry`.
MOL_FILL = None

#: How much of the per-frame fit the push-in keeps. 0 = a constant scale, 1 = a full
#: auto-zoom. 0.88 after Clancy: *"Zoom in a bit, most of the frame should be occupied by a
#: molecule or the area around it."* High on purpose - these scenes change size by 2x over the
#: clip, so a low beta spends most of the frame on the widest instant and leaves the rest small.
MOL_BETA = 0.88

#: Separation between the two nuclei, IN UNITS OF R, at the start and at the end of the
#: approach - and the fraction of the clip the approach occupies.
#:
#: ⚠ IN UNITS OF R, NOT IN WORLD PX, AND THAT IS WHAT KEEPS `fit_scales` SOLVABLE. Its fit is
#: a fixed-point iteration `R <- R / ratio`, which converges only because the whole
#: configuration scales with `R`. A separation quoted in world units would be an additive term
#: the iteration does not see, and the guard would settle on the wrong scale without failing.
SEP_FAR, SEP_NEAR = 3.2, 1.5
#: Hold apart, approach, hold bonded. The longest hold is on the BONDED state - that is the
#: thing the clip is about, and a 5 s clip has no room to be even-handed about it.
SEP_IN, SEP_OUT = 0.14, 0.66

#: Camera overrides. None = the shipped behaviour (0.62 turns, or a whole turn under LOOP).
#:
#: ⚠ THE CAMERA'S RATE IS PER CLIP, NOT PER SECOND - `camera_at` works in `u = t / DURATION` -
#: so a 5 s clip performs the whole 0.62-turn sweep in 5 s instead of 20, four times faster,
#: and the demo whips round. Any scene that shortens `seconds` has to say what it wants the
#: camera to do or it inherits a rate that was chosen for a different duration.
#:
#: ⚠ AND A NON-INTEGER `turns` CANNOT LOOP. Under `LOOP` these must be whole numbers; the
#: bond scenes do not loop, which is why they are free to pick a fraction.
TURNS = None
EL_CYC = None
#: Starting azimuth. Irrelevant for an axisymmetric (m=0) scene and load-bearing for m>0: the
#: void card is a plane containing world +Y and the camera's own RIGHT vector, and that vector
#: sits at world azimuth `-az`. A pi bond made of p_x orbitals has its lobes on world +/-X, so
#: an azimuth sweep that carries the card round to the Z axis slices the field where its
#: amplitude is zero and the hole disappears. The pi scenes centre their sweep on X instead.
AZ0 = 0.55
#: Camera elevation, as (base, breath) in degrees.
#:
#: ⚠ THE TILT DECIDES HOW MUCH OF A PLANAR SCENE MEETS THE VOID CARD, which is not obvious
#: until it bites. The card is a thin slab perpendicular to the view; at 13 degrees of
#: elevation a molecule 2.8 A out along the ring sits 0.62 A off that plane, against a slab
#: half-depth of ~0.33 A - so it falls outside, gets no carved hole, and renders as a soft blob
#: while the two molecules near the origin get variation C's rimmed silhouette. Five of seven
#: waters looked like a different treatment from the other two. A planar scene wants a shallow
#: tilt; a compact one does not care.
EL_DEG = (13.0, 7.0)

#: LOOP MODE. When on, every rate in the clip is made periodic over `DURATION` so the last
#: frame hands off to the first with no step. Default OFF, so the 15 gated scenes keep the
#: camera move they were gated with.
#:
#: ⚠ THE CAMERA IS THE HALF OF THIS THAT IS NOT IN `flow.loop_snap`. Its azimuth sweep and its
#: elevation breath are both non-integer fractions of a turn by design (see `camera_at`), which
#: is exactly what a loop cannot have.
LOOP = False


#: Import-time values of the framing numbers `size` scales, so a second `configure()` in the
#: same process (the viewer server does this) cannot compound the scaling.
_BASE_FRAME = None


def configure(name=None, seconds=None, **over):
    """Point the module at one scene. Returns the resolved override dict."""
    global DURATION, HOLD, RATE, DUR_LO, DUR_HI, R_NOMINAL, CAM_DIST, ELECTRONS, FLOW
    global FIT_BETA
    global MESH, MESH_ROWS, MESH_COLS, MESH_GAIN, VOID, VOID_MESH, LOOP
    global BOND, SEP_FAR, SEP_NEAR, TURNS, EL_CYC, AZ0, NUC_GAIN, MOL, MOL_SCALE, EL_DEG
    global MOL_FILL, GRADE, FRAME_X, FRAME_Y, FIT_X, FIT_Y
    spec = dict(SCENES[name]) if name else {}
    spec.update({k: v for k, v in over.items() if v is not None})
    if seconds:
        spec["seconds"] = seconds
    DURATION = float(spec.get("seconds", DURATION))
    HOLD = float(spec.get("hold", HOLD))
    RATE = float(spec.get("rate", RATE))
    DUR_LO = float(spec.get("dur_lo", DUR_LO))
    DUR_HI = float(spec.get("dur_hi", DUR_HI))
    R_NOMINAL = float(spec.get("r_nominal", R_NOMINAL))
    FIT_BETA = float(spec.get("fit_beta", FIT_BETA))
    # ⚠ `size` SCALES THE NOMINAL AND BOTH BOXES TOGETHER. Raising `r_nominal` alone moves
    # the size by only nominal^(1 - FIT_BETA) = 4% for a 10% ask, and the guard would then take
    # much of that back. The hard box is capped at 0.95 of the half-frame; `dry` still proves
    # nothing reaches the edge on every frame.
    global _BASE_FRAME
    if _BASE_FRAME is None:
        _BASE_FRAME = (FRAME_X, FRAME_Y, FIT_X, FIT_Y)
    sz = float(spec.get("size", 1.0))
    fx, fy, hx, hy = _BASE_FRAME
    FRAME_X, FRAME_Y = fx * sz, fy * sz
    FIT_X, FIT_Y = min(0.95, hx * sz), min(0.95, hy * sz)
    if sz != 1.0:
        R_NOMINAL = float(spec.get("r_nominal", 205.0)) * sz
    CAM_DIST = float(spec.get("cam_dist", CAM_DIST))
    ELECTRONS = str(spec.get("electrons", ELECTRONS)).lower()
    if ELECTRONS not in _MODES:
        raise ValueError(f"electrons={ELECTRONS!r}; expected one of {_MODES}")
    FLOW = spec.get("flow", FLOW)
    F.spec(FLOW)                      # validate NOW, not on the first frame of a render
    MESH = str(spec.get("mesh", MESH)).lower()
    if MESH not in _MESHES:
        raise ValueError(f"mesh={MESH!r}; expected one of {_MESHES}")
    MESH_ROWS = int(spec.get("mesh_rows", MESH_ROWS))
    MESH_COLS = int(spec.get("mesh_cols", MESH_COLS))
    MESH_GAIN = float(spec.get("mesh_gain", 2.6 if MESH == "dots" else 1.0))
    VOID = float(spec.get("void", VOID))
    VOID_MESH = float(spec.get("void_mesh", VOID_MESH))
    LOOP = bool(spec.get("loop", LOOP))
    BOND = spec.get("bond", BOND)
    SEP_FAR = float(spec.get("sep_far", SEP_FAR))
    SEP_NEAR = float(spec.get("sep_near", SEP_NEAR))
    TURNS = spec.get("turns", TURNS)
    EL_CYC = spec.get("el_cyc", EL_CYC)
    EL_DEG = tuple(spec.get("el_deg", EL_DEG))
    AZ0 = float(spec.get("az0", AZ0))
    NUC_GAIN = spec.get("nuc_gain", NUC_GAIN)
    MOL = spec.get("mol", MOL)
    MOL_SCALE = spec.get("mol_scale", MOL_SCALE)
    MOL_FILL = spec.get("mol_fill", MOL_FILL)
    if MOL is not None and BOND is not None:
        raise ValueError("a scene is either a bond (signed) or a molecule (density), not both")
    if MESH == "cloud" and BOND is None and MOL is None:
        raise ValueError("mesh='cloud' needs a bond or a molecule - it draws an isosurface")
    if BOND is not None and ELECTRONS != "off":
        # The electron term deforms and lights ONE cloud about the origin; with two translated
        # atoms it would pull both of them toward a point that belongs to neither.
        raise ValueError("bond scenes require electrons='off'")
    if spec.get("states"):
        H.STATES = list(spec["states"])
    # ⚠ LAST, because it OVERRIDES `DURATION` and `H.STATES` from the song's own file. A
    # `--seconds` on a sync scene would desynchronise picture from audio, so it is ignored.
    import os as _os
    sy = SY.activate(spec.get("sync"), _os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))), vib=spec.get("vib"), vib_gamma=spec.get("vib_gamma"),
        react=spec.get("react", True), bloom=spec.get("bloom", False),
        core=spec.get("core", False))
    from . import hq as _HQ
    _HQ.activate(spec.get("hq"))
    GRADE = None
    if sy is not None:
        DURATION = sy.duration
        GRADE = sy.grade
    return spec


def plan(amps, seed=20260825):
    """`plan_electrons` with the scene's own rate, or NO events at all in mode `"off"`.

    Returning `[]` rather than planning events and then ignoring them is deliberate: an empty
    list is the one state every consumer already handles (`points_at` iterates it, `dry` slices
    it, the flash-step warning never fires), and it means the RNG is not advanced - so a scene
    is reproducible from its seed whether or not electrons are on.
    """
    if ELECTRONS == "off":
        return []
    return plan_electrons(amps, seed=seed, duration=DURATION,
                          rate=RATE, dur_lo=DUR_LO, dur_hi=DUR_HI)


# --------------------------------------------------------------------------------------
# the choreography
# --------------------------------------------------------------------------------------
def blend(t: float, duration: float = None, n_states: int = None):
    """(i, j, w) - interpolate state i toward state j by w, with a hold at each state.

    ⚠ BOTH DEFAULTS ARE RESOLVED AT CALL TIME, and they have to be. Written as
    `duration=DURATION, n_states=len(H.STATES)` the values are bound once, when the module is
    imported — so a scene that shortens the state list or changes the duration is ignored by
    every call that does not pass them explicitly, and the failure is an IndexError deep in
    `lm_at` rather than anything that names the cause. That is the whole reason the registry
    could not simply overwrite the globals.
    """
    if SY.ACTIVE is not None and duration is None and n_states is None:
        return SY.ACTIVE.blend(t)
    duration = DURATION if duration is None else duration
    n_states = len(H.STATES) if n_states is None else n_states
    segs = max(n_states - 1, 1)
    u = min(max(t / duration, 0.0), 1.0) * segs
    i = min(int(u), segs - 1)
    f = u - i
    lo, hi = HOLD * 0.5, 1.0 - HOLD * 0.5
    if f <= lo:
        w = 0.0
    elif f >= hi:
        w = 1.0
    else:
        x = (f - lo) / (hi - lo)
        w = x * x * (3.0 - 2.0 * x)
    return i, i + 1, float(w)


def walk_u(t: float) -> float:
    """Clip progress as the split arc and the void crossfade see it. Canon: `t / DURATION`.
    Under a sync: the WALK's position on canon's clock (see `orb/sync.py`)."""
    if SY.ACTIVE is not None:
        return SY.ACTIVE.walk_u(t)
    return t / max(DURATION, 1e-9)


def sep_at(t: float) -> float:
    """Distance between the two nuclei at time `t`, IN UNITS OF R. 0 when there is no bond.

    Hold apart, approach, hold bonded - and nothing else in the clip changes, which is the
    point. The state does not morph, the exposure does not ramp, the field's total flux is
    static by construction (`void.py`); the ONLY thing that happens in five seconds is that two
    orbitals get close enough to interfere. That is the whole subject.
    """
    if BOND is None:
        return 0.0
    u = min(max(t / max(DURATION, 1e-9), 0.0), 1.0)
    return SEP_FAR + (SEP_NEAR - SEP_FAR) * F._smoothstep(u, SEP_IN, SEP_OUT)


def lm_at(t: float):
    """The (l, m) pair the source would have PRINTED at this instant. Nothing renders it -
    the brief says no text - but the dry gate prints it, and it is the clearest single
    statement of what the choreography is doing at a given second."""
    i, j, w = blend(t)
    li, mi, _ = H.STATES[i]
    lj, mj, _ = H.STATES[j]
    return (1 - w) * li + w * lj, (1 - w) * mi + w * mj


# --------------------------------------------------------------------------------------
# the camera
# --------------------------------------------------------------------------------------
def camera_at(t: float, ref=(1080, 1920)):
    """A slow orbit about the vertical axis with a gentle elevation breath.

    vdx measured the source at `drifting_camera`, 63.6 px/s of phase-correlation drift on a
    716-wide frame (= 96 px/s referred to 1080). A 0.62-turn sweep over 20 s puts the equator
    of a 205 px-radius object at ~40 px/s, and the rest of the source's measured drift is the
    morph itself, not the camera. Deliberately under the measurement rather than over it: the
    brief's subject is the shape change, and vdx cannot separate the two.
    """
    from ._pf import Camera
    u = t / DURATION
    # ⚠ IN LOOP MODE BOTH RATES BECOME WHOLE CYCLES. 0.62 of a turn and 0.75 of an elevation
    # cycle were chosen to stay under the source's measured drift on a clip that played once;
    # on a clip that repeats they are the seam. One full turn over the clip is close to the
    # same on-screen speed at 15 s as 0.62 of a turn was at 20 s, so the look barely moves.
    turns = float(TURNS) if TURNS is not None else (1.0 if LOOP else 0.62)
    el_cyc = float(EL_CYC) if EL_CYC is not None else (1.0 if LOOP else 0.75)
    az = AZ0 + 2.0 * math.pi * turns * u
    el = math.radians(EL_DEG[0] + EL_DEG[1]
                      * math.sin(2.0 * math.pi * (el_cyc * u + 0.12)))
    d = CAM_DIST
    eye = (d * math.cos(el) * math.sin(az), d * math.sin(el), d * math.cos(el) * math.cos(az))
    cam = Camera(eye, (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=CAM_FOV, ref=ref)
    cam.depth_ref = d
    return cam


# --------------------------------------------------------------------------------------
# the electrons
# --------------------------------------------------------------------------------------
class Electron:
    """One lit point: a mesh index, a window in time, and the pull it exerts while lit."""

    __slots__ = ("t0", "dur", "idx", "amp", "peak")

    def __init__(self, t0, dur, idx, amp):
        self.t0, self.dur, self.idx, self.amp = t0, dur, idx, amp
        self.peak = t0 + dur * 0.5

    def env(self, t: float) -> float:
        """0 outside the window, a soft flash inside it.

        `sin(pi x) ** 1.5` rather than a triangle: the cloud has to arrive at the point and
        leave it with zero velocity or the release reads as a snap, which is exactly the
        'velocity jump at a join' every gate in this library hard-fails.
        """
        x = (t - self.t0) / self.dur
        if x <= 0.0 or x >= 1.0:
            return 0.0
        return self.amp * math.sin(math.pi * x) ** 1.5


_MESH_TH = None          # set by build_mesh(); plan_electrons needs the polar angles


def build_mesh():
    """(theta, phi, dirs, amps, shape) - the static side of the engine, computed once on CPU.

    ⚠ `shape` is COPIED out of `harmonics.SHAPE` here and not read from the module later.
    `fit_scales` builds a second, coarser mesh through the same function, which overwrites the
    module-level dict - so anything that reads it lazily gets the coarse layout and reshapes
    160k points as if they were 8k.
    """
    global _MESH_TH
    # ⚠ IN CLOUD MODE THIS LATTICE IS NOT DRAWN. It is still built, cheaply, because
    # `plan_electrons` reads `_MESH_TH` and `fit_scales` wants an amplitude table - and building
    # the full 160k line mesh for something nothing renders is pure waste.
    th, ph = (H.curve_mesh(n_mer=0, n_th=0, n_lat=MESH_ROWS, n_ph=MESH_COLS)
              if MESH in ("dots", "cloud") else H.curve_mesh())
    _MESH_TH = th
    return th, ph, H.directions(th, ph), H.amplitudes(th, ph), dict(H.SHAPE)


def plan_electrons(amps: np.ndarray, seed=20260825, duration=None,
                   rate=1.30, dur_lo=0.42, dur_hi=0.72):
    """Sample the whole clip's electron events up front.

    The lit point is drawn from |psi|^2 over the mesh AT THE EVENT'S OWN START TIME, which is
    the honest thing to do and also the good-looking one: it lights up inside a lobe, where
    the cloud is dense, instead of on a node where nothing is.

    WARNING: weighting by r^2 alone would bias hard toward the two poles, where the meridian
    family piles 112 curves into a shrinking circle - the mesh is denser there than the
    physics is. `sin(theta)` is the metric factor that cancels it.
    """
    duration = DURATION if duration is None else duration   # resolved at CALL time; see blend
    rng = np.random.default_rng(seed)
    st = np.sin(np.asarray(_MESH_TH, np.float64))
    out, t = [], 0.35
    while t < duration - 0.55:
        d = float(rng.uniform(dur_lo, dur_hi))
        i, j, w = blend(t)
        r = (1.0 - w) * amps[i] + w * amps[j]
        p = (r.astype(np.float64) ** 2) * st
        s = p.sum()
        idx = int(rng.integers(0, p.size)) if s <= 0 else int(rng.choice(p.size, p=p / s))
        out.append(Electron(t, d, idx, float(rng.uniform(0.72, 1.0))))
        t += float(rng.exponential(1.0 / rate)) + 0.18
    return out


# --------------------------------------------------------------------------------------
# the size guard
# --------------------------------------------------------------------------------------
def _coarse_mesh():
    th, ph = H.curve_mesh(n_mer=48, n_th=96, n_lat=32, n_ph=128)
    return th, ph, H.directions(th, ph), H.amplitudes(th, ph)


def mol_scale(sc, n_frames, fps=FPS, ref=(1080, 1920), verbose=False):
    """One CONSTANT world-px-per-Angstrom for the whole clip, solved against the worst frame.

    ⚠ CONSTANT, AND THAT IS THE WHOLE POINT. Both molecular scenes are about things MOVING - a
    cluster assembling, a molecule flying apart - so a per-frame fit would shrink the frame
    exactly as the subject spreads and the viewer could not tell a dispersal from a zoom. S5's
    auto-zoom warning, and here the subject's extent IS the story rather than a nuisance.

    The worst frame is the START for the water (scattered) and the END for the combustion
    (dispersed products); solved over every frame rather than either assumption.
    """
    rad = sc.radii()
    # 26 directions per atom is a cheap conservative hull - the drawn surface sits a little
    # outside the bare atom radius where neighbours push it out, hence the 1.18.
    dirs = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                if a or b or c:
                    dirs.append([a, b, c])
    dirs = np.asarray(dirs, np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    hx, hy = FIT_X * ref[0] * 0.5, FIT_Y * ref[1] * 0.5
    if MOL_FILL:
        # ⚠ SOLVED AGAINST THE FINAL LATTICE AND HELD CONSTANT. The sheet contracts as it
        # orders, so a per-frame fit would push in as it froze - and the whole point is that
        # the frame stays full. Sizing on the END state means the ordered sheet exceeds the
        # frame by `MOL_FILL`, and the disordered start (which is larger) exceeds it by more.
        cam = camera_at(sc.duration, ref=ref)
        ext = getattr(sc, "fill_extent", None)
        if ext is not None:
            # ⚠ FRAMED ON THE DECLARED EXTENT, NOT ON THE ATOM HULL. The hull runs a lattice
            # period past the frame on purpose - that margin is what gives the visible molecules
            # their bonding partners - so framing on it would pull the margin into view and put
            # the crystal's edge back on screen.
            hull = np.asarray([[ext[0], ext[1], 0.0], [-ext[0], ext[1], 0.0],
                               [ext[0], -ext[1], 0.0], [-ext[0], -ext[1], 0.0]], np.float64)
        else:
            Pe = sc.positions(sc.duration)
            hull = (Pe[:, None, :] + np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                                                np.float64)[None, :, :]
                    * (rad[:, None, None] * 1.18)).reshape(-1, 3)
        R = 90.0
        for _ in range(5):
            x, y, _z = cam.project((hull * R).astype(np.float32))
            got = min(np.abs(x - ref[0] * 0.5).max() / (ref[0] * 0.5),
                      np.abs(y - ref[1] * 0.5).max() / (ref[1] * 0.5))
            R = R * float(MOL_FILL) / max(got, 1e-6)
        if verbose:
            print(f"[fit] FILL: {R:.1f} px per Angstrom, constant - the sheet exceeds the "
                  f"frame by {float(MOL_FILL):.2f}")
        return np.full(n_frames, R, np.float32)
    per = np.empty(n_frames, np.float64)
    for f in range(n_frames):
        t = f / float(fps)
        P = sc.positions(t)
        hull = (P[:, None, :] + dirs[None, :, :] * (rad[:, None, None] * 1.18)).reshape(-1, 3)
        cam = camera_at(t, ref=ref)
        R = 90.0
        for _ in range(4):
            x, y, _z = cam.project((hull * R).astype(np.float32))
            ratio = max(np.abs(x - ref[0] * 0.5).max() / hx,
                        np.abs(y - ref[1] * 0.5).max() / hy)
            R = R / max(ratio, 1e-6)
        per[f] = R
    # ⚠ A PARTIAL FIT, exactly as `FIT_BETA` does for the orbitals (S5), and for the same
    # reason in reverse. A CONSTANT scale sized to the worst frame leaves the assembled cluster
    # - the thing the clip is about - at a third of the frame, because the scattered start is
    # twice its size. A FULL per-frame fit is the auto-zoom S5 forbids: the subject's extent IS
    # the story here, and normalising it away would make an assembly indistinguishable from a
    # push-in. `MOL_BETA` keeps most of the correction, so the cluster fills the frame AND
    # visibly closes. Smoothed over a second first, so the ramp is a camera move rather than a
    # per-frame correction.
    k = max(int(round(fps)) | 1, 3)
    pad = np.concatenate([np.full(k // 2, per[0]), per, np.full(k // 2, per[-1])])
    sm = np.convolve(pad, np.ones(k) / k, mode="valid")
    out = (per.min() ** (1.0 - MOL_BETA)) * (sm ** MOL_BETA)
    out = np.minimum(out, per)                    # never above the frame's own bound
    if verbose:
        print(f"[fit] molecule scale {out.min():.1f} -> {out.max():.1f} px per Angstrom "
              f"(authored push-in, beta {MOL_BETA:.2f}; a constant fit would be "
              f"{per.min():.1f})")
    return out.astype(np.float32)


def fit_scales(n_frames, fps=FPS, ref=(1080, 1920), verbose=False):
    """(n_frames,) world scale per frame: `min(R_NOMINAL, smoothed exact-fit scale)`.

    Solved rather than estimated. Projected extent is not linear in the world scale - a
    perspective camera grows the near side faster than the far - so an orthographic guess is
    not a bound. Three fixed-point corrections `R <- R / ratio` converge to well under a pixel
    at this object-to-distance ratio.
    """
    if MOL is not None:
        from . import molecule as _M
        sc = _M.SCENES[MOL](duration=DURATION)
        if MOL_SCALE:
            return np.full(n_frames, float(MOL_SCALE), np.float32)
        return mol_scale(sc, n_frames, fps, ref, verbose)
    th, ph, U, A = _coarse_mesh()

    def solve(r, cam, hx, hy, sep=0.0):
        """The world scale at which this cloud EXACTLY fills the (hx, hy) half-extent box.

        ⚠ WITH A BOND IT IS THE UNION OF BOTH ATOMS, and it has to be. The guard's job is to
        prove nothing leaves the frame; measuring one centred copy of a two-atom configuration
        measures a shape that is never on screen, and it understates the vertical extent by
        half the separation - which is the largest length in these clips. `sep` is in units of
        R precisely so this iteration stays homogeneous in R (see `SEP_FAR`).
        """
        R = R_NOMINAL
        for _ in range(3):
            P = (R * r)[:, None] * U
            if sep > 0.0:
                off = np.zeros((1, 3), P.dtype)
                off[0, 1] = 0.5 * sep * R
                P = np.concatenate([P + off, P - off])
            # ⚠ `Camera.project` returns PIXEL coordinates with the origin at the TOP-LEFT
            # (it feeds a splat that indexes an image), not offsets from the optical axis.
            # Forgetting the principal point makes every extent ~half a frame too big, the
            # fit divides R by ~2.4 three times, and the whole orbital renders as a 16 px
            # dot with every statistic still inside its bounds.
            x, y, z = cam.project(P)
            ratio = max(np.abs(x - ref[0] * 0.5).max() / hx,
                        np.abs(y - ref[1] * 0.5).max() / hy)
            R = R / max(ratio, 1e-6)
        return R

    frame_ = np.empty(n_frames, np.float64)
    hard = np.empty(n_frames, np.float64)
    for f in range(n_frames):
        t = f / float(fps)
        i, j, w = blend(t)
        r = ((1.0 - w) * A[i] + w * A[j]).astype(np.float64)
        cam = camera_at(t, ref=ref)
        sp = sep_at(t)
        frame_[f] = solve(r, cam, FRAME_X * ref[0] * 0.5, FRAME_Y * ref[1] * 0.5, sp)
        hard[f] = solve(r, cam, FIT_X * ref[0] * 0.5, FIT_Y * ref[1] * 0.5, sp)
        # ⚠ THE VIBRATION GROWS THE SILHOUETTE, so the HARD bound is solved against the worst
        # case. Only the hard bound: putting it into `frame_` would make the framing pump.
        if SY.ACTIVE is not None:
            hard[f] /= SY.ACTIVE.max_inflation()

    # ⚠ A PARTIAL fit, not a full one, and this is the interesting number in the file.
    # Normalising every |Y_lm| to max 1 is what stops the high-l states leaving the frame -
    # but it OVERCORRECTS: an f or g state spends its unit radius on eight lobes pointing
    # sideways, so its projected footprint is much smaller than a sphere of the same radius,
    # and the second half of the clip visibly shrank. A full fit-to-frame fixes that and
    # throws away the honest size differences with it (a p_z really is twice as tall as it is
    # wide, and auto-zooming it to the same footprint as a sphere is a lie about the physics).
    # The geometric blend keeps `FIT_BETA` of the correction: shapes still differ in size,
    # they just no longer differ by a factor of two.
    b = float(FIT_BETA)
    want = (R_NOMINAL ** (1.0 - b)) * (frame_ ** b)

    # ⚠ A SOFT MINIMUM, not a hard one followed by a blur. The first pass blurred
    # `min(nominal, hard)` over 0.30 s to keep the scale from kinking - and a Gaussian is
    # symmetric, so every frame within 0.9 s of a BINDING frame got pulled down too. The
    # opening `s` sphere, which does not bind at all, rendered visibly smaller than the
    # identical closing `s` sphere purely because a p_z transition 0.6 s later did. This
    # is C-infinity, is never above either operand, and touches a frame only when that
    # frame's own bound is near.
    p = float(SOFTMIN_P)
    used = (want ** -p + hard ** -p) ** (-1.0 / p)
    if verbose:
        nb = int((hard < want - 1e-6).sum())
        print(f"[fit] guard binds on {nb}/{n_frames} frames ({nb / n_frames:.0%}); "
              f"R in [{used.min():.1f}, {used.max():.1f}] of nominal {R_NOMINAL:.0f}")
    return used.astype(np.float32)
