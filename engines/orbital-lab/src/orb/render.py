# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""render.py - the per-frame point build, the palettes, the DRY GATE and the ffmpeg pipe.

This module owns the only hot loop in the project. Per frame, on the device:

    r      = lerp of two precomputed |Y_lm| vectors            (N,)
    P      = (R * r) * U                                        (N,3)
    for each lit electron:  gain *= ...;  key += ...;  and in mode "pull" only, P += ...
    three DEPOSITS into one accumulator - mesh, nucleus, electrons - each with its own LUT

and nothing else. There is no integration, so a frame at t=12 costs exactly what a frame at
t=0 costs and `still` is as cheap as it looks.

WHY THREE DEPOSITS AND NOT ONE COLOUR KEY
-----------------------------------------
`Canvas` colours a deposit through a single 256-entry LUT indexed by `key`. The mesh is a teal
ramp and the nucleus/electron are a magenta one, and those are not two ends of one ramp - a
ramp that had to reach magenta at the top would tint the mesh's own bright cores magenta,
which is precisely the fault the source does NOT have (its hot mesh knots are white). So the
LUT is swapped between passes into the same accumulator. Additive blending makes that exactly
equivalent to one pass with two colour spaces, and it costs one extra scatter over a few
thousand points.

WHAT THIS BORROWS AND WHY IT MUST NOT BE COPIED
-----------------------------------------------
`..\\Particle_Fields\\pf\\` is imported READ-ONLY: `gpu` (the CuPy backend and its CUDA_PATH
scrub), `canvas` (the splat/tonemap/bloom chain), `camera` (the pinhole) and `config`
(RenderConfig + the NVENC probe). Vendoring a copy would fork four files that have already
been debugged expensively - `canvas.to_rgb`'s clamp before the fractional power alone is a bug
that shipped black pixels - and this project has no reason to diverge from any of them.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from . import flow as _F
from . import bond as _B
from . import molecule as _M
from . import scene as S
from . import sync as _SY
from . import void as _V
from ._pf import Canvas, RenderConfig, encoder_args, find_ffmpeg, gpu as _G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out")

REF_W, REF_H = 1080, 1920

# --------------------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------------------
#: The mesh. Deep teal -> turquoise -> white-hot, monotone in luminance. This is
#: Particle_Fields' `cyan` ramp with its dark end pulled down: vdx measured the source's stage
#: at bg #010103 and 71% dark, and pf's ramp starts at (4,18,26) because ITS scenes fill the
#: frame with particles. A wireframe leaves most of the frame empty and a lifted dark end
#: would grey the void out.
#:
#: ⚠ The MIDS are the colour of the clip, and they are TEAL, not blue. A wireframe puts almost
#: all of its area in the 0.3-0.7 band, and pf's `cyan` ramp is B-dominant there (26,152,178):
#: the first pass came back an even electric blue with none of the source's green. The source
#: measures G > B through its whole body and only goes white at the knots, so every mid stop
#: below leads on GREEN. Same lesson Particle_Fields S7 wrote up as "secondary hues in the
#: midtones" - it just happens to be the primary hue that was wrong here.
#:
#: ⚠ And the ramp runs BLUE -> TEAL -> WHITE from the outside in, because `key` is driven by
#: (1 - r): the outer skin of every lobe sits at the bottom of the ramp and the origin sits at
#: the top. The source does exactly this - its lobe tips are indigo, its body is teal-green and
#: only the knots are white - and a ramp that is teal all the way down renders every lobe the
#: same colour, which is what the second pass here did.
RAMP_MESH = [(0.00, (2, 10, 18)), (0.22, (10, 46, 88)), (0.40, (14, 118, 142)),
             (0.60, (34, 186, 176)), (0.80, (140, 236, 228)), (0.92, (206, 248, 250)),
             (1.00, (250, 255, 255))]

#: The nucleus and the lit electron. Magenta into white - the source's centre dot and its
#: bright points are both pink, and they are the only non-teal thing in the frame.
RAMP_CORE = [(0.00, (36, 4, 26)), (0.35, (150, 18, 108)), (0.70, (238, 74, 186)),
             (0.90, (255, 168, 232)), (1.00, (255, 240, 252))]


#: THE CHARGE RAMP - blue for positive, red for negative.
#:
#: ⚠ THE COLOURS ARE THE CONVENTION, NOT A CHOICE. Every electrostatic-potential map in
#: chemistry reads RED = electron-rich = negative and BLUE = electron-poor = positive, so a
#: hydrogen bond runs BLUE at the hydrogen (delta+) to RED at the oxygen (delta-). Reversing it
#: would be a plausible, completely wrong picture - this project's recurring failure - so
#: `MolLines` derives the direction from the atoms' ELEMENTS, never from bond bookkeeping.
#:
#: ⚠ DIVERGING, WITH A PALE MIDDLE, because the ask was for two charges and not a gradient of
#: one. A ramp that just darkened between the ends would read as a single glowing tube; the
#: near-white waist is what makes it read as two poles meeting.
#: ⚠ THE PALE WAIST IS NARROW ON PURPOSE. Spread over 0.40-0.60 it swallowed the middle fifth
#: of every line, and with eleven lines per bond that pale band was most of what the eye saw -
#: the charges stopped reading. Squeezed to 0.45-0.55, each line is blue for nearly half its
#: length and red for nearly half, which is the statement the colours are there to make.
RAMP_CHARGE = [(0.00, (20, 56, 255)), (0.28, (52, 112, 255)), (0.45, (140, 180, 255)),
               (0.50, (226, 224, 236)), (0.55, (255, 176, 162)), (0.72, (255, 92, 76)),
               (1.00, (255, 26, 34))]


def violet_ramp(base):
    """The bloom variant's second-layer colour: HALFWAY between the glow blue and the
    background purple of whatever the mesh ramp is (Clancy, 2026-09-16).

    Derived from `base` - the GRADED mesh LUT - rather than authored, so it stays between
    those two colours if the grade moves. Background = the ramp's field key (0.22), glow = the
    rim body (0.68), measured against a rendered frame (glow px median (49, 38, 213) against
    LUT(0.68)). Monotone in luminance like every ramp here, whitening only at the very top.
    """
    bg = base[int(0.22 * 255)]
    glow = base[int(0.68 * 255)]
    mid = 0.5 * (bg + glow)
    hi = mid + 0.45 * (np.array([255.0, 255.0, 255.0]) - mid)
    stops = [(0.00, tuple(base[0])), (0.22, tuple(bg)), (0.60, tuple(mid)),
             (0.88, tuple(hi)), (1.00, (235.0, 230.0, 255.0))]
    return ramp(stops)


def split_luts(base):
    """(A, B) whose SUM IS `base`, entry for entry - the chromatic separation pair.

    ⚠ DERIVED FROM THE PARENT RAMP, NOT AUTHORED. Two hand-made magenta/cyan ramps would be a
    second palette to keep in step with `RAMP_MESH`, and worse, they would not SUM to it - so
    wherever the two split copies register (every node, every knot, the polar axis, and every
    frame at split 0) the colour would be something other than the shipped teal. Splitting the
    parent per channel makes registration exact by construction:

        A = (R, 0, B/2)      magenta / violet
        B = (0, G, B/2)      cyan / green

    Blue is halved into both because it is the channel both hues share; R and G carry the
    separation. At the teal mids that is violet (34,0,88) against green-cyan (0,186,88); at the
    white knots, magenta against spring-green. Total flux is conserved exactly, so the only
    exposure risk left is the CONCAVE tonemap - two half-bright dots tonemap slightly brighter
    than one full dot - which is measured, not assumed.
    """
    a = base.copy()
    b = base.copy()
    a[:, 1] = 0.0
    b[:, 0] = 0.0
    a[:, 2] *= 0.5
    b[:, 2] *= 0.5
    return a, b


def ramp(stops, n=256):
    ts = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    u = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(u, ts, cs[:, c]) for c in range(3)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def config(tier="hd", **kw) -> RenderConfig:
    """1080x1920 is BOTH the reference frame and the delivery frame, so `scale` is 1.0.

    The tiers exist only so a review pass can be cheap. `hd` is the deliverable.
    """
    # ⚠ THE REFERENCE FRAME FOLLOWS THE ORIENTATION, and it has to. Every length in this
    # project is quoted in the 1080x1920 reference frame, and `camera_at` / `fit_scales` take
    # that frame as an argument — so a 16:9 tier needs a 16:9 reference (1920x1080) or the
    # object is fitted to a portrait box inside a landscape frame and sits in the middle third
    # of it. The orbital is roughly square, so the HEIGHT is what binds either way; what
    # changes is which dimension the guard measures against.
    w, h, fps, ref = {
        "preview": (540, 960, 30, (REF_W, REF_H)),
        # ⚠ `720` IS A PROOF TIER, not a delivery one. 720x1280@30 is the smallest frame the
        # look survives at (the wireframe is ~1 px wide at `point_sigma` 1.6, and half of REF
        # is where that stops being true), and it renders the whole 20 s in about a minute, so
        # a choreography change can be judged on a MOVING clip instead of on a contact sheet.
        # The library's finals rule is still 4K/60 - `uhd` - and this is not that.
        "720": (720, 1280, 30, (REF_W, REF_H)),
        "hd": (1080, 1920, 60, (REF_W, REF_H)),
        "uhd": (2160, 3840, 60, (REF_W, REF_H)),
        # 2K vertical. Added 2026-09-16 for the first music final.
        "2k": (1440, 2560, 60, (REF_W, REF_H)),
        "wide": (1920, 1080, 60, (REF_H, REF_W)),
        "final": (3840, 2160, 60, (REF_H, REF_W)),
    }[tier]
    cfg = RenderConfig(width=w, height=h, fps=fps, ref_w=ref[0], ref_h=ref[1])
    cfg.bg = (1, 1, 3)                 # vdx: source stage #010103
    cfg.palette = "cyan"               # replaced per pass; kept valid so Canvas can build
    # ⚠ EXPOSURE IS MATCHED TO THE SOURCE, MEASURED, not set by eye. The source at its native
    # 716x1274 runs dark 41-68%, luma mean 14-32, p99 208-254, blown 0.0001-0.010. The first
    # pass here rendered dark 94% / luma 9 / p99 250 - the right SHAPE inside a dead frame,
    # because a wireframe on black only fills the void through its bloom and the bloom was
    # both too tight and too high-threshold. What moved the numbers, in order of effect:
    # the threshold (62 -> 44 lets the BODY of the mesh bloom, not just the knots), the wide
    # halo gain, the line width, and only then `ink`.
    # ⚠ Do NOT chase the last of the gap with `ink`. Above ~1.8 the knots clip before the
    # void lights up: `blown` passes 0.02 while `dark` barely moves, which is the same
    # "exposure vs concentration" confusion Particle_Fields S5 spent three targets on.
    # ⚠ ...and the correction pass, which is the more useful half of the lesson. Pushing to
    # dark 57% / luma 26 by opening the bloom (threshold 44, ink 1.55) DID hit the numbers and
    # destroyed the picture: every lobe became one solid white mass with no wireframe left in
    # it. The source's numbers are high because its object is big and its knots are hot, NOT
    # because its body is washed out - so matching the aggregate by flooding the body matched
    # the wrong thing. Settled between the two passes, by eye, on the still.
    cfg.point_sigma = 1.60
    cfg.trail_tau = 0.045              # a short smear, not a comet - the mesh must stay crisp
    cfg.tone_gamma = 1.62
    cfg.bloom_threshold = 66.0
    cfg.glow_sigma = 4.5
    cfg.glow_gain = 0.32               # vdx: source glow index 0.418
    cfg.glow2_sigma = 32.0
    cfg.glow2_gain = 0.26
    cfg.vignette = 0.18
    cfg.ink = 1.30
    # ⚠ DOT MODE RE-TUNES THE BLOOM, NOT THE EXPOSURE. Separated glowing dots and continuous
    # lines are two different pictures asking the same tone curve for two different things: a
    # dot has to have a HOT CORE that crosses `bloom_threshold` so the glow pass gives it a
    # halo - that halo IS the "nodes will likely have to glow" - while a line only ever needed
    # its knots to bloom. Settled on the still, like every other exposure number here: at gain
    # 4.5 / threshold 52 the lobes flooded into solid glowing masses with no dots left in them,
    # which is exactly S7's "matching the aggregate matched the wrong thing" a second time.
    # These sit at the point where the dots are still individually legible.
    if S.MESH == "dots":
        # ⚠ THE PERSISTENCE HAS TO COME DOWN, and this is the non-obvious half of dot mode.
        # `trail_tau` 0.045 leaves a dot visible for ~2 frames while the drift carries it
        # ~4 ref px - so every dot renders as a SHORT STREAK and the picture goes straight
        # back to being made of little lines, which is the one thing dot mode exists to stop.
        # It was plainly visible on the first dot still and invisible in the statistics.
        cfg.trail_tau = 0.012
        cfg.point_sigma = 2.20
        cfg.bloom_threshold = 60.0
        cfg.glow_gain = 0.40
        cfg.glow2_gain = 0.28
    # ⚠ THE VOID INVERTS THE FRAME, SO THE BLOOM HAS TO BE RE-AIMED. Every glow number above
    # was chosen to light an EMPTY void around a small bright object. With the frame filled,
    # a 32 px halo over every field dot is a uniform lift - the flat glowing rectangle - so
    # `glow2_gain` comes down hard, the threshold goes UP so only the carve rim and the knots
    # bloom rather than the whole body, and the vignette becomes load-bearing: it is what
    # re-establishes a dark border so the frame does not read as a lit card.
    if S.VOID > 0.0:
        # ⚠ THE WIDE HALO IS THE ENEMY OF A FILLED FRAME. `glow2` is a 32 ref-px blur that
        # exists to light an EMPTY void around a small bright object; over a field that covers
        # the whole frame it is a uniform lift that erases the dot texture and leaves a flat
        # glowing rectangle. It comes down hard, the threshold goes UP so only the carve rim
        # and the knots bloom rather than the whole field, and the vignette becomes
        # load-bearing - it is what re-establishes a dark border.
        cfg.glow2_gain = 0.06
        cfg.glow_gain = 0.18
        cfg.bloom_threshold = 105.0
        cfg.vignette = 0.34
    # THE HQ "FINE" MASTER (`orb/hq.py`, item 2): pitch and point blur shrink together.
    from . import hq as _HQ
    if _HQ.fine_active(cfg):
        cfg.point_sigma *= float(_HQ.HQ["fine"])
        _V.PITCH_SCALE = float(_HQ.HQ["fine"])
    else:
        _V.PITCH_SCALE = 1.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------------------
# the point build
# --------------------------------------------------------------------------------------
class _Points:
    """The four attributes `Canvas.deposit` reads, plus WHICH LUT this pass wants.

    ⚠ `split` IS CARRIED ON THE PASS AND NOT INFERRED FROM THE LIST LENGTH, and that is a fix.
    The frame loop used to pick the complementary LUTs by `lut_split[pass_index]`, which is
    correct only while the number of passes IS the number of split copies. A bond scene
    deposits the wireframe once PER ATOM, so a split bond clip has four passes and the old
    expression indexed a two-entry list with 3. Saying which LUT a pass wants is the thing that
    was actually meant; the count was only ever a proxy for it.
    """
    __slots__ = ("P", "key", "gain", "dt", "split")

    def __init__(self, P, key, gain, dt, split=None):
        self.P, self.key, self.gain, self.dt = P, key, gain, dt
        self.split = split


#: How hard a lit electron pulls - APPLIED ONLY IN `S.ELECTRONS == "pull"`. `GLOBAL` moves the
#: WHOLE cloud toward the point (this was the "converge to that point" of the original brief);
#: `LOCAL` is an extra gaussian pucker so the point reads as an attractor with a neighbourhood
#: rather than as a rigid translation of the whole shape.
PULL_GLOBAL = 0.24
PULL_LOCAL = 0.48
PULL_SIGMA = 0.55        # in units of R
#: The brightening halo around a lit point, and how far up the LUT it pushes the mesh there.
#: ⚠ Kept TIGHT, and the KEY lift kept SMALL. At sigma 0.30 R and gain 2.6 the halo was a third
#: of the orbital wide and rendered as a featureless white ellipse hanging off one side - it
#: read as a blown highlight, not as an electron. Worse, a large `LIT_KEY` drives the mesh to
#: the white end of its ramp exactly where the magenta electron sits, so the one element that
#: is supposed to be a different COLOUR disappears into a white patch of its own making.
#: The event has to be legible as a POINT; the convergence is what makes it large-scale, and
#: that is `PULL_*`, not this.
LIT_SIGMA = 0.19
LIT_GAIN = 1.3
LIT_KEY = 0.22

#: Bounds on the arc-length weight (in units of the mesh's mean spacing). See `_arc_gain`.
ARC_FLOOR = 0.06
ARC_CEIL = 3.20

#: The nucleus. The source does not have a magenta DOT, it has a magenta ATMOSPHERE - a soft
#: pink cast through the inner third of the orbital with a hot point at its centre - and that
#: cast is most of what stops the middle of the frame reading as a plain white star.
#:
#: ⚠ THAT ATMOSPHERE IS MADE BY THE BLOOM, NOT BY POINTS. The obvious construction - a second,
#: wide, dim shell of points - does not work at any density this project can afford: a splatted
#: cloud is smooth only when many points land in every pixel, and a sigma-46 shell covers
#: ~70,000 px, so even 44,000 points give 0.6 per pixel and the "haze" renders as a field of
#: individual magenta specks 350 px across. (Flattening their gains removes the sparkle and
#: keeps the grain - the fluctuation is Poisson, not the gain distribution.) A small core
#: driven hard ABOVE `bloom_threshold` gets the same magenta light spread by a true Gaussian
#: over `glow2_sigma`, perfectly smooth, for a tenth of the points. The colour survives because
#: the bloom is taken per channel on the tonemapped frame: a clipped magenta core still has
#: much less green than red or blue, so its halo is magenta and not white.
#: ⚠ And the gain is bounded ABOVE by colour, not by exposure. Drive the core hard enough and
#: all three channels clip, the halo goes white, and the one magenta element in the frame
#: disappears - at gain 120 the "pink nucleus" rendered as a plain white star. What is wanted
#: is a small clipped centre inside an annulus where R and B clip and G does not; that annulus
#: is what tints the bloom.
#: Deposit gain for the charge field lines.
#:
#: ⚠ SIZED SO THEY CROSS `bloom_threshold`, which is what Clancy asked for by name: *"make sure
#: that red and blue glow and somewhat bloom around their lines."* A glow is not a property of
#: the ramp - it is what the bloom pass does to anything whose accumulated value clears the
#: threshold - so a line that reads as the right colour and sits under it has no halo at all.
#: In void mode that threshold is 105, well above the field, so the lines have to be deposited
#: hot enough to pass it and are then the only thing in frame besides the carve rim that blooms.
#: ⚠ AND THE TENSION IS REAL: hot enough to bloom, dim enough not to take over. Resolved by
#: cutting the line COUNT and the bow rather than the gain - fewer, tighter lines put the same
#: peak on the accumulator (so the bloom still fires) while depositing a fraction of the total
#: flux across the frame.
LINE_GAIN = 1.15

NUC_N = 60000
NUC_SIGMA = 20.0         # ref px
NUC_GAIN = 0.40
NUC_KEY = 0.70

#: The lit electron, built the same way and for the same reason: one compact shell driven well
#: past the bloom threshold, so the pink corona that makes it read as an ELECTRON rather than a
#: hot pixel comes out of the glow pass instead of out of a second sparse point cloud.
ELE_N = 20000
ELE_SIGMA = 10.0
ELE_GAIN = 1.30
ELE_KEY = 0.74


class Orbital:
    """The whole clip, as data. Built once; `points_at(t)` is a pure function after that."""

    def __init__(self, on_gpu=True, seed=20260825):
        th, ph, U, A, shape = S.build_mesh()
        self.events = S.plan(A, seed=seed)
        self.N = int(U.shape[0])
        self.shape = shape
        self.U_cpu = U
        self.A_cpu = A
        self.xp = _G.cp if (on_gpu and _G.GPU) else np
        self.U = _G.asarray(U, np.float32) if on_gpu else U
        self.A = _G.asarray(A, np.float32) if on_gpu else A

        # ---- flow ---------------------------------------------------------------------
        # ⚠ RESOLVED ONCE HERE, and unlike the electron mode it is safe to cache: the spec is
        # geometry the Manifold is BUILT from, and `configure()` is required to run before any
        # build regardless (the registry's standing rule). The electron mode is read per frame
        # because it is a branch, not a build.
        self.flow = _F.spec(S.FLOW)
        if S.LOOP:
            # ⚠ SNAPPED ONCE, HERE, so every consumer sees the same rates. Snapping inside
            # `rebuild` would re-derive them per frame from a duration that is already fixed,
            # and `dry` would then be gating numbers the render does not use.
            self.flow = _F.loop_snap(self.flow, S.DURATION, shape,
                                     float(np.pi) / (shape["n_lat"] + 1))
        self.flow_static = _F.is_static(self.flow)
        self.man = None
        self.th_dev = self.ph_dev = self.por = None
        if self.flow:
            self.man = _F.Manifold(th, ph, shape, self.xp,
                                   (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                   else (lambda a: a))
            self.th_dev = _G.asarray(th.astype(np.float32)) if on_gpu                 else th.astype(np.float32)
            self.ph_dev = _G.asarray(ph.astype(np.float32)) if on_gpu                 else ph.astype(np.float32)
            # static, so built once - it is a property of the GRID, not of the frame
            self.por = self.man.porosity(self.flow)
        # ---- the bond ------------------------------------------------------------------
        self.bond = (_B.BondField(S.BOND, self.xp,
                                  (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                  else (lambda a: a))
                     if S.BOND is not None else None)
        # ---- the molecular scene ---------------------------------------------------------
        self.mol = None
        if S.MOL is not None:
            self.molsc = _M.SCENES[S.MOL](duration=S.DURATION)
            self.mol = _M.MolField(self.molsc, self.xp,
                                   (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                   else (lambda a: a))
        # ---- the shell cloud - what is DRAWN in bond scenes -----------------------------
        self.cloud = (_B.ShellCloud(self.bond, self.xp,
                                    (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                    else (lambda a: a))
                      if (S.MESH == "cloud" and self.bond is not None) else None)
        # ⚠ THE PER-ATOM DOT COUNT SCALES DOWN WITH THE ATOM COUNT. 2,600 x 22 dots is right
        # for two atoms and ruinous for 144; the total is what costs, and each atom is a much
        # smaller thing on screen when there are that many.
        _na = len(self.molsc.atoms) if self.mol is not None else 1
        _nd = 2600 if _na <= 30 else (1500 if _na <= 60 else 950)
        _nr = 22 if _na <= 30 else (18 if _na <= 60 else 15)
        self.mcloud = (_M.MolCloud(self.molsc, self.mol, self.xp,
                                   (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                   else (lambda a: a), n_dirs=_nd, n_rad=_nr)
                       if (S.MESH == "cloud" and self.mol is not None) else None)
        self.mlines = (_M.MolLines(self.molsc, self.xp,
                                   (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                   else (lambda a: a))
                       if (self.mol is not None
                           and any(b.kind == "hb" for b in self.molsc.bonds)) else None)
        self.void = None
        # ⚠ THE VOID NO LONGER NEEDS A `Manifold`. Its carve comes from the bond field, and a
        # cloud scene builds no Manifold at all (there are no curve families to drift), so the
        # old `self.man is not None` test silently switched variation C's whole field off.
        if S.VOID > 0.0 and (self.man is not None or self.bond is not None
                             or self.mol is not None):
            ref = (REF_W, REF_H)
            self.void = _V.VoidField(self.man, self.xp,
                                     (lambda a: _G.asarray(a, a.dtype)) if on_gpu
                                     else (lambda a: a),
                                     S.camera_at(0.0, ref=ref), ref)

        rng = np.random.default_rng(seed + 7)
        self.nuc = _G.asarray(rng.normal(0.0, 1.0, (NUC_N, 3)).astype(np.float32))
        self.ele = _G.asarray(rng.normal(0.0, 1.0, (ELE_N, 3)).astype(np.float32))
        #: FLAT gains for the blobs. A heavy tail is what gives a sparse cloud its sparkle,
        #: and sparkle is exactly wrong for something whose job is to be a smooth glow.
        self.nuc_g = _G.asarray(np.ones(NUC_N, np.float32))
        self.ele_g = _G.asarray(np.ones(ELE_N, np.float32))

    # ---- the arc-length weight ---------------------------------------------------------
    def _arc_gain(self, P):
        """Per-point gain proportional to that point's SPACING along its own curve.

        ⚠ THIS IS THE SINGLE BIGGEST LOOK FIX IN THE PROJECT, and it is a sampling correction,
        not a style choice. A polyline drawn by splatting samples has flux per unit length
        `g / ds`: uniform `g` therefore makes brightness a function of how the curve happened
        to be PARAMETERISED. On these surfaces `r -> 0` at every node and at the origin, so the
        theta samples of all 112 meridians collapse into a few pixels and lay down thousands of
        deposits there while a stretched lobe tip gets a handful. That rendered as one enormous
        white star swallowing the middle of the frame, with lobe tips fading to black - and no
        exposure, gamma or bloom setting can fix it, because it is two different line densities
        being asked to share one brightness curve.
        `g` proportional to `ds` makes the flux per unit length constant, which is what a
        stroked wireframe actually looks like, and it brightens the tips and darkens the core in
        exactly the right proportion at the same time.

        The clip is a guard on both ends: `ds -> 0` at a node would otherwise erase the knot
        entirely (the source has bright knots and so should this), and one badly stretched
        segment would otherwise draw as a bright dash.
        """
        xp = self.xp
        sh = self.shape
        n1 = sh["n1"]
        # meridians: open curves, so the last sample copies its neighbour's spacing
        a = P[:n1].reshape(sh["n_mer"], sh["n_th"], 3)
        d1 = xp.sqrt(((a[:, 1:] - a[:, :-1]) ** 2).sum(axis=2))
        d1 = xp.concatenate([d1, d1[:, -1:]], axis=1).reshape(-1)
        # latitudes: CLOSED curves - the wrap-around segment is a real segment
        b = P[n1:].reshape(sh["n_lat"], sh["n_ph"], 3)
        d2 = xp.sqrt(((xp.roll(b, -1, axis=1) - b) ** 2).sum(axis=2)).reshape(-1)
        ds = xp.concatenate([d1, d2])
        return xp.clip(ds / xp.maximum(ds.mean(), np.float32(1e-6)),
                       np.float32(ARC_FLOOR), np.float32(ARC_CEIL))

    # ---- one frame ------------------------------------------------------------------
    def points_at(self, t, R, dt):
        """(meshes, nucleus, electrons). `meshes` is ONE `_Points`, or TWO when a split is on.

        ⚠ IT IS A LIST BECAUSE THE CHROMATIC SPLIT IS TWO DEPOSITS OF THE SAME WIREFRAME, and
        the caller has to swap the LUT between them. Returning a list rather than a second
        return value keeps every consumer - the frame loop and all three of `dry`'s checks -
        written once against "however many passes there are", which is what stopped the
        overflow check from silently measuring only half the drawn cloud.
        """
        xp = self.xp
        self._t = t
        i, j, w = S.blend(t)
        signs = (-1, +1) if (self.man is not None
                             and _F.has_split(self.flow)) else (0,)
        # ⚠ THE WIREFRAME IS DEPOSITED ONCE PER ATOM, TRANSLATED - which is the literal
        # meaning of "linear combination of ATOMIC orbitals". The hole in the void field is the
        # MOLECULAR orbital and it is what changes as the two approach; these two faint
        # wireframes are what say WHICH atomic orbitals made it. Same division of labour
        # `void_mesh` 0.15 already carries in variation C.
        offs = (None,)
        if self.bond is not None:
            half = np.float32(0.5 * S.sep_at(t) * R)
            offs = tuple(_G.asarray(np.asarray([0.0, s * half, 0.0], np.float32))
                         for s in (+1.0, -1.0))
        if self.mcloud is not None:
            h = 0.5 * _B.CLOUD_SPLIT
            out = []
            for si, sh in enumerate((-h, +h)):
                Pm, km, gm = self.mcloud.build(R, t, split=sh)
                out.append(_Points(Pm, km, gm, dt, split=si))
            return (out, self._nucleus(dt, offs), None)
        if self.cloud is not None:
            return (self._cloud(t, R, dt), self._nucleus(dt, offs), None)
        return ([self._pass(t, R, dt, i, j, w, sg, off)
                 for sg in signs for off in offs],
                self._nucleus(dt, offs), self._electrons(t, R, dt, i, j, w))

    def _cloud(self, t, R, dt):
        """The molecular surface, as two chromatically split shells of the SAME dot set.

        ⚠ THE SPLIT IS FREE HERE and it is a true separation. Both passes read the one `dist`
        array and light shells offset either side of the surface, so the two coloured copies are
        PHYSICALLY DISPLACED (along the surface normal) rather than recoloured - and their LUTs
        sum to `RAMP_MESH`, so wherever they register the colour is the shipped teal.
        """
        sep = S.sep_at(t)
        h = 0.5 * _B.CLOUD_SPLIT
        out = []
        for si, sh in enumerate((-h, +h)):
            P, key, gain = self.cloud.build(R, sep, split=sh)
            out.append(_Points(P, key, gain, dt, split=si))
        return out

    def lines_at(self, t, R, dt):
        """The hydrogen bonds as charge field lines, or None."""
        if self.mlines is None:
            return None
        got = self.mlines.build(R, t)
        if got is None:
            return None
        P, key, gain = got
        return _Points(P, key, gain, dt)

    def void_at(self, t, R, dt, cam):
        """The void field as a `_Points`, or None. Separate from `points_at` because it is a
        different object with its own deposit gain, not another pass of the wireframe."""
        if self.void is None:
            return None
        i, j, w = S.blend(t)
        if self.mol is not None:
            self.mol.set_time(t)
        P, key, gain = self.void.build(i, j, w, R, S.walk_u(t), cam,
                                       bond=self.bond or self.mol, sep=S.sep_at(t), t=t)
        self._react_core = None
        self._extra = [(name, _Points(P, k_, g_, dt))
                       for name, k_, g_ in getattr(self.void, "last_extra", [])]
        if _SY.ACTIVE is not None:
            got = _SY.ACTIVE.react(self.xp, self.void.last_L, self.void.last_d, t)
            if got is not None:
                mkey, lit1, mmult, ckey, cgain = got
                self._react_core = _Points(P, ckey, gain * cgain, dt)
                key = key * (1 - lit1) + mkey * lit1
                gain = gain * mmult
        return _Points(P, key, gain, dt)

    def _pass(self, t, R, dt, i, j, w, sign, off=None):
        xp = self.xp
        # ⚠ THE FAST PATH IS THE DEFAULT AND IT IS BIT-FOR-BIT THE OLD ENGINE. A flow that
        # only modulates brightness (`pulse`) also takes it: `stream_*` is a gain on the
        # existing points, and only a DRIFT moves a sample off the precomputed mesh.
        fade = None
        if self.man is not None and not self.flow_static:
            U, r, fade = self.man.rebuild(i, j, w, t, self.flow, sign=sign,
                                          u=S.walk_u(t))
        else:
            U = self.U
            r = self.A[i] * np.float32(1.0 - w) + self.A[j] * np.float32(w)
        if _SY.ACTIVE is not None:
            fac = _SY.ACTIVE.factor(xp, U[:, 0], U[:, 1], U[:, 2], t)
            if fac is not None:
                r = r * fac
        P = (r * np.float32(R))[:, None] * U
        if off is not None:
            P = P + off
        # ⚠ The FLOOR matters more than the span. At 0.20 the outer skin of every lobe landed
        # in the ramp's dead zone and the lobes read as dark shells around one white core;
        # the source's lobes are lit across their whole surface. 0.22 with a shorter span puts
        # the skin on the indigo stop and the body on the teal one.
        key = xp.clip(np.float32(0.22) + np.float32(0.62) * (np.float32(1.0) - r),
                      0.0, 1.0)
        gain = xp.ones(self.N, np.float32)

        # ⚠ THE MODE IS READ HERE, PER FRAME, and not captured in `__init__`. `configure()` is
        # allowed to run at any point before the first build, and a mode cached on the object
        # would be the previous scene's. In mode "off" `self.events` is empty, so this whole
        # block is one failed loop test per frame and costs nothing measurable.
        P, gain, key = self._lit_gain(t, R, i, j, w, P, gain, key)

        gain = gain * self._arc_gain(P)
        if self.man is not None:
            sg = self.man.stream_gain(t, self.flow, self.th_dev, self.ph_dev)
            if sg is not None:
                gain = gain * sg
            if fade is not None:
                gain = gain * fade
            if self.por is not None:
                gain = gain * self.por
        return _Points(P, key, gain, dt,
                       split=None if sign == 0 else (0 if sign < 0 else 1))

    def _lit_gain(self, t, R, i, j, w, P, gain, key):
        """The electron term -> (P, gain, key). Records the lit points on `self._lit` for
        `_electrons` to turn into its own deposit.

        ⚠ CALLED ONCE PER SPLIT PASS, and `self._lit` is overwritten each time rather than
        appended to - the two passes light the SAME events, so accumulating would deposit every
        electron twice. `points_at` reads it after the last pass."""
        xp = self.xp
        deform = S.ELECTRONS == "pull"
        self._lit = []
        lit = self._lit
        for ev in self.events:
            a = ev.env(t)
            if a <= 1e-4:
                continue
            re = float((1.0 - w) * self.A_cpu[i][ev.idx] + w * self.A_cpu[j][ev.idx])
            E = (R * re) * self.U_cpu[ev.idx]
            Eg = xp.asarray(E.astype(np.float32))
            d = Eg[None, :] - P
            d2 = (d * d).sum(axis=1)
            b2 = np.float32(2.0 * (LIT_SIGMA * R) ** 2)
            hot = xp.exp(-d2 / b2)
            if deform:
                # ⚠ ORDER IS LOAD-BEARING: `hot` is measured on the UNDISPLACED cloud, then the
                # cloud moves. Recomputing the halo after the pull would light the points the
                # pull had just dragged inward, which is a different (and much larger) patch.
                s2 = np.float32(2.0 * (PULL_SIGMA * R) ** 2)
                loc = xp.exp(-d2 / s2)
                pull = np.float32(a) * (np.float32(PULL_GLOBAL)
                                        + np.float32(PULL_LOCAL) * loc)
                P = P + pull[:, None] * d
            gain = gain * (np.float32(1.0) + np.float32(LIT_GAIN * a) * hot)
            key = xp.minimum(np.float32(1.0), key + np.float32(LIT_KEY * a) * hot)
            lit.append((E, a))
        return P, gain, key

    def _nucleus(self, dt, offs=(None,)):
        """The magenta core - ONE PER ATOM.

        ⚠ THIS WAS A REAL DEFECT ON THE FIRST BOND STILL, and it is the kind that only shows up
        by looking. The blob is built at the world origin, which for a single orbital IS its
        nucleus - and for a bond is the MIDPOINT BETWEEN two nuclei, where there is no nucleus
        at all. It rendered as a bright pink core sitting in the middle of the neck, which reads
        as "the bond glows" and is a statement about the physics that is simply false. Two
        atoms, two nuclei, each at its own centre.
        """
        xp = self.xp
        g = np.float32(NUC_GAIN if S.NUC_GAIN is None else S.NUC_GAIN)
        # ⚠ A MOLECULE HAS NO ONE NUCLEUS. The magenta core marks THE centre of a single
        # orbital; over 45 atoms there is no such point, and a blob at the world origin would
        # sit in the middle of a heptane chain claiming to be something. `nuc_gain=0`.
        if float(g) <= 0.0:
            return None
        if _SY.ACTIVE is not None and _SY.CORE is not None and offs == (None,):
            # THE LIVE CORE - a list of (lut, points) instead of one magenta blob
            return [(lut, _Points(P, k_, g_, dt))
                    for lut, P, k_, g_ in _SY.ACTIVE.core(xp, self.nuc, self._t,
                                                           NUC_SIGMA, float(g))]
        base = self.nuc * np.float32(NUC_SIGMA)
        Ps = [base if o is None else base + o for o in offs]
        n = NUC_N * len(Ps)
        return _Points(xp.concatenate(Ps) if len(Ps) > 1 else Ps[0],
                       xp.full(n, np.float32(NUC_KEY)),
                       (xp.concatenate([self.nuc_g] * len(Ps)) if len(Ps) > 1
                        else self.nuc_g) * g, dt)

    def _electrons(self, t, R, dt, i, j, w):
        xp = self.xp
        lit = getattr(self, "_lit", [])
        if not lit:
            return None
        Ps, ks, gs = [], [], []
        for E, a in lit:
            Eg = xp.asarray(E.astype(np.float32))[None, :]
            Ps.append(self.ele * np.float32(ELE_SIGMA) + Eg)
            ks.append(xp.full(ELE_N, np.float32(ELE_KEY)))
            gs.append(self.ele_g * np.float32(ELE_GAIN * a))
        return _Points(xp.concatenate(Ps), xp.concatenate(ks), xp.concatenate(gs), dt)


# --------------------------------------------------------------------------------------
# the frame loop
# --------------------------------------------------------------------------------------
def frames_of(cfg, times, orb=None, scales=None, on_gpu=True, preroll=0.25, bits16=False):
    """Yield (t, rgb) for each t in `times`, which must be NON-DECREASING.

    `times` need not be every frame - the point build is closed form - but the PERSISTENCE
    buffer is not, so skipping frames shortens the visible trail. `dry` accounts for that by
    probing consecutive pairs; a still is rendered with a short warm-up for the same reason.
    """
    orb = orb or Orbital(on_gpu=on_gpu)
    from . import hq as _HQ
    can = (_HQ.HQCanvas if _HQ.HQ else Canvas)(cfg, on_gpu=on_gpu)
    can.bits16 = bool(bits16)
    # a PYTHON float on purpose: an np.float32 1.0 changes canvas rounding and broke canon
    fg = float(_HQ.fine_gain(cfg)) if _HQ.HQ else 1.0
    fgc = float(_HQ.core_gain(cfg)) if _HQ.HQ else 1.0
    xp = can.xp
    # ⚠ A sync scene's grade is COLOUR ONLY (`orb/grade.py`); None leaves canon untouched.
    if S.GRADE:
        from . import grade as _GR
        base = ramp(_GR.apply(RAMP_MESH, S.GRADE))
    else:
        base = ramp(RAMP_MESH)
    lut_mesh = xp.asarray(base)
    _a, _b = split_luts(base)
    lut_split = [xp.asarray(_a), xp.asarray(_b)]
    if S.GRADE:
        lut_core = xp.asarray(ramp(_GR.apply(RAMP_CORE, S.GRADE, core=True)))
    else:
        lut_core = xp.asarray(ramp(RAMP_CORE))
    lut_charge = xp.asarray(ramp(RAMP_CHARGE))
    lut_violet = xp.asarray(violet_ramp(base))
    # the canon nucleus ramp, UNGRADED - the live core's magenta heart (`sync.CORE`)
    lut_magenta = xp.asarray(ramp(RAMP_CORE))
    cam_ref = (cfg.ref_w, cfg.ref_h)
    n = int(round(S.DURATION * cfg.fps))
    scales = scales if scales is not None else S.fit_scales(n, cfg.fps, cam_ref)
    h = 1.0 / cfg.fps

    def _step(t):
        f = min(n - 1, max(0, int(round(t * cfg.fps))))
        cam = S.camera_at(t, ref=cam_ref)
        meshes, nuc, ele = orb.points_at(t, float(scales[f]), h)
        can.decay(h)
        vf = orb.void_at(t, float(scales[f]), h, cam)
        if vf is not None:
            # ⚠ THE FIELD GOES DOWN FIRST and through the SHIPPED ramp, at its own low gain.
            # Its own key already runs hot at the carve rim and cool outward, so it needs no
            # second LUT and no boost pass.
            can.lut = lut_mesh
            can.deposit(vf, cam, gain=S.VOID * fg)
            for name, ep in getattr(orb, "_extra", []):
                # a bloom layer with its OWN colour (`void._bloom`, `violet_ramp`)
                can.lut = lut_violet
                can.deposit(ep, cam, gain=S.VOID * fg)
            rc = getattr(orb, "_react_core", None)
            if rc is not None:
                # the music-reactive field dots that take the NUCLEUS colours
                can.lut = lut_core
                can.deposit(rc, cam, gain=S.VOID * fg)
        gmesh = S.MESH_GAIN * (S.VOID_MESH if vf is not None else 1.0) * fg
        for mp in meshes:
            # no split -> the shipped ramp; split -> the complementary pair, whose sum IS it
            can.lut = lut_mesh if mp.split is None else lut_split[mp.split]
            can.deposit(mp, cam, gain=gmesh)
        # ⚠ THE FIELD LINES GO DOWN AFTER THE WIREFRAME AND BEFORE THE CORE, through their
        # own diverging LUT. Additive blending makes the ORDER irrelevant to the result; what
        # matters is that each pass gets its own colour space, which is the same argument
        # `render.py`'s module docstring makes about the mesh and the nucleus.
        lp = orb.lines_at(t, float(scales[f]), h)
        if lp is not None:
            can.lut = lut_charge
            can.deposit(lp, cam, gain=LINE_GAIN)
        luts = dict(core=lut_core, violet=lut_violet, mesh=lut_mesh, magenta=lut_magenta)
        can.lut = lut_core
        if isinstance(nuc, list):
            for lname, np_ in nuc:
                can.lut = luts[lname]
                can.deposit(np_, cam, gain=fgc)
            can.lut = lut_core
        elif nuc is not None:
            can.deposit(nuc, cam)
        if ele is not None:
            can.deposit(ele, cam)

    # ⚠ PRE-ROLL, and it is not cosmetic. The persistence buffer reaches steady state at
    # `1 / (1 - exp(-h/tau))` = 3.2x a single frame's deposit, so a COLD frame 0 is a third as
    # bright as the same picture one fifth of a second later. It is not subtle: on the first
    # contact sheet the opening `s` sphere looked measurably SMALLER than the identical closing
    # one, and the difference was entirely this. The pre-roll frames are integrated and thrown
    # away, which costs 15 frames and makes frame 0 the frame it is supposed to be.
    t0 = times[0]
    k = int(round(preroll * cfg.fps))
    for i in range(k, 0, -1):
        _step(max(0.0, t0 - i * h))

    for t in times:
        _step(t)
        yield t, can.to_rgb()


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------
def _luma(rgb):
    a = rgb.astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def dry(cfg, n=7, verbose=True):
    """The pre-render gate. There is no reference clip to score against, so - the
    Particle_Fields rule - it may only assert things that are ALWAYS bugs, and it reports
    everything else for a human to read.

    The one check that is specific to THIS engine, and the reason the gate exists at all, is
    OVERFLOW: the brief's single explicit prohibition is orbitals growing past the frame.

    ⚠ It is checked on EVERY frame, on the FULL 160k mesh, at the real camera - not on the
    probe grid, and not by trusting that `fit_scales` was called. Two reasons. The guard solves
    against an 8.7k COARSE mesh, and a coarser sample of the same surface can only UNDERSTATE
    its extent, so the guard's own bound is not a proof about the mesh that actually renders.
    And an overflow is not a look note: one frame with a lobe cut off by the frame edge is the
    single failure this clip was commissioned to avoid, and seven probe frames out of 1,200
    would find it only by luck. On the GPU the whole sweep costs a few seconds.
    """
    dur = S.DURATION
    nfr = int(round(dur * cfg.fps))
    print(f"[dry] orbital  {cfg.width}x{cfg.height} @{cfg.fps}  {dur:.1f}s  {nfr} frames")
    t0 = time.time()
    orb = Orbital(on_gpu=True)
    scales = S.fit_scales(nfr, cfg.fps, (cfg.ref_w, cfg.ref_h), verbose=True)
    print(f"      mesh N={orb.N:,}  electrons={len(orb.events)} (mode {S.ELECTRONS})  "
          f"flow={S.FLOW}{'' if orb.flow_static else ' (rebuild)'}"
          f"{'' if orb.por is None else ' +porosity'}"
          f"{'' if orb.void is None else f' +void({orb.void.n_bars:,}/{orb.void.n_rings:,})'}  "
          f"(setup {time.time() - t0:.1f}s)")
    if orb.mol is not None:
        sc = orb.molsc
        from collections import Counter
        c = Counter(a.elem for a in sc.atoms)
        print(f"      MOLECULE '{S.MOL}'  {len(sc.atoms)} atoms "
              f"({' '.join(f'{k}{v}' for k, v in sorted(c.items()))})  "
              f"{len(sc.bonds)} bonds  scale {float(scales[0]):.1f} px/A  "
              f"(run `cli.py moltest` - `dry` CANNOT see a wrong reaction)")
    if orb.mcloud is not None:
        print(f"      MOL CLOUD {orb.mcloud.n:,} dots "
              f"({_M.MOL_DIRS:,} dirs x {_M.MOL_RAD} layers x {orb.mcloud.n_at} atoms)")
    if orb.cloud is not None:
        print(f"      SHELL CLOUD {orb.cloud.n * 2:,} dots "
              f"({_B.CLOUD_DIRS:,} dirs x {_B.CLOUD_RAD} radial layers x 2 atoms), "
              f"reach {orb.cloud.reach:.2f}R")
    if orb.bond is not None:
        b = orb.bond
        print(f"      BOND (l,m)=({b.l},{b.m}) phase=({b.pa:+.0f},{b.pb:+.0f}) "
              f"{S.BOND.get('kind', '?')}  sep {S.sep_at(0.0):.2f}R -> "
              f"{S.sep_at(dur):.2f}R  (run `cli.py bondtest` - `dry` CANNOT tell "
              f"bonding from antibonding)")

    h = 1.0 / cfg.fps
    probes = [dur * i / float(n) for i in range(n)]
    # every electron peak is a probe too - the flash is the shortest-lived thing in the clip
    # and a uniform probe grid can miss all of them
    probes += [e.peak for e in orb.events[:6]]
    times = sorted(set(round(x, 6) for x in probes + [p + h for p in probes]
                       + [0.0, dur - h]))

    got, sizes = {}, []
    t1 = time.time()
    for t, rgb in frames_of(cfg, times, orb=orb, scales=scales):
        got[t] = rgb
    el = time.time() - t1
    print(f"      {el / len(times) * 1000:.0f} ms/probe -> full render approx "
          f"{el / len(times) * nfr:.0f}s for {nfr} frames")

    fail, warn = [], []

    # ---- motion (computed early: the loop-seam check below is quoted against it) --------
    steps = []
    for t in probes:
        a = min(got, key=lambda x: abs(x - t))
        b = min(got, key=lambda x: abs(x - (t + h)))
        if a != b:
            steps.append(float(np.abs(_luma(got[a]) - _luma(got[b])).mean()))
    if steps and float(np.mean(steps)) < 0.02:
        fail.append(f"consecutive frames are identical (|d| {np.mean(steps):.4f})")
    if steps and max(steps) > 6.0 * (np.median(steps) + 1e-6):
        warn.append(f"one step {max(steps):.2f} vs median {np.median(steps):.2f} - "
                    "an un-eased join? (an electron flash can do this legitimately)")

    # ---- non-finite state ------------------------------------------------------------
    for t in (0.0, dur * 0.5):
        f = min(nfr - 1, int(round(t * cfg.fps)))
        meshes, _n, _e = orb.points_at(t, float(scales[f]), h)
        bad = sum(int((~np.isfinite(_G.asnumpy(m.P))).sum()) for m in meshes)
        if bad:
            fail.append(f"t={t:.2f}s: {bad} non-finite values in the point cloud")

    # ---- THE OVERFLOW CHECK ----------------------------------------------------------
    HX, HY = cfg.ref_w * 0.5, cfg.ref_h * 0.5
    t2 = time.time()
    over = []
    for f in range(nfr):
        t = f / float(cfg.fps)
        cam = S.camera_at(t, ref=(cfg.ref_w, cfg.ref_h))
        meshes, _n, _e = orb.points_at(t, float(scales[f]), h)
        # ⚠ THE MAX OVER EVERY PASS. With a split there are two clouds and only their UNION is
        # what the frame shows; measuring `meshes[0]` would check half the drawn points and
        # pass a frame whose other half is off the edge.
        mx = my = 0.0
        # ⚠ THE VOID FIELD IS DELIBERATELY EXCLUDED FROM THIS CHECK, AND IT IS NOT EXEMPT -
        # it is measured by the OPPOSITE test below. This check exists to prove the ORBITAL is
        # never clipped, which is the one thing the brief banned. The field's whole purpose is
        # to bleed off every edge and fill the frame, so including it here would hard-fail
        # every frame of variation C for doing exactly what it was asked to do. Two different
        # requirements need two different assertions; what is forbidden is dropping one.
        for m in meshes:
            x, y, z = cam.project(m.P)
            # ⚠ MEASURED OVER THE VISIBLE POINTS, NOT EVERY POINT, and cloud mode is why. A
            # shell cloud carries dots out to 2.35x the atom's radius so that a MERGED surface
            # still has dots at it - and all but a shell of them have gain 0 on any given frame.
            # A point that deposits nothing cannot clip the frame, so measuring it would
            # hard-fail every frame for geometry the viewer never sees. For `lines` and `dots`
            # every gain is O(1), so this changes nothing there - verified against the three
            # shipped finals, which still report 0.673W.
            vis = m.gain > (m.gain.max() * np.float32(1e-3))
            if not bool(_G.asnumpy(vis.any())):
                continue
            mx = max(mx, float(_G.asnumpy(abs(x - HX)[vis].max())) / HX)
            my = max(my, float(_G.asnumpy(abs(y - HY)[vis].max())) / HY)
        sizes.append((t, mx, my))
        if mx > 0.985 or my > 0.985:
            over.append((t, mx, my))
    if S.MOL_FILL:
        # ⚠ INVERTED, NOT DISABLED - the standing rule, and the third time this engine has met
        # it (the void field, then the bond set, now this). Clancy asked for the sheet to *"fill
        # the screen and to go a little bit outside the frame"*, which is precisely what the
        # overflow check hard-fails. So the check still runs, unchanged, everywhere else; for
        # this scene the assertion is the OPPOSITE one - the subject must COVER the frame - and
        # a scene that quietly stopped filling it would now fail rather than pass.
        worst_fill = min(min(a, b) for (_t, a, b) in sizes)
        print(f"      MOL FILL: the sheet covers {worst_fill:.3f} of the half-frame at its "
              f"worst frame (asked to exceed 1.0)")
        if worst_fill < 1.0:
            fail.append(f"the sheet does not fill the frame: only {worst_fill:.3f} of the "
                        f"half-frame at its worst, leaving a dead border")
    elif over:
        fail.append(f"{len(over)} frames touch the frame edge, first at t={over[0][0]:.2f}s "
                    f"({over[0][1]:.3f}W {over[0][2]:.3f}H of the half-frame)")
    # ---- THE INVERTED CHECK: the void field must COVER the frame -------------------
    if orb.void is not None:
        cov = []
        for t in probes:
            f = min(nfr - 1, int(round(t * cfg.fps)))
            cam = S.camera_at(t, ref=(cfg.ref_w, cfg.ref_h))
            vf = orb.void_at(t, float(scales[f]), h, cam)
            x, y, _z = cam.project(vf.P)
            cov.append((float(_G.asnumpy(abs(x - HX).max())) / HX,
                        float(_G.asnumpy(abs(y - HY).max())) / HY))
        worst_cov = min(min(c) for c in cov)
        print(f"      void field covers {worst_cov:.3f} of the half-frame at its worst probe "
              f"({orb.void.n_bars:,} bars / {orb.void.n_rings:,} rings)")
        if worst_cov < 1.0:
            fail.append(f"void field does not cover the frame: reaches only "
                        f"{worst_cov:.3f} of the half-frame, leaving a dead border")

    # ---- THE LOOP SEAM ---------------------------------------------------------------
    # ⚠ CHECKED BY RENDERING t=DURATION AND COMPARING IT TO t=0, because "a loop is exact or
    # it is not a loop" and every rate in this engine is independently periodic. A single
    # un-snapped phase - the camera azimuth, one stream, the void crossfade - shows up here as
    # a mean |delta| the eye would read as a jump, and nowhere else in the gate.
    if S.LOOP:
        a = min(got, key=lambda x: abs(x - 0.0))
        seam = None
        for _t, rgb in frames_of(cfg, [dur], orb=orb, scales=scales):
            seam = rgb
        d = float(np.abs(_luma(seam).astype(np.float32)
                         - _luma(got[a]).astype(np.float32)).mean())
        ref_step = float(np.median(steps)) if steps else 1.0
        print(f"      LOOP seam |t=0 vs t={dur:.0f}s| = {d:.3f} "
              f"(a normal frame step is {ref_step:.3f})")
        if d > max(2.0 * ref_step, 0.5):
            fail.append(f"loop seam is visible: |delta| {d:.3f} vs a normal step "
                        f"of {ref_step:.3f} - some rate is not periodic over the clip")

    worst = max(sizes, key=lambda s: max(s[1], s[2]))
    print(f"      widest of ALL {nfr} frames: t={worst[0]:.2f}s at {worst[1]:.3f}W "
          f"{worst[2]:.3f}H of the half-frame (guard box {S.FIT_X:.2f}/{S.FIT_Y:.2f}) "
          f"[{time.time() - t2:.0f}s]")

    # ---- exposure --------------------------------------------------------------------
    for t in probes:
        k = min(got, key=lambda x: abs(x - t))
        lum = _luma(got[k])
        if lum.max() < 24:
            fail.append(f"t={t:.2f}s frame is black (max luma {lum.max():.0f})")
        blown = float((got[k] >= 254).all(axis=2).mean())
        if blown > 0.05:
            fail.append(f"t={t:.2f}s frame is blown ({blown:.1%} clipped)")

    # ---- the readout the source would have printed -----------------------------------
    if verbose:
        print("      choreography:")
        for t in probes[:n]:
            k = min(got, key=lambda x: abs(x - t))
            l, m = S.lm_at(t)
            st = _luma(got[k])
            if orb.mol is not None:
                print(f"        t={t:5.2f}s                      "
                      f"dark {float((st < 8).mean()):4.0%}  luma {st.mean():5.1f}  "
                      f"p99 {np.percentile(st, 99):5.1f}")
                continue
            if orb.bond is not None:
                print(f"        t={t:5.2f}s  sep={S.sep_at(t):4.2f}R           "
                      f"dark {float((st < 8).mean()):4.0%}  luma {st.mean():5.1f}  "
                      f"p99 {np.percentile(st, 99):5.1f}")
                continue
            print(f"        t={t:5.2f}s  l={l:4.2f} m={m:4.2f}   "
                  f"dark {float((st < 8).mean()):4.0%}  luma {st.mean():5.1f}  "
                  f"p99 {np.percentile(st, 99):5.1f}")

    for wmsg in warn:
        print("[dry] warn -", wmsg)
    if fail:
        print("[dry] FAIL")
        for fmsg in fail:
            print("       -", fmsg)
        return False
    print("[dry] ok")
    return True


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------
def stills(cfg, times, prefix="orbital"):
    """Render PNGs. Each `t` gets its own pre-rolled canvas, so a still is exactly the frame
    the full render would write at that instant - a cold still is a different picture."""
    os.makedirs(OUT, exist_ok=True)
    orb = Orbital(on_gpu=True)
    n = int(round(S.DURATION * cfg.fps))
    scales = S.fit_scales(n, cfg.fps, (cfg.ref_w, cfg.ref_h))
    paths = []
    for t in sorted(times):
        for _t, rgb in frames_of(cfg, [t], orb=orb, scales=scales):
            p = os.path.join(OUT, f"{prefix}_t{t:05.2f}.png")
            Image.fromarray(rgb).save(p)
            paths.append(p)
            print("[still]", p)
    return paths


def _hevc(path):
    """True when the video stream is HEVC - it then needs the `hvc1` tag to play on Apple."""
    ff = find_ffmpeg()
    probe = os.path.join(os.path.dirname(ff), "ffprobe") if os.path.dirname(ff) else "ffprobe"
    out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return out.strip() == "hevc"


def mux_audio(video, sy, start=0.0, seconds=None):
    """Put the song under a sync scene's video, SAMPLE-EXACT, from the ONE shared WAV.

    ⚠ `atrim` BY SAMPLE INDEX on PCM, never `-ss` on the mp3 (MUSIC_PLAN S4): the section's
    first sample is `round((trim.start + start) * sr)`, the same arithmetic the analysis and
    the viewer used, so picture frame 0 and audio sample 0 are the same instant.
    ⚠ And not `deliver.py`, which writes a SILENT track on purpose for the stock finals.
    """
    import wave
    with wave.open(sy.doc["wav"], "rb") as w:
        sr = w.getframerate()
    seconds = sy.duration - start if seconds is None else seconds
    a0 = int(round((sy.t0 + start) * sr))
    a1 = a0 + int(round(seconds * sr))
    tmp = video + ".mux.mp4"
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", video, "-i", sy.doc["wav"],
           "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-af", f"atrim=start_sample={a0}:end_sample={a1},asetpts=PTS-STARTPTS",
           "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart",
           *(["-tag:v", "hvc1"] if _hevc(video) else []), tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, video)
    print(f"[mux] {os.path.basename(video)} <- song {sy.t0 + start:.3f}s "
          f"+ {seconds:.3f}s (samples {a0}..{a1})")
    return video


def render(cfg, out=None, seconds=None, start=0.0):
    dur = seconds or S.DURATION
    nfull = int(round(S.DURATION * cfg.fps))
    n = int(round(dur * cfg.fps))
    out = out or os.path.join(OUT, "orbital.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part.mp4"

    import copy
    from . import hq as _HQ
    hq = _HQ.HQ
    sub = int(hq["sub"]) if hq else 1
    bits16 = bool(hq and int(hq.get("bits", 8)) >= 16)
    cfg_r = copy.copy(cfg)
    cfg_r.fps = cfg.fps * sub
    orb = Orbital(on_gpu=True)
    scales = S.fit_scales(nfull * sub, cfg_r.fps, (cfg.ref_w, cfg.ref_h), verbose=True)

    keyframes = f"expr:eq(n,0)+eq(n,{max(n - 1, 0)})"
    if bits16:
        # ⚠ A 10-bit MASTER. The canvas quantises once, to 16 bits; NVENC HEVC Main10 keeps 10.
        enc = ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p6", "-rc", "vbr",
               "-cq", os.environ.get("PF_CQ", "14"), "-b:v", "0", "-maxrate", "300M",
               "-bufsize", "600M", "-g", "60", "-tag:v", "hvc1"]
        in_fmt, out_fmt = "rgb48le", "p010le"
    else:
        enc, in_fmt, out_fmt = encoder_args(), "rgb24", "yuv420p"
    cmd = [find_ffmpeg(), "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", in_fmt,
           "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "-",
           *enc, "-force_key_frames", keyframes,
           "-pix_fmt", out_fmt, "-movflags", "+faststart", part]
    print(f"[render] orbital  {cfg.width}x{cfg.height} @{cfg.fps}  {dur:.1f}s  {n} frames  "
          f"N={orb.N:,} -> {os.path.basename(out)}")

    # ⚠ MOTION BLUR: `sub` samples spread over HALF a frame interval (a 180-degree shutter),
    # CENTRED on the frame's instant, so the picture does not shift against the audio.
    offs = [(-0.25 + 0.5 * (j + 0.5) / sub) / cfg.fps for j in range(sub)] if sub > 1 else [0.0]
    times = [max(0.0, start + i / cfg.fps + o) for i in range(n) for o in offs]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    i = 0
    try:
        accum, got = None, 0
        for _t, rgb in frames_of(cfg_r, times, orb=orb, scales=scales, bits16=bits16):
            if sub > 1:
                accum = rgb.astype(np.float32) if accum is None else accum + rgb
                got += 1
                if got < sub:
                    continue
                dt_ = np.uint16 if bits16 else np.uint8
                rgb = np.rint(accum / np.float32(sub)).astype(dt_)
                accum, got = None, 0
            proc.stdin.write(rgb.tobytes())
            i += 1
            if i % 30 == 0:
                el = time.time() - t0
                sys.stdout.write(f"\r  {i}/{n}  {el / i * 1000:.0f} ms/frame  "
                                 f"eta {el / i * (n - i):5.0f}s   ")
                sys.stdout.flush()
        proc.stdin.close()
        rc = proc.wait()
    except BaseException:
        try:
            proc.stdin.close()
            proc.kill()
        finally:
            if os.path.exists(part):
                os.remove(part)
        raise
    print()
    if rc != 0:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError(f"ffmpeg exited {rc}")
    if os.path.exists(out):
        os.remove(out)
    os.replace(part, out)
    if _SY.ACTIVE is not None:
        mux_audio(out, _SY.ACTIVE, start=start, seconds=dur)
    el = time.time() - t0
    print(f"[done] {out}   {el:.0f}s  ({el / max(n, 1) * 1000:.0f} ms/frame)")
    return out
