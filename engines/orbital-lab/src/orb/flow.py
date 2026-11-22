# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""flow.py - MOVING THE GRID WITHOUT MOVING THE SHAPE.

Added 2026-09-01, and this module is the one Clancy called the crux of the engine: *"the 'grid'
or 'manifold' that covers the orbital shells needs to be somewhat dynamically moving ... make
the grid lines flow in different ways."*

THE ONE RULE THIS MODULE EXISTS TO ENFORCE
------------------------------------------
**Flow moves the GRID. It never moves the SHAPE.** Every point this module produces still
satisfies `r = |Y_lm(theta, phi)|` exactly - it is drawn at a DIFFERENT (theta, phi) on the
same surface, never off it. That is the whole design, and it is the direct descendant of the
lesson that got the electron convergence withdrawn: a treatment that deforms the silhouette
makes the subject a function of the treatment, and the subject here is the orbital. Texture is
allowed to move. The shape is not.

Three consequences fall straight out of that rule, and they are why this was affordable:

1. **The size guard needs no change at all.** The drawn points are a subset of the same
   surface, so the envelope `fit_scales` solves against is the same envelope. (`cli.py dry`
   re-measures the real drawn points on every frame regardless, so this is checked, not
   assumed.)
2. **`_arc_gain` needs no change.** Drift changes the parameter VALUES a curve is sampled at,
   never the block layout it reshapes by, and the spacing it measures is still the real
   spacing of the real curve.
3. **Nothing here is random and nothing here has state.** Every quantity is closed form in `t`,
   like the rest of the engine; per-curve phase offsets come from the curve INDEX via the
   golden ratio, not from an RNG, so a scene is still reproducible from its seed and a seek
   backward is still free.

WHY SLIDING SAMPLES ALONG A CURVE IS A NO-OP, WHICH IS THE NON-OBVIOUS PART
--------------------------------------------------------------------------
The intuitive implementation of "make the lines flow" is to advect the samples along their own
curve. **That renders as absolutely nothing.** A latitude ring is sampled at all 1152 azimuths;
re-parameterising it in phi maps that sample set onto itself, and the accumulator only ever
sees the SET. Sliding a meridian's samples in theta is the same story. So flow has to be one
of exactly three things, and this module implements the two safe ones:

| primitive | what actually moves | safe? |
|---|---|---|
| `stream` | brightness travels along the curve; geometry untouched | yes - pure gain |
| `drift` | the curve family moves TRANSVERSALLY: meridians in phi, rings in theta | yes - stays on the surface |
| *radial ripple* | the surface itself displaces | **NO - deferred.** It moves the shape, and it would need an explicit bound folded into `fit_scales` rather than a hope |

Note the asymmetry in `drift`, because it is the whole trick: a meridian is a curve of constant
phi, so moving it in phi is transverse (visible) and moving it in theta is along itself
(invisible). For a latitude ring it is the other way round. Each family drifts in the parameter
the OTHER family varies.

THE SEPARABILITY THAT MADE DRIFT CHEAP
--------------------------------------
Drift needs `|Y|` at parameters that are not on the precomputed mesh, and re-evaluating an
associated Legendre function for 160k points per frame is not affordable. It is also not
necessary, because for the cos-form real harmonic this file's `harmonics.real_Y` returns,

    |Y_lm(theta, phi)| = |Y_lm(theta, 0)| * |cos(m phi)|

**exactly** - the harmonic is SEPARABLE, and each mesh family is a tensor product of its own
theta and phi samples. So one `(n_theta,)` profile per state plus a closed-form cosine
reproduces the whole amplitude field and evaluates anywhere, and a drifted frame is two outer
products instead of a special-function call.

⚠ THE PROFILE IS TAKEN AT phi = 0 AND THAT IS NOT A SHORTCUT, it is the definition:
`cos(m*0) = 1`, so `real_Y(l, m, theta, 0)` IS the theta profile, sign and the sqrt(2) real-form
factor included.

⚠ AND THE NORMALISATION IS MEASURED ON THE REAL MESH, not taken as `max|P| * 1`. `harmonics.
amplitudes` divides each state by its maximum over the SAMPLED mesh, which is a hair under the
continuous maximum. Reproducing that exactly is the difference between the flow path being a
drop-in for the precomputed one and being a 0.3% brightness change on every state - which
would be invisible in the gate and visible as a look regression next to the shipped clip.

THE HAZARD THIS MODULE PAID FOR: TWO IMPLEMENTATIONS OF THE AXIS SWAP
---------------------------------------------------------------------
`harmonics.directions` maps the quantum z-axis to world +Y. Drift means rebuilding directions
on the device, i.e. writing that mapping a SECOND time - which is exactly the class of bug
`harmonics` S-documents for scipy's argument order: it does not raise, it renders a plausible
wrong orbital. The defence is `selftest()` below: **with every flow amplitude set to zero, the
rebuilt cloud must equal the precomputed one to float32 tolerance.** It is three lines, it runs
in a second, and it catches every error in this class including the ones not thought of yet.
`cli.py flowtest` runs it.

⚠ A KNOWN INTERACTION, RECORDED RATHER THAN FIXED: `electrons="pull"` COMBINED WITH A DRIFT.
`render.Orbital.points_at` locates a lit electron from the UNDRIFTED `A_cpu`/`U_cpu` - the
event carries a mesh INDEX, and an index means a fixed (theta, phi). Under a drift the cloud
is rebuilt at moved parameters while the lit point is not, so `E` stays on the surface but
drifts out of register with the grid line it was sampled on. No scene does this today: the 15
gated scenes inherit flow "none", and the one scene that names a flow has electrons off. It is
left alone because the fix (carrying the drift into the event's parameters) is only meaningful
once a scene actually wants both, and guessing at the semantics now would be building for a
caller that does not exist. **If you write that scene, start here.**
"""
from __future__ import annotations

import numpy as np

from . import harmonics as H

#: Golden-ratio conjugate. Per-curve phase offsets are `i * PHI mod 1`, which spreads any
#: number of curves about as evenly as a low-discrepancy sequence can - the point is that
#: neighbouring curves must NOT share a phase (that reads as one rigid band sweeping the
#: object, and a rigid full-mesh band is a luminance flash by another name) and that no RNG
#: enters an engine whose entire selling point is being closed form in `t`.
PHI = 0.61803398875


# ======================================================================================
# THE FLOW REGISTRY
#
# A preset is a dict of primitives, in exactly the shape of `scene.SCENES` and for the same
# reason. `None` / "none" is the identity and takes the FAST PATH - the precomputed mesh, not
# one operation of this module - so the flow-free clip is bit-for-bit what it was.
#
# ⚠ THE 15 SCENES GATED ON 2026-08-29 NAME NO FLOW and therefore inherit "none". That is by
# construction, not by luck: `configure()` resolves flow from the scene spec, and a spec
# without a `flow` key gets the module default, which is "none". Unlike the electron mode,
# nothing had to be re-pinned - the default IS their behaviour.
#
# keys, all optional, all zero-by-default:
#   mer_drift   revolutions/second the MERIDIAN family turns in phi (transverse -> visible)
#   lat_drift   radians/second the LATITUDE rings march in theta (transverse -> visible)
#   lat_fade    exponent on sin(theta) dimming rings near the poles; insurance against a ring
#               wrapping theta pi->0 reading as a pop. 0 disables.
#   por_amp     0..1 POROSITY: how deep the random transparency dips. See `porosity()`.
#   por_runs    how many random runs along each curve - LOW makes long dashes, high makes fuzz
#   por_point   0..1 per-POINT random opacity. THE DOT-MODE TERM - one node, one transmittance
#   por_curve   0..1 per-CURVE random opacity: whole meridians/rings dim independently. This
#               is the term that actually destroys the enclosing outline - see `porosity()`
#   mer_split   CHROMATIC SPLIT of the meridian family, in radians of phi. See `SPLIT` below
#   lat_split   the same for the latitude rings, in radians of theta
#   split_arc   1 = the split follows the state walk's own clock; 0 = hold it constant
#   stream_mer  (amp, cycles along the curve, Hz) brightness travelling pole-to-pole
#   stream_lat  (amp, cycles around the ring, Hz) brightness travelling azimuthally
# ======================================================================================
FLOWS = {
    "none": {},

    #: THE DELIVERABLE. Both families move, and they move DIFFERENTLY - which is the literal
    #: ask ("make the grid lines flow in different ways"). The meridians turn slowly one way
    #: while the rings march the other, so the grid reads as a woven surface with a current in
    #: it rather than as an object being spun. Rates are deliberately under what looks
    #: impressive on a still frame: on a 20 s clip the drift has to be legible without
    #: competing with the morph, which is the actual subject.
    "weave": dict(mer_drift=0.030, lat_drift=-0.115, lat_fade=0.55,
                  stream_mer=(0.42, 3.0, 0.16), stream_lat=(0.34, 2.0, -0.11)),

    #: Geometry untouched - light only. The cheapest primitive and the safest: nothing moves
    #: off the precomputed mesh, so this takes the fast path too.
    "pulse": dict(stream_mer=(0.55, 4.0, 0.22), stream_lat=(0.45, 3.0, -0.15)),

    #: The meridian family alone, turning. On an m=0 state (an axisymmetric surface) this
    #: reads as the object spinning; on m>0 it reads as spokes sweeping across a stationary
    #: shape, which is the more interesting half.
    "spin": dict(mer_drift=0.075, stream_mer=(0.30, 3.0, 0.12)),

    #: POROUS - the 2026-09-01 second ask. `weave` with the grid broken up so no unbroken line
    #: encloses a lobe. The drift rates are `weave`'s exactly: the two variations differ in ONE
    #: thing, which is the only way a side-by-side judgement means anything.
    "porous": dict(mer_drift=0.030, lat_drift=-0.115, lat_fade=0.55,
                   stream_mer=(0.42, 3.0, 0.16), stream_lat=(0.34, 2.0, -0.11),
                   por_amp=0.80, por_runs=52.0, por_curve=0.72, por_point=0.0),

    #: DOT MODE's porosity. `por_curve` comes right down: dimming a whole ROW of a lattice is
    #: visible banding, where dimming a whole LINE of a wireframe was welcome raggedness.
    #: `por_point` carries it instead - one node, one transmittance, which is what "each unit"
    #: means once there are no connections.
    "porous_dots": dict(mer_drift=0.030, lat_drift=-0.115, lat_fade=0.55,
                        stream_mer=(0.42, 3.0, 0.16), stream_lat=(0.34, 2.0, -0.11),
                        por_amp=0.55, por_runs=44.0, por_curve=0.14, por_point=0.62),

    #: THE REEL. `weave`'s motion plus the two-pass chromatic split, on the state walk's clock.
    "emerge": dict(mer_drift=0.030, lat_drift=-0.115, lat_fade=0.55,
                   stream_mer=(0.42, 3.0, 0.16), stream_lat=(0.34, 2.0, -0.11),
                   por_amp=0.55, por_runs=44.0, por_curve=0.14, por_point=0.62,
                   mer_split=0.024, lat_split=0.020),

    #: The rings alone, marching pole to pole. The strongest "flow" of the three and the one
    #: that most changes where the bright knots sit, because a ring crossing a NODAL CONE
    #: collapses to a point - so the knots travel with the rings.
    "tide": dict(lat_drift=-0.200, lat_fade=0.65, stream_lat=(0.40, 2.0, -0.18)),
}


# ======================================================================================
# THE CHROMATIC SPLIT - the reel translation (2026-09-01)
#
# Reference: an Instagram reel by `decayingsun`, a TouchDesigner capture captioned "Figured out
# how to control the point of emergence". A portrait dissolves into a dense field of fine
# vertical lines with heavy magenta/violet against cyan/green separation, then the comb bends
# into CONCENTRIC RINGS about a centre and the subject dissolves into the interference.
#
# THE TRANSLATION: deposit the SAME wireframe TWICE, at transverse parameter offsets +/-s/2,
# through two LUTs whose sum is exactly `RAMP_MESH`. That is a true chromatic separation - two
# physically displaced copies in different colours - and not a palette shift.
#
# ⚠ WHY THE ORBITAL SURVIVES ANY SPLIT VALUE, which is the whole reason this is allowed: both
# copies still lie on r=|Y|, so the union of them IS the same surface. There is no `s` at which
# a d_xz stops having four lobes. What dissolves is the CRISPNESS of the grid, which is exactly
# the "subject readable through the striping" the reel shows. This is the inverse of the
# electron withdrawal - that made the SUBJECT a function of the treatment; this makes only the
# TEXTURE one.
#
# ⚠ AND THE COHERENCE FIELD IS NOT AUTHORED - it falls out of the geometry. The physical
# separation between the two copies is |dP| ~ r * sin(theta) * s, which vanishes wherever
# r -> 0: every nodal cone, every knot, the origin, and the polar axis. So the nodal structure
# (the thing that distinguishes a d from an f) stays registered and colourless while the lobe
# SKINS fringe. The reel's "point of emergence" is the nucleus, and coherence radiates from it
# for free.
#
# ⚠ THE USEFUL RANGE IS BOUNDED BY THE PITCH, and past it the effect REVERSES. Meridian pitch
# is 2*pi/112 = 0.056 rad, so beyond mer_split ~ 0.028 (half-pitch) the +/- copies re-register
# against their NEIGHBOURS - the comb simply looks shifted and the interference decreases.
# Ring pitch is pi/66 = 0.048, so lat_split tops out ~0.024. At 0.020 the first ring's lower
# copy sits at theta ~ 0.038, comfortably inside the pole epsilon (2e-3), so the wrap and the
# drift-gated `lat_fade` never fire. VERIFIED against `curve_mesh`, not assumed.
#
# ⚠ s = 0 REDUCES TO THE SHIPPED CLIP EXACTLY. The two deposits register, the two LUTs sum to
# RAMP_MESH, and the total flux is unchanged - so `flowtest` extends to the split path for
# free, and the arc below can open and close on a bit-identical frame.
# ======================================================================================
def _smoothstep(x, a, b):
    u = min(max((x - a) / max(b - a, 1e-9), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def split_at(u, sp):
    """(mer, lat) split in radians at normalised clip time `u` in [0,1].

    ⚠ THE ARC IS BOUND TO THE STATE WALK, NEVER TO A THIRD TIMELINE. The correspondence is
    already sitting there: the opening `s` sphere is the reel's clean portrait. So the split
    holds at ZERO through s->p, the MERIDIAN split ramps in across the d states (a comb of
    fine near-vertical striping), crossfades to the LATITUDE split through f and g (the reel's
    "circular modulation" - concentric ring fringing, and f/g is exactly where the rings are
    densest and most structured), then collapses to zero into the closing `s`.

    Because the clip already ends where it began, zero at BOTH ends means the bookend frames
    are bit-identical to the flagship and nothing pops. The split is a treatment OF the morph,
    on the morph's own clock, so it never competes with it for the viewer's attention.
    """
    ms = float(sp.get("mer_split", 0.0))
    ls = float(sp.get("lat_split", 0.0))
    if not float(sp.get("split_arc", 1.0)):
        return ms, ls
    mer = _smoothstep(u, 0.15, 0.35) * (1.0 - _smoothstep(u, 0.50, 0.72))
    lat = _smoothstep(u, 0.45, 0.65) * (1.0 - _smoothstep(u, 0.86, 0.97))
    return ms * mer, ls * lat


def loop_snap(sp, duration, shape, th_step):
    """Round every RATE so the clip is periodic over `duration`. Returns a new spec.

    ⚠ A LOOP IS EXACT OR IT IS NOT A LOOP (OUTPUT_RULES), so this rounds rather than hopes.
    Each quantity has its own period, and they are not the same period:

    * `lat_drift` marches rings in theta and the ring lattice repeats every `th_step`, so the
      drift over the clip must be a whole number of STEPS - not a whole number of laps. That is
      a very fine quantum (one 200th of the span here), so the snapped rate is within a
      fraction of a percent of the one that was art-directed.
    * `mer_drift` turns the meridian family in phi, and that family repeats every `2*pi/n_mer`,
      so the same argument applies with the column spacing as the quantum.
    * `stream_*` are phases in Hz: a whole number of cycles over the clip.
    * The SPLITS need no snapping - `split_at` is already zero at both ends by construction.

    Anything that rounds to zero is pinned to one quantum instead, because silently switching
    a treatment off is a worse failure than a slightly wrong rate.
    """
    out = dict(sp)
    n_mer = max(int(shape.get("n_mer", 0)), 0)

    def snap(v, quantum):
        if not v:
            return v
        k = round(float(v) * duration / quantum)
        if k == 0:
            k = 1 if v > 0 else -1
        return k * quantum / duration

    if out.get("lat_drift"):
        out["lat_drift"] = snap(out["lat_drift"], th_step)
    if out.get("mer_drift") and n_mer:
        out["mer_drift"] = snap(out["mer_drift"], 1.0 / n_mer)
    for k in ("stream_mer", "stream_lat"):
        if out.get(k):
            a, c, hz = (float(x) for x in out[k])
            out[k] = (a, c, snap(hz, 1.0))
    return out


def has_split(sp) -> bool:
    return (abs(float(sp.get("mer_split", 0.0))) > 0.0
            or abs(float(sp.get("lat_split", 0.0))) > 0.0)


def spec(name):
    """Resolve a preset name (or a dict, or None) to a flow spec. Unknown names raise."""
    if name is None or name == "none":
        return {}
    if isinstance(name, dict):
        return dict(name)
    if name not in FLOWS:
        raise ValueError(f"flow={name!r}; expected one of {tuple(FLOWS)}")
    return dict(FLOWS[name])


def is_static(sp) -> bool:
    """True when the spec needs no geometry rebuild, so the caller can use the fast path.

    ⚠ STREAMS DO NOT COUNT AS MOTION HERE. `stream_*` is a pure gain on the existing points,
    so a stream-only preset still uses the precomputed mesh and costs one sine per point. Only
    a DRIFT moves a sample off the mesh it was built on.
    """
    return not (abs(float(sp.get("mer_drift", 0.0))) > 0.0
                or abs(float(sp.get("lat_drift", 0.0))) > 0.0
                # ⚠ A SPLIT REBUILDS GEOMETRY EVEN AT ZERO DRIFT - it IS a transverse offset,
                # just a signed one. Forgetting this returns the precomputed mesh for both
                # passes, which registers them perfectly and renders the un-split clip in
                # two half-bright halves: no fringing, and no error anywhere.
                or has_split(sp))


class Manifold:
    """The separable rebuild of the wireframe, evaluable at drifted parameters.

    Built once from the same mesh `harmonics.curve_mesh` produced, so the two paths cannot
    disagree about layout. `rebuild()` is a pure function of `(i, j, w, t)`.
    """

    def __init__(self, th, ph, shape, xp, asarray, states=None):
        self.xp = xp
        self.sh = dict(shape)
        st = list(states if states is not None else H.STATES)
        self.states = st
        n_mer, n_th = self.sh["n_mer"], self.sh["n_th"]
        n_lat, n_ph = self.sh["n_lat"], self.sh["n_ph"]
        n1 = self.sh["n1"]

        # ---- the four parameter axes, recovered from the flat mesh ----------------------
        # ⚠ RECOVERED, not recomputed. Re-deriving `linspace(eps, pi-eps, n_th)` here would be
        # a second definition of the mesh that has to be kept in step with `curve_mesh` by
        # hand, and the failure mode is a silent half-sample offset between the two paths.
        # ⚠ THE MERIDIAN FAMILY MAY BE EMPTY - that is DOT MODE, where the latitude family
        # alone carries the whole lattice. Every meridian-side term below is then a no-op, and
        # the pole bound has to come from the named constant rather than from `th1[0]`, which
        # does not exist. (The naive `ph[:n1:n_th]` is also a zero-step slice and raises.)
        self.has_mer = n_mer > 0 and n_th > 0
        if self.has_mer:
            self.th1 = np.asarray(th[:n_th], np.float64)                # meridian theta
            self.ph1 = np.asarray(ph[:n1:n_th], np.float64)             # meridian phi
            assert self.ph1.size == n_mer
        else:
            self.th1 = np.zeros(0)
            self.ph1 = np.zeros(0)
        self.th2 = np.asarray(th[n1::n_ph], np.float64)                 # latitude theta
        self.ph2 = np.asarray(ph[n1:n1 + n_ph], np.float64)             # latitude phi
        assert self.th2.size == n_lat

        # ⚠ THE WRAP SPAN IS THE RING LATTICE'S OWN PERIOD, not the pole-epsilon band, and
        # that is what makes a marching family able to LOOP. Wrapping over `[eps, pi-eps]`
        # while the rings are spaced `pi/(n_lat+1)` is wrapping over a span that is not a whole
        # number of spacings, so after one lap the set lands BETWEEN its old positions and the
        # picture never repeats. Wrapping over exactly `n_lat` spacings means a drift of one
        # spacing maps the set onto itself, so any integer number of spacings is a perfect
        # loop. The lowest position is `th2[0]` = pi/(n_lat+1), comfortably outside POLE_EPS.
        self.th_step = float(self.th2[1] - self.th2[0]) if n_lat > 1 else float(np.pi)
        self.th_lo = float(self.th2[0])
        self.th_span = self.th_step * n_lat
        self.th_hi = self.th_lo + self.th_span

        # ---- the separable pieces, per state --------------------------------------------
        # f(theta) = |Y(theta, 0)| ; |Y(theta, phi)| = f(theta) * |cos(m phi)|
        n_lut = 8192
        self.th_lut = np.linspace(self.th_lo, self.th_hi, n_lut)
        f1, flut, cph1, cph2, norms, ms = [], [], [], [], [], []
        for (l, m, _n) in st:
            f_mer = np.abs(H.real_Y(l, m, self.th1, np.zeros_like(self.th1)))
            f_lut = np.abs(H.real_Y(l, m, self.th_lut, np.zeros_like(self.th_lut)))
            f_lat = np.abs(H.real_Y(l, m, self.th2, np.zeros_like(self.th2)))
            c1 = np.abs(np.cos(m * self.ph1))
            c2 = np.abs(np.cos(m * self.ph2))
            # ⚠ THE NORMALISATION IS THE MAX OVER THE REAL MESH, matching `H.amplitudes`
            # exactly. Both families see the same surface, so the max is the larger of the
            # two blocks' maxima - and it is NOT `f.max() * 1.0`, because no sampled phi need
            # land on a maximum of |cos(m phi)|.
            # ⚠ THE MAX OVER WHICHEVER FAMILIES EXIST. In dot mode the meridian block is
            # empty and `.max()` on a zero-size array raises - and the latitude block is then
            # the whole mesh anyway, so it alone defines the normalisation. Guarding here
            # rather than seeding with 0.0 keeps the lines-mode value bit-identical.
            cands = [float((f_lat[:, None] * c2[None, :]).max()), 1e-12]
            if f_mer.size and c1.size:
                cands.append(float((f_mer[None, :] * c1[:, None]).max()))
            nrm = max(cands)
            f1.append(f_mer)
            flut.append(f_lut)
            cph1.append(c1)
            cph2.append(c2)
            norms.append(nrm)
            ms.append(int(m))
        self.m = np.asarray(ms, np.int32)
        self.norm = np.asarray(norms, np.float64)
        self.f_mer = asarray(np.asarray(f1, np.float32))                # (S, n_th)
        self.f_lut = asarray(np.asarray(flut, np.float32))              # (S, n_lut)
        self.c_ph2 = asarray(np.asarray(cph2, np.float32))              # (S, n_ph)
        self.g_th_lut = asarray(self.th_lut.astype(np.float32))
        self.d_ph1 = asarray(self.ph1.astype(np.float32))
        self.d_ph2 = asarray(self.ph2.astype(np.float32))
        self.d_st1 = asarray(np.sin(self.th1).astype(np.float32))
        self.d_ct1 = asarray(np.cos(self.th1).astype(np.float32))
        self.d_cos_ph2 = asarray(np.cos(self.ph2).astype(np.float32))
        self.d_sin_ph2 = asarray(np.sin(self.ph2).astype(np.float32))

    # ---- the per-frame rebuild ---------------------------------------------------------
    def rebuild(self, i, j, w, t, sp, sign=0, u=0.0):
        """(U, r, fade) at the drifted parameters. `U` is (N,3) unit, `r` is (N,) amplitude.

        `fade` is None or an (N,) multiplier - the near-pole ring dimming, which is the one
        piece of this that is a gain rather than a geometry.
        """
        xp = self.xp
        n_mer, n_th = self.sh["n_mer"], self.sh["n_th"]
        n_lat, n_ph = self.sh["n_lat"], self.sh["n_ph"]
        f32 = np.float32

        ms, ls = split_at(u, sp)
        d_phi = f32(2.0 * np.pi * float(sp.get("mer_drift", 0.0)) * t + sign * 0.5 * ms)
        d_th = float(sp.get("lat_drift", 0.0)) * t + sign * 0.5 * ls

        # ---- meridians: theta fixed, phi drifts (TRANSVERSE) ---------------------------
        # ⚠ SKIPPED ENTIRELY IN DOT MODE, where this family is empty and the lattice lives in
        # the latitude block alone. Building it anyway would be a (0, 0, 3) reshape and a
        # zero-length concatenate - harmless today, and exactly the sort of thing that starts
        # raising the moment someone indexes `[0]` for a diagnostic.
        if self.has_mer:
            ph1 = self.d_ph1 + d_phi
            A_mer = 0.0
            for s, ww in ((i, 1.0 - w), (j, w)):
                if ww == 0.0:
                    continue
                c = xp.abs(xp.cos(f32(self.m[s]) * ph1))                # (n_mer,)
                A_mer = A_mer + f32(ww / self.norm[s]) * (c[:, None] * self.f_mer[s][None, :])
            st1, ct1 = self.d_st1, self.d_ct1
            cp1, sp1 = xp.cos(ph1), xp.sin(ph1)
            U_mer = xp.empty((n_mer, n_th, 3), np.float32)
            # ⚠ THE SAME AXIS SWAP `harmonics.directions` MAKES: quantum z -> world +Y.
            # Written a second time here because the device rebuild cannot call the numpy one;
            # `selftest()` is what keeps the two honest.
            U_mer[:, :, 0] = cp1[:, None] * st1[None, :]
            U_mer[:, :, 1] = xp.broadcast_to(ct1[None, :], (n_mer, n_th))
            U_mer[:, :, 2] = sp1[:, None] * st1[None, :]
            mer_U = U_mer.reshape(-1, 3)
            mer_r = xp.asarray(A_mer, np.float32).reshape(-1)
        else:
            mer_U = xp.empty((0, 3), np.float32)
            mer_r = xp.empty(0, np.float32)

        # ---- latitudes: phi fixed, theta marches (TRANSVERSE) --------------------------
        # wrapped INSIDE the pole epsilon, never across it
        th2 = self.th_lo + np.mod((self.th2 - self.th_lo) + d_th, self.th_span)
        g_th2 = xp.asarray(th2.astype(np.float32))
        A_lat = 0.0
        for s, ww in ((i, 1.0 - w), (j, w)):
            if ww == 0.0:
                continue
            f = xp.interp(g_th2, self.g_th_lut, self.f_lut[s])          # (n_lat,)
            A_lat = A_lat + f32(ww / self.norm[s]) * (f[:, None] * self.c_ph2[s][None, :])
        st2, ct2 = xp.sin(g_th2), xp.cos(g_th2)
        U_lat = xp.empty((n_lat, n_ph, 3), np.float32)
        U_lat[:, :, 0] = st2[:, None] * self.d_cos_ph2[None, :]
        U_lat[:, :, 1] = xp.broadcast_to(ct2[:, None], (n_lat, n_ph))
        U_lat[:, :, 2] = st2[:, None] * self.d_sin_ph2[None, :]

        U = xp.concatenate([mer_U, U_lat.reshape(-1, 3)])
        r = xp.concatenate([mer_r, xp.asarray(A_lat, np.float32).reshape(-1)])

        fade = None
        fe = float(sp.get("lat_fade", 0.0))
        if fe > 0.0 and abs(float(sp.get("lat_drift", 0.0))) > 0.0:
            # ⚠ GATED ON THE DRIFT BEING ON. A ring that never moves must not be dimmed - that
            # would be a look change to a still frame, and this whole feature is supposed to
            # be invisible when it is switched off.
            #
            # Insurance, not the mechanism: a ring approaching a pole ALREADY collapses to a
            # point (its radius goes as sin theta) and `_arc_gain` already dims it as its
            # spacing shrinks, so the birth and death are smooth on geometry alone. This just
            # guarantees it, because "nothing enters by popping into existence" is a hard rule
            # and an accumulator is unforgiving of a bright point appearing at a pole.
            fl = xp.maximum(xp.sin(g_th2), f32(1e-4)) ** f32(fe)
            fade = xp.concatenate([xp.ones(n_mer * n_th, np.float32),
                                   xp.broadcast_to(fl[:, None], (n_lat, n_ph)).reshape(-1)])
        return U, r, fade

    # ---- POROSITY: the grid stops being a set of solid lines ---------------------------
    def porosity(self, sp, seed=20260901):
        """(N,) static transmittance per point, MEAN 1, or None.

        THE ASK, in Clancy's words: *"the 'ring' shape (from the viewer's perspective) is
        completely solid ... what we're going for is that the grid, and its lines (but more
        specifically each unit), are partially transparent, randomly. This way there will not
        be a solid 'line' that encloses every cloud, but rather a bit of a fuzzy layer."*

        WHAT THE SOLID RING ACTUALLY IS, because that determines the fix. It is the LIMB - the
        locus where the surface turns away from the camera. There the curves are seen edge-on,
        so a whole band of them projects into a few pixels and the additive accumulator sums
        them into one hard bright outline around every lobe. It is not a line anyone drew; it
        is a density artefact, and that is why no exposure or colour change removes it.

        ⚠ SO THE FIX HAS TO BREAK THE CURVES, NOT DIM THEM. Dimming the whole mesh keeps the
        limb the brightest thing in the frame - it just moves everything down together and the
        outline survives, now on a deadened body. What removes an outline is making it
        DISCONTINUOUS: each curve becomes a run of dashes with random gaps, and because every
        curve's gaps fall in a different place, the limb becomes a dense scatter of short
        segments - a fuzzy layer - instead of a continuous edge.

        ⚠ RUNS, NOT PER-POINT NOISE. Independent noise per sample is the obvious
        implementation and it is wrong twice over: at sub-pixel sample spacing it averages
        straight back out to a uniform dimming inside a single pixel (so the outline returns),
        and what little survives reads as film grain rather than as a porous grid. The mask is
        therefore a SMOOTH random field along the curve - `por_runs` control points per curve,
        interpolated - so the transparent stretches are longer than a pixel and read as gaps.

        ⚠ MEAN 1 BY CONSTRUCTION, like `stream_gain` and for the same reason: this is a look
        change, not an exposure change. The mask is divided by its own mean, so the light the
        gaps remove is handed back to the segments that survive. Without that, `por_amp` 0.82
        is a two-thirds cut to the whole clip and every judgement made next to it is really a
        judgement about brightness.

        ⚠ AND IT IS INDEXED BY CURVE AND POSITION-ALONG-CURVE, so it is ATTACHED TO THE GRID
        and travels with it under a drift. A mask in world or screen space would be a texture
        the orbital slides underneath, which reads as dirt on the lens.

        Deterministic from `seed`: precomputed once on the CPU, no per-frame RNG - the engine's
        closed-form-in-t property is not negotiable.
        """
        xp = self.xp
        amp = float(sp.get("por_amp", 0.0))
        if amp <= 0.0:
            return None
        runs = max(float(sp.get("por_runs", 24.0)), 2.0)
        n_mer, n_th = self.sh["n_mer"], self.sh["n_th"]
        n_lat, n_ph = self.sh["n_lat"], self.sh["n_ph"]
        rng = np.random.default_rng(seed)

        def field(n_curves, n_samp, wrap):
            """A smooth random field per curve: `k` control points, cosine-interpolated.

            `wrap` closes the field on itself for the latitude family, whose curves are closed
            - without it every ring carries a visible seam at phi=0 where the noise jumps, and
            112 of those seams line up into exactly the kind of artificial meridian this
            feature exists to remove.
            """
            k = max(int(round(runs)), 2)
            c = rng.random((n_curves, k + 1)).astype(np.float64)
            if wrap:
                c[:, -1] = c[:, 0]
            u = np.linspace(0.0, k, n_samp, endpoint=False if wrap else True)
            i0 = np.clip(u.astype(np.int64), 0, k - 1)
            f = (u - i0)[None, :]
            f = 0.5 - 0.5 * np.cos(np.pi * f)                     # smoothstep between knots
            a = np.take_along_axis(c, np.broadcast_to(i0[None, :], (n_curves, n_samp)), 1)
            b = np.take_along_axis(c, np.broadcast_to(i0[None, :] + 1, (n_curves, n_samp)), 1)
            return a + (b - a) * f

        m1 = field(n_mer, n_th, False)
        m2 = field(n_lat, n_ph, True)
        mask = np.concatenate([m1.reshape(-1), m2.reshape(-1)])
        # ⚠ SHAPED, not used raw. A uniform field spends most of its range in the middle, which
        # is a haze; a curve pushed toward the ends spends its time either OPEN or CLOSED,
        # which is a dash. `smoothstep` twice is enough to do that without hard edges.
        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = 1.0 - amp * (1.0 - mask)

        # ⚠ THE PER-CURVE TERM IS THE ONE THAT ACTUALLY REMOVES THE OUTLINE, and it was added
        # after LOOKING at the first pass. Dashing each curve broke the latitude rings nicely
        # and left the enclosing edge of every lobe as solid as before - because that edge is
        # not one curve, it is the ENVELOPE OF MANY, and gaps punched independently in 112
        # equally bright meridians are covered by their neighbours. Giving each whole curve its
        # own opacity makes the family itself ragged: the envelope is now assembled from lines
        # of visibly different strength, so it reads as a layer of depth rather than as a
        # drawn boundary. Dashes give the texture; THIS gives the fuzz.
        # ⚠ THE PER-POINT TERM IS THE DOT-MODE ANALOGUE, and it is the one term that was WRONG
        # for lines and is RIGHT for dots. On a line, independent per-sample noise averages
        # back out inside a single pixel (see above) - which is why the line version needed
        # runs. A dot IS one sample and one pixel-ish blob, so "each unit partially
        # transparent, randomly" is literally per-point here, with nothing to average against.
        pt = float(sp.get("por_point", 0.0))
        if pt > 0.0:
            mask = mask * (1.0 - pt * rng.random(mask.size))

        cur = float(sp.get("por_curve", 0.0))
        if cur > 0.0:
            c1 = rng.random(n_mer)[:, None].repeat(n_th, 1).reshape(-1)
            c2 = rng.random(n_lat)[:, None].repeat(n_ph, 1).reshape(-1)
            cv = np.concatenate([c1, c2])
            mask = mask * (1.0 - cur * (1.0 - cv))
        mask = mask / max(mask.mean(), 1e-6)
        mask = mask / max(mask.mean(), 1e-6)
        # ⚠ NO ANIMATION KNOB HERE ON PURPOSE. The mask is STATIC and attached to the grid, so
        # under a drift the fuzz travels with the lines it belongs to and the surface reads as
        # a porous material. Crawling the pattern along the curve as well was tried on paper
        # and dropped: at these sample densities it is a shimmer, and a shimmering wireframe is
        # the sparkle Particle_Fields S7 spent a pass removing.
        return xp.asarray(mask.astype(np.float32))

    # ---- the travelling brightness -----------------------------------------------------
    def stream_gain(self, t, sp, th_dev, ph_dev):
        """(N,) multiplicative gain, MEAN 1 BY CONSTRUCTION, or None.

        ⚠ THE UNIT MEAN IS THE WHOLE SAFETY ARGUMENT. Exposure on this engine was settled by
        eye against the source and must not move, so the modulation is `1 + a sin(...)`: `sin`
        has zero mean over the curve, so the total flux the accumulator receives is unchanged
        and only its DISTRIBUTION travels. A one-sided envelope (`1 + a * bump`) would raise
        the mean and re-look the clip, which is the mistake this comment exists to prevent.

        The two families get different wavelengths, different rates AND OPPOSITE SIGNS,
        because "flow in different ways" is the ask - a single global drift direction reads as
        the object turning, not as the grid flowing over it.
        """
        xp = self.xp
        f32 = np.float32
        n_mer, n_th = self.sh["n_mer"], self.sh["n_th"]
        n_lat, n_ph = self.sh["n_lat"], self.sh["n_ph"]
        n1 = self.sh["n1"]
        sm = sp.get("stream_mer")
        sl = sp.get("stream_lat")
        if not sm and not sl:
            return None
        g = xp.ones(th_dev.size, np.float32)
        if sm and n1 > 0:
            a, k, hz = (float(x) for x in sm)
            # phase per MERIDIAN, from its index - see PHI
            idx = xp.arange(n_mer, dtype=np.float32) * f32(PHI)
            ph = (idx - xp.floor(idx))[:, None]
            u = th_dev[:n1].reshape(n_mer, n_th) / f32(np.pi)
            g[:n1] = (f32(1.0) + f32(a) * xp.sin(
                f32(2.0 * np.pi) * (f32(k) * u - f32(hz * t) + ph))).reshape(-1)
        if sl:
            a, k, hz = (float(x) for x in sl)
            idx = xp.arange(n_lat, dtype=np.float32) * f32(PHI)
            ph = (idx - xp.floor(idx))[:, None]
            u = ph_dev[n1:].reshape(n_lat, n_ph) / f32(2.0 * np.pi)
            g[n1:] = (f32(1.0) + f32(a) * xp.sin(
                f32(2.0 * np.pi) * (f32(k) * u - f32(hz * t) + ph))).reshape(-1)
        return g


# --------------------------------------------------------------------------------------
# THE TEST THAT KEEPS THE TWO PATHS HONEST
# --------------------------------------------------------------------------------------
def selftest(verbose=True):
    """At ZERO drift the rebuilt cloud must equal the precomputed one.

    ⚠ THIS IS THE ONLY THING STANDING BETWEEN THIS MODULE AND THE CLASS OF BUG THAT COST S7 A
    PASS. `rebuild` writes the quantum-z -> world-+Y axis swap a second time and reconstructs
    the amplitude field from a separable identity; either can be wrong in a way that renders a
    perfectly plausible, completely different orbital, and no exposure or overflow statistic
    would notice. Equality at zero drift exercises the separability, the normalisation, the
    axis mapping, the block layout and the parameter recovery in three lines.

    It is checked at a BLENDED instant (w strictly between 0 and 1) as well as on a pure
    state, because the blend is the one place the two paths could agree per-state and still
    disagree - the states carry different `m`, so the separable form has to be blended after
    the product, never before.
    """
    th, ph = H.curve_mesh()
    U = H.directions(th, ph)
    A = H.amplitudes(th, ph)
    man = Manifold(th, ph, H.SHAPE, np, lambda a: np.asarray(a))
    worst = 0.0
    for (i, j, w) in ((0, 1, 0.0), (3, 4, 0.5), (9, 10, 0.37), (12, 13, 1.0)):
        rU, rr, fade = man.rebuild(i, j, w, 3.7, {})       # t is arbitrary: no drift, no use
        ref = (1.0 - w) * A[i] + w * A[j]
        du = float(np.abs(rU - U).max())
        da = float(np.abs(rr - ref).max())
        worst = max(worst, du, da)
        assert fade is None, "fade must be off when the drift is off"
        assert du < 2e-6, f"directions diverge at ({i},{j},{w}): {du:.2e}"
        assert da < 3e-5, f"amplitudes diverge at ({i},{j},{w}): {da:.2e}"
    if verbose:
        print(f"[flow] zero-drift equality ok - worst |delta| {worst:.2e} "
              f"over {A.shape[0]} states, N={U.shape[0]:,}")
    return True
