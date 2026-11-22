# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""void.py - THE NEGATIVE SPACE INVERTED. Variation C, 2026-09-01.

Clancy: *"the use of negative space REVERSED (i.e. its the UNOCCUPIED space that primarily has
the effect, and its the lack or major reduction of the effect that reveals the shape of the
orbital)."* Then, on the first finals: *"I was hoping that the use of negative space would
extend to the whole frame of the video! Right now, C only has a small portion of the frame
illuminated, it should extend to the whole screen and should be somewhat vertically offset, so
that the glow from the orbitals is still viewable."*

So the treatment fills the space the orbital does not occupy - THE WHOLE FRAME - and the
orbital is a HOLE in it.

WHY THIS IS STILL "FLOW MOVES THE GRID, NEVER THE SHAPE"
--------------------------------------------------------
The field is NEW GEOMETRY that is not the orbital. `r = |Y|` is still evaluated exactly, and
the field is defined BY the surface - never the surface by the field.

⚠ THE CARVE MARGIN IS ZERO AND THE TRANSITION IS SYMMETRIC ABOUT r = |Y|. Carving
`r < |Y|*(1+m)` for any m > 0 inflates the apparent silhouette by m - and then the silhouette
IS a function of the treatment, which is the electron withdrawal all over again. Widen
`VOID_EDGE` (a symmetric softness) if the rim is harsh; never bias the centre.

⚠ SOLID INTERIOR, NOT A SHELL BAND. A band leaves the field lit on both sides, and a dark shell
over a bright interior projects as a faint veil with soft rims - a contour drawing. Interior
suppression projects as a HARD SILHOUETTE, and the silhouette is what identifies an orbital.
Better than it sounds, too: the lobes are SEPARATE solids, so the field reaches in through
every nodal plane and the lobe count and nodal structure are delivered by the bright field
BETWEEN the lobes.

⚠ A SCREEN-FACING CARD, NOT A BALL - this is what makes the hole read at all. The first build
filled a ball of radius 1.45 R and the orbital did NOT read as a hole: a ray through the
silhouette is carved where it crosses the surface, but it still crosses a ball's worth of
UNCARVED field in front of and behind it, and those deposits fill the hole straight back in.
The carve is a volume operation; the frame is a projection; a deep field defeats it. The
lattice is therefore built in CAMERA space and re-oriented every frame. `VOID_SLAB` is the
crux knob: 0.045 is Clancy's *"much more shallow, it should read like a flat card"*.

That also made the reference translate properly rather than approximately. In camera space
`bars` ARE the reel's fine vertical stripes and `rings` ARE its concentric circular modulation
about a screen point.

⚠ LAYER COUNT AND LATTICE PITCH BOTH FOLLOW SLAB DEPTH, and getting it wrong washed the first
flat-card build into a smooth glowing disc with no dots in it. At 0.045 the depth layers sit
within a pixel of each other, so they stop SPREADING the dots and start STACKING them -
several stacked layers on a fine lattice is one bright layer whose glows merge. A card wants
FEW layers and a pitch comfortably wider than `point_sigma`. `VOID_PITCH` is quoted in
REFERENCE px for exactly that reason: it is the number that has to be compared against
`cfg.point_sigma` (2.2 ref px in dot mode), and both scale with the output tier together, so
the ratio - and therefore the look - is resolution independent.

⚠ THE FIELD IS SIZED IN THE FRAME, NOT IN `R`, AND THE GATE IS WHAT PROVED IT HAD TO BE. The
first build scaled the lattice by the frame's own world scale `R`, and `dry` hard-failed at
1.672 W of the half-frame. The reasoning that looked sound and was not: "the widest frame
reaches 0.673 W, so a field at 1.45 R reaches 0.976". That silently assumes the widest-extent
frame is also the largest-`R` frame. It is not. `fit_scales` returns `R` over a 220..367 range
and it grows PRECISELY for the states whose projected footprint is small - that is what
`FIT_BETA` is for (S5) - so the field ballooned exactly on the frames where the orbital was
compact. Everything here is now quoted in HALF-FRAME WIDTHS and one solve converts that to
world units against the real camera. The consequence is a decision, not an accident: THE FIELD
IS A STABLE FRAME-FILLING CARD AND THE HOLE IS WHAT CHANGES SIZE.

⚠ IT COVERS THE WHOLE FRAME, WHICH MEANS IT MUST REACH THE FRAME EDGE ON PURPOSE - and that is
the exact condition `cli.py dry`'s overflow check hard-fails. The check is NOT disabled: it
still runs, unchanged, on the wireframe passes, because its job is to prove the ORBITAL is not
clipped. The field is measured by its own INVERTED test instead - `render.dry` asserts that it
COVERS the frame. A background whose purpose is to bleed off every edge and a subject that must
never touch one are two different requirements, and they need two different assertions.

⚠ THE FALLOFF CENTRE IS OFFSET VERTICALLY, and that is the answer to "so that the glow from the
orbitals is still viewable". A frame-filling field centred on the orbital swamps it - the
orbital's own bloom has nowhere to be seen against. Offsetting the field's bright centre up by
`VOID_OFFSET` half-widths leaves the lower part of the frame dimmer, so the orbital's magenta
core and its halo still read. It is also what stops a full-frame field being a flat lit
rectangle: the brightness gradient IS the composition.

⚠ ITS TOTAL FLUX IS STATIC IN `t`. A field covering the whole frame IS full-frame luminance, so
ramping its overall level would be the banned flash with extra steps. The arc is carried by the
lattice crossfade and the chromatic split, never by brightness - and the crossfade uses
COMPLEMENTARY gains (`1-mix` and `mix`) so the two lattices sum to a constant. (The radial
falloff is a fixed spatial envelope, not a time-varying modulation, so the unit-mean rule does
not apply to it: that rule governs things that change with `t`.)

⚠ AND `scene.fit_scales` IS LEFT ALONE, STILL SOLVING AGAINST THE SURFACE. Re-solving it
against the field would make it bind on nearly every frame, and per development notes a guard that
binds everywhere stops being a clamp and becomes an auto-zoom. Leaving it makes `R` - and
therefore the orbital's framing - identical across all three variations, which is the only
reason a side-by-side judgement means anything.
"""
from __future__ import annotations

import numpy as np

from .flow import PHI, _smoothstep

#: EVERY LENGTH BELOW IS IN HALF-FRAME WIDTHS. 1.0 = half the reference frame's width, and
#: `_solve_radius` converts that to world units once, against the real camera.

#: How far past the frame the lattice is generated. The frame's own half-height is
#: `ref_h/ref_w` = 1.778 half-widths, so the rectangle is much taller than it is wide.
VOID_COVER = 1.06
#: Vertical offset of the field's bright centre, in half-widths. See the offset note.
VOID_OFFSET = 0.42
#: The falloff, measured from the offset centre: full inside `IN`, zero past `OUT`. `OUT` sits
#: past the far corner on purpose, so no part of the frame goes fully black.
VOID_FALL_IN = 1.05
VOID_FALL_OUT = 2.95
#: Lattice pitch in REFERENCE px - the number to compare against `cfg.point_sigma` (2.2).
#:
#: 5.8 as of 2026-09-01, after Clancy asked for the resolution to come back roughly halfway from
#: the flat card (6.5) toward the original deep slab. It is bounded from BELOW by legibility,
#: not by cost: probed at 4.4 / 5.8 / 7.2 on one still, 4.4 renders as a completely flat wash
#: with no dots in it at all, 7.2 is plainly dotted, 5.8 is the finest pitch that still carries
#: visible texture. The binding ratio is pitch/point_sigma and BOTH scale with the output tier,
#: so this reads identically at 720 and at 4K - it cannot be fixed by rendering larger.
VOID_PITCH = 5.8
#: Set by `render.config` for the HQ "fine" master only (`orb/hq.py`); 1.0 everywhere else.
PITCH_SCALE = 1.0
#: HALF-DEPTH of the screen-facing slab, in half-widths. Clancy's "flat card".
VOID_SLAB = 0.045
#: HALF-width of the carve transition, in units of R. SYMMETRIC about the surface.
VOID_EDGE = 0.020
#: Outward falloff of the rim key, in units of R.
VOID_FADE = 0.35


class VoidField:
    """A frame-filling lattice of dots, carved by |Y|, rebuilt into the camera basis.

    Two lattices, crossfaded across the arc with complementary gains:

    * `bars`  - a rectangular dot grid in the screen plane (the reel's straight comb).
    * `rings` - concentric circles of dots about the OFFSET centre (its circular modulation).

    Both are static point sets in half-width units, built once. Points are never compacted -
    the carve sets `gain -> 0` - so every buffer stays a fixed size for the whole clip, which
    is what `Canvas.deposit` already assumes.
    """

    def __init__(self, man, xp, asarray, cam, ref, n_z=2):
        self.man = man
        self.xp = xp
        AR = float(ref[1]) / float(ref[0])          # half-height, in half-widths
        HX, HY = float(VOID_COVER), float(VOID_COVER) * AR
        pitch = float(VOID_PITCH) * float(PITCH_SCALE) / (0.5 * float(ref[0]))  # -> half-widths
        zs = (((np.arange(n_z) + 0.5) / n_z * 2.0 - 1.0) * float(VOID_SLAB)
              if n_z > 1 else np.zeros(1))

        def layered(XY):
            n = XY.shape[0]
            P = np.empty((n * n_z, 3), np.float64)
            P[:, 0] = np.repeat(XY[:, 0], n_z)
            P[:, 1] = np.repeat(XY[:, 1], n_z)
            P[:, 2] = np.tile(zs, n)
            return P

        nx = max(int(round(2.0 * HX / pitch)), 4)
        ny = max(int(round(2.0 * HY / pitch)), 4)
        gx = (np.arange(nx) + 0.5) / nx * 2.0 * HX - HX
        gy = (np.arange(ny) + 0.5) / ny * 2.0 * HY - HY
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        XY1 = np.stack([X.ravel(), Y.ravel()], 1)

        # ⚠ RINGS ARE CONCENTRIC ABOUT THE OFFSET CENTRE, not the frame centre - so the
        # "circular modulation" radiates from the same point the brightness does, and the two
        # halves of the composition agree with each other.
        # ⚠ DOT COUNT PER RING PROPORTIONAL TO ITS RADIUS, or the inner rings crowd into a
        # bright spot and the outer ones thin to nothing. `_arc_gain` solves the same problem
        # on the mesh and cannot help here, because the field is not a curve family.
        y0 = float(VOID_OFFSET)
        far = float(np.hypot(HX, HY + abs(y0))) * 1.02
        rows, n_rho = [], max(int(round(far / pitch)), 4)
        for a in range(n_rho):
            rho = (a + 0.5) * pitch
            n_c = max(int(round(2.0 * np.pi * rho / pitch)), 6)
            ang = (np.arange(n_c) + (a * PHI % 1.0)) * (2.0 * np.pi / n_c)
            rows.append(np.stack([rho * np.cos(ang), y0 + rho * np.sin(ang)], 1))
        XY2 = np.concatenate(rows)
        XY2 = XY2[(np.abs(XY2[:, 0]) <= HX) & (np.abs(XY2[:, 1]) <= HY)]

        L1, L2 = layered(XY1), layered(XY2)
        self.d_bars = asarray(L1.astype(np.float32))
        self.d_rings = asarray(L2.astype(np.float32))
        self.f_bars = asarray(self._falloff(L1))
        self.f_rings = asarray(self._falloff(L2))
        self.n_bars = int(L1.shape[0])
        self.n_rings = int(L2.shape[0])
        self.W = self._solve_radius(cam, ref)

    # ---------------------------------------------------------------------------------
    def _falloff(self, P):
        """Static brightness envelope about the OFFSET centre. Not time varying, so the
        unit-mean rule does not apply - see the module note."""
        d = np.hypot(P[:, 0], P[:, 1] - float(VOID_OFFSET))
        u = np.clip((d - VOID_FALL_IN) / (VOID_FALL_OUT - VOID_FALL_IN), 0.0, 1.0)
        return (1.0 - u * u * (3.0 - 2.0 * u)).astype(np.float32)

    def _solve_radius(self, cam, ref):
        """World length that projects to exactly ONE half-frame WIDTH.

        Solved the way `scene.fit_scales` solves its own bound, and for the same reason: a
        perspective camera does not make projected extent linear in world size, so the ratio
        correction is ITERATED rather than assumed. One solve serves the whole clip, because
        the camera distance, the fov and the card's facing are all constant.
        """
        right, up, fwd = self._basis(cam)
        a = np.linspace(0.0, 2.0 * np.pi, 256)
        ring = np.stack([np.cos(a), np.sin(a), np.zeros(256)], 1)
        B = np.stack([right, up, fwd])
        W = 300.0
        for _ in range(5):
            x, _y, _z = cam.project(((ring @ B) * W).astype(np.float32))
            got = float(np.abs(np.asarray(x) - ref[0] * 0.5).max()) / (ref[0] * 0.5)
            W = W / max(got, 1e-6)
        return float(W)

    def _basis(self, cam):
        """(right, up, fwd) of the camera, as world unit vectors.

        ⚠ THE FIELD IS REBUILT INTO THIS BASIS EVERY FRAME, which is the whole point of a card:
        the camera orbits, and a card fixed in WORLD space would turn edge-on and vanish.
        Derived from `cam.eye` rather than stored, so it cannot drift out of step with
        `scene.camera_at` - and because it follows the camera, the field loops exactly when the
        camera does.
        """
        eye = np.asarray(cam.eye, np.float64)
        fwd = -eye / max(np.linalg.norm(eye), 1e-9)          # the target is the origin
        wup = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(fwd, wup)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return right, up, fwd

    # ---------------------------------------------------------------------------------
    @staticmethod
    def mix_at(u):
        """Bars -> rings -> bars. ZERO AT BOTH ENDS, which is what lets the clip loop.

        ⚠ A HARD SWAP WOULD BE A FULL-FRAME LUMINANCE EVENT - the banned flash by another name,
        since this field IS most of the frame. The crossfade uses COMPLEMENTARY gains so the
        two lattices always sum to one field's worth of flux, and it returns to bars before the
        end so the last frame hands off to the first.
        """
        return _smoothstep(u, 0.20, 0.42) * (1.0 - _smoothstep(u, 0.62, 0.84))

    def build(self, i, j, w, R, u, cam, bond=None, sep=0.0, t=None):
        """(P, key, gain) for the field at this instant, in WORLD units.

        ⚠ `bond` REPLACES THE CARVE, NOT THE FIELD. Everything above - the two lattices, the
        crossfade, the falloff envelope, the solved half-frame width, the rim key - is the
        composition Clancy approved, and none of it knows or cares what shape the hole is. A
        molecular orbital is a different ANSWER to `is this point inside?`, so it enters at
        exactly one line and the rest of variation C is untouched. See `orb/bond.py`.
        """
        xp = self.xp
        f32 = np.float32
        mix = float(self.mix_at(u))
        right, up, fwd = self._basis(cam)
        B = xp.asarray(np.stack([right, up, fwd]).astype(np.float32))

        parts, fall, wts = [], [], []
        if mix < 0.999:
            parts.append(self.d_bars); fall.append(self.f_bars); wts.append(1.0 - mix)
        if mix > 0.001:
            parts.append(self.d_rings); fall.append(self.f_rings); wts.append(mix)
        if len(parts) == 1:
            L, F = parts[0], fall[0]
            Wt = xp.full(L.shape[0], f32(wts[0]))
        else:
            L = xp.concatenate(parts)
            F = xp.concatenate(fall)
            Wt = xp.concatenate([xp.full(p.shape[0], f32(k))
                                 for p, k in zip(parts, wts)])

        # ⚠ SCALED BY `self.W`, THE SOLVED HALF-FRAME WIDTH - NOT by the frame's `R`. See the
        # sizing note. `R` is still used below for the CARVE, which does have to track the
        # orbital.
        P = (L @ B) * f32(self.W)
        self.last_L, self.last_d = L, None
        self.last_extra = []

        # ---- the implicit carve: |Y| at each field point's own (theta, phi) --------------
        # Reuses the SAME separable representation the mesh rebuild uses - the theta-profile
        # LUT and a closed-form cosine - so there is no second amplitude path to keep in step,
        # and the carve is blend-aware for free.
        if bond is not None:
            # ⚠ `d` IS IN DIFFERENT UNITS ON THIS BRANCH, which is why the edge and the fade
            # come off the bond object instead of off this module's constants. `VOID_EDGE` is
            # a fraction of R; `bond.dist` returns a fraction of the LOCAL surface radius,
            # because a two-centre field has no single R to be a fraction of. Sharing the
            # number would silently sharpen or smear the rim by whatever |Y| happens to be.
            d = bond.dist(P, R, sep)
            e = f32(bond.edge)
            fade = f32(bond.fade)
            return P, self._key(xp, d, fade), self._carve(xp, d, e) * F * Wt

        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        rad = xp.sqrt(x * x + y * y + z * z) + f32(1e-6)
        # ⚠ +Y IS THE QUANTUM z AXIS. Third place that has to know this; the other two are
        # `harmonics.directions` and `flow.Manifold.rebuild`.
        th = xp.arccos(xp.clip(y / rad, f32(-1.0), f32(1.0)))
        ph = xp.arctan2(z, x)
        from . import sync as _SY
        layers = (_SY.LAYERS if (_SY.ACTIVE is not None and t is not None) else None)
        amp = xp.zeros(rad.shape, np.float32)
        comp = {}
        for s, ww in ((i, 1.0 - w), (j, w)):
            if ww == 0.0 and not layers:
                continue
            f = xp.interp(th, self.man.g_th_lut, self.man.f_lut[s])
            c = xp.abs(xp.cos(f32(self.man.m[s]) * ph))
            if layers:
                comp[s] = f32(1.0 / self.man.norm[s]) * f * c
            if ww:
                # ⚠ the ORIGINAL expression, kept exactly: canon is gated byte-identical
                amp = amp + f32(ww / self.man.norm[s]) * f * c
        # ⚠ THE SAME VIBRATION AS THE DOTS, or the hole - most of what is visible in this
        # look - would sit still while its faint residual wireframe shook. `orb/sync.py`.
        if _SY.ACTIVE is not None and t is not None:
            fac = _SY.ACTIVE.factor(xp, x / rad, y / rad, z / rad, t)
            if fac is not None:
                amp = amp * fac
        surf = amp * f32(R)

        d = (rad - surf) / f32(R)              # signed distance OUTWARD, in units of R
        self.last_d = d
        carve = self._carve(xp, d, f32(VOID_EDGE))
        if layers:
            key = self._key(xp, d, f32(VOID_FADE))
            carve, key = self._bloom(xp, x / rad, y / rad, z / rad, rad, comp, i, j, w, R, t,
                                     carve, key, layers)
            self.last_extra = [(n_, k_, g_ * F * Wt) for n_, k_, g_ in self.last_extra]
            return P, key, carve * F * Wt

        # ⚠ THE RIM IS THE KEY, and it is free because `d` is already computed for the carve.
        # Keying the field by distance-from-the-surface puts the hottest, whitest points ON the
        # carve rim and cools them outward - the same outside-in logic `RAMP_MESH` is built
        # around - so the hole's edge is the most defined thing in the frame, with no extra
        # pass and no boost.
        return P, self._key(xp, d, f32(VOID_FADE)), carve * F * Wt

    def _bloom(self, xp, ux, uy, uz, rad, comp, i, j, w, R, t, carve, key, layers):
        """THE BLOOM VARIANT (2026-09-16): echo shells of the orbital, larger and more distorted.

        Clancy: *"fan out orbitals 3 to 4 times as much (add other halo layers, other orbital
        inflections), with larger radius's ... most of the screen should be covered in these
        blossoming, distorted, orbital fields"*. Each layer is the same |Y| surface at `scale`
        x R, with its own distortion (`sync.factor_layer`: stronger, rotated, re-phased) and
        leaning `lean` of the way toward the NEXT orbital in the walk - the "inflection".

        ⚠ FIELD-ONLY. The layers exist in the void field, which is already required to cover
        the frame; the dot mesh (the real orbital) is not echoed, so the size guard and the
        overflow check keep meaning what they meant.

        Each layer draws a DARK OUTLINE just inside its surface and a HOT RIM that cools
        outward - the same hole-then-rim grammar as the real orbital, so the echoes read as
        more of the same object rather than a new effect laid on top.
        """
        from . import sync as _SY
        f32 = np.float32
        out_carve, out_key = carve, key
        self.last_extra = []
        th = xp.arccos(xp.clip(uy, f32(-1.0), f32(1.0)))
        ph = xp.arctan2(uz, ux)

        def state_amp(s):
            if s not in comp:
                f = xp.interp(th, self.man.g_th_lut, self.man.f_lut[s])
                comp[s] = f32(1.0 / self.man.norm[s]) * f * xp.abs(
                    xp.cos(f32(self.man.m[s]) * ph))
            return comp[s]

        for L in layers:
            sa, sb, fl = _SY.ACTIVE.layer_walk(t, L["lean"])
            a = f32(1.0 - fl) * state_amp(sa) + f32(fl) * state_amp(sb)
            fac = _SY.ACTIVE.factor_layer(xp, ux, uy, uz, t, L)
            if fac is not None:
                a = a * fac
            kick = _SY.ACTIVE.kick(t, L.get("tone"))
            sR = f32(float(L["scale"]) * (1.0 + float(L.get("kick", 0.0)) * kick) * R)
            boost = float(L["boost"]) * (1.0 + float(L.get("flare", 0.0)) * kick)
            dk = (rad - a * sR) / sR
            wd = f32(L["width"])
            dark = xp.exp(-((dk + wd) / wd) ** 2)
            rim = xp.where(dk > 0, xp.exp(-dk / f32(L["fade"])), xp.exp(-(dk / wd) ** 2))
            out_carve = out_carve * (f32(1.0) - f32(L["dip"]) * dark)
            lkey = f32(0.22) + f32(0.62 * L["heat"]) * rim
            if L.get("lut"):
                # its own deposit through its own ramp (`render.void_at`)
                self.last_extra.append((L["lut"], xp.clip(lkey, f32(0), f32(1)),
                                        carve * f32(boost) * rim))
                continue
            out_carve = out_carve + carve * f32(boost) * rim
            out_key = xp.maximum(out_key, lkey)
        return out_carve, xp.clip(out_key, f32(0.0), f32(1.0))

    @staticmethod
    def _carve(xp, d, e):
        """A smoothstep across the surface, SYMMETRIC about d = 0. Factored out so the
        single-centre and molecular branches cannot drift apart - the transition shape is the
        look, and only the WIDTH is allowed to differ between them."""
        f32 = np.float32
        c = xp.clip((d + e) / (f32(2.0) * e), f32(0.0), f32(1.0))
        return c * c * (f32(3.0) - f32(2.0) * c)

    @staticmethod
    def _key(xp, d, fade):
        """The rim key: hottest ON the carve rim, cooling outward. Free - `d` is already
        computed for the carve - and it is what makes the hole's edge the most defined thing in
        the frame with no extra pass and no boost."""
        f32 = np.float32
        return xp.clip(f32(0.22) + f32(0.62) * (f32(1.0) - xp.clip(d / fade,
                                                                  f32(0.0), f32(1.0))),
                       f32(0.0), f32(1.0))
