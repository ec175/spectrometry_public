# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""bond.py - TWO CENTRES, AND THE SIGN THAT MAKES IT BONDING. 2026-09-02.

Clancy: *"id like to use this engine to create videos of more than 1 of these structures
'bonding' with another ... Show me a few 5 second 720p 30fps clips of the different kinds of
bonding that the orbitals can demonstrate. Use the negative space final as a design template."*

So: variation C's format, with the hole made by TWO atoms instead of one.

WHY THIS RIDES ON `void.py` AND NOT ON THE MESH
------------------------------------------------
Every other surface in this engine is a RADIAL GRAPH: one origin, and `r = |Y(theta, phi)|`
solved per direction. That representation cannot express a molecular orbital, and the way it
fails is the dangerous kind. A ray cast from the bond midpoint through an ANTIBONDING sigma*
enters the far side of one lobe, and the largest crossing is its OUTER boundary - so the gap
between the two lobes, which is the entire content of the word "antibonding", is silently
filled in and sigma* renders as a peanut identical to sigma. That is this project's standing
failure class (`harmonics`' scipy-argument-order note): a plausible, completely wrong picture
with every exposure and framing statistic inside its bounds.

`void.py`'s carve is already the thing that does not have that problem. It is a MEMBERSHIP TEST
evaluated at an arbitrary 3-D point - `is this point inside the orbital?` - which makes no
star-shape assumption, needs no origin, and handles disconnected lobes exactly. Swapping its
single-centre `rad > R|Y|` for a genuine two-centre implicit field is therefore the smallest
honest change that can express a bond, and it lands inside the format that was asked for.

THE TRAP, AND IT DECIDES WHETHER ANY OF THIS IS BONDING AT ALL: SIGN
--------------------------------------------------------------------
Bonding and antibonding are the SAME two orbitals differing only in relative phase. Every
existing amplitude path in this engine is a MAGNITUDE - `harmonics.amplitudes` takes `abs`,
`flow.Manifold` stores `|Y(theta,0)|` and `|cos(m phi)|`, `void`'s carve reuses both - so a
field built the existing way cannot tell sigma from sigma*: they would render identically. This
module evaluates the SIGNED real harmonic, and it carries a per-atom phase.

⚠ AND THE SIGN OF "BONDING" IS NOT ALWAYS `+`. For two p_z orbitals head-on, the bond axis is
each atom's own z, so the lobes that face each other have OPPOSITE sign (one is the +z lobe,
the other the -z lobe). The CONSTRUCTIVE combination is therefore `A - B`, and a hardcoded plus
ships sigma* with sigma's name on it. Which combination is bonding is a property of the
geometry, not a constant, so `phase_b` is spelled out per scene and `cli.py bondtest` asserts
it - constructive scenes must reinforce at the overlap point, antibonding scenes must cancel
there. That assertion is the only check in the gate that can tell these five clips apart.

THE FIELD
---------
Each atom contributes a signed scalar that is 1 exactly on the surface this engine already
draws, and decays outward:

    q   = rad / (R * |Y_hat(theta, phi)|)        1 on the atom's own surface
    g   = phase * sign(Y_hat) * exp(K * (1 - q))
    F   = g_A + g_B                              the molecular field
    inside  <=>  |F| > 1

⚠ CHOSEN SO THE LOOK IS CONTINUOUS WITH WHAT SHIPPED. At large separation `g_B` is ~0 near atom
A, so `|F| > 1` reduces to `rad < R|Y|` - the EXACT silhouette of the three shipped finals. The
clip therefore opens on two of the objects Clancy already approved and the bond is something
that happens to them, rather than a new look that has to be re-judged from scratch. `FALLOFF`
(K) is the one knob that decides how close they must come before a neck forms.

⚠ HONESTY CAVEAT, and it is the same one `air_frog` carries. This is a STYLISED LCAO: real
interference of the ANGULAR parts, with a decay envelope standing in for the radial function.
There is no `R_nl`, so no radial nodes and no shell structure - `JUMP_PLAN.md` is where that
lives. What IS honest is the thing the clips are about: which combination piles amplitude up
between the nuclei and which one puts a node there.
"""
from __future__ import annotations

import numpy as np

from . import harmonics as H

#: Decay rate of the envelope outside an atom's own surface, per unit of that surface's radius.
#: Sets how close the two atoms must come before the field between them crosses the level and
#: a neck appears. Low = they merge into one blob while still visibly apart; high = nothing
#: happens until the surfaces touch. 4.0 puts the neck's arrival inside the approach.
FALLOFF = 4.0

#: Half-width of the carve transition, in the same relative units `dist` returns (a fraction of
#: the local surface radius), and SYMMETRIC about the surface - `void.py`'s rule, for the same
#: reason: any asymmetry inflates or shrinks the apparent silhouette, which would make the
#: shape a function of the treatment.
BOND_EDGE = 0.055
#: Outward falloff of the rim key, same units.
#:
#: ⚠ 0.85 WAS WRONG AND IT LOOKED WRONG, and the arithmetic says why. Far from both atoms
#: `dist -> rad/R - 1`, so for the s case these units ARE `void.VOID_FADE`'s units - and 0.85
#: against C's 0.35 is a rim 2.4x too wide. It rendered as a huge smooth teal gradient with the
#: field's dot texture flooded out of it entirely: variation C's whole composition is a TIGHT
#: hot rim over a dark dotted field, and a wide rim is a different picture wearing its colours.
#: 0.30 rather than 0.35 because a bond has TWO rims and they overlap at the neck.
BOND_FADE = 0.30


class BondField:
    """The signed two-centre field. Evaluated at arbitrary world points, on the device."""

    def __init__(self, spec, xp, asarray):
        l, m = int(spec["state"][0]), int(spec["state"][1])
        self.l, self.m = l, m
        self.pa = float(spec.get("phase_a", 1.0))
        self.pb = float(spec.get("phase_b", 1.0))
        self.k = float(spec.get("falloff", FALLOFF))
        self.edge = float(spec.get("edge", BOND_EDGE))
        self.fade = float(spec.get("fade", BOND_FADE))
        self.xp = xp

        # ⚠ THE SIGNED THETA PROFILE. `real_Y(l, m, theta, phi) = f(theta) * cos(m phi)` with
        # `f(theta) = real_Y(l, m, theta, 0)` - exactly separable and exactly signed, which is
        # the same factorisation `flow.Manifold` uses with an `abs` around both halves.
        # ⚠ THE NORMALISATION IS EXACT HERE, not a mesh maximum. `max |cos(m phi)|` is 1 for a
        # CONTINUOUS phi, so the peak of |Y| over the sphere is just `max |f|`. (The mesh path
        # normalises by the largest value any SAMPLED phi happens to reach, which is the same
        # number to within the phi spacing; the two differ by <0.1% and neither is wrong.)
        th = np.linspace(1e-6, np.pi - 1e-6, 16384)
        f = H.real_Y(l, m, th, np.zeros_like(th)).astype(np.float64)
        self.norm = float(np.abs(f).max())
        self.f_np = (f / max(self.norm, 1e-12)).astype(np.float64)
        self.th_np = th
        self.f_lut = asarray(self.f_np.astype(np.float32))
        self.g_th = asarray(th.astype(np.float32))

    def y_hat_np(self, th, ph):
        """`y_hat` on the CPU, for BUILDING a canonical point set. Same LUT, same values -
        it is `self.f_np` the device copy is made from, not a second table."""
        return np.interp(th, self.th_np, self.f_np) * np.cos(self.m * np.asarray(ph))

    # ---------------------------------------------------------------------------------
    def y_hat(self, th, ph):
        """SIGNED normalised harmonic at (theta, phi); |y_hat| <= 1, peak exactly 1."""
        xp = self.xp
        f = xp.interp(th, self.g_th, self.f_lut)
        return f * xp.cos(np.float32(self.m) * ph)

    def _atom(self, P, cy, R):
        """One atom's signed contribution at every point of `P`. `cy` is its world +Y offset."""
        xp = self.xp
        f32 = np.float32
        x = P[:, 0]
        y = P[:, 1] - f32(cy)
        z = P[:, 2]
        rad = xp.sqrt(x * x + y * y + z * z) + f32(1e-6)
        # ⚠ +Y IS THE QUANTUM z AXIS, and the BOND AXIS TOO. Fourth place that has to know the
        # first half; `harmonics.directions`, `flow.Manifold.rebuild` and `void.build` are the
        # others. Putting the bond on the same axis is what lets one (l, m) serve every scene:
        # (1,0) is then head-on and (1,1) is side-by-side, with no rotation anywhere.
        th = xp.arccos(xp.clip(y / rad, f32(-1.0), f32(1.0)))
        ph = xp.arctan2(z, x)
        ys = self.y_hat(th, ph)
        a = xp.abs(ys) + f32(1e-6)
        q = rad / (f32(R) * a)
        # ⚠ CLIPPED BEFORE THE EXP, not after. `q` runs to infinity along every nodal direction
        # (`a -> 0` there), and `exp` of a large positive argument is `inf` - which then
        # propagates through the sum into the carve as a NaN and paints a frame of noise. The
        # upper clip is `k`, the value at rad = 0, so nothing legitimate is touched.
        e = xp.exp(xp.clip(f32(self.k) * (f32(1.0) - q), f32(-60.0), f32(self.k)))
        return xp.sign(ys) * e

    def field(self, P, R, sep):
        """The signed molecular field `F` at every point of `P`. `sep` is in units of R."""
        half = 0.5 * float(sep) * float(R)
        return (np.float32(self.pa) * self._atom(P, +half, R)
                + np.float32(self.pb) * self._atom(P, -half, R))

    def dist(self, P, R, sep):
        """Signed OUTWARD distance from the molecular surface, 0 on it.

        `|F| = 1` is the surface, and for an isolated atom `|F| = exp(k(1-q))` exactly - so
        `-ln|F| / k` is `q - 1`, the distance out from that atom's own surface as a FRACTION of
        the local surface radius. Which is why `BOND_EDGE` is not `void.VOID_EDGE`: one is in
        units of R, the other is relative. They are not interchangeable and must not be shared.
        """
        xp = self.xp
        phi = xp.abs(self.field(P, R, sep))
        return -xp.log(xp.maximum(phi, np.float32(1e-12))) / np.float32(self.k)


# ======================================================================================
# THE SHELL CLOUD - what actually gets DRAWN, and the reason this was rebuilt (2026-09-02b)
# ======================================================================================
#
# Clancy, on the first bond set: *"frankly... very bad. I was hoping that the orbitals would at
# least change/morph with each other. the shape of the structures should be changing, especially
# when they start to get close to each other."*
#
# HE IS RIGHT, AND IT WAS A REPRESENTATION PROBLEM, NOT A TUNING ONE. The first build drew two
# RIGID copies of `r = |Y|`, translated toward each other, and let the void field's hole carry
# the whole story. So the only thing that ever changed shape was the negative space: the two
# structures themselves were perfect spheres sliding together, exactly as they had been at
# t = 0. Everything the physics did was in the hole, and the hole is the least of it.
#
# THE MOLECULAR SURFACE WAS ALREADY IN THE FILE. `|F| = 1` is a genuine two-centre isosurface
# that bulges toward the partner, necks, and merges - it was simply never the thing drawn. So
# the fix is to draw IT.
#
# ⚠ AND IT CANNOT BE DRAWN AS A RADIAL GRAPH, which is why this is a cloud and not a mesh.
# `r(theta, phi)` per direction stores ONE crossing, and a bond changes TOPOLOGY: before the
# neck closes a ray along the bond axis leaves atom A, crosses a gap, and enters atom B - three
# crossings - and at the instant it closes the "first crossing" jumps discontinuously from A's
# own surface to B's far side. That is a snap, and a velocity jump at a join is the one thing
# every gate in this library hard-fails. There is no radius function that is continuous through
# a merge, because the merge is not a deformation of a sphere.
#
# WHAT A SHELL CLOUD DOES INSTEAD: fix a dense 3-D dot set ONCE, and per frame light the dots
# the surface currently passes through. The lit set changes continuously as the surface moves,
# handles ANY topology (a neck closing is just neighbouring dots joining the lit set), and never
# pops - because no dot is ever created or destroyed, only dimmed. **The samples are fixed, the
# parameters move** - `flow.py`'s rule, applied to a volume instead of a lattice, and the same
# principle `JUMP_PLAN.md` is built around.
#
# ⚠ THE CANONICAL CLOUD IS SHAPED LIKE THE ATOM, NOT LIKE A BOX. Points sit at
# `rho * |Y_hat(u)| * u` for `rho` spanning roughly 0.35 .. 2.3, so they are dense exactly where
# the molecular surface can go and absent where it cannot. A uniform box of the same dot spacing
# would need ~40x the points for the same shell density, and would spend nearly all of them
# somewhere the surface never reaches.

#: Directions on the Fibonacci sphere, and radial layers per direction.
CLOUD_DIRS, CLOUD_RAD = 20000, 30
#: Radial span in units of the atom's own surface radius. Must reach far enough that a MERGED
#: surface still has dots at it - at the neck's closure the surface near the bond axis runs out
#: past the partner's centre.
CLOUD_RHO = (0.30, 2.35)
#: A direction's amplitude is floored here so a nodal direction does not collapse every one of
#: its radial layers onto the origin and burn a hole in the accumulator - the same hazard
#: `harmonics.POLE_EPS` guards on the lat/long mesh.
CLOUD_AMP_FLOOR = 0.11

#: The shell kernel, in `dist` units (a fraction of the local surface radius). ASYMMETRIC on
#: purpose: a thin hot rim OUTSIDE, a slower falloff INSIDE, so the body of the orbital carries
#: some structure instead of being an empty bubble. Variation C's hole is dark, and a pure
#: surface shell would leave nothing in it.
#: ⚠ 0.055/0.20 WAS WRONG AND THE FIRST CLOUD STILL SHOWED WHY. A shell centred on the surface
#: puts its hottest dots exactly where the void field's own carve rim already is - the two
#: brightest things in the frame land on the same pixels, the rim blows out, and the subject
#: stops being a distinct object. Variation C's composition has the field's rim carry the
#: BOUNDARY and the wireframe carry the INTERIOR, and that division still holds: the cloud
#: belongs almost entirely INSIDE the surface, filling the hole with structure that deforms.
CLOUD_W_OUT, CLOUD_W_IN = 0.018, 0.46
#: Per-dot random transmittance, mean 1 - `flow.porosity`'s `por_point` term, which is the one
#: porosity primitive that means anything without curves.
CLOUD_POR = 0.55
#: The chromatic split, in `dist` units. Two passes light shells offset either side of the
#: surface through LUTs that sum to `RAMP_MESH` - a true separation, and free here because both
#: passes read the SAME `dist` array.
CLOUD_SPLIT = 0.030


class ShellCloud:
    """A fixed 3-D dot cloud; the molecular surface lights up wherever it currently passes."""

    def __init__(self, bf, xp, asarray, seed=20260902):
        self.bf = bf
        self.xp = xp
        n_d, n_r = int(CLOUD_DIRS), int(CLOUD_RAD)

        # ---- directions: a Fibonacci sphere -----------------------------------------
        # ⚠ NOT A LAT/LONG LATTICE. Lat/long piles directions at the poles, and on a bond that
        # is exactly the bond axis - the one place the surface does the most interesting thing.
        # A Fibonacci sphere is uniform in solid angle, so no direction is privileged, and the
        # rim concentration that makes these read as shells still comes free from the
        # projection at the limb (S-DOTS).
        i = np.arange(n_d, dtype=np.float64) + 0.5
        cz = 1.0 - 2.0 * i / n_d
        sr = np.sqrt(np.maximum(1.0 - cz * cz, 0.0))
        a = np.pi * (3.0 - np.sqrt(5.0)) * i
        # +Y is the quantum z axis - the fifth place that knows it
        U = np.stack([sr * np.cos(a), cz, sr * np.sin(a)], 1)
        th = np.arccos(np.clip(cz, -1.0, 1.0))
        ph = np.arctan2(U[:, 2], U[:, 0])
        amp = np.maximum(np.abs(bf.y_hat_np(th, ph)), CLOUD_AMP_FLOOR)

        lo, hi = CLOUD_RHO
        rho = lo + (np.arange(n_r, dtype=np.float64) + 0.5) / n_r * (hi - lo)

        # canonical positions, IN UNITS OF R, relative to the atom's own centre
        P0 = (rho[None, :, None] * amp[:, None, None]) * U[:, None, :]
        self.n = n_d * n_r

        # ⚠ THE PER-DOT WEIGHT IS A DENSITY CORRECTION, not a style choice - `_arc_gain`'s
        # argument in a volume. This cloud's dots are NOT uniformly spaced: angular spacing
        # grows with radius and radial spacing grows with the direction's own amplitude, so a
        # cell's volume goes as `rho^2 * amp^3`. Weighting by it makes the deposited flux per
        # unit of SURFACE area constant, which is what a shell actually looks like; without it
        # the lobe tips wash out and the small-amplitude directions burn.
        # (One factor of `amp` comes back off because the kernel's physical thickness is itself
        # proportional to the local surface radius - `dist` is relative.)
        w = (rho[None, :] ** 2) * (amp[:, None] ** 2)
        rng = np.random.default_rng(seed)
        por = 1.0 + CLOUD_POR * (rng.random((n_d, n_r)) * 2.0 - 1.0)
        w = w * por
        w = w / max(float(w.mean()), 1e-12)

        self.P0 = asarray(P0.reshape(-1, 3).astype(np.float32))
        self.w = asarray(w.reshape(-1).astype(np.float32))
        #: the largest canonical radius, for the size guard
        self.reach = float((rho.max() * amp.max()))

    # ---------------------------------------------------------------------------------
    def build(self, R, sep, split=0.0):
        """(P, key, gain) for the whole two-atom cloud at this separation."""
        xp = self.xp
        f32 = np.float32
        half = f32(0.5 * float(sep) * float(R))
        base = self.P0 * f32(R)
        off = xp.zeros((2, 3), np.float32)
        off[0, 1] = half
        off[1, 1] = -half
        P = xp.concatenate([base + off[0], base + off[1]])
        d = self.bf.dist(P, R, sep) - f32(split)
        # asymmetric shell: thin and hot outside, slower inside
        wo, wi = f32(CLOUD_W_OUT), f32(CLOUD_W_IN)
        s = xp.where(d >= 0, d / wo, d / wi)
        k = xp.exp(-s * s)
        gain = xp.concatenate([self.w, self.w]) * k
        # ⚠ THE KEY IS DISTANCE-FROM-THE-SURFACE, exactly as `void._key` keys the field - so the
        # hottest, whitest dots sit ON the surface and cool inward. Same outside-in logic
        # `RAMP_MESH` is built around, and it is free: `d` is already computed.
        # ⚠ THE CEILING IS 0.64, NOT 1.0, AND THAT IS DELIBERATE. `RAMP_MESH` runs to white at
        # the top, and the cloud's hottest dots sit ON the surface - exactly where the void
        # field's carve rim already goes white. Two white things on the same pixels is a blown
        # rim and no subject. Capping the cloud in the TEAL part of the ramp keeps white as the
        # rim's alone, which is the outside-in reading variation C is built on.
        key = xp.clip(f32(0.20) + f32(0.44)
                      * (f32(1.0) - xp.clip(xp.abs(d) / f32(0.5), f32(0.0), f32(1.0))),
                      f32(0.0), f32(1.0))
        return P, key, gain


# ======================================================================================
# THE GATE'S OWN CHECK
# ======================================================================================
#: Where two atoms of this state actually OVERLAP, as (x, y, z) in units of R with the bond on
#: +/-Y - and where each atom's amplitude PEAKS, for the isolated-atom check.
#:
#: ⚠ THE OVERLAP POINT IS NOT ALWAYS THE MIDPOINT. For p_x side-by-side the bond axis is a
#: NODAL direction of both atoms, so the field is identically zero on it and a midpoint probe
#: measures 0 == 0 and passes every scene. The pi overlap is off-axis, where the two lobes
#: actually face each other.
PROBES = {
    (0, 0): dict(overlap=(0.0, 0.0, 0.0), peak=(0.0, 1.0, 0.0)),
    (1, 0): dict(overlap=(0.0, 0.0, 0.0), peak=(0.0, 1.0, 0.0)),
    (1, 1): dict(overlap=(0.45, 0.0, 0.0), peak=(1.0, 0.0, 0.0)),
    (2, 0): dict(overlap=(0.0, 0.0, 0.0), peak=(0.0, 1.0, 0.0)),
}


def selftest(verbose=True):
    """Assert every registered bond scene is the bond its NAME claims.

    ⚠ THIS IS THE ONLY CHECK IN THE GATE THAT CAN TELL THESE CLIPS APART. `dry` measures
    framing, exposure, motion and coverage, and sigma / sigma* differ in NONE of them - they
    are the same two atoms at the same separation with one sign flipped, so every aggregate
    statistic is nearly identical while the picture means the opposite thing. Same argument as
    `harmonics._selftest`: the cheapest statement that separates the two conventions.

    Three assertions per scene, at the BONDED separation:

    1. CONSTRUCTIVE or DESTRUCTIVE at the overlap point. Bonding must reinforce there;
       antibonding must cancel there. This is what the words mean.
    2. A NODAL SURFACE between the nuclei, or the absence of one - the field must change sign
       across the midplane for an antibonding scene and must not for a bonding one.
    3. ISOLATED-ATOM CONTINUITY at the FAR separation: a point on one atom's own surface must
       still be on the molecular surface to within 2%, which is what makes these clips open on
       the object that already shipped.
    """
    from . import scene as S

    ok = True
    names = [k for k, v in S.SCENES.items() if v.get("bond")]
    for name in names:
        sp = dict(S.SCENES[name])
        b = dict(sp["bond"])
        bf = BondField(b, np, lambda a: a)
        key = (bf.l, bf.m)
        if key not in PROBES:
            print(f"[bondtest] FAIL {name}: no probe point registered for (l,m)={key}")
            ok = False
            continue
        pr = PROBES[key]
        want = str(b.get("kind", "")).lower()
        if want not in ("bonding", "antibonding"):
            print(f"[bondtest] FAIL {name}: bond spec must name kind=bonding|antibonding")
            ok = False
            continue

        R = 100.0
        near = float(sp.get("sep_near", S.SEP_NEAR))
        far = float(sp.get("sep_far", S.SEP_FAR))
        half = 0.5 * near * R

        # ---- 1. reinforce or cancel at the overlap point ---------------------------
        p = np.asarray([pr["overlap"]], np.float32) * np.float32(R)
        gA = float(bf.pa * bf._atom(p, +half, R)[0])
        gB = float(bf.pb * bf._atom(p, -half, R)[0])
        tot = abs(gA + gB)
        mag = abs(gA) + abs(gB)
        if want == "bonding":
            good = tot > 1.25 * max(abs(gA), abs(gB))
        else:
            good = tot < 0.05 * mag
        if not good:
            print(f"[bondtest] FAIL {name}: wants {want} but at the overlap point "
                  f"gA={gA:+.4f} gB={gB:+.4f} -> |sum|={tot:.4f} of {mag:.4f}")
            ok = False

        # ---- 2. the nodal surface between the nuclei -------------------------------
        o = np.asarray(pr["overlap"], np.float64)
        q = np.asarray([[o[0] * R, +0.12 * half, o[2] * R],
                        [o[0] * R, -0.12 * half, o[2] * R]], np.float32)
        F = bf.field(q, R, near)
        flip = bool(np.sign(F[0]) * np.sign(F[1]) < 0)
        if flip != (want == "antibonding"):
            print(f"[bondtest] FAIL {name}: sign across the midplane is "
                  f"{'flipped' if flip else 'constant'} ({F[0]:+.4f}, {F[1]:+.4f}), "
                  f"which is not what {want} means")
            ok = False

        # ---- 3. isolated-atom continuity at the far separation ---------------------
        hf = 0.5 * far * R
        s = np.asarray([pr["peak"]], np.float64) * R
        s[0, 1] += hf
        d = float(bf.dist(s.astype(np.float32), R, far)[0])
        if abs(d) > 0.02:
            print(f"[bondtest] FAIL {name}: at sep={far:.1f}R a point on atom A's own "
                  f"surface sits {d:+.3f} off the molecular surface (want |d|<0.02)")
            ok = False

        if verbose:
            print(f"[bondtest] {name:20s} (l,m)=({bf.l},{bf.m}) phase=({bf.pa:+.0f},"
                  f"{bf.pb:+.0f})  {want:11s}  overlap |sum|={tot:6.3f} of {mag:6.3f}  "
                  f"midplane {'node' if flip else 'filled'}  far-continuity {d:+.4f}")

    if not names:
        print("[bondtest] no bond scenes registered")
        return False
    print("[bondtest] ok" if ok else "[bondtest] FAIL")
    return ok
