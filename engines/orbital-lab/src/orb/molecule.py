# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""molecule.py - THE TEMPLATE BOND STRUCTURES, and the two molecular scenes. 2026-09-02c.

Clancy: *"Make two new low quality videos, of some small molecules bonding with this engine ...
6-7 H2O molecules exhibiting the bonding that they would over 10 seconds ... the burning of a
long chain hydrocarbon with some oxygen around ... **its important that we build "template" bond
structures though.**"*

So the deliverable is two clips AND a reusable layer, and the clips are compositions of it.

WHAT IS A TEMPLATE HERE
-----------------------
Three tiers, each usable without the one above it:

  ELEMENTS   an element's display radius, in Angstrom
  SPECIES    `water()`, `o2()`, `co2()`, `alkane(n)` -> local-frame atoms + a bond list, built
             from published bond lengths and angles. Pure geometry, no timeline.
  SCENE      `MolScene` - a flat list of ATOMS THAT PERSIST and BONDS THAT DO NOT. Molecules are
             poses of a species stamped into it; a reaction is a re-pose of the same atoms.

⚠ ATOMS PERSIST; ONLY BONDS ARE CREATED AND DESTROYED. That single decision is doing three jobs
at once. It is the honest physics (combustion rearranges bonds, it does not make or unmake
nuclei), it is what makes the animation continuous (every atom has one unbroken path, so nothing
ever pops into or out of existence), and it is what makes the gate ASSERTABLE - `selftest` can
check that every element's count is the same at the end as at the start, which is the one
statement that would catch a mis-specified reaction.

⚠ THE FIELD HERE IS TOTAL ELECTRON DENSITY, NOT A SIGNED WAVEFUNCTION - and that is a different
object from the five sigma/pi clips, deliberately. `bond.py` carries a SIGNED field because
bonding-vs-antibonding IS a sign. A molecule at rest has no such question: what a viewer reads
as its shape is total density, which is positive everywhere and is well modelled as a union of
atom-centred envelopes with build-up between bonded pairs. Mixing the two would be the worse
mistake - a signed field over 45 atoms would carve nodal surfaces nobody asked to see.

⚠ COVALENT NECKS ARE FREE; HYDROGEN BONDS ARE NOT. Two atoms whose envelopes overlap neck on
their own - an O-H at 0.96 A with radii 0.70 + 0.42 is already merged, so the molecule's shape
comes out of the atom list with no bond term at all. A hydrogen bond does NOT: O...H at 1.9 A
leaves the field at ~0.25 between them, well under the surface. So H-bonds carry an explicit
TUBE, and that is the honest reading as well as the convenient one - a hydrogen bond really is a
weak, separate, directional interaction, and giving it its own thin neck that fades in says
exactly that. Covalent bonds are still registered in the bond list, with no tube: the list is
the TEMPLATE, and `selftest` measures it.

⚠ AND THE SPURIOUS NECK IS THIS BUILD'S VERSION OF THE pi LESSON. Seven waters flying to a
cluster will pass near each other, and the field necks between ANY two atoms that come close -
fake bonds forming and breaking that a viewer cannot tell from real ones. `selftest` probes the
whole trajectory for it. (The trajectories are also built RADIALLY, each molecule starting
directly outward from where it ends, so the paths cannot cross in the first place.)
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# ELEMENTS - display radius in Angstrom
# --------------------------------------------------------------------------------------
#: ⚠ These are LOOK radii, not covalent radii, and they are chosen so the covalent necks come
#: out right. A bond necks when the two envelopes overlap, so what matters is `R_i + R_j`
#: against the published bond length: O-H 0.96 against 0.70+0.42 = 1.12 (merged, a fat neck),
#: C-C 1.54 against 0.80+0.80 = 1.60 (merged, a fatter one), C-H 1.09 against 0.80+0.42 = 1.22.
#: Ordered by real size, so the picture still says which atom is which.
ELEMENTS = {
    "H": dict(radius=0.42),
    "C": dict(radius=0.80),
    "O": dict(radius=0.70),
}

#: Published geometry, in Angstrom and degrees. Every number below is quoted here once.
GEOM = dict(
    OH=0.9572, HOH=104.5,          # water
    OO=1.2075,                     # dioxygen
    CO2=1.1600,                    # carbon dioxide, linear
    CC=1.5400, CH=1.0900, CCC=112.0, TET=109.47,   # alkanes
    HB_OO=2.80,                    # hydrogen bond, O...O
)


def _u(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)


def rot_axis(axis, ang):
    """Rodrigues. `axis` need not be normalised."""
    k = _u(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rot_align(a, b):
    """The rotation taking unit `a` onto unit `b`. Handles the antiparallel case."""
    a, b = _u(a), _u(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # antiparallel: any perpendicular axis, half turn
        p = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            p = np.array([0.0, 1.0, 0.0])
        return rot_axis(np.cross(a, p), np.pi)
    return rot_axis(v, np.arctan2(np.linalg.norm(v), c))


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z). Shepperd's branch-safe form."""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S,
             (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S,
             (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S,
             (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S,
             (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q, np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def quat_to_mat(q):
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(qa, qb, u):
    """Shortest-arc spherical interpolation, with the near-parallel case linearised."""
    qa = qa / max(np.linalg.norm(qa), 1e-12)
    qb = qb / max(np.linalg.norm(qb), 1e-12)
    d = float(np.dot(qa, qb))
    if d < 0.0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + u * (qb - qa)
        return q / max(np.linalg.norm(q), 1e-12)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    st = np.sin(th)
    return (np.sin((1.0 - u) * th) * qa + np.sin(u * th) * qb) / st


# --------------------------------------------------------------------------------------
# SPECIES - local-frame geometry + the bond list
# --------------------------------------------------------------------------------------
class Species:
    """`elems`, local `pos` (n,3) in Angstrom, and `bonds` as (i, j, order)."""

    __slots__ = ("name", "elems", "pos", "bonds")

    def __init__(self, name, elems, pos, bonds):
        self.name = name
        self.elems = list(elems)
        self.pos = np.asarray(pos, np.float64).reshape(-1, 3)
        self.bonds = [tuple(b) for b in bonds]
        assert len(self.elems) == self.pos.shape[0]

    def __len__(self):
        return len(self.elems)


def water():
    """O at the origin, both H in the local xz plane, the C2 axis along local +z.

    ⚠ THE LOCAL FRAME IS A CONTRACT, not a convenience: `hbond_pose` builds a donor's
    orientation by aligning ATOM 1's direction onto the acceptor, so atom 1 must be an H and the
    template must not be reordered."""
    h = np.radians(GEOM["HOH"] * 0.5)
    d = GEOM["OH"]
    p = [[0.0, 0.0, 0.0],
         [d * np.sin(h), 0.0, d * np.cos(h)],
         [-d * np.sin(h), 0.0, d * np.cos(h)]]
    return Species("H2O", ["O", "H", "H"], p, [(0, 1, 1), (0, 2, 1)])


def o2():
    d = GEOM["OO"] * 0.5
    return Species("O2", ["O", "O"], [[0, 0, d], [0, 0, -d]], [(0, 1, 2)])


def co2():
    d = GEOM["CO2"]
    return Species("CO2", ["C", "O", "O"], [[0, 0, 0], [0, 0, d], [0, 0, -d]],
                   [(0, 1, 2), (0, 2, 2)])


def alkane(n):
    """C_n H_(2n+2), as a planar zig-zag along local x with the kink in z.

    Carbons alternate +/- in z so consecutive C-C vectors close the published 112 degree angle;
    the step along the axis and the half-kink are SOLVED from the bond length and that angle
    rather than assumed, so `selftest` measuring them back is a real check.
    """
    L, ang = GEOM["CC"], np.radians(GEOM["CCC"])
    # |step| = L, and the angle between (+ax, +kk) and (-ax, +kk) is `ang`
    ax = L * np.sin(ang * 0.5)
    kk = np.sqrt(max(L * L - ax * ax, 0.0))
    C = np.array([[i * ax, 0.0, (kk * 0.5) * (1 if i % 2 == 0 else -1)] for i in range(n)])
    C[:, 0] -= C[:, 0].mean()
    C[:, 2] -= C[:, 2].mean()

    elems = ["C"] * n
    pos = list(C)
    bonds = [(i, i + 1, 1) for i in range(n - 1)]
    half = np.radians(GEOM["TET"] * 0.5)
    for i in range(n):
        nb = [C[j] - C[i] for j in (i - 1, i + 1) if 0 <= j < n]
        if len(nb) == 2:
            u1, u2 = _u(nb[0]), _u(nb[1])
            bis = _u(-(u1 + u2))
            perp = _u(np.cross(u1, u2))
            dirs = [bis * np.cos(half) + perp * np.sin(half),
                    bis * np.cos(half) - perp * np.sin(half)]
        else:
            # a terminal carbon: three H tetrahedral about the one C-C direction
            u1 = _u(nb[0])
            p = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(u1, p))) > 0.9:
                p = np.array([0.0, 0.0, 1.0])
            e1 = _u(np.cross(u1, p))
            e2 = np.cross(u1, e1)
            dirs = []
            for k in range(3):
                a = 2.0 * np.pi * k / 3.0
                dirs.append(-u1 * np.cos(np.pi - np.radians(GEOM["TET"]))
                            + (e1 * np.cos(a) + e2 * np.sin(a))
                            * np.sin(np.pi - np.radians(GEOM["TET"])))
            dirs = [_u(d) for d in dirs]
        for d in dirs:
            bonds.append((i, len(pos), 1))
            pos.append(C[i] + GEOM["CH"] * d)
            elems.append("H")
    return Species(f"C{n}H{2 * n + 2}", elems, pos, bonds)


# --------------------------------------------------------------------------------------
# THE SCENE - persistent atoms, transient bonds
# --------------------------------------------------------------------------------------
def _smooth(x, a, b):
    if b <= a:
        return 0.0 if x < a else 1.0
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Atom:
    """One nucleus for the whole clip. `keys` is [(t, xyz), ...], smoothstepped between."""

    __slots__ = ("elem", "keys", "mol", "local", "ckeys", "qkeys")

    def __init__(self, elem, p0, mol=0):
        self.elem = elem
        self.keys = [(0.0, np.asarray(p0, np.float64))]
        self.mol = mol
        #: THE RIGID PATH. When set, `at()` interpolates the molecule's POSE and re-places this
        #: atom from `local`, instead of interpolating the atom's world position.
        self.local = None
        self.ckeys = None
        self.qkeys = None

    def rigid(self, local, ckeys, qkeys):
        """Attach a rigid pose path: centre keyframes + orientation keyframes.

        ⚠ INTERPOLATING ATOM POSITIONS THROUGH A ROTATION SHRINKS THE MOLECULE, and the gate
        found it as a 15% bond stretch that no amount of tuning the thermal motion would fix.
        A straight line between an atom's start and end positions is a CHORD of the arc it
        should travel, so every bond contracts mid-turn - by `1 - cos(theta/2)`, which is 9% for
        a 50 degree tumble and 100% for a half turn, where the whole molecule collapses to a
        point and comes back. Crystallisation is almost entirely reorientation, so nearly every
        molecule in the scene was doing this.

        The fix is to interpolate the POSE and rebuild the atom from it: the centre moves in a
        straight line and the orientation SLERPS, so the molecule is rigid at every instant by
        construction rather than at the two ends by luck.
        """
        self.local = np.asarray(local, np.float64)
        self.ckeys = [(float(t), np.asarray(p, np.float64)) for (t, p) in ckeys]
        self.qkeys = [(float(t), np.asarray(q, np.float64)) for (t, q) in qkeys]
        self.keys = [(t, p + quat_to_mat(q) @ self.local)
                     for (t, p), (_t, q) in zip(self.ckeys, self.qkeys)]

    def hold_to(self, t, p):
        self.keys.append((float(t), np.asarray(p, np.float64)))

    def at(self, t):
        if self.qkeys is not None:
            c = self._interp(self.ckeys, t)
            q = self._slerp_keys(t)
            return c + quat_to_mat(q) @ self.local
        k = self.keys
        if len(k) == 1 or t <= k[0][0]:
            return k[0][1]
        for a in range(len(k) - 1):
            t0, p0 = k[a]
            t1, p1 = k[a + 1]
            if t <= t1:
                w = _smooth(t, t0, t1)
                return p0 + (p1 - p0) * w
        return k[-1][1]

    @staticmethod
    def _interp(keys, t):
        if len(keys) == 1 or t <= keys[0][0]:
            return keys[0][1]
        for a in range(len(keys) - 1):
            t0, p0 = keys[a]
            t1, p1 = keys[a + 1]
            if t <= t1:
                return p0 + (p1 - p0) * _smooth(t, t0, t1)
        return keys[-1][1]

    def _slerp_keys(self, t):
        k = self.qkeys
        if len(k) == 1 or t <= k[0][0]:
            return k[0][1]
        for a in range(len(k) - 1):
            t0, q0 = k[a]
            t1, q1 = k[a + 1]
            if t <= t1:
                return slerp(q0, q1, _smooth(t, t0, t1))
        return k[-1][1]


class Bond:
    """A bond that may fade in or out. `tube` 0 means it contributes NO field - the neck comes
    from the two atoms overlapping - and the entry exists so `selftest` can measure it."""

    __slots__ = ("i", "j", "order", "tube", "win", "phase", "kind")

    def __init__(self, i, j, order=1, tube=0.0, win=None, phase=0, kind="cov"):
        #: "cov" = a covalent bond, whose neck comes free from the two envelopes overlapping.
        #: "hb"  = a hydrogen bond. ⚠ NO LONGER A TUBE IN THE FIELD - it is DRAWN, as charge
        #: field lines (`MolLines`), so `tube` is 0 and the kind is what tells them apart. The
        #: geometry test keys off THIS, not off the tube: an H-bond with tube 0 read as covalent
        #: and got measured against the O-H bond length.
        self.kind = str(kind)
        self.i, self.j, self.order, self.tube = int(i), int(j), int(order), float(tube)
        self.win = win                      # (t_on0, t_on1, t_off0, t_off1) or None
        #: 0 = present at t=0 (a reactant, or a molecule that never changes), 1 = formed later.
        #: ⚠ THE GEOMETRY TEST NEEDS THIS AND GOT IT WRONG WITHOUT IT. A reactant bond is only
        #: at its published length at the START and a product bond only at the END, so a test
        #: that picks the frame from the ATOM's molecule id measured every dissociated fuel
        #: bond against the intact value and reported twenty failures for one missing tag.
        self.phase = int(phase)

    def strength(self, t):
        if self.win is None:
            return 1.0
        a, b, c, d = self.win
        s = _smooth(t, a, b)
        if d > c:
            s *= 1.0 - _smooth(t, c, d)
        return s


class MolScene:
    def __init__(self, name, duration, seed=20260902):
        self.name = name
        self.duration = float(duration)
        self.atoms = []
        self.bonds = []
        self.n_mol = 0
        self._seed = int(seed)
        self._jit = None
        #: (half-width, half-height) in Angstrom that should map to the FRAME. None = use the
        #: atom hull. ⚠ For a sheet these differ on purpose: the hull runs a lattice period past
        #: the frame so the visible molecules all have partners, and framing on the hull would
        #: pull that margin into view - which is the edge the margin exists to hide.
        self.fill_extent = None

    # ---- the thermal motion ----------------------------------------------------------
    def _build_jitter(self):
        """Fix the wave set once. Deterministic from the seed, like everything else here."""
        rng = np.random.default_rng(self._seed + 991)
        n = len(self.atoms)
        w = int(JIT_SLOW["waves"])
        k = rng.normal(size=(w, 3))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        k *= 2.0 * np.pi / float(JIT_SLOW["wavelen"])
        d = rng.normal(size=(w, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        lo, hi = JIT_SLOW["hz"]
        self._jit = dict(
            k=k, d=d * (float(JIT_SLOW["amp"]) / np.sqrt(w * 0.5)),
            f=rng.uniform(lo, hi, w), p=rng.uniform(0, 2 * np.pi, w),
            # ⚠ UNIFORM, NOT NORMAL, AND THAT IS A BOUND RATHER THAN A STYLE. A normal
            # draw has no maximum: with 21 atoms nothing reached 3 sigma, but at 144 atoms
            # several did, and two of them on the same bond stretched it 26% - past the
            # assertion, from a distribution whose *mean* amplitude was fine. An amplitude that
            # a test asserts a bound on should be drawn from something bounded.
            vd=rng.uniform(-1.0, 1.0, (n, 3)) * float(JIT_FAST["amp"]),
            vf=rng.uniform(*JIT_FAST["hz"], size=(n, 3)),
            vp=rng.uniform(0, 2 * np.pi, (n, 3)))
        # the measured worst excursion - `relax_paths` needs a BOUND, not a sample
        m = 0.0
        for t in np.linspace(0.0, self.duration, 121):
            base = np.stack([a.at(t) for a in self.atoms]) if n else np.zeros((0, 3))
            m = max(m, float(np.abs(self._offset(base, t)).max()))
        self._jit["bound"] = m

    def _offset(self, base, t):
        """Brownian (rigid per molecule) + vibration (per atom).

        ⚠ THE BROWNIAN FIELD IS SAMPLED AT A NEIGHBOURHOOD MEAN, NOT AT THE ATOM ITSELF, and
        that is what makes it a DRIFT rather than a second vibration. Sampled per atom, a field
        of wavelength L differs across a bond of length b by roughly `2*pi*b/L` of a period -
        for a 1.09 A C-H in a 6.5 A field that is a full radian, so the "coherent" term was
        pulling bonded atoms apart by 0.4 A and the gate measured a 41% stretch. Smoothing the
        sample point over a ~1.6 A Gaussian means every atom of a molecule reads essentially
        ONE value of the field, so the molecule translates as a body - which is what Brownian
        motion is - while different molecules, further apart than the smoothing width, still
        get their own. Bond stretching is then left entirely to the vibration term, where it
        belongs and where it is bounded.
        """
        j = self._jit
        # ⚠ SMOOTHED OVER THE MOLECULE'S OWN BOND GRAPH, NOT OVER A DISTANCE GAUSSIAN. A
        # Gaussian is a guess at "which atoms belong together", and in a packed lattice it is a
        # bad one: neighbouring molecules 2.76 A away carry real weight, so an oxygen and its
        # own hydrogen end up with different sample points and the "coherent" term stretches
        # the bond anyway - 15% after two rounds of tuning the width. The COVALENT BONDS
        # already say exactly which atoms are one molecule. `(I + A)^3` row-normalised averages
        # each atom over its own connected component and nothing else, so every atom of a
        # molecule reads ONE value of the field and the residual is zero by construction.
        # ⚠ And it degrades correctly through a reaction: as a bond's strength fades, its entry
        # in `A` fades with it, so a fragment smoothly stops sharing its parent's motion instead
        # of popping out of it the way a hard membership test would.
        n = base.shape[0]
        A = np.eye(n)
        for b in self.bonds:
            if b.kind != "cov":
                continue
            w = b.strength(t)
            if w > 1e-3:
                A[b.i, b.j] += w
                A[b.j, b.i] += w
        W = A @ A @ A
        q = (W @ base) / W.sum(1)[:, None]
        out = np.sin(q @ j["k"].T + 2.0 * np.pi * j["f"][None, :] * t
                     + j["p"][None, :]) @ j["d"]
        out += j["vd"] * np.sin(2.0 * np.pi * j["vf"] * t + j["vp"])
        return out

    def jitter_bound(self):
        if self._jit is None:
            self._build_jitter()
        return float(self._jit["bound"])

    def stamp(self, sp: Species, centre, R=None, mol=None):
        """Place a species at a pose. Returns the global indices of its atoms."""
        R = np.eye(3) if R is None else np.asarray(R, np.float64)
        c = np.asarray(centre, np.float64)
        m = self.n_mol if mol is None else mol
        if mol is None:
            self.n_mol += 1
        base = len(self.atoms)
        for e, p in zip(sp.elems, sp.pos):
            self.atoms.append(Atom(e, c + R @ p, mol=m))
        for (i, j, o) in sp.bonds:
            self.bonds.append(Bond(base + i, base + j, o))
        return list(range(base, base + len(sp)))

    # ---- per-frame state -------------------------------------------------------------
    def positions(self, t, jitter=True):
        """⚠ `jitter=False` is THE TEMPLATE, and it is what the geometry assertions measure.
        A vibrating bond is not its published length at any instant, so a test that checked the
        drawn positions would be asserting that the molecule is not vibrating."""
        P = np.stack([a.at(t) for a in self.atoms])
        if not jitter:
            return P
        if self._jit is None:
            self._build_jitter()
        return P + self._offset(P, t)

    def radii(self):
        return np.array([ELEMENTS[a.elem]["radius"] for a in self.atoms], np.float64)

    def tubes(self, t):
        """(m,3),(m,3),(m,),(m,) - the segments with a live tube at this instant."""
        A, B, rr, ss = [], [], [], []
        for b in self.bonds:
            if b.tube <= 0.0:
                continue
            s = b.strength(t)
            if s <= 1e-3:
                continue
            A.append(self.atoms[b.i].at(t))
            B.append(self.atoms[b.j].at(t))
            rr.append(b.tube)
            ss.append(s)
        if not A:
            z = np.zeros((0, 3))
            return z, z, np.zeros(0), np.zeros(0)
        return (np.stack(A), np.stack(B), np.asarray(rr), np.asarray(ss))

    def counts(self, t=None):
        from collections import Counter
        return dict(Counter(a.elem for a in self.atoms))


# --------------------------------------------------------------------------------------
# SCENE 1 - the water hydrogen-bond network
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# NOTHING IS EVER STILL
# --------------------------------------------------------------------------------------
# Clancy: *"Atoms/molecules often stop moving.. they should never stop moving. when in molecular
# structure, they should still be exhibiting molecular vibration/slight brownian motion."*
#
# He is right and it is the honest picture: a molecule at 300 K is never at rest. Two motions,
# because they are two different physical things and they must not be one knob:
#
#   BROWNIAN   the whole molecule wanders and librates. SLOW, and COHERENT ACROSS A MOLECULE -
#              a water does not tear itself apart drifting, it moves as a body.
#   VIBRATION  the atoms move RELATIVE to each other, stretching and bending their bonds. FAST,
#              small, and INCOHERENT between atoms - that is what makes it a vibration and not
#              a second drift.
#
# ⚠ THE BROWNIAN TERM IS A FIELD IN SPACE, NOT A PER-MOLECULE OFFSET, and that is what makes it
# survive a reaction. "Which molecule is this atom in" is not constant over `mol_burn` - the
# whole point is that it changes - so an offset keyed to a molecule id would have the fuel's
# atoms already wandering in their product groups while the fuel is still intact, tearing it.
# A smooth vector field sampled at the atom's own position is coherent for anything close
# together, at every instant, with no notion of membership at all. Wavelength ~6.5 A: a molecule
# is 1-3 A across, so its atoms sit in essentially one value of the field.
#
# ⚠ AND BOTH ARE CLOSED FORM IN `t` - sums of sinusoids, never an integrator. Same rule the rest
# of this engine runs on: a frame at t=9 costs what a frame at t=0 costs and a backward seek is
# free. A random walk would be neither.
#
# ⚠ THE AMPLITUDES ARE BOUNDED AND THE GATE MEASURES THEM. Vibration that stretches a bond by a
# tenth of its length is a molecule breathing; by half its length it is a different molecule.
# `selftest` asserts the worst bond-length excursion stays under 12%.

#: Brownian: amplitude in Angstrom, number of waves, spatial wavelength in Angstrom, Hz range.
#: ⚠ THE WAVELENGTH HAS TO GROW WITH THE PACKING DENSITY. The neighbourhood mean that keeps a
#: molecule rigid is only approximate - two atoms of one molecule see slightly different means -
#: and how much that residual matters is set by `2*pi*(delta q)/wavelen`. In a crystallising
#: sheet the molecules are packed 2.76 A apart, the means differ more than in a loose cluster,
#: and at 7 A the residual stretched an O-H by 25%. At 16 A it is negligible - and a long
#: wavelength is the RIGHT physics for a solid anyway: a crystal's thermal motion is collective
#: long-wavelength modes plus local vibration, which is exactly these two terms.
JIT_SLOW = dict(amp=0.13, waves=5, wavelen=16.0, hz=(0.11, 0.31))
#: Vibration: amplitude in Angstrom and Hz range, per atom, incoherent.
JIT_FAST = dict(amp=0.022, hz=(1.9, 4.4))
#: A bound on how much the FAST term alone can close a gap. The slow term is <= 0.31 Hz and is
#: sampled directly; the fast term runs to 4.4 Hz and would alias, so it is carried as a margin.
JIT_FAST_BOUND = 2.0 * 0.022 * 1.7321

#: The falloff of an atom's envelope, per unit of its own radius. Shared with the tubes so
#: `dist` stays one consistent quantity.
K_FALL = 4.0
#: Hydrogen-bond tube radius, in Angstrom. Deliberately thin - a hydrogen bond is a fifth the
#: strength of the O-H it sits next to, and the picture should say so.
HB_TUBE = 0.155


def water_network(duration=10.0, n_ring=6, scatter=2.05):
    """Six waters assembling into the RING HEXAMER, with a seventh arriving on top.

    ⚠ THE END STATE IS AUTHORED FIRST AND THE START IS DERIVED FROM IT. A hydrogen bond is
    DIRECTIONAL - the donor's O-H has to point at the acceptor's O - so a cluster cannot be
    arrived at by scattering molecules and hoping. The ring is built with valid donor
    orientations, then each molecule is walked backwards along its own OUTWARD RADIUS to a start
    pose. That also happens to make the trajectories non-crossing by construction, which is what
    stops spurious necks between molecules in flight (see the module note).

    The ring hexamer is a real, named structure - the smallest water cluster whose hydrogen
    bonds form a closed loop, each molecule donating one H to the next and accepting one from
    the previous.
    """
    sc = MolScene("water_network", duration)
    sp = water()
    ring_r = GEOM["HB_OO"]                       # hexagon: side == circumradius
    O = np.array([[ring_r * np.cos(2 * np.pi * i / n_ring),
                   ring_r * np.sin(2 * np.pi * i / n_ring), 0.0] for i in range(n_ring)])

    h1_local = _u(sp.pos[1])
    poses = []
    for i in range(n_ring):
        d = _u(O[(i + 1) % n_ring] - O[i])        # this molecule DONATES along here
        R0 = rot_align(h1_local, d)
        # ⚠ THE SPIN ABOUT THE DONATED O-H IS SOLVED, NOT GUESSED, and the first guess put two
        # molecules' hydrogens 1.07 A apart - a spurious neck the gate caught. Rotating about
        # `d` leaves the donated H on the acceptor, so the free H is free: scan the spin and
        # take the one that throws it furthest OUT of the ring plane, alternating side. That is
        # also what a real ring hexamer does with its free hydrogens.
        # ⚠ THE ALTERNATION STARTS DOWN, NOT UP, and the gate is why. The seventh molecule
        # arrives from ABOVE onto ring molecule 0, so a molecule-0 free hydrogen thrown UP is
        # thrown straight into the arriving donor - 1.32 A apart at t=6.75s. Which side the
        # alternation starts on is not arbitrary once something lands on the ring.
        sgn = -1.0 if i % 2 == 0 else 1.0
        best, bR = -9e9, R0
        for phi in np.linspace(0.0, 2.0 * np.pi, 361):
            Rf = rot_axis(d, phi) @ R0
            z = float((Rf @ sp.pos[2])[2])
            if sgn * z > best:
                best, bR = sgn * z, Rf
        poses.append((O[i], bR))

    # ⚠ THE SEVENTH SITS AT ONE HYDROGEN BOND FROM ITS ACCEPTOR, SOLVED. Placed by eye at
    # `[0, 0, 2.69]` it was 3.88 A from O[0] - the ring's circumradius and its height are two
    # different lengths - and its H...O came out 2.92 A, half an Angstrom past anything anyone
    # would call a hydrogen bond. Built by stepping O...O away from the acceptor instead.
    top = O[0] + GEOM["HB_OO"] * _u(-_u(O[0]) + 1.45 * np.array([0.0, 0.0, 1.0]))
    d7 = _u(O[0] - top)
    R7 = rot_align(h1_local, d7)
    best, bR = -9e9, R7
    for phi in np.linspace(0.0, 2.0 * np.pi, 361):
        Rf = rot_axis(d7, phi) @ R7
        z = float((Rf @ sp.pos[2])[2])
        if z > best:
            best, bR = z, Rf
    poses.append((top, bR))

    idx = []
    for (c, R) in poses:
        idx.append(sc.stamp(sp, c, R))

    # ---- the walk in: each molecule starts OUTWARD along its own radius ----------------
    n = len(poses)
    for m, (c, R) in enumerate(poses):
        out = _u(c if np.linalg.norm(c) > 1e-6 else np.array([0.0, 0.0, 1.0]))
        start = c + out * scatter
        # a slow tumble on the way in, so it does not read as a rigid slide
        spin = rot_axis(np.array([0.31, -0.62, 0.72]), np.radians(52.0 if m % 2 else -46.0))
        t0 = duration * (0.04 + 0.055 * m)
        t1 = t0 + duration * 0.30
        # a mid-flight waypoint so `relax_paths` has a key it may move, and so the walk in
        # reads as an arc rather than a slide - as a POSE, so the tumble stays rigid
        q0, q1 = mat_to_quat(spin @ R), mat_to_quat(R)
        tm = t0 + (t1 - t0) * 0.5
        ck = [(0.0, start), (t0, start), (tm, 0.5 * (start + c)), (t1, c)]
        qk = [(0.0, q0), (t0, q0), (tm, slerp(q0, q1, 0.5)), (t1, q1)]
        for k, gi in enumerate(idx[m]):
            sc.atoms[gi].rigid(sp.pos[k], ck, qk)
            sc.atoms[gi].mol = m

    # ---- the hydrogen bonds, each fading in as its donor lands -----------------------
    #: donor molecule m gives its atom-1 H to acceptor (m+1)'s O
    for m in range(n_ring):
        acc = (m + 1) % n_ring
        t1 = duration * (0.04 + 0.055 * max(m, acc)) + duration * 0.30
        sc.bonds.append(Bond(idx[m][1], idx[acc][0], order=0, kind="hb",
                             win=(t1 - duration * 0.06, t1 + duration * 0.05, 0.0, 0.0)))
    t1 = duration * (0.04 + 0.055 * (n - 1)) + duration * 0.30
    sc.bonds.append(Bond(idx[n - 1][1], idx[0][0], order=0, kind="hb",
                         win=(t1 - duration * 0.06, t1 + duration * 0.05, 0.0, 0.0)))
    relax_paths(sc, key=2)
    return sc


# --------------------------------------------------------------------------------------
# SCENE 2 - combustion
# --------------------------------------------------------------------------------------
def combustion(duration=10.0, n_c=7, seed=20260902):
    """C7H16 + 11 O2 -> 7 CO2 + 8 H2O, as a burn front travelling along the chain.

    ⚠ THE REACTION IS BALANCED AND THE TEST CHECKS IT. C_n H_(2n+2) needs (3n+1)/2 O2, which is
    an integer only for ODD n - heptane needs 11, octane needs 12.5. Picking octane and rounding
    to 12 would have rendered a reaction that does not conserve oxygen, with every framing and
    exposure statistic inside its bounds. `selftest` counts every element at both ends.

        C7H16 + 11 O2  ->  7 CO2 + 8 H2O
        C   7 = 7                          H  16 = 16
        O  22 = 14 (CO2) + 8 (H2O) = 22

    ⚠ IT BURNS AS A FRONT, NOT A FLASH. Every bond breaking at once is a full-frame event, which
    OUTPUT_RULES treats as a defect, and it also reads as an explosion rather than as burning.
    Staggering the break along the chain gives a travelling front, keeps any single frame's
    change small, and is closer to what a flame actually is.
    """
    assert n_c % 2 == 1, "an even-n alkane needs a half-integer number of O2"
    n_o2 = (3 * n_c + 1) // 2
    rng = np.random.default_rng(seed)
    sc = MolScene("combustion", duration)
    fuel = alkane(n_c)
    f_idx = sc.stamp(fuel, np.zeros(3))
    C_of = [i for i, e in enumerate(fuel.elems) if e == "C"]
    H_of = [i for i, e in enumerate(fuel.elems) if e == "H"]

    # ---- the oxygen, on a shell around the fuel -------------------------------------
    o_idx = []
    sp_o2 = o2()
    for k in range(n_o2):
        # a deterministic spiral on the sphere, so the O2 never bunch
        # ⚠ AN ANNULUS IN THE SCENE PLANE, NOT A SPHERE. The void field is a thin
        # screen-facing card, so anything far off the plane gets no carved hole and reads as a
        # soft blob rather than as variation C's rimmed silhouette. A spherical shell of oxygen
        # put most of it off-card. The small z spread keeps it from being a flat diagram.
        a = 2.0 * np.pi * (k + 0.5) / n_o2
        u = _u(np.array([np.cos(a), np.sin(a), 0.34 * np.sin(3.0 * a)]))
        # ⚠ STRETCHED ALONG THE CHAIN. A sphere of oxygen round a 7.7 A rod puts the
        # equatorial O2 on top of the chain's ends; the long axis needs more room than the
        # short ones, so the shell is an ellipsoid with the same aspect as the fuel.
        c = u * (4.7 + 0.7 * float(rng.random()))
        c[0] *= 1.38
        R = rot_axis(rng.normal(size=3), float(rng.random()) * np.pi)
        o_idx.append(sc.stamp(sp_o2, c, R))
    O_all = [g for pair in o_idx for g in pair]
    assert len(O_all) == 2 * n_o2

    # ---- the products, placed near where their atoms already are --------------------
    # ⚠ ASSIGNED BY PROXIMITY, and that is a look decision with a physical excuse. Atoms are
    # persistent, so a bad assignment sends an oxygen straight across the frame to a carbon it
    # was never near - which reads as teleporting, not as reacting. Greedy nearest keeps every
    # path short and local, which is also what a flame front actually does.
    # ⚠ `jitter=False`, and getting this wrong baked the thermal motion INTO the template:
    # `p_start` became a vibrating snapshot rather than the published geometry, and every C-C
    # bond in the fuel came out 1.42-1.63 A instead of 1.540. Anything that BUILDS the scene
    # must read the template; only the renderer wants the drawn positions.
    P0 = sc.positions(0.0, jitter=False)
    free = list(O_all)

    def take_nearest(p, k):
        got = []
        for _ in range(k):
            d = [float(np.linalg.norm(P0[g] - p)) for g in free]
            g = free.pop(int(np.argmin(d)))
            got.append(g)
        return got

    sp_co2, sp_h2o = co2(), water()
    groups = []        # (species, [global atom indices in template order])
    for ci in C_of:
        g = f_idx[ci]
        groups.append((sp_co2, [g] + take_nearest(P0[g], 2)))
    hs = [f_idx[i] for i in H_of]
    while hs:
        a = hs.pop(0)
        d = [float(np.linalg.norm(P0[b] - P0[a])) for b in hs]
        b = hs.pop(int(np.argmin(d)))
        p = 0.5 * (P0[a] + P0[b])
        groups.append((sp_h2o, take_nearest(p, 1) + [a, b]))

    # ⚠ PRODUCTS GO TO SLOTS ON A SPHERE, ASSIGNED BY DIRECTION - and the first build, which
    # just pushed each product outward from where its own atoms were, put an H and an O from two
    # DIFFERENT product molecules 0.53 A apart. That is a spurious neck between two things that
    # are not bonded, and it is indistinguishable from a real bond. Slots on a Fibonacci sphere
    # are separated by construction (15 of them at 7.0 A are ~6.4 A apart, against a 2.3 A CO2),
    # and assigning each product to the free slot nearest its OWN atoms' direction keeps every
    # path short and radial, so the trajectories diverge instead of crossing.
    # ⚠ TWO RINGS, ONE PER SPECIES, AND A RING IS NOT A SPHERE. Fifteen slots spread over a
    # SPHERE of radius 6.6 sit ~6 A apart; the same fifteen on a planar RING - which is what
    # they had to become so they would meet the void card - sit `2*pi*6.6/15` = 2.8 A apart,
    # and a CO2 is 2.3 A long. The gate found two product oxygens merged at their FINAL poses,
    # which no amount of path relaxation can fix because the end keyframes are the chemistry.
    # Splitting by species gives each ring a spacing matched to what sits on it: the eight
    # small waters inside, the seven long CO2 outside, where the arc is longer.
    R_IN, R_OUT = 4.0, 7.0
    rings = {}
    for (sp, gidx) in groups:
        rings.setdefault(sp.name, []).append((sp, gidx))
    prod = []
    for nm, items in rings.items():
        rad_r = R_IN if nm == "H2O" else R_OUT
        m = len(items)
        slots = []
        for k in range(m):
            a = 2.0 * np.pi * (k + 0.5) / m + (0.0 if nm == "H2O" else np.pi / max(m, 1))
            slots.append(np.array([rad_r * np.cos(a), rad_r * np.sin(a),
                                   0.30 * np.sin(2.0 * a)]))
        free_slot = list(range(m))
        for (sp, gidx) in items:
            cen0 = np.mean([P0[g] for g in gidx], axis=0)
            u = _u(cen0 if np.linalg.norm(cen0) > 1e-6 else np.array([1.0, 0.0, 0.0]))
            k = max(free_slot, key=lambda q: float(np.dot(_u(slots[q]), u)))
            free_slot.remove(k)
            R = rot_axis(rng.normal(size=3), float(rng.random()) * np.pi)
            prod.append((sp, slots[k], R, gidx))

    # ---- the front, and the walk to the products ------------------------------------
    # ⚠ THE FRONT IS CONTINUOUS IN POSITION, NOT ONE TIME PER PRODUCT, and the gate is what
    # forced that. A per-product break time puts ADJACENT atoms in completely different states -
    # one hydrogen departs at t=2.6 while the one next to it on the same carbon waits until
    # t=5.0, so the moving one sweeps past the stationary one (0.77 A at t=3.00s). A front that
    # is a smooth function of x gives neighbours nearly equal departure times, so they leave
    # together and then diverge toward their own products, which never brings them closer than
    # they already were. It is also the more honest picture of a flame front: the front reaches
    # an atom, that atom lets go.
    t_lo, t_hi = duration * 0.24, duration * 0.66
    xs = [float(P0[g][0]) for g in f_idx]
    x0, x1 = min(xs), max(xs)

    def front(x):
        u = (float(x) - x0) / max(x1 - x0, 1e-6)
        return t_lo + (t_hi - t_lo) * min(max(u, 0.0), 1.0)

    #: every atom of a product ARRIVES together even though they LEAVE when the front reaches
    #: each of them - otherwise a molecule assembles limb by limb.
    when = {}
    for q, (_s, _c, _R, gidx) in enumerate(prod):
        when[q] = max(front(P0[g][0]) for g in gidx)

    for q, (sp, cen, R, gidx) in enumerate(prod):
        t_b = when[q]
        for k, g in enumerate(gidx):
            a = sc.atoms[g]
            t_g = front(P0[g][0])          # this atom's own departure
            p_start = P0[g]
            p_end = cen + R @ sp.pos[k]
            # ⚠ THE FRAGMENTS BOW OUTWARD ON THE WAY, and it is not decoration. Straight lines
            # from reactant to product cross each other - an oxygen leaving the shell for a
            # carbon in the middle cuts across an oxygen heading the other way, and the gate
            # caught two of them 0.84 A apart. Pushing every path out along its own radius makes
            # them diverge, which is also what fragments flying off a burn front actually do.
            mid = 0.5 * (p_start + p_end)
            # ⚠ THE BOW IS PERPENDICULAR TO THE CHAIN, and that is the fix the gate forced. A
            # fragment leaving an early carbon travels ALONG the rod toward its slot and passes
            # straight through the part of the fuel that has not burned yet - a carbon and a
            # hydrogen 0.94 A apart at t=3.75s, which reads as a bond forming between a burnt
            # fragment and an intact one. Throwing every fragment SIDEWAYS off the chain first
            # clears the rod immediately, and it is what a front blowing pieces off actually
            # looks like. `cen` is the product's slot, so each fragment leaves on its own
            # bearing and they diverge rather than sharing a lane.
            w = np.array([0.0, cen[1], cen[2]])
            w = _u(w) if np.linalg.norm(w) > 1e-6 else np.array([0.0, 0.0, 1.0])
            bow = mid + w * 2.2
            t_e = min(t_b + duration * 0.26, duration)
            t_g = min(t_g, t_e - 1e-3)
            a.keys = [(0.0, p_start), (t_g, p_start),
                      (t_g + (t_e - t_g) * 0.55, bow), (t_e, p_end)]
            a.mol = 100 + q
        t_end = min(t_b + duration * 0.26, duration)
        for (i, j, o) in sp.bonds:
            sc.bonds.append(Bond(gidx[i], gidx[j], o, phase=1,
                                 win=(t_end - duration * 0.06, t_end, 0.0, 0.0)))

    # every REACTANT bond fades out as its own front passes
    owner = {}
    for q, (_s, _c, _R, gidx) in enumerate(prod):
        for g in gidx:
            owner[g] = q
    for b in sc.bonds[:len(fuel.bonds) + n_o2]:
        t_b = min(front(P0[b.i][0]), front(P0[b.j][0]))
        b.win = (0.0, 0.0, t_b, t_b + duration * 0.08)
    relax_paths(sc, key=2)
    return sc


def merge_distance(ra, rb, k=K_FALL):
    """The separation at which two envelopes stop being one blob. Solved, not assumed."""
    lo, hi = 1e-3, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _neck(ra, rb, mid, k) >= 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def relax_paths(sc: MolScene, key=2, iters=130, n_t=201, pad=1.10, step=0.45):
    """Push TRANSIT waypoints apart until no two unbonded atoms of different molecules merge.

    ⚠ THIS EXISTS BECAUSE FIXING SPURIOUS NECKS ONE AT A TIME DOES NOT CONVERGE. Four separate
    choreography changes each cleared the collision the gate had just named and produced a new
    one somewhere else - a carbon and a hydrogen, then two hydrogens, then a hydrogen and an
    oxygen. Forty-five atoms flying between two arrangements have ~1000 pairs and no amount of
    art-directing the departure order addresses the general case. This does: sample the whole
    trajectory, find every pair that merges, and push the pair apart at the one keyframe that is
    free to move.

    ⚠ ONLY THE MID-FLIGHT WAYPOINT MOVES. The first and last keyframes ARE the published
    molecular geometry - `selftest` measures every bond length against the template at both ends
    - so relaxing them would fix the picture by falsifying the chemistry. The waypoint is the
    only key that no assertion depends on, which is exactly why it is the one to solve with.

    ⚠ AND ONLY ATOMS ACTUALLY IN TRANSIT ARE PUSHED. Moving the waypoint of an atom that is
    parked at either end does nothing at that instant, so pushing it is a wasted correction that
    can only make some other frame worse.
    """
    n = len(sc.atoms)
    if n < 2:
        return 0
    rad = sc.radii()
    elems = [a.elem for a in sc.atoms]
    uniq = sorted(set(elems))
    need = {}
    for a in uniq:
        for b in uniq:
            need[(a, b)] = merge_distance(ELEMENTS[a]["radius"], ELEMENTS[b]["radius"])
    NEED = np.array([[need[(elems[i], elems[j])] for j in range(n)] for i in range(n)])
    # ⚠ THE THERMAL MOTION IS A MARGIN, NOT SOMETHING TO SAMPLE AGAINST. The vibration runs at
    # up to 4.4 Hz, so 81 samples over a 10 s clip alias it hopelessly and relaxing against the
    # sampled positions would be chasing a value the next frame does not have. Relax the
    # TEMPLATE paths instead and widen the requirement by twice the measured worst excursion,
    # which is a bound rather than a hope.
    # ⚠ A MEASURED MARGIN, NOT A WORST-CASE BOUND. Adding twice the largest thermal
    # excursion assumes the two atoms are exactly anti-phase at their closest approach, which
    # a spatially coherent field almost never is - it inflated every requirement by ~0.7 A and
    # failed pairs sitting a comfortable 1.6 A apart. The slow term is at most 0.31 Hz, so it
    # is sampled honestly below; only the fast term aliases, and only it needs a margin.
    NEED = NEED * pad + JIT_FAST_BOUND
    np.fill_diagonal(NEED, 0.0)

    mol = np.array([a.mol for a in sc.atoms])
    same = mol[:, None] == mol[None, :]
    for b in sc.bonds:
        same[b.i, b.j] = same[b.j, b.i] = True
    np.fill_diagonal(same, True)

    # which atoms have a movable waypoint, and over what window it is live
    movable = np.array([len(a.keys) > key + 1 for a in sc.atoms])
    t_lo = np.array([a.keys[1][0] if len(a.keys) > key + 1 else 1e9 for a in sc.atoms])
    t_hi = np.array([a.keys[-1][0] if len(a.keys) > key + 1 else -1e9 for a in sc.atoms])

    ts = np.linspace(0.0, sc.duration, n_t)
    worst = 0.0
    for _ in range(iters):
        worst, moved = 0.0, False
        for t in ts:
            P = sc.positions(t)
            D = P[:, None, :] - P[None, :, :]
            d = np.sqrt((D * D).sum(2)) + np.eye(n) * 1e3
            bad = (d < NEED) & (~same)
            if not bad.any():
                continue
            live = movable & (t > t_lo) & (t < t_hi)
            ii, jj = np.nonzero(np.triu(bad))
            for i, j in zip(ii, jj):
                gap = float(NEED[i, j] - d[i, j])
                worst = max(worst, gap)
                u = D[i, j] / max(float(d[i, j]), 1e-9)
                for a, sgn in ((i, +1.0), (j, -1.0)):
                    if live[a]:
                        tk, pk = sc.atoms[a].keys[key]
                        sc.atoms[a].keys[key] = (tk, pk + sgn * u * gap * step)
                        moved = True
        if not moved or worst < 1e-4:
            break
    return worst


# --------------------------------------------------------------------------------------
# SCENE 3 - CRYSTALLIZATION
# --------------------------------------------------------------------------------------
#: How far the lines bow off the straight H...O axis, as a fraction of its length.
#:
#: ⚠ 0.46 WITH NINE LINES WAS A WALLPAPER, NOT A BOND. At that bow the family opens into a lens
#: as wide as the gap is long, and forty-five of them tiled across the frame became the subject:
#: the water they were supposed to be joining had disappeared behind them. A hydrogen bond is a
#: weak interaction between two molecules and the picture has to keep saying that - the lines
#: are a tracery over the structure, never the structure.
HB_BOW = 0.20
#: Field lines per hydrogen bond, and samples along each.
#:
#: ⚠ MORE LINES AND EACH ONE DIMMER, WHICH IS THE ONLY WAY BOTH ASKS FIT. Clancy wanted *"more
#: field lines"* and *"the many charge lines should be semi transparent"*, and those pull
#: opposite ways on total flux - eleven lines at five lines' brightness is a solid object. In an
#: additive accumulator "semi transparent" is not an alpha channel, it is simply a per-line gain
#: low enough that the line never saturates: everything behind it still adds through, so the
#: structure reads past the lines instead of being occluded by them. The count went 5 -> 11 and
#: `render.LINE_GAIN` came down to match, leaving total flux roughly where it was.
HB_LINES, HB_SAMPLES = 11, 96


#: How far past the FRAMED region the lattice is generated, in Angstrom.
#:
#: ⚠ A SHEET HAS TO HAVE NEIGHBOURS IT NEVER SHOWS YOU. Generating the lattice only as far as
#: the frame leaves the outermost molecules with nothing to bond to on the outward side, so they
#: sit there with dangling hydrogens and the picture reads as the EDGE OF A CRYSTAL rather than
#: as a sheet continuing past the frame - which is exactly what Clancy saw down the left side.
#: The margin has to be at least one nearest-neighbour distance (2.76 A) so that every molecule
#: the viewer can see has its full set of three; those extra molecules are then incomplete in
#: turn, one full period off-screen where nobody looks.
SHEET_MARGIN = 3.1


def ice_sheet(hx=8.2, hy=14.6, d=None, margin=None):
    """A honeycomb sheet of water sites - ice Ih seen down its c axis.

    ⚠ THE RING HEXAMER GENERALISES INTO THIS, which is why the earlier clip was worth building:
    ice Ih's motif IS the six-membered ring, and a honeycomb lattice of oxygens at the same
    O...O distance is what you see looking down the c axis. So "crystallization" is the same
    structure the previous scene assembled ONE of, tiled.

    ⚠ HONESTY CAVEAT: a real ice Ih layer is PUCKERED and every oxygen has FOUR neighbours -
    three in the layer and one out of it. This is a flat, three-coordinate projection, which is
    what a planar view can carry (and what the thin void card requires). The consequence is
    visible in the bond count and is not hidden: three neighbours per site is 1.5 hydrogen bonds
    per molecule, against the two hydrogens a water actually has, so some hydrogens DANGLE. That
    is a real feature of a surface layer, and it is why `_donors` caps donation at two rather
    than assuming every hydrogen finds a partner.
    """
    d = float(GEOM["HB_OO"] if d is None else d)
    m_ = float(SHEET_MARGIN if margin is None else margin)
    hx, hy = float(hx) + m_, float(hy) + m_
    a1 = np.array([1.5 * d, 0.8660254 * d, 0.0])
    a2 = np.array([1.5 * d, -0.8660254 * d, 0.0])
    basis = [np.zeros(3), np.array([d, 0.0, 0.0])]
    sites, rng = [], np.random.default_rng(4242)
    m = int(hx / d) + int(hy / d) + 4
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            for b in basis:
                p = i * a1 + j * a2 + b
                if abs(p[0]) <= hx and abs(p[1]) <= hy:
                    # a small out-of-plane offset, the flat sheet's nod to the real pucker -
                    # kept well inside the void card's depth
                    sites.append(np.array([p[0], p[1], 0.18 * ((i + j) % 2 * 2 - 1)]))
    S = np.stack(sites)
    # nearest-neighbour edges
    D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    edges = [(i, j) for i in range(len(S)) for j in range(i + 1, len(S))
             if D[i, j] < d * 1.18]
    return S, edges


def _donors(n, edges, cap=2):
    """Give every edge a donor, no site donating more than `cap` hydrogens.

    ⚠ GREEDY WITH A CAP, and the cap is the physics. A honeycomb site has three neighbours but a
    water has two hydrogens, so the edges cannot all be donated from one end - assigning them
    blindly gives some molecule three O-H bonds pointing outward, which is not water. Total
    edges are `1.5 n` against a capacity of `2 n`, so a greedy pass always fits.
    """
    out = {i: [] for i in range(n)}
    acc = {i: 0 for i in range(n)}
    for (i, j) in sorted(edges, key=lambda e: (len(out[e[0]]) + len(out[e[1]]))):
        a, b = (i, j) if len(out[i]) <= len(out[j]) else (j, i)
        if len(out[a]) < cap:
            out[a].append(b)
        elif len(out[b]) < cap:
            out[b].append(a)
        acc[b] = acc.get(b, 0) + 1
    return out


def crystallization(duration=10.0, hx=7.6, hy=13.5, seed=20260903, expand=1.20):
    """A disordered sheet of water ORDERING into ice, from a nucleus outward.

    ⚠ CRYSTALLIZATION IS ORDERING, NOT TRANSPORT - which is both the honest picture and the
    reason this scene needs no path relaxation. Each molecule ends within ~1 A of where it
    started; what changes is its ORIENTATION, from random to the one its hydrogen bonds demand.
    Nothing travels across the frame, so no two molecules can swap places or pass through each
    other, and `relax_paths` (which is O(n^2) over 201 sample times and would be hopeless at 150
    atoms) is not needed at all.

    ⚠ THE FRONT SPREADS FROM A NUCLEUS. Ordering everywhere at once is a full-frame event, and
    it is also not what freezing looks like: a crystal has a seed and it grows.
    """
    rng = np.random.default_rng(seed)
    S, edges = ice_sheet(hx, hy)
    n = len(S)
    donors = _donors(n, edges)
    sp = water()
    h1, h2 = _u(sp.pos[1]), _u(sp.pos[2])
    bl, nl = _u(h1 + h2), _u(np.cross(h1, h2))
    F_loc = np.stack([bl, nl, np.cross(bl, nl)])

    sc = MolScene("crystallization", duration, seed=seed)
    #: 7.6 x 13.5 is 9:16, so width and height reach the frame edge together.
    sc.fill_extent = (float(hx), float(hy))
    idx = []
    poses = []
    for i in range(n):
        tg = [_u(S[j] - S[i]) for j in donors[i]]
        if len(tg) >= 2:
            # ⚠ BOTH HYDROGENS BEST-FIT AT ONCE, not one aimed and the other left. The lattice
            # asks for 120 degrees between neighbours and a water offers 104.5, so neither H can
            # point exactly at its acceptor; matching the BISECTOR and the plane splits the
            # residual evenly and gives each bond the same ~8 degree bend. Real hydrogen bonds
            # in ice are bent by about this much for exactly this reason.
            b, nn = _u(tg[0] + tg[1]), _u(np.cross(tg[0], tg[1]))
            R = np.stack([b, nn, np.cross(b, nn)]).T @ F_loc
        elif len(tg) == 1:
            # ⚠ THE SPARE HYDROGEN POINTS OUT OF THE SHEET, and spinning it at random put it
            # down an edge some other molecule had already donated along - two hydrogens on one
            # bond, 0.81 A apart, which is a merged pair and a fake bond. This is the
            # Bernal-Fowler ice rule in the only form a flat sheet can honour: one hydrogen per
            # bond, and whatever is left over dangles perpendicular to the layer. Solved by
            # scanning the spin, the same way the hexamer's free hydrogen is placed.
            R = rot_align(h1, tg[0])
            sgn = 1.0 if i % 2 == 0 else -1.0
            best, bR = -9e9, R
            for phi in np.linspace(0.0, 2.0 * np.pi, 181):
                Rf = rot_axis(tg[0], phi) @ R
                z = float((Rf @ sp.pos[2])[2])
                if sgn * z > best:
                    best, bR = sgn * z, Rf
            R = bR
        else:
            # no partner at all: both hydrogens out of the layer
            R = rot_align(_u(h1 + h2), np.array([0.0, 0.0, 1.0 if i % 2 == 0 else -1.0]))
        poses.append(R)
        idx.append(sc.stamp(sp, S[i], R, mol=i))

    # ---- the front ------------------------------------------------------------------
    nuc = np.array([0.0, -hy * 0.55, 0.0])
    dist = np.linalg.norm(S - nuc[None, :], axis=1)
    far = float(dist.max()) + 1e-6
    t_lo, t_hi = duration * 0.10, duration * 0.80
    freeze = t_lo + (t_hi - t_lo) * (dist / far)

    for i in range(n):
        R = poses[i]
        # the disordered pose: same site, nudged, and turned any which way
        # ⚠ THE DISORDERED SHEET IS EXPANDED, AND THAT IS WHAT MAKES RANDOM ORIENTATION
        # LEGAL. On the lattice the oxygens sit 2.76 A apart, so two molecules turned to face
        # each other put their hydrogens 2.76 - 2*0.96 = 0.84 A apart - under the 0.99 A at
        # which two hydrogens merge, so the disordered state grew fake bonds everywhere. At
        # 1.20x spacing the same worst case is 1.55 A and nothing touches. It also reads
        # correctly: the sheet CONTRACTS into order, which is what condensing into a solid
        # looks like, and it keeps the frame overfilled at the start.
        off = rng.normal(size=3) * np.array([0.30, 0.30, 0.14])
        P_dis = S[i] * float(expand) + off
        R0 = rot_axis(rng.normal(size=3), float(rng.random()) * 2 * np.pi)
        t1 = float(freeze[i])
        t0 = max(t1 - duration * 0.20, 0.0)
        q0, q1 = mat_to_quat(R0), mat_to_quat(R)
        ck = [(0.0, P_dis), (t0, P_dis), (t1, S[i])]
        qk = [(0.0, q0), (t0, q0), (t1, q1)]
        for k, g in enumerate(idx[i]):
            sc.atoms[g].rigid(sp.pos[k], ck, qk)

    # ---- the hydrogen bonds, drawn rather than fielded ------------------------------
    for i in range(n):
        for jj, j in enumerate(donors[i]):
            t1 = max(float(freeze[i]), float(freeze[j]))
            sc.bonds.append(Bond(idx[i][1 + min(jj, 1)], idx[j][0], order=0, kind="hb",
                                 win=(t1, t1 + duration * 0.06, 0.0, 0.0)))
    return sc


# ======================================================================================
# THE CHARGE FIELD LINES
# ======================================================================================
class MolLines:
    """Hydrogen bonds drawn as bowed field lines, coloured by charge.

    Clancy: *"Change the solid 'bond' line ... to red and blue, based on their charge. and rather
    than a straight line, make sure these are 'magnetic field' lines showing red/blue. dont show
    arrows."*

    ⚠ RED IS THE OXYGEN AND BLUE IS THE HYDROGEN, which is not arbitrary - it is the convention
    every electrostatic-potential map in chemistry uses: RED = electron-rich = negative, BLUE =
    electron-poor = positive. A water's oxygen carries the partial negative and its hydrogens
    the partial positive, so a hydrogen bond runs blue at the H end to red at the O end. Getting
    it backwards would be a plausible, completely wrong picture - the failure class this project
    keeps meeting - so the key is derived from the ATOM'S ELEMENT, never from bond order.

    ⚠ A FAMILY OF BOWED CURVES, NOT ONE LINE, is what makes it read as a FIELD rather than a
    stick. The lines leave the hydrogen spread around the axis, bow outward, and converge on the
    oxygen - which is the shape of a dipole field between two opposite charges. Amplitudes are
    staggered so the family nests instead of forming a tube.

    ⚠ AND NO ARROWS, ASKED FOR BY NAME - so nothing here is directional. The colour ramp does
    the whole job of saying which end is which, which is why it must be a DIVERGING ramp and not
    a brightness gradient.
    """

    def __init__(self, sc: MolScene, xp, asarray):
        self.sc, self.xp, self.asarray = sc, xp, asarray
        k, m = int(HB_LINES), int(HB_SAMPLES)
        self.k, self.m = k, m
        self.s = np.linspace(0.0, 1.0, m)
        self.phi = np.arange(k) * (2.0 * np.pi / k)
        # staggered bow so the family nests rather than forming a tube
        # ⚠ THE AMPLITUDE LADDER HAS TO BE FINER AS THE COUNT GROWS. At three distinct bows,
        # eleven lines are three coincident bundles rather than a spread family - and coincident
        # lines defeat the translucency, because they stack back up to a solid core.
        self.bow = HB_BOW * (0.22 + 0.78 * (np.arange(k) / max(k - 1, 1)))

    def build(self, R, t):
        """(P, key, gain) in world units, or None when no hydrogen bond is live."""
        sc = self.sc
        live = [(b, b.strength(t)) for b in sc.bonds if b.kind == "hb"]
        live = [(b, w) for (b, w) in live if w > 1e-3]
        if not live:
            return None
        P = sc.positions(t)
        s, k, m = self.s, self.k, self.m
        env = np.sin(np.pi * s)                       # bow profile, zero at both ends
        segs, keys, gains = [], [], []
        for (b, w) in live:
            #: ⚠ FROM THE HYDROGEN TO THE OXYGEN, resolved by ELEMENT. `b.i`/`b.j` order is a
            #: bookkeeping detail; the colour is a physical fact about which atom is which.
            i, j = (b.i, b.j) if sc.atoms[b.i].elem == "H" else (b.j, b.i)
            A, B = P[i], P[j]
            ax = B - A
            L = float(np.linalg.norm(ax)) + 1e-9
            u = ax / L
            e1 = np.cross(u, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(u, np.array([0.0, 1.0, 0.0]))
            e1 = _u(e1)
            e2 = np.cross(u, e1)
            base = A[None, :] + ax[None, :] * s[:, None]              # (m,3)
            for q in range(k):
                p = e1 * np.cos(self.phi[q]) + e2 * np.sin(self.phi[q])
                segs.append(base + (p[None, :] * (self.bow[q] * L))
                            * env[:, None])
                keys.append(s)
                gains.append(np.full(m, w))
        Pw = np.concatenate(segs) * R
        return (self.asarray(Pw.astype(np.float32)),
                self.asarray(np.concatenate(keys).astype(np.float32)),
                self.asarray(np.concatenate(gains).astype(np.float32)))


#: ⚠ `combustion` IS SCRAPPED AS A DELIVERABLE (Clancy, 2026-09-03: *"Lets scrap the combustion
#: one. we will for now, focus on making this a 'crystallization' view."*) but it is KEPT HERE
#: on purpose - "for now" is not "never", the alkane / CO2 / O2 templates are the reusable layer
#: he asked for, and a builder nobody tests is a builder that rots. `moltest` still runs it.
_BUILDERS = {"water_network": water_network, "combustion": combustion,
             "crystallization": crystallization}
_CACHE = {}


def build(name, duration=10.0):
    """A scene, built once per process.

    ⚠ CACHED BECAUSE `relax_paths` IS NOT CHEAP - it sweeps 201 sample times up to 130 times to
    push ~1000 atom pairs apart. `fit_scales` and `render.Orbital` each ask for the scene, and
    a dry run asks again, so an uncached build pays that three times for an identical result.
    Deterministic from its seed, so one instance is the same as three.
    """
    key = (name, round(float(duration), 6))
    if key not in _CACHE:
        _CACHE[key] = _BUILDERS[name](duration=duration)
    return _CACHE[key]


class _Scenes(dict):
    """`SCENES[name](duration=...)` still works, and now goes through the cache."""

    def __getitem__(self, k):
        return lambda duration=10.0, _k=k: build(_k, duration)

    def items(self):
        return [(k, self[k]) for k in _BUILDERS]

    def __iter__(self):
        return iter(_BUILDERS)

    def keys(self):
        return _BUILDERS.keys()


SCENES = _Scenes()


# ======================================================================================
# THE FIELD
# ======================================================================================
class MolField:
    """Total electron density as a union of atom envelopes plus hydrogen-bond tubes.

    Exposes `dist(P, R, sep)` / `edge` / `fade` so `void.VoidField.build` can carve with it
    exactly as it carves with a `bond.BondField` - the carve is a membership test, and it does
    not care what answers it.
    """

    def __init__(self, sc: MolScene, xp, asarray, edge=0.055, fade=0.70):
        self.sc = sc
        self.xp = xp
        self.asarray = asarray
        self.k = float(K_FALL)
        self.edge, self.fade = float(edge), float(fade)
        self.rad = sc.radii()
        self.t = 0.0
        self.set_time(0.0)

    def set_time(self, t):
        """Cache this frame's geometry. Called once per frame, before any evaluation."""
        self.t = float(t)
        self.pos = self.sc.positions(t)
        self.tA, self.tB, self.tR, self.tS = self.sc.tubes(t)

    # ---------------------------------------------------------------------------------
    def field(self, P, R):
        """Sum of every primitive at every point of `P`. `R` is world px per Angstrom.

        ⚠ ACCUMULATED IN A LOOP OVER PRIMITIVES, NEVER BROADCAST. The obvious
        `(n_points, n_atoms)` intermediate is 1.5M x 45 floats - a quarter of a gigabyte per
        frame, allocated and freed 300 times a clip. The loop is the same arithmetic with one
        temporary.
        """
        xp, f32 = self.xp, np.float32
        F = xp.zeros(P.shape[0], np.float32)
        k = f32(self.k)
        for c, rr in zip(self.pos, self.rad):
            dx = P[:, 0] - f32(c[0] * R)
            dy = P[:, 1] - f32(c[1] * R)
            dz = P[:, 2] - f32(c[2] * R)
            d = xp.sqrt(dx * dx + dy * dy + dz * dz)
            F = F + xp.exp(xp.clip(k * (f32(1.0) - d / f32(rr * R)), f32(-40.0), k))
        for a, b, rr, ss in zip(self.tA, self.tB, self.tR, self.tS):
            F = F + f32(ss) * xp.exp(xp.clip(
                k * (f32(1.0) - self._seg(P, a * R, b * R) / f32(rr * R)),
                f32(-40.0), k))
        return F

    def _seg(self, P, a, b):
        """Distance from every point of `P` to the segment ab."""
        xp, f32 = self.xp, np.float32
        ab = b - a
        L2 = float(np.dot(ab, ab)) + 1e-12
        px = P[:, 0] - f32(a[0])
        py = P[:, 1] - f32(a[1])
        pz = P[:, 2] - f32(a[2])
        s = xp.clip((px * f32(ab[0]) + py * f32(ab[1]) + pz * f32(ab[2])) / f32(L2),
                    f32(0.0), f32(1.0))
        qx = px - s * f32(ab[0])
        qy = py - s * f32(ab[1])
        qz = pz - s * f32(ab[2])
        return xp.sqrt(qx * qx + qy * qy + qz * qz)

    def dist_blocks(self, P, R, pos, rad, per):
        """`dist` on a cloud laid out as one contiguous block per atom, culled by neighbour."""
        xp, f32 = self.xp, np.float32
        n = len(pos)
        Dm = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        reach = 1.90 * rad[:, None] + 2.25 * rad[None, :]
        near = Dm <= reach
        out = xp.zeros(P.shape[0], np.float32)
        k = f32(self.k)
        for i in range(n):
            lo, hi = i * per, (i + 1) * per
            blk = P[lo:hi]
            F = xp.zeros(per, np.float32)
            for j in np.nonzero(near[i])[0]:
                c = pos[j]
                dx = blk[:, 0] - f32(c[0] * R)
                dy = blk[:, 1] - f32(c[1] * R)
                dz = blk[:, 2] - f32(c[2] * R)
                dd = xp.sqrt(dx * dx + dy * dy + dz * dz)
                F = F + xp.exp(xp.clip(k * (f32(1.0) - dd / f32(rad[j] * R)),
                                       f32(-40.0), k))
            out[lo:hi] = -xp.log(xp.maximum(F, f32(1e-12))) / k
        return out

    def dist(self, P, R, sep=0.0):
        """Signed OUTWARD distance from the density surface, 0 on it - `bond.dist`'s units."""
        xp = self.xp
        return -xp.log(xp.maximum(self.field(P, R), np.float32(1e-12))) / np.float32(self.k)


# ======================================================================================
# THE CLOUD - one canonical sphere per ELEMENT, instanced per atom
# ======================================================================================
#: Directions and radial layers per atom. Far smaller than `bond.ShellCloud`'s because there
#: are 21-45 atoms rather than 2, and each is a much smaller thing on screen.
#: ⚠ AND A RADIAL JITTER, WHICH IS NOT COSMETIC. Without it every dot of an atom sits on one
#: of `MOL_RAD` exact shells, so wherever the surface crosses a shell it lights a whole ring at
#: once and the atom renders as a set of concentric circles - the Fibonacci spiral read end-on.
#: Jittering each dot by half a layer spacing turns the shells into a continuum. Free, and the
#: same lesson `flow.porosity` learned needing RUNS rather than per-point noise: what a regular
#: sampling structure does to an accumulator is visible long before it is statistical.
MOL_DIRS, MOL_RAD, MOL_JIT = 2600, 22, 1.0
MOL_RHO = (0.28, 1.85)
#: The shell kernel, in `dist` units - the same asymmetric shape `bond.py` settled on: a thin
#: hot skin outside, a slow falloff inside, so the body carries structure instead of being an
#: empty bubble and the void field's rim keeps the boundary to itself.
MOL_W_OUT, MOL_W_IN = 0.020, 0.42
MOL_POR = 0.55


class MolCloud:
    """A fixed dot cloud per atom; the density surface lights the dots it passes through.

    ⚠ ONE CANONICAL SPHERE PER ELEMENT, INSTANCED BY TRANSLATION - not one cloud per atom.
    Forty-five atoms of three elements is three point sets and forty-five offsets; building it
    per atom would be forty-five times the setup and forty-five times the memory for the same
    45 x 21k points, and every H would carry its own private copy of an identical sphere.
    """

    def __init__(self, sc: MolScene, fld, xp, asarray, seed=20260902,
                 n_dirs=None, n_rad=None):
        self.sc, self.fld, self.xp, self.asarray = sc, fld, xp, asarray
        n_d = int(MOL_DIRS if n_dirs is None else n_dirs)
        n_r = int(MOL_RAD if n_rad is None else n_rad)
        i = np.arange(n_d, dtype=np.float64) + 0.5
        cz = 1.0 - 2.0 * i / n_d
        sr = np.sqrt(np.maximum(1.0 - cz * cz, 0.0))
        a = np.pi * (3.0 - np.sqrt(5.0)) * i
        U = np.stack([sr * np.cos(a), cz, sr * np.sin(a)], 1)
        lo, hi = MOL_RHO
        rho = lo + (np.arange(n_r, dtype=np.float64) + 0.5) / n_r * (hi - lo)
        rng = np.random.default_rng(seed)
        step = (hi - lo) / n_r
        rj = rho[None, :] + (rng.random((n_d, n_r)) - 0.5) * step * MOL_JIT
        unit = (rj[:, :, None] * U[:, None, :]).reshape(-1, 3)         # per unit radius
        # ⚠ `rho^2` is the volume of the shell that layer represents - `_arc_gain`'s argument
        # again. Without it the innermost layers, which stand for almost no volume, carry the
        # same flux as the outer ones and every atom renders with a bright core.
        w = (rj ** 2).reshape(-1)

        self.n_at = len(sc.atoms)
        self.per = unit.shape[0]
        self.n = self.per * self.n_at
        self.rad = sc.radii()
        big = np.concatenate([unit * r for r in self.rad])             # (n, 3), in Angstrom
        ww = np.concatenate([w * (r ** 2) for r in self.rad])
        ww = ww * (1.0 + MOL_POR * (rng.random(ww.size) * 2.0 - 1.0))
        self.unit = asarray(big.astype(np.float32))
        self.w = asarray((ww / max(float(ww.mean()), 1e-12)).astype(np.float32))
        self.reach = float(hi * self.rad.max())

    def build(self, R, t, split=0.0):
        """(P, key, gain) - the whole assembly's cloud in world units at time `t`.

        ⚠ THE FIELD IS EVALUATED PER ATOM BLOCK OVER ITS NEIGHBOURS ONLY, and at 144 atoms that
        is the difference between a render and an afternoon. The envelope is `exp(k(1-q))` with
        k=4, so an atom contributes under 1% of the surface level past ~2.2 of its own radii -
        about 1.5 A. A dot belonging to atom `i` lies within 1.85 of `i`'s radius, so only atoms
        within ~3.5 A can matter to it, and in a molecular assembly that is a handful rather
        than all 144. Naively: 2M dots x 144 atoms = 290M evaluations a frame. Culled: ~7x
        fewer atoms each, and the neighbour list is a 144x144 distance matrix on the CPU.
        """
        xp, f32 = self.xp, np.float32
        self.fld.set_time(t)
        pos = self.sc.positions(t)
        off = self.asarray((np.repeat(pos, self.per, axis=0) * R).astype(np.float32))
        P = self.unit * f32(R) + off
        d = self.fld.dist_blocks(P, R, pos, self.rad, self.per) - f32(split)
        wo, wi = f32(MOL_W_OUT), f32(MOL_W_IN)
        sgl = xp.where(d >= 0, d / wo, d / wi)
        gain = self.w * xp.exp(-sgl * sgl)
        key = xp.clip(f32(0.20) + f32(0.44)
                      * (f32(1.0) - xp.clip(xp.abs(d) / f32(0.45), f32(0.0), f32(1.0))),
                      f32(0.0), f32(1.0))
        return P, key, gain


# ======================================================================================
# THE GATE - `cli.py moltest`
# ======================================================================================
def _neck(ra, rb, d, k=K_FALL, n=257):
    """The SADDLE of the two-atom field on the line between the centres. >= 1 means MERGED.

    ⚠ THE MINIMUM, NOT THE MAXIMUM, AND THE FIRST VERSION TOOK THE MAXIMUM. Two blobs are one
    blob exactly when the field never dips below the surface level on the line joining them -
    that is what connected MEANS. The maximum along that line is the value at an atom's own
    centre, `exp(k)` = 54.6, for every pair at every distance: the check reported the same
    number whether the atoms were 0.5 A or 3 A apart, so it failed everything and proved
    nothing. Two of the failures it reported were real and the rest were the metric.

    A test can be broken in the direction of FAILING, and it looks like diligence.
    """
    x = np.linspace(0.0, d, n)
    f = np.exp(np.clip(k * (1 - x / ra), -40, k)) + np.exp(np.clip(k * (1 - (d - x) / rb),
                                                                  -40, k))
    return float(f.min())


def selftest(verbose=True):
    """Assert both molecular scenes are the molecules and the reaction they claim.

    ⚠ NONE OF THIS IS VISIBLE TO `dry`, which is the same argument `bondtest` makes. A scene
    with a mis-specified reaction, a distorted template or a fake bond has entirely ordinary
    framing, exposure, motion and coverage.

    1. STOICHIOMETRY. Atoms persist, so every element's count is trivially constant - what is
       NOT trivial is that the products actually consume all of them. Checked by counting each
       element's atoms and asserting the product molecules use every one exactly once.
    2. GEOMETRY. Every registered covalent bond's length must match the template's published
       value while its molecule is intact - at t=0 for reactants, at the end for products.
    3. HYDROGEN BONDS are H...O, BETWEEN DIFFERENT MOLECULES, at a sane distance.
    4. NO SPURIOUS NECKS. Over the whole trajectory, no unbonded pair of atoms in DIFFERENT
       molecules may come close enough for the field to neck between them - a fake bond a
       viewer cannot tell from a real one. This is the pi lesson: the test that proves the
       specification does not prove the picture, so probe the geometry separately.
    """
    ok = True
    tol = 0.02

    for name, fn in SCENES.items():
        sc = fn()
        P0, PE = sc.positions(0.0, jitter=False), sc.positions(sc.duration, jitter=False)
        cnt = sc.counts()

        # ---- 1. stoichiometry ------------------------------------------------------
        seen = {}
        for a in sc.atoms:
            seen[a.elem] = seen.get(a.elem, 0) + 1
        if name == "combustion":
            n_c = cnt.get("C", 0)
            want_h, want_o = 2 * n_c + 2, 2 * ((3 * n_c + 1) // 2)
            if cnt.get("H") != want_h or cnt.get("O") != want_o:
                print(f"[moltest] FAIL {name}: C{n_c} needs H{want_h} O{want_o}, "
                      f"got H{cnt.get('H')} O{cnt.get('O')}")
                ok = False
            # products must consume every atom exactly once
            mols = {}
            for i, a in enumerate(sc.atoms):
                mols.setdefault(a.mol, []).append(i)
            prod = {m: v for m, v in mols.items() if m >= 100}
            used = sum(len(v) for v in prod.values())
            if used != len(sc.atoms):
                print(f"[moltest] FAIL {name}: {used} of {len(sc.atoms)} atoms end up in a "
                      f"product - the rest would be left as loose nuclei")
                ok = False
            n_co2 = sum(1 for v in prod.values()
                        if sorted(sc.atoms[i].elem for i in v) == ["C", "O", "O"])
            n_h2o = sum(1 for v in prod.values()
                        if sorted(sc.atoms[i].elem for i in v) == ["H", "H", "O"])
            if (n_co2, n_h2o) != (n_c, n_c + 1):
                print(f"[moltest] FAIL {name}: products are {n_co2} CO2 + {n_h2o} H2O, "
                      f"want {n_c} + {n_c + 1}")
                ok = False
            if verbose:
                print(f"[moltest] {name:14s} C{n_c}H{cnt.get('H')} + {cnt.get('O') // 2} O2 "
                      f"-> {n_co2} CO2 + {n_h2o} H2O   ({len(sc.atoms)} atoms, "
                      f"{len(sc.bonds)} bonds)")

        # ---- 2. covalent geometry --------------------------------------------------
        want = {("O", "H"): GEOM["OH"], ("H", "O"): GEOM["OH"],
                ("O", "O"): None, ("C", "O"): GEOM["CO2"], ("O", "C"): GEOM["CO2"],
                ("C", "C"): GEOM["CC"], ("C", "H"): GEOM["CH"], ("H", "C"): GEOM["CH"]}
        worst = 0.0
        for b in sc.bonds:
            if b.kind != "cov":
                continue
            ei, ej = sc.atoms[b.i].elem, sc.atoms[b.j].elem
            w = want.get((ei, ej))
            if w is None:
                continue
            P = PE if b.phase == 1 else P0
            got = float(np.linalg.norm(P[b.i] - P[b.j]))
            worst = max(worst, abs(got - w))
            if abs(got - w) > tol:
                print(f"[moltest] FAIL {name}: {ei}-{ej} bond is {got:.3f} A, want {w:.3f}")
                ok = False
        # water's angle, measured back
        if name == "water_network":
            g = [i for i, a in enumerate(sc.atoms) if a.elem == "O"][0]
            hs = [b.j if b.i == g else b.i for b in sc.bonds
                  if b.kind == "cov" and g in (b.i, b.j)]
            v1, v2 = P0[hs[0]] - P0[g], P0[hs[1]] - P0[g]
            ang = np.degrees(np.arccos(float(np.dot(_u(v1), _u(v2)))))
            if abs(ang - GEOM["HOH"]) > 0.3:
                print(f"[moltest] FAIL {name}: H-O-H is {ang:.2f} deg, want {GEOM['HOH']}")
                ok = False
            if verbose:
                print(f"[moltest] {name:14s} {len(sc.atoms)} atoms, "
                      f"{sum(1 for b in sc.bonds if b.kind == 'hb')} H-bonds; "
                      f"H-O-H {ang:.2f} deg; worst covalent error {worst:.4f} A")

        # ---- 3. the hydrogen bonds -------------------------------------------------
        for b in sc.bonds:
            if b.kind != "hb":
                continue
            ai, aj = sc.atoms[b.i], sc.atoms[b.j]
            if {ai.elem, aj.elem} != {"H", "O"}:
                print(f"[moltest] FAIL {name}: an H-bond joins {ai.elem}-{aj.elem}")
                ok = False
            if ai.mol == aj.mol:
                print(f"[moltest] FAIL {name}: an H-bond is INTRAmolecular (mol {ai.mol})")
                ok = False
            d = float(np.linalg.norm(PE[b.i] - PE[b.j]))
            if not (1.5 < d < 2.4):
                print(f"[moltest] FAIL {name}: H-bond H...O is {d:.2f} A at the end "
                      "(want 1.5-2.4)")
                ok = False

        # ---- 3b. the thermal motion must BREATHE a bond, not break it ---------------
        # ⚠ A NEW ASSERTION, because "never stop moving" is one knob away from "shake the
        # molecule apart". Vibration that stretches a bond by a tenth of its length is a
        # molecule at room temperature; by half its length it is a different molecule, and no
        # framing or exposure statistic would notice.
        vw, vwb = 0.0, None
        for t in np.linspace(0.0, sc.duration, 121):
            Pj = sc.positions(t, jitter=True)
            Pt = sc.positions(t, jitter=False)
            for b in sc.bonds:
                if b.kind != "cov" or b.strength(t) < 0.99:
                    continue
                d0 = float(np.linalg.norm(Pt[b.i] - Pt[b.j]))
                d1 = float(np.linalg.norm(Pj[b.i] - Pj[b.j]))
                if d0 > 1e-6 and abs(d1 - d0) / d0 > vw:
                    vw, vwb = abs(d1 - d0) / d0, (t, b.i, b.j)
        if vw > 0.12:
            print(f"[moltest] FAIL {name}: thermal motion stretches a bond by {vw:.1%} "
                  f"(t={vwb[0]:.2f}s, atoms {vwb[1]}-{vwb[2]}); that is not a vibration")
            ok = False
        elif verbose:
            print(f"[moltest] {name:14s} worst bond breathing {vw:.1%}; "
                  f"thermal excursion bound {sc.jitter_bound():.3f} A")

        # ---- 4. no spurious necks --------------------------------------------------
        rad = sc.radii()
        bonded = {(min(b.i, b.j), max(b.i, b.j)) for b in sc.bonds}
        bad = None
        margin = JIT_FAST_BOUND
        for t in np.linspace(0.0, sc.duration, 241):
            P = sc.positions(t)
            for i in range(len(sc.atoms)):
                for j in range(i + 1, len(sc.atoms)):
                    if sc.atoms[i].mol == sc.atoms[j].mol or (i, j) in bonded:
                        continue
                    d = float(np.linalg.norm(P[i] - P[j])) - margin
                    if d > 3.4:
                        continue
                    f = _neck(rad[i], rad[j], max(d, 1e-3))
                    if f >= 1.0 and (bad is None or f > bad[0]):
                        bad = (f, t, i, j, d, sc.atoms[i].elem, sc.atoms[j].elem)
        if bad:
            print(f"[moltest] FAIL {name}: a SPURIOUS NECK - {bad[5]}{bad[2]} and "
                  f"{bad[6]}{bad[3]} in different molecules reach field {bad[0]:.2f} "
                  f"at t={bad[1]:.2f}s ({bad[4]:.2f} A apart); a viewer cannot tell that "
                  "from a real bond")
            ok = False
        elif verbose:
            print(f"[moltest] {name:14s} no spurious neck over 241 probes of the "
                  f"trajectory (thermal motion included)")

    print("[moltest] ok" if ok else "[moltest] FAIL")
    return ok
