# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""sync.py - drive the orbital walk and a two-tone vibration from a song. MUSIC_PLAN S5/S11.

A scene opts in with `sync="audio/<slug>.proposal.session.json"`. When no scene does,
`ACTIVE` is None and not one line below runs: every other scene is untouched BY CONSTRUCTION.

THREE THINGS, THREE LAYERS

1. `blend(t)` - TIMING. The walk follows the session's `schedule`: state k is fully arrived AT
   its mark and its morph starts `morph` seconds earlier. Morphs arrive ON the note, because a
   morph that starts on the note lands late and reads as behind the music.
2. `factor(xp, U, t)` - GEOMETRY. A radial multiplier `1 + A1 e1 V1 + A2 e2 V2` on the surface,
   one term per synth tone (Clancy: "two interacting disturbances"). They interact by plain
   superposition on the same r - reinforcing where both push out, cancelling where they
   disagree. Applied to the dot mesh AND to the void carve, because in `qm_negative` the
   visible shape is mostly the carved HOLE (`void_mesh` 0.15); displacing only the dots would
   move 15% of the picture.
3. `walk_u(t)` - the chromatic split and the void lattice crossfade are bound to the STATE
   WALK in canon (`flow.split_at`). The schedule's states are canon's first 13 in canon order,
   so the walk position maps onto canon's clock exactly: u = (i + w) / 13.

⚠ ONE CLOCK. Everything in the session is SONG time. `t` everywhere in the engine is VIDEO
time. The conversion `song = t + trim.start` happens in `_Sync.__init__` and nowhere else.

⚠ THE ENVELOPES ARE PRECOMPUTED, ONE VALUE PER VIDEO FRAME, and normalised ONCE over the
clip (floor p20, scale p97). That keeps the engine a pure function of t (a still at any
instant is exact) and it is OUTPUT_RULES' stable-reference rule: a rolling normalisation
would make a quiet bar move as hard as a loud one.

⚠ THE VIBRATION IS NOT A WOBBLE OF THE WHOLE OBJECT. V1 and V2 are standing waves over the
sphere of directions, so lobes move against each other ("movement decomposes"). And they
modulate r on the same parameterisation, so a nodal direction (r = 0) stays at zero: a d_xz
does not stop having four lobes at any amplitude.

⚠ AN OUTWARD DISPLACEMENT GROWS THE SILHOUETTE, so `max_inflation()` is folded into
`scene.fit_scales`' hard bound. The electron pull never needed that because it only shrinks.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

#: the live sync, or None. Set by `scene.configure`.
ACTIVE = None

#: Look knobs - fractions of the surface radius. Worst case inflation is 1 + A1 + A2.
#: ⚠ 0.07 / 0.04 WAS INVISIBLE. Clancy, on the first 720: *"distortion needs to be much much
#: more significant, this is hardly noticable."* The amplitude was not the only reason: the
#: tone envelopes sit at a MEDIAN of ~0.15 (they are peaky), so a typical frame moved ~1.5%
#: of the radius. Now 4x the amplitude AND a gamma on the envelope (`VIB_GAMMA`) so mid-level
#: notes register. Scene-overridable via `vib=(A1, A2)` / `vib_gamma=`.
A1, A2 = 0.28, 0.18
VIB_GAMMA = 0.6
#: How long a note's phase step takes to glide in. A step, not a jump: a pattern that
#: teleports on every note is a "sudden motion" OUTPUT_RULES rejects.
STEP_TAU = 0.14
STEP1, STEP2 = 0.90, 1.30          # rad per note
N_CANON = 13                        # segments in canon's 14-state walk


#: THE REACTIVE PIXELS (Clancy, 2026-09-16): *"some of the square units being reactive to the
#: music as well, primarily the 'pixels' around the edges of the orbital and around its glow.
#: this should change the pixels to colors that are used in the orbital. make sure this is
#: pretty distinct"*. Field dots in a band from just inside the carve rim out through the glow
#: (`REACT_BAND`, in units of R) light up: tone 1 in the MESH ramp's colours, tone 2 in the
#: NUCLEUS ramp's. The lit fraction follows each tone's envelope, and the chosen subset is
#: re-drawn on each of that tone's notes. Boosted past `bloom_threshold` so they glow.
#: ⚠ LOCAL BY CONSTRUCTION - only the band can light, so this is never a full-frame flash.
REACT_BAND = (-0.03, 0.01, 0.22, 0.55)     # rise from, rise to, fall from, fall to
REACT_DENSITY = 0.20                        # fraction of band dots lit at full drive
REACT_BOOST = 3.0
#: ⚠ The first cut (band to 1.05 R, 0.30, x4) BURIED the orbital in a glowing cloud - the lobes
#: and the hole were gone. Tuned on stills: the band now hugs the rim + near glow, and the
#: frame lifts by at most ~40% luma, all of it in the halo.
REACT_SOFT = 0.04
#: Scene switch (`react=`). Clancy removed the effect after the first 2K final (2026-09-16),
#: so `music_4morant` sets it False; the code stays for a scene that wants it.
REACT_ON = True

#: THE BLOOM LAYERS - echo shells in the void field (`void._bloom`). None = off (every scene
#: but the bloom variant). Per layer: `scale` (x R), `amp` (x A1/A2), `rot` (azimuth, rad),
#: `p1`/`p2` (phase offsets), `lean` (fraction toward the NEXT orbital), and the look:
#: `width` (dark outline, units of the layer's R), `dip`, `boost`, `fade`, `heat`.
LAYERS = None
#:
#: ⚠ v2 (Clancy: *"there are definitely some portions where there are drastic 'cuts' between
#: blooms ... reduce the number of blooms to 2. make the color of the second bloom between the
#: blue of the glow and the purple of the background"*). Two layers, and `lut="violet"` routes
#: the second through its own ramp (`render.violet_ramp`). `lean` is now in STATES along the
#: walk (continuous), not a fraction of the current morph - see `layer_walk`.
BLOOM_LAYERS = [
    dict(scale=1.70, amp=2.2, rot=0.55, p1=1.7, p2=-2.3, lean=0.30,
         width=0.030, dip=0.70, boost=0.72, fade=0.22, heat=0.95,
         tone="tone1", kick=0.140, flare=0.85),
    dict(scale=2.60, amp=3.2, rot=1.60, p1=3.7, p2=-4.6, lean=0.60,
         width=0.026, dip=0.50, boost=0.62, fade=0.20, heat=1.00, lut="violet",
         tone="tone2", kick=0.170, flare=0.85),
]
#: The layers' OWN, slower drive. The v1 cuts had three causes and two were here: a 10 ms
#: attack is a one-frame step, and at 3-4x the amplitude a one-frame step in the envelope is a
#: visible jump of the outer shells; and a note's pattern shift glided in over only 0.14 s,
#: which at 4x reads as a snap. The real orbital keeps the fast drive (it was approved).
LAYER_ATTACK, LAYER_RELEASE, LAYER_STEP_TAU = 0.080, 0.450, 0.40
#: ⚠ v3 - THE SLOW DRIVE ABOVE KILLED THE SYNC, and it was the one v2 fix that was wrong.
#: Clancy: *"it doesnt look like the music is particularly synced anymore.. the notes are not as
#: noticable"*. Measured (scratchpad `syncscore.py`: frame motion in the 7 frames after each
#: note / motion before it): the non-bloom final x1.31, bloom v2 x1.05 - no response at all,
#: and its median frame step was 9x the final's, so notes drowned in constant motion. The cuts
#: were the lean snap and the simultaneous morphs; those fixes stay. The layers now take a
#: FAST-ATTACK, FAST-DECAY drive with no pulse breathing, so their motion bunches up right
#: after each note, plus a KICK: on its own tone's notes a shell swells outward and its rim
#: flares. Knobs per layer: `tone`, `kick` (fraction of the shell's scale), `flare`.
LAYER_ATTACK, LAYER_RELEASE, LAYER_STEP_TAU = 0.030, 0.200, 0.22
KICK_ATTACK, KICK_RELEASE = 0.035, 0.220
#: The pulse-grid breathing, as (floor, swing). v1/v2 used (0.55, 0.45) on the orbital, which
#: damps an OFF-beat note to 10% at the trough - and this synth plays off the beat.
BREATH = (0.85, 0.15)
PRE_ROLL = 1.0

#: THE LIVE CORE (Clancy, 2026-09-16): *"add some more dynamic elements to the 'core' that has
#: that cloud of dots rotating. lets try having the dot cloud illuminate with the music sync.
#: Allow the core to have more than one color."* None = the canon nucleus (every other scene).
#:
#: The core's 60k gaussian dots are split into THREE COLOUR SHELLS by radius (in units of the
#: gaussian sigma), dithered so the boundaries blend: `pink` (the nucleus ramp) inside,
#: `violet` (the outer bloom's ramp) between, `blue` (the mesh ramp) outside. Then:
#: * SPIN - the cloud really rotates about a tilted axis (it only ever turned with the camera),
#:   inner shells faster than outer, so it winds into a swirl. Angle = the time-integral of a
#:   base rate plus the note kicks, so a note spins it up and it can never jump.
#: * NOTE RINGS - each note sends a bright ring out through the cloud; tone 1 lights the pink
#:   and blue dots, tone 2 the violet ones, so the two synths read as two colours.
#: * SPARKLE - a random few percent of dots flare on each note, re-drawn per note.
#: * SWELL - the cloud's radius and brightness breathe with the envelopes.
#: ⚠ The nucleus is the one element the tone curve clips first (development notes: a dense blob
#: turns into a white star), so every boost here is measured against blown pixels.
CORE = None
#:
#: ⚠ v2 of the core, after the first crops. (1) A SPHERICAL cloud looks identical at every
#: rotation - the spin was invisible - so the groups got STRUCTURE the spin can show: a compact
#: magenta HEART, a tilted violet RING that precesses, and a blue DISK with two spiral ARMS.
#: (2) Three colours overlapping in one ball summed to a single lavender, and the graded
#: nucleus ramp is near-white (core_sat 0.44) - so the groups are separated in space and the
#: heart uses the CANON magenta ramp (`lut="magenta"`).
CORE_DEFAULT = dict(
    sigma=1.35,                        # x render.NUC_SIGMA - a bigger core to carry detail
    shells=(1.15, 1.85, 1.55, 2.55),   # heart->ring (from, to), ring->disk (from, to)
    keys=dict(pink=0.66, violet=0.76, blue=0.74),
    gains=dict(pink=0.34, violet=3.60, blue=4.80),
    form=dict(pink=(0.80, 1.0), violet=(1.80, 0.14), blue=(2.40, 0.10)),   # (scale, flatten)
    ring_tilt=(1.0, 0.35, -0.55), arms=2, arm_wind=2.4, arm_sharp=2.0,
    spin=0.55, spin_kick=5.0, spin_axis=(0.30, 1.0, 0.20), shear=0.40,
    ring_tau=0.38, ring_reach=3.2, ring_width=0.32, ring_gain=2.2,
    sparkle=0.05, sparkle_gain=2.6,
    swell=0.14, glow=0.45,
)
#: The layers' walk is the orbital's walk SMOOTHED IN TIME (gaussian, seconds). Without it
#: every shell morphs inside the orbital's own 0.53 s sub-morph, all at once, and those
#: windows were the largest frame steps left after the lean fix. Symmetric, so the shells
#: start easing slightly BEFORE the orbital does - they already lean ahead of it.
LAYER_WALK_SIGMA = 0.35


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _follow(r, fps, attack=0.010, release=0.160):
    ka = 1.0 - math.exp(-1.0 / max(attack * fps, 1e-6))
    kr = 1.0 - math.exp(-1.0 / max(release * fps, 1e-6))
    out = np.empty_like(r)
    v = 0.0
    for i, x in enumerate(r):
        v += (x - v) * (ka if x > v else kr)
        out[i] = v
    return out


class _Sync:
    def __init__(self, path, root, fps=60):
        from . import audio as A
        from . import harmonics as H
        session_path = os.path.join(root, path)
        with open(session_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["wav"] = os.path.join(os.path.dirname(session_path), doc["wav"])
        self.doc = doc
        tr = doc["trim"]
        self.t0 = float(tr["start"])                 # ⚠ THE one song->video conversion
        self.duration = float(tr["end"]) - self.t0
        self.fps = fps
        sch = doc["schedule"]
        # video-time arrivals and morph lengths
        self.T = [float(s["t"]) - self.t0 for s in sch]
        self.M = [float(s["morph"]) for s in sch]
        self.states = [(int(s["lm"][0]), int(s["lm"][1]), s["state"]) for s in sch]
        self.grade = doc.get("grade")
        pulse = doc.get("pulse") or {}
        self.P = float(pulse.get("period", 0.5))
        self.ph = float(pulse.get("phase", 0.0)) - self.t0      # grid phase in video time

        # ---- the two tones' drivers, from the ONE shared WAV ---------------------------
        # ⚠ PRE-ROLLED. The followers start from zero, so reading from the clip's first
        # sample made every envelope RAMP UP over the opening frames - a pop at t = 0 that the
        # smoothness probe measured as the largest step of the whole clip. The audio is read
        # PRE seconds early, the followers run through it, and those frames are dropped.
        pre = min(PRE_ROLL, self.t0)
        npre = int(round(pre * fps))
        x = A.read_mono(doc["wav"], self.t0 - pre, self.t0 + self.duration)
        win, hop = 4096, 128
        n = (len(x) - win) // hop
        fr = np.lib.stride_tricks.as_strided(
            x, (n, win), (x.strides[0] * hop, x.strides[0])) * np.hanning(win)
        mag = np.abs(np.fft.rfft(fr, axis=1))
        f = np.fft.rfftfreq(win, 1.0 / A.SR)
        tt = np.arange(n) * hop / A.SR + win / 2.0 / A.SR - pre
        nfr = int(round(self.duration * fps))
        idx = np.clip(((tt + pre) * fps).astype(int), 0, nfr + npre - 1)
        cut = slice(npre, npre + nfr)
        tones = doc["tones"]
        self.env, self.phase = {}, {}
        for key, step in (("tone1", STEP1), ("tone2", STEP2)):
            m = np.zeros_like(f, bool)
            for q in tones[key]:
                m |= (f > q * 0.97) & (f < q * 1.03)
            e = mag[:, m].sum(1)
            per = np.zeros(nfr + npre)
            np.maximum.at(per, idx, e)
            raw = per
            per = _follow(raw, fps)[cut]
            lo, hi = np.percentile(per, 20), np.percentile(per, 97)
            self.env[key] = (np.clip((per - lo) / max(hi - lo, 1e-9), 0, 1)
                             ** VIB_GAMMA).astype(np.float32)
            sl = _follow(raw, fps, LAYER_ATTACK, LAYER_RELEASE)[cut]
            lo, hi = np.percentile(sl, 20), np.percentile(sl, 97)
            self.env[key + "_slow"] = (np.clip((sl - lo) / max(hi - lo, 1e-9), 0, 1)
                                       ** VIB_GAMMA).astype(np.float32)
            kk = _follow(raw, fps, KICK_ATTACK, KICK_RELEASE)[cut]
            lo, hi = np.percentile(kk, 30), np.percentile(kk, 98)
            self.env[key + "_kick"] = np.clip((kk - lo) / max(hi - lo, 1e-9),
                                              0, 1).astype(np.float32)
            # note onsets of THIS tone -> a gliding phase
            lg = np.log1p(mag[:, m] * 100.0)
            d = np.maximum(0.0, np.diff(lg, axis=0)).sum(1)
            picks = A.pick(tt[1:].astype(np.float32), d.astype(np.float32), min_sep=0.2)
            ons_all = [a for a, s in picks if s > 0.25]
            ons = [a for a in ons_all if a >= 0.0]
            self.ons_all = getattr(self, "ons_all", {})
            self.ons_all[key] = ons_all
            ph = np.zeros(nfr, np.float32)
            phs = np.zeros(nfr, np.float32)
            ft = np.arange(nfr) / fps
            # notes in the pre-roll still count toward the phase, so it does not start at a
            # value it could never have reached a frame later
            for o in ons_all:
                ph += step * np.clip((ft - o) / STEP_TAU, 0, 1) ** 2 * (
                    3 - 2 * np.clip((ft - o) / STEP_TAU, 0, 1))
                u = np.clip((ft - o) / LAYER_STEP_TAU, 0, 1)
                phs += step * u * u * (3 - 2 * u)
            self.phase[key] = ph
            self.phase[key + "_slow"] = phs
            self.env[key + "_onsets"] = ons
        self.zero = False                     # the regression gate forces the drivers off
        H.STATES = list(self.states)
        from scipy.ndimage import gaussian_filter1d
        walk = np.array([sum(v * c for v, c in zip(self.blend(k / fps)[:2],
                                                   (1.0 - self.blend(k / fps)[2],
                                                    self.blend(k / fps)[2])))
                         for k in range(nfr)], np.float64)
        self.walk_slow = gaussian_filter1d(walk, LAYER_WALK_SIGMA * fps, mode="nearest")

    # ------------------------------------------------------------------ timing
    def blend(self, t):
        T, M, n = self.T, self.M, len(self.T)
        if n == 1:
            return 0, 0, 0.0
        for k in range(n - 1, 0, -1):
            if t >= T[k] - M[k]:
                if t >= T[k] or M[k] <= 0:
                    return k, min(k + 1, n - 1), 0.0
                return k - 1, k, float(_smooth((t - (T[k] - M[k])) / M[k]))
        return 0, 1, 0.0

    def walk_u(self, t):
        i, j, w = self.blend(t)
        return min(1.0, (i + (j - i) * w) / N_CANON)

    # ------------------------------------------------------------------ geometry
    def max_inflation(self):
        return 1.0 + A1 + A2

    # ------------------------------------------------------------------ the live core
    def _core_prep(self):
        """Per-frame arrays for the core: spin angle, and time since each tone's last note."""
        fps, nfr = self.fps, len(self.env["tone1"])
        c = CORE
        kick = np.maximum(self.env["tone1_kick"], self.env["tone2_kick"]).astype(np.float64)
        rate = c["spin"] + c["spin_kick"] * kick
        self.core_angle = np.cumsum(rate) / fps
        ft = np.arange(nfr) / fps
        self.since = {}
        for key in ("tone1", "tone2"):
            ons = np.asarray(self.ons_all[key], np.float64)
            idx = np.searchsorted(ons, ft, side="right") - 1
            last = np.where(idx >= 0, ons[np.clip(idx, 0, None)], -1e9)
            self.since[key] = (ft - last).astype(np.float32)
        self._core_ready = True

    def core(self, xp, U, t, base_sigma, base_gain):
        """[(lut_name, P, key, gain)] - the live core. `U` is the (N, 3) unit-gaussian cloud."""
        if not getattr(self, "_core_ready", False):
            self._core_prep()
        c = CORE
        f32 = np.float32
        k = min(max(int(round(t * self.fps)), 0), len(self.env["tone1"]) - 1)
        e1 = 0.0 if self.zero else float(self.env["tone1"][k])
        e2 = 0.0 if self.zero else float(self.env["tone2"][k])
        kk = 0.0 if self.zero else max(float(self.env["tone1_kick"][k]),
                                        float(self.env["tone2_kick"][k]))
        if not hasattr(self, "_core_static"):
            rho = xp.sqrt((U * U).sum(1))
            v = xp.sin(U[:, 0] * f32(12.9898) + U[:, 1] * f32(78.233)
                       + U[:, 2] * f32(37.719)) * f32(43758.5453)
            h = v - xp.floor(v)
            a0, a1, b0, b1 = c["shells"]
            p_in = xp.clip((a1 - rho) / f32(a1 - a0), 0, 1)
            p_out = xp.clip((rho - b0) / f32(b1 - b0), 0, 1)
            pink = h < p_in
            blue = (~pink) & (h > f32(1.0) - p_out * (f32(1.0) - p_in))
            violet = ~(pink | blue)
            ax = np.asarray(c["spin_axis"], np.float64)
            ax = ax / np.linalg.norm(ax)
            tl = np.asarray(c["ring_tilt"], np.float64)
            tl = tl - ax * (tl @ ax) * 0.35          # mostly off the spin axis, so it precesses
            tl = tl / np.linalg.norm(tl)
            axd = xp.asarray(ax.astype(np.float32))
            tld = xp.asarray(tl.astype(np.float32))
            # BODY-FRAME SHAPES, static: each group is scaled and flattened along its own axis
            U0 = U * f32(1.0)
            forms = c["form"]

            def flat(V, n, s, k):
                return (V - (f32(1.0) - f32(k)) * (V @ n)[:, None] * n[None, :]) * f32(s)
            mk = dict(pink=pink, violet=violet, blue=blue)
            Ub = xp.where(mk["pink"][:, None], U0 * f32(forms["pink"][0]), U0)
            Ub = xp.where(mk["violet"][:, None],
                          flat(U0, tld, forms["violet"][0], forms["violet"][1]), Ub)
            Ub = xp.where(mk["blue"][:, None],
                          flat(U0, axd, forms["blue"][0], forms["blue"][1]), Ub)
            # spiral arms on the disk: azimuth in the disk plane
            bx = np.cross(ax, [0.0, 0.0, 1.0])
            bx = bx / np.linalg.norm(bx)
            by = np.cross(ax, bx)
            e1d, e2d = xp.asarray(bx.astype(np.float32)), xp.asarray(by.astype(np.float32))
            phi = xp.arctan2(U0 @ e2d, U0 @ e1d)
            arm = f32(0.5) + f32(0.5) * xp.cos(f32(c["arms"]) * phi - f32(c["arm_wind"]) * rho)
            arm = arm ** f32(c["arm_sharp"])
            self._core_static = dict(rho=rho, h=h, masks=mk, axis=axd, Ub=Ub,
                                     arm=f32(0.25) + f32(0.75) * arm)
        st = self._core_static
        rho, h = st["rho"], st["h"]
        U = st["Ub"]

        # SPIN - differential: inner shells turn faster, so the cloud winds up
        ang = f32(self.core_angle[k]) * (f32(1.0) + f32(c["shear"])
                                         * xp.clip(f32(1.8) - rho, f32(-1.0), f32(1.0)))
        a = st["axis"]
        ca, sa = xp.cos(ang)[:, None], xp.sin(ang)[:, None]
        cross = xp.stack([a[1] * U[:, 2] - a[2] * U[:, 1],
                          a[2] * U[:, 0] - a[0] * U[:, 2],
                          a[0] * U[:, 1] - a[1] * U[:, 0]], 1)
        dot = (U @ a)[:, None]
        Ur = U * ca + cross * sa + a[None, :] * dot * (f32(1.0) - ca)

        swell = 1.0 + c["swell"] * max(e1, e2)
        P = Ur * f32(base_sigma * c["sigma"] * swell)
        glow = f32(1.0 + c["glow"] * 0.5 * (e1 + e2))

        # NOTE RINGS, one per tone
        def ring(key):
            if self.zero:
                return 0.0
            dt = float(self.since[key][k])
            if dt < 0 or dt > c["ring_tau"]:
                return 0.0
            x = dt / c["ring_tau"]
            r0 = f32(0.2 + c["ring_reach"] * x)
            return (f32((1.0 - x) ** 2 * c["ring_gain"])
                    * xp.exp(-((rho - r0) / f32(c["ring_width"])) ** 2))
        r1, r2 = ring("tone1"), ring("tone2")

        # SPARKLE - a per-note subset, decaying with the kick
        n1 = self.note_index("tone1", t) + 7 * self.note_index("tone2", t)
        v = xp.sin(h * f32(91.7) + f32(n1 * 1.618)) * f32(43758.5453)
        sp = (v - xp.floor(v)) < f32(c["sparkle"])
        spark = sp.astype(np.float32) * f32(c["sparkle_gain"] * kk)

        out = []
        for name, lut in (("pink", "magenta"), ("violet", "violet"), ("blue", "mesh")):
            m = st["masks"][name].astype(np.float32)
            if name == "blue":
                m = m * st["arm"]
            rr = r2 if name == "violet" else r1
            g = m * f32(base_gain * c["gains"][name]) * glow * (f32(1.0) + rr + spark)
            key = xp.full(U.shape[0], f32(c["keys"][name]))
            out.append((lut, P, key, g))
        return out

    def kick(self, t, tone):
        if self.zero or not tone:
            return 0.0
        k = min(max(int(round(t * self.fps)), 0), len(self.env["tone1"]) - 1)
        return float(self.env[tone + "_kick"][k])

    def layer_walk(self, t, lean):
        """(a, b, f): the layer's position on the walk, `lean` STATES ahead of the orbital.

        ⚠ THIS WAS THE CUT. v1 leaned by `min(1, w + lean)` - a fraction of the CURRENT morph.
        At the frame an orbital finishes arriving, `blend` hands over from (k-1, k, w=1) to
        (k, k+1, w=0), so the lean jumped from "all of k" to "lean of the way to k+1": every
        echo shell snapped on every arrival. Measured: the six worst frame steps of the clip
        sat exactly on the schedule's arrival times. The walk coordinate `i + w` is continuous
        through that hand-over, so offsetting IT is continuous too.
        """
        k = min(max(t * self.fps, 0.0), len(self.walk_slow) - 1.0)
        k0 = int(k)
        k1 = min(k0 + 1, len(self.walk_slow) - 1)
        base = self.walk_slow[k0] + (self.walk_slow[k1] - self.walk_slow[k0]) * (k - k0)
        u = min(float(len(self.T) - 1), float(base) + float(lean))
        a = int(np.floor(u))
        b = min(a + 1, len(self.T) - 1)
        return a, b, u - a

    def drivers(self, t, slow=False):
        if self.zero:
            return 0.0, 0.0, 0.0, 0.0
        k = min(max(int(round(t * self.fps)), 0), len(self.env["tone1"]) - 1)
        if slow:
            # no pulse breathing on the shells: their motion should BE the notes
            return (float(self.env["tone1_slow"][k]), float(self.env["tone2_slow"][k]),
                    float(self.phase["tone1_slow"][k]), float(self.phase["tone2_slow"][k]))
        # the sway breathes WITH the pulse grid: peak on every pulse (tone 1), twice per
        # pulse (tone 2). Continuous in t - never reset on a note, so it cannot step.
        g = 2.0 * math.pi * (t - self.ph) / self.P
        b0, b1 = BREATH
        e1 = float(self.env["tone1"][k]) * (b0 + b1 * math.cos(g))
        e2 = float(self.env["tone2"][k]) * (b0 + b1 * math.cos(2.0 * g))
        return e1, e2, float(self.phase["tone1"][k]), float(self.phase["tone2"][k])

    def note_index(self, key, t):
        ons = self.env[key + "_onsets"]
        return int(np.searchsorted(np.asarray(ons), t, side="right"))

    def react(self, xp, L, d, t):
        """(mesh_key, mesh_mult, core_key, core_gain) for the void field's dots, or None.

        `mesh_mult` multiplies the field's own gain (1 = untouched); lit tone-1 dots get a key
        up the mesh ramp and REACT_BOOST. Lit tone-2 dots are removed from the mesh pass
        (mult 0) and returned as `core_gain` for a second deposit through the nucleus LUT.
        """
        if self.zero or d is None or not REACT_ON:
            return None
        k = min(max(int(round(t * self.fps)), 0), len(self.env["tone1"]) - 1)
        e1, e2 = float(self.env["tone1"][k]), float(self.env["tone2"][k])
        if e1 <= 0.0 and e2 <= 0.0:
            return None
        f64 = np.float64
        a, b, c, dd = REACT_BAND
        x = xp.clip((d - a) / (b - a), 0, 1)
        y = xp.clip((d - c) / (dd - c), 0, 1)
        band = (x * x * (3 - 2 * x)) * (1 - y * y * (3 - 2 * y))
        # ⚠ HASHED ON THE LATTICE COORDINATE, not the index and not the world position: the
        # index order changes when the bars/rings crossfade drops a family, and the world
        # position moves with the camera every frame - either would re-roll every frame.
        Ld = L.astype(f64)
        base = Ld[:, 0] * 127.1 + Ld[:, 1] * 311.7 + Ld[:, 2] * 74.7

        def h(seed):
            v = xp.sin(base + f64(seed) * 17.137) * 43758.5453
            return (v - xp.floor(v)).astype(np.float32)
        n1, n2 = self.note_index("tone1", t), self.note_index("tone2", t)
        s1, s2 = h(n1 * 1.0 + 0.5), h(n2 * 1.0 + 911.5)
        thr1 = band * np.float32(REACT_DENSITY * e1)
        thr2 = band * np.float32(REACT_DENSITY * e2)
        lit1 = xp.clip((thr1 - s1) / np.float32(REACT_SOFT), 0, 1)
        lit2 = xp.clip((thr2 - s2) / np.float32(REACT_SOFT), 0, 1)
        lit2 = lit2 * (1 - lit1)                       # one colour per dot
        kh1 = h(n1 + 3.25)
        mesh_key = np.float32(0.40) + np.float32(0.35) * kh1
        core_key = np.float32(0.55) + np.float32(0.40) * h(n2 + 7.75)
        mesh_mult = (1 - lit1 - lit2) + lit1 * np.float32(REACT_BOOST)
        core_gain = lit2 * np.float32(REACT_BOOST)
        return mesh_key, lit1, mesh_mult, core_key, core_gain

    def factor_layer(self, xp, ux, uy, uz, t, L):
        """`factor` for an echo layer: stronger, rotated and re-phased, clamped positive."""
        e1, e2, p1, p2 = self.drivers(t, slow=True)
        if e1 == 0.0 and e2 == 0.0:
            return None
        f32 = np.float32
        k = float(L["amp"])
        psi = xp.arctan2(uz, ux) + f32(L["rot"])
        yy = xp.clip(uy, f32(-1), f32(1))
        st = xp.sqrt(xp.maximum(f32(0), f32(1) - yy * yy))
        q1, q2 = p1 + float(L["p1"]), p2 + float(L["p2"])
        v1 = (f32(0.75) * st * xp.cos(f32(2) * psi + f32(q1))
              + f32(0.25) * yy * f32(math.sin(q1)))
        v2 = xp.cos(f32(5) * psi - f32(q2)) * xp.sin(f32(3) * xp.arccos(yy) + f32(0.5 * q2))
        f = f32(1) + f32(k * A1 * e1) * v1 + f32(k * A2 * e2) * v2
        # ⚠ CLAMPED: at 4x the amplitude the raw factor can go negative, which would fold the
        # surface through the origin and draw a shell on the wrong side.
        return xp.maximum(f, f32(0.20))

    def factor(self, xp, ux, uy, uz, t):
        """(N,) radial multiplier for unit directions (ux, uy, uz). +Y is the quantum z."""
        e1, e2, p1, p2 = self.drivers(t)
        if e1 == 0.0 and e2 == 0.0:
            return None
        f32 = np.float32
        psi = xp.arctan2(uz, ux)
        yy = xp.clip(uy, f32(-1), f32(1))
        st = xp.sqrt(xp.maximum(f32(0), f32(1) - yy * yy))
        v1 = f32(0.75) * st * xp.cos(f32(2) * psi + f32(p1)) + f32(0.25) * yy * f32(math.sin(p1))
        v2 = xp.cos(f32(5) * psi - f32(p2)) * xp.sin(f32(3) * xp.arccos(yy) + f32(0.5 * p2))
        return f32(1) + f32(A1 * e1) * v1 + f32(A2 * e2) * v2


def activate(path, root, vib=None, vib_gamma=None, react=True, bloom=False, core=False):
    global ACTIVE, A1, A2, VIB_GAMMA, REACT_ON, LAYERS, CORE
    CORE = ({**CORE_DEFAULT, **core} if isinstance(core, dict)
            else (dict(CORE_DEFAULT) if core else None)) if path else None
    REACT_ON = bool(react)
    LAYERS = (BLOOM_LAYERS if bloom is True else (bloom or None)) if path else None
    if vib is not None:
        A1, A2 = float(vib[0]), float(vib[1])
    if vib_gamma is not None:
        VIB_GAMMA = float(vib_gamma)
    ACTIVE = _Sync(path, root) if path else None
    return ACTIVE
