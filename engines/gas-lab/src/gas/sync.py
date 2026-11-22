# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""sync.py - the beat-locked drive, and which combustion carries which word's flash.

THE LOCK. A radial fires every 144 deg of crank. If the crank turns at a constant rate of
144 deg per grid unit, then every combustion lands exactly on a grid line of the song - and
the grid, measured in audio.py, is where the vocal hits actually sit (a triplet of the beat,
0.221 s). So the engine runs at ONE STEADY SPEED and is nonetheless sample-locked to the
music. Nothing is keyframed and no frame is nudged: the only thing chosen is how fast the
crank turns, which is a thing an engine has.

    deg(t) = 124 + (t - t_fire0) * 144 / period

124 deg is where the first cylinder in the firing order reaches its spark angle. The whole
mapping from crank angle to cylinder is in `_fire_angles`, derived from the same clock phases
catalog.radial builds, so if the firing order changes this follows it.

WHICH WORD. The onsets fall on two of the three sub-positions of a beat (measured: 10 on
sub 0, 9 on sub 2, none on sub 1). Those two series are the two words. Which of them is
"flashing" and which is "lights" is the ONE thing measurement cannot settle - it needs an ear.
`scratch\\mark_words.py` serves a page that plays the extracted WAV and writes the answer to
`audio\\<slug>.labels.json`; until it does, DEFAULT_LABELS applies and the note below stands.

    DEFAULT, and the reasoning: the measured pairs run sub 2 FIRST and sub 0 one grid unit
    later (28.325 then 28.546, 29.651 then 29.872, ...). "Flashing lights" is a two-syllable
    pickup into a stressed one-syllable word, so the EARLIER hit should be "flashing" and the
    on-beat one "lights" - making sub 2 the 'a' (strobe / core) flash and sub 0 the 'b'
    (wash / metal / rays) one. That is an inference from the ORDER of the hits, not a
    measurement of the words, and it is the one thing in this build that wants an ear. If it
    is backwards it is one line in the labels file and a re-render of the DRAW PASS only -
    the physics, the lock and the gate are untouched.
"""
from __future__ import annotations

import math

import numpy as np

from . import audio as A

# sub-position on the grid -> which word's flash. Overridden by audio\<slug>.labels.json.
DEFAULT_LABELS = {"2": "a", "0": "b"}


def _fire_angles(n=5):
    """Crank angle (deg of the 720 cycle) at which each cylinder sparks, as catalog.radial
    phases its clocks: clock j carries phase -360j/n (+360 on even j), and the igniter fires
    at SPARK. So cylinder j fires when deg(t) + phase_j == SPARK."""
    from . import catalog as K
    out = {}
    for j in range(n):
        ph = -360.0 * j / n + (360.0 if j % 2 == 0 else 0.0)
        out[j] = (K.SPARK - ph) % 720.0
    return out                      # {cyl: angle}; the angles are 144 deg apart


class BeatDrive:
    """A constant-rate drive whose combustions land on the song's grid.

    Duck-types engine.Drive (rev0 / mult / configure / s / revs / theta / deg / omega) so
    every clock, cam lobe, valve and the drawn crank read it without knowing it is different.
    """

    def __init__(self, period, t_fire0, start_angle=124.0, rev0=None):
        self.period = float(period)             # seconds between combustions
        self.rate = 144.0 / self.period         # crank deg per second
        self.rev_seconds = 360.0 / self.rate    # seconds per crank revolution
        self.rev0 = float(rev0 if rev0 is not None else self.rev_seconds)
        self.t0 = float(t_fire0)                # absolute time of the first combustion
        self.a0 = float(start_angle)
        self.mult = self.rev0 / self.rev_seconds
        self.warm = 0.0
        self.ramp = 0.0
        self.locked = True      # render.report: there is no ramp, so do not scale the ratio

    # engine.Drive's surface -------------------------------------------------
    def configure(self, warm=0.0, ramp=0.0, mult=1.0):
        """render.schedule() calls this. The lock is absolute, so only `warm` is kept - the
        ramp and multiplier belong to the eight-engine batch, not to a synced clip."""
        self.warm = float(warm)

    def s(self, t):
        return self.mult

    def deg(self, t):
        return self.a0 + (t - self.t0) * self.rate

    def revs(self, t):
        return self.deg(t) / 360.0

    def theta(self, t):
        return math.radians(self.deg(t))

    def omega(self, t):
        return math.radians(self.rate)



class SplineDrive:
    """A crank that turns through 144 deg between one word and the next.

    `BeatDrive` above holds ONE speed, which lands a bang on every grid line - including the
    two thirds of them that carry no word. Clancy, 2026-09-20: *"flashes are way too frequent.
    each word should have its own stroke and flash. The speed/rate of the previous video was
    nearly perfect! this is way too fast."* So the knots become the ONSETS THEMSELVES: fire i
    at t_i, at crank angle 124 + 144 i. Nineteen combustions instead of forty-five, averaging
    0.53 s apart against the previous radial's 0.665 s.

    theta(t) is a PCHIP monotone cubic through those knots, so it passes through every one
    exactly, never runs backwards, and does not overshoot between them the way a natural
    spline would. `omega` is its analytic derivative - which matters, because the moving-wall
    reflection (v' = -v + 2u) reads the piston velocity every substep, and a finite difference
    there puts a step in the gas.

    HONESTY: the gaps run 1 to 5 grid units, so the crank surges roughly 3x between the
    tightest pair of words and the widest gap. The GAS is exact for whatever theta(t) it is
    given, and every valve, cam and drawn part reads the same curve - but that angular
    acceleration is not a torque an iron crank could deliver. It is a music video's engine,
    and the project doc says so.
    """

    def __init__(self, times, warm, start_angle=124.0, rev0=5.0, pad=3):
        from scipy.interpolate import PchipInterpolator
        t = [float(v) for v in times]
        a = [start_angle + 144.0 * i for i in range(len(t))]
        # extend both ends at the local rate so the warm-up and the tail are well defined
        r0 = (a[1] - a[0]) / (t[1] - t[0])
        r1 = (a[-1] - a[-2]) / (t[-1] - t[-2])
        span0 = max(warm + 1.0, 2.0)
        pre_t = [t[0] - span0 * (k + 1) / pad for k in range(pad)][::-1]
        pre_a = [a[0] - r0 * (t[0] - v) for v in pre_t]
        post_t = [t[-1] + 2.0 * (k + 1) / pad for k in range(pad)]
        post_a = [a[-1] + r1 * (v - t[-1]) for v in post_t]
        self._f = PchipInterpolator(np.asarray(pre_t + t + post_t),
                                    np.asarray(pre_a + a + post_a), extrapolate=True)
        self._d = self._f.derivative()
        self.times = t
        self.rev0 = float(rev0)
        gaps = [t[i + 1] - t[i] for i in range(len(t) - 1)]
        self.period = sum(gaps) / len(gaps)              # mean seconds per combustion
        self.rev_seconds = 2.5 * self.period             # 5 fires per 2 crank revolutions
        self.mult = self.rev0 / self.rev_seconds
        self.warm, self.ramp = 0.0, 0.0
        self.locked = True
        self.span = (pre_t[0], post_t[-1])

    def configure(self, warm=0.0, ramp=0.0, mult=1.0):
        self.warm = float(warm)

    def deg(self, t):
        return float(self._f(t))

    def revs(self, t):
        return self.deg(t) / 360.0

    def theta(self, t):
        return math.radians(self.deg(t))

    def omega(self, t):
        return math.radians(float(self._d(t)))

    def s(self, t):
        return float(self._d(t)) / (360.0 / self.rev0)

    def check(self, n=4000):
        """Monotone, and how hard it surges. Printed by the gate; never asserted away."""
        lo, hi = self.times[0] - 0.5, self.times[-1] + 0.5
        ts = np.linspace(lo, hi, n)
        w = np.array([float(self._d(v)) for v in ts])
        err = max(abs(self.deg(t) - (124.0 + 144.0 * i)) for i, t in enumerate(self.times))
        return dict(min_rate=float(w.min()), max_rate=float(w.max()),
                    surge=float(w.max() / max(w.min(), 1e-6)), knot_err_deg=float(err),
                    monotone=bool(w.min() > 0.0))


class Schedule:
    """The song, the firing plan, and the per-combustion word labels - one object the engine
    builder, the flash renderer and the muxer all read."""

    def __init__(self, slug, warm):
        self.slug = slug
        self.doc = A.load(slug)
        self.warm = float(warm)
        self.win = self.doc["window"]
        pl = self.doc["plan"]
        self.period = pl["period"]
        lab = {**DEFAULT_LABELS, **{str(k): v for k, v in A.labels(slug).items()}}
        self.labels = lab
        # clip time = song time - window start, then offset by the (unshown) warm-up
        self.fires = []
        marks = pl.get("mode") == "marks"
        for f in pl["fires"]:
            # a marks plan already quotes CLIP time (it came from the viewer); the older
            # plans quote SONG time and need the window subtracted
            tc = (f["t"] if marks else f["t"] - self.win[0]) + self.warm
            kind = (f.get("kind", "") if marks
                    else lab.get(str(f["sub"]), "") if f["on_vocal"] else "")
            self.fires.append(dict(t=round(tc, 4), song_t=f["t"], kind=kind,
                                   strength=f["strength"], sub=f["sub"]))
        self.t_fire0 = self.fires[0]["t"]
        self.n_word = sum(1 for f in self.fires if f["kind"])
        # THE FLASH LIST IS NOT THE COMBUSTION LIST. A word gets its own stroke; an echo is a
        # dimmer pulse on gas that is already burning, because two bangs 0.14 s apart is a
        # lurch. Each flash is anchored on the combustion at or before it, so it lights the
        # chamber that is actually alight.
        self.flashes = []
        for fl in pl.get("flashes", []):
            # SAME rule as the fires above: a marks plan is already in clip time, an older
            # plan is in song time. Guessing from the magnitude put every flash at -20 s.
            tc = (fl["t"] if marks else fl["t"] - self.win[0]) + self.warm
            cyl, best = None, 1e9
            for i, f in enumerate(self.fires):
                d = tc - f["t"]
                if -0.02 <= d < best:
                    best, cyl = d, self.cylinder_of(i)
            self.flashes.append(dict(t=round(tc, 4), kind=fl["kind"],
                                     strength=float(fl["strength"]),
                                     primary=bool(fl.get("primary", True)), cyl=cyl))
        if not self.flashes:                      # older plans: the fires ARE the flashes
            self.flashes = [dict(t=f["t"], kind=f["kind"], strength=f["strength"],
                                 primary=True, cyl=self.cylinder_of(i))
                            for i, f in enumerate(self.fires) if f["kind"]]

    def drive(self, rev0=5.0):
        """A SplineDrive when the plan fires on the words themselves (their spacing varies),
        a BeatDrive when it fires at one constant rate."""
        if self.doc["plan"].get("mode") in ("words", "marks"):
            return SplineDrive([f["t"] for f in self.fires], self.warm, 124.0, rev0)
        return BeatDrive(self.period, self.t_fire0, 124.0, rev0)

    def word_fires(self, kind=None):
        return [f for f in self.fires if f["kind"] and (kind is None or f["kind"] == kind)]

    def word_flashes(self, kind=None, primary=None):
        return [f for f in self.flashes
                if (kind is None or f["kind"] == kind)
                and (primary is None or f["primary"] == primary)]

    def summary(self):
        a, b = len(self.word_fires("a")), len(self.word_fires("b"))
        pl = self.doc["plan"]
        how = ("one per WORD" if pl.get("mode") == "words"
               else f"one every {self.period:.4f}s")
        rng = (f", gaps {pl['gap_min']:.3f}-{pl['gap_max']:.3f}s" if pl.get("mode") == "words"
               else "")
        wa = len(self.word_flashes("a", True))
        wb = len(self.word_flashes("b", True))
        ec = len(self.word_flashes(primary=False))
        return (f"[sync] {self.slug}: {len(self.fires)} combustions in "
                f"{self.win[1] - self.win[0]:.1f}s, {how} (mean {self.period:.3f}s{rng}; "
                f"{self.doc['bpm']:.2f} BPM); {self.n_word} land on a word; "
                f"FLASHES {wa} 'flashing' + {wb} 'lights' + {ec} echoes; "
                f"crank {2.5 * self.period:.3f}s/rev mean")

    # ---- what the draw pass asks, per frame ----
    ECHO = 0.46          # an echo is a pulse, not a bang

    def flash(self, t, kind, rise=0.012, fall=0.26):
        """Envelope 0..1 for flashes of `kind` at clip time t. A hard rise and an exponential
        fall - the shape of a real ignition, and short enough that it reads as an event rather
        than a level change. Echoes are scaled down: he marked them as echoes."""
        best = 0.0
        for f in self.flashes:
            if f["kind"] != kind:
                continue
            d = t - f["t"]
            if d < -rise or d > fall * 4.0:
                continue
            e = ((d + rise) / rise) if d < 0 else math.exp(-d / fall)
            lvl = (0.70 + 0.30 * min(1.0, f["strength"] / 0.7)) * (1.0 if f["primary"] else self.ECHO)
            best = max(best, min(1.0, e) * lvl)
        return best

    def flash_anchor(self, t, kind, fall=0.26):
        """The cylinder the most recent flash of `kind` belongs to."""
        cyl, best = None, 1e9
        for f in self.flashes:
            if f["kind"] != kind:
                continue
            d = t - f["t"]
            if -0.02 <= d < best and d <= fall * 4.0:
                best, cyl = d, f["cyl"]
        return cyl

    def any_fire(self, t, rise=0.010, fall=0.16):
        """The same envelope over EVERY combustion, word or not - the ordinary bang."""
        best = 0.0
        for f in self.fires:
            d = t - f["t"]
            if d < -rise or d > fall * 4.0:
                continue
            e = ((d + rise) / rise) if d < 0 else math.exp(-d / fall)
            best = max(best, min(1.0, e))
        return best

    def cylinder_of(self, fire_index, n=5):
        """Which cylinder carries fire i. Angles are 144 deg apart in firing order."""
        ang = _fire_angles(n)
        order = sorted(ang, key=lambda j: ang[j])
        return order[fire_index % len(order)]

    def fire_cylinders(self, n=5):
        return [self.cylinder_of(i, n) for i in range(len(self.fires))]
