"""field.py - the ILLUMINATION field and the PALETTE field.

Two smooth functions of (position, time) evaluated at the lattice sites. Together they are
what makes the picture: the lattice and its content are almost static, and essentially all
the motion on screen is these two fields sweeping across a fixed substrate. That is not a
stylistic guess - vdx measured the source clip as `locked camera, 0.0 px/s drift` while its
frame luminance swings 1.6 -> 144.5, so whatever moves is not the geometry.

  Illumination(P, t) -> (N,) in [0,1]
      A band-limited sum of travelling plane waves plus rotating spiral arms plus a few
      drifting flares. Gates WHICH elements are drawn (below `cut` an element is skipped
      entirely, which is also the render's main speed control) and how hot they are.

  Palette(P, t) -> (N,3) in [0,1]
      A slow smooth scalar field mapped through a measured BIMODAL LUT. The bimodality is
      the point: k-means over the source's bright pixels found a warm cluster (hue 6-37 deg,
      37% of saturated pixels) and a cool cluster (hue 182-241 deg, 55%) with a magenta
      accent (298 deg, 8%) and almost nothing in between. A continuous rainbow ramp is the
      obvious implementation and it is wrong - it fills the empty hues and the result reads
      as generic RGB rather than as this clip.

`_smooth_noise` is a random-phase Fourier sum rather than value/Perlin noise: it is exactly
band-limited (so the field cannot develop detail at the lattice pitch and alias into a
flicker), it is 8 lines, and it is trivially evaluable at arbitrary scattered points.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

# --- measured palette (k-means over bright saturated pixels of reel_51d5e4f3) -------------
COOL = [(0x37, 0x3c, 0xff), (0x43, 0xa4, 0xfe), (0x83, 0xed, 0xf1), (0x97, 0x95, 0xff)]
WARM = [(0xfe, 0x83, 0x58), (0xff, 0xa0, 0x95), (0xfb, 0xd6, 0x9a)]
ACCENT = [(0xeb, 0x9e, 0xee)]
COOL_FRAC = 0.55        # share of saturated PIXELS in the cool cluster, as measured
WARM_FRAC = 0.37

# LUT segment widths are NOT the measured pixel shares, for the same reason `sat_boost`
# exists: the measured shares describe the source's OUTPUT pixels, and the map from
# "what hue an element was assigned" to "what hue its pixels read as" is not the identity.
# Additive bloom lifts a warm element's blue channel (warm is R-dominant, so its weakest
# channel is the one every neighbouring cool halo feeds), which drags it below the sat > 0.3
# threshold the measurement uses and drops it out of the histogram entirely. Cool elements
# are B-dominant and keep their saturation. Measured: LUT warm 0.371 rendered as 0.278, LUT
# cool 0.551 rendered as 0.669. These pre-image shares are calibrated so the RENDER lands on
# the measured 0.346 / 0.511.
COOL_LUT = 0.48
WARM_LUT = 0.43
BG = (0x06, 0x03, 0x08)


def _ramp(colors, n):
    c = np.array(colors, np.float32) / 255.0
    if len(c) == 1:
        return np.repeat(c, n, 0)
    x = np.linspace(0, len(c) - 1, n)
    i = np.clip(x.astype(int), 0, len(c) - 2)
    f = (x - i)[:, None]
    return c[i] * (1 - f) + c[i + 1] * f


def build_lut(n=256, cool_frac=COOL_LUT, warm_frac=WARM_LUT, sat_boost=0.85):
    """The bimodal palette LUT. Segment widths are the measured cluster shares CALIBRATED
    through the renderer - see the COOL_LUT / WARM_LUT note above.

    `sat_boost` pushes each swatch away from its own luminance. The k-means centroids are
    measured off pixels that had ALREADY been through the source's bloom, so they are the
    washed-out result, not the input - feeding them back in unmodified and blooming a second
    time compounds the desaturation and the frame comes out pastel. Boosting recovers the
    pre-bloom ink. This is the one place the pipeline must not use a measurement literally.
    """
    n_cool = int(round(n * cool_frac))
    n_warm = int(round(n * warm_frac))
    n_acc = max(1, n - n_cool - n_warm)
    lut = np.concatenate([_ramp(COOL, n_cool), _ramp(WARM, n_warm),
                          _ramp(ACCENT, n_acc)]).astype(np.float32)
    lum = lut @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lut = np.clip(lum[:, None] + (lut - lum[:, None]) * (1.0 + sat_boost), 0.0, 1.0)
    return lut.astype(np.float32)


# ------------------------------------------------------------------------------------------
def _smooth_noise(P, t, rng, n_modes=6, wavelength=900.0, speed=0.05):
    """Band-limited scalar field on scattered points. Returns roughly [-1, 1]."""
    ang = rng.uniform(0, 2 * np.pi, n_modes)
    k = 2 * np.pi / (wavelength * rng.uniform(0.6, 1.8, n_modes))
    kx, ky = np.cos(ang) * k, np.sin(ang) * k
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    w = rng.uniform(-1, 1, n_modes) * speed * 2 * np.pi
    v = np.sin(P[:, 0:1] * kx[None] + P[:, 1:2] * ky[None] + ph[None] + w[None] * t)
    return v.mean(1) * np.sqrt(n_modes) * 0.7


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Illumination:
    """Travelling waves + spiral arms + drifting flares -> a [0,1] gate at every site."""

    def __init__(self, w, h, seed=0, n_waves=4, n_spirals=2, n_flares=3,
                 wavelength=520.0, drift=0.22, spin=0.13,
                 axis_deg=52.0, axis_spread=38.0,
                 octaves=3, octave_gain=0.55, octave_speed=1.3):
        self.w, self.h = float(w), float(h)
        rng = np.random.default_rng(seed)
        self.rng_seed = seed

        # Plane-wave directions are CLUSTERED about `axis_deg`, not uniform on the circle.
        # Uniform directions interfere into an isotropic blob field; the source's lit regions
        # are elongated diagonal streaks, which is what a narrow angular spread produces.
        a = np.deg2rad(axis_deg + rng.uniform(-axis_spread, axis_spread, n_waves))
        lam = wavelength * rng.uniform(0.55, 1.9, n_waves)
        k = 2 * np.pi / lam
        self.kx, self.ky = np.cos(a) * k, np.sin(a) * k
        self.wph = rng.uniform(0, 2 * np.pi, n_waves)
        self.ww = rng.uniform(-1, 1, n_waves) * drift * 2 * np.pi
        self.wamp = rng.uniform(0.7, 1.0, n_waves)

        # spiral arms: sin(2*pi*r/lam + m*theta - w t). `m` is the arm count.
        self.sc = np.stack([rng.uniform(0.15, 0.85, n_spirals) * w,
                            rng.uniform(0.15, 0.85, n_spirals) * h], 1).astype(np.float32)
        self.sc0 = self.sc.copy()
        self.slam = rng.uniform(360.0, 900.0, n_spirals)
        self.sm = rng.choice([2, 3, 3, 4, 5], n_spirals).astype(np.float32)
        self.sw = rng.choice([-1.0, 1.0], n_spirals) * rng.uniform(0.5, 1.0, n_spirals) \
            * spin * 2 * np.pi
        self.samp = rng.uniform(0.8, 1.3, n_spirals)
        self.swander = rng.uniform(0, 2 * np.pi, (n_spirals, 2))
        self.swrate = rng.uniform(0.03, 0.08, (n_spirals, 2)) * 2 * np.pi

        # flares: slow bright bumps that sweep across and blow their neighbourhood white
        self.fc = np.stack([rng.uniform(0.1, 0.9, n_flares) * w,
                            rng.uniform(0.1, 0.9, n_flares) * h], 1).astype(np.float32)
        self.fv = rng.normal(0, 1, (n_flares, 2)).astype(np.float32)
        self.fv /= np.maximum(np.hypot(*self.fv.T)[:, None], 1e-6)
        self.fv *= rng.uniform(28.0, 75.0, (n_flares, 1))
        self.fr = rng.uniform(0.16, 0.34, n_flares) * w
        self.fper = rng.uniform(3.4, 7.5, n_flares)
        self.fph = rng.uniform(0, 1, n_flares)

        # OCTAVES. The base layer (a handful of long waves plus spirals) is smooth by
        # construction, and measurement says the source is not: at 161 px the source's
        # light-field power is 0.093 against this field's 0.034, and it resolves 12.9 distinct
        # lit regions per frame against 7.1. Each octave halves the wavelength, cuts the
        # amplitude by `octave_gain` and moves `octave_speed` times faster - small structures
        # that live and die quickly inside the big arms, which is what "detailed lighting"
        # actually is here.
        #
        # `octave_speed` is deliberately LOW (1.3, not the 1.9 first tried). The octaves are
        # here for SPATIAL detail; if they also move fast they generate high-frequency
        # temporal energy that is spatially smooth, and the neighbour-HF correlation goes to
        # 0.44 against the source's 0.20. The high-frequency motion has to come from Twinkle,
        # which is per-element and therefore uncorrelated - that is the division of labour.
        self.oct = []
        for o in range(octaves):
            n = 5
            a = np.deg2rad(rng.uniform(0, 360, n))
            k = 2 * np.pi / (wavelength / (2.0 ** (o + 1)) * rng.uniform(0.7, 1.5, n))
            self.oct.append(dict(
                kx=np.cos(a) * k, ky=np.sin(a) * k,
                ph=rng.uniform(0, 2 * np.pi, n),
                w=rng.uniform(-1, 1, n) * drift * (octave_speed ** (o + 1)) * 2 * np.pi,
                amp=(octave_gain ** (o + 1)) / np.sqrt(n)))

    def raw(self, P, t):
        """The unnormalised field, roughly [-1.5, 1.5]."""
        v = (np.sin(P[:, 0:1] * self.kx[None] + P[:, 1:2] * self.ky[None]
                    + self.wph[None] + self.ww[None] * t) * self.wamp[None]).sum(1)
        # spiral centres wander so the arms never settle into a fixed rosette
        cx = self.sc0[:, 0] + 0.10 * self.w * np.sin(self.swander[:, 0] + self.swrate[:, 0] * t)
        cy = self.sc0[:, 1] + 0.07 * self.h * np.sin(self.swander[:, 1] + self.swrate[:, 1] * t)
        for i in range(len(self.slam)):
            dx = P[:, 0] - cx[i]
            dy = P[:, 1] - cy[i]
            r = np.hypot(dx, dy)
            th = np.arctan2(dy, dx)
            v += self.samp[i] * np.sin(2 * np.pi * r / self.slam[i]
                                       + self.sm[i] * th - self.sw[i] * t)
        n = len(self.wamp) + self.samp.sum()
        v = v / max(n, 1e-6) * 2.2
        for o in self.oct:
            v += o["amp"] * np.sin(P[:, 0:1] * o["kx"][None] + P[:, 1:2] * o["ky"][None]
                                   + o["ph"][None] + o["w"][None] * t).sum(1)
        return v

    def flare(self, P, t):
        """(N,) additive hot-spot term in [0, ~1.4]. Wraps around the frame."""
        out = np.zeros(len(P), np.float32)
        for i in range(len(self.fr)):
            cx = (self.fc[i, 0] + self.fv[i, 0] * t) % (self.w * 1.4) - 0.2 * self.w
            cy = (self.fc[i, 1] + self.fv[i, 1] * t) % (self.h * 1.4) - 0.2 * self.h
            amp = 0.5 + 0.5 * np.sin(2 * np.pi * (t / self.fper[i] + self.fph[i]))
            d2 = (P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2
            out += (amp ** 2) * np.exp(-d2 / (2.0 * self.fr[i] ** 2))
        return out

    def __call__(self, P, t, lo=-0.15, hi=0.85, flare_gain=0.85, gamma=1.0,
                 ambient=0.0, hot_gain=2.6):
        """The gate, in [ambient, >1].

        `lo`/`hi` are the exposure handles: raising `lo` thins the lit regions into arms
        rather than shrinking them uniformly, narrowing the gap hardens their edges.

        Three things happen here that a plain smoothstep does not do, and each one is a
        measured property of the source rather than a preference:

        `gamma`   applied INSIDE, before ambient. Mid-lit elements stay dim and saturated
                  (source: mean saturation of bright pixels 0.389) instead of crowding the
                  top of the range where the tonemap desaturates everything equally.
        `ambient` a floor, so the field's dark regions still carry faint structure. The
                  source is not black there - it measures 40.8% of pixels below luma 20,
                  which means ~59% are above it, and a pure gate cannot produce that without
                  also destroying the arms.
        `hot`     the gate SATURATES at 1 but the field does not. Past `hi` the excess is
                  returned as a multiplier > 1, so crests and flares drive their elements far
                  enough up the tonemap to bleach to white. Without this the render has no
                  white cores at all, and the source spends 5.4% of its pixels above luma 200.
        """
        v = self.raw(P, t) + flare_gain * self.flare(P, t)
        g = smoothstep(lo, hi, v) ** gamma
        hot = 1.0 + hot_gain * np.maximum(v - hi, 0.0)
        return (ambient + (1.0 - ambient) * g) * hot


class Twinkle:
    """Independent per-element flicker.

    Measured on the source: high-frequency (>~1.7 Hz) motion carries 12.8% of each cell's
    temporal variance, the global mean of that component is flat (0.93 against a per-cell
    8.62, so it is not a global exposure flicker), and the correlation between neighbouring
    cells' high-frequency components is only 0.20. Every element is doing its own thing.

    NO smooth field can produce that. `Illumination` is band-limited by construction -
    neighbouring sites are 19.25 px apart under a field whose finest octave is ~40 px, so
    their high-frequency components are necessarily near-identical. The twinkle has to be
    indexed by ELEMENT, not by position, and that is the whole reason this class exists.

    Two incommensurate oscillators per element, at rates around 4-13 Hz. Two rather than one
    because a single sine is visibly periodic at these rates - the element pulses like a
    metronome - while two beat against each other and read as flicker.
    """

    def __init__(self, n, seed=0, rate=(3.5, 13.0), depth=0.62, n_osc=2):
        rng = np.random.default_rng(seed + 5309)
        self.f = rng.uniform(rate[0], rate[1], (n_osc, n)).astype(np.float32)
        self.ph = rng.uniform(0, 2 * np.pi, (n_osc, n)).astype(np.float32)
        self.depth = float(depth)

    def __call__(self, t):
        """(n,) multiplier, mean 1, clipped at 0 so an element can go fully dark."""
        s = np.sin(self.f * (2 * np.pi * t) + self.ph).mean(0)
        return np.maximum(1.0 + self.depth * s * np.sqrt(2.0), 0.0)


class Palette:
    """Spatially coherent hue assignment through the measured bimodal LUT.

    The hue field must vary on a SHORTER scale than the illumination field, and this is the
    one coupling in the project that has to be designed rather than left to chance:

    1. Correlation. Both fields are low-mode random Fourier sums. Two of those at comparable
       wavelengths correlate substantially on any given frame purely by small-sample luck -
       measured at wavelength 760 against illumination's 620, mean illumination came out
       0.343 on warm sites against 0.454 on cool ones, which dragged the rendered hue balance
       to 0.30/0.65 even though the LUT shares and the `u` distribution were both correct.
       Separating the scales decorrelates them.
    2. Look. At a comparable wavelength each lit region comes out a single colour. The source
       carries warm and cool WITHIN one arm, which only happens if hue turns over faster than
       illumination does.

    Both point the same way, so `wavelength` is well under Illumination's and `n_modes` is
    higher.
    """

    # `wavelength` sets a CEILING on the far-field plateau, independently of `fine_weight`:
    # a 900 px regional field is itself only ~0.48 correlated at eight pitches (154 px), so
    # no weighting can push the measured plateau above that. The source plateaus AT 0.484, so
    # the regional field has to be much longer than the distance being measured.
    def __init__(self, w, h, seed=0, wavelength=1700.0, speed=0.045, lut=None, n_modes=5,
                 jitter_amp=0.10, scatter=0.07,
                 fine_wavelength=78.0, fine_weight=0.46, fine_modes=6):
        self.lut = build_lut() if lut is None else lut
        self.seed = seed + 9161
        self.wavelength, self.speed, self.n_modes = wavelength, speed, n_modes
        self.jitter_amp = float(jitter_amp)
        self.scatter = float(scatter)
        self.fine_wavelength = float(fine_wavelength)
        self.fine_weight = float(fine_weight)
        self.fine_modes = int(fine_modes)
        self.w, self.h = w, h
        # Measure the noise's own spread once, so `scalar`'s CDF transform is calibrated to
        # this mode set rather than to an assumed variance.
        probe = np.stack([np.random.default_rng(4).uniform(0, w, 4000),
                          np.random.default_rng(5).uniform(0, h, 4000)], 1)
        self.sigma = float(np.std(_smooth_noise(
            probe, 0.0, np.random.default_rng(self.seed), n_modes=n_modes,
            wavelength=wavelength, speed=speed))) or 1.0
        self.fine_sigma = float(np.std(_smooth_noise(
            probe, 0.0, np.random.default_rng(self.seed + 5501), n_modes=self.fine_modes,
            wavelength=self.fine_wavelength, speed=speed * 2.2))) or 1.0

    def scalar(self, P, t):
        """[0,1], and UNIFORMLY distributed on it.

        The noise RNG is re-seeded identically every call ON PURPOSE - it selects a fixed set
        of Fourier modes, so the field is a deterministic function of (P, t) and a still at t
        matches the same t in a render.

        The erf is not decoration. A sum of sines is bell-shaped, so an affine map into [0,1]
        piles most sites near 0.5 and they all land in whichever LUT segment straddles the
        middle - measured as hue_cool 0.72 against the source's 0.511, with the accent segment
        never reached at all. Pushing the field through its own CDF makes `u` uniform, and
        only then do the LUT's segment widths actually deliver the measured cluster shares.
        """
        v = _smooth_noise(P, t, np.random.default_rng(self.seed), n_modes=self.n_modes,
                          wavelength=self.wavelength, speed=self.speed)
        if self.fine_weight > 1e-4:
            # TWO SCALES, and this is what the source's hue statistics actually describe.
            # Correlation between the hues of two elements: 0.765 one pitch apart, falling to
            # a PLATEAU near 0.484 by eight pitches and staying there. A single field cannot
            # do that - a long one gives ~0.98 next door, a short one gives ~0 far away. A
            # long regional field sets the plateau; a fine field on top (~5 pitches) supplies
            # the local variety, and the variance split IS the plateau height.
            #
            # Crucially the fine field is SMOOTH, not random. Neighbouring elements land on
            # nearby LUT entries, so where their halos overlap they blend between related
            # hues. Randomising instead (the first attempt, 58% of elements taking an
            # arbitrary hue) hit the correlation target but put unrelated colours on top of
            # each other, and additive overlap turned them white: white_of_bright 0.51 against
            # the source's 0.321, saturation 0.26 against 0.389.
            f = _smooth_noise(P, t, np.random.default_rng(self.seed + 5501),
                              n_modes=self.fine_modes,
                              wavelength=self.fine_wavelength, speed=self.speed * 2.2)
            w = self.fine_weight
            v = v * (1.0 - w) + f * w * (self.sigma / max(self.fine_sigma, 1e-6))
            s = self.sigma * np.sqrt((1.0 - w) ** 2 + w ** 2)
        else:
            s = self.sigma
        return np.clip(0.5 * (1.0 + erf(v / (s * np.sqrt(2.0)))), 0.0, 1.0)

    def __call__(self, P, t, jitter=None, scatter_u=None):
        """(N,3) RGB in [0,1]. `jitter` (N,) in [-1,1] is a fixed PER-ELEMENT hue offset;
        `scatter_u` (N,) in [0,1] lets a fraction of elements take a hue from anywhere.

        `jitter_amp` is the single most important colour constant in the project, and the
        first build had it 1.7x too small. Measured on the source, the correlation between
        the hues of two elements a given distance apart is:

            distance      1 pitch   2      4      8
            source         +0.77  +0.61  +0.54  +0.48
            first build    +0.98  +0.92  +0.69  +0.04

        Two separate errors, pulling opposite ways. Neighbouring elements in the source only
        agree 77% - it assigns colour with real PER-ELEMENT variety, which is why an isolated
        green dot sits happily next to an orange ring. And its correlation then PLATEAUS near
        0.5 out to eight pitches and beyond, because underneath the variety there is a broad
        regional tint. The first build had it exactly backwards: a smooth short-wavelength
        field, so neighbours were identical and any regional bias was gone by 150 px.

        The fix is both at once - a LONG regional wavelength for the plateau, plus a strong
        per-element offset for the local variety. For an offset uniform in +/-a the adjacent
        correlation is (sin(2*pi*a)/(2*pi*a))^2, so 0.77 wants a ~ 0.135.

        The offset WRAPS rather than clipping. Clipping piles every jittered element at the
        two ends of the LUT, which is a colour cast, not variety.
        """
        u = self.scalar(P, t)
        if jitter is not None:
            u = np.mod(u + self.jitter_amp * jitter, 1.0)
        if scatter_u is not None and self.scatter > 1e-4:
            # A fraction of elements ignore the regional field and take a hue from anywhere
            # in the palette. On a BIMODAL LUT a bounded offset is not enough on its own: a
            # +/-0.145 shift inside the 0.47-wide cool segment still lands on a blue, so
            # adjacent elements stay in the same family and the measured hue correlation
            # sticks at 0.94 against the source's 0.765. The source clearly puts an isolated
            # green beside an orange - it jumps CLUSTERS, and only a scatter term does that.
            # One array serves as both the decision and the value, so the two cannot
            # accidentally correlate.
            hit = scatter_u < self.scatter
            u = np.where(hit, scatter_u / max(self.scatter, 1e-6), u)
        idx = np.clip((u * (len(self.lut) - 1)).astype(np.int32), 0, len(self.lut) - 1)
        return self.lut[idx]


def envelope(t, dur, fade_in=0.55, fade_out=1.1):
    """The tempo arc, compressed from the source's measured per-second luminance.

    The source runs 46 s: builds to a hard climax at 37% (mean luma 40 -> 137), settles to a
    long plateau around 45, then decays to black over the last 3 s. vdx reports
    `climax @37%`. Both peaks are kept so a 15 s cut still has the source's shape rather than
    a single lazy swell.
    """
    x = np.clip(t / max(dur, 1e-6), 0.0, 1.0)
    base = 0.42
    main = 0.62 * np.exp(-((x - 0.37) ** 2) / (2 * 0.085 ** 2))
    second = 0.26 * np.exp(-((x - 0.76) ** 2) / (2 * 0.10 ** 2))
    e = base + main + second
    e *= smoothstep(0.0, fade_in, t) * smoothstep(0.0, fade_out, dur - t)
    return float(e)
