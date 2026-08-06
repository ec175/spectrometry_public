"""crtfilm.py — the "filmed off a real bright CRT" post-filter.

Not a video-filter preset: a three-stage physical model, applied per frame in the correct
light domains, so the result reads as a *camera pointed at a glowing glass screen* rather
than a digitized script:

  SCREEN (linear light, things that happen at/inside the glass):
    - multi-scale HALATION: light from the trace scatters inside the faceplate glass —
      a tight reinforcement + a mid halo + a very wide, slightly desaturated glass glow.
    - PHOSPHOR GRAIN: the phosphor coating is granular; a *static* multiplicative texture
      that is only visible where the screen is lit (grain does not swim frame to frame).
    - DUST + SMUDGES on the glass: static specks/smears that light up when bright trace
      passes near them (multiplied by the local low-frequency luminance).
    - GLASS SHEEN: a faint static room-light reflection gradient on the curved glass.

  LENS (geometry + optics of the camera lens):
    - BARREL distortion from the curved faceplate + lens (graticule edges bow slightly).
    - transverse CHROMATIC ABERRATION: R and B sampled at slightly different radial
      scales -> coloured fringes on high-contrast edges toward the corners.
    - CORNER DEFOCUS (field curvature): edges go slightly soft, centre stays sharp.
    - optical VIGNETTE (cos^4-ish falloff).
    - micro HAND-SHAKE: a sub-pixel smoothed random-walk translation.

  CAMERA (sensor + response):
    - EXPOSURE FLICKER: slow ±1% breathing (shutter beating against the CRT refresh).
    - HUM BAND: an extremely faint broad horizontal brightness band drifting vertically.
    - HIGHLIGHT ROLLOFF: Reinhard-style soft knee; hot trace cores overexpose toward
      white with channel bleed (saturated green pushes into R/B) like a blown sensor.
    - BLACK LIFT: filmed blacks are never 0 (room light + sensor floor).
    - SENSOR NOISE: per-frame gaussian read noise in display space (fine luma + coarse
      blotchy chroma, the high-ISO look).

Backend: per-frame math runs on `osc.gpu.xp` — CuPy on the RTX 2060 when OSC_CUPY=1
(≈20-60× on the blurs/warps), transparently numpy/scipy otherwise. All STATIC textures and
schedules are generated with numpy from the seed, so the look is identical either way (only
the per-frame sensor noise differs by backend RNG, which is invisible by construction).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter as _gf_cpu

from .gpu import GPU, asnumpy, gaussian_filter, map_coordinates, randn, xp


def _blur_ds(a, sigma, ds):
    """Gaussian blur computed at 1/ds resolution (glow is low-frequency; ~ds^2 cheaper)."""
    if ds <= 1:
        return gaussian_filter(a, sigma=(sigma, sigma, 0))
    small = a[::ds, ::ds]
    b = gaussian_filter(small, sigma=(sigma / ds, sigma / ds, 0))
    up = xp.repeat(xp.repeat(b, ds, axis=0), ds, axis=1)
    return up[: a.shape[0], : a.shape[1]]


class FilmLook:
    # --- screen ---
    exposure = 1.75           # linear gain before everything: camera exposed for a BRIGHT trace
    halation = ((1.6, 1, 0.30), (9.0, 2, 0.22), (34.0, 4, 0.14))  # (sigma, downscale, gain)
    halation_desat = 0.50     # widest halo desaturates toward grey (glass scatter)
    grain_amp = 0.11          # phosphor granularity (multiplicative, lit areas only)
    grain_sigma = 0.6         # grain feature size (px)
    dust_amp = 0.16           # dust/smudge visibility under bright trace
    sheen_amp = 0.010         # static room-light sheen on the glass (linear add)
    ambient_gain = 0.0        # AMBILIGHT: colour of the content spilling through the glass into
    ambient_sigma = 150.0     #   the dark surround (0 = off; the ambient stays neutral white)
    # --- lens ---
    barrel_k = 0.030          # radial distortion coefficient (the whole-face bow)
    bevel = 0.0               # beveled-edge ridge amp (0 = off): a chamfer that seats the glass
    bevel_frac = 0.055        #   into the surrounding bezel; band width as frac of min(w,h)
    bevel_tint = (0.55, 1.30, 0.75)   # colour the bevel: green-glass ridge, deeper (darker) trough
    ca_scale = 0.0013         # R/B radial scale offset (transverse CA)
    defocus_sigma = 2.2       # corner softness blur
    defocus_amt = 0.85        # how fully corners blend to the blurred copy
    vignette_amt = 0.30       # corner darkening strength
    shake_step = 0.22         # px/frame random-walk step (smoothed)
    shake_max = 1.6           # px clamp
    # --- camera ---
    flicker_amp = 0.010       # exposure breathing
    hum_amp = 0.006           # drifting horizontal band strength
    hum_sigma = 110.0         # band half-thickness (px)
    hum_speed = 55.0          # px/s drift
    white_pt = 1.35           # Reinhard extended white point (lower = cores blow out harder)
    bleed_thr = 0.78          # green level where the sensor starts blowing out
    bleed_rb = (0.60, 0.50)   # how much excess green bleeds into R, B
    black_lift = (0.009, 0.013, 0.010)   # filmed black floor, green-tinted (phosphor face)
    noise_amp = 0.0065        # per-frame fine sensor noise (display space)
    cnoise_amp = 0.0045       # coarse blotchy chroma noise (high-ISO look), 1/4-res blurred

    def __init__(self, w, h, fps, n_frames, seed=7, *, halation_boost=1.0,
                 bleed_measure=None, bleed_into=None, black_lift=None,
                 tint_dust=None, tint_sheen=None, ambient_gain=None, ambient_sigma=None,
                 barrel_k=None, bevel=None, bevel_frac=None, bevel_tint=None):
        """Phosphor-hue overrides — ALL default to the green-phosphor behaviour, so every existing
        render (green/pink scenes) is byte-for-byte unchanged unless it opts in:
          halation_boost  scales the in-glass glow (>1 = more bloom/glow through the lens).
          bleed_measure   3-ch weights -> the 'how hot' scalar that drives highlight blowout
                          (green phosphor: (0,1,0); magenta: (0.5,0,0.5), keyed off R&B).
          bleed_into      per-ch gain the excess bleeds into so hot cores blow toward WHITE
                          (green: into R,B; magenta: into G).
          black_lift      filmed-black floor tint (green-tinted by default; magenta-tint for magenta).
          tint_dust/tint_sheen  colour of the lit dust + glass sheen."""
        self.w, self.h, self.fps = w, h, fps
        rng = np.random.default_rng(seed)
        # per-instance halation (glow) gains, scaled by halation_boost for "more glow through the lens"
        self._halation = [(s, ds, g * halation_boost) for (s, ds, g) in self.halation]
        if ambient_gain is not None:
            self.ambient_gain = float(ambient_gain)
        if ambient_sigma is not None:
            self.ambient_sigma = float(ambient_sigma)
        if barrel_k is not None:                    # stronger = the glass bows OUT further
            self.barrel_k = float(barrel_k)
        if bevel is not None:
            self.bevel = float(bevel)
        if bevel_frac is not None:
            self.bevel_frac = float(bevel_frac)
        if bevel_tint is not None:
            self.bevel_tint = bevel_tint

        # -- static screen textures (numpy-generated: identical on CPU and GPU) ------
        g = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        g = _gf_cpu(g, self.grain_sigma)
        self._grain = xp.asarray(g / (g.std() + 1e-9))

        dust = np.zeros((h, w), np.float32)
        n_specks = int(w * h / 12000)                                     # ~170 at 1080x1920
        ys = rng.integers(0, h, n_specks)
        xs = rng.integers(0, w, n_specks)
        dust[ys, xs] = rng.uniform(0.4, 1.0, n_specks).astype(np.float32)
        dust = _gf_cpu(dust, 1.1) * 9.0                                   # tiny soft specks
        for _ in range(3):                                                # a few faint smears
            sm = np.zeros((h, w), np.float32)
            cy, cx = rng.integers(int(h * .15), int(h * .85)), rng.integers(int(w * .15), int(w * .85))
            sm[cy, cx] = 1.0
            dust += _gf_cpu(sm, rng.uniform(45, 90)) * rng.uniform(2e3, 5e3) * 0.12
        self._dust = xp.asarray(np.clip(dust, 0, 1.2))

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sy, sx = (yy - h * 0.20) / (0.55 * h), (xx - w * 0.28) / (0.50 * w)
        self._sheen = xp.asarray(np.exp(-(sx * sx + sy * sy) * 2.2).astype(np.float32))

        # -- lens geometry: barrel + CA sampling grids per channel, stacked (2,h,w) --
        cx, cy = w / 2.0, h / 2.0
        rmax = np.hypot(cx, cy)
        xn, yn = (xx - cx) / rmax, (yy - cy) / rmax
        r2 = xn * xn + yn * yn
        self._maps = []
        for s in (1.0 + self.ca_scale, 1.0, 1.0 - self.ca_scale):        # R, G, B
            f = (1.0 + self.barrel_k * r2) * s
            self._maps.append(xp.asarray(
                np.stack([cy + yn * f * rmax, cx + xn * f * rmax]).astype(np.float32)))

        rn2 = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2              # 0 centre -> ~2 corner
        self._vig = xp.asarray(
            (1.0 - self.vignette_amt * np.clip(rn2 / 2.0, 0, 1) ** 1.3)[..., None].astype(np.float32))
        self._defmask = xp.asarray(
            (np.clip(rn2 / 2.0, 0, 1) ** 1.6 * self.defocus_amt)[..., None].astype(np.float32))

        # -- beveled edge ridge: a chamfer that seats the (nearly-flat) glass into the bezel.
        # From the rectangular border inward: a dark trough right at the edge, then a bright
        # highlight ridge, blending to flat in the interior — so the four edges read as a raised
        # bevel fitting into the terminal, without bowing the whole face.
        if self.bevel > 0.0:
            d_edge = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]).astype(np.float32)
            u = np.clip(d_edge / (self.bevel_frac * min(w, h)), 0.0, 1.0)   # 0 edge -> 1 interior
            ridge = np.exp(-((u - 0.60) / 0.16) ** 2)                       # bright chamfer crest
            trough = np.exp(-((u - 0.08) / 0.11) ** 2)                      # shadow into the bezel
            prof = self.bevel * (ridge - 0.85 * trough)
            prof[u >= 1.0] = 0.0
            self._bevel = xp.asarray(prof[..., None].astype(np.float32))
            self._bevel_tintv = xp.asarray(np.asarray(self.bevel_tint, np.float32))
        else:
            self._bevel = None

        # -- camera temporal tracks (numpy: scalar lookups per frame) ---------------
        steps = rng.normal(0, 1, (2, n_frames + 8)).astype(np.float32)
        walk = np.cumsum(steps, axis=1) * self.shake_step
        walk = _gf_cpu(walk, sigma=(0, 3.0))
        self._shake = np.clip(walk - walk.mean(axis=1, keepdims=True),
                              -self.shake_max, self.shake_max)
        fl = _gf_cpu(rng.normal(0, 1, n_frames + 8).astype(np.float32), 6.0)
        self._flicker = 1.0 + self.flicker_amp * fl / (fl.std() + 1e-9)
        self._ygrid = xp.asarray(yy[:, :1])                              # (h,1) for the hum band
        # colour-tint constants on the backend (numpy*cupy mixing is not allowed). All configurable
        # so the filter matches the phosphor hue (green default; magenta for the circle format).
        self._tint_dust = xp.asarray(tint_dust if tint_dust is not None else [0.85, 1.0, 0.9],
                                     dtype=xp.float32)
        self._tint_sheen = xp.asarray(tint_sheen if tint_sheen is not None else [0.9, 1.0, 0.95],
                                      dtype=xp.float32)
        self._lift = xp.asarray(black_lift if black_lift is not None else self.black_lift,
                                dtype=xp.float32)
        # highlight-blowout config: default keys off GREEN and bleeds into R,B (green phosphor)
        self._bleed_measure = bleed_measure if bleed_measure is not None else (0.0, 1.0, 0.0)
        self._bleed_into = (bleed_into if bleed_into is not None
                            else (self.bleed_rb[0], 0.0, self.bleed_rb[1]))

    # ---------------------------------------------------------------------
    def process(self, frame_u8, i):
        f = (xp.asarray(frame_u8).astype(xp.float32) / 255.0) ** 2.2      # -> linear light
        f *= self.exposure                                                # exposed for the trace

        # SCREEN ----------------------------------------------------------
        halo = xp.zeros_like(f)
        for sigma, ds, gain in self._halation:
            halo += _blur_ds(f, sigma, ds) * gain
        wide = halo.mean(axis=2, keepdims=True)                           # desaturate widest part
        halo = halo * (1 - self.halation_desat) + wide * self.halation_desat
        f = f + halo

        # AMBILIGHT: a very wide, COLOUR-PRESERVING blur of the lit screen spills the content's
        # own colour into the dark surround / onto the glass — so the ambient light is tinted by
        # the video (a cyan face glows cyan around it) instead of a neutral white sheen.
        if self.ambient_gain > 0.0:
            f = f + _blur_ds(f, self.ambient_sigma, 8) * self.ambient_gain

        lum = f.mean(axis=2)
        lit = xp.clip(lum * 3.0, 0, 1)
        f *= (1.0 + self.grain_amp * self._grain * lit)[..., None]

        glow = gaussian_filter(lum[::4, ::4], 6.0)
        glow = xp.repeat(xp.repeat(glow, 4, axis=0), 4, axis=1)[: self.h, : self.w]
        f += (self._dust * glow * self.dust_amp)[..., None] * self._tint_dust
        f += (self._sheen * self.sheen_amp)[..., None] * self._tint_sheen

        # LENS ------------------------------------------------------------
        dy, dx = float(self._shake[0, i]), float(self._shake[1, i])
        out = xp.empty_like(f)
        for c in range(3):
            co = xp.empty_like(self._maps[c])
            co[0] = self._maps[c][0] + dy
            co[1] = self._maps[c][1] + dx
            out[:, :, c] = map_coordinates(f[:, :, c], co,
                                           order=1, mode="constant", cval=0.0)
        soft = _blur_ds(out, self.defocus_sigma, 2)
        out = out * (1 - self._defmask) + soft * self._defmask
        out *= self._vig
        if self._bevel is not None:                                       # tinted beveled edge ridge
            out *= (1.0 + self._bevel * self._bevel_tintv)

        # CAMERA ----------------------------------------------------------
        t = i / self.fps
        y0 = (t * self.hum_speed) % (self.h + 4 * self.hum_sigma) - 2 * self.hum_sigma
        band = 1.0 + self.hum_amp * xp.exp(-((self._ygrid - y0) ** 2) / (2 * self.hum_sigma ** 2))
        out *= (band * float(self._flicker[i]))[..., None]

        wp2 = self.white_pt * self.white_pt
        out = out * (1.0 + out / wp2) / (1.0 + out)                       # Reinhard extended
        m = self._bleed_measure                                          # 'how hot' per phosphor hue
        hot = out[:, :, 0] * m[0] + out[:, :, 1] * m[1] + out[:, :, 2] * m[2]
        excess = xp.clip(hot - self.bleed_thr, 0, None)
        bi = self._bleed_into                                            # bleed the excess -> white
        if bi[0]:
            out[:, :, 0] += excess * bi[0]
        if bi[1]:
            out[:, :, 1] += excess * bi[1]
        if bi[2]:
            out[:, :, 2] += excess * bi[2]

        out = xp.clip(out, 0, 1) ** (1 / 2.2)                             # -> display space
        out = self._lift + (1.0 - self._lift) * out
        out += xp.asarray(randn(i * 7919 + 13, out.shape)) * self.noise_amp
        cn = xp.asarray(randn(i * 7919 + 14, (self.h // 4 + 1, self.w // 4 + 1, 3)))
        cn = gaussian_filter(cn, sigma=(1.2, 1.2, 0))
        cn = xp.repeat(xp.repeat(cn, 4, axis=0), 4, axis=1)[: self.h, : self.w]
        out += cn * self.cnoise_amp
        return asnumpy((xp.clip(out, 0, 1) * 255.0 + 0.5).astype(xp.uint8))


def film_config_for_phosphor(color, halation_boost=1.0, bleed_strength=0.85, lift_level=0.010):
    """Derive FilmLook phosphor-hue kwargs FROM THE TRACE / PHOSPHOR COLOUR (RGB 0..255), so the
    "filmed off a CRT" look automatically matches whatever colour the trace is — green, magenta,
    amber, cyan, or a hue that changes between renders. Returns a dict to splat into FilmLook(...).

    The physics: a lit phosphor emits in its own hue, and a camera filming a HOT core overexposes
    that core toward WHITE (the deficient channels fill in). So:
      - `bleed_measure` (the 'how hot' scalar) is weighted by the colour's LIT channels;
      - `bleed_into` pushes the excess into the colour's DEFICIENT channels (∝ how far each sits
        below the brightest one) -> any hue blows out to white, not to a tint;
      - filmed blacks + lit dust/sheen are tinted gently toward the hue.
    Feed it green -> the module's original green look; magenta -> the magenta look; and it is ready
    to be called PER-FRAME later for dynamic/animated trace colours (it is pure + cheap)."""
    c = np.asarray(color, dtype=np.float64)
    c = c / max(float(c.max()), 1e-9)                  # unit colour (brightest channel -> 1)
    weights = c / max(float(c.sum()), 1e-9)            # 'how hot' measure (sums to 1)
    into = bleed_strength * (1.0 - c)                  # bleed into the deficient channels -> white
    dust = 0.85 + 0.15 * c
    sheen = 0.90 + 0.10 * c
    lift = lift_level * (0.6 + 0.6 * c)                # filmed-black floor, tinted toward the hue
    return dict(
        halation_boost=halation_boost,
        bleed_measure=tuple(float(x) for x in weights),
        bleed_into=tuple(float(x) for x in into),
        black_lift=tuple(float(x) for x in lift),
        tint_dust=tuple(float(x) for x in dust),
        tint_sheen=tuple(float(x) for x in sheen),
    )
