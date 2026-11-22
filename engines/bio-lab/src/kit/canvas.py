# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""canvas.py - the shared render surface. The engine draws through this and nothing else.

TWO LAYERS, AND WHY
-------------------
A frame is built from two buffers that composite differently:

* ``base``  - OPAQUE paint. The background, and any solid fill. Later paint replaces
  earlier paint. This is what the Objects engine is mostly made of: crisp, readable
  shapes that should look like themselves and not like light.
* ``emit``  - ADDITIVE light. Overlapping deposits SUM, and only this layer blooms. This
  is what the Bio engine is almost entirely made of: a glowing junctional network where
  two edges crossing genuinely is brighter than one.

Keeping them apart is what lets one post chain serve a flat vector look and a luminous
microscopy look without either being a special case. `final = tonemap(base + emit +
bloom(emit))`.


THE BLOOM IS THRESHOLDED, ALWAYS
--------------------------------
``bloom = blur(max(emit - threshold, 0))``, never ``blur(emit)``.

Blooming the whole emissive layer takes every faint deposit and smears it over roughly
``2*pi*sigma^2`` pixels; with a few hundred glowing edges those tails SUM into a broad
haze that lifts the entire frame off black, and then no exposure setting recovers the
contrast because the wash is not noise, it is signal in the wrong place. A thresholded
bloom is also what the real optical effect IS - veiling glare comes off highlights, not
off everything.


SUPERSAMPLING IS THE ANTI-ALIASING
----------------------------------
Nothing here anti-aliases. Everything rasterises hard-edged at ``cfg.render_size``
(= output size x ``cfg.ss``) and `to_rgb` box-downsamples at the end. That is why
delivery tiers use ``ss=2`` and the live preview uses ``ss=1``: the live view is
deliberately a little jaggy, and paying 4x the pixels to smooth a thing you are dragging
a slider on is a bad trade.

Consequence worth knowing: a 1 ru hairline is 1 px at ss=1 and 2 px at ss=2, and after
downsampling the ss=2 version is a smooth grey line while the ss=1 version is a hard
black one. Lines therefore look SOFTER in the final than in the preview. That is correct
and is not a bug to chase.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import gpu as G

try:
    from scipy.ndimage import gaussian_filter as _gf
    HAVE_SCIPY = True
except Exception:                                       # pragma: no cover - fallback path
    HAVE_SCIPY = False
    _gf = None


def _blur_direct(a, sigma):
    """Gaussian blur at full resolution, sigma in PIXELS.

    Falls back to three box blurs when scipy is missing - three boxes approximate a
    gaussian to within a few percent (central limit theorem), far closer than the eye
    needs for a glow. The engine must not hard-require scipy just to make light soft.
    """
    if HAVE_SCIPY:
        s = (sigma, sigma, 0) if a.ndim == 3 else (sigma, sigma)
        return _gf(a, s, mode="nearest")
    r = max(1, int(round(sigma * 1.2)))
    out = np.ascontiguousarray(a, np.float32)
    for _ in range(3):
        k = 2 * r + 1
        pad = [(r, r), (r, r)] + ([(0, 0)] if out.ndim == 3 else [])
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=0)
        out = (c[k:, ...] - c[:-k, ...]) / k
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=1)
        out = (c[:, k:, ...] - c[:, :-k, ...]) / k
    return out


#: Cache of tone curves, keyed by gamma. See `_tone_lut_apply`.
_TONE_LUTS = {}
#: Accumulator values above this map to essentially 1.0 through the Reinhard shoulder, so
#: the table does not need to extend further.
_TONE_MAX = 24.0
_TONE_N = 8192


def _tone_lut_apply(img, gamma):
    """`255 * a**g / (1 + a**g)` where `a = img/255`, evaluated through a lookup table.

    Mathematically identical to calling `np.power` directly, to within the table's
    resolution - and much faster. `np.power` on the 25 million floats of a 4K frame is one
    of the most expensive single operations in the whole post chain; a table lookup is an
    integer index and a gather.

    The table covers `a` in [0, 24]; beyond that the Reinhard shoulder is within 0.2% of
    its limit, so everything higher is clamped to the last entry with no visible error.
    """
    lut = _TONE_LUTS.get(gamma)
    if lut is None:
        a = np.linspace(0.0, _TONE_MAX, _TONE_N, dtype=np.float32)
        ag = np.power(a, gamma)
        lut = (255.0 * ag / (1.0 + ag)).astype(np.float32)
        if len(_TONE_LUTS) > 32:
            _TONE_LUTS.clear()
        _TONE_LUTS[gamma] = lut
    a = np.maximum(img, 0.0) * (1.0 / 255.0)
    idx = np.minimum(a * (_TONE_N / _TONE_MAX), _TONE_N - 1).astype(np.int32)
    return lut[idx]


def box_down(a, s):
    """Box-average an (H, W, 3) array by an integer factor `s`. H and W must divide by s.

    Written as a sum of strided slices rather than the obvious
    ``a.reshape(h, s, w, s, 3).mean(axis=(1, 3))``.

    Measured on a 3840x2160x3 float32 frame: the reshape-and-mean takes **0.438 s**, this
    takes **0.060 s** - 7.3x faster - and the two results are BIT-IDENTICAL (max absolute
    difference 0.0). The reshape version has to reduce over two non-contiguous axes, which
    defeats numpy's fast paths; the slice version is `s*s` contiguous adds. At two
    downsamples per frame this alone was three quarters of a second of every 1080p frame.
    """
    if s == 1:
        return a
    out = a[0::s, 0::s].copy()
    for i in range(s):
        for j in range(s):
            if i or j:
                out += a[i::s, j::s]
    out *= 1.0 / (s * s)
    return out


def _resize_f32(a, w, h):
    """Bilinear resize of a float32 (H, W) or (H, W, 3) array."""
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a, mode="F").resize((w, h), Image.BILINEAR),
                          np.float32)
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(a[..., i]), mode="F")
                        .resize((w, h), Image.BILINEAR), np.float32) for i in range(a.shape[2])]
    return np.stack(chans, axis=-1)


def blur(a, sigma):
    """Gaussian blur, sigma in PIXELS - downsampled when the kernel is wide.

    A wide blur at full resolution is the single most expensive operation in this file.
    Measured: a sigma=52 px blur over a 3840x2160x3 buffer took **11.8 seconds**, which
    made the whole post chain slower than the simulation it was decorating.

    The fix is the standard one, and it is essentially free of visual cost: a heavily
    blurred image has no high-frequency content left, so computing it at full resolution
    is arithmetic spent on detail that the blur itself is in the process of destroying.
    We box-downsample by `f`, blur with `sigma/f`, and bilinearly upsample. Cost falls by
    roughly `f**2` - a factor of 64 at f=8 - and the difference is not visible because
    the output is, by construction, smooth at the scale of `f` pixels.

    `f` is capped at 8 and chosen so the reduced sigma stays >= ~3 px, below which the
    downsampled blur would start to alias.
    """
    if sigma <= 0:
        return a
    if sigma <= 6:
        return _blur_direct(a, sigma)

    f = int(min(8, max(1, sigma // 3)))
    if f <= 1:
        return _blur_direct(a, sigma)

    h, w = a.shape[:2]
    sh, sw = max(1, h // f), max(1, w // f)
    small = _resize_f32(np.ascontiguousarray(a, np.float32), sw, sh)
    small = _blur_direct(small, sigma / float(f))
    return _resize_f32(small, w, h)


class Canvas:
    """One frame in progress. Create it, draw into it, call `to_rgb`, throw it away.

    Reusing a Canvas across frames is supported ONLY through `clear`, which keeps the
    trail buffer. That is the whole point of the trail buffer.
    """

    def __init__(self, cfg, bg=(0, 0, 0)):
        self.cfg = cfg
        self.rw, self.rh = cfg.render_size
        self.bg = np.asarray(bg, np.float32)
        self.base = np.zeros((self.rh, self.rw, 3), np.float32) + self.bg
        self.emit = np.zeros((self.rh, self.rw, 3), np.float32)
        #: Radial splats (film, nuclei, granules) accumulate here when CuPy is present.
        #: PIL-drawn primitives cannot live on the GPU, so the two are kept separate and
        #: summed in to_rgb. Splats are the bulk of the light and are pure array adds.
        self.emit_g = G.zeros((self.rh, self.rw, 3)) if G.GPU else None
        #: Persistence buffer for trails. Lives ACROSS frames; decayed per SUBSTEP by the
        #: sim (see `decay_trail`), never per frame - contract 2 in config.py.
        self.trail = None
        self._bg_image = None

    # --- lifecycle ---------------------------------------------------------------------
    def clear(self):
        """Reset the per-frame layers. Keeps `trail` and the background image."""
        if self._bg_image is not None:
            self.base = self._bg_image.copy()
        else:
            self.base = np.zeros((self.rh, self.rw, 3), np.float32) + self.bg
        self.emit[:] = 0.0
        if self.emit_g is not None:
            self.emit_g[:] = 0.0

    def set_background_colour(self, rgb):
        self.bg = np.asarray(rgb, np.float32)
        self._bg_image = None
        self.base[:] = self.bg

    def set_background_image(self, path_or_array, fit="cover"):
        """A still image behind everything. `fit` is 'cover' (fill, crop overflow) or
        'stretch'. Stored at render size so `clear` is a memcpy, not a resample."""
        if isinstance(path_or_array, str):
            im = Image.open(path_or_array).convert("RGB")
        else:
            im = Image.fromarray(np.asarray(path_or_array, np.uint8)).convert("RGB")
        if fit == "cover":
            s = max(self.rw / im.width, self.rh / im.height)
            im = im.resize((max(1, int(round(im.width * s))),
                            max(1, int(round(im.height * s)))), Image.LANCZOS)
            left = (im.width - self.rw) // 2
            top = (im.height - self.rh) // 2
            im = im.crop((left, top, left + self.rw, top + self.rh))
        else:
            im = im.resize((self.rw, self.rh), Image.LANCZOS)
        self._bg_image = np.asarray(im, np.float32)
        self.base = self._bg_image.copy()

    # --- trails ------------------------------------------------------------------------
    def ensure_trail(self):
        if self.trail is None:
            self.trail = np.zeros((self.rh, self.rw, 3), np.float32)
        return self.trail

    def decay_trail(self, dt, tau):
        """Decay by a TIME CONSTANT. Call once per SUBSTEP with that substep's dt.

        `tau` is the time for the trail to fall to 1/e (37%) of its brightness. tau=0
        clears instantly, which is how "trails off" is expressed without a branch.
        """
        if self.trail is None:
            return
        if tau <= 0:
            self.trail[:] = 0.0
        else:
            self.trail *= float(np.exp(-dt / tau))

    # --- drawing -----------------------------------------------------------------------
    #
    # EVERY primitive stamps into its own BOUNDING BOX, never a full-frame mask.
    #
    # This is not a micro-optimisation, it is the difference between an interactive engine
    # and a slideshow. The Bio engine draws one segment per cell edge - several hundred per
    # frame - and a full-frame mask at a 1080p ss=2 tier is an 8.3 megapixel allocation.
    # Six hundred of those is 5 gigapixels of allocate-draw-composite-discard per frame,
    # and essentially all of it is on pixels the segment does not touch. Stamping a 40x40
    # box instead makes the same frame roughly three orders of magnitude cheaper.
    #
    # The rule for anything you add here: compute the bbox, clip it to the canvas, bail if
    # it is empty, and composite into the SLICE.

    def _draw_target(self, layer):
        return {"base": self.base, "emit": self.emit,
                "trail": self.ensure_trail()}[layer]

    def _stamp(self, bbox, draw_fn, colour, layer, additive, alpha=1.0):
        """Rasterise `draw_fn` inside `bbox` and composite it into `layer`.

        `bbox` is (x0, y0, x1, y1) in render pixels, exclusive at the far edge.
        `draw_fn(d, ox, oy)` receives an ImageDraw for a box-sized mask plus the offset to
        subtract from absolute coordinates.
        """
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(np.floor(x0)))
        y0 = max(0, int(np.floor(y0)))
        x1 = min(self.rw, int(np.ceil(x1)))
        y1 = min(self.rh, int(np.ceil(y1)))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        im = Image.new("L", (w, h), 0)
        draw_fn(ImageDraw.Draw(im), x0, y0)
        m = np.asarray(im, np.float32) / 255.0
        if not m.any():
            return
        tgt = self._draw_target(layer)[y0:y1, x0:x1]
        c = np.asarray(colour, np.float32)
        if additive:
            tgt += m[..., None] * c * float(alpha)
        else:
            a = (m * float(alpha))[..., None]
            tgt *= (1.0 - a)
            tgt += a * c

    @staticmethod
    def _bbox_of(pts, pad):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        return (p[:, 0].min() - pad, p[:, 1].min() - pad,
                p[:, 0].max() + pad + 1, p[:, 1].max() + pad + 1)

    def polygon(self, pts_px, colour, layer="base", additive=None, alpha=1.0,
                outline_px=0.0, outline_colour=None):
        """Filled polygon. `pts_px` is an (N, 2) array of RENDER pixels."""
        additive = (layer != "base") if additive is None else additive
        p = np.asarray(pts_px, np.float64).reshape(-1, 2)
        if len(p) < 3:
            return
        ow = max(1, int(round(outline_px))) if outline_px else 0
        if len(p) >= 3:
            self._stamp(self._bbox_of(p, 1),
                        lambda d, ox, oy: d.polygon(
                            [(x - ox, y - oy) for x, y in p], fill=255),
                        colour, layer, additive, alpha)
        if ow and outline_colour is not None:
            self._stamp(self._bbox_of(p, ow + 1),
                        lambda d, ox, oy: d.line(
                            [(x - ox, y - oy) for x, y in p]
                            + [(p[0][0] - ox, p[0][1] - oy)],
                            fill=255, width=ow, joint="curve"),
                        outline_colour, layer, additive, alpha)

    def disc(self, cx, cy, r_px, colour, layer="base", additive=None, alpha=1.0,
             outline_px=0.0, outline_colour=None):
        additive = (layer != "base") if additive is None else additive
        r = float(r_px)
        ow = max(1, int(round(outline_px))) if outline_px else 0
        box = (cx - r - ow - 1, cy - r - ow - 1, cx + r + ow + 2, cy + r + ow + 2)
        self._stamp(box,
                    lambda d, ox, oy: d.ellipse(
                        [cx - r - ox, cy - r - oy, cx + r - ox, cy + r - oy], fill=255),
                    colour, layer, additive, alpha)
        if ow and outline_colour is not None:
            self._stamp(box,
                        lambda d, ox, oy: d.ellipse(
                            [cx - r - ox, cy - r - oy, cx + r - ox, cy + r - oy],
                            outline=255, width=ow),
                        outline_colour, layer, additive, alpha)

    def segment(self, x0, y0, x1, y1, width_px, colour, layer="emit", additive=None,
                alpha=1.0):
        """One line segment. Bio calls this once per cell edge with a per-edge colour."""
        additive = (layer != "base") if additive is None else additive
        w = max(1, int(round(width_px)))
        pts = ((x0, y0), (x1, y1))
        self._stamp(self._bbox_of(pts, w * 0.5 + 1),
                    lambda d, ox, oy: d.line(
                        [(x0 - ox, y0 - oy), (x1 - ox, y1 - oy)], fill=255, width=w),
                    colour, layer, additive, alpha)

    def polyline(self, pts_px, width_px, colour, layer="emit", additive=None, alpha=1.0,
                 closed=False):
        additive = (layer != "base") if additive is None else additive
        p = np.asarray(pts_px, np.float64).reshape(-1, 2)
        if len(p) < 2:
            return
        q = np.vstack([p, p[:1]]) if (closed and len(p) >= 3) else p
        w = max(1, int(round(width_px)))
        self._stamp(self._bbox_of(q, w * 0.5 + 1),
                    lambda d, ox, oy: d.line(
                        [(x - ox, y - oy) for x, y in q], fill=255, width=w,
                        joint="curve"),
                    colour, layer, additive, alpha)

    # --- radial splats -------------------------------------------------------------
    #
    # THE CELL FILM AND THE NUCLEI ARE DRAWN WITH THESE, AND THEY HAVE TO BE CHEAP.
    #
    # A colony can hold several hundred cells, each needing a smooth radial gradient, at
    # 3840x2160, sixty times a second. Rasterising a gradient per cell per frame is
    # hopeless, and clipping one to each cell's polygon is worse.
    #
    # The trick is that every cell wants the SAME function of r/R - only the radius,
    # colour and position differ. So the profile is rasterised ONCE per integer radius
    # and cached; drawing a cell is then a slice, a multiply and an add. Radii bucket to
    # integers, so a few dozen distinct images serve any number of cells.
    #
    # Profiles are named, not passed as arrays, so the cache key stays small.

    #: name -> callable(r) -> intensity, r in [0, 1] from centre to rim.
    PROFILES = {
        # A BUBBLE SEEN IN CROSS-POLARISED LIGHT. Dim through the middle, brightening
        # toward the rim where the curved surface is edge-on and the optical path through
        # the film is longest, then falling to nothing at the very edge. This is the shape
        # that reads as "a dome bulging toward the viewer" rather than "a flat disc".
        "shell": lambda r: (r ** 2.2) * np.exp(-((r - 0.86) / 0.30) ** 2) * 1.9
                           + 0.10 * np.exp(-(r / 0.55) ** 2),
        # Softer, fuller - reads as a thicker, more gelatinous cell.
        "dome": lambda r: np.clip(1.0 - r * r, 0.0, 1.0) ** 0.75,
        # A tight core with a soft halo - the nucleus.
        "core": lambda r: np.exp(-(r / 0.34) ** 2) + 0.22 * np.exp(-(r / 0.85) ** 2),
        # A clean ring, for accents.
        "ring": lambda r: np.exp(-((r - 0.80) / 0.16) ** 2),
        # BLOB: a soft lump with no edge, for organelles. `core` has a visible centre and
        # reads as a point; this reads as a piece of something.
        "blob": lambda r: np.exp(-(r / 0.62) ** 2) * (1.0 - 0.25 * r),
        # LENS: a BROAD bright band inside the wall over a dark centre - light taking the
        # long path through the cell and emerging concentrated near the boundary. `refract`
        # puts its peak at 0.94, which was right when the film was 4-6 discrete rings and
        # is a 3 px hairline now that the field resolves it exactly. This is the same idea
        # with a band you can actually see, and it stops just short of the wall so the
        # membrane keeps an edge of its own.
        "lens": lambda r: (r ** 1.9) * np.exp(-((r - 0.78) / 0.27) ** 2) * 1.55
                          + 0.05 * np.exp(-(r / 0.5) ** 2),
        # REFRACTIVE: the close-up profile. A narrow, bright band hard against the wall
        # and a nearly black interior - light taking the long path through the cell,
        # emerging concentrated at the boundary. Much less total flux than `shell`, but
        # far more of it where the eye reads structure.
        "refract": lambda r: (r ** 3.4) * np.exp(-((r - 0.94) / 0.13) ** 2) * 2.9
                             + 0.04 * np.exp(-(r / 0.45) ** 2),
    }

    _PROFILE_CACHE = {}

    @classmethod
    def profile_image(cls, name, r_px):
        """(2r+1, 2r+1) float32 image of a named radial profile. Cached per radius."""
        r_px = int(max(1, round(r_px)))
        key = (name, r_px)
        img = cls._PROFILE_CACHE.get(key)
        if img is not None:
            return img
        n = 2 * r_px + 1
        y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
        rr = np.sqrt(x * x + y * y) / float(r_px)
        f = cls.PROFILES[name]
        img = np.asarray(f(np.clip(rr, 0.0, 1.0)), np.float32)
        img[rr > 1.0] = 0.0
        # Guard the cache: a long render with hundreds of distinct radii would otherwise
        # grow it without bound. Dropping it wholesale is fine - it refills in a frame.
        if len(cls._PROFILE_CACHE) > 4096:
            cls._PROFILE_CACHE.clear()
        cls._PROFILE_CACHE[key] = img
        return img

    def radial(self, cx, cy, r_px, colour, profile="shell", gain=1.0, layer="emit"):
        """Add a cached radial profile, tinted by `colour`, centred on (cx, cy)."""
        r = int(max(1, round(r_px)))
        if gain <= 0:
            return
        img = self.profile_image(profile, r)
        x0 = int(round(cx)) - r
        y0 = int(round(cy)) - r
        n = img.shape[0]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(n - sx0, self.rw - dx0)
        h = min(n - sy0, self.rh - dy0)
        if w <= 0 or h <= 0:
            return
        c = np.asarray(colour, np.float32) * float(gain)
        if layer == "emit" and self.emit_g is not None:
            key = ("g", profile, r)
            gimg = self._PROFILE_CACHE.get(key)
            if gimg is None:
                gimg = G.to_gpu(img)
                if len(self._PROFILE_CACHE) > 4096:
                    self._PROFILE_CACHE.clear()
                self._PROFILE_CACHE[key] = gimg
            self.emit_g[dy0:dy0 + h, dx0:dx0 + w] += (gimg[sy0:sy0 + h, sx0:sx0 + w, None]
                                                      * G.to_gpu(c))
            return
        src = img[sy0:sy0 + h, sx0:sx0 + w, None]
        tgt = self._draw_target(layer)[dy0:dy0 + h, dx0:dx0 + w]
        tgt += src * c

    def radial_batch(self, cx, cy, r_px, cols, gains, profile="core"):
        """Many radial splats in one launch. See kit/splats.py."""
        from . import splats as SP
        tgt = self.emit_g if self.emit_g is not None else self.emit
        SP.draw(tgt, cx, cy, r_px, cols, gains, profile, self.PROFILES[profile],
                self.rw, self.rh)

    def cell_field(self, polys, sites, cols, R, ksoft, fgain, wgain, wwidth, fill,
                   profile="shell"):
        """The Bio film: one analytic pass per cell. See kit/cellfield.py.

        Goes into `emit_g` when CuPy is present, which is where the radial splats already
        live, so the whole additive layer stays on the card until to_rgb.
        """
        from . import cellfield as CF
        tgt = self.emit_g if self.emit_g is not None else self.emit
        CF.draw(tgt, polys, sites, cols, R, ksoft, fgain, wgain, wwidth, fill,
                profile, self.PROFILES[profile], self.rw, self.rh)

    def sprite(self, img_rgba, cx, cy, w_px, h_px, angle_deg=0.0, alpha=1.0,
               layer="base"):
        """Paste an RGBA sprite centred on (cx, cy), rotated. Alpha-composited into
        `base` (or added into `emit` if you want a glowing sprite)."""
        w_px = max(1, int(round(w_px)))
        h_px = max(1, int(round(h_px)))
        im = img_rgba.resize((w_px, h_px), Image.LANCZOS)
        if angle_deg:
            im = im.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
        a = np.asarray(im, np.float32) / 255.0
        sh, sw = a.shape[:2]
        x0 = int(round(cx - sw / 2.0))
        y0 = int(round(cy - sh / 2.0))
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        cw = min(sw - sx0, self.rw - dx0)
        ch = min(sh - sy0, self.rh - dy0)
        if cw <= 0 or ch <= 0:
            return
        src = a[sy0:sy0 + ch, sx0:sx0 + cw]
        tgt = self._draw_target(layer)
        dst = tgt[dy0:dy0 + ch, dx0:dx0 + cw]
        av = src[..., 3:4] * float(alpha)
        if layer == "base":
            dst *= (1.0 - av)
            dst += av * src[..., :3] * 255.0
        else:
            dst += av * src[..., :3] * 255.0

    # --- output ------------------------------------------------------------------------
    def to_rgb(self, post=None):
        """Composite, downsample, post-process -> (out_h, out_w, 3) uint8.

        GPU when CuPy is present (~17x faster at 4K).

        WRITTEN IN PLACE. One 4K frame buffer is 99.5 MB, and the obvious version allocates
        about ten of them, which overflows a shared GPU pool and kills the render mid-batch.
        Every step below either writes into an existing buffer or frees its temporary.
        """
        post = post or {}
        use_gpu = self.emit_g is not None
        xp = G.cp if use_gpu else np

        thr = float(post.get("bloom_threshold", 70.0))
        g1, gg1 = float(post.get("glow_sigma", 0.0)), float(post.get("glow_gain", 0.0))
        g2, gg2 = float(post.get("glow2_sigma", 0.0)), float(post.get("glow2_gain", 0.0))
        want1, want2 = (gg1 > 0 and g1 > 0), (gg2 > 0 and g2 > 0)

        if use_gpu:
            emit = G.to_gpu(self.emit)
            emit += self.emit_g
            img = G.to_gpu(self.base)
            if self.trail is not None:
                trail = G.to_gpu(self.trail)
                img += trail
                emit += trail
                del trail
        else:
            img = self.base
            emit = self.emit
            if self.trail is not None:
                img = img + self.trail
                emit = emit + self.trail

        hi = None
        if want1 or want2:
            hi = emit - thr
            xp.maximum(hi, 0.0, out=hi)

        img += emit
        if use_gpu:
            del emit

        s_ = self.cfg.ss
        h, w = self.cfg.height, self.cfg.width
        if s_ > 1:
            img = _box_down_xp(img, s_, xp)
            if hi is not None:
                hi = _box_down_xp(hi, s_, xp)

        px = self.cfg.scale
        if hi is not None:
            s1, s2 = g1 * px, g2 * px
            if use_gpu:
                if want1:
                    b = _blur_ds_xp(hi, s1, xp)
                    b *= gg1
                    img += b
                    del b
                if want2:
                    b = _blur_ds_xp(hi, s2, xp)
                    b *= gg2
                    img += b
                    del b
                del hi
            else:
                ref = s1 if want1 else s2
                f = int(min(8, max(1, ref // 3))) if ref > 6 else 1
                if f > 1:
                    sh, sw = max(1, h // f), max(1, w // f)
                    small = _resize_f32(hi, sw, sh)
                    acc = _blur_direct(small, max(0.6, s1 / f)) * gg1 if want1 else None
                    if want2:
                        b2 = _blur_direct(small, max(0.6, s2 / f)) * gg2
                        acc = b2 if acc is None else (acc + b2)
                    img = img + _resize_f32(acc, w, h)
                else:
                    if want1:
                        img = img + _blur_direct(hi, s1) * gg1
                    if want2:
                        img = img + _blur_direct(hi, s2) * gg2

        gamma = float(post.get("tone_gamma", 0.0))
        if gamma > 0:
            if use_gpu:
                xp.maximum(img, 0.0, out=img)
                img *= (1.0 / 255.0)
                xp.power(img, gamma, out=img)
                d = img + 1.0
                img /= d
                img *= 255.0
                del d
            else:
                img = _tone_lut_apply(img, gamma)

        vig = float(post.get("vignette", 0.0))
        if vig > 0:
            yy = xp.linspace(-1.0, 1.0, h, dtype=xp.float32)[:, None]
            xx = xp.linspace(-1.0, 1.0, w, dtype=xp.float32)[None, :]
            m = 1.0 - vig * xp.clip((xx * xx + yy * yy) * 0.5, 0.0, 1.0)
            if use_gpu:
                img *= m[..., None]
            else:
                img = img * m[..., None]
            del m

        grain = float(post.get("grain", 0.0))
        if grain > 0:
            gh, gw = max(1, h // 2), max(1, w // 2)
            if use_gpu:
                rs = xp.random.RandomState(int(post.get("grain_seed", 0)) % (2 ** 31))
                g = rs.normal(0.0, grain * 255.0, (gh, gw, 3)).astype(xp.float32)
                up = _upsample_xp(g, w, h, xp)
                img += up
                del g, up
            else:
                rng = np.random.default_rng(post.get("grain_seed", 0))
                g = rng.normal(0.0, grain * 255.0, (gh, gw, 3)).astype(np.float32)
                img = img + _resize_f32(g, w, h)

        xp.clip(img, 0.0, 255.0, out=img)
        out = G.to_cpu(img.astype(xp.uint8))
        if use_gpu:
            del img
        return out


def _box_down_xp(a, s, xp):
    if s == 1:
        return a
    out = a[0::s, 0::s].copy()
    for i in range(s):
        for j in range(s):
            if i or j:
                out += a[i::s, j::s]
    return out * (1.0 / (s * s))


def _upsample_xp(a, w, h, xp):
    """Nearest-neighbour upsample, GPU-friendly. Only used for grain, where the blockiness
    is indistinguishable from grain clump structure."""
    fy = int(np.ceil(h / a.shape[0]))
    fx = int(np.ceil(w / a.shape[1]))
    return xp.repeat(xp.repeat(a, fy, axis=0), fx, axis=1)[:h, :w]


def _blur_ds_xp(a, sigma, xp):
    """Blur, downsampling first when the kernel is wide. Same trick as the CPU path."""
    if sigma <= 6:
        return G.gaussian(a, sigma)
    f = int(min(8, max(1, sigma // 3)))
    h, w = a.shape[:2]
    sh, sw = max(1, h // f), max(1, w // f)
    small = a[:sh * f, :sw * f].reshape(sh, f, sw, f, 3).mean(axis=(1, 3))
    small = G.gaussian(small, sigma / f)
    up = xp.repeat(xp.repeat(small, f, axis=0), f, axis=1)
    if up.shape[0] < h or up.shape[1] < w:
        pad_y, pad_x = h - up.shape[0], w - up.shape[1]
        up = xp.pad(up, ((0, max(0, pad_y)), (0, max(0, pad_x)), (0, 0)), mode="edge")
    return up[:h, :w]
