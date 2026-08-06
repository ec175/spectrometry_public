"""draw.py - the compositor.

One float32 accumulation buffer, ADDITIVE. Additive is not a stylistic choice here: in the
source clip, wherever the content gets dense the colour runs to white (k-means found 32% of
bright pixels at saturation < 0.25, against a palette that is otherwise strongly saturated).
That is what accumulation does for free and what alpha compositing cannot do at all - PIL's
draw calls OVERWRITE, so a hot cluster would come out the colour of whichever glyph was drawn
last. Same lesson as Field_Lines' density channel, reached from the other direction.

The buffer is PADDED by the largest stamp on all four sides so a stamp can never fall partly
outside it. That removes every clipping branch from the splat - which matters, because the
splat is the whole render.

    splat -> [pad crop] -> two-scale bloom -> exposure tonemap -> uint8 RGB

BOTH halos are computed on a decimated copy and re-expanded by pixel replication. A halo has
no content above the Nyquist of its own decimation - that is what makes it a halo - so this
is not an approximation of the blur so much as a change of the grid it is computed on. The
SHARP image is added back at full resolution separately, so nothing the viewer resolves ever
passes through the decimation.

Measured at 1080x1920 (RTX 2060 / i7-9700K, single core):

    gaussian_filter at full res + scipy.zoom upsample     677 ms/frame
    decimated blurs + np.repeat upsample                   38 ms/frame

`scipy.ndimage.zoom` was the larger half of that: it fits a spline per axis, which is wasted
work on an image that is already smooth. Pixel replication of a blurred image is invisible
underneath the sharp layer.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .atlas import phase_index
from .field import BG


class Frame:
    def __init__(self, cfg, atlas):
        self.cfg = cfg
        self.atlas = atlas
        self.W, self.H = cfg.width, cfg.height
        self.pad = max([v.shape[-1] for v in atlas.node.values()]
                       + [v.shape[-1] for v in atlas.link]) + 2
        self.Wp = self.W + 2 * self.pad
        self.Hp = self.H + 2 * self.pad
        self.buf = np.zeros((self.Hp, self.Wp, 3), np.float32)
        self.splatted = 0

    def clear(self):
        self.buf[:] = 0.0
        self.splatted = 0

    # -- the splat -------------------------------------------------------------------------
    def splat(self, bank, pos, colors, inten, cut=0.012, disjoint=False):
        """Add `bank` stamps at world positions `pos`, tinted `colors`, scaled by `inten`.

        bank    (P*P, s, s) float32 alpha, sub-pixel phase-baked (atlas.py)
        pos     (K,2) float world px   -- stamp CENTRES
        colors  (K,3) float [0,1]
        inten   (K,)  float
        """
        if len(pos) == 0:
            return
        keep = inten > cut
        if not keep.any():
            return
        pos, colors, inten = pos[keep], colors[keep], inten[keep]

        s = bank.shape[-1]
        sc = self.cfg.scale
        x = pos[:, 0] * sc - 0.5 * s + self.pad
        y = pos[:, 1] * sc - 0.5 * s + self.pad
        ix = np.floor(x).astype(np.int64)
        iy = np.floor(y).astype(np.int64)
        ph = phase_index(x - ix, y - iy)

        # Cull anything whose whole stamp is outside the padded buffer.
        ok = (ix >= 0) & (iy >= 0) & (ix + s <= self.Wp) & (iy + s <= self.Hp)
        if not ok.all():
            ix, iy, ph = ix[ok], iy[ok], ph[ok]
            colors, inten = colors[ok], inten[ok]
            if not len(ix):
                return

        ar = np.arange(s, dtype=np.int64)
        rows = iy[:, None] + ar[None, :]                       # (K,s)
        cols = ix[:, None] + ar[None, :]                       # (K,s)
        flat = (rows[:, :, None] * self.Wp + cols[:, None, :]).ravel()

        a = bank[ph]                                           # (K,s,s)
        tint = (colors * inten[:, None]).astype(np.float32)    # (K,3)
        vals = a[:, :, :, None] * tint[:, None, None, :]       # (K,s,s,3)

        # `disjoint` says no two stamps in THIS call can touch the same pixel, which lets the
        # accumulation use buffered fancy indexing instead of the unbuffered np.add.at.
        # It is true for nodes and false for links:
        #   nodes - sites are >= pitch (19.25 px) apart and the largest node stamp is 16 px,
        #           and a scene never splats one site twice in one call (see content.blend),
        #           so node stamps in one bank provably cannot overlap.
        #   links - two collinear links sit 19.25 px apart under a 23 px stamp. They DO
        #           overlap, and plain `+=` would apply each duplicate index once, silently
        #           dropping exactly the overlaps that make dense clusters run white.
        # bincount was tried for the general case and is ~2.7x SLOWER than np.add.at here: it
        # allocates a whole-buffer float64 accumulator per channel per call, and a frame makes
        # ~20 calls.
        if disjoint:
            self.buf[rows[:, :, None], cols[:, None, :]] += vals
        else:
            np.add.at(self.buf.reshape(-1, 3), flat, vals.reshape(-1, 3))
        self.splatted += len(ix)

    def splat_nodes(self, sites, glyph_name, colors, inten):
        self.splat(self.atlas.node[glyph_name], sites, colors, inten, disjoint=True)

    def splat_links(self, mids, kind, colors, inten):
        """Links are batched per stamp kind: a 45-degree diagonal and a 0-degree orthogonal
        are different stamps and cannot share a splat call."""
        for k in range(len(self.atlas.link)):
            m = kind == k
            if m.any():
                self.splat(self.atlas.link[k], mids[m], colors[m], inten[m])

    # -- output ----------------------------------------------------------------------------
    @staticmethod
    def _half(a):
        """2x box decimation as two contiguous slice-adds.

        The obvious spelling - a.reshape(H//2, 2, W//2, 2, 3).mean((1, 3)) - is a strided
        reduction over a non-contiguous 5-D view and measured 117 ms at 1080x1920, four times
        the cost of the gaussian it was feeding. Slice-adds walk memory in order. Odd
        dimensions drop their last row/column rather than failing, so any --width works.
        """
        h, w = a.shape[0] & ~1, a.shape[1] & ~1
        a = a[:h, :w]
        a = a[0::2] + a[1::2]
        a = a[:, 0::2] + a[:, 1::2]
        return a * 0.25

    @staticmethod
    def _up2(a, shape):
        """Double `a` and soften, cropped/padded to `shape`. One rung of the mip cascade."""
        out = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        dh, dw = shape[0] - out.shape[0], shape[1] - out.shape[1]
        if dh > 0 or dw > 0:
            out = np.pad(out, ((0, max(dh, 0)), (0, max(dw, 0)), (0, 0)), mode="edge")
        return gaussian_filter(out[:shape[0], :shape[1]], (1.0, 1.0, 0), mode="nearest")

    def _expand(self, b, d):
        """Pixel-replicate back to full frame. The decimation chain drops odd rows/columns,
        so the expansion can land a few pixels short (540 -> 67 -> 536 at preview res); pad
        by edge replication rather than truncating, which would fail to broadcast."""
        out = np.repeat(np.repeat(b, d, axis=0), d, axis=1)
        dh, dw = self.H - out.shape[0], self.W - out.shape[1]
        if dh > 0 or dw > 0:
            out = np.pad(out, ((0, max(dh, 0)), (0, max(dw, 0)), (0, 0)), mode="edge")
        return out[:self.H, :self.W]

    def add_source(self, rgb):
        """Add the under-surface layer (already transmission-masked) to the buffer."""
        self.buf[self.pad:self.pad + self.H, self.pad:self.pad + self.W] += rgb

    def finish(self, rgb, frame_index=0):
        """Background lift + grain, applied AFTER the RGB-delay pass."""
        c = self.cfg
        bg = np.array(getattr(c, "bg", BG), np.float32) / 255.0
        out = bg + (1.0 - bg) * np.clip(rgb, 0.0, 1.0)
        if c.grain > 1e-5:
            g = np.random.default_rng(9973 + int(frame_index))
            n = g.standard_normal(out.shape[:2] + (1,)).astype(np.float32)
            if c.grain_chroma > 1e-5:
                n = n * (1.0 - c.grain_chroma) + c.grain_chroma * \
                    g.standard_normal(out.shape).astype(np.float32)
            f = c.grain_floor
            out += n * c.grain * (f + (1.0 - f) * np.sqrt(np.clip(out, 0, 1)))
        return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    def to_rgb(self, exposure=None, glow_mul=1.0, glow2_mul=1.0, streak_mul=1.0,
               frame_index=0, mip_gain=None, mono=False):
        c = self.cfg
        img = self.buf[self.pad:self.pad + self.H, self.pad:self.pad + self.W]
        sc = c.scale
        e = c.exposure if exposure is None else exposure

        # One decimation CHAIN feeds every halo: each stage starts from the previous stage's
        # output instead of touching the full-res buffer again.
        s2 = self._half(img)
        core = self._expand(gaussian_filter(
            s2, (max(0.5, c.glow_sigma * sc / 2),) * 2 + (0,), mode="nearest"), 2)
        s4 = self._half(s2)
        s8 = self._half(s4)
        wide = self._expand(gaussian_filter(
            s8, (max(0.5, c.glow2_sigma * sc / 8),) * 2 + (0,), mode="nearest"), 8)

        lit = img + (c.glow_gain * glow_mul) * core + (c.glow2_gain * glow2_mul) * wide

        # MIP-CHAIN GLOW. Two gaussians give two humps and a visible seam between them - the
        # "lighting doesn't flush well" complaint. A geometric ladder of blurs approximates the
        # smooth power-law falloff real bloom has, so light grades continuously from a core out
        # to the far field with no scale reading as a distinct ring.
        #
        # The upsample must CASCADE (double, blur, add the next-finer level, repeat) rather
        # than expanding each level straight to full size. Replicating a 4-px-wide level by
        # 128x paints 128 px blocks, and no amount of blurring afterwards removes them - the
        # first attempt at this put a visible rectangle around the whole subject. Doubling
        # with a small blur at every rung is both smooth and cheaper.
        g = c.mip_gain if mip_gain is None else mip_gain
        if g > 1e-4:
            chain = [s8]
            for _ in range(c.mip_levels):
                if min(chain[-1].shape[:2]) < 8:
                    break
                chain.append(self._half(chain[-1]))
            acc = gaussian_filter(chain[-1], (1.1, 1.1, 0), mode="nearest")
            for k in range(len(chain) - 2, -1, -1):
                acc = self._up2(acc, chain[k].shape)
                acc = acc + c.mip_falloff * gaussian_filter(
                    chain[k], (1.1, 1.1, 0), mode="nearest")
            lit += g * self._expand(acc, 8)

        # ANAMORPHIC STREAK: a very wide, very short horizontal gaussian. Computed on the
        # 1/4 chain, where a 110 px sigma costs 27 px - and where the 2.6 px vertical sigma
        # still resolves, which is why it is not taken from the 1/8 stage.
        if c.streak_gain * streak_mul > 1e-4:
            st = gaussian_filter(s4, (max(0.4, c.streak_sigma_y * sc / 4),
                                      max(0.5, c.streak_sigma_x * sc / 4), 0), mode="nearest")
            lit += (c.streak_gain * streak_mul) * self._expand(st, 4)

        np.maximum(lit, 0.0, out=lit)
        lit *= -e
        np.exp(lit, out=lit)                       # lit is now exp(-x*e), i.e. 1 - rgb
        if mono:
            # Single channel, pre-background, pre-grain: the black-and-white render that the
            # RGB-delay glitch pass consumes. Background and grain must come AFTER the delay,
            # or the background gets split into fringes and the grain becomes coloured.
            return (1.0 - lit).mean(2)
        bg = np.array(getattr(c, "bg", BG), np.float32) / 255.0
        rgb = bg + (1.0 - bg) * (1.0 - lit)

        # GRAIN, post-tonemap so it survives in the blacks. Seeded by FRAME INDEX, not by a
        # global RNG, so a still at t reproduces the render's frame at t exactly - the same
        # determinism rule the rest of the engine follows.
        if c.grain > 1e-5:
            g = np.random.default_rng(9973 + int(frame_index))
            n = g.standard_normal((self.H, self.W, 1)).astype(np.float32)
            if c.grain_chroma > 1e-5:
                n = n * (1.0 - c.grain_chroma) + c.grain_chroma * \
                    g.standard_normal((self.H, self.W, 3)).astype(np.float32)
            # Scaled by sqrt(luma) with a configurable floor. At the source-matching 0.80 the
            # grain is barely attenuated in the blacks (measured std 7.14 below luma 12) -
            # correct for a recreation, wrong for anything that wants true black, because
            # per-frame noise in a dark region is exactly "unilluminated parts blinking".
            f = c.grain_floor
            rgb += n * c.grain * (f + (1.0 - f) * np.sqrt(np.clip(rgb, 0, 1)))
        return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
