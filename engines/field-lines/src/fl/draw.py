"""draw.py - the look. PIL vector strokes at SS x resolution, box-downsampled, plus a
two-scale additive bloom and an additive DENSITY term.

Ported from Shape_Physics\\sim\\draw.py, with one addition that this project needs and that one
does not. PIL line drawing OVERWRITES; the reference clip is clearly ADDITIVE - hundreds of
lines converging on a charge blow the core to white, and a frame corner 500 px from any line
still measures (3,5,12) rather than (0,0,0), which only a wide-sigma blur summed back in can
produce. Rather than pay for a real additive rasteriser, the integrator's own samples are
splatted into a scalar density histogram at OUTPUT resolution, blurred, and added. It is one
bincount, it is the only genuinely additive channel, and it is what makes a dense fan glow.

Bloom runs on the DOWNSAMPLED frame, never the supersampled buffer - the glow is low-frequency,
so it is visually identical and ~4x cheaper (Oscilloscope's half-res-bloom argument).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def mix(c1, c2, u):
    u = max(0.0, min(1.0, u))
    return (c1[0] + (c2[0] - c1[0]) * u,
            c1[1] + (c2[1] - c1[1]) * u,
            c1[2] + (c2[2] - c1[2]) * u)


def dim(c, k):
    return (int(max(0, min(255, c[0] * k))),
            int(max(0, min(255, c[1] * k))),
            int(max(0, min(255, c[2] * k))))


class Frame:
    """Supersampled canvas -> box downsample -> density + two-scale bloom."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.k = cfg.scale * cfg.ss          # world px -> supersampled px
        self.W, self.H = cfg.width * cfg.ss, cfg.height * cfg.ss
        self.img = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.dens = np.zeros(cfg.width * cfg.height, dtype=np.float32)
        self.dens_tint = (90, 170, 255)
        # Per-scene bloom exposure. A RADIAL fan and a PARALLEL bundle need different amounts
        # and there is no single right value: a fan leaves most of the frame black, so a wide
        # halo reads as a glow around a bright core, while an inlet rake covers a third of the
        # frame in lit pixels and the same halo integrates into a flat blue wash with no
        # contrast anywhere. Measured on the first stream_glass still - the wash was the bloom,
        # not the density term.
        self.glow_mul = 1.0
        self.glow2_mul = 1.0

    def clear(self):
        self.d.rectangle([0, 0, self.W, self.H], fill=(0, 0, 0))
        self.dens[:] = 0.0

    # ---- primitives (world px) ---------------------------------------------------
    def disc(self, cx, cy, r, color):
        k = self.k
        x, y, rr = cx * k, cy * k, r * k
        if rr < 0.35:
            return
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=color)

    def ring(self, cx, cy, r, width, color):
        k = self.k
        x, y, rr = cx * k, cy * k, r * k
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr],
                       outline=color, width=max(1, int(round(width * k))))

    def seg(self, x0, y0, x1, y1, width, color):
        k = self.k
        self.d.line([x0 * k, y0 * k, x1 * k, y1 * k], fill=color,
                    width=max(1, int(round(width * k))))

    def glyph(self, cx, cy, r, sign, color, width=None):
        """The + / - bar glyph, drawn as strokes rather than text so it scales exactly with the
        marker and never depends on a font being present."""
        w = width if width is not None else max(1.6, 0.26 * r)
        a = 0.58 * r
        self.seg(cx - a, cy, cx + a, cy, w, color)
        if sign > 0:
            self.seg(cx, cy - a, cx, cy + a, w, color)

    # ---- the field lines ---------------------------------------------------------
    def polylines(self, pts, length, colors, alphas, width=1.6, alpha_floor=0.02):
        """pts (N,S+1,2) world px, length (N,), colors (N,3) 0-255, alphas (N,)."""
        k = self.k
        w = max(1, int(round(width * k)))
        for i in range(len(length)):
            a = alphas[i]
            if a < alpha_floor or length[i] < 2:
                continue
            xy = (pts[i, :length[i]] * k).ravel().tolist()
            self.d.line(xy, fill=dim(colors[i], a), width=w)

    def strokes(self, pts, idx, k0, k1, color, gains, width=2.4):
        """Stroke slice [k0, k1] of selected lines - the pulse packets."""
        k = self.k
        w = max(1, int(round(width * k)))
        for j in range(len(idx)):
            xy = (pts[idx[j], k0[j]:k1[j] + 1] * k).ravel().tolist()
            if len(xy) >= 4:
                self.d.line(xy, fill=dim(color, gains[j]), width=w)

    def accumulate(self, pts, length, alphas, sub=3, gain=1.0):
        """Additive density: splat the integrator's samples (bilinear) at OUTPUT resolution.

        The polylines are sampled every `ds` world px, which is too sparse to look continuous -
        but this term is blurred by `dens_sigma` before use, so 2-3 px subsampling is ample.
        """
        cfg = self.cfg
        W, H, s = cfg.width, cfg.height, cfg.scale
        n, S1, _ = pts.shape
        kk = np.arange(S1)[None, :]
        valid = kk < length[:, None]
        if not valid.any():
            return
        # linear subsample along the arclength axis
        A = pts[:, :-1, :]
        B = pts[:, 1:, :]
        seg_ok = valid[:, 1:]
        w_line = np.repeat(alphas[:, None], S1 - 1, axis=1)[seg_ok] * gain
        A, B = A[seg_ok], B[seg_ok]
        if not len(A):
            return
        us = (np.arange(sub) / float(sub))[None, :, None]
        P = (A[:, None, :] + (B - A)[:, None, :] * us).reshape(-1, 2) * s
        wt = np.repeat(w_line, sub) / float(sub)

        x, y = P[:, 0], P[:, 1]
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - ix, y - iy
        for dx in (0, 1):
            for dy in (0, 1):
                gx, gy = ix + dx, iy + dy
                m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
                if not m.any():
                    continue
                ww = ((1.0 - fx) if dx == 0 else fx) * ((1.0 - fy) if dy == 0 else fy) * wt
                self.dens += np.bincount(gy[m] * W + gx[m], weights=ww[m],
                                         minlength=W * H).astype(np.float32)

    # ---- finish ------------------------------------------------------------------
    def to_rgb(self) -> np.ndarray:
        cfg = self.cfg
        small = self.img.resize((cfg.width, cfg.height), Image.BOX)
        out = np.asarray(small, dtype=np.float32)

        if cfg.dens_gain > 0 and self.dens.any():
            d = self.dens.reshape(cfg.height, cfg.width) / cfg.dens_ref
            di = Image.fromarray(np.clip(d * 255.0, 0, 255).astype(np.uint8), "L")
            db = np.asarray(di.filter(ImageFilter.GaussianBlur(cfg.dens_sigma)),
                            dtype=np.float32) / 255.0
            tint = np.array(self.dens_tint, dtype=np.float32)
            out = out + cfg.dens_gain * db[:, :, None] * tint[None, None, :]

        base8 = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        g1, g2 = cfg.glow_gain * self.glow_mul, cfg.glow2_gain * self.glow2_mul
        if g1 > 0:
            out = out + g1 * np.asarray(
                base8.filter(ImageFilter.GaussianBlur(cfg.glow_sigma)), dtype=np.float32)
        if g2 > 0:
            out = out + g2 * np.asarray(
                base8.filter(ImageFilter.GaussianBlur(cfg.glow2_sigma)), dtype=np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------------------
CYAN = (86, 196, 230)
BLUE = (44, 66, 214)
MAGENTA = (255, 109, 242)
WHITE = (255, 255, 255)
AMBER = (255, 186, 92)
MINT = (120, 255, 200)
ROSE = (255, 120, 160)


def palette_alternate(n, a=CYAN, b=BLUE):
    """The reference clip's scheme: adjacent seeds alternate between two blues. At high line
    counts the two families beat against each other and the fan breaks into a radial MOIRE -
    an emergent interference pattern nobody authored. It is the single cheapest source of
    visual interest in the whole format, and it is why this is a palette MODE and not a
    hard-coded pair."""
    c = np.empty((n, 3))
    c[0::2] = a
    c[1::2] = b
    return c


def palette_angle(n, stops=(CYAN, BLUE, MINT, BLUE)):
    """Cycle a colour ramp around the launch angle - reads as a rotating rainbow wheel."""
    u = np.linspace(0.0, len(stops), n, endpoint=False)
    i0 = np.floor(u).astype(int) % len(stops)
    i1 = (i0 + 1) % len(stops)
    f = (u - np.floor(u))[:, None]
    S = np.array(stops, dtype=float)
    return S[i0] * (1 - f) + S[i1] * f


def palette_fate(ended, captured=BLUE, escaped=CYAN, null=ROSE):
    """Colour a line by HOW IT ENDED. Only possible because the integrator reports it, and it
    makes the field's topology directly readable: which flux is bound and which gets away."""
    c = np.tile(np.array(escaped, dtype=float), (len(ended), 1))
    c[ended == 0] = captured
    c[ended == 1] = null
    return c
