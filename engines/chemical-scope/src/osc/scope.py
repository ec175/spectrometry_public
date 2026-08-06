"""scope.py — the CRT phosphor screen: a persistence buffer + beam stroking + glow compose.

Beams are stroked into a *colour* persistence buffer (RGB float) which decays a little every
frame (P31-style afterglow), so traces can be green (the scope trace) OR red/blue (electron-
orbital wavefunction sign). The buffer is bloomed (blurred + added) for the glow. A separate
transient buffer holds the single bright *head* (redrawn every frame so it never smears into a
chain of beads); it can be faded out entirely.

Graticule has two states the scene crossfades between: cartesian grid -> axes only (a faint
centre cross). Coordinates: signals live in normalised [-1,1] scope space
(centre origin, +y up); `to_px` maps that onto the rectangular face (anisotropic: y stretched).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import (BEZEL, DIVISIONS, GRATICULE, GRATICULE_AXIS, RenderConfig)

from .gpu import asnumpy, gaussian_filter, xp


def _blur(a, sigma):
    """Backend-dispatched gaussian (CuPy on GPU when OSC_CUPY=1, scipy otherwise)."""
    s = (sigma, sigma, 0) if a.ndim == 3 else sigma
    return gaussian_filter(a, sigma=s)


class Scope:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self.w, self.h = cfg.width, cfg.height
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.Rx = 0.5 * cfg.face_w_frac * self.w        # half face width  (px per x-unit)
        self.Ry = 0.5 * cfg.face_h_frac * self.h        # half face height (px per y-unit) > Rx
        # persistence + head buffers and the graticules live on the GPU when OSC_CUPY=1
        # (all heavy per-frame math: decay, accumulate, bloom, compose). PIL rasterisation
        # of beam polylines stays CPU; each stroke uploads one (h,w) uint8 mask (~2 MB).
        self.phos = xp.zeros((self.h, self.w, 3), dtype=xp.float32)      # colour persistence
        self.head_buf = xp.zeros((self.h, self.w, 3), dtype=xp.float32)  # transient this-frame head
        # graticule states the scene crossfades between: cartesian(0) -> axes-only(1). (No radial.)
        self._grats = [xp.asarray(self._build_cartesian()), xp.asarray(self._build_axes())]

    # --- coordinate map: norm [-1,1] -> pixel (col,row). Anisotropic: y stretched to fill. ---
    def to_px(self, x, y):
        return self.cx + x * self.Rx, self.cy - y * self.Ry

    def _face_box(self):
        return (self.cx - self.Rx, self.cy - self.Ry, self.cx + self.Rx, self.cy + self.Ry)

    def _mask_face(self, arr):
        x0, y0, x1, y1 = self._face_box()
        m = np.zeros((self.h, self.w, 1), np.float32)
        m[int(y0):int(y1), int(x0):int(x1)] = 1.0
        return arr * m

    # --- graticule state 0: cartesian grid, square cells, NO ticks ---------------------------
    def _build_cartesian(self):
        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self._face_box()
        step = (2 * self.Rx) / DIVISIONS
        nx = DIVISIONS // 2
        for i in range(-nx, nx + 1):
            gx = self.cx + i * step
            col = GRATICULE_AXIS if i == 0 else GRATICULE
            d.line([(gx, y0), (gx, y1)], fill=col, width=2 if i == 0 else 1)
        ny = int(self.Ry / step)
        for j in range(-ny, ny + 1):
            gy = self.cy + j * step
            if y0 <= gy <= y1:
                col = GRATICULE_AXIS if j == 0 else GRATICULE
                d.line([(x0, gy), (x1, gy)], fill=col, width=2 if j == 0 else 1)
        d.rectangle([x0, y0, x1, y1], outline=BEZEL, width=3)
        return np.asarray(img).astype(np.float32)

    # --- graticule state 1: radial grid (rings + spokes) -------------------------------------
    def _build_polar(self):
        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self._face_box()
        step = (2 * self.Rx) / DIVISIONS
        rmax = np.hypot(self.Rx, self.Ry)
        k = 1
        while k * step <= rmax:
            r = k * step
            d.ellipse([self.cx - r, self.cy - r, self.cx + r, self.cy + r],
                      outline=GRATICULE, width=1)
            k += 1
        for deg in range(0, 360, 30):
            a = np.deg2rad(deg)
            col = GRATICULE_AXIS if deg % 90 == 0 else GRATICULE
            d.line([(self.cx, self.cy),
                    (self.cx + rmax * np.cos(a), self.cy + rmax * np.sin(a))], fill=col, width=1)
        d.rectangle([x0, y0, x1, y1], outline=BEZEL, width=3)
        return self._mask_face(np.asarray(img).astype(np.float32))

    # --- graticule state 2: axes only (faint centre cross) -----------------------------------
    def _build_axes(self):
        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self._face_box()
        d.line([(self.cx, y0), (self.cx, y1)], fill=GRATICULE_AXIS, width=2)   # vertical axis
        d.line([(x0, self.cy), (x1, self.cy)], fill=GRATICULE_AXIS, width=2)   # horizontal axis
        d.rectangle([x0, y0, x1, y1], outline=BEZEL, width=3)
        return np.asarray(img).astype(np.float32)

    def _graticule(self, state):
        if state < 0:                                    # sentinel: no graticule at all (pure black)
            return xp.zeros((self.h, self.w, 3), xp.float32)
        state = float(np.clip(state, 0, len(self._grats) - 1))
        lo = int(np.floor(state))
        if lo >= len(self._grats) - 1:
            return self._grats[-1]
        f = state - lo
        return self._grats[lo] * (1 - f) + self._grats[lo + 1] * f

    # --- per-frame API ------------------------------------------------------
    def new_frame(self):
        self.phos *= self.cfg.decay
        self.head_buf[:] = 0.0

    def beam(self, pts, color=None, gain=1.0, closed=False, speed_shade=False, width=None):
        """Stroke a polyline (pts = Nx2 in norm [-1,1]) into the colour persistence buffer.
        `color` = RGB 0..255 (defaults to the green scope phosphor). `speed_shade` brightens
        slow (dense) parts ∝ 1/length; otherwise the whole polyline is one fast uniform stroke."""
        if len(pts) < 2:
            return
        color = xp.asarray(color if color is not None else self.cfg.beam_color, dtype=xp.float32)
        px = np.empty((len(pts), 2), dtype=np.float32)
        px[:, 0], px[:, 1] = self.to_px(pts[:, 0], pts[:, 1])
        if closed:
            px = np.vstack([px, px[:1]])
        layer = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(layer)
        wd = width if width is not None else self.cfg.beam_width
        if speed_shade:
            seg = np.diff(px, axis=0)
            seglen = np.hypot(seg[:, 0], seg[:, 1]) + 1e-3
            inten = np.clip(220.0 * (np.median(seglen) / seglen), 40, 255)
            for i in range(len(seg)):
                d.line([tuple(px[i]), tuple(px[i + 1])], fill=int(inten[i]), width=wd)
        else:
            d.line([tuple(p) for p in px], fill=200, width=wd, joint="curve")
        inten = xp.asarray(np.asarray(layer)).astype(xp.float32) / 255.0
        self.phos += inten[:, :, None] * color * gain

    def beam_transient(self, pts, color=None, gain=1.0, width=None):
        """Stroke a polyline into the TRANSIENT buffer (head_buf) — cleared every frame, so it does
        NOT persist/accumulate. Use for brief flashed overlays that must vanish cleanly."""
        if len(pts) < 2:
            return
        color = xp.asarray(color if color is not None else self.cfg.head_color, dtype=xp.float32)
        px = np.empty((len(pts), 2), dtype=np.float32)
        px[:, 0], px[:, 1] = self.to_px(pts[:, 0], pts[:, 1])
        layer = Image.new("L", (self.w, self.h), 0)
        wd = width if width is not None else self.cfg.beam_width
        ImageDraw.Draw(layer).line([tuple(p) for p in px], fill=220, width=wd, joint="curve")
        inten = xp.asarray(np.asarray(layer)).astype(xp.float32) / 255.0
        self.head_buf += inten[:, :, None] * color * gain

    def head(self, pt, color=None, gain=1.0, r=None):
        """Draw the single bright head into the transient buffer (this frame only)."""
        color = xp.asarray(color if color is not None else self.cfg.head_color, dtype=xp.float32)
        r = r if r is not None else self.cfg.beam_width * 1.7
        px, py = self.to_px(pt[0], pt[1])
        layer = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(layer).ellipse([px - r, py - r, px + r, py + r], fill=255)
        inten = xp.asarray(np.asarray(layer)).astype(xp.float32) / 255.0
        self.head_buf += inten[:, :, None] * color * gain

    def _bloom(self, buf, sigma, gain):
        """A soft additive halo. Blur at HALF resolution (the glow is low-frequency, so this is
        visually identical) — ~4x cheaper, which matters a lot at full res over thousands of
        frames."""
        small = xp.clip(buf[::2, ::2], 0, 255)
        b = _blur(small, sigma / 2.0)
        up = xp.repeat(xp.repeat(b, 2, axis=0), 2, axis=1)[:buf.shape[0], :buf.shape[1]]
        return up * gain

    def render(self, grid_state=0.0) -> np.ndarray:
        """Compose: graticule (state 0..1 crossfaded: cartesian->axes) + coloured trail + its
        bloom, then (only if present) the head + its wider bloom. -> uint8 RGB."""
        cfg = self.cfg
        out = self._graticule(grid_state) + self.phos + self._bloom(self.phos, cfg.glow_sigma,
                                                                     cfg.glow_gain)
        if bool(self.head_buf.any()):                    # skip the (wasted) head bloom when no dot
            out = out + self.head_buf + self._bloom(self.head_buf, cfg.head_glow_sigma,
                                                    cfg.head_glow_gain)
        return asnumpy(xp.clip(out, 0, 255).astype(xp.uint8))
