# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""hq.py - the high-quality render path (Clancy, 2026-09-16: suggestions 1, 2, 4, 5).

Opt-in per scene with `hq=dict(...)`; `HQ` is None for every other scene, and then nothing
here runs. Four things, and each is a separate knob because each has its own cost:

1. `sub`   - MOTION BLUR by temporal supersampling. Each output frame is the mean of `sub`
             sub-frames spread evenly across half a frame interval, centred on the frame's own
             instant (a 180-degree shutter), so the audio alignment does not move.
             The engine is closed-form in t, so a sub-frame costs a frame and nothing else.
2. `fine`  - FINER DOT TEXTURE, only on tiers at or above `fine_min_scale` (the 4K master).
             The void lattice pitch and the point blur shrink TOGETHER by `fine`, because the
             legibility bound is their RATIO (void.py: pitch 4.4 vs sigma 2.2 is a flat wash).
             Deposit gains scale by `fine**2` so the frame's level does not move - measured,
             see MUSIC_PLAN S16.
4. `bits`  - 16-bit out of the canvas (the float image is quantised once, to 16 bits, not 8),
             piped as rgb48 into a 10-bit HEVC master. The violet-to-purple gradients are what
             an 8-bit path turns into visible bands.
5. `glow3` - a third, WIDE and FAINT bloom (sigma, gain, threshold) on top of pf's two, built
             at quarter resolution (a 4K sigma-180 blur at full size is the cost of a frame).

⚠ `pf` IS NOT MODIFIED. `HQCanvas` subclasses `pf.canvas.Canvas` and reproduces its
`to_rgb` step for step (point blur -> clamp -> gamma -> Reinhard -> two glows -> vignette ->
floor), adding the third glow after pf's two and before the vignette, and changing only the
final quantisation. Same rule as `orb/_pf.py`: the engine next door is imported, never edited.
"""
from __future__ import annotations

import numpy as np

from ._pf import Canvas, gpu as _G

HQ = None
DEFAULT = dict(sub=2, fine=0.72, fine_min_scale=1.9, bits=16,
               # glow3: 0.055 was invisible (+1% luma); 0.14 / thr 50 = +4%, blown <= 0.19%;
               # 0.20 / thr 45 started clipping (0.76%)
               glow3=(90.0, 0.14, 50.0))


def fine_active(cfg):
    return bool(HQ and HQ.get("fine", 1.0) != 1.0
                and cfg.scale >= float(HQ.get("fine_min_scale", 1.9)))


def fine_gain(cfg):
    """Deposit gain for the field and the mesh on the fine master.

    `fine**2` keeps an ISOLATED dot's peak where it was (count per area x 1/fine**2, blur area
    x fine**2). Measured against the regular 4K master downscaled to 1080: luma -3.1%, so the
    small `FIELD_TRIM` closes the gap.
    """
    return float(HQ["fine"]) ** 2 * FIELD_TRIM if fine_active(cfg) else 1.0


def core_gain(cfg):
    """The core is DENSE - overlapping dots, where a narrower blur preserves the mean and a
    `fine**2` cut only darkens it (the first fine frames showed a visibly dimmer heart). Its
    sparse arms are the exception, so it gets a partial cut, `fine**1`."""
    return float(HQ["fine"]) if fine_active(cfg) else 1.0


FIELD_TRIM = 1.03


class HQCanvas(Canvas):
    bits16 = False

    def to_rgb(self, k=1.0):
        xp = self.xp
        cfg = self.cfg
        s = cfg.scale
        f32 = np.float32
        acc = self.acc if k == 1.0 else self.acc * f32(k)
        a = _G.gaussian_filter(acc, (0.0, cfg.point_sigma * s, cfg.point_sigma * s))
        a = xp.maximum(a, f32(0.0)) * f32(1.0 / 255.0)
        if cfg.tone_gamma != 1.0:
            a = a ** f32(cfg.tone_gamma)
        out = f32(255.0) * (a / (f32(1.0) + a))
        base = out
        g1, g2 = cfg.glow_gain, cfg.glow2_gain
        if g1 > 0 or g2 > 0:
            hi = xp.maximum(base - f32(cfg.bloom_threshold), f32(0.0))
            if g1 > 0:
                out = out + f32(g1) * _G.gaussian_filter(
                    hi, (0.0, cfg.glow_sigma * s, cfg.glow_sigma * s))
            if g2 > 0:
                out = out + f32(g2) * _G.gaussian_filter(
                    hi, (0.0, cfg.glow2_sigma * s, cfg.glow2_sigma * s))
        g3 = (HQ or {}).get("glow3")
        if g3:
            sig, gain, thr = g3
            hi3 = xp.maximum(base - f32(thr), f32(0.0))
            C, H, W = hi3.shape
            q = 4
            Hq, Wq = H // q, W // q
            small = hi3[:, :Hq * q, :Wq * q].reshape(C, Hq, q, Wq, q).mean(axis=(2, 4))
            small = _G.gaussian_filter(small, (0.0, sig * s / q, sig * s / q))
            if self.on_gpu:
                from cupyx.scipy.ndimage import zoom
            else:
                from scipy.ndimage import zoom
            big = zoom(small, (1, H / Hq, W / Wq), order=1)[:, :H, :W]
            out = out + f32(gain) * big
        if cfg.vignette > 0:
            out = out * self._vignette()
        bg = xp.asarray(np.asarray(cfg.bg, np.float32).reshape(3, 1, 1))
        out = xp.clip(out + bg, 0, 255)
        if self.bits16:
            out = xp.rint(out * f32(257.0)).astype(xp.uint16)
        else:
            out = out.astype(xp.uint8)
        if self.on_gpu:
            return _G.asnumpy(xp.ascontiguousarray(xp.transpose(out, (1, 2, 0))))
        return np.ascontiguousarray(np.transpose(out, (1, 2, 0)))


def activate(spec):
    global HQ
    HQ = ({**DEFAULT, **spec} if isinstance(spec, dict)
          else (dict(DEFAULT) if spec else None))
    return HQ
