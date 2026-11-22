# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""canvas.py — additive splat, persistence, tonemap, bloom. ALL ON THE GPU.

Nothing here is a draw call. A quarter-million particles per substep is far past what PIL
vector calls can do, so every particle is bilinearly SPLATTED into a float accumulator with a
scatter-add — the same primitive `..\\Attractors\\at\\draw.py`, `..\\Wind_Tunnel\\wt\\streaks.py`
and `..\\Field_Lines\\` all use, and for the same reason: it is genuinely ADDITIVE. A dense
filament is bright *because* many particles are there, and an overwriting rasteriser would draw
a uniform sheet instead.

⚠ **DIVERGENCE FROM Neon_Trace, ON PURPOSE.** That engine keeps its bloom on the HOST because
PIL's `GaussianBlur` is a three-pass BOX approximation and swapping it for the true Gaussian in
`cupyx` would change the glow of clips it had already shipped. This project has shipped
nothing, and the host bloom is ~80% of a Neon_Trace frame — the single biggest cost in it. So
the bloom is a true Gaussian, on the device, and the ONLY thing that crosses to the host is the
finished uint8 frame on its way to the ffmpeg pipe. Do not "restore" the PIL path for
consistency: the two projects are not colour-matched to each other and never were.

FOUR THINGS THAT HAVE TO BE RIGHT TOGETHER, and each is invisible alone
----------------------------------------------------------------------
1. **The point profile is one blur of the accumulator, not a sprite per particle.** Every
   particle has the same profile, blur is linear and the decay is a scalar, so
   `blur(sum of deposits) == sum of blurred deposits` exactly. One separable Gaussian replaces
   N gathers, and the picture is identical — not similar.
2. **Deposit scales with `scale^2`.** A bilinear splat lays down a fixed amount of flux
   regardless of resolution, so at 4K that flux is spread over four times the pixels and the
   final comes out a quarter as bright as the preview it was approved from. That reads as a
   grading fault and is really just pixel area.
3. **Decay is a TIME CONSTANT.** `acc *= exp(-dt/tau)` per substep, never a per-frame factor,
   or trail length becomes a function of frame rate.
4. **Tonemap before bloom, vignette after it.** The bloom must see the rolled-off image or it
   blooms the raw dynamic range and the whole frame goes white; the vignette is a lens
   property and must not be blurred back out (Neon_Trace's `_vig` argument, kept).
"""
from __future__ import annotations

import numpy as np

from . import gpu as _G
from .gpu import xp_of

# --------------------------------------------------------------------------------------
# palettes — cyan / teal / turquoise / white, with restrained secondary accents
# --------------------------------------------------------------------------------------
# Built dark-to-bright and MONOTONE IN LUMINANCE, so a viewer reads "more" from "brighter"
# with no legend. The dark end is not black: an additive buffer spends most of its area at
# very low values, and a ramp that starts at #000 renders that majority as literally nothing,
# which is how a particle field ends up looking sparse no matter how many particles are in it.
PALETTES = {
    # ⚠ RESTRAINT IS THE BRIEF, and it took two passes to actually obey it. "Cyan, teal,
    # turquoise, white, and OCCASIONALLY very subtle secondary hues."
    #
    # Pass 1 put each secondary hue at full saturation through the MIDTONES. A particle field
    # has nearly all of its area in the midtones, so `turbulence` rendered as a flat violet
    # carpet with no cyan anywhere. Pass 2 pulled the saturation down but left the hue in the
    # same place, and the frame came back royal BLUE — better, still not the brief.
    #
    # The lesson is about where a colour key actually lands, not about saturation. `key` is
    # speed-derived and most particles are slow, so the population piles up in the lower half
    # of the ramp: whatever colour lives at 0.2-0.6 IS the colour of the clip, regardless of
    # what the top of the ramp says. So every ramp below is TEAL from 0.25 to 0.65 without
    # exception, and each one differs only in the bottom 20% (little energy, few photons) and
    # the top 15% (few pixels). Read them side by side: they are one ramp, nudged four ways.
    # That is what a secondary hue is supposed to be.

    # the base look — deep teal through turquoise to a white-hot core
    "cyan":   [(0.00, (4, 18, 26)), (0.28, (8, 72, 92)), (0.58, (28, 158, 180)),
               (0.80, (130, 228, 238)), (1.00, (240, 255, 255))],
    # deep indigo SHADOW — blue only in the darkest quarter, teal by the time it matters
    "indigo": [(0.00, (7, 15, 38)), (0.24, (14, 66, 104)), (0.58, (30, 152, 184)),
               (0.80, (132, 226, 240)), (1.00, (238, 252, 255))],
    # pale gold HIGHLIGHT — the warm lift is the last 15% only, so tips read warm on a teal body
    "gold":   [(0.00, (4, 18, 26)), (0.28, (10, 76, 96)), (0.62, (44, 168, 184)),
               (0.86, (170, 216, 208)), (1.00, (255, 236, 186))],
    # violet — a cool lean in the HIGHLIGHTS, nothing more. Mids are the same teal as `cyan`.
    "violet": [(0.00, (8, 16, 32)), (0.28, (10, 70, 96)), (0.60, (34, 156, 186)),
               (0.84, (150, 214, 246)), (1.00, (236, 240, 255))],
}


def ramp(name, n=256):
    stops = PALETTES[name]
    ts = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    u = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(u, ts, cs[:, c]) for c in range(3)], axis=1).astype(np.float32)


class Canvas:
    def __init__(self, cfg, on_gpu=True):
        self.cfg = cfg
        self.on_gpu = bool(on_gpu) and _G.GPU
        xp = _G.cp if self.on_gpu else np
        self.xp = xp
        self.W, self.H = cfg.width, cfg.height
        # CHANNEL-PLANAR (3,H,W), like Neon_Trace's accumulator and for the same reason: the
        # scatter is the hot call, and an interleaved layout makes every channel a stride-3
        # write. Planar gives each channel a contiguous run that `reshape(-1)` views for free.
        self.acc = xp.zeros((3, self.H, self.W), np.float32)
        self.lut = xp.asarray(ramp(cfg.palette))
        self._vig = None
        #: reference px -> output px. Every length a scene quotes goes through this.
        self.s = cfg.scale
        #: see docstring point 2 — flux per unit REFERENCE area, held constant across tiers
        self.flux = np.float32(self.s * self.s)

    def clear(self):
        self.acc[:] = 0.0

    def decay(self, dt, tau=None):
        tau = self.cfg.trail_tau if tau is None else tau
        if tau <= 0:
            self.acc[:] = 0.0
        else:
            self.acc *= np.float32(np.exp(-dt / tau))

    # ---- the splat --------------------------------------------------------------------
    def deposit(self, sim, cam, gain=1.0, dt=None):
        """Bilinearly splat every particle, coloured by `sim.key`, weighted by `sim.gain`.

        Out-of-frame and behind-camera particles are handled by ZEROING THEIR WEIGHT and
        clamping their index, never by compacting the arrays. Compaction costs a device
        synchronisation and a variable-size allocation every substep; a zero weight costs one
        multiply and lets every buffer stay a fixed size for the whole clip.
        """
        xp = self.xp
        dt = sim.dt if dt is None else dt
        x, y, z = cam.project(sim.P)
        x = x * np.float32(self.s)
        y = y * np.float32(self.s)

        ok = (z > np.float32(1.0)) & (x > np.float32(-2)) & (x < np.float32(self.W + 1)) \
            & (y > np.float32(-2)) & (y < np.float32(self.H + 1))

        # Depth cue. The ONLY one that survives additive blending — there is no occlusion in
        # this project and there should not be (see camera.py). Near particles brighter, with
        # a floor so the far field does not vanish entirely and leave the frame hollow.
        d = z / np.float32(max(getattr(cam, "depth_ref", 1600.0), 1.0))
        wdepth = np.float32(0.45) + np.float32(0.55) / (np.float32(1.0) + d * d)

        w = (sim.gain * wdepth * xp.where(ok, np.float32(1.0), np.float32(0.0))
             * np.float32(gain * self.cfg.ink * dt * 60.0) * self.flux)

        col = self.lut[xp.clip((sim.key * np.float32(255.0)).astype(xp.int32), 0, 255)]

        ix = xp.clip(xp.floor(x).astype(xp.int32), 0, self.W - 2)
        iy = xp.clip(xp.floor(y).astype(xp.int32), 0, self.H - 2)
        fx = x - ix.astype(xp.float32)
        fy = y - iy.astype(xp.float32)

        base = iy * np.int32(self.W) + ix
        for dxi in (0, 1):
            wx = fx if dxi else (np.float32(1.0) - fx)
            for dyi in (0, 1):
                ww = w * wx * (fy if dyi else (np.float32(1.0) - fy))
                idx = base + np.int32(dyi * self.W + dxi)
                for c in range(3):
                    _G.scatter_add(self.acc[c].reshape(-1), idx, ww * col[:, c])

    # ---- finish ------------------------------------------------------------------------
    def _vignette(self):
        if self._vig is None:
            xp = self.xp
            y, x = xp.meshgrid(xp.arange(self.H, dtype=xp.float32),
                               xp.arange(self.W, dtype=xp.float32), indexing="ij")
            nx = (x - self.W * 0.5) / (self.W * 0.5)
            ny = (y - self.H * 0.5) / (self.H * 0.5)
            r = xp.sqrt(nx * nx + ny * ny) / np.float32(np.sqrt(2.0))
            self._vig = xp.clip(1.0 - self.cfg.vignette * (r ** np.float32(1.7)),
                                0.0, 1.0)[None, :, :]
        return self._vig

    def to_rgb(self, k=1.0) -> np.ndarray:
        """Composite and quantise. ALWAYS returns numpy — it feeds the ffmpeg pipe.

        `k` scales the accumulator on the way in, and exists for ONE caller:
        `tools/calibrate.py`. Deposit is exactly linear in `cfg.ink`, so the entire brightness
        curve of a scene can be evaluated from a SINGLE integration by re-tonemapping one
        captured accumulator at many `k` — which turns a 12-integration bisection per scene
        into one. Nothing on the render path passes it.
        """
        xp = self.xp
        cfg = self.cfg
        s = cfg.scale

        # 1. the point profile — one blur, see docstring point 1
        acc = self.acc if k == 1.0 else self.acc * np.float32(k)
        a = _G.gaussian_filter(acc, (0.0, cfg.point_sigma * s, cfg.point_sigma * s))

        # 2. Contrast, then the Reinhard roll-off. An additive swarm has a huge dynamic range
        #    and a hard clip turns every dense filament into the same flat white blob; the
        #    gamma ahead of it separates resolved cores from the faint overlap wash, which no
        #    amount of scaling can do (see config.tone_gamma).
        # ⚠ CLAMP BEFORE THE FRACTIONAL POWER. A Gaussian kernel is everywhere positive, so
        # blurring a non-negative accumulator is non-negative *mathematically* — but in float32
        # it returns values a few ulp BELOW zero, and `(-1e-9) ** 1.85` is NaN. NaN survives
        # `clip` untouched and casts to 0 on the way to uint8, so the defect renders as
        # perfectly black pixels: it looks exactly like background and is invisible in every
        # statistic that background is allowed to affect. Measured before the clamp: 14.6% of a
        # `vortex_core` frame was exact (0,0,0), including per-channel cases like (0,6,0) where
        # only two of the three channels had gone NaN. That asymmetry is what gave it away.
        a = xp.maximum(a, np.float32(0.0)) * np.float32(1.0 / 255.0)
        if cfg.tone_gamma != 1.0:
            a = a ** np.float32(cfg.tone_gamma)
        out = np.float32(255.0) * (a / (np.float32(1.0) + a))

        # 3. bloom — of the HIGHLIGHTS only. See config.bloom_threshold: blooming the whole
        #    frame sums a quarter-million faint tails into a haze that swamps the negative
        #    space, and it is not what veiling glare does optically either.
        g1, g2 = cfg.glow_gain, cfg.glow2_gain
        if g1 > 0 or g2 > 0:
            hi = xp.maximum(out - np.float32(cfg.bloom_threshold), np.float32(0.0))
            if g1 > 0:
                out = out + np.float32(g1) * _G.gaussian_filter(
                    hi, (0.0, cfg.glow_sigma * s, cfg.glow_sigma * s))
            if g2 > 0:
                out = out + np.float32(g2) * _G.gaussian_filter(
                    hi, (0.0, cfg.glow2_sigma * s, cfg.glow2_sigma * s))

        # 4. vignette (a lens property — after the bloom, never before)
        if cfg.vignette > 0:
            out = out * self._vignette()

        # 5. the near-black floor. Added last so it is not blooming or vignetting itself.
        bg = xp.asarray(np.asarray(cfg.bg, np.float32).reshape(3, 1, 1))
        out = xp.clip(out + bg, 0, 255).astype(xp.uint8)

        # Transpose planar->interleaved ON THE DEVICE. Identical values, but the host copy
        # costs ~50 ms at 3840x2160 against ~5 ms here, and it lands in the layout the pipe
        # wants with no second pass. (Neon_Trace's measurement, and it holds.)
        if self.on_gpu:
            return _G.asnumpy(xp.ascontiguousarray(xp.transpose(out, (1, 2, 0))))
        return np.ascontiguousarray(np.transpose(out, (1, 2, 0)))
