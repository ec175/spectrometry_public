# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""config.py — render config + ffmpeg/NVENC resolution.

Ported from `..\\Neon_Trace\\neon\\config.py` (itself from Field_Lines, itself from
Oscilloscope): same `find_ffmpeg` chain, same one-time NVENC probe with automatic libx264
fallback. Env vars here are PF_-prefixed (PF_FFMPEG / PF_NVENC / PF_CQ).

WORLD UNITS: every scene is quoted in the REFERENCE frame **1920x1080** px, whatever the
output resolution is. `RenderConfig.scale` maps world -> output px. This project is
horizontal-native for the same reason Neon_Trace is — it exists to make 16:9 stock footage.

⚠ THE FPS-INVARIANCE CONTRACT. A 1280x720@30 preview and a 3840x2160@60 final must be the
SAME animation sampled differently, or a preview is worthless and the whole tier structure is
a lie. For a *field* engine like Neon_Trace that is nearly free. For a particle engine with a
persistence buffer it is not, and it is bought in three places:

1. `sim.py` runs on a FIXED SUBSTEP CLOCK (`SUBSTEP_HZ = 120`), and 120 is divisible by both
   30 and 60, so both tiers sample one identical trajectory rather than integrating their own.
2. The trail decay is quoted as a TIME CONSTANT in seconds and applied per substep, never as
   a per-frame factor.
3. `canvas.py` scales splat gain by `scale**2` so a particle deposits the same flux per unit
   REFERENCE area at every resolution. Without it 4K is a quarter as bright as its own
   preview, which reads as "the finals came out wrong" and is really just pixel area.

Points 1 and 2 also make the sim independent of the render at all: `dry` at 5 probe frames
integrates exactly the same trajectory the full render does.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LIB = os.path.normpath(os.path.join(ROOT, ".."))
OUT = os.path.join(ROOT, "out")

REF_W = 1920
REF_H = 1080

#: The sim clock. Divisible by 24, 30 and 60 — every delivery frame rate lands exactly on a
#: substep, so no tier ever interpolates between two states the physics never occupied.
SUBSTEP_HZ = 120


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60

    ref_w: int = REF_W
    ref_h: int = REF_H

    # --- the look ------------------------------------------------------------------
    #: Radius of a single particle, in REFERENCE px, applied as one gaussian over the whole
    #: accumulator rather than per-particle. A per-particle sprite would be N gathers; this is
    #: one separable blur and gives an identical picture because every particle has the same
    #: profile.
    #:
    #: ⚠ Raised 1.10 -> 1.45 after looking at the first frames. At 1.10 reference px a particle
    #: is 0.73 px at the 720p preview tier — SUB-PIXEL — so a quarter million of them do not
    #: read as "fine luminous points" at all, they integrate into a smooth continuum, and no
    #: exposure or contrast setting recovers a point that was never resolved. A point has to be
    #: bigger than a pixel to look like a point.
    point_sigma: float = 1.45
    #: Trail persistence TIME CONSTANT, in seconds — never a per-frame decay factor. See the
    #: fps-invariance note above; this is the one that silently changes trail length with fps.
    #: 0 means no persistence at all (the wave/dust scenes, which are points, not paths).
    trail_tau: float = 0.10
    #: LUMINOUS POWER per particle, in accumulator units per second. THE global brightness
    #: knob, and the one number that is genuinely per-scene: a dense trail-heavy field lays
    #: down two orders of magnitude more flux per pixel than a sparse dust one, because every
    #: trail crossing a pixel adds to it. `tools\calibrate.py` solves for it per scene against
    #: a measured target rather than anyone guessing twenty times.
    ink: float = 1.0
    #: BLOOM THRESHOLD, 0-255. Only the part of the frame ABOVE this blooms:
    #: `bloom = blur(max(img - thresh, 0))`.
    #:
    #: ⚠ This is the fix for the biggest mistake made building this engine, and it is worth
    #: knowing why an unthresholded bloom fails so badly HERE specifically. Blooming the whole
    #: frame takes every faint particle's light and spreads it over ~2*pi*sigma^2 = 1800 px;
    #: with a quarter-million particles those tails SUM into a broad haze that lifts the
    #: entire frame above the lit threshold. Measured: cutting `turbulence` from 260k
    #: particles to 26k — a factor of TEN — moved lit_frac only 0.70 -> 0.53, because the lit
    #: area was never the particles. It was their bloom. A thresholded bloom is also simply
    #: what the real optical effect is: veiling glare comes off HIGHLIGHTS, not off everything.
    bloom_threshold: float = 70.0
    #: Two-scale additive bloom, in REFERENCE px. Tight halo + wide atmosphere.
    glow_sigma: float = 4.0
    glow_gain: float = 0.30
    glow2_sigma: float = 26.0
    glow2_gain: float = 0.16
    #: TONE GAMMA — the CONTRAST control, applied to the accumulator before the Reinhard
    #: shoulder: `out = 255 * a^g / (1 + a^g)`.
    #:
    #: ⚠ There is deliberately NO `exposure` field. An `exposure` multiplier and `ink` both
    #: scale the accumulator on the way into the same curve, so they are algebraically ONE
    #: knob under two names — exactly how a project ends up with a scene whose brightness is
    #: set in two places that disagree. `ink` is the calibrated one; this changes the SHAPE of
    #: the response rather than its scale, which is a genuinely different thing.
    #:
    #: Why a gamma is needed at all: a field of faint overlapping particles and a field of
    #: resolved bright points can have the same mean AND the same lit area. They differ in the
    #: SPREAD of the accumulator, and no scale factor changes a spread. `g > 1` crushes the low
    #: end (0.1 -> 0.025 at g=1.6) while leaving anything at or above 1.0 almost alone —
    #: "keep the cores, lose the wash". Swept, not guessed: `tools/gamma_sweep.py`.
    tone_gamma: float = 1.85
    vignette: float = 0.22
    #: Near-black, not black. A pure #000 background reads as a dead pixel region on OLED and
    #: crushes the bloom's own tail; #02060A keeps the frame alive without lifting the blacks.
    bg: tuple = (2, 6, 10)
    palette: str = "cyan"
    accent: str = ""
    accent_gain: float = 0.0

    @property
    def scale(self) -> float:
        """world (reference) px -> output px."""
        return self.width / float(self.ref_w)

    @classmethod
    def preview(cls, **kw):
        """1280x720@30 — two thirds linear. The review tier."""
        kw.setdefault("width", 1280)
        kw.setdefault("height", 720)
        kw.setdefault("fps", 30)
        return cls(**kw)

    @classmethod
    def hd(cls, **kw):
        kw.setdefault("width", 1920)
        kw.setdefault("height", 1080)
        kw.setdefault("fps", 60)
        return cls(**kw)

    @classmethod
    def final(cls, **kw):
        """3840x2160@60 — exactly 2x LINEAR on the reference frame.

        The integer scale is the point, and it is the same argument Neon_Trace makes: every
        length here is a reference length times `scale`, so 2x carries point radius, bloom
        radii and camera framing across untouched. 2560x1440 is 1.333x and lands them on
        sub-pixel boundaries. 4K is also the tier the stock agencies actually whitelist.
        """
        kw.setdefault("width", 3840)
        kw.setdefault("height", 2160)
        kw.setdefault("fps", 60)
        return cls(**kw)

    @classmethod
    def for_scene(cls, scene, tier="preview", **kw):
        maker = {"preview": cls.preview, "hd": cls.hd, "final": cls.final}[tier]
        rw, rh = getattr(scene, "REF", (REF_W, REF_H))
        kw.setdefault("ref_w", rw)
        kw.setdefault("ref_h", rh)
        cfg = maker(**kw)
        for k, v in getattr(scene, "LOOK", {}).items():
            setattr(cfg, k, v)
        return cfg


_NVENC_OK = None


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process. GPU encode is the library default, so a missing or
    busy NVENC must degrade to libx264 rather than kill a render mid-pipe."""
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            r = subprocess.run(
                [find_ffmpeg(), "-v", "error", "-f", "lavfi",
                 "-i", "color=black:size=256x256:rate=30:duration=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=25)
            _NVENC_OK = r.returncode == 0
        except Exception:
            _NVENC_OK = False
        if not _NVENC_OK:
            print("[config] h264_nvenc unavailable -> libx264 (CPU encode)")
    return _NVENC_OK


def encoder_args() -> list:
    """NVENC by default; PF_NVENC=0 forces libx264.

    The bitrate ceiling is deliberately far above what `cq` wants. Two projects in this
    library have already paid for a low one: Lattice_Grid lost half its measured grain to a
    40 Mbit/s cap, and Neon_Trace measured an 80M cap being HIT exactly at 4K, which means the
    encoder had stopped being quality-limited. Isolated bright points on near-black is
    *precisely* the content a starved encoder throws away first — it is all high-frequency
    energy with no neighbours to predict it from.
    """
    cq = os.environ.get("PF_CQ", "17")
    if os.environ.get("PF_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", cq,
                "-b:v", "0", "-maxrate", "250M", "-bufsize", "500M", "-g", "60"]
    return ["-c:v", "libx264", "-crf", str(max(0, int(cq) - 2)), "-preset", "medium",
            "-g", "60"]


def find_ffmpeg() -> str:
    """PF_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("PF_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    bundled = os.path.normpath(os.path.join(
        LIB, "Video Rendering", "Chemistry", "manim", "bin", "ffmpeg.exe"))
    if os.path.exists(bundled):
        return bundled
    return "ffmpeg"
