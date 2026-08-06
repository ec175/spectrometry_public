"""config.py - render config + ffmpeg/NVENC resolution.

Ported from Shape_Physics\\sim\\config.py (itself ported from Oscilloscope\\osc\\config.py):
same find_ffmpeg chain, same one-time NVENC probe with automatic libx264 fallback. Env vars
here are FL_-prefixed (FL_FFMPEG / FL_NVENC).

WORLD UNITS: every scene is quoted in the REFERENCE frame 1080x1920 px, y DOWN, whatever the
output resolution is. `RenderConfig.scale` maps world -> output px. That is what makes a
540x960 preview and a 1080x1920 final the SAME picture, sampled differently - and it is why a
preview is trustworthy. Same discipline as the rest of the library.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LIB = os.path.normpath(os.path.join(ROOT, ".."))
OUT = os.path.join(ROOT, "out")

REF_W = 1080
REF_H = 1920


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60
    ss: int = 2                  # supersample factor for the vector draws (anti-aliasing)

    # --- the look ------------------------------------------------------------------
    # Two-scale additive bloom, measured off the source clip: a tight core halo plus a wide
    # atmosphere. The wide one is what lifts a black corner to ~(3,5,12) with no line near it.
    glow_sigma: float = 5.0
    glow_gain: float = 0.60
    glow2_sigma: float = 26.0
    glow2_gain: float = 0.42
    # DENSITY term: field lines are drawn by PIL, which OVERWRITES. Real accumulation where
    # hundreds of lines converge is supplied by splatting the integrator's own samples into a
    # histogram and blurring it. Cheap (one bincount) and it is the only additive channel.
    dens_sigma: float = 3.0
    dens_gain: float = 0.85
    dens_ref: float = 6.0        # samples per output px that counts as "fully lit"

    @property
    def scale(self) -> float:
        """world px -> output px."""
        return self.width / float(REF_W)

    @classmethod
    def preview(cls, **kw):
        kw.setdefault("width", REF_W // 2)
        kw.setdefault("height", REF_H // 2)
        kw.setdefault("fps", 30)
        return cls(**kw)


_NVENC_OK = None


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process. GPU encode is the library default, so a missing or
    busy NVENC must degrade to libx264 rather than kill a render mid-pipe. (Consumer NVENC
    allows only 3-5 concurrent sessions - that is why render_five.ps1 caps at 3 lanes.)"""
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
    """NVENC by default; FL_NVENC=0 forces libx264 CRF 16 (archival single renders)."""
    if os.environ.get("FL_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
                "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M"]
    return ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]


def find_ffmpeg() -> str:
    """FL_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("FL_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"
