"""config.py - render config, song paths and ffmpeg/NVENC resolution.

Ported from Oscilloscope\\osc\\config.py (same find_ffmpeg chain, same one-time NVENC probe
with automatic libx264 fallback, same env-var opt-out names but SP_-prefixed).

WORLD UNITS: the simulation always runs in the REFERENCE frame 1080x1920 px, whatever the
output resolution is. `RenderConfig.scale` maps world -> output px. That is what makes a
540x960 preview and a 1080x1920 final the same physics, frame for frame.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LIB = os.path.normpath(os.path.join(ROOT, ".."))
OUT = os.path.join(ROOT, "out")

# world/reference frame - all scene geometry is quoted in these px
REF_W = 1080
REF_H = 1920

# Audio is NOT shipped with this repository. Drop your own files in songs\ (or
# pass an absolute path to --song) and add a key here to give it a short name.
SONGS_DIR = os.path.join(ROOT, "songs")
SONGS = {}


def resolve_song(name_or_path: str) -> str:
    """A key from SONGS, or a path straight through."""
    if not name_or_path:
        return ""
    key = name_or_path.lower().replace(" ", "").replace("_", "")
    if key in SONGS:
        return SONGS[key]
    return name_or_path


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60
    ss: int = 2                 # supersample factor for the vector draws (anti-aliasing)
    substeps_per_sec: int = 900  # PHYSICS rate, independent of fps (preview == final)
    # additive bloom: a tight core halo + a wide atmospheric one
    glow_sigma: float = 6.0
    glow_gain: float = 0.55
    glow2_sigma: float = 22.0
    glow2_gain: float = 0.38

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
    """Probe h264_nvenc ONCE per process. GPU encode is the default here (library policy), so
    a missing/busy NVENC must degrade to libx264 rather than kill the render mid-pipe."""
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            r = subprocess.run(
                [find_ffmpeg(), "-v", "error", "-f", "lavfi",
                 "-i", "color=black:size=256x256:rate=30:duration=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=20)
            _NVENC_OK = r.returncode == 0
        except Exception:
            _NVENC_OK = False
        if not _NVENC_OK:
            print("[config] h264_nvenc unavailable -> libx264 (CPU encode)")
    return _NVENC_OK


def encoder_args() -> list:
    """NVENC by default; SP_NVENC=0 forces libx264 CRF 16."""
    if os.environ.get("SP_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
                "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M"]
    return ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]


def find_ffmpeg() -> str:
    """SP_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("SP_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"
