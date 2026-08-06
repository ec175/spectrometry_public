"""config.py — render config, palette, and ffmpeg resolution for the scope engine.

Vertical 1080x1920 by default (these are shorts). The oscilloscope "face" is a square
graticule region centred in that tall frame; everything else is dark bezel.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out")


@dataclass
class RenderConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # Scope face: a RECTANGULAR graticule filling the tall 9:16 frame. x maps across the width,
    # y across the (taller) height -> vertical deflection is stretched to fill the screen, and the
    # beam can pass through the centre (X-Y deflection model). Grid cells stay square in pixels.
    face_w_frac: float = 0.95       # face width  as fraction of frame width
    face_h_frac: float = 0.93       # face height as fraction of frame height (Ry > Rx -> v-stretch)
    # Phosphor / CRT look
    decay: float = 0.86             # per-frame persistence multiplier (afterglow length)
    beam_width: int = 5             # px thickness of the electron beam core
    glow_sigma: float = 9.0         # bloom blur radius
    glow_gain: float = 0.9          # bloom strength
    # P31-ish green phosphor (bright core + halo)
    beam_color: tuple = (150, 255, 170)
    glow_color: tuple = (40, 230, 90)
    # the live beam head: hotter near-white core + a wider, stronger bloom
    head_color: tuple = (210, 255, 220)
    head_glow_sigma: float = 16.0
    head_glow_gain: float = 1.4


# electron-orbital wavefunction-sign colours (blue = +, red = -), per the standard convention
ORB_BLUE = (70, 130, 255)
ORB_RED = (255, 70, 70)


# graticule (grid) colour — faint green, static (kept subtle so the trace dominates)
GRATICULE = (9, 34, 16)
GRATICULE_AXIS = (14, 52, 24)       # slightly brighter centre cross / rings
BEZEL = (24, 74, 36)
DIVISIONS = 8                       # grid cells across the width


_NVENC_OK = None                        # cached one-time probe result (per process)


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process by encoding a tiny test clip to null. GPU encode is
    the default, so a missing/busy NVENC (driver update, >3 concurrent sessions on GeForce)
    must degrade to libx264 instead of killing the render mid-pipe."""
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
            print("[config] h264_nvenc unavailable -> falling back to libx264 (CPU encode)")
    return _NVENC_OK


def encoder_args() -> list:
    """Video-encode args. GPU encode (h264_nvenc on the RTX 2060) is the DEFAULT — it frees
    ~2-3 CPU threads per render and is required for the parallel worker lanes. Set
    OSC_NVENC=0 to force libx264 CRF 16 (e.g. for an archival-quality single render).
    Falls back to libx264 automatically if the NVENC probe fails."""
    if os.environ.get("OSC_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
                "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M"]
    return ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]


def find_ffmpeg() -> str:
    """OSC_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("OSC_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"   # assume on PATH
