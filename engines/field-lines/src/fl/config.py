# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""config.py - render config + ffmpeg/NVENC resolution.

Same find_ffmpeg chain and one-time NVENC probe pattern used elsewhere in this project: probe
h264_nvenc once per process and fall back to libx264 automatically rather than kill a render
mid-pipe. Env vars here are FL_-prefixed (FL_FFMPEG / FL_NVENC).

WORLD UNITS: every scene is quoted in the REFERENCE frame 1080x1920 px, y DOWN, whatever the
output resolution is. `RenderConfig.scale` maps world -> output px. That is what makes a
540x960 preview and a 1080x1920 final the SAME picture, sampled differently - and it is why a
preview is trustworthy.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out")

REF_W = 1080
REF_H = 1920


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60
    ss: int = 2                  # supersample factor for the vector draws (anti-aliasing)

    # The REFERENCE frame this scene's world coordinates were authored in. Vertical scenes
    # keep 1080x1920 and are unaffected. A scene that declares a 1920x1080 reference renders
    # at scale 1.0 instead of 1.78 - without this a 16:9 layout would draw 9:16 geometry blown
    # up nearly twice and hanging off the frame.
    ref_w: int = REF_W
    ref_h: int = REF_H

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
        return self.width / float(self.ref_w)

    @classmethod
    def preview(cls, **kw):
        """Half the SCENE's reference frame, not a hardcoded 9:16 half."""
        rw = kw.get("ref_w", REF_W)
        rh = kw.get("ref_h", REF_H)
        kw.setdefault("width", rw // 2)
        kw.setdefault("height", rh // 2)
        kw.setdefault("fps", 30)
        return cls(**kw)

    @classmethod
    def for_scene(cls, scene, **kw):
        """Build a config in the reference frame the scene declares.

        A scene opts into 16:9 with `REF = (1920, 1080)`; anything without the attribute keeps
        the original 1080x1920, so every existing vertical scene is untouched.
        """
        rw, rh = getattr(scene, "REF", (REF_W, REF_H))
        kw.setdefault("ref_w", rw)
        kw.setdefault("ref_h", rh)
        kw.setdefault("width", rw)
        kw.setdefault("height", rh)
        return cls(**kw)


_NVENC_OK = None


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process. GPU encode is the default, so a missing or busy
    NVENC must degrade to libx264 rather than kill a render mid-pipe. (Consumer NVENC allows
    only 3-5 concurrent sessions - cap a batch of renders at 3 lanes.)"""
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
    """FL_FFMPEG env -> local bin\\ffmpeg.exe -> PATH."""
    env = os.environ.get("FL_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"
