# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""config.py - render config, physical calibration, ffmpeg/NVENC resolution.

Ported from ..\Shape_Physics\sim\config.py (same find_ffmpeg chain, same one-time NVENC
probe with libx264 fallback); env vars are GL_-prefixed.

WORLD UNITS: the simulation always runs in the REFERENCE frame 1080x1920 px, y DOWN,
whatever the output resolution is. `RenderConfig.scale` maps world -> output px. That plus a
physics rate independent of fps is what makes a 540x960@30 preview and a 1080x1920@60 final
the SAME simulation, frame for frame.

THE CALIBRATION (read this before changing V_TH or N0)
-----------------------------------------------------
Mass m = 1 for every species, Boltzmann k = 1. A 2D Maxwellian at temperature T has
<v^2> = 2kT/m, so if V_TH is the most-probable speed at 300 K then

    T_K  =  300 * <v^2> / V_TH**2                 (gas.world.temperature_K)

and the component std at T is  sigma = V_TH/sqrt(2) * sqrt(T/300).

Ideal-gas pressure in 2D is P*A = N*k*T, and the wall momentum flux of a Maxwellian is
n*k*T, so both reduce to the same reference P0 = N0 * 300. One bar == P0. `--dry` prints
BOTH the ideal-gas pressure and the independently measured wall-impulse pressure; their
ratio is the check that the physics is real, not that the code ran.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
LIB = os.path.normpath(os.path.join(ROOT, ".."))
OUT = os.path.join(ROOT, "out")

# world / reference frame - ALL scene geometry is quoted in these px
REF_W = 1080
REF_H = 1920

# ---------------------------------------------------------------- gas calibration
# 2026-09-20: 3600 -> 9000. The clips now accelerate to 3x the base crank speed, and below a
# thermal/piston speed ratio of ~8 the ports choke. At 9000 the reference single sits at ~25
# at base speed and ~8 at 3x. Substeps go 1200 -> 3000 with it so a molecule still moves 3 px
# per substep. q_rxn values in the catalogue are quoted AT Q_REF_VTH and rescaled by the world
# (a reaction's energy is in px^2/s^2, so its temperature rise scales with 1/V_TH^2).
V_TH = 9000.0      # px/s: most-probable molecular speed at 300 K.  THE look/physics tradeoff.
Q_REF_VTH = 3600.0
N0 = 0.055         # particles per px^2 in the ambient reservoir (1 bar, 300 K)
T0 = 300.0         # K, reservoir temperature
T_REF_HOT = 2600.0 # K: the FIXED top of the colour ramp. Never normalise to the frame peak.

# species
AIR, FUEL, CO2, H2O = 0, 1, 2, 3
N2 = 4                 # 2026-10-02: inert nitrogen for the gasoline-in-AIR ledger set (AIR = O2 there)
SPECIES_NAMES = {AIR: "air", FUEL: "fuel", CO2: "CO2", H2O: "H2O", N2: "N2"}


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60
    ss: int = 2                    # supersample factor for the SOLIDS (vector strokes)
    substeps_per_sec: int = 3000    # PHYSICS rate, independent of fps (preview == final)
    coll_every: int = 12           # DSMC pass every N substeps (4 ms at 3000/s)
    # additive bloom on the particle buffer: tight core halo + wide atmosphere
    dot_sigma: float = 1.4
    glow_sigma: float = 5.0
    glow_gain: float = 0.34
    glow2_sigma: float = 20.0
    glow2_gain: float = 0.14

    @property
    def scale(self) -> float:
        """world px -> output px."""
        return self.width / float(REF_W)

    @classmethod
    def preview(cls, **kw):
        """720p/30, CONVENTIONS 13g: always rendered first."""
        kw.setdefault("width", 720)
        kw.setdefault("height", 1280)
        kw.setdefault("fps", 30)
        return cls(**kw)

    @classmethod
    def final(cls, **kw):
        """1440x2560 @60 - Clancy's 2026-09-20 ruling for this set: not 4K, the step above
        1080 that the platforms transcode at a higher bitrate."""
        kw.setdefault("width", 1440)
        kw.setdefault("height", 2560)
        kw.setdefault("fps", 60)
        return cls(**kw)


_NVENC_OK = None


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process. GPU encode is the default (library policy), so a
    missing or busy NVENC must degrade to libx264 rather than kill the render mid-pipe."""
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


def encoder_args(cfg=None) -> list:
    """NVENC by default; GL_NVENC=0 forces libx264 CRF 16. The bitrate ceiling scales with
    the pixel count so a 1440p final is not starved by a cap tuned for 1080p."""
    px = (cfg.width * cfg.height) if cfg is not None else (REF_W * REF_H)
    mx = int(round(40 * px / (REF_W * REF_H)))
    if os.environ.get("GL_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "17",
                "-b:v", "0", "-maxrate", f"{mx}M", "-bufsize", f"{2 * mx}M"]
    return ["-c:v", "libx264", "-crf", "15", "-preset", "medium"]


def find_ffmpeg() -> str:
    """GL_FFMPEG env -> local bin\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("GL_FFMPEG")
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
