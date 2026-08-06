"""config.py - render config + ffmpeg/NVENC resolution.

Ported from Field_Lines\\fl\\config.py (itself from Oscilloscope\\osc\\config.py): same
find_ffmpeg chain, same one-time NVENC probe with automatic libx264 fallback. Env vars here
are LG_-prefixed (LG_FFMPEG / LG_NVENC / LG_GPU).

WORLD UNITS: every scene is quoted in the REFERENCE frame 1080x1920 px, y DOWN, whatever the
output resolution is. `RenderConfig.scale` maps world -> output px, so a 540x960 preview and a
1080x1920 final are the SAME picture sampled differently. Library-wide discipline.

The one number this project is built around is `pitch`: the lattice spacing in WORLD px.
Measured off reel_51d5e4f3 as 12.83 px at 720 wide => 19.25 px at 1080 wide. See CLAUDE.md S2.
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

# Measured lattice pitch of the source clip, in the 1080-wide reference frame.
SOURCE_PITCH = 19.25


@dataclass
class RenderConfig:
    width: int = REF_W
    height: int = REF_H
    fps: int = 60

    # --- the look ---------------------------------------------------------------------
    # Two-scale additive bloom, same shape as Field_Lines: a tight core halo that turns a
    # dense cluster white, plus a wide atmosphere that lifts the whole frame off pure black.
    # Sigmas are quoted in WORLD px and scaled with the output, so a preview blooms the same.
    # The tight halo is what produces WHITE. A single palette swatch cannot tonemap to white
    # (its weakest channel saturates far too late), so every near-white pixel in the source is
    # neighbouring elements of different hue overlapping through their core halos. Measured:
    # at glow_gain 1.0 the render held 0.167 of bright pixels near-neutral against the
    # source's 0.321; the fix is halo strength, not exposure - raising exposure instead just
    # makes the frame brighter and MORE saturated.
    #
    # `glow_sigma` was re-measured directly off the source in the deep pass: the
    # background-subtracted halo around cores isolated in dark surroundings falls to 0.289 at
    # r=8 and 0.094 at r=16 (720-wide), both of which fit a gaussian of sigma ~6.6 px there =
    # 9.9 px at 1080. The first build's 4.6 was less than half the real halo.
    # 9.9 is the halo sigma measured directly off the source, but that measurement is
    # contaminated by the neighbouring elements it could not exclude, so it reads WIDE. The
    # honest constraint is downstream: the halo is what smears adjacent elements' hues
    # together, and at 9.9 it caps `hue_corr_1p` near 0.88 (source 0.765) while pushing
    # `white_of_bright` to 0.48 (source 0.321). Both move the right way when it narrows.
    glow_sigma: float = 7.4
    glow_gain: float = 1.05
    # The wide halo is the `dark_frac` control: it is what lifts a black corner off zero, and
    # the source is only 40.8% below luma 20 despite being mostly empty.
    glow2_sigma: float = 34.0
    # Raising this lifts the blacks (the `dark_frac` control) but the halo also smears
    # adjacent elements' hues, so it fights `hue_corr_2p`. 0.66 is where both sit inside
    # tolerance; there is no setting that centres both.
    glow2_gain: float = 0.66

    # ANAMORPHIC STREAK. The source's halo is not round: sampled at r=10..60 around 141 cores
    # isolated in dark surroundings, its angular profile peaks hard at 0/180 deg and troughs
    # at 90, max/min 1.62. That is a horizontal lens streak, and it is the single most
    # recognisable piece of the clip's "VFX" character. Not to be confused with lattice bias -
    # the lattice is symmetric in x and y, so it cannot make 0 deg brighter than 90.
    # Anisotropy is very nearly linear in the gain: 0.42 measured 3.84 against a round-halo
    # baseline of 1.0, so the source's 1.62 wants ~0.09. A streak you can obviously SEE is
    # already about four times too strong for this clip.
    streak_sigma_x: float = 110.0
    streak_sigma_y: float = 2.6
    streak_gain: float = 0.072

    # MIP-CHAIN GLOW. Two gaussians leave a visible seam between their two humps; a geometric
    # ladder of blurs with a decaying weight approximates the smooth power-law falloff real
    # bloom has. This is the "lighting doesn't flush well" fix.
    mip_levels: int = 4
    mip_gain: float = 0.34
    mip_falloff: float = 0.62

    # GRAIN. Measured on the source: high-frequency spatial residual std 11.45/255 = 0.045,
    # and temporal noise std in DARK regions 7.14/255 = 0.028 - i.e. it is real per-frame
    # noise, not a static overlay. Applied post-tonemap so it stays visible in the blacks
    # instead of being crushed by the exposure curve.
    grain: float = 0.044
    grain_chroma: float = 0.45      # how much of the grain is per-channel vs luma-only
    # How much grain survives in the BLACKS. 0.80 reproduces the source clip's measured
    # noise, which is barely attenuated there - correct for `circuit_*`, which is a
    # recreation. But per-frame noise in a black region IS "unilluminated parts blinking at
    # low intensity", so any composition that wants true blacks must drop this: the eye
    # scenes use ~0.10 and their unlit frame goes genuinely out.
    grain_floor: float = 0.80

    # The lift applied to an otherwise-black pixel. #060308 is the source clip's measured
    # stage colour and is right for `circuit_*`; but it is a FLOOR applied everywhere, so it
    # alone puts the unlit frame at mean luma 5.7. Anything wanting true black must lower it.
    bg: tuple = (0x06, 0x03, 0x08)

    exposure: float = 1.0

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


# ------------------------------------------------------------------------------------------
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
    """NVENC by default; LG_NVENC=0 forces libx264 CRF 14 (archival single renders).

    The bitrate ceiling is HIGH on purpose. This format's grain is per-frame white noise -
    incompressible by construction - and at the library's usual 40 Mbit/s cap a 15 s render
    pinned the ceiling exactly (76 MB) and the encoder started throwing detail away to fit.
    Measured on the encoded file against the source: grain_spatial 7.5 vs 9.8, grain_dark 1.4
    vs 2.8, and - the tell - neighbour-HF correlation 0.44 against 0.21, because what h.264
    leaves behind when it discards independent noise is CORRELATED blocking. The engine was
    producing the right picture and the encode was undoing it.
    """
    cq = os.environ.get("LG_CQ", "17")
    mx = os.environ.get("LG_MAXRATE", "70M")
    if os.environ.get("LG_NVENC", "1") != "0" and _nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", cq,
                "-b:v", "0", "-maxrate", mx, "-bufsize", "180M",
                "-rc-lookahead", "20", "-spatial-aq", "1", "-aq-strength", "12"]
    return ["-c:v", "libx264", "-crf", str(max(0, int(cq) - 3)), "-preset", "medium"]


def find_ffmpeg() -> str:
    """LG_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    env = os.environ.get("LG_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return "ffmpeg"
