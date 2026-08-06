"""asciivid.py — video frame -> coloured ASCII, vectorized via a glyph atlas.

Ports the Ascii_Studio `ascii_art.frame_to_ascii` mapping (luminance -> ramp index, KEEP the
source pixel colour, drop near-black cells) but renders a whole frame at once with a
precomputed glyph atlas, so it's fast enough for full video instead of a per-cell blit loop:

    frame ──resize to (cols,rows)──> small RGB
          ──luminance ──> ramp-index grid (rows,cols)   (dark cells -> space)
          ──atlas[idx] ──> per-cell alpha masks (rows,cols,ch,cw)
          ──reshape ──> full-band alpha (band_h,band_w)
          ──× block-upsampled cell colour ──> coloured ASCII band
          ──place in a centred black frame ──> (out_h,out_w,3) uint8

The output frame is pure black except the ASCII band, so it feeds straight into
`crtfilm.FilmLook` (the Oscilloscope "screen" filter) — the CRT glass halation/lens/camera
model then makes it read as coloured ASCII glowing on a real CRT face.

Look matches the Ascii_Studio album icon (same RAMP, same keep-colour rule); the only change
is that a whole frame is assembled with numpy fancy-indexing rather than glyph-by-glyph.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# Luminance ramp (dark -> light) — the SAME one Ascii_Studio / the manim ASCII art use.
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

_WINFONTS = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
_FONT_CANDIDATES = [
    # bundled with the Ascii_Studio project, then Windows Terminal's Cascadia, then Consolas.
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "Ascii_Studio", "assets", "fonts", "CascadiaMono.ttf")),
    os.path.join(_WINFONTS, "CascadiaMono.ttf"),
    os.path.join(_WINFONTS, "CascadiaCode.ttf"),
    os.path.join(_WINFONTS, "consola.ttf"),
]


def _load_font(px: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


class AsciiAtlas:
    """Alpha masks (white glyph on black, 0..1) for every RAMP char at a fixed monospace cell.

    The cell box is the font's "M" advance x a leaded line height, so masks tile on a grid;
    RAMP[0] is a space -> an all-zero mask (dark cells render to nothing)."""

    def __init__(self, font_px: int = 18, leading: float = 1.35):
        font = _load_font(font_px)
        bbox = font.getbbox("M")
        cw = max(1, int(round(font.getlength("M"))))
        ch = max(1, int(round((bbox[3] - bbox[1]) * leading)))
        atlas = np.zeros((len(RAMP), ch, cw), np.float32)
        for k, c in enumerate(RAMP):
            if c == " ":
                continue
            im = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(im).text((0, -bbox[1]), c, fill=255, font=font)
            atlas[k] = np.asarray(im, np.float32) / 255.0
        self.atlas = atlas
        self.cw, self.ch, self.n = cw, ch, len(RAMP)


class AsciiVideo:
    """Turn each source video frame into a coloured-ASCII RGB frame on a black canvas.

    Subject-fit for a vertical short: the interesting action is in the MIDDLE THIRD of a wide
    source, so we take a centred `crop_frac` slice of the width (full height) as the ASCII
    source, and SQUEEZE it horizontally by `hsqueeze` (~0.8) — some centre objects clip past
    the third's edges, and squeezing pulls them in while making the slice stand tall enough to
    FILL the vertical screen. The squeeze is purely a DISPLAY-aspect change (same pixels, just
    narrower), so nothing is lost. The ASCII band is fit inside the frame minus a `margin`
    border (the "terminal window" gap). Colour is the source pixel colour (album-icon look).
    """

    _LUM = np.array([0.299, 0.587, 0.114], np.float32)

    def __init__(self, out_w: int, out_h: int, src_w: int, src_h: int, *,
                 font_px: int = 16, lum_floor: float = 14.0,
                 color_boost: float = 1.5, contrast: float = 1.12, bright: float = 1.15,
                 crop_frac: float = 1.0 / 3.0, hsqueeze: float = 0.8, margin: float = 0.04):
        self.out_w, self.out_h = out_w, out_h
        self.atlas = AsciiAtlas(font_px)
        cw, ch = self.atlas.cw, self.atlas.ch
        self._src_w, self._src_h = src_w, src_h
        crop_frac = min(max(crop_frac, 0.05), 1.0)
        self._crop_frac = crop_frac

        # DISPLAY aspect of the centre-cropped, horizontally-squeezed slice (w:h).
        region_aspect = crop_frac * src_w * hsqueeze / src_h
        avail_w, avail_h = out_w * (1 - margin), out_h * (1 - margin)
        if avail_w / avail_h > region_aspect:     # slice narrower than frame -> fill height
            band_h = avail_h
            band_w = band_h * region_aspect
        else:                                      # slice wider -> fill width
            band_w = avail_w
            band_h = band_w / region_aspect
        cols = max(1, int(band_w // cw))
        rows = max(1, int(band_h // ch))

        self.cols, self.rows = cols, rows
        self.cw, self.ch = cw, ch
        self.band_w, self.band_h = cols * cw, rows * ch
        self.x0 = (out_w - self.band_w) // 2
        self.y0 = (out_h - self.band_h) // 2
        self.lum_floor = lum_floor
        self.color_boost, self.contrast, self.bright = color_boost, contrast, bright

    # --- subject tracking -------------------------------------------------------------
    # `pan` is the crop centre as a fraction of source width (0.5 = the old fixed centre crop).
    # A FIXED centre slice is only right when the subject stays centred; on a real music video
    # the performer walks out of the middle third constantly and the ASCII then renders an empty
    # wall while the subject is off-frame. `ascii_video.py --track` measures where the action
    # actually is in a cheap pre-pass and drives this per frame.
    pan: float = 0.5

    def _prep(self, frame_rgb: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame_rgb)
        if self._crop_frac < 1.0:                  # slice of the width, full height
            cwpx = int(round(self._src_w * self._crop_frac))
            # clamp so the window always lies inside the source, whatever the tracker asks for
            x = int(round(self.pan * self._src_w - 0.5 * cwpx))
            x = max(0, min(self._src_w - cwpx, x))
            img = img.crop((x, 0, x + cwpx, self._src_h))
        return img.resize((self.cols, self.rows), Image.BOX)

    def render(self, frame_rgb: np.ndarray, pan: float | None = None) -> np.ndarray:
        """(src_h,src_w,3) uint8 -> (out_h,out_w,3) uint8 coloured-ASCII on black."""
        if pan is not None:
            self.pan = float(pan)
        small = self._prep(frame_rgb)
        small = ImageEnhance.Color(small).enhance(self.color_boost)
        small = ImageEnhance.Contrast(small).enhance(self.contrast)
        small = ImageEnhance.Brightness(small).enhance(self.bright)
        arr = np.asarray(small, np.float32)                     # (rows,cols,3)

        lum = arr @ self._LUM                                   # (rows,cols)
        idx = np.clip((lum / 255.0 * (self.atlas.n - 1)).astype(np.int32), 0, self.atlas.n - 1)
        idx[lum < self.lum_floor] = 0                           # dark cells -> space (empty)

        masks = self.atlas.atlas[idx]                           # (rows,cols,ch,cw)
        alpha = masks.transpose(0, 2, 1, 3).reshape(self.band_h, self.band_w)
        colour = np.repeat(np.repeat(arr, self.ch, axis=0), self.cw, axis=1)   # (band_h,band_w,3)
        band = alpha[..., None] * colour

        out = np.zeros((self.out_h, self.out_w, 3), np.float32)
        out[self.y0:self.y0 + self.band_h, self.x0:self.x0 + self.band_w] = band
        return np.clip(out, 0, 255).astype(np.uint8)
