"""pick_color.py — pull a vibrant, non-dark accent colour out of an album cover.

Downsamples the image, drops dark + washed-out pixels, then takes the dominant hue
among what's left and averages it -> a saturated colour that actually appears in the art.
Optional hue window (degrees) constrains the pick (e.g. greens for one cover).

Usage:
  python pick_color.py cover.png                 # -> #rrggbb
  python pick_color.py cover.png 70 170          # constrain hue to [70,170] deg (greens)
Run in the manim venv (Pillow is a manim dep).
"""
import colorsys
import sys

from PIL import Image


def pick(path, hue_lo=None, hue_hi=None):
    im = Image.open(path).convert("RGB").resize((120, 120))
    px = list(im.getdata())
    buckets = {}                                   # hue bin -> [count, r, g, b]
    for r, g, b in px:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v < 0.30 or s < 0.32:                   # skip darks + greys
            continue
        deg = h * 360.0
        if hue_lo is not None and not (hue_lo <= deg <= hue_hi):
            continue
        b_key = int(deg // 15)                      # 24 hue bins
        acc = buckets.setdefault(b_key, [0, 0, 0, 0])
        acc[0] += 1; acc[1] += r; acc[2] += g; acc[3] += b
    if not buckets:                                 # nothing vibrant (or no hue match) -> brightest pixel
        r, g, b = max(px, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        return "#%02x%02x%02x" % (r, g, b)
    n, r, g, b = max(buckets.values(), key=lambda a: a[0])
    r, g, b = r / n, g / n, b / n
    # lift saturation/brightness a touch so it reads as a glow colour on black
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    s = min(1.0, s * 1.15); v = max(v, 0.65)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    lo = float(args[1]) if len(args) > 1 else None
    hi = float(args[2]) if len(args) > 2 else None
    print(pick(path, lo, hi))
