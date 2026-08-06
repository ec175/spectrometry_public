"""atlas.py - the glyph stamps, baked once.

The whole format is a few thousand copies of about eight small shapes. So bake each shape ONCE
into an anti-aliased alpha stamp and composite copies of it additively, rather than asking
Pillow to draw 3000 primitives per frame. Measured: ~2 ms/frame of splatting against ~1400
ms/frame of per-glyph Pillow calls at 1080x1920 (CLAUDE.md S6).

Two things make this work and both are easy to get wrong:

1. SUB-PIXEL PHASE. The lattice pitch is 19.25 px - fractional on purpose, it is what was
   measured - so sites do not land on integer pixels. Rounding every stamp to the nearest
   pixel puts a +/-0.5 px jitter on a 5 px ring and the lattice reads as mush. Each stamp is
   therefore baked at PHASES x PHASES sub-pixel offsets and the splat picks the bucket from
   the fractional part of the position. This is the single highest-value 20 lines in the file.

2. STAMPS ARE ALPHA, NOT COLOUR. Colour and intensity are applied at splat time, so one ring
   stamp serves every hue in the palette.

Stamp geometry is quoted as a fraction of `pitch`, all measured off reel_51d5e4f3 rescaled to
the 1080-wide reference frame (CLAUDE.md S2).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SS = 8               # supersample factor used to bake a stamp (anti-aliasing quality)
PHASES = 4           # sub-pixel buckets per axis

# --- glyph geometry, in units of `pitch` --------------------------------------------------
DOT_R = 0.17
RING_R = 0.27
RING_W = 0.11
PAD_S = 0.26
LINK_W = 0.085
BOX_W = 0.055        # outlined-cell stroke (drawn as a polyline, not a stamp)


def _bake(size, draw_fn):
    """Render one shape at PHASES^2 sub-pixel offsets -> (PHASES*PHASES, size, size) f32.

    `draw_fn(d, cx, cy)` draws into an SS-times-oversampled L image with the shape centred on
    (cx, cy) in oversampled coordinates.
    """
    out = np.empty((PHASES * PHASES, size, size), np.float32)
    for py in range(PHASES):
        for px in range(PHASES):
            img = Image.new("L", (size * SS, size * SS), 0)
            d = ImageDraw.Draw(img)
            cx = (size * 0.5 + px / PHASES) * SS
            cy = (size * 0.5 + py / PHASES) * SS
            draw_fn(d, cx, cy)
            a = np.asarray(img, np.float32).reshape(size, SS, size, SS).mean((1, 3)) / 255.0
            out[py * PHASES + px] = a
    return out


def _size_for(extent):
    """Odd-ish box that fits `extent` world px plus a pixel of slack each side and one more
    for the sub-pixel offset, which only ever pushes +1 px."""
    return int(np.ceil(extent)) + 3


class Atlas:
    """Baked stamps for one (pitch, lattice) pair.

    node stamps: 'dot' (filled), 'ring' (hollow - the clip's most common glyph), 'pad'
                 (filled square, or filled hexagon on a hex lattice - the SMD-pad look)
    link stamps: one per entry of `lattice.link_specs`, so a 45-degree sqrt(2)-long diagonal
                 and a 0-degree pitch-long orthogonal are different stamps.
    """

    def __init__(self, pitch, link_specs, pad_shape="square"):
        self.pitch = float(pitch)
        p = self.pitch
        self.node = {}

        r = DOT_R * p
        s = _size_for(2 * r)
        self.node["dot"] = _bake(s, lambda d, cx, cy: d.ellipse(
            [cx - r * SS, cy - r * SS, cx + r * SS, cy + r * SS], fill=255))

        ro, wl = RING_R * p, RING_W * p
        s = _size_for(2 * ro + wl)
        self.node["ring"] = _bake(s, lambda d, cx, cy: d.ellipse(
            [cx - ro * SS, cy - ro * SS, cx + ro * SS, cy + ro * SS],
            outline=255, width=max(1, int(round(wl * SS)))))

        half = 0.5 * PAD_S * p
        s = _size_for(2 * half * (1.16 if pad_shape == "hex" else 1.0))
        if pad_shape == "hex":
            ang = np.deg2rad(90.0 + 60.0 * np.arange(6))
            rr = PAD_S * p * 0.62

            def _pad(d, cx, cy):
                d.polygon([(cx + rr * SS * np.cos(a), cy + rr * SS * np.sin(a))
                           for a in ang], fill=255)
        else:
            def _pad(d, cx, cy):
                d.rectangle([cx - half * SS, cy - half * SS,
                             cx + half * SS, cy + half * SS], fill=255)
        self.node["pad"] = _bake(s, _pad)

        # --- link stamps ------------------------------------------------------------------
        self.link = []
        self.link_vec = []
        lw = LINK_W * p
        for ang_deg, length in link_specs:
            a = np.deg2rad(ang_deg)
            dx, dy = np.cos(a) * length, np.sin(a) * length
            self.link_vec.append((dx, dy))
            s = _size_for(max(abs(dx), abs(dy)) + lw + 1)

            def _lk(d, cx, cy, dx=dx, dy=dy, lw=lw):
                d.line([cx - dx * SS / 2, cy - dy * SS / 2,
                        cx + dx * SS / 2, cy + dy * SS / 2],
                       fill=255, width=max(1, int(round(lw * SS))))
            self.link.append(_bake(s, _lk))

        self.stamp_bytes = sum(v.nbytes for v in self.node.values()) + \
            sum(v.nbytes for v in self.link)

    def __repr__(self):
        ns = {k: v.shape[-1] for k, v in self.node.items()}
        return (f"<Atlas pitch={self.pitch:.2f} nodes={ns} "
                f"links={[v.shape[-1] for v in self.link]} {self.stamp_bytes/1024:.0f} KiB>")


def phase_index(frac_x, frac_y):
    """Fractional position -> sub-pixel stamp bucket."""
    px = np.clip((frac_x * PHASES).astype(np.int32), 0, PHASES - 1)
    py = np.clip((frac_y * PHASES).astype(np.int32), 0, PHASES - 1)
    return py * PHASES + px
