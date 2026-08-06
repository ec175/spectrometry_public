"""draw.py - the look. Pillow vector draws at SS x resolution, box-downsampled, plus an
additive two-scale bloom (tight halo + wide atmosphere).

Same immediate-mode idea as Ascii_Studio / Oscilloscope: no scene graph, no cache. Each frame
is drawn from the world state at time t and handed to ffmpeg as raw RGB24.

Bloom is done on the DOWNSAMPLED frame (the glow is low-frequency, so blurring at output res
is visually identical to blurring the supersampled buffer and ~4x cheaper - the same argument
as Oscilloscope's half-res bloom).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "Ascii_Studio", "assets", "fonts", "CascadiaMono.ttf"),
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\consola.ttf",
]
_font_cache: dict[int, object] = {}


def _font(px: int):
    """A bold monospace face for the multiplier labels. Falls back through the list and finally
    to None - a missing font must not stop a render, the pit just loses its text."""
    px = max(8, int(px))
    if px not in _font_cache:
        f = None
        for p in _FONT_PATHS:
            try:
                f = ImageFont.truetype(os.path.normpath(p), px)
                break
            except Exception:
                continue
        _font_cache[px] = f
    return _font_cache[px]

TAU = 2.0 * math.pi


def _mix(c1, c2, u):
    u = max(0.0, min(1.0, u))
    return (int(c1[0] + (c2[0] - c1[0]) * u),
            int(c1[1] + (c2[1] - c1[1]) * u),
            int(c1[2] + (c2[2] - c1[2]) * u))


def _dim(c, k):
    return (int(max(0, min(255, c[0] * k))),
            int(max(0, min(255, c[1] * k))),
            int(max(0, min(255, c[2] * k))))


@dataclass
class Style:
    pad_color: tuple = (255, 200, 90)          # the trigger markers on the outer gaps
    pad_idle: float = 0.42                     # brightness when armed but not firing
    zone_color: tuple = (120, 255, 190)        # scoring-zone outlines
    guide_gain: float = 0.0                    # brightness of the orbit-guide ellipses
    # multiplier pits are colour-coded by their factor, so the payoff is readable at a glance
    mult_colors: dict = field(default_factory=lambda: {
        1: (120, 255, 190), 2: (120, 215, 255), 3: (255, 200, 90),
        4: (255, 140, 210), 5: (200, 255, 110),
    })
    pad_flash: float = 0.28                    # seconds of flash after a trigger
    ball_palette: tuple = ((255, 255, 255), (130, 225, 255), (255, 175, 95),
                           (205, 150, 255), (150, 255, 190), (255, 130, 165))
    trail_len: int = 7
    trail_gain: float = 0.55
    flash_life: float = 0.38                   # seconds an exit shockwave lives
    spawn_life: float = 0.30
    ring_pulse: float = 0.22                   # extra ring brightness while audio is sounding
    ball_core: float = 0.55                    # white-hot core as a fraction of ball radius


class Frame:
    """Reusable supersampled canvas -> downsample -> bloom."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.k = cfg.scale * cfg.ss             # world px -> supersampled px
        self.W = cfg.width * cfg.ss
        self.H = cfg.height * cfg.ss
        self.img = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def clear(self):
        self.d.rectangle([0, 0, self.W, self.H], fill=(0, 0, 0))

    # ---- primitives (all arguments in WORLD px / radians)
    def arc(self, cx, cy, radius, a0, a1, width, color):
        k = self.k
        r = radius * k
        w = max(1, int(round(width * k)))
        x, y = cx * k, cy * k
        self.d.arc([x - r, y - r, x + r, y + r],
                   math.degrees(a0), math.degrees(a1), fill=color, width=w)
        # rounded caps (PIL arcs are butt-ended); also matches the physics' cap circles
        h = 0.5 * w
        for a in (a0, a1):
            px, py = x + r * math.cos(a), y + r * math.sin(a)
            self.d.ellipse([px - h, py - h, px + h, py + h], fill=color)

    def disc(self, cx, cy, radius, color):
        k = self.k
        x, y, r = cx * k, cy * k, radius * k
        if r < 0.4:
            return
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def ring_outline(self, cx, cy, radius, width, color):
        k = self.k
        x, y, r = cx * k, cy * k, radius * k
        w = max(1, int(round(width * k)))
        self.d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=w)

    def seg(self, x0, y0, x1, y1, width, color):
        k = self.k
        self.d.line([x0 * k, y0 * k, x1 * k, y1 * k],
                    fill=color, width=max(1, int(round(width * k))))

    def text_c(self, cx, cy, s, px, color):
        """Centred text. px is a WORLD size, so labels scale with the output resolution."""
        f = _font(px * self.k)
        if f is None:
            return
        k = self.k
        self.d.text((cx * k, cy * k), s, font=f, fill=color, anchor="mm")

    def capsule(self, x0, y0, x1, y1, thickness, color):
        """A thick segment with ROUND ends - matches the collider exactly (the collider is a
        segment swept by a disc), so what you see is what a ball hits."""
        self.seg(x0, y0, x1, y1, thickness, color)
        h = 0.5 * thickness
        self.disc(x0, y0, h, color)
        self.disc(x1, y1, h, color)

    # ---- finish
    def to_rgb(self) -> np.ndarray:
        cfg = self.cfg
        small = self.img.resize((cfg.width, cfg.height), Image.BOX)
        base = np.asarray(small, dtype=np.float32)
        out = base
        if cfg.glow_gain > 0:
            b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(cfg.glow_sigma)),
                            dtype=np.float32)
            out = out + cfg.glow_gain * b1
        if cfg.glow2_gain > 0:
            b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(cfg.glow2_sigma)),
                            dtype=np.float32)
            out = out + cfg.glow2_gain * b2
        return np.clip(out, 0, 255).astype(np.uint8)


def draw_world(fr: Frame, world, style: Style, t: float, score=None) -> None:
    """One frame, from world state at time t. Dispatches on obstacle kind, so a scene made of
    capsules or pegs draws with no scene-specific code."""
    fr.clear()
    cx, cy = world.cx, world.cy
    pulse = 1.0 + (style.ring_pulse if (score is not None and score.active(t)) else 0.0)
    outer = world._outer

    # --- scoring zones sit UNDER everything (they are targets, not objects). A multiplier pit
    # brightens for `pad_flash` after it is hit, and carries its own "Nx" label.
    for zi, z in enumerate(world.zones):
        if z.get("hidden"):
            continue                    # the trigger still fires; nothing is drawn
        mult = int(z.get("mult", 0))
        hot = 0.0
        for ev in world.triggers:
            if ev.get("kind") == "zone" and ev.get("zone") == zi:
                age = t - ev["t"]
                if 0.0 <= age < style.pad_flash:
                    hot = max(hot, 1.0 - age / style.pad_flash)
        base = style.mult_colors.get(mult, style.zone_color) if mult else style.zone_color
        col = _mix(_dim(base, 0.55 + 0.45 * hot), (255, 255, 255), 0.75 * hot)
        if mult:
            # NO surrounding circle (Ethan 2026-07-29) - the multiplier is the label alone.
            # The catch area is unchanged; only its outline is gone.
            fr.text_c(z["x"], z["y"], f"{mult}x", z["r"] * 0.72, col)
        else:
            fr.ring_outline(z["x"], z["y"], z["r"], 3.0 + 3.0 * hot, col)

    # --- orbit-guide ellipses, drawn very faint UNDER everything. Three staggered ellipses
    # read as a six-lobed petal rosette; they are a force field, not an object, so they must
    # never look solid enough to be mistaken for a wall. style.guide_gain = 0 hides them.
    if style.guide_gain > 0 and getattr(world, "guides", None):
        an = getattr(world, "aniso", 1.0)
        for gi, (A, B, phi) in enumerate(world.guides):
            col = _dim(style.ball_palette[gi % len(style.ball_palette)], style.guide_gain)
            pts = []
            for i in range(97):
                th = TAU * i / 96.0
                u, v = A * math.cos(th), B * math.sin(th)
                # tracks live in the stretched metric - undo it to draw them on screen
                pts.append((cx + u * math.cos(phi) - v * math.sin(phi),
                            cy + (u * math.sin(phi) + v * math.cos(phi)) * an))
            for i in range(96):
                fr.seg(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 2.0, col)

    # --- obstacles. Rings outermost-first so inner shells read as being in front.
    for obs in sorted(world.obstacles,
                      key=lambda o: -getattr(o, "radius", 0.0) if o.kind == "ring" else 1.0):
        if not obs.alive:
            continue
        if obs.kind == "ring":
            col = _dim(obs.color, pulse if obs is outer else 1.0)
            for a0, a1 in obs.solid_arcs(t):
                fr.arc(cx, cy, obs.radius, a0, a1, obs.thickness, col)
        elif obs.kind == "capsule":
            col = obs.color
            if obs.hp0 > 0:                     # a damaged brick reads dimmer
                col = _dim(col, 0.45 + 0.55 * max(0, obs.hp) / obs.hp0)
            ax, ay, bx, by = obs.ends(t)
            fr.capsule(ax, ay, bx, by, obs.thickness, col)
        elif obs.kind == "peg":
            px, py = obs.pos(t)
            fr.disc(px, py, obs.radius, obs.color)
            fr.disc(px, py, obs.radius * 0.42, (255, 255, 255))

    # --- the TRIGGERS: markers flanking each gap of the escape ring, on its OUTSIDE face, so
    # a ball only reaches one by getting all the way out. Skipped when the scene has no ring.
    fired = {}
    for ev in world.triggers:
        age = t - ev["t"]
        if 0.0 <= age < style.pad_flash:
            g = ev.get("gap", 0)
            fired[g] = min(fired.get(g, 1.0), age / style.pad_flash)
    if outer is not None and outer.n_gaps > 0:
        hw = 0.5 * outer.gap_width
        r0 = outer.radius + outer.h + 4.0
        r1 = r0 + 17.0
        for k in range(outer.n_gaps):
            c = outer.gap_center(k, t)
            u = fired.get(k)
            if u is None:
                col, w = _dim(style.pad_color, style.pad_idle), 6.0
            else:
                hot = 1.0 - u
                col, w = _mix(style.pad_color, (255, 255, 255), hot), 6.0 + 5.0 * hot
            for a in (c - hw, c + hw):
                ca, sa = math.cos(a), math.sin(a)
                fr.seg(cx + r0 * ca, cy + r0 * sa, cx + r1 * ca, cy + r1 * sa, w, col)

    # --- shockwaves: an expanding hoop where a ball punched out (or spawned)
    for f in world.flashes:
        age = t - f["t"]
        life = style.flash_life if f["kind"] == "trigger" else style.spawn_life
        if not (0.0 <= age < life):
            continue
        u = age / life
        rr = f["r"] * (1.0 + 5.5 * u)
        fade = (1.0 - u) ** 1.6
        base = style.pad_color if f["kind"] == "trigger" else (140, 220, 255)
        fr.ring_outline(f["x"], f["y"], rr, max(1.5, 5.0 * (1.0 - u)), _dim(base, fade))

    # --- balls (trail first, then body, then the hot core). Once the tiers separate in depth,
    # draw FAR planes first so a nearer sphere occludes a further one - the honest depth cue,
    # and the only one that does not fight the palette (colour and size already encode tier).
    pal = style.ball_palette
    balls = [b for b in world.balls if b.alive]
    if getattr(world, "plane_z", None) and world.plane_sep(t) > 0.0:
        balls.sort(key=lambda bb: world.ball_z(bb, t))
    for b in balls:
        col = pal[b.gen % len(pal)]
        n = len(b.trail)
        for i, (tx, ty) in enumerate(b.trail):
            u = (i + 1) / (n + 1)
            fr.disc(tx, ty, b.r * (0.30 + 0.55 * u), _dim(col, style.trail_gain * u * u))
        fr.disc(b.x, b.y, b.r, col)
        fr.disc(b.x, b.y, b.r * style.ball_core, (255, 255, 255))
