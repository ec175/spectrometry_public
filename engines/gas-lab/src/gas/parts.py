# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""parts.py - the 2D car parts. Opaque, filled, cross-hatched where the metal is cut.

The 2026-09-15 solids were thin strokes over an additive gas, composited with `maximum`, so
bright gas shone through metal and nothing could hide anything. The 2026-09-20 pass draws the
cutaway the way the reference cutaways do: FILLED metal with a light edge, the sectioned
block and head cross-hatched at 45 degrees, moving parts smooth and two-toned, and an alpha
channel so a part can OCCLUDE gas (which is what lets a reservoir end hide under a manifold
flange instead of having to leave the frame).

Everything is drawn through a Group transform, so a V-twin bank or a radial cylinder is drawn
in its own upright local frame and lands rotated. Hatching stays at 45 degrees to the PAGE,
as in a real section drawing. Coordinates are world (reference) px; the canvas is at
output x SS resolution and downsampled once with a box filter.

The static layer (block, head, ducts, ring) is drawn ONCE and copied every frame; only the
moving parts are redrawn. That is the whole render-cost story for the solids.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import config as C
from . import engine as E

# ---------------------------------------------------------------- palette (dark theme)
BLOCK = (66, 74, 86)
BLOCK_HATCH = (96, 106, 121)
BLOCK_EDGE = (156, 168, 184)
HEAD = (72, 80, 93)
PISTON = (150, 158, 170)
PISTON_HI = (196, 204, 214)
PISTON_LO = (108, 116, 128)
RING = (34, 38, 44)
ROD = (164, 150, 124)
ROD_HI = (206, 194, 168)
ROD_LO = (118, 106, 84)
CRANK = (118, 126, 138)
CRANK_HI = (170, 178, 190)
CRANK_LO = (76, 82, 92)
PIN = (44, 48, 56)
PIN_RIM = (200, 206, 214)
VALVE = (204, 210, 220)
VALVE_LO = (140, 146, 156)
SPRING = (186, 192, 202)
CAM = (128, 136, 150)
CAM_HI = (186, 194, 206)
PLUG_HEX = (172, 178, 188)
PLUG_CERAMIC = (232, 228, 218)
PLUG_ELEC = (236, 200, 128)
INJECTOR = (160, 166, 176)
FIN = (86, 94, 106)
BOLT = (150, 156, 166)
BOLT_LO = (90, 96, 106)
DUCT_EDGE = (150, 162, 178)
RUBBER = (54, 58, 64)
SPARK = (215, 235, 255)


def _rot(px, py, ang, cx=0.0, cy=0.0):
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = px - cx, py - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


class Canvas:
    """A Pillow RGBA canvas at (output x SS) px, drawn in world px through a Group."""

    def __init__(self, W, H, scale, ss):
        self.W, self.H, self.ss = W, H, ss
        self.k = scale * ss
        self.img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.g = E.Group()
        self.zoom, self.vcx, self.vcy = 1.0, C.REF_W * 0.5, C.REF_H * 0.5

    def set_view(self, zoom=1.0, cx=None, cy=None):
        """Scale the whole picture by `zoom` about the world point (cx, cy), which lands at
        the centre of the frame. Geometry is untouched; this is the map to output px."""
        self.zoom = float(zoom)
        self.vcx = C.REF_W * 0.5 if cx is None else float(cx)
        self.vcy = C.REF_H * 0.5 if cy is None else float(cy)

    # ---- frames ----
    def set_group(self, g: E.Group):
        self.g = g

    def P(self, x, y):
        X, Y = self.g.to_world(x, y)
        X = (X - self.vcx) * self.zoom + C.REF_W * 0.5
        Y = (Y - self.vcy) * self.zoom + C.REF_H * 0.5
        return (X * self.k, Y * self.k)

    def px(self, v):
        """A length in world px -> canvas px."""
        return v * self.k * self.zoom

    def snapshot(self):
        return self.img.copy()

    def restore(self, img):
        self.img = img.copy()
        self.d = ImageDraw.Draw(self.img)

    def finish(self):
        """Downsample to output res. Returns (rgb float32 HxWx3, alpha float32 HxWx1)."""
        im = self.img.resize((self.W, self.H), Image.BOX)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return a[:, :, :3], a[:, :, 3:4]

    # ---- primitives ----
    def poly(self, pts, fill, outline=None, width=1.0):
        P = [self.P(x, y) for x, y in pts]
        if fill is not None:
            self.d.polygon(P, fill=fill + (255,) if len(fill) == 3 else fill)
        if outline is not None:
            self.d.line(P + [P[0]], fill=outline + (255,), width=max(1, int(round(self.px(width)))),
                        joint="curve")

    def rect(self, x0, y0, x1, y1, fill, outline=None, width=1.0):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill, outline, width)

    def cut(self, x0, y0, x1, y1):
        """Make a rectangle transparent (a gas passage through the metal)."""
        P = [self.P(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        self.d.polygon(P, fill=(0, 0, 0, 0))

    def cut_poly(self, pts):
        self.d.polygon([self.P(x, y) for x, y in pts], fill=(0, 0, 0, 0))

    def circle(self, cx, cy, r, fill, outline=None, width=1.0):
        X, Y = self.P(cx, cy)
        R = self.px(r)
        self.d.ellipse([X - R, Y - R, X + R, Y + R],
                       fill=fill + (255,) if fill is not None else None,
                       outline=outline + (255,) if outline is not None else None,
                       width=max(1, int(round(self.px(width)))))

    def line(self, p, q, fill, width=1.0):
        self.d.line([self.P(*p), self.P(*q)], fill=fill + (255,),
                    width=max(1, int(round(self.px(width)))))

    def polyline(self, pts, fill, width=1.0):
        self.d.line([self.P(x, y) for x, y in pts], fill=fill + (255,),
                    width=max(1, int(round(self.px(width)))), joint="curve")

    def hatched(self, pts, fill=BLOCK, hatch=BLOCK_HATCH, spacing=13.0, angle=45.0,
                lw=1.4, outline=BLOCK_EDGE, owidth=2.2):
        """A sectioned solid: filled, 45-degree hatch lines clipped to it, and a light edge."""
        P = [self.P(x, y) for x, y in pts]
        xs, ys = [p[0] for p in P], [p[1] for p in P]
        x0, y0 = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - 2
        x1, y1 = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in P], fill=255)
        layer = Image.new("RGBA", (w, h), fill + (255,))
        ld = ImageDraw.Draw(layer)
        sp = self.px(spacing)
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)          # hatch direction
        nx, ny = -dy, dx                           # step direction
        diag = math.hypot(w, h)
        n = int(diag / sp) + 2
        cx, cy = w * 0.5, h * 0.5
        lwp = max(1, int(round(self.px(lw))))
        for i in range(-n, n + 1):
            ox, oy = cx + nx * i * sp, cy + ny * i * sp
            ld.line([(ox - dx * diag, oy - dy * diag), (ox + dx * diag, oy + dy * diag)],
                    fill=hatch + (255,), width=lwp)
        self.img.paste(layer, (x0, y0), mask)
        if outline is not None:
            self.d.line(P + [P[0]], fill=outline + (255,),
                        width=max(1, int(round(self.px(owidth)))), joint="curve")

    def hatched_rect(self, x0, y0, x1, y1, **kw):
        self.hatched([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], **kw)

    def edge(self, x0, y0, x1, y1, col=DUCT_EDGE, width=1.6):
        """A light outline round a cut passage."""
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], None, col, width)

    # ---- compound parts ----
    def bolt(self, x, y, r=9.0, dark=False):
        self.circle(x, y, r, BOLT_LO if dark else BOLT, BOLT_LO, 1.0)
        self.line((x - r * 0.55, y), (x + r * 0.55, y), BOLT_LO if not dark else BOLT, 1.6)

    def fins(self, x_in, x_out, y_top, y_bot, n, depth_col=FIN, th=9.0):
        """Cooling fins on one side of a cylinder wall, from x_in (the wall) out to x_out."""
        for i in range(n):
            y = y_top + (y_bot - y_top) * (i + 0.5) / n
            self.rect(min(x_in, x_out), y - th * 0.5, max(x_in, x_out), y + th * 0.5, depth_col,
                      BLOCK_EDGE, 1.0)

    def piston(self, x0, x1, crown, height, pin_y, rings=3, dish=0.0, up=True):
        """A ringed piston: crown at `crown`, skirt to crown + height (or up if not `up`)."""
        s = 1.0 if up else -1.0
        top, bot = crown, crown + s * height
        self.rect(x0 + 3, min(top, bot), x1 - 3, max(top, bot), PISTON, PISTON_LO, 1.6)
        # crown band and a shaded skirt half
        self.rect(x0 + 3, min(top, top + s * 22), x1 - 3, max(top, top + s * 22), PISTON_HI)
        self.rect(x0 + 3, min(top + s * 0.62 * height, bot), x1 - 3,
                  max(top + s * 0.62 * height, bot), PISTON_LO)
        for i in range(rings):
            ry = top + s * (30 + i * 16)
            self.rect(x0 + 3, ry - 3, x1 - 3, ry + 3, RING)
        # wrist pin boss
        cx = 0.5 * (x0 + x1)
        self.circle(cx, pin_y, 26, PISTON_LO, PIN_RIM, 1.4)
        self.circle(cx, pin_y, 16, PIN, PIN_RIM, 1.2)
        self.circle(cx, pin_y, 7, PISTON_LO)

    def rod(self, small, big, w=34.0, big_r=30.0, small_r=18.0, bolts=True):
        """A con-rod: I-beam shank, small end, big end with a bolted cap."""
        (sx, sy), (bx, by) = small, big
        ang = math.atan2(by - sy, bx - sx)
        nx, ny = -math.sin(ang), math.cos(ang)
        L = math.hypot(bx - sx, by - sy)
        # shank tapers from the big end to the small end
        wb, ws = w, w * 0.72
        p = [(sx + nx * ws * 0.5, sy + ny * ws * 0.5), (bx + nx * wb * 0.5, by + ny * wb * 0.5),
             (bx - nx * wb * 0.5, by - ny * wb * 0.5), (sx - nx * ws * 0.5, sy - ny * ws * 0.5)]
        self.poly(p, ROD, ROD_LO, 1.4)
        # I-beam web: a lighter stripe down the middle
        p2 = [(sx + nx * ws * 0.17, sy + ny * ws * 0.17), (bx + nx * wb * 0.17, by + ny * wb * 0.17),
              (bx - nx * wb * 0.17, by - ny * wb * 0.17), (sx - nx * ws * 0.17, sy - ny * ws * 0.17)]
        self.poly(p2, ROD_HI)
        self.circle(sx, sy, small_r, ROD, ROD_LO, 1.4)
        self.circle(sx, sy, small_r * 0.5, PIN, PIN_RIM, 1.0)
        self.circle(bx, by, big_r, ROD, ROD_LO, 1.6)
        self.circle(bx, by, big_r * 0.62, PIN, PIN_RIM, 1.2)
        if bolts:
            for sgn in (-1, 1):
                self.bolt(bx + nx * sgn * (big_r + 6), by + ny * sgn * (big_r + 6), 7.0, True)

    def crank(self, C, R, theta, journal=36.0, web_w=64.0, counter=True):
        """A crank throw: main journal, web to the pin, a counterweight opposite the pin."""
        cx, cy = C
        px, py = cx + R * math.sin(theta), cy - R * math.cos(theta)
        ang = math.atan2(py - cy, px - cx)
        if counter:
            # counterweight: a fat lobe opposite the pin
            rw = max(0.7 * R, journal + 30.0)
            pts = []
            for i in range(25):
                a = ang + math.pi + math.radians(-58 + 116 * i / 24)
                pts.append((cx + rw * math.cos(a), cy + rw * math.sin(a)))
            pts.append((cx + 0.55 * journal * math.cos(ang + math.pi + math.radians(100)),
                        cy + 0.55 * journal * math.sin(ang + math.pi + math.radians(100))))
            pts.append((cx + 0.55 * journal * math.cos(ang + math.pi - math.radians(100)),
                        cy + 0.55 * journal * math.sin(ang + math.pi - math.radians(100))))
            self.poly(pts, CRANK, CRANK_LO, 1.6)
        nx, ny = -math.sin(ang), math.cos(ang)
        w = web_w * 0.5
        self.poly([(cx + nx * w, cy + ny * w), (px + nx * w * 0.8, py + ny * w * 0.8),
                   (px - nx * w * 0.8, py - ny * w * 0.8), (cx - nx * w, cy - ny * w)],
                  CRANK, CRANK_LO, 1.6)
        self.circle(cx, cy, journal, CRANK_HI, CRANK_LO, 1.6)
        self.circle(cx, cy, journal * 0.45, PIN, PIN_RIM, 1.2)
        return (px, py)

    def valve(self, cx, seat_y, half, lift_px, stem_top, tilt_deg=0.0, disc_h=14.0,
              stem_w=9.0, up=False):
        """A poppet valve: tulip head at the seat (+ lift), stem up to `stem_top`.
        `up`: the valve opens UPWARD (flathead)."""
        s = -1.0 if up else 1.0
        t = math.radians(tilt_deg)
        disc = seat_y + s * lift_px

        def T(x, y):
            return _rot(x, y, t, cx, seat_y)
        # tulip: wide face, chamfer up to the stem
        face = [T(cx - half, disc), T(cx + half, disc),
                T(cx + half, disc - s * disc_h), T(cx + half * 0.55, disc - s * (disc_h + 22)),
                T(cx + stem_w * 0.5, disc - s * (disc_h + 34)),
                T(cx - stem_w * 0.5, disc - s * (disc_h + 34)),
                T(cx - half * 0.55, disc - s * (disc_h + 22)), T(cx - half, disc - s * disc_h)]
        self.poly(face, VALVE, VALVE_LO, 1.2)
        # stem
        top = stem_top + s * lift_px * 0 + (s * lift_px)     # the whole valve moves with lift
        self.poly([T(cx - stem_w * 0.5, disc - s * (disc_h + 30)), T(cx + stem_w * 0.5, disc - s * (disc_h + 30)),
                   T(cx + stem_w * 0.5, top), T(cx - stem_w * 0.5, top)], VALVE, VALVE_LO, 1.0)
        return T, top

    def spring(self, p, q, coils=7, radius=22.0, width=3.4, col=SPRING):
        """A coil spring as a zigzag between p and q (any direction)."""
        (x0, y0), (x1, y1) = p, q
        ang = math.atan2(y1 - y0, x1 - x0)
        nx, ny = -math.sin(ang), math.cos(ang)
        n = coils * 2
        pts = [(x0, y0)]
        for i in range(1, n):
            u = i / n
            sgn = 1 if i % 2 else -1
            pts.append((x0 + (x1 - x0) * u + nx * radius * sgn, y0 + (y1 - y0) * u + ny * radius * sgn))
        pts.append((x1, y1))
        self.polyline(pts, col, width)
        # seats
        for (x, y) in (p, q):
            self.line((x - nx * radius, y - ny * radius), (x + nx * radius, y + ny * radius), col, width)

    def cam_lobe(self, c, base_r, lift_max, valve: E.Valve, t, contact_deg=90.0, n=90):
        """A cam lobe generated FROM the valve's lift curve, turning at half crank speed, with
        the profile point that corresponds to `now` sitting in the `contact_deg` direction
        (screen degrees, 90 = straight down). The follower there therefore rides exactly the
        lift the physics is using at this instant."""
        cx, cy = c
        now = valve.clock(t)
        cyc = valve.cycle
        rate = 360.0 / cyc                       # cam degrees per crank degree (0.5 or 1)
        pts = []
        for i in range(n):
            d = cyc * i / n
            r = base_r + lift_max * E.cam_lift(d, valve.opens, valve.closes, cyc)
            a = math.radians(contact_deg + (d - now) * rate)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        self.poly(pts, CAM, CAM_HI, 1.6)
        self.circle(cx, cy, base_r * 0.42, CAM_HI, CAM_HI, 1.0)
        self.circle(cx, cy, base_r * 0.16, PIN, PIN_RIM, 1.0)

    def plug(self, x, deck_y, up=True, flash=0.0):
        """A spark plug in the head: hex, ceramic, centre electrode below the deck, and a
        small spark when `flash` > 0 (a local event, never a frame-wide one)."""
        s = -1.0 if up else 1.0                  # up: the plug body points UP out of the deck
        self.poly([(x - 8, deck_y), (x + 8, deck_y), (x + 6, deck_y + s * -1 * 0)], None)
        # electrode + ground strap (inside the chamber)
        self.line((x, deck_y), (x, deck_y - s * 14), PLUG_ELEC, 3.0)
        self.line((x + 10, deck_y), (x + 10, deck_y - s * 16), PLUG_ELEC, 2.6)
        self.line((x + 10, deck_y - s * 16), (x + 1, deck_y - s * 16), PLUG_ELEC, 2.6)
        # threaded shell and hex in the head
        self.rect(x - 12, min(deck_y, deck_y + s * 40), x + 12, max(deck_y, deck_y + s * 40), PLUG_HEX, BLOCK_EDGE, 1.0)
        self.rect(x - 18, min(deck_y + s * 40, deck_y + s * 62), x + 18, max(deck_y + s * 40, deck_y + s * 62), PLUG_HEX, BLOCK_EDGE, 1.2)
        # ceramic
        self.poly([(x - 11, deck_y + s * 62), (x + 11, deck_y + s * 62), (x + 8, deck_y + s * 118),
                   (x - 8, deck_y + s * 118)], PLUG_CERAMIC, VALVE_LO, 1.0)
        self.rect(x - 5, min(deck_y + s * 118, deck_y + s * 132), x + 5, max(deck_y + s * 118, deck_y + s * 132), PLUG_HEX, BLOCK_EDGE, 1.0)
        if flash > 0:
            r = 8.0 + 10.0 * flash
            cx, cy = x + 5, deck_y - s * 15
            for i in range(6):
                a = math.radians(i * 60 + 15)
                self.line((cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a)), SPARK, 1.4)
            self.circle(cx, cy, 3.0, SPARK)

    def injector(self, x, deck_y, spraying=0.0):
        """A diesel injector: body above the deck, nozzle tip just below it."""
        self.poly([(x - 6, deck_y), (x + 6, deck_y), (x + 4, deck_y - 12), (x - 4, deck_y - 12)],
                  INJECTOR, VALVE_LO, 1.0)
        self.rect(x - 11, deck_y - 40, x + 11, deck_y, INJECTOR, BLOCK_EDGE, 1.0)
        self.rect(x - 17, deck_y - 62, x + 17, deck_y - 40, INJECTOR, BLOCK_EDGE, 1.2)
        self.rect(x - 9, deck_y - 128, x + 9, deck_y - 62, INJECTOR, BLOCK_EDGE, 1.0)
        self.rect(x - 14, deck_y - 140, x + 14, deck_y - 128, PLUG_HEX, BLOCK_EDGE, 1.0)
        self.line((x + 9, deck_y - 100), (x + 34, deck_y - 100), INJECTOR, 5.0)   # the fuel line

    def finger(self, pivot, tip, lift_px, w=16.0):
        """A finger follower / rocker: a lever from `pivot` to `tip`, drooped by the lift."""
        (px, py), (tx, ty) = pivot, tip
        L = math.hypot(tx - px, ty - py)
        base = math.atan2(ty - py, tx - px)
        a = base + math.atan2(lift_px, L)
        ex, ey = px + L * math.cos(a), py + L * math.sin(a)
        nx, ny = -math.sin(a), math.cos(a)
        self.poly([(px + nx * w * 0.5, py + ny * w * 0.5), (ex + nx * w * 0.35, ey + ny * w * 0.35),
                   (ex - nx * w * 0.35, ey - ny * w * 0.35), (px - nx * w * 0.5, py - ny * w * 0.5)],
                  CAM_HI, CAM, 1.2)
        self.circle(px, py, w * 0.55, CAM, CAM_HI, 1.2)
        self.circle(px, py, w * 0.22, PIN, PIN_RIM, 1.0)
        return (ex, ey)

    def flange(self, x0, y0, x1, y1, bolts=2):
        """An opaque manifold flange: what hides a reservoir end that cannot leave the frame."""
        self.hatched_rect(x0, y0, x1, y1, spacing=10.0)
        w, h = x1 - x0, y1 - y0
        if w >= h:
            for i in range(bolts):
                self.bolt(x0 + w * (i + 1) / (bolts + 1), 0.5 * (y0 + y1), 6.0)
        else:
            for i in range(bolts):
                self.bolt(0.5 * (x0 + x1), y0 + h * (i + 1) / (bolts + 1), 6.0)
