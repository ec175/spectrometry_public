# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""metal.py - the parts set's metal, shaded on the GPU from the solids' own grids.

parts.py (the 2026-09-20 look) drew flat Pillow polygons with a light outline. That is what
Clancy meant by the models lacking detail: every part was a filled shape. Here each body is a
sdf.Solid, and its picture is SAMPLED from the same grid the gas bounces off:

  alpha    from the signed distance, anti-aliased over one output pixel
  normal   from the part's height field: a quarter-round bevel on every edge, recessed faces,
           raised bosses - relief, not extra parts
  albedo   the material, its paint zones, and a faint machining texture (turned, ground, cast)
  light    ONE fixed lamp, upper left, never moving with the part: diffuse + a Blinn highlight.
           Because the normals are rotated with the part but the lamp is not, a turning rotor's
           highlight stays put while its lobes sweep through it - which is what reads as metal
  section  a cut casting (sectioned=True) gets 45-degree hatching in SCREEN space, as in a real
           section drawing

Static bodies are drawn once into a static layer; moving bodies are sampled per frame over their
bounding box only. A fast body is drawn at several instants across the shutter and averaged
(`blur`), so a spinning wheel smears instead of strobing backwards.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .gpu import GPU, xp

if GPU:
    from cupyx.scipy.ndimage import map_coordinates
else:
    from scipy.ndimage import map_coordinates

LIGHT = np.array([-0.46, -0.62, 0.64])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0.0, 0.0, 1.0])
HALF = HALF / np.linalg.norm(HALF)


class Metal:
    def __init__(self, cfg, eng, zoom=1.0, view=None):
        self.W, self.H = cfg.width, cfg.height
        self.k = cfg.scale
        self.zoom = float(zoom)
        self.vc = view or (C.REF_W * 0.5, C.REF_H * 0.5)
        self.eng = eng
        self.fps = cfg.fps
        self.kz = self.k * self.zoom
        self.hatch_px = 12.0 * self.kz
        bodies = [b for b in eng.bodies if b.draw]
        self.static = sorted([b for b in bodies if b.static], key=lambda b: b.z)
        # moving bodies with z < 0 pass BEHIND the static castings (the back of a coil spring)
        self.behind = sorted([b for b in bodies if not b.static and b.z < 0], key=lambda b: b.z)
        self.moving = sorted([b for b in bodies if not b.static and b.z >= 0], key=lambda b: b.z)
        self.rgb0 = xp.zeros((self.H, self.W, 3), dtype=xp.float32)
        self.a0 = xp.zeros((self.H, self.W, 1), dtype=xp.float32)
        for b in self.static:
            self._draw(self.rgb0, self.a0, b, 0.0, samples=1)

    # ---- view maps ----
    def to_out(self, x, y):
        return (((x - self.vc[0]) * self.zoom) * self.k + 0.5 * self.W,
                ((y - self.vc[1]) * self.zoom) * self.k + 0.5 * self.H)

    def to_world(self, X, Y):
        return ((X - 0.5 * self.W) / self.k / self.zoom + self.vc[0],
                (Y - 0.5 * self.H) / self.k / self.zoom + self.vc[1])

    # ---- one body ----
    def _bbox(self, b, pose):
        """The output-pixel box of the body's GRID, posed (not a circle of its radius: a
        needle 600 px long from its origin would otherwise sample the whole frame)."""
        s = b.solid
        g = self.eng.groups[b.group]
        ca, sa = math.cos(pose[2]), math.sin(pose[2])
        xs, ys = [], []
        for (lx, ly) in ((s.x0, s.y0), (s.x0 + s.gw * s.h, s.y0),
                         (s.x0, s.y0 + s.gh * s.h), (s.x0 + s.gw * s.h, s.y0 + s.gh * s.h)):
            px_, py_ = pose[0] + ca * lx - sa * ly, pose[1] + sa * lx + ca * ly
            X, Y = self.to_out(*g.to_world(px_, py_))
            xs.append(X)
            ys.append(Y)
        x0, y0 = max(0, int(math.floor(min(xs))) - 1), max(0, int(math.floor(min(ys))) - 1)
        x1, y1 = min(self.W, int(math.ceil(max(xs))) + 2), min(self.H, int(math.ceil(max(ys))) + 2)
        return x0, y0, x1, y1

    def _sample(self, b, pose, x0, y0, x1, y1):
        """Premultiplied colour and alpha of body b at `pose`, over the output box."""
        s = b.solid
        g = self.eng.groups[b.group]
        ys = xp.arange(y0, y1, dtype=xp.float32) + 0.5
        xs = xp.arange(x0, x1, dtype=xp.float32) + 0.5
        Xo, Yo = xp.meshgrid(xs, ys)
        wx, wy = self.to_world(Xo, Yo)
        lx, ly = g.to_local(wx, wy)
        px, py, ang = pose[0], pose[1], pose[2]
        ca, sa = math.cos(ang), math.sin(ang)
        dx, dy = lx - px, ly - py
        bx = ca * dx + sa * dy
        by = -sa * dx + ca * dy
        fx = (bx - s.x0) / s.h
        fy = (by - s.y0) / s.h
        coords = xp.stack([fy.ravel(), fx.ravel()])
        shp = Xo.shape
        D = map_coordinates(s.d.reshape(s.gh, s.gw), coords, order=1, mode="constant",
                            cval=1e3).reshape(shp)
        a = xp.clip(0.5 - D * self.kz, 0.0, 1.0)
        if float(a.max()) <= 0.0:
            return None
        nxl = map_coordinates(s.nx_l, coords, order=1, mode="constant", cval=0.0).reshape(shp)
        nyl = map_coordinates(s.ny_l, coords, order=1, mode="constant", cval=0.0).reshape(shp)
        alb = [map_coordinates(s.albedo[c], coords, order=1, mode="constant", cval=0.0).reshape(shp)
               for c in range(3)]
        # normals: body angle + group angle into the page frame
        ga = math.radians(g.angle_deg) + ang
        cg, sg = math.cos(ga), math.sin(ga)
        nx = cg * nxl - sg * nyl
        ny = sg * nxl + cg * nyl
        nz = xp.sqrt(xp.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        dif = xp.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0.0, 1.0)
        nh = xp.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0.0, 1.0)
        spec = s.gloss * nh ** 36 + 0.25 * s.gloss * nh ** 6
        lit = 0.34 + 0.78 * dif
        if s.sectioned:
            u = (Xo + Yo) / self.hatch_px
            fr = u - xp.floor(u)
            hatch = xp.exp(-((fr - 0.5) ** 2) / (2 * 0.07 ** 2))
            lit = lit * (1.0 + 0.30 * hatch * (nz > 0.97))
        # a light rim at the very edge: the reference cutaways' outline, now from the geometry
        rim = xp.exp(-((D * self.kz + 0.9) ** 2) / 0.9) * 0.22
        out = xp.stack([alb[c] * lit + spec * 0.85 + rim for c in range(3)], axis=-1)
        return out * a[..., None], a[..., None]

    def _draw(self, rgb, al, b, t, samples=None):
        pose = b.pose(t)
        n = samples or self._samples(b, pose)
        if n <= 1:
            poses = [pose]
        else:
            sh = 0.5 / self.fps                    # a 180-degree shutter
            poses = [b.pose(t - sh * (i + 0.5) / n) for i in range(n)]
        boxes = [self._bbox(b, p) for p in poses]
        x0 = min(q[0] for q in boxes); y0 = min(q[1] for q in boxes)
        x1 = max(q[2] for q in boxes); y1 = max(q[3] for q in boxes)
        if x1 <= x0 or y1 <= y0:
            return
        acc_c, acc_a = None, None
        for p in poses:
            r = self._sample(b, p, x0, y0, x1, y1)
            if r is None:
                continue
            c, a = r
            acc_c = c if acc_c is None else acc_c + c
            acc_a = a if acc_a is None else acc_a + a
        if acc_c is None:
            return
        acc_c, acc_a = acc_c / len(poses), acc_a / len(poses)
        # premultiplied OVER
        sub_c, sub_a = rgb[y0:y1, x0:x1], al[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = acc_c + sub_c * (1.0 - acc_a)
        al[y0:y1, x0:x1] = acc_a + sub_a * (1.0 - acc_a)

    def _samples(self, b, pose):
        """Enough instants that the rim moves < ~2.5 output px between them."""
        om = abs(pose[5])
        v = math.hypot(pose[3], pose[4])
        travel = (om * b.solid.rad + v) / self.fps * 0.5 * self.kz
        return int(min(10, max(1, math.ceil(travel / 2.5))))

    def render(self, t):
        """(premultiplied rgb HxWx3, alpha HxWx1) on the backend."""
        if self.behind:
            rgb = xp.zeros_like(self.rgb0)
            al = xp.zeros_like(self.a0)
            for b in self.behind:
                self._draw(rgb, al, b, t)
            rgb = self.rgb0 + rgb * (1.0 - self.a0)          # static OVER the behind layer
            al = self.a0 + al * (1.0 - self.a0)
        else:
            rgb, al = self.rgb0.copy(), self.a0.copy()
        for b in self.moving:
            self._draw(rgb, al, b, t)
        return rgb, al
