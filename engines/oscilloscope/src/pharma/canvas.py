# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""canvas.py — the shared phosphor canvas + molecule drawer every PHARMA template draws through.

Two light buffers, the drugscope recipe:
  phos   — persistence (decays every frame): moving things leave a phosphor comet; sweeps refresh.
  sharp  — rebuilt each frame: crisp text, labels, the steady parts of a molecule.
Each gets a half-res bloom, then the static graticule, then crtfilm.FilmLook (applied by the CLI).

Speed: drugscope allocates one full-frame PIL layer PER STROKE. Here every (buffer, colour) pair
shares ONE layer per frame, uploaded once — a frame with 300 strokes is ~10 uploads, not 300.
Overlapping strokes of one colour take the MAX (PIL) rather than adding; that is the look we
want anyway (no hot knots where bonds meet).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from osc.drugscope import ROLE, _blur_ds, _font
from osc.gpu import asnumpy, gaussian_filter, xp

HOT = (255, 92, 80)            # playhead / beam head (the drugscope red)
WHITE = (238, 244, 240)        # "difference" highlight: what changes between class members
ELEM = {"N": ROLE["N"], "O": ROLE["O"], "S": ROLE["S"], "Cl": (150, 250, 140),
        "F": (190, 255, 235), "Br": (230, 150, 90), "P": (255, 170, 90)}


def ease(u):
    """smootherstep, clamped."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


def rot3(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Canvas:
    # grid=False by default: Ethan 2026-10-19, "just get rid of the background grid, it looks even worse"
    # (static was fine but plain; the dynamic version was disliked twice). Pure black background.
    def __init__(self, w=1080, h=1920, *, decay=0.80, glow_sigma=9.0, glow_gain=0.85, grid=False):
        self.w, self.h = w, h
        self.s = w / 1080.0
        self.decay = decay
        self.glow_sigma, self.glow_gain = glow_sigma, glow_gain
        self.phos = xp.zeros((h, w, 3), np.float32)
        self._layers = {}
        self._fonts = {}
        # DYNAMIC grid (Ethan 2026-10-19: "a little dynamic rather than static"): the grid is drawn one
        # cell larger than the frame and slowly DRIFTS (crop offset), and a soft diagonal light wave
        # travels across it. Both are slow and low-amplitude: no flash, no full-frame jump.
        self.grid = grid
        self.t = 0.0
        self._step = self.w / 10.0
        self._grat = xp.asarray(self._graticule(pad=int(self._step) + 2) if grid else np.zeros((h, w, 3), np.float32))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self._diag = xp.asarray(0.35 * xx + yy)

    # ---- frame lifecycle ----
    def begin(self, decay=None, t=None):
        self.phos *= self.decay if decay is None else decay
        self._layers = {}
        self._patches = []
        self._knock = []
        if t is not None:
            self.t = t

    def _grid_now(self):
        if not self.grid:
            return 0.0
        st = self._step
        # SUBTLE (Ethan 2026-10-19 round 3: "changing way too fast ... just needs to be a subtle background"):
        # ~1.5 px/s drift, and a faint wave that takes ~70 s to cross (a 14 s clip sees ~1/5 of one pass)
        ox = int(round((self.t * 1.6) % st))
        oy = int(round((self.t * 0.9) % st))
        g = self._grat[oy:oy + self.h, ox:ox + self.w]
        span = 1.35 * self.h + 0.35 * self.w
        c = (self.t * 38.0 + 0.35 * span) % span - 0.2 * self.h
        wave = 1.0 + 0.45 * xp.exp(-((self._diag - c) / 420.0) ** 2)
        return g * wave[:, :, None]

    def _layer(self, buf, color):
        key = (buf, tuple(int(max(0, min(255, c))) for c in color))
        if key not in self._layers:
            im = Image.new("L", (self.w, self.h), 0)
            self._layers[key] = (im, ImageDraw.Draw(im))
        return self._layers[key][1]

    def patch(self, buf, x0, y0, arr):
        """An RGB float patch (canon_draw's per-molecule buffer) added onto `buf` at (x0, y0)."""
        self._patches.append((buf, int(x0), int(y0), arr))

    def finish(self):
        sharp = xp.zeros((self.h, self.w, 3), np.float32)
        for (buf, col), (im, _) in self._layers.items():
            if buf == "label":                                  # composited last, over the knockout mask
                continue
            m = xp.asarray(np.asarray(im, np.float32)) / 255.0
            tgt = self.phos if buf == "phos" else sharp
            tgt += m[:, :, None] * xp.asarray(np.asarray(col, np.float32))
        for buf, x0, y0, a in getattr(self, "_patches", []):
            tgt = self.phos if buf == "phos" else sharp
            ya, xa = max(0, y0), max(0, x0)
            yb, xb = min(self.h, y0 + a.shape[0]), min(self.w, x0 + a.shape[1])
            if yb > ya and xb > xa:
                tgt[ya:yb, xa:xb] += xp.asarray(a[ya - y0:yb - y0, xa - x0:xb - x0])
        out = (self._grid_now() + self.phos + _blur_ds(self.phos, self.glow_sigma) * self.glow_gain
               + sharp + _blur_ds(sharp, self.glow_sigma) * self.glow_gain)
        # LABELS over data (Ethan 2026-10-20: "the text and the trace overlap ... give the notations an invisible
        # background (or at least an opacity mask)"): everything drawn so far is darkened under a soft, feathered
        # mask around each knocked label, then the label layer (text + its own glow) goes on top, so the text reads
        # and the trace behind it fades out instead of running through the letters.
        lab = None
        for (buf, col), (im, _) in self._layers.items():
            if buf != "label":
                continue
            if lab is None:
                lab = xp.zeros((self.h, self.w, 3), np.float32)
            lab += (xp.asarray(np.asarray(im, np.float32)) / 255.0)[:, :, None] * xp.asarray(np.asarray(col, np.float32))
        if getattr(self, "_knock", None):
            hw, hh = self.w // 2, self.h // 2
            mim = Image.new("L", (hw, hh), 0)
            md = ImageDraw.Draw(mim)
            for x0, y0, x1, y1, a in self._knock:
                md.rounded_rectangle([x0 / 2, y0 / 2, x1 / 2, y1 / 2], radius=max(1, (y1 - y0) / 6), fill=int(255 * a))
            m = gaussian_filter(xp.asarray(np.asarray(mim, np.float32)) / 255.0, sigma=3.0 * self.s)
            m = xp.clip(m * 1.15, 0, 1)
            m = xp.repeat(xp.repeat(m, 2, axis=0), 2, axis=1)[:self.h, :self.w]
            out = out * (1.0 - 0.92 * m[:, :, None])
        if lab is not None:
            out = out + lab + _blur_ds(lab, self.glow_sigma) * self.glow_gain
        return asnumpy(xp.clip(out, 0, 255).astype(xp.uint8))

    def knock_rect(self, x0, y0, x1, y1, alpha=1.0):
        """Darken whatever lies under this box (soft edges) before the labels are added."""
        self._knock.append((x0, y0, x1, y1, alpha))

    # ---- primitives (gain scales the colour; >1 pushes toward bloom/blowout) ----
    @staticmethod
    def _c(color, gain):
        return tuple(c * gain for c in color)

    def line(self, buf, pts, color, width=4, gain=1.0):
        if gain <= 0.004 or len(pts) < 2:
            return
        self._layer(buf, self._c(color, gain)).line([tuple(map(float, p)) for p in pts], fill=255,
                                                     width=max(1, int(round(width))), joint="curve")

    def disc(self, buf, x, y, r, color, gain=1.0):
        if gain <= 0.004:
            return
        self._layer(buf, self._c(color, gain)).ellipse([x - r, y - r, x + r, y + r], fill=255)

    def ring(self, buf, x, y, r, color, width=3, gain=1.0):
        if gain <= 0.004:
            return
        self._layer(buf, self._c(color, gain)).ellipse([x - r, y - r, x + r, y + r], outline=255,
                                                        width=max(1, int(width)))

    def font(self, px):
        px = max(6, int(round(px * self.s)))
        if px not in self._fonts:
            self._fonts[px] = _font(px)
        return self._fonts[px]

    def text(self, buf, s, x, y, color, px, gain=1.0, anchor="la", knock=False):
        """knock=True: the text goes on the LABEL layer over a soft dark mask, so data under it fades out."""
        if gain <= 0.004 or not s:
            return
        if knock:
            if not hasattr(self, "_td"):
                self._td = ImageDraw.Draw(Image.new("L", (4, 4)))
            b = self._td.textbbox((x, y), s, font=self.font(px), anchor=anchor)
            pad = 7 * self.s
            self.knock_rect(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, min(1.0, gain))
            buf = "label"
        self._layer(buf, self._c(color, gain)).text((x, y), s, fill=255, font=self.font(px), anchor=anchor)

    def text_w(self, s, px):
        b = self.font(px).getbbox(s)
        return b[2] - b[0]

    def _graticule(self, pad=0):
        W, H = self.w + pad, self.h + pad
        img = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        step = self.w / 10.0
        for i in range(int(W / step) + 2):
            d.line([(i * step, 0), (i * step, H)], fill=(6, 20, 11), width=1)
        for j in range(int(H / step) + 2):
            d.line([(0, j * step), (W, j * step)], fill=(6, 20, 11), width=1)
        return np.asarray(img).astype(np.float32)


# ---------------------------------------------------------------- scramble text (house effect)
GLYPHS = "!@#$%&*+=/\\|<>[]{}?~^01"


def scramble(s, frac, seed):
    """Replace a fraction of s's glyphs with noise, re-picked per seed (=per frame)."""
    if frac <= 0.06 or not s:
        return s
    r = np.random.default_rng(abs(int(seed)) + 17)
    idx = [i for i, c in enumerate(s) if c != " "]
    k = min(len(idx), max(1, int(round(frac * len(idx)))))
    ch = list(s)
    for p in r.choice(idx, size=k, replace=False):
        ch[p] = GLYPHS[int(r.integers(0, len(GLYPHS)))]
    return "".join(ch)


# ---------------------------------------------------------------- molecules
def label_of(a):
    """Skeletal-formula label for an atom dict, or None for an unlabelled carbon."""
    s = a["sym"]
    if s == "C":
        return None
    h = a["nH"]
    return s + ("H" if h == 1 else f"H{h}" if h > 1 else "")


class Pose:
    """Card-style 3D motion for a flat 2D molecule: bounded rock about the vertical axis + tilt
    + slow in-plane sway. Never edge-on (|yaw| <= ~34 deg), so labels stay apart."""

    def __init__(self, yaw=30.0, pitch=14.0, roll=7.0, period=7.0, phase=0.0):
        self.yaw, self.pitch, self.roll, self.period, self.phase = yaw, pitch, roll, period, phase

    def R(self, t):
        w = 2 * math.pi / self.period
        return rot3(math.radians(self.pitch) * math.sin(0.73 * w * t + 1.3 + self.phase),
                    math.radians(self.yaw) * math.sin(w * t + self.phase),
                    math.radians(self.roll) * math.sin(0.41 * w * t + 0.4 + self.phase))


def project(xy, R, cx, cy, scale):
    """xy (N,2) molecule units (y up) -> screen px (N,2) under rotation R."""
    p = np.column_stack([xy, np.zeros(len(xy))]) @ R.T
    return np.column_stack([cx + p[:, 0] * scale, cy - p[:, 1] * scale])


def project3(xyz, R, cx, cy, scale):
    """xyz (N,3) molecule units -> (screen px (N,2), depth z (N,) toward the viewer)."""
    p = np.asarray(xyz, float) @ R.T
    return np.column_stack([cx + p[:, 0] * scale, cy - p[:, 1] * scale]), p[:, 2]


def draw_bond(cv, buf, pa, pb, order, color, width, gain, trim_a=0.0, trim_b=0.0, gap=7.0):
    d = pb - pa
    L = float(np.hypot(*d))
    if L < 1e-3:
        return
    u = d / L
    pa = pa + u * min(trim_a, 0.45 * L)
    pb = pb - u * min(trim_b, 0.45 * L)
    n = np.array([-u[1], u[0]])
    offs = {1: [0.0], 2: [-0.5, 0.5], 3: [-1.0, 0.0, 1.0]}.get(order, [0.0])
    for o in offs:
        cv.line(buf, [pa + n * o * gap, pb + n * o * gap], color, width, gain)


def draw_molecule(cv, mem, P, *, color, buf="sharp", width=None, gain=1.0, diff=False,
                  atom_gain=None, bond_gain=None, label_gain=1.0, show_labels=True,
                  label_px=None, scale=60.0, lit=None, lit_color=WHITE, lit_buf="phos", lit_r=0.30,
                  lit_gain=0.35, gap=None, depth=None, **_):
    """Draw member `mem` at projected positions P (N,2 px) — through the STRUCTURE CANON
    (pharma/canon_draw.py, canon/CANON.md): mitred joins, ring doubles inside, element-anchored labels
    with H away from the bonds, subscripts, charges, 2D wedges; with depth= (0 far..1 near per atom) the
    3D rules too (taper, crossing gaps, joint caps; no wedges).
    diff=True   -> core atoms/bonds in the class colour, the rest (what makes this drug THIS drug) in
                   white (A4: class videos only).
    atom_gain   -> per-atom gain (morph fade-in/out, fragments, depth dimming); a bond takes the min.
    bond_gain   -> per-bond gain, same order as mem["bonds"] (cut bonds).
    lit         -> per-atom 0..1 highlight (13C link, vibration): bonds blend to lit_color and brighten;
                   lit_gain > 0 also adds a glow disc. `gap` is ignored: the canon owns it (B1)."""
    from . import canon_draw as CD
    atoms = mem["atoms"]
    n = len(atoms)
    ag = np.ones(n) if atom_gain is None else np.asarray(atom_gain, float)
    M = CD.Model(n, "3d" if depth is not None else "2d", float(scale), tuple(color))
    M.p = np.asarray(P, float)
    if depth is not None:
        M.d = np.asarray(depth, float)
    M.gain = gain * ag
    lg = np.broadcast_to(np.asarray(label_gain, float), (n,))
    for k, a in enumerate(atoms):
        M.lab[k] = CD.label_parts(a["sym"], a["nH"], a.get("q", 0)) if show_labels else None
        M.lab_gain[k] = float(lg[k])
    bonds = mem["bonds"]
    bg = [gain * min(ag[i], ag[j]) * (1.0 if bond_gain is None else bond_gain[k]) for k, (i, j, o) in enumerate(bonds)]
    nb = [[] for _ in range(n)]
    for (i, j, o), g in zip(bonds, bg):
        if g > 0.004:
            nb[i].append(j); nb[j].append(i)
    rings = mem.get("rings", [])
    wed = {}
    if depth is None and CD.P["stereo"] == "wedge":
        for c, o_, ty in mem.get("wedges", []):
            wed[CD._key(c, o_)] = dict(frm=c, type=ty, w=1.0)
    for (i, j, o), g in zip(bonds, bg):
        core = atoms[i]["core"] and atoms[j]["core"]
        col = color if (not diff or core) else WHITE
        if lit is not None:
            lv = 0.5 * (lit[i] + lit[j])
            col = tuple(c + (lc - c) * lv for c, lc in zip(col, lit_color))
            g *= 1.0 + 0.9 * lv
        side = CD.side_of(M.p, nb, rings, i, j) if o == 2 else 0
        M.bonds.append(dict(i=i, j=j, lines=CD.bond_lines(o, side), gain=g, color=col, hot=0.0,
                            wedge=wed.get(CD._key(i, j))))
    M.nbrs = nb
    if lit is not None and lit_gain > 0:
        for k in range(n):
            if lit[k] > 0.02:
                cv.disc(lit_buf, P[k][0], P[k][1], lit_r * scale, lit_color, lit_gain * lit[k] * ag[k])
    CD.draw_model(cv, M, label_px=label_px, width=width, buf=buf)


def fit_scale(members, box_w, box_h, margin=1.3):
    """One px-per-bond scale that fits EVERY member in the box (core drawn the same size in all)."""
    ext = []
    for m in members:
        xy = np.array([[a["x"], a["y"]] for a in m["atoms"]])
        r = np.max(np.hypot(xy[:, 0], xy[:, 1])) + margin * 0.5
        ext.append(r)
    r = max(ext)
    return min(box_w, box_h) / (2 * r)


def centroid(mem):
    xy = np.array([[a["x"], a["y"]] for a in mem["atoms"]])
    return xy.mean(axis=0)


def xy_of(mem, recenter=True):
    xy = np.array([[a["x"], a["y"]] for a in mem["atoms"]], float)
    if recenter:
        xy = xy - 0.5 * (xy.max(axis=0) + xy.min(axis=0))
    return xy
