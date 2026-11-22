# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""canon_draw.py — the STRUCTURE CANON, drawn by the render engine.

A faithful Python port of `canon/viewer/canon.js` `drawModel` (the canon's reference implementation;
port its geometry, never re-derive it — canon/CANON.md "Porting the canon"). Parameters are read from
canon/rules.json ('v' = the canon value), the single source.

  A1-A6  one bond length, stroke ~ L, class colour + element colours, difference colour, bloom as halo,
         MITRED joins (single bonds are trapezoids; solid 3-bond junctions; a centred double lands flush)
  B1-B4  double gap 0.13 L; ring doubles INSIDE the ring (short inner line); triples; circular clearance
  C1-C6  element glyph ON the atom, H trailing AWAY from the bonds, subscript digits, formal charges,
         label ~ L, methyls bare
  E1-E3  wedges / hashes in 2D (never in 3D)
  F4,F6,F7  3D: depth taper, a front bond cuts a dark gap in the bond behind (never in a bond it shares
         an atom with — the erased band is repainted, clipped), solid joint caps

How it composites: every bond polygon is rasterised into a cropped mask and folded into ONE RGB float
buffer per molecule with np.maximum (the canon's "lighten": overlaps take the MAX, like the per-colour PIL
layers). That makes the F6 erase (zero a band) and the F7 clipped repaint (max(buf, poly x band)) exact.
The finished buffer is handed to the canvas as a patch on the 'sharp' layer; labels and fx go through the
canvas as before.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

HOT = (255, 92, 80)
WHITE = (238, 244, 240)
ELEM = {"N": (110, 175, 255), "O": (255, 110, 110), "S": (245, 225, 110), "Cl": (150, 250, 140),
        "F": (190, 255, 235), "Br": (230, 150, 90), "P": (255, 170, 90)}

_RULES = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "canon", "rules.json"))
P = {k: v["v"] for k, v in json.load(open(_RULES, encoding="utf-8"))["params"].items()}


def pose_R(t, R0, P_=None):
    """The canon 3D motion (F2/F3, Ethan 2026-10-19): FRONT-FACING dwell, then one bounded excursion per
    cycle, direction stepping by the golden angle. R0 = the base pose (R_ring). Identical to canon.js poseR."""
    P_ = P_ or P
    cyc, dw = float(P_["yaw_s"]), float(P_.get("dwell_s", 2.0))
    k = math.floor(t / cyc)
    v = ((t - k * cyc) - dw) / max(1e-6, cyc - dw)
    env = math.sin(math.pi * v) ** 2 if 0.0 < v < 1.0 else 0.0
    th = math.radians(137.5) * k + 0.6
    br = math.radians(float(P_.get("breathe_deg", 3.0)))
    tb = float(P_["tilt_s"])
    yaw = math.radians(P_["yaw_deg"]) * env * math.cos(th) + br * math.sin(2 * math.pi * t / tb)
    tilt = math.radians(P_["tilt_deg"]) * env * math.sin(th) + 0.7 * br * math.sin(2 * math.pi * t / (0.73 * tb) + 1.0)

    def rot(ax, ay):
        cx, sx, cy, sy = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Rx @ Ry
    return rot(tilt, yaw) @ np.asarray(R0, float)


def _key(i, j):
    return (i, j) if i < j else (j, i)


def mix(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def crs(a, b):
    return a[0] * b[1] - a[1] * b[0]


# ------------------------------------------------------------------ bond line recipes (B1-B3, G3)
def bond_lines(order, side):
    if order == 3:
        return [dict(off=-1.0), dict(off=0.0), dict(off=1.0)]
    if order == 2:
        if P["dbl_style"] == "inner" and side:
            return [dict(off=0.0), dict(off=float(side), short=True)]
        return [dict(off=-0.5), dict(off=0.5)]
    if order == 1:
        return [dict(off=0.0)]
    return []


def order_lines(oa, ob, g, side):
    """Partial order change oa -> ob at progress g: the extra line unzips from the MIDPOINT (G3)."""
    if oa == ob or g <= 0:
        return bond_lines(oa, side)
    if g >= 1:
        return bond_lines(ob, side)
    s = g if ob > oa else 1 - g
    lo, hi = min(oa, ob), max(oa, ob)
    if (lo, hi) == (1, 2):
        if P["dbl_style"] == "inner" and side:
            return [dict(off=0.0), dict(off=float(side), short=True, t0=0.5 - 0.5 * s, t1=0.5 + 0.5 * s)]
        return [dict(off=-0.5 * s), dict(off=0.5 * s, t0=0.5 - 0.5 * s, t1=0.5 + 0.5 * s)]
    if (lo, hi) == (2, 3):
        return [dict(off=-(0.5 + 0.5 * s)), dict(off=0.0), dict(off=0.5 + 0.5 * s, t0=0.5 - 0.5 * s, t1=0.5 + 0.5 * s)]
    return bond_lines(hi if s > 0.5 else lo, side)


def side_of(pos, nbrs, rings, i, j):
    """Which side a double bond's second line goes (B2): +1/-1 along rot90(u), 0 = centred."""
    pa, pb = pos[i], pos[j]
    u = pb - pa
    L = float(np.hypot(*u)) or 1.0
    n = np.array([-u[1] / L, u[0] / L])
    mid = 0.5 * (pa + pb)
    rs = [r for r in rings if i in r["atoms"] and j in r["atoms"]]
    if rs:
        rs.sort(key=lambda r: (-int(bool(r["ar"])), len(r["atoms"])))
        c = np.mean([pos[k] for k in rs[0]["atoms"]], axis=0)
        return 1 if float(np.dot(c - mid, n)) >= 0 else -1
    nbi = [k for k in nbrs[i] if k != j]
    nbj = [k for k in nbrs[j] if k != i]
    if not nbi or not nbj:
        return 0                                                    # terminal (C=O, =CH2): centred
    sc = sum(np.sign(float(np.dot(pos[k] - mid, n))) for k in nbi + nbj)
    return 1 if sc > 0 else -1 if sc < 0 else 0


# ------------------------------------------------------------------ the model
class Model:
    """One frame's molecule. atoms: p (px), d (depth 0 far..1 near), gain, lab {el,h,q}|None, lab_gain,
    lab_text, lab_color, hot. bonds: i, j, lines, gain, color, hot, wedge {frm,type,w}|None.
    fx: dicts kind spark|pulse|tip."""

    def __init__(self, n, space, L, color):
        self.n, self.space, self.L, self.color = n, space, L, color
        self.p = np.zeros((n, 2))
        self.d = np.ones(n)
        self.gain = np.ones(n)
        self.lab = [None] * n
        self.lab_gain = np.ones(n)
        self.lab_text = [None] * n
        self.lab_color = [None] * n
        self.hot = np.zeros(n)
        self.bonds = []
        self.nbrs = [[] for _ in range(n)]
        self.fx = []

    def link(self):
        self.nbrs = [[] for _ in range(self.n)]
        for b in self.bonds:
            if b["gain"] > 0.004:
                self.nbrs[b["i"]].append(b["j"]); self.nbrs[b["j"]].append(b["i"])


def label_parts(sym, nH, q):
    if sym == "C":
        return None                                                 # C6: carbons (and methyls) are bare
    return dict(el=sym, h=int(nH), q=int(q or 0))


# ------------------------------------------------------------------ raster helpers (cropped masks)
class Buf:
    """The molecule's RGB float buffer, cropped to the molecule (+ margin)."""

    def __init__(self, P_, margin):
        lo = P_.min(axis=0) - margin
        hi = P_.max(axis=0) + margin
        self.x0, self.y0 = int(math.floor(lo[0])), int(math.floor(lo[1]))
        self.w = max(1, int(math.ceil(hi[0])) - self.x0)
        self.h = max(1, int(math.ceil(hi[1])) - self.y0)
        self.a = np.zeros((self.h, self.w, 3), np.float32)

    def mask(self, draw_fn, pts):
        """Rasterise into a cropped L mask over the bbox of pts -> (y0, x0, mask float 0..1) local coords."""
        pts = np.asarray(pts, float)
        lo = np.floor(pts.min(axis=0) - 3).astype(int)
        hi = np.ceil(pts.max(axis=0) + 3).astype(int)
        lo = np.maximum(lo, [self.x0, self.y0])
        hi = np.minimum(hi, [self.x0 + self.w, self.y0 + self.h])
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            return None
        im = Image.new("L", (int(hi[0] - lo[0]), int(hi[1] - lo[1])), 0)
        draw_fn(ImageDraw.Draw(im), [(float(x - lo[0]), float(y - lo[1])) for x, y in pts])
        return int(lo[1] - self.y0), int(lo[0] - self.x0), np.asarray(im, np.float32) / 255.0

    def put(self, m, color, gain, clip=None):
        if m is None or gain <= 0.004:
            return
        y, x, mk = m
        if clip is not None:
            cy, cx, cm = clip
            full = np.zeros((self.h, self.w), np.float32)
            full[cy:cy + cm.shape[0], cx:cx + cm.shape[1]] = cm
            mk = mk * full[y:y + mk.shape[0], x:x + mk.shape[1]]
        c = np.minimum(255.0, np.asarray(color, np.float32) * gain)
        sl = self.a[y:y + mk.shape[0], x:x + mk.shape[1]]
        np.maximum(sl, mk[:, :, None] * c, out=sl)

    def erase(self, m):
        if m is None:
            return
        y, x, mk = m
        sl = self.a[y:y + mk.shape[0], x:x + mk.shape[1]]
        sl *= (1.0 - mk)[:, :, None]


def _poly(d, pts):
    d.polygon(pts, fill=255)


def _disc_fn(r):
    def f(d, pts):
        (x, y), = pts[:1]
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return f


def rect(s0, s1, w):
    u = s1 - s0
    L = float(np.hypot(*u)) or 1.0
    n = np.array([-u[1], u[0]]) / L * (w / 2)
    return [s0 + n, s1 + n, s1 - n, s0 - n]


# ------------------------------------------------------------------ A6 mitre quad (port of mitreQuad)
def mitre_quad(M, axial, s0, s1, u, nn, w, ea, eb, bi, bj, off_px):
    h = w / 2

    def cut(atom, other, d, edge_off, base):
        e0 = base + nn * edge_off
        if atom < 0:
            return e0
        ap = M.p[atom]
        allnb = [k for k in M.nbrs[atom] if k != other]
        ax = [k for k in allnb if _key(atom, k) in axial]
        nb = ax if ax else allnb
        if not nb:
            return e0
        n2 = np.array([-d[1], d[0]])
        want_left = float(np.dot(nn, n2)) * (off_px if abs(off_px) > 1e-6 else edge_off) > 0
        lo = hi = None
        for k in nb:
            v = M.p[k] - ap
            th = math.atan2(float(np.dot(v, n2)), float(np.dot(v, d)))
            if th < 0:
                th += 2 * math.pi
            if lo is None or th < lo[0]:
                lo = (th, k)
            if hi is None or th > hi[0]:
                hi = (th, k)
        if abs(off_px) > 1e-6 and _key(atom, other) not in axial:
            # a line of a centred double: end FLUSH on the facing edge of the single bond on its side (A6c)
            k = (lo if want_left else hi)[1]
            vN = M.p[k] - ap
            uN = vN / (float(np.hypot(*vN)) or 1.0)
            pN = np.array([-uN[1], uN[0]])
            if float(np.dot(pN, d)) < 0:
                pN = -pN
            P0 = ap + pN * h
            den = crs(d, uN)
            if abs(den) > 0.08:
                s = min(6 * w, max(-4 * w, crs(P0 - e0, uN) / den))
                return e0 + d * s
            return e0
        half = lo[0] / 2 if want_left else (hi[0] + 2 * math.pi) / 2
        bis = d * math.cos(half) + n2 * math.sin(half)
        den = crs(d, bis)
        if abs(den) < 0.08:
            return e0                                              # straight through: square end
        s = min(3 * w, max(-3 * w, crs(ap - e0, bis) / den))
        return e0 + d * s

    base_a, base_b = s0 - nn * off_px, s1 - nn * off_px
    aL = cut(ea, bj, u, off_px + h, base_a)
    aR = cut(ea, bj, u, off_px - h, base_a)
    bL = cut(eb, bi, -u, off_px + h, base_b)
    bR = cut(eb, bi, -u, off_px - h, base_b)
    pts = [aL, bL]
    if eb >= 0 and abs(off_px) < 1e-6:
        pts.append(M.p[eb])                                        # A6b: a centred line reaches the atom point
    pts += [bR, aR]
    if ea >= 0 and abs(off_px) < 1e-6:
        pts.append(M.p[ea])
    return pts


def _wedge_polys(b, qa, qb, w, L):
    pa, pb = (qa, qb) if b["wedge"]["frm"] == b["i"] else (qb, qa)
    d = pb - pa
    Lb = float(np.hypot(*d)) or 1.0
    u = d / Lb
    n = np.array([-u[1], u[0]])
    W = max(w, P["wedge_k"] * L * b["wedge"]["w"])
    if b["wedge"]["type"] == "wedge":
        return [("poly", [pa + n * w * 0.3, pb + n * W / 2, pb - n * W / 2, pa - n * w * 0.3])]
    out = []
    N = max(2, int(round(P["hash_n"])))
    for k in range(N):
        f = (k + 0.6) / N
        c = pa + (pb - pa) * f
        hw = max(w * 0.5, f * W / 2)
        out.append(("poly", rect(c + n * hw, c - n * hw, max(1.5, w * 0.62))))
    return out


# ------------------------------------------------------------------ DRAW (port of drawModel)
def draw_model(cv, M, *, label_px=None, width=None, buf="sharp"):
    s = cv.s
    L = M.L
    width = width or max(2.0, P["width_k"] * L)
    fpx = label_px if label_px is not None else float(np.clip(P["label_k"] * L, 18, 64)) / s
    gap = max(2.0, P["gap_k"] * L)
    # ---- label layout first (bond clearance depends on it)
    lay = [None] * M.n
    for k in range(M.n):
        lab = M.lab[k]
        if not lab or M.lab_gain[k] <= 0.01:
            continue
        hs = "H" if lab["h"] else ""
        digit = str(lab["h"]) if lab["h"] > 1 else ""
        clear = P["clear_k"] * fpx * s
        side = "right"
        if hs and P["h_side"] == "away":
            dirs = []
            for j in M.nbrs[k]:
                v = M.p[j] - M.p[k]
                l_ = float(np.hypot(*v)) or 1.0
                dirs.append(v / l_)

            def sc(v):
                return max((float(np.dot(d_, v)) for d_ in dirs), default=-1.0)
            r_, l_ = sc((1, 0)), sc((-1, 0))
            if min(r_, l_) < 0.55:
                side = "right" if r_ <= l_ + 0.05 else "left"
            else:
                side = "down" if sc((0, 1)) <= sc((0, -1)) else "up"
        lay[k] = dict(el=lab["el"], hs=hs, digit=digit, side=side, clear=clear * M.lab_gain[k])
    clear_of = [l_["clear"] if l_ else 0.0 for l_ in lay]
    # ---- bonds
    order = list(range(len(M.bonds)))
    if M.space == "3d":                                            # back to front (F6)
        order.sort(key=lambda n: M.d[M.bonds[n]["i"]] + M.d[M.bonds[n]["j"]])
    axial = {_key(b["i"], b["j"]) for b in M.bonds if b["gain"] > 0.004 and
             (b.get("wedge") or any(abs(l_["off"]) < 1e-6 and not l_.get("short") for l_ in b["lines"]))}
    margin = 0.6 * L + 20 * s
    B = Buf(M.p, margin)
    geo = {}

    def geom(n):
        if n in geo:
            return geo[n]
        b = M.bonds[n]
        if b["gain"] <= 0.004:
            geo[n] = None
            return None
        pa, pb = M.p[b["i"]], M.p[b["j"]]
        d = pb - pa
        Lb = float(np.hypot(*d))
        if Lb < 1e-3:
            geo[n] = None
            return None
        u = d / Lb
        nn = np.array([-u[1], u[0]])
        ta, tb = min(clear_of[b["i"]], 0.45 * Lb), min(clear_of[b["j"]], 0.45 * Lb)
        qa, qb = pa + u * ta, pb - u * tb
        dmean = 0.5 * (M.d[b["i"]] + M.d[b["j"]]) if M.space == "3d" else 1.0
        w = width * (1 - P["depth_taper"] * (1 - dmean) * (1 if M.space == "3d" else 0))
        col = b["color"]
        hot = b.get("hot", 0.0)
        if hot > 0:
            col = mix(col, HOT, 0.85 * hot)
        g = b["gain"] * (1 + 0.6 * hot)
        geo[n] = G = dict(b=b, u=u, nn=nn, ta=ta, tb=tb, qa=qa, qb=qb, w=w, col=col, g=g)
        return G

    def shapes(n):
        """The polygons of bond n (list of point lists)."""
        G = geom(n)
        if G is None:
            return []
        b, u, nn, ta, tb, qa, qb, w = G["b"], G["u"], G["nn"], G["ta"], G["tb"], G["qa"], G["qb"], G["w"]
        if b.get("wedge") and b["wedge"]["w"] > 0.001 and M.space == "2d":
            return [p for _, p in _wedge_polys(b, qa, qb, w, L)]
        out = []
        for ln in b["lines"]:
            t0, t1 = ln.get("t0", 0.0), ln.get("t1", 1.0)
            if t1 - t0 < 1e-3:
                continue
            off = nn * ln["off"] * gap
            s0 = qa + (qb - qa) * t0 + off
            s1 = qa + (qb - qa) * t1 + off
            if ln.get("short"):                                    # B2 inner line: shorter, square ends
                k = P["inner_trim"] * L
                if float(np.hypot(*(s1 - s0))) <= 2 * k + 2:
                    continue
                if t0 == 0.0:
                    s0 = s0 + u * k
                if t1 == 1.0:
                    s1 = s1 - u * k
                out.append(rect(s0, s1, w))
                continue
            if P["joins"] == "mitre":
                ea = b["i"] if (t0 == 0.0 and ta < 0.5) else -1
                eb = b["j"] if (t1 == 1.0 and tb < 0.5) else -1
                out.append(mitre_quad(M, axial, s0, s1, u, nn, w, ea, eb, b["i"], b["j"], ln["off"] * gap))
            else:
                out.append(rect(s0, s1, w))
        return out

    def paint(n, clip=None):
        G = geom(n)
        if G is None:
            return
        for pts in shapes(n):
            B.put(B.mask(_poly, pts), G["col"], G["g"], clip)

    drawn = []
    crossing = M.space == "3d" and P["cross_gap"] > 0
    for n in order:
        G = geom(n)
        if G is None:
            continue
        if crossing:                                               # F6: dark gap around a near bond
            e0 = G["qa"] + (G["qb"] - G["qa"]) * 0.16
            e1 = G["qa"] + (G["qb"] - G["qa"]) * 0.84
            hw = 0.5 * G["w"] + P["cross_gap"] * s
            band = B.mask(_poly, rect(e0, e1, 2 * hw))
            B.erase(band)
            bi, bj = G["b"]["i"], G["b"]["j"]
            for m in drawn:                                        # F7: never cut a bond sharing an atom
                c = M.bonds[m]
                if c["i"] in (bi, bj) or c["j"] in (bi, bj):
                    paint(m, clip=band)
        paint(n)
        drawn.append(n)
    if P["joins"] != "butt":                                       # F7 / A6b: solid joint caps
        at = {}
        for n in drawn:
            G = geo[n]
            b = G["b"]
            full = any(abs(l_["off"]) < 1e-6 and l_.get("t0", 0.0) == 0.0 and l_.get("t1", 1.0) == 1.0
                       for l_ in b["lines"])
            if not full:
                continue
            for k, t in ((b["i"], G["ta"]), (b["j"], G["tb"])):
                if t > 0.5:
                    continue
                e = at.setdefault(k, dict(n=0, w=1e9, g=0.0, col=None))
                e["n"] += 1
                e["w"] = min(e["w"], G["w"])
                if G["g"] > e["g"]:
                    e["g"], e["col"] = G["g"], G["col"]
        for k, e in at.items():
            if e["n"] >= 2:
                r = 0.5 * e["w"]
                B.put(B.mask(_disc_fn(r), [M.p[k]]), e["col"], e["g"])
    cv.patch(buf, B.x0, B.y0, B.a)
    # ---- labels (C1-C5)
    for k, l_ in enumerate(lay):
        if not l_:
            continue
        col0 = M.lab_color[k] or ELEM.get(M.lab[k]["el"], M.color)
        col = mix(col0, HOT, 0.7 * M.hot[k]) if M.hot[k] > 0 else col0
        g = M.lab_gain[k] * M.gain[k] * 1.1
        x, y = float(M.p[k][0]), float(M.p[k][1])
        el_txt = M.lab_text[k] if M.lab_text[k] is not None else l_["el"]
        cv.text(buf, el_txt, x, y, col, fpx, g, anchor="mm")
        ew, hw = cv.text_w(l_["el"], fpx), cv.text_w("H", fpx)
        sub_px = fpx * 0.62
        dw = cv.text_w(l_["digit"], sub_px if P["subscripts"] == "on" else fpx) if l_["digit"] else 0.0
        right = x + ew / 2
        if l_["hs"]:
            hy = y
            if l_["side"] == "right":
                hx = x + ew / 2 + hw / 2
            elif l_["side"] == "left":
                hx = x - ew / 2 - dw - hw / 2
            elif l_["side"] == "down":
                hx, hy = x, y + fpx * s * 0.95
            else:
                hx, hy = x, y - fpx * s * 0.95
            cv.text(buf, "H", hx, hy, col, fpx, g, anchor="mm")
            if l_["digit"]:
                if P["subscripts"] == "on":
                    cv.text(buf, l_["digit"], hx + hw / 2, hy + fpx * s * 0.30, col, sub_px, g, anchor="lm")
                else:
                    cv.text(buf, l_["digit"], hx + hw / 2, hy, col, fpx, g, anchor="lm")
            if l_["side"] == "right":
                right = hx + hw / 2 + dw
        q = M.lab[k]["q"]
        if P["charges"] == "on" and q:
            mark = ("+" if q == 1 else f"{q}+") if q > 0 else ("−" if q == -1 else f"{-q}−")
            cv.text(buf, mark, right + 1, y - fpx * s * 0.42, col, fpx * 0.6, g, anchor="lm")
    # ---- fx: sparks (local glows, H2), swap pulses, sprout tips
    for f in M.fx:
        if f["kind"] == "spark":
            cv.disc("phos", f["p"][0], f["p"][1], f["r"], f.get("color", WHITE), 0.30 * f["gain"])
            cv.disc("sharp", f["p"][0], f["p"][1], max(2.0, width * 0.9), WHITE, min(1.0, f["gain"]))
        elif f["kind"] == "pulse":
            cv.ring("phos", f["p"][0], f["p"][1], f["r"], f["color"], max(2.0, width * 0.7), f["gain"])
        elif f["kind"] == "tip":
            cv.disc("phos", f["p"][0], f["p"][1], f["r"], WHITE, f["gain"])
