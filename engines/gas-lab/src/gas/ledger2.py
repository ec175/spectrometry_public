# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""ledger2.py - the GASOLINE ledger films: engine on top, INPUT | OUTPUT ledger with real charts.

Clancy, 2026-10-02: "the output is kinda hard to understand, the bar graph doesnt properly show
whats happening here. Do another pass on the detailed graphs. we have hundreds, and hundreds of
examples of graph usage in the past and should certainly use it."

What was wrong with ledger.py's columns: no axis values, no axis titles, no x-axis at all (a bar
was "a cycle" with nothing saying so), and the shrinking-per-cycle story had no variable beside
it to explain it. Every chart here therefore has:

  - a TITLE in three words or fewer saying what is plotted, and axis titles WITH UNITS
  - numbered ticks on both axes on a FIXED scale (never the frame's own peak - OUTPUT_RULES),
    a 0.5/1/2/2.5/5 tick ladder, the chrome present on frame one, only the data growing
  - an x-axis that is the film's own variable (time, mixture, intake pressure, firing number),
    so the reason a quantity changes is ON the chart
  - direct labels at the line heads instead of a legend (the library's rule, and Clancy's),
    the live point brightest and in the hot colour, older cycles dimmer

House style ported from the library (survey 2026-10-02): the manim spectra palette
(Video Rendering/Chemistry/manim/spectro_lib.py: BG #0a0d12, INK #e8edf2, MUTED #8a97a8, GRID
#2a3444, a thick axis spine and thin ticks) and Wind_Tunnel's PIL instrument block (render.py
_draw_axes: right-aligned fixed numeric slots, the live reading in (255,214,120)).

The panel is drawn at 2x and box-downsampled once, so lines and text are anti-aliased.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from . import world as W
from .gpu import xp

FONT_DIR = r"C:\Windows\Fonts"
BG = (10, 13, 18)
INK = (232, 237, 242)
MUTED = (138, 151, 168)
GRID = (42, 52, 68)
SPINE = (92, 104, 122)
HOT = (255, 214, 120)
CARD = (17, 21, 28)
CARD_EDGE = (40, 48, 60)
COL = {"fuel": (255, 196, 70), "O2": (96, 160, 255), "CO2": (255, 118, 60), "H2O": (255, 160, 150),
       "N2": (138, 151, 168)}

# iso-octane in air; textbook molar masses and gas-phase enthalpies of formation (kJ/mol)
M = {"fuel": 114.23, "O2": 32.00, "CO2": 44.01, "H2O": 18.02, "N2": 28.01}
HF = {"fuel": -208.4, "O2": 0.0, "CO2": -393.5, "H2O": -241.8, "N2": 0.0}
DH_FUEL = (8 * HF["CO2"] + 9 * HF["H2O"] - HF["fuel"])          # -5115.8 kJ per mol C8H18
MOL_CHARGE = 1e5 * 500e-6 / (8.314 * 300.0)                     # 500 cc at 1 bar, 300 K


def _font(name, px):
    for f in (name, "segoeui.ttf"):
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, f), max(6, int(px)))
        except OSError:
            continue
    return ImageFont.load_default()


def nice_ticks(lo, hi, n=5):
    """A 1/2/2.5/5 x 10^k ladder (the finance scenes' rule), at most ~n ticks."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= raw:
            step = m * p
            break
    t0 = math.ceil(lo / step - 1e-9) * step
    out = []
    v = t0
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v):
    if abs(v) >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
    return f"{v:.1f}" if abs(v * 10 - round(v * 10)) < 1e-6 else f"{v:.2f}"


# ====================================================================== the data
class Ledger2:
    """Everything the panel shows, measured off the World each frame from the clip's start."""

    def __init__(self, eng, world):
        self.eng, self.w = eng, world
        self.scale = MOL_CHARGE / (float(eng.notes["swept_area"]) * C.N0)    # mol per molecule
        self.U0 = world.reactions
        self.cycles = []
        self.cur = None
        self.n_ev = len(world.events)
        self.T_in, self.T_ex, self.T_ch, self.P_ch, self.V_ch = 300.0, 700.0, 300.0, 1.0, 0.0
        self.hist = []                     # (t_clip, units) per frame - the cumulative charts
        self.t_clip = 0.0
        self.firing_no = 0
        self.last_clock = None

    def _open_cycle(self):
        e = self.eng
        phi = e.notes.get("phi_fn")
        bst = e.notes.get("boost_fn")
        # the mixture IN THE CYLINDER at the spark, measured: fuel / O2 against stoichiometric
        # (12.5 O2 per C8H18). It lags the intake setting badly - the residual exhaust of lean
        # cycles carries their spare oxygen - so the mixture chart is plotted against THIS
        ch = e.notes["chamber_fn"](self.w)
        nf = int((ch & (self.w.sp == C.FUEL)).sum())
        no = int((ch & (self.w.sp == C.AIR)).sum())
        phic = 12.5 * nf / max(no, 1)
        self.cur = dict(U0=self.w.reactions, peak=0.0, pmax=0.0, pv=[], phic=phic,
                        phi=phi(self.w.t) if phi else 1.0, boost=bst(self.w.t) if bst else 1.0,
                        left_fuel=None, left_o2=None, face=self.firing_no % 3, t=self.t_clip)
        self.firing_no += 1

    def _close_cycle(self):
        c = self.cur
        c["U"] = self.w.reactions - c["U0"]
        self.cycles.append(c)

    def _mean_T(self, m):
        n = int(m.sum())
        if not n:
            return None, 0
        return (C.T0 * float(xp.where(m, self.w.vx ** 2 + self.w.vy ** 2, 0.0).sum()) / n
                / C.V_TH ** 2), n

    def update(self, t_clip):
        w, e = self.w, self.eng
        self.t_clip = t_clip
        for ev in w.events[self.n_ev:]:
            if ev[1] == "spark" and (not e.notes.get("dual_plug") or "leading" in ev[2]):
                if self.cur is not None:
                    self._close_cycle()
                self._open_cycle()
        self.n_ev = len(w.events)
        ch = e.notes["chamber_fn"](w)
        T, n = self._mean_T(ch)
        if T is not None:
            self.T_ch = T
        vfn = e.notes.get("volume_fn")
        if vfn is not None:
            A = vfn(w.t)
            self.V_ch = A
            self.P_ch = (n * self.T_ch / max(A, 1.0)) / (C.N0 * C.T0)
        if self.cur is not None and n >= 2500:
            # a chamber squeezed through the Wankel's waist briefly holds a few hundred
            # molecules and their mean spiked to 4247 K: a peak needs a real sample
            self.cur["peak"] = max(self.cur["peak"], self.T_ch)
            self.cur["pmax"] = max(self.cur["pmax"], self.P_ch)
            if vfn is not None:
                self.cur["pv"].append((self.V_ch, self.P_ch))
            # leftovers: what is still unburnt in the chamber when the exhaust valve opens
            evo = e.notes.get("evo_deg")
            if evo is not None and self.cur["left_fuel"] is None:
                d = e.notes["clock"](w.t)
                if self.last_clock is not None and (self.last_clock - evo) % 720 > 360 >= (d - evo) % 720:
                    self.cur["left_fuel"] = int((ch & (w.sp == C.FUEL)).sum())
                    self.cur["left_o2"] = int((ch & (w.sp == C.AIR)).sum())
                self.last_clock = d
        r_in = e.notes.get("intake_region")
        r_ex = e.notes.get("exhaust_region")
        if r_in is not None:
            t_, _ = self._mean_T(r_in(w))
            if t_:
                self.T_in += (t_ - self.T_in) * 0.15
        if r_ex is not None:
            t_, _ = self._mean_T(r_ex(w))
            if t_:
                self.T_ex += (t_ - self.T_ex) * 0.15
        self.hist.append((t_clip, w.reactions - self.U0))

    # ---- amounts (moles -> grams) ----
    def mol_from_units(self, U):
        s = self.scale
        return {"fuel": 2 * U * s, "O2": 25 * U * s, "CO2": 16 * U * s, "H2O": 18 * U * s}

    def mass_g(self, U):
        m = self.mol_from_units(U)
        return {k: m[k] * M[k] for k in m}

    def mg_molecules(self, n, key):
        return n * self.scale * M[key] * 1e3


# ====================================================================== the chart
class Chart:
    """A fixed-scale chart in panel units (1080 x 960 for the half-frame), drawn at 2x."""

    def __init__(self, P, box, xr, yr, title, xlabel, ylabel, xticks=None, yticks=None,
                 xfmt=_fmt, yfmt=_fmt):
        self.P = P
        self.x0, self.y0, self.x1, self.y1 = box             # plot area
        self.xr, self.yr = xr, yr
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.pending = []                                      # head labels, placed together
        self.xt = xticks or nice_ticks(*xr, 4)
        self.yt = yticks or nice_ticks(*yr, 4)
        self.xfmt, self.yfmt = xfmt, yfmt
        P._charts.append(self)

    def X(self, x):
        return self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)

    def Y(self, y):
        y = min(max(y, self.yr[0] - (self.yr[1] - self.yr[0]) * 0.02), self.yr[1] * 1.04)
        return self.y1 - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.y1 - self.y0)

    def chrome(self, d):
        P = self.P
        P.text(d, (self.x0 - 64, self.y0 - 78), self.title, P.f_title, INK)
        P.text(d, (self.x0 - 64, self.y0 - 40), self.ylabel, P.f_axis, MUTED)
        for v in self.yt:
            y = self.Y(v)
            P.line(d, [(self.x0, y), (self.x1, y)], GRID, 1.2)
            P.text(d, (self.x0 - 12, y), self.yfmt(v), P.f_tick, MUTED, "rm")
        for v in self.xt:
            x = self.X(v)
            P.line(d, [(x, self.y1), (x, self.y1 + 9)], SPINE, 2.0)
            P.text(d, (x, self.y1 + 16), self.xfmt(v), P.f_tick, MUTED, "ma")
        P.line(d, [(self.x0, self.y1), (self.x1, self.y1)], SPINE, 3.2)      # the spine
        P.line(d, [(self.x0, self.y0 - 4), (self.x0, self.y1)], SPINE, 3.2)
        P.text(d, (self.x1, self.y1 + 50), self.xlabel, P.f_axis, MUTED, "ra")

    def series(self, d, pts, col, width=3.4, label=None, value=None, live=True, dot=True,
               label_dy=0.0):
        """A line through pts with its head dotted and DIRECTLY labelled (no legend)."""
        if not pts:
            return
        P = self.P
        q = [(self.X(x), self.Y(y)) for x, y in pts]
        if len(q) > 1:
            P.line(d, q, col, width)
        hx, hy = q[-1]
        if dot:
            P.dot(d, (hx, hy), 7.5 if live else 5.5, col, ring=BG)
        if label or value:
            self.pending.append(dict(x=hx, y=hy + label_dy, label=label, value=value, col=col,
                                     live=live))

    def labels(self, d, gap=50.0):
        """Place every head label at once: sorted by height, pushed apart to at least `gap`,
        kept inside the plot, with a leader back to its point when it had to move. Labels that
        collide are unreadable - the first build stacked five of them on the origin."""
        P = self.P
        if not self.pending:
            return
        # a label that cannot fit to the RIGHT of its point goes ABOVE-left of it: placed plainly to
        # the left it sat on its own line (the line arrives from below-left, the text is above it)
        for r in self.pending:
            r["right"] = r["x"] + 150 < self.x1
            if not r["right"]:
                r["y"] = r["y"] - 46
        items = sorted(self.pending, key=lambda r: r["y"])
        ys = [min(max(r["y"], self.y0 + 14), self.y1 - 36) for r in items]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        over = ys[-1] - (self.y1 - 36)
        if over > 0:
            ys = [y - over for y in ys]
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        for r, y in zip(items, ys):
            right = r["right"]
            lx = r["x"] + 16 if right else r["x"] - 10
            anchor_l, anchor_v = ("lm", "lm") if right else ("rm", "rm")
            if right and abs(y - r["y"]) > 8:
                P.line(d, [(r["x"], r["y"]), (lx - (6 if right else -6), y + 6)], r["col"], 1.4)
            if r["label"]:
                P.text(d, (lx, y - 6), r["label"], P.f_lab, r["col"], anchor_l)
            if r["value"]:
                P.text(d, (lx, y + 20), r["value"], P.f_val, HOT if r["live"] else INK, anchor_v)
        self.pending = []

    def vline(self, d, x, text, col=MUTED):
        X = self.X(x)
        self.P.line(d, [(X, self.y0), (X, self.y1)], col, 1.6)
        self.P.text(d, (X + 8, self.y0 + 6), text, self.P.f_tick, col)


# ====================================================================== the panel
class Panel2:
    SS = 2

    def __init__(self, Wpx, Hpx, film):
        self.Wpx, self.Hpx = Wpx, Hpx
        self.k = Wpx / 1080.0 * self.SS                     # panel units -> canvas px
        self.film = film
        k = self.k
        self.f_head = _font("bahnschrift.ttf", 38 * k)
        self.f_name = _font("segoeuib.ttf", 28 * k)
        self.f_form = _font("segoeui.ttf", 24 * k)
        self.f_big = _font("bahnschrift.ttf", 52 * k)
        self.f_unit = _font("bahnschrift.ttf", 24 * k)
        self.f_mid = _font("bahnschrift.ttf", 22 * k)
        self.f_title = _font("bahnschrift.ttf", 30 * k)
        self.f_axis = _font("segoeui.ttf", 21 * k)
        self.f_tick = _font("bahnschrift.ttf", 20 * k)
        self.f_lab = _font("segoeuib.ttf", 21 * k)
        self.f_val = _font("bahnschrift.ttf", 21 * k)

    # ---- primitives in panel units ----
    def text(self, d, xy, s, f, fill, anchor="la"):
        d.text((xy[0] * self.k, xy[1] * self.k), s, font=f, fill=fill, anchor=anchor)

    def line(self, d, pts, col, w):
        d.line([(x * self.k, y * self.k) for x, y in pts], fill=col, width=max(1, int(round(w * self.k))),
               joint="curve")

    def dot(self, d, xy, r, col, ring=None):
        x, y = xy[0] * self.k, xy[1] * self.k
        R = r * self.k
        if ring is not None:
            d.ellipse([x - R - 2 * self.k, y - R - 2 * self.k, x + R + 2 * self.k, y + R + 2 * self.k],
                      fill=ring)
        d.ellipse([x - R, y - R, x + R, y + R], fill=col)

    def rect(self, d, box, fill, r=10, outline=None):
        d.rounded_rectangle([v * self.k for v in box], radius=r * self.k, fill=fill,
                            outline=outline, width=max(1, int(1.5 * self.k)) if outline else 0)

    # ---- the stat cards ----
    def card(self, d, x0, y0, key, name, formula, mg, mmol):
        c = COL[key]
        self.rect(d, (x0, y0, x0 + 250, y0 + 132), CARD, 12, CARD_EDGE)
        self.rect(d, (x0, y0, x0 + 7, y0 + 132), c, 3)
        self.text(d, (x0 + 22, y0 + 10), name, self.f_name, c)
        self.text(d, (x0 + 236, y0 + 14), formula, self.f_form, MUTED, "ra")
        val = f"{mg / 1000:.2f}" if mg >= 1000 else f"{mg:.1f}"
        unit = "g" if mg >= 1000 else "mg"
        # value right-aligned in a fixed slot, unit after it, mmol on its own line below
        self.text(d, (x0 + 188, y0 + 92), val, self.f_big, INK, "rs")
        self.text(d, (x0 + 196, y0 + 92), unit, self.f_unit, MUTED, "ls")
        self.text(d, (x0 + 22, y0 + 122), f"{mmol:.2f} mmol", self.f_mid, MUTED, "ls")

    def draw(self, L: Ledger2):
        S = self.SS
        img = Image.new("RGB", (self.Wpx * S, self.Hpx * S), BG)
        d = ImageDraw.Draw(img)
        U = L.w.reactions - L.U0
        mol = L.mol_from_units(U)
        mg = {k: mol[k] * M[k] * 1e3 for k in mol}
        # headers + divider
        self.line(d, [(540, 18), (540, 942)], (70, 82, 100), 2.4)
        self.text(d, (270, 18), "INPUT", self.f_head, INK, "ma")
        self.text(d, (810, 18), "OUTPUT", self.f_head, INK, "ma")
        # reaction badge + heat released, on the divider
        q = -DH_FUEL * mol["fuel"]
        self.rect(d, (540 - 128, 66, 540 + 128, 102), (24, 30, 40), 16, (70, 82, 100))
        self.text(d, (540, 84), f"\u0394H {DH_FUEL:.0f} kJ/mol", self.f_mid, HOT, "mm")
        self.rect(d, (540 - 100, 108, 540 + 100, 140), (44, 28, 16), 14)
        self.text(d, (540, 124), f"Heat {q:.2f} kJ", self.f_mid, (255, 190, 120), "mm")
        # cards
        self.card(d, 16, 150, "fuel", "Gasoline", "C\u2088H\u2081\u2088", mg["fuel"], mol["fuel"] * 1e3)
        self.card(d, 274, 150, "O2", "Oxygen", "O\u2082", mg["O2"], mol["O2"] * 1e3)
        self.card(d, 556, 150, "CO2", "CO\u2082", "", mg["CO2"], mol["CO2"] * 1e3)
        self.card(d, 814, 150, "H2O", "Water", "H\u2082O", mg["H2O"], mol["H2O"] * 1e3)
        # the strip under the cards: temperatures, mass balance
        m_in = mg["fuel"] + mg["O2"]
        m_out = mg["CO2"] + mg["H2O"]
        peak = L.cycles[-1]["peak"] if L.cycles else (L.cur["peak"] if L.cur else L.T_ch)
        self.text(d, (24, 306), f"Intake {L.T_in:.0f} K", self.f_mid, COL["O2"])
        self.text(d, (520, 306), f"{m_in:.1f} mg in", self.f_mid, INK, "ra")
        if self.show_peak:
            self.text(d, (562, 306), f"Peak {peak:.0f} K", self.f_mid, (255, 170, 110))
        else:
            self.text(d, (562, 306), f"Exhaust {L.T_ex:.0f} K", self.f_mid, (255, 170, 110))
        self.text(d, (1056, 306), f"{m_out:.1f} mg out", self.f_mid, INK, "ra")
        self._charts = []
        getattr(self, "_charts_" + self.film)(d, L, mg)
        for ch in self._charts:
            ch.labels(d)
        return np.asarray(img.resize((self.Wpx, self.Hpx), Image.BOX))

    # ---------------------------------------------------------------- per film
    def _cum(self, L, key, step=1):
        out = []
        for t, U in L.hist[::step] + [L.hist[-1]] if L.hist else []:
            out.append((t, L.mass_g(U)[key] * 1e3))                 # mg
        return out

    def _charts_revs(self, d, L, mg, title_in="Mass burned", title_out="Mass made"):
        ymax = self.film_ymax
        left = Chart(self, (92, 440, 506, 880), (0, 16), (0, ymax), title_in, "time (s)", "mg")
        right = Chart(self, (632, 440, 1046, 880), (0, 16), (0, ymax), title_out, "time (s)", "mg")
        for ch in (left, right):
            ch.chrome(d)
        step = max(1, len(L.hist) // 240)
        tot_in = [(t, (L.mass_g(U)["fuel"] + L.mass_g(U)["O2"]) * 1e3) for t, U in L.hist[::step] + L.hist[-1:]]
        left.series(d, tot_in, MUTED, 2.0, "Total", f"{tot_in[-1][1]:.0f} mg", live=False, dot=False,
                    label_dy=-30)
        right.series(d, tot_in, MUTED, 2.0, None, None, live=False, dot=False)
        for ch, keys in ((left, ("O2", "fuel")), (right, ("CO2", "H2O"))):
            for j, k in enumerate(keys):
                pts = self._cum(L, k, step)
                nm = {"fuel": "Gasoline", "O2": "Oxygen", "CO2": "CO\u2082", "H2O": "Water"}[k]
                ch.series(d, pts, COL[k], 3.6, nm, f"{pts[-1][1]:.1f} mg", label_dy=(-6 if j == 0 else 4))
        # the burn RATE as a running average since the clip began. A short window read 0 between
        # burns and spiked during them - a number that jumps is noise, not a reading.
        if L.hist and L.hist[-1][0] > 0.5:
            t1, u1 = L.hist[-1]
            rate = L.mass_g(u1)["fuel"] * 1e3 / t1
            self.text(d, (506, 404), f"{rate:.1f} mg/s fuel", self.f_val, HOT, "ra")

    def _charts_wankel(self, d, L, mg):
        """Left: what has burned, over time. Right: what each FIRING made, one column per
        firing, stacked CO2 under water, lettered by the rotor face that fired - a rotary
        fires once per shaft turn, cycling through its three faces A, B, C."""
        ymax = self.film_ymax
        left = Chart(self, (92, 440, 506, 880), (0, 16), (0, ymax), "Mass burned", "time (s)", "mg")
        nmax = self.n_firings
        right = Chart(self, (632, 440, 1046, 880), (0.4, nmax + 0.6), (0, self.film_ymax_out),
                      "Made per firing", "firing number", "mg",
                      xticks=list(range(2, nmax + 1, 2)), xfmt=lambda v: f"{v:.0f}")
        for ch in (left, right):
            ch.chrome(d)
        step = max(1, len(L.hist) // 240)
        for j, k in enumerate(("O2", "fuel")):
            pts = self._cum(L, k, step)
            nm = {"fuel": "Gasoline", "O2": "Oxygen"}[k]
            left.series(d, pts, COL[k], 3.6, nm, f"{pts[-1][1]:.1f} mg", label_dy=(-6 if j == 0 else 4))
        rows = self._per_cycle(L)
        bw = (right.X(1.0) - right.X(0.0)) * 0.62
        for i, r in enumerate(rows):
            n = i + 1
            if n > nmax:
                break
            g = L.mass_g(r.get("U", 0))
            live = i == len(rows) - 1
            a = 1.0 if live else 0.62
            x = right.X(n)
            y0 = right.Y(0.0)
            y1 = right.Y(g["CO2"] * 1e3)
            y2 = right.Y((g["CO2"] + g["H2O"]) * 1e3)
            if y0 - y1 > 0.5:
                self.rect(d, (x - bw / 2, y1, x + bw / 2, y0), tuple(int(c * a) for c in COL["CO2"]), 3)
            if y1 - y2 > 2.5:
                self.rect(d, (x - bw / 2, y2, x + bw / 2, y1 - 2), tuple(int(c * a) for c in COL["H2O"]), 3)
            self.text(d, (x, y0 - 4 if y0 - y2 < 26 else y2 - 8), "ABC"[r["face"]], self.f_tick,
                      HOT if live else MUTED, "mb")
        if rows:
            g = L.mass_g(rows[-1].get("U", L.w.reactions - rows[-1]["U0"]))
            self.text(d, (1046, 404), f"{(g['CO2'] + g['H2O']) * 1e3:.1f} mg this firing",
                      self.f_val, HOT, "ra")
        self.text(d, (640, 404), "CO₂", self.f_lab, COL["CO2"])
        self.text(d, (700, 404), "Water", self.f_lab, COL["H2O"])

    def _per_cycle(self, L):
        rows = list(L.cycles)
        if L.cur is not None:
            c = dict(L.cur)
            c["U"] = L.w.reactions - c["U0"]
            rows.append(c)
        return rows

    def _charts_mixture(self, d, L, mg):
        # measured 2026-10-02: the cylinder sits at phi 0.33 while the intake reads 0.9 - the
        # axis spans what the CYLINDER actually does, lean residual and all
        lo, hi = 0.2, 2.0
        left = Chart(self, (92, 440, 506, 880), (lo, hi), (0, self.film_ymax_in), "Burned per cycle",
                     "cylinder mixture \u03c6", "mg per cycle", xticks=[0.2, 0.6, 1.0, 1.4, 1.8])
        right = Chart(self, (632, 440, 1046, 880), (lo, hi), (0, self.film_ymax_out), "Left unburned",
                      "cylinder mixture \u03c6", "mg per cycle", xticks=[0.2, 0.6, 1.0, 1.4, 1.8])
        for ch in (left, right):
            ch.chrome(d)
            ch.vline(d, 1.0, "stoichiometric")
        rows = self._per_cycle(L)
        done = [r for r in rows[:-1] if r.get("U")]
        # burned per cycle, fuel AND oxygen: both rise with phi until stoichiometric, then flatten
        # once the oxygen runs out - the left chart's whole story
        fuel = [(r["phic"], L.mass_g(r["U"])["fuel"] * 1e3) for r in done]
        oxy = [(r["phic"], L.mass_g(r["U"])["O2"] * 1e3) for r in done]
        for pts, k, nm in ((oxy, "O2", "Oxygen"), (fuel, "fuel", "Gasoline")):
            for x, y in pts[:-1]:
                self.dot(d, (left.X(x), left.Y(y)), 5.5, tuple(int(c * 0.6) for c in COL[k]))
            if pts:
                left.series(d, pts, COL[k], 3.4, nm, f"{pts[-1][1]:.1f} mg")
        lf = [(r["phic"], L.mg_molecules(r["left_fuel"], "fuel")) for r in done if r.get("left_fuel") is not None]
        lo2 = [(r["phic"], L.mg_molecules(r["left_o2"], "O2")) for r in done if r.get("left_o2") is not None]
        if lo2:
            right.series(d, lo2, COL["O2"], 3.4, "Spare oxygen", f"{lo2[-1][1]:.1f} mg", label_dy=-8)
        if lf:
            right.series(d, lf, COL["fuel"], 3.4, "Unburnt fuel", f"{lf[-1][1]:.1f} mg", label_dy=8)
        if L.cur is not None:
            for ch in (left, right):
                X = ch.X(min(max(L.cur["phic"], ch.xr[0]), ch.xr[1]))
                self.line(d, [(X, ch.y1 - 6), (X, ch.y1 + 6)], HOT, 4)

    def _charts_boost(self, d, L, mg):
        lo, hi = 0.5, 1.8
        left = Chart(self, (92, 440, 506, 880), (lo, hi), (0, self.film_ymax_in), "Fuel per cycle",
                     "intake pressure (bar)", "mg", xticks=[0.5, 1.0, 1.5])
        right = Chart(self, (632, 440, 1046, 880), (0, self.v_max), (0, self.p_max), "Pressure vs volume",
                      "volume (cc)", "bar")
        for ch in (left, right):
            ch.chrome(d)
        left.vline(d, 1.0, "atmospheric")
        rows = self._per_cycle(L)
        done = [r for r in rows[:-1] if r.get("U")]
        pts = [(r["boost"], L.mass_g(r["U"])["fuel"] * 1e3) for r in done]
        for x, y in pts[:-1]:
            self.dot(d, (left.X(x), left.Y(y)), 5.5, tuple(int(c * 0.6) for c in COL["fuel"]))
        if pts:
            left.series(d, pts, COL["fuel"], 3.4, "Gasoline", f"{pts[-1][1]:.1f} mg")
        # P-V loops: the last three, older dimmer, the live one drawn to its head
        cc = 500.0 / self.swept
        loops = rows[-4:]
        for i, r in enumerate(loops):
            pv = [(v * cc, p) for v, p in r["pv"]][::2]
            if len(pv) < 2:
                continue
            live = i == len(loops) - 1
            a = 1.0 if live else 0.25 + 0.2 * i
            col = tuple(int(c * a) for c in COL["CO2"])
            right.series(d, pv, col, 3.2 if live else 2.0, None, None, live=live, dot=live)
        if loops and loops[-1]["pv"]:
            v, p = loops[-1]["pv"][-1]
            self.text(d, (1046, 404), f"{p:.1f} bar  {v * cc:.0f} cc", self.f_val, HOT, "ra")


def top_overlay(img, film, L, eng, t, stroke):
    """Stroke + cycle on the left margin, the film's variable on the right margin."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    u = im.width / 1080.0
    f1 = _font("bahnschrift.ttf", 44 * u)
    f2 = _font("bahnschrift.ttf", 28 * u)
    col = {"intake": (120, 176, 255), "compression": (200, 206, 216), "power": (255, 170, 90),
           "exhaust": (255, 130, 90)}.get(stroke, INK)
    d.text((40 * u, 60 * u), stroke.upper(), font=f1, fill=col)
    d.text((40 * u, 116 * u), f"{'Firing' if film == 'wankel' else 'Cycle'} "
                              f"{len(L.cycles) + (1 if L.cur else 0)}", font=f2, fill=MUTED)
    if film in ("revs", "wankel"):
        big, small = f"\u00d7{eng.drive.s(t):.2f}", "engine speed"
    elif film == "mixture":
        # the CYLINDER's mixture (measured at the last spark), the same number the charts plot
        phi = L.cur["phic"] if L.cur is not None else eng.notes["phi_fn"](t)
        big = f"\u03c6 {phi:.2f}"
        small = "lean mixture" if phi < 0.97 else ("rich mixture" if phi > 1.03 else "stoichiometric")
    else:
        b = eng.notes["boost_fn"](t)
        big, small = f"{b:.2f} bar", ("throttled intake" if b < 0.97 else
                                      ("boosted intake" if b > 1.03 else "atmospheric intake"))
    d.text((1040 * u, 60 * u), big, font=f1, fill=INK, anchor="ra")
    d.text((1040 * u, 116 * u), small, font=f2, fill=MUTED, anchor="ra")
    return np.asarray(im)


# ====================================================================== render
def render_ledger(eng, out_path, seconds=14.0, stills=(), video=True, log_every=120, seed=7,
                  ranges=None):
    from .pframe import PartFrame
    from .render import _advance, _ffmpeg_proc
    film = eng.notes["film"]
    warm = float(eng.notes.get("warm", 6.0))
    eng.drive.configure(warm=warm, ramp=float(eng.notes.get("ramp_frac", 1.3)) * seconds,
                        mult=float(eng.notes.get("mult", 1.0)))
    top_cfg = C.RenderConfig(width=1440, height=1280, fps=60)
    top_cfg.substeps_per_sec = int(eng.notes.get("substeps", top_cfg.substeps_per_sec))
    world = W.World(eng, seed=seed, coll_every=top_cfg.coll_every, cap=900_000)
    fr = PartFrame(top_cfg, eng)
    panel = Panel2(1440, 1280, film)
    for k, v in (ranges or {}).items():
        setattr(panel, k, v)
    panel.swept = float(eng.notes["swept_area"])
    panel.show_peak = bool(eng.notes.get("show_peak", True))
    panel.eng_phi = lambda L: eng.notes["phi_fn"](L.w.t) if "phi_fn" in eng.notes else 1.0
    t0 = time.time()
    _advance(world, top_cfg, warm - 1.0 / 60)
    world.mark_frame()
    L = Ledger2(eng, world)
    print(f"[warm] {warm:.1f}s in {time.time() - t0:.0f}s N={world.x.size} V_TH={C.V_TH:.0f} "
          f"scale={L.scale:.3e} mol/molecule", flush=True)
    N = int(round(seconds * 60))
    proc = tmp = None
    if video:
        tmp = os.path.splitext(out_path)[0] + ".part.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        proc = _ffmpeg_proc(C.RenderConfig.final(), tmp)
    want = sorted(int(round(s * 60)) for s in stills)
    try:
        for i in range(N):
            _advance(world, top_cfg, warm + i / 60.0)
            L.update(i / 60.0)
            if video or i in want:
                stroke = (W.stroke_name(eng.notes["clock"](world.t), eng.cycle_deg)
                          if eng.cycle_deg >= 720 else eng.notes.get("phase_fn", lambda t: "power")(world.t))
                top = fr.compose(fr.gas(world.snapshot()), world.t)
                top = top_overlay(top, film, L, eng, world.t, stroke)
                frame = np.concatenate([top, panel.draw(L)], axis=0)
                if video:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                if i in want:
                    p = os.path.splitext(out_path)[0] + f"_t{i / 60:.1f}.png"
                    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                    Image.fromarray(frame).save(p)
                    print(f"[still] {p}", flush=True)
            world.mark_frame()
            if log_every and (i + 1) % log_every == 0:
                print(f"[render] {i + 1}/{N} {time.time() - t0:6.0f}s cycles {len(L.cycles)} "
                      f"units {world.reactions - L.U0} peak {L.cur['peak'] if L.cur else 0:.0f} K "
                      f"pmax {L.cur['pmax'] if L.cur else 0:.1f}", flush=True)
            if not video and want and i >= want[-1]:
                break
        if video:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            os.replace(tmp, out_path)
            print(f"[done] {out_path} ({time.time() - t0:.0f}s)", flush=True)
    except BaseException:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        raise
    return L
