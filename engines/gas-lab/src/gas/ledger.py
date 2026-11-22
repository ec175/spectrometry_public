# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""ledger.py - the four-stroke with its chemistry kept as a LEDGER (2026-09-26).

Clancy: "keep the four stroke video as the top half of a new video, and on the bottom half, split
that section down the middle with a vertical line. The left is the input, the right is the
output ... fuel, and oxygen on the left, and the gas, and water on the right with amounts like
enthalpy across the reaction, mass reacted, temperature, etc ... have the four stroke cylinder
speed up throughout the video, so that the bottom half's amounts/visuals are changing".

WHAT IS COUNTED, AND HOW IT BECOMES GRAMS
-----------------------------------------
Every number is read off the running simulation, the same one the top half draws. The model's
reaction is count-conserving 1 FUEL + 2 OXIDISER -> 1 CO2 + 2 H2O, which is methane burning in
oxygen exactly:  CH4 + 2 O2 -> CO2 + 2 H2O. So a reaction in the model is one of those, and
the ledger is its stoichiometry.

The model has ~72k molecules in one swept volume at 1 bar; a real 500 cc cylinder at 1 bar and
300 K holds 0.0200 mol. That ratio (SCALE, mol per simulated molecule) turns counts into
millimoles and milligrams. Molar masses and standard enthalpies of formation (gas-phase water,
as in an engine) are textbook values:

    CH4  16.04 g/mol   -74.87 kJ/mol          CO2  44.01 g/mol   -393.51 kJ/mol
    O2   32.00 g/mol     0.00                  H2O  18.02 g/mol   -241.83 kJ/mol
    => dH(reaction) = -802.3 kJ per mol CH4, and 80.04 g in = 80.05 g out per mol.

Temperatures are MEASURED (mean kinetic energy): the intake charge in the intake port, the
chamber at the spark (after compression), the per-cycle peak, and the exhaust port.

THE PANEL follows OUTPUT_RULES: short labels (<= three words), numbers where he asked for
amounts, light text on a dark panel so it always contrasts, colours from the gas LUT (fuel gold,
oxygen blue, CO2 ember, water rose), and a FIXED chart scale (0-100 mg per cycle for every
species, so left and right columns are comparable and nothing breathes with its own peak).
"""
from __future__ import annotations

import math
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from . import world as W
from .gpu import GPU, xp

MOL_CHARGE = 1e5 * 500e-6 / (8.314 * 300.0)          # 500 cc at 1 bar, 300 K: 0.0200 mol
M = {"CH4": 16.04, "O2": 32.00, "CO2": 44.01, "H2O": 18.02}
HF = {"CH4": -74.87, "O2": 0.0, "CO2": -393.51, "H2O": -241.83}
DH = HF["CO2"] + 2 * HF["H2O"] - HF["CH4"]                       # -802.3 kJ/mol
COL = {"CH4": (255, 196, 70), "O2": (96, 160, 255), "CO2": (255, 118, 60), "H2O": (255, 160, 150)}
FONT_DIR = r"C:\Windows\Fonts"


def _font(name, px):
    for f in (name, "segoeui.ttf"):
        try:
            return ImageFont.truetype(os.path.join(FONT_DIR, f), int(px))
        except OSError:
            continue
    return ImageFont.load_default()


class Ledger:
    """Tracks the reaction off a World, per frame, from the start of the clip."""

    def __init__(self, eng, world):
        self.eng, self.w = eng, world
        x0, x1 = eng.notes["bore"]
        self.scale = MOL_CHARGE / ((x1 - x0) * eng.notes["stroke"] * C.N0)   # mol per molecule
        self.R0 = world.reactions
        self.cycles = []          # closed cycles: dict(R, peak, comp)
        self.cur = dict(R0=world.reactions, peak=0.0, comp=0.0)
        self.n_ev = len(world.events)
        self.T_in = 300.0
        self.T_ex = 700.0
        self.T_ch = 300.0
        self.started = False

    def _mean_T(self, m):
        n = int(m.sum())
        if not n:
            return None
        return C.T0 * float(xp.where(m, self.w.vx ** 2 + self.w.vy ** 2, 0.0).sum()) / n / C.V_TH ** 2

    def update(self):
        w, eng = self.w, self.eng
        # spark events close a cycle
        for ev in w.events[self.n_ev:]:
            if ev[1] == "spark":
                if self.started:
                    self.cycles.append(dict(R=w.reactions - self.cur["R0"], peak=self.cur["peak"],
                                            comp=self.cur["comp"]))
                self.started = True
                self.cur = dict(R0=w.reactions, peak=0.0, comp=self.T_ch)
        self.n_ev = len(w.events)
        T = self._mean_T(eng.notes["chamber_fn"](w))
        if T is not None:
            self.T_ch = T
            self.cur["peak"] = max(self.cur["peak"], T)
        xm = 0.5 * sum(eng.notes["bore"])
        hb = eng.notes["head_box"]
        t_in = self._mean_T((w.box == hb) & (w.x < xm - 250.0))
        t_ex = self._mean_T((w.box == hb) & (w.x > xm + 250.0))
        if t_in is not None:
            self.T_in += (t_in - self.T_in) * 0.15
        if t_ex is not None:
            self.T_ex += (t_ex - self.T_ex) * 0.15

    # ---- amounts, from the start of the clip ----
    def mol(self, R):
        return R * self.scale

    def totals(self):
        R = self.w.reactions - self.R0
        n = self.mol(R)
        return dict(R=R, n=n, CH4=n, O2=2 * n, CO2=n, H2O=2 * n)

    def per_cycle(self, k_last=8):
        """mg of each species per cycle for the last k cycles, the live one last."""
        rows = [c["R"] for c in self.cycles] + [self.w.reactions - self.cur["R0"]]
        rows = rows[-k_last:]
        out = []
        for R in rows:
            n = self.mol(R)
            out.append({"CH4": n * M["CH4"] * 1e3, "O2": 2 * n * M["O2"] * 1e3,
                        "CO2": n * M["CO2"] * 1e3, "H2O": 2 * n * M["H2O"] * 1e3})
        return out


class Panel:
    """Draws the bottom half (W x H px) from a Ledger."""

    def __init__(self, W_, H_):
        self.W, self.H = W_, H_
        u = W_ / 1080.0
        self.u = u
        self.f_head = _font("bahnschrift.ttf", 40 * u)
        self.f_name = _font("segoeuib.ttf", 34 * u)
        self.f_form = _font("segoeui.ttf", 30 * u)
        self.f_big = _font("bahnschrift.ttf", 68 * u)
        self.f_mid = _font("bahnschrift.ttf", 30 * u)
        self.f_small = _font("segoeui.ttf", 24 * u)
        self.bg = (10, 12, 16)

    def _t(self, d, xy, s, f, fill=(232, 236, 242), anchor="la"):
        d.text((xy[0] * self.u, xy[1] * self.u), s, font=f, fill=fill, anchor=anchor)

    def _species(self, d, x0, y0, key, name, formula, mg, mmol, per, cyc):
        c = COL[key]
        u = self.u
        hh = 320
        d.rounded_rectangle([(x0 + 24) * u, y0 * u, (x0 + 516) * u, (y0 + hh) * u], radius=18 * u,
                            fill=(18, 21, 27), outline=(44, 50, 60), width=max(1, int(2 * u)))
        d.rounded_rectangle([(x0 + 24) * u, y0 * u, (x0 + 34) * u, (y0 + hh) * u], radius=4 * u, fill=c)
        self._t(d, (x0 + 54, y0 + 18), name, self.f_name, c)
        self._t(d, (x0 + 496, y0 + 22), formula, self.f_form, (180, 188, 200), "ra")
        # the number right-aligned to a fixed column, the unit just after it: no floating gap
        self._t(d, (x0 + 300, y0 + 62), f"{mg:.1f}", self.f_big, (240, 244, 250), "ra")
        self._t(d, (x0 + 312, y0 + 98), "mg", self.f_mid, (180, 188, 200))
        self._t(d, (x0 + 54, y0 + 148), f"{mmol:.2f} mmol", self.f_mid, (200, 206, 216))
        self._t(d, (x0 + 496, y0 + 148), f"{per:.1f} mg/cycle", self.f_mid, (200, 206, 216), "ra")
        # per-cycle columns, FIXED scale 0-100 mg, the live cycle brightest
        bx0, bx1, by0, by1 = x0 + 54, x0 + 496, y0 + 196, y0 + 302
        d.line([(bx0 * u, by1 * u), (bx1 * u, by1 * u)], fill=(60, 66, 78), width=max(1, int(2 * u)))
        for gy in (25, 50, 75, 100):
            yy = by1 - (by1 - by0) * gy / 100.0
            d.line([(bx0 * u, yy * u), (bx1 * u, yy * u)], fill=(30, 34, 42), width=1)
        n = 8
        cw = (bx1 - bx0) / n
        for i, row in enumerate(cyc):
            v = min(row[key], 100.0)
            h = (by1 - by0) * v / 100.0
            live = i == len(cyc) - 1
            a = 1.0 if live else 0.45 + 0.4 * (i + 1) / len(cyc)
            fill = tuple(int(ch * a) for ch in c)
            xa = bx0 + i * cw + cw * 0.14
            xb = bx0 + (i + 1) * cw - cw * 0.14
            if h > 0.5:
                d.rounded_rectangle([xa * u, (by1 - h) * u, xb * u, by1 * u], radius=4 * u, fill=fill)

    def draw(self, L, speed, cycle_no):
        img = Image.new("RGB", (self.W, self.H), self.bg)
        d = ImageDraw.Draw(img)
        u = self.u
        tot = L.totals()
        cyc = L.per_cycle(8)
        last = cyc[-2] if len(cyc) >= 2 else cyc[-1]
        # headers and the divider
        d.line([(540 * u, 26 * u), (540 * u, 934 * u)], fill=(120, 130, 146), width=max(2, int(3 * u)))
        self._t(d, (270, 34), "INPUT", self.f_head, (240, 244, 250), "ma")
        self._t(d, (810, 34), "OUTPUT", self.f_head, (240, 244, 250), "ma")
        # the reaction badge on the divider
        d.rounded_rectangle([(540 - 150) * u, 84 * u, (540 + 150) * u, 128 * u], radius=20 * u,
                            fill=(26, 30, 38), outline=(120, 130, 146), width=max(1, int(2 * u)))
        self._t(d, (540, 92), f"\u0394H  {DH:.0f} kJ/mol", self.f_small, (255, 214, 150), "ma")
        # species
        mg = {k: tot[k] * M[k] * 1e3 for k in ("CH4", "O2", "CO2", "H2O")}
        mm = {k: tot[k] * 1e3 for k in ("CH4", "O2", "CO2", "H2O")}
        self._species(d, 0, 186, "CH4", "Methane", "CH\u2084", mg["CH4"], mm["CH4"], last["CH4"], cyc)
        self._species(d, 0, 522, "O2", "Oxygen", "O\u2082", mg["O2"], mm["O2"], last["O2"], cyc)
        self._species(d, 540, 186, "CO2", "Carbon dioxide", "CO\u2082", mg["CO2"], mm["CO2"], last["CO2"], cyc)
        self._species(d, 540, 522, "H2O", "Water vapour", "H\u2082O", mg["H2O"], mm["H2O"], last["H2O"], cyc)
        # the bottom strip: temperatures, enthalpy, mass balance, heat
        H_in = tot["n"] * HF["CH4"]
        H_out = tot["n"] * (HF["CO2"] + 2 * HF["H2O"])
        m_in = mg["CH4"] + mg["O2"]
        m_out = mg["CO2"] + mg["H2O"]
        peak = max([c["peak"] for c in L.cycles[-1:]] + [L.cur["peak"]]) if L.started else L.T_ch
        comp = L.cur["comp"] if L.started else L.T_ch
        y = 860
        self._t(d, (30, y), f"Intake {L.T_in:4.0f} K", self.f_small, (150, 190, 255))
        self._t(d, (30, y + 30), f"Compressed {comp:4.0f} K", self.f_small, (200, 206, 216))
        self._t(d, (510, y), f"H {H_in:6.2f} kJ", self.f_small, (200, 206, 216), "ra")
        self._t(d, (510, y + 30), f"{m_in:6.1f} mg in", self.f_small, (200, 206, 216), "ra")
        self._t(d, (570, y), f"Peak {peak:4.0f} K", self.f_small, (255, 170, 110))
        self._t(d, (570, y + 30), f"Exhaust {L.T_ex:4.0f} K", self.f_small, (200, 206, 216))
        self._t(d, (1050, y), f"H {H_out:6.2f} kJ", self.f_small, (200, 206, 216), "ra")
        self._t(d, (1050, y + 30), f"{m_out:6.1f} mg out", self.f_small, (200, 206, 216), "ra")
        # heat released, on the divider under the badge
        q = -DH * tot["n"]
        d.rounded_rectangle([(540 - 110) * u, 136 * u, (540 + 110) * u, 170 * u], radius=14 * u,
                            fill=(40, 26, 18))
        self._t(d, (540, 140), f"Heat {q:5.2f} kJ", self.f_small, (255, 190, 120), "ma")
        return np.asarray(img)


def top_overlay(img, eng, t, speed, cycle_no, stroke):
    """Stroke name and speed in the black margins beside the engine (the top half)."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    u = im.width / 1080.0
    f1 = _font("bahnschrift.ttf", 44 * u)
    f2 = _font("bahnschrift.ttf", 30 * u)
    col = {"intake": (120, 176, 255), "compression": (200, 206, 216), "power": (255, 170, 90),
           "exhaust": (255, 130, 90)}.get(stroke, (220, 224, 230))
    d.text((40 * u, 60 * u), stroke.upper(), font=f1, fill=col)
    d.text((40 * u, 118 * u), f"Cycle {cycle_no}", font=f2, fill=(200, 206, 216))
    d.text((1040 * u, 60 * u), f"\u00d7{speed:.2f}", font=f1, fill=(240, 244, 250), anchor="ra")
    d.text((1040 * u, 118 * u), "speed", font=f2, fill=(160, 168, 180), anchor="ra")
    return np.asarray(im)


# ====================================================================== the split render
def render_split(eng, out_path, seconds=20.0, warm=None, mult=3.5, stills=(), video=True,
                 log_every=120, seed=7):
    """Top half: the engine (1440x1280). Bottom half: the ledger. One simulation for both."""
    from .pframe import PartFrame
    from .render import _advance, _ffmpeg_proc
    warm = float(eng.notes.get("warm", 6.0)) if warm is None else float(warm)
    # speed up THROUGHOUT: a smoothstep that would finish at 1.3x the clip, so it is still
    # climbing on the last frame (a ramp that ends in the clip would read as "then it stops")
    eng.drive.configure(warm=warm, ramp=1.3 * seconds, mult=mult)
    top_cfg = C.RenderConfig(width=1440, height=1280, fps=60)
    top_cfg.substeps_per_sec = int(eng.notes.get("substeps", top_cfg.substeps_per_sec))
    out_cfg = C.RenderConfig.final()
    world = W.World(eng, seed=seed, coll_every=top_cfg.coll_every, cap=900_000)
    fr = PartFrame(top_cfg, eng)
    panel = Panel(1440, 1280)
    t0 = time.time()
    _advance(world, top_cfg, warm - 1.0 / 60)
    world.mark_frame()
    led = Ledger(eng, world)
    print(f"[warm] {warm:.1f}s in {time.time() - t0:.0f}s, N={world.x.size}, V_TH={C.V_TH:.0f}, "
          f"substeps={top_cfg.substeps_per_sec}, scale={led.scale:.3e} mol/molecule", flush=True)
    N = int(round(seconds * 60))
    proc = tmp = None
    if video:
        tmp = os.path.splitext(out_path)[0] + ".part.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        proc = _ffmpeg_proc(out_cfg, tmp)
    want = sorted(int(round(s * 60)) for s in stills)
    shots = []
    try:
        for i in range(N):
            t = warm + i / 60.0
            _advance(world, top_cfg, t)
            led.update()
            if video or i in want:
                spd = eng.drive.s(world.t)
                cyc = len(led.cycles) + (1 if led.started else 0)
                stroke = W.stroke_name(eng.notes["clock"](world.t), 720.0)
                top = fr.compose(fr.gas(world.snapshot()), world.t)
                top = top_overlay(top, eng, world.t, spd, cyc, stroke)
                bot = panel.draw(led, spd, cyc)
                frame = np.concatenate([top, bot], axis=0)
                if video:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                if i in want:
                    p = os.path.splitext(out_path)[0] + f"_t{i / 60:.1f}.png"
                    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                    Image.fromarray(frame).save(p)
                    shots.append(p)
                    print(f"[still] {p}", flush=True)
            world.mark_frame()
            if log_every and (i + 1) % log_every == 0:
                tot = led.totals()
                print(f"[render] {i + 1}/{N} {time.time() - t0:6.0f}s x{eng.drive.s(world.t):.2f} "
                      f"cycles {len(led.cycles)} reacted {tot['R']} "
                      f"({tot['CH4'] * M['CH4'] * 1e3:.1f} mg CH4) peak {led.cur['peak']:.0f} K",
                      flush=True)
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
    return shots
