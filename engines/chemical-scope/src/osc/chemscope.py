"""chemscope.py — CRT-scope "chemical profile" compositor (molecule above, spectrum trace below).

A manim-free remake of the vertical chemical-profile format, drawn as an oscilloscope screen:
  TOP half   — the molecule, its skeletal BONDS stroked as GREEN scope traces and its
               functional-group SYMBOLS drawn in per-role trace colours (N blue, O red,
               CHx dim-green). Symbol styling MIRRORS the manim `styled_molecule` recipe:
               a size scaled to the on-screen molecule, a radial OUTWARD push (weaker near
               the centre via `push_power`), and a BG halo that KNOCKS OUT the bond behind
               each symbol so letters never sit on a bond. Molecule gently rocks in 3D
               (billboard, width ~ cos theta) — a bounded rock, never edge-on, so symbols
               stay separated.
  BOTTOM half — the spectrum as a live GREEN scope trace in a phosphor-persistence buffer
               with a hot red beam head sweeping along it (playhead). Two METHODS play in
               sequence — FTIR then RAMAN — cross-dissolving through the phosphor decay.
  TEXT       — title / subtitle / method header + band legend / axis, all GREEN ASCII glyphs,
               laid out so NOTHING overlaps the trace (all text lives ABOVE the plot box; only
               small numeric markers sit in the clear gap just above each peak).

Two buffers, same idea as osc.scope:
  * `phos`  — RGB persistence buffer (decays each frame); the spectrum beam + head live here,
              so switching FTIR->Raman dissolves naturally as the old trace decays.
  * `sharp` — rebuilt every frame (no decay); molecule + all text live here, crisp.
Compose = faint graticule + each buffer + its half-res bloom, then crtfilm.FilmLook.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .gpu import asnumpy, gaussian_filter, xp

# --- scope palette ---
TRACE_G = (150, 255, 170)     # green phosphor trace (spectrum + bonds + main text)
TRACE_DIM = (95, 190, 130)    # dimmer green (secondary text / markers)
HOT = (255, 90, 90)           # red playhead head
BOND_C = (120, 235, 150)      # skeletal bond trace
ROLE = {                      # functional-group symbol colours (per-group trace hue, CPK-ish)
    "C":   (150, 255, 170),   # carbon skeleton  -> green
    "N":   (110, 175, 255),   # nitrogen -> blue
    "O":   (255, 110, 110),   # oxygen   -> red
    "S":   (245, 225, 110),   # sulfur   -> yellow
    "Cl":  (140, 245, 140),   # chlorine -> bright green
    "H":   (200, 245, 210),   # hydrogen -> pale green-white
    "dim": (120, 200, 150),   # CHx groups -> dim green
}

_WINFONTS = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
_FONT_CANDIDATES = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "Ascii_Studio", "assets", "fonts", "CascadiaMono.ttf")),
    os.path.join(_WINFONTS, "CascadiaMono.ttf"),
    os.path.join(_WINFONTS, "CascadiaCode.ttf"),
    os.path.join(_WINFONTS, "consola.ttf"),
]


def _font(px):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def _lorentz(x, x0, amp, fwhm):
    g = fwhm / 2.0
    return amp * (g * g) / ((x - x0) ** 2 + g * g)


def _blur_ds(a, sigma):
    """Half-res additive-glow gaussian (glow is low-frequency -> visually identical, ~4x cheaper)."""
    small = xp.clip(a[::2, ::2], 0, 1e6)
    b = gaussian_filter(small, sigma=(sigma / 2.0, sigma / 2.0, 0))
    up = xp.repeat(xp.repeat(b, 2, axis=0), 2, axis=1)
    return up[:a.shape[0], :a.shape[1]]


def _rot3(ax, ay, az):
    """Rz @ Ry @ Rx — the 3D tumble rotation (ported verbatim from mitragynine_4method.py)."""
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class ChemScope:
    # symbol placement (mirrors styled_molecule / STRUCTURE_PRESET) ------------
    label_push = 1.0           # symbols sit ON their atom (honest placement; halo masks bonds).
    push_power = 1.4           # (>1 pushes symbols radially out — over-pushed far atoms off bonds)
    halo_r = 1.0               # overall scale on the per-glyph knockout ellipse (clears bonds only)
    sym_scale = 0.46           # symbol size as a fraction of the molecule's per-unit scale
    sym_px_min, sym_px_max = 40, 104   # symbol size clamp (at 1080 wide)
    rock_deg = 32.0            # billboard rock amplitude (never edge-on -> symbols stay apart)
    rock_period = 13.0         # seconds per full rock cycle
    # motion="tumble" -> the manim mitragynine_4method 3D tumble (multi-axis, incommensurate freqs
    # + a fixed in-plane spin) instead of the single-axis rock. Set per-molecule via mol["motion"].
    motion = "rock"
    TUMBLE = dict(ampx=32.0, ampy=48.0, ampz=16.0, fx=0.0725, fy=0.0515, fz=0.0322,
                  phx=1.7, phz=0.6, base_rot=-8.0)
    # vertical layout (fractions of h). 2026-07-25 (Ethan): the molecule NAME title moved OUT of
    # the top block down to `title_y`, so it sits directly ABOVE the method title ("FTIR SPECTRUM"
    # / "RAMAN SPECTRUM") — name and method read as one stacked heading. The top block is now just
    # the two metadata subtitles (pulled up into the vacated slot), so the molecule gets a taller
    # gap: `header_bot` (end of the subtitles) -> `method_top` (top of the name title).
    sub1_y = 0.042
    sub2_y = 0.068
    header_bot = 0.098
    title_y = 0.383           # molecule name — one line above the method header
    method_top = 0.378        # molecule zone stops here (just above the name title)
    target_w_frac = 0.50       # default molecule width as a fraction of frame width
    intro_t = 1.5              # CRT power-on / accelerating-sweep intro length (s; 0 = off)

    def __init__(self, mol, w=1080, h=1920, *, decay=0.80, glow_sigma=9.0, glow_gain=0.85,
                 duration=30.0, method_switch=None):
        self.mol = mol
        self.w, self.h = w, h
        self.decay = decay
        self.glow_sigma, self.glow_gain = glow_sigma, glow_gain
        self.duration = duration
        if method_switch is not None:
            self.t_switch = method_switch
        elif "OSC_TSWITCH" in os.environ:                  # demo override (push the FTIR->Raman swap out)
            self.t_switch = float(os.environ["OSC_TSWITCH"])
        else:
            self.t_switch = duration * 0.5
        self.intro_style = os.environ.get("OSC_INTRO_STYLE", "speedup")
        self.phos = xp.zeros((h, w, 3), np.float32)

        s = w / 1080.0
        self._s = s
        self._f_title = _font(max(10, int(64 * s)))
        self._f_sub = _font(max(8, int(30 * s)))
        self._f_sec = _font(max(8, int(34 * s)))
        self._f_axis = _font(max(7, int(26 * s)))
        self._f_num = _font(max(6, int(22 * s)))
        self._f_leg = _font(max(6, int(23 * s)))

        size = mol.get("size", {})
        self._twf = size.get("target_w_frac", self.target_w_frac)
        self.label_push = size.get("label_push", self.label_push)
        self.halo_r = size.get("halo_r", self.halo_r)
        self.motion = mol.get("motion", type(self).motion)   # "rock" (default) or "tumble"

        self._graticule = xp.asarray(self._build_graticule())
        self._mol_geom()
        if "sym_px" in size:                              # explicit symbol size (big molecules)
            sym_px = size["sym_px"] * s
        else:
            sym_px = np.clip(self.sym_scale * self._mscale, self.sym_px_min * s, self.sym_px_max * s)
        self._f_lbl = _font(int(sym_px))
        self._build_spectra()
        # occasional "text flare": brief windows where a few glyphs of a line scramble then
        # return. Applies to the subtitles AND the stretching-mode legend lines (leg0..). The
        # windows are STRING-AGNOSTIC (time + glyph count only) so the same channel works even
        # when the legend text differs between the FTIR and Raman phases.
        chans = ["sub1", "sub2", "sub_ftir"] + [f"leg{i}" for i in range(6)]
        self._flares = {k: self._build_flares(200 + j) for j, k in enumerate(chans)}

    # ---------- text-flare scheduling ----------
    FLARE_CHARS = "!@#$%&*+=/\\|<>[]{}?~^01"

    def _build_flares(self, seed):
        """Sparse, brief flare windows (start, end, glyph_count) — string-agnostic, so a channel
        can flare whatever text is currently on that line."""
        rng = np.random.default_rng(seed)
        ev, t = [], float(rng.uniform(0.8, 2.5))
        while t < self.duration:
            cnt = int(rng.integers(1, 4))                  # 1-3 glyphs
            ev.append((t, t + float(rng.uniform(0.08, 0.16)), cnt))
            t += float(rng.uniform(1.8, 3.6))              # gap (slightly more frequent)
        return ev

    def _flare(self, s, key, t, frame):
        """Return `s` with a couple of glyphs scrambled to random chars while a flare is live.
        Positions are drawn from THIS string's non-space glyphs (stable per window), chars
        re-pick each frame for a shimmer."""
        ev = self._flares.get(key)
        if not ev or not s:
            return s
        live = [e for e in ev if e[0] <= t < e[1]]
        if not live:
            return s
        idx = [i for i, c in enumerate(s) if c != ' ']
        if not idx:
            return s
        chars = list(s)
        fr = np.random.default_rng(frame * 131 + 7)        # per-frame shimmer
        for t0, _, cnt in live:
            wr = np.random.default_rng(int(t0 * 1000))     # stable positions per window
            for p in wr.choice(idx, size=min(cnt, len(idx)), replace=False):
                chars[p] = self.FLARE_CHARS[int(fr.integers(0, len(self.FLARE_CHARS)))]
        return "".join(chars)

    def _scramble(self, s, frac, frame, salt=0):
        """Scramble a FRACTION of `s`'s glyphs to random chars, re-picked each frame (rapid
        flicker). Used across the whole bottom-half text during the FTIR->Raman transition."""
        if frac <= 0.0 or not s:
            return s
        idx = [i for i, c in enumerate(s) if c != ' ']
        if not idx:
            return s
        r = np.random.default_rng(frame * 911 + salt + 3)
        k = max(1, int(round(frac * len(idx))))
        chars = list(s)
        for p in r.choice(idx, size=min(k, len(idx)), replace=False):
            chars[p] = self.FLARE_CHARS[int(r.integers(0, len(self.FLARE_CHARS)))]
        return "".join(chars)

    def _fx(self, s, key, t, frame, scr, salt):
        """Bottom-text effect: heavy transition scramble when scr>0, else the occasional flare."""
        return self._scramble(s, scr, frame, salt) if scr > 0.0 else self._flare(s, key, t, frame)

    # ---------- static faint scope graticule ----------
    def _build_graticule(self):
        img = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        grid, axis = (7, 26, 13), (11, 40, 20)
        step = self.w / 10.0
        for i in range(int(self.w / step) + 2):
            gx = i * step
            d.line([(gx, 0), (gx, self.h)], fill=grid, width=1)
        for j in range(int(self.h / step) + 2):
            gy = j * step
            col = axis if abs(gy - self.h / 2) < 1 else grid
            d.line([(0, gy), (self.w, gy)], fill=col, width=2 if col == axis else 1)
        return np.asarray(img).astype(np.float32)

    # ---------- molecule geometry ----------
    def _mol_geom(self):
        BB = np.asarray(self.mol["BB"], float)
        bw, bh = BB[1][0] - BB[0][0], BB[1][1] - BB[0][1]
        zone_c = (self.header_bot + self.method_top) / 2.0          # centre of the free gap
        zone_half = (self.method_top - self.header_bot) / 2.0
        target_w = self._twf * self.w
        max_h = (2 * zone_half - 0.03) * self.h            # leave ~3% margin top+bottom in the gap
        self._mcx = self.w * 0.5
        self._mcy = zone_c * self.h                        # molecule centred in the gap
        if self.motion == "tumble":
            # scale + centre from the TUMBLE ENVELOPE: sample the whole 3D rotation over the clip,
            # project every atom, and fit the union bbox -> the molecule never clips or drifts as it
            # tumbles (mirrors mitragynine_4method's sampled SCALE/CBASE).
            pts3 = np.array([[v[0], v[1], v[2] if len(v) > 2 else 0.0]
                             for v in self.mol["AT"].values()], float)
            self._cen3 = pts3.mean(axis=0)
            xs, ys = [], []
            for tt in np.linspace(0.0, max(self.duration, 1.0), 96):
                q = (pts3 - self._cen3) @ self._tumble_R(tt).T
                xr, yr = self._base_spin(q[:, 0], q[:, 1])
                xs.append(xr); ys.append(yr)
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            xext, yext = xs.max() - xs.min(), ys.max() - ys.min()
            self._mscale = min(target_w / xext, max_h / yext)
            self._pcx = 0.5 * (xs.max() + xs.min())        # projected-bbox centre (molecule units)
            self._pcy = 0.5 * (ys.max() + ys.min())
        else:
            self._mscale = min(target_w / bw, max_h / bh)
            # pivot on the molecule's own centroid midpoint so it's truly centred (x,y only)
            self._bmid = np.array([(BB[0][0] + BB[1][0]) / 2.0, (BB[0][1] + BB[1][1]) / 2.0])

    def _tumble_R(self, t):
        """3D tumble rotation at time t (multi-axis, incommensurate freqs) — the manim WOB."""
        T = self.TUMBLE
        ax = math.radians(T["ampx"]) * math.sin(2 * math.pi * T["fx"] * t + T["phx"])
        ay = math.radians(T["ampy"]) * math.sin(2 * math.pi * T["fy"] * t)
        az = math.radians(T["ampz"]) * math.sin(2 * math.pi * T["fz"] * t + T["phz"])
        return _rot3(ax, ay, az)

    def _base_spin(self, x, y):
        """Fixed in-plane spin applied after the 3D projection (manim BASE_ROT)."""
        br = math.radians(self.TUMBLE["base_rot"])
        c, s = math.cos(br), math.sin(br)
        return c * x - s * y, s * x + c * y

    def _proj(self, v, t):
        """Project atom `v` (x,y[,z]) to screen px at time `t`. motion='tumble' = the manim 3D
        multi-axis tumble; 'rock' = the single-axis billboard spin (z rotates INTO x)."""
        if self.motion == "tumble":
            p = np.array([v[0], v[1], v[2] if len(v) > 2 else 0.0], float) - self._cen3
            q = self._tumble_R(t) @ p
            x, y = self._base_spin(q[0], q[1])
            return self._mcx + (x - self._pcx) * self._mscale, self._mcy - (y - self._pcy) * self._mscale
        theta = math.radians(self.rock_deg) * math.sin(2 * math.pi * t / self.rock_period)
        z = v[2] if len(v) > 2 else 0.0
        x = (v[0] - self._bmid[0]) * math.cos(theta) + z * math.sin(theta)
        y = (v[1] - self._bmid[1])
        return self._mcx + x * self._mscale, self._mcy - y * self._mscale

    # ---------- spectra (both methods pre-built) ----------
    def _make_curve(self, lines):
        HI, LO = self.mol["ir_window"]
        xg = np.linspace(HI, LO, 900)
        yv = np.zeros_like(xg)
        for cm, inten in lines:
            yv += _lorentz(xg, cm, inten, 16.0)
        sig = yv / (yv.max() or 1.0)
        px = self._sx_l + (HI - xg) / (HI - LO) * (self._sx_r - self._sx_l)
        return dict(xg=xg, sig=sig, px=px)

    def _build_spectra(self):
        self._sx_l, self._sx_r = self.w * 0.06, self.w * 0.94
        self._sy_base = self.h * 0.905
        self._sy_h = self.h * 0.205                       # peak tops reach y = 0.70 h
        self.spectra = {
            "ftir": dict(header="FTIR SPECTRUM", axis="wavenumber  (cm-1)",
                         bands=self.mol["ftir_bands"], **self._make_curve(self.mol["ir_lines"])),
            "raman": dict(header="RAMAN SPECTRUM", axis="Raman shift  (cm-1)",
                          bands=self.mol["raman_bands"], **self._make_curve(self.mol["raman_lines"])),
        }

    def _curve_points(self, sp, t):
        # gentle "live" motion only — small enough that real peak-height changes stay readable
        # (was ±5% breathing + ripple, which bounced the peaks too much to compare methods).
        n = len(sp["sig"])
        breath = 0.005 * np.sin(1.6 * t + 1.2 * np.linspace(0, 6, n))     # ~±0.5% peak height
        ripple = 0.0015 * np.sin(2.4 * np.linspace(0, 9, n) - 0.8 * t)
        y = self._sy_base - (sp["sig"] * (1.0 + breath) + ripple) * self._sy_h
        return np.column_stack([sp["px"], y])

    # ---------- low-level stroking ----------
    def _stroke(self, buf, pts, color, width, gain=1.0, speed_shade=False):
        if len(pts) < 2:
            return
        layer = Image.new("L", (self.w, self.h), 0)
        d = ImageDraw.Draw(layer)
        if speed_shade:
            px = np.asarray(pts, float)
            seg = np.diff(px, axis=0)
            seglen = np.hypot(seg[:, 0], seg[:, 1]) + 1e-3
            inten = np.clip(210.0 * (np.median(seglen) / seglen), 45, 255)
            for i in range(len(seg)):
                d.line([tuple(px[i]), tuple(px[i + 1])], fill=int(inten[i]), width=width)
        else:
            d.line([tuple(p) for p in pts], fill=210, width=width, joint="curve")
        m = xp.asarray(np.asarray(layer, np.float32)) / 255.0
        buf += m[:, :, None] * xp.asarray(np.asarray(color, np.float32)) * gain

    def _dot(self, buf, x, y, r, color, gain=1.0):
        layer = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(layer).ellipse([x - r, y - r, x + r, y + r], fill=255)
        m = xp.asarray(np.asarray(layer, np.float32)) / 255.0
        buf += m[:, :, None] * xp.asarray(np.asarray(color, np.float32)) * gain

    def _knockout(self, buf, x, y, rx, ry):
        """Zero a glyph-sized ELLIPSE of `buf` (the knockout halo). `buf` holds ONLY the bonds,
        so a symbol clears the bond behind it — never the axis grid (a separate buffer)."""
        layer = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(layer).ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)
        m = xp.asarray(np.asarray(layer, np.float32)) / 255.0
        buf *= (1.0 - m[:, :, None])

    def _text(self, buf, s, x, y, color, font, gain=1.0, anchor="la"):
        if gain <= 0.0 or not s:
            return
        layer = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(layer).text((x, y), s, fill=235, font=font, anchor=anchor)
        m = xp.asarray(np.asarray(layer, np.float32)) / 255.0
        buf += m[:, :, None] * xp.asarray(np.asarray(color, np.float32)) * gain

    def _text_size(self, s, font):
        b = font.getbbox(s)
        return b[2] - b[0], b[3] - b[1]

    # ---------- molecule (bonds -> bmol, symbols -> tmol; halo clears bonds only) ----------
    def _draw_molecule(self, bmol, tmol, t):
        AT = self.mol["AT"]
        P = {k: np.array(self._proj(v, t)) for k, v in AT.items()}
        for a, b, dbl in self.mol["BONDS"]:
            pa, pb = P[a], P[b]
            if dbl:
                d = pb - pa
                nrm = np.array([-d[1], d[0]])
                nrm = nrm / (np.linalg.norm(nrm) + 1e-9) * max(3.5, 0.055 * self._mscale)
                for off in (nrm, -nrm):
                    self._stroke(bmol, [pa + off, pb + off], BOND_C, self._bond_w, gain=0.9)
            else:
                self._stroke(bmol, [pa, pb], BOND_C, self._bond_w, gain=0.9)

        # labelled atoms: push radially out, knock the bond out under the glyph, draw the symbol
        ctr = np.mean(np.stack(list(P.values())), axis=0)
        lbl = self.mol["LBL"]
        pts = [P[name] for name, _, _ in lbl]
        dists = [float(np.linalg.norm(p - ctr)) for p in pts]
        dmax = max(dists) or 1.0
        _, lblh = self._text_size("O", self._f_lbl)
        for (name, txt, role), p, d in zip(lbl, pts, dists):
            w = (d / dmax) ** self.push_power
            pos = ctr + (p - ctr) * (1.0 + (self.label_push - 1.0) * w)
            tw, _ = self._text_size(txt, self._f_lbl)
            rx = (0.56 * tw + 0.14 * lblh) * self.halo_r   # snug glyph-sized ellipse (bonds only)
            ry = 0.66 * lblh * self.halo_r
            self._knockout(bmol, pos[0], pos[1], rx, ry)
            self._text(tmol, txt, pos[0], pos[1], ROLE[role], self._f_lbl, gain=1.1, anchor="mm")

    # ---------- text / legend / markers (all ABOVE the plot box) ----------
    def _draw_text(self, tmol, t, sp, alpha, frame, scr=0.0):
        m, h, gx = self.mol, self.h, self.w * 0.06
        # TOP metadata block (identical across methods) — occasional subtitle flare.
        self._text(tmol, self._flare(m["sub1"], "sub1", t, frame), gx, h * self.sub1_y, TRACE_G,
                   self._f_sub, gain=0.95)
        self._text(tmol, self._flare(m["sub2"], "sub2", t, frame), gx, h * self.sub2_y, TRACE_DIM,
                   self._f_sub, gain=0.9)
        # Molecule NAME, stacked directly on top of the method title below it.
        self._text(tmol, m["title"], gx, h * self.title_y, TRACE_G, self._f_title, gain=1.15)
        # BOTTOM-half method text. During the FTIR->Raman transition (scr>0) ALL of it rapidly
        # ASCII-scrambles at full opacity as the swap indication (instead of a fade out/in).
        self._text(tmol, self._fx(sp["header"], None, t, frame, scr, 1), gx, h * 0.440,
                   TRACE_G, self._f_sec, gain=1.05 * alpha)
        self._text(tmol, self._fx(m["sub_ftir"], "sub_ftir", t, frame, scr, 2), gx, h * 0.478,
                   TRACE_DIM, self._f_sub, gain=0.85 * alpha)
        for i, (name, cm) in enumerate(sp["bands"]):
            leg = self._fx(f"{i+1}. {name}  {cm}", f"leg{i}", t, frame, scr, 10 + i)
            self._text(tmol, leg, gx, h * (0.512 + 0.0225 * i),
                       TRACE_DIM, self._f_leg, gain=0.85 * alpha)
        self._text(tmol, self._fx(sp["axis"], None, t, frame, scr, 3), self.w * 0.5, h * 0.945,
                   TRACE_DIM, self._f_axis, gain=0.9 * alpha, anchor="ma")
        # numeric markers in the CLEAR gap just above each peak (never over text or trace body)
        HI, LO = m["ir_window"]
        placed = []                                       # (x, y) of numbers, for anti-collision
        for i in range(len(sp["bands"])):
            cm = sp["bands"][i][1]
            px = self._sx_l + (HI - cm) / (HI - LO) * (self._sx_r - self._sx_l)
            sig = float(np.interp(cm, sp["xg"][::-1], sp["sig"][::-1]))
            peak_y = self._sy_base - sig * self._sy_h
            ny = peak_y - 34 * self._s
            for (ox, oy) in placed:                        # lift if crowding a neighbour
                if abs(ox - px) < 34 * self._s and abs(oy - ny) < 26 * self._s:
                    ny = oy - 26 * self._s
            placed.append((px, ny))
            self._stroke(tmol, [(px, peak_y - 6 * self._s), (px, ny + 12 * self._s)],
                         TRACE_DIM, max(1, self._bond_w - 2), gain=0.6 * alpha)
            self._text(tmol, str(i + 1), px, ny, TRACE_G, self._f_num, gain=1.0 * alpha, anchor="md")

    # ---------- intro: CRT power-on + accelerating beam sweep ----------
    @staticmethod
    def _flick(t):
        """Global brightness of a CRT flicking on in the first ~0.07 s (black -> stutter -> lit)."""
        if t < 0.012:
            return 0.0
        if t < 0.028:
            return 1.30
        if t < 0.046:
            return 0.28
        if t < 0.066:
            return 1.12
        return 1.0

    # intro sweep styles: an INVISIBLE vertical line sweeps L->R; only where it crosses an object
    # is that object lit, into the phosphor. As the sweep accelerates, the moving lit slice smears
    # (persistence) into a solid line. 'sweep' = accelerating repeat passes; 'drawon' = one
    # accelerating left->right wipe.
    # tuned for intro_t = 1.5 s (45 frames @30): a bright slice sweeps L->R, faster each pass,
    # smearing (phosphor) into a solid line by the end. K*p ~= T/dt (=45) so it solidifies near the end.
    INTRO_STYLES = {
        "speedup":      dict(mode="sweep", p=2.0, K=24.0),  # slow first trace -> accelerating -> solid
        "speedup_soft": dict(mode="sweep", p=1.6, K=30.0),  # gentler accel, more even sweeping
        "drawon":       dict(mode="drawon", q=1.3),          # single accelerating left->right wipe
    }

    def _intro_bonds(self, buf, P, x_lo, x_hi, gain):
        """Draw each bond's portion whose screen-x lies in [x_lo, x_hi] (the lit sweep slice)."""
        for a, b, dbl in self.mol["BONDS"]:
            pa, pb = P[a], P[b]
            if pa[0] > pb[0]:
                pa, pb = pb, pa
            dx = pb[0] - pa[0]
            if abs(dx) < 1e-6:
                if not (x_lo <= pa[0] <= x_hi):
                    continue
                s0, s1 = pa, pb
            else:
                t0 = max(0.0, (x_lo - pa[0]) / dx)
                t1 = min(1.0, (x_hi - pa[0]) / dx)
                if t1 <= t0:
                    continue
                s0, s1 = pa + t0 * (pb - pa), pa + t1 * (pb - pa)
            if dbl:
                d = s1 - s0
                nrm = np.array([-d[1], d[0]])
                nrm = nrm / (np.linalg.norm(nrm) + 1e-9) * max(3.5, 0.055 * self._mscale)
                for off in (nrm, -nrm):
                    self._stroke(buf, [s0 + off, s1 + off], BOND_C, self._bond_w, gain=gain)
            else:
                self._stroke(buf, [s0, s1], BOND_C, self._bond_w, gain=gain)

    def _intro_band(self, buf, pts, P, x_lo, x_hi, gain):
        """Light every object (spectrum curve, baseline, bonds) only over screen-x in [x_lo, x_hi]."""
        px = pts[:, 0]
        i0, i1 = int(np.searchsorted(px, x_lo)), int(np.searchsorted(px, x_hi))
        if i1 - i0 >= 2:
            self._stroke(buf, pts[i0:i1], TRACE_G, self._bond_w, gain=gain, speed_shade=True)
        bl_lo, bl_hi = max(self._sx_l, x_lo), min(self._sx_r, x_hi)
        if bl_hi > bl_lo:
            self._stroke(buf, [(bl_lo, self._sy_base), (bl_hi, self._sy_base)],
                         TRACE_G, max(1, self._bond_w - 1), gain=0.16)
        self._intro_bonds(buf, P, x_lo, x_hi, gain)

    def _intro_text(self, tmol, t, frame, scr, P):
        """All text + molecule symbols ASCII-flicker in (scramble ramping to clean)."""
        m, h, gx = self.mol, self.h, self.w * 0.06
        sp = self.spectra["ftir"]
        self._text(tmol, self._scramble(m["sub1"], scr, frame, 2), gx, h * self.sub1_y, TRACE_G, self._f_sub, 0.95)
        self._text(tmol, self._scramble(m["sub2"], scr, frame, 3), gx, h * self.sub2_y, TRACE_DIM, self._f_sub, 0.9)
        self._text(tmol, self._scramble(m["title"], scr, frame, 1), gx, h * self.title_y, TRACE_G, self._f_title, 1.15)
        self._text(tmol, self._scramble(sp["header"], scr, frame, 4), gx, h * 0.440, TRACE_G, self._f_sec, 1.05)
        self._text(tmol, self._scramble(m["sub_ftir"], scr, frame, 5), gx, h * 0.478, TRACE_DIM, self._f_sub, 0.85)
        for idx, (name, cm) in enumerate(sp["bands"]):
            self._text(tmol, self._scramble(f"{idx+1}. {name}  {cm}", scr, frame, 10 + idx),
                       gx, h * (0.512 + 0.0225 * idx), TRACE_DIM, self._f_leg, 0.85)
        self._text(tmol, self._scramble(sp["axis"], scr, frame, 6), self.w * 0.5, h * 0.945,
                   TRACE_DIM, self._f_axis, 0.9, anchor="ma")
        for name, txt, role in m["LBL"]:
            pos = P[name]
            self._text(tmol, self._scramble(txt, scr, frame, ord(name[0]) + 20), pos[0], pos[1],
                       ROLE[role], self._f_lbl, 1.05, anchor="mm")

    def _render_intro(self, i, fps, t):
        """First `intro_t` seconds: screen flicks on, an INVISIBLE vertical sweep lights each object
        only where it crosses; as the sweep accelerates the moving lit slice smears (phosphor) into
        a solid line — the traces "speed up" until solid. Text flickers in. Reaches normal by intro_t."""
        T = self.intro_t
        self._bond_w = max(2, int(4 * self._s))
        cfg = self.INTRO_STYLES.get(self.intro_style, self.INTRO_STYLES["speedup"])

        self.phos *= self.decay
        pts = self._curve_points(self.spectra["ftir"], t)
        P = {k: np.array(self._proj(v, t)) for k, v in self.mol["AT"].items()}

        if cfg["mode"] == "drawon":                        # one accelerating left->right wipe
            xb = min(1.0, (t / T) ** cfg["q"]) * self.w
            self._intro_band(self.phos, pts, P, self._sx_l - 2, xb, 0.5)
        else:                                              # accelerating repeat sweeps
            p, K = cfg["p"], cfg["K"]
            ph0, ph1 = K * (t / T) ** p, K * ((t + 1.0 / fps) / T) ** p
            if ph1 - ph0 >= 1.0:                            # >= a full pass this frame -> whole object lit
                self._intro_band(self.phos, pts, P, -2, self.w + 2, 0.5)
            else:
                a, b = ph0 - math.floor(ph0), ph1 - math.floor(ph1)
                if a <= b:
                    self._intro_band(self.phos, pts, P, a * self.w, b * self.w, 0.6)
                else:                                       # wrapped past the right edge
                    self._intro_band(self.phos, pts, P, a * self.w, self.w + 2, 0.6)
                    self._intro_band(self.phos, pts, P, -2, b * self.w, 0.6)

        tmol = xp.zeros((self.h, self.w, 3), np.float32)
        if t > 0.066:                                      # text + symbols flicker in once screen is lit
            t_done = 0.7 * T                               # text resolves by 70% in -> readable at the end
            scr = max(0.0, (t_done - t) / (t_done - 0.066)) * 0.9
            self._intro_text(tmol, t, i, scr, P)

        out = (self._graticule + self.phos + _blur_ds(self.phos, self.glow_sigma) * self.glow_gain
               + tmol + _blur_ds(tmol, self.glow_sigma) * self.glow_gain) * self._flick(t)
        return asnumpy(xp.clip(out, 0, 255).astype(xp.uint8))

    # ---------- one frame ----------
    def render_frame(self, i, fps):
        self._bond_w = max(2, int(4 * self._s))
        t = i / fps
        if self.intro_t > 0 and t < self.intro_t:
            return self._render_intro(i, fps, t)

        # method phase: FTIR, a quick transition window (cur=None), or RAMAN. The transition is
        # signalled by ASCII-scrambling ALL the bottom text (scr), NOT a fade — text stays lit.
        fade = 0.5
        scr = 0.0
        if t < self.t_switch - fade:
            cur = "ftir"
        elif t <= self.t_switch + fade:
            cur = None
        else:
            cur = "raman"

        self.phos *= self.decay
        # --- spectrum trace(s) into persistence ---
        if cur is not None:
            sp = self.spectra[cur]
            self._stroke(self.phos, self._curve_points(sp, t), TRACE_G, self._bond_w,
                         gain=0.42, speed_shade=True)
            active = sp
        else:                                              # transition: cross-dissolve the traces
            f = (t - (self.t_switch - fade)) / (2 * fade)  # 0..1
            self._stroke(self.phos, self._curve_points(self.spectra["ftir"], t), TRACE_G,
                         self._bond_w, gain=0.42 * (1 - f), speed_shade=True)
            self._stroke(self.phos, self._curve_points(self.spectra["raman"], t), TRACE_G,
                         self._bond_w, gain=0.42 * f, speed_shade=True)
            active = self.spectra["ftir" if f < 0.5 else "raman"]
            scr = 0.6                                       # heavy scramble, tapering at the ends
            if f < 0.12:
                scr *= f / 0.12
            elif f > 0.78:
                scr *= max(0.0, (1.0 - f) / 0.22)
        # baseline axis (steady) + a red LINE segment tracing along the active curve (a bright
        # leading run of the trace itself, not a dot) — the phosphor decay gives it a short comet
        # tail so it reads as a red line gliding over the green.
        self._stroke(self.phos, [(self._sx_l, self._sy_base), (self._sx_r, self._sy_base)],
                     TRACE_G, max(1, self._bond_w - 1), gain=0.16)
        pts = self._curve_points(active, t)
        k = int(((t * 0.55) % 1.0) * (len(pts) - 1))
        win = max(3, int(0.032 * len(pts)))                # length of the red leading run
        lo, hi = max(0, k - win), min(len(pts), k + 1)
        self._stroke(self.phos, pts[lo:hi], HOT, self._bond_w, gain=0.7)

        # --- bonds + text in separate buffers so the symbol halo clears ONLY bonds ---
        bmol = xp.zeros((self.h, self.w, 3), np.float32)   # skeletal bonds (knocked out under glyphs)
        tmol = xp.zeros((self.h, self.w, 3), np.float32)   # symbols + all text (crisp, on top)
        self._draw_molecule(bmol, tmol, t)
        self._draw_text(tmol, t, active, 1.0, i, scr)     # alpha=1 always; scr signals the swap

        out = (self._graticule                              # grid: never knocked out
               + self.phos + _blur_ds(self.phos, self.glow_sigma) * self.glow_gain
               + bmol + _blur_ds(bmol, self.glow_sigma) * self.glow_gain
               + tmol + _blur_ds(tmol, self.glow_sigma) * self.glow_gain)
        return asnumpy(xp.clip(out, 0, 255).astype(xp.uint8))
