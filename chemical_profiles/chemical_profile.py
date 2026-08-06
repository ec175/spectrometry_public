"""
chemical_profile.py — the vertical "chemical profile" format, as one runnable scene file.

A chemical profile is a 9:16 short built in three phases:

  Phase 1  the molecule eases in over the top half, NUMBER-labelled for NMR
           assignment, and spins one full turn above its 13C and 1H NMR spectra.
           Peak markers on both spectra carry the same numbers as the atoms.
  Phase 2  the NMR spectra cross-fade to a compact FTIR trace, and the molecule
           reverts from numbers to atom symbols.
  Phase 3  each localised IR vibration mode plays in turn -- the motion falls off
           sharply in the surrounding bonds -- while a cursor and numbered band
           markers track the matching FTIR peak.

EVERY molecule-specific value is a class attribute, so a new molecule is just a
subclass that overrides them. `VerticalProfile_Aspirin` at the bottom of this file
is the worked example: read it top to bottom and you have the whole data contract.

    manim -qh -r 1080,1920 chemical_profile.py VerticalProfile_Aspirin

See GUIDE.md for installation, every flag, and how to author a new molecule.

----------------------------------------------------------------------------------
This file is choreography only. The reusable objects, the palette, the molecule
builders and the spectroscopy maths all live in `spectro_lib.py`, imported below
with `from spectro_lib import *` (which also applies the canonical dark background
and camera config).
"""
import os

import numpy as np
from manim import *

from spectro_lib import *


class VerticalProfile(Scene):
    """Reusable vertical 'profile' format (9:16). TOP half = molecule + a structure title/subtitle
    tag (bottom-left, same house format as the Structures folder); BOTTOM half = spectra.
    Phase 1: the molecule (large preset, width `target_w_frac` of screen, centred top half,
    NUMBER-LABELLED for NMR assignment) eases in and spins one full turn over ¹³C + ¹H NMR
    spectra whose peaks carry slanted markers + numbers matching the atom numbers on the molecule.
    Phase 2: the spectra cross-fade to a compact FTIR trace ('… Spectra' title) and the molecule
    reverts to atom symbols. Phase 3: the localised IR vibration modes play in turn — the motion
    drops off sharply in surrounding bonds — while a cursor + numbered band markers track the
    matching FTIR peaks.

    EVERY molecule-specific value lives in the class attributes below, so a new molecule is just a
    subclass that overrides them -- see VerticalProfile_Aspirin at the bottom of this file for a
    fully worked one, and GUIDE.md section 7 for what each attribute means.

    ⚠️  BATCHING NEW MOLECULES — MUST VERIFY PER FILE (these are NOT auto-derived):
        • subtitle / subtitle2 / subtitle_ftir / subtitle2_ftir / solvent  — every value is
          molecule-specific text; copy-paste from caffeine WILL be wrong. Update them all.
        • nmrc / nmrh  — the ¹³C and ¹H peak lists AND their assignment numbers must match THIS
          molecule's real spectrum and the LBL_NUM atom numbering. A wrong/stale assignment is a
          factual error on screen. Check each shift against a reference before rendering.
        • vib_modes (bands + wn ranges) and ir_lines must be this molecule's IR, not the
          template's.
    Render: manim -qh -r 1080,1920 chemical_profile.py VerticalProfile_Aspirin"""

    # ---- per-molecule config (override in a subclass) ----
    title = "MOLECULE"
    # two phase-swapped subtitle pairs below the title: line 1 = QUANTITATIVE values (white),
    # line 2 = QUALITIES (shade colour). The NMR pair shows during phase 1, then cross-fades to
    # the FTIR pair ("additional info") when the spectra switch.
    subtitle = ""                # NMR phase, quantitative (formula · MW · mp …)
    subtitle2 = ""               # NMR phase, qualities (fallback if no solvent set)
    solvent = ""                 # NMR acquisition solvent (deuterated -> "d#-Solvent", e.g. d6-DMSO)
    spin_rate = ""               # NMR sample spin rate, e.g. "20 Hz"
    subtitle_ftir = ""           # FTIR phase, quantitative
    subtitle2_ftir = ""          # FTIR phase, qualities
    tag_title_fs = 70            # bottom-left tag: title / subtitle font sizes + height
    tag_sub_fs = 30
    tag_y = 0.96
    curve_shade = "green"        # SHADE family for the light->dark gradient under every curve
    AT = {}                      # atom -> (x, y) build-space coords
    BONDS = []                   # (a, b, double?) skeletal bonds
    LBL_ATOM = []                # (atom, text, color) symbol labels (IR phase)
    LBL_NUM = []                 # (atom, text, color) numbered labels (NMR phase)
    BB = [[-2.95, -2.35, 0], [2.4, 2.4, 0]]   # invisible bbox -> fixed scale (no breathing)
    target_w_frac = 0.50
    mol_height = 2.0
    label_fs = 16
    label_font_size = 30
    nmrc = []                    # ¹³C: (ppm, inten, num)
    nmrh = []                    # ¹H:  (ppm, inten, num)
    nmrc_window = (200.0, 0.0)
    nmrh_window = (10.0, 0.0)
    nmrc_width = 1.6
    nmrh_width = 0.08
    ir_lines = []                # FTIR: (cm⁻¹, inten)
    ir_window = (1900.0, 600.0)
    ir_fwhm = 16.0
    vib_amp = 0.35               # vibration amplitude + falloff radius (build units)
    vib_sigma = 0.65
    # ordered sweep list: each dict(name=, band=cm⁻¹, drivers=[(atom, kind, ref, sign), …]);
    # kind 'stretch' = along atom→ref, 'perp' = perpendicular to it.
    vib_modes = []
    # ---- opt-in audio-reactive TOP "light bar" (default off -> unchanged flat lightbar) ----
    # When set to a [n_frames, n_bars] .npy (built by make_audio_bars.py), the top bar's flat
    # trace becomes audio-reactive EQ bars in the same palette. The BOTTOM spectra bar is never
    # touched. Read by REAL time, so render fps is irrelevant. Mux audio with the SAME --start.
    audio_bars = None
    audio_fps = 30

    def construct(self):
        fw, fh = config.frame_width, config.frame_height
        ROUND = LineJointType.ROUND
        XL = -fw * 0.42                                 # common left edge (header + bottom-half tags)
        _msty = {k: STRUCTURE_PRESET[k] for k in
                 ("label_push", "push_power", "base_label_fs", "label_halo",
                  "halo_color", "halo_r", "min_double_sep")}
        TARGET_W = self.target_w_frac * fw
        TOP_C = [0, fh / 4, 0]
        LIGHT, DARK = SHADE[self.curve_shade]         # light->dark gradient for every curve
        GREEN = LIGHT                                  # accent colour (mode labels, qualities)

        # ---- subtle, fluid spectral motion (small; never shifts a trace far) ----
        gt = ValueTracker(0.0)                         # global seconds clock (runs every phase)
        gt.add_updater(lambda m, dt: m.increment_value(dt))

        def ripple_y(xs, t):
            """Low-amplitude travelling baseline waves (sum of sines -> organic, not a clean
            sine); a slow swell makes them come and go rather than churn constantly."""
            swell = 0.42 + 0.58 * (0.5 + 0.5 * np.sin(0.21 * t))      # 0.42..1.0
            r = (0.030 * np.sin(3.0 * xs - 1.25 * t)
                 + 0.018 * np.sin(5.3 * xs + 0.85 * t + 1.1)
                 + 0.011 * np.sin(8.1 * xs - 0.55 * t + 2.0))
            return swell * r

        def breath(xs, t):
            """±5% peak-height 'breathing', phase rolling along x so peaks look fluid not rigid."""
            return 0.05 * np.sin(2.5 * t + 1.7 * xs)

        AT, BONDS, BB = self.AT, self.BONDS, self.BB
        VA, SIGMA = self.vib_amp, self.vib_sigma
        P_BASE = {k: np.array([v[0], v[1], 0.0]) for k, v in AT.items()}

        def nrm(v):
            n = np.linalg.norm(v); return v / n if n > 1e-9 else v

        def perturb(drivers_spec, ph):
            """Displace each driver atom by ph·VA (along its bond for 'stretch', perpendicular for
            'perp'), then spread to every atom with a SHARP Gaussian falloff (sigma SIGMA) so
            surrounding bonds barely move and the bulk stays put."""
            drivers = []
            for atom, kind, ref, sign in drivers_spec:
                d = nrm(P_BASE[atom] - P_BASE[ref])
                vec = d if kind == 'stretch' else np.array([-d[1], d[0], 0.0])
                drivers.append((atom, sign * ph * VA * vec))
            P = {}
            for k, pos in P_BASE.items():
                disp = np.zeros(3)
                for da, dv in drivers:
                    disp = disp + dv * np.exp(-(np.linalg.norm(pos - P_BASE[da]) / SIGMA) ** 2)
                P[k] = pos + disp
            return P

        sc_box = [1.0]                                # styled_mol reads the scale late

        def styled_mol(P, label_set, angle=0.0):
            mobs = [Line(BB[0], BB[1]).set_opacity(0)]   # fixed BB -> stable scale (no breathing)
            bonds = []                                    # the skeleton, identical across label sets
            _id = [0]
            for a, b, dbl in BONDS:
                pa, pb = P[a], P[b]
                if dbl:
                    pn = np.array([-(pb - pa)[1], (pb - pa)[0], 0.0])
                    pn = pn / (np.linalg.norm(pn) + 1e-9) * 0.06
                    pid = _id[0]; _id[0] += 1
                    for off in (pn, -pn):
                        ln = Line(pa + off, pb + off, color=MOL_DIM, stroke_width=3)
                        ln._double_pair = pid; mobs.append(ln); bonds.append(ln)
                else:
                    ln = Line(pa, pb, color=MOL_DIM, stroke_width=3)
                    mobs.append(ln); bonds.append(ln)
            for name, txt, col in label_set:
                mobs.append(Text(txt, font_size=self.label_font_size, color=col).move_to(P[name]))
            g = VGroup(*mobs)
            styled_molecule(g, mol_height=self.mol_height, label_fs=self.label_fs, **_msty)
            g.scale(sc_box[0])
            if abs(angle) > 1e-9:
                billboard_rotate(g, angle, axis_point=g.get_center(), axis_dir=UP)
            # centre on the BOND SKELETON (identical for every label set) so swapping numbers <-> CH₃
            # symbols (which are wider) does NOT nudge the molecule between the NMR and FTIR phases
            g.shift(np.array(TOP_C, float) - VGroup(*bonds).get_center())
            return g

        ref = styled_mol(P_BASE, self.LBL_ATOM)
        sc_box[0] = TARGET_W / ref.width             # uniform scale -> width = target_w_frac

        # ----- structure title + phase-swapped subtitles, bottom-left of the TOP half -----
        # line 1 = quantitative (WHITE), line 2 = qualities (accent). The NMR pair cross-fades to
        # the FTIR pair when the spectra switch.
        title_t = Text(self.title, font_size=self.tag_title_fs, color=INK, weight=BOLD)

        def sub_block(quant, qual):
            items = []
            if quant:
                items.append(Text(quant, font_size=self.tag_sub_fs, color=WHITE))
            if qual:
                items.append(Text(qual, font_size=self.tag_sub_fs, color=GREEN))
            return VGroup(*items).arrange(DOWN, buff=0.10, aligned_edge=LEFT)

        # NMR phase line 2 = the acquisition solvent (+ optional spin rate); falls back to qualities
        _nmr_bits = [b for b in (self.solvent, (f"spin {self.spin_rate}" if self.spin_rate else "")) if b]
        nmr_qual = "  ·  ".join(_nmr_bits) if _nmr_bits else self.subtitle2
        sub_nmr = sub_block(self.subtitle, nmr_qual)
        sub_ftir = sub_block(self.subtitle_ftir, self.subtitle2_ftir)
        head = VGroup(title_t, sub_nmr).arrange(DOWN, buff=0.13, aligned_edge=LEFT)
        head.move_to([-fw * 0.42 + head.width / 2, self.tag_y, 0])
        if len(sub_ftir):                              # park the FTIR pair under the title (hidden)
            sub_ftir.next_to(title_t, DOWN, buff=0.13, aligned_edge=LEFT)

        # ----- NMR panels (bottom half) with numbered, slanted peak markers -----
        xLn, xRn = -fw * 0.42, fw * 0.42

        def peak_markers(plist, base_lift=0.42, aura=0.50, slant=0.80, cluster_gap=0.60,
                         fs=30, halo=False):
            # plist: (px, peak_top, num). Peaks closer than `cluster_gap` form a cluster; the
            # cluster's numbers spread symmetrically about its centre, so the OUTER leaders slant
            # LEFT and RIGHT (away from the crowd) while isolated peaks get a straight leader. The
            # bigger the horizontal slant, the longer the leader (so the angle stays readable).
            # halo=True drops a BG disk behind each number so a crossing line is knocked out.
            plist = sorted(plist, key=lambda t: t[0])
            n = len(plist)
            if n == 0:
                return VGroup()
            clusters, cur = [], [0]
            for i in range(1, n):
                if plist[i][0] - plist[i - 1][0] < cluster_gap:
                    cur.append(i)
                else:
                    clusters.append(cur); cur = [i]
            clusters.append(cur)
            tx = [0.0] * n
            for cl in clusters:
                centre = sum(plist[i][0] for i in cl) / len(cl)
                k = len(cl)
                for j, i in enumerate(cl):
                    tx[i] = centre + (j - (k - 1) / 2.0) * aura      # spread -> outer slant L/R
            g = VGroup()
            for i, (px, pty, num) in enumerate(plist):
                x = tx[i]
                lift = base_lift + slant * abs(x - px) + (0.0 if i % 2 == 0 else 0.14)
                tip = [x, pty + lift, 0]
                g.add(Line([px, pty + 0.04, 0], tip, color=INK, stroke_width=2.5))
                num_t = Text(num, font_size=fs, color=INK).move_to([x, pty + lift + 0.22, 0])
                if halo:
                    g.add(Circle(radius=0.92 * num_t.height, color=BG, fill_color=BG,
                                 fill_opacity=1.0, stroke_width=0).move_to(num_t.get_center()))
                g.add(num_t)
            return g

        def peak_markers_grouped(plist, x_lo, x_hi, peak_lift=0.24, leader_h=0.42,
                                 group_gap=0.28, max_g=6, min_dx=0.44, fs=22, halo=True):
            # Crowd-friendly markers for the ¹³C panel. THREE ideas, all aimed at the dense
            # steroid aliphatic envelope:
            #   1) GROUP peaks whose screen-x are within `group_gap` into ONE annotation (a big
            #      proximity run is chunked into pieces of <= max_g). Each annotation lists its
            #      carbon numbers as "#,#" per line, STACKED in rows of two — so a 4-carbon group
            #      reads "1,3 / 7,8" and stays narrow instead of one long overflowing string.
            #   2) KINKED leaders: a vertical stub rises from each group's tallest peak to a single
            #      shared rail height S, then ONE diagonal runs out to the label at a single shared
            #      row height R.
            #   3) NON-CROSSING by construction: every diagonal joins the same two parallel lines
            #      (y=S -> y=R) with start- and end-x in the SAME left-to-right order, so no two can
            #      cross; label x-positions are packed monotonically (>= min_dx apart) within bounds.
            plist = sorted(plist, key=lambda t: t[0])
            if not plist:
                return VGroup()
            prox = [[plist[0]]]
            for p in plist[1:]:
                if p[0] - prox[-1][-1][0] <= group_gap:
                    prox[-1].append(p)
                else:
                    prox.append([p])
            groups = []                                     # split each proximity run into <= max_g
            for grp in prox:
                for k in range(0, len(grp), max_g):
                    groups.append(grp[k:k + max_g])

            def _key(s):
                t = s.lstrip("-")
                return int(s) if t.isdigit() else 10 ** 9
            gx = [sum(p[0] for p in grp) / len(grp) for grp in groups]
            gtop = [max(p[1] for p in grp) for grp in groups]
            gnums = [sorted((p[2] for p in grp), key=_key) for grp in groups]
            n = len(groups)
            S = max(gtop) + peak_lift                       # common stub rail (above all peaks)
            R = S + leader_h                                # common label row (leader tips land here)
            # Label x-positions: spread EVENLY across the full panel width in peak-x order. Steroid
            # ¹³C has a big empty mid-field (≈60–150 ppm), so fanning the labels across it — rather
            # than stacking them above their own crowded peaks — gives every label room. Order is
            # preserved, so the diagonals (all S->R) still cannot cross. Even spacing >= min_dx
            # unless there are too many groups to fit (then it compresses uniformly).
            if n == 1:
                lx = [min(max(gx[0], x_lo), x_hi)]
            else:
                span = x_hi - x_lo
                step = max(min_dx, span / n)
                base = (x_lo + x_hi) / 2.0 - step * (n - 1) / 2.0
                lx = [base + step * i for i in range(n)]
            g = VGroup()
            for i in range(n):
                nums = gnums[i]
                rows = [",".join(nums[k:k + 2]) for k in range(0, len(nums), 2)]   # "#,#" per line
                lbl = VGroup(*[Text(r, font_size=fs, color=INK) for r in rows]).arrange(DOWN, buff=0.04)
                lbl.move_to([lx[i], R + lbl.height / 2.0, 0])    # sit just above the leader tip
                g.add(Line([gx[i], gtop[i] + 0.04, 0], [gx[i], S, 0], color=INK, stroke_width=2.5))
                g.add(Line([gx[i], S, 0], [lx[i], R, 0], color=INK, stroke_width=2.5))
                if halo:
                    g.add(Rectangle(width=lbl.width + 0.13, height=lbl.height + 0.10, color=BG,
                                    fill_color=BG, fill_opacity=1.0, stroke_width=0).move_to(lbl.get_center()))
                g.add(lbl)
            return g

        GLOW = interpolate_color(ManimColor(LIGHT), WHITE, 0.30)   # bright tint for the glow haze

        def baseline_glow(base_y, xl, xr, height, layers=14, mcols=8):
            """A soft glow rising from a trace's baseline. BRIGHT at the baseline, fading to nothing
            over `height` (vertical). FULL strength across the graph (one seamless band), then
            CONTINUES past the left/right bounds and trails off — dropping toward 0 just beyond the
            screen edges (horizontal margins). Static; peaks rise THROUGH it and it's gone by the
            peak tops. A NEGATIVE `height` makes the glow face DOWNWARD (the top mirror bar)."""
            g = VGroup()
            cx = (xl + xr) / 2.0
            gh = (xr - xl) / 2.0                                # graph half-width
            ext = gh * 1.55                                     # extend past the screen edges
            band_h = abs(height) / layers * 1.9
            mcw = (ext - gh) / mcols
            for k in range(layers):
                frac = k / (layers - 1)
                vop = 0.17 * (1.0 - frac) ** 1.9               # vertical falloff
                if vop < 0.004:
                    continue
                yy = base_y + frac * height
                # one seamless full-strength band across the graph
                g.add(Rectangle(width=2 * gh, height=band_h, fill_color=GLOW, fill_opacity=vop,
                                stroke_width=0).move_to([cx, yy, 0]))
                # left + right margins continue the glow past the bounds and trail off
                for side in (-1, 1):
                    for j in range(mcols):
                        d = gh + (ext - gh) * (j + 0.5) / mcols
                        hop = max(0.0, 1.0 - (d - gh) / (ext - gh)) ** 1.4
                        op = vop * hop
                        if op < 0.004:
                            continue
                        g.add(Rectangle(width=mcw * 1.08, height=band_h, fill_color=GLOW,
                                        fill_opacity=op, stroke_width=0).move_to([cx + side * d, yy, 0]))
            return g

        def make_curve(xs, base_y, sig, h):
            """Build VGroup(fill, stroke) for a trace at time gt, with fluid breathing + a
            rippling baseline. The gradient FILL uses the SAME dense points as the stroke, so its
            slices hug the curve exactly (no shading spilling past the peaks)."""
            n = len(xs)
            def build():
                t = gt.get_value()
                y = base_y + sig * h * (1.0 + breath(xs, t)) + ripple_y(xs, t)
                pts = [[xs[i], y[i], 0] for i in range(n)]
                fill = horizontal_gradient_fill(pts, base_y, LIGHT, DARK, xs[0], xs[-1], opacity=0.85)
                stroke = VMobject(color=LIGHT, stroke_width=4, joint_type=ROUND)
                stroke.set_points_as_corners(pts)
                return VGroup(fill, stroke)
            return build

        def nmr_panel(peaks, hi, lo, base_y, h, width, xlabel, npts=440, grouped=False):
            def X(p):
                return xLn + ((hi - p) / (hi - lo)) * (xRn - xLn)
            xg = np.linspace(hi, lo, npts); yv = np.zeros_like(xg)
            for ppm, inten, _ in peaks:
                yv = yv + lorentzian(xg, ppm, inten, width)
            sig = yv / (yv.max() or 1.0)
            xs = np.array([X(p) for p in xg])
            glow = baseline_glow(base_y, xLn, xRn, h * 0.60)        # glow behind the trace
            axis = axis_line([xLn, base_y, 0], [xRn, base_y, 0], tip=False)
            maxi = max(i for _, i, _ in peaks)
            tops = [(X(ppm), base_y + (inten / maxi) * h, num) for ppm, inten, num in peaks]
            if grouped:
                # ¹³C: group nearby peaks into one comma-joined label + non-crossing kinked leaders
                mk = peak_markers_grouped(tops, x_lo=xLn + 0.12, x_hi=xRn - 0.12)
            else:
                # ¹H: slanted leaders that fan by local crowding; halos keep numbers readable
                mk = peak_markers(tops, base_lift=0.38, aura=0.62, slant=0.52, halo=True)
            lab = Text(xlabel, font_size=30, color=WHITE).move_to([0, base_y - 0.32, 0])
            return VGroup(glow, axis, mk, lab), make_curve(xs, base_y, sig, h)

        hiC, loC = self.nmrc_window
        hiH, loH = self.nmrh_window
        # peak height is a FRACTION of the frame; halved from the (too-big) ~3x version
        cstat, cbuild = nmr_panel(self.nmrc, hiC, loC, -fh * 0.205, fh * 0.082, self.nmrc_width, "¹³C   (ppm)", grouped=True)
        hstat, hbuild = nmr_panel(self.nmrh, hiH, loH, -fh * 0.390, fh * 0.082, self.nmrh_width, "¹H   (ppm)")
        # decorative top mirror bar: a glowing axis like the NMR baseline but with its glow facing
        # DOWN, plus a FLAT trace held at 0 (no peaks). Static position for the whole animation.
        # Its glow has an NMR size and an FTIR size; it cross-fades from one to the other when the
        # spectra switch, so the top bar mirrors the FTIR lighting too.
        TOPBAR_Y = fh * 0.448
        xs_top = np.linspace(xLn, xRn, 200)
        topbuild = make_curve(xs_top, TOPBAR_Y, np.zeros(200), 1.0)   # sig=0 -> flat, just ripple
        top_glow_nmr = baseline_glow(TOPBAR_Y, xLn, xRn, -fh * 0.055)
        top_glow_ftir = baseline_glow(TOPBAR_Y, xLn, xRn, -fh * 0.115)  # bigger, mirrors the FTIR glow
        top_axis = axis_line([xLn, TOPBAR_Y, 0], [xRn, TOPBAR_Y, 0], tip=False)
        top_bar = VGroup(top_glow_nmr, top_axis)
        # section title in the BOTTOM half, LEFT-aligned (same left edge as the molecule tag)
        TITLE_Y = -fh * 0.052
        nmr_title = Text("NMR Spectra", font_size=36, color=INK, weight=BOLD)
        nmr_title.move_to([XL + nmr_title.width / 2, TITLE_Y, 0])

        # ===================== PHASE 1 : ROTATION over NMR =====================
        spin = ValueTracker(0.0)
        rotmol = always_redraw(lambda: styled_mol(P_BASE, self.LBL_NUM, angle=spin.get_value()))
        # ---- top-bar trace: audio-reactive EQ bars (opt-in) OR the flat rippling trace ----
        # The audio bars hang DOWN from the top axis (same direction as the existing top glow),
        # in the curve's LIGHT palette with a bright GLOW cap — so it reads as the same light bar,
        # now segmented and dancing to the song. The bottom spectra bar is untouched.
        if getattr(self, "audio_bars", None):
            _feat = np.load(self.audio_bars)               # [n_frames, n_bars] in 0..1
            _nb = _feat.shape[1]
            _afps = self.audio_fps
            _edges = np.linspace(xLn, xRn, _nb + 1)
            _bw = (_edges[1] - _edges[0]) * 0.62           # bar width (leaves gaps)
            _bcx = (_edges[:-1] + _edges[1:]) / 2.0        # bar centres
            _bmax = fh * 0.072                             # max downward bar length
            _bmin = fh * 0.006                             # always-lit minimum
            _aclock = ValueTracker(0.0)
            _aclock.add_updater(lambda m, dt: m.increment_value(dt))

            def _bars():
                t = _aclock.get_value()
                k = int(t * _afps)
                k = 0 if k < 0 else (len(_feat) - 1 if k >= len(_feat) else k)
                row = _feat[k]
                g = VGroup()
                for i in range(_nb):
                    a = float(row[i])
                    ln = _bmin + a * _bmax
                    body = RoundedRectangle(width=_bw, height=ln, stroke_width=0,
                                            corner_radius=min(_bw, ln) * 0.3,
                                            fill_color=LIGHT, fill_opacity=0.55 + 0.4 * a)
                    body.move_to([_bcx[i], TOPBAR_Y - ln / 2.0, 0])
                    cap = Line([_bcx[i] - _bw / 2, TOPBAR_Y - ln, 0],
                               [_bcx[i] + _bw / 2, TOPBAR_Y - ln, 0],
                               color=GLOW, stroke_width=3).set_stroke(opacity=0.9)
                    g.add(body, cap)
                return g
            topcurve = always_redraw(_bars)
            self.add(_aclock)
        else:
            topcurve = always_redraw(topbuild)            # flat rippling trace on the top bar
        self.add(top_bar, topcurve, title_t, sub_nmr, rotmol, gt)
        csnap, hsnap = cbuild(), hbuild()             # static snapshots for a clean fade-in
        ccurve = always_redraw(cbuild); hcurve = always_redraw(hbuild)
        # ---- spin profile: ONE full turn that is FASTEST at t=0 and eases to a stop ----
        # The molecule is unmistakably rotating the instant the video opens — no slow ramp-up and
        # no near-face-on crawl (a constant rate looks static at the start because billboard width
        # ∝ cos θ is flat near 0°). It decelerates smoothly and finishes face-on (angle TAU ≡ 0)
        # exactly as the NMR phase ends, so the phase-2 freeze to the static numbered molecule is
        # seamless. Driven by a per-frame updater (not a play) so the ease-out is one continuous
        # curve while the panels fade in over the opening beat:
        #     θ(s) = TAU·(1−(1−s)²),  s = t/TURN_T   →   θ'(0) = 2·TAU/TURN_T ≈ 45°/s
        TURN_T = 16.0
        spin_clock = ValueTracker(0.0)
        spin_clock.add_updater(lambda m, dt: m.increment_value(dt))

        def drive_spin(m):
            s = min(spin_clock.get_value() / TURN_T, 1.0)
            m.set_value(TAU * (1.0 - (1.0 - s) ** 2))
        spin.add_updater(drive_spin)
        self.add(spin_clock, spin)                    # spin must be in-scene for its updater to fire
        # panels + title fade in over the opening beat while the molecule is ALREADY turning fast
        self.play(FadeIn(cstat), FadeIn(hstat), FadeIn(csnap), FadeIn(hsnap), FadeIn(nmr_title),
                  run_time=2.0)
        self.remove(csnap, hsnap)                     # swap to the live, rippling traces
        self.add(ccurve, hcurve)
        self.wait(TURN_T - 2.0)                        # rotation decelerates to a stop (one full turn)
        spin.remove_updater(drive_spin)               # freeze before the phase-2 swap (angle = TAU)
        spin_clock.clear_updaters(); self.remove(spin_clock, spin)
        self.wait(1.0)

        # ===================== PHASE 2 : NMR -> FTIR, numbers -> atoms =====================
        xL_s, xR_s = -fw * 0.42, fw * 0.42
        # FTIR baseline sits lower than the NMR panels so the gap above the curve has room for the
        # title + the (now bottom-half) vibration-mode subtitle + the band numbers.
        yb_s, yh_s = -fh * 0.45, fh * 0.215
        HI, LO = self.ir_window
        def Xs(cm):
            return xL_s + ((HI - cm) / (HI - LO)) * (xR_s - xL_s)
        xg = np.linspace(HI, LO, 280); yv = np.zeros_like(xg)
        for c, i in self.ir_lines:
            yv = yv + lorentzian(xg, c, i, self.ir_fwhm)
        sig_s = yv / (yv.max() or 1.0)
        xs_s = np.array([Xs(cm) for cm in xg])
        specbuild = make_curve(xs_s, yb_s, sig_s, yh_s)
        spec_static = specbuild()                          # snapshot for the draw-on entrance
        spec_glow = baseline_glow(yb_s, xL_s, xR_s, yh_s * 0.55)
        axis_s = axis_line([xL_s, yb_s, 0], [xR_s, yb_s, 0], tip=False)
        # FTIR title sits where the NMR title was (same place + left-aligned, per house format)
        ftir_title = Text("FTIR Spectra", font_size=36, color=INK, weight=BOLD)
        ftir_title.move_to([XL + ftir_title.width / 2, TITLE_Y, 0])
        xlab = Text("ν̃   (cm⁻¹)", font_size=30, color=WHITE).move_to([0, yb_s - 0.32, 0])
        # numbered markers at the bands where the vibration modes occur (sweep order = 1..N);
        # halo=True so the red cursor line is knocked out behind each number
        bands = [m['band'] for m in self.vib_modes]
        band_h = np.interp(bands, xg[::-1], sig_s[::-1])   # curve height at each band
        band_mk = peak_markers([(Xs(cm), yb_s + hh * yh_s, str(i + 1))
                                for i, (cm, hh) in enumerate(zip(bands, band_h))],
                               base_lift=0.16, aura=0.46, slant=0.70, halo=True)

        numstat = styled_mol(P_BASE, self.LBL_NUM)      # freeze the numbered molecule
        self.remove(rotmol); self.add(numstat)
        staticmol = styled_mol(P_BASE, self.LBL_ATOM)
        # freeze the live NMR traces so they can fade out cleanly
        ccurve.clear_updaters(); hcurve.clear_updaters()
        # smooth crossfade: numbers -> functional groups; NMR subtitles -> FTIR ("additional info")
        t1 = [FadeOut(numstat), FadeIn(staticmol),
              FadeOut(cstat), FadeOut(hstat), FadeOut(ccurve), FadeOut(hcurve), FadeOut(nmr_title),
              FadeOut(top_glow_nmr), FadeIn(top_glow_ftir)]   # top bar grows into the FTIR lighting
        if len(sub_ftir):
            t1 += [FadeOut(sub_nmr), FadeIn(sub_ftir)]
        self.play(*t1, run_time=0.9)
        self.bring_to_front(topcurve)                 # keep the flat trace above its grown glow
        self.play(FadeIn(ftir_title), Create(axis_s), FadeIn(xlab), FadeIn(spec_glow), run_time=0.7)
        self.play(FadeIn(spec_static[0]), Create(spec_static[1]), run_time=1.2)
        self.play(FadeIn(band_mk), run_time=0.6)
        self.remove(spec_static)                          # swap the snapshot for the live trace
        speccurve = always_redraw(specbuild)
        self.add(speccurve)
        self.bring_to_front(band_mk)

        # ===================== PHASE 3 : slow, localised vibration modes =====================
        clock = ValueTracker(0.0); clock.add_updater(lambda m, dt: m.increment_value(dt))
        VIBF = 0.6                                    # slow
        cur_drivers = [self.vib_modes[0]['drivers']]
        vibmol = always_redraw(lambda: styled_mol(
            perturb(cur_drivers[0], float(np.sin(2 * np.pi * VIBF * clock.get_value()))),
            self.LBL_ATOM))
        self.remove(staticmol); self.add(vibmol, clock)
        cur = ValueTracker(Xs(self.vib_modes[0]['band']))
        cursor = always_redraw(lambda: DashedLine(
            [cur.get_value(), yb_s, 0], [cur.get_value(), yb_s + yh_s + 0.12, 0],
            color=HOT, stroke_width=3, dash_length=0.08).set_stroke(opacity=0.85))
        self.add(cursor)
        self.bring_to_front(band_mk)                  # numbers (with halos) ride OVER the cursor
        mlabel = None
        for m in self.vib_modes:
            cur_drivers[0] = m['drivers']
            # mode subtitle now lives in the BOTTOM half, just under the FTIR title
            # mode subtitle (name) + the characteristic wavenumber range beneath it, LEFT-aligned
            # (same left edge as the title) with the range as large as the name above it
            nl = VGroup(
                Text(m['name'], font_size=28, color=GREEN),
                Text(m.get('wn', ''), font_size=28, color="#b8c3d1"),
            ).arrange(DOWN, buff=0.10, aligned_edge=LEFT)
            nl.move_to([XL + nl.width / 2, TITLE_Y - 0.62, 0])
            anims = [cur.animate.set_value(Xs(m['band'])), FadeIn(nl)]
            if mlabel is not None:
                anims.append(FadeOut(mlabel))
            self.play(*anims, run_time=0.8)
            mlabel = nl
            self.wait(3.4)
        self.play(FadeOut(mlabel), run_time=0.5)
        self.wait(0.5)


class VerticalProfile_Aspirin(VerticalProfile):
    """Aspirin — acetylsalicylic acid, the archetypal NSAID.

    THE WORKED EXAMPLE. Every attribute a molecule needs is set here, in the order the
    format uses them: identity text, then the structure (AT/BB/BONDS and the two label
    sets), then the spectra (13C, 1H, FTIR), then the vibration sweep. A new molecule is
    a copy of this class with every value replaced -- see GUIDE.md section 6.

    Render: manim -qh -r 1080,1920 chemical_profile.py VerticalProfile_Aspirin"""
    title = "ASPIRIN"
    subtitle = "C₉H₈O₄  ·  180.16 g/mol  ·  Melting Point 135 °C"
    subtitle2 = "Salicylate  ·  Acidic  ·  Analgesic"
    solvent = "d6-DMSO"
    subtitle_ftir = "logP +1.19  ·  TPSA 63.6 Å²  ·  H₂O sol. 3 g/L"
    subtitle2_ftir = "Ester  ·  Carboxylic acid  ·  Aromatic"
    curve_shade = 'red'
    AT = {'C1': (-2.54, -0.86), 'C2': (-1.8, -0.19), 'O1': (-2.01, 0.79), 'O2': (-0.85, -0.5), 'C3': (-0.11, 0.17), 'C4': (-0.31, 1.15), 'C5': (0.43, 1.82), 'C6': (1.38, 1.51), 'C7': (1.59, 0.54), 'C8': (0.85, -0.13), 'C9': (1.06, -1.11), 'O3': (2.01, -1.42), 'O4': (0.31, -1.78)}
    BB = [[-2.99, -2.23, 0], [2.46, 2.27, 0]]
    BONDS = [('C1', 'C2', 0), ('C2', 'O1', 1), ('C2', 'O2', 0), ('O2', 'C3', 0), ('C3', 'C4', 1), ('C4', 'C5', 0), ('C5', 'C6', 1), ('C6', 'C7', 0), ('C7', 'C8', 1), ('C8', 'C9', 0), ('C9', 'O3', 1), ('C9', 'O4', 0), ('C8', 'C3', 0)]
    LBL_ATOM = [('C1', 'CH₃', MOL_DIM), ('O1', 'O', O_COLOR), ('O2', 'O', O_COLOR), ('O3', 'O', O_COLOR), ('O4', 'O', O_COLOR)]
    LBL_NUM = [('C1', '1', INK), ('C2', '2', INK), ('O1', 'O', O_COLOR), ('O2', 'O', O_COLOR), ('C3', '3', INK), ('C4', '4', INK), ('C5', '5', INK), ('C6', '6', INK), ('C7', '7', INK), ('C8', '8', INK), ('C9', '9', INK), ('O3', 'O', O_COLOR), ('O4', 'O', O_COLOR)]
    # ¹³C (DMSO-d6): COOH (9); ester C=O (2); C–OAc ipso (3); aromatic CH (4–7); C–COOH ipso (8); OAc CH₃ (1)
    nmrc = [(169.9, 0.6, '9'), (169.4, 0.6, '2'), (150.7, 0.55, '3'), (134.0, 0.6, '6'), (132.0, 0.6, '7'), (126.4, 0.6, '5'), (124.0, 0.6, '4'), (123.6, 0.5, '8'), (20.9, 0.6, '1')]
    nmrh = [(7.99, 1.0, '7'), (7.63, 1.0, '6'), (7.36, 1.0, '5'), (7.27, 1.0, '4'), (2.27, 3.0, '1')]
    ir_lines = [(1750, 0.9), (1690, 0.95), (1605, 0.45), (1575, 0.4), (1485, 0.4), (1455, 0.4), (1420, 0.45), (1370, 0.4), (1305, 0.5), (1190, 0.6), (1095, 0.45), (1015, 0.4), (920, 0.35), (840, 0.3), (755, 0.45), (705, 0.35)]
    vib_modes = [
        {'name': "Ester C=O stretch", 'band': 1750, 'wn': "1730–1770 cm⁻¹", 'drivers': [('O1', 'stretch', 'C2', +1)]},
        {'name': "Carboxylic acid C=O stretch", 'band': 1690, 'wn': "1680–1710 cm⁻¹", 'drivers': [('O3', 'stretch', 'C9', +1)]},
        {'name': "Aromatic C=C stretch", 'band': 1605, 'wn': "1580–1620 cm⁻¹", 'drivers': [('C4', 'stretch', 'C5', +1), ('C6', 'stretch', 'C7', -1)]},
        {'name': "C–O ester stretch", 'band': 1190, 'wn': "1160–1220 cm⁻¹", 'drivers': [('O2', 'stretch', 'C2', +1), ('O2', 'stretch', 'C3', -1)]},
        {'name': "Aromatic C–H out-of-plane", 'band': 755, 'wn': "740–770 cm⁻¹", 'drivers': [('C5', 'perp', 'C4', +1), ('C6', 'perp', 'C7', +1)]},
    ]
