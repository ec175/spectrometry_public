"""
spectro_lib.py — reusable object library + CANONICAL FORMAT for the spectroscopy
manim explainers. Import everything with `from spectro_lib import *`.

This module is the single source of truth for the house style; the scene file
(`chemical_profile.py`) is just choreography on top of these objects. Importing this
module ALSO applies the canonical camera/background config (side effects below), so
the same scene composes in both 16:9 and 9:16 just by changing `-r`.

NB this library backs a wider set of spectroscopy scenes than the chemical profiles
published here, so it carries helpers the profiles never call (the STFT/heatmap
machinery, the optics primitives). They are harmless; leave them or strip them.

================================ CANONICAL FORMAT ================================
THEME / PALETTE (dark; shared with the finance + piano renders so they sit together)
  BG #0a0d12 background · INK #e8edf2 primary text · MUTED #8a97a8 labels/secondary
  GRID #2a3444 axes/ticks/frames
  Data roles: STICK #ffd24a yellow = discrete line list · CURVE #4ea1ff blue =
    broadened/measured spectrum (LINEFILL = same) · LORENTZ #5be0a0 green =
    individual line shapes · HOT #ff5b5b red = playhead/accents · TEMP #ff8a4c
    orange = temperature read-out
  Molecule: MOL_DIM #7d8b9f baseline · MOL_GLOW #ffd24a glow · O_COLOR #ff6b6b
    oxygen · H_COLOR #e4ecf5 hydrogen

FRAME / DUAL-ORIENTATION
  config.frame_width = 14.222, frame_height = frame_width * pixel_height/pixel_width.
  The logical frame follows the OUTPUT pixel aspect, so ONE file renders 16:9 and
  9:16 (square pixels, no letterboxing). Every scene branches on
  `portrait = fh > fw` and expresses geometry as FRACTIONS of fw/fh (never hard
  units).

AXES & MARKS (the look)
  No title/subtitle — explanation is carried by equations (MathTex) + compact inline
  labels (Text). Thick axis lines AXIS_W=9, deliberately small arrowheads
  TIP_LEN=TIP_W=0.11, tick thickness TICK_W=4, heatmap box frame FRAME_W=7, 3D axes
  AXIS_W3D=4. All in GRID color. Axis numbers/labels use Text (Pango, no LaTeX).
  Some scenes elsewhere use MathTex for formulae, which needs a LaTeX install — the
  chemical profiles do NOT, so you can render them with no LaTeX on the machine.

HEATMAPS
  Color via the inferno-like `_colormap` (6 anchor stops) with nearest-neighbor
  resampling; magnitudes normalized to [0,1] and gamma-lifted (sqrt) for visibility.

ANIMATION IDIOMS
  All live graphics are ValueTracker + always_redraw; static intro mobjects are
  swapped for live always_redraw versions at the seam so transitions are seamless.
  Canonical "axis reveal" move: show a labeled axis during the build, then once the
  data trace is drawn, fade the spine + title and stretch the plot to full width
  (`reveal_then_collapse_axis`).

DRAFT / QUALITY
  A single DRAFT flag gates sample counts (N_ENV/N_COMP/N_SPEC) and STFT window
  size; finals render at 30fps.

RENDER
  manim.cfg sets media_dir = ./renders, so output lands in
  renders/videos/<file>/<height>p<fps>/<ClassName>.mp4. See GUIDE.md.
=================================================================================
"""
import numpy as np
from manim import *

# ---- palette ---------------------------------------------------------------
BG      = "#0a0d12"
INK     = "#e8edf2"
MUTED   = "#8a97a8"
GRID    = "#2a3444"
STICK   = "#ffd24a"   # yellow  - the discrete line list
CURVE   = "#4ea1ff"   # blue    - the broadened (measured) spectrum
LINEFILL= "#4ea1ff"
LORENTZ = "#5be0a0"   # green   - the individual line shapes
HOT      = "#ff5b5b"  # red     - playhead / accents
TEMP     = "#ff8a4c"  # orange  - temperature read-out

# molecule colours
MOL_DIM  = "#7d8b9f"   # baseline carbon/bond colour (lightly shaded)
MOL_GLOW = "#ffd24a"   # warm glow at full proximity
O_COLOR  = "#ff6b6b"   # CPK-ish oxygen red
H_COLOR  = "#e4ecf5"   # hydrogen near-white
N_COLOR  = "#6f9bff"   # CPK-ish nitrogen blue

# ---- canonical-format line weights -----------------------------------------
AXIS_W   = 9.0     # axis line thickness
TIP_LEN  = 0.11    # arrowhead length — deliberately small
TIP_W    = 0.11    # arrowhead width
TICK_W   = 4.0     # tick-mark thickness
FRAME_W  = 7.0     # box-frame thickness for heatmap scenes
AXIS_W3D = 4.0     # ThreeDAxes line thickness

# ---- camera / background (applied on import) -------------------------------
config.background_color = BG
# Logical camera frame follows the OUTPUT pixel aspect, so one scene composes in
# both 16:9 and 9:16. pixel_width/height already reflect any -r flag by import time.
config.frame_width  = 14.222
config.frame_height = config.frame_width * config.pixel_height / config.pixel_width

# ---- DRAFT MODE (fast iteration) -------------------------------------------
# While True: fewer points per line, lower STFT resolution. Render `-q l`. Flip to
# False (and render `-q h`) for the final pass — every quality knob keys off this.
DRAFT = False
N_ENV  = 1600 if DRAFT else 5000   # Resolution_Temperature: envelope sample points
N_COMP = 110  if DRAFT else 260    # Resolution_Temperature: points per line shape
N_SPEC = 90   if DRAFT else 200    # 2D spectrum-panel smoothing samples


# =============================================================================
#  CANONICAL-FORMAT MOBJECT HELPERS
# =============================================================================

def ease_out_back(t, s=1.70158):
    """Ease-out with a slight overshoot past 1.0 that settles back exactly to 1.0 at t=1.
    Use as a manim `rate_func` for "snap/land" intro moves (an imploding lattice that
    overshoots its sites, a title that punches in). Shared by every vertical-format intro
    so they land with the same energy."""
    u = t - 1.0
    return u * u * ((s + 1.0) * u + s) + 1.0


def axis_line(start, end, tip=True, width=AXIS_W):
    """A canonical axis: thick line, optional small arrowhead."""
    ln = Line(start, end, color=GRID, stroke_width=width)
    if tip:
        ln.add_tip(tip_length=TIP_LEN, tip_width=TIP_W)
    return ln


def tick_marks(positions, labels, *, axis="x", baseline=0.0, tick_minus=0.08,
               tick_plus=0.0, label_gap=0.28, font_size=20, label_mode="move_to",
               label_buff=0.12, color=GRID, label_color=MUTED):
    """Canonical tick marks + numeric labels along one axis.

    positions  screen coords along the axis (x for axis='x', y for axis='y')
    labels     the value drawn at each position
    baseline   the perpendicular coord of the axis line
    axis='x'   vertical ticks from baseline-tick_minus .. baseline+tick_plus,
               labels below (move_to baseline-label_gap, or next_to DOWN)
    axis='y'   horizontal ticks from baseline-tick_minus .. baseline+tick_plus,
               labels left (next_to LEFT buff=label_buff, or move_to baseline-label_gap)

    Reproduces the three original tick loops exactly:
      Res_Temp x:    tick_minus=0.09, tick_plus=0.09, label_gap=0.34
      Spectrogram x: tick_minus=0.08, tick_plus=0.0,  label_gap=0.28
      Spectrogram y: axis='y', tick_minus=0.08, label_mode='next_to', label_buff=0.12
    """
    g = VGroup()
    for p, lab in zip(positions, labels):
        t = Text(str(lab), font_size=font_size, color=label_color)
        if axis == "x":
            g.add(Line([p, baseline - tick_minus, 0], [p, baseline + tick_plus, 0],
                       color=color, stroke_width=TICK_W))
            if label_mode == "next_to":
                t.next_to([p, baseline, 0], DOWN, buff=label_buff)
            else:
                t.move_to([p, baseline - label_gap, 0])
        else:
            g.add(Line([baseline - tick_minus, p, 0], [baseline + tick_plus, p, 0],
                       color=color, stroke_width=TICK_W))
            if label_mode == "next_to":
                t.next_to([baseline, p, 0], LEFT, buff=label_buff)
            else:
                t.move_to([baseline - label_gap, p, 0])
        g.add(t)
    return g


def frame_box(w, h, center):
    """Canonical box frame around a heatmap."""
    return Rectangle(width=w, height=h, color=GRID, stroke_width=FRAME_W).move_to(center)


def readout_pill(anchor_fn, grow_tracker, *, width, h1, row_dy, pad_top=0.14,
                 corner_radius=0.14, fill_color="#0f141b", fill_opacity=0.92,
                 cx_inset=0.22, z_index=15):
    """Factory for the morphing read-out HUD pill. Returns a builder suitable for
    `always_redraw`. The pill grows DOWN as `grow_tracker` rises 0->1 (e.g. a second
    read-out row fading in). `anchor_fn()` gives the top-left text anchor."""
    def make_pill():
        a = anchor_fn()
        h = h1 + grow_tracker.get_value() * row_dy
        top_edge = a[1] + pad_top
        cx = a[0] + width / 2 - cx_inset
        return RoundedRectangle(
            width=width, height=h, corner_radius=corner_radius, fill_color=fill_color,
            fill_opacity=fill_opacity, stroke_color=GRID, stroke_width=2
        ).move_to([cx, top_edge - h / 2, 0]).set_z_index(z_index)
    return make_pill


def reveal_then_collapse_axis(scene, fade_mobs, lt, rt, left1, right1, *,
                              between=None, fade_rt=0.8, expand_rt=1.4,
                              expand_rate=smooth):
    """Canonical 'show the axis, then drop it and fit to screen' move.

    1) fade out `fade_mobs` (the axis spine + its title, plus any extras like the
       stick spectrum), then
    2) stretch the plot to full width by driving the left/right edge trackers
       `lt`/`rt` to `left1`/`right1`.

    `between` (optional callable) runs in the gap between the two plays — use it to
    swap the static intro mobjects for their live always_redraw versions so the
    expansion reflows cleanly (see Resolution_Temperature)."""
    scene.play(*[FadeOut(m) for m in fade_mobs], run_time=fade_rt)
    if between is not None:
        between()
    scene.play(lt.animate.set_value(left1), rt.animate.set_value(right1),
               run_time=expand_rt, rate_func=expand_rate)


# =============================================================================
#  STANDARD CURVE COLOURS  (set 2026-06-10 — THE house palette for data curves)
#  5 colour families; every curve fill shades left=LIGHT -> right=DARK. Use these
#  for all spectra / trace fills going forward (one family per axis/technique).
# =============================================================================
SHADE = {                       # light end -> dark end (contrast bumped 2026-06-10)
    "blue":   ("#c0e0ff", "#184f9e"),
    "red":    ("#ffc4c1", "#971a1a"),
    "green":  ("#bbf3d5", "#16834e"),
    "purple": ("#dac6ff", "#4d2495"),
    "pink":   ("#ffcae8", "#a3186a"),
}
SHADE_ORDER = ["blue", "red", "green", "purple", "pink"]


def horizontal_gradient_fill(pts_top, baseline_y, light, dark, xleft, xright, *, opacity=1.0,
                             max_slices=None):
    """Fill under a curve (its top points) with a LEFT->RIGHT light->dark gradient, built
    from vertical slices. Each slice UNDERLAPS the next (extends right under it) and
    carries a matching-colour stroke, so there are NO antialiasing seams ("dead lines")
    between columns — the fill reads as one fluid gradient.

    `max_slices` (perf): cap the number of gradient columns. The curve SILHOUETTE stays
    exact (every top point is still a polygon vertex); only the number of separately
    coloured fill bands drops — imperceptible for a smooth gradient but ~ (n-1)/max_slices
    fewer submobjects per frame. Leave None for the exact per-point behaviour (default;
    every existing caller is byte-identical)."""
    g = VGroup()
    L, D = ManimColor(light), ManimColor(dark)
    span = (xright - xleft) or 1.0
    n = len(pts_top)
    if n < 2:
        return g
    xs = [p[0] for p in pts_top]
    ov = (max(xs) - min(xs)) / max(1, n - 1) if n > 1 else 0.0     # ~one strip width

    if not max_slices or max_slices >= n - 1:
        # -------- exact per-point gradient (original behaviour) --------
        for i in range(n - 1):
            a, b = pts_top[i], pts_top[i + 1]
            f = min(1.0, max(0.0, (0.5 * (a[0] + b[0]) - xleft) / span))
            col = interpolate_color(L, D, f)
            rx = b[0] + (ov if i < n - 2 else 0.0)    # run under the next slice (hides the seam)
            q = Polygon([a[0], baseline_y, 0], [a[0], a[1], 0], [rx, b[1], 0], [rx, baseline_y, 0],
                        color=col, fill_color=col, fill_opacity=opacity, stroke_width=0)
            q.set_stroke(col, width=1.0, opacity=opacity)
            g.add(q)
        return g

    # -------- coarse gradient: fewer bands, silhouette still follows every point --------
    edges = [round(k * (n - 1) / max_slices) for k in range(max_slices + 1)]
    for bi in range(max_slices):
        i0, i1 = edges[bi], edges[bi + 1]
        if i1 <= i0:
            continue
        seg = pts_top[i0:i1 + 1]
        f = min(1.0, max(0.0, (0.5 * (seg[0][0] + seg[-1][0]) - xleft) / span))
        col = interpolate_color(L, D, f)
        rx = seg[-1][0] + (ov if i1 < n - 1 else 0.0)          # underlap the next band
        verts = [[seg[0][0], baseline_y, 0]] + [[p[0], p[1], 0] for p in seg] \
                + [[rx, seg[-1][1], 0], [rx, baseline_y, 0]]
        q = Polygon(*verts, color=col, fill_color=col, fill_opacity=opacity, stroke_width=0)
        q.set_stroke(col, width=1.0, opacity=opacity)
        g.add(q)
    return g


def gradient_swatch(center, w, h, light, dark, *, n=26, border=GRID):
    """A small horizontal light->dark gradient bar — a legend swatch for a colour
    family. Returns a VGroup (gradient slices + a thin border)."""
    cx, cy = center[0], center[1]
    x0, x1 = cx - w / 2, cx + w / 2
    pts_top = [[x0 + (x1 - x0) * i / (n - 1), cy + h / 2, 0] for i in range(n)]
    g = horizontal_gradient_fill(pts_top, cy - h / 2, light, dark, x0, x1)
    g.add(Rectangle(width=w, height=h, color=border, stroke_width=2).move_to([cx, cy, 0]))
    return g


# =============================================================================
#  DOMAIN: LINE LIST -> BROADENED SPECTRUM
# =============================================================================

# Representative IR line list (center cm^-1, relative intensity 0-1) at real
# functional-group band positions. Swap for a DFT line list to make it exact:
#   centers, intens = gaussian.parse_ir(log)
LINE_LIST = [
    (3300, 0.50),   # O-H / N-H stretch
    (2960, 0.80),   # asym C-H stretch
    (2920, 0.95),   # C-H stretch
    (2855, 0.55),   # sym C-H stretch
    (1710, 1.00),   # C=O stretch (carbonyl)
    (1610, 0.70),   # aromatic C=C
    (1580, 0.42),   # aromatic C=C
    (1455, 0.60),   # CH2 / CH3 bend
    (1375, 0.40),   # CH3 sym bend
    (1235, 0.78),   # C-O stretch
    (1160, 0.50),   # C-O stretch
    (1040, 0.55),   # C-O / C-C
    (880,  0.33),   # aromatic C-H oop
    (760,  0.46),   # aromatic C-H oop
]


def lorentzian(x, x0, inten, fwhm):
    """Peak-normalised Lorentzian: value == `inten` at x == x0."""
    hw = fwhm / 2.0
    return inten * hw ** 2 / ((x - x0) ** 2 + hw ** 2)


def broaden(x, lines, fwhm):
    """Sum of Lorentzians (the lab's FTIR/Raman line shape)."""
    y = np.zeros_like(x, dtype=float)
    for x0, inten in lines:
        y += lorentzian(x, x0, inten, fwhm)
    return y


def gaussian(x, x0, inten, fwhm):
    """Peak-normalised Gaussian: value == `inten` at x == x0. The natural broad band
    shape for UV-Vis electronic transitions (vibronic + solvent broadening)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return inten * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


# Aspirin (acetylsalicylic acid) UV-Vis absorption bands in ethanol:
# (center nm, rel absorbance at l = 1 cm, FWHM nm). Broad electronic transitions —
# a strong pi->pi* (K-band) near 229 nm plus a weaker benzenoid B-band near 277 nm.
ASPIRIN_UV = [
    (229.0, 0.55, 22.0),
    (277.0, 0.16, 26.0),
]


# Aspirin (acetylsalicylic acid) IR line list: (center cm^-1, rel intensity, group).
# The `group` tags tie each band to a part of the molecule model so it can glow.
ASPIRIN = [
    (3060, 0.22, "ring"),     # aromatic C-H stretch
    (1750, 1.00, "esterCO"),  # ester C=O stretch
    (1690, 0.92, "acidCO"),   # carboxylic-acid C=O stretch
    (1605, 0.45, "ring"),     # aromatic C=C
    (1575, 0.30, "ring"),
    (1485, 0.42, "ring"),
    (1458, 0.52, "ring"),     # aromatic C=C / CH
    (1418, 0.46, "OH"),       # O-H bend
    (1370, 0.45, "CH3"),      # CH3 sym bend
    (1307, 0.56, "CO"),       # C-O stretch
    (1220, 0.72, "CO"),       # C-O (ester)
    (1190, 0.85, "CO"),       # C-O stretch
    (1095, 0.50, "CO"),
    (1015, 0.42, "ring"),     # ring in-plane
    (915,  0.34, "OH"),       # O-H out-of-plane
    (840,  0.30, "ring"),
    (755,  0.66, "ring"),     # ortho aromatic C-H oop
    (705,  0.50, "ring"),
]


def build_aspirin(s=1.0):
    """Skeletal Kekule stick model of aspirin. Returns (VGroup, groups); `groups`
    maps a band group -> list of mobjects to glow. Bonds gray, O red, H white; each
    mob carries `.glow_base` (its dim colour) so the glow blends toward MOL_GLOW."""
    def Lp(p):
        return np.array([p[0] * s, p[1] * s, 0.0])
    R = 0.60
    V = {k: (R * np.cos(np.deg2rad(60 * k)), R * np.sin(np.deg2rad(60 * k))) for k in range(6)}
    groups = {"ring": [], "esterCO": [], "acidCO": [], "CO": [], "OH": [], "CH3": []}
    mobs = []

    def reg(mo, grp, base):
        mo.glow_base = base
        groups[grp].append(mo); mobs.append(mo)

    def bond(p1, p2, grp, double=False, w=3.0, base=MOL_DIM):
        a, b = Lp(p1), Lp(p2)
        if not double:
            reg(Line(a, b, color=base, stroke_width=w), grp, base); return
        d = b - a
        n = np.array([-d[1], d[0], 0.0]); n = n / (np.linalg.norm(n) + 1e-9) * 0.055 * s
        for off in (n, -n):
            reg(Line(a + off, b + off, color=base, stroke_width=w), grp, base)

    def lab(p, txt, grp, base, fs=22):
        reg(Text(txt, font_size=fs, color=base).move_to(Lp(p)), grp, base)

    # benzene: 6 outer single bonds + 3 inner Kekule double bonds (no circle)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        bond(V[i], V[j], "ring")
    for i, j in [(0, 1), (2, 3), (4, 5)]:
        a, b = np.array(V[i]), np.array(V[j])
        inw = -((a + b) / 2.0); inw = inw / (np.linalg.norm(inw) + 1e-9) * 0.14
        bond(tuple(a + (b - a) * 0.14 + inw), tuple(b + (a - b) * 0.14 + inw), "ring")

    # acetyl ester  Ar-O-C(=O)-CH3  off V1 (upper-right)
    Oe, Ce = (0.62, 1.12), (1.24, 1.20)
    bond(V[1], Oe, "CO"); bond(Oe, Ce, "CO")
    bond(Ce, (1.24, 1.82), "esterCO", double=True)        # C=O up
    bond(Ce, (1.86, 1.02), "CH3")                         # C-CH3
    lab((0.44, 1.20), "O", "CO", O_COLOR)
    lab((1.24, 2.08), "O", "esterCO", O_COLOR)
    lab((2.18, 1.06), "CH₃", "CH3", MOL_DIM)

    # carboxylic acid  Ar-C(=O)-O-H  off V0 (right)
    Cc = (1.22, -0.06)
    bond(V[0], Cc, "acidCO")
    bond(Cc, (1.22, -0.68), "acidCO", double=True)        # C=O down
    bond(Cc, (1.80, 0.26), "OH")                          # C-O
    lab((1.22, -0.94), "O", "acidCO", O_COLOR)
    lab((1.98, 0.28), "O", "OH", O_COLOR)
    lab((2.38, 0.28), "H", "OH", H_COLOR)

    return VGroup(*mobs), groups


# ---- more skeletal models (same hand-built style as build_aspirin) ----------
def _skeletal(s=1.0):
    """Shared drawing canvas for hand-built skeletal molecule models. Returns
    (mobs, bond, lab, ring): bonds gray, O red, N blue, H white; `ring` draws a
    Kekule benzene at a center and returns its 6 vertices."""
    mobs = []

    def Lp(p):
        return np.array([p[0] * s, p[1] * s, 0.0])

    _dbl = [0]

    def bond(p1, p2, double=False, w=3.0, base=MOL_DIM):
        a, b = Lp(p1), Lp(p2)
        if not double:
            mobs.append(Line(a, b, color=base, stroke_width=w)); return
        d = b - a
        n = np.array([-d[1], d[0], 0.0]); n = n / (np.linalg.norm(n) + 1e-9) * 0.06 * s
        pid = _dbl[0]; _dbl[0] += 1                   # tag both lines of this double bond
        for off in (n, -n):
            ln = Line(a + off, b + off, color=base, stroke_width=w)
            ln._double_pair = pid
            mobs.append(ln)

    def lab(p, txt, base, fs=30):
        mobs.append(Text(txt, font_size=fs, color=base).move_to(Lp(p)))

    def ring(center, R=0.6, rot=0.0, doubles=(0, 2, 4)):
        V = [(center[0] + R * np.cos(rot + np.deg2rad(60 * k)),
              center[1] + R * np.sin(rot + np.deg2rad(60 * k))) for k in range(6)]
        for k in range(6):
            bond(V[k], V[(k + 1) % 6])
        for k in doubles:                       # inner Kekule double bonds
            a, b, c = np.array(V[k]), np.array(V[(k + 1) % 6]), np.array(center)
            bond(tuple(a + (c - a) * 0.16), tuple(b + (c - b) * 0.16))
        return V

    return mobs, bond, lab, ring


def build_acetaminophen(s=1.0):
    """Skeletal model of acetaminophen (paracetamol): para-aminophenol N-acetyl."""
    mobs, bond, lab, ring = _skeletal(s)
    V = ring((0, 0), R=0.6)                       # V0 right, V3 left (para)
    bond(V[3], (-1.18, 0.0)); lab((-1.32, 0.0), "O", O_COLOR); lab((-1.74, 0.0), "H", H_COLOR)
    bond(V[0], (1.18, 0.12)); lab((1.2, 0.26), "N", N_COLOR); lab((1.2, 0.68), "H", H_COLOR)
    bond((1.2, 0.12), (1.8, -0.22))                          # N-C
    bond((1.8, -0.22), (1.8, -0.86), double=True)           # C=O
    lab((1.8, -1.06), "O", O_COLOR)
    bond((1.8, -0.22), (2.4, 0.12)); lab((2.7, 0.16), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_caffeine(s=1.0):
    """Skeletal model of caffeine (1,3,7-trimethylxanthine): purine bicyclic with
    two ring carbonyls, three N-CH3 and a C8-H."""
    mobs, bond, lab, ring = _skeletal(s)
    C5, C6, N1 = (0, 0.5), (-0.87, 1.0), (-1.74, 0.5)
    C2, N3, C4 = (-1.74, -0.5), (-0.87, -1.0), (0, -0.5)
    N9, C8, N7 = (0.95, -0.81), (1.54, 0.0), (0.95, 0.81)
    # 6-ring (pyrimidinedione) + 5-ring (imidazole), fused C4-C5
    bond(N1, C2); bond(C2, N3); bond(N3, C4); bond(C4, C5, double=True)
    bond(C5, C6); bond(C6, N1)
    bond(C4, N9); bond(N9, C8, double=True); bond(C8, N7); bond(N7, C5)
    bond(C2, (-2.52, -0.95), double=True); bond(C6, (-0.87, 1.95), double=True)  # C=O
    lab((-2.66, -1.05), "O", O_COLOR); lab((-0.87, 2.12), "O", O_COLOR)
    for p in (N1, N3, N7, N9):
        lab(p, "N", N_COLOR)
    bond(N1, (-2.52, 0.95)); lab((-2.92, 1.05), "CH₃", MOL_DIM)
    bond(N3, (-0.87, -1.9)); lab((-0.87, -2.12), "CH₃", MOL_DIM)
    bond(N7, (1.55, 1.5)); lab((1.78, 1.66), "CH₃", MOL_DIM)
    bond(C8, (2.18, 0.0)); lab((2.34, 0.0), "H", H_COLOR)
    return VGroup(*mobs)


def build_diphenhydramine(s=1.0):
    """Skeletal model of diphenhydramine (Benadryl): two phenyls on a CH, an ether
    O, and an -O-CH2-CH2-N(CH3)2 chain."""
    mobs, bond, lab, ring = _skeletal(s)
    C0 = (0.0, 0.0)
    VA = ring((-1.15, 1.05), R=0.55, rot=np.deg2rad(-30))
    VB = ring((1.15, 1.05), R=0.55, rot=np.deg2rad(-150))
    bond(C0, VA[0]); bond(C0, VB[0])                         # CH to each phenyl
    # ether C-O-C: clear bond from the CH carbon down to O, then O down-right to CH2
    bond(C0, (0.0, -0.98))
    lab((0.0, -1.0), "O", O_COLOR)
    bond((0.0, -0.98), (0.70, -1.40))                        # O - CH2
    bond((0.70, -1.40), (0.70, -2.16))                       # CH2 - CH2
    bond((0.70, -2.16), (1.34, -2.52)); lab((1.46, -2.60), "N", N_COLOR)
    bond((1.34, -2.52), (2.02, -2.29)); lab((2.34, -2.29), "CH₃", MOL_DIM)
    bond((1.34, -2.52), (1.34, -3.32)); lab((1.34, -3.52), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def _catechol(bond, lab, ring):
    """Shared catechol head (benzene-1,2-diol): a Kekule ring with two adjacent
    -OH on the left/lower-left vertices. Returns the ring vertices (V0 = right,
    free for a side chain)."""
    V = ring((0, 0), R=0.6)                       # V0 right, V3 left, V4 lower-left
    bond(V[3], (-1.25, 0.0)); lab((-1.40, 0.0), "O", O_COLOR); lab((-1.78, 0.0), "H", H_COLOR)
    bond(V[4], (-0.62, -1.12)); lab((-0.74, -1.30), "O", O_COLOR); lab((-1.12, -1.30), "H", H_COLOR)
    return V


def build_dopamine(s=1.0):
    """Skeletal model of dopamine: a catechol ring with a -CH2-CH2-NH2 chain."""
    mobs, bond, lab, ring = _skeletal(s)
    V = _catechol(bond, lab, ring)
    bond(V[0], (1.25, 0.35)); bond((1.25, 0.35), (1.90, 0.0))      # CH2-CH2
    bond((1.90, 0.0), (2.55, 0.35)); lab((2.80, 0.56), "NH₂", N_COLOR)
    return VGroup(*mobs)


def build_adrenaline(s=1.0):
    """Skeletal model of adrenaline (epinephrine): a catechol ring with a
    -CH(OH)-CH2-NH-CH3 chain (beta-hydroxyl + secondary N-methyl amine)."""
    mobs, bond, lab, ring = _skeletal(s)
    V = _catechol(bond, lab, ring)
    bond(V[0], (1.25, 0.35))                                       # ring - CH(OH)
    bond((1.25, 0.35), (1.25, 1.08)); lab((1.25, 1.26), "O", O_COLOR); lab((1.62, 1.26), "H", H_COLOR)
    bond((1.25, 0.35), (1.90, 0.0))                               # CH - CH2
    bond((1.90, 0.0), (2.55, 0.35)); lab((2.70, 0.52), "NH", N_COLOR)
    bond((2.55, 0.35), (3.20, 0.0)); lab((3.50, -0.06), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_serotonin(s=1.0):
    """Skeletal model of serotonin (5-hydroxytryptamine): a 5-hydroxy indole
    (fused benzene + pyrrole) with a 3-(2-aminoethyl) chain."""
    mobs, bond, lab, ring = _skeletal(s)
    C7a, C3a = (0.0, 0.5), (0.0, -0.5)                            # shared fused edge
    C7, C6, C5, C4 = (-0.87, 1.0), (-1.74, 0.5), (-1.74, -0.5), (-0.87, -1.0)
    N1, C2, C3 = (0.95, 0.81), (1.54, 0.0), (0.95, -0.81)
    # benzene ring (Kekule) + pyrrole ring, fused along C3a-C7a
    bond(C7a, C7, double=True); bond(C7, C6); bond(C6, C5, double=True)
    bond(C5, C4); bond(C4, C3a, double=True); bond(C3a, C7a)
    bond(C7a, N1); bond(N1, C2); bond(C2, C3, double=True); bond(C3, C3a)
    lab((1.42, 1.14), "NH", N_COLOR, fs=38)                       # indole N-H (nudged right)
    bond(C5, (-2.46, -0.85)); lab((-2.62, -1.00), "O", O_COLOR, fs=38); lab((-3.05, -1.00), "H", H_COLOR, fs=38)
    bond(C3, (1.55, -1.25)); bond((1.55, -1.25), (2.20, -0.95))   # CH2-CH2
    bond((2.20, -0.95), (2.85, -1.35)); lab((3.40, -1.46), "NH₂", N_COLOR, fs=38)
    return VGroup(*mobs)


def build_gaba(s=1.0):
    """Skeletal model of GABA (gamma-aminobutyric acid): H2N-CH2-CH2-CH2-COOH."""
    mobs, bond, lab, ring = _skeletal(s)
    P0, P1, P2, P3, P4 = (-2.45, 0.18), (-1.83, -0.18), (-1.21, 0.18), (-0.59, -0.18), (0.03, 0.18)
    bond(P0, P1); bond(P1, P2); bond(P2, P3); bond(P3, P4)
    lab((-2.66, 0.40), "H₂N", N_COLOR)
    bond(P4, (0.03, 0.92), double=True); lab((0.03, 1.12), "O", O_COLOR)   # C=O
    bond(P4, (0.65, -0.18)); lab((0.78, -0.22), "O", O_COLOR); lab((1.16, -0.22), "H", H_COLOR)
    return VGroup(*mobs)


def build_glutamate(s=1.0):
    """Skeletal model of glutamic acid (glutamate): HOOC-CH(NH2)-CH2-CH2-COOH."""
    mobs, bond, lab, ring = _skeletal(s)
    C1, Ca, Cb, Cg, C5 = (-2.25, 0.20), (-1.63, -0.16), (-1.01, 0.20), (-0.39, -0.16), (0.23, 0.20)
    bond(C1, Ca); bond(Ca, Cb); bond(Cb, Cg); bond(Cg, C5)
    bond(C1, (-2.25, 0.94), double=True); lab((-2.25, 1.14), "O", O_COLOR)         # alpha C=O
    bond(C1, (-2.87, -0.16)); lab((-3.00, -0.20), "O", O_COLOR); lab((-3.00, -0.58), "H", H_COLOR)
    bond(Ca, (-1.63, -0.90)); lab((-1.63, -1.10), "NH₂", N_COLOR)                  # alpha amine
    bond(C5, (0.23, 0.94), double=True); lab((0.23, 1.14), "O", O_COLOR)           # gamma C=O
    bond(C5, (0.85, -0.16)); lab((0.98, -0.20), "O", O_COLOR); lab((1.36, -0.20), "H", H_COLOR)
    return VGroup(*mobs)


def build_glutamine(s=1.0):
    """Skeletal model of glutamine: HOOC-CH(NH2)-CH2-CH2-C(=O)-NH2 (glutamate's gamma
    acid replaced by a primary amide)."""
    mobs, bond, lab, ring = _skeletal(s)
    C1, Ca, Cb, Cg, C5 = (-2.25, 0.20), (-1.63, -0.16), (-1.01, 0.20), (-0.39, -0.16), (0.23, 0.20)
    bond(C1, Ca); bond(Ca, Cb); bond(Cb, Cg); bond(Cg, C5)
    bond(C1, (-2.25, 0.94), double=True); lab((-2.25, 1.14), "O", O_COLOR)         # alpha C=O
    bond(C1, (-2.87, -0.16)); lab((-3.00, -0.20), "O", O_COLOR); lab((-3.00, -0.58), "H", H_COLOR)
    bond(Ca, (-1.63, -0.90)); lab((-1.63, -1.10), "NH₂", N_COLOR)                  # alpha amine
    bond(C5, (0.23, 0.94), double=True); lab((0.23, 1.14), "O", O_COLOR)           # gamma C=O (amide)
    bond(C5, (0.85, -0.16)); lab((1.02, -0.20), "NH₂", N_COLOR)                    # amide -NH2
    return VGroup(*mobs)


# ========================== THE 20 STANDARD AMINO ACIDS =======================
# All share the backbone HOOC-CH(NH2)-R; only the side chain R differs. _aa_core
# draws the backbone and returns the alpha carbon; each builder adds R from there.
_S_COLOR = "#d8c24a"            # sulfur (Cys/Met)


def _aa_core(s=1.0):
    """Amino-acid backbone HOOC-CH(NH2)- . Returns (mobs, bond, lab, ring, Ca); the
    side chain attaches at the alpha carbon Ca and extends to the right."""
    mobs, bond, lab, ring = _skeletal(s)
    C1, Ca = (-2.25, 0.20), (-1.63, -0.16)
    bond(C1, Ca)
    bond(C1, (-2.25, 0.94), double=True); lab((-2.25, 1.14), "O", O_COLOR)
    bond(C1, (-2.87, -0.16)); lab((-3.00, -0.20), "O", O_COLOR); lab((-3.00, -0.58), "H", H_COLOR)
    bond(Ca, (-1.63, -0.90)); lab((-1.63, -1.10), "NH₂", N_COLOR)
    return mobs, bond, lab, ring, Ca


def build_glycine(s=1.0):
    """Glycine — R = H (the simplest amino acid)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    bond(Ca, (-1.01, 0.20)); lab((-0.86, 0.26), "H", H_COLOR)
    return VGroup(*mobs)


def build_alanine(s=1.0):
    """Alanine — R = -CH3."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    bond(Ca, (-1.01, 0.20)); lab((-0.74, 0.30), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_valine(s=1.0):
    """Valine — R = isopropyl -CH(CH3)2."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    bond(Cb, (-1.01, 0.94)); lab((-1.01, 1.14), "CH₃", MOL_DIM)
    bond(Cb, (-0.39, -0.16)); lab((-0.10, -0.20), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_leucine(s=1.0):
    """Leucine — R = isobutyl -CH2-CH(CH3)2."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb, Cg = (-1.01, 0.20), (-0.39, -0.16); bond(Ca, Cb); bond(Cb, Cg)
    bond(Cg, (-0.39, -0.90)); lab((-0.39, -1.10), "CH₃", MOL_DIM)
    bond(Cg, (0.23, 0.20)); lab((0.52, 0.24), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_isoleucine(s=1.0):
    """Isoleucine — R = sec-butyl -CH(CH3)-CH2-CH3."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    bond(Cb, (-1.01, 0.94)); lab((-1.01, 1.14), "CH₃", MOL_DIM)
    bond(Cb, (-0.39, -0.16)); bond((-0.39, -0.16), (0.23, 0.20)); lab((0.52, 0.24), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_serine(s=1.0):
    """Serine — R = -CH2-OH."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    bond(Cb, (-1.01, 0.94)); lab((-1.01, 1.14), "O", O_COLOR); lab((-0.64, 1.14), "H", H_COLOR)
    return VGroup(*mobs)


def build_threonine(s=1.0):
    """Threonine — R = -CH(OH)-CH3."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    bond(Cb, (-1.01, 0.94)); lab((-1.01, 1.14), "O", O_COLOR); lab((-0.64, 1.14), "H", H_COLOR)
    bond(Cb, (-0.39, -0.16)); lab((-0.10, -0.20), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_cysteine(s=1.0):
    """Cysteine — R = -CH2-SH (thiol)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    bond(Cb, (-1.01, 0.94)); lab((-1.01, 1.14), "S", _S_COLOR); lab((-0.62, 1.14), "H", H_COLOR)
    return VGroup(*mobs)


def build_methionine(s=1.0):
    """Methionine — R = -CH2-CH2-S-CH3 (thioether)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb, Cg = (-1.01, 0.20), (-0.39, -0.16); bond(Ca, Cb); bond(Cb, Cg)
    bond(Cg, (0.23, 0.20)); lab((0.30, 0.30), "S", _S_COLOR)
    bond((0.23, 0.20), (0.85, -0.16)); lab((1.14, -0.20), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_asparagine(s=1.0):
    """Asparagine — R = -CH2-C(=O)-NH2 (side-chain amide)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb, Cg = (-1.01, 0.20), (-0.39, -0.16); bond(Ca, Cb); bond(Cb, Cg)
    bond(Cg, (-0.39, -0.90), double=True); lab((-0.39, -1.10), "O", O_COLOR)
    bond(Cg, (0.23, 0.20)); lab((0.48, 0.24), "NH₂", N_COLOR)
    return VGroup(*mobs)


def build_aspartate(s=1.0):
    """Aspartate (aspartic acid) — R = -CH2-COOH."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb, Cg = (-1.01, 0.20), (-0.39, -0.16); bond(Ca, Cb); bond(Cb, Cg)
    bond(Cg, (-0.39, -0.90), double=True); lab((-0.39, -1.10), "O", O_COLOR)
    bond(Cg, (0.23, 0.20)); lab((0.36, 0.24), "O", O_COLOR); lab((0.74, 0.24), "H", H_COLOR)
    return VGroup(*mobs)


def build_lysine(s=1.0):
    """Lysine — R = -(CH2)4-NH2."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    chain = [(-1.01, 0.20), (-0.39, -0.16), (0.23, 0.20), (0.85, -0.16)]
    prev = Ca
    for p in chain:
        bond(prev, p); prev = p
    bond(chain[-1], (1.47, 0.20)); lab((1.76, 0.24), "NH₂", N_COLOR)
    return VGroup(*mobs)


def build_arginine(s=1.0):
    """Arginine — R = -(CH2)3-NH-C(=NH)-NH2 (guanidinium)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    chain = [(-1.01, 0.20), (-0.39, -0.16), (0.23, 0.20)]
    prev = Ca
    for p in chain:
        bond(prev, p); prev = p
    bond(chain[-1], (0.85, -0.16)); lab((0.99, -0.22), "NH", N_COLOR)
    Cz = (1.47, 0.20); bond((0.85, -0.16), Cz)
    bond(Cz, (1.47, 0.94), double=True); lab((1.47, 1.14), "NH", N_COLOR)
    bond(Cz, (2.09, -0.16)); lab((2.36, -0.20), "NH₂", N_COLOR)
    return VGroup(*mobs)


def build_phenylalanine(s=1.0):
    """Phenylalanine — R = -CH2-C6H5 (benzyl)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    V = ring((-0.15, 0.98), R=0.55, rot=np.deg2rad(-90))   # benzene; V0 bottom
    bond(Cb, V[0])
    return VGroup(*mobs)


def build_tyrosine(s=1.0):
    """Tyrosine — R = -CH2-C6H4-OH (4-hydroxybenzyl)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    cen = (-0.15, 1.05)
    V = ring(cen, R=0.55, rot=np.deg2rad(-90))             # V0 bottom, V3 top (para)
    bond(Cb, V[0])
    bond(V[3], (cen[0], cen[1] + 1.0)); lab((cen[0], cen[1] + 1.20), "O", O_COLOR)
    lab((cen[0] + 0.37, cen[1] + 1.20), "H", H_COLOR)
    return VGroup(*mobs)


def build_histidine(s=1.0):
    """Histidine — R = -CH2-(imidazole, two ring N)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    cx, cy, R = 0.05, 0.78, 0.52
    ang = [np.deg2rad(a) for a in (90, 162, 234, 306, 18)]
    V = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]   # V0 top..V2 lower-left..V3 lower-right
    for k in range(5):
        bond(V[k], V[(k + 1) % 5])
    bond(Cb, V[2])                                              # CH2 -> ring (lower-left)
    lab((V[0][0], V[0][1] + 0.04), "N", N_COLOR)                 # one ring N (top)
    lab((V[3][0] + 0.04, V[3][1] - 0.06), "N", N_COLOR); lab((V[3][0] + 0.04, V[3][1] - 0.40), "H", H_COLOR)
    a, b, c = np.array(V[0]), np.array(V[1]), np.array((cx, cy))  # an inner double bond
    bond(tuple(a + (c - a) * 0.16), tuple(b + (c - b) * 0.16))
    return VGroup(*mobs)


def build_proline(s=1.0):
    """Proline — pyrrolidine ring (secondary amine) bearing the alpha-COOH."""
    mobs, bond, lab, ring = _skeletal(s)
    cx, cy, R = -0.55, 0.0, 0.62
    ang = [np.deg2rad(a) for a in (90, 162, 234, 306, 18)]
    V = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]
    for k in range(5):
        bond(V[k], V[(k + 1) % 5])
    N = V[1]                                          # ring nitrogen (upper-left)
    lab((N[0] - 0.16, N[1] + 0.06), "N", N_COLOR); lab((N[0] - 0.16, N[1] - 0.30), "H", H_COLOR)
    Cax = V[0]                                        # alpha carbon (top), adjacent to N
    bond(Cax, (Cax[0] + 0.66, Cax[1] + 0.34))         # to COOH carbon
    Cc = (Cax[0] + 0.66, Cax[1] + 0.34)
    bond(Cc, (Cc[0] + 0.10, Cc[1] + 0.74), double=True); lab((Cc[0] + 0.10, Cc[1] + 0.94), "O", O_COLOR)
    bond(Cc, (Cc[0] + 0.66, Cc[1] - 0.10)); lab((Cc[0] + 0.82, Cc[1] - 0.14), "O", O_COLOR)
    lab((Cc[0] + 1.20, Cc[1] - 0.14), "H", H_COLOR)
    return VGroup(*mobs)


def build_tryptophan(s=1.0):
    """Tryptophan — R = -CH2-(indole; benzene fused to a pyrrole)."""
    mobs, bond, lab, ring, Ca = _aa_core(s)
    Cb = (-1.01, 0.20); bond(Ca, Cb)
    # indole (mirrored serotonin geometry, shifted so C3 sits near the CH2)
    C3, C2, N1 = (-0.30, 0.10), (-0.89, 0.91), (-0.30, 1.72)
    C7a, C3a = (0.65, 1.41), (0.65, 0.41)
    C7, C6, C5, C4 = (1.52, 1.91), (2.39, 1.41), (2.39, 0.41), (1.52, -0.09)
    bond(Cb, C3)
    bond(C3, C2, double=True); bond(C2, N1); bond(N1, C7a); bond(C7a, C3a); bond(C3a, C3)
    lab((N1[0] - 0.02, N1[1] + 0.16), "N", N_COLOR); lab((N1[0] - 0.40, N1[1] + 0.16), "H", H_COLOR)
    bond(C7a, C7, double=True); bond(C7, C6); bond(C6, C5, double=True)
    bond(C5, C4); bond(C4, C3a, double=True)
    return VGroup(*mobs)


def _hexring(bond, center, R=0.62, rot=0.0):
    """Plain (non-aromatic) hexagon; returns its 6 vertices (no double bonds)."""
    V = [(center[0] + R * np.cos(rot + np.deg2rad(60 * k)),
          center[1] + R * np.sin(rot + np.deg2rad(60 * k))) for k in range(6)]
    for k in range(6):
        bond(V[k], V[(k + 1) % 6])
    return V


def build_glucose(s=1.0):
    """β-D-glucopyranose: a pyranose ring (ring O top) with four OH and a CH2OH."""
    mobs, bond, lab, ring = _skeletal(s)
    V = [(0.70, 0.0), (0.35, 0.62), (-0.35, 0.62), (-0.70, 0.0), (-0.35, -0.62), (0.35, -0.62)]
    for k in range(6):
        if k != 1 and (k + 1) % 6 != 1:           # skip drawing through the O label twice
            bond(V[k], V[(k + 1) % 6])
    bond(V[0], V[1]); bond(V[1], V[2])
    lab(V[1], "O", O_COLOR)                                       # ring oxygen
    bond(V[0], (1.30, 0.0)); lab((1.55, 0.0), "OH", O_COLOR)      # anomeric OH
    bond(V[5], (0.35, -1.25)); lab((0.35, -1.45), "OH", O_COLOR)
    bond(V[4], (-0.35, -1.25)); lab((-0.35, -1.45), "OH", O_COLOR)
    bond(V[3], (-1.30, 0.0)); lab((-1.58, 0.0), "HO", O_COLOR)
    bond(V[2], (-0.35, 1.25)); bond((-0.35, 1.25), (0.22, 1.6))   # CH2-OH
    lab((0.5, 1.7), "OH", O_COLOR)
    return VGroup(*mobs)


def build_sucrose(s=1.0):
    """Sucrose: a glucopyranose (6-ring) and a fructofuranose (5-ring) linked by an
    ether O. Schematic but reads as two linked sugar rings with OH/CH2OH."""
    mobs, bond, lab, ring = _skeletal(s)
    # left pyranose
    G = [(-1.2, 0.0), (-1.5, 0.6), (-2.2, 0.6), (-2.5, 0.0), (-2.2, -0.6), (-1.5, -0.6)]
    for k in range(6):
        a, b = G[k], G[(k + 1) % 6]
        if not (a == G[1] or b == G[1]):
            bond(a, b)
    bond(G[0], G[1]); bond(G[1], G[2]); lab(G[1], "O", O_COLOR)
    bond(G[3], (-3.1, 0.0)); lab((-3.35, 0.0), "HO", O_COLOR)
    bond(G[4], (-2.2, -1.2)); lab((-2.2, -1.4), "OH", O_COLOR)
    bond(G[2], (-2.2, 1.2)); bond((-2.2, 1.2), (-2.75, 1.55)); lab((-3.0, 1.65), "OH", O_COLOR)
    # bridging ether O
    bond(G[0], (-0.5, -0.1)); lab((-0.35, -0.18), "O", O_COLOR)
    # right furanose (5-ring)
    F = [(0.2, 0.0), (0.95, 0.35), (1.55, -0.15), (1.2, -0.85), (0.4, -0.7)]
    for k in range(5):
        a, b = F[k], F[(k + 1) % 5]
        if not (a == F[1] or b == F[1]):
            bond(a, b)
    bond(F[0], F[1]); bond(F[1], F[2]); lab(F[1], "O", O_COLOR)
    bond((-0.35, -0.05), F[0])
    bond(F[2], (2.15, 0.1)); bond((2.15, 0.1), (2.4, 0.6)); lab((2.55, 0.72), "OH", O_COLOR)
    bond(F[3], (1.4, -1.45)); lab((1.4, -1.65), "OH", O_COLOR)
    bond(F[4], (0.0, -1.25)); lab((-0.15, -1.4), "OH", O_COLOR)
    return VGroup(*mobs)


def build_citric_acid(s=1.0):
    """Citric acid: central C(OH) bearing a COOH and two CH2COOH arms."""
    mobs, bond, lab, ring = _skeletal(s)
    Cc = (0.0, 0.0)
    bond(Cc, (0.0, 0.7)); lab((0.0, 0.92), "OH", O_COLOR)                    # central OH

    def cooh(a, b, odir, hdir):
        bond(a, b)
        bond(b, (b[0] + odir[0], b[1] + odir[1]), double=True)
        lab((b[0] + odir[0] * 1.25, b[1] + odir[1] * 1.25), "O", O_COLOR)
        bond(b, (b[0] + hdir[0], b[1] + hdir[1])); lab((b[0] + hdir[0] * 1.35, b[1] + hdir[1] * 1.35), "OH", O_COLOR)

    cooh(Cc, (0.78, 0.42), (0.5, 0.3), (0.55, -0.25))                        # top-right COOH (direct)
    bond(Cc, (-0.62, -0.45)); cooh((-0.62, -0.45), (-1.3, -0.1), (-0.1, 0.55), (-0.55, -0.25))  # left CH2COOH
    bond(Cc, (0.62, -0.45)); cooh((0.62, -0.45), (0.55, -1.15), (0.55, -0.3), (-0.55, -0.15))   # down CH2COOH
    return VGroup(*mobs)


def build_vanillin(s=1.0):
    """Vanillin: 4-hydroxy-3-methoxybenzaldehyde (CHO + OH + OCH3 on benzene)."""
    mobs, bond, lab, ring = _skeletal(s)
    V = ring((0, 0), R=0.62)                                      # V0 right, V3 left
    bond(V[0], (1.2, 0.0)); bond((1.2, 0.0), (1.2, 0.66), double=True); lab((1.2, 0.86), "O", O_COLOR)
    bond((1.2, 0.0), (1.78, -0.28)); lab((2.0, -0.34), "H", H_COLOR)         # CHO
    bond(V[3], (-1.2, 0.0)); lab((-1.48, 0.0), "HO", O_COLOR)                # 4-OH
    bond(V[2], (-0.62, 1.05)); lab((-0.78, 1.16), "O", O_COLOR)              # 3-OCH3
    bond((-0.62, 1.05), (-0.1, 1.4)); lab((0.18, 1.5), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_menthol(s=1.0):
    """Menthol: 2-isopropyl-5-methylcyclohexan-1-ol."""
    mobs, bond, lab, ring = _skeletal(s)
    V = _hexring(bond, (0, 0), R=0.65)
    bond(V[0], (1.22, 0.0)); lab((1.48, 0.0), "OH", O_COLOR)                 # C1-OH
    bond(V[1], (0.32, 1.12)); bond((0.32, 1.12), (-0.12, 1.5)); lab((-0.34, 1.62), "CH₃", MOL_DIM)
    bond((0.32, 1.12), (0.78, 1.5)); lab((1.0, 1.62), "CH₃", MOL_DIM)        # C2-isopropyl
    bond(V[4], (-0.32, -1.12)); lab((-0.32, -1.34), "CH₃", MOL_DIM)         # C5-methyl
    return VGroup(*mobs)


def build_ibuprofen(s=1.0):
    """Ibuprofen: p-isobutyl ring + CH(CH3)COOH."""
    mobs, bond, lab, ring = _skeletal(s)
    V = ring((0, 0), R=0.62)
    bond(V[3], (-1.2, 0.0)); bond((-1.2, 0.0), (-1.72, 0.34))               # CH2-CH(
    bond((-1.72, 0.34), (-2.24, 0.06)); lab((-2.5, 0.0), "CH₃", MOL_DIM)
    bond((-1.72, 0.34), (-1.72, 0.96)); lab((-1.72, 1.16), "CH₃", MOL_DIM)
    bond(V[0], (1.2, 0.0)); bond((1.2, 0.0), (1.2, 0.66)); lab((1.2, 0.86), "CH₃", MOL_DIM)
    bond((1.2, 0.0), (1.78, -0.34))                                          # CH-COOH
    bond((1.78, -0.34), (1.78, -1.0), double=True); lab((1.78, -1.2), "O", O_COLOR)
    bond((1.78, -0.34), (2.4, -0.1)); lab((2.66, -0.12), "OH", O_COLOR)
    return VGroup(*mobs)


def build_vitamin_c(s=1.0):
    """Ascorbic acid: a γ-lactone furanone with a 2,3-enediol and a CH(OH)CH2OH tail."""
    mobs, bond, lab, ring = _skeletal(s)
    O1, C1, C2, C3, C4 = (0.0, 0.7), (-0.7, 0.35), (-0.55, -0.45), (0.3, -0.55), (0.65, 0.15)
    bond(O1, C1); bond(C1, C2); bond(C2, C3, double=True); bond(C3, C4); bond(C4, O1)
    lab((0.05, 0.88), "O", O_COLOR)                                          # ring O
    bond(C1, (-1.25, 0.7), double=True); lab((-1.45, 0.85), "O", O_COLOR)    # lactone C=O
    bond(C2, (-0.95, -1.05)); lab((-1.12, -1.2), "OH", O_COLOR)              # enediol OH
    bond(C3, (0.55, -1.25)); lab((0.7, -1.4), "OH", O_COLOR)                 # enediol OH
    bond(C4, (1.3, 0.4)); bond((1.3, 0.4), (1.55, -0.2)); lab((1.75, -0.32), "OH", O_COLOR)  # CH(OH)
    bond((1.3, 0.4), (1.85, 0.75)); lab((2.05, 0.88), "OH", O_COLOR)         # CH2OH
    return VGroup(*mobs)


def _steroid_nucleus(bond, lab, *, dx=0.0, dy=0.0):
    """The cyclopentanoperhydrophenanthrene steroid core: rings A,B,C (6) + D (5),
    fused. Returns key positions for substituents. Draws bonds (one B-ring C=C)."""
    P = lambda x, y: (x + dx, y + dy)
    A = [P(-3.0, 0.30), P(-3.0, -0.50), P(-2.3, -0.92), P(-1.6, -0.50), P(-1.6, 0.30), P(-2.3, 0.72)]
    B = [P(-1.6, 0.30), P(-1.6, -0.50), P(-0.9, -0.92), P(-0.2, -0.50), P(-0.2, 0.30), P(-0.9, 0.72)]
    C = [P(-0.2, 0.30), P(-0.2, -0.50), P(0.5, -0.92), P(1.2, -0.50), P(1.2, 0.30), P(0.5, 0.72)]
    D = [P(1.2, 0.30), P(1.2, -0.50), P(1.95, -0.62), P(2.38, 0.0), P(1.95, 0.62)]
    for R in (A, B, C):
        for k in range(6):
            bond(R[k], R[(k + 1) % 6])
    for k in range(5):
        bond(D[k], D[(k + 1) % 5])
    bond(A[5], (A[5][0], A[5][1] + 0.72)); lab((A[5][0], A[5][1] + 0.92), "CH₃", MOL_DIM)  # C19
    bond(C[5], (C[5][0], C[5][1] + 0.72)); lab((C[5][0], C[5][1] + 0.92), "CH₃", MOL_DIM)  # C18
    return dict(A=A, B=B, C=C, D=D, P=P)


def build_cholesterol(s=1.0):
    """Cholesterol: steroid nucleus, 3β-OH, Δ5 double bond, iso-octyl tail."""
    mobs, bond, lab, ring = _skeletal(s)
    n = _steroid_nucleus(bond, lab)
    A, B, C, D, P = n["A"], n["B"], n["C"], n["D"], n["P"]
    bond(B[4], B[5], double=True)                                            # Δ5,6
    bond(A[2], P(-2.7, -1.45)); lab(P(-2.95, -1.6), "HO", O_COLOR)           # 3-OH
    # iso-octyl tail off D apex
    t = [P(2.38, 0.0), P(2.95, 0.32), P(3.5, 0.0), P(4.05, 0.32), P(4.6, 0.0), P(5.15, 0.32)]
    for k in range(len(t) - 1):
        bond(t[k], t[k + 1])
    bond(t[1], P(2.95, 0.95)); lab(P(2.95, 1.12), "CH₃", MOL_DIM)
    bond(t[5], P(5.15, 0.95)); lab(P(5.15, 1.12), "CH₃", MOL_DIM)
    bond(t[5], P(5.7, 0.0)); lab(P(5.95, -0.05), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_testosterone(s=1.0):
    """Testosterone: steroid nucleus, 17β-OH, 4-en-3-one (A-ring enone)."""
    mobs, bond, lab, ring = _skeletal(s)
    n = _steroid_nucleus(bond, lab)
    A, D, P = n["A"], n["D"], n["P"]
    bond(A[2], P(-2.3, -1.6), double=True); lab(P(-2.3, -1.8), "O", O_COLOR)  # 3-one
    bond(A[3], A[4], double=True)                                            # Δ4
    bond(D[3], P(2.95, 0.3)); lab(P(3.2, 0.34), "OH", O_COLOR)               # 17-OH
    return VGroup(*mobs)


def build_melatonin(s=1.0):
    """Melatonin: 5-methoxy-N-acetyltryptamine (methoxy-indole + acetamide chain)."""
    mobs, bond, lab, ring = _skeletal(s)
    C7a, C3a = (0.0, 0.5), (0.0, -0.5)
    C7, C6, C5, C4 = (-0.87, 1.0), (-1.74, 0.5), (-1.74, -0.5), (-0.87, -1.0)
    N1, C2, C3 = (0.95, 0.81), (1.54, 0.0), (0.95, -0.81)
    bond(C7a, C7, double=True); bond(C7, C6); bond(C6, C5, double=True)
    bond(C5, C4); bond(C4, C3a, double=True); bond(C3a, C7a)
    bond(C7a, N1); bond(N1, C2); bond(C2, C3, double=True); bond(C3, C3a)
    lab((1.18, 1.12), "NH", N_COLOR, fs=34)
    # 5-OCH3
    bond(C5, (-2.5, -0.85)); lab((-2.66, -0.97), "O", O_COLOR, fs=34)
    bond((-2.5, -0.85), (-2.5, -1.5)); lab((-2.5, -1.68), "CH₃", MOL_DIM, fs=30)
    # 3-(2-acetamidoethyl)
    bond(C3, (1.55, -1.25)); bond((1.55, -1.25), (2.2, -0.95))               # CH2-CH2
    bond((2.2, -0.95), (2.85, -1.3)); lab((3.02, -1.42), "NH", N_COLOR, fs=34)
    bond((2.85, -1.3), (3.5, -0.98)); bond((3.5, -0.98), (3.5, -0.32), double=True)
    lab((3.5, -0.14), "O", O_COLOR, fs=34); bond((3.5, -0.98), (4.15, -1.3)); lab((4.4, -1.34), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_nicotine(s=1.0):
    """Nicotine: 3-pyridyl + N-methylpyrrolidine, joined by a single bond."""
    mobs, bond, lab, ring = _skeletal(s)
    # pyridine (aromatic) on the left; replace one vertex with N
    Py = ring((-1.25, 0.0), R=0.62)                                          # Kekulé benzene drawn
    lab(Py[4], "N", N_COLOR)                                                 # pyridine N (lower-left)
    # pyrrolidine (saturated 5-ring) on the right, N-CH3
    Pr = [(0.55, 0.0), (1.25, 0.32), (1.7, -0.28), (1.25, -0.85), (0.6, -0.6)]
    for k in range(5):
        bond(Pr[k], Pr[(k + 1) % 5])
    lab(Pr[2], "N", N_COLOR)                                                 # pyrrolidine N
    bond(Pr[2], (2.35, -0.1)); lab((2.6, -0.05), "CH₃", MOL_DIM)
    bond(Py[0], Pr[0])                                                       # link C3(py)-C2(pyrrolidine)
    return VGroup(*mobs)


def build_cbd(s=1.0):
    """Cannabidiol: a pentyl-resorcinol ring linked to a methyl/isopropenyl cyclohexene."""
    mobs, bond, lab, ring = _skeletal(s)
    Ar = ring((-1.7, 0.0), R=0.62)                  # resorcinol; Ar0 right vertex
    bond(Ar[2], (-2.4, 0.95)); lab((-2.58, 1.06), "OH", O_COLOR)             # phenolic OH (1)
    bond(Ar[4], (-2.4, -0.95)); lab((-2.58, -1.06), "HO", O_COLOR)           # phenolic OH (3)
    # pentyl tail (down-right from the lower vertex)
    p = [Ar[5], (-1.0, -1.05), (-0.4, -1.4), (0.2, -1.05), (0.8, -1.4), (1.4, -1.05)]
    for k in range(len(p) - 1):
        bond(p[k], p[k + 1])
    # cyclohexene ring on the right, linked Ar0 -> Cy left vertex
    Cy = _hexring(bond, (0.35, 0.5), R=0.6)         # Cy3 = (-0.25, 0.5) left vertex
    bond(Ar[0], Cy[3])
    bond(Cy[1], Cy[2], double=True)                 # ring C=C
    bond(Cy[1], (1.15, 1.4)); lab((1.35, 1.5), "CH₃", MOL_DIM)               # methyl on the ene
    bond(Cy[4], (0.05, -0.55)); bond((0.05, -0.55), (-0.45, -0.95), double=True)  # isopropenyl =CH2
    bond((0.05, -0.55), (0.55, -0.95)); lab((0.75, -1.05), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def build_morphine(s=1.0):
    """Morphine — clean recognizable representation: a row of fused rings (aromatic A
    → C → piperidine N-CH3), the furan ether O bridging below, phenolic + allylic OH
    (schematic, not a rigorous stereo drawing)."""
    mobs, bond, lab, ring = _skeletal(s)
    # aromatic A-ring (explicit, with Kekulé doubles), right edge shared with ring C
    A_TR, A_BR = (-1.2, 0.85), (-1.2, 0.05)
    A_BL2, A_BL, A_TL, A_T = (-1.9, -0.35), (-2.6, 0.05), (-2.6, 0.85), (-1.9, 1.25)
    Aedges = [(A_TR, A_BR), (A_BR, A_BL2), (A_BL2, A_BL), (A_BL, A_TL), (A_TL, A_T), (A_T, A_TR)]
    for a, b in Aedges:
        bond(a, b)
    # inner Kekulé bars
    cen = np.array([-1.9, 0.45])
    for a, b in [(A_BR, A_BL2), (A_BL, A_TL), (A_T, A_TR)]:
        aa, bb = np.array(a), np.array(b)
        bond(tuple(aa + (cen - aa) * 0.16), tuple(bb + (cen - bb) * 0.16))
    # ring C (cyclohexane) fused on A_TR-A_BR
    Cb, Cc, Cd, Ce = (-0.5, -0.35), (0.2, 0.05), (0.2, 0.85), (-0.5, 1.25)
    for a, b in [(A_BR, Cb), (Cb, Cc), (Cc, Cd), (Cd, Ce), (Ce, A_TR)]:
        bond(a, b)
    # furan ether O bridging A_BL2 — C ring (below)
    bond(A_BL2, (-1.2, -0.7)); lab((-1.35, -0.82), "O", O_COLOR); bond((-1.2, -0.7), Cb)
    # piperidine N-CH3 ring fused on Cc-Cd
    Na, Nb, Nc, Nd = (0.9, -0.35), (1.55, 0.05), (1.55, 0.85), (0.9, 1.25)
    for a, b in [(Cc, Na), (Na, Nb), (Nb, Nc), (Nc, Nd), (Nd, Cd)]:
        bond(a, b)
    lab((1.72, 0.05), "N", N_COLOR); bond(Nb, (2.2, -0.28)); lab((2.45, -0.34), "CH₃", MOL_DIM)
    # phenolic + allylic OH
    bond(A_TL, (-3.2, 1.15)); lab((-3.45, 1.25), "HO", O_COLOR)
    bond(Na, (0.9, -1.0)); lab((0.9, -1.2), "OH", O_COLOR)
    return VGroup(*mobs)


def build_penicillin(s=1.0):
    """Penicillin G — the iconic β-lactam (4-ring) fused to a thiazolidine (5-ring,
    sulfur), with the gem-dimethyl, COOH, and phenylacetamide side chain."""
    mobs, bond, lab, ring = _skeletal(s)
    # β-lactam (4-membered): N-C(=O)-C-C
    L = [(0.0, 0.5), (0.0, -0.3), (0.8, -0.3), (0.8, 0.5)]
    bond(L[0], L[1]); bond(L[1], L[2]); bond(L[2], L[3]); bond(L[3], L[0])
    lab(L[0], "N", N_COLOR)
    bond(L[3], (0.8, 1.15), double=True); lab((0.8, 1.35), "O", O_COLOR)     # lactam C=O
    # thiazolidine (5-membered, shares N-C edge L0-L1): N-CH-S-C(Me)2-CH
    T = [L[0], L[1], (-0.7, -0.7), (-1.25, 0.0), (-0.7, 0.7)]
    bond(L[1], T[2]); bond(T[2], T[3]); bond(T[3], T[4]); bond(T[4], L[0])
    lab((-0.78, -0.82), "S", "#e0c850")                                      # sulfur (yellow-ish)
    bond(T[3], (-1.9, -0.3)); lab((-2.15, -0.38), "CH₃", MOL_DIM)            # gem-dimethyl
    bond(T[3], (-1.9, 0.3)); lab((-2.15, 0.4), "CH₃", MOL_DIM)
    bond(T[4], (-0.7, 1.4)); bond((-0.7, 1.4), (-0.05, 1.7), double=True); lab((-0.7, 1.6), "O", O_COLOR)
    bond((-0.7, 1.4), (0.0, 1.55))                                           # (sketch COOH near ring)
    lab((0.2, 1.62), "OH", O_COLOR)
    # phenylacetamide side chain off the lactam C (L2)
    bond(L[2], (1.5, -0.7)); lab((1.62, -0.85), "NH", N_COLOR)
    bond((1.5, -0.7), (2.15, -0.4)); bond((2.15, -0.4), (2.15, 0.25), double=True); lab((2.15, 0.43), "O", O_COLOR)
    bond((2.15, -0.4), (2.8, -0.7))                                          # CH2
    Ph = ring((3.5, -0.9), R=0.55)
    bond((2.8, -0.7), Ph[2])
    return VGroup(*mobs)


def build_capsaicin(s=1.0):
    """Capsaicin: vanillyl (4-OH-3-OCH3 benzyl) + amide + acyl chain to a branched tail."""
    mobs, bond, lab, ring = _skeletal(s)
    Ar = ring((-2.8, 0.0), R=0.58)
    bond(Ar[3], (-3.45, 0.0)); lab((-3.72, 0.0), "HO", O_COLOR)              # 4-OH
    bond(Ar[2], (-3.2, 0.95)); lab((-3.36, 1.06), "O", O_COLOR)              # 3-OMe
    bond((-3.2, 0.95), (-2.75, 1.35)); lab((-2.55, 1.46), "CH₃", MOL_DIM)
    # CH2-NH-C(=O)- then chain
    bond(Ar[0], (-2.05, -0.35)); bond((-2.05, -0.35), (-1.45, 0.0)); lab((-1.4, -0.2), "NH", N_COLOR)
    bond((-1.45, 0.0), (-0.85, -0.35)); bond((-0.85, -0.35), (-0.85, 0.3), double=True)
    lab((-0.85, 0.5), "O", O_COLOR)
    chain = [(-0.85, -0.35), (-0.2, -0.0), (0.45, -0.35), (1.1, 0.0), (1.75, -0.35), (2.4, 0.0)]
    for k in range(len(chain) - 1):
        bond(chain[k], chain[k + 1])
    bond(chain[5], (3.05, -0.35), double=True)                              # C=C (trans)
    bond((3.05, -0.35), (3.7, 0.0))                                          # to CH(CH3)2
    bond((3.7, 0.0), (4.2, 0.4)); lab((4.4, 0.5), "CH₃", MOL_DIM)
    bond((3.7, 0.0), (4.2, -0.4)); lab((4.4, -0.5), "CH₃", MOL_DIM)
    return VGroup(*mobs)


def styled_molecule(mol, *, mol_height=1.5, label_fs=None, base_label_fs=30,
                    label_push=1.0, push_power=1.6, label_halo=False, halo_color=BG,
                    halo_r=0.82, min_double_sep=0.06):
    """Apply the house structure styling to a molecule mobject IN PLACE and return it:
    scale the BODY to `mol_height`, force a fixed on-screen symbol size (`label_fs`,
    decoupled from `mol_height`), push atom labels radially OUT (`label_push`, weaker near
    the centre via `push_power`), drop a BG-colour halo behind each symbol (`label_halo`),
    and keep double-bond lines a minimum gap apart (`min_double_sep`). This is the styling
    core shared with `molecule_tag` — use it to drop a preset-styled structure ANYWHERE
    (e.g. a header) without the title/subtitle/tag. Pair with `STRUCTURE_PRESET`."""
    raw_h = mol.height
    sf = (mol_height / raw_h) if raw_h else 1.0
    mol.scale(sf)
    labels = [s for s in mol.submobjects if isinstance(s, Text)]
    if labels and (label_fs is not None or label_push != 1.0 or label_halo):
        ctr = np.array(mol.get_center())
        dists = [float(np.linalg.norm(np.array(s.get_center()) - ctr)) for s in labels]
        dmax = max(dists) or 1.0
        halos = VGroup()
        for s, d in zip(labels, dists):
            if label_fs is not None:
                s.scale(label_fs / (base_label_fs * sf))           # fixed symbol size
            if label_push != 1.0:                                  # nudge OUT, weaker near centre
                w = (d / dmax) ** push_power
                s.move_to(ctr + (np.array(s.get_center()) - ctr) * (1.0 + (label_push - 1.0) * w))
            s.set_z_index(3)
            if label_halo:                                         # BG disk behind the symbol
                halos.add(Circle(radius=halo_r * s.height, color=halo_color, fill_color=halo_color,
                                 fill_opacity=1.0, stroke_width=0).move_to(s.get_center()).set_z_index(2))
        if label_halo:
            mol.add(halos)
    if min_double_sep > 0:                                  # keep double bonds visibly split
        pairs = {}
        for sub in mol.submobjects:
            pid = getattr(sub, "_double_pair", None)
            if pid is not None:
                pairs.setdefault(pid, []).append(sub)
        for lns in pairs.values():
            if len(lns) != 2:
                continue
            c0 = np.array(lns[0].get_center()); c1 = np.array(lns[1].get_center())
            sep_vec = c1 - c0; sep = float(np.linalg.norm(sep_vec))
            if 1e-6 < sep < min_double_sep:                 # too close -> widen to the floor
                u = sep_vec / sep; mid = (c0 + c1) / 2.0
                lns[0].move_to(mid - u * (min_double_sep / 2.0))
                lns[1].move_to(mid + u * (min_double_sep / 2.0))
    return mol


def molecule_tag(mol, title, subtitle, center, *, mol_height=1.5, mol_dy=-0.4,
                 header_dy=1.55, title_fs=30, sub_fs=18, title_color=INK, sub_color=MUTED,
                 label_fs=None, base_label_fs=30, label_push=1.0, push_power=1.6,
                 label_halo=False, halo_color=BG, halo_r=0.82, min_double_sep=0.06,
                 tag=None, tag_dy=-1.85, tag_fs=17):
    """An Information/Tag block for a molecule (landscape): a bold TITLE + subtitle near
    the top, the molecule centred below — the BODY scaled to `mol_height` so structure
    size is independent of the (fixed) text sizes.

    Functional-group symbols are controllable too: `label_fs` forces every atom label to a
    fixed on-screen font size (DECOUPLED from `mol_height`, assuming they were drawn at
    `base_label_fs`), and `label_push` (>1) moves each label radially OUT from the molecule
    centre by that factor so the symbols clear the bonds. Optional `tag` at the bottom.
    Returns a VGroup at `center`. House template for consistent structure sizing."""
    cx, cy = center[0], center[1]
    g = VGroup()
    header = VGroup(
        Text(title, font_size=title_fs, color=title_color, weight=BOLD),
        Text(subtitle, font_size=sub_fs, color=sub_color),
    ).arrange(DOWN, buff=0.12).move_to([cx, cy + header_dy, 0])
    g.add(header)
    styled_molecule(mol, mol_height=mol_height, label_fs=label_fs, base_label_fs=base_label_fs,
                    label_push=label_push, push_power=push_power, label_halo=label_halo,
                    halo_color=halo_color, halo_r=halo_r, min_double_sep=min_double_sep)
    mol.move_to([cx, cy + mol_dy, 0])
    g.add(mol)
    if tag:
        g.add(Text(tag, font_size=tag_fs, color=MUTED).move_to([cx, cy + tag_dy, 0]))
    return g


def billboard_rotate(mol, angle, *, axis_point=ORIGIN, axis_dir=UP):
    """Rotate `mol` about the IN-PLANE axis (axis_point, axis_dir) by `angle`, in 3D, and
    let the 2D camera project it orthographically (z is dropped at render). BONDS rotate
    fully (they foreshorten as the molecule turns); TEXT labels only have their POSITION
    rotated — the glyphs are NOT rotated, so they stay UPRIGHT facing the camera while
    moving as if rotating (billboard). Mutates and returns `mol`. Call BEFORE molecule_tag."""
    ap = np.array(axis_point, float)
    k = np.array(axis_dir, float); k = k / (np.linalg.norm(k) + 1e-12)
    ca, sa = np.cos(angle), np.sin(angle)
    for sub in mol.submobjects:
        if isinstance(sub, Text):
            v = np.array(sub.get_center()) - ap                    # Rodrigues rotation of the centre
            nv = v * ca + np.cross(k, v) * sa + k * np.dot(k, v) * (1 - ca)
            sub.move_to(ap + nv)
        else:
            sub.rotate(angle, axis=axis_dir, about_point=ap)
    return mol


# Saved house preset for molecule structure tags (locked 2026-06-10). Apply with
# `molecule_tag(mol, title, sub, center, mol_height=…, label_fs=…, tag=…, **STRUCTURE_PRESET)`.
STRUCTURE_PRESET = dict(
    title_fs=30, sub_fs=18, tag_fs=17,
    label_push=1.13, push_power=1.6, base_label_fs=30,
    label_halo=True, halo_color=BG, halo_r=0.82,
    min_double_sep=0.06,
    mol_dy=-0.3, header_dy=1.5, tag_dy=-1.9,
)


def molecule_glow_updater(groups, grp_centers, scan, scan_amt, scan_cutoff):
    """Factory returning an updater that glows molecular `groups` by proximity of a
    scan line (ValueTracker `scan`) to each group's band centers; `scan_amt` (0..1)
    gates the whole effect. Each mob's dim colour is read from `.glow_base`."""
    def grp_inten(g):
        sx = scan.get_value()
        best = 0.0
        for c in grp_centers.get(g, []):
            d = abs(sx - c)
            if d < scan_cutoff:
                best = max(best, 1.0 - d / scan_cutoff)
        return best

    def update_mol(m):
        a = scan_amt.get_value()
        for g, mobs in groups.items():
            inten = grp_inten(g) * a
            for mo in mobs:
                base = getattr(mo, "glow_base", MOL_DIM)   # O red / H white / bond gray
                col = interpolate_color(ManimColor(base), ManimColor(MOL_GLOW), inten)
                mo.set_color(col)
                if not isinstance(mo, Text):
                    mo.set_stroke(width=3.0 + 2.6 * inten)
    return update_mol


# =============================================================================
#  DOMAIN: SPECTROGRAPHY PRIMITIVES
#  Theme A — where a spectrum comes from (dispersion, optical bench, FT-IR).
#  Theme B — why peaks exist (energy levels, photons, molecular vibrations).
#  Composable building blocks: stateless static builders + factory closures that
#  return an always_redraw builder driven by ValueTrackers (canonical idiom).
# =============================================================================

def visible_color(nm):
    """Approximate sRGB for a visible wavelength (380-750 nm) as a manim color
    (Bruton's piecewise map + end-of-range intensity falloff + mild gamma)."""
    w = float(np.clip(nm, 380.0, 750.0))
    if   w < 440: r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif w < 490: r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif w < 510: r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif w < 580: r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif w < 645: r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    else:         r, g, b = 1.0, 0.0, 0.0
    if   w < 420: f = 0.30 + 0.70 * (w - 380) / (420 - 380)
    elif w > 700: f = 0.30 + 0.70 * (750 - w) / (750 - 700)
    else:         f = 1.0
    return rgb_to_color([(max(0.0, c) * f) ** 0.80 for c in (r, g, b)])


def energy_to_color(frac):
    """Map a normalised energy gap (0..1) to a photon colour: larger gap -> bluer
    (shorter wavelength). frac 0 -> ~720 nm (red), frac 1 -> ~400 nm (violet)."""
    return visible_color(720.0 - 320.0 * float(np.clip(frac, 0.0, 1.0)))


# ---- Theme A: where a spectrum comes from ----------------------------------

def prism(center=ORIGIN, *, size=1.7, color=GRID, fill="#cfe0f5", fill_opacity=0.06):
    """Equilateral dispersing prism (apex up). Reusable glyph."""
    c = np.array([center[0], center[1], 0.0])
    h = size * np.sqrt(3.0) / 2.0
    pts = [c + np.array([0.0, 2 * h / 3, 0.0]),
           c + np.array([-size / 2, -h / 3, 0.0]),
           c + np.array([size / 2, -h / 3, 0.0])]
    return Polygon(*pts, color=color, stroke_width=FRAME_W,
                   fill_color=fill, fill_opacity=fill_opacity)


def disperse_spectrum(prism_center=ORIGIN, *, size=1.7, in_len=3.0, n=60,
                      spread=28 * DEGREES, length=4.4, tilt=-18 * DEGREES,
                      lam_lo=400, lam_hi=700, white=INK, with_prism=True,
                      style="filled", sw=5, cauchy_A=1.50, cauchy_B=4.0e3):
    """White beam -> prism -> visible spectrum, dispersed by a REAL Cauchy index
    n(λ)=A+B/λ² (λ in nm). Because dn/dλ steepens toward the blue, the violet end fans
    out far more than the red — and the IR band (just past red) bunches into a tiny
    angular range while the UV band (just past violet) spreads widely. Each wavelength
    maps to a deviation angle through n(λ); `spread` is the 400->700 nm visible fan
    angle (red = least-deviated top, violet = most-deviated bottom). Exposes
    `.angle_of(λ)` so callers place UV/IR markers on the SAME physical curve, plus
    .exit_pt/.length/.ang_red/.ang_violet/.n_of. style='filled' = gap-free wedges."""
    pc = np.array([prism_center[0], prism_center[1], 0.0])
    g = VGroup()
    pr = prism(pc, size=size) if with_prism else None
    entry = pc + np.array([-size * 0.18, 0.0, 0.0])
    in_beam = Line(entry + np.array([-in_len, 0.0, 0.0]), entry, color=white, stroke_width=7)
    exit_pt = pc + np.array([size * 0.20, -size * 0.05, 0.0])

    def n_of(lam):
        return cauchy_A + cauchy_B / (lam * lam)

    n_red = n_of(lam_hi)
    dn = (n_of(lam_lo) - n_red) or 1e-9
    ang_red = tilt + 0.5 * spread

    def angle_of(lam):                                # higher n -> more deviation (downward)
        return ang_red - (n_of(lam) - n_red) / dn * spread

    def dir_of(lam):
        a = angle_of(lam)
        return np.array([np.cos(a), np.sin(a), 0.0])

    fan = VGroup()
    if style == "filled":
        for i in range(n - 1):
            l0 = lam_hi + (lam_lo - lam_hi) * i / (n - 1)
            l1 = lam_hi + (lam_lo - lam_hi) * (i + 1) / (n - 1)
            col = visible_color(0.5 * (l0 + l1))
            fan.add(Polygon(exit_pt, exit_pt + dir_of(l0) * length, exit_pt + dir_of(l1) * length,
                            color=col, fill_color=col, fill_opacity=1.0, stroke_width=1.2
                            ).set_stroke(col, width=1.2))
    else:
        for i in range(n):
            lam = lam_hi + (lam_lo - lam_hi) * i / (n - 1)
            fan.add(Line(exit_pt, exit_pt + dir_of(lam) * length, color=visible_color(lam), stroke_width=sw))
    if pr is not None:
        g.add(pr)
    g.add(in_beam, fan)
    g.prism, g.in_beam, g.fan = pr, in_beam, fan
    g.exit_pt, g.length, g.angle_of, g.n_of = exit_pt, length, angle_of, n_of
    g.ang_red, g.ang_violet = angle_of(lam_hi), angle_of(lam_lo)
    return g


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _ray_seg(O, d, A, B):
    """Distance t>0 along ray O+t*d to its intersection with segment A-B, or None."""
    O = np.asarray(O, float)[:2]; d = np.asarray(d, float)[:2]
    A = np.asarray(A, float)[:2]; B = np.asarray(B, float)[:2]
    e = B - A
    den = _cross2(d, e)
    if abs(den) < 1e-12:
        return None
    w = A - O
    t = _cross2(w, e) / den
    u = _cross2(w, d) / den
    return t if (t > 1e-7 and -1e-9 <= u <= 1 + 1e-9) else None


def _refract(d, nrm, eta):
    """Refract unit dir d across a surface with unit normal nrm (any orientation);
    eta = n_from/n_to. Returns the refracted unit dir, or None on total internal
    reflection (Snell, vector form)."""
    d = np.asarray(d, float)[:2]; n = np.asarray(nrm, float)[:2]
    if np.dot(n, d) > 0:                          # make the normal oppose the ray
        n = -n
    cosi = -np.dot(n, d)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return None                               # TIR
    out = eta * d + (eta * cosi - np.sqrt(k)) * n
    return out / (np.linalg.norm(out) + 1e-12)


def _edge_normal(A, B):
    e = np.asarray(B, float)[:2] - np.asarray(A, float)[:2]
    nrm = np.array([-e[1], e[0]])
    return nrm / (np.linalg.norm(nrm) + 1e-12)


def _prism_exit(verts, O, d, n_glass, max_bounce=5):
    """Trace ray (O,d) INTO a triangular glass prism and out — refracting at the entry
    face, then walking face to face, refracting OUT where it can and TOTAL-INTERNAL-
    REFLECTING (bouncing) where it can't, until it escapes. Returns (path_pts, exit_pt,
    out_dir): path_pts is the internal polyline [entry, bounce…, exit]. None if no entry."""
    edges = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
    hits = [(t, i) for i, (A, B) in enumerate(edges) for t in [_ray_seg(O, d, A, B)] if t]
    if not hits:
        return None
    t1, ei = min(hits)
    P1 = np.asarray(O, float)[:2] + t1 * np.asarray(d, float)[:2]
    d1 = _refract(d, _edge_normal(*edges[ei]), 1.0 / n_glass)      # air -> glass
    if d1 is None:
        return None
    pts = [P1]
    cur_p, cur_d, last = P1 + d1 * 1e-4, d1, ei
    for _ in range(max_bounce):
        hits2 = [(t, j) for j, (A, B) in enumerate(edges) if j != last
                 for t in [_ray_seg(cur_p, cur_d, A, B)] if t]
        if not hits2:
            return None
        t2, ej = min(hits2)
        P2 = cur_p + t2 * cur_d
        pts.append(P2)
        nrm = _edge_normal(*edges[ej])
        d_out = _refract(cur_d, nrm, n_glass)                     # try to exit (glass -> air)
        if d_out is not None:
            return pts, P2, d_out
        nu = nrm / (np.linalg.norm(nrm) + 1e-12)                  # TIR: reflect internally
        cur_d = cur_d - 2.0 * np.dot(cur_d, nu) * nu
        cur_p, last = P2 + cur_d * 1e-4, ej
    return None


def refracting_prism(center, phi, *, R=1.5, lam_lo=400, lam_hi=700, n_rays=26,
                     out_len=4.5, white=INK, glass=GRID, cauchy_A=1.50, cauchy_B=1.1e4,
                     beam_y=0.0, in_x=-5.0, beam_angle=0.0, bands=True):
    """Factory: a white beam refracted THROUGH a triangular prism that rotates by `phi`
    (tracker). Each wavelength is ray-traced with Snell's law at the entry AND exit
    faces using n(λ)=A+B/λ², so the fan deviates and disperses with orientation and
    VANISHES at total-internal-reflection angles. Returns an always_redraw builder
    drawing prism + incoming beam + internal segments + exit fan (+ dashed IR/UV).

    `beam_angle` (rad) tilts the INCOMING beam off horizontal (the beam is back-projected
    so it still crosses the aim point at the centroid height + `beam_y`). beam_angle=0 is
    the classic horizontal beam — needed so an UPRIGHT (apex-up) prism still disperses at
    minimum deviation instead of total-internal-reflecting the short wavelengths."""
    C = np.array([center[0], center[1], 0.0])

    def n_of(lam):
        return cauchy_A + cauchy_B / (lam * lam)

    def build():
        ph = phi.get_value()
        verts = [(C[0] + R * np.cos(np.pi / 2 + 2 * np.pi / 3 * k + ph),
                  C[1] + R * np.sin(np.pi / 2 + 2 * np.pi / 3 * k + ph)) for k in range(3)]
        g = VGroup()
        g.add(Polygon(*[[v[0], v[1], 0] for v in verts], color=glass, stroke_width=FRAME_W,
                      fill_color="#cfe0f5", fill_opacity=0.06))
        d = np.array([np.cos(beam_angle), np.sin(beam_angle)])     # tiltable incoming beam
        O = np.array([C[0], C[1] + beam_y]) - abs(in_x) * d        # back-project entry along d
        # incoming white beam to the entry point (entry always exists through the centroid)
        edges = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
        ehits = [(t, i) for i, (A, B) in enumerate(edges) for t in [_ray_seg(O, d, A, B)] if t]
        if ehits:
            t1, _ = min(ehits); P1m = O + t1 * d
            g.add(Line([O[0], O[1], 0], [P1m[0], P1m[1], 0], color=white, stroke_width=6))
        internal, fan = VGroup(), VGroup()
        for lam in np.linspace(lam_hi, lam_lo, n_rays):
            res = _prism_exit(verts, O, d, n_of(lam), max_bounce=1)   # direct refraction only
            if res is None:                                          # TIR -> this colour can't exit
                continue
            pts, P2, d2 = res; col = visible_color(lam)
            for a, b in zip(pts[:-1], pts[1:]):
                internal.add(Line([a[0], a[1], 0], [b[0], b[1], 0], color=col,
                                  stroke_width=2).set_stroke(opacity=0.45))
            fan.add(Line([P2[0], P2[1], 0], [P2[0] + d2[0] * out_len, P2[1] + d2[1] * out_len, 0],
                         color=col, stroke_width=4))
        g.add(internal, fan)
        if bands:
            for lam, col in [(950, HOT), (340, "#9b6bff")]:
                res = _prism_exit(verts, O, d, n_of(lam), max_bounce=1)
                if res is None:
                    continue
                _, P2, d2 = res
                g.add(DashedLine([P2[0], P2[1], 0], [P2[0] + d2[0] * out_len, P2[1] + d2[1] * out_len, 0],
                                 color=col, stroke_width=3, dash_length=0.13).set_stroke(opacity=0.7))
        return g
    return build


def optical_bench(*, y=0.0, x0=-5.6, x1=5.6, glyph=0.6, labels=True, fs=22,
                  stages=("Source", "Lens", "Sample", "Grating", "Detector")):
    """Horizontal instrument schematic: evenly spaced labelled stations joined by a
    beam line. A generic header for any method scene. Returns a VGroup with
    .stations (dict name -> center point) and .beam (the connecting line)."""
    n = len(stages)
    xs = np.linspace(x0, x1, n)
    g = VGroup()
    beam = Line([x0 - 0.2, y, 0], [x1 + 0.2, y, 0], color=INK, stroke_width=3).set_stroke(opacity=0.35)
    g.add(beam)
    stations = {}
    for name, x in zip(stages, xs):
        c = np.array([x, y, 0.0])
        key = name.lower()
        if key == "source":
            m = VGroup(Circle(radius=glyph * 0.42, color=STICK, fill_color=STICK, fill_opacity=0.9, stroke_width=2),
                       *[Line(c + 0.001, c, color=STICK, stroke_width=2) for _ in range(0)]).move_to(c)
        elif key == "lens":
            m = Ellipse(width=glyph * 0.42, height=glyph * 1.25, color=CURVE, stroke_width=4).move_to(c)
        elif key == "sample":
            m = Rectangle(width=glyph * 0.85, height=glyph * 1.25, color=INK, stroke_width=4,
                          fill_color=CURVE, fill_opacity=0.10).move_to(c)
        elif key in ("grating", "prism", "disperser"):
            m = Triangle(color=CURVE, stroke_width=4, fill_color=CURVE, fill_opacity=0.08
                         ).scale(glyph * 0.78).move_to(c)
        elif key == "detector":
            body = Rectangle(width=glyph * 0.9, height=glyph * 1.3, color=INK, stroke_width=4,
                             fill_color="#1b2433", fill_opacity=1.0)
            sensor = VGroup(*[Line([-glyph * 0.3 + j * glyph * 0.2, -glyph * 0.45, 0],
                                   [-glyph * 0.3 + j * glyph * 0.2, glyph * 0.45, 0],
                                   color=MUTED, stroke_width=2) for j in range(4)])
            m = VGroup(body, sensor).move_to(c)            # build at origin, THEN place
        else:
            m = Dot(c, color=INK)
        g.add(m)
        stations[name] = c
        if labels:
            g.add(Text(name, font_size=fs, color=MUTED).next_to(c, DOWN, buff=glyph * 0.9))
    g.stations, g.beam = stations, beam
    return g


def michelson(center=ORIGIN, *, arm=1.7, mirror=0.0, beam_color=STICK, part_color=GRID,
              labels=True, fs=20):
    """Michelson interferometer schematic (the heart of FT-IR). `mirror` (~ -1..1)
    slides the moving mirror along its arm — the scanned path difference. Static
    build; wrap in always_redraw with a tracker for motion. Returns a VGroup."""
    c = np.array([center[0], center[1], 0.0])
    BS = c
    src = c + np.array([-arm, 0.0, 0.0])
    fixed = c + np.array([0.0, arm, 0.0])
    mov = c + np.array([arm + mirror * 0.45, 0.0, 0.0])
    det = c + np.array([0.0, -arm, 0.0])
    g = VGroup()
    # beams
    g.add(Line(src, BS, color=beam_color, stroke_width=5),
          Line(BS, fixed, color=beam_color, stroke_width=5).set_stroke(opacity=0.85),
          Line(BS, mov, color=beam_color, stroke_width=5).set_stroke(opacity=0.85),
          Line(BS, det, color=beam_color, stroke_width=5))
    # beamsplitter (45 deg) + mirrors (perpendicular bars)
    g.add(Line(BS + np.array([-0.32, -0.32, 0]), BS + np.array([0.32, 0.32, 0]),
               color=INK, stroke_width=4).set_stroke(opacity=0.8))
    g.add(Line(fixed + np.array([-0.45, 0, 0]), fixed + np.array([0.45, 0, 0]), color=part_color, stroke_width=7))
    g.add(Line(mov + np.array([0, -0.45, 0]), mov + np.array([0, 0.45, 0]), color=HOT, stroke_width=7))
    g.add(Arrow(mov + np.array([0.18, 0.62, 0]), mov + np.array([0.18 + mirror * 0.0 + 0.5, 0.62, 0]),
                color=HOT, buff=0, stroke_width=4, tip_length=0.14).set_opacity(0.0))
    # source + detector glyphs
    g.add(Circle(radius=0.22, color=STICK, fill_color=STICK, fill_opacity=0.9, stroke_width=2).move_to(src))
    g.add(Rectangle(width=0.5, height=0.7, color=GRID, stroke_width=4, fill_color="#11161d",
                    fill_opacity=0.9).move_to(det))
    if labels:
        g.add(Text("Source", font_size=fs, color=MUTED).next_to(src, LEFT, buff=0.34),
              Text("Fixed", font_size=fs, color=MUTED).next_to(fixed, UP, buff=0.16),
              Text("Moving mirror", font_size=fs, color=HOT).next_to(mov, RIGHT, buff=0.20),
              Text("Detector", font_size=fs, color=MUTED).next_to(det, DOWN, buff=0.5))
    g.points_of_interest = dict(BS=BS, src=src, fixed=fixed, mov=mov, det=det)
    return g


def interferogram_value(d, lines, span=1.0, cycles=9.0):
    """Interferogram amplitude I(delta) at a single path difference (peak 1.0 at the
    centre burst delta=0). Used for a moving cursor that tracks the mirror position."""
    maxnu = max(nu for nu, _ in lines)
    tot = sum(inten for _, inten in lines) or 1.0
    return float(sum(inten * np.cos(2 * np.pi * (nu / maxnu) * cycles * (d / span))
                     for nu, inten in lines) / tot)


def interferogram(P, lines, *, span=1.0, reveal=1.0, dmax=None, n=420, color=CURVE,
                  sw=4, cycles=9.0):
    """Interferogram I(delta) = sum inten*cos(2*pi*nu*delta) for a (nu cm^-1, inten)
    line list, drawn through mapper P(delta, I). delta runs [-span, span] (arbitrary
    path-difference units). Reveal either SYMMETRIC (|delta| <= reveal*span) or
    DIRECTIONAL left->right (delta <= dmax) — the latter ties the trace to a moving
    mirror scanning from -span to +span. The centre-burst at delta=0 is every cosine
    in phase (its Fourier transform is the spectrum)."""
    d = np.linspace(-span, span, n)
    I = np.array([interferogram_value(di, lines, span, cycles) for di in d])
    if dmax is not None:
        keep = d <= dmax + 1e-9
    else:
        keep = np.abs(d) <= reveal * span + 1e-9
    pts = [P(di, Ii) for di, Ii, k in zip(d, I, keep) if k]
    m = VMobject(color=color, stroke_width=sw, joint_type=LineJointType.ROUND)
    if len(pts) >= 2:
        m.set_points_as_corners(pts)
    return m


# ---- Theme B: why peaks exist ----------------------------------------------

def wave_packet(start, end, *, cycles=6.0, amplitude=0.16, color=INK, n=140,
                envelope=True, sw=4):
    """A transverse sinusoid (optionally Gaussian-enveloped wave packet) from start
    to end — the building block for beams, emission, and scattering."""
    start = np.array(start, float); end = np.array(end, float)
    axis = end - start; L = np.linalg.norm(axis) + 1e-9
    u = axis / L; nrm = np.array([-u[1], u[0], 0.0])
    ts = np.linspace(0.0, 1.0, n)
    env = np.exp(-((ts - 0.5) / 0.22) ** 2) if envelope else np.ones_like(ts)
    pts = [start + u * (t * L) + nrm * (amplitude * e * np.sin(2 * np.pi * cycles * t))
           for t, e in zip(ts, env)]
    m = VMobject(color=color, stroke_width=sw, joint_type=LineJointType.ROUND)
    m.set_points_as_corners(pts)
    return m


def traveling_photon(start, end, prog, *, span=0.9, cycles=5.0, amplitude=0.16,
                     color=INK, n=70, sw=4):
    """Factory: a wave packet of length `span` (scene units) whose CENTRE travels
    start -> end as `prog` (tracker) goes 0..1. Returns an always_redraw builder."""
    start = np.array(start, float); end = np.array(end, float)
    axis = end - start; L = np.linalg.norm(axis) + 1e-9
    u = axis / L; nrm = np.array([-u[1], u[0], 0.0])

    def build():
        cs = prog.get_value() * L
        s0 = max(0.0, cs - span / 2); s1 = min(L, cs + span / 2)
        if s1 - s0 < 1e-3:
            return VMobject()
        ss = np.linspace(s0, s1, n)
        env = np.exp(-((ss - cs) / (span * 0.30)) ** 2)
        pts = [start + u * s + nrm * (amplitude * e * np.sin(2 * np.pi * cycles * (s - s0) / span))
               for s, e in zip(ss, env)]
        m = VMobject(color=color, stroke_width=sw, joint_type=LineJointType.ROUND)
        m.set_points_as_corners(pts)
        return m
    return build


def energy_levels(energies, *, x0=-1.4, x1=1.4, y0=-1.6, height=3.2, color=INK,
                  labels=None, dashed=(), sw=5, label_color=MUTED, fs=24):
    """Horizontal energy levels. `energies` are fractions in [0,1] mapped to
    y0..y0+height. `dashed` = indices drawn dashed (e.g. a Raman virtual state).
    Returns a VGroup with .ys (level y's), .xmid, .x0, .x1, .height for arrows."""
    g = VGroup(); ys = []
    for i, e in enumerate(energies):
        y = y0 + float(e) * height; ys.append(y)
        if i in dashed:
            ln = DashedLine([x0, y, 0], [x1, y, 0], color=color, stroke_width=sw,
                            dash_length=0.12).set_stroke(opacity=0.6)
        else:
            ln = Line([x0, y, 0], [x1, y, 0], color=color, stroke_width=sw)
        g.add(ln)
        if labels:
            g.add(Text(str(labels[i]), font_size=fs, color=label_color).next_to([x1, y, 0], RIGHT, buff=0.18))
    g.ys, g.xmid, g.x0, g.x1, g.height = ys, (x0 + x1) / 2, x0, x1, height
    return g


def transition_arrow(x, y_from, y_to, *, color=None, label=None, span=3.2, sw=7,
                     fs=24, tip=0.20, side=RIGHT):
    """Vertical transition arrow between two level y's (up = absorption, down =
    emission). Colour auto from |dE|/span if None (larger gap -> bluer). Returns a
    VGroup(arrow[, label])."""
    frac = min(1.0, abs(y_to - y_from) / span)
    col = color or energy_to_color(frac)
    ar = Arrow([x, y_from, 0], [x, y_to, 0], color=col, buff=0, stroke_width=sw,
               max_tip_length_to_length_ratio=0.4, tip_length=tip)
    g = VGroup(ar)
    if label:
        g.add(Text(label, font_size=fs, color=col).next_to(ar, side, buff=0.14))
    g.arrow = ar
    return g


def spring(p1, p2, *, coils=6, width=0.16, color=MUTED, sw=4, lead=0.16):
    """Zigzag spring / bond between two points (transverse triangle wave)."""
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    axis = p2 - p1; L = np.linalg.norm(axis) + 1e-9
    u = axis / L; nrm = np.array([-u[1], u[0], 0.0])
    seg = max(2, coils * 2)
    pts = [p1]
    for i in range(1, seg):
        s = lead + (L - 2 * lead) * i / seg
        off = width * (1 if i % 2 else -1)
        pts.append(p1 + u * s + nrm * off)
    pts.append(p2)
    m = VMobject(color=color, stroke_width=sw, joint_type=LineJointType.ROUND)
    m.set_points_as_corners(pts)
    return m


def vibration_mode(center, phase, *, mode="stretch", bond=1.3, amp=0.18, axis=RIGHT,
                   atom_r=0.32, a_color=CURVE, b_color=HOT, spring_color=MUTED, coils=6):
    """Factory: an oscillating ball-and-spring normal mode driven by `phase` (tracker;
    one full period per unit). Connects a molecular motion to its band. Modes:
      'stretch' — A~B antiphase along `axis` (the bond stretches/compresses)
      'asym'    — linear A-B-A asymmetric stretch
      'bend'    — A-B-A scissor (apex angle oscillates)
    Returns an always_redraw builder."""
    u = np.array(axis, float)[:3]; u = u / (np.linalg.norm(u) + 1e-9)
    c = np.array([center[0], center[1], 0.0])

    def atom(p, col):
        return Circle(radius=atom_r, color=col, fill_color=col, fill_opacity=0.92,
                      stroke_width=2).move_to(p)

    def build():
        ph = 2 * np.pi * phase.get_value()
        d = amp * np.sin(ph)
        g = VGroup()
        if mode == "stretch":
            A = c - u * (bond / 2) - u * d
            B = c + u * (bond / 2) + u * d
            g.add(spring(A, B, coils=coils, color=spring_color), atom(A, a_color), atom(B, b_color))
        elif mode == "asym":
            B = c
            A1 = c - u * bond - u * d
            A2 = c + u * bond - u * d                  # both arms shift together -> asym
            g.add(spring(A1, B, coils=coils, color=spring_color),
                  spring(B, A2, coils=coils, color=spring_color),
                  atom(A1, a_color), atom(B, b_color), atom(A2, a_color))
        else:  # bend (scissor); arms point downward at +/- theta from vertical
            th = 50 * DEGREES + (amp * 1.6) * np.sin(ph)
            B = c
            AL = B + bond * np.array([-np.sin(th), -np.cos(th), 0.0])
            AR = B + bond * np.array([np.sin(th), -np.cos(th), 0.0])
            g.add(spring(B, AL, coils=coils, color=spring_color),
                  spring(B, AR, coils=coils, color=spring_color),
                  atom(AL, a_color), atom(B, b_color), atom(AR, a_color))
        return g
    return build


def _fold(x, lo, hi):
    """Reflect x into [lo, hi] (ideal elastic bounce as a triangle wave) — lets a
    drifting body bounce off walls as a pure function of time."""
    span = hi - lo
    if span <= 1e-9:
        return (lo + hi) / 2.0
    y = (x - lo) % (2 * span)
    if y > span:
        y = 2 * span - y
    return lo + y


def vibrating_molecule(center, t, *, mode="stretch", ax=1.3, ay=0.7, cdy=-0.35,
                       orbit_tilt=0.0, orbit_w=0.7, orbit_phase=0.0, ang_vel=0.12,
                       vib_freq=1.6, vib_amp=0.16, phase0=0.0, bond=0.62, atom_r=0.20,
                       bar_w=0.14, a_color=CURVE, b_color=HOT, bond_color=MOL_DIM):
    """Factory: a little molecule that ORBITS on an ELLIPSE (semi-axes ax/ay, centred at
    (cx, cy+cdy)) — its gravitational path — speeding up gently along the way, while
    slowly tumbling (ang_vel) and exhibiting a normal mode. All a pure function of the
    time tracker `t`. Draw the matching ellipse separately (Ellipse width=2*ax,
    height=2*ay at the same centre). Atoms are SOLID (opaque, over the bonds); each bond
    is a RECTANGLE bar whose length stretches. Modes: 'stretch' (diatomic), 'sym'/'asym'
    stretch, 'bend' (scissor), 'rock' (rigid libration), 'wag' (central atom bobs)."""
    cx, cy = center[0], center[1]

    def bar(p1, p2):
        p1 = np.array([p1[0], p1[1], 0.0]); p2 = np.array([p2[0], p2[1], 0.0])
        dd = p2 - p1; L = float(np.linalg.norm(dd)) + 1e-9
        return Rectangle(width=L, height=bar_w, color=bond_color, fill_color=bond_color,
                         fill_opacity=1.0, stroke_width=0).move_to((p1 + p2) / 2).rotate(
                         np.arctan2(dd[1], dd[0]))

    def atom(p, col):
        return Circle(radius=atom_r, color=col, fill_color=col, fill_opacity=1.0,
                      stroke_width=0).move_to([p[0], p[1], 0.0])

    def build():
        tt = t.get_value()
        theta = orbit_w * tt + orbit_phase
        ang = theta + 0.4 * np.sin(theta)              # gentle gravitational speed-up
        ex, ey = ax * np.cos(ang), ay * np.sin(ang)    # point on the axis-aligned ellipse
        ct, st = np.cos(orbit_tilt), np.sin(orbit_tilt)
        px = cx + ex * ct - ey * st                    # ellipse tilted by orbit_tilt
        py = (cy + cdy) + ex * st + ey * ct
        th = ang_vel * tt + 0.5 * np.sin(0.3 * tt + phase0)   # slow, meandering tumble (never fast)
        cth, sth = np.cos(th), np.sin(th)
        ph = 2 * np.pi * vib_freq * tt + phase0; d = vib_amp * np.sin(ph)
        b = bond
        if mode == "stretch":
            locs = [(-(b / 2 + d), 0, a_color), (b / 2 + d, 0, b_color)]; bonds = [(0, 1)]
        elif mode == "sym":
            locs = [(-(b + d), 0, a_color), (0, 0, b_color), (b + d, 0, a_color)]; bonds = [(0, 1), (1, 2)]
        elif mode == "asym":
            locs = [(-b - d, 0, a_color), (0, 0, b_color), (b - d, 0, a_color)]; bonds = [(0, 1), (1, 2)]
        elif mode == "bend":
            a0 = 55 * DEGREES + (vib_amp * 2.2) * np.sin(ph)
            locs = [(-b * np.sin(a0), -b * np.cos(a0), a_color), (0, 0, b_color),
                    (b * np.sin(a0), -b * np.cos(a0), a_color)]; bonds = [(0, 1), (1, 2)]
        elif mode == "rock":
            a0 = 55 * DEGREES; rk = (vib_amp * 1.8) * np.sin(ph); cr, sr = np.cos(rk), np.sin(rk)
            base = [(-b * np.sin(a0), -b * np.cos(a0)), (0, 0), (b * np.sin(a0), -b * np.cos(a0))]
            cols = [a_color, b_color, a_color]
            locs = [(x * cr - y * sr, x * sr + y * cr, cols[k]) for k, (x, y) in enumerate(base)]
            bonds = [(0, 1), (1, 2)]
        else:  # wag — central atom bobs perpendicular to the molecular axis
            a0 = 55 * DEGREES
            locs = [(-b * np.sin(a0), -b * np.cos(a0), a_color), (0, d, b_color),
                    (b * np.sin(a0), -b * np.cos(a0), a_color)]; bonds = [(0, 1), (1, 2)]
        world = [(lx * cth - ly * sth + px, lx * sth + ly * cth + py, col) for (lx, ly, col) in locs]
        g = VGroup()
        for i, j in bonds:
            g.add(bar(world[i][:2], world[j][:2]))
        for wx, wy, col in world:
            g.add(atom((wx, wy), col))
        return g
    return build


# =============================================================================
#  DOMAIN: STFT / SPECTROGRAM
# =============================================================================

def make_signal(fs=200.0, dur=4.0):
    """Illustrative time-varying signal: a rising chirp plus two gated tones.

    Returns (t, signal). Repoint this at real time-resolved data (e.g. UV-Vis
    dissolution kinetics or a DSC temperature ramp) to make it lab-specific.
    """
    t = np.arange(0, dur, 1.0 / fs)
    sig = np.zeros_like(t)
    f0, f1 = 6.0, 42.0                          # chirp sweep (Hz)
    k = (f1 - f0) / dur
    sig += 0.85 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))
    m1 = t < dur * 0.55                          # tone 1: on early
    sig[m1] += 0.65 * np.sin(2 * np.pi * 18.0 * t[m1])
    m2 = t > dur * 0.45                          # tone 2: on late
    sig[m2] += 0.55 * np.sin(2 * np.pi * 72.0 * t[m2])
    return t, sig


def _colormap(v):
    """Map v in [0,1] -> RGB (0-255), an inferno-like ramp on the dark theme."""
    anchors = [
        (0.00, (10, 13, 18)),
        (0.18, (38, 26, 72)),
        (0.42, (120, 32, 110)),
        (0.66, (224, 88, 64)),
        (0.85, (248, 168, 52)),
        (1.00, (252, 246, 196)),
    ]
    pos = np.array([a[0] for a in anchors])
    cols = np.array([a[1] for a in anchors], dtype=float)
    out = np.empty(v.shape + (3,), dtype=float)
    for ch in range(3):
        out[..., ch] = np.interp(v, pos, cols[:, ch])
    return out.astype(np.uint8)


def _hex_from_mag(m):
    """Scalar magnitude in [0,1] -> '#rrggbb' via the inferno-like colormap."""
    r, g, b = _colormap(np.array([float(m)]))[0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _stft(fs, dur, nper, noverlap, fmax_show, gamma=0.5):
    """Shared STFT: returns (t, sig, freqs, times, mag) with mag normalised to
    [0,1] and gamma-lifted for visibility. mag is indexed [freq, time]."""
    from scipy.signal import stft
    t, sig = make_signal(fs, dur)
    f, tt, Z = stft(sig, fs=fs, nperseg=nper, noverlap=noverlap, window="hann")
    mag = np.abs(Z)
    fmask = f <= fmax_show
    f, mag = f[fmask], mag[fmask, :]
    mag = mag / (mag.max() + 1e-12)
    mag = mag ** gamma
    return t, sig, f, tt, mag


def heatmap_image(mag, w, h, center):
    """Build the heatmap ImageMobject from a [freq, time] magnitude array: flip freq
    so HIGH frequency is on top, colorize via `_colormap`, nearest resampling, then
    stretch to EXACT (w, h) at `center`. (The .width/.height setters preserve aspect,
    which would leave the height wrong and let the heatmap bleed past its frame.)"""
    img_rgb = _colormap(mag[::-1, :])
    heat = ImageMobject(img_rgb)
    heat.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    heat.stretch_to_fit_height(h)
    heat.stretch_to_fit_width(w)
    heat.move_to(center)
    return heat


class HeatmapGeom:
    """Geometry of a heatmap panel: center/size, derived pixel bounds, and the
    freq->y / time->x mappers. Bundles the 8 locals the playhead/curtain/spectrum
    factories need into one object."""
    def __init__(self, center, w, h, fmax, dur):
        self.center = np.array([center[0], center[1], 0.0]) if len(center) == 2 \
            else np.array(center, dtype=float)
        self.w, self.h = w, h
        self.fmax, self.dur = fmax, dur
        cx, cy = center[0], center[1]
        self.hx0 = cx - w / 2.0
        self.hx1 = cx + w / 2.0
        self.hy0 = cy - h / 2.0
        self.hy1 = cy + h / 2.0

    def fy(self, freq):                # frequency -> screen y
        return self.hy0 + (freq / self.fmax) * self.h

    def tx(self, time):                # time -> screen x
        return self.hx0 + (time / self.dur) * self.w


def reveal_curtain(rev, geom):
    """Factory: a BG rectangle that hides the not-yet-revealed (right) portion of the
    heatmap. `rev` in [0,1] is how far the reveal has advanced. Wrap in always_redraw."""
    def curtain():
        x = geom.hx0 + rev.get_value() * geom.w
        w = max(geom.hx1 - x, 1e-3)
        r = Rectangle(width=w, height=geom.h + 0.02, stroke_width=0,
                      fill_color=BG, fill_opacity=1.0)
        r.move_to([(x + geom.hx1) / 2, geom.center[1], 0])
        return r
    return curtain


def playhead_line(ph, geom, top_y):
    """Factory: the red playhead line at fractional position `ph` (0..1), spanning
    from the heatmap bottom up to `top_y`. Wrap in always_redraw."""
    def playhead():
        x = geom.hx0 + ph.get_value() * geom.w
        return Line([x, geom.hy0, 0], [x, top_y, 0], color=HOT, stroke_width=4)
    return playhead


def instant_spectrum(ph, geom, mag, f, tt, sp_x0, sp_w, n_spec=N_SPEC):
    """Factory: the live instantaneous-spectrum side panel — the STFT column under
    the playhead, drawn as a shape-preserving (PCHIP) cubic so peaks are rounded.
    Wrap in always_redraw. `f` are the frequency bins, `tt` the STFT time columns."""
    from scipy.interpolate import PchipInterpolator
    yc = np.array([geom.fy(fr) for fr in f])

    def builder():
        idx = int(round(ph.get_value() * (len(tt) - 1)))
        idx = max(0, min(idx, len(tt) - 1))
        col = mag[:, idx]
        cmax = col.max() + 1e-9
        xc = sp_x0 + (col / cmax) * sp_w
        yy = np.linspace(yc[0], yc[-1], n_spec)
        xx = PchipInterpolator(yc, xc)(yy)
        m = VMobject(color=CURVE, stroke_width=3, joint_type=LineJointType.ROUND)
        m.set_points_as_corners([[x, y, 0] for x, y in zip(xx, yy)])
        return m
    return builder


# What `from spectro_lib import *` exposes. Underscore names are listed explicitly
# so the scene file (and Spectrogram3D) can use them via the star import.
__all__ = [
    # palette
    "BG", "INK", "MUTED", "GRID", "STICK", "CURVE", "LINEFILL", "LORENTZ", "HOT",
    "TEMP", "MOL_DIM", "MOL_GLOW", "O_COLOR", "H_COLOR", "N_COLOR",
    # line weights
    "AXIS_W", "TIP_LEN", "TIP_W", "TICK_W", "FRAME_W", "AXIS_W3D",
    # quality knobs
    "DRAFT", "N_ENV", "N_COMP", "N_SPEC",
    # canonical-format helpers
    "ease_out_back",
    "axis_line", "tick_marks", "frame_box", "readout_pill", "reveal_then_collapse_axis",
    # line list / broadening
    "LINE_LIST", "ASPIRIN", "ASPIRIN_UV", "lorentzian", "broaden", "gaussian",
    "build_aspirin", "build_acetaminophen", "build_caffeine", "build_diphenhydramine",
    "build_dopamine", "build_adrenaline", "build_serotonin", "build_gaba", "build_glutamate",
    "build_glutamine",
    "build_glycine", "build_alanine", "build_valine", "build_leucine", "build_isoleucine",
    "build_serine", "build_threonine", "build_cysteine", "build_methionine", "build_asparagine",
    "build_aspartate", "build_lysine", "build_arginine", "build_phenylalanine", "build_tyrosine",
    "build_histidine", "build_proline", "build_tryptophan",
    "build_glucose", "build_sucrose", "build_citric_acid", "build_vanillin", "build_menthol",
    "build_ibuprofen", "build_vitamin_c", "build_cholesterol", "build_testosterone",
    "build_melatonin", "build_nicotine", "build_cbd", "build_morphine", "build_penicillin",
    "build_capsaicin",
    "molecule_tag", "styled_molecule", "STRUCTURE_PRESET", "billboard_rotate", "molecule_glow_updater",
    # spectrography primitives — Theme A (origin) + Theme B (why peaks exist)
    "visible_color", "energy_to_color", "prism", "disperse_spectrum", "refracting_prism", "optical_bench",
    "michelson", "interferogram", "interferogram_value", "wave_packet", "traveling_photon",
    "energy_levels", "transition_arrow", "spring", "vibration_mode", "vibrating_molecule",
    # data-shading palette
    "SHADE", "SHADE_ORDER", "horizontal_gradient_fill", "gradient_swatch",
    # STFT / spectrogram
    "make_signal", "_stft", "_colormap", "_hex_from_mag", "heatmap_image",
    "HeatmapGeom", "reveal_curtain", "playhead_line", "instant_spectrum",
]
