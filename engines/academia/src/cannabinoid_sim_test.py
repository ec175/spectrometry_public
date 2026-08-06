"""cannabinoid_sim_test.py — simulate FTIR/Raman/UV/1H-NMR for ~10 cannabinoids,
then a second set overlaying each cannabinoid with its carboxylic-ACID variant.

Same pipeline-test spirit as the amino-acid tests (empirical group-frequency /
shift line lists, not DFT). Cannabinoids share a resorcinol + alkyl-terpenoid
scaffold, so the data is BASE (shared) + per-structural-type modifier. The acid
variant (THCA, CBDA, ...) is the neutral's line list PLUS a carboxyl modifier:
its hallmark is a chelated (H-bonded) C=O ~1650 cm-1 and a far-downfield ~12 ppm
COOH/phenol proton -- features the neutral simply lacks. So the overlay shows a
genuine CHEMICAL difference (new peaks), unlike the crystal/amorphous broadening.

Lab conditions: FTIR/Raman powder, UV in ethanol, 1H NMR in CDCl3 (cannabinoids
are lipophilic -> CDCl3, not D2O). Rows are ordered by DESCENDING molar mass.

Set 1 -> cannabinoids_<TECH>.pdf            (neutral only)
Set 2 -> cannabinoids_<TECH>_neutral_vs_acid.pdf

Run:  .\.venv\Scripts\python.exe cannabinoid_sim_test.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

import spectra as sp
import raman as raman_mod
import uv as uv_mod
import nmr as nmr_mod
import stacks
import molecules
try:
    from chem_data import CANNABINOID_SMILES
except Exception:
    CANNABINOID_SMILES = {}

FIG = Path(__file__).resolve().parent / "figures"
OUT = FIG / "cannabinoids"
OUT2 = FIG / "cannabinoids_neutral_vs_acid"
OUT.mkdir(parents=True, exist_ok=True)
OUT2.mkdir(parents=True, exist_ok=True)

NEUTRAL_COLOR, ACID_COLOR = "black", "#1f77b4"

# (code, display, neutral MW, structural type) — DESCENDING molar mass.
# Acid variant MW = neutral + 44.01 (adds CO2). Isomers (314.46) kept in a
# consistent secondary order.
CANN = [
    ("CBG",  "CBG (cannabigerol)",        316.5, "alkene"),
    ("THC",  "$\\Delta^{9}$-THC",          314.5, "pyran"),
    ("D8THC", "$\\Delta^{8}$-THC",         314.5, "pyran"),
    ("CBD",  "CBD (cannabidiol)",          314.5, "alkene"),
    ("CBC",  "CBC (cannabichromene)",      314.5, "chromene"),
    ("CBL",  "CBL (cannabicyclol)",        314.5, "cyclol"),
    ("CBN",  "CBN (cannabinol)",           310.4, "aromatic"),
    ("CBGV", "CBGV (cannabigerovarin)",    288.4, "alkene"),
    ("CBDV", "CBDV (cannabidivarin)",      286.4, "alkene"),
    ("THCV", "THCV (tetrahydrocannabivarin)", 286.4, "pyran"),
]
# sort by ring system: most rings on top, fewest on bottom (3-ring → 2 → 1)
CANN.sort(key=lambda r: (-molecules.ring_count(CANNABINOID_SMILES.get(r[0], {}).get("neutral", "")),
                         -r[2]))
TYPE = {c: t for c, _, _, t in CANN}
ITEMS = [(c, disp) for c, disp, mw, _ in CANN]      # labels: name only (no MW)
ITEMS_ACID = ITEMS

# ── FTIR (cm-1, rel intensity) ───────────────────────────────────────────────
FTIR_BASE = [(3380, .50), (2955, .70), (2925, .85), (2870, .55), (1620, .55),
             (1580, .50), (1460, .50), (1420, .40), (1370, .35), (1230, .50),
             (1180, .40), (1140, .35), (1040, .35), (820, .35), (780, .30)]
FTIR_TYPE = {
    "alkene":   [(3080, .25), (1640, .45), (885, .45)],
    "pyran":    [(1150, .50), (1058, .45), (1188, .40), (1378, .40), (940, .30)],
    "chromene": [(1640, .40), (1118, .45), (1045, .40), (950, .35), (810, .35)],
    "cyclol":   [(2960, .40), (1300, .30), (1095, .40), (1035, .35)],
    "aromatic": [(3050, .35), (1625, .60), (1572, .55), (1500, .45), (1380, .40),
                 (885, .40), (808, .40), (745, .40)],
}
ACID_FTIR = [(2560, .30), (1650, .95), (1610, .40), (1440, .35), (1300, .50), (1260, .45)]

# ── Raman (cm-1) ─────────────────────────────────────────────────────────────
RAMAN_BASE = [(2930, .80), (2870, .55), (1620, .60), (1580, .45), (1450, .45),
              (1300, .40), (1160, .35), (1040, .40), (820, .35)]
RAMAN_TYPE = {
    "alkene":   [(1668, .65), (1640, .45)],
    "pyran":    [(1150, .40), (800, .30)],
    "chromene": [(1640, .45), (1120, .35)],
    "cyclol":   [(1040, .30), (1000, .30)],
    "aromatic": [(1605, .85), (1578, .60), (1378, .50), (1030, .40), (760, .40)],
}
ACID_RAMAN = [(1650, .40), (1420, .45)]

# ── UV-Vis (nm, oscillator strength) · ethanol ──────────────────────────────
UV_BASE = [(206, 1.0), (230, .18), (274, .28), (282, .20)]
UV_TYPE = {
    "aromatic": [(220, .55), (286, .40), (310, .28)],     # CBN: extended conjugation, red-shift
    "chromene": [(258, .12)],
    "alkene": [], "pyran": [], "cyclol": [],
}
ACID_UV = [(268, .35), (304, .40)]                         # conjugated COOH -> ~305 nm band
UV_GRID = np.linspace(195, 360, 3000)

# ── 1H NMR (ppm, # protons) · CDCl3 ──────────────────────────────────────────
NMR_BASE = [(6.27, 1), (6.13, 1), (4.85, 1), (2.43, 2), (1.96, 2), (1.66, 3),
            (1.56, 2), (1.30, 4), (0.88, 3)]
NMR_TYPE = {
    "alkene":   [(5.57, 1), (4.66, 1), (4.56, 1), (1.79, 3)],
    "pyran":    [(6.31, 1), (3.20, 1), (1.41, 3), (1.10, 3)],
    "chromene": [(6.62, 1), (5.48, 1), (1.38, 6)],
    "cyclol":   [(2.55, 1), (1.32, 6), (1.18, 3)],
    "aromatic": [(8.16, 1), (7.10, 1), (6.96, 1), (2.40, 3), (1.59, 6)],
}
ACID_NMR = [(12.10, 1)]                                    # chelated COOH/phenol, very downfield
NMR_GRID = np.linspace(0, 14, 4500)


def _lines(base, typemap, ctype, extra=None):
    pts = list(base) + list(typemap.get(ctype, []))
    if extra:
        pts = pts + list(extra)
    return [p[0] for p in pts], [p[1] for p in pts]


def sim_ftir(code, acid=False):
    c, a = _lines(FTIR_BASE, FTIR_TYPE, TYPE[code], ACID_FTIR if acid else None)
    return sp.simulate(c, a, "FTIR", fwhm=9)


def sim_raman(code, acid=False):
    c, a = _lines(RAMAN_BASE, RAMAN_TYPE, TYPE[code], ACID_RAMAN if acid else None)
    return raman_mod.simulate(c, a, fwhm=8)


def sim_uv(code, acid=False):
    c, a = _lines(UV_BASE, UV_TYPE, TYPE[code], ACID_UV if acid else None)
    return uv_mod.simulate(c, a, grid=UV_GRID, fwhm_ev=0.30)


def sim_nmr(code, acid=False):
    c, a = _lines(NMR_BASE, NMR_TYPE, TYPE[code], ACID_NMR if acid else None)
    return nmr_mod.simulate(c, a, nucleus="1H", grid=NMR_GRID, fwhm=0.05)


SET1 = [
    (sim_ftir,  "Simulated FTIR · cannabinoids · powder", "Wavenumber (cm$^{-1}$)",
     (600, 3600), True, "cannabinoids_FTIR.pdf"),
    (sim_raman, "Simulated Raman · cannabinoids · powder", "Raman shift (cm$^{-1}$)",
     (300, 3200), False, "cannabinoids_RAMAN.pdf"),
    (sim_uv,    "Simulated UV–Vis · cannabinoids · ethanol", "Wavelength (nm)",
     (205, 345), False, "cannabinoids_UV.pdf"),
    (sim_nmr,   "Simulated $^{1}$H NMR · cannabinoids · CDCl$_3$", "$^{1}$H chemical shift (ppm)",
     (0, 13), True, "cannabinoids_NMR_1H.pdf"),
]


def _img(code, which, **kw):
    d = CANNABINOID_SMILES.get(code)
    return molecules.render_rgba(d[which], **kw) if d else None


def main():
    # neutral structures (default colors); acid structures (black skeleton, COOH haloed blue)
    neutral_imgs = [_img(c, "neutral") for c, _ in ITEMS]
    acid_imgs = [_img(c, "acid", bw_base=True, highlight_smarts=molecules.ACID_SMARTS)
                 for c, _ in ITEMS]
    # Set 1 — neutral cannabinoids only
    for sim, title, xlabel, xrange, desc, fname in SET1:
        stacks.stack_plot(ITEMS, lambda code, s=sim: s(code, acid=False),
                          title, xlabel, xrange, desc, OUT / fname,
                          offset=1.8, figsize=(14, 20), images=neutral_imgs,
                          image_zoom=0.12, image_dy=0.28, left_margin=0.26)
    # Set 2 — neutral (black) overlaid with acid variant (blue)
    for sim, title, xlabel, xrange, desc, fname in SET1:
        out = fname.replace(".pdf", "_neutral_vs_acid.pdf")
        stacks.stack_overlay(
            ITEMS_ACID, lambda code, s=sim: (s(code, acid=False), s(code, acid=True)),
            title + " · neutral vs acid", xlabel, xrange, desc, OUT2 / out,
            color_a=NEUTRAL_COLOR, color_b=ACID_COLOR,
            label_a="neutral (decarboxylated)", label_b="acid (–COOH)",
            offset=1.95, figsize=(14, 22), images=acid_imgs,
            image_zoom=0.12, image_dy=0.28, left_margin=0.26)
    print(f"8 cannabinoid PDFs in {OUT}")


if __name__ == "__main__":
    main()
