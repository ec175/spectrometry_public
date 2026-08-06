"""amino_acid_crystal_vs_amorphous_test.py — overlay a crystalline and an
amorphous simulated trace for each of the 20 canonical amino acids, per technique.

Builds on amino_acid_sim_test.py (reuses its FTIR/Raman/UV line lists) and models
the crystalline -> amorphous change the honest way: SAME line list, much larger
broadening. Structural disorder spreads every local environment over a
distribution, so sharp crystalline peaks merge into broad amorphous bands -- the
exact FTIR/ssNMR signature the lab uses to confirm an amorphous solid dispersion.

All four techniques are put in SOLID-STATE conditions here (crystal vs amorphous
is only meaningful for a solid -- in solution both forms dissolve identically):
  FTIR / Raman - powder
  UV           - solid diffuse reflectance
  NMR          - 13C CP-MAS ssNMR (not D2O; switched so the comparison exists)

Each row overlays:  black = crystalline (sharp),  red = amorphous (broadened).

Run:  .\.venv\Scripts\python.exe amino_acid_crystal_vs_amorphous_test.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import spectra as sp
import raman as raman_mod
import uv as uv_mod
import nmr as nmr_mod
import stacks
import amino_acid_sim_test as base          # reuse AA, BB_*, SIDE_*, _merge

OUT = base.OUT.parent / "amino_acids_xtal_vs_amorphous"
OUT.mkdir(parents=True, exist_ok=True)
AA = base.AA

CRYST_COLOR, AMOR_COLOR = "black", "#d62728"

# ── 13C CP-MAS shifts (ppm, ~carbon count) — solid/zwitterion, representative ─
NMR_13C = {
    "Gly": [(174, 1), (42, 1)],
    "Ala": [(176, 1), (51, 1), (17, 1)],
    "Val": [(175, 1), (61, 1), (30, 1), (18, 2)],
    "Leu": [(176, 1), (54, 1), (40, 1), (25, 1), (22, 2)],
    "Ile": [(175, 1), (60, 1), (37, 1), (25, 1), (15, 1), (12, 1)],
    "Pro": [(175, 1), (62, 1), (47, 1), (30, 1), (25, 1)],
    "Met": [(175, 1), (55, 1), (31, 1), (30, 1), (15, 1)],
    "Phe": [(175, 1), (56, 1), (37, 1), (137, 1), (130, 2), (129, 2), (128, 1)],
    "Tyr": [(175, 1), (56, 1), (37, 1), (156, 1), (131, 2), (128, 1), (116, 2)],
    "Trp": [(175, 1), (56, 1), (28, 1), (136, 1), (127, 1), (124, 1), (122, 1),
            (119, 1), (118, 1), (112, 1), (110, 1)],
    "His": [(174, 1), (55, 1), (28, 1), (136, 1), (130, 1), (117, 1)],
    "Ser": [(173, 1), (61, 1), (57, 1)],
    "Thr": [(174, 1), (67, 1), (60, 1), (20, 1)],
    "Cys": [(173, 1), (56, 1), (26, 1)],
    "Asn": [(175, 1), (174, 1), (51, 1), (36, 1)],
    "Gln": [(178, 1), (175, 1), (55, 1), (32, 1), (27, 1)],
    "Asp": [(178, 1), (175, 1), (53, 1), (37, 1)],
    "Glu": [(182, 1), (175, 1), (55, 1), (34, 1), (28, 1)],
    "Lys": [(175, 1), (55, 1), (40, 1), (31, 1), (27, 1), (22, 1)],
    "Arg": [(175, 1), (157, 1), (55, 1), (41, 1), (28, 1), (25, 1)],
}

UV_GRID = np.linspace(185, 320, 3000)


# ── per-technique crystalline/amorphous simulation pairs ────────────────────
def _pair_ftir(code):
    c, a = base._merge(base.BB_FTIR, base.SIDE_FTIR, code)
    return sp.simulate(c, a, "FTIR", fwhm=7), sp.simulate(c, a, "FTIR", fwhm=26)


def _pair_raman(code):
    c, a = base._merge(base.BB_RAMAN, base.SIDE_RAMAN, code)
    return raman_mod.simulate(c, a, fwhm=6), raman_mod.simulate(c, a, fwhm=20)


def _pair_uv(code):
    c, a = base._merge(base.BB_UV, base.SIDE_UV, code)
    return (uv_mod.simulate(c, a, grid=UV_GRID, fwhm_ev=0.35),
            uv_mod.simulate(c, a, grid=UV_GRID, fwhm_ev=0.60))


def _pair_nmr13c(code):
    c, a = zip(*NMR_13C[code])
    return (nmr_mod.simulate(c, a, nucleus="13C", fwhm=1.0),
            nmr_mod.simulate(c, a, nucleus="13C", fwhm=5.0))


CONFIGS = [
    ("Simulated FTIR · 20 amino acids · powder · crystalline vs amorphous",
     "Wavenumber (cm$^{-1}$)", _pair_ftir, (600, 3600), True, "amino_acids_FTIR_xtal_vs_amorphous.pdf"),
    ("Simulated Raman · 20 amino acids · powder · crystalline vs amorphous",
     "Raman shift (cm$^{-1}$)", _pair_raman, (300, 3150), False, "amino_acids_RAMAN_xtal_vs_amorphous.pdf"),
    ("Simulated UV–Vis · 20 amino acids · solid (diffuse reflectance) · crystalline vs amorphous",
     "Wavelength (nm)", _pair_uv, (190, 310), False, "amino_acids_UV_xtal_vs_amorphous.pdf"),
    ("Simulated $^{13}$C CP-MAS ssNMR · 20 amino acids · crystalline vs amorphous",
     "$^{13}$C chemical shift (ppm)", _pair_nmr13c, (5, 190), True, "amino_acids_NMR_13C_xtal_vs_amorphous.pdf"),
]


def main():
    imgs = base.amino_images()
    for title, xlabel, pair_fn, xrange, descending, fname in CONFIGS:
        stacks.stack_overlay(AA, pair_fn, title, xlabel, xrange, descending, OUT / fname,
                             color_a=CRYST_COLOR, color_b=AMOR_COLOR,
                             label_a="crystalline", label_b="amorphous",
                             images=imgs, image_zoom=0.085, image_dy=0.25,
                             left_margin=0.24, offset=1.9, figsize=(14, 38))
    print(f"4 crystalline-vs-amorphous PDFs in {OUT}")


if __name__ == "__main__":
    main()
