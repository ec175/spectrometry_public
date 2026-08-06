"""amino_acid_sim_test.py — simulate FTIR/Raman/UV/1H-NMR for the 20 canonical
amino acids and render one large stacked PDF per technique.

This is a PIPELINE TEST, not a DFT calculation: the line lists are representative
group-frequency / chemical-shift tables (empirical, already "observed" so no
freq_scale), pushed through the library's simulate() + broaden() + overlay infra
to confirm every technique's simulation path produces sane stacked figures.

Common lab conditions are baked into the data:
  FTIR  - zwitterionic powder      (COO-/NH3+ bands, broad)
  Raman - powder                   (ring-breathing, C-S/S-S, COO- sym)
  UV    - aqueous, pH 7            (far-UV backbone + aromatic near-UV bands)
  1H NMR- D2O                       (exchangeable NH3+/OH/COOH/NH not observed)

Run:  .\.venv\Scripts\python.exe amino_acid_sim_test.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import spectra as sp
import raman as raman_mod
import uv as uv_mod
import nmr as nmr_mod
import stacks
import molecules
try:
    from chem_data import AMINO_SMILES
except Exception:
    AMINO_SMILES = {}

OUT = Path(__file__).resolve().parent / "figures" / "amino_acids"
OUT.mkdir(parents=True, exist_ok=True)


def amino_images():
    """Structure thumbnail per amino acid (aligned with AA order), or None."""
    return [molecules.render_rgba(AMINO_SMILES.get(code)) if AMINO_SMILES.get(code) else None
            for code, _ in AA]
OUT.mkdir(exist_ok=True)

# Grouped by side-chain class so the stack tells a story (aromatics cluster, etc.)
AA = [
    ("Gly", "Glycine"), ("Ala", "Alanine"), ("Val", "Valine"), ("Leu", "Leucine"),
    ("Ile", "Isoleucine"), ("Pro", "Proline"), ("Met", "Methionine"),
    ("Phe", "Phenylalanine"), ("Tyr", "Tyrosine"), ("Trp", "Tryptophan"),
    ("His", "Histidine"), ("Ser", "Serine"), ("Thr", "Threonine"), ("Cys", "Cysteine"),
    ("Asn", "Asparagine"), ("Gln", "Glutamine"), ("Asp", "Aspartate"),
    ("Glu", "Glutamate"), ("Lys", "Lysine"), ("Arg", "Arginine"),
]
# order by ascending molar mass: smallest on top, largest on bottom (MW not shown)
AA.sort(key=lambda t: molecules.mol_weight(AMINO_SMILES.get(t[0], "")) or 9e9)

# ── FTIR: zwitterionic backbone + side-chain bands (cm-1, rel. intensity) ────
BB_FTIR = [(3080, .45), (2965, .55), (2925, .45), (2120, .15), (1605, 1.0),
           (1585, .55), (1505, .50), (1410, .80), (1330, .35), (1130, .30),
           (1040, .30), (870, .30)]
SIDE_FTIR = {
    "Gly": [], "Ala": [(1455, .35), (1375, .30)],
    "Val": [(1468, .35), (1390, .25), (1370, .25)], "Leu": [(1467, .35), (1385, .25), (1367, .25)],
    "Ile": [(1460, .35), (1380, .25)], "Pro": [(2985, .40), (1450, .35), (920, .25)],
    "Met": [(1440, .25), (700, .35), (660, .25)],
    "Phe": [(3030, .35), (1605, .45), (1498, .45), (750, .55), (700, .55)],
    "Tyr": [(3200, .40), (3050, .30), (1615, .40), (1515, .65), (1245, .55), (830, .55)],
    "Trp": [(3040, .35), (1620, .35), (1550, .45), (1455, .30), (1340, .40), (760, .55), (740, .40)],
    "His": [(3120, .35), (1590, .45), (1495, .30), (1090, .30), (625, .30)],
    "Ser": [(3300, .55), (1080, .45), (1030, .45)], "Thr": [(3350, .55), (1100, .40), (1040, .35)],
    "Cys": [(2550, .25), (680, .35)],
    "Asn": [(3380, .40), (3185, .30), (1670, .60), (1620, .40), (1410, .30)],
    "Gln": [(3380, .40), (3185, .30), (1668, .60), (1610, .40)],
    "Asp": [(1580, .55), (1402, .55)], "Glu": [(1560, .55), (1400, .55)],
    "Lys": [(3000, .45), (1630, .45), (1525, .40)], "Arg": [(3340, .45), (1673, .55), (1633, .55)],
}

# ── Raman: backbone + side chains (cm-1, rel. intensity) ─────────────────────
BB_RAMAN = [(2960, .80), (1600, .25), (1410, .60), (1330, .40), (1130, .20),
            (1030, .40), (925, .60), (850, .35)]
SIDE_RAMAN = {
    "Gly": [], "Ala": [(1460, .35), (1360, .25)], "Val": [(1450, .35), (1330, .25)],
    "Leu": [(1450, .35), (1340, .25)], "Ile": [(1450, .35)], "Pro": [(1450, .35), (900, .35)],
    "Met": [(700, .55), (655, .45), (1320, .25)],
    "Phe": [(1004, 1.0), (1032, .50), (1604, .50), (622, .30), (3060, .40)],
    "Tyr": [(644, .35), (830, .60), (853, .60), (1207, .45), (1615, .55), (3060, .35)],
    "Trp": [(760, 1.0), (880, .45), (1014, .55), (1340, .50), (1360, .50), (1552, .60), (1582, .40)],
    "His": [(1572, .40), (1410, .30), (990, .35)],
    "Ser": [(1050, .35), (880, .30)], "Thr": [(1045, .35), (875, .30)],
    "Cys": [(2570, .45), (670, .65), (500, .30)],
    "Asn": [(1670, .35), (1420, .30)], "Gln": [(1668, .35), (1420, .30)],
    "Asp": [(1418, .55), (940, .30)], "Glu": [(1415, .55), (940, .30)],
    "Lys": [(1440, .35), (1305, .25)], "Arg": [(1440, .35), (1180, .25)],
}

# ── UV-Vis: aqueous (nm, oscillator strength). Most absorb only far-UV; the
#    three aromatics (+His) add near-UV bands. ─────────────────────────────────
BB_UV = [(192, 1.0), (215, 0.06)]
SIDE_UV = {
    "Phe": [(188, .55), (206, .30), (243, .015), (252, .020), (257, .022), (263, .017)],
    "Tyr": [(193, .55), (223, .35), (275, .045), (282, .030)],
    "Trp": [(195, .65), (219, .45), (272, .045), (280, .050), (288, .040)],
    "His": [(211, .05)],
}

# ── 1H NMR in D2O (ppm, # of non-exchangeable protons). NH3+/OH/COOH/NH omitted
#    (exchange with D). Representative shifts from HMDB/BMRB-style references. ──
NMR_1H = {
    "Gly": [(3.55, 2)],
    "Ala": [(3.78, 1), (1.47, 3)],
    "Val": [(3.60, 1), (2.27, 1), (0.99, 6)],
    "Leu": [(3.73, 1), (1.70, 3), (0.96, 6)],
    "Ile": [(3.67, 1), (1.97, 1), (1.30, 2), (0.95, 6)],
    "Pro": [(4.12, 1), (3.35, 2), (2.35, 1), (2.00, 3)],
    "Met": [(3.86, 1), (2.64, 2), (2.15, 2), (2.13, 3)],
    "Phe": [(3.98, 1), (3.12, 2), (7.35, 5)],
    "Tyr": [(3.93, 1), (3.05, 2), (7.18, 2), (6.89, 2)],
    "Trp": [(4.05, 1), (3.30, 2), (7.20, 1), (7.28, 1), (7.53, 1), (7.65, 1), (7.73, 1)],
    "His": [(3.98, 1), (3.20, 2), (7.05, 1), (7.90, 1)],
    "Ser": [(3.84, 1), (3.96, 2)],
    "Thr": [(3.58, 1), (4.25, 1), (1.32, 3)],
    "Cys": [(3.97, 1), (3.05, 2)],
    "Asn": [(4.00, 1), (2.84, 2)],
    "Gln": [(3.77, 1), (2.45, 2), (2.13, 2)],
    "Asp": [(3.89, 1), (2.80, 2)],
    "Glu": [(3.75, 1), (2.35, 2), (2.12, 2)],
    "Lys": [(3.75, 1), (3.02, 2), (1.89, 2), (1.70, 2), (1.45, 2)],
    "Arg": [(3.76, 1), (3.23, 2), (1.90, 2), (1.65, 2)],
}


def _merge(backbone, side, code):
    pts = list(backbone) + list(side.get(code, []))
    c = [p[0] for p in pts]
    a = [p[1] for p in pts]
    return c, a


def main():
    uv_grid = np.linspace(185, 320, 3000)
    imgs = amino_images()
    img_kw = dict(images=imgs, image_zoom=0.085, image_dy=0.25, left_margin=0.24,
                  offset=1.8, figsize=(14, 34))
    # FTIR — zwitterionic powder
    stacks.stack_plot(
        AA, lambda code: sp.simulate(*_merge(BB_FTIR, SIDE_FTIR, code), "FTIR", fwhm=12),
        "Simulated FTIR · 20 canonical amino acids · zwitterionic powder",
        "Wavenumber (cm$^{-1}$)", (600, 3600), True, OUT / "amino_acids_FTIR.pdf", **img_kw)
    # Raman — powder
    stacks.stack_plot(
        AA, lambda code: raman_mod.simulate(*_merge(BB_RAMAN, SIDE_RAMAN, code), fwhm=10),
        "Simulated Raman · 20 canonical amino acids · powder",
        "Raman shift (cm$^{-1}$)", (300, 3150), False, OUT / "amino_acids_RAMAN.pdf", **img_kw)
    # UV-Vis — aqueous, pH 7
    stacks.stack_plot(
        AA, lambda code: uv_mod.simulate(*_merge(BB_UV, SIDE_UV, code), grid=uv_grid, fwhm_ev=0.45),
        "Simulated UV–Vis · 20 canonical amino acids · aqueous (pH 7)",
        "Wavelength (nm)", (190, 310), False, OUT / "amino_acids_UV.pdf", **img_kw)
    # 1H NMR — D2O, non-exchangeable protons only
    stacks.stack_plot(
        AA, lambda code: nmr_mod.simulate(*zip(*NMR_1H[code]), nucleus="1H", fwhm=0.025),
        "Simulated $^{1}$H NMR · 20 canonical amino acids · D$_2$O (non-exchangeable $^{1}$H)",
        "$^{1}$H chemical shift (ppm)", (0.3, 8.3), True, OUT / "amino_acids_NMR_1H.pdf", **img_kw)
    print(f"4 stacked PDFs in {OUT}")


if __name__ == "__main__":
    main()
