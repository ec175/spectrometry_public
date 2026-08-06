"""bcs_sim_test.py — crystalline-vs-amorphous simulated spectra for BCS Class 1-4
oral drugs, one stacked figure per (class, technique), with a structure thumbnail
beside each row.

Spectra come from predict.py (SMILES -> group-contribution stick lists), so any
drug works without hand-typing peaks. Crystalline = sharp broadening, amorphous =
broad (the disorder signature), identical to the amino-acid crystal/amorphous set.
NMR uses predicted 13C (CP-MAS style, the solid-state comparison). Rows are ordered
by DESCENDING molar mass. Drug list + SMILES live in bcs_drugs.py (workflow-verified;
RDKit-validated; too-large-to-render molecules excluded).

Run:  .\.venv\Scripts\python.exe bcs_sim_test.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

import spectra as sp
import uv as uv_mod
import nmr as nmr_mod
import predict
import molecules
import stacks

try:
    from bcs_drugs import DRUGS          # {class:int -> [(name, smiles, mw), ...]}
except Exception:
    DRUGS = {}

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
UV_GRID = np.linspace(195, 360, 3000)


def sim_ftir(smi, fwhm):  return sp.simulate(*predict.ir(smi), "FTIR", fwhm=fwhm)
def sim_raman(smi, fwhm): return sp.simulate(*predict.raman(smi), "RAMAN", fwhm=fwhm)
def sim_uv(smi, fwhm):    return uv_mod.simulate(*predict.uv(smi), grid=UV_GRID, fwhm_ev=fwhm)
def sim_13c(smi, fwhm):   return nmr_mod.simulate(*predict.c13(smi), nucleus="13C", fwhm=fwhm)


# (tech tag, simulate(smi,fwhm), crystalline fwhm, amorphous fwhm, xlabel, xrange, descending)
TECHS = [
    ("FTIR",  sim_ftir,  7.0,  26.0, "Wavenumber (cm$^{-1}$)",        (600, 3600), True),
    ("Raman", sim_raman, 6.0,  20.0, "Raman shift (cm$^{-1}$)",       (300, 3200), False),
    ("UV",    sim_uv,    0.30, 0.55, "Wavelength (nm)",               (205, 355),  False),
    ("13C",   sim_13c,   1.0,  5.0,  "$^{13}$C chemical shift (ppm)", (0, 200),    True),
]


def build_class(cls, drugs):
    drugs = sorted(drugs, key=lambda d: -d[2])               # descending molar mass
    items = [(name, name) for name, _smi, _mw in drugs]      # label: name only (no MW)
    smap = {name: smi for name, smi, _mw in drugs}
    images = [molecules.render_rgba(smi) for _name, smi, _mw in drugs]
    n = len(drugs)
    height = max(16, n * 1.7)
    clsdir = OUT / f"bcs_class{cls}"
    clsdir.mkdir(parents=True, exist_ok=True)
    for tag, simfn, cf, af, xlabel, xrange, desc in TECHS:
        def pair(code, f=simfn, cf=cf, af=af):
            return f(smap[code], cf), f(smap[code], af)
        stacks.stack_overlay(
            items, pair,
            f"Simulated {tag} · BCS Class {cls} · crystalline vs amorphous",
            xlabel, xrange, desc, clsdir / f"bcs_class{cls}_{tag}_xtal_vs_amorphous.pdf",
            color_a="black", color_b="#d62728", label_a="crystalline", label_b="amorphous",
            images=images, offset=1.8, figsize=(14, height),
            image_zoom=0.085, image_dy=0.25, left_margin=0.24)


def main():
    if not DRUGS:
        print("bcs_drugs.py not found / empty — populate DRUGS first.")
        return
    for cls in sorted(DRUGS):
        print(f"BCS Class {cls}: {len(DRUGS[cls])} drugs")
        build_class(cls, DRUGS[cls])
    print(f"{4 * len(DRUGS)} BCS PDFs in {OUT}")


if __name__ == "__main__":
    main()
