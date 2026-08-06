"""extra_classes_sim_test.py — simulate FTIR / Raman / UV / 13C for four new
compound classes (neurotransmitters, steroids, small peptides, fats/lipids) from
SMILES via predict.py, and render stacked PDFs per technique.

Three sets per class (mirrors the amino-acid / cannabinoid layout):
  <class>/<class>_<TECH>.pdf                                    base  (neutral powder, single trace)
  <class>_xtal_vs_amorphous/<class>_<TECH>_xtal_vs_amorphous.pdf  crystalline vs amorphous  (shared branch)
  <class>_<branch>/<class>_<TECH>_<token>.pdf                   per-class form pair

SHARED branch = the SAME predicted line list, sharp vs broad FWHM (the honest
crystalline->amorphous broadening model). PER-CLASS branch = a second chemical
form derived by an RDKit transform of the SMILES (protonation / esterification /
oxidation / deprotonation), re-run through predict.py, PLUS a small modifier line
list supplying the new group's bands that predict.py's group vocabulary lacks
(ammonium / disulfide / carboxylate; the ester appears in predict on its own).

Structure thumbnails use image_zoom=0.085 to match the existing amino-acid figures.

Run:  .\.venv\Scripts\python.exe extra_classes_sim_test.py [class ...]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

import predict
import spectra as sp
import raman as raman_mod
import uv as uv_mod
import nmr as nmr_mod
import stacks
import molecules
from chem_data import (NEUROTRANSMITTER_SMILES, STEROID_SMILES,
                       PEPTIDE_SMILES, LIPID_SMILES)

FIG = Path(__file__).resolve().parent / "figures"
UV_GRID = np.linspace(185, 360, 3000)

CRYST_COLOR, AMOR_COLOR = "black", "#d62728"
BASE_COLOR, ALT_COLOR = "black", "#1f77b4"


# ── techniques: predict fn + broadener + axes + sharp/broad FWHM ─────────────
def _sim_ftir(c, a, fwhm):  return sp.simulate(c, a, "FTIR", fwhm=fwhm)
def _sim_raman(c, a, fwhm): return raman_mod.simulate(c, a, fwhm=fwhm)
def _sim_uv(c, a, fwhm):    return uv_mod.simulate(c, a, grid=UV_GRID, fwhm_ev=fwhm)
def _sim_nmr(c, a, fwhm):   return nmr_mod.simulate(c, a, nucleus="13C", fwhm=fwhm)

TECHS = [
    dict(name="FTIR",   pred="ir",    sim=_sim_ftir,  token="FTIR",
         xlabel="Wavenumber (cm$^{-1}$)",          xrange=(600, 3600), desc=True,  sharp=9,    broad=26),
    dict(name="RAMAN",  pred="raman", sim=_sim_raman, token="RAMAN",
         xlabel="Raman shift (cm$^{-1}$)",         xrange=(300, 3150), desc=False, sharp=8,    broad=20),
    dict(name="UV",     pred="uv",    sim=_sim_uv,    token="UV",
         xlabel="Wavelength (nm)",                 xrange=(190, 345),  desc=False, sharp=0.35, broad=0.60),
    dict(name="13C",    pred="c13",   sim=_sim_nmr,   token="NMR_13C",
         xlabel="$^{13}$C chemical shift (ppm)",   xrange=(0, 210),    desc=True,  sharp=1.5,  broad=5.0),
]


# ── per-class "individual" branch: an RDKit SMILES transform + a band modifier ─
def _rxn(smiles, smarts):
    """Apply a one-product reaction SMARTS; return (new_smiles, changed)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles, False
    prods = AllChem.ReactionFromSmarts(smarts).RunReactants((m,))
    if not prods:
        return smiles, False
    p = prods[0][0]
    try:
        Chem.SanitizeMol(p)
        return Chem.MolToSmiles(p), True
    except Exception:
        return smiles, False


def protonate_amine(smiles):
    """Protonate the first basic aliphatic amine (freebase -> R-NH3+ salt form)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles, False
    patt = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!$(N=O);!$([N+])]")
    matches = m.GetSubstructMatches(patt)
    if not matches:
        return smiles, False
    rw = Chem.RWMol(m)
    a = rw.GetAtomWithIdx(matches[0][0])
    a.SetFormalCharge(1)
    a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
    try:
        mm = rw.GetMol()
        Chem.SanitizeMol(mm)
        return Chem.MolToSmiles(mm), True
    except Exception:
        return smiles, False


def esterify_oh(smiles):   return _rxn(smiles, "[OX2H:1]>>[O:1]C(C)=O")        # free -OH -> acetate
def oxidize_thiol(smiles): return _rxn(smiles, "[SX2H:1]>>[S:1]SC")           # -SH -> disulfide
def deprotonate_acid(smiles): return _rxn(smiles, "[CX3:1](=O)[OX2H:2]>>[C:1](=O)[O-:2]")  # -COOH -> -COO-

# bands the new group adds that predict.py has no vocabulary for (by technique name)
MOD_AMMONIUM   = {"FTIR": [(3000, .45), (2700, .35), (2150, .25), (1600, .40), (1490, .35)],
                  "RAMAN": [(3000, .20), (1600, .20)]}
MOD_DISULFIDE  = {"RAMAN": [(510, .55)], "FTIR": [(510, .20)]}
MOD_CARBOXYLATE = {"FTIR": [(1560, .85), (1410, .60)], "RAMAN": [(1410, .45)]}
MOD_NONE = {}   # esterification: predict detects the ester C=O on its own

CLASSES = [
    dict(slug="neurotransmitters", title="neurotransmitters", smiles=NEUROTRANSMITTER_SMILES,
         branch_dir="neurotransmitters_protonation", token="freebase_vs_protonated",
         branch_title="freebase vs protonated", label_a="freebase", label_b="protonated (·H$^+$)",
         transform=protonate_amine, modifier=MOD_AMMONIUM),
    dict(slug="steroids", title="steroids", smiles=STEROID_SMILES,
         branch_dir="steroids_ester", token="free_vs_ester",
         branch_title="free vs esterified", label_a="free (–OH)", label_b="esterified (acetate)",
         transform=esterify_oh, modifier=MOD_NONE),
    dict(slug="peptides", title="small peptides", smiles=PEPTIDE_SMILES,
         branch_dir="peptides_redox", token="reduced_vs_oxidized",
         branch_title="reduced vs oxidized", label_a="reduced (–SH)", label_b="oxidized (S–S)",
         transform=oxidize_thiol, modifier=MOD_DISULFIDE),
    dict(slug="lipids", title="fats / lipids", smiles=LIPID_SMILES,
         branch_dir="lipids_salt", token="acid_vs_salt",
         branch_title="free acid vs salt", label_a="free acid", label_b="carboxylate salt",
         transform=deprotonate_acid, modifier=MOD_CARBOXYLATE),
]


# ── line-list helpers ────────────────────────────────────────────────────────
def base_lines(smiles, pred_name):
    c, a = getattr(predict, pred_name)(smiles)
    return list(map(float, c)), list(map(float, a))


def alt_lines(smiles, pred_name, transform, modifier):
    """The transformed-form line list: predict on the new SMILES + the modifier."""
    new_smiles, changed = transform(smiles)
    c, a = base_lines(new_smiles, pred_name)
    if changed:
        for cc, aa in modifier.get(_cur_tech_name, []):
            c.append(float(cc)); a.append(float(aa))
    return c, a


_cur_tech_name = "FTIR"  # set per technique inside the loop (modifier lookup key)


def class_images(smiles_map, items):
    return [molecules.render_rgba(smiles_map[name]) for name, _ in items]


def ordered_items(smiles_map):
    """(name, name) tuples, smallest molecule on top (ascending MW)."""
    names = sorted(smiles_map, key=lambda n: molecules.mol_weight(smiles_map[n]) or 9e9)
    return [(n, n) for n in names]


def build_class(cfg):
    smiles_map = cfg["smiles"]
    items = ordered_items(smiles_map)
    imgs = class_images(smiles_map, items)
    base_dir = FIG / cfg["slug"]
    xtal_dir = FIG / f"{cfg['slug']}_xtal_vs_amorphous"
    branch_dir = FIG / cfg["branch_dir"]
    for d in (base_dir, xtal_dir, branch_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_kw = dict(images=imgs, image_zoom=0.085, image_dy=0.25, left_margin=0.24,
                   offset=1.8, figsize=(14, 36))
    ovl_kw = dict(images=imgs, image_zoom=0.085, image_dy=0.25, left_margin=0.24,
                  offset=1.9, figsize=(14, 38))

    global _cur_tech_name
    for t in TECHS:
        _cur_tech_name = t["name"]
        pred, sim = t["pred"], t["sim"]
        sharp, broad = t["sharp"], t["broad"]
        cap = f" · {t['xlabel'].split(' (')[0]}"

        # 1) base — neutral powder, single sharp trace
        stacks.stack_plot(
            items, lambda code: sim(*base_lines(smiles_map[code], pred), sharp),
            f"Simulated {t['name']} · {cfg['title']} · powder", t["xlabel"],
            t["xrange"], t["desc"], base_dir / f"{cfg['slug']}_{t['token']}.pdf", **base_kw)

        # 2) crystalline vs amorphous — same lines, sharp vs broad
        def _xtal_pair(code, _p=pred, _s=sim, _sh=sharp, _br=broad):
            c, a = base_lines(smiles_map[code], _p)
            return _s(c, a, _sh), _s(c, a, _br)
        stacks.stack_overlay(
            items, _xtal_pair,
            f"Simulated {t['name']} · {cfg['title']} · crystalline vs amorphous", t["xlabel"],
            t["xrange"], t["desc"], xtal_dir / f"{cfg['slug']}_{t['token']}_xtal_vs_amorphous.pdf",
            color_a=CRYST_COLOR, color_b=AMOR_COLOR, label_a="crystalline", label_b="amorphous", **ovl_kw)

        # 3) per-class form pair — neutral vs transformed form
        def _branch_pair(code, _p=pred, _s=sim, _sh=sharp, _cfg=cfg):
            bc, ba = base_lines(smiles_map[code], _p)
            ac, aa = alt_lines(smiles_map[code], _p, _cfg["transform"], _cfg["modifier"])
            return _s(bc, ba, _sh), _s(ac, aa, _sh)
        stacks.stack_overlay(
            items, _branch_pair,
            f"Simulated {t['name']} · {cfg['title']} · {cfg['branch_title']}", t["xlabel"],
            t["xrange"], t["desc"], branch_dir / f"{cfg['slug']}_{t['token']}_{cfg['token']}.pdf",
            color_a=BASE_COLOR, color_b=ALT_COLOR, label_a=cfg["label_a"], label_b=cfg["label_b"], **ovl_kw)

    print(f"  {cfg['slug']}: 12 PDFs (base + xtal/amorphous + {cfg['branch_dir'].split('_',1)[-1]})")


def main():
    want = set(a.lower() for a in sys.argv[1:])
    todo = [c for c in CLASSES if not want or c["slug"] in want or c["title"] in want]
    for cfg in todo:
        print(f"building {cfg['title']} ...")
        build_class(cfg)
    print(f"done: {len(todo)} classes")


if __name__ == "__main__":
    main()
