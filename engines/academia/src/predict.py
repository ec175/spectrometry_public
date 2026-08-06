"""predict.py — crude SMILES -> spectroscopic line lists via functional-group rules.

A *qualitative* group-contribution estimator so structurally diverse molecules
(e.g. ~80 BCS drugs) get plausible, differentiated stick spectra without
hand-typing each. NOT quantitative — peak positions follow standard
group-frequency / chemical-shift tables; intensities are schematic. Good enough
to drive the simulate()/broaden() pipeline and a crystalline-vs-amorphous
(broadening) comparison.

    import predict, spectra as sp
    c, a = predict.ir("CC(C)Cc1ccc(cc1)C(C)C(=O)O")     # ibuprofen IR sticks
    gx, gy = sp.simulate(c, a, "FTIR", fwhm=8)

Functions: ir, raman, uv, c13  -> (centers, intensities). All take a SMILES.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ── functional-group SMARTS (order matters: specific carbonyls before generic) ─
_SMARTS = {k: Chem.MolFromSmarts(v) for k, v in {
    "acid":        "[CX3](=O)[OX2H1]",
    "ester":       "[#6][CX3](=O)[OX2H0][#6]",
    "amide":       "[NX3][CX3](=O)",
    "aldehyde":    "[CX3H1](=O)",
    "ketone":      "[#6][CX3](=O)[#6]",
    "anhydride":   "[CX3](=O)[OX2][CX3](=O)",
    "alcohol":     "[#6;!$([#6]=O)][OX2H1]",
    "phenol":      "[c][OX2H1]",
    "ether":       "[OD2]([#6])[#6]",
    "amine1":      "[NX3;H2;!$(NC=O);!$(N=O)]",
    "amine2":      "[NX3;H1;!$(NC=O);!$(N=O)]",
    "nitro":       "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile":     "[NX1]#[CX2]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "sulfone":     "[#6][SX4](=O)(=O)[#6]",
    "alkene":      "[CX3]=[CX3]",
    "fluoride":    "[#6][F]",
    "chloride":    "[#6][Cl]",
    "thiol":       "[#16X2H1]",
    "pyridine":    "[$(n1ccccc1)]",
}.items()}


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return m


def _has(m, key):
    p = _SMARTS[key]
    return 0 if p is None else len(m.GetSubstructMatches(p))


def _flags(m):
    f = {k: _has(m, k) for k in _SMARTS}
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(m)
    n_aliph_CH = sum(1 for a in m.GetAtoms()
                     if a.GetSymbol() == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() > 0)
    return f, n_arom_rings, n_aliph_CH


# ── FTIR ────────────────────────────────────────────────────────────────────
def ir(smiles):
    m = _mol(smiles)
    f, n_ar, n_chr = _flags(m)
    pts = []
    if n_chr:
        pts += [(2960, .45), (2920, .55), (2870, .35)]          # aliphatic C-H
    if n_ar:
        pts += [(3055, .25), (1600, .55), (1585, .40), (1495, .45),
                (1450, .35), (830, .30), (760, .35), (695, .30)]  # aromatic
    if f["acid"]:
        pts += [(1710, .95), (3000, .45), (2600, .30), (1420, .40), (1280, .50)]
    if f["ester"] or f["anhydride"]:
        pts += [(1735, .85), (1250, .55), (1100, .40)]
    if f["amide"]:
        pts += [(1660, .85), (1540, .55), (3320, .50)]
    if f["ketone"]:
        pts += [(1715, .75)]
    if f["aldehyde"]:
        pts += [(1725, .65), (2720, .20)]
    if f["alcohol"]:
        pts += [(3350, .50), (1050, .40)]
    if f["phenol"]:
        pts += [(3300, .50), (1230, .40)]
    if f["amine1"]:
        pts += [(3370, .40), (3300, .35), (1610, .30)]
    if f["amine2"]:
        pts += [(3300, .30)]
    if f["ether"]:
        pts += [(1110, .40)]
    if f["nitro"]:
        pts += [(1520, .60), (1345, .55)]
    if f["nitrile"]:
        pts += [(2240, .40)]
    if f["sulfonamide"] or f["sulfone"]:
        pts += [(1330, .50), (1150, .50)]
    if f["alkene"]:
        pts += [(1640, .35)]
    if f["fluoride"]:
        pts += [(1200, .50), (1140, .40)]
    if f["chloride"]:
        pts += [(740, .40)]
    if f["pyridine"]:
        pts += [(1590, .40), (1440, .30)]
    if not pts:
        pts = [(2950, .5), (1450, .4)]
    return _unzip(pts)


# ── Raman ───────────────────────────────────────────────────────────────────
def raman(smiles):
    m = _mol(smiles)
    f, n_ar, n_chr = _flags(m)
    pts = []
    if n_chr:
        pts += [(2930, .70), (2870, .45), (1450, .40)]
    if n_ar:
        pts += [(1600, .65), (1030, .35), (1003, .75 if n_ar == 1 else .45), (785, .30)]
    if f["acid"] or f["ester"] or f["ketone"] or f["amide"] or f["aldehyde"]:
        pts += [(1700, .35)]
    if f["alkene"]:
        pts += [(1650, .55), (1620, .40)]
    if f["nitro"]:
        pts += [(1350, .60)]
    if f["nitrile"]:
        pts += [(2240, .50)]
    if f["sulfone"] or f["sulfonamide"]:
        pts += [(1150, .40)]
    if f["chloride"]:
        pts += [(650, .40)]
    if f["thiol"]:
        pts += [(2570, .45), (660, .50)]
    if not pts:
        pts = [(2930, .7), (1450, .4)]
    return _unzip(pts)


# ── UV-Vis (nm, oscillator strength) ────────────────────────────────────────
def uv(smiles):
    m = _mol(smiles)
    f, n_ar, _ = _flags(m)
    n_conj = _conjugated_double_bonds(m)
    pts = [(205, 1.0)]                                  # sigma->sigma* / backbone
    if n_ar >= 1:
        pts += [(255, 0.30)]                            # benzene B-band
    if n_ar >= 2:
        pts += [(280, 0.45), (305, 0.25)]               # fused/extended aromatic
    if n_ar >= 3:
        pts += [(330, 0.30)]
    if f["nitro"]:
        pts += [(270, 0.40), (335, 0.20)]
    if f["amide"] or f["ester"] or f["acid"]:
        pts += [(215, 0.10)]                            # n->pi*
    if f["ketone"] or f["aldehyde"]:
        pts += [(280, 0.08)]
    if n_conj >= 3:
        pts += [(250 + 12 * min(n_conj, 8), 0.35)]      # extended conjugation red-shift
    return _unzip(pts)


# ── 13C (CP-MAS style stick list) ───────────────────────────────────────────
def c13(smiles):
    m = _mol(smiles)
    shifts = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        shifts.append(_carbon_shift(a))
    if not shifts:
        shifts = [30.0]
    return list(map(float, shifts)), [1.0] * len(shifts)


def _carbon_shift(a):
    nbrs = a.GetNeighbors()
    has_dbl_O = any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "O"
                    for b in a.GetBonds())
    nbr_syms = [n.GetSymbol() for n in nbrs]
    if has_dbl_O:                                        # carbonyl
        if "N" in nbr_syms:
            return 170.0                                 # amide
        if sum(n.GetSymbol() == "O" for n in nbrs) >= 2:
            return 172.0                                 # acid/ester
        if a.GetTotalNumHs() == 1:
            return 192.0                                 # aldehyde
        return 205.0                                     # ketone
    if a.GetIsAromatic():
        if any(n.GetSymbol() == "O" for n in nbrs):
            return 156.0
        if any(n.GetSymbol() == "N" for n in nbrs):
            return 149.0
        return 129.0
    # sp2 alkene
    if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "C"
           for b in a.GetBonds()):
        return 130.0
    # sp3
    if any(n.GetSymbol() == "O" for n in nbrs):
        return 66.0
    if any(n.GetSymbol() == "N" for n in nbrs):
        return 45.0
    n_heavy_C = sum(n.GetSymbol() == "C" for n in nbrs)
    if n_heavy_C <= 1 and a.GetTotalNumHs() >= 2:
        return 18.0                                      # CH3-ish
    return 32.0


# ── helpers ──────────────────────────────────────────────────────────────────
def _conjugated_double_bonds(m):
    return sum(1 for b in m.GetBonds()
               if b.GetIsConjugated() and b.GetBondTypeAsDouble() == 2)


def _unzip(pts):
    return [p[0] for p in pts], [p[1] for p in pts]


if __name__ == "__main__":
    for name, smi in [("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
                      ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
                      ("nifedipine", "CCOC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC")]:
        c, a = ir(smi)
        cs, _ = c13(smi)
        print(f"{name:11s} IR lines={len(c):2d}  13C carbons={len(cs):2d}  "
              f"strongest IR~{c[int(np.argmax(a))]:.0f}cm-1")
