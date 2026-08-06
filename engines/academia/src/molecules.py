"""molecules.py — render molecule thumbnails (transparent RGBA) from SMILES.

Used to put a structure image next to each trace in the stacked sim figures.
Transparent background (so it sits on the figure's white space), optional
substructure highlight (e.g. a cannabinoid acid's -COOH in blue over an
otherwise black skeleton).

    import molecules as mol
    img  = mol.render_rgba("CC(C)Cc1ccc(cc1)C(C)C(=O)O")          # ibuprofen, default colors
    acid = mol.render_rgba(cbda_smiles, bw_base=True,             # black skeleton,
                           highlight_smarts="[CX3](=O)[OX2H1]")   # COOH haloed blue
    big  = mol.render_rgba(smi, font_scale=2.9, halo=True)        # big symbols + knockout halo

`halo=True` reproduces the functional-group treatment from the manim and
Oscilloscope videos (see Oscilloscope\\osc\\drugscope.py): symbols are enlarged
and a snug ring of skeleton ink is cleared around each one, so a letter never
sits on a bond. Defaults are unchanged, so the existing figure sets (BCS,
cannabinoids, amino acids, steroids, ...) render byte-identical.
"""
import io
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D

rdDepictor.SetPreferCoordGen(True)          # cleaner, more consistent 2D layouts

ACID_SMARTS = "[CX3](=O)[OX2H1]"
ACID_BLUE = (0.62, 0.80, 0.96)          # light, soft blue for the acid-group halo


def to_mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    rdDepictor.Compute2DCoords(m)
    try:
        # align to principal axes so molecules aren't drawn at random tilts
        rdMolTransforms.CanonicalizeConformer(m.GetConformer())
    except Exception:
        pass
    return m


def heavy_atoms(smiles):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else m.GetNumHeavyAtoms()


def ring_count(smiles):
    """Number of SSSR rings (for sorting cannabinoids by ring system)."""
    from rdkit.Chem import rdMolDescriptors
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return rdMolDescriptors.CalcNumRings(m) if m else 0


def mol_weight(smiles):
    """Molecular weight (for sorting; not shown on figures)."""
    from rdkit.Chem import Descriptors
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Descriptors.MolWt(m) if m else 0.0


def _png_to_rgba(png_bytes):
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"), float) / 255.0
    except Exception:
        import matplotlib.image as mpimg
        return mpimg.imread(io.BytesIO(png_bytes), format="png")


def _disk(r):
    """Circular structuring element of radius r (for the halo dilation)."""
    r = int(max(1, round(r)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _knockout_halo(rgba_labelled, rgba_bare, halo_px, inner_px):
    """Clear a snug ring of skeleton ink around every atom symbol.

    Mirrors the knockout halo in Oscilloscope\\osc\\drugscope.py: bonds are drawn
    full-length, then a glyph-sized region is zeroed so a letter never sits on a
    bond. Here the glyph footprint is recovered by differencing the labelled
    render against a `noAtomLabels` render of the same molecule -- RDKit shortens
    bonds under a label, so `labelled - bare` lights up on the glyph.

    The difference goes blind along the thin strip where the glyph overlaps the
    bond it replaced, which is exactly why `inner_px` exists: dilating by a bit
    more than a bond half-width closes that strip, so the protected core covers
    the whole glyph and only the ring beyond it is cleared.
    """
    from scipy import ndimage

    a_lab = rgba_labelled[..., 3]
    a_bare = rgba_bare[..., 3]
    glyph = np.clip(a_lab - a_bare, 0.0, 1.0) > 0.2
    if not glyph.any():                         # nothing labelled (e.g. pure hydrocarbon)
        return rgba_labelled

    inner = ndimage.binary_dilation(glyph, _disk(inner_px))
    outer = ndimage.binary_dilation(glyph, _disk(inner_px + halo_px))
    clear = outer & ~inner

    out = rgba_labelled.copy()
    out[..., 3] *= ~clear
    return out


def _draw(m, size, *, pad, bond_width, font_scale, bw_base, label_pad,
          max_font_px, no_labels, hl_atoms, hl_bonds, a_colors, b_colors):
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = bond_width
    o.padding = pad
    # maxFontSize defaults to 40 px, which silently caps baseFontSize on a large
    # canvas -- raise it or font_scale above ~1.5 does nothing at 1200 px wide.
    # Do NOT uncap it (-1): baseFontSize is in MOLECULE units, so RDKit scales the
    # font with the drawing scale and a tiny molecule (isoxazole, 5 atoms) gets
    # letters big enough to swallow its own ring. The clamp is the same trick as
    # drugscope's sym_px_max.
    o.maxFontSize = int(max_font_px)
    o.baseFontSize = 0.6 * font_scale          # larger O / OH / N / H labels
    o.additionalAtomLabelPadding = label_pad   # RDKit's own bond-shortening gap
    o.noAtomLabels = no_labels
    if bw_base:
        o.useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, m, highlightAtoms=hl_atoms, highlightBonds=hl_bonds,
        highlightAtomColors=a_colors, highlightBondColors=b_colors)
    d.FinishDrawing()
    return _png_to_rgba(d.GetDrawingText())


def render_rgba(smiles, *, size=(1200, 900), bw_base=False, highlight_smarts=None,
                highlight_color=ACID_BLUE, bond_width=None, pad=0.06, font_scale=1.5,
                halo=False, halo_px=None, label_pad=None, max_font_px=None):
    """SMILES -> high-resolution RGBA float array (transparent bg). None on failure.

    Rendered large (default 1200x900 px) so it stays crisp when embedded small in a
    PDF. bw_base draws the skeleton black; highlight_smarts haloes a substructure
    (e.g. an acid -COOH in blue) with RDKit's default per-atom circles + per-bond
    bars. font_scale enlarges heteroatom labels (O/OH/N/H) everywhere.

    halo=True adds the knockout halo used in the manim / Oscilloscope videos: a
    snug transparent ring is cleared around every functional-group symbol so no
    bond ever runs into a letter. Costs a second RDKit render. halo_px (ring
    width) and label_pad (RDKit's own bond-shortening gap) default to values
    scaled off the bond width, so `halo=True` alone is usually enough.

    max_font_px caps label size in pixels. It matters because baseFontSize is in
    MOLECULE units: a 5-atom molecule is drawn at a much larger scale than a
    25-atom one, so without a clamp the same font_scale gives a tiny molecule
    letters that swallow its own ring.

    Defaults are unchanged with halo=False, so existing figure sets (BCS,
    cannabinoids, amino acids) render byte-identical.
    """
    try:
        m = to_mol(smiles)
    except ValueError:
        return None

    bw = bond_width if bond_width else max(2.0, size[0] / 180.0)
    # Scale the ring off BOND WIDTH, not font size: the halo only has to clear a
    # stroke, so a couple of line-widths is enough. Scaling it off the font (or
    # off canvas width) overshoots badly at large font_scale and eats whole bonds.
    if halo_px is None:
        halo_px = max(2.5, 0.5 * bw)
    if label_pad is None:
        label_pad = 0.06 if halo else 0.0      # fixed: RDKit's pad is already font-relative
    if max_font_px is None:
        # ~40 px at the library default 1200x900 (RDKit's own default), rising
        # with font_scale so the clamp only bites on very small molecules.
        max_font_px = max(12.0, size[0] / 45.0 * font_scale)

    hl_atoms, hl_bonds, a_colors, b_colors = [], [], {}, {}
    if highlight_smarts:
        patt = Chem.MolFromSmarts(highlight_smarts)
        atoms = set()
        if patt is not None:
            for match in m.GetSubstructMatches(patt):
                atoms.update(match)
        hl_atoms = list(atoms)
        a_colors = {i: highlight_color for i in hl_atoms}
        for b in m.GetBonds():
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
                hl_bonds.append(b.GetIdx())
                b_colors[b.GetIdx()] = highlight_color

    kw = dict(pad=pad, bond_width=bw, font_scale=font_scale, bw_base=bw_base,
              label_pad=label_pad, max_font_px=max_font_px,
              hl_atoms=hl_atoms, hl_bonds=hl_bonds,
              a_colors=a_colors, b_colors=b_colors)
    labelled = _draw(m, size, no_labels=False, **kw)
    if not halo:
        return labelled
    try:
        bare = _draw(m, size, no_labels=True, **kw)
        return _knockout_halo(labelled, bare, halo_px, inner_px=0.75 * bw)
    except Exception:
        return labelled            # halo is cosmetic -- never lose the thumbnail


if __name__ == "__main__":
    for name, smi in [("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
                      ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]:
        img = render_rgba(smi)
        print(f"{name}: {None if img is None else img.shape}, heavy={heavy_atoms(smi)}")
