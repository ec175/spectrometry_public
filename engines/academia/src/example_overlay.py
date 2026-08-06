"""example_overlay.py — template: overlay a simulated spectrum on an experimental scan.

Run:  .\.venv\Scripts\python.exe example_overlay.py
This uses a placeholder line list. Tomorrow, swap it for a computed one:

    import orca
    centers, inten = orca.parse_ir("simulations/orca_outputs/cbn_freq.out")   # IR
    # centers, inten = orca.parse_raman(...)      # Raman
    # centers, inten = orca.parse_uvvis(...)      # UV-Vis (use technique="UV")
"""
import matplotlib
matplotlib.use("Agg")
import registry as reg
import spectra as sp

TECHNIQUE = "FTIR"

# pick an experimental sample that has this technique (here: first available)
sample_id = reg.samples_with(TECHNIQUE)["sample_id"].iloc[0]
exp = sp.experimental(sample_id, TECHNIQUE)            # averaged, fetched via registry

# --- placeholder simulated line list (replace with orca.parse_*(...)) -------
centers = [3300, 2920, 1735, 1620, 1450, 1100]
inten   = [0.5,  0.8,  1.0,  0.7,  0.5,  0.6]
gx, gy = sp.simulate(centers, inten, TECHNIQUE, freq_scale=0.97)   # scale for DFT IR

fig, ax = sp.overlay(exp, (gx, gy), TECHNIQUE,
                     title=f"{sample_id} — simulated vs experimental",
                     sticks=(centers, inten))
out = f"figures/overlay_{TECHNIQUE.lower()}_example.png"
fig.savefig(out, dpi=110)
print("wrote", out)
