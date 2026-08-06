"""examples.py — runnable end-to-end demos for every technique.

Each function shows the canonical flow: pull/parse data through the library,
simulate from a DFT line list, and overlay sim-on-experimental. Used as living
documentation and a network smoke test. Run all and save figures:

    .\.venv\Scripts\python.exe examples.py            # writes to figures/

Simulation-only demos need no network. The experimental/overlay demos fetch from
whatever `SPECTRA_DATA_URL` points at and skip gracefully when it is unset or
unreachable — measured data is not part of this repository.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import spectra as sp
import gaussian as g16
import registry
import ftir, raman, uv, xrd, nmr, dsc

OUT = Path(__file__).resolve().parent / "figures" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

# Measured scans and DFT outputs come from your own store; see registry.py.
RAW = MAIN = registry.DATA_BASE_URL
NIF_IR = RAW + "DFT/Nifedipine_DFT_B3LYP_6311pp2d3p_OptFreq_Raman_Hirshfeld_ir.txt"
NIF_RAMAN = RAW + "DFT/Nifedipine_DFT_B3LYP_6311pp2d3p_OptFreq_Raman_Hirshfeld_raman_act.txt"
B3LYP_SCALE = 0.967          # B3LYP/6-311++G(2d,3p) harmonic scaling


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def ftir_overlay():
    """FTIR: overlay a Gaussian DFT IR spectrum on a measured PVP29 powder scan."""
    exp = sp.average([MAIN + "FTIR/Clancy/PVP29_2025_10_4_ECE_scan1.csv"])
    c, a = g16.parse_ir(NIF_IR)
    sim = sp.simulate(c, a, "FTIR", freq_scale=B3LYP_SCALE)
    fig, ax = sp.overlay(exp, sim, "FTIR", title="NIF DFT IR vs PVP29 powder",
                         xrange=(1800, 600), sticks=(c, a))
    _save(fig, "example_ftir_overlay.png")


def raman_sim():
    """Raman: simulate from the Gaussian Raman activity line list (no exp data yet)."""
    c, a = g16.parse_raman(NIF_RAMAN)
    gx, gy = raman.simulate(c, a, freq_scale=B3LYP_SCALE)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gx, sp.normalize01(gy), color="#d62728", lw=1.5)
    ax.vlines(c, 0, sp.normalize01(a), color="gray", lw=0.8, alpha=0.5)
    ax.set_xlim(200, 3600); ax.set_xlabel(sp.TECH["RAMAN"]["x_label"])
    ax.set_title("NIF simulated Raman (B3LYP)", fontweight="bold")
    _save(fig, "example_raman_sim.png")


def uv_demos():
    """UV-Vis: a simulated absorption band + a Beer-Lambert dissolution series."""
    gx, gy = uv.simulate([235, 290, 340], [1.0, 0.6, 0.25])     # TD-DFT-style sticks
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gx, sp.normalize01(gy), color="#1f77b4", lw=1.8)
    ax.set_xlim(200, 500); ax.set_xlabel(sp.TECH["UV"]["x_label"])
    ax.set_title("Simulated UV-Vis (energy-space Gaussian)", fontweight="bold")
    _save(fig, "example_uv_sim.png")

    series = {"CBD:PVP 1:1": [(0, .05), (5, .30), (10, .55), (20, .80), (40, .92)],
              "CBD:PVP 1:2": [(0, .04), (5, .45), (10, .74), (20, .95), (40, .99)]}
    fig, ax = uv.plot_dissolution(series, epsilon=1.8e3, title="CBD ASD dissolution")
    _save(fig, "example_uv_dissolution.png")


def xrd_demos():
    """XRD: crystalline-vs-amorphous overlay with NIF-alpha polymorph reflections."""
    fig, ax = xrd.plot_xrd(
        [("crystalline NIF", "black", MAIN + "XRD/2025_06_07_Nifedipine_3_60_8min_pXRD.xy")],
        title="NIF pXRD + alpha reflections", xrange=(3, 40))
    tt, inten = xrd.load_reflections(
        RAW + "XRD/CrystalDiffract/NIF_Alpha_XRD_CrystalDiffract_Calculated_Reflection_List.txt")
    xrd.overlay_reflections(ax, tt, inten, color="red", label="NIF alpha (calc)")
    _save(fig, "example_xrd_polymorph.png")

    gx, gy = xrd.simulate(tt, inten)         # calc reflections -> powder pattern
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gx, sp.normalize01(gy), color="black", lw=1.2)
    ax.set_xlim(3, 40); ax.set_xlabel(sp.TECH["XRD"]["x_label"])
    ax.set_title("NIF-alpha simulated powder pattern", fontweight="bold")
    _save(fig, "example_xrd_sim.png")


def nmr_demos():
    """NMR: measured liquid 13C trace + a simulated 13C shift list."""
    fig, ax = nmr.plot_nmr(
        [("FEL liquid 13C", "black",
          RAW + "NMR/2025_05_26_felodipine_cdcl3_tms_5mm_bruker500_19.tsv")],
        nucleus="13C", xrange=(0, 200), title="Felodipine 13C (CDCl3)")
    _save(fig, "example_nmr_liquid.png")

    gx, gy = nmr.simulate([20, 24, 60, 105, 145, 167], nucleus="13C")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gx, sp.normalize01(gy), color="#d62728", lw=1.2)
    ax.set_xlim(200, 0); ax.set_xlabel("13C shift (ppm)")
    ax.set_title("Simulated 13C from a shift list", fontweight="bold")
    _save(fig, "example_nmr_sim.png")


def dsc_demo():
    """DSC: load + smooth + plot a single [step] heating trace."""
    files = [{"label": "PVP29kDa", "url": MAIN + "DSC/Earl/2025_10_06_PVP29_5Cmin_n60C_200C_x3.csv",
              "y_scale": 6, "x_min": 60, "x_max": 200}]
    traces = dsc.load_dsc_traces(files, -15, 200)
    offset = dsc.apply_per_trace_offset(traces, files, offset_step=0.2)
    fig, ax = plt.subplots(figsize=(9, 5))
    dsc.plot_dsc_with_cutoffs(ax, offset, x_min=40, x_max=200)
    dsc.format_dsc_axes(ax, "PVP29 DSC", "Temperature (°C)", "Heat Flow", 40, 200,
                        title_font_size=14, axis_label_font_size=14, axis_numbers_font_size=12)
    _save(fig, "example_dsc.png")


DEMOS = [ftir_overlay, raman_sim, uv_demos, xrd_demos, nmr_demos, dsc_demo]

if __name__ == "__main__":
    for demo in DEMOS:
        print(f"{demo.__name__}: {demo.__doc__.splitlines()[0]}")
        try:
            demo()
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}: {e})")
    print(f"figures in {OUT}")
