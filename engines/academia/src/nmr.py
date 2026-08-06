"""nmr.py — solid- and liquid-state NMR loading, plotting, and simulation.

NMR probes local bonding/H-bonding directly. The lab has Varian ssNMR (MAS) and
Bruker liquid-state data; this module reads both, draws the conventional
ppm-reversed stacked comparison (crystalline vs amorphous vs ASD), and can
broaden a chemical-shift list (reference table or DFT GIAO) into a spectrum.

  load           - tolerant (ppm, intensity) reader for ssNMR .txt or liquid .tsv
  import_varian  - ssNMR convenience (Varian "writedata" export)
  plot_nmr       - ppm-reversed stacked traces, 1H or 13C
  simulate       - broaden a shift list; nucleus="1H" or "13C" sets the window
  overlay        - simulated over measured (both normalized)

No NMR scans are registered yet; pass explicit URLs as (label, color, source)
tuples (registry sample_ids work once NMR is registered).

    import nmr
    nmr.plot_nmr([("crystalline NIF", "blue", NIF_x_C13_url),
                  ("amorphous NIF",  "black", NIF_a_C13_url)],
                 nucleus="13C", title="NIF 13C CP-MAS")
"""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import spectra as sp

TECHNIQUE = "NMR"
_NUCLEUS_TECH = {"1H": "NMR-1H", "13C": "NMR-13C", "H": "NMR-1H", "C": "NMR-13C"}


# === loading ================================================================
def _read(source, *, timeout=30):
    s = str(source)
    if s.startswith(("http://", "https://")):
        import requests
        return requests.get(s, timeout=timeout).text
    return Path(source).read_text(errors="ignore")


def load(source, *, skip_rows=None):
    """Read a 2-column (ppm, intensity) NMR file, oriented ascending in ppm.

    Tolerant of large instrument headers (Varian/Bruker): non-numeric lines are
    skipped automatically. Pass skip_rows to force a fixed header size instead.
    """
    lines = _read(source).splitlines()
    if skip_rows is not None:
        lines = lines[skip_rows:]
    rows = []
    for line in lines:
        parts = re.split(r"[\s,]+", line.strip())
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not rows:
        raise ValueError(f"No (ppm, intensity) data in {source!r}")
    arr = np.asarray(rows, float)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def import_varian(source, skip_rows=482):
    """Varian ssNMR 'writedata' export -> (ppm, intensity). Header is ~482 lines;
    override skip_rows if the export differs (load() also auto-skips non-numeric)."""
    try:
        return load(source, skip_rows=skip_rows)
    except ValueError:
        return load(source)          # fall back to auto-skip


# === simulation =============================================================
def shieldings_to_shifts(shieldings, reference_shielding):
    """Convert DFT GIAO isotropic shieldings (sigma) to chemical shifts (delta ppm).

    delta = sigma_ref - sigma, where sigma_ref is the reference compound's
    computed isotropic shielding AT THE SAME LEVEL OF THEORY (e.g. TMS for 1H/13C).
    `shieldings` may be a list of floats or of (index, element, sigma) tuples
    (as returned by gaussian.parse_nmr_shieldings / orca.parse_nmr).
    """
    sig = np.array([s[2] if isinstance(s, (tuple, list)) else s for s in shieldings], float)
    return reference_shielding - sig


def simulate(shifts, intensities=None, *, nucleus="13C", fwhm=None, grid=None,
             npoints=4000):
    """Broaden a chemical-shift list into a spectrum.

    shifts      : ppm positions (DFT GIAO-derived or a reference table).
    intensities : per-shift heights; defaults to all ones.
    nucleus     : "1H" or "13C" -> sets the ppm window and default linewidth.
    Returns (ppm grid, intensity).
    """
    tech = _NUCLEUS_TECH.get(str(nucleus), TECHNIQUE)
    shifts = np.asarray(shifts, float)
    if intensities is None:
        intensities = np.ones_like(shifts)
    return sp.simulate(shifts, intensities, tech, grid=grid, fwhm=fwhm, npoints=npoints)


def overlay(experimental_xy, simulated_xy, *, nucleus="13C", **kw):
    """Overlay a simulated NMR spectrum on a measured one. See spectra.overlay."""
    tech = _NUCLEUS_TECH.get(str(nucleus), TECHNIQUE)
    kw.setdefault("exp_label", "measured")
    return sp.overlay(experimental_xy, simulated_xy, tech, **kw)


# === plotting ===============================================================
def plot_nmr(samples, *, nucleus="13C", xrange=None, title="", vertical_offset=0.05,
             default_color="black", linewidth=0.8, figsize=(9, 6),
             skip_rows=None, label_traces=True):
    """Stack measured NMR traces with a reversed ppm axis (high ppm on the left).

    samples : (label, color, source) tuples or registry sample_ids.
    nucleus : "1H" or "13C" — only sets the default ppm window when xrange is None.
    Returns (fig, ax).
    """
    tech = _NUCLEUS_TECH.get(str(nucleus), TECHNIQUE)
    lo, hi = sorted(xrange or sp.TECH[tech]["default_range"])
    fig, ax = plt.subplots(figsize=figsize)
    for i, item in enumerate(samples):
        label, color, sources = sp.resolve(item, TECHNIQUE)
        if len(sources) == 1:
            x, y = load(sources[0], skip_rows=skip_rows)
        else:
            xs, ys = zip(*[load(s, skip_rows=skip_rows) for s in sources])
            x = xs[0]
            y = np.mean([np.interp(x, xi, yi) for xi, yi in zip(xs, ys)], axis=0)
        m = (x >= lo) & (x <= hi)
        y = sp.normalize01(y, lo=y[m].min(), hi=y[m].max()) if m.any() else sp.normalize01(y)
        y = y + i * vertical_offset
        ax.plot(x, y, color=color or default_color, lw=linewidth, label=label)
        if label_traces and m.any():
            ax.text(lo, y[m][np.argmin(np.abs(x[m] - lo))], f"{label} ",
                    va="bottom", ha="right", fontsize=10, color=color or default_color)
    ax.set_xlim(hi, lo)                      # reversed: high ppm on the left
    ax.set_xlabel(f"$^{{{nucleus[:-1]}}}${nucleus[-1]} chemical shift (ppm)"
                  if nucleus[-1].isalpha() else sp.TECH[TECHNIQUE]["x_label"], fontsize=13)
    ax.set_ylabel(sp.TECH[TECHNIQUE]["y_label"], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    gx, gy = simulate([22, 36, 110, 145, 168], nucleus="13C")
    print(f"NMR 13C sim {gx.min():.0f}-{gx.max():.0f} ppm, peak {gx[int(np.argmax(gy))]:.0f} ppm")
    gx, gy = simulate([1.2, 2.3, 7.1], nucleus="1H")
    print(f"NMR 1H sim {gx.min():.0f}-{gx.max():.0f} ppm, {len(gx)} pts")
