"""spectra.py — technique-agnostic spectrum loading, simulation, and overlay.

Generalizes the FTIR conventions in ftir.py to UV-Vis and Raman: every spectrum is
a 2-column (x, y) trace. This module adds

  * load_xy / average / normalize01 — load (URL or local), average replicates, scale
  * broaden                         — line list (positions + intensities) -> curve
  * simulate                        — broaden with per-technique defaults + freq_scale
  * overlay                         — draw a SIMULATED spectrum over the EXPERIMENTAL
  * experimental(sample_id, tech)   — averaged experimental trace via registry.py

The simulated line list can come from anywhere: a reference table, or a computed
quantum-chemistry run (see orca.py). Nothing here depends on ORCA.

Example
-------
    import spectra as sp
    ex = sp.experimental("CBN-PVP29-1_2", "FTIR")        # averaged powder scan
    gx, gy = sp.simulate([1735, 1620, 1450], [1.0, .6, .4], "FTIR", freq_scale=0.97)
    sp.overlay(ex, (gx, gy), "FTIR", title="CBN:PVP 1:2 — sim vs powder")
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import registry as reg
except Exception:                      # registry optional (only for experimental())
    reg = None

HERE = Path(__file__).resolve().parent

# --- per-technique defaults -------------------------------------------------
# descending = high value on the LEFT (FTIR cm-1, NMR ppm); shape/default_fwhm
# seed simulate(); default_range is the plotted window (overlay sorts it).
TECH = {
    "FTIR":  dict(x_label="Wavenumber (cm$^{-1}$)", y_label="Absorbance",
                  descending=True,  shape="lorentzian", default_fwhm=6.0,
                  default_range=(4000, 400)),
    "RAMAN": dict(x_label="Raman shift (cm$^{-1}$)", y_label="Intensity",
                  descending=False, shape="lorentzian", default_fwhm=6.0,
                  default_range=(200, 3600)),
    "UV":    dict(x_label="Wavelength (nm)",          y_label="Absorbance",
                  descending=False, shape="gaussian",  default_fwhm=20.0,
                  default_range=(200, 800)),
    # Powder XRD: a stick reflection list (2theta, intensity) broadened with a
    # pseudo-Voigt instrument profile. FWHM is in degrees 2theta (~0.1-0.3 deg
    # on a lab diffractometer). See xrd.py for Q-conversion and polymorph lines.
    "XRD":   dict(x_label="2$\\theta$ (degree)",      y_label="Intensity (a.u.)",
                  descending=False, shape="pseudovoigt", default_fwhm=0.15,
                  default_range=(3, 50)),
    # NMR: a chemical-shift stick list broadened with a Lorentzian. Defaults suit
    # 13C (0-200 ppm, ~0.5 ppm lines); override range/fwhm for 1H (e.g. (0, 12),
    # ~0.02 ppm). High ppm on the left (descending). See nmr.py.
    "NMR":   dict(x_label="Chemical shift (ppm)",     y_label="Intensity",
                  descending=True,  shape="lorentzian", default_fwhm=0.5,
                  default_range=(0, 200)),
}
TECH["UV-VIS"] = TECH["UVVIS"] = TECH["UV"]      # aliases
TECH["PXRD"] = TECH["XRD"]                       # alias
TECH["NMR-1H"] = dict(TECH["NMR"], default_fwhm=0.02, default_range=(0, 12))
TECH["NMR-13C"] = dict(TECH["NMR"])              # 13C == NMR defaults


def _cfg(technique):
    t = technique.upper()
    if t not in TECH:
        raise KeyError(f"Unknown technique {technique!r}; known: {sorted(TECH)}")
    return TECH[t]


# --- loading ----------------------------------------------------------------
def load_xy(source, *, timeout=15):
    """Load a 2-column spectrum from a URL or local path -> (x, y), ascending in x.

    Non-numeric header lines (instrument metadata) are skipped, matching the lab's
    scan format. Accepts comma- or tab-separated.
    """
    s = str(source)
    if s.startswith(("http://", "https://")):
        import requests
        text = requests.get(s, timeout=timeout).text
    else:
        text = Path(source).read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        parts = line.replace("\t", ",").strip().split(",")
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not rows:
        raise ValueError(f"No numeric 2-column data found in {source!r}")
    arr = np.asarray(rows, float)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def average(sources):
    """Pointwise mean of several scans, interpolated onto the first scan's grid."""
    xs, ys = zip(*[load_xy(s) for s in sources])
    x = xs[0]
    yavg = np.mean([np.interp(x, xi, yi) for xi, yi in zip(xs, ys)], axis=0)
    return x, yavg


def normalize01(y, lo=None, hi=None):
    """Scale to 0-1. Pass lo/hi to normalize within a sub-range's min/max."""
    y = np.asarray(y, float)
    lo = y.min() if lo is None else lo
    hi = y.max() if hi is None else hi
    return (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)


def experimental(sample_id, technique, *, include_only=True):
    """Averaged experimental trace for a registry sample_id + technique."""
    if reg is None:
        raise RuntimeError("registry not importable; load files manually with load_xy")
    urls = reg.urls_for(sample_id, technique, include_only=include_only)
    if not urls:
        raise LookupError(f"No {technique} files for sample {sample_id!r}")
    return average(urls)


# --- shared sample resolution + stacking (used by raman/uv/xrd/nmr modules) -
def label_for(sample_id):
    """Build a readable label from a registry sample's compound/polymer/ratio."""
    if reg is None:
        return sample_id
    try:
        s = reg.get_sample(sample_id)
    except Exception:
        return sample_id
    comp, poly, ratio = s.get("compound", ""), s.get("polymer", ""), s.get("ratio", "")
    label = comp or sample_id
    if poly and poly != comp:
        label = f"{label}:{poly}"
    return f"{label} {ratio}".strip()


def resolve(item, technique):
    """Normalize a sample spec to (label, color, sources).

    item may be a registry sample_id (str), or a (label, color, sources) tuple
    where sources is a URL/path or a list of them. This is the same contract
    ftir.py uses, generalized to any technique.
    """
    if isinstance(item, str):
        if reg is None:
            raise RuntimeError("registry not importable; pass (label, color, sources)")
        return label_for(item), None, reg.urls_for(item, technique)
    label, color, sources = item
    return label, color, ([sources] if isinstance(sources, str) else list(sources))


def stack(samples, technique, *, offset=0.0, normalize=True, default_color="black",
          norm_range=None):
    """Load/average/normalize a list of sample specs into stacked traces.

    Returns a list of (label, color, x, y). Trace i is lifted by i*offset.
    norm_range=(lo, hi) normalizes each trace within that x-window's min/max.
    """
    traces = []
    for i, item in enumerate(samples):
        label, color, sources = resolve(item, technique)
        x, y = average(sources)
        if normalize:
            if norm_range is not None:
                lo, hi = sorted(norm_range)
                m = (x >= lo) & (x <= hi)
                y = normalize01(y, lo=y[m].min(), hi=y[m].max()) if m.any() else normalize01(y)
            else:
                y = normalize01(y)
        traces.append((label, color or default_color, x, y + i * offset))
    return traces


# --- simulation: line list -> broadened curve -------------------------------
def _lineshape(grid, center, fwhm, shape):
    shape = shape.lower()
    if shape == "gaussian":
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-(grid - center) ** 2 / (2 * sigma ** 2))
    if shape == "lorentzian":
        g = fwhm / 2.0
        return g * g / ((grid - center) ** 2 + g * g)
    if shape in ("voigt", "pseudovoigt", "pseudo-voigt"):
        return 0.5 * _lineshape(grid, center, fwhm, "gaussian") \
             + 0.5 * _lineshape(grid, center, fwhm, "lorentzian")
    raise ValueError(f"Unknown line shape {shape!r}")


def broaden(centers, intensities, grid, fwhm, shape="lorentzian"):
    """Sum intensity-weighted unit-height line shapes onto `grid`.

    Each line peaks at ~its intensity (shapes are unit-height, not unit-area), so a
    stick spectrum reads intuitively. normalize01() the result before overlaying.
    """
    grid = np.asarray(grid, float)
    y = np.zeros_like(grid)
    for c, a in zip(centers, intensities):
        y += float(a) * _lineshape(grid, float(c), fwhm, shape)
    return y


def simulate(centers, intensities, technique="FTIR", *, grid=None, fwhm=None,
             shape=None, npoints=4000, freq_scale=1.0):
    """Broaden a line list into a (grid, curve) for a technique.

    centers/intensities : the stick spectrum (e.g. ORCA modes, or a reference table).
    freq_scale          : multiply positions (apply DFT harmonic scaling for IR/Raman).
    grid/fwhm/shape     : default to the technique's TECH config if omitted.
    """
    cfg = _cfg(technique)
    fwhm = cfg["default_fwhm"] if fwhm is None else fwhm
    shape = cfg["shape"] if shape is None else shape
    centers = np.asarray(centers, float) * freq_scale
    intensities = np.asarray(intensities, float)
    if grid is None:
        lo, hi = sorted(cfg["default_range"])
        grid = np.linspace(lo, hi, npoints)
    return np.asarray(grid, float), broaden(centers, intensities, grid, fwhm, shape)


# --- overlay plot -----------------------------------------------------------
def overlay(experimental_xy, simulated_xy, technique="FTIR", *, title="",
            exp_label="experimental", sim_label="simulated",
            exp_color="black", sim_color="#d62728", sim_fill=True,
            normalize=True, xrange=None, ax=None, figsize=(12, 6),
            sticks=None, stick_color=None):
    """Overlay a simulated spectrum on an experimental trace (both normalized 0-1).

    experimental_xy / simulated_xy : (x, y) tuples (use experimental()/simulate()).
    xrange  : (lo, hi) data range to show; defaults to the technique range.
    sticks  : optional (centers, intensities) to also draw the raw stick spectrum.
    Returns (fig, ax).
    """
    cfg = _cfg(technique)
    ex, ey = experimental_xy
    sx, sy = simulated_xy
    if normalize:
        ey, sy = normalize01(ey), normalize01(sy)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(ex, ey, color=exp_color, lw=1.8, label=exp_label, zorder=3)
    if sim_fill:
        ax.fill_between(sx, sy, 0, color=sim_color, alpha=0.16, zorder=1)
    ax.plot(sx, sy, color=sim_color, lw=1.6, label=sim_label, zorder=2)
    if sticks is not None:
        c, a = sticks
        a = normalize01(a) if normalize else np.asarray(a, float)
        ax.vlines(c, 0, a, color=stick_color or sim_color, lw=1.0, alpha=0.5, zorder=2)

    lo, hi = sorted(xrange or cfg["default_range"])
    ax.set_xlim((hi, lo) if cfg["descending"] else (lo, hi))
    ax.set_xlabel(cfg["x_label"], fontsize=13)
    ax.set_ylabel(cfg["y_label"], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.margins(y=0.05)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # offline smoke test: simulate a 3-line FTIR stick spectrum and broaden it
    gx, gy = simulate([1735, 1620, 1450], [1.0, 0.6, 0.4], "FTIR", freq_scale=0.97)
    print(f"FTIR sim grid {gx[0]:.0f}->{gx[-1]:.0f} cm-1, peak {gy.max():.3f}")
    for t in ("FTIR", "RAMAN", "UV"):
        gx, gy = simulate([1000, 1500], [1, 1], t)
        print(f"{t:5s} ok: {len(gx)} pts, range {gx.min():.0f}-{gx.max():.0f}")
