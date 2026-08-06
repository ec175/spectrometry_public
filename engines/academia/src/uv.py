"""uv.py — UV-Vis loading, plotting, TD-DFT overlay, and dissolution kinetics.

Two jobs:
  1. Spectra: load/plot absorbance vs wavelength; overlay a TD-DFT stick list
     (wavelength, oscillator strength) broadened into a curve. UV-Vis broadening
     is physically Gaussian in ENERGY, so simulate() broadens in eV by default
     and maps back to nm (set in_energy=False to broaden directly in nm).
  2. Dissolution kinetics: Beer-Lambert A = eps * c * l to turn the measured
     absorbance of an ASD dissolution series into concentration vs time -- the
     bioavailability comparison the cannabinoid paper is built around.

    import uv, orca, spectra as sp
    c, f = orca.parse_uvvis("simulations/orca_outputs/cbn_tddft.out")
    gx, gy = uv.simulate(c, f)                       # energy-space Gaussian
    uv.overlay(sp.experimental("cbd_in_meoh", "UV"), (gx, gy), title="CBD")

    series = [(0, 0.05), (5, 0.42), (10, 0.71), (20, 0.93)]   # (min, A)
    uv.plot_dissolution({"CBD:PVP 1:2": series}, epsilon=1.8e3, path_length=1.0)
"""
import numpy as np
import matplotlib.pyplot as plt

import spectra as sp

TECHNIQUE = "UV"
_HC_EV_NM = 1239.841984        # E(eV) = 1239.84 / lambda(nm)


# === spectra ================================================================
def simulate(centers_nm, intensities, *, fwhm=None, fwhm_ev=0.30, in_energy=True,
             grid=None, npoints=4000):
    """Broaden a TD-DFT line list (wavelengths nm, oscillator strengths) -> (nm, curve).

    in_energy=True (default): convert centers to eV, broaden with a Gaussian of
    width fwhm_ev there, evaluate on an nm grid -- the physically correct shape.
    in_energy=False: broaden directly in nm with `fwhm` (spectra.TECH default).
    """
    centers_nm = np.asarray(centers_nm, float)
    intensities = np.asarray(intensities, float)
    cfg = sp.TECH[TECHNIQUE]
    if grid is None:
        lo, hi = sorted(cfg["default_range"])
        grid = np.linspace(lo, hi, npoints)
    grid = np.asarray(grid, float)

    if not in_energy:
        return sp.simulate(centers_nm, intensities, TECHNIQUE, grid=grid, fwhm=fwhm)

    e_grid = _HC_EV_NM / grid                       # nm -> eV (per grid point)
    e_centers = _HC_EV_NM / centers_nm
    y = sp.broaden(e_centers, intensities, e_grid, fwhm_ev, shape="gaussian")
    return grid, y


def overlay(experimental_xy, simulated_xy, **kw):
    """Overlay a simulated UV-Vis spectrum on a measured one. See spectra.overlay."""
    kw.setdefault("exp_label", "measured")
    return sp.overlay(experimental_xy, simulated_xy, TECHNIQUE, **kw)


def plot_uv(samples, *, xrange=(200, 800), title="", vertical_offset=0.0,
            default_color="black", linewidth=1.8, figsize=(10, 6), mark_lambda_max=False):
    """Overlay/stack averaged UV-Vis absorbance traces.

    samples : registry sample_ids and/or (label, color, sources) tuples.
    mark_lambda_max : annotate each trace's peak-absorbance wavelength.
    Returns (fig, ax, lambda_max) with lambda_max = {label: nm}.
    """
    lo, hi = sorted(xrange)
    traces = sp.stack(samples, TECHNIQUE, offset=vertical_offset,
                      default_color=default_color, norm_range=(lo, hi))
    fig, ax = plt.subplots(figsize=figsize)
    lambda_max = {}
    for label, color, x, y in traces:
        m = (x >= lo) & (x <= hi)
        ax.plot(x[m], y[m], color=color, lw=linewidth, label=label)
        if m.any():
            lmax = float(x[m][np.argmax(y[m])])
            lambda_max[label] = lmax
            if mark_lambda_max:
                ax.axvline(lmax, color=color, ls=":", lw=1, alpha=0.5)
                ax.text(lmax, y[m].max(), f" {lmax:.0f} nm", fontsize=10,
                        va="bottom", ha="left", color=color)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(sp.TECH[TECHNIQUE]["x_label"], fontsize=13)
    ax.set_ylabel(sp.TECH[TECHNIQUE]["y_label"], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax, lambda_max


# === dissolution kinetics (Beer-Lambert) ====================================
def concentration_from_absorbance(absorbance, epsilon, path_length=1.0):
    """Beer-Lambert: c = A / (eps * l). eps in M^-1 cm^-1, l in cm -> c in M."""
    return np.asarray(absorbance, float) / (epsilon * path_length)


def plot_dissolution(series_by_sample, *, epsilon=None, path_length=1.0,
                     time_unit="min", title="Dissolution kinetics",
                     figsize=(9, 6), markers=True):
    """Plot dissolution curves from time/absorbance series.

    series_by_sample : {label: [(time, absorbance), ...]}.
    epsilon : molar absorptivity (M^-1 cm^-1). If given, the y-axis is converted
              to concentration via Beer-Lambert; otherwise raw absorbance.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, series in series_by_sample.items():
        arr = np.asarray(series, float)
        t, a = arr[:, 0], arr[:, 1]
        y = concentration_from_absorbance(a, epsilon, path_length) if epsilon else a
        ax.plot(t, y, marker="o" if markers else None, label=label)
    ax.set_xlabel(f"Time ({time_unit})", fontsize=13)
    ax.set_ylabel("Concentration (M)" if epsilon else "Absorbance", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    gx, gy = simulate([210, 270, 310], [1.0, 0.4, 0.2])
    print(f"UV sim {gx.min():.0f}-{gx.max():.0f} nm, peak at {gx[int(np.argmax(gy))]:.0f} nm")
    c = concentration_from_absorbance([0.05, 0.42, 0.93], epsilon=1.8e3)
    print("Beer-Lambert c (M):", c.round(6).tolist())
