"""xrd.py — powder XRD loading, Q-conversion, polymorph overlays, simulation.

Powder XRD is the lab's amorphous-vs-crystalline arbiter: a broad halo means
amorphous, sharp Bragg peaks mean (re)crystallized. This module:

  load_pattern   - read a 2-column .xy (2theta, intensity)
  calculate_Q    - 2theta -> Q (Cu K-alpha by default)
  plot_xrd       - stack/overlay measured patterns (2theta or Q axis)
  load_reflections / overlay_reflections - CrystalDiffract calculated stick
                   lines for polymorph identification
  simulate       - broaden a reflection list into a powder pattern

No XRD scans are registered yet, so pass explicit .xy URLs/paths as
(label, color, source) tuples (registry sample_ids work once XRD is registered).

    import xrd
    xrd.plot_xrd([("crystalline FEL", "black", FEL_xy),
                  ("amorphous FEL",  "red",   aFEL_xy)], title="FEL pXRD")
    refl = xrd.load_reflections(NIF_alpha_reflections_url)   # (2theta, intensity)
    gx, gy = xrd.simulate(*refl)                              # calc powder pattern
"""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import spectra as sp

TECHNIQUE = "XRD"
CU_KALPHA = 1.542512        # Angstrom (Cu K-alpha weighted average), per the lab Colab


# === loading ================================================================
def _read(source, *, timeout=30):
    s = str(source)
    if s.startswith(("http://", "https://")):
        import requests
        return requests.get(s, timeout=timeout).text
    return Path(source).read_text(errors="ignore")


def load_pattern(source):
    """Read a whitespace- or comma-separated 2-column .xy -> (2theta, intensity).

    Skips instrument/header lines (anything that doesn't start with two numbers).
    """
    rows = []
    for line in _read(source).splitlines():
        parts = re.split(r"[\s,]+", line.strip())
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not rows:
        raise ValueError(f"No 2-column numeric data in {source!r}")
    arr = np.asarray(rows, float)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def calculate_Q(two_theta, wavelength=CU_KALPHA):
    """Convert 2theta (degrees) to scattering vector Q = 4*pi*sin(theta)/lambda."""
    return (4 * np.pi / wavelength) * np.sin(np.deg2rad(np.asarray(two_theta, float) / 2))


def load_reflections(source, *, two_theta_col=None, intensity_col=None):
    """Read a CrystalDiffract calculated reflection list -> (2theta, intensity).

    Finds the header row (the one naming a '2-theta'/'2-Theta' column), then reads
    the whitespace columns under it. Column names are matched case-insensitively;
    override two_theta_col / intensity_col with 0-based indices if a file differs.
    """
    lines = [ln for ln in _read(source).splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines)
                       if re.search(r"2[\s_-]?theta", ln, re.I)), None)
    if header_idx is not None and two_theta_col is None:
        cols = re.split(r"\s{2,}|\t", lines[header_idx].strip()) or lines[header_idx].split()
        cols = [c.strip() for c in cols if c.strip()]
        norm = [re.sub(r"[\s_-]", "", c).lower() for c in cols]
        ti = next((j for j, c in enumerate(norm) if "2theta" in c), 0)
        ii = next((j for j, c in enumerate(norm) if "inten" in c), len(cols) - 1)
        two_theta_col, intensity_col = ti, ii
        data_start = header_idx + 1
    else:
        two_theta_col = 0 if two_theta_col is None else two_theta_col
        intensity_col = -1 if intensity_col is None else intensity_col
        data_start = 0

    tt, inten = [], []
    for ln in lines[data_start:]:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ln)
        if len(nums) > max(two_theta_col, intensity_col):
            try:
                tt.append(float(nums[two_theta_col]))
                inten.append(float(nums[intensity_col]))
            except (ValueError, IndexError):
                pass
    if not tt:
        raise ValueError(f"No reflection rows parsed from {source!r}")
    return np.asarray(tt), np.asarray(inten)


# === simulation =============================================================
def simulate(two_theta, intensity, *, fwhm=None, grid=None, npoints=4000):
    """Broaden a reflection list (2theta, intensity) into a powder pattern.

    Returns (2theta grid, intensity). Uses the XRD pseudo-Voigt default in
    spectra.TECH; pass fwhm (degrees 2theta) to widen/narrow the instrument profile.
    """
    return sp.simulate(two_theta, intensity, TECHNIQUE, grid=grid, fwhm=fwhm, npoints=npoints)


# === plotting ===============================================================
def plot_xrd(samples, *, xrange=(3, 50), use_Q=False, wavelength=CU_KALPHA,
             title="", vertical_offset=0.0, default_color="black",
             linewidth=1.5, figsize=(8, 6), normalize=True):
    """Stack/overlay measured powder patterns.

    samples : (label, color, source) tuples or registry sample_ids.
    use_Q   : plot against Q (1/Angstrom) instead of 2theta.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = sorted(xrange)
    for i, item in enumerate(samples):
        label, color, sources = sp.resolve(item, TECHNIQUE)
        x, y = (load_pattern(sources[0]) if len(sources) == 1 else sp.average(sources))
        if normalize:
            m = (x >= lo) & (x <= hi)
            y = sp.normalize01(y, lo=y[m].min(), hi=y[m].max()) if m.any() else sp.normalize01(y)
        y = y + i * vertical_offset
        xx = calculate_Q(x, wavelength) if use_Q else x
        ax.plot(xx, y, color=color or default_color, lw=linewidth, label=label)
    if use_Q:
        qlo, qhi = calculate_Q([lo, hi], wavelength)
        ax.set_xlim(qlo, qhi)
        ax.set_xlabel("Q ($\\AA^{-1}$)", fontsize=13)
    else:
        ax.set_xlim(lo, hi)
        ax.set_xlabel(sp.TECH[TECHNIQUE]["x_label"], fontsize=13)
    ax.set_ylabel(sp.TECH[TECHNIQUE]["y_label"], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if normalize or vertical_offset:
        ax.set_yticks([])
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig, ax


def overlay_reflections(ax, two_theta, intensity, *, color="green", label=None,
                        scale=None, alpha=0.85, linewidth=0.9, use_Q=False,
                        wavelength=CU_KALPHA):
    """Draw calculated reflection lines as vertical sticks for polymorph ID.

    scale : peak height in axes data units; defaults to the current y-limit so
            sticks span the plot. Call after plot_xrd to compare a measured
            pattern against a candidate polymorph's reflections.
    """
    x = calculate_Q(two_theta, wavelength) if use_Q else np.asarray(two_theta, float)
    inten = np.asarray(intensity, float)
    inten = inten / inten.max() if inten.max() else inten
    top = scale if scale is not None else ax.get_ylim()[1]
    ax.vlines(x, 0, inten * top, color=color, linewidth=linewidth, alpha=alpha,
              label=label)
    if label:
        ax.legend(fontsize=10)
    return ax


if __name__ == "__main__":
    refl_tt = np.array([8.5, 11.6, 13.2, 19.8, 24.1, 26.5])
    refl_i = np.array([100, 45, 70, 30, 55, 20.0])
    gx, gy = simulate(refl_tt, refl_i)
    print(f"XRD sim {gx.min():.0f}-{gx.max():.0f} deg, peak 2theta {gx[int(np.argmax(gy))]:.1f}")
    print("Q of 8.5,26.5 deg:", calculate_Q([8.5, 26.5]).round(3).tolist())
