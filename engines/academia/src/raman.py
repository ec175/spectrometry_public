"""raman.py — Raman experimental loading, plotting, and DFT overlay.

Raman is FTIR's vibrational sibling: same line lists, same broadening, but the
shift axis runs low->high and there's no broken-axis convention, so traces are
drawn in a single panel. Data flows through registry.py (technique "RAMAN") or
explicit (label, color, sources) tuples, exactly like ftir.py.

No Raman scans are registered yet, so the common use today is SIMULATION:

    import raman, gaussian as g16, spectra as sp
    c, a = g16.parse_raman(NIF_raman_url)
    gx, gy = raman.simulate(c, a, freq_scale=0.967)
    # once measured data exists:
    # raman.overlay(sp.experimental("nif_xtal", "RAMAN"), (gx, gy), title="NIF")

Peak finding reuses ftir.find_peaks_derivative (technique-agnostic).
"""
import numpy as np
import matplotlib.pyplot as plt

import spectra as sp
from ftir import find_peaks_derivative, add_vlines

TECHNIQUE = "RAMAN"


def simulate(centers, intensities, *, freq_scale=1.0, **kw):
    """Broaden a Raman line list into (shift, curve). See spectra.simulate."""
    return sp.simulate(centers, intensities, TECHNIQUE, freq_scale=freq_scale, **kw)


def overlay(experimental_xy, simulated_xy, **kw):
    """Overlay a simulated Raman spectrum on a measured one. See spectra.overlay."""
    kw.setdefault("exp_label", "measured")
    return sp.overlay(experimental_xy, simulated_xy, TECHNIQUE, **kw)


def plot_raman(samples, *, xrange=(200, 3600), title="", vertical_offset=1.1,
               find_peaks=True, peak_ref=0, peak_kwargs=None, default_color="black",
               linewidth=1.6, figsize=(14, 7), add_peak_labels=True):
    """Overlay/stack averaged Raman traces and (optionally) label peaks.

    samples : registry sample_ids and/or (label, color, sources) tuples.
    peak_ref: which trace to label peaks on -- int index, label, or "all".
    Returns (fig, ax, peaks) with peaks = {label: [peak dicts]}.
    """
    peak_kwargs = peak_kwargs or {}
    lo, hi = sorted(xrange)
    traces = sp.stack(samples, TECHNIQUE, offset=vertical_offset,
                      default_color=default_color, norm_range=(lo, hi))

    peaks = {}
    fig, ax = plt.subplots(figsize=figsize)
    for label, color, x, y in traces:
        m = (x >= lo) & (x <= hi)
        ax.plot(x[m], y[m], color=color, lw=linewidth)
        seg = y[m]
        if seg.size:
            ax.text(hi, seg.min() + 0.05, f" {label}", va="bottom", ha="left", fontsize=12)
        found = find_peaks_derivative(x[m], y[m], **peak_kwargs) if find_peaks else []
        peaks[label] = found

    if add_peak_labels and find_peaks:
        if peak_ref == "all":
            draw = [p["wavenumber"] for ps in peaks.values() for p in ps]
        else:
            ref = traces[peak_ref][0] if isinstance(peak_ref, int) else peak_ref
            draw = [p["wavenumber"] for p in peaks.get(ref, [])]
        add_vlines(ax, draw)

    ax.set_xlim(lo, hi)
    ax.set_xlabel(sp.TECH[TECHNIQUE]["x_label"], fontsize=14, fontweight="bold")
    ax.set_ylabel(sp.TECH[TECHNIQUE]["y_label"], fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax, peaks


if __name__ == "__main__":
    gx, gy = simulate([1003, 1605, 2950], [1.0, 0.7, 0.5], freq_scale=0.97)
    print(f"RAMAN sim {gx.min():.0f}-{gx.max():.0f} cm-1, peak {gy.max():.3f}")
