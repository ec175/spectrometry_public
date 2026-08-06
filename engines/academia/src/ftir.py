"""FTIR parsing, averaging, peak-finding and plotting for the research library.

Scans are pulled through the registry (reg.urls_for(sample_id, "FTIR")), averaged,
normalized, and drawn in the lab's dual-panel broken-axis style. Vertical peak
lines are found automatically from the first derivative of the averaged trace
(the point where the derivative falls through zero, i.e. a local maximum) rather
than typed by hand.

Peak locations are refined to better than ~0.1 cm-1 by resampling onto a fine
grid before differentiating, so native 4 cm-1 spacing is not a limit.
"""
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter

try:
    import registry as reg          # optional: only needed when passing sample_ids
except Exception:
    reg = None


# ── load / average (ported from the lab Colab) ──────────────────────────────
def load_ftir_csv(url):
    """Fetch one scan CSV -> (wavenumber, value), oriented high -> low."""
    text = requests.get(url, timeout=15).text
    rows = []
    for line in text.split("\n"):
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    data = np.array(rows)
    if data[0, 0] < data[-1, 0]:
        data = data[::-1]
    return data[:, 0], data[:, 1]


def average_ftir_scans(urls):
    """Pointwise mean of several scans, interpolated onto the first scan's grid."""
    xs, ys = zip(*[load_ftir_csv(u) for u in urls])
    x = xs[0]
    y_avg = np.mean([np.interp(x, xi[::-1], yi[::-1]) for xi, yi in zip(xs, ys)], axis=0)
    return x, y_avg


# ── peak finding: derivative zero-crossing (NEW) ────────────────────────────
def find_peaks_derivative(x, y, smooth_cm=8.0, poly=3, grid_step=0.05,
                          min_prominence=0.02, normalize=True):
    """Peaks of an averaged FTIR trace via the first-derivative zero crossing.

    The trace is resampled onto a uniform `grid_step` (cm-1) grid and lightly
    smoothed (Savitzky-Golay over a `smooth_cm`-wide window). A peak is any point
    where the smoothed derivative passes from positive to <= 0 as wavenumber
    increases; its location is linearly interpolated between the bracketing grid
    points, giving sub-grid_step (<0.1 cm-1) accuracy.

    Returns a list of dicts: {"wavenumber", "height", "prominence"}, sorted by
    wavenumber. `min_prominence` (on a 0-1 normalized trace) drops noise.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)                       # ascending wavenumber
    xs, ys = x[order], y[order]

    xg = np.arange(xs[0], xs[-1], grid_step)    # uniform fine grid
    yg = np.interp(xg, xs, ys)
    if normalize:
        rng = yg.max() - yg.min()
        yg = (yg - yg.min()) / rng if rng else yg * 0.0

    win = int(round(smooth_cm / grid_step))
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(yg) - (1 - len(yg) % 2)))
    p = min(poly, win - 1)

    dy = savgol_filter(yg, win, p, deriv=1, delta=grid_step)   # smoothed 1st derivative
    ysm = savgol_filter(yg, win, p)                             # smoothed signal

    maxima, minima = [], []
    for i in range(len(xg) - 1):
        if dy[i] > 0 >= dy[i + 1]:                              # + -> - : maximum
            f = dy[i] / (dy[i] - dy[i + 1])
            maxima.append((xg[i] + f * grid_step, ysm[i] + f * (ysm[i + 1] - ysm[i])))
        elif dy[i] < 0 <= dy[i + 1]:                            # - -> + : minimum
            f = dy[i] / (dy[i] - dy[i + 1])
            minima.append((xg[i] + f * grid_step, ysm[i] + f * (ysm[i + 1] - ysm[i])))

    floor = float(yg.min())
    min_x = [m[0] for m in minima]
    min_y = [m[1] for m in minima]
    peaks = []
    for xp, yp in maxima:
        left = [my for mx, my in zip(min_x, min_y) if mx < xp]
        right = [my for mx, my in zip(min_x, min_y) if mx > xp]
        base = max(left[-1] if left else floor, right[0] if right else floor)
        prom = yp - base
        if prom >= min_prominence:
            peaks.append({"wavenumber": round(xp, 2), "height": yp, "prominence": prom})
    peaks.sort(key=lambda d: d["wavenumber"])
    return peaks


# ── registry glue ───────────────────────────────────────────────────────────
def _label_from_registry(sample_id):
    s = reg.get_sample(sample_id)
    comp, poly, ratio = s.get("compound", ""), s.get("polymer", ""), s.get("ratio", "")
    label = comp or sample_id
    if poly and poly != comp:
        label = f"{label}:{poly}"
    if ratio:
        label = f"{label} {ratio}"
    return label


def _resolve(item):
    """Accept a registry sample_id, or the lab's (label, color, urls) tuple."""
    if isinstance(item, str):
        if reg is None:
            raise RuntimeError("registry not importable; pass (label, color, urls) instead")
        return _label_from_registry(item), None, reg.urls_for(item, "FTIR")
    label, color, urls = item
    return label, color, urls


# ── vertical peak lines (fed by the auto finder) ────────────────────────────
def _cluster(values, tol):
    """Single-linkage cluster of sorted values; consecutive items within `tol`
    join the same cluster. Returns a list of lists."""
    clusters = []
    for v in sorted(values):
        if clusters and v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def add_vlines(ax, wavenumbers, merge_cm=5.0, bold_shared=True,
               min_label_gap_cm=14.0, auto_kink=True, bold_all=False,
               color="gray", linestyle="--", linewidth=1.0, alpha=1.0,
               bold_color="black", bold_linewidth=2.0,
               fontsize=12, bold_fontsize=13, label_y=0.02, rotation=90,
               label_bg="white", label_pad=2,
               kink_color="gray", kink_linewidth=0.9, kink_linestyle="--",
               kink_zone_height=0.08):
    """Draw vertical peak lines + labels for the wavenumbers visible on `ax`.

    Two declutter steps, aimed at the "peaks on all traces" union:
      1. Merge: peaks within `merge_cm` of each other collapse to ONE line at
         their mean. A merged cluster (a band shared by >1 trace) is drawn bold
         when `bold_shared` is set.
      2. Kink: remaining labels closer than `min_label_gap_cm` are nudged apart
         and tied back to their true peak with a vertical-then-diagonal leader,
         the same trick used by hand in the original Colab.

    `wavenumbers` may contain duplicates (one per trace); the multiplicity is
    what tells a shared band from a single-trace one.
    """
    xmin, xmax = sorted(ax.get_xlim())
    trans = ax.get_xaxis_transform()
    vis = [w for w in wavenumbers if xmin <= w <= xmax]
    if not vis:
        return

    # 1) merge near-coincident peaks -> (representative wavenumber, n members)
    reps = [(sum(cl) / len(cl), len(cl)) for cl in _cluster(vis, merge_cm)]
    reps.sort()
    label_x = [x for x, _ in reps]

    # 2) spread labels that would still collide, recording a kink per label
    if auto_kink and min_label_gap_cm > 0 and len(reps) > 1:
        group = [0]
        for i in range(1, len(reps)):
            if reps[i][0] - reps[i - 1][0] < min_label_gap_cm:
                group.append(i)
                if i < len(reps) - 1:
                    continue
            if len(group) > 1:
                center = sum(reps[j][0] for j in group) / len(group)
                n = len(group)
                for k, j in enumerate(group):
                    label_x[j] = center + (k - (n - 1) / 2) * min_label_gap_cm
            group = [i]

    # 3) draw lines, leaders, and labels
    for (x_rep, n), x_lab in zip(reps, label_x):
        shared = bold_all or (bold_shared and n >= 2)
        lw = bold_linewidth if shared else linewidth
        lcol = bold_color if shared else color
        fw = "bold" if shared else "normal"
        fs = bold_fontsize if shared else fontsize
        if abs(x_lab - x_rep) < 1e-6:
            ax.axvline(x_rep, color=lcol, linestyle=linestyle,
                       linewidth=lw, alpha=alpha, zorder=1)
        else:
            vb = label_y + kink_zone_height
            ax.plot([x_rep, x_rep], [1.0, vb], transform=trans, color=lcol,
                    linestyle=linestyle, linewidth=lw, alpha=alpha,
                    zorder=1, clip_on=False)
            ax.plot([x_rep, x_lab], [vb, label_y + 0.005], transform=trans,
                    color=kink_color, linestyle=kink_linestyle,
                    linewidth=kink_linewidth, alpha=alpha, zorder=2, clip_on=False)
        ax.text(x_lab, label_y, f"{x_rep:.0f}", transform=trans, rotation=rotation,
                fontsize=fs, fontweight=fw, va="bottom", ha="center",
                color="black", zorder=3,
                bbox=dict(facecolor=label_bg, edgecolor="none", pad=label_pad))


# ── plotting (lab dual-panel style) ─────────────────────────────────────────
def plot_ftir(samples, left_range=(3600, 2700), right_range=(1800, 600),
              title="", vertical_offset=1.1, vertical_scale=1.25,
              peak_ref=0, peak_kwargs=None, default_color="black",
              merge_cm=5.0, bold_shared=True, min_label_gap_cm=14.0, auto_kink=True,
              find_peaks=True, manual_peaks=None, include_ranges=None, exclude_ranges=None,
              vline_linewidth=1.0, vline_fontsize=12, vline_bold=False,
              major_tick_len=12.0, minor_tick_len=8.0, tick_label_fontsize=20):
    """Overlay/stack averaged FTIR traces and mark peaks.

    samples   : list of registry sample_ids and/or (label, color, urls) tuples.
    peak_ref  : which trace to find peaks on -- an int index, a sample label,
                or "all" to take the union across every trace.
    find_peaks : run the auto derivative peak finder (per-sample).
    manual_peaks   : {sample_id: [wavenumbers]} added regardless of find_peaks.
    include_ranges : {sample_id: [(lo, hi), ...]} keep only auto peaks inside.
    exclude_ranges : {sample_id: [(lo, hi), ...]} drop auto peaks inside.
    vline_linewidth / vline_fontsize / vline_bold : annotation styling.
    major_tick_len / minor_tick_len / tick_label_fontsize : x-axis ticks.
    Returns (fig, (ax_l, ax_r), peaks) where peaks is {label: [peak dicts]}.
    """
    peak_kwargs = peak_kwargs or {}
    manual_peaks = manual_peaks or {}
    include_ranges = include_ranges or {}
    exclude_ranges = exclude_ranges or {}
    x_lo = min(left_range[1], right_range[1])
    x_hi = max(left_range[0], right_range[0])

    traces, peaks = [], {}
    for i, item in enumerate(samples):
        label, color, urls = _resolve(item)
        key = item if isinstance(item, str) else label
        x, y_avg = average_ftir_scans(urls)
        mask = (x >= x_lo) & (x <= x_hi)
        yr = y_avg[mask]
        y_norm = (y_avg - yr.min()) / (yr.max() - yr.min()) + i * vertical_offset
        traces.append((label, color or default_color, x, y_norm * vertical_scale))

        found = find_peaks_derivative(x[mask], y_avg[mask], **peak_kwargs) if find_peaks else []
        inc = [tuple(sorted(r)) for r in include_ranges.get(key, [])]
        exc = [tuple(sorted(r)) for r in exclude_ranges.get(key, [])]
        def _keep(w):
            if inc and not any(lo <= w <= hi for lo, hi in inc):
                return False
            return not any(lo <= w <= hi for lo, hi in exc)
        kept = [p for p in found if _keep(p["wavenumber"])]
        for w in manual_peaks.get(key, []):
            kept.append({"wavenumber": float(w), "height": None, "prominence": None})
        kept.sort(key=lambda d: d["wavenumber"])
        peaks[label] = kept

    # choose which peaks to draw
    if peak_ref == "all":
        draw = [p["wavenumber"] for ps in peaks.values() for p in ps]
    else:
        ref_label = (traces[peak_ref][0] if isinstance(peak_ref, int)
                     else peak_ref)
        draw = [p["wavenumber"] for p in peaks[ref_label]]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(24, 11),
                                     gridspec_kw={"width_ratios": [1, 2]}, sharey=True)
    for label, color, x, y in traces:
        for ax, lo, hi in [(ax_l, left_range[1], left_range[0]),
                           (ax_r, right_range[1], right_range[0])]:
            m = (x >= lo) & (x <= hi)
            ax.plot(x[m], y[m], color=color, lw=1.8)
        seg = y[(x >= left_range[1]) & (x <= left_range[0])]
        if seg.size:
            ax_l.text(left_range[0] - 20, seg.min() + 0.11, label,
                      va="bottom", ha="left", fontsize=14)

    _format_axes(fig, ax_l, ax_r, traces, left_range, right_range, title,
                 major_tick_len=major_tick_len, minor_tick_len=minor_tick_len,
                 tick_label_fontsize=tick_label_fontsize)
    for ax in (ax_l, ax_r):
        add_vlines(ax, draw, merge_cm=merge_cm, bold_shared=bold_shared,
                   min_label_gap_cm=min_label_gap_cm, auto_kink=auto_kink,
                   bold_all=vline_bold, linewidth=vline_linewidth,
                   bold_linewidth=max(vline_linewidth * 2, vline_linewidth + 0.8),
                   fontsize=vline_fontsize, bold_fontsize=vline_fontsize + 1)
    return fig, (ax_l, ax_r), peaks


def _format_axes(fig, ax_l, ax_r, traces, left_range, right_range, title,
                 x_label="Wavenumber (cm$^{-1}$)", y_label="Absorbance",
                 y_pad_low=0.80, y_pad_high=0.15,
                 major_tick_len=12.0, minor_tick_len=8.0, tick_label_fontsize=20):
    ax_l.set_xlim(left_range[0], left_range[1])
    ax_r.set_xlim(right_range[0], right_range[1])
    all_y = np.concatenate([y for _, _, _, y in traces])
    ax_l.set_ylim(all_y.min() - y_pad_low, all_y.max() + y_pad_high)
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    for ax in (ax_l, ax_r):
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis="x", which="major", labelsize=tick_label_fontsize,
                       length=major_tick_len, width=1.5)
        ax.tick_params(axis="x", which="minor", length=minor_tick_len, width=1)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    fig.text(0.5, 0.02, x_label, ha="center", fontsize=28, fontweight="bold")
    ax_l.set_ylabel(y_label, fontsize=24, fontweight="bold")
    fig.suptitle(title, fontsize=28, fontweight="bold")
    plt.subplots_adjust(wspace=0.08, left=0.07, right=0.97, bottom=0.10, top=0.92)
    fig.canvas.draw()
    for ax, x_side in ((ax_l, 1.0), (ax_r, 0.0)):
        bx = ax.get_position()
        xf = bx.x0 + x_side * bx.width
        for ya in (bx.y0, bx.y1):
            fig.add_artist(plt.Line2D([xf - .004, xf + .004], [ya - .012, ya + .012],
                                      transform=fig.transFigure, color="k", lw=1.5, clip_on=False))