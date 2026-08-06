"""dsc.py — DSC parsing, smoothing, stacked plotting, and Tg/melt annotations.

Ported from the cannabinoid ASD Colab (the more capable of the lab's two DSC
codepaths): it understands the instrument's multi-'[step]' export, picks the
heating ramp, smooths it, and stacks several samples with per-trace vertical
scaling, cutoffs, and temperature annotations for Tg / melting marks.

DSC is thermal, not a spectrum, so it stands apart from spectra.py. Samples are
described by config dicts (proven flexible for figure work); URLs can also come
from registry.urls_for(sample_id, "DSC") once DSC data is registered.

    import dsc
    files = [
        {"label": "CBN 1:2", "url": URL1, "y_scale": 6, "x_min": 15, "x_max": 120,
         "cutoff_x": 120, "label_x": 127, "label_y": 0.9},
        {"label": "PVP29",   "url": URL2, "y_scale": 6, "x_min": 60, "x_max": 200},
    ]
    traces = dsc.load_dsc_traces(files, x_min=-15, x_max=200)
    offset = dsc.apply_per_trace_offset(traces, files, offset_step=0.2)
    fig, ax = plt.subplots(figsize=(10, 6))
    dsc.plot_dsc_with_cutoffs(ax, offset, x_min=-15, x_max=200)
    dsc.plot_temperature_annotations(ax, offset, [{"trace_label": "CBN 1:2", "x_temp": 94}])
    dsc.format_dsc_axes(ax, "CBN:PVP", "Temperature (°C)", "Heat Flow", -15, 200)
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator

INTERP_POINTS = 500
SAVGOL_WINDOW = 51
SAVGOL_POLYORDER = 3
HTTP_TIMEOUT = 30.0


# === parse + smooth =========================================================
def _split_into_steps(lines):
    step_indices = [i for i, ln in enumerate(lines) if ln.strip() == "[step]"]
    return [
        (s, step_indices[k + 1] if k + 1 < len(step_indices) else len(lines))
        for k, s in enumerate(step_indices)
    ]


def _extract_step_pairs(lines, start, end):
    pairs = []
    for j in range(start + 4, end):
        ln = lines[j].strip()
        if not ln or "[step]" in ln:
            continue
        try:
            parts = ln.split(",")
            pairs.append((float(parts[1]), float(parts[2])))
        except (ValueError, IndexError):
            continue
    return pairs


def _clean_and_smooth(temps, heats, x_min, x_max,
                      interp_points, savgol_window, savgol_polyorder):
    mask = (temps >= x_min) & (temps <= x_max)
    temps_f, heats_f = temps[mask], heats[mask]

    df = pd.DataFrame({"T": temps_f, "Q": heats_f})
    df = df.groupby("T", as_index=False)["Q"].mean()
    t, unique_idx = np.unique(df["T"].values, return_index=True)
    h = df["Q"].values[unique_idx]

    tg = np.linspace(t.min(), t.max(), interp_points)
    hi = interp1d(t, h, kind="cubic", fill_value="extrapolate")(tg)

    win = max(min(savgol_window, len(tg) - (1 - len(tg) % 2)), 5)
    hs = savgol_filter(hi, window_length=win, polyorder=min(savgol_polyorder, win - 1))
    return tg, hs


def _select_heating_step(step_data, keep_step):
    if keep_step == "all":
        return step_data[0][1], step_data[0][2]
    deg_idx = max(range(len(step_data)), key=lambda i: step_data[i][3])
    remaining = [s for i, s in enumerate(step_data) if i != deg_idx and s[3] >= 0]
    return remaining[0][1], remaining[0][2]


def parse_dsc_text(text, x_min, x_max, interp_points=INTERP_POINTS,
                   savgol_window=SAVGOL_WINDOW, savgol_polyorder=SAVGOL_POLYORDER,
                   keep_step="heating"):
    """Parse a multi-'[step]' DSC export -> (temperature, smoothed heat flow)."""
    lines = text.splitlines()
    step_ranges = _split_into_steps(lines)
    step_data = []
    for step_idx, (start, end) in enumerate(step_ranges):
        pairs = _extract_step_pairs(lines, start, end)
        temps = np.array([p[0] for p in pairs])
        heats = np.array([p[1] for p in pairs])
        tg, hs = _clean_and_smooth(temps, heats, x_min, x_max,
                                   interp_points, savgol_window, savgol_polyorder)
        step_data.append((step_idx, tg, hs, hs.mean()))
    return _select_heating_step(step_data, keep_step)


# === load (requests, to match the pinned venv) ==============================
def _fetch_url(url):
    import requests
    resp = requests.get(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def load_dsc_url(entry, x_min, x_max, **parse_kwargs):
    text = _fetch_url(entry["url"])
    return parse_dsc_text(text, entry.get("x_min", x_min), entry.get("x_max", x_max),
                          **parse_kwargs)


def load_dsc_traces(dsc_files, x_min, x_max, **parse_kwargs):
    """Load + parse each config entry -> list of (temperature, heat_flow, label)."""
    traces = []
    for entry in dsc_files:
        tg, hs = load_dsc_url(entry, x_min, x_max, **parse_kwargs)
        traces.append((tg, hs, entry.get("label", "?")))
    return traces


def apply_per_trace_offset(traces, dsc_files, offset_step, default_y_scale=5.0):
    """Scale each trace by its 'y_scale' and lift it by index*offset_step."""
    offset_traces = []
    for i, ((tg, hs, label), entry) in enumerate(zip(traces, dsc_files)):
        y_scale = entry.get("y_scale", default_y_scale)
        hs_scaled = (hs - hs.min()) * y_scale
        offset_traces.append((np.asarray(tg), hs_scaled + i * offset_step, label, entry))
    return offset_traces


# === plotting ===============================================================
def _trim_trace(tg, hs, cutoff_x):
    if cutoff_x is None:
        return tg, hs
    mask = tg <= cutoff_x
    return tg[mask], hs[mask]


def _resolve_label_placement(entry, tg_plot, hs_plot, default_label_x, label_y_nudge):
    cutoff_x = entry.get("cutoff_x")
    lx = entry.get("label_x", default_label_x)
    ly = entry.get("label_y", hs_plot[-1] + label_y_nudge)
    default_ha = "left" if (cutoff_x is not None and "label_x" not in entry) else "right"
    ha = entry.get("label_ha", default_ha)
    va = entry.get("label_va", "center")
    return lx, ly, ha, va


def plot_dsc_with_cutoffs(ax, offset_traces, x_min, x_max, label_x_frac=0.98,
                          label_fontsize=16, label_y_nudge=0.0, line_width=2.0,
                          line_color="k"):
    """Draw stacked DSC traces (honoring per-trace 'cutoff_x' and 'linestyle')."""
    default_label_x = x_min + label_x_frac * (x_max - x_min)
    for (tg, hs, label, entry) in offset_traces:
        tg_plot, hs_plot = _trim_trace(tg, hs, entry.get("cutoff_x"))
        ax.plot(tg_plot, hs_plot, color=line_color, linewidth=line_width,
                linestyle=entry.get("linestyle", "-"))
        lx, ly, ha, va = _resolve_label_placement(entry, tg_plot, hs_plot,
                                                   default_label_x, label_y_nudge)
        ax.text(lx, ly, label, fontsize=label_fontsize, ha=ha, va=va)


def _find_trace_by_label(offset_traces, label):
    for trace in offset_traces:
        if trace[2] == label:
            return trace


def _trace_y_center(tg, hs, entry, x_temp):
    tg_plot, hs_plot = _trim_trace(tg, hs, entry.get("cutoff_x"))
    return float(np.interp(x_temp, tg_plot, hs_plot))


def plot_temperature_annotations(ax, offset_traces, annotations, default_color="gray",
                                 default_linewidth=1.5, default_half_length=0.04,
                                 default_label_fontsize=12, default_label_offset_x=0.5,
                                 default_label_offset_y=0.02, temp_unit="°C"):
    """Mark a temperature (Tg, Tm) on a named trace with a tick + label."""
    for ann in annotations:
        found = _find_trace_by_label(offset_traces, ann["trace_label"])
        if found is None:
            continue
        tg, hs, _label, entry = found
        x_temp = ann["x_temp"]
        y_center = _trace_y_center(tg, hs, entry, x_temp)
        half_length = ann.get("half_length", default_half_length)
        y_bottom, y_top = y_center - half_length, y_center + half_length
        color = ann.get("color", default_color)
        ax.plot([x_temp, x_temp], [y_bottom, y_top], color=color,
                linewidth=ann.get("linewidth", default_linewidth))
        label_text = ann.get("label_text", f"{x_temp:g} {temp_unit}")
        ax.text(x_temp + ann.get("label_offset_x", default_label_offset_x),
                y_bottom - ann.get("label_offset_y", default_label_offset_y),
                label_text, fontsize=ann.get("label_fontsize", default_label_fontsize),
                ha=ann.get("label_ha", "left"), va=ann.get("label_va", "top"), color=color)


def format_dsc_axes(ax, plot_title, x_axis_label, y_axis_label, x_min, x_max,
                    x_axis_scale="linear", title_font_size=20, axis_label_font_size=24,
                    axis_numbers_font_size=24, hide_y_ticks=True, show_minor_x_ticks=True,
                    minor_x_tick_subdivisions=5):
    """Title, labels, limits, ticks for a DSC axis."""
    ax.set_title(plot_title, fontsize=title_font_size)
    ax.set_xlabel(x_axis_label, fontsize=axis_label_font_size)
    ax.set_ylabel(y_axis_label, fontsize=axis_label_font_size)
    ax.set_xlim(x_min, x_max)
    ax.set_xscale(x_axis_scale)
    ax.tick_params(axis="x", which="major", labelsize=axis_numbers_font_size,
                   length=7, width=1.2)
    if show_minor_x_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x_tick_subdivisions))
        ax.tick_params(axis="x", which="minor", length=4, width=1.0)
    if hide_y_ticks:
        ax.set_yticks([])
    else:
        ax.tick_params(axis="y", labelsize=axis_numbers_font_size)


if __name__ == "__main__":
    # offline smoke test of the [step] parser
    demo = "[step]\nh1\nh2\nh3\n" + "\n".join(
        f"i,{t},{np.sin(t / 20.0):.4f}" for t in range(0, 200))
    tg, hs = parse_dsc_text(demo, 0, 200, keep_step="all")
    print(f"DSC parsed {len(tg)} pts, T {tg.min():.0f}-{tg.max():.0f}, Q range {hs.ptp():.3f}")
