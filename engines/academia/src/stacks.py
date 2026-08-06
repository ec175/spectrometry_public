"""stacks.py — reusable stacked-trace plots for simulated-spectra libraries.

Two plotters, both normalize each trace 0-1 and lift it by index*offset so many
samples read as a waterfall. Used by the amino-acid, cannabinoid and BCS sim tests.

  stack_plot(items, sim_for, ...)    one trace per item (e.g. one spectrum each)
  stack_overlay(items, pair_for, ...) two overlaid traces per item (A on top of B)

items   : list of (code, label); label is the y-axis tick text.
sim_for : code -> (x, y).   pair_for : code -> ((xa, ya), (xb, yb)).
Items are drawn top-to-bottom in list order (items[0] at the top).

Optional `images` (list of RGBA arrays aligned with items, None to skip one) places
a molecule thumbnail whose RIGHT edge is anchored to the left plot spine, sitting
just above that row's name (so the structure reads as "the molecule of this name").
Use a generous `offset` so there is clear empty space above each thumbnail.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import spectra as sp


def _place_images(ax, ys, images, image_zoom):
    """Anchor each thumbnail's lower-right corner to the spine at height y."""
    trans = ax.get_yaxis_transform()       # x: axes fraction (0 == left spine), y: data
    for y, img in zip(ys, images):
        if img is None:
            continue
        ab = AnnotationBbox(OffsetImage(img, zoom=image_zoom), (0.0, y),
                            xycoords=trans, frameon=False,
                            box_alignment=(1.0, 0.0),   # (right, bottom) corner at (spine, y)
                            annotation_clip=False)
        ax.add_artist(ab)


def _finish(ax, fig, yticks, ylabels, xlabel, title, xrange, descending, n, offset,
            top_pad, outfile, bases=None, images=None, image_zoom=0.09,
            image_dy=0.22, left_margin=0.24):
    lo, hi = sorted(xrange)
    ax.set_xlim((hi, lo) if descending else (lo, hi))
    ax.set_ylim(-0.4, (n - 1) * offset + top_pad)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if images is not None:
        _place_images(ax, [b + image_dy for b in bases], images, image_zoom)
        fig.subplots_adjust(left=left_margin, right=0.97, top=0.97, bottom=0.04)
    else:
        fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def stack_plot(items, sim_for, title, xlabel, xrange, descending, outfile, *,
               offset=1.7, figsize=(13, 24), color="black", trace_lw=1.0,
               images=None, image_zoom=0.09, image_dy=0.22, left_margin=0.24):
    """One simulated trace per item, stacked and labelled."""
    n = len(items)
    lo, hi = sorted(xrange)
    fig, ax = plt.subplots(figsize=figsize)
    yticks, ylabels, bases = [], [], []
    for i, (code, label) in enumerate(items):
        pos = (n - 1 - i) * offset
        gx, gy = sim_for(code)
        m = (gx >= lo) & (gx <= hi)
        ax.plot(gx[m], sp.normalize01(gy[m]) + pos, color=color, lw=trace_lw)
        yticks.append(pos)               # name at the trace's baseline (its "start")
        ylabels.append(label)
        bases.append(pos)
    _finish(ax, fig, yticks, ylabels, xlabel, title, xrange, descending, n, offset,
            1.25, outfile, bases, images, image_zoom, image_dy, left_margin)


def stack_overlay(items, pair_for, title, xlabel, xrange, descending, outfile, *,
                  offset=1.85, figsize=(13, 26), color_a="black", color_b="#d62728",
                  label_a="A", label_b="B", lw_a=1.0, lw_b=1.5, fill=True,
                  images=None, image_zoom=0.09, image_dy=0.22, left_margin=0.24):
    """Two overlaid traces per item: A (color_a, on top) over B (color_b, filled)."""
    n = len(items)
    lo, hi = sorted(xrange)
    fig, ax = plt.subplots(figsize=figsize)
    yticks, ylabels, bases = [], [], []
    for i, (code, label) in enumerate(items):
        pos = (n - 1 - i) * offset
        (xa, ya), (xb, yb) = pair_for(code)
        ma, mb = (xa >= lo) & (xa <= hi), (xb >= lo) & (xb <= hi)
        yb_n = sp.normalize01(yb[mb]) + pos
        if fill:
            ax.fill_between(xb[mb], yb_n, pos, color=color_b, alpha=0.10, zorder=2)
        ax.plot(xb[mb], yb_n, color=color_b, lw=lw_b, alpha=0.9, zorder=3)
        ax.plot(xa[ma], sp.normalize01(ya[ma]) + pos, color=color_a, lw=lw_a, zorder=4)
        yticks.append(pos)
        ylabels.append(label)
        bases.append(pos)
    ax.legend(handles=[Line2D([0], [0], color=color_a, lw=1.5, label=label_a),
                       Line2D([0], [0], color=color_b, lw=1.5, label=label_b)],
              loc="upper right", fontsize=12, framealpha=0.9)
    _finish(ax, fig, yticks, ylabels, xlabel, title, xrange, descending, n, offset,
            1.4, outfile, bases, images, image_zoom, image_dy, left_margin)
