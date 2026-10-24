# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""lattice_check.py - does the substrate close?

The one question a render cannot answer: is the lattice actually the geometry it claims?
A honeycomb whose vertices failed to weld looks fine in a still (the edges are still drawn)
but has 6 neighbours per site instead of 3, no closed hexagons, and every `walk` degenerates.
So check the invariants numerically, before drawing anything.

    python tools\\lattice_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lg import lattice as L                              # noqa: E402
from lg.config import REF_H, REF_W, SOURCE_PITCH         # noqa: E402

EXPECT = {                     # kind -> (degree, n_link_kinds, n_edge_lengths)
    "square":     (8, 4, 2),
    "hex":        (3, 3, 1),
    "triangular": (6, 3, 1),
}


def main():
    bad = 0
    for kind, (deg_x, nk_x, nl_x) in EXPECT.items():
        lat = L.build(kind, REF_W, REF_H, SOURCE_PITCH)
        d = lat.sites[lat.edges[:, 1]] - lat.sites[lat.edges[:, 0]]
        ln = np.hypot(d[:, 0], d[:, 1])
        # Interior degree: sites away from the boundary carry the lattice's true coordination.
        s = lat.sites
        inner = ((s[:, 0] > 3 * lat.pitch) & (s[:, 0] < REF_W - 3 * lat.pitch) &
                 (s[:, 1] > 3 * lat.pitch) & (s[:, 1] < REF_H - 3 * lat.pitch))
        degs = np.diff(lat.nbr_start)[inner]
        deg = int(np.bincount(degs).argmax())
        nl = len(np.unique(np.round(ln, 3)))
        dup = len(lat.edges) - len(np.unique(np.sort(lat.edges, 1), axis=0))

        ok = (deg == deg_x and len(lat.link_specs) == nk_x and nl == nl_x and dup == 0)
        bad += not ok
        print(f"  {kind:11s} sites {lat.n_sites:6d}  edges {lat.n_edges:6d}  "
              f"deg {deg} (want {deg_x})  kinds {len(lat.link_specs)} (want {nk_x})  "
              f"lengths {nl} (want {nl_x})  dup {dup}   {'OK' if ok else '<-- FAIL'}")
        print(f"              link specs {[(round(a, 1), round(v, 2)) for a, v in lat.link_specs]}")

        # every link stamp length must match a real edge length, or the atlas draws the
        # wrong-size segment and links stop meeting their nodes.
        for a, v in lat.link_specs:
            if not np.any(np.isclose(ln, v, atol=1e-3)):
                print(f"              <-- FAIL spec length {v:.3f} matches no edge")
                bad += 1

        # A box must come back as a CLOSED ring of real edges: every site it touches must be
        # touched exactly twice, or `loop_edges` silently dropped a side and the box is a
        # broken bracket rather than an outline.
        rng = np.random.default_rng(0)
        want = {"square": None, "hex": 6, "triangular": 6}[kind]
        for _ in range(60):
            e = lat.sample_box(rng)
            if len(e) == 0:
                print("              <-- FAIL empty box")
                bad += 1
                break
            deg = np.bincount(lat.edges[e].ravel())
            if deg[deg > 0].min() != 2 or deg.max() != 2:
                print(f"              <-- FAIL box not a closed ring (degrees "
                      f"{sorted(set(deg[deg > 0].tolist()))})")
                bad += 1
                break
            if want is not None and len(e) != want:
                print(f"              <-- FAIL box has {len(e)} sides, want {want}")
                bad += 1
                break
        else:
            print(f"              boxes OK (closed rings, "
                  f"{len(lat.sample_box(np.random.default_rng(1)))} sides sampled)")

    bad += _scale_invariance()
    print("\n  ALL OK" if not bad else f"\n  {bad} FAILURES")
    return 1 if bad else 0


def _scale_invariance():
    """Is a preview actually a preview?

    The standing promise of this format is that a 540x960 preview and a 1080x1920 final are the
    SAME picture sampled differently. It was silently false here for a whole tuning pass: the
    glyph atlas is baked in pixels and was not scaled with the output, so a 540-wide frame drew
    every glyph at twice its correct relative size. `look_check` read mean_luma 58 where the
    final delivered 27, and every constant tuned against it was wrong.

    So check it numerically: render the same scene at two resolutions, downsample the larger,
    and compare bulk statistics. They cannot be identical (different sampling, and grain is
    per-pixel) but they must be close.
    """
    import numpy as np
    from lg.config import RenderConfig
    from lg.draw import Frame
    from lg.scenes import SCENES

    print("\n  scale invariance (preview must match final):")
    bad = 0
    for name in ("circuit_square", "circuit_hex"):
        sc = SCENES[name]()
        out = {}
        for w in (1080, 540):
            cfg = RenderConfig(width=w, height=w * 1920 // 1080)
            g0, cfg.grain = cfg.grain, 0.0        # grain is per-pixel; it cannot survive
            fr = Frame(cfg, sc.atlas_for(cfg))
            rgb = sc.render_frame(fr, 0.42 * sc.duration).astype(np.float32)
            cfg.grain = g0
            lum = rgb.mean(2)
            out[w] = (float(lum.mean()), float((lum < 20).mean()), float((lum > 200).mean()))
        (m1, d1, h1), (m2, d2, h2) = out[1080], out[540]
        ok = abs(m1 - m2) < 6.0 and abs(d1 - d2) < 0.09 and abs(h1 - h2) < 0.035
        bad += not ok
        print(f"    {name:15s} 1080: luma {m1:5.1f} dark {d1:.3f} hot {h1:.4f}  |  "
              f"540: luma {m2:5.1f} dark {d2:.3f} hot {h2:.4f}   {'OK' if ok else '<-- FAIL'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
