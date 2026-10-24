# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""pf_check.py - is a `PotentialFlow.cylinder` actually a cylinder?

The exact solution has two properties that a wrongly-oriented doublet destroys, and neither is
visible in a still: the surface-normal velocity is ZERO everywhere on the body, and (at zero
circulation) the equator speed is exactly 2|U| with stagnation at the poles.

The first build of `stream_glass` failed both - it built a +x doublet for a +y stream, so the
flow ran straight through both bodies. `flux_report` caught it indirectly (52% of streamlines
"captured" inside a body that potential flow can never let them enter); this catches it
directly, in a tenth of a second, and it is the check to repeat after touching the kernel.

    python tools/pf_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from fl.field import PotentialFlow                     # noqa: E402
from fl.scenes import StreamGlass                      # noqa: E402


def check(U, a=168.0, gamma=0.0, label=""):
    pf = PotentialFlow()
    pf.U = np.array(U, dtype=float)
    c = np.array([500.0, 900.0])
    pf.cylinder(c, a, gamma)
    th = np.linspace(0, 2 * np.pi, 721, endpoint=False)
    P = c + a * np.stack([np.cos(th), np.sin(th)], axis=1)
    V = pf(P)
    nrm = np.stack([np.cos(th), np.sin(th)], axis=1)
    vn = np.abs((V * nrm).sum(axis=1))
    sp = np.hypot(V[:, 0], V[:, 1])
    u = float(np.hypot(*U))
    print(f"  {label:26s} max|v.n|/|U| = {vn.max()/u:8.5f}   "
          f"max speed/|U| = {sp.max()/u:6.3f}   min = {sp.min()/u:6.3f}")
    return vn.max() / u


print("PotentialFlow.cylinder - surface-normal velocity must be ~0")
bad = 0
for U, lab in [((0.0, 1.0), "stream +y (StreamGlass)"),
               ((1.0, 0.0), "stream +x"),
               ((0.6, -0.8), "stream oblique")]:
    if check(U, label=lab) > 1e-3:
        bad += 1
check((0.0, 1.0), gamma=235.0, label="stream +y, gamma=235")

print("\nthe deliberately WRONG version the first build shipped:")
pf = PotentialFlow()
pf.U = np.array([0.0, 1.0])
c = np.array([500.0, 900.0])
pf.doublets = [(c, 168.0 ** 2, 0.0)]           # axis for a +x stream, in a +y stream
th = np.linspace(0, 2 * np.pi, 721, endpoint=False)
P = c + 168.0 * np.stack([np.cos(th), np.sin(th)], axis=1)
V = pf(P)
vn = np.abs((V * np.stack([np.cos(th), np.sin(th)], axis=1)).sum(axis=1))
print(f"  {'doublet axis 0 in +y stream':26s} max|v.n|/|U| = {vn.max():8.5f}"
      f"   <-- flow passes straight through the body")

print("\nlive scene, sampled over the clip:")
sg = StreamGlass()
worst = 0.0
for t in np.linspace(0, sg.duration, 25):
    sg.spec(t)
    for cen, _, _ in sg.pf.doublets:
        th = np.linspace(0, 2 * np.pi, 361, endpoint=False)
        P = cen + sg.A_R * np.stack([np.cos(th), np.sin(th)], axis=1)
        V = sg.pf(P)
        n = np.stack([np.cos(th), np.sin(th)], axis=1)
        worst = max(worst, float(np.abs((V * n).sum(axis=1)).max()))
print(f"  worst |v.n| on either body over 15 s = {worst:.4f}  (|U| = 1.0)")
print("  " + ("OK - bodies are streamline surfaces" if worst < 0.06
              else "FAIL - flow is crossing a body"))
sys.exit(1 if bad else 0)
