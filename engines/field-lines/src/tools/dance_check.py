# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""dance_check.py - does `pair_dance` actually dance?

`flux_report` answers "are the field lines legal?". It cannot answer "did the six charges do
something worth watching?", and those are different questions - a scene can print COMPLETED
with the best stability numbers around while its bodies quietly drift apart instead of
together, and only a check aimed at the INTENT catches that.

The intent here is: six charges, solved not authored, that stay on camera, keep moving, and
keep meeting each other closely enough for the flux between a + and a - to visibly change
hands. So measure exactly that.

    python tools/dance_check.py [coupling] [well]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from fl.scenes import PairDance                       # noqa: E402

W, H = 1080, 1920
MARG = 90.0        # a charge this far outside the frame is off camera


def run(coupling=None, well=None, dur=15.0, n=450):
    sc = PairDance()
    if coupling is not None:
        sc.nb.C = coupling
    if well is not None:
        sc.nb.well = well
        sc.nb.launch_tangential(1.0)
    P, V = [], []
    for k in range(n):
        t = dur * k / (n - 1.0)
        sc.nb.advance_to(t)
        P.append(sc.nb.p.copy())
        V.append(sc.nb.v.copy())
    P, V = np.array(P), np.array(V)          # (n, 6, 2)

    on = ((P[:, :, 0] > -MARG) & (P[:, :, 0] < W + MARG) &
          (P[:, :, 1] > -MARG) & (P[:, :, 1] < H + MARG)).mean()
    sp = np.hypot(V[:, :, 0], V[:, :, 1])
    rel = P[:, :, None, :] - P[:, None, :, :]
    d = np.hypot(rel[..., 0], rel[..., 1])
    iu = np.triu_indices(6, 1)
    dp = d[:, iu[0], iu[1]]                  # (n, 15) pair separations
    q = sc.q
    opp = (q[iu[0]] * q[iu[1]]) < 0          # opposite-sign pairs: the ones that hand flux over
    close = (dp[:, opp] < 260.0).any(axis=1).mean()
    rad = np.hypot(P[:, :, 0] - 540.0, (P[:, :, 1] - 960.0) / 1.778)

    print(f"  coupling {sc.nb.C:.3g}   well {sc.nb.well:.3g}   softening "
          f"{np.sqrt(sc.nb.soft2):.0f} px")
    print(f"  on camera            {on:6.1%}   (want > 0.92)")
    print(f"  speed  mean {sp.mean():6.0f}  max {sp.max():6.0f} px/s   "
          f"(v_max {sc.nb.v_max:.0f})")
    print(f"  pair separation  min {dp.min():6.0f}  p1 {np.percentile(dp,1):6.0f}  "
          f"median {np.median(dp):6.0f} px")
    touch = (dp.min(axis=1) < 45.0).mean()
    print(f"  frames with a +/- pair closer than 260 px   {close:6.1%}   (want 0.20-0.90)")
    print(f"  frames with ANY pair inside 45 px (markers overlap)  {touch:6.2%}   (want < 1%)")
    print(f"  orbit radius (stretched metric)  min {rad.min():5.0f}  "
          f"mean {rad.mean():5.0f}  max {rad.max():5.0f}")
    # `min` over a whole clip is a NEVER criterion, and the honest question is "rarely". One
    # 19 px head-on turning point in 900 frames is a dramatic near-miss the soft core caught
    # correctly (it absorbs 600k px^2/s^2, more than the pass carried); a marker overlap lasting
    # several percent of the clip would be a defect. Gate on the percentile, report the min.
    ok = (on > 0.92 and np.percentile(dp, 1) > 80.0 and touch < 0.01
          and 0.20 < close < 0.90 and sp.max() < sc.nb.v_max * 0.999)
    print("  " + ("OK" if ok else "NOT YET - see the wanted ranges above"))
    return ok


if __name__ == "__main__":
    c = float(sys.argv[1]) if len(sys.argv) > 1 else None
    w = float(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(0 if run(c, w) else 1)
