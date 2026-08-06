"""pulses.py - the flux packets. The whole module is short because `lines.trace` steps in
CONSTANT ARCLENGTH.

Every field line gets one white dash launched from its seed at the same instant. Because the
polyline is uniformly sampled in arclength, a packet of age tau sits at array index
`round(speed*tau/ds)`. There is no integration here, no particle state, and no way for a packet
to drift off the line it belongs to. Do NOT "improve" this by integrating markers separately -
they desynchronise from the line under them within a few frames.

What falls out for free, and is the strongest visual hook in the source clip: at tau = 0 every
packet is at the same arclength from a common source, so the set is a PERFECT CIRCLE. As the
lines diverge the circle splays and shatters into comet dashes, and the ones heading toward a
sink get funnelled into it. One scalar (emission time) produces both a shockwave and a comet
field.

Measured off the reference clip: period 2.500 s (one 4/4 bar at 96 BPM), arclength speed
~650 px/s at 608 px wide == ~1155 px/s in the 1080-wide reference frame, dash ~14 px == ~25 px
reference. Those are the defaults.
"""
from __future__ import annotations

import numpy as np


class PulseTrain:
    def __init__(self, period=2.5, speed=1155.0, dash=26.0, lead_in=0.0, fade=0.18,
                 stagger=0.0):
        self.period = float(period)
        self.speed = float(speed)
        self.dash = float(dash)
        self.lead_in = float(lead_in)   # first emission at t = lead_in
        self.fade = float(fade)         # seconds of fade-in after emission
        # STAGGER breaks the synchrony deliberately, per line, by a fraction of the period.
        #
        # Synchrony is the whole point on a RADIAL bundle - every packet at the same arclength
        # from a common source IS a circle, and that expanding ring is the reference clip's
        # strongest hook. On a PARALLEL bundle it is a disaster: an inlet rake launches lines
        # that stay side by side, so the packets stay collinear and sweep the frame as a solid
        # horizontal bar that reads as a scan artefact, not as flow. A golden-ratio offset per
        # line scatters them into tracer particles while every packet still rides its own line
        # at exactly the same speed.
        self.stagger = float(stagger)

    def phases(self, n):
        if self.stagger <= 0.0:
            return np.zeros(n)
        return (np.arange(n) * 0.6180339887 % 1.0) * self.stagger

    def ages(self, t, s_max):
        """Ages of every packet currently in flight (a packet dies when it runs off the end of
        the longest line, so the count is set by geometry, not by a lifetime parameter)."""
        life = s_max / self.speed
        out = []
        k = int(np.floor((t - self.lead_in) / self.period))
        while k >= 0:
            tau = t - (self.lead_in + k * self.period)
            if tau > life:
                break
            if tau >= 0.0:
                out.append(tau)
            k -= 1
        return out

    def slices(self, t, length, ds, n_steps, seed_ids=None):
        """-> (idx, k0, k1, gain) arrays: for each (line, packet) pair still on its line, the
        index range to stroke white and how bright it is.

        `seed_ids` maps each traced row back to its ORIGINAL seed slot. It matters whenever the
        renderer drops dark lines before tracing: without it a line's stagger phase would change
        every time the live set changed, and the tracers would visibly jump.
        """
        s_max = ds * n_steps
        n = len(length)
        ids = np.arange(n) if seed_ids is None else np.asarray(seed_ids)
        ph = self.phases(int(ids.max()) + 1 if len(ids) else 1)[ids] if self.stagger > 0 \
            else np.zeros(n)
        dsteps = max(1, int(round(self.dash / ds)))
        life = s_max / self.speed
        nback = int(np.ceil(life / self.period)) + 2

        I, K0, K1, G = [], [], [], []
        kmax = int(np.floor((t - self.lead_in) / self.period)) + 1
        for k in range(kmax, kmax - nback - 1, -1):
            if k < 0:
                continue
            tau = t - (self.lead_in + (k + ph) * self.period)
            k0 = np.round(self.speed * tau / ds).astype(np.int64)
            hi = np.minimum(k0 + dsteps, length - 1)
            ok = (tau >= 0.0) & (tau <= life) & (k0 < n_steps) & (hi > k0)
            live = np.nonzero(ok)[0]
            if not len(live):
                continue
            g = np.ones(len(live)) if self.fade <= 0 else \
                np.clip(tau[live] / self.fade, 0.0, 1.0)
            I.append(live)
            K0.append(k0[live])
            K1.append(hi[live])
            G.append(g)
        if not I:
            e = np.zeros(0, dtype=np.int64)
            return e, e, e, np.zeros(0)
        return (np.concatenate(I), np.concatenate(K0),
                np.concatenate(K1), np.concatenate(G))
