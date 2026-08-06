"""glitch.py - the glitch is a LAYER OVER the picture, not a property of the nodes.

Ethan: "the glitch is not associated with the nodes... it seems to be placed over the video,
and has the color delays" - and then, decisively: "glitches need to be synced and be the same
size (three displaced identical glitches, all different colors corresponding to color delay)."

That last sentence is the whole design. A glitch is ONE small block whose POSITION JUMPS on
exactly the delay interval, while its size and its content stay fixed. Then the RGB delay does
the rest for free:

    blue  shows the block where it is now          (step s)
    green shows it where it was 0.1 s ago          (step s-1)
    red   shows it where it was 0.2 s ago          (step s-2)

Three identical rectangles at three positions, in blue, green and red. Nothing here draws
three copies - there is only ever one block per event per frame, and the triple is produced
entirely by the channel delay downstream. `hold` MUST therefore equal the green delay in
frames (0.1 s), or the three channels sample the same position and the effect collapses.

The earlier version re-rolled size AND position together and used blocks up to 24% of the
frame, which produced huge unrelated rectangles instead of a synced triple.

Blocks brighten what they cover rather than replacing it: a glitch momentarily ILLUMINATES
the dark lattice underneath, which is the part of the effect Ethan wanted kept. The same idea
is applied to the network itself in `network.py` (`border_flash`).
"""
from __future__ import annotations

import numpy as np


class Glitch:
    def __init__(self, fps, seed=0, green_delay=0.1, n_events=(5, 14), life=(2, 5),
                 size=(0.012, 0.055), aspect=(0.35, 2.6), jump=(0.02, 0.13),
                 gain=(0.55, 1.5), lift=0.30, active=0.85):
        # The jump interval IS the green delay - see the module docstring.
        self.hold = max(1, int(round(green_delay * fps)))
        self.seed = int(seed)
        self.n_events, self.life = n_events, life
        self.size, self.aspect, self.jump = size, aspect, jump
        self.gain, self.lift = gain, float(lift)
        self.active = float(active)

    def __call__(self, mono, frame_index):
        H, W = mono.shape
        step = int(frame_index) // self.hold
        out = None
        # An event lives `life` steps. Bucket steps into eras so an event keeps its SIZE for
        # its whole life while its POSITION is re-rolled every step.
        for era_off in (0, 1):
            era = (step - era_off * (self.life[1] // 2)) // max(self.life[1], 1)
            ergs = np.random.default_rng(self.seed * 104729 + era * 7919 + era_off)
            if ergs.random() > self.active:
                continue
            n = int(ergs.integers(self.n_events[0], self.n_events[1] + 1))
            for k in range(n):
                krng = np.random.default_rng(self.seed * 104729 + era * 7919
                                             + era_off * 31 + k * 977)
                born = era * max(self.life[1], 1) + int(
                    krng.integers(0, max(self.life[1], 1)))
                span = int(krng.integers(self.life[0], self.life[1] + 1))
                if not (born <= step < born + span):
                    continue
                # FIXED for the event's whole life
                a = float(krng.uniform(*self.aspect))
                base = float(krng.uniform(*self.size))
                bw = max(2, int(base * W * a))
                bh = max(2, int(base * H * 0.55 / max(a, 1e-3)))
                bw, bh = min(bw, W - 1), min(bh, H - 1)
                g = float(krng.uniform(*self.gain))
                x0 = int(krng.integers(0, max(1, W - bw)))
                y0 = int(krng.integers(0, max(1, H - bh)))
                # POSITION re-rolled each step: this is what the delay turns into a triple
                srng = np.random.default_rng(self.seed * 104729 + k * 977 + step * 6151)
                jx = int(srng.uniform(*self.jump) * W) * (1 if srng.random() < 0.5 else -1)
                jy = int(srng.uniform(*self.jump) * H * 0.45) * \
                    (1 if srng.random() < 0.5 else -1)
                x = int(np.clip(x0 + jx, 0, W - bw))
                y = int(np.clip(y0 + jy, 0, H - bh))
                if out is None:
                    out = mono.copy()
                blk = out[y:y + bh, x:x + bw]
                # BRIGHTEN, do not replace: the glitch lights up whatever dark lattice it
                # lands on, which is the behaviour Ethan asked to keep.
                np.maximum(blk, blk * (1.0 + g) + self.lift * g, out=blk)
        return mono if out is None else out
