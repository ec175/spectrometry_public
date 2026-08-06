"""rgbdelay.py - the colour.

The creator's own account, confirmed by measurement: the render was **black and white**, and
all of the colour comes from a glitch pass that delays the channels against each other -
blue 0 frames, green 3, red 6 (at 30 fps).

Cross-correlating the channels of the reference reels as time series:

    reel_466a0f52   R vs B peaks at +6 frames (0.716, against 0.241 at lag 0)
                    G vs B peaks at +3 (0.746)
                    R vs G peaks at +3 (0.721)
    reel_51d5e4f3   R vs B peaks at +5..6, G vs B at +2..3, R vs G at +3

Exactly the claim. This one fact explains everything the palette model could not:

  * **The bimodal warm/cool palette is an ARTEFACT, not a cause.** An element that is
    appearing shows blue first (B is undelayed) and reads cyan; one that is fading still has
    its red present after the newer channels have gone and reads orange; anything steady has
    all three channels equal and is white. Two clusters plus white - which is precisely the
    k-means result the first build spent so long trying to reproduce with a hand-built LUT.
  * **Why hue correlates the way it does across space.** Hue here is a function of how much a
    pixel CHANGED over the last 200 ms, so it inherits the illumination field's spatial
    structure automatically - a broad regional agreement with real per-element variation. No
    two-scale hue field required; that was an epicycle.
  * **Why saturation and whiteness could never be tuned together.** They are not free
    parameters at all; they follow from the rate of change.

The delays are held in SECONDS, not frames, so a 60 fps render matches a 30 fps reference:
3 frames at 30 = 0.1 s = 6 frames at 60.
"""
from __future__ import annotations

import numpy as np


class ChannelDelay:
    """Ring buffer turning a stream of MONO frames into RGB by delaying the channels.

    Feed it consecutive frames with `push`; each call returns the composed RGB frame for the
    newest input. Frames pushed before the buffer is full reuse the oldest frame available,
    so an un-primed start degrades to greyscale rather than to garbage - but callers should
    PRE-ROLL (see `render.py`) so frame 0 is already correct.
    """

    # seconds of delay per channel, as (R, G, B)
    DEFAULT = (6.0 / 30.0, 3.0 / 30.0, 0.0)

    def __init__(self, fps, delays=None, shape=None):
        r, g, b = self.DEFAULT if delays is None else delays
        self.lag = [max(0, int(round(d * fps))) for d in (r, g, b)]
        self.n = max(self.lag) + 1
        self.buf = [None] * self.n
        self.i = 0
        self.shape = shape

    def prime(self, mono):
        """Fill the whole history with one frame - used for a still, where there is no
        preceding animation to draw from."""
        for k in range(self.n):
            self.buf[k] = mono
        self.i = 0

    def push(self, mono):
        """Add the newest mono frame and return (H,W,3) float32."""
        self.buf[self.i] = mono
        out = np.empty(mono.shape + (3,), np.float32)
        for c, lag in enumerate(self.lag):
            j = (self.i - lag) % self.n
            f = self.buf[j]
            if f is None:                    # not primed yet: fall back to the newest
                f = mono
            out[..., c] = f
        self.i = (self.i + 1) % self.n
        return out
