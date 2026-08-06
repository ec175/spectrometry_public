"""eye.py - the IMAGE the lattice is asked to draw.

A detailed, deliberately NON-SPHERICAL eye: the outline is an almond built from two different
curves (the upper lid is higher and rounder than the lower), the iris is an ellipse that
foreshortens as the gaze moves off-axis, and the pupil sits slightly nasal of the iris centre
the way a real one does. Nothing here is a circle on a circle.

It returns THREE fields per point, because the lattice has three independent things it can do
with them, and using all three is what produces depth on a flat grid:

    lit    (N,) [0,1]   how bright this point should be
    conn   (N,) [0,1]   how strongly nodes here want to be CONNECTED. Ethan's note: bright
                        regions should be more wired-up. Iris > sclera, and the limbal ring is
                        the most connected line in the picture.
    layer  (N,) [-1,1]  a DEPTH coordinate, fed to the holofoil as a hue offset. The sclera
                        reads as the outermost surface, the iris sits below it, the pupil is a
                        hole. Because holofoil hue shifts with viewing depth in real foil, a
                        hue offset per layer is exactly the cue that sells "these are separate
                        sheets", and it costs nothing.

Anatomy that matters for the illusion:

  limbal ring   the dark/bright rim at the iris edge. The single strongest cue that the iris
                is a disc UNDER a curved cornea rather than a painted circle.
  collarette    the raised ring ~1/3 out from the pupil where iris fibres change direction.
  fibres        radial, and they must converge on the PUPIL, not on the iris centre - which
                is only visible because the pupil is offset. This is the detail that stops it
                reading as a dartboard.
  specular      one small highlight, fixed to the light and NOT to the gaze, so it slides
                across the cornea as the eye looks around. Depth cue, nearly free.
  lid shadow    the upper lid casts onto the top of the eyeball. Without it the eye looks
                pasted on rather than set into a socket.

Saccades are real saccades: the gaze HOLDS, then moves in ~60 ms. A smoothly drifting gaze
reads as a floating balloon; the hold-jump-hold rhythm is most of what makes it read as alive.
"""
from __future__ import annotations

import numpy as np

REF_W, REF_H = 1080, 1920


def _smoothstep(a, b, x):
    """Smoothstep that accepts ARRAY edges AND descending ones (b < a).

    Both matter here. The lid curves vary along x, so the edges are arrays; and the iris,
    pupil and fibre envelopes are all written as high-to-low ramps. Clamping the denominator
    with max(b - a, eps) - the obvious guard - turns every descending ramp into a hard step at
    `a` in the WRONG direction, which silently inverted the iris mask: the interior of the eye
    came out black and only a thin ring survived.
    """
    d = np.subtract(b, a)
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t = np.clip((x - a) / d, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Eye:
    def __init__(self, seed=0, cx=None, cy=None, half_w=478.0, up=292.0, dn=224.0,
                 iris_r=196.0, pupil_r=72.0, duration=15.0,
                 saccade_s=0.06, hold=(0.55, 1.9), blink_every=(2.4, 5.2), blink_s=0.16,
                 gaze_range=(0.52, 0.30)):
        self.cx = REF_W * 0.5 if cx is None else cx
        self.cy = REF_H * 0.5 if cy is None else cy
        self.half_w, self.up, self.dn = half_w, up, dn
        self.iris_r, self.pupil_r = iris_r, pupil_r
        self.saccade_s, self.blink_s = saccade_s, blink_s
        self.gaze_range = gaze_range
        rng = np.random.default_rng(seed + 771)

        # --- gaze schedule: hold, then jump ------------------------------------------------
        t, self.gz = 0.0, [(0.0, 0.0, 0.0)]
        while t < duration + 3.0:
            t += rng.uniform(*hold)
            gx = rng.uniform(-1, 1) * gaze_range[0]
            gy = rng.uniform(-1, 1) * gaze_range[1]
            # a real eye returns to centre often; pure random walk looks drunk
            if rng.random() < 0.34:
                gx, gy = gx * 0.2, gy * 0.2
            self.gz.append((t, gx, gy))
        self.gz = np.array(self.gz)

        # --- blink schedule ------------------------------------------------------------------
        t, self.bl = rng.uniform(0.8, 2.0), []
        while t < duration + 2.0:
            self.bl.append(t)
            t += rng.uniform(*blink_every)
        self.bl = np.array(self.bl) if self.bl else np.zeros(0)

        # iris fibre phases - fixed to the IRIS, so they rotate with gaze and never crawl
        self.fib_n = 34
        self.fib_ph = rng.uniform(0, 2 * np.pi, self.fib_n)
        self.fib_a = rng.uniform(0.55, 1.0, self.fib_n)
        self.fib_k = rng.integers(9, 34, self.fib_n).astype(np.float32)

    # -- animation ---------------------------------------------------------------------------
    def gaze(self, t):
        """Saccadic gaze in [-1,1]^2. Holds, then moves in `saccade_s`."""
        i = int(np.searchsorted(self.gz[:, 0], t, side="right") - 1)
        i = max(0, min(i, len(self.gz) - 2))
        t0, x0, y0 = self.gz[i]
        t1, x1, y1 = self.gz[i + 1]
        # the jump happens at the START of the next hold, not spread across it
        u = _smoothstep(t1 - self.saccade_s, t1, t)
        return float(x0 + (x1 - x0) * u), float(y0 + (y1 - y0) * u)

    def blink(self, t):
        """0 = open, 1 = shut. Fast close, slower open - as real lids do."""
        if not len(self.bl):
            return 0.0
        d = t - self.bl
        d = d[(d > -0.02) & (d < self.blink_s * 2.4)]
        if not len(d):
            return 0.0
        u = float(d[0])
        close = self.blink_s * 0.38
        if u < close:
            return float(_smoothstep(0.0, close, u))
        return float(1.0 - _smoothstep(close, self.blink_s * 2.2, u))

    # -- the fields ---------------------------------------------------------------------------
    def __call__(self, P, t):
        x = P[:, 0] - self.cx
        y = P[:, 1] - self.cy
        gx, gy = self.gaze(t)
        bk = self.blink(t)

        # --- lid aperture -----------------------------------------------------------------
        # Two DIFFERENT exponents: the upper lid is a fuller curve than the lower, which is
        # what makes an eye an almond instead of a lens. The canthi stay pinned as the lids
        # close, so a blink sweeps rather than scales.
        u = np.clip(x / self.half_w, -1.0, 1.0)
        w = np.maximum(1.0 - u * u, 0.0)
        up = self.up * w ** 0.62
        dn = self.dn * w ** 1.05
        # Blink: BOTH lids converge on a single meeting line, which sits below centre the way
        # a real closed eye does. Driving them by independent fractions (the first attempt)
        # leaves a band still open at bk = 1 - the eye never actually shuts.
        meet = self.dn * 0.26 * w ** 1.05
        up_e = up * (1.0 - bk) - meet * bk
        dn_e = dn * (1.0 - bk) + meet * bk
        inside_y = _smoothstep(-2.0, 9.0, y + up_e) * _smoothstep(-2.0, 9.0, dn_e - y)
        inside_x = _smoothstep(-1.0, 10.0, self.half_w - np.abs(x))
        aperture = inside_y * inside_x

        # --- iris: an ELLIPSE, foreshortened by the gaze ------------------------------------
        ix = gx * self.half_w * 0.40
        iy = gy * self.up * 0.42
        dx, dy = x - ix, y - iy
        # off-axis gaze compresses the iris along the gaze direction (a sphere seen obliquely)
        fx = np.sqrt(max(1.0 - 0.36 * gx * gx, 0.12))
        fy = np.sqrt(max(1.0 - 0.30 * gy * gy, 0.12))
        ex, ey = dx / fx, dy / fy
        r = np.hypot(ex, ey)
        th = np.arctan2(ey, ex)

        iris = _smoothstep(self.iris_r + 5.0, self.iris_r - 12.0, r)
        # pupil sits slightly NASAL and slightly high, as real pupils do
        pr = np.hypot(ex + 7.0, ey - 4.0)
        pupil = _smoothstep(self.pupil_r + 4.0, self.pupil_r - 7.0, pr)

        # --- iris texture: fibres converging on the PUPIL, plus the collarette -------------
        fib = np.zeros_like(r)
        for k in range(self.fib_n):
            fib += self.fib_a[k] * np.sin(self.fib_k[k] * th + self.fib_ph[k])
        fib /= np.sqrt(self.fib_n)
        # fibres fade toward the pupil and toward the limbus
        fib_env = _smoothstep(self.pupil_r, self.pupil_r + 30.0, r) * \
            _smoothstep(self.iris_r + 4.0, self.iris_r - 26.0, r)
        collar = np.exp(-((r - (self.pupil_r + self.iris_r * 0.42)) ** 2) / (2 * 15.0 ** 2))

        # --- limbal ring: the strongest single depth cue -------------------------------------
        limbal = np.exp(-((r - self.iris_r) ** 2) / (2 * 11.0 ** 2))

        # --- specular: fixed to the LIGHT, slides over the cornea as the eye turns ----------
        sx = -self.half_w * 0.20 - ix * 0.45
        sy = -self.up * 0.36 - iy * 0.45
        spec = np.exp(-(((x - sx) ** 2) / (2 * 26.0 ** 2) + ((y - sy) ** 2) / (2 * 19.0 ** 2)))

        # --- lid shadow: the socket. Dark at the top (y negative), open toward the bottom. ---
        lid_sh = _smoothstep(-up_e * 0.92, -up_e * 0.10, y)

        # --- assemble -------------------------------------------------------------------------
        sclera = np.clip(aperture - iris, 0.0, 1.0)
        iris_body = iris * (1.0 - pupil)

        # The sclera is deliberately much dimmer than the iris. Equal weighting reads as one
        # flat colour disc with a hole in it; the iris has to be the subject.
        lit = (0.17 * sclera * (0.40 + 0.60 * lid_sh)
               + iris_body * (0.46 + 0.52 * fib * fib_env + 0.30 * collar)
               + 1.55 * limbal * aperture
               + 1.30 * spec * aperture)
        lit *= aperture
        lit = np.clip(lit - 0.97 * pupil * aperture, 0.0, None)   # pupil is a HOLE

        # Connectivity carries the FIBRES too, so the iris wires up along its radial structure
        # instead of uniformly - the mesh itself draws the iris rather than merely filling it.
        conn = np.clip(0.14 * sclera
                       + iris_body * (0.62 + 0.42 * fib * fib_env)
                       + 1.0 * limbal * aperture
                       + 0.50 * collar * iris_body, 0.0, 1.0) * aperture

        # depth: sclera outermost (+), iris below (-), pupil deepest
        layer = np.clip(0.55 * sclera - 0.45 * iris_body - 1.0 * pupil * aperture, -1.0, 1.0)

        return dict(lit=lit.astype(np.float32), conn=conn.astype(np.float32),
                    layer=layer.astype(np.float32), aperture=aperture.astype(np.float32),
                    blink=bk, gaze=(gx, gy))
