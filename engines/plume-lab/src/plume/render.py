# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""render.py - solve -> mirror -> per-frame LUT -> overlays -> compose -> ffmpeg. And the gate.

There is deliberately NO particle layer here. In Wind_Tunnel the streak splat is 43% of a static
frame; this look has no streaks in it at all, so the whole stage is absent rather than disabled.
Nor is there a bloom pass: the LUT was measured off an ENCODED clip, so it is already downstream
of whatever that renderer did, and blooming a post-bloom palette compounds the desaturation
(Video_Interpreter development notes, the trap that bit Lattice_Grid).

THREE FRAMINGS, ONE SOLVER
--------------------------
`cfg.frame` picks the format and nothing below it changes:

  `panel`    - the source's own shape. One horizontal stream, 1080x604, flow left to right.
  `vertical` - 1080x1920, ONE stream, nozzle at the TOP spraying DOWN. The solver is unchanged:
               it always works in (z, r) with z along the stream, and the render transposes.
               Gravity is not modelled and would be invisible here anyway - a plume at Mach 2
               crosses the frame in milliseconds, during which g moves it by microns.
  `triple`   - 1080x1920, THREE independent streams stacked with two divider bars.

WHY THE TRIPLE'S BARS CAN BE DRAWN, WHEN WIND_TUNNEL'S COULD NOT
----------------------------------------------------------------
The road set had to make its divider a REAL bounce-back plate spanning the domain, because its
two halves were regions of ONE lattice and a wake crosses a notional line as happily as anything
else. Here each stream is a **separate `AxiEuler` instance** with its own arrays, ghosts and
absorbers. They cannot physically interact - there is no shared field to leak through - so the
bars are pure composition and carry no physics. The layout is still SOLVED rather than placed
(`3*D + 2*B = height`, see `Cfg.panel_h`) for the road set's other reason: three ducts of
unequal height would confound the comparison the format exists to make.

THE DRY GATE
------------
`cli.py dry <scene>` runs the FULL solve at the scene's own resolution with no drawing and no
encode. It exists because this library's most expensive mistakes are the ones that look fine for
forty minutes: peak |u| against the thermodynamic ceiling, achieved CFL against the design CFL,
WHEN in the clip the front arrives (not just how far it got), and whether the active box grew.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np

from . import nozzle as noz
from .axi import AxiEuler
from .colormap import Mapper, build_lut
from .gpu import GPU, asnumpy, sync, xp
from .heat import HeatShield
from .sky import Sky
from .stars import StarField


def smoothstep(x):
    """C1 at BOTH ends - the rate is zero where the schedule starts and stops.

    Every ramp in this library is this shape, and the reason is always the same: a corner in a
    schedule is an impulse in its derivative, and an impulse is what a viewer reads as a jolt
    (and what a solver reads as an acoustic pulse). Here it is the attitude and the star speed,
    neither of which touches the solver - but a heading that arrives at level with its rate still
    finite reads as the cow being caught rather than settling.
    """
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def tilt_at(cfg, tc):
    """Heading above horizontal, in DEGREES, at clip time `tc` seconds.

    Two shapes. `tilt_ease = 0` is the smoothstep every other schedule in this library uses -
    symmetric, zero rate at both ends. A positive `tilt_ease` selects `(1 - u) ** ease`, which
    turns hardest at the start and eases into level; its rate at the START is not zero, and that
    is deliberate rather than overlooked - the clip OPENS there, so there is no discontinuity
    inside it, and a body that is already rotating when you meet it is what settling looks like.
    """
    if cfg.tilt_end <= 0:
        return float(cfg.tilt_deg)
    u = min(1.0, max(0.0, tc / cfg.tilt_end))
    if cfg.tilt_ease > 0:
        return float(cfg.tilt_deg) * (1.0 - u) ** float(cfg.tilt_ease)
    return float(cfg.tilt_deg) * (1.0 - smoothstep(u))


# A vibration spectrum, not a random walk. Four incommensurate lines per channel: white noise
# per frame reads as the subject TELEPORTING (and aliases differently at every frame rate),
# where a few close-but-unrelated frequencies beat against each other and read as a machine
# under load. The top line is kept under half the frame rate so 60 fps still resolves it.
_SHAKE_HZ = np.array([[6.7, 11.3, 17.9, 26.3],
                      [7.9, 13.1, 19.7, 24.1],
                      [5.3, 10.7, 15.1, 22.9]], np.float64)
_SHAKE_PH = np.array([[0.00, 1.91, 3.77, 5.12],
                      [2.44, 0.63, 4.85, 1.28],
                      [3.31, 5.60, 0.94, 2.77]], np.float64)


def shake_at(cfg, tc):
    """(dx, dy, ddeg) - the subject's vibration at clip time `tc`.

    The envelope is `(1 - u) ** shake_ease` over `shake_end` seconds, so it is violent at t = 0,
    already well down by the last couple of seconds before it ends, and exactly zero on arrival.
    At the shipped settings that reads 100% / 52% / 14% / 0% at 0 / 3.5 / 7 / 9 s - Clancy's
    "intense at the beginning, slightly dampened over 7-9, eased into 0 by 9".
    """
    if cfg.shake_end <= 0 or (cfg.shake_px <= 0 and cfg.shake_deg <= 0):
        return 0.0, 0.0, 0.0
    u = min(1.0, max(0.0, tc / cfg.shake_end))
    env = (1.0 - u) ** float(cfg.shake_ease)
    # /2.4, not /4: four unit sines sum to an RMS of 1.41 and only rarely approach their
    # arithmetic peak of 4, so dividing by the line count leaves the shake at a third of the
    # amplitude asked for. This puts the RMS near 0.6 of `shake_px` and lets the rare
    # constructive peaks overshoot it, which is what a real vibration record looks like.
    s = np.sin(2 * np.pi * _SHAKE_HZ * float(tc) + _SHAKE_PH).sum(axis=1) / 2.4
    return (env * cfg.shake_px * s[0], env * cfg.shake_px * s[1],
            env * cfg.shake_deg * s[2])


def drift_at(cfg, tc):
    """(dx, dy) - how far the pivot still is from where it belongs, at clip time `tc`."""
    if cfg.drift_end <= 0:
        return 0.0, 0.0
    u = min(1.0, max(0.0, tc / cfg.drift_end))
    k = (1.0 - u) ** float(cfg.drift_ease) if cfg.drift_ease > 0 else 1.0 - smoothstep(u)
    return float(cfg.drift_px[0]) * k, float(cfg.drift_px[1]) * k


def _edge(a, k, axis):
    """Shift by one with the EDGE value repeated - `xp.roll` with the wrap removed."""
    if axis == 0:
        return (xp.concatenate([a[:1], a[:-1]], 0) if k > 0
                else xp.concatenate([a[1:], a[-1:]], 0))
    return (xp.concatenate([a[:, :1], a[:, :-1]], 1) if k > 0
            else xp.concatenate([a[:, 1:], a[:, -1:]], 1))


def star_speed_at(cfg, tc):
    """Star speed in px/s for a multiplier-1.0 star, at clip time `tc` seconds."""
    v0, v1 = float(cfg.star_speed), float(cfg.star_speed_end)
    if v1 <= 0 or cfg.seconds <= cfg.star_accel_from:
        return v0
    return v0 + (v1 - v0) * smoothstep((tc - cfg.star_accel_from)
                                       / (cfg.seconds - cfg.star_accel_from))


def ffmpeg_bin():
    for k in ("PL_FFMPEG", "WT_FFMPEG", "OSC_FFMPEG"):
        p = os.environ.get(k)
        if p and os.path.exists(p):
            return p
    local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    return shutil.which("ffmpeg") or "ffmpeg"


def _nvenc_ok(ff):
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             timeout=25).stdout
        return "h264_nvenc" in out
    except Exception:
        return False


def encoder_args(cfg, ff):
    """NVENC, with the two flags this library learned to set explicitly.

    `-profile:v high`: NVENC emits **Main** by default, which Wind_Tunnel measured and had to
    correct for its own HQ renders. High enables the 8x8 transform, i.e. better quality per bit
    at the same `cq` - it does not raise the bitrate, it spends it better.

    `-spatial-aq`: these clips are mostly BLACK with wide, very shallow gradients - a bloom halo,
    an entrained plume edge, a star field. That is the textbook banding case, and a quantiser
    that treats a near-flat dark region the same as a busy one is what produces the rings.
    """
    if cfg.nvenc and _nvenc_ok(ff):
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-profile:v", "high",
                "-rc", "vbr", "-cq", str(cfg.crf), "-b:v", "0",
                "-spatial-aq", "1", "-aq-strength", "8", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(cfg.crf), "-pix_fmt", "yuv420p"]


def ambient_value(field, gamma):
    """The value the chosen field takes in undisturbed ambient, in solver units."""
    return {"speed": 0.0, "mach": 0.0, "density": 1.0, "pressure": 1.0 / gamma,
            "temperature": 1.0, "schlieren": 0.0}[field]


class Panel:
    """ONE stream: a solver, its painted nozzle, its colour mapping, and its own image.

    `over` overrides scene parameters for this panel only - which is how the triple format runs
    three genuinely different streams off one clip clock.
    """

    def __init__(self, cfg, over=None, sim=None):
        self.cfg = c = cfg
        o = dict(over or {})
        self.mach_e = o.get("mach_e", c.mach_e)
        self.pr_e = o.get("pr_e", c.pr_e)
        self.tr_e = o.get("tr_e", c.tr_e)
        self.gamma = o.get("gamma", c.gamma)
        self.field = o.get("field", c.field)
        self.label = o.get("label", "")

        self.vertical = (c.frame == "vertical")
        # panel pixel size, and which side the stream runs along
        self.pw = c.width
        self.ph = c.height if self.vertical else c.panel_h
        self.long_px = self.ph if self.vertical else self.pw
        self.short_px = self.pw if self.vertical else self.ph

        # A SHARED sim is how "same flow, three instruments" costs one solve instead of three.
        # Only legitimate when the panels differ solely in what they PLOT - `Scene` checks that.
        self.shared = sim is not None
        if self.shared:
            self.sim = sim
        else:
            self.sim = AxiEuler(
                c.nz, c.nr, gamma=self.gamma, cfl=c.cfl, dz=1.0, dr=1.0,
                sponge_r=c.sponge_r, sponge_z=c.sponge_z,
                active=c.active, active_start=c.active_start, fused=c.fused)
            self.sim.set_nozzle(c.r_exit_cells, self.mach_e, pr_e=self.pr_e, tr_e=self.tr_e)
            self.sim.t_ramp = c.t_ramp

        # --- painted divergent section, on the SAME grid and from the SAME exit state
        self.headc = int(round(c.head_px / c.scale))
        self.nzc = int(round(c.nozzle_px / c.scale))
        cone, wall = noz.paint(self.nzc, c.nr, c.r_exit_cells, self.mach_e, gamma=self.gamma,
                               r_throat=(None if c.r_throat_frac is None
                                         else c.r_exit_cells * c.r_throat_frac),
                               field=self.field, tr_e=self.tr_e, pr_e=self.pr_e)
        self.cone = xp.asarray(cone)
        self.cone_mask = xp.asarray(np.isnan(cone))
        self.amb = ambient_value(self.field, self.gamma)

        lut = build_lut(o.get("lut", c.lut), 256, c.lut_gamma, c.lut_clip_hi)
        # THE COLOUR RANGE IS DERIVED FROM THE PHYSICS, NOT FROM THE FRAME'S OWN PEAK.
        #
        # `auto` pins the top of the range to `vmax_mult * u_jet`, the speed of the fully
        # expanded jet. Two problems fall out at once:
        #
        #   - a per-frame range makes the SAME physical speed a different colour in different
        #     frames, which is the flicker; a derived range makes speed mean one thing for the
        #     whole clip, and makes two clips comparable to each other;
        #   - it puts the jet core at a CHOSEN place on the ramp instead of wherever the frame's
        #     max happens to leave it. With a per-frame max the core sat at ~0.77 of the range,
        #     which after `clip_hi` lands in the greens - the vertical template's first still came
        #     out cyan when the source's core is orange, because the peak was in the exit
        #     expansion and not in the core the eye reads.
        #
        # Anything faster than the cap clips to the top of the ramp, and that is wanted: it is
        # exactly the fastest gas that was inverting into dark red.
        #
        # Same discipline as Wind_Tunnel's road set ("THE PALETTE IS DERIVED, NOT PICKED"), which
        # solved for its `vmax` so a named lane landed at a named place on the ramp.
        mode, lo, hi = c.norm, c.norm_lo, c.norm_hi
        self.u_jet = noz.jet_speed(self.mach_e, self.pr_e, self.tr_e, self.gamma)
        self.auto_hi = None
        if mode == "auto":
            if self.field in ("speed", "mach"):
                hi = c.vmax_mult * self.u_jet
                if self.field == "mach":
                    # a Mach field is already normalised by the LOCAL sound speed
                    hi = hi / max(np.sqrt(self.tr_e), 1e-6)
                mode, lo = "fixed", self.amb
                self.auto_hi = hi
            else:
                mode = "stable"          # schlieren/pressure have no such natural scale
        self.mapper = Mapper(lut, mode=mode, lo=lo, hi=hi, floor=self.amb)
        self._ov = None

    def set_range_hi(self, hi):
        self.mapper.hi = float(hi)

    # ------------------------------------------------------------------ field -> image
    def _half(self):
        """(2*nr, headc+nzc+nz) scalar, mirrored about the axis, plate cells set to ambient."""
        f = self.sim.field(self.field)
        ramp = self.sim._ramp_now()
        cone = self.amb + ramp * (self.cone - self.amb)
        parts = [cone, f]
        if self.headc:
            # Undisturbed ambient AHEAD of the nozzle. Not solved and not drawn - the engine's
            # exterior is not part of the solve, exactly as for the cells outside the cone wall.
            parts.insert(0, xp.full((self.cfg.nr, self.headc), self.amb, dtype=cone.dtype))
        half = xp.concatenate(parts, axis=1)
        fld = xp.concatenate([half[::-1], half], axis=0)
        m = xp.concatenate([self.cone_mask[::-1], self.cone_mask], axis=0)
        # THE MASK MUST BE ALIGNED THE SAME WAY THE FIELD IS. It only covers the cone, so it needs
        # False for the head band on the LEFT as well as for the solved region on the right;
        # padding one side only silently paints ambient over the wrong columns.
        lead = xp.zeros((m.shape[0], self.headc), dtype=bool)
        pad = xp.zeros((m.shape[0], fld.shape[1] - m.shape[1] - self.headc), dtype=bool)
        m = xp.concatenate([lead, m, pad], axis=1)
        # Outside the nozzle wall reads as UNDISTURBED AMBIENT, not as a drawn plate - which is
        # what the source does, and is the honest value: the engine's exterior is not solved.
        return xp.where(m, self.amb, fld)

    def _resize(self, img):
        """(short_cells, long_cells) -> (short_px, long_px). Box mean when integral, else nearest."""
        h, w = img.shape[:2]
        th, tw = self.short_px, self.long_px
        if h == th and w == tw:
            return img
        if th and tw and h % th == 0 and w % tw == 0:
            return img.reshape(th, h // th, tw, w // tw).mean(axis=(1, 3))
        yy = xp.clip((xp.arange(th, dtype=xp.float32) * (h / th)).astype(xp.int32), 0, h - 1)
        xx = xp.clip((xp.arange(tw, dtype=xp.float32) * (w / tw)).astype(xp.int32), 0, w - 1)
        return img[yy][:, xx]

    def frame_rgb(self):
        """(ph, pw, 3) uint8 for this panel, already oriented."""
        fld = self._resize(self._half())
        rgb = self.mapper.rgb(fld)
        if self.vertical:
            # z runs DOWN the image. The solver never knows: it always works in (z, r).
            rgb = xp.transpose(rgb, (1, 0, 2))
        return self._overlays(rgb)

    def _overlays(self, rgb):
        """The dotted axis and the exit-plane line, both measured off the source panel."""
        c = self.cfg
        h, w = rgb.shape[:2]
        if self._ov is None:
            a_axis = np.zeros((h, w), np.float32)
            a_exit = np.zeros((h, w), np.float32)
            noz_px = int(round(c.exit_px))
            # The dash is suppressed over the HEAD BAND only, never over the cone. The source
            # draws its centreline across the painted nozzle too, so cutting it there would move
            # a measured element; the head band is new, is not solved, and a row of white dashes
            # on undisturbed ambient beside the object reads as a drawn line over nothing.
            # `head_px` is 0 everywhere else, so every earlier scene is bit-identical.
            head = int(round(c.head_px))
            if self.vertical:
                dash = (np.arange(h) % 8) < 4
                dash[:min(head, h)] = False
                if c.axis_line:
                    a_axis[:, w // 2 - 1] = dash * 0.20
                    a_axis[:, w // 2] = dash * 0.20
                if c.exit_line and noz_px > 0:
                    a_exit[min(noz_px, h - 1), :] = 0.45
            else:
                dash = (np.arange(w) % 8) < 4
                dash[:min(head, w)] = False
                if c.axis_line:
                    a_axis[h // 2 - 1] = dash * 0.20
                    a_axis[h // 2] = dash * 0.20
                if c.exit_line and noz_px > 0:
                    a_exit[:, min(noz_px, w - 1)] = 0.45
            self._ov = (xp.asarray(a_axis[..., None]), xp.asarray(a_exit[..., None]))
        a_axis, a_exit = self._ov
        f = rgb.astype(xp.float32)
        f = f + a_axis * (255.0 - f)          # lighten toward white
        f = f * (1.0 - a_exit)                # darken
        return xp.clip(f, 0, 255).astype(xp.uint8)

    # ------------------------------------------------------------------ diagnostics
    def front_z(self, frac=0.06):
        """Furthest column above `frac` of the frame's OWN peak speed - the VISIBLE front.

        The threshold is a fraction of the current peak, not an absolute floor, and that is not a
        detail: an absolute 0.02 triggers on the ACOUSTIC PRECURSOR, which outruns everything
        visible, and made `dry` and `look_check` report the front arriving at 48% and 100% of the
        same clip. Under a per-frame colour range a luma threshold IS a fraction of the frame's
        own peak, so this is the comparable definition.
        """
        col = asnumpy(self.sim.speed().max(axis=0))
        peak = float(col.max())
        if peak <= 1e-9:
            return 0
        idx = np.nonzero(col > frac * peak)[0]
        return int(idx.max()) if len(idx) else 0


class ObjectOverlay:
    """A cut-out PNG composited over the finished frame, straddling the exit plane.

    Everything expensive happens ONCE, here: the alpha crop, the rotation, the resample. A frame
    then costs one alpha blend over the object's own bounding box.

    THE PLUME IS BEHIND IT, WHICH IS THE POINT. The field is drawn first, so the jet is hidden
    from the exit plane until it clears the silhouette - the exhaust appears to leave the object
    rather than to start at a line drawn across empty space. The painted nozzle behind it is
    still painted and the solver still has it; see `Cfg.obj_png`.
    """

    def __init__(self, cfg):
        from PIL import Image
        c = self.cfg = cfg
        path = c.obj_png
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.normpath(os.path.join(root, path))
        im = Image.open(path).convert("RGBA")
        im = self._crop(im)
        if c.obj_rotate:
            # expand=True so nothing is clipped; BICUBIC then re-crop, because a rotation leaves
            # transparent corners and the fit below must be to the SUBJECT, not to its box.
            im = self._crop(im.rotate(float(c.obj_rotate), expand=True, resample=Image.BICUBIC))
        h = max(1, int(c.obj_h_px))
        w = max(1, int(round(im.width * h / im.height)))
        arr = np.asarray(im.resize((w, h), Image.LANCZOS)).astype(np.float32)
        self.h, self.w = h, w
        self.rgb = xp.asarray(np.ascontiguousarray(arr[..., :3]))
        self.a = xp.asarray(np.ascontiguousarray(arr[..., 3:4] / 255.0))
        self.a_np = arr[..., 3]
        # top-left corner in output pixels, along-stream axis first
        along = int(round(c.exit_px - c.obj_exit_frac * (h if c.frame == "vertical" else w)))
        if c.frame == "vertical":
            self.y0, self.x0 = along, (c.width - w) // 2 + int(c.obj_dx_px)
        else:
            self.y0, self.x0 = (c.panel_h - h) // 2 + int(c.obj_dx_px), along

    @staticmethod
    def _crop(im):
        a = np.asarray(im)[..., 3]
        ys, xs = np.nonzero(a > 8)
        if not len(ys):
            return im
        return im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    def draw(self, img):
        H, W = img.shape[:2]
        y0, x0 = self.y0, self.x0
        sy0, sx0 = max(0, -y0), max(0, -x0)
        y0, x0 = max(0, y0), max(0, x0)
        sy1 = min(self.h, sy0 + (H - y0))
        sx1 = min(self.w, sx0 + (W - x0))
        if sy1 <= sy0 or sx1 <= sx0:
            return img
        a = self.a[sy0:sy1, sx0:sx1]
        src = self.rgb[sy0:sy1, sx0:sx1]
        dst = img[y0:y0 + (sy1 - sy0), x0:x0 + (sx1 - sx0)].astype(xp.float32)
        img[y0:y0 + (sy1 - sy0), x0:x0 + (sx1 - sx0)] = \
            xp.clip(dst + a * (src - dst), 0, 255).astype(xp.uint8)
        return img

    def cover(self):
        """Worst alpha over the painted cone's footprint - the "is the nozzle hidden" number.

        MEASURED, not asserted, and deliberately conservative: the rectangle is the cone's full
        exit radius plus 2 px of margin, from the head band's end to just past the exit plane.
        Returns (min_alpha, mean_alpha, n_pixels_below_250).
        """
        c = self.cfg
        r = int(round(c.r_exit)) + 2
        lo, hi = int(round(c.head_px)) - 2, int(round(c.exit_px)) + 2
        if c.frame == "vertical":
            ay0, ay1, ax0, ax1 = lo, hi, c.width // 2 - r, c.width // 2 + r
        else:
            ay0, ay1, ax0, ax1 = c.panel_h // 2 - r, c.panel_h // 2 + r, lo, hi
        # into the object's own coordinates
        oy0, oy1 = ay0 - self.y0, ay1 - self.y0
        ox0, ox1 = ax0 - self.x0, ax1 - self.x0
        n = max(0, ay1 - ay0) * max(0, ax1 - ax0)
        if oy0 < 0 or ox0 < 0 or oy1 > self.h or ox1 > self.w:
            return 0.0, 0.0, n          # the rectangle is not even inside the object's box
        sub = self.a_np[oy0:oy1, ox0:ox1]
        return float(sub.min()), float(sub.mean()), int((sub < 250).sum())


class StageObject:
    """The subject in the `free` framing: rotatable about the point the nozzle is bolted to.

    The object is padded onto a square canvas with `obj_anchor` at its CENTRE, so a rotation
    about that point can never clip a corner and the anchor's position in the canvas is a
    constant. The frame then just pastes the canvas centred on the stage anchor - which is why
    the jet stays welded to the body at every heading instead of drifting across it.
    """

    def __init__(self, cfg):
        from PIL import Image
        self.cfg = c = cfg
        path = c.obj_png
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.normpath(os.path.join(root, path))
        im = ObjectOverlay._crop(Image.open(path).convert("RGBA"))
        if c.obj_flip_x:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if c.obj_rotate:
            im = ObjectOverlay._crop(
                im.rotate(float(c.obj_rotate), expand=True, resample=Image.BICUBIC))
        if c.obj_w_px:
            w = int(c.obj_w_px)
            h = max(1, int(round(im.height * w / im.width)))
        else:
            h = max(1, int(c.obj_h_px))
            w = max(1, int(round(im.width * h / im.height)))
        im = im.resize((w, h), Image.LANCZOS)
        self.ow, self.oh = w, h
        ax, ay = float(c.obj_anchor[0]) * w, float(c.obj_anchor[1]) * h
        self.ax_o, self.ay_o = ax, ay
        # square canvas, anchor dead centre, side = twice the furthest corner + slack
        rad = max(np.hypot(ax, ay), np.hypot(w - ax, ay),
                  np.hypot(ax, h - ay), np.hypot(w - ax, h - ay))
        self.s = int(np.ceil(2 * rad)) + 4
        canvas = Image.new("RGBA", (self.s, self.s), (0, 0, 0, 0))
        canvas.paste(im, (int(round(self.s / 2 - ax)), int(round(self.s / 2 - ay))))
        self.canvas = canvas
        self.alpha_o = np.asarray(im)[..., 3].astype(np.float32)
        self._cache = {}

    def at(self, deg):
        """(rgb, alpha) on the square canvas, rotated `deg` CCW about the anchor."""
        from PIL import Image
        key = round(float(deg), 2)
        hit = self._cache.get(key)
        if hit is None:
            im = self.canvas if key == 0.0 else self.canvas.rotate(
                key, resample=Image.BICUBIC, expand=False)
            arr = np.asarray(im).astype(np.float32)
            hit = (xp.asarray(np.ascontiguousarray(arr[..., :3])),
                   xp.asarray(np.ascontiguousarray(arr[..., 3:4] / 255.0)))
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def draw(self, img, deg, ax, ay):
        rgb, a = self.at(deg)
        x0, y0 = int(round(ax - self.s / 2)), int(round(ay - self.s / 2))
        H, W = img.shape[:2]
        sy0, sx0 = max(0, -y0), max(0, -x0)
        y0, x0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(self.s, sy0 + (H - y0)), min(self.s, sx0 + (W - x0))
        if sy1 <= sy0 or sx1 <= sx0:
            return img
        aa, ss = a[sy0:sy1, sx0:sx1], rgb[sy0:sy1, sx0:sx1]
        sl = (slice(y0, y0 + sy1 - sy0), slice(x0, x0 + sx1 - sx0))
        img[sl] = xp.clip(img[sl] + aa * (ss - img[sl]), 0, 255)
        return img

    def lead_profile(self, pad=30.0, step=2.0):
        """(v, forward) - the subject's LEADING SURFACE, measured off its own alpha.

        For each lateral offset `v` in pixels from the pivot, how far forward along the heading
        the silhouette reaches. This is what ties the friction sheath to the shape of the animal
        instead of to a circle on its nose: the profile bulges at the shoulder, pinches at the
        neck and stops where the body does.

        Rows with no body (above the back, below the feet) are filled from the nearest row that
        HAS one, pulled back by `pad`, so the sheath closes over the ends smoothly rather than
        falling off a cliff at the topline.
        """
        v = np.arange(-self.ay_o - pad, (self.oh - self.ay_o) + pad, step, dtype=np.float32)
        rows = np.round(self.ay_o + v).astype(int)
        fwd = np.full(v.shape, np.nan, np.float32)
        for i, r in enumerate(rows):
            if 0 <= r < self.oh:
                xs = np.nonzero(self.alpha_o[r] > 96)[0]
                if len(xs):
                    fwd[i] = float(xs.max()) - self.ax_o
        have = ~np.isnan(fwd)
        if not have.any():
            return v, np.zeros_like(v)
        idx = np.arange(len(v))
        src = idx[have][np.abs(idx[:, None] - idx[have][None, :]).argmin(1)]
        out = np.where(have, np.nan_to_num(fwd),
                       fwd[src] - np.abs(idx - src) * step * 0.9)
        return v, np.maximum(out, 0.0).astype(np.float32)

    def cover(self):
        """Worst alpha over the painted cone, in the OBJECT's own pixels.

        Angle-independent by construction: the cone and the object rotate about the same point,
        so if the cone is inside the silhouette at level it is inside it at every heading. The
        cone lies UPSTREAM of the exit, which in object space is +x (the subject faces +x and
        fires out of its back), spanning +-(r_exit + 2) across.
        """
        c = self.cfg
        px = 1.0 / c.scale          # object pixels are output pixels here
        x0 = int(round(self.ax_o)) - 2
        x1 = int(round(self.ax_o + c.nozzle_px)) + 2
        y0 = int(round(self.ay_o - c.r_exit - 2))
        y1 = int(round(self.ay_o + c.r_exit + 2)) + 1
        n = max(0, x1 - x0) * max(0, y1 - y0)
        if x0 < 0 or y0 < 0 or x1 > self.ow or y1 > self.oh:
            return 0.0, 0.0, n
        sub = self.alpha_o[y0:y1, x0:x1]
        return float(sub.min()), float(sub.mean()), int((sub < 250).sum())


class Stage:
    """`frame="free"`: the exit plane anchored anywhere, at any heading, over a background.

    WHY ROTATING THE RENDERED FIELD IS NOT A FUDGE
    ----------------------------------------------
    The solve has no gravity, a uniform isotropic ambient, and axial symmetry about the jet - so
    it has NO PREFERRED DIRECTION. Drawing it at an angle is a rigid rotation of a rotationally
    invariant problem, and the solver could not tell the difference if it were asked. Contrast
    Wind_Tunnel, where the same move would be illegal: an inclined body in a duct with real side
    walls is a different problem from a level one, which is why that project rotates the MODEL
    and never the picture. Here there is no duct and no wall.

    THE ONE APPROXIMATION, AND ITS SIZE
    -----------------------------------
    The plume rotates RIGIDLY with the object. Physically the gas at the far end of the plume
    left the nozzle a transit time ago and should still be lying along the OLD heading, so a
    turning rocket trails a bent plume. Measured on this rig: the visible plume is ~750 cells and
    the jet runs at u_jet, so a transit is ~1.4 s of clip, and the peak turn rate is
    `tilt_deg / tilt_end * 1.5` (smoothstep's peak is 1.5x its mean) = 3.75 deg/s at 15 deg over
    6 s. **The tail therefore lags by up to ~5 deg and this build does not show it.** Fixing it
    means sampling with a z-dependent heading, which has no closed-form inverse - one fixed-point
    iteration would do it, and it is deliberately not in v1.

    COMPOSITING IS ADDITIVE, AND AN OVER-COMPOSITE WAS WRONG
    --------------------------------------------------------
    The first cut used the field's own value as an OPACITY. It looked right in principle - halo
    translucent, core opaque - and it produced a large **dark navy lens** around the plume where
    the faint halo was blending the stars toward `plumefire`'s near-black low end. A halo that
    puts out almost no light was subtracting light, which is the one thing an exhaust cannot do.

    So the field ADDS, and its OPACITY is a separate, much steeper function of the same value:

        img = img * (1 - t**occlude_gamma) + rgb * t**alpha_gamma

    Emission is nearly linear in the field (`alpha_gamma` 0.85), so the halo glows. Occlusion is
    cubic (`occlude_gamma` 3.0), so the halo hides **0.3%** of what is behind it and the core
    hides **73%** - which is what actually reads as a solid exhaust with a glow around it. One
    exponent cannot do both: at 0.85 the stars streamed straight through the jet core, and at 3.0
    alone the halo disappears. `stage_feather_px` still takes the domain's own boundary smoothly
    to nothing, because the sponge leaves a small but non-zero disturbance right up to the edge.
    """

    def __init__(self, cfg, panel):
        self.cfg, self.p = cfg, panel
        c = cfg
        self.ax = float(c.anchor[0]) * c.width
        self.ay = float(c.anchor[1]) * c.height
        Y, X = np.meshgrid(np.arange(c.height, dtype=np.float32),
                           np.arange(c.width, dtype=np.float32), indexing="ij")
        self.dx = xp.asarray(X - self.ax)
        self.dy = xp.asarray(Y - self.ay)
        self.nc = c.nz + int(round(c.nozzle_px / c.scale))     # columns in the half image
        self.stars = (StarField(c.width, c.height, c.stars, c.star_layers, c.star_seed)
                      if c.stars else None)
        self.obj = StageObject(c) if c.obj_png else None
        # The sheath is shaped by the SUBJECT, so it cannot exist without one.
        self.heat = (HeatShield(c.heat, self.obj.lead_profile(), c.heat_ahead_px, c.heat_len_px,
                                c.heat_spread, c.heat_thick_px, c.heat_seed, c.heat_rate,
                                c.heat_gain, c.heat_writhe)
                     if (c.heat and self.obj is not None) else None)
        self._feather = None

        # ---- ALTITUDE, integrated from the subject's own motion.
        # `dh/dt = v * sin(theta)`, i.e. the vertical component of the velocity the star field is
        # already using, with the heading the subject is already pitching through - a gravity
        # turn. The vehicle climbs hard while vertical and stops climbing as it lays over, so the
        # background changes because of what it is DOING. `sky_alt_floor` is the residual climb a
        # real vehicle keeps after pitchover; without it the sky freezes for the last three
        # seconds. Only the total is normalised; the shape is the trajectory's.
        self.sky = self.alt = None
        if c.sky:
            tcs = np.arange(c.frames, dtype=np.float64) / c.fps
            th = np.radians([tilt_at(c, t) for t in tcs])
            v = np.array([star_speed_at(c, t) for t in tcs])
            fl = float(c.sky_alt_floor)
            vz = v * (np.sin(th) * (1.0 - fl) + fl)
            cum = np.cumsum(vz) / c.fps
            self.alt = c.sky_alt_max * cum / max(cum[-1], 1e-9)
            # cloud drift shares the same velocity: sideways with cos, and DOWN the screen with
            # sin, because climbing past a cloud deck pushes it down the frame.
            self.cdx = np.cumsum(v * np.cos(th)) / c.fps * c.sky_cloud_par
            self.cdy = np.cumsum(v * np.sin(th)) / c.fps * c.sky_cloud_par
            self.sky = Sky(c.width, c.height, c.sky_seed, c.sky_clouds, c.sky_cloud_km,
                           c.sky_scale_km, c.sky_limb_km, c.sky_horizon_km)

    def alt_at(self, tc):
        if self.alt is None:
            return 0.0
        i = int(np.clip(round(tc * self.cfg.fps), 0, len(self.alt) - 1))
        return float(self.alt[i])

    # ------------------------------------------------------------------ the affine gather
    def _sample(self, deg, sx=0.0, sy=0.0):
        """(height, width) scalar gathered from the mirrored half image, + a validity mask."""
        c = self.cfg
        th = np.radians(float(deg))
        ct, st = float(np.cos(th)), float(np.sin(th))
        # The subject travels along (+cos, -sin) in image coordinates (y is DOWN, so a positive
        # heading is up the screen). The jet fires the other way; `p` completes the pair.
        dx, dy = self.dx - sx, self.dy - sy
        z = -dx * ct + dy * st
        r = dx * st + dy * ct
        col = (z + c.nozzle_px) / c.scale
        row = r / c.scale + c.nr
        ci = xp.clip(col.astype(xp.int32), 0, self.nc - 1)
        ri = xp.clip(row.astype(xp.int32), 0, 2 * c.nr - 1)
        inside = ((col >= 0) & (col < self.nc - 1) & (row >= 0) & (row < 2 * c.nr - 1))
        # feather: 1 in the interior, falling to 0 over `stage_feather_px` at every boundary
        f = float(max(1, c.stage_feather_px)) / c.scale
        w = xp.minimum(xp.minimum(col, self.nc - 1 - col),
                       xp.minimum(row, 2 * c.nr - 1 - row)) / f
        return ci, ri, xp.clip(w, 0.0, 1.0) * inside

    def _bloom(self, img, src):
        """Two-scale glow around `src`, added to `img`. All on the GPU.

        `src` IS THE PLUME'S OWN EMISSION, not the finished frame, and that distinction only
        became visible once there was a daylit sky behind it: thresholding the composite blooms
        anything bright, and a blue sky near the horizon is brighter than most of the plume, so
        the whole frame hazed over. Glow belongs to what is emitting - the flame - not to what is
        merely lit. On a black background the two are the same thing, which is why the first
        version got away with it.

        Blurring at full resolution would be the obvious thing and is the expensive thing; a
        wide gaussian is 100+ taps over 2 megapixels. Blurring an 8x-downsampled copy costs
        1/64th of that and, once it is blurred, the block structure of the upsample is far below
        the amplitude anything is added at. The second, 2x scale carries the tight core halo,
        which is where blockiness WOULD show.
        """
        c = self.cfg
        H, W = img.shape[:2]
        lum = src.max(axis=2)
        m = xp.clip(lum - float(c.bloom_thr), 0.0, None) / max(255.0 - c.bloom_thr, 1.0)
        bright = src * m[..., None]
        out = img
        for f, g in ((8, 1.0), (2, float(c.bloom_tight))):
            if g <= 0 or H % f or W % f:
                continue
            s = bright.reshape(H // f, f, W // f, f, 3).mean(axis=(1, 3))
            # CLAMPED, not wrapped. `xp.roll` was the obvious way to write this and it is a real
            # bug: the plume runs off the LEFT edge, so a wrapping blur painted its glow down the
            # RIGHT edge of the frame as a faint vertical band with nothing there to make it.
            for _ in range(3):                      # box passes -> near-gaussian
                s = (s + _edge(s, 1, 0) + _edge(s, -1, 0)) / 3.0
                s = (s + _edge(s, 1, 1) + _edge(s, -1, 1)) / 3.0
            out = out + xp.repeat(xp.repeat(s, f, axis=0), f, axis=1) * (float(c.bloom) * g)
        return xp.clip(out, 0, 255)

    def frame_rgb(self, tc):
        c = self.cfg
        deg = tilt_at(c, tc)
        sx, sy, sdeg = shake_at(c, tc)
        dx0, dy0 = drift_at(c, tc)
        sx, sy = sx + dx0, sy + dy0
        deg += sdeg
        ax, ay = self.ax + sx, self.ay + sy
        ci, ri, w = self._sample(deg, sx, sy)

        # THE PLUME'S ALTITUDE RESPONSE. Same solved field, more or less of it luminous - see
        # Cfg.plume_vmax_lo. `atm` is 1 in thick air and 0 in vacuum.
        alt = self.alt_at(tc)
        a_gam = float(c.stage_alpha_gamma)
        if c.plume_vmax_lo > 0 or c.plume_alpha_lo > 0:
            atm = float(np.exp(-alt / max(c.plume_atm_km, 1e-6)))
            if c.plume_vmax_lo > 0 and self.p.auto_hi is not None:
                vm = c.vmax_mult + (c.plume_vmax_lo - c.vmax_mult) * atm
                self.p.mapper.hi = vm * self.p.u_jet
            if c.plume_alpha_lo > 0:
                a_gam = a_gam + (c.plume_alpha_lo - a_gam) * atm

        fld = self.p._half()[ri, ci]
        t = self.p.mapper.normalise(fld)
        idx = xp.clip((t * (self.p.mapper.n - 1) + 0.5).astype(xp.int32), 0,
                      self.p.mapper.n - 1)
        rgb = self.p.mapper.lut[idx].astype(xp.float32)
        w3 = w[..., None]
        a = xp.clip(t ** a_gam, 0.0, 1.0)[..., None] * w3
        occ = xp.clip(t ** float(c.stage_occlude_gamma), 0.0, 1.0)[..., None] * w3

        i = int(np.clip(round(tc * c.fps), 0, c.frames - 1))
        if self.sky is not None:
            img = self.sky.render(alt, (self.cdx[i], self.cdy[i]))
        else:
            img = xp.zeros((c.height, c.width, 3), xp.float32)

        if self.stars is not None:
            v = star_speed_at(c, tc) / float(c.fps)
            th = np.radians(deg)
            # Stars stream OPPOSITE the heading, i.e. along the exhaust. Both turn together -
            # a star field that kept its direction while the subject pitched would be the one
            # thing that gives the whole shot away.
            dxs, dys = -v * float(np.cos(th)), v * float(np.sin(th))
            s = xp.asarray(self.stars.render(dxs, dys, c.star_streak, c.star_gain))
            # Stars arrive because the SKY LEFT, not on a cue of their own - and they sit BEHIND
            # everything the sky draws, the planet most of all.
            if self.sky is not None:
                s = s * (self.sky.star_fade(alt) * (1.0 - self.sky.block)[..., None])
            img = img + s
            self.stars.advance(dxs, dys)

        emis = a * rgb
        img = xp.clip(img * (1.0 - occ) + emis, 0, 255)
        if self.obj is not None:
            img = self.obj.draw(img, deg, ax, ay)
        if self.heat is not None:
            # AFTER the body and ADDITIVE, so the filaments veil it instead of being hidden
            # behind it. Drawn underneath, the sheath reads as a separate object in front of the
            # animal; drawn over it, the leading surface glows and the trail streams back across
            # its flank, which is what the reference shows.
            img = xp.clip(img + self._warp_heat(deg, tc, ax, ay), 0, 255)
        if c.bloom > 0:
            img = self._bloom(img, emis)
        return img.astype(xp.uint8)

    def _warp_heat(self, deg, tc, ax, ay):
        """The local glow buffer, rotated into the frame by the same gather the field uses."""
        h = self.heat
        (ox, oy), back, across = h.place(deg, ax, ay)
        dx, dy = self.dx - (ox - self.ax), self.dy - (oy - self.ay)
        lx = dx * back[0] + dy * back[1] + h.fwd
        ly = dx * across[0] + dy * across[1] + h.half
        # Feathered, not just masked. The sheath's grain and the filament tails are small but not
        # zero at the buffer's rim, and a hard cut there would draw the buffer's own RECTANGLE
        # across the sky - the same failure the field's `stage_feather_px` exists to prevent.
        ok = (xp.clip(xp.minimum(xp.minimum(lx, h.wl - 1 - lx),
                                 xp.minimum(ly, h.hl - 1 - ly)) / 28.0, 0.0, 1.0))
        xi = xp.clip(lx.astype(xp.int32), 0, h.wl - 1)
        yi = xp.clip(ly.astype(xp.int32), 0, h.hl - 1)
        loc = xp.asarray(h.local(tc)) * 255.0
        return loc[yi, xi] * ok[..., None]

    def cover_report(self):
        if self.obj is None:
            return True
        mn, mean, bad = self.obj.cover()
        ok = mn >= 250.0
        print(f"  object: {self.obj.ow}x{self.obj.oh} px, anchor {self.cfg.obj_anchor} on a "
              f"{self.obj.s} px turntable at ({self.ax:.0f}, {self.ay:.0f})\n"
              f"  cone cover: min alpha {mn:.0f}/255, mean {mean:.1f}, {bad} px under 250"
              f"  -> {'HIDDEN' if ok else 'NOZZLE VISIBLE'}")
        return ok


class Scene:
    """One clip: one panel (`panel` / `vertical`) or three (`triple`), on ONE clock."""

    # parameters that change the FLOW, as opposed to how it is drawn
    _PHYS = ("mach_e", "pr_e", "tr_e", "gamma")

    def __init__(self, cfg):
        self.cfg = cfg
        specs = cfg.variants if (cfg.frame == "triple" and cfg.variants) else (None,)
        if cfg.frame == "triple" and len(specs) != 3:
            raise SystemExit("frame='triple' needs exactly 3 entries in cfg.variants")

        # --- SHARED SOLVER, when the panels differ only in what they PLOT.
        # Checked, not trusted: if any panel overrides a physics parameter, sharing would be a
        # silent lie - three panels labelled as different gases all showing the first one.
        share = bool(cfg.share_sim) and cfg.frame == "triple"
        if share:
            bad = [k for s in specs for k in (s or {}) if k in self._PHYS]
            if bad:
                raise SystemExit(
                    f"share_sim=True but variants override physics {sorted(set(bad))} - "
                    "a shared solve can only differ in field/lut/label")
        self.panels = []
        for s in specs:
            reuse = self.panels[0].sim if (share and self.panels) else None
            self.panels.append(Panel(cfg, s, sim=reuse))

        # --- SHARED COLOUR RANGE across panels plotting the SAME field.
        #
        # THE TRAP THIS EXISTS FOR: `norm="auto"` pins each panel's range to ITS OWN jet speed.
        # In a single clip that is exactly right. In a COMPARISON it is a disaster - a panel
        # twice as fast as its neighbour gets scaled by twice the number and the two render
        # IDENTICALLY, so the format normalises away the very difference it was built to show.
        # Grouped by field because a Mach number and a speed have no common scale.
        if cfg.frame == "triple" and cfg.share_range:
            groups = {}
            for p in self.panels:
                if p.auto_hi is not None:
                    groups.setdefault(p.field, []).append(p)
            for ps in groups.values():
                hi = max(p.auto_hi for p in ps)
                for p in ps:
                    p.set_range_hi(hi)

        self.stage = Stage(cfg, self.panels[0]) if cfg.frame == "free" else None
        self.obj = (ObjectOverlay(cfg) if (self.stage is None and cfg.obj_png and cfg.obj_h_px)
                    else None)
        self.settled = False
        self.bar = None
        if cfg.frame == "triple":
            self.bar = xp.asarray(
                np.tile(np.array(cfg.bar_rgb, np.uint8), (cfg.bar_px, cfg.width, 1)))

    # ------------------------------------------------------------------ per-frame
    def advance_to(self, t):
        seen = set()
        for p in self.panels:
            if id(p.sim) in seen:      # a shared solver is stepped ONCE, not once per panel
                continue
            seen.add(id(p.sim))
            p.sim.advance_to(t)

    def health(self):
        return max(p.sim.health() for p in self.panels)

    def over_ceiling(self):
        return [p for p in self.panels
                if not np.isfinite(p.sim.health()) or p.sim.health() > p.sim.health_limit]

    def t_of(self, i):
        """Solver time at output frame `i` (0-based). The clip starts AFTER the settle."""
        c = self.cfg
        return c.t_settle + (i + 1) * c.t_end / c.frames

    def settle(self, verbose=True):
        """Run the pre-roll once, so frame 0 opens on a developed jet rather than on a bang."""
        c = self.cfg
        if self.settled or c.t_settle <= 0:
            self.settled = True
            return
        t0 = time.time()
        self.advance_to(c.t_settle)
        sync()
        self.settled = True
        if verbose:
            print(f"  settled to t={c.t_settle:.0f} in {time.time() - t0:.0f}s "
                  f"({self.panels[0].sim.steps_done} steps, peak |u| {self.health():.2f})",
                  flush=True)

    def frame_rgb(self, tc=0.0):
        if self.stage is not None:
            return self.stage.frame_rgb(tc)
        imgs = [p.frame_rgb() for p in self.panels]
        if self.bar is None:
            img = imgs[0]
        else:
            out = [imgs[0]]
            for im in imgs[1:]:
                out.extend([self.bar, im])
            img = xp.concatenate(out, axis=0)
            # solved layout can leave a row or two over; pad to the exact output height
            short = self.cfg.height - img.shape[0]
            if short > 0:
                img = xp.concatenate([img, xp.repeat(self.bar[:1], short, axis=0)], axis=0)
            img = img[:self.cfg.height]
        if self.obj is not None:
            img = self.obj.draw(img)
        return img

    def cover_report(self):
        """Print the measured occlusion of the painted nozzle, or say there is no object."""
        if self.stage is not None:
            return self.stage.cover_report()
        if self.obj is None:
            return True
        mn, mean, bad = self.obj.cover()
        ok = mn >= 250.0
        print(f"  object: {self.obj.w}x{self.obj.h} px at ({self.obj.x0}, {self.obj.y0})"
              f"  cone cover: min alpha {mn:.0f}/255, mean {mean:.1f}, "
              f"{bad} px under 250  -> {'HIDDEN' if ok else 'NOZZLE VISIBLE'}")
        return ok

    # ------------------------------------------------------------------ gates / output
    def dry(self, verbose=True):
        c = self.cfg
        if verbose:
            print(c.summary())
            for i, p in enumerate(self.panels):
                tag = f"  [{i}] " if len(self.panels) > 1 else "  "
                print(tag + p.sim.describe().replace("\n", "\n  " + " " * len(tag)))
            self.cover_report()
        t0 = time.time()
        self.settle(verbose)
        peaks = [0.0] * len(self.panels)
        peak_cfl = 0.0
        bad = None
        fronts = []
        for i in range(c.frames):
            self.advance_to(self.t_of(i))
            hs = [p.sim.health() for p in self.panels]
            peaks = [max(a, b) for a, b in zip(peaks, hs)]
            h = max(hs)
            peak_cfl = max(peak_cfl, max(p.sim.cfl_now() for p in self.panels))
            fronts.append([p.front_z() for p in self.panels])
            if self.over_ceiling():
                bad = (i, h)
                break
        sync()
        el = time.time() - t0
        n = len(fronts)
        # PER PANEL. Printing max-over-panels |u| against min-over-panels ceiling compares two
        # different panels and reads as a breach when there is none - `triple_temp` showed 7.957
        # against 7.770 while every panel was inside its own limit, because the peak was the hot
        # panel's (ceiling 19.0) and the ceiling was the cold panel's. `over_ceiling()` always
        # checked per panel, so the GUARD was right and only this line was wrong; that is the
        # same class of mistake as every other detector bug in this project.
        print(f"\n  steps={self.panels[0].sim.steps_done}"
              f"  ({self.panels[0].sim.steps_done / max(n,1):.1f}/frame)"
              f"  dt_final={self.panels[0].sim.dt:.4f}")
        for k, p in enumerate(self.panels):
            tag = f"panel {k}" if len(self.panels) > 1 else "peak"
            h = peaks[k]
            print(f"  {tag} peak |u| = {h:.3f}  (unsteady ceiling {p.sim.health_limit:.3f}; "
                  f"steady vacuum limit {p.sim.steady_limit:.3f}"
                  f"{' - EXCEEDED, expected in a starting jet' if h > p.sim.steady_limit else ''})")
        print(f"  peak CFL = {peak_cfl:.3f}  (design {c.cfl})")
        print(f"  active box: {self.panels[0].sim.nz_act}/{c.nz} cells, "
              f"grew {self.panels[0].sim.grew}x")
        # Reported PER PANEL. A pressure-ratio ladder has three different jet speeds by
        # construction, so a single max-over-panels arrival is not a pacing number - it is the
        # fastest panel's, and tuning t_end to put THAT at 67% would mean the slowest panel never
        # arrives at all. The 55-80% band is a single-stream target; for a ladder, read the
        # spread and check that the SLOWEST one gets somewhere.
        for k in range(len(self.panels)):
            seq = [f[k] for f in fronts]
            tag = f"panel {k}" if len(self.panels) > 1 else "front"
            lab = self.panels[k].label
            print(f"  {tag}{' (' + lab + ')' if lab else ''}: reached "
                  f"{seq[-1]}/{c.nz} cells ({seq[-1] / max(c.nz,1) * 100:.0f}% of the panel)",
                  end="")
            arr = next((i for i, f in enumerate(seq) if f >= 0.97 * c.nz), None)
            if arr is None:
                print("  - never reached the end")
            else:
                pct = (arr + 1) / max(n, 1) * 100.0
                # arrival TIME is fixed physics; only its fraction of the clip moves with t_end,
                # so t_end_new = t_end * (pct/67) - this-over-67, NOT 67-over-this.
                flag = ("" if (55 <= pct <= 80 or len(self.panels) > 1) else
                        f"  <- outside 55-80%; set t_end = {c.t_end * pct / 67.0:.0f}")
                print(f", ARRIVES at {pct:.0f}% of the clip (source: 67%){flag}")
        adv = np.diff([max(f) for f in fronts]) if n > 1 else np.array([0.0])
        print(f"  fastest-panel advance: mean {adv.mean():.2f} cells/frame, max {adv.max():.2f}")
        print(f"  solve wall clock {el:.1f}s  ({el / max(n,1) * 1000:.0f} ms/frame)")
        if bad:
            print(f"\n  FAILED at frame {bad[0]}: |u| = {bad[1]:.3f} exceeds the ceiling.")
            return False
        print("\n  COMPLETED")
        return True

    def render(self, out_path, verbose=True):
        c = self.cfg
        ff = ffmpeg_bin()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        part = out_path + ".part.mp4"                 # atomic: rename only on success
        cmd = [ff, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{c.width}x{c.height}", "-r", str(c.fps), "-i", "-",
               "-an", *encoder_args(c, ff), "-movflags", "+faststart", part]
        if verbose:
            print(c.summary())
            for i, p in enumerate(self.panels):
                tag = f"  [{i}] " if len(self.panels) > 1 else "  "
                print(tag + p.sim.describe().replace("\n", "\n  " + " " * len(tag)))
            print(f"  encoder: {'NVENC' if 'nvenc' in ' '.join(cmd) else 'libx264'}"
                  f"  GPU solve: {GPU}")
        self.settle(verbose)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        peak = 0.0
        try:
            for i in range(c.frames):
                self.advance_to(self.t_of(i))
                h = self.health()
                peak = max(peak, h)
                if self.over_ceiling():
                    raise RuntimeError(
                        f"frame {i}: |u| = {h:.3f} exceeds the thermodynamic ceiling - "
                        f"aborting rather than encoding garbage")
                frame = asnumpy(self.frame_rgb(i / c.fps))
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                if verbose and (i % 60 == 0 or i == c.frames - 1):
                    el = time.time() - t0
                    print(f"    frame {i + 1}/{c.frames}  |u|max={h:.2f}  {el:.0f}s "
                          f"(eta {el / (i + 1) * (c.frames - i - 1):.0f}s)", flush=True)
        except Exception:
            proc.stdin.close()
            proc.wait()
            if os.path.exists(part):
                os.remove(part)
            raise
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0 or not os.path.exists(part):
            raise RuntimeError(f"ffmpeg exited {rc}")
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(part, out_path)
        el = time.time() - t0
        print(f"  wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB, "
              f"{el:.0f}s, {el / c.frames * 1000:.0f} ms/frame, peak |u| {peak:.2f})")
        return out_path

    def still(self, path, at_frac=1.0):
        return self.stills({at_frac: path})[0]

    def stills(self, at):
        """ONE solve, several frames: `at` maps frame fraction -> output path.

        Two stills from one clip are two taps on the same causal run, exactly as Wind_Tunnel's
        `--still T,T` is - not two solves. At 600 frames the difference is minutes.
        """
        from PIL import Image
        c = self.cfg
        print(c.summary())
        self.cover_report()
        self.settle()
        # frame INDEX is 0-based, as `render` writes them; fraction 0 is the first written frame
        want = {min(c.frames - 1, max(0, int(round(c.frames * f)))): p
                for f, p in sorted(at.items())}
        out = []
        for i in range(max(want) + 1):
            self.advance_to(self.t_of(i))
            if i in want:
                # The star field is stepped inside frame_rgb, so every frame up to the last tap
                # must be COMPOSED, not just solved - skipping one would leave the parallax
                # behind by that frame's travel. Cheap next to the solve, and correct.
                img = self.frame_rgb(i / c.fps)
                Image.fromarray(asnumpy(img)).save(want[i])
                print(f"  wrote {want[i]}  clip t={i / c.fps:.2f}s  solver t="
                      f"{self.panels[0].sim.t:.0f}  tilt {tilt_at(c, i / c.fps):.2f} deg  "
                      f"peak |u| {self.health():.2f}", flush=True)
                out.append(want[i])
            elif self.stage is not None and self.stage.stars is not None:
                v = star_speed_at(c, i / c.fps) / float(c.fps)
                th = np.radians(tilt_at(c, i / c.fps))
                self.stage.stars.advance(-v * float(np.cos(th)), v * float(np.sin(th)))
        return out
