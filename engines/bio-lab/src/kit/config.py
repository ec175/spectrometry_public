# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""config.py - the world, the tiers, and the crop model. ONE place.

Read this file before you change anything about resolution, frame rate or framing. Almost
every "why does the final look different from the preview" bug in a video engine is a
violation of one of the three contracts below.


THE WORLD IS A SQUARE
---------------------
Scenes are authored in a single **1920 x 1920 reference-unit** square centred on (0, 0).
Nothing in a scene ever mentions pixels or an aspect ratio.

Every output shape CROPS a centred rectangle whose LONGEST side is `WORLD` units:

    horizontal 16:9  ->  crop 1920 x 1080 ru   (full width, loses top and bottom)
    vertical   9:16  ->  crop 1080 x 1920 ru   (full height, loses left and right)

So the centred **1080 x 1080 square is visible in every shape** - the SAFE SQUARE. Put
anything that must always be on screen inside it; the live preview draws its outline.

That is why one scene renders to all four delivery shapes with no re-authoring. The cost,
stated honestly: you cannot compose tightly for one shape. If you ever need to, add a
per-shape `crop` override to the scene JSON rather than authoring the scene twice.


THE THREE CONTRACTS
-------------------
1. **FIXED SUBSTEP CLOCK.** The simulation advances in steps of 1/`SUBSTEP_HZ` seconds,
   never in steps of one output frame. 120 is divisible by 24, 30 and 60, so every
   delivery frame rate lands exactly on a substep and no tier interpolates between two
   states the physics never occupied. A 30 fps preview and a 60 fps final are then the
   SAME animation sampled differently - the only thing that makes a preview worth looking
   at.

2. **DECAY IS A TIME CONSTANT, NEVER A PER-FRAME FACTOR.** Write `trail_tau = 0.25`
   seconds and apply `exp(-dt/tau)` per substep. If you write `trail *= 0.9` per frame,
   trails come out exactly half as long when you switch 30 -> 60 fps, and the bug is
   invisible until someone puts the two files side by side.

3. **ADDITIVE GAIN SCALES WITH PIXEL AREA.** Anything that deposits light must deposit
   the same flux per unit REFERENCE area at every resolution, so additive gains carry a
   `scale**2` factor (`RenderConfig.area_scale`). Without it a 1080p render is ~2.2x
   dimmer than its own 720p preview and everyone concludes "the finals came out wrong"
   when it is really just pixel area.


TIERS
-----
`quality` x `orientation` x `fps` - the four output shapes, plus `live`, the low
resolution tier the browser preview streams at. `live` obeys all three contracts, so what
you tune in the browser is what renders.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out")
SCENES = os.path.join(ROOT, "scenes")
ASSETS = os.path.join(ROOT, "assets")

#: Side of the square authoring world, in reference units (ru).
WORLD = 1920.0
#: The region visible in EVERY output shape. Keep the subject inside it.
SAFE = 1080.0

#: The simulation clock. Divisible by 24, 30 and 60 - see contract 1 above.
SUBSTEP_HZ = 120

#: SHORT side in pixels for each named quality. "720p" and "1080p" name the short side by
#: universal convention: 1080p horizontal is 1920 x 1080, and vertical 1080 is 1080 x 1920.
#: Naming the LONG side instead gives you a 1080p horizontal of 1080 x 608, which is what
#: this constant looked like for about ten minutes and is the reason it is commented.
QUALITY = {"720": 720, "1080": 1080, "2160": 2160}
ORIENTATIONS = ("horizontal", "vertical")
FPS_CHOICES = (30, 60)

#: The browser preview tier, as a SHORT side. 360 -> 640x360 or 360x640, which keeps a
#: CPU-only laptop interactive. It is a real render at a real tier, not a special path.
LIVE_SHORT = 360
LIVE_FPS = 30


def shape_for(quality: str = "1080", orientation: str = "horizontal") -> tuple:
    """(width, height) in pixels for a named quality + orientation.

        720  horizontal -> 1280 x  720        720  vertical ->  720 x 1280
        1080 horizontal -> 1920 x 1080        1080 vertical -> 1080 x 1920
    """
    if quality not in QUALITY:
        raise ValueError("quality must be one of %s, got %r" % (sorted(QUALITY), quality))
    if orientation not in ORIENTATIONS:
        raise ValueError("orientation must be one of %s, got %r" % (ORIENTATIONS, orientation))
    short_px = QUALITY[quality]
    long_px = int(round(short_px * 16 / 9))
    if long_px % 2:                       # h264 needs even dimensions on both axes
        long_px += 1
    return (long_px, short_px) if orientation == "horizontal" else (short_px, long_px)


@dataclass
class RenderConfig:
    """Everything the renderer needs that is NOT the scene's own parameters.

    Built from a tier name plus the scene's framing, then handed around read-only. If you
    find yourself mutating one mid-render, something has gone wrong: a changed `scale`
    halfway through silently rescales every gain from that frame on.
    """

    width: int = 1920
    height: int = 1080
    fps: int = 60

    #: Framing, in reference units. `zoom` > 1 moves the camera IN (a smaller crop), so
    #: the subject gets bigger. `centre` shifts the crop off the world origin.
    zoom: float = 1.0
    centre: tuple = (0.0, 0.0)

    #: Anti-aliasing supersample factor. 1 for live, 2 for delivery. Costs 4x the pixels.
    ss: int = 1

    #: Encoder knobs. `crf` is libx264's, `cq` is NVENC's. Lower is better quality.
    crf: int = 17
    cq: int = 19
    #: TARGET BITRATE in bits per second, or 0 to let crf/cq decide alone.
    #:
    #: Set this when the file is going to be re-encoded by someone else - Instagram,
    #: TikTok and YouTube all transcode on upload, and what survives that pass is roughly
    #: proportional to what you gave them. A quality-targeted encode (crf/cq alone) will
    #: happily spend 8 Mbit/s on a dark frame that compresses well, and the platform's
    #: encoder then has little to work from. Forcing a high floor costs disk and nothing
    #: else.
    bitrate: int = 0
    #: Peak allowed, as a multiple of `bitrate`. Headroom for the busy frames.
    bitrate_peak: float = 1.6

    @property
    def crop(self) -> tuple:
        """(crop_w, crop_h) in reference units - the window this shape cuts out."""
        a = self.width / float(self.height)
        cw, ch = (WORLD, WORLD / a) if a >= 1.0 else (WORLD * a, WORLD)
        return cw / self.zoom, ch / self.zoom

    @property
    def scale(self) -> float:
        """reference units -> output pixels (before supersampling)."""
        return self.width / self.crop[0]

    @property
    def area_scale(self) -> float:
        """Multiply every ADDITIVE gain by this. Contract 3 above."""
        return self.scale ** 2

    @property
    def render_size(self) -> tuple:
        """The size actually rasterised, i.e. output size x supersample."""
        return self.width * self.ss, self.height * self.ss

    def world_to_px(self, x, y):
        """Vectorised world (ru) -> render pixel.

        y is FLIPPED: world y is up, pixel y is down. Getting this wrong is the classic
        "my gravity is inverted" afternoon.
        """
        cx, cy = self.centre
        s = self.scale * self.ss
        rw, rh = self.render_size
        return (x - cx) * s + rw * 0.5, (cy - y) * s + rh * 0.5

    def px_per_ru(self) -> float:
        """Render pixels per reference unit, supersampling included. Use this to convert
        a radius or a stroke width quoted in ru into something to rasterise."""
        return self.scale * self.ss

    # --- tier constructors -------------------------------------------------------------
    @classmethod
    def live(cls, orientation: str = "horizontal", **kw):
        """The browser preview: 640x360 or 360x640 @30, no supersampling."""
        long_px = int(round(LIVE_SHORT * 16 / 9))
        long_px += long_px % 2
        w, h = (long_px, LIVE_SHORT) if orientation == "horizontal" else (LIVE_SHORT, long_px)
        kw.setdefault("width", w)
        kw.setdefault("height", h)
        kw.setdefault("fps", LIVE_FPS)
        kw.setdefault("ss", 1)
        return cls(**kw)

    #: Default target bitrates per quality, in bits/s, chosen for platforms that will
    #: re-encode the file. Generous on purpose - see the `bitrate` note above.
    BITRATES = {"720": 12_000_000, "1080": 28_000_000, "2160": 90_000_000}

    @classmethod
    def tier(cls, quality: str = "1080", orientation: str = "horizontal",
             fps: int = 60, ss: int = 2, **kw):
        """A delivery tier: quality x orientation x fps.

        At 2160 the default supersample is 1, not 2. An ss=2 4K frame is a 7680x4320
        float accumulator - roughly 400 MB per layer - and it is not worth it: the whole
        look is soft glows and blooms, which have no high-frequency content for the extra
        samples to resolve.
        """
        if fps not in FPS_CHOICES:
            raise ValueError("fps must be one of %s, got %r" % (FPS_CHOICES, fps))
        if quality == "2160":
            ss = min(ss, 1)
        kw.setdefault("bitrate", cls.BITRATES.get(quality, 0))
        w, h = shape_for(quality, orientation)
        kw.setdefault("width", w)
        kw.setdefault("height", h)
        kw.setdefault("fps", fps)
        kw.setdefault("ss", ss)
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)
