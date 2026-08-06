"""config.py ââ‚¬â€ render/solver config, ffmpeg resolution, encoder args.

Vertical 1080x1920 by default (these are shorts, same target as ``Oscilloscope``). The SOLVER
runs in its own "wind coordinates" lattice ââ‚¬â€ x is ALWAYS the streamwise axis, y the cross-stream
axis ââ‚¬â€ and `flow` only decides how that lattice is mapped onto the screen:

    flow="up"     streamwise -> screen VERTICAL (bottom to top).  lattice nx = H/scale, ny = W/scale
                  The default: a tall frame gives ~1920 px of downstream wake to develop in.
    flow="right"  streamwise -> screen HORIZONTAL (left to right). lattice nx = W/scale, ny = H/scale
                  The classic wind-tunnel orientation (the source video), letterboxed into 9:16.

Nothing in `lbm.py` / `shapes.py` knows about the screen; only `render.py` applies the mapping.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from dataclasses import field as _dc_field      # `field` is taken: it's a config option below

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "out")
# the Oscilloscope project supplies the CRT "filmed off a screen" post-filter (see wt/film.py)
OSC_ROOT = os.path.normpath(os.path.join(ROOT, "..", "..", "oscilloscope", "src"))


@dataclass
class RenderConfig:
    # --- output ---
    width: int = 1080
    height: int = 1920
    fps: int = 60
    flow: str = "up"                # "up" | "down" (streamwise = screen vertical) | "right"
                                    # "down" is "up" with the stream running top-to-bottom: same
                                    # lattice, same physics, only render.py's mapping differs.
                                    # It also mirrors the cross-stream axis (lattice +y -> screen
                                    # LEFT) so the mapping stays orientation-PRESERVING; without
                                    # that the picture would be a reflection and the sign of
                                    # `--field vort` would silently invert.

    # --- solver ---
    solver: str = "lbm"             # "lbm" = D2Q9 lattice-Boltzmann (wt/lbm.py), the default and
                                    # the right choice for almost everything here.
                                    # "cns" = compressible Navier-Stokes, HLLC+MUSCL (wt/cns.py):
                                    # a real ideal gas with gamma=1.4 and genuine shock capture,
                                    # for scenes that need Mach numbers the LBM cannot reach.
                                    # NOTE `u0` MEANS A DIFFERENT THING in each. Under "lbm" it
                                    # is a lattice velocity and Mach is u0/(1/sqrt(3)); under
                                    # "cns" the freestream sound speed is 1 by construction, so
                                    # u0 IS the Mach number. Porting a scene between solvers
                                    # without converting it is the easy mistake.
                                    # "cns" has no free-body support (see CNS.force_field) and no
                                    # sealed-box mode, so the blue-pool scenes stay on "lbm".

    # --- lattice ---
    scale: float = 3.0              # screen px per lattice cell (3 -> 640x360 lattice @1080p).
                                    # --preview halves BOTH the frame size and this, so the
                                    # lattice ââ‚¬â€ and therefore the physics ââ‚¬â€ is unchanged.
    u0: float = 0.10                # inlet speed in lattice units/step (keep <= 0.12: Mach limit)
    re: float = 6000.0              # Reynolds number based on the scene's reference length
    csm: float = 0.16               # Smagorinsky constant (LES closure; lets tau sit near 0.5)
    steps: int = 14                 # LBM steps per rendered frame (sets the flow's apparent speed)
    settle: float = 1.2             # seconds of flow simulated BEFORE frame 0 (prime the field)
    perturb: float = 0.02           # one-shot symmetry-breaking velocity noise during settling,
                                    # as a fraction of u0. A perfectly symmetric case (centred
                                    # cylinder) will NOT shed without it. 0 disables.
    sponge_side: int = 0            # side absorbing band, in cells (only used by side_bc="free")
    sponge_out: int = 14            # graded outlet absorber (damps reflections back upstream)
    wall_clamp: float = 0.08        # cap on the wall speed the moving-boundary term sees; must
                                    # exceed the speeds free bodies actually reach (see lbm.py)
    closed: bool = False            # sealed box (no inlet/outlet) for recirculating scenes
    side_bc: str = "slip"           # "slip" = free-slip tunnel walls (no edge stripe, default)
                                    # "free" = graded relax-to-freestream band (sponge_side)

    # --- colour field ---
    field: str = "speed"            # "speed" | "vort" | "pressure"
                                    # solver="cns" adds "mach" and "schlieren". Schlieren
                                    # (|grad rho|/rho) is what a real tunnel shows you OF AIR: it
                                    # is blind to velocity and sees only density gradient, so it
                                    # renders shocks and expansion fans and nothing else.
    cmap: str = "jet"               # see colormap.py (jet matches the source video)
    vmax: float = 2.20              # field is normalised by u0*vmax (freestream lands mid-map)
    gamma: float = 0.95             # tone curve on the normalised field
    saturation: float = 1.0         # LUT chroma, 1 = the shipped palettes untouched. Below 1
                                    # blends each entry toward its own luminance.
    brightness: float = 1.0         # LUT luminance scale, 1 = untouched.
                                    # Both act on the COLOURMAP only, so the white tracer dashes
                                    # and the body keep full contrast against a tamed field -
                                    # see colormap._tone for why `jet`'s middle is the problem.

    # --- streaklines (the advected white dashes) ---
    n_streaks: int = 32000          # quoted AT 1080x1920; render.py scales it by frame area so
                                    # the on-screen dash DENSITY is resolution-independent (a
                                    # fixed count looks dense in a preview and sparse in a final)
    streak_tail: int = 22           # backward-integration samples per streak
    streak_span: float = 4.0        # tail length in FRAMES of travel. One frame of motion is
                                    # only ~8 px, which renders as a dot-field that reads as
                                    # noise; ~4 frames gives the ~30 px dash of the source.
    streak_gain: float = 0.46
    streak_speed_gain: float = 0.38    # how much a fast tracer outshines a slow one (0 = flat)
    streak_sigma: float = 0.85      # px blur on the streak layer
    streak_life: tuple = (0.55, 1.5)   # seconds; randomised per particle, then respawn

    # --- body ---
    body_fill: tuple = (12, 14, 18)
    body_edge: tuple = (238, 240, 236)
    body_edge_px: int = 3

    loop: bool = False              # this scene is a SEAMLESS LOOP of length scene.duration.
                                    # Puts the tracer layer into its periodic mode (see
                                    # streaks.py) and lengthens the pre-roll accordingly. It does
                                    # NOT make the flow loop - only the scene can do that, by
                                    # bringing its bodies to rest early enough that the field
                                    # flushes back to the frame-0 state. Verify with
                                    # tools\\loop_check.py, which renders one frame PAST the end
                                    # and diffs it against frame 0.

    # --- overlays ---
    fbd: bool = False               # burn in the FREE-BODY DIAGRAM: labelled force arrows on a
                                    # body, supplied by the scene's `arrows(t)` in LATTICE
                                    # coords. Opt-in, and deliberately so - the source clip this
                                    # project was built from had force arrows on its foil and
                                    # Ethan disliked them (CLAUDE.md preamble), so no scene gets
                                    # them unless it is ABOUT them. `surf_wave` is.

    # --- legend ---
    legend: bool = False           # burn in a colour-bar key (top left) reading the LUT back as
                                    # a speed scale. Opt-in: the shipped clips are deliberately
                                    # bare, but a scene whose SUBJECT is the colour mapping (e.g.
                                    # tri_foil_rates, three flow rates at once) needs the key.
                                    # Content comes from the scene's legend() method.

    # --- post ---
    film: bool = False              # CRT "filmed off a screen" pass (Oscilloscope crtfilm)
    glow: float = 0.55              # film halation boost (LOW: the field is bright edge to edge)
    exposure: float = 0.72          # film pre-exposure. FilmLook's 1.75 default exposes for a
                                    # thin trace on a black scope screen; a wind-tunnel frame is
                                    # bright everywhere and blows out to pastel at that level.
    ambient: float = 0.16           # film ambilight (content colour spilling into the surround)
    bow: float = 0.045              # film barrel distortion
    seed: int = 7

    extras: dict = _dc_field(default_factory=dict)

    # OVERSCAN: simulate a domain LARGER than the visible frame and show only the middle of it.
    # A body entering or leaving does so off-camera, so it never pops into shot, and the pressure
    # transient it makes while appearing has room to decay before it reaches the visible area.
    # Expressed as a fraction of the visible extent added to EACH side (0.12 = 12% margin all
    # round -> ~24% more cells per axis). 0 = the visible frame is the whole domain.
    overscan: float = 0.0

    # -- derived lattice dimensions ---------------------------------------------------
    @property
    def vis_nx(self) -> int:        # VISIBLE streamwise cells (what the screen shows)
        return int(round((self.height if self.flow in ("up", "down") else self.width) / self.scale))

    @property
    def vis_ny(self) -> int:        # VISIBLE cross-stream cells
        return int(round((self.width if self.flow in ("up", "down") else self.height) / self.scale))

    @property
    def nx(self) -> int:            # SIMULATED streamwise cells (visible + overscan margins)
        return int(round(self.vis_nx * (1.0 + 2.0 * self.overscan)))

    @property
    def ny(self) -> int:            # SIMULATED cross-stream cells
        return int(round(self.vis_ny * (1.0 + 2.0 * self.overscan)))

    @property
    def vis_x0(self) -> int:        # lattice index of the visible window's lower streamwise edge
        return (self.nx - self.vis_nx) // 2

    @property
    def vis_y0(self) -> int:
        return (self.ny - self.vis_ny) // 2


def encoder_args(quality: str = "normal") -> list:
    """GPU encode (h264_nvenc) is the DEFAULT. WT_NVENC=0 (or OSC_NVENC=0) forces libx264.

    `quality="high"` (cli `--hq`) exists because these clips are pathological for a video codec:
    a full-frame field of fine white streaks over a smooth colour ramp is high-entropy
    EVERYWHERE, so there is no quiet region for the rate controller to steal bits from, and the
    first thing to go is exactly the streak texture the picture is made of. High mode drops cq
    19 -> 14, raises the ceiling 40 -> 90 Mb/s and moves to the slowest NVENC preset with two
    B-frames of lookahead; on the x264 fallback it is crf 16 -> 12 at `slow`. Encode time is a
    rounding error next to a half-hour solve, so the only real cost is file size.
    """
    hq = str(quality).lower() in ("high", "hq")
    if os.environ.get("WT_NVENC", os.environ.get("OSC_NVENC", "1")) != "0" and _nvenc_available():
        if hq:
            # MEASURED, not assumed: the normal-mode render came out at 41.0 Mb/s against its
            # own 40M ceiling, i.e. the cap was BINDING and the rate controller was throwing
            # away detail it wanted to keep. So the ceiling here is set high enough to stop
            # binding (cq 14 is then the only thing deciding quality, which is the point of
            # constant-quality mode) rather than to a number that sounds generous.
            # `-profile:v high` is free efficiency: NVENC was emitting MAIN, which gives up the
            # 8x8 transform - worth a few percent on exactly this kind of fine texture.
            return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
                    "-profile:v", "high", "-cq", "14", "-b:v", "0",
                    "-maxrate", "220M", "-bufsize", "440M",
                    "-bf", "2", "-rc-lookahead", "20", "-spatial-aq", "1", "-aq-strength", "12",
                    "-pix_fmt", "yuv420p"]
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
                "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M"]
    return ["-c:v", "libx264", "-crf", "12" if hq else "16",
            "-preset", "slow" if hq else "medium"]


_NVENC_OK = None


def _nvenc_available() -> bool:
    """Probe h264_nvenc ONCE per process (256x256 test encode ââ‚¬â€ NVENC rejects tiny frames, don't
    shrink it). A missing/busy NVENC must degrade to libx264, never kill a render mid-pipe."""
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            r = subprocess.run(
                [find_ffmpeg(), "-v", "error", "-f", "lavfi",
                 "-i", "color=black:size=256x256:rate=30:duration=0.1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=20)
            _NVENC_OK = r.returncode == 0
        except Exception:
            _NVENC_OK = False
        if not _NVENC_OK:
            print("[config] h264_nvenc unavailable -> falling back to libx264 (CPU encode)")
    return _NVENC_OK


def find_ffmpeg() -> str:
    """WT_FFMPEG/OSC_FFMPEG env -> local bin\\ffmpeg.exe -> the Chemistry bundled binary -> PATH."""
    for key in ("WT_FFMPEG", "OSC_FFMPEG"):
        env = os.environ.get(key)
        if env and os.path.exists(env):
            return env
    local = os.path.join(ROOT, "bin", "ffmpeg.exe")
    if os.path.exists(local):
        return local
    for cand in (os.path.join(OSC_ROOT, "bin", "ffmpeg.exe"),):
        if os.path.exists(cand):
            return cand
    return "ffmpeg"





