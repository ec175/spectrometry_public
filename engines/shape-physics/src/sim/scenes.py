# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""scenes.py - the scene catalogue. A scene is geometry + materials + audio choice, nothing
more; it cannot place a ball on a path (see world.py's note).

`escape_rings` is the original brief. The ten that follow it are the LIBRARY set: no music, a
different structure and a different trigger scheme each, all built from the same three
primitives so anything that reads well can be lifted straight into a music cut.

Each scene's headline event (whatever "something happened" means there) lands in
`world.triggers`, which drives both the shockwave on screen and the lower event voice in the
audio. What counts as the headline event is the interesting design choice, and it differs on
purpose across the set: escaping a shell, reaching a bin, grazing the centre, breaking a
brick, or simply reaching the floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import build, config
from .draw import Style
from .world import Peg, Ring, World

TAU = 2.0 * math.pi
DEG = math.pi / 180.0
CX, CY = config.REF_W * 0.5, config.REF_H * 0.5

# palettes, so the set does not read as ten shades of the same video
CYAN = (120, 215, 255)
MAGENTA = (255, 110, 195)
AMBER = (255, 185, 80)
MINT = (120, 255, 190)
VIOLET = (185, 140, 255)
LIME = (200, 255, 110)
CORAL = (255, 130, 120)
ICE = (200, 235, 255)
WHITE = (255, 255, 255)
BLUE = (95, 175, 255)
RED = (255, 85, 85)
PURPLE = (185, 120, 255)
GREEN = (110, 245, 130)
YELLOW = (255, 225, 90)

# THE TIER LADDER (2026-07-29). `Ball.gen` is the tier index and draw.py
# colours a ball by gen, so ordering this palette names the tiers: white is the parent size,
# and every mechanic that makes a ball smaller (a multiplier pit, a scatter) walks down it.
TIER_PALETTE = (WHITE, BLUE, RED, PURPLE, GREEN, YELLOW)

# Vertical stretch of the orbital field, so orbits fill a 9:16 frame instead of sitting in a
# circle inside it. 1920/1080 = 1.78 is the frame's own aspect, which makes an orbit that fits
# the width also fit the height.
ANISO = 1.78


@dataclass
class Scene:
    name: str
    duration: float
    make_world: object                  # callable() -> World
    style: Style = field(default_factory=Style)
    song: str = ""
    song_start: float = 0.0
    slice_len: float = 0.5
    bounce_gain: float = 0.13       # 0 = no bounce blips
    bounce_duck: float = 0.55       # blip level while the song is sounding
    event_gain: float = 0.0         # the headline-event voice (0 = off)
    substeps: int | None = None     # physics rate override (obstacle-heavy scenes)
    blurb: str = ""

    def build(self) -> World:
        return self.make_world()


# --------------------------------------------------------------------------- the brief

def escape_rings(seconds: float = 20.0, seed: int = 7, song: str = "",
                 song_start: float = 7.6, slice_len: float = 0.5,
                 r0: float = 26.0, shrink: float = 0.78, max_balls: int = 20,
                 r_min: float = 7.0, split: int = 2, split_min_r: float = 9.5,
                 bounce_gain: float = 0.13, bounce_duck: float = 0.55) -> Scene:
    """Two hollow spinning shells, gaps in both, the outer gaps carrying the audio triggers.

    No song ships with this engine (`song=""` renders silently but for the bounce/event
    voices) - pass `--song PATH` to a file of your own to hear the escape-triggered
    progressive playback this scene was built to show off.

    Geometry notes (why these numbers):
    - The gap must clear the BALL, not a point: a ball of radius r at shell radius R needs
      2*asin(r/R) of arc just to fit. 32 deg at R=210 against the 14 deg a 26 px ball needs
      is comfortable without being a doorway.
    - The shells counter-rotate, and the inner one turns faster. Both matter: counter-rotation
      keeps the two gap sets from phase-locking (a ball that escapes the inner shell would
      otherwise meet the outer gap at nearly the same place every time), and the speed
      difference means the annulus is a genuinely different problem from the core.
    - Annulus clearance is (400-8) - (210+7) = 175 px, ~3.4 ball diameters, so a ball in
      there has room to bounce rather than being wedged.
    - `mu` is what makes this alive: the shells are spinning, and bounces resolve in the WALL
      frame, so a solid arc sweeping past kicks the ball along with it. Set mu = 0 and balls
      just rattle down to the bottom and sit there waiting for a gap.
    """
    inner = Ring(radius=210.0, thickness=14.0, n_gaps=2, gap_width=32 * DEG,
                 omega=+1.90, phase0=0.55, e=0.92, mu=0.16, color=CYAN, name="inner")
    outer = Ring(radius=400.0, thickness=16.0, n_gaps=3, gap_width=26 * DEG,
                 omega=-1.25, phase0=-0.30, e=0.90, mu=0.14, color=MAGENTA, name="outer")

    def make_world():
        return World([inner, outer], CX, CY, config.REF_W, config.REF_H,
                     gravity=1500.0, seed=seed, ball_e=0.90,
                     r0=r0, shrink=shrink, r_min=r_min, max_balls=max_balls,
                     spawn_speed=260.0, spawn_jitter=30.0, split=split,
                     split_min_r=split_min_r)

    return Scene(name="escape_rings", duration=float(seconds), make_world=make_world,
                 style=Style(), song=config.resolve_song(song),
                 song_start=song_start, slice_len=slice_len,
                 bounce_gain=bounce_gain, bounce_duck=bounce_duck,
                 blurb="two counter-rotating gapped shells; escaping the outer one plays the song")


# --------------------------------------------------------------------------- the library set
# All silent (no music asset): bounce blips pitched by ball size, plus the event voice.

def _pit_row(mults, y=1760.0, r=86.0, x0=170.0, x1=910.0, wall_top=1600.0,
             color=MINT, thickness=13.0, dividers=True, tile=False):
    """A row of multiplier pits, optionally with dividers between them. Returns
    (zones, obstacles).

    A ball that drops into an Nx pit is consumed and comes back as N balls one tier down (see
    World's multiplier-pit note) - so the pit row is the scene's payoff AND its spawner.

    `tile=True` sets the radius so neighbouring pits TOUCH, leaving no gap for a ball to fall
    through. This matters more than it looks: with r below half the pit spacing, `funnel_cascade`
    caught only 3 of 34 balls - the rest poured down the ~40 px chutes either side of each
    divider and out of the frame. A pit row that is meant to catch everything must tile.
    """
    n = len(mults)
    xs = [x0 + (x1 - x0) * (k / max(1, n - 1)) for k in range(n)]
    if tile and n > 1:
        r = max(r, 0.5 * (xs[1] - xs[0]) + 2.0)
    zones, obs = [], []
    for k, (x, m) in enumerate(zip(xs, mults)):
        zones.append({"x": x, "y": y, "r": r, "mult": int(m)})
        if dividers and k:
            xm = 0.5 * (x + xs[k - 1])
            obs.append(build.Capsule(xm, wall_top, xm, y + r, thickness, color=color,
                                     name=f"pitdiv{k}"))
    return zones, obs


def _lib(name, make_world, seconds, style, blurb, substeps=None, bounce_gain=0.14,
         event_gain=0.30):
    return Scene(name=name, duration=float(seconds), make_world=make_world, style=style,
                 song="", bounce_gain=bounce_gain, event_gain=event_gain,
                 substeps=substeps, blurb=blurb)


def hex_nest(seconds: float = 15.0, seed: int = 3, **kw) -> Scene:
    """Three nested POLYGON cages, each missing one side, counter-rotating at different rates.
    The corners are the character here: a polygon concentrates a ball into its vertices, so
    escapes come in sudden bursts rather than the even trickle a circle gives."""
    def make_world():
        obs = []
        # Spin rates are the liveliness lever, and they have to be read against the CLIP
        # LENGTH: at the first build's -0.55 rad/s the hexagon's missing side swept past the
        # bottom of the frame once every 11 s, so in a 15 s clip the ball simply waited. These
        # rates put every opening past the low point several times per clip.
        obs += build.polygon_shell(CX, CY, 400, 6, 16, missing=(1, 4), omega=-1.15,
                                   color=MAGENTA, name="hex")
        obs += build.polygon_shell(CX, CY, 275, 5, 15, missing=(2,), omega=+1.70,
                                   rot=0.4, color=VIOLET, name="pent")
        obs += build.polygon_shell(CX, CY, 155, 4, 14, missing=(0,), omega=-2.40,
                                   rot=0.8, color=CYAN, name="sq")
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1500.0, seed=seed,
                     r0=23.0, shrink=0.80, r_min=8.0, max_balls=18, split=2,
                     split_min_r=11.0, sensor_r=430.0)
    return _lib("hex_nest", make_world, seconds,
                Style(ball_palette=(ICE, CYAN, VIOLET, MAGENTA)),
                "3 nested spinning polygon cages, one side missing each; escape the outermost")


def plinko(seconds: float = 15.0, seed: int = 11, **kw) -> Scene:
    """A Galton board over five MULTIPLIER pits, sharing orbit_swarm's tier ladder.

    A white sphere landing in a 2x comes back as two smaller BLUE spheres, those in turn as
    RED, then purple, green, yellow. So one ball feeds an escalating cascade, and the pin walk
    decides how far down the ladder it gets. The pits are 1x / 2x / 3x / 2x / 1x, which puts
    the jackpot dead centre - exactly where the Galton distribution piles up, so the cascade
    builds by design rather than by luck.

    `tier_shrink = 0.72` keeps it bounded: 2 children at 0.72 radius carry 2*0.72^2 = 1.04 of
    the parent's area and 3 carry 1.56, so area grows only slowly per generation while the
    r_min floor and the ball cap catch the tail.
    """
    def make_world():
        obs = build.peg_grid(140, 940, 500, 112, rows=9, cols=8, radius=11.0,
                             e=0.72, color=ICE)
        obs += build.walls(60, 1020, 440, 1900, 16, color=CYAN, sides="lr")
        zones, pit_obs = _pit_row((1, 2, 3, 2, 1), y=1780.0, r=80.0, x0=175.0, x1=905.0,
                                  wall_top=1620.0, color=MINT)
        obs += pit_obs
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1350.0, seed=seed,
                     r0=22.0, max_balls=26, spawn_mode="top", spawn_every=0.8, initial=2,
                     respawn="fresh", zones=zones, sensor_r=0.0, ball_e=0.55,
                     tier_shrink=0.72, n_tiers=6, r_min=6.0)
    return _lib("plinko", make_world, seconds,
                Style(ball_palette=TIER_PALETTE, zone_color=MINT),
                "Galton board over 1x/2x/3x/2x/1x multiplier pits; tier ladder white>yellow",
                substeps=700)


def paddle_wheel(seconds: float = 15.0, seed: int = 5, **kw) -> Scene:
    """A bowl with a four-blade wheel turning inside it. The blades do the work: a ball that
    settles at the bottom is picked up, carried and thrown, so the scene never goes quiet the
    way a plain bowl does. Escapes happen when a throw lines up with the rim's single gap."""
    def make_world():
        bowl = Ring(radius=430.0, thickness=18.0, n_gaps=2, gap_width=36 * DEG,
                    omega=+0.45, e=0.88, mu=0.12, color=AMBER, name="bowl")
        obs = [bowl]
        obs += build.spokes(CX, CY, 45, 250, 4, 15, omega=+2.60, color=CORAL)
        obs += [Peg(CX, CY, 34.0, e=0.9, color=ICE, name="hub")]
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1500.0, seed=seed,
                     r0=24.0, shrink=0.80, r_min=9.0, max_balls=16, split=2,
                     split_min_r=12.0, escape_ring=bowl)
    return _lib("paddle_wheel", make_world, seconds,
                Style(ball_palette=(ICE, MINT, CYAN, VIOLET), pad_color=AMBER),
                "4-blade wheel flinging balls inside a bowl with one rim gap")


def spiral_drop(seconds: float = 15.0, seed: int = 9, **kw) -> Scene:
    """One long spiral chute, turning slowly. The rotation is what makes it a chute rather
    than a set of parking spots (see build.spiral) - the coil carries a ball around while
    gravity walks it outward, coil by coil, until it clears the last one. Reaching the floor
    is the event."""
    def make_world():
        # SIGN MATTERS: a rotating spiral is an Archimedes screw, and only one direction of
        # rotation conveys OUTWARD. Measured over 15 s: -1.25 gives 16 escapes and 8 balls on
        # the floor; +1.25 gives ZERO and just stirs them around the hub (max radius reached
        # 188 of 425). If this scene ever goes dead, check the sign before anything else.
        obs = build.spiral(CX, CY, 110, 425, 1.5, 58, 13, rot=-1.2, color=MINT, omega=-1.45)
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1400.0, seed=seed,
                     r0=21.0, max_balls=6, spawn_every=2.1, initial=2, respawn="fresh",
                     bottom_triggers=True, sensor_r=468.0, ball_e=0.7)
    return _lib("spiral_drop", make_world, seconds,
                Style(ball_palette=(AMBER, CORAL, LIME, ICE)),
                "a single spiral chute; balls work outward and down", substeps=800)


def orbit_swarm(seconds: float = 15.0, seed: int = 13, **kw) -> Scene:
    """Central-pull orbits inside a DESTRUCTIBLE brick shell, with spheres that scatter and
    recombine. Built out 2026-07-29 from the first set's version of this scene.

    Five mechanics, all of them properties of the bodies rather than scripted beats:
    - **Pink orbs KICK.** The three orbiting pegs have `e = 1.35`, so a ball leaves them with
      more energy than it arrived with - the same device as `bumper_pit`'s solenoids, and what
      keeps a decaying swarm from going flat.
    - **Scatter.** Any sphere-sphere hit has a small chance of fragmenting the PAIR into three
      smaller spheres one tier down (white -> blue -> red). Momentum carries across and the
      outward kicks sum to zero, so the fragmentation adds internal energy, not net drift.
    - **Recombine.** Two of the SMALLEST (red) spheres may instead fuse into one intermediate
      (blue). Radius goes as sqrt(ra^2 + rb^2), i.e. area is conserved. Only the smallest tier
      may fuse, which is what makes the ladder a cycle rather than a one-way slide to dust.
    - **Size sets speed.** The central pull scales as (r/r0)^0.85, so a small sphere is held
      far less tightly and tears around the frame while the big ones sit in slow close orbits.
    - **Size sets damage.** `dmg_ref = 9` px of radius per hp, so a white sphere smashes a
      brick a red one merely chips.
    """
    def make_world():
        # NO containing rim (rev 4): the spheres orbit freely and are never
        # removed for leaving the frame. What keeps them on camera is now the physics rather
        # than a wall - the central pull grows linearly with radius so every orbit is bound,
        # and each tier's guide ellipse pulls it back toward its own track. Reds are MEANT to
        # swing out past the frame edge (the "exosphere") and come back.
        # The breakout shell stays a TRUE CIRCLE even though the orbital
        # field is stretched. The two are deliberately mismatched: vertically elongated orbits
        # cross a round shell near the top and bottom, so that is where the bricks come down
        # first, instead of the shell being hit evenly all round.
        obs = build.arc_bricks(CX, CY, 400, 44, 17, gap_deg=2.0, hp=2, color=MAGENTA,
                               omega=-0.28, name="shell")
        # The three orbiting pink KICKER spheres are CUT (rev 5). They were
        # the swarm's energy source at e=1.35; `swirl_a` already sustains the circulation on its
        # own, so removing them costs motion but not liveliness.
        # the centre sphere is the GRAVITY POINT, drawn but not solid - spheres pass straight
        # through it, which also reads as depth: they fly in front of and
        # behind it rather than bouncing off it
        obs += [Peg(CX, CY, 30.0, e=0.96, color=ICE, name="core", ghost=True)]
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=0.0, central_g=2400.0,
                     seed=seed, r0=23.0, max_balls=24, initial=6, respawn="same",
                     spawn_mode="orbit", ball_e=0.95, r_min=7.0,
                     # INTRINSIC ANGULAR MOMENTUM: one shared direction of travel, launched a
                     # little above circular speed, and scatter fragments biased along the
                     # circulation - so the swarm keeps spinning instead of cancelling itself.
                     # The swirl drive needs DRAG to balance it, or it pumps energy in and every
                     # sphere migrates out and rides the rim (measured: mean radius 479 of 498,
                     # a single ring hugging the wall). With drag 0.25 the pair settles into
                     # circular orbits spread across the whole disc - radii 76-487, mean 400,
                     # 16 of 17 co-rotating, and the shell still comes fully down.
                     # BLUE COUNTER-ROTATES: white +, blue -, red +. It also
                     # composes with the angle rule - a counter-running tier meets every other
                     # tier head-on, which is precisely the case that never interacts.
                     orbit_dir=+1, tier_dir=[+1, -1, +1],
                     orbit_boost=1.0, tan_kick=150.0, swirl_a=500.0, drag=0.25,
                     # THREE ROTATIONALLY STAGGERED ELLIPSES, one per tier, 60 deg apart -
                     # overlaid they trace a six-lobed petal rosette (an ellipse already has
                     # two-fold symmetry, so 3 x 60 deg gives 6 evenly spaced lobes). Assigned
                     # by SIZE: white on the inner track, blue mid-to-outer, red out in the
                     # exosphere beyond where the old rim used to sit (505 px).
                     # TRACKS ARE DESIGNED IN SCREEN SPACE, THEN CONVERTED - the red track
                     # first came out far too flat, swinging back and forth along a diagonal
                     # line.
                     #
                     # A track quoted directly in the stretched metric does NOT look like that
                     # shape on screen: the 60 deg stagger and the 1.78 vertical stretch compound
                     # each other, and a mildly elliptical track becomes a thin diagonal. The old
                     # hand-picked numbers measured 826 x 369 on screen for red (ratio 2.24) and
                     # 598 x 264 for blue (2.26) - flat slashes, exactly as reported.
                     #
                     # So pick the SCREEN oval you want, then invert the mapping:
                     #     M_metric = diag(1, 1/aniso) @ R(tilt) @ diag(major, minor)
                     # and read (A, B, phi) off its SVD. A small helper script (not included
                     # here) does this and prints the resulting on-screen size, which is the
                     # only number worth judging.
                     # These three are 380x290, 620x450 and 820x600 on screen - ratios 1.31-1.38,
                     # proper ovals - still staggered 60 deg apart for the petal rosette.
                     aniso=ANISO,
                     guides=[(375.5, 164.9, -3.0669),
                             (552.9, 283.5, 2.9062),
                             (625.5, 441.9, -2.9095)],
                     guide_a=900.0, guide_soft=150.0,
                     # blue spheres climb back to white after 3 s - closes the ladder
                     decay_after=3.0, decay_rate=0.28, decay_tier=1,
                     # a slow trickle of fresh WHITES: recombination only climbs red -> blue,
                     # so without a spawner the swarm ends up entirely blue/red and the top
                     # of the ladder disappears from the picture
                     spawn_every=3.0,
                     tier_shrink=0.68, n_tiers=3, g_size_exp=0.85,
                     # interaction is decided by HEADING ALIGNMENT, not by contact: head-on
                     # pairs never interact, rear-end pairs always fragment (into 4)
                     angle_rule=True, align_exp=1.0,
                     scatter_chance=0.35, scatter_into=4, recombine_chance=0.30,
                     # v_max caps the excursions a scatter kick can produce. Measured over 30 s:
                     # 2600 let spheres reach 1163 px from centre horizontally, 1900 holds them
                     # to 911 - and it also RAISED the scatter and recombine counts, because
                     # spheres that stay in the disc keep meeting each other.
                     dmg_ref=9.0, size_speed=150.0, v_max=1900.0,
                     # DEPTH PLANES, one per tier, eased in over 3 s once the shell is down.
                     # White holds the centre plane; blue and red split fore and aft. The gaps
                     # must exceed the pair's radius SUM or the tiers still clip: white+blue
                     # need 38.6 px, white+red 33.6, blue+red 26 - so 58 px per side clears all
                     # three, with blue and red 116 px apart from each other.
                     plane_z=[0.0, 58.0, -58.0], plane_ramp=3.0,
                     kill_below=False, sensor_r=0.0,
                     # the core zone still fires its event; `hidden` drops the yellow ring
                     # that used to be drawn round it
                     zones=[{"x": CX, "y": CY, "r": 76.0, "hidden": True}])
    return _lib("orbit_swarm", make_world, seconds,
                Style(ball_palette=(WHITE, BLUE, RED), zone_color=AMBER, guide_gain=0.0),
                "orbits in a destructible shell; spheres scatter (white>blue>red) and recombine")


def breakout_shell(seconds: float = 15.0, seed: int = 17, **kw) -> Scene:
    """TWO counter-rotating shells of destructible bricks, ball loose inside.

    The shells must ROTATE, or the ball settles into one spot and chews a single hole while
    the rest of the shell is never touched - turning them presents fresh bricks to the same
    impact point. The outer shell turns the other way and is bigger, so breaking out of the
    inner one only earns you the next problem.
    """
    def make_world():
        obs = build.arc_bricks(CX, CY, 265, 24, 18, gap_deg=2.2, hp=1, color=CORAL,
                               omega=+0.75, name="inner")
        obs += build.arc_bricks(CX, CY, 440, 40, 18, gap_deg=2.2, hp=2, color=MAGENTA,
                                omega=-0.50, name="outer")
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1550.0, seed=seed,
                     r0=24.0, shrink=0.82, r_min=10.0, max_balls=12, split=2,
                     split_min_r=13.0, sensor_r=480.0, ball_e=0.95, spawn_every=4.5)
    return _lib("breakout_shell", make_world, seconds,
                Style(ball_palette=(ICE, AMBER, LIME, MINT)),
                "two counter-rotating destructible brick shells")


def funnel_cascade(seconds: float = 15.0, seed: int = 23, **kw) -> Scene:
    """Three V funnels staggered down the frame, each SLIDING horizontally at its own slow
    rate, over a 1x / 3x / 1x pit row.

    The sway is what stops the cascade being deterministic: a fixed funnel drops every ball
    through the same point, so the whole clip repeats. Moving mouths mean the hand-off between
    funnels lands somewhere different each pass. The funnels carry the ball with them too - a
    sliding capsule contributes its own velocity at the contact (see Capsule.sway_v).
    """
    def make_world():
        obs = []
        # Sway PEAK SPEED is amp*2*pi*freq, and it has to stay modest: a sliding capsule can
        # pinch a ball against its opposite wall and eject it at the speed cap. At amp 120 /
        # freq 0.16 (121 px/s) balls were fired clean out of the frame - measured bottom-exit
        # x positions ran from -1158 to +2633 on a 1080-wide frame, and only 3 of 33 ever
        # reached a pit. These rates peak near 60 px/s, and the side walls below contain the
        # rest (a real rig has walls).
        specs = [(370, 470, CYAN, 100.0, 0.095), (710, 830, MINT, 115.0, 0.075),
                 (420, 1190, LIME, 105.0, 0.085)]
        for i, (cx, y, col, amp, freq) in enumerate(specs):
            for c in build.funnel(cx, y, 55, 300, 170, 13, color=col, name=f"f{i}"):
                c.sway_amp = amp
                c.sway_freq = freq
                c.sway_phase = 1.7 * i
                obs.append(c)
        # Pits must TILE the full width or balls simply fall past them: at r=108 on
        # x0=250/x1=830 the row left ~90 px open at each edge and between neighbours, and only
        # 3 of 34 balls ever landed in one. Wider pits plus side walls close every bypass.
        zones, pit_obs = _pit_row((1, 3, 1), y=1745.0, r=125.0, x0=200.0, x1=880.0,
                                  wall_top=1470.0, color=AMBER, tile=True, dividers=False)
        obs += pit_obs
        obs += build.walls(70, 1010, 380, 1900, 16, color=AMBER, sides="lr", name="pitwall")
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1500.0, seed=seed,
                     r0=21.0, max_balls=16, spawn_mode="top", spawn_every=0.6, initial=2,
                     respawn="fresh", bottom_triggers=True, sensor_r=0.0, ball_e=0.6,
                     zones=zones, tier_shrink=0.78, n_tiers=6, r_min=6.0)
    return _lib("funnel_cascade", make_world, seconds,
                Style(ball_palette=TIER_PALETTE),
                "3 sliding funnels over a 1x / 3x / 1x pit row")


def bumper_pit(seconds: float = 15.0, seed: int = 29, **kw) -> Scene:
    """Pinball bumpers. These pegs have restitution ABOVE one, so they genuinely add energy -
    which is what a real solenoid bumper does, and why the pit stays lively instead of
    settling. The speed cap is what keeps that honest rather than divergent."""
    def make_world():
        pts = [(300, 620), (780, 620), (540, 830), (250, 1040), (830, 1040),
               (420, 1250), (660, 1250), (540, 1460), (300, 1620), (780, 1620)]
        obs = [Peg(x, y, 27.0, e=1.12, mu=0.02, color=MAGENTA, name=f"bump{i}")
               for i, (x, y) in enumerate(pts)]
        obs += [build.Capsule(60, 500, 150, 1560, 16, color=VIOLET, name="wallL"),
                build.Capsule(1020, 500, 930, 1560, 16, color=VIOLET, name="wallR")]
        zones, pit_obs = _pit_row((1, 3, 1), y=1740.0, r=92.0, x0=230.0, x1=850.0,
                                  wall_top=1560.0, color=VIOLET)
        obs += pit_obs
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1500.0, seed=seed,
                     r0=22.0, max_balls=26, spawn_mode="top", spawn_every=1.1, initial=3,
                     respawn="fresh", bottom_triggers=True, sensor_r=0.0, ball_e=0.9,
                     v_max=1500.0, zones=zones, tier_shrink=0.78, n_tiers=6, r_min=6.0)
    return _lib("bumper_pit", make_world, seconds,
                Style(ball_palette=TIER_PALETTE),
                "energising bumpers (e > 1) over a 1x / 3x / 1x pit row")


def gear_cage(seconds: float = 15.0, seed: int = 31, **kw) -> Scene:
    """An octagon cage turning one way, a six-tooth gear turning the other inside it. Balls get
    trapped between a tooth and the wall and are squeezed out along the gap - a completely
    different escape mechanism from a rotating slot, and it looks mechanical rather than
    random."""
    def make_world():
        obs = build.polygon_shell(CX, CY, 425, 8, 17, missing=(3, 7), omega=+1.30,
                                  color=AMBER, name="oct")
        obs += build.spokes(CX, CY, 55, 300, 6, 14, omega=-2.10, color=CORAL)
        obs += [Peg(CX, CY, 42.0, e=0.9, color=ICE, name="hub")]
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1500.0, seed=seed,
                     r0=22.0, shrink=0.80, r_min=9.0, max_balls=16, split=2,
                     split_min_r=11.0, sensor_r=455.0)
    return _lib("gear_cage", make_world, seconds,
                Style(ball_palette=(ICE, LIME, CYAN, MINT), pad_color=AMBER),
                "counter-rotating octagon cage + 6-tooth gear")


def pendulum_rows(seconds: float = 15.0, seed: int = 37, **kw) -> Scene:
    """Three rows of swinging bars. The rows are deliberately NOT in phase - each bar's swing
    rate is offset a few percent from its neighbour's, so the row never re-aligns into a rigid
    comb and a falling ball meets a different arrangement every time."""
    def make_world():
        obs = build.pendulum_row(430, 5, 150, 930, 250, 14, amp=52 * DEG, freq=0.30,
                                 color=CYAN, name="p0")
        obs += build.pendulum_row(950, 4, 240, 840, 250, 14, amp=60 * DEG, freq=0.38,
                                  color=VIOLET, name="p1")
        obs += build.pendulum_row(1450, 5, 150, 930, 250, 14, amp=52 * DEG, freq=0.26,
                                  color=MAGENTA, name="p2")
        return World(obs, CX, CY, config.REF_W, config.REF_H, gravity=1450.0, seed=seed,
                     r0=19.0, max_balls=12, spawn_mode="top", spawn_every=0.55, initial=2,
                     respawn="fresh", bottom_triggers=True, sensor_r=0.0, ball_e=0.7)
    return _lib("pendulum_rows", make_world, seconds,
                Style(ball_palette=(ICE, AMBER, LIME, CORAL)),
                "3 rows of out-of-phase swinging bars")


SCENES = {
    "escape_rings": escape_rings,
    "hex_nest": hex_nest,
    "plinko": plinko,
    "paddle_wheel": paddle_wheel,
    "spiral_drop": spiral_drop,
    "orbit_swarm": orbit_swarm,
    "breakout_shell": breakout_shell,
    "funnel_cascade": funnel_cascade,
    "bumper_pit": bumper_pit,
    "gear_cage": gear_cage,
    "pendulum_rows": pendulum_rows,
}

LIBRARY = ["hex_nest", "plinko", "paddle_wheel", "spiral_drop", "orbit_swarm",
           "breakout_shell", "funnel_cascade", "bumper_pit", "gear_cage", "pendulum_rows"]
