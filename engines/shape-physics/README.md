# Shape Physics

**A rigid-body-lite engine for "satisfying shapes" clips** — discs, spinning shells with gaps,
destructible structures, collisions, gravity — **and audio that the simulation triggers.** That
last part is the inverse of how the rest of this repository works, and it is the reason the engine
exists.

📖 **[Object catalogue](CATALOG.md)** · 🖼 **[Reference frames](frames/)** · 💾 **[Source](src/)**

---

## The audio rule

Everywhere else in this repository, a song is analysed into a band track and the render *reads*
it. Here it runs the other way: the simulation fires events, and the events decide when the song
plays.

> Each event plays the **next** `slice` seconds of the song. An event arriving while a segment is
> already sounding **extends that segment** by another slice — it does not retrigger it.

Videos of this kind normally retrigger, so two events a few frames apart stack two copies of the
sample or hard-cut back to its start. Extending instead means an extended segment is **one
continuous read of the song** — no overlap, and no abrupt stop while events keep arriving. Song
offsets advance consecutively across segments, so the clip walks forward through the track however
the events clump.

`Score` is **causal**: it only ever looks at events already fired. That is what lets one object
serve both the on-screen "audio is live" indicator and the final track, instead of keeping two
schedules in step.

Segment edges get an 8 ms fade *inside* the segment, so an extended segment stays continuous while
an isolated cut does not click.

**Verify it by measuring the track, not by listening.** Take the per-100 ms RMS of the built WAV
against the segment list, and check the *minimum* level inside an extended segment — a zero there
means the extension broke into two reads. On the shipped clips that minimum is 0.013 and the peak
is 0.81, which a summed overlap would have pushed past 1.0.

## The physics

World units are **reference pixels (1080×1920, y down) and seconds**, whatever the output
resolution is. That, plus a physics rate independent of frame rate, is why a half-size preview at
30 fps and a full-size final at 60 fps are the same simulation, sampled differently.

Angles are radians from +x increasing toward +y, which matches both `atan2` and Pillow's arc
convention — **a physics angle is a drawing angle**, with no conversion anywhere.

Three primitives. Everything in the catalogue is built from them, and they all reduce to the same
two resolvers, which is why they interact correctly with each other for free:

| primitive | is | gives you |
|---|---|---|
| `Ring` | a hollow shell, thickness *h*, N gaps, optional spin | the escape-shell family |
| `Capsule` | a segment swept by a disc | polygons, funnels, chutes, spirals, paddles, gear teeth, pendulums, destructible bricks |
| `Peg` | a disc, optionally orbiting | plinko pins, bumpers, hubs, attractors |

**What a composition may and may not do**, carried over deliberately from the fluid engine: it
chooses geometry, materials and forcing — radius, gap width, spin rate, restitution, friction,
gravity, what spawns and when. It does **not** place a ball on a path. Every bounce, escape and
fall is solved. That is what makes the motion read as real.

## Quick start

```bash
pip install -r src/requirements.txt   # numpy + Pillow, nothing else
```

```python
import sys; sys.path.insert(0, "src")
from sim import world, build, audio

obstacles  = build.polygon_shell(540, 960, 210, n_sides=6, thickness=14., missing=(0,), omega=1.2)
obstacles += build.walls(60, 1020, 100, 1820)

w = world.World(obstacles, cx=540, cy=960, width=1080, height=1920, gravity=1500., initial=1)
for _ in range(2 * 900):          # 2 s at the 900 Hz physics rate
    w.step(1 / 900.)

print(len(w.bounces), "bounces,", len(w.triggers), "headline events")

score = audio.Score(slice_len=0.5)
for t in (1.0, 1.2, 3.0):
    score.fire(t)
print(len(score.segments), "segments")   # 2 — the first two events merged
```

**Run the simulation with no drawing before you render anything.** It is about 0.3 s for a 20 s
clip, and it prints the escape counts, every trigger time and the resulting audio segments. Tune
gap width, spin, friction and gravity against *that*, never against renders. A composition whose
brief is "balls escape and fire sounds" needs a check that measures exactly that.

## Eleven rules that are load-bearing

Each is a bug if you undo it.

1. **Bounces resolve in the WALL's frame** — subtract the wall velocity, reflect, add it back. A
   spinning shell must *fling* a ball, not merely stop it. With zero friction, or with the
   reflection done in the world frame, balls rattle to the bottom and sit there waiting for a gap.
2. **Every gap edge is a real object.** Each edge carries a cap circle of radius *h*, collided
   against every substep whether or not the ball is in the gap. Without it a ball clipping the
   edge slides through the shell wall, and a solid arc sweeping onto a half-escaped ball has
   nothing to shove it back with.
3. **A gap must clear the BALL, not a point.** A ball of radius *r* at shell radius *R* needs
   `2·asin(r/R)` of arc just to fit through.
4. **Radial early-out before the cap loop.** Without it every ball tests every cap every substep
   and the simulation is about 5× slower for nothing.
5. **Floats, not numpy, in the inner loop.** Roughly a million substep-body iterations for a 20 s
   clip, and a numpy scalar operation costs about 10× a float multiply. Rewriting `Ball` "cleanly"
   with arrays makes a 0.3 s simulation take 15 s.
6. **A rotating spiral is an Archimedes SCREW, and only one sign conveys outward.** Measured over
   15 s: one sign gave 16 escapes, the other gave **zero**. A spiral that does not rotate at all is
   worse than either — every coil has a local low point, a ball parks in it, and the clip dies
   after two seconds. If a chute goes dead, check the sign before touching anything else.
7. **Read spin rates against the CLIP LENGTH, not against taste.** A hexagon whose one missing side
   passes the bottom of frame once every 11 s means a ball simply waits through a 15 s clip. An
   opening has to pass the low point several times.
8. **A pit row meant to catch everything must TILE.** With radii below half the pit spacing, one
   composition caught 3 of 34 balls — the rest poured down the gaps between dividers.
9. **A SLIDING capsule can pinch a ball and fire it out of frame.** Measured bottom exits ran from
   −1158 to +2633 on a 1080-wide frame. Keep sway peak speed modest and contain the play area with
   side walls, which a real rig has anyway. Diagnose by printing the *x* of each bottom exit.
10. **Read `scatter_chance` against the PAIR-HIT count.** Ball-ball impacts are far rarer than
    bounces (62 of 179 in one clip), so 0.10 produced literally zero scatters where 0.35 gives
    about nine.
11. **Balls are never removed on the frame they die.** Deaths set a flag, spawns happen after the
    per-ball loop, and the list is compacted once — mutating it mid-iteration silently skips
    collision pairs.

## Population control, and why a cap is not enough

Splitting one ball into two on every exit is **exponential in the number of escape cycles**, and
shrinking the children makes it *worse*, not better: a smaller ball clears a gap more easily, so
it escapes sooner. Measured at 45 s:

| setting | triggers | live at end | spawns dropped by the cap | music sounding |
|---|---|---|---|---|
| split 2, cap 28 | 116 | 28 | **85** | 100% (one 51 s segment) |
| split 2, cap 60 | 189 | 60 | **130** | 100% (one 88 s segment) |
| split 1 | 7 | 1 | 0 | 8% — never builds at all |
| **split 2, `split_min_r` = 12** | **46** | **8** | **0** | **51%** |

So the cap is not a fix — raising it makes the runaway worse, and either way it silently drops
most of the spawns the brief asked for. Bound the population by **geometry** instead: a ball
splits only while its children would still be big enough to matter. The count then plateaus on its
own and the cap never bites.

This is also what keeps the AUDIO eventful. With an unbounded population the escape rate outruns
the slice length within about twenty seconds, and `Score` — correctly, per its own rule — extends
one segment over the entire rest of the clip. **Punctuation is a property of the physics, not of
the audio code.**

## The tier ladder

One size and colour ladder shared by every mechanic that makes a ball smaller: **white → blue →
red → purple → green → yellow**. `Ball.gen` *is* the tier index and the renderer colours by it, so
a composition names its tiers just by ordering its palette.

Three mechanics walk it and they compose: a **multiplier pit** consumes a ball and returns N one
tier down; a **scatter** may fragment a colliding pair; a **recombine** fuses two of the smallest
tier into one of the tier above. Recombination is restricted to the smallest tier on purpose —
that is what makes the ladder a **cycle** rather than a one-way slide to dust. A fourth mechanic,
**decay**, promotes a ball that has lived long enough, in place, so it visibly grows and recolours
instead of teleporting.

Area is conserved on a fuse (`r = sqrt(ra² + rb²)`), and a scatter carries the pair's momentum
across plus a symmetric outward kick whose vector sum is about zero — internal energy in, no net
drift, which is what a real fragmentation does.

## Two mechanisms worth lifting whole

**Orbital planes without a switch.** Give each tier a depth offset and let a pair collide only
when `d_2d² < (ra + rb)² − dz²`. The effective contact radius shrinks as depth grows and reaches
zero at `dz = ra + rb`, so ramping the offset makes cross-tier hits first glancing, then rare,
then impossible. Nothing is faked and nothing is disabled — the pass-through emerges from the
geometry, which is exactly what makes it rampable. Same-tier pairs share a plane and collide as
before. Plane gaps must exceed each pair's radius *sum*, and draw order must become depth order.

**Interaction by heading, not by contact.** `p = ((cos(angle between velocities) + 1)/2)^exp` —
head-on pairs pass through, rear-end pairs always fragment. ⚠️ The roll must happen **once per
contact episode, not per substep**: a touching pair stays overlapped for many substeps, and
re-rolling each step turns any per-step chance into a near-certainty — 32,525 interactions in one
30 s clip before this was guarded. Fragments are also born overlapping and must be pre-marked as
already in contact, with a short spawn immunity on top.

## Performance

Measured on an RTX 2060 / i7-9700K. The cost is **Pillow vector rasterisation**, not array maths —
a 20 s preview is 37 s wall clock at about 60 ms/frame, and the two bloom blurs are a few
milliseconds of it. There is no GPU path and no need for one. Bloom runs on the **downsampled**
frame, not the supersampled buffer: the glow is low-frequency, so it is visually identical and
about 4× cheaper.

## Orientation

World units are reference pixels with a scale factor to output pixels, so any output resolution is
one number away. A landscape world means different centre coordinates and different structures —
the builders all take explicit geometry — not different code. The one thing tied to the vertical
frame is the default anisotropy used to stretch orbital fields into 9:16; set it to 1.0 for
isotropic, or to your own aspect.

## What is not here

- **`sim/scenes.py`** — the twelve compositions. See [CATALOG.md](CATALOG.md) for what each one's
  headline event is, its measured event and bounce counts, and frames.
- **Audio files.** `SONGS` is empty and `SONGS_DIR` points at a `songs/` directory beside the
  source. Drop your own files there, or pass an absolute path.
