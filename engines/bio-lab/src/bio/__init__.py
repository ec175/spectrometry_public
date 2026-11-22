# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""The BIO engine - 2-D cell colony growth, coloured the way a confocal stack is coloured.

WHAT THIS IS TRYING TO LOOK LIKE
--------------------------------
A real clip of intestinal epithelial cells from an organoid (HHMI / Matthew Tyska,
Vanderbilt - see `reference/docs/CELL_REFERENCE.md` for the link, the credit and every
number below). Two measurements off that clip decide almost everything here:

1. **THE GLOW IS THE MEMBRANE, NOT THE CELL.** Cell interiors are near-black - measured
   mean value 35/255 - and every bright, saturated pixel is the junctional network
   BETWEEN cells. Measured edge-to-interior brightness ratio: 3.3 to 3.7x.

   If you take one thing from this file, take that. Painting bright cells on a dark
   background is the obvious way to draw a cell colony and it looks nothing like this;
   the source looks the way it does because it is a dark field with a lit MESH over it.

2. **THE COLOUR RAMP SKIPS BLUE.** Hue weight in the 210-270 degree band measured 0.4%,
   1.0% and 1.3% across three frames. See `kit/colour.py`, which carries the ramp and a
   test that keeps it honest.

The source is a frame-filling sheet; this engine draws a FINITE COLONY on black, which
is what was asked for, so surface tension and outside pressure genuinely shape the
visible outline rather than only the interior.


COLOUR DRIVERS
--------------
Three, selectable. All of them run through the same measured ramp, so switching driver
changes what the colour MEANS without changing the palette:

* `pressure` - how crowded a cell is. Raising outside pressure sweeps the whole colony up
  the ramp at once, which is the most dramatic of the three.
* `depth`    - a synthetic dome: the colony is treated as a curved cap and colour is
  height. The most faithful to the source, and the smoothest, but it is decorative - it
  does not respond to the physics.
* `age`      - time since the cell last divided, blended with its lineage tag. Division
  events flash and growth fronts read as colour waves moving outward.
"""
from __future__ import annotations

import numpy as np

from kit import colour as C
from kit.params import P, coerce_all, hex_to_rgb

from .sim import Colony
from .tess import polygon_area

def hash01(seed, k):
    """Pseudo-random floats in [0, 1) from (per-cell seed, slot).

    Vectorised, so a whole frame's organelle assortment costs no per-cell Python. The
    unsigned wraparound IS the mixing step, hence the errstate.
    """
    with np.errstate(over="ignore"):
        x = (np.asarray(seed, np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             ) ^ (np.uint64(k + 1) * np.uint64(0xBF58476D1CE4E5B9))
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


NAME = "bio"
TITLE = "Bio"
BLURB = "Cell colony growth: tension, pressure, division, rich depth colour."

#: THE DEFAULTS BELOW ARE CALIBRATED, NOT CHOSEN - and they are HONESTLY NOT ALL IN RANGE.
#:
#: `tools/look_check.py` measures a render with the same code that measured the reference
#: clip. Re-measured after the v3 draw stack (the film and the membrane are one analytic
#: field now - see kit/cellfield.py), at 720p horizontal with the shipped defaults:
#:
#:     t=9s   26-85 cells   sat 0.585   e/i 2.87   p99 180   blue 1.8
#:     t=13s  216 cells     sat 0.420   e/i 2.80   p99 188   blue 7.9
#:     targets              sat .45-.60 e/i 3.0-4.0 p99 120-175 blue <=3.0
#:
#: So at the density the reference actually shows (its envelope says 60-200 cells in
#: frame) saturation is now INSIDE the band, blue is well inside it, and edge contrast and
#: p99 are near-misses. The v3 stack halved the blue miss - the old draw painted every
#: shared junction twice, once from each side, and manufactured blue out of cyan plus
#: magenta; the field paints each cell only inside its own wall, so that cannot happen.
#:
#: TWO THINGS TO KNOW BEFORE YOU "FIX" THE REMAINING MISSES.
#:
#: 1. **The numbers drift with CELL COUNT, and t=13 is outside the reference regime.** At
#:    216 cells the colony is denser than anything the envelope was measured on, bloom
#:    starts bridging neighbouring membranes, and saturation falls while blue rises. That
#:    is the brief (a colony that grows) meeting a target measured from a static sheet.
#:
#: 2. **The contrast and p99 misses are bloom filling the cell interiors.** Dropping
#:    `glow_gain` below about 0.06 closes them, at the cost of the membranes no longer
#:    reading as light. That trade was made toward the look, deliberately.
#:
#: If you change any of `wall_gain`, `film_gain`, `film_profile`, `interior`,
#: `colour_smooth`, `tone_gamma`, the glow gains OR anything in the Growth group, RE-RUN
#: `python tools/look_check.py` and update this block. A stale measurement in a comment is
#: worse than no measurement - an earlier version of it quoted numbers taken before
#: `heterogeneity` existed.
PARAMS = [
    # --- Growth ------------------------------------------------------------------------
    P("growth_rate", "float", 0.28, 0.0, 1.5, group="Growth",
      help="Exponential growth rate of a cell's TARGET area, per second. This is the "
           "master pace of the whole scene."),
    P("divide_at", "float", 1.9, 1.1, 4.0, group="Growth",
      help="A cell divides once its target area reaches this multiple of the base area."),
    P("cycle_min", "float", 0.7, 0.0, 6.0, group="Growth",
      help="Minimum seconds between divisions, whatever the area says. Stops a burst of "
           "cells all dividing on the same frame."),
    P("inhibit", "float", 0.34, 0.0, 1.5, group="Growth",
      help="CONTACT INHIBITION: a cell will not divide if its local pressure exceeds "
           "this. Pressure is a DIMENSIONLESS mean fractional overlap with neighbours, "
           "typically 0.05 loose to 0.5 jammed. Real biology, and also what stops the "
           "population exploding."),
    P("inhibit_growth", "float", 1.2, 0.0, 8.0, group="Growth",
      help="How much crowding slows growth. High values make the colony interior go "
           "quiet while the rim keeps expanding."),
    P("division_time", "float", 1.0, 0.0, 4.0, group="Growth",
      help="Seconds a cell takes to pinch in two, at the reference cell size; scales with "
           "radius, so big cells divide slowly and visibly and small ones snap. 0 is the "
           "old instantaneous split."),
    P("size_pareto", "float", 2.2, 0.0, 6.0, group="Growth",
      help="Pareto tail index for per-cell size. Lower is a heavier tail - mostly small "
           "cells and a few much larger ones, which is what a real culture looks like. "
           "0 turns it off. The spread is normalised so `cell_area` stays the MEAN."),
    P("size_max", "float", 6.0, 1.0, 20.0, group="Growth",
      help="Cap on the size multiplier, as a multiple of cell_area. A Pareto tail is "
           "unbounded and one 40x cell would own the frame."),
    P("size_drift", "float", 0.06, 0.0, 0.5, group="Growth",
      help="How much a daughter's size may differ from its parent's. Small values make "
           "size a LINEAGE trait rather than a per-cell lottery."),
    P("size_drag", "float", 2.6, 0.0, 5.0, group="Mechanics",
      help="Exponent on cell radius in the drag. THE MASS-TRANSFER KNOB - but the "
           "interesting value is higher than it looks. Measured, path length over 3 s, "
           "small tercile / large tercile: 0.0 -> 0.64, 1.0 -> 0.77, 1.5 -> 0.86, "
           "2.0 -> 0.93, 2.5 -> 1.01, 3.0 -> 1.17. Big cells move MORE by default (the "
           "contact force on them scales with radius), so 1.0 only cancels that; small "
           "cells do not actually start flowing between the large ones until about 2.45. "
           "2.0 is also the physical reading - friction proportional to contact area."),
    P("drag_floor", "float", 0.12, 0.01, 1.0, group="Mechanics",
      help="Lower bound on drag, as a fraction of `viscosity`. `size_drag` is a power "
           "law and it runs away at the small end - without a floor the smallest cells "
           "are flung by contact faster than any prescribed motion can steer them."),
    P("heterogeneity", "float", 0.22, 0.0, 1.2, group="Growth",
      help="Spread of per-cell growth rates, log-normal and INHERITED. 0 makes every "
           "cell identical, which divides the colony in synchronised waves and looks "
           "like a diagram. Around 0.4 gives the varied cell sizes real tissue has."),
    P("seed_cells", "int", 12, 1, 200, group="Growth",
      help="How many cells the colony starts from."),
    P("cell_area", "float", 2600.0, 300.0, 20000.0, group="Growth",
      help="Base cell area in square reference units. THIS IS THE ZOOM CONTROL: smaller "
           "cells means more of them on screen. The reference clip, zoomed out as asked, "
           "is around 2000-4000."),
    P("max_cells", "int", 700, 10, 3000, group="Growth",
      help="Hard population cap. Cost per frame is roughly linear in this."),

    # --- Mechanics ---------------------------------------------------------------------
    P("tension", "float", 0.38, 0.0, 3.0, group="Mechanics",
      help="SURFACE TENSION. Smooths the colony outline by pulling rim cells toward "
           "their neighbours. High tension gives a round colony; low tension lets it "
           "grow lobes and fingers."),
    P("pressure", "float", 0.35, 0.0, 4.0, group="Mechanics",
      help="OUTSIDE PRESSURE, pushing inward on the rim only. Compacts the colony and "
           "raises the pressure colouring everywhere. Different from tension: this "
           "squeezes, tension smooths."),
    P("adhesion", "float", 1.1, 0.0, 6.0, group="Mechanics",
      help="How strongly cells hold on to each other once separated. Too low and the "
           "colony sheds cells; too high and it never expands."),
    P("repulsion", "float", 3.2, 0.2, 20.0, group="Mechanics",
      help="Resistance to overlap - cell incompressibility."),
    P("contact", "float", 1.0, 0.5, 1.8, group="Mechanics",
      help="Multiplier on the touching distance. Below 1 cells interpenetrate before "
           "pushing back, which reads as soft, squashy cells."),
    P("viscosity", "float", 1.0, 0.05, 8.0, group="Mechanics",
      help="Damping. Everything is overdamped - there is no inertia - so this sets how "
           "fast the colony responds, not how much it wobbles."),
    P("motility", "float", 0.12, 0.0, 2.0, group="Mechanics",
      help="Random cell motion. A little keeps a jammed colony breathing; a lot "
           "fluidises it and cells start swapping neighbours."),

    # --- Colour ------------------------------------------------------------------------
    P("driver", "choice", "pressure", choices=["pressure", "depth", "age"],
      group="Colour", help="What the colour MEANS. See the module docstring."),
    P("palette", "choice", "depth_lut", choices=sorted(C.RAMPS), group="Colour",
      help="depth_lut is measured off the reference clip and deliberately skips blue."),
    P("palettes", "text", "", group="Colour",
      help="One ramp name per CULTURE, comma separated, e.g. "
           "'polar_bloom,polar_amber,polar_rich'. Each strain then samples its OWN ramp. "
           "Hue-rotating one shared ramp cannot express two cultures that are meant to "
           "keep their own colour identity - it only shifts the same ramp around the "
           "circle. Empty falls back to `palette` plus `strain_hue_spread`."),
    P("colour_gamma", "float", 1.0, 0.25, 4.0, group="Colour",
      help="Biases where values land on the ramp without rescaling the driver."),
    P("colour_lo", "float", 0.0, 0.0, 1.0, group="Colour",
      help="Low end of the driver's mapped range."),
    P("colour_hi", "float", 1.0, 0.0, 1.0, group="Colour",
      help="High end. Set below the low end to reverse the ramp."),
    P("lineage_split", "float", 0.35, 0.0, 1.0, group="Colour",
      help="Chance that a daughter cell takes a NEW lineage tag rather than inheriting "
           "its parent's. Only affects the 'age' driver. Around 0.3 gives contiguous "
           "patches of common descent, which is what real lineage tracing looks like; "
           "1.0 gives confetti."),
    P("colour_stretch", "float", 0.8, 0.0, 1.0, group="Colour",
      help="Stretch the driver's own value distribution to span the whole ramp before "
           "colouring. Without it a driver whose values happen to cluster - which is most "
           "of them, most of the time - uses a narrow slice of the palette and the scene "
           "comes out one colour. 0 uses raw driver values; 1 always spans the full ramp."),
    P("colour_tau", "float", 1.2, 0.0, 6.0, group="Colour",
      help="Seconds for a cell's colour to catch up with its physics. The driver is "
           "recomputed every frame through a percentile stretch, so without this the whole "
           "field can shift hue the moment the population changes, and a newborn cell "
           "arrives at a colour of its own. Daughters inherit the smoothed value."),
    P("colour_smooth", "float", 0.72, 0.0, 1.0, group="Colour",
      help="Blend each cell's colour toward its neighbours' mean. The source's colour "
           "is a SMOOTH SPATIAL FIELD, not per-cell noise - this is what buys that."),

    # --- Membrane (the look) ------------------------------------------------------------
    P("edge_width", "float", 7.0, 0.5, 40.0, group="Membrane",
      help="Junction line width in reference units. The reference measures about 8.5 px "
           "at 720 wide including its glow, i.e. roughly 23 reference units."),
    P("edge_gain", "float", 0.0, 0.0, 12.0, group="Membrane",
      help="Membrane brightness. The single biggest look control."),
    P("vertex_gain", "float", 0.0, 0.0, 8.0, group="Membrane",
      help="Extra brightness at tri-cellular junctions, which are visibly the brightest "
           "points in the reference."),
    P("interior", "float", 0.05, 0.0, 0.6, group="Membrane",
      help="Cell interior brightness as a fraction of its membrane colour. The measured "
           "ratio is 1/3.5, so about 0.10-0.14 keeps the reference's contrast."),
    P("interior_texture", "float", 0.35, 0.0, 1.0, group="Membrane",
      help="Mottling inside cells - the reference has visible speckle from the "
           "fluorophore, and a perfectly flat interior reads as vector art."),
    P("opacity_falloff", "float", 0.55, 0.0, 2.0, group="Membrane",
      help="Fades membranes toward the colony rim, as if the tissue curves away. Set to "
           "0 for a flat, evenly lit sheet."),

    # --- Bubble (the film that makes cells read as 3-D) ---------------------------------
    P("film_gain", "float", 0.70, 0.0, 5.0, group="Bubble",
      help="Brightness of the surface film over each cell. This is the layer that makes "
           "a cell look like a dome bulging toward you rather than a flat tile."),
    P("film_size", "float", 1.0, 0.4, 1.0, group="Bubble",
      help="Film extent as a fraction of the cell's own wall. 1.0 fills the cell exactly. "
           "Capped at 1.0 on purpose: the film is CLIPPED TO THE CELL, so a value above 1 "
           "would put a cell's light on the far side of its neighbour's wall."),
    P("wall_soft", "float", 0.10, 0.0, 0.35, group="Bubble",
      help="Corner rounding, as a fraction of the cell radius. Applied as a SMOOTH "
           "minimum in the film field, so it rounds corners without moving the walls - "
           "neighbours keep sharing their wall and no junction void can open."),
    P("wall_gain", "float", 0.85, 0.0, 8.0, group="Bubble",
      help="Membrane brightness drawn BY THE FIELD, so it follows the rounded wall "
           "exactly. Prefer this over edge_gain, which is straight PIL segments."),
    P("wall_width", "float", 6.0, 0.5, 40.0, group="Bubble",
      help="Membrane thickness in reference units - absolute, not scaled by cell size."),
    P("wall_round", "int", 0, 0, 3, group="Bubble",
      help="DEPRECATED, ignored. Rounding is `wall_soft` now. Was: "
           "(fine when cells are small); 2-3 gives smooth curved walls for close-ups, at "
           "4x the vertices per pass."),
    P("film_rings", "int", 20, 3, 32, group="Bubble",
      help="DEPRECATED, ignored. The film is an analytic field now, so it is smooth at "
           "every zoom and there are no steps to count."),
    P("film_profile", "choice", "lens", choices=["shell", "dome", "refract", "lens"],
      group="Bubble",
      help="shell: dim centre brightening to a rim, the cross-polarised look. "
           "dome: full and soft. refract: a narrow bright band hard against the wall over "
           "a near-black interior. lens: the same idea with a BAND wide enough to read - "
           "the close-up profile."),
    P("film_hue_shift", "float", 0.0, -60.0, 60.0, group="Bubble",
      help="Rotate the film's hue away from the membrane's, in degrees. A small offset "
           "reads as the film and the junction being different materials."),

    # --- Nucleus ------------------------------------------------------------------------
    P("nucleus_gain", "float", 2.0, 0.0, 8.0, group="Nucleus",
      help="Brightness of the glowing point at each cell's centre."),
    P("nucleus_size", "float", 0.21, 0.04, 0.6, group="Nucleus",
      help="Nucleus radius as a fraction of the cell radius."),
    P("nucleus_hue", "float", 0.55, 0.0, 1.0, group="Nucleus",
      help="How far each nucleus's hue may wander from its cell's. 0 makes every "
           "nucleus the colour of its cell; 1 gives every cell its own distinct point."),
    P("nucleus_drift", "float", 0.06, 0.0, 0.4, group="Nucleus",
      help="How much a daughter's nucleus hue differs from its sister's. Small values "
           "keep a lineage recognisable while still making a split visible."),
    P("division_flash", "float", 2.2, 0.0, 8.0, group="Nucleus",
      help="Extra nucleus brightness immediately after a division, decaying over "
           "`bridge_time`. This is what makes a split read as an EVENT."),
    P("bridge_time", "float", 1.0, 0.0, 2.0, group="Nucleus",
      help="How long the midbody between two separating sisters lasts, as a MULTIPLE of "
           "that cell's own `division_time` - not seconds. Tied to it so the bridge and "
           "the division flash track the pinch instead of outlasting it on a big cell "
           "and being over before it starts on a small one."),
    P("bridge_gain", "float", 1.8, 0.0, 8.0, group="Nucleus",
      help="Brightness of that bridge."),

    # --- Detail (close-up scenes) ---------------------------------------------------------
    P("granules", "int", 0, 0, 16, group="Detail",
      help="ORGANELLES per cell: a soft blurry chunk plus a small glowing dot, both "
           "dimmer than the nucleus. Position, size and brightness come from a per-cell "
           "seed, so no two cells share an assortment, and BOTH daughters get a fresh "
           "seed at division - a shared one smears the pattern across the cytokinesis."),
    P("granule_gain", "float", 0.55, 0.0, 3.0, group="Detail",
      help="Brightness of the blurry chunk."),
    P("granule_dot", "float", 0.35, 0.0, 3.0, group="Detail",
      help="Brightness of the glowing dot at each organelle. Keep it under "
           "`nucleus_gain` or the cell reads as having a dozen nuclei."),
    P("granule_size", "float", 0.075, 0.02, 0.4, group="Detail",
      help="Organelle size as a fraction of the cell radius. Randomised per organelle "
           "about this."),
    P("granule_spread", "float", 0.52, 0.1, 0.9, group="Detail",
      help="How far from the centre organelles may sit, as a fraction of cell radius. "
           "A fraction, so they stay inside a cell that grows or shrinks."),

    # --- Delivery (v4: producers, couriers, and the culture outside the frame) ----------
    P("producer_frac", "float", 0.0, 0.0, 0.3, group="Delivery",
      help="Fraction of cells that BUD small motile cells. Scattered over the whole "
           "seeded culture, so most of them are off screen."),
    P("emit_period", "float", 2.0, 0.2, 10.0, group="Delivery",
      help="Seconds between buds, jittered per event."),
    P("courier_frac", "float", 0.06, 0.0, 0.3, group="Delivery",
      help="A courier's area as a fraction of its parent's. The area comes OUT of the "
           "parent, so a producer that keeps emitting visibly shrinks."),
    P("courier_speed", "float", 90.0, 0.0, 400.0, group="Delivery",
      help="How fast a courier swims toward its destination, in ru/s. Applied as a "
           "VELOCITY, not a force: a courier's drag is tiny, so a force big enough to "
           "push it past a wall of large cells is big enough to make the timestep decide "
           "where it lands."),
    P("courier_adhesion", "float", 0.1, 0.0, 1.0, group="Delivery",
      help="Adhesion multiplier for any pair involving a courier. Near 1 they glue to the "
           "first membrane they touch and never arrive."),
    P("courier_life", "float", 8.0, 0.0, 30.0, group="Delivery",
      help="Seconds a courier lasts before it dissolves. Bounds the population: without "
           "it the count is a race between the bud rate and the travel time to the rim, "
           "and it walks into `max_cells` and stops every division in the scene."),
    P("courier_spread", "float", 60.0, 0.0, 400.0, group="Delivery",
      help="How far off its destination a courier aims, in reference units. Zero and they "
           "all steer at one point and pile up there; a spread turns the approach into a "
           "lane."),
    P("courier_squeeze", "float", 0.35, 0.05, 1.0, group="Delivery",
      help="Contact distance for any pair involving a courier, relative to normal. At 1 "
           "a courier is a hard disc that has to displace a large cell to pass, and it "
           "simply parks at the nearest junction where the repulsions cancel. Low values "
           "let it thread the interstices."),
    P("courier_offscreen", "float", 0.5, 0.0, 1.0, group="Delivery",
      help="Fraction of couriers sent to an off-screen attractor rather than to a cell "
           "in frame. The ones that leave are what make the shot feel like a window."),
    P("attractors", "int", 3, 0, 8, group="Delivery",
      help="Destinations on the seed rim. Keep it to two or three: shared destinations "
           "make couriers form visible LANES threading between the big cells, which is "
           "the shot. A random target per courier reads as noise."),
    P("recipients", "int", 3, 0, 12, group="Delivery",
      help="Cells that couriers feed. Chosen from the innermost cells, so the one that "
           "visibly swells is one the frame is looking at."),
    P("producer_speed", "float", 0.0, 0.0, 150.0, group="Delivery",
      help="A producer's own crawl, ru/s along its heading. This, not the bulk flow, is "
           "what carries producers off screen: bulk flow at a plausible speed crosses "
           "only a quarter of the frame in eighteen seconds."),
    P("cull_radius", "float", 0.0, 0.0, 3.0, group="Delivery",
      help="Cells past this multiple of the seed ellipse are gone for good. Bounds the "
           "population once the culture is larger than the frame. 0 disables."),
    P("flow_speed", "float", 0.0, 0.0, 120.0, group="Delivery",
      help="Bulk tissue flow, ru/s."),
    P("flow_angle", "float", 90.0, 0.0, 360.0, group="Delivery",
      help="Direction of the bulk flow, degrees."),
    P("flow_shear", "float", 0.0, -80.0, 80.0, group="Delivery",
      help="Velocity gradient across the flow. Uniform flow alone is only a camera pan; "
           "shear is what makes neighbouring cells actually slide past each other."),
    P("courier_glow", "float", 1.8, 0.5, 5.0, group="Delivery",
      help="Brightness multiplier on couriers, so a cell a twentieth the area of its "
           "neighbours still reads."),

    # --- Clusters -------------------------------------------------------------------------
    P("seed_mode", "choice", "blob", choices=["blob", "filled"], group="Clusters",
      help="blob: a few founder cells that grow into the frame. filled: the frame starts "
           "FULL of mature interlocking cultures, in contact from the first frame."),
    P("clip_scale", "float", 1.50, 1.0, 2.6, group="Bubble",
      help="How far a cell may fill its Voronoi region, as a multiple of its own radius. "
           "Low values leave black margins around cells whose region is larger than they "
           "are - common once sizes vary - and round off the colony's rim. Raise it for a "
           "frame-filling sheet, where the rim is off screen anyway."),
    P("seed_aspect", "float", 1.0, 0.2, 3.0, group="Clusters",
      help="Height/width of the seeded region when seed_mode is 'filled'. 1 is a disc; "
           "1.78 matches a 9:16 frame, which is what stops black wedges appearing at top "
           "and bottom of a vertical clip that is meant to open full."),
    P("settle", "float", 0.0, 0.0, 6.0, group="Clusters",
      help="Seconds of pre-roll before frame 0, with growth and division off. A `filled` "
           "seed is a lattice at one pitch; with per-cell sizes it is a mis-packed state, "
           "and without a settle the first seconds of the clip are that relaxation. Also "
           "what 'starts already developed' actually means - walls deformed at frame one."),
    P("seed_radius", "float", 700.0, 200.0, 1500.0, group="Clusters",
      help="Radius seeded when seed_mode is 'filled', in reference units."),
    P("strain_size_spread", "float", 0.0, 0.0, 0.8, group="Clusters",
      help="Spread of per-culture CELL SIZE. Different-sized cells is most of what makes "
           "two cultures read as different organisms rather than two colours."),
    P("affinity_spread", "float", 0.0, 0.0, 0.9, group="Clusters",
      help="Spread of per-culture affinity for its OWN kind. High-affinity cultures round "
           "up and refuse to mix; low-affinity ones spread and interdigitate."),
    P("cross_affinity", "float", 1.0, 0.0, 2.0, group="Clusters",
      help="Adhesion between DIFFERENT cultures, relative to within one. Below ~0.5 the "
           "interface goes sharp and taut and cultures sort out; near 1 they mix."),
    P("clusters", "int", 1, 1, 8, group="Clusters",
      help="Independent colonies, each its own STRAIN with its own colour and growth. "
           "More than one and they grow into each other and compete for space."),
    P("cluster_spread", "float", 520.0, 80.0, 900.0, group="Clusters",
      help="Radius of the ring the clusters are seeded on, in reference units."),
    P("strain_hue_spread", "float", 110.0, 0.0, 180.0, group="Clusters",
      help="How far apart the strains' colours are, in degrees of hue. Zero makes them "
           "identical twins fighting invisibly."),
    P("strain_growth_spread", "float", 0.20, 0.0, 0.8, group="Clusters",
      help="Spread of per-strain growth rates. This decides whether the clusters are "
           "evenly matched or one of them takes the frame."),

    # --- Stage -------------------------------------------------------------------------
    P("bg_colour", "colour", "#040306", group="Stage",
      help="Near-black, not black. The reference's background measures (23, 20, 20)."),
    P("bloom_threshold", "float", 64.0, 0.0, 255.0, group="Stage"),
    P("glow_sigma", "float", 5.0, 0.0, 40.0, group="Stage"),
    P("glow_gain", "float", 0.10, 0.0, 2.0, group="Stage"),
    P("glow2_sigma", "float", 34.0, 0.0, 160.0, group="Stage"),
    P("glow2_gain", "float", 0.045, 0.0, 2.0, group="Stage"),
    P("tone_gamma", "float", 2.30, 0.0, 3.0, group="Stage",
      help="Contrast. Above 1 crushes the low end and keeps the interiors dark while "
           "the membranes stay bright."),
    P("vignette", "float", 0.22, 0.0, 1.0, group="Stage"),
    P("grain", "float", 0.006, 0.0, 0.06, group="Stage"),
    P("drift", "float", 0.0, 0.0, 60.0, group="Stage",
      help="Sway AMPLITUDE in reference units - the colony oscillates gently about centre. "
           "Not a rate: it never accumulates, so the subject cannot walk out of frame."),
]


def _clean_ramps(ramps, params):
    """Validate a scene's `ramps` block. Unknown or malformed keys are REFUSED, not dropped.

    The rest of the scene format drops an unrecognised key with a warning, which is right
    for a parameter that has a default. A ramp has no default: the scene asked for something
    to change over time and if the key is wrong NOTHING changes and the clip renders as a
    still-looking version of itself, which passes every check except a human watching it.
    """
    if not ramps:
        return {}
    out = {}
    for key, knots in dict(ramps).items():
        if key not in params:
            raise KeyError("ramps: no such parameter %r" % key)
        pts = [(float(a), float(b)) for a, b in knots]
        if len(pts) < 2:
            raise ValueError("ramps[%r] needs at least two knots" % key)
        out[key] = sorted(pts)
    return out


def _ramp_at(knots, t):
    """Piecewise SMOOTHSTEP between knots, held flat outside them.

    Smoothstep rather than linear because every gate in this library hard-fails a velocity
    jump, and a linear ramp has a corner in its derivative at every knot -- which is exactly
    the un-eased transition those checks exist to catch.
    """
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    for (t0, v0), (t1, v1) in zip(knots, knots[1:]):
        if t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            return v0 + (v1 - v0) * (u * u * (3.0 - 2.0 * u))
    return knots[-1][1]


#: Scenes may carry a `ramps` block; `kit.scene.build_engine` only passes it to engines
#: that advertise it, so a second engine without the facility is unaffected.
SUPPORTS_RAMPS = True


class Engine:
    NAME = NAME
    PARAMS = PARAMS

    def __init__(self, params=None, seed=7, ramps=None):
        self.p = coerce_all(PARAMS, params or {})
        self.seed = int(seed)
        self.ramps = _clean_ramps(ramps, self.p)
        self.cfg = None
        self.canvas = None
        self.colony = None
        self._rng = np.random.default_rng(self.seed + 991)
        self._col_t = 0.0
        self._stretch = None

    # --- lifecycle ----------------------------------------------------------------------
    def bind(self, cfg, canvas):
        self.cfg = cfg
        self.canvas = canvas
        # Compile the field and splat kernels now. NVRTC's first compile is seconds, and
        # three render lanes hitting it mid-frame stall three lanes.
        from kit import cellfield as _cf, splats as _sp
        _cf.warm()
        _sp.warm()

    def background_rgb(self):
        return hex_to_rgb(self.p["bg_colour"])

    def post(self):
        p = self.p
        return {"bloom_threshold": p["bloom_threshold"],
                "glow_sigma": p["glow_sigma"], "glow_gain": p["glow_gain"],
                "glow2_sigma": p["glow2_sigma"], "glow2_gain": p["glow2_gain"],
                "tone_gamma": p["tone_gamma"], "vignette": p["vignette"],
                "grain": p["grain"], "grain_seed": self._grain_seed()}

    def reset(self):
        self.colony = Colony(dict(self.p, _seed=self.seed), seed=self.seed)
        self._col_t = 0.0
        self._stretch = None
        self._settle()

    def step(self, dt):
        self._apply_ramps()
        self.colony.step(dt)

    def _apply_ramps(self):
        """Move any time-varying parameters to their value at the colony's current time.

        ⚠ Written onto `self.colony.p`, not `self.p`. The colony is handed a COPY of the
        parameter dict at construction (`Colony(dict(self.p, _seed=...))`), so a scene that
        ramped `self.p` would change nothing at all and do it silently -- which is the worst
        possible failure for a facility whose entire job is to make something visibly change.
        """
        if not self.ramps or self.colony is None:
            return
        t = float(getattr(self.colony, "t", 0.0))
        for key, knots in self.ramps.items():
            self.colony.p[key] = _ramp_at(knots, t)

    def _settle(self):
        """Pre-roll with growth and division off, so frame 0 is a RELAXED colony.

        A `filled` seed is a hexagonal lattice at one pitch, and per-cell Pareto sizes mean
        the big cells start deeply overlapped: pressure spikes, the colour driver runs hot,
        and the opening seconds of the clip are that relaxation rather than the scene. The
        clock is rewound afterwards, so `settle` costs sim time and no screen time.
        """
        ss = float(self.p.get("settle", 0.0))
        if ss <= 0 or self.colony is None:
            return
        from kit.config import SUBSTEP_HZ
        cp = self.colony.p                      # the colony's OWN dict, not self.p
        g, mc = cp["growth_rate"], cp["max_cells"]
        # Growth off is not enough: a `filled` seed spreads cells across the cell cycle, so
        # some are already over their division area and would divide during the settle.
        # Pinning the cap to the current count freezes the population without touching
        # divide_at or cycle_min, which the scene owns.
        cp["growth_rate"] = 0.0
        cp["max_cells"] = len(self.colony.pos)
        dt = 1.0 / SUBSTEP_HZ
        for _ in range(int(round(ss / dt))):
            self.colony.step(dt)
        cp["growth_rate"], cp["max_cells"] = g, mc
        self.colony.t = 0.0
        self.colony.div_t[:] = -1e9
        self.colony.sister[:] = -1
        self.colony._poly_t = None

    def _grain_seed(self):
        """Advances with the sim clock so the grain MOVES. See the note in the Objects
        engine: a constant seed is a static overlay, not grain."""
        t = self.colony.t if self.colony is not None else 0.0
        return int(self.seed * 1000003 + t * 1000.0) & 0x7FFFFFFF

    def apply_live(self):
        """Push `self.p` into the running colony. Cells are not rebuilt.

        `_seed` is re-applied because the colony's params dict is a COPY made at
        construction; updating it from `self.p` alone would drop that key and the
        tessellation would silently start clipping at the wrong radius."""
        if self.colony is not None:
            self.colony.p.update(dict(self.p, _seed=self.seed))

    def state_arrays(self):
        return self.colony.state_arrays() if self.colony else {}

    def check_invariants(self):
        return self.colony.check_invariants() if self.colony else []

    def summary(self):
        return self.colony.summary() if self.colony else {}

    # --- colour -------------------------------------------------------------------------
    def _driver_value(self):
        """Per-cell scalar in [0, 1] before the ramp."""
        c = self.colony
        p = self.p
        n = len(c.pos)
        if n == 0:
            return np.zeros(0)

        which = p["driver"]
        if which == "pressure":
            v = c.pressure.copy()
            hi = np.percentile(v, 92) if n > 4 else (v.max() if n else 1.0)
            v = v / max(hi, 1e-6)
        elif which == "depth":
            # A synthetic dome: distance from the colony centre, mapped to the height of
            # a spherical cap. Highest (nearest) in the middle, falling to the rim -
            # which is the same sense as the reference, where the middle of the organoid
            # bulges toward the objective.
            d = np.hypot(*(c.pos - c.pos.mean(axis=0)).T)
            R = max(float(d.max()), 1e-6) * 1.06
            v = np.sqrt(np.maximum(1.0 - (d / R) ** 2, 0.0))
        else:                                    # age
            cyc = max(1e-3, float(p["divide_at"]) / max(float(p["growth_rate"]), 1e-3))
            v = np.clip(c.age / cyc, 0.0, 1.0) * 0.65 + (c.line / 7.0) * 0.35

        v = np.clip(v, 0.0, 1.0)

        # CONTRAST-STRETCH THE DRIVER so the palette is actually used.
        #
        # Measured across ten authored scenes, the circular spread of hue over the lit
        # region ranged from 90 degrees down to FIVE - and the narrow ones had perfectly
        # wide `colour_lo..colour_hi` settings. The palette range was never the
        # constraint; the DRIVER was. Pressure in a uniformly-packed colony, or age in a
        # steadily-dividing one, produces values crowded into a narrow band, and a narrow
        # band of a ramp is one colour.
        #
        # Stretching between the 3rd and 97th percentiles (not min and max, which one
        # outlier cell would dominate) maps whatever spread the driver actually has onto
        # the whole ramp. `colour_lo`/`colour_hi` then still choose WHICH part of the ramp
        # to use, so an author's deliberate restriction is preserved - this only stops the
        # driver from silently wasting most of it.
        st = float(p["colour_stretch"])
        if st > 0 and len(v) > 8:
            a3, a97 = np.percentile(v, 3), np.percentile(v, 97)
            # The STRETCH CONSTANTS get a much longer time constant than the per-cell
            # value. They are a calibration of the driver's range, not a property of a
            # cell, and at the per-cell tau they drift fast enough to swing the whole
            # frame's hue while every individual cell is changing slowly - which is the
            # complaint `colour_tau` was added to fix, wearing a different hat.
            tau0 = float(p["colour_tau"]) * 8.0
            if tau0 > 0 and self._stretch is not None:
                w = np.exp(-max(c.t - self._col_t, 0.0) / tau0)
                a3 = self._stretch[0] * w + a3 * (1.0 - w)
                a97 = self._stretch[1] * w + a97 * (1.0 - w)
            self._stretch = (a3, a97)
            if a97 - a3 > 1e-4:
                stretched = np.clip((v - a3) / (a97 - a3), 0.0, 1.0)
                v = v * (1.0 - st) + stretched * st

        # SMOOTH IT ACROSS NEIGHBOURS. The reference's colour is a smooth spatial field;
        # a raw per-cell driver is salt-and-pepper and reads as a data visualisation.
        s = float(p["colour_smooth"])
        if s > 0 and len(c.nb):
            acc = np.zeros(n)
            cnt = np.zeros(n)
            i, j = c.nb[:, 0], c.nb[:, 1]
            # A courier crossing the frame would otherwise drag its lane's colour with
            # it - it touches a different cell every few frames and averages into each.
            w = (c.kind != 2).astype(np.float64)
            np.add.at(acc, i, v[j] * w[j])
            np.add.at(acc, j, v[i] * w[i])
            np.add.at(cnt, i, w[j])
            np.add.at(cnt, j, w[i])
            mean = acc / np.maximum(cnt, 1)
            v = v * (1.0 - s) + mean * s

        # RELAX TOWARD THE NEW VALUE instead of jumping to it.
        #
        # The driver is recomputed every frame and normalised by percentiles of the CURRENT
        # population, so a burst of divisions moves every cell's colour at once. Holding a
        # per-cell smoothed value with a time constant makes colour lag the physics, and
        # because a daughter inherits its parent's value at division, a newborn cell does
        # not arrive already a different colour from the cell it came out of.
        tau = float(p["colour_tau"])
        if tau > 0:
            unset = c.cval < 0
            if unset.any():
                c.cval[unset] = v[unset]
                # A courier with no inherited colour takes the sheet's, not its own: a
                # cell a twentieth the area of its neighbours sits at whichever end of
                # every driver, so its own value is not a colour, it is an artefact.
                uc = unset & (c.kind == 2)
                if uc.any():
                    ref = c.cval[(c.kind != 2) & (c.cval >= 0)]
                    if len(ref):
                        c.cval[uc] = float(np.median(ref))
            dt = c.t - self._col_t
            if dt > 0:
                # Couriers keep the colour they were budded with. Their own driver value
                # is meaningless - a cell a twentieth the area of its neighbours reads as
                # maximally crowded or maximally shallow whatever it is doing - and it
                # came out as a hot magenta disc pasted over the sheet.
                u = c.kind != 2
                c.cval[u] += (v[u] - c.cval[u]) * (1.0 - np.exp(-dt / tau))
            v = c.cval.copy()
        self._col_t = c.t

        lo, hi = float(p["colour_lo"]), float(p["colour_hi"])
        return np.clip(lo + v * (hi - lo), 0.0, 1.0)

    # --- drawing ------------------------------------------------------------------------
    def draw(self, cv):
        c = self.colony
        cfg = self.cfg
        p = self.p
        if c is None or len(c.pos) == 0:
            return

        # BOUNDED SWAY. `drift` is an AMPLITUDE in reference units, not a rate.
        #
        # It used to be `unit_vector(t) * drift * t`, which is a spiral that grows without
        # limit: at drift=14 the colony had translated 250+ ru by t=18, marching off the
        # left edge while leaving a quarter of the frame empty on the right. Every clip was
        # framed for a centred subject and then the subject walked out of the frame.
        drift = np.array([0.0, 0.0])
        if p["drift"] > 0:
            a = c.t * 0.31
            drift = np.array([np.cos(a), np.sin(a * 0.7)]) * float(p["drift"])

        polys = c.polygons()
        val = self._driver_value()
        gam = float(p["colour_gamma"])
        ns = int(c.strain.max()) + 1 if len(c.strain) else 1

        # PER-CULTURE RAMPS when `palettes` names them, otherwise one ramp hue-rotated.
        #
        # Rotating a single ramp keeps luma equal between strains, which is why it was the
        # original choice - no culture looks dimmer, and therefore further away, than
        # another. But it cannot put crimson's ramp next to amber's in one frame, because
        # both ends of a rotation come off the same ramp. Naming a ramp per culture can.
        names = [x.strip() for x in str(p.get("palettes", "")).split(",") if x.strip()]
        good = [x for x in names if x in C.RAMPS]
        if names and not good and not getattr(self, "_pal_warned", False):
            # Silently falling back here cost a whole build: `palettes` was handed culture
            # names instead of ramp names, every strain sampled one ramp, and six cultures
            # came out as a single radial gradient that looked deliberate.
            print("[bio] palettes %r names no known ramp - falling back to %r. Have: %s"
                  % (names, p["palette"], ", ".join(sorted(C.RAMPS))))
            self._pal_warned = True
        names = good
        if names and ns > 1:
            cols = np.zeros((len(val), 3))
            for si in range(ns):
                m = c.strain == si
                if m.any():
                    cols[m] = C.sample(names[si % len(names)], val[m], gamma=gam)
        else:
            cols = C.sample(names[0] if names else p["palette"], val, gamma=gam)

        if ns > 1 and float(p["strain_hue_spread"]) > 0:
            off = (np.arange(ns) - (ns - 1) * 0.5) * (float(p["strain_hue_spread"])
                                                      / max(1, ns - 1)) * 2.0
            cols = C.rotate_hue(cols, off[c.strain])

        # Opacity falls toward the rim, as if the tissue curves away from the objective.
        expose = np.hypot(c.free[:, 0], c.free[:, 1]) if len(c.free) else np.zeros(len(c.pos))
        alpha = np.clip(1.0 - expose * float(p["opacity_falloff"]), 0.12, 1.0)

        ew = max(1.0, float(p["edge_width"]) * cfg.px_per_ru())
        eg = float(p["edge_gain"])
        interior = float(p["interior"])
        tex = float(p["interior_texture"])
        rng = np.random.default_rng(self.seed + 17)

        # --- 1. the interior term, handed to the field below -----------------------------
        # The reference's cells are near-black inside; this is that floor. It used to be
        # one PIL polygon per cell into the opaque base layer. The field already knows
        # inside from outside, so it carries the fill for free.
        fill = interior * alpha
        if tex > 0:
            fill = fill * (1.0 + tex * (rng.random(len(polys)) - 0.5))

        # --- 2. THE FILM - one analytic field pass, see kit/cellfield.py ---------------
        #
        # A cell is a convex polygon, so the distance from any pixel to its wall is
        # min_e(d_e - (q-s).n_e) over its half-planes, and the sign of that is the inside
        # test. A SMOOTH min rounds the corners without moving the walls, so neighbours
        # keep sharing their wall exactly. This replaced concentric polygon fills, which
        # banded at wide framing and opened black voids at every tri-cellular junction.
        fg = float(p["film_gain"])
        wg = float(p["wall_gain"])
        film_cols = cols
        if abs(float(p["film_hue_shift"])) > 0.5:
            film_cols = C.rotate_hue(cols, float(p["film_hue_shift"]))

        if fg > 0 or wg > 0 or interior > 0:
            ppr = cfg.px_per_ru()
            sx, sy = cfg.world_to_px(c.pos[:, 0] + drift[0], c.pos[:, 1] + drift[1])
            sites = np.stack([sx, sy], 1)
            ppolys = []
            areas = np.empty(len(polys))
            for k, q in enumerate(polys):
                areas[k] = polygon_area(q) if len(q) >= 3 else 0.0
                if len(q) < 3:
                    ppolys.append(np.zeros((0, 2)))
                    continue
                px_, py_ = cfg.world_to_px(q[:, 0] + drift[0], q[:, 1] + drift[1])
                ppolys.append(np.stack([px_, py_], 1))
            # R is the cell's OWN equivalent radius, so the profile is cell-relative and a
            # squeezed cell reads as squeezed rather than as a smaller round one.
            R = np.maximum(np.sqrt(np.maximum(areas, 1e-6) / np.pi) * ppr
                           * float(p["film_size"]), 1.0)
            ks = np.clip(R * float(p["wall_soft"]), 0.0, R * 0.35)
            # The membrane is an ABSOLUTE thickness in reference units - a membrane does
            # not get thicker because its cell is bigger.
            # Absolute thickness, but capped against the cell's own radius. With a
            # Pareto size spread the smallest cells are a third the median, and an
            # uncapped membrane swallowed them whole - they came out as solid beads.
            ww = np.minimum(max(0.6, float(p["wall_width"]) * ppr), R * 0.30)
            # A courier is a twentieth of its neighbours' area; at the same gain it is a
            # speck. The boost is on the CELL, not on the profile, so its shape is the
            # same shape everything else has.
            # A COURIER IS DRAWN AS A SMALL FILLED CELL, not as a scaled-down big one.
            # `lens` puts its band just inside the wall and `wall_width` is capped at 30%
            # of the cell radius, so on a cell a fifth the normal radius the band IS the
            # cell: they came out as tiny bright donuts, and a queue of them read as one
            # blown blob. Flat fill carries them instead, with the rim turned down.
            cr = c.kind == 2
            cg = float(p["courier_glow"])
            fga = np.where(cr, fg * 0.35, fg) * alpha
            wga = np.where(cr, wg * 0.25, wg) * alpha
            fila = np.where(cr, cg * 0.55, fill)
            cv.cell_field(ppolys, sites, film_cols, R, ks, fga, wga, ww, fila,
                          p["film_profile"])

        # --- 3. membranes, into EMIT (additive, blooms) -----------------------------------
        #
        # EACH JUNCTION IS DRAWN EXACTLY ONCE, with the MEAN colour of the two cells that
        # share it. This is the single most important line in the file's look, and it was
        # wrong first, so the reasoning is worth keeping.
        #
        # The obvious implementation draws every cell's outline, which paints each shared
        # junction twice - once from each side. That seems harmless and even desirable
        # (junctions between two lit cells come out brighter than rim ones). It is not.
        # Adding two DIFFERENT hues bleaches the result toward white, and it is a
        # junction's own two cells that differ most in the drivers that vary spatially.
        # Measured on the raw composite, before any post-processing: edge saturation
        # 0.314 against a reference of 0.45-0.60, and 5.8% of hue weight pushed into the
        # 210-270 band that the reference does not have at all - blue manufactured out of
        # cyan plus magenta, entirely at draw time.
        #
        # Edges are deduplicated GEOMETRICALLY, by rounded midpoint, rather than through
        # the neighbour list - a clipped polygon's edges do not map one-to-one onto
        # Voronoi ridges (the outer arcs belong to a single cell), and the midpoint key
        # handles both cases without a special case.
        eg = float(p["edge_gain"])
        edges = {}
        for k, poly in enumerate(polys) if eg > 0 else ():
            if len(poly) < 3:
                continue
            q = poly + drift
            for a, b in zip(q, np.roll(q, -1, axis=0)):
                key = (round((a[0] + b[0]) * 0.5, 1), round((a[1] + b[1]) * 0.5, 1))
                e = edges.get(key)
                if e is None:
                    edges[key] = [a, b, cols[k].copy(), alpha[k], 1]
                else:
                    e[2] += cols[k]
                    e[3] += alpha[k]
                    e[4] += 1

        if eg > 0:
            for a, b, col, al, n in edges.values():
                col = col / n
                al = al / n
                ax, ay = cfg.world_to_px(a[0], a[1])
                bx, by = cfg.world_to_px(b[0], b[1])
                cv.segment(float(ax), float(ay), float(bx), float(by), ew,
                           tuple((col * eg * al).tolist()), layer="emit", additive=True)

        # --- 4. tri-cellular junctions ----------------------------------------------------
        #
        # MEAN colour and MEAN alpha, not the sums: written with a trailing `* n` the
        # first time, which cancelled one of the divisions and made every junction three
        # times too bright (luma_p99 202 against a reference of 120-175).
        #
        # Batched into ONE kernel launch. A packed sheet has ~2 junctions per cell, so a
        # 700-cell colony draws 1400 of these a frame and the per-call overhead was most
        # of their cost.
        vg = float(p["vertex_gain"])
        if vg > 0:
            verts = {}
            for k, poly in enumerate(polys):
                if len(poly) < 3:
                    continue
                for v in poly + drift:
                    key = (round(v[0], 0), round(v[1], 0))
                    e = verts.get(key)
                    if e is None:
                        verts[key] = [v, cols[k].copy(), alpha[k], 1]
                    else:
                        e[1] += cols[k]
                        e[2] += alpha[k]
                        e[3] += 1
            if verts:
                V = np.array([e[0] for e in verts.values()])
                VC = np.array([e[1] / e[3] for e in verts.values()])
                VA = np.array([e[2] / e[3] for e in verts.values()])
                vx, vy = cfg.world_to_px(V[:, 0], V[:, 1])
                cv.radial_batch(vx, vy, np.full(len(V), ew * 0.62), VC,
                                vg * VA, profile="core")

        # --- 5. NUCLEI, and the division split --------------------------------------------
        #
        # Each cell carries a hue offset of its own (`colony.nuc`), inherited with a small
        # drift, so the field reads as a population of individuals rather than one
        # substance. A daughter's nucleus is a slightly different colour from its sister's,
        # which is what lets the eye see ONE POINT BECOME TWO rather than one point
        # smearing - and because the two sites now separate over `division_time` instead
        # of jumping apart, that is exactly what it looks like.
        ng = float(p["nucleus_gain"])
        if ng > 0:
            nuc_cols = C.rotate_hue(cols, (c.nuc - 0.5) * 360.0 * float(p["nucleus_hue"]))
            # Nuclei are the brightest thing in a cell, so they are pushed toward white.
            nuc_cols = np.clip(nuc_cols * 0.72 + 70.0, 0, 255)
            nr = np.maximum(1.0, c.radii * float(p["nucleus_size"]) * cfg.px_per_ru())

            # The flash is a BELL over the cytokinesis window, not a step at t=0 decaying
            # away. A step reads as a pop; a bell reads as an event with a beginning.
            bt = np.maximum(c.divT * float(p["bridge_time"]), 1e-6)
            u = np.clip((c.t - c.div_t) / bt, 0.0, 1.0)
            flash = 1.0 + float(p["division_flash"]) * 4.0 * u * (1.0 - u)

            nx_, ny_ = cfg.world_to_px(c.pos[:, 0] + drift[0], c.pos[:, 1] + drift[1])
            cv.radial_batch(nx_, ny_, nr, nuc_cols, ng * flash * alpha, profile="core")

            # THE MIDBODY - the middle third of the furrow, not nucleus to nucleus.
            # Full-length bridges measured 64-93 ru against a 55 ru cell radius, so each
            # was a bright bar across an entire neighbouring cell; two reviewers called
            # them dead pixels.
            bg = float(p["bridge_gain"])
            if bg > 0:
                live = np.nonzero((c.sister >= 0) & (u < 1.0)
                                  & (np.arange(len(c.pos)) < c.sister))[0]
                for k in live:
                    sis = int(c.sister[k])
                    if sis >= len(c.pos):
                        continue
                    fade = (1.0 - u[k]) ** 2
                    pk = c.pos[k] + drift
                    ps = c.pos[sis] + drift
                    q0 = pk + (ps - pk) * 0.36
                    q1 = pk + (ps - pk) * 0.64
                    ax, ay = cfg.world_to_px(q0[0], q0[1])
                    bx, by = cfg.world_to_px(q1[0], q1[1])
                    col = (nuc_cols[k] + nuc_cols[sis]) * 0.5 * bg * fade
                    cv.segment(float(ax), float(ay), float(bx), float(by),
                               max(1.0, nr[k] * 0.7), tuple(col.tolist()),
                               layer="emit", additive=True)

        # --- 6. ORGANELLES ----------------------------------------------------------------
        #
        # A soft chunk plus a small glowing dot, both dimmer than the nucleus. Position,
        # size and brightness come from the cell's OWN seed through `hash01`, so every
        # cell has a different assortment and it is stable for that cell's lifetime.
        # Offsets are fractions of the CURRENT radius, so they stay inside a cell that
        # grows or shrinks, and both daughters get a fresh seed at division - the previous
        # version used one offset table for every cell in the frame.
        gn = int(p["granules"])
        gg, gd = float(p["granule_gain"]), float(p["granule_dot"])
        if gn > 0 and (gg > 0 or gd > 0):
            ppr = cfg.px_per_ru()
            rr = c.radii
            # Skip anything too small on screen to resolve one: couriers, mostly.
            live = np.nonzero((rr * ppr > 14.0) & (c.kind != 2))[0]
            if len(live):
                sd = c.org[live]
                rl = rr[live]
                xs, ys, rs, gb, gc = [], [], [], [], []
                for kk in range(gn):
                    u1 = hash01(sd, kk * 4 + 0)
                    u2 = hash01(sd, kk * 4 + 1)
                    u3 = hash01(sd, kk * 4 + 2)
                    u4 = hash01(sd, kk * 4 + 3)
                    rad = np.sqrt(u1) * float(p["granule_spread"]) * rl
                    th = u2 * (2.0 * np.pi)
                    px_, py_ = cfg.world_to_px(c.pos[live, 0] + np.cos(th) * rad + drift[0],
                                               c.pos[live, 1] + np.sin(th) * rad + drift[1])
                    xs.append(px_)
                    ys.append(py_)
                    rs.append(np.maximum(1.5, rl * float(p["granule_size"])
                                         * (0.55 + 0.95 * u3) * ppr))
                    gb.append(0.45 + 1.1 * u4)
                    gc.append(live)
                X = np.concatenate(xs)
                Y = np.concatenate(ys)
                S = np.concatenate(rs)
                B = np.concatenate(gb)
                I = np.concatenate(gc)
                al = alpha[I]
                if gg > 0:
                    cv.radial_batch(X, Y, S, cols[I], gg * B * al, profile="blob")
                if gg > 0 or gd > 0:
                    if gd > 0:
                        dot = np.clip(C.rotate_hue(cols[I], 0.0) * 0.75 + 55.0, 0, 255)
                        cv.radial_batch(X, Y, np.maximum(1.0, S * 0.34), dot,
                                        gd * B * al, profile="core")


def make(params=None, seed=7, ramps=None):
    return Engine(params, seed, ramps)
