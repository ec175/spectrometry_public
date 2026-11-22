# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cultures.py - the template cultures, split into QUALITIES and PALETTES.

Clancy reviewed the first twenty clips and named three keepers to build from. The useful
observation in that review was that he wanted one of them for how its CELLS behave and
are lit, and the other two only for their COLOUR. So the templates are split along that
line and the two halves compose:

    QUALITIES   mechanics, film, membrane, nucleus, granules - how a cell moves, deforms
                and catches light. One per scene, because the mechanics are global.
    PALETTES    ramp, gamma, film hue shift, nucleus hue - the colour identity of ONE
                strain. A scene names several, one per culture, and cells sample their
                own strain's ramp (see `palettes` in the engine).

So "taut_interface's colour on crimson_pressure's cells" is PALETTES["amber"] over
QUALITIES["crimson"], which is what was asked for and is one line to write.

Numbers below are lifted from the shipped scenes that passed review, not re-invented.
"""

#: The reference cells: heavy confinement, taut membranes, a bright refractive rim.
#: From `crimson_pressure`, which was the pick of the twenty.
QUALITIES = {
    "crimson": dict(
        # mechanics - firm, slow, strongly adherent, so cells press and hold their shape
        tension=2.4, pressure=2.4, adhesion=4.0, repulsion=9.0, contact=0.98,
        viscosity=3.4, motility=0.13,
        # the film and the membrane
        film_profile="lens", film_gain=1.35, film_size=1.0,
        wall_soft=0.11, wall_gain=1.5, wall_width=5.0,
        edge_gain=0.0, vertex_gain=0.0,
        interior=0.05, interior_texture=0.30, opacity_falloff=0.40,
        # nucleus and sub-cellular detail
        nucleus_gain=1.05, nucleus_size=0.16, nucleus_drift=0.06,
        division_flash=2.6, bridge_time=1.0, bridge_gain=0.9,
        granules=5, granule_gain=0.22, granule_size=0.065,
        # growth character
        divide_at=2.2, cycle_min=1.0, inhibit=0.52, inhibit_growth=4.0,
        heterogeneity=0.28,
        # v3: size spread, mass transfer, progressive division, colour inertia
        size_pareto=2.1, size_max=5.0, size_drift=0.07, size_drag=1.1,
        division_time=1.1, colour_tau=1.6, settle=1.6,
    ),
}

#: One entry per CULTURE colour. `palette` names a ramp in kit/colour.py.
PALETTES = {
    "crimson": dict(palette="polar_bloom", colour_gamma=0.60, film_hue_shift=20.0,
                    nucleus_hue=0.35),
    "amber":   dict(palette="polar_amber", colour_gamma=0.90, film_hue_shift=22.0,
                    nucleus_hue=0.45),
    "rich":    dict(palette="polar_rich",  colour_gamma=0.90, film_hue_shift=12.0,
                    nucleus_hue=0.55),
    "jade":    dict(palette="polar_jade",  colour_gamma=0.85, film_hue_shift=-14.0,
                    nucleus_hue=0.50),
    "orchid":  dict(palette="polar_orchid", colour_gamma=0.85, film_hue_shift=-8.0,
                    nucleus_hue=0.60),
}


def apply(params, quality="crimson", palettes=("crimson",)):
    """Merge a QUALITIES block and a list of PALETTES into a scene's params.

    Scene-authored keys WIN: a scene sets `cell_area`, `clusters`, `max_cells`, framing
    and pacing, and the template supplies everything it does not mention. `cell_area` in
    particular must stay scene-owned - a close-up and a wide shot are the same cells at
    different absolute sizes.
    """
    out = dict(QUALITIES[quality])
    names = [p for p in palettes if p in PALETTES] or ["crimson"]
    out.update(PALETTES[names[0]])
    # RAMP names, not culture names. Written as the culture keys first time, which the
    # engine filtered against `C.RAMPS`, found nothing, and silently fell back to one
    # shared ramp - six cultures rendered as one radial gradient with no territories in it.
    out["palettes"] = ",".join(PALETTES[n]["palette"] for n in names)
    # Several palettes already differ in hue; rotating them apart as well double-counts
    # and pushes strains off their own ramp.
    out["strain_hue_spread"] = 0.0 if len(names) > 1 else out.get("strain_hue_spread", 0.0)
    out.update(params)
    return out
