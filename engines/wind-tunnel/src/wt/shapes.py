"""shapes.py — the bodies placed in the tunnel, as closed polygons in LATTICE coordinates.

Every body is just an ordered closed polygon. The solver only ever sees a rasterised boolean
mask (`rasterize`), so adding a new body = adding a function that returns points — no meshing,
no signed-distance field, no special cases in the physics.

Profiles are generated in unit-chord space (x from 0 at the leading edge to 1 at the trailing
edge, y up) and then `place()`d: scaled by the chord, rotated by the angle of attack about the
quarter-chord point (the conventional aerodynamic centre, so a sweeping AoA pivots the way a
real foil in a rig does — it does not swing the nose around the centroid), and translated.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


# --- profile generators (unit chord) ------------------------------------------------
def naca4(code="2412", n=140, closed_te=True):
    """NACA 4-digit section, standard formulation. code = MPXX:
    M = max camber (% chord), P = its position (tenths), XX = thickness (% chord)."""
    code = str(code).zfill(4)
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    # cosine spacing -> points cluster at the leading edge where curvature is extreme
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if closed_te else -0.1015          # closed trailing edge variant
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 + a4 * x**4)
    if m > 0 and p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x),
                       2 * m / (1 - p)**2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    # upper surface TE->LE then lower LE->TE = one closed loop
    pts = np.concatenate([np.stack([xu[::-1], yu[::-1]], 1), np.stack([xl[1:], yl[1:]], 1)])
    return pts


def circle(n=160):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([0.5 + 0.5 * np.cos(a), 0.5 * np.sin(a)], 1)


def square(n=4):
    return np.array([[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5]])


def plate(thick=0.045):
    return np.array([[0.0, -thick / 2], [1.0, -thick / 2], [1.0, thick / 2], [0.0, thick / 2]])


def wedge(half=0.30):
    """Nose-forward triangle (a classic low-drag / high-drag pair with `wedge_rev`)."""
    return np.array([[0.0, 0.0], [1.0, -half], [1.0, half]])


def wedge_rev(half=0.30):
    """Flat face into the wind — the bluffest body in the set."""
    return np.array([[1.0, 0.0], [0.0, -half], [0.0, half]])


def ellipse(thick=0.30, n=140):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([0.5 + 0.5 * np.cos(a), 0.5 * thick * np.sin(a)], 1)


def teardrop(thick=0.34, n=150):
    """Streamlined body: an ellipse nose faired into a long tapering tail."""
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    y = 0.5 * thick * np.sqrt(np.clip(1 - (2 * x - 1) ** 2, 0, None)) * (1 - x) ** 0.55 * 2.05
    return np.concatenate([np.stack([x, y], 1), np.stack([x[::-1][1:], -y[::-1][1:]], 1)])


def cow(scale_y=1.0):
    """A COW in side profile, head into the wind, as ONE closed polygon in unit-chord space.

    x = 0 is the muzzle (the leading edge), x = 1 the tail tuft; +y is the BACK and -y the
    hooves, i.e. the same sign convention every other profile here uses, so `place()` puts it
    the right way up with no special case.

    Why a single loop and not a body plus four legs: the solver only ever sees a rasterised
    mask, and separate polygons that nearly touch leave sub-cell cracks the lattice cannot
    represent (CLAUDE.md section 6 rule 4). Tracing the whole animal in one circuit - over the
    back, down the tail, then weaving down and up each leg along the underside - makes the mask
    one connected island by construction.

    THE SILHOUETTE IS THE REQUIREMENT, exactly as it was for `turbofan` (AGENT_GUIDE 2a.29): a
    physics check cannot fail on "does it read as a cow". So the proportions are set by what
    survives rasterisation at the ~119 cells of chord this scene gives it, not by anatomy:

      - **legs are 0.055-0.065 of the length wide** (~7 cells) rather than a real cow's ~0.03.
        At 3 cells a leg is a jagged ribbon that sheds numerical noise and reads as a scratch;
        the inter-leg gaps are held at 6+ cells for the same reason.
      - **the head is one wedge with a chunky horn-and-ear bump.** Separate horns and ears at
        this resolution merge into a blob anyway, so they are drawn as the blob deliberately.
      - **the tail hangs, with a tuft.** A tail held out behind would fair into the rump and
        disappear; hanging, it is the one feature that says "animal" from across a room, and it
        is also a bluff trailing appendage that gives the wake something to do.
      - **the udder** fills the gap between the hind legs, which stops that gap from acting as a
        little nozzle between two thin bluff bodies.

    `scale_y` stretches the body about the spine for a taller or shallower animal without
    touching the length the chord is quoted against.
    """
    p = [
        # -- head, up the neck, along the back, to the tail head -------------------------
        (0.000, 0.104), (0.026, 0.148), (0.056, 0.170),          # muzzle -> forehead
        (0.052, 0.232), (0.090, 0.196),                          # horn
        (0.118, 0.240), (0.146, 0.182),                          # ear
        (0.206, 0.212), (0.268, 0.250),                          # crest of the neck, withers
        (0.390, 0.234), (0.570, 0.230), (0.706, 0.248),          # back -> hip
        (0.806, 0.240), (0.866, 0.212),                          # rump -> tail head
        # -- the tail: down its back edge, round the tuft, up its front edge -------------
        (0.906, 0.062), (0.926, -0.116), (0.950, -0.208),
        (0.908, -0.176), (0.886, -0.020), (0.864, 0.104),
        # -- thigh + hock, then the hind pair (rear leg first) ---------------------------
        # Every inter-leg gap below is >= 0.055 of the length, i.e. >= 6 cells at the ~119-cell
        # chord this is drawn at. That is not styling: a sub-cell passage is a channel the
        # lattice cannot represent and the speed in it runs away (CLAUDE.md section 6 rule 4).
        # A first pass drew anatomically thin legs 1.7 cells apart.
        (0.842, 0.030), (0.834, -0.072),
        (0.850, -0.150), (0.844, -0.230), (0.846, -0.300),
        (0.798, -0.302), (0.800, -0.210), (0.804, -0.110), (0.808, -0.070),
        (0.748, -0.062),
        (0.740, -0.150), (0.734, -0.300), (0.686, -0.302), (0.692, -0.150), (0.698, -0.066),
        # -- udder, then the belly line --------------------------------------------------
        (0.676, -0.114), (0.642, -0.124), (0.614, -0.084),
        (0.520, -0.070), (0.442, -0.068),
        # -- the fore pair (rear leg first) ----------------------------------------------
        (0.424, -0.080), (0.416, -0.190), (0.412, -0.300),
        (0.364, -0.302), (0.368, -0.190), (0.372, -0.076),
        (0.318, -0.066),
        (0.310, -0.190), (0.306, -0.300), (0.258, -0.302), (0.264, -0.190), (0.270, -0.060),
        # -- brisket, throat, jaw, back to the muzzle ------------------------------------
        (0.246, -0.056), (0.222, -0.026), (0.190, 0.010),
        (0.150, 0.036), (0.104, 0.042), (0.062, 0.030), (0.020, 0.038),
    ]
    a = np.asarray(p, dtype=np.float64)
    a[:, 1] *= float(scale_y)
    return a


def surfboard(width=0.28, tail=0.36, tip=0.012, nose_p=0.70, x_n=0.12, n=240):
    """TOP-DOWN surfboard PLANFORM, unit-LENGTH space: x = 0 nose -> 1 tail, +-y the two rails.

    Same contract as `naca4` - one closed polygon in unit-chord space - so `place`, `rasterize`,
    `FreeBody` and the body draw need no special case at all.

    It is drawn as the OUTLINE SEEN FROM ABOVE, which is the view these scenes simulate: the
    plane of the lattice is the water surface (or, in `surf_wave`, the face of the wave), so the
    silhouette the solver is handed is the board's planshape and its incidence is the board's
    YAW. Fins and rocker are out of plane and therefore genuinely absent, not omitted for
    convenience - see the scene docstrings for what that costs.

    `width` is the maximum width as a fraction of LENGTH. 0.28 is a real shortboard (a 6'2" x
    20" is 0.27); it is quoted rather than hardcoded because it is the single number that decides
    how bluff this body is, and blockage is what sets peak `max|u|` in this project
    (AGENT_GUIDE 2a.27). At +-34 deg a 0.28-wide board projects 0.28*cos34 + 1.0*sin34 = 0.792
    chords across the stream against a NACA 0015's 0.683 - i.e. **16% more blockage than the
    section it replaces**, which is why the tri-lane scene re-runs its own stability gate rather
    than inheriting one.

    THE SILHOUETTE IS A REQUIREMENT, not a garnish (AGENT_GUIDE 2a.29 - a duct built out of
    aerofoils IS two aerofoils, and no physics check can fail on "does it read as the thing").
    Four features carry the reading, and each is set by what survives rasterisation:

      - **the wide point sits AFT of centre**, at x = 0.55. A shape whose widest point is at 0.5
        reads as a leaf or a lens; a surfboard's volume is behind the middle and that asymmetry
        is most of what says "board" at a glance.
      - **a rounded, not needle, nose.** `r ~ (x/x_n)**nose_p` has infinite slope at the tip, so
        the nose is blunt in the way an ellipse is blunt - a `_knots` ramp to zero would leave a
        zero-slope needle. And the tip is TRUNCATED at `tip` x the length rather than run to a
        point: a half-width that goes to zero is a sub-cell sliver, and the first mask drawn here
        came out as the body plus a DETACHED SPECK where the sliver crossed the 2x-supersampled
        threshold and back. That is CLAUDE.md section 6 rule 4 in miniature - the same reason
        `cow`'s legs are drawn fatter than a real cow's and `image_body` welds at 1.5 cells. At
        0.012 the tip is 1.8 cells wide on the tri-lane lattice and 2.3 on the single-board one,
        and it is invisible at any size these clips are watched at. `mask_check` in
        `tools\\board_check.py` counts the islands, so a regression here is caught rather than
        rendered.
      - **a SQUASH TAIL**: the outline ends at a straight edge of half-width `tail` x the max
        half-width, i.e. a full tail width of `tail * width` = 0.10 of the length. A pintail
        tapering to nothing looks like the trailing edge of an aerofoil, which is the one reading
        this shape must not have. It is also a real bluff base, so the wake gets a proper
        separated region instead of a fairing.
      - **it is exactly symmetric about the stringer**, so like the NACA 0015 it makes no lift at
        zero incidence and everything on screen is bought by yaw alone.
    """
    half = 0.5 * float(width)
    r_tip = float(tip) / max(half, 1e-9)            # tip half-width, in units of the max
    # Normalised half-width law, 1.0 at the wide point. The FIRST knot is the join to the nose
    # ogive and must carry the real half-width there, not zero: `_knots` hits every knot exactly,
    # so anchoring it at (x_n, 0.0) pinched the board to ZERO WIDTH at 12% of its length - the
    # body then rasterised as two or more islands and `board_check` caught it before any render.
    s_knots = [(x_n, 0.50), (0.18, 0.63), (0.26, 0.77), (0.36, 0.90), (0.46, 0.98),
               (0.55, 1.00), (0.64, 0.98), (0.73, 0.92), (0.82, 0.82), (0.89, 0.70),
               (0.95, 0.55), (1.00, float(tail))]
    r_n = float(s_knots[0][1])          # the ogive's end value == the aft law's start value

    def half_w(x):
        x = np.asarray(x, dtype=np.float64)
        u = np.clip(x / max(x_n, 1e-9), 0.0, 1.0)
        # ogive from the truncated tip up to the aft law - r_tip at x=0, r_n at x=x_n
        nose = r_tip + (r_n - r_tip) * u ** float(nose_p)
        return half * np.where(x < x_n, nose, _knots(x, s_knots))

    # cosine spacing clusters points at the nose, where the curvature is
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))
    r = half_w(x)
    # one rail nose->tail, straight across the squash tail, the other rail tail->nose. Both ends
    # are BLUNT (a tip edge and a squash tail), so the loop closes across a real edge at each.
    return np.concatenate([np.stack([x, r], 1), np.stack([x[::-1], -r[::-1]], 1)])


def board_texture(px=768, deck=(238, 241, 244), rail=(22, 96, 168), stringer=(92, 64, 38),
                  pad=(28, 34, 44), width=0.28):
    """Procedural deck art for `surfboard`, as an RGB array in the board's OWN unit space.

    u runs 0 (nose) -> 1 (tail) along the array's width, v across the rails. A scene hands the
    renderer an exact `to_uv` (the analytic inverse of `place`), so unlike a bounding-box texture
    this stays glued to the board through any rotation - which a swept body needs and
    `tri_shapes`' stationary cow did not.

    Why paint it at all, when every aerofoil in this project is flat black: a black planform is a
    leaf. A white deck with a dark stringer down the middle and a traction pad at the tail is
    what makes a viewer see a surfboard in the first half second, and it is generated rather than
    photographed so it costs no asset and inherits the outline exactly.
    """
    h = max(16, int(round(px * float(width))))
    img = np.zeros((h, int(px), 3), dtype=np.float64)
    u = np.linspace(0.0, 1.0, int(px))[None, :]
    v = np.linspace(-1.0, 1.0, h)[:, None]          # -1 / +1 = the two rails
    img[:] = np.asarray(deck, dtype=np.float64)
    # rails: a slim colour band that thickens toward the tail, the way a spray does. Kept THIN on
    # purpose - a wide band swallows the white deck, and at 40 px of on-screen board width the
    # deck is the only thing carrying the "board, seen from above" reading.
    band = _sstep(0.80, 0.99, np.abs(v) + 0.07 * u)
    img = img * (1 - band[:, :, None]) + np.asarray(rail, np.float64) * band[:, :, None]
    # stringer: a thin wood line on the centreline, from just behind the nose to the tail
    st = (1.0 - _sstep(0.008, 0.026, np.abs(v))) * _sstep(0.03, 0.09, u)
    img = img * (1 - st[:, :, None]) + np.asarray(stringer, np.float64) * st[:, :, None]
    # traction pad over the back foot
    tp = _sstep(0.74, 0.79, u) * (1.0 - _sstep(0.95, 0.99, u)) * (1.0 - _sstep(0.52, 0.72, np.abs(v)))
    img = img * (1 - tp[:, :, None]) + np.asarray(pad, np.float64) * tp[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _sstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _knots(x, pts):
    """Smoothstep interpolation through (x, r) knots - flat outside the ends.

    Segment-wise smoothstep rather than a spline: every knot is hit EXACTLY and the slope is
    zero at each one, so a station quoted here (the throat, the step, the nozzle) is the radius
    that actually appears on the body, and a scene reading `duct_r` gets the same number the
    polygon was drawn with.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.full_like(x, float(pts[0][1]))
    for (xa, ra), (xb, rb) in zip(pts[:-1], pts[1:]):
        r = r + (rb - ra) * _sstep(xa, xb, x)
    return r


def turbofan(r_lip=0.175, r_throat=0.160, r_exit=0.105, r_out=0.330, r_spin=0.055,
             t_te=0.026, n=280):
    """2-D SECTION through a jet engine, in unit-LENGTH space: x = 0 intake -> 1 nozzle exit.

    Returns dict(cowl, spinner, duct_r, spin_r). `cowl` is the +y half of the pod wall (a scene
    mirrors it for the other side); `spinner` is the intake cone, one closed loop straddling
    the axis. `duct_r(x)` / `spin_r(x)` are the radius laws, so a scene can find the free
    passage at any station without duplicating these constants.

    THE SILHOUETTE IS THE POINT, and getting it wrong is what this shape exists to fix. The
    first version wrapped the NACA thickness law around the duct wall - a perfectly defensible
    way to build a cowl, which on screen looked like two airfoils facing each other, because
    that is exactly what it was. Ethan, 2026-08-03: *"very confused on why its two airfoils...
    even a simple jet engine block with a funnel at the end? ... do not leave an airfoil in the
    video."* So this is built as the thing he asked for, and four features carry the reading:

      - **a thick-walled POD, not a thin cowl.** The wall is `r_out - r_throat` = 0.16 of the
        engine length against a duct radius of 0.14 - the solid is WIDER than the hole. This is
        the single biggest change and it is what stops the two halves reading as rails or as
        wing sections. An intermediate version with realistic (thin, ~0.4 of duct radius)
        nacelle walls was tried and looked like two blades with a cigar between them.
      - **a flat barrel.** `r_out` is held constant over 0.10-0.50. An airfoil's surface curves
        continuously from nose to tail and never has a straight stretch; a pod is a can.
      - **the FUNNEL.** At 0.50 the outer surface turns down and converges hard onto a narrow
        exhaust tube - outer radius 0.137 against the pod's 0.300 - which then runs straight to
        the exit. A silhouette that is fat, then conical, then a narrow tube is a jet exhaust
        and very little else. It costs a ~26 deg boat-tail, which is steeper than a real
        nacelle's and will separate at idle; the jet entrains that region back in as thrust
        rises, and it frames the plume rather than spoiling it.
      - **the spinner** - a cone on the axis inside the mouth, ending at 0.48. It says
        "machinery" through the intake, and stopping it well short of the nozzle is deliberate:
        a full-length centrebody would split the exhaust into two thin annular sheets, and one
        round jet with one shear layer on each side gives far cleaner roll-up (which is the
        actual subject of the clip) at this lattice resolution.

    The intake lip is an explicit half-ellipse of semi-axes (0.10, t_lip) where t_lip is set by
    the wall thickness itself, so the lip is a fat rounded-over rim that fairs into the barrel
    with no kink and will not separate when the fan pulls its streamtube in from outside the
    capture area.

    Everything the duct then does is still the solver's: the engine is a pipe with a pump in
    it, so intake suction, entrainment and the jet are consequences, not drawings.
    """
    x_n = 0.10                       # length of the elliptical lip region
    t_lip = 0.5 * (r_out - r_lip)    # lip half-thickness: fairs the rim exactly into the barrel
    m_n = r_lip + t_lip              # lip mid-line radius

    def duct_r(x):
        """Inner (wetted) duct radius: mouth -> throat -> convergence -> straight nozzle."""
        x = np.asarray(x, dtype=np.float64)
        nose = m_n - t_lip * np.sqrt(np.clip(1.0 - ((x_n - x) / x_n) ** 2, 0.0, None))
        aft = _knots(x, [(x_n, r_lip), (0.30, r_throat), (0.58, r_throat),
                         (0.92, r_exit), (1.00, r_exit)])
        return np.where(x < x_n, nose, aft)

    def outer_r(x):
        """Outer pod radius: round rim -> flat barrel -> FUNNEL -> short nozzle lip.

        Inner and outer converge TOGETHER over 0.58-0.92, so the whole aft third of the body is
        one cone rather than a taper on the outside and a straight pipe on the inside. That is
        the fix for the failure mode a rendered cut showed: with the convergence outside only,
        each half ended in a long thin wall and the pair read as two legs, not as one funnel.
        The nozzle lip that follows is deliberately SHORT - just enough to end the wall bluntly.
        """
        x = np.asarray(x, dtype=np.float64)
        nose = m_n + t_lip * np.sqrt(np.clip(1.0 - ((x_n - x) / x_n) ** 2, 0.0, None))
        aft = _knots(x, [(x_n, r_out), (0.58, r_out), (0.92, r_exit + t_te),
                         (1.00, r_exit + t_te)])
        return np.where(x < x_n, nose, aft)

    def spin_r(x):
        """Spinner radius: ogive nose, short parallel section, conical tail.

        Both ends reach r = 0 with a FINITE slope - the nose as an ogive, the tail as
        (1 - s^2). Running the ends through `_knots` instead would give zero slope at r = 0,
        i.e. a needle rather than a cone.
        """
        x = np.asarray(x, dtype=np.float64)
        x_h, x_a, x_b, x_t = 0.05, 0.22, 0.36, 0.55
        u = np.clip((x - x_h) / (x_a - x_h), 0.0, 1.0)
        nose = r_spin * u ** 0.62 * (1.62 - 0.62 * u)
        s = np.clip((x - x_b) / (x_t - x_b), 0.0, 1.0)
        tail = r_spin * (1.0 - s * s)
        r = np.where(x < x_a, nose, np.where(x < x_b, r_spin, tail))
        return np.where((x < x_h) | (x > x_t), 0.0, r)

    def blades():
        """Interleaved rotor/stator stubs - the compressor-cutaway cue.

        WHY THESE EXIST AT ALL. A 2-D section through a flow-through duct is topologically
        forced to be two or more DISCONNECTED solids, and disconnected solids read as separate
        objects however they are shaped: three renders of pod outlines all came back as "two
        boots and a seed". In 3-D the fix is a strut from the centrebody to the wall, but in 2-D
        there is no circumferential direction for flow to get around one, so a strut is a
        complete blockage. Blade stubs are the version that survives the dimension: each spans
        only PART of the passage, so the eye completes the row into a stage while the fluid
        weaves axially through the gaps.

        Rotors hang off the spinner and point out, stators hang off the wall and point in, and
        the two rows overlap in radius but never in x - which is exactly what a compressor
        cutaway looks like, and is also why the duct is not choked. Each is a parallelogram,
        slanted `slant` in x from root to tip: a straight radial bar reads as a peg, a slanted
        one reads as a blade.
        """
        out = []
        rows = [(0.285, "rotor"), (0.360, "stator"), (0.435, "rotor"), (0.510, "stator")]
        dx, slant = 0.020, 0.016
        for xc, kind in rows:
            wall, hub = float(duct_r(xc)), float(spin_r(xc))
            # roots are pushed 0.010 INTO the part they hang off, so the rasterised mask fuses
            # them to it instead of leaving a one-cell crack that would leak flow through the
            # blade root and, worse, read as a floating bar
            if kind == "rotor":
                r0, r1 = hub - 0.010, hub + 0.62 * (wall - hub)
            else:
                r0, r1 = wall + 0.010, wall - 0.62 * (wall - hub)
            quad = np.array([[xc - dx + slant, r1], [xc + dx + slant, r1],
                             [xc + dx - slant, r0], [xc - dx - slant, r0]])
            out.append(quad)
            out.append(np.stack([quad[:, 0], -quad[:, 1]], 1))
        return out

    # cosine spacing clusters points at the lip, where the curvature is
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))
    cowl = np.concatenate([np.stack([x, outer_r(x)], 1),
                           np.stack([x[::-1], duct_r(x[::-1])], 1)])

    xs = np.linspace(0.05, 0.55, int(n) // 2)
    rs = spin_r(xs)
    spinner = np.concatenate([np.stack([xs, rs], 1),
                              np.stack([xs[::-1], -rs[::-1]], 1)])
    return dict(cowl=cowl, spinner=spinner, blades=blades(), duct_r=duct_r, spin_r=spin_r,
                outer_r=outer_r, r_out=float(r_out))


PROFILES = {
    "naca": naca4, "circle": circle, "square": square, "plate": plate,
    "wedge": wedge, "wedge_rev": wedge_rev, "ellipse": ellipse, "teardrop": teardrop,
    "cow": cow, "surfboard": surfboard,
}


def metaball(blobs, n=192, iters=36):
    """Smooth-union outline of a set of circles - the shape two droplets make while they fuse.

    `blobs` is [(x, y, r), ...] in any frame; the returned closed polygon is in that same frame.

    Implicit field f(p) = sum_i r_i^2 / |p - c_i|^2, contoured at f = 1. A lone blob returns its
    own circle exactly; two blobs grow a neck between them the way surface tension does. The
    f = 1 level set of overlapping blobs is star-shaped about their r^2-weighted centre, so the
    contour follows from one radial bisection per angle - no marching squares, no scikit-image,
    and the result is already an ordered closed polygon, which is the only shape representation
    the rest of this engine understands (`place`, `rasterize`, `FreeBody`).

    The property the coalescence relies on: as two centres are brought together the f = 1 surface
    tends to a circle of radius sqrt(r1^2 + r2^2) - exactly the AREA-conserving radius. So a
    merge that simply animates the separation to zero conserves the droplet's volume for free,
    with no fudge factor anywhere.
    """
    c = np.asarray([(b[0], b[1]) for b in blobs], dtype=np.float64)
    r = np.asarray([b[2] for b in blobs], dtype=np.float64)
    w = r ** 2
    o = (c * w[:, None]).sum(0) / w.sum()
    th = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    ux, uy = np.cos(th), np.sin(th)
    lo = np.zeros(int(n))
    hi = np.full(int(n), float(np.hypot(*(c - o).T).max() + 2.0 * r.max()))
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        px, py = o[0] + mid * ux, o[1] + mid * uy
        f = np.zeros(int(n))
        for k in range(len(r)):
            f += w[k] / np.maximum((px - c[k, 0]) ** 2 + (py - c[k, 1]) ** 2, 1e-9)
        inside = f >= 1.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    rr = 0.5 * (lo + hi)
    return np.stack([o[0] + rr * ux, o[1] + rr * uy], 1)


def _trace_boundary(mask):
    """Ordered outer boundary of a binary mask, as (row, col) pixel coords.

    Moore-neighbour tracing. There is no contour library in this venv (no skimage, no cv2) and
    pulling one in for forty lines would be the wrong trade - `requirements.txt` is deliberately
    numpy/Pillow/scipy.

    Walk the 8-neighbourhood clockwise from the direction we arrived from, step to the first
    solid cell found, and stop when we come back to the start. The start is the topmost-then-
    leftmost solid cell, whose west neighbour is guaranteed to be background, which is what makes
    the initial direction well defined.
    """
    m = np.pad(np.asarray(mask, bool), 1)
    ys, xs = np.nonzero(m)
    start = (int(ys[0]), int(xs[0]))
    n8 = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    out = [start]
    b, prev = start, (start[0], start[1] - 1)
    cap = 8 * int(m.sum()) + 64
    while len(out) < cap:
        k = n8.index((prev[0] - b[0], prev[1] - b[1]))
        nxt = None
        for i in range(1, 9):
            c = n8[(k + i) % 8]
            p = (b[0] + c[0], b[1] + c[1])
            if m[p]:
                prev = (b[0] + n8[(k + i - 1) % 8][0], b[1] + n8[(k + i - 1) % 8][1])
                nxt = p
                break
        if nxt is None:                       # a single isolated pixel
            break
        b = nxt
        if b == start:
            break
        out.append(b)
    return np.asarray(out, dtype=np.float64)


def image_body(path, cells=120.0, flip_x=False, close_cells=1.5, detail=3.0, alpha_thr=128,
               tex_px=1024):
    """A cut-out PNG -> (closed unit-space polygon, RGB texture) that the solver and the
    renderer agree on by construction.

    Returns dict(points, rgb, ybox). `points` is in the same unit-chord space every other profile
    here uses (x 0..1 along the body, y centred on 0, +y screen-down in the horizontal format) so
    `place()` and `rasterize()` need no special case. `rgb` is the texture, already mirrored and
    cropped to match `points`' bounding box exactly.

    THREE THINGS THIS HAS TO GET RIGHT, and each is a rule this project already learned:

      - **No sub-cell gaps** (CLAUDE.md section 6 rule 4). A photographic cut-out has legs a few
        pixels apart at the FINAL lattice size, and a passage narrower than a cell is one the
        lattice cannot represent - the speed in it runs away. So the mask is closed with a disk
        of `close_cells` LATTICE CELLS, which welds shut exactly the channels that would be
        unresolvable and leaves everything wider alone. The closing radius is quoted in cells and
        converted here, rather than in pixels, so it keeps its meaning if the chord or the source
        image changes.
      - **The polygon is traced at `detail`x the lattice, on purpose.** The solver rasterises it
        at lattice resolution, but `render._draw_bodies` rasterises it at 2x SCREEN resolution -
        which for this body is several times finer. Tracing at the lattice would throw away
        detail the picture can show (the tail, the gap between the leg pairs) to satisfy a
        constraint that only the solver has. Trace fine, weld coarse: each stage then gets what
        it can actually use.
      - **The picture and the physics must be the same object.** The drawn alpha comes from the
        traced POLYGON, not from the source alpha, so a gap the closing welded shut is solid in
        the picture too. To make that look right rather than like a hole, the RGB is
        nearest-neighbour-extended outside the original silhouette (one EDT with `return_indices`)
        - so a welded-over gap picks up the colour of the leg beside it instead of the cut-out's
        background.
      - **It is still just a polygon.** The whole engine rests on "a body is an ordered closed
        polygon" (see this module's docstring); a body that arrived as pixels must join that
        contract rather than fork it, or every downstream stage - `place`, `rasterize`,
        `FreeBody`, the body draw - needs a second code path.
    """
    from scipy.ndimage import (binary_closing, binary_fill_holes, distance_transform_edt,
                               label as _label)
    im = Image.open(path).convert("RGBA")
    if flip_x:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im)
    solid = a[:, :, 3] >= int(alpha_thr)
    solid = binary_fill_holes(solid)
    lab, n = _label(solid)
    if n > 1:                                  # drop stray specks from the cut-out
        solid = lab == (1 + np.argmax([(lab == k + 1).sum() for k in range(n)]))
    ys, xs = np.nonzero(solid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    solid = solid[y0:y1, x0:x1]
    rgb = a[y0:y1, x0:x1, :3]

    # extend the colour outward so anything the closing welds shut is cow-coloured, not
    # background-coloured
    _, idx = distance_transform_edt(~solid, return_indices=True)
    rgb = rgb[idx[0], idx[1]]

    # work at `detail` x the lattice: fine enough for the 2x-screen body draw, coarse enough that
    # the closing radius below is a small integer
    w = max(8, int(round(float(cells) * float(detail))))
    h = max(8, int(round(w * solid.shape[0] / solid.shape[1])))
    small = np.asarray(Image.fromarray((solid * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR)) >= 128
    r = int(max(1, round(float(close_cells) * float(detail))))     # radius in LATTICE cells
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    small = binary_fill_holes(binary_closing(small, structure=(xx ** 2 + yy ** 2) <= r * r))
    lab, n = _label(small)
    if n > 1:
        small = lab == (1 + np.argmax([(lab == k + 1).sum() for k in range(n)]))

    c = _trace_boundary(small)                 # (row, col)
    px, py = c[:, 1], c[:, 0]
    sx0, sx1 = px.min(), px.max()
    span = max(sx1 - sx0, 1.0)
    pts = np.stack([(px - sx0) / span, (py - 0.5 * (py.min() + py.max())) / span], 1)

    # The texture is sampled NEAREST at draw time, so hand back something close to the size it
    # will be drawn at rather than a 21 MB original that would alias.
    if rgb.shape[1] > int(tex_px):
        th = max(1, int(round(int(tex_px) * rgb.shape[0] / rgb.shape[1])))
        rgb = np.asarray(Image.fromarray(rgb).resize((int(tex_px), th), Image.LANCZOS))
    return dict(points=pts, rgb=rgb,
                ybox=(float(pts[:, 1].min()), float(pts[:, 1].max())))


# --- placement + rasterisation --------------------------------------------------------
def flip_y(pts):
    """Mirror a profile about its own chord line.

    Needed because "which way up does this shape appear" is a property of the FORMAT, not of the
    shape: `place()` maps profile +y straight onto lattice +y, and lattice +y is screen-DOWN in
    the horizontal (`flow="right"`) format and screen-RIGHT in the vertical ones. So a profile
    authored the natural way - back up, hooves down; camber up, flat side down - lands upside
    down in the horizontal format and has to be mirrored there.

    It does NOT touch the angle-of-attack convention, which lives in `place`'s rotation: a
    positive AoA still sends the leading edge to screen-up either way. Mirroring only swaps which
    surface carries the camber, so a symmetric section is unchanged by it.
    """
    p = np.asarray(pts, dtype=np.float64).copy()
    p[:, 1] = -p[:, 1]
    return p


def place(pts, chord, cx, cy, aoa_deg=0.0, pivot=0.25):
    """Unit-chord profile -> lattice coordinates.

    Rotation is about `pivot` (quarter-chord by default). Lattice +y is SCREEN-DOWN in the
    horizontal format, so a positive angle of attack — nose up, into the oncoming flow — is a
    POSITIVE rotation here: the leading edge sits at x < 0, and +a sends it to y = -sin(a) < 0,
    i.e. upward. (Getting this backwards silently flips lift, suction side and all.)
    """
    a = np.deg2rad(float(aoa_deg))
    p = np.asarray(pts, dtype=np.float64).copy()
    p[:, 0] -= pivot
    ca, sa = np.cos(a), np.sin(a)
    r = np.stack([p[:, 0] * ca - p[:, 1] * sa, p[:, 0] * sa + p[:, 1] * ca], 1)
    r *= float(chord)
    r[:, 0] += float(cx)
    r[:, 1] += float(cy)
    return r


def place_uv(chord, cx, cy, aoa_deg=0.0, ylim=(-0.5, 0.5), pivot=0.25):
    """The exact analytic INVERSE of `place`, as the `to_uv(lattice_x, lattice_y)` closure that
    `render._draw_texture` wants for a body that ROTATES.

    The renderer's default texture map is the polygon's axis-aligned bounding box, which is right
    only while the body does not turn: the box of a rotated shape shears the picture off the
    outline (`tri_shapes`' cow got away with it because it is stationary, and a swept surfboard
    cannot). `place` is a similarity transform, so its inverse is closed-form and exact - no
    iteration, unlike the deforming body in `wt/rig.py`, which has to solve for an angle.

    `ylim` is the profile's own (min, max) y in unit-chord space, so v spans the body exactly.
    """
    a = np.deg2rad(float(aoa_deg))
    ca, sa = float(np.cos(a)), float(np.sin(a))
    y0, y1 = float(ylim[0]), float(ylim[1])
    span = max(y1 - y0, 1e-9)

    def to_uv(ii, jj):
        q0 = (np.asarray(ii, dtype=np.float64) - float(cx)) / float(chord)
        q1 = (np.asarray(jj, dtype=np.float64) - float(cy)) / float(chord)
        return (q0 * ca + q1 * sa) + float(pivot), ((-q0 * sa + q1 * ca) - y0) / span

    return to_uv


def rasterize(polys, nx, ny, supersample=2):
    """Closed polygons (lattice coords) -> boolean solid mask of shape (ny, nx).

    Rasterised at `supersample`x and thresholded, so a rotating body's mask grows/shrinks
    smoothly instead of jittering a whole cell at a time as the angle creeps.
    """
    s = max(1, int(supersample))
    img = Image.new("L", (nx * s, ny * s), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        if len(p) >= 3:
            d.polygon([(float(x) * s, float(y) * s) for x, y in p], fill=255)
    a = np.asarray(img, dtype=np.uint8)
    if s > 1:
        a = a.reshape(ny, s, nx, s).mean(axis=(1, 3))
    return a >= 128
