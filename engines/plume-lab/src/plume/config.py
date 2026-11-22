# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""config.py - one dataclass per clip. Everything a scene can decide lives here.

The layout numbers default to the MEASURED reference panel:
``Video_Interpreter\\rocket_pressure.mp4`` middle panel is 720x402 (aspect 1.7910), exit plane at
x = 40 px, exit radius 19 px, panel half-height 201 px. Scaled to a 1080-wide output those become
``nozzle_px = 60``, ``r_exit = 28.5``, ``nr = 302`` - i.e. **28 cells across the exit radius at
1:1**, comfortably past the ~20 needed for the Mach disk to read as a disk rather than a smudge,
so v1 needs no supersampling.
"""
from __future__ import annotations

# NOTE: `dataclasses.field` is deliberately NOT imported - `field` is a Cfg attribute name (the
# scalar being plotted) and importing it would shadow the helper at class-body evaluation time.
from dataclasses import dataclass


@dataclass
class Cfg:
    # ---- output -------------------------------------------------------------------------
    name: str = "plume"
    width: int = 1080                 # panel width (nozzle overlay + solved region)
    height: int = 604                 # 1080/604 = 1.7881, the reference panel's 1.7910
    fps: int = 60
    seconds: float = 3.0
    scale: float = 1.0                # cells per output pixel; <1 = supersample the solve

    # ---- framing ------------------------------------------------------------------------
    # "panel"    - the source's shape: ONE horizontal stream, flow left -> right.
    # "vertical" - 1080x1920, ONE stream, nozzle at the TOP spraying DOWN.
    # "triple"   - 1080x1920, THREE independent streams stacked, two divider bars.
    # "free"     - the STAGE: the exit plane is anchored anywhere in the frame at any HEADING,
    #              the field composites over a background with its own value as opacity, and the
    #              object rotates with it. See render.Stage.
    frame: str = "panel"
    bar_px: int = 12                  # divider thickness (triple only)
    bar_rgb: tuple = (7, 6, 11)
    lut_clip_hi: float = 1.0          # stop short of the ramp's dark-red top (see build_lut)

    # ---- geometry -----------------------------------------------------------------------
    # UNDISTURBED AMBIENT ahead of the painted cone, in output pixels. It pushes the exit plane
    # down the frame (`vertical`) without lengthening the drawn nozzle, so "put the head at 25%"
    # and "shrink the nozzle" are two independent numbers instead of one compromise. The solve
    # loses exactly these cells - `nz` subtracts them - so the panel length is unchanged and the
    # band costs nothing but the cells it occupies. Keep `head_px + nozzle_px` a multiple of
    # `scale` so the drawn exit line and the cone's last cell land on the same pixel.
    head_px: int = 0
    nozzle_px: int = 60               # painted divergent section, left of the exit plane
    r_exit: float = 28.5              # exit radius in output pixels
    # None = derive the throat from mach_e, so the drawn cone IS the nozzle the solver is given.
    # An override is honoured for art reasons, but note the reference's own 0.29 taper implies
    # M_e ~ 3.95 against its M_jet ~ 1.97 - its cone is decorative. See nozzle.paint.
    r_throat_frac: float = None

    # ---- nozzle / gas -------------------------------------------------------------------
    gamma: float = 1.4
    mach_e: float = 1.8               # exit Mach number
    pr_e: float = 1.30                # p_exit / p_ambient   (>1 under-, <1 over-expanded)
    tr_e: float = 2.00                # T_exit / T_ambient
    t_ramp: float = 1.0               # inflow ramp, in ambient sound-crossing times

    # ---- run ----------------------------------------------------------------------------
    t_end: float = 700.0              # simulated time, in cells / ambient sound speed
    # SOLVER TIME RUN BEFORE FRAME 0. Every clip in this project so far opens on a nozzle that
    # has just been switched on, because the starting transient IS the subject. A clip that opens
    # on a vehicle already at speed cannot: its first frame has to be a DEVELOPED jet with its
    # shock-cell train established and its blast wave already gone out through the sponge.
    # Wind_Tunnel's `settle` for the same reason, and its rule applies - a scene that opens at
    # speed needs the pre-roll, or the first seconds are a startup artefact dressed as content.
    t_settle: float = 0.0
    cfl: float = 0.40
    fused: bool = True                # fused HLLC ElementwiseKernel (GPU only)
    active: bool = True               # growing active box
    active_start: int = 192
    sponge_r: int = 28
    sponge_z: int = 32

    # ---- look ---------------------------------------------------------------------------
    field: str = "speed"
    lut: str = "plumefire"
    lut_gamma: float = 1.0
    norm: str = "minmax"              # minmax | p999 | fixed | stable | auto
    # `auto` only: top of the colour range = vmax_mult * the fully-expanded jet speed.
    # 1.00 puts the jet core exactly at the top of the (clipped) ramp; above 1 leaves
    # headroom so only genuinely faster-than-jet gas clips.
    vmax_mult: float = 1.02
    norm_lo: float = None
    norm_hi: float = None
    axis_line: bool = True            # the dotted centreline the reference draws
    exit_line: bool = True            # the dark vertical line at the exit plane
    plate_rgb: tuple = None            # None = outside the cone reads as ambient (the reference)

    # ---- the OBJECT overlay -------------------------------------------------------------
    # A cut-out PNG composited over the finished frame, straddling the exit plane, so the thing
    # the plume comes out of is a picture rather than a drawn cone.
    #
    # THE NOZZLE IS STILL PAINTED AND STILL SOLVED - it is OCCLUDED, not removed. The cone is
    # what makes the painted interior agree with the exit state the solver is given (nozzle.paint,
    # and plume_check test A gates that agreement at 0.124%); deleting it to hide it would break
    # the one thing that keeps the drawing honest. So the cone stays and the object covers it,
    # and `Scene.cover_report()` MEASURES the cover rather than assuming it.
    obj_png: str = None               # path, absolute or relative to the project root
    obj_rotate: float = 0.0           # degrees COUNTERCLOCKWISE, applied after the alpha crop
    obj_h_px: int = 0                 # height on screen after rotation; 0 = do not draw
    # Where the exit plane falls on the object's OWN height. 0 = its top edge, 1 = its bottom.
    # This is the placement knob that matters: it decides how much solid body sits over the cone
    # and how far the plume travels hidden before it clears the silhouette.
    obj_exit_frac: float = 0.70
    obj_dx_px: int = 0                # lateral nudge from the axis (+ = right)
    obj_w_px: int = 0                 # fit the WIDTH instead of the height, if set
    obj_flip_x: bool = False          # mirror the subject (which way it faces)
    # `free` only: where the exit plane sits in the object's OWN image, as (fx, fy) fractions.
    # The object is rotated about exactly this point, so the jet stays glued to it at any
    # heading - which is what a nozzle bolted to a body does.
    obj_anchor: tuple = (0.06, 0.20)

    # ---- the FREE framing (frame="free") -------------------------------------------------
    # The solved domain quoted in OUTPUT PIXELS rather than derived from the frame, because in
    # this framing the stream runs diagonally and the panel's own edges mean nothing. Size it to
    # cover the picture and no more: cells outside the frame are cells nobody sees.
    stage_z_px: int = 1200            # solved stream length, from the exit plane downstream
    stage_r_px: int = 420             # solved half-width, either side of the jet axis
    anchor: tuple = (0.72, 0.40)      # exit plane in the frame, as fractions of (width, height)
    # Heading above horizontal, in degrees, at t = 0. It rolls smoothly to zero by `tilt_end`
    # seconds and holds. Smoothstep, so the RATE is zero at both ends too - a corner in the
    # attitude reads as a jolt, the same reason every ramp in this library is C1.
    tilt_deg: float = 0.0
    tilt_end: float = 0.0
    # 0 = smoothstep (C1 at both ends). Any positive value selects `(1 - u) ** tilt_ease`, an
    # EASE-OUT: it turns fastest at the start and settles to level with zero rate. Use it when
    # the brief names a waypoint - the exponent is then solved from that waypoint rather than
    # dialled in (see `cow_reentry`, where "two thirds done by 3.5 s of 9" gives 2.232).
    tilt_ease: float = 0.0
    # The field is ADDED to the background, weighted by (normalised value) ** this. Ambient is
    # exactly the bottom of the range, so it adds exactly nothing and the solved domain's
    # rectangle can never show. Below 1 lifts the faint halo into a visible wash; above 1 keeps
    # the shot to the jet itself. See render.Stage for why this is additive and not an opacity.
    stage_alpha_gamma: float = 0.85
    # How much the field HIDES what is behind it, as (normalised value) ** this. Emission alone
    # leaves stars streaming straight through the jet core, which is the one thing an optically
    # thick exhaust does not allow. A high exponent is the point: at 3.0 the halo occludes 0.3%
    # (so it cannot darken the sky - that was the first cut's bug) while the core occludes 73%.
    stage_occlude_gamma: float = 3.0
    stage_feather_px: int = 90        # smooth the domain's own boundary to nothing

    # ---- the star field (drawn, never solved - see stars.py) -----------------------------
    stars: int = 0                    # 0 = no star field
    star_layers: tuple = ((1.00, 1.7, 1.00), (0.62, 1.2, 0.80), (0.34, 0.9, 0.62),
                          (0.16, 0.7, 0.46))
    star_speed: float = 0.0           # px/s for a multiplier-1.0 star, at t = 0
    star_speed_end: float = 0.0       # px/s at the end of the clip (0 = hold `star_speed`)
    star_accel_from: float = 0.0      # seconds at which the acceleration begins
    star_streak: float = 1.0          # exposure, as a multiple of one frame's travel
    star_gain: float = 1.0            # 1.0 = a brightness-1 star peaks at white
    star_seed: int = 7

    # ---- the atmospheric-friction glow (drawn, never solved - see heat.py) ---------------
    # There is no oncoming freestream in this solve, so there is no bow shock and nothing that
    # could produce aerodynamic heating. This is a drawn stand-in, matched to a reference frame.
    heat: int = 0                     # filament count; 0 = off
    # Stand-off of the sheath AHEAD OF THE SUBJECT'S OWN SURFACE - the shape comes from the
    # silhouette (`StageObject.lead_profile`), not from a circle, so this is the one number
    # that says how far off the skin the glow sits.
    heat_ahead_px: float = 26.0
    heat_len_px: float = 620.0        # how far the trail sweeps back
    heat_spread: float = 1.5          # lateral growth of the trail from body to tail
    heat_rate: float = 1.6            # how fast material appears to pour off the cap
    heat_gain: float = 1.0
    heat_seed: int = 5
    heat_thick_px: float = 34.0       # how far a filament wanders across the flow
    heat_writhe: float = 1.0          # how fast that wander travels - 0 freezes the shapes

    # ---- vibration ------------------------------------------------------------------------
    # Shakes the PIVOT, so the subject, its jet and its sheath move together against a steady
    # star field - which reads as the vehicle shuddering rather than as a handheld camera.
    shake_px: float = 0.0             # peak translation at t = 0
    shake_deg: float = 0.0            # peak rotation at t = 0
    shake_end: float = 0.0            # seconds at which it reaches exactly zero
    shake_ease: float = 1.3

    # ---- pivot drift ----------------------------------------------------------------------
    # A starting OFFSET of the pivot that decays to nothing by `drift_end`, so the subject opens
    # displaced and works its way back to where it belongs. Smoothstep by default: it is held
    # early, recovers through the middle and ARRIVES with zero velocity, which reads as settling
    # into place rather than stopping dead against a wall. Moves the subject, its jet and its
    # sheath together, because they are bolted to each other.
    drift_px: tuple = (0.0, 0.0)      # (x, y) offset at t = 0; +y is DOWN the screen
    drift_end: float = 0.0            # seconds at which the offset reaches exactly zero
    drift_ease: float = 0.0           # 0 = smoothstep; >0 selects (1 - u) ** drift_ease

    # ---- bloom ----------------------------------------------------------------------------
    # DELIBERATELY OFF BY DEFAULT, and this is a departure worth flagging: `plumefire` was
    # measured off an ENCODED clip, so it is already downstream of whatever that renderer did,
    # and blooming a post-bloom palette compounds the desaturation (the trap that bit
    # Lattice_Grid). That argument holds for every clip that is trying to MATCH the reference.
    # It does not hold for `rocketfire`, which is authored rather than measured, and where the
    # glow is the point.
    bloom: float = 0.0                # 0 = off; gain on the blurred bright pass
    bloom_thr: float = 118.0          # luma above which a pixel contributes
    bloom_tight: float = 0.55         # weight of the second, tighter scale

    # ---- the ASCENT background (drawn, never solved - see sky.py) -------------------------
    # Everything here is a function of ALTITUDE, and altitude is integrated from the subject's
    # OWN speed and heading (`dh/dt = v * sin(theta)`), so the sky changes because of what the
    # vehicle is doing rather than because of the clock.
    sky: bool = False
    sky_alt_max: float = 120.0        # km at the last frame; the SHAPE is the gravity turn's
    sky_alt_floor: float = 0.13       # residual climb after pitchover, as a fraction of v
    sky_clouds: int = 8
    sky_cloud_km: float = 6.5         # cumulus scale height - clouds are gone by ~20 km
    sky_scale_km: float = 14.0        # the diffuse blue column
    sky_limb_km: float = 30.0         # a limb cannot exist until you are above the air
    sky_horizon_km: float = 30.0      # how fast the horizon rises into frame and bows
    sky_cloud_par: float = 6.0        # cloud drift, as a multiple of the star velocity
    sky_seed: int = 3

    # ---- how the PLUME responds to altitude -----------------------------------------------
    # NOT a change to the solve, and not a fake: the same field is rendered with more or less of
    # it luminous. In thick air the exhaust entrains and afterburns the surrounding atmosphere,
    # so what you SEE is a big turbulent mass far wider than the nozzle; in vacuum there is
    # nothing to entrain and only the exhaust itself shows, which is fine and sharp. Lowering
    # `vmax` lights the slower outer gas (the solved shear layer, with its real Kelvin-Helmholtz
    # billows) and lowering the alpha exponent makes even faint field opaque - between them that
    # IS the fat/turbulent look, and it is made of solved structure rather than drawn smoke.
    # 0 disables the coupling and the plain `vmax_mult` / `stage_alpha_gamma` are used.
    plume_vmax_lo: float = 0.0        # vmax_mult at sea level (vs `vmax_mult` in vacuum)
    plume_alpha_lo: float = 0.0       # stage_alpha_gamma at sea level
    plume_atm_km: float = 20.0        # scale height over which it hands over

    # ---- encode -------------------------------------------------------------------------
    crf: int = 16
    nvenc: bool = True

    # ---- triple only --------------------------------------------------------------------
    # Exactly 3 dicts of per-panel overrides (mach_e / pr_e / tr_e / gamma / field / lut / label).
    # A tuple, not a list: dataclass defaults must be immutable.
    variants: tuple = None
    # Share ONE colour range across panels plotting the same field. Keep this ON for any
    # comparison: per-panel `auto` ranges normalise away the difference being compared.
    share_range: bool = True
    # Share ONE solver across all three panels. Only for "same flow, three instruments" - Scene
    # refuses it if any variant overrides a physics parameter.
    share_sim: bool = False

    # ---- derived ------------------------------------------------------------------------
    @property
    def frames(self):
        return int(round(self.seconds * self.fps))

    @property
    def panel_h(self):
        """Height of ONE stream panel in output pixels.

        The triple layout is SOLVED, not placed - `3*D + 2*B = height` - because three ducts of
        unequal height would confound the comparison the format exists to make. Same discipline
        as Wind_Tunnel's road set solving `2D + 2B` for D.
        """
        if self.frame == "triple":
            return (self.height - 2 * self.bar_px) // 3
        return self.height

    @property
    def _free(self):
        return self.frame == "free"

    @property
    def nr(self):
        """Radial cells - HALF the domain; the render mirrors it.

        The stream runs along the panel's LONG axis in the panel framings: across in
        `panel`/`triple`, down in `vertical`. So the radial extent is the panel's short side.
        `free` states it outright instead, because a diagonal stream has no such side.
        """
        if self._free:
            return int(round(self.stage_r_px / self.scale))
        short = self.width if self.frame == "vertical" else self.panel_h
        return int(round(short / 2 / self.scale))

    @property
    def nz(self):
        """Axial cells, minus the painted nozzle AND the ambient head band above it."""
        if self._free:
            return int(round((self.stage_z_px - self.nozzle_px) / self.scale))
        long = self.height if self.frame == "vertical" else self.width
        return int(round((long - self.nozzle_px - self.head_px) / self.scale))

    @property
    def exit_px(self):
        """Where the exit plane sits along the stream, in output pixels."""
        return self.head_px + self.nozzle_px

    @property
    def r_exit_cells(self):
        return self.r_exit / self.scale

    def summary(self):
        # cells reflect SOLVES, not panels: share_sim means one solve feeding three panels
        n = 1 if (self.share_sim or self.frame != "triple") else 3
        return (f"{self.name}: {self.width}x{self.height}@{self.fps} {self.seconds}s "
                f"({self.frames} frames) frame={self.frame} | solve {self.nz}x{self.nr} "
                f"half-domain x{n} ({n * self.nz * self.nr / 1e3:.0f}k cells) | "
                f"field={self.field} lut={self.lut} norm={self.norm} clip_hi={self.lut_clip_hi}"
                + (f"\n  stage: anchor {self.anchor} heading {self.tilt_deg} deg -> 0 by "
                   f"{self.tilt_end}s | settle {self.t_settle} | stars {self.stars} "
                   f"{self.star_speed:.0f}->{self.star_speed_end:.0f} px/s from {self.star_accel_from}s"
                   if self._free else
                   f"\n  exit plane at {self.exit_px} px "
                   f"({self.exit_px / max(self.height,1) * 100:.0f}% down; head {self.head_px} "
                   f"+ cone {self.nozzle_px}) r_exit={self.r_exit}")
                + f"\n  M_e={self.mach_e} p_e/p_inf={self.pr_e} T_e/T_inf={self.tr_e} "
                  f"gamma={self.gamma} | t_end={self.t_end}")
