# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""render.py — the pipeline: solve -> colourise -> streaks -> body -> film -> ffmpeg.

manim-free, same shape as the oscilloscope engine's render.py: this module owns the clock and the rawvideo
pipe, and nothing else knows about video.

Per output frame
----------------
    scene.bodies(t)            closed polygons in lattice coords - what is DRAWN
    scene.solid_polys(t)       what the SOLVER is given (defaults to the same list)
        -> shapes.rasterize    boolean solid mask -> lbm.set_solid
    lbm.run(steps_per_frame)   the actual physics
    field = speed|vort|p       (ny, nx) lattice field
        -> map_coordinates     bilinear upsample straight into SCREEN orientation (one warp
                               does both the resize and the flow="up" 90-degree mapping)
        -> colormap LUT        uint8 RGB at 1080x1920
    streaks.draw(...)          additive white comet-tails, splatted at screen res
    body                       filled + outlined at 2x and box-downsampled (clean AA edge)
    [FilmLook]                 optional CRT/camera pass
        -> raw RGB24 -pipe-> ffmpeg -> out\\<scene>.mp4

Timebase
--------
`cfg.steps` is LBM steps per frame **at cfg.fps_ref (60) fps**, and the renderer scales it by
fps_ref/fps with a fractional accumulator. That is what makes a 30 fps preview and a 60 fps
final show the *same flow at the same moment* instead of one running at double speed — the same
"frames driven by real seconds" rule the oscilloscope engine uses.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from . import colormap, shapes
from .config import OUT, RenderConfig, encoder_args, find_ffmpeg
from .gpu import GPU, asnumpy, gaussian_filter, map_coordinates, xp
from .lbm import LBM
from .streaks import Streaks

FPS_REF = 60.0          # cfg.steps is quoted per frame at this rate


class Tunnel:
    """One configured simulation + its renderer. Reusable for stills and for full renders."""

    def __init__(self, scene, cfg: RenderConfig):
        self.scene, self.cfg = scene, cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        self.W, self.H = cfg.width, cfg.height
        self.steps_pf = cfg.steps * FPS_REF / cfg.fps
        self._step_acc = 0.0

        # `self.lbm` keeps its name whichever solver is in it: every call site downstream (the
        # streaks, the field maps, the scenes' prepare()/substep()) speaks one interface, and
        # renaming it would touch far more than it would clarify.
        if getattr(cfg, "solver", "lbm") == "cns":
            from .cns import CNS
            self.lbm = CNS(self.nx, self.ny, u0=cfg.u0, re=cfg.re,
                           ref_len=scene.ref_len(), csm=cfg.csm,
                           cfl=cfg.extras.get("cfl", 0.50), sponge_out=cfg.sponge_out,
                           side_bc=cfg.side_bc, wall_clamp=cfg.wall_clamp, closed=cfg.closed,
                           blockage=cfg.extras.get("blockage", 4.2))
        else:
            self.lbm = LBM(self.nx, self.ny, u0=cfg.u0, re=cfg.re,
                           ref_len=scene.ref_len(), csm=cfg.csm,
                           sponge_side=cfg.sponge_side, sponge_out=cfg.sponge_out,
                           side_bc=cfg.side_bc, wall_clamp=cfg.wall_clamp, closed=cfg.closed)
        # cfg.n_streaks is quoted at 1080x1920; scale by area so density, not count, is fixed
        # ...and again by the overscan ratio, since tracers are seeded over the WHOLE lattice:
        # without this the on-screen streaks would thin out as the off-screen margin grew
        n_str = max(200, int(round(cfg.n_streaks * (self.W * self.H) / (1080 * 1920)
                                   * (self.nx * self.ny)
                                   / float(cfg.vis_nx * cfg.vis_ny))))
        self.streaks = Streaks(self.nx, self.ny, n=n_str, fps=cfg.fps,
                               steps=self.steps_pf, tail=cfg.streak_tail,
                               span=cfg.streak_span, life=cfg.streak_life, seed=cfg.seed,
                               period=float(scene.duration) if cfg.loop else 0.0,
                               # one LBM step IS one time unit; the CNS holds a fixed dt ~0.135,
                               # and without this the tracers outrun the fluid by 1/dt
                               step_dt=float(getattr(self.lbm, "dt", 1.0)))
        # Optional transported species. A scene opts in by defining `dye_spec()`; nothing else
        # in the registry pays for it, in memory or in time.
        self.dye = None
        spec = getattr(scene, "dye_spec", None)
        if spec is not None:
            from .dye import Dye
            s = spec()
            self.dye = Dye(self.nx, self.ny, species=len(s["colors"]),
                           tau=float(s.get("tau", 0.515)))
            self.dye_colors = xp.asarray(np.asarray(s["colors"], dtype=np.float32))
            self.dye_gain = float(s.get("gain", 1.0))

        self._build_maps()
        self._coords_np = None            # host copy, built on first textured body draw
        self._font = None
        self._last_poly_key = None
        self._legend = False              # False = not built yet; None = this scene has no legend

    # -- lattice <-> screen ------------------------------------------------------------
    def _build_maps(self):
        """Precompute the map_coordinates sampling grid (screen px -> lattice coords).

        flow="up":    lattice +x (streamwise) points UP the screen, +y points right.
        flow="down":  lattice +x points DOWN the screen, +y points LEFT.
        flow="right": lattice +x points right, +y points down. (classic tunnel view)
        flow="left":  lattice +x points LEFT, +y points UP.

        Each of the four is orientation-PRESERVING (the 2x2 map has det > 0). That is not
        cosmetic: a mirrored mapping would flip the sign of every vortex on screen, so
        `--field vort` would lie and a cambered section would appear to lift the wrong way. For
        "down" that is why +y goes LEFT rather than right - taking the obvious "+x down, +y
        right" would have been a reflection. "left" is the same story on the horizontal axis: it
        is `right` turned through 180 degrees (BOTH axes negated), not `right` with the stream
        mirrored, so nothing about the handedness of the picture changes.

        "left" exists for a body that has to face RIGHT while the air comes at it - the bullet
        series, where the projectile travels downrange to the right and the relative wind
        therefore blows to the left. Mirroring the body instead would have been the reflection
        this method exists to avoid.
        """
        H, W = self.H, self.W
        cfg = self.cfg
        # the screen shows only the VISIBLE window of the lattice; with overscan=0 that is the
        # whole domain and this reduces to the original edge-to-edge mapping
        self.x0, self.y0 = cfg.vis_x0, cfg.vis_y0
        vnx, vny = cfg.vis_nx, cfg.vis_ny
        r = np.arange(H, dtype=np.float32)
        c = np.arange(W, dtype=np.float32)
        if cfg.flow == "up":
            self.fx_c, self.fx_r = (vny - 1) / (W - 1), (vnx - 1) / (H - 1)
            j = self.y0 + c[None, :] * self.fx_c + np.zeros((H, 1), np.float32)
            i = self.x0 + (vnx - 1) - r[:, None] * self.fx_r + np.zeros((1, W), np.float32)
        elif cfg.flow == "down":
            self.fx_c, self.fx_r = (vny - 1) / (W - 1), (vnx - 1) / (H - 1)
            j = self.y0 + (vny - 1) - c[None, :] * self.fx_c + np.zeros((H, 1), np.float32)
            i = self.x0 + r[:, None] * self.fx_r + np.zeros((1, W), np.float32)
        elif cfg.flow == "left":
            self.fx_c, self.fx_r = (vnx - 1) / (W - 1), (vny - 1) / (H - 1)
            j = self.y0 + (vny - 1) - r[:, None] * self.fx_r + np.zeros((1, W), np.float32)
            i = self.x0 + (vnx - 1) - c[None, :] * self.fx_c + np.zeros((H, 1), np.float32)
        else:
            self.fx_c, self.fx_r = (vnx - 1) / (W - 1), (vny - 1) / (H - 1)
            j = self.y0 + r[:, None] * self.fx_r + np.zeros((1, W), np.float32)
            i = self.x0 + c[None, :] * self.fx_c + np.zeros((H, 1), np.float32)
        self._coords = xp.asarray(np.stack([j, i]))          # field is indexed [j, i]

    def to_screen(self, px, py):
        """Lattice point arrays -> screen pixel arrays (used by the streak splatter)."""
        if self.cfg.flow == "up":
            return ((py - self.y0) / self.fx_c,
                    (self.x0 + self.cfg.vis_nx - 1 - px) / self.fx_r)
        if self.cfg.flow == "down":
            return ((self.y0 + self.cfg.vis_ny - 1 - py) / self.fx_c,
                    (px - self.x0) / self.fx_r)
        if self.cfg.flow == "left":
            return ((self.x0 + self.cfg.vis_nx - 1 - px) / self.fx_c,
                    (self.y0 + self.cfg.vis_ny - 1 - py) / self.fx_r)
        return (px - self.x0) / self.fx_c, (py - self.y0) / self.fx_r

    def poly_to_screen(self, p):
        sx, sy = self.to_screen(p[:, 0], p[:, 1])
        return np.stack([asnumpy(sx), asnumpy(sy)], 1)

    # -- simulation ---------------------------------------------------------------------
    def set_bodies(self, t):
        # The DRAWN set and the SOLID set are separate questions. `Scene.solid_polys` defaults to
        # `bodies(t)` - the same list - so every scene written before that hook is untouched; a
        # scene overrides it when part of what the viewer sees must not be a wall (the cow's legs,
        # see the hook's docstring). Keyed on the SOLID list, because that is what the raster
        # depends on: a scene whose drawn body moves over a stationary solid one correctly skips
        # the re-raster.
        polys = self.scene.bodies(t)
        solid_polys = self.scene.solid_polys(t)
        key = tuple(np.round(p, 2).tobytes() for p in solid_polys)
        if key != self._last_poly_key:                        # skip the raster when nothing moved
            mask = shapes.rasterize(solid_polys, self.nx, self.ny)
            self.lbm.set_solid(mask)
            if self.dye is not None:
                self.dye.set_solid(self.lbm.solid)
            self._last_poly_key = key
        self._polys = polys
        return polys

    def _solve(self, n):
        """`n` solver steps, with the species lattices marched in lockstep beside the fluid.

        Lockstep, not once-per-frame: at ~95 steps/frame a parcel crosses a third of the engine
        between frames, so advecting the dye with a single frame-end velocity would put the
        colour somewhere the fluid never was.
        """
        if self.dye is None:
            self.lbm.run(n)
            return
        for _ in range(int(n)):
            self.lbm.step()
            self.dye.step(self.lbm.ux, self.lbm.uy)

    def _dyn_step(self, t, freeze=False):
        """One LBM step for a DYNAMIC scene: move the free bodies, re-mask, then solve."""
        self.lbm.set_solid(self.scene.substep(self.lbm, t, 1.0, freeze=freeze))
        sysm = getattr(self.scene, "sys", None)
        if sysm is not None and os.environ.get("WT_WALLU", "1") != "0":
            self.lbm.set_wall_velocity(sysm.wux, sysm.wuy)
        self._polys = self.scene.bodies(t)
        self.lbm.step()

    def settle(self, seconds, progress=True):
        """Run the flow BEFORE frame 0 so the field opens established, not from a uniform slab."""
        n = int(round(seconds * self.cfg.steps * FPS_REF))
        if n <= 0:
            return
        if hasattr(self.scene, "prepare"):
            self.scene.prepare(self.lbm)   # seed a starting field (e.g. a standing vortex)
        self.set_bodies(0.0)
        if progress:
            print(f"  settling {seconds:g}s of flow ({n} LBM steps)...", flush=True)
        dyn = getattr(self.scene, "dynamic", False)
        # OPT-IN pre-roll hook. A scene whose body has an internal degree of freedom the FLOW
        # sets - `_FarmSet`'s hinged tail - cannot know where that lands until there is a flow
        # to read, so it has to converge during the settle like everything else. Without it the
        # clip opens at whatever the scene guessed and spends the first seconds of a 10 s take
        # ringing back to the real answer, which is a startup transient dressed as physics.
        # Called once per FRAME-equivalent, not per step: these are slow
        # degrees of freedom and the mask is only re-rasterised when it actually moved.
        # Nothing else in the registry pays for it - no `settle_step`, no calls, no cost.
        pre = getattr(self.scene, "settle_step", None)
        every = max(1, int(self.cfg.steps))
        kick_at = int(0.35 * n)              # early enough that the transient is gone by frame 0
        for k in range(n):
            if k == kick_at and self.cfg.perturb > 0:
                self.lbm.perturb(self.cfg.perturb, seed=self.cfg.seed)
            # bodies already on stage at t=0 are PRESENT but frozen while the flow builds up
            # around them; anything with a later `born` time is still absent
            if dyn:
                self._dyn_step(0.0, freeze=True)
            else:
                # the species are NOT marched during settling: the point of settling is to
                # establish the velocity field, and a scene's dye schedule starts at t=0
                self.lbm.step()
            if pre is not None and (k + 1) % every == 0:
                pre(self.lbm, every)
                self.set_bodies(0.0)
            if progress and n >= 400 and k % (n // 4) == 0 and k:
                print(f"    {100 * k // n}%", flush=True)
        # Let the tracers find the established field instead of starting on a uniform grid. In
        # LOOPING mode this is not cosmetic: the pre-roll must be at least one longest tracer
        # cycle so that every slot has been reborn on schedule and no memory of its arbitrary
        # initial placement survives into frame 0 - which is what makes the ensemble at t=0 and
        # at t=T the same ensemble. The clock lands exactly on 0 at the end of it.
        warm = self.streaks.warmup_frames()
        if self.streaks.period > 0:
            self.streaks.reset_clock(-warm / float(self.cfg.fps))
            if progress:
                print(f"  loop: {warm} tracer pre-roll frames "
                      f"(period {self.streaks.period:g}s)", flush=True)
        for _ in range(warm):
            self.streaks.advance(self.lbm.ux, self.lbm.uy, self.lbm.solid)

    def advance(self, t):
        """One output frame of physics."""
        self._step_acc += self.steps_pf
        n = int(self._step_acc)
        self._step_acc -= n
        if getattr(self.scene, "dynamic", False):
            dt = 1.0 / (self.cfg.fps * self.steps_pf)      # scene-seconds per LBM step
            for k in range(n):
                self._dyn_step(t + k * dt)
        else:
            self.set_bodies(t)
            # A STATIC scene may still be forcing the fluid - a fan spooling up, a jet cutting
            # out. `prepare()` only fires once before frame 0, and `substep()` belongs to the
            # free-body path, so without this hook a scene whose bodies do not move has no way to
            # change anything over time. Called once per FRAME, which is ample: the schedules
            # this drives are seconds long, so the per-frame increment is a fraction of a percent.
            if hasattr(self.scene, "forcing"):
                self.scene.forcing(self.lbm, t)
            if self.dye is not None and hasattr(self.scene, "dye_forcing"):
                self.scene.dye_forcing(self.dye, t)
            self._solve(n)
        self.streaks.advance(self.lbm.ux, self.lbm.uy, self.lbm.solid)

    # -- image ---------------------------------------------------------------------------
    def _field01(self):
        cfg = self.cfg
        if cfg.field == "vort":
            v = self.lbm.vorticity()
            s = 0.5 + 0.5 * xp.clip(v / (cfg.u0 * 0.55), -1.0, 1.0)
        elif cfg.field == "pressure":
            p = self.lbm.pressure()
            s = 0.5 + 0.5 * xp.clip(p / (cfg.u0 * cfg.u0 * 1.6), -1.0, 1.0)
        elif cfg.field == "mach":
            # CNS only. Normalised so that the top of the ramp is exactly M = vmax_mach, and the
            # SONIC LINE therefore lands at a fixed, quotable place on the colour bar - which is
            # the whole reason to plot Mach rather than speed: "where is M = 1" is the question a
            # compressible picture has to be able to answer.
            s = xp.clip(self.lbm.mach() / cfg.extras.get("vmax_mach", 1.4), 0.0, 1.0)
            if abs(cfg.gamma - 1.0) > 1e-3:
                s = s ** cfg.gamma
        elif cfg.field == "stag":
            # Stagnation pressure, normalised so the top of the ramp is the dynamic pressure of
            # the SAME speed the `speed` field would have put there (u0*vmax). That pairing is
            # what makes the two fields comparable at a glance instead of each having its own
            # invented scale, and it means `vmax` keeps meaning one thing across both.
            full = cfg.extras.get("stag_max", 0.5 * (cfg.u0 * cfg.vmax) ** 2)
            s = xp.clip(self.lbm.stagnation() / full, 0.0, 1.0)
            if abs(cfg.gamma - 1.0) > 1e-3:
                s = s ** cfg.gamma
        elif cfg.field == "schlieren":
            # CNS only. |grad rho| / rho, tone-mapped with 1-exp(-k x) rather than clipped: a
            # shock's density gradient is orders of magnitude above the wake's, so a linear scale
            # shows the shock and nothing else, which is the opposite of what a schlieren image
            # is for.
            s = 1.0 - xp.exp(-self.lbm.schlieren() * cfg.extras.get("schlieren_gain", 260.0))
        else:
            s = xp.clip(self.lbm.speed() / (cfg.u0 * cfg.vmax), 0.0, 1.0)
            if abs(cfg.gamma - 1.0) > 1e-3:
                s = s ** cfg.gamma
        return s.astype(xp.float32)

    def _streak_intensity(self):
        """Raw streak layer at screen res, before `streak_gain`/clip - split out of `frame()`
        (2026-09-05, unchanged behaviour there) so a sky-panels compositor (not included here)
        can gate it by the disturbance alpha without duplicating the draw
        call's own parameters and risking the two disagreeing."""
        cfg = self.cfg
        return self.streaks.draw(self.lbm.ux, self.lbm.uy, self.to_screen, self.W, self.H,
                                 sigma=cfg.streak_sigma, speed_ref=cfg.u0,
                                 speed_gain=cfg.streak_speed_gain)

    def field_and_alpha(self):
        """For the sky-panels format (`cfg.field == "disturb"`): the coloured field AND a
        disturbance alpha, at screen res, float - (rgb 0-255, alpha 0-1). NOT used by `frame()`
        and touches nothing there; a compositor not included here composites this over a
        background itself, because a `Tunnel` here has no background of its own to composite
        against. Colour is `speed01`, unchanged from every other clip's mapping -
        only WHERE it shows is new, so the familiar jet ramp still means what it always has.
        """
        cfg = self.cfg
        big = map_coordinates(self._field01(), self._coords, order=1, mode="nearest")
        rgb = colormap.apply(big, cfg.cmap, cfg.saturation, cfg.brightness).astype(xp.float32)

        # d = |u - u_inf| / u0, zeroed on solid first (lbm.ux/uy are NOT zero
        # inside a solid cell - only the accessors mask it, and there is no accessor for this).
        ux = xp.where(self.lbm.solid, cfg.u0, self.lbm.ux)
        uy = xp.where(self.lbm.solid, 0.0, self.lbm.uy)
        d = xp.sqrt((ux - cfg.u0) ** 2 + uy ** 2) / max(cfg.u0, 1e-9)
        d_big = map_coordinates(d, self._coords, order=1, mode="nearest")
        lo = float(cfg.extras.get("alpha_lo", 0.08))
        hi = float(cfg.extras.get("alpha_hi", 0.30))
        s = xp.clip((d_big - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        alpha = s * s * (3.0 - 2.0 * s)                          # smoothstep
        blur_px = float(cfg.extras.get("alpha_blur_px", 3.0))
        if blur_px > 0:
            alpha = gaussian_filter(alpha, blur_px)
        return rgb, xp.clip(alpha, 0.0, 1.0)

    def frame(self, t):
        """Compose one RGB uint8 (H, W, 3) screen frame from the CURRENT solver state."""
        cfg = self.cfg
        big = map_coordinates(self._field01(), self._coords, order=1, mode="nearest")
        rgb = colormap.apply(big, cfg.cmap, cfg.saturation, cfg.brightness).astype(xp.float32)
        if self.dye is not None:
            rgb = self._blend_dye(rgb, big)

        st = self._streak_intensity()
        rgb = rgb + (xp.clip(st * cfg.streak_gain, 0.0, 1.6) * 255.0)[:, :, None]
        if cfg.extras.get("bloom", 0.0) > 0.0:
            rgb = self._add_bloom(rgb)
        out = asnumpy(xp.clip(rgb, 0, 255).astype(xp.uint8))

        out = self._draw_bodies(out, t)
        if cfg.fbd:
            out = self._draw_arrows(out, t)
        if cfg.legend:
            out = self._draw_legend(out)
        if cfg.axes:
            out = self._draw_axes(out, t)
        if cfg.panel:
            out = self._draw_panel(out, t)
        if cfg.extras.get("hud"):
            out = self._draw_hud(out, t)
        return out

    def _add_bloom(self, rgb):
        """Additive two-scale bloom over the FAST region.

        Keyed to SPEED, deliberately, even when the picture is showing something else. Bloom is
        standing in for the light a hot high-velocity exhaust would actually throw, and that is a
        property of the flow, not of whichever scalar is currently on screen - so pointing it at
        `cfg.field` would make the glow move when only the colour scheme changed.

        Two sigmas rather than one: a tight core that reads as brightness on the plume itself and
        a wide halo that spills onto the surrounding tunnel. A single blur can do one or the
        other and looks like a smudge attempting both. Everything below `bloom_thr` of full scale
        contributes nothing, so the ambient tunnel - which covers most of the frame - stays crisp
        instead of being washed into haze.
        """
        cfg = self.cfg
        e = cfg.extras
        thr = float(e.get("bloom_thr", 0.45))
        sp = xp.clip(self.lbm.speed() / (cfg.u0 * cfg.vmax), 0.0, 1.0)
        big = map_coordinates(sp, self._coords, order=1, mode="nearest")
        hot = xp.clip((big - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0) ** 1.5
        s1 = float(e.get("bloom_sigma", 7.0))
        g = 0.62 * gaussian_filter(hot, s1) + 0.38 * gaussian_filter(hot, s1 * 3.4)
        tint = xp.asarray(np.asarray(e.get("bloom_tint", (1.00, 0.86, 0.66)), dtype=np.float32))
        return rgb + (g * float(e.get("bloom", 0.0)) * 255.0)[:, :, None] * tint

    def _blend_dye(self, rgb, speed01):
        """Tint the frame by species concentration, keeping the speed field as the brightness.

        The two jobs are split on purpose. HUE says WHICH fluid a parcel is; VALUE still says how
        fast it is going. So a propellant sitting nearly still in its tank is a dark version of
        its own colour and the plume is a blazing version of the mixture, and the clip does not
        have to give up the speed reading to gain the material reading.

        Mixing falls out of the sum rather than being a special case: the tint is the
        concentration-weighted mean of the species colours, so equal parts of an amber and a
        cyan propellant land between them, and a parcel that is mostly one reads as mostly that
        one. `a` is the total concentration, clipped, so undyed tunnel fluid keeps the ordinary
        colormap and there is no seam where the dye front ends.
        """
        c = self.dye.concentration()                                    # (K, ny, nx)
        big = xp.stack([map_coordinates(ck, self._coords, order=1, mode="nearest")
                        for ck in c])                                   # (K, H, W)
        tot = big.sum(axis=0)
        a = xp.clip(tot * self.dye_gain, 0.0, 1.0)[:, :, None]
        # concentration-weighted mean colour; the max() guards the undyed cells, where the
        # weights are all zero and the ratio would be 0/0.
        # The 255 is load-bearing: `colormap.apply` works on 0-255 and `dye_spec` quotes colours
        # as 0-1 floats, so without it every dyed cell composites toward BLACK - which is exactly
        # what the first render did, over the whole plume.
        tint = 255.0 * (big[:, :, :, None] * self.dye_colors[:, None, None, :]).sum(axis=0) \
            / xp.maximum(tot, 1e-6)[:, :, None]
        # brightness from the speed field, floored so a slow-moving propellant is still legible
        v = (0.35 + 0.65 * xp.clip(speed01, 0.0, 1.0))[:, :, None]
        return rgb * (1.0 - a) + a * tint * v

    def _draw_bodies(self, rgb, t=0.0):
        """Filled body + bright outline, rasterised at 2x and box-downsampled for a clean edge.

        Only the BOUNDING BOX of the bodies is rasterised, not the whole frame. Bodies cover a
        few percent of a 1080x1920 picture, but allocating two full-frame 2x images (2160x3840
        'L' each) and box-resizing them every frame cost ~230 ms — 41% of total frame time, more
        than the entire fluid solve. Cropping to the bodies makes it ~10 ms for pixel-identical
        output. (Measured directly.)
        """
        cfg = self.cfg
        if not self._polys:
            return rgb
        s = 2
        scr = [self.poly_to_screen(p) for p in self._polys]
        tex = self._textures(t)
        pad = cfg.body_edge_px + 3
        # body_edge_px = 0 means NO outline at all - the body is a hole punched in the field, not
        # an object drawn on top of it. `black_holes` needs that; everything else wants the rim.
        rim = cfg.body_edge_px > 0

        def crop(q):
            """One polygon's own screen box, clipped to the frame. None if it is off-camera."""
            a = max(0, int(np.floor(q[:, 0].min())) - pad)
            b = max(0, int(np.floor(q[:, 1].min())) - pad)
            c = min(self.W, int(np.ceil(q[:, 0].max())) + pad)
            d = min(self.H, int(np.ceil(q[:, 1].max())) + pad)
            return None if (c <= a or d <= b) else (a, b, c, d)

        def stroke(q, box, draw_edge):
            """Composite one flat polygon (fill, or its rim) inside `box`."""
            a, b, c, d = box
            w, h = c - a, d - b
            m = Image.new("L", (w * s, h * s), 0)
            pts = [((float(x) - a) * s, (float(y) - b) * s) for x, y in q]
            if draw_edge:
                ImageDraw.Draw(m).line(pts + [pts[0]], fill=255,
                                       width=cfg.body_edge_px * s, joint="curve")
            else:
                ImageDraw.Draw(m).polygon(pts, fill=255)
            al = np.asarray(m.resize((w, h), Image.BOX), np.float32)[:, :, None] / 255.0
            sub = (slice(b, d), slice(a, c))
            col = np.asarray(cfg.body_edge if draw_edge else cfg.body_fill, np.float32)
            rgb[sub] = np.clip(rgb[sub].astype(np.float32) * (1 - al) + col * al,
                               0, 255).astype(np.uint8)

        # PER-POLYGON crops, not one union box over all of them. Cropping at all is the ~230 ms ->
        # ~10 ms optimisation this function was built around, but a UNION box only stays small
        # while every body is small: the road set's tunnel dividers are full-lattice-width
        # rectangles, so their union spans the entire frame and hands the saving straight back
        # (measured 1.87 s/frame against the ~1.54 the solve alone needs). Each divider's OWN box
        # is 1080x12. Fills still run before rims, exactly as one shared pair of buffers did, so
        # the composite order is unchanged.
        flat = [(k, q) for k, q in enumerate(scr) if tex[k] is None]
        boxes = {k: crop(q) for k, q in flat}
        for k, q in flat:
            if boxes[k] is not None:
                stroke(q, boxes[k], False)
        if rim:
            for k, q in flat:
                if boxes[k] is not None:
                    stroke(q, boxes[k], True)

        for k, t in enumerate(tex):
            if t is not None:
                img, to_uv = t if isinstance(t, tuple) else (t, None)
                self._draw_texture(rgb, self._polys[k], scr[k], img, to_uv)
        return rgb

    def _textures(self, when=0.0):
        """Per-body textures, or None for the usual flat silhouette.

        A scene opts in with `body_textures()` returning one entry per polygon: an RGB array, or
        `(rgb, to_uv)` where `to_uv(lattice_x, lattice_y) -> (u, v)` in [0,1]. The second form
        exists for a body that DEFORMS - a bounding box is a fine texture map for a rigid shape
        and a wrong one for a bending neck, because the box itself moves. A scene that deforms
        knows its own rig and placement, so it is the right place to own that map; the renderer
        stops guessing.

        Nothing else in the registry pays for any of it: a scene without `body_textures` never
        allocates or samples anything.

        A texture map may also depend on TIME - the road set's Jeep spins its wheels, which is a
        rotating sample of a parked photograph and nothing else. So `body_textures` is called
        with the frame time if it declares a parameter for it, and without if it does not: every
        scene written before this took no argument and still takes none, rather than all of them
        growing an unused `t`.
        """
        fn = getattr(self.scene, "body_textures", None)
        if fn is None:
            return [None] * len(self._polys)
        wants_t = getattr(fn, "__code__", None) is not None and fn.__code__.co_argcount > 1
        t = list(fn(when) if wants_t else fn())
        return (t + [None] * len(self._polys))[:len(self._polys)]

    def _draw_texture(self, rgb, poly, scr, tex, to_uv=None):
        """Composite an image onto one body, sampled through the SAME lattice->screen map.

        Not "resize the image into the body's screen box": that would only be right for
        `flow="right"`, and would silently transpose the picture in the vertical formats. Going
        back through `self._coords` - the grid `frame()` already uses to warp the field - means
        the texture follows whatever orientation the format has, including `down`'s mirrored
        cross-stream axis, for free.

        The alpha is the rasterised POLYGON, not the source cut-out's own alpha, so what is drawn
        is exactly the body the solver was given (see `shapes.image_body` for why those can
        differ at the edges).
        """
        s = 2
        pad = 2
        x0 = max(0, int(np.floor(scr[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(scr[:, 1].min())) - pad)
        x1 = min(self.W, int(np.ceil(scr[:, 0].max())) + pad)
        y1 = min(self.H, int(np.ceil(scr[:, 1].max())) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        w, h = x1 - x0, y1 - y0
        m = Image.new("L", (w * s, h * s), 0)
        ImageDraw.Draw(m).polygon([((float(x) - x0) * s, (float(y) - y0) * s) for x, y in scr],
                                  fill=255)
        a = np.asarray(m.resize((w, h), Image.BOX), np.float32)[:, :, None] / 255.0
        if self._coords_np is None:
            self._coords_np = asnumpy(self._coords)
        jj = self._coords_np[0, y0:y1, x0:x1]            # lattice y at each screen pixel
        ii = self._coords_np[1, y0:y1, x0:x1]            # lattice x
        Ht, Wt = tex.shape[:2]
        if to_uv is None:
            lx0, lx1 = poly[:, 0].min(), poly[:, 0].max()
            ly0, ly1 = poly[:, 1].min(), poly[:, 1].max()
            u = (ii - lx0) / max(lx1 - lx0, 1e-6)
            v = (jj - ly0) / max(ly1 - ly0, 1e-6)
            maps = [None]
        else:
            # `to_uv` may be a LIST of maps, which are averaged - i.e. MOTION BLUR over the
            # frame's exposure. Added 2026-08-11 for the road set's wheels at real-time speed.
            #
            # WHY IT IS NEEDED, and it is physics rather than polish: a wheel drawn sharp is only
            # right while it turns less than half its own symmetry period per frame. An F1 wheel
            # at 350 km/h turns **0.73 of a revolution per frame** at 60 fps - measured 263
            # deg/frame against a 5-spoke wheel's 72 deg period - so a sharp draw does not merely
            # look fast, it ALIASES: the wagon-wheel effect, where the wheel appears to crawl or
            # run backwards. Slowing the wheel to beat the limit would be a lie about the speed,
            # and leaving it sharp is a lie about what a camera sees. A real shutter integrates
            # over the exposure, so the honest picture of a wheel that fast IS a rotational smear,
            # and averaging N samples across the frame's own angular sweep is exactly that
            # integral. The Jeep at 129 km/h needs it too (75 deg/frame against 72).
            maps = list(to_uv) if isinstance(to_uv, (list, tuple)) else [to_uv]
        sub = (slice(y0, y1), slice(x0, x1))
        base = rgb[sub].astype(np.float32)
        acc = np.zeros_like(base)
        for m in maps:
            if m is None:
                uu, vv = u, v
            else:
                uu, vv = m(ii, jj)
            tx = np.clip(uu * (Wt - 1), 0, Wt - 1).astype(np.int32)
            ty = np.clip(vv * (Ht - 1), 0, Ht - 1).astype(np.int32)
            acc += tex[ty, tx].astype(np.float32)
        acc /= float(len(maps))
        rgb[sub] = np.clip(base * (1 - a) + acc * a, 0, 255).astype(np.uint8)

    def _draw_bodies_fullframe(self, rgb):
        """Reference implementation kept for comparison — see _draw_bodies for why it is unused."""
        cfg = self.cfg
        s = 2
        fill = Image.new("L", (self.W * s, self.H * s), 0)
        edge = Image.new("L", (self.W * s, self.H * s), 0)
        df, de = ImageDraw.Draw(fill), ImageDraw.Draw(edge)
        for p in self._polys:
            q = [(float(x) * s, float(y) * s) for x, y in self.poly_to_screen(p)]
            df.polygon(q, fill=255)
            de.line(q + [q[0]], fill=255, width=max(1, cfg.body_edge_px * s), joint="curve")
        af = np.asarray(fill.resize((self.W, self.H), Image.BOX), np.float32)[:, :, None] / 255.0
        ae = np.asarray(edge.resize((self.W, self.H), Image.BOX), np.float32)[:, :, None] / 255.0
        base = rgb.astype(np.float32)
        base = base * (1 - af) + np.asarray(cfg.body_fill, np.float32) * af
        base = base * (1 - ae) + np.asarray(cfg.body_edge, np.float32) * ae
        return np.clip(base, 0, 255).astype(np.uint8)

    # -- free-body diagram ----------------------------------------------------------------
    def _draw_arrows(self, rgb, t):
        """Labelled vector arrows from `scene.arrows(t)`, given in LATTICE coordinates.

        The scene hands over `(x0, y0, x1, y1, colour, label)` per arrow and this maps BOTH
        endpoints through `to_screen`. Mapping the endpoints rather than the tail plus a rotated
        direction is what keeps it correct in all three formats for free: the lattice->screen map
        is affine, so a straight lattice segment is a straight screen segment, and `flow="down"`'s
        mirrored cross-stream axis needs no special case.

        The scene owns the FORCE-TO-LENGTH scale, and must use ONE scale for every force arrow it
        returns - otherwise the arrows are three unrelated pictures and their relative lengths,
        which is the entire content of a free-body diagram, mean nothing.
        """
        fn = getattr(self.scene, "arrows", None)
        if fn is None:
            return rgb
        items = fn(t)
        if not items:
            return rgb
        s = self.H / 1920.0
        if self._font is None:
            self._font = _load_font(max(16, self.H // 68))
        font = _load_font(max(12, int(round(26 * s))))
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img, "RGBA")
        head = max(6.0, 22.0 * s)                 # arrowhead length in px
        for it in items:
            x0, y0, x1, y1 = it[0], it[1], it[2], it[3]
            col = tuple(int(c) for c in it[4])
            label = it[5] if len(it) > 5 else ""
            wid = int(it[6]) if len(it) > 6 else max(2, int(round(5 * s)))
            (sx0, sy0) = self.to_screen(np.asarray([x0]), np.asarray([y0]))
            (sx1, sy1) = self.to_screen(np.asarray([x1]), np.asarray([y1]))
            ax, ay = float(asnumpy(sx0)[0]), float(asnumpy(sy0)[0])
            bx, by = float(asnumpy(sx1)[0]), float(asnumpy(sy1)[0])
            L = float(np.hypot(bx - ax, by - ay))
            if L < 2.0:
                continue                          # a zero-length arrow has no direction to draw
            ux, uy = (bx - ax) / L, (by - ay) / L
            hl = min(head, 0.45 * L)              # never let the head swallow a short arrow
            # shaft stops where the head begins, so the two do not overdraw and thicken the tip
            d.line([ax, ay, bx - ux * hl, by - uy * hl], fill=col + (255,), width=wid)
            px_, py_ = -uy, ux
            d.polygon([(bx, by),
                       (bx - ux * hl + px_ * hl * 0.42, by - uy * hl + py_ * hl * 0.42),
                       (bx - ux * hl - px_ * hl * 0.42, by - uy * hl - py_ * hl * 0.42)],
                      fill=col + (255,))
            if label:
                # Offset the label CLEAR of the head, not merely away from the shaft. The plate is
                # centred on its anchor, so the perpendicular offset has to carry half the plate's
                # own extent as well - without that term the box sits on the arrowhead for any
                # arrow pointing up-and-right, which is exactly what the synthetic check showed.
                b = d.textbbox((0, 0), label, font=font)
                tw_, th_ = b[2] - b[0], b[3] - b[1]
                off = hl * 0.8 + 0.5 * float(np.hypot(px_ * tw_, py_ * th_))
                tx = bx + ux * hl * 0.7 + px_ * off
                ty = by + uy * hl * 0.7 + py_ * off
                tx = float(np.clip(tx - 0.5 * tw_, 4, self.W - tw_ - 4))
                ty = float(np.clip(ty - 0.5 * th_, 4, self.H - th_ - 4))
                # a dark plate under the text: these arrows sit on a bright colour field and
                # unbacked text is illegible over the pale end of any of the palettes
                d.rounded_rectangle([tx - 6 * s, ty - 4 * s,
                                     tx + tw_ + 6 * s, ty + th_ + 8 * s],
                                    radius=4 * s, fill=(6, 8, 12, 165))
                d.text((tx, ty), label, font=font, fill=col + (255,))
        # np.array, not np.asarray: a PIL image exports a READ-ONLY buffer, and unlike `_draw_hud`
        # this overlay is not the last one - `_draw_legend` writes into the frame after it and
        # fails on a read-only destination. Caught by the smoke run, which is what it is for.
        return np.array(img)

    # -- legend --------------------------------------------------------------------------
    def _build_legend(self):
        """Compose the colour-bar key ONCE into an RGBA patch, cached for the whole render.

        The bar is the shipped LUT read back through the SAME tone curve the frame gets
        (`s ** gamma` before the lookup), so a colour in the key is the colour the field takes at
        that speed - not an idealised ramp that happens to use the same palette. Ticks are placed
        linearly in SPEED, which is why the gamma has to be undone here rather than ignored: with
        gamma != 1 an evenly-spaced set of colours is NOT an evenly-spaced set of speeds.
        """
        cfg = self.cfg
        spec = self.scene.legend() if hasattr(self.scene, "legend") else None
        if not spec:
            return None
        s = self.H / 1920.0
        # Sized to stay OUT OF THE PICTURE: the key is a caption, not a panel. On tri_foil_rates
        # the top lane's leading edge sits ~50 px to the right of this at 1080x1920, which is the
        # constraint these numbers are set against - grow any of them and the legend starts
        # covering the one foil the scene most wants you to look at.
        fs = max(10, int(round(22 * s)))
        fst = max(11, int(round(27 * s)))
        font, ftitle = _load_font(fs), _load_font(fst)
        pad = max(5, int(round(15 * s)))
        gap = max(3, int(round(10 * s)))
        barw = max(7, int(round(26 * s)))
        barh = max(50, int(round(340 * s)))
        tick = max(3, int(round(8 * s)))

        title = spec.get("title", "SPEED")
        marks = list(spec.get("marks", []))
        note = spec.get("note")

        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

        def tw(txt, f):
            b = probe.textbbox((0, 0), txt, font=f)
            return b[2] - b[0], b[3] - b[1]

        lw = max([tw(m[1], font)[0] for m in marks] or [0])
        th = tw(title, ftitle)[1]
        nw, nh = tw(note, font) if note else (0, 0)
        inner_w = max(tw(title, ftitle)[0], barw + tick + gap + lw, nw)
        inner_h = th + gap + barh + (gap + nh if note else 0)
        pw, ph = inner_w + 2 * pad, inner_h + 2 * pad

        img = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=max(4, int(round(10 * s))),
                            fill=(6, 8, 12, 150), outline=(210, 220, 226, 70), width=1)
        d.text((pad, pad), title, font=ftitle, fill=(232, 238, 236, 255))

        # the ramp itself: top of the bar = full scale (u0 * vmax), bottom = zero
        bx, by = pad, pad + th + gap
        lutn = asnumpy(colormap.lut(cfg.cmap, cfg.saturation, cfg.brightness))
        lin = np.linspace(1.0, 0.0, barh, dtype=np.float32)
        # The tone curve has to be undone for EVERY field that applies one, not just "speed".
        # `_field01` raises speed, mach and stag to cfg.gamma; listing only speed here drew a
        # legend whose colours were not the frame's for the other two - latent until a mach
        # scene turned the key on (mach_sweep, 2026-08-04).
        toned = cfg.field in ("speed", "mach", "stag") and abs(cfg.gamma - 1.0) > 1e-3
        shaped = lin ** cfg.gamma if toned else lin
        col = lutn[np.clip(shaped * 255.0, 0, 255).astype(np.int32)]
        bar = np.repeat(col[:, None, :], barw, axis=1).astype(np.uint8)
        img.paste(Image.fromarray(bar).convert("RGBA"), (bx, by))
        d.rectangle([bx, by, bx + barw - 1, by + barh - 1], outline=(226, 232, 230, 120), width=1)

        for frac, label in marks:
            y = by + int(round((1.0 - float(np.clip(frac, 0.0, 1.0))) * (barh - 1)))
            d.line([bx, y, bx + barw + tick, y], fill=(238, 242, 240, 235), width=max(1, int(2 * s)))
            # labels centre on their tick, but a mark AT either end of the ramp would otherwise
            # push its text out of the bar's span - into the title above, or the note below
            ty = min(max(y - 0.62 * fs, by), by + barh - fs)
            d.text((bx + barw + tick + gap, ty), label, font=font, fill=(232, 238, 236, 255))
        if note:
            d.text((pad, by + barh + gap), note, font=font, fill=(186, 198, 200, 235))

        a = np.asarray(img, np.float32)
        x0 = int(round(0.035 * self.W))
        y0 = int(round(0.026 * self.H))
        x0 = min(x0, max(0, self.W - pw))
        y0 = min(y0, max(0, self.H - ph))
        return x0, y0, a[:, :, :3], a[:, :, 3:4] / 255.0

    def _draw_legend(self, rgb):
        if self._legend is False:
            self._legend = self._build_legend()
        if not self._legend:
            return rgb
        x0, y0, src, alpha = self._legend
        h, w = src.shape[:2]
        if y0 + h > self.H or x0 + w > self.W:
            return rgb
        sub = (slice(y0, y0 + h), slice(x0, x0 + w))
        base = rgb[sub].astype(np.float32)
        rgb[sub] = np.clip(base * (1.0 - alpha) + src * alpha, 0, 255).astype(np.uint8)
        return rgb

    # -- instrument axes ------------------------------------------------------------------
    def _draw_axes(self, rgb, t):
        """The BULLET SERIES' instrument block: a scrolling ruler + a filling progress bar.

        A colour-bar legend answers "what does this colour mean". These answer a different and,
        for a projectile clip, more useful question - "where is it and how long has it been
        flying" - so the two are alternatives rather than companions, and `mach_bullet` runs
        `legend=False` / `axes=True`.

        THE RULER IS THE SPEED READING, and that is why it scrolls at a FIXED pixels-per-metre
        instead of rescaling to fit the distance travelled. A bar that stretched to always span
        0..distance would move at a nearly constant rate however hard the round decelerated - it
        would show the total and hide the derivative. Pinned to the ground, the ruler runs at
        the projectile's own speed: 9x19 opens at ~7 major ticks a second and ends at under
        three, which is the deceleration the clip is about, drawn without a number having to
        say it.

        Everything comes from `scene.axes(t)`; the renderer owns no ballistics. Spec:

            ruler = dict(label, value, span, major, minor, unit, marker, fmt)
            bar   = dict(label, value, total, tag, fmt)
            note  = str      (a caption line - where a scale distortion gets DECLARED)
            corner= str      (a live right-aligned readout on the ruler row)

        Sizes are quoted against a 1920-px frame and scaled, so a `--preview` at 540x960 lays
        out identically at half size rather than losing the block off the bottom.
        """
        fn = getattr(self.scene, "axes", None)
        spec = fn(t) if fn is not None else None
        if not spec:
            return rgb
        s = self.H / 1920.0
        f_lab = _load_font(max(9, int(round(26 * s))))
        f_num = _load_font(max(8, int(round(23 * s))))
        f_big = _load_font(max(10, int(round(34 * s))))
        f_note = _load_font(max(8, int(round(21 * s))))

        x0 = int(round(0.060 * self.W))
        x1 = int(round(0.940 * self.W))
        w = x1 - x0
        pad = int(round(26 * s))
        # `note` may be a string or a LIST of lines - the comparison clips need a second line for
        # the standings, and growing the block is cheaper than shrinking the type.
        notes = spec.get("note") or []
        if isinstance(notes, str):
            notes = [notes]
        # laid out from the BOTTOM up, so the block hugs the frame edge whatever it contains
        y_bot = self.H - int(round(40 * s))
        h_note = int(round(30 * s)) * len(notes)
        h_bar = int(round(126 * s))
        h_rul = int(round(162 * s))
        y_top = y_bot - h_note - h_bar - h_rul - pad

        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img, "RGBA")
        # The plate has to hold text over WHATEVER the field is doing behind it, and the field is
        # the whole frame. 190 rather than the legend panel's 150: that one sits top-left over a
        # single body's near-field, this one spans the full width at the bottom where the wake
        # goes, and it was measurably washed out against a mid-blue `blues` freestream.
        d.rounded_rectangle([x0 - pad, y_top - pad, x1 + pad, y_bot + int(round(10 * s))],
                            radius=int(round(14 * s)), fill=(5, 7, 11, 190),
                            outline=(210, 220, 226, 74), width=1)

        ink = (233, 239, 237, 255)
        dim = (150, 163, 170, 255)
        hot = (255, 214, 120, 255)

        def right(text, font, xr, y, fill):
            b = d.textbbox((0, 0), text, font=font)
            d.text((xr - (b[2] - b[0]), y), text, font=font, fill=fill)

        # --- the scrolling ruler ----------------------------------------------------------
        r = spec.get("ruler")
        y = y_top
        if r:
            d.text((x0, y), r.get("label", "DISTANCE"), font=f_lab, fill=dim)
            if spec.get("corner"):
                right(spec["corner"], f_big, x1, y - int(round(6 * s)), hot)
            y += int(round(40 * s))
            base = y + int(round(44 * s))                 # the ruler's own baseline; the gap
                                                          # carries the live reading above it
            span = max(float(r.get("span", 100.0)), 1e-6)
            ppu = w / span                                # pixels per metre - FIXED, see above
            mark = x0 + float(r.get("marker", 0.42)) * w
            val = float(r.get("value", 0.0))
            minor = float(r.get("minor", 5.0))
            major = float(r.get("major", 25.0))
            fmt = r.get("fmt", "{:.0f}")
            d.line([x0, base, x1, base], fill=(206, 216, 220, 210), width=max(1, int(2 * s)))
            # Ghost positions are needed BEFORE the ticks are labelled, because a ghost sits on
            # the same baseline as the tick numbers - it IS a distance, so that is where it
            # belongs - and a major tick that would collide with one gets dropped instead.
            # Putting them above the axis was tried and collides three ways (the corner readout,
            # the row header, and the live value when the round is passing one, which is exactly
            # the moment that matters). Below the axis there is no competition but the ticks, and
            # a tick is the thing worth losing.
            ghosts = []
            for gu, glab in (r.get("ghosts") or ()):
                gx = mark + (float(gu) - val) * ppu
                if x0 - 1 <= gx <= x1 + 1:
                    gb = d.textbbox((0, 0), glab, font=f_num)
                    ghosts.append((gx, glab, gb[2] - gb[0]))
            lo = val - (mark - x0) / ppu
            k0 = int(np.floor(lo / minor)) - 1
            k1 = int(np.ceil((lo + span) / minor)) + 1
            for k in range(k0, k1 + 1):
                u = k * minor
                if u < -1e-9:                             # behind the muzzle: nothing there yet
                    continue
                px = mark + (u - val) * ppu
                if px < x0 - 1 or px > x1 + 1:
                    continue
                is_major = abs(u / major - round(u / major)) < 1e-6
                hgt = int(round((26 if is_major else 12) * s))
                d.line([px, base, px, base + hgt],
                       fill=(226, 234, 232, 235) if is_major else (170, 182, 188, 190),
                       width=max(1, int((2 if is_major else 1) * s)))
                if is_major:
                    lab = fmt.format(u)
                    b = d.textbbox((0, 0), lab, font=f_num)
                    tw_ = b[2] - b[0]
                    tx = px - 0.5 * tw_
                    # drop a tick number a ghost is about to sit on - see the note above
                    if any(abs(px - gx) < 0.5 * (tw_ + gw) + 10 * s for gx, _, gw in ghosts):
                        continue
                    if x0 - 2 <= tx and tx + tw_ <= x1 + 2:
                        d.text((tx, base + hgt + int(round(6 * s))), lab, font=f_num, fill=ink)
            # GHOSTS: where a previous test finished, pinned to the same ground as the ticks.
            # Drawn before the marker so a live round overtaking one passes OVER it, which is the
            # comparison happening rather than being asserted. Off-window ghosts simply are not
            # drawn - the ruler is a window on the ground, not a summary of the whole range.
            # A ghost is a POSITION, so it is labelled with the round and nothing else - the
            # distances live in the standings line, where they can be read rather than caught in
            # passing. Carrying "45 ACP 571" on the marker collided three ways at once (with the
            # corner readout, with the row header, and with the other ghost 125 m away), and the
            # fix is to stop the marker doing the standings' job. The label also sits low, hugging
            # its own dashes, so the band above the axis belongs to the LIVE reading alone.
            for gx, glab, gw in ghosts:
                gh = int(round(26 * s))
                for yy in range(base, base + gh, max(2, int(round(8 * s)))):   # dashed, BELOW
                    d.line([gx, yy, gx, min(yy + int(round(4 * s)), base + gh)],
                           fill=(150, 214, 236, 235), width=max(1, int(round(2 * s))))
                d.text((float(np.clip(gx - 0.5 * gw, x0, x1 - gw)),
                        base + gh + int(round(6 * s))), glab, font=f_num,
                       fill=(150, 214, 236, 255))
            # the marker is the PROJECTILE's own station: everything left of it is behind it
            mh = int(round(20 * s))
            d.polygon([(mark, base), (mark - 0.5 * mh, base - mh), (mark + 0.5 * mh, base - mh)],
                      fill=hot)
            # The live reading gets the SAME weight as the timer's, because it is the same kind
            # of number: the ticks say where the ruler is, this says where the subject is.
            cur = (fmt.format(val) + " " + r.get("unit", "")).strip()
            b = d.textbbox((0, 0), cur, font=f_big)
            d.text((float(np.clip(mark - 0.5 * (b[2] - b[0]), x0, x1 - (b[2] - b[0]))),
                    base - mh - int(round(40 * s))), cur, font=f_big, fill=hot)
            y = y_top + h_rul

        # --- the solid time bar -----------------------------------------------------------
        bp = spec.get("bar")
        if bp:
            lab = bp.get("label", "TIME")
            d.text((x0, y), lab, font=f_lab, fill=dim)
            if bp.get("tag"):
                b = d.textbbox((0, 0), lab, font=f_lab)
                d.text((x0 + (b[2] - b[0]) + int(round(18 * s)), y), bp["tag"],
                       font=f_num, fill=(126, 190, 220, 255))
            fmt = bp.get("fmt", "{:.2f}")
            total = max(float(bp.get("total", 1.0)), 1e-9)
            val = float(np.clip(bp.get("value", 0.0), 0.0, total))
            right(fmt.format(val), f_big, x1, y - int(round(6 * s)), ink)
            y += int(round(44 * s))
            bh = int(round(30 * s))
            d.rectangle([x0, y, x1, y + bh], fill=(24, 30, 38, 220),
                        outline=(150, 163, 170, 150), width=1)
            fill_to = x0 + int(round((val / total) * w))
            if fill_to > x0:
                d.rectangle([x0, y, fill_to, y + bh], fill=(122, 196, 226, 255))
            y += bh + int(round(8 * s))
            d.text((x0, y), "0", font=f_num, fill=dim)
            right(fmt.format(total), f_num, x1, y, dim)

        for k, line in enumerate(notes):
            d.text((x0, y_bot - int(round(24 * s)) - (len(notes) - 1 - k) * int(round(30 * s))),
                   line, font=f_note,
                   fill=(163, 176, 182, 255) if k == 0 else (126, 190, 220, 255))
        # np.array, not np.asarray: PIL exports a READ-ONLY buffer and `_draw_hud` may still
        # write after this one (the same trap `_draw_arrows` fell into).
        return np.array(img)

    def _draw_panel(self, rgb, t):
        """The ROAD SET's caption panel: a solid bottom BAND plus a corner TAG.

        Deliberately not `_draw_legend` or `_draw_axes` re-dressed. Those two answer questions
        about the physics - what does this colour mean, where is the projectile - and both float
        over the field. This one is a CAPTION for a social post, and the difference that actually
        matters in the code is that it OWNS a band of the frame rather than floating: the scene
        reads the same fraction back (`_RoadSet.LEGEND_FRAC`) and lays its tunnels out in what is
        left, so nothing the solver draws is ever hidden underneath it. A floating overlay cannot
        offer that, which is why this is a third element and not a fourth argument to one of them.

        Everything comes from `scene.panel(t)`; the renderer owns no schedule. Spec:

            band  = float           fraction of the frame HEIGHT the panel occupies, bottom-up
            title = str             left of the band
            label = str             small dim heading over the reading
            value = str             the reading itself, big
            unit  = str             drawn beside `value` at label size
            frac  = float 0..1      optional progress bar under the reading
            tscale= float           optional multiplier on every TYPE size in the band. Default
                                    1.0, so no clip written before it moves by a pixel. It exists
                                    because the sizes below were chosen against the road set's
                                    20% band, and `band` is a scene's own decision - `air_frog`
                                    asks for a THIRD of the frame, where the same type floats in
                                    the middle of 640 px of ink. Deliberately NOT derived from
                                    `band`: a bigger band is not always a request for bigger type
                                    (a scene might want the room for a second row), so the two
                                    stay separate decisions.
            tag   = dict(text, at=(fx, fy))    a boxed handle placed anywhere on the frame

        THE READING IS RIGHT-ALIGNED ON PURPOSE. My note, 2026-08-11: it needed to sit in the
        right half of the bottom legend so short-form UI chrome would not cover it - a vertical clip's
        caption furniture sits bottom-LEFT, so anything that must stay readable goes to the other
        side. The title takes the left because it is the one line a viewer can lose without
        losing the clip.

        Sizes are quoted against a 1920-px frame and scaled, so a `--preview` lays out identically
        at half size rather than losing the band off the bottom.
        """
        fn = getattr(self.scene, "panel", None)
        spec = fn(t) if fn is not None else None
        if not spec:
            return rgb
        s = self.H / 1920.0
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img, "RGBA")

        # --- the band ---------------------------------------------------------------------
        band = float(spec.get("band", 0.20))
        if band > 0:
            y0 = int(round((1.0 - band) * self.H))
            # The same ink as the scene's divider bars, so the band and the bars read as ONE
            # graphic system rather than as a caption sitting on top of a picture that has
            # black lines in it. Opaque, not 190/255 like the axes plate: this band covers
            # lattice the scene deliberately composed OUT of frame, so there is nothing behind
            # it worth letting through.
            d.rectangle([0, y0, self.W, self.H], fill=(6, 8, 12, 255))
            d.line([0, y0, self.W, y0], fill=(210, 220, 226, 60), width=max(1, int(round(2 * s))))

            ts = s * float(spec.get("tscale", 1.0))
            f_title = _load_font(max(12, int(round(54 * ts))))
            f_lab = _load_font(max(8, int(round(24 * ts))))
            f_val = _load_font(max(14, int(round(88 * ts))))
            f_unit = _load_font(max(9, int(round(34 * ts))))
            ink = (236, 241, 239, 255)
            dim = (146, 159, 166, 255)
            hot = (255, 214, 120, 255)

            xl = int(round(0.055 * self.W))
            xr = int(round(0.945 * self.W))
            mid = y0 + (self.H - y0) // 2

            title = spec.get("title") or ""
            if title:
                b = d.textbbox((0, 0), title, font=f_title)
                d.text((xl, mid - (b[3] - b[1]) // 2 - b[1]), title, font=f_title, fill=ink)

            # --- the reading, laid out from the RIGHT edge inward
            val = str(spec.get("value", ""))
            unit = spec.get("unit") or ""
            lab = spec.get("label") or ""
            bv = d.textbbox((0, 0), val, font=f_val)
            bu = d.textbbox((0, 0), unit, font=f_unit) if unit else (0, 0, 0, 0)
            gap = int(round(12 * ts))
            wv = (bv[2] - bv[0]) + (gap + (bu[2] - bu[0]) if unit else 0)
            y_val = mid - (bv[3] - bv[1]) // 2 - bv[1]
            x_val = xr - wv
            d.text((x_val, y_val), val, font=f_val, fill=hot)
            if unit:
                d.text((x_val + (bv[2] - bv[0]) + gap,
                        y_val + (bv[3] - bv[1]) - (bu[3] - bu[1]) - int(round(6 * ts))),
                       unit, font=f_unit, fill=dim)
            if lab:
                bl = d.textbbox((0, 0), lab, font=f_lab)
                d.text((xr - (bl[2] - bl[0]), y_val - (bl[3] - bl[1]) - int(round(20 * ts))),
                       lab, font=f_lab, fill=dim)

            frac = spec.get("frac")
            if frac is not None:
                bw = int(round(0.30 * self.W))
                bh = max(2, int(round(7 * ts)))
                bx1, by1 = xr, y_val + (bv[3] - bv[1]) + int(round(26 * ts))
                d.rectangle([bx1 - bw, by1, bx1, by1 + bh], fill=(255, 255, 255, 40))
                d.rectangle([bx1 - bw, by1, bx1 - bw + int(round(bw * max(0.0, min(1.0, frac)))),
                             by1 + bh], fill=hot)

        # --- the tag ----------------------------------------------------------------------
        tag = spec.get("tag")
        if tag and tag.get("text"):
            f_tag = _load_font(max(9, int(round(30 * s))))
            tx = int(round(float(tag.get("at", (0.045, 0.030))[0]) * self.W))
            ty = int(round(float(tag.get("at", (0.045, 0.030))[1]) * self.H))
            b = d.textbbox((0, 0), tag["text"], font=f_tag)
            padx, pady = int(round(16 * s)), int(round(10 * s))
            # A background, because the field behind it is the picture and a bare handle over a
            # mid-blue freestream is unreadable at phone size (the same finding that took the
            # axes plate from 150 to 190 alpha).
            d.rounded_rectangle([tx, ty, tx + (b[2] - b[0]) + 2 * padx,
                                 ty + (b[3] - b[1]) + 2 * pady],
                                radius=int(round(10 * s)), fill=(6, 8, 12, 205),
                                outline=(210, 220, 226, 70), width=1)
            d.text((tx + padx - b[0], ty + pady - b[1]), tag["text"], font=f_tag,
                   fill=(236, 241, 239, 255))

        return np.asarray(img)

    def _draw_hud(self, rgb, t):
        if self._font is None:
            self._font = _load_font(max(16, self.H // 68))
        img = Image.fromarray(rgb).convert("RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        lines = ([self.scene.title] if self.scene.title else []) + \
                [f"{k}  {v}" for k, v in self.scene.readout(t)]
        step = int(self._font.size * 1.45)
        x = int(self.W * 0.045)
        # A scene that turns its LEGEND on cannot also put the readout in the top-left corner:
        # `_build_legend` anchors at (0.035W, 0.026H) and this anchors at (0.045W, 0.028H), so
        # the two draw straight over each other. Opting into the opposite corner is per-scene and
        # the default is exactly what it was, so no existing clip moves by a pixel.
        if getattr(self.scene, "hud_anchor", "top") == "bottom":
            y = self.H - int(self.H * 0.05) - step * len(lines)
            pad = int(self.H * 0.012)
            wmax = max(d.textbbox((0, 0), s, font=self._font)[2] for s in lines) if lines else 0
            d.rounded_rectangle([x - pad, y - pad, x + wmax + pad, y + step * len(lines) + pad],
                                radius=max(4, pad // 2), fill=(6, 8, 12, 150),
                                outline=(210, 220, 226, 70), width=1)
        else:
            y = int(self.H * 0.028)
        for line in lines:
            d.text((x, y), line, font=self._font, fill=(226, 234, 230))
            y += step
        return np.asarray(img.convert("RGB"))


def _load_font(px):
    from PIL import ImageFont
    for p in (os.path.join(os.path.dirname(__file__), "..", "fonts", "CascadiaMono.ttf"),
              r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(os.path.normpath(p), px)
        except Exception:
            continue
    return ImageFont.load_default()


# --- drivers ----------------------------------------------------------------------------
def render_stills(scene, cfg: RenderConfig, times, out_png):
    """Dump a PNG at each time in `times` from ONE simulation run — the tuning loop. Re-simulating
    from t=0 for every sample would dominate iteration time; the flow is causal, so we just tap it
    as it goes past. `out_png` gets `_<t>s` appended when more than one time is asked for."""
    times = sorted(float(t) for t in times)
    tun = Tunnel(scene, cfg)
    tun.settle(cfg.settle)
    stem, ext = os.path.splitext(out_png)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    written = []
    n = max(1, int(round(times[-1] * cfg.fps)))
    nxt = 0
    for i in range(n + 1):
        t = i / cfg.fps
        tun.advance(t)
        while nxt < len(times) and t >= times[nxt]:
            p = out_png if len(times) == 1 else f"{stem}_{times[nxt]:g}s{ext}"
            Image.fromarray(tun.frame(t)).save(p)
            # `tau` is an LBM relaxation time and does not exist on the compressible solver;
            # printing it unconditionally made --still crash on any cns scene, which is why
            # tri_foil_gas never got a look at a frame. The CNS analogue is the achieved CFL.
            extra = (f"tau={tun.lbm.tau:.4f}" if hasattr(tun.lbm, "tau")
                     else f"CFL={tun.lbm.cfl_now():.3f}")
            print(f"still: {p}  (t={t:.2f}s, max|u|={tun.lbm.health():.4f}, {extra})",
                  flush=True)
            written.append(p)
            nxt += 1
    return written


def render_still(scene, cfg: RenderConfig, t, out_png):
    return render_stills(scene, cfg, [t], out_png)[0]


def render_scene(scene, out_path, cfg: RenderConfig, seconds=None, progress=True):
    """Full render. Atomic: encodes to <out>.part.mp4 and renames only on success, so a file
    bearing its final name is ALWAYS finalized/playable (the oscilloscope engine's convention)."""
    seconds = float(seconds if seconds is not None else scene.duration)
    n_frames = max(1, int(round(seconds * cfg.fps)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or OUT, exist_ok=True)
    tmp = out_path + ".part.mp4"

    tun = Tunnel(scene, cfg)
    look = None
    if cfg.film:
        from .film import make_look
        look = make_look(cfg.width, cfg.height, cfg.fps, n_frames,
                         glow=cfg.glow, ambient=cfg.ambient, bow=cfg.bow, seed=cfg.seed,
                         exposure=cfg.exposure)

    print(f"  lattice {tun.nx}x{tun.ny} cells, {tun.lbm.describe()}, "
          f"{tun.steps_pf:.2f} steps/frame, backend={'GPU' if GPU else 'CPU'}")
    tun.settle(cfg.settle, progress=progress)

    proc = subprocess.Popen(
        [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
         "-an", *encoder_args(cfg.extras.get("quality", "normal")), "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)
    try:
        for i in range(n_frames):
            t = i / cfg.fps
            tun.advance(t)
            img = tun.frame(t)
            if look is not None:
                img = look.process(img, i)
            proc.stdin.write(img.tobytes())
            # Abort on divergence rather than encoding garbage. The threshold belongs to the
            # SOLVER, not to this loop: for the LBM it is 0.45, safely under the lattice sound
            # speed of 1/sqrt(3); for the CNS a velocity of 1 IS Mach 1, and a local supersonic
            # pocket is the physics it was written to capture, so the same number there would
            # abort a correct transonic render.
            if i % 10 == 0:
                h = tun.lbm.health()
                if not np.isfinite(h) or h > tun.lbm.health_limit:
                    raise RuntimeError(
                        f"solver diverged at frame {i} (t={t:.2f}s, max|u|={h:.4f} against a "
                        f"limit of {tun.lbm.health_limit}). "
                        f"Lower --u0 / --re, raise --csm, or reduce body sizes.")
            if progress and i % cfg.fps == 0:
                print(f"  frame {i}/{n_frames}  (t={t:.1f}s, max|u|={tun.lbm.health():.4f})",
                      flush=True)
        proc.stdin.close()
        proc.wait()
    except BaseException:                     # incl. KeyboardInterrupt: never leave a .part
        try:
            proc.kill(); proc.wait()
        except Exception:
            pass
        _rm(tmp)
        raise
    if proc.returncode != 0:
        _rm(tmp)
        raise RuntimeError(f"ffmpeg pipe failed (exit {proc.returncode})")
    os.replace(tmp, out_path)
    print("done:", out_path)
    return out_path


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass



