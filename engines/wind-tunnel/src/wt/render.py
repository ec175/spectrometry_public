"""render.py ââ‚¬â€ the pipeline: solve -> colourise -> streaks -> body -> film -> ffmpeg.

manim-free, same shape as Oscilloscope's render.py: this module owns the clock and the rawvideo
pipe, and nothing else knows about video.

Per output frame
----------------
    scene.bodies(t)            closed polygons in lattice coords
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
final show the *same flow at the same moment* instead of one running at double speed ââ‚¬â€ the same
"frames driven by real seconds" rule the Oscilloscope engine uses.
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

        Each of the three is orientation-PRESERVING (the 2x2 map has det > 0). That is not
        cosmetic: a mirrored mapping would flip the sign of every vortex on screen, so
        `--field vort` would lie and a cambered section would appear to lift the wrong way. For
        "down" that is why +y goes LEFT rather than right - taking the obvious "+x down, +y
        right" would have been a reflection.
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
        return (px - self.x0) / self.fx_c, (py - self.y0) / self.fx_r

    def poly_to_screen(self, p):
        sx, sy = self.to_screen(p[:, 0], p[:, 1])
        return np.stack([asnumpy(sx), asnumpy(sy)], 1)

    # -- simulation ---------------------------------------------------------------------
    def set_bodies(self, t):
        polys = self.scene.bodies(t)
        key = tuple(np.round(p, 2).tobytes() for p in polys)
        if key != self._last_poly_key:                        # skip the raster when nothing moved
            mask = shapes.rasterize(polys, self.nx, self.ny)
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

    def frame(self, t):
        """Compose one RGB uint8 (H, W, 3) screen frame from the CURRENT solver state."""
        cfg = self.cfg
        big = map_coordinates(self._field01(), self._coords, order=1, mode="nearest")
        rgb = colormap.apply(big, cfg.cmap, cfg.saturation, cfg.brightness).astype(xp.float32)
        if self.dye is not None:
            rgb = self._blend_dye(rgb, big)

        st = self.streaks.draw(self.lbm.ux, self.lbm.uy, self.to_screen, self.W, self.H,
                               sigma=cfg.streak_sigma, speed_ref=cfg.u0,
                               speed_gain=cfg.streak_speed_gain)
        rgb = rgb + (xp.clip(st * cfg.streak_gain, 0.0, 1.6) * 255.0)[:, :, None]
        if cfg.extras.get("bloom", 0.0) > 0.0:
            rgb = self._add_bloom(rgb)
        out = asnumpy(xp.clip(rgb, 0, 255).astype(xp.uint8))

        out = self._draw_bodies(out)
        if cfg.fbd:
            out = self._draw_arrows(out, t)
        if cfg.legend:
            out = self._draw_legend(out)
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

    def _draw_bodies(self, rgb):
        """Filled body + bright outline, rasterised at 2x and box-downsampled for a clean edge.

        Only the BOUNDING BOX of the bodies is rasterised, not the whole frame. Bodies cover a
        few percent of a 1080x1920 picture, but allocating two full-frame 2x images (2160x3840
        'L' each) and box-resizing them every frame cost ~230 ms — 41% of total frame time, more
        than the entire fluid solve. Cropping to the bodies makes it ~10 ms for pixel-identical
        output. (Measured with scratch/profile_render.py; see CLAUDE.md "Render performance".)
        """
        cfg = self.cfg
        if not self._polys:
            return rgb
        s = 2
        scr = [self.poly_to_screen(p) for p in self._polys]
        tex = self._textures()
        pad = cfg.body_edge_px + 3
        x0 = max(0, int(np.floor(min(q[:, 0].min() for q in scr))) - pad)
        y0 = max(0, int(np.floor(min(q[:, 1].min() for q in scr))) - pad)
        x1 = min(self.W, int(np.ceil(max(q[:, 0].max() for q in scr))) + pad)
        y1 = min(self.H, int(np.ceil(max(q[:, 1].max() for q in scr))) + pad)
        if x1 <= x0 or y1 <= y0:
            return rgb                                  # every body is off-screen this frame
        w, h = x1 - x0, y1 - y0

        fill = Image.new("L", (w * s, h * s), 0)
        edge = Image.new("L", (w * s, h * s), 0)
        df, de = ImageDraw.Draw(fill), ImageDraw.Draw(edge)
        # body_edge_px = 0 means NO outline at all - the body is a hole punched in the field, not
        # an object drawn on top of it. `black_holes` needs that; everything else wants the rim.
        rim = cfg.body_edge_px > 0
        for k, q in enumerate(scr):
            if tex[k] is not None:
                continue                                # textured bodies are composited below
            pts = [((float(x) - x0) * s, (float(y) - y0) * s) for x, y in q]
            df.polygon(pts, fill=255)
            if rim:
                de.line(pts + [pts[0]], fill=255, width=cfg.body_edge_px * s, joint="curve")
        af = np.asarray(fill.resize((w, h), Image.BOX), np.float32)[:, :, None] / 255.0
        sub = (slice(y0, y1), slice(x0, x1))
        base = rgb[sub].astype(np.float32)
        base = base * (1 - af) + np.asarray(cfg.body_fill, np.float32) * af
        if rim:
            ae = np.asarray(edge.resize((w, h), Image.BOX), np.float32)[:, :, None] / 255.0
            base = base * (1 - ae) + np.asarray(cfg.body_edge, np.float32) * ae
        rgb[sub] = np.clip(base, 0, 255).astype(np.uint8)

        for k, t in enumerate(tex):
            if t is not None:
                img, to_uv = t if isinstance(t, tuple) else (t, None)
                self._draw_texture(rgb, self._polys[k], scr[k], img, to_uv)
        return rgb

    def _textures(self):
        """Per-body textures, or None for the usual flat silhouette.

        A scene opts in with `body_textures()` returning one entry per polygon: an RGB array, or
        `(rgb, to_uv)` where `to_uv(lattice_x, lattice_y) -> (u, v)` in [0,1]. The second form
        exists for a body that DEFORMS - a bounding box is a fine texture map for a rigid shape
        and a wrong one for a bending neck, because the box itself moves. A scene that deforms
        knows its own rig and placement, so it is the right place to own that map; the renderer
        stops guessing.

        Nothing else in the registry pays for any of it: a scene without `body_textures` never
        allocates or samples anything.
        """
        fn = getattr(self.scene, "body_textures", None)
        t = list(fn()) if fn is not None else []
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
        else:
            u, v = to_uv(ii, jj)
        tx = np.clip(u * (Wt - 1), 0, Wt - 1).astype(np.int32)
        ty = np.clip(v * (Ht - 1), 0, Ht - 1).astype(np.int32)
        sub = (slice(y0, y1), slice(x0, x1))
        base = rgb[sub].astype(np.float32)
        rgb[sub] = np.clip(base * (1 - a) + tex[ty, tx].astype(np.float32) * a,
                           0, 255).astype(np.uint8)

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

    def _draw_hud(self, rgb, t):
        if self._font is None:
            self._font = _load_font(max(16, self.H // 68))
        img = Image.fromarray(rgb)
        d = ImageDraw.Draw(img)
        y = int(self.H * 0.028)
        lines = ([self.scene.title] if self.scene.title else []) + \
                [f"{k}  {v}" for k, v in self.scene.readout(t)]
        for line in lines:
            d.text((int(self.W * 0.045), y), line, font=self._font, fill=(226, 234, 230))
            y += int(self._font.size * 1.45)
        return np.asarray(img)


def _load_font(px):
    from PIL import ImageFont
    for p in (os.path.join(os.path.dirname(__file__), "..", "..", "Ascii_Studio", "assets",
                           "fonts", "CascadiaMono.ttf"),
              r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(os.path.normpath(p), px)
        except Exception:
            continue
    return ImageFont.load_default()


# --- drivers ----------------------------------------------------------------------------
def render_stills(scene, cfg: RenderConfig, times, out_png):
    """Dump a PNG at each time in `times` from ONE simulation run ââ‚¬â€ the tuning loop. Re-simulating
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
    bearing its final name is ALWAYS finalized/playable (the Oscilloscope convention)."""
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



