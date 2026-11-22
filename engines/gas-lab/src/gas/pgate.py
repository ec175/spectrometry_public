# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""pgate.py - the dry gate and the clip renderer for the 2026-09-25 PARTS set.

render.py's gate is built round cylinders (PV/NkT per cylinder box, volumetric efficiency).
A Roots blower has no cylinder; what makes it a SUPERCHARGER is measurable all the same:

  PROBES      named rectangles in the ducts. Per frame: particle count, temperature (mean
              kinetic energy), and pressure n*k*T over the probe's FREE area (the area not
              inside metal, found by sampling the solids), in bar.
  THROUGHPUT  net particles out of the outlet reservoirs per second - positive = it pumps.
  HEATING     outlet T minus inlet T. A Roots blower heats its charge by backflow; a scroll by
              internal compression; a damper by throttling. None of it is set anywhere.
  WALL CHECK  pressure on the metal: the impulse the gas delivered to each body per unit of its
              wetted perimeter, against n*k*T next to it (where a part has a probe beside it).
  MASS        (in - out) against the domain change: residual 0 or particles are leaking.
  RESCUES     mirrors deeper than one substep can explain. Too many = a part too thin, or
              moving too fast, for the substep.

`simulate()` runs the same schedule the clip does (warm-up at base speed, not shown, then the
clip: ramp to `mult` x by the midpoint and hold), and reports the whole run and the HOLD.
"""
from __future__ import annotations

import os
import subprocess
import time

import numpy as np

from . import config as C
from . import world as W
from .gpu import GPU, xp
from .render import _advance, _ffmpeg_proc


def schedule(eng, seconds, warm=None, mult=3.0, ramp=None):
    warm = float(eng.notes.get("warm", 3.0)) if warm is None else float(warm)
    ramp = float(eng.notes.get("ramp_frac", 0.5)) * seconds if ramp is None else float(ramp)
    mult = float(eng.notes.get("mult", mult))
    eng.drive.configure(warm=warm, ramp=ramp, mult=mult)
    return warm, ramp, mult


class Probes:
    """Rectangle probes with their free (non-metal) area measured once at t=0."""

    def __init__(self, world, eng):
        self.items = []
        for name, box, (x0, y0, x1, y1) in eng.probes:
            gx, gy = np.meshgrid(np.linspace(x0, x1, 48), np.linspace(y0, y1, 48))
            X = xp.asarray(gx.ravel().astype(np.float32))
            Y = xp.asarray(gy.ravel().astype(np.float32))
            B = xp.full(X.size, box, dtype=xp.int32)
            free = 1.0
            if world.solids is not None:
                free = float((world.solids.dmin(X, Y, B) > 0).mean())
            self.items.append((name, box, (x0, y0, x1, y1), (x1 - x0) * (y1 - y0) * free))

    def read(self, world):
        out = {}
        for name, box, (x0, y0, x1, y1), area in self.items:
            m = ((world.box == box) & (world.x >= x0) & (world.x < x1)
                 & (world.y >= y0) & (world.y < y1))
            n = int(m.sum())
            T = (W.temperature_K(float(xp.where(m, world.vx ** 2 + world.vy ** 2, 0.0).sum()) / n)
                 if n else C.T0)
            out[name] = dict(n=n, T=T, p=(n * T / max(area, 1.0)) / (C.N0 * C.T0))
        return out


def simulate(eng, cfg=None, seconds=13.0, warm=None, mult=3.0, seed=7, verbose=True, every=1):
    cfg = cfg or C.RenderConfig()
    warm, ramp, mult = schedule(eng, seconds, warm, mult)
    total = warm + seconds
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every, cap=900_000)
    probes = Probes(world, eng)
    extra = eng.notes.get("gate_probe")          # optional fn(world) -> dict of named numbers
    n = int(round(total * cfg.fps))
    trace = []
    t0 = time.time()
    for i in range(n):
        _advance(world, cfg, i / cfg.fps)
        if i % every == 0:
            f = dict(t=world.t, n_total=int(world.x.size), probes=probes.read(world),
                     n_out_ex=world.n_out_ex, n_in_ex=world.n_in_ex,
                     n_in=world.n_in, n_out_in=world.n_out_in)
            if world.solids is not None:
                f["tally"] = world.solids.take_tally()
            if extra is not None:
                f["extra"] = extra(world)
            trace.append(f)
        world.mark_frame()
    if verbose:
        report(world, trace, eng, cfg, total, warm, ramp, mult, time.time() - t0)
    return world, trace


def _stats(vals):
    v = np.asarray(vals, dtype=float)
    return v.min(), v.mean(), v.max()


def report(world, trace, eng, cfg, total, warm, ramp, mult, wall=None):
    d = eng.drive
    print(f"[dry] {eng.name}  {total:.1f}s = warm {warm:.1f}s + clip {total - warm:.1f}s "
          f"(ramp {ramp:.1f}s to x{mult:.1f}, then hold)  base {eng.rev_seconds:.3f} s/rev  "
          f"revs in clip {d.revs(total) - d.revs(warm):.2f}  bodies {len(eng.bodies)} "
          f"({sum(b.physics for b in eng.bodies)} physics)  holes {len(eng.holes)}")
    vmax = 0.0
    for b in eng.bodies:
        if not b.physics:
            continue
        for tt in np.linspace(warm, total, 60):
            x, y, a, vx, vy, om = b.pose(tt)
            vmax = max(vmax, abs(om) * b.solid.rad + np.hypot(vx, vy))
    if vmax > 0:
        print(f"[dry] SCALED GAS: v_thermal {C.V_TH:.0f} px/s, fastest metal {vmax:.0f} px/s "
              f"-> ratio {C.V_TH / vmax:.1f}; metal moves {vmax / cfg.substeps_per_sec:.2f} px/substep")
    phases = [("whole run", [f for f in trace if f["t"] >= warm * 0.5]),
              (f"hold x{mult:.1f}", [f for f in trace if f["t"] >= warm + ramp])]
    for tag, fr in phases:
        if len(fr) < 2:
            continue
        dt = fr[-1]["t"] - fr[0]["t"]
        print(f"[dry] ---- {tag} ({fr[0]['t']:.1f}-{fr[-1]['t']:.1f}s) ----")
        for name in (fr[0]["probes"] or {}):
            T = _stats([f["probes"][name]["T"] for f in fr])
            P = _stats([f["probes"][name]["p"] for f in fr])
            print(f"[dry]   probe {name:<12s} T {T[0]:6.0f} / {T[1]:6.0f} / {T[2]:6.0f} K   "
                  f"P {P[0]:5.2f} / {P[1]:5.2f} / {P[2]:5.2f} bar  (min/mean/max)")
        out_ex = (fr[-1]["n_out_ex"] - fr[0]["n_out_ex"]) - (fr[-1]["n_in_ex"] - fr[0]["n_in_ex"])
        in_in = (fr[-1]["n_in"] - fr[0]["n_in"]) - (fr[-1]["n_out_in"] - fr[0]["n_out_in"])
        print(f"[dry]   throughput  : {out_ex / dt:8.0f} /s out the outlet, {in_in / dt:8.0f} /s in at "
              f"the inlet  ({'PUMPS' if out_ex > 0 else 'no net delivery'})")
        if "extra" in fr[0]:
            for key in fr[0]["extra"]:
                v = _stats([f["extra"][key] for f in fr])
                print(f"[dry]   {key:<12s}: {v[0]:9.3f} / {v[1]:9.3f} / {v[2]:9.3f}")
        if "tally" in fr[0] and world.solids is not None:
            imp = sum(f["tally"][:, 0] for f in fr)
            for k, b in enumerate(world.solids.bodies):
                print(f"[dry]   body {b.name:<14s} impulse/s {imp[k] / dt:12.4g}")
    npk = max(f["n_total"] for f in trace)
    print(f"[dry] population    : {trace[0]['n_total']} -> {trace[-1]['n_total']}, peak {npk}"
          f"{'   ** AT THE CAP **' if npk >= world.cap - 64 else ''}")
    n_in = world.n_in + world.n_in_ex + world.n_injected + getattr(world, "n_created", 0)
    n_out = world.n_out_in + world.n_out_ex
    # the domain at t=0 is the seed; the trace starts one frame in, so use the world's own count
    dom0 = world._seeded
    resid = (n_in - n_out) - (int(world.x.size) - dom0)
    print(f"[dry] mass balance  : in {n_in} out {n_out} (net {n_in - n_out:+d}) vs domain change "
          f"{int(world.x.size) - dom0:+d}   residual {resid:+d}  "
          f"{'OK' if abs(resid) <= 2 else '** PARTICLES LEAKING **'}")
    if eng.igniters:
        bal = (world.made_co2 == world.reactions and world.made_h2o == 2 * world.reactions)
        print(f"[dry] reactions     : {world.reactions} burnt   ledger {'OK' if bal else '** BROKEN **'}")
        ig = [e for e in world.events if e[1] in ("spark", "autoignite", "inject")]
        print(f"[dry] ignition      : {len(ig)} event(s): "
              + ", ".join(f"{e[1]} {e[0]:.2f}s" for e in ig[:30]))
    resc = world.solids.rescues() if world.solids is not None else 0
    tot = sum(f["n_total"] for f in trace) * max(1, cfg.substeps_per_sec // cfg.fps)
    print(f"[dry] tunnelling    : {C.V_TH / cfg.substeps_per_sec:.2f} px/substep at v_thermal   "
          f"box rescues {world.n_clamped}   metal rescues {resc} "
          f"({resc / max(tot, 1) * 1e6:.2f} per million particle-substeps)"
          f"{'   ** too many **' if resc > 2e-4 * tot else ''}")
    if wall:
        print(f"[dry] {wall:.1f}s wall clock ({wall / total:.1f} s per simulated second)")


# ====================================================================== rendering
def render_clip(eng, cfg, out_path, seconds=13.0, warm=None, mult=3.0, seed=7, log_every=60,
                style=None):
    from .pframe import PartFrame
    warm, ramp, mult = schedule(eng, seconds, warm, mult)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every, cap=900_000)
    fr = PartFrame(cfg, eng, style)
    N = int(round(seconds * cfg.fps))
    t0 = time.time()
    _advance(world, cfg, warm - 1.0 / cfg.fps)
    world.mark_frame()
    print(f"[warm] {warm:.1f}s simulated in {time.time() - t0:.0f}s, N={world.x.size}", flush=True)
    tmp = os.path.splitext(out_path)[0] + ".part.mp4"
    proc = _ffmpeg_proc(cfg, tmp)
    print(f"[render] {eng.name} {cfg.width}x{cfg.height} @{cfg.fps} ({N} frames, ramp {ramp:.1f}s "
          f"to x{mult:.1f}, physics {cfg.substeps_per_sec}/s, gpu={GPU})", flush=True)
    try:
        for i in range(N):
            t = warm + i / cfg.fps
            _advance(world, cfg, t)
            img = fr.compose(fr.gas(world.snapshot()), world.t)
            proc.stdin.write(img.tobytes())
            world.mark_frame()
            if log_every and (i + 1) % log_every == 0:
                print(f"[render] {i + 1}/{N}  {time.time() - t0:6.1f}s  x{eng.drive.s(world.t):.2f}  "
                      f"N={world.x.size}", flush=True)
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
    except BaseException:
        try:
            proc.kill()
        except Exception:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    print(f"[done] {out_path}  ({time.time() - t0:.1f}s)", flush=True)
    return out_path


def render_still(eng, cfg, t, out_path, seed=7, seconds=13.0, warm=None, mult=3.0, style=None):
    """One PNG at clip time t (0 = first frame of the clip, after the warm-up)."""
    from PIL import Image
    from .pframe import PartFrame
    warm, ramp, mult = schedule(eng, seconds, warm, mult)
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every, cap=900_000)
    fr = PartFrame(cfg, eng, style)
    _advance(world, cfg, warm + t - 1.0 / cfg.fps)
    world.mark_frame()
    _advance(world, cfg, warm + t)
    img = fr.compose(fr.gas(world.snapshot()), world.t)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"[still] clip t={t:.2f}s (sim {world.t:.2f}s) x{eng.drive.s(world.t):.2f} "
          f"N={world.x.size} -> {out_path}")
    return out_path


def layout(eng, cfg, out_path, t=0.0):
    """Metal only, plus the RENDERED-EXTENT check: frame_fit() only ever checked gas boxes,
    and radial.mp4 shipped with drawn metal off both edges. This measures the metal's alpha
    against the frame edge, per side."""
    from PIL import Image
    from .pframe import PartFrame
    schedule(eng, 13.0)
    fr = PartFrame(cfg, eng)
    img, a = fr.layout(t)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    Image.fromarray(img).save(out_path)
    edge = dict(left=float(a[:, :2].max()), right=float(a[:, -2:].max()),
                top=float(a[:2, :].max()), bottom=float(a[-2:, :].max()))
    print(f"[layout] -> {out_path}   metal at the frame edge: "
          + "  ".join(f"{k} {v:.2f}" for k, v in edge.items()))
    return edge
