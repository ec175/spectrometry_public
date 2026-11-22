# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""render.py - the clock, the dry gate, the ramp, and the ffmpeg pipe.

  simulate(eng, cfg)       THE DRY GATE. Full physics, no drawing, over the SAME schedule the
                           clip uses (warm-up at base speed, then the ramp and the hold).
                           Measures whether each cylinder is an ENGINE: PV/NkT against the
                           independently measured wall pressure, the T swing per stroke,
                           volumetric efficiency, the reaction ledger, the mass balance
                           (including injected fuel), and the geometry-rescue count - and
                           then the same again for the HOLD phase alone, because an engine
                           that breathes at base speed can choke at three times it.
  render_clip(...)         frames -> raw RGB24 -> ffmpeg -> atomic rename off a .part.mp4.

THE SCHEDULE (Clancy, 2026-09-20): warm up a full period at base speed (the domain starts at
rest with no residual, and a two-stroke's first cycle has no charge in it), then the clip:
the crank accelerates from t = 0 to the halfway point and holds at `mult` x (>= 3) to the end.
That is engine.Drive; every clock, piston and cam reads it. The clip therefore plays ONCE - a
speed ramp cannot wrap - which is the library's default rule for narrative clips anyway.
"""
from __future__ import annotations

import os
import subprocess
import time

from . import config as C
from . import world as W


def _advance(world, cfg, t):
    dt = 1.0 / cfg.substeps_per_sec
    while world.t < t - 1e-9:
        world.step(dt)


def schedule(eng, seconds=10.0, warm=None, mult=3.0, ramp=None):
    """Configure the engine's drive for a clip: returns (warm, ramp)."""
    warm = max(eng.period, 10.0) if warm is None else float(warm)
    ramp = 0.5 * seconds if ramp is None else float(ramp)
    eng.drive.configure(warm=warm, ramp=ramp, mult=mult)
    return warm, ramp


# ====================================================================== the gate
def simulate(eng, cfg=None, seconds=10.0, warm=None, mult=3.0, seed=7, verbose=True):
    """The dry gate over warm-up + clip. Returns (world, trace)."""
    cfg = cfg or C.RenderConfig()
    warm, ramp = schedule(eng, seconds, warm, mult)
    total = warm + seconds
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every)
    n = int(round(total * cfg.fps))
    trace = []
    t0 = time.time()
    for i in range(n):
        _advance(world, cfg, i / cfg.fps)
        trace.append(world.frame_state())
        world.reset_impulse()
        world.mark_frame()
    if verbose:
        report(world, trace, eng, cfg, total, warm, ramp, time.time() - t0)
    return world, trace


def _cyl_block(trace, eng, cfg, tag):
    cyls = eng.cyl_boxes()
    for ci, i in enumerate(cyls):
        b = eng.boxes[i]
        Ts = [f["cyl"][ci]["T"] for f in trace]
        pi = [f["cyl"][ci]["p_ideal"] for f in trace]
        pw = [f["cyl"][ci]["p_wall"] for f in trace]
        ns = [f["cyl"][ci]["n"] for f in trace]
        ok = [a / bb for a, bb in zip(pw, pi) if bb > 0.02]
        hi = max(range(len(Ts)), key=lambda k: Ts[k])
        m = sum(ok) / len(ok) if ok else 0.0
        areas = [f["cyl"][ci]["area"] for f in trace]
        kmax = max(range(len(areas)), key=lambda k: areas[k])
        boost = max([bx.boost for bx in eng.boxes if bx.role == "intake"] or [1.0])
        ve = max(ns) / (C.N0 * boost * areas[kmax])
        # volumetric efficiency only means something for a box a piston sweeps; a fixed side
        # chamber (flathead) fills at compression and would read as "blowing through"
        ve_s = (f"VE {ve:.2f}{' (x' + format(boost, '.1f') + ' boost)' if boost != 1.0 else ''}"
                f"{'  ** BLOWING THROUGH **' if ve > 1.15 else ''}") if b.floor >= 0 else "fixed"
        print(f"[dry{tag}] {b.label:<12s} T {min(Ts):5.0f}..{max(Ts):5.0f} K (peak t={trace[hi]['t']:5.2f}s) "
              f"P {min(pi):5.2f}..{max(pi):6.2f} bar  PV/NkT {m:.3f} "
              f"{'OK' if 0.75 < m < 1.35 else '** OFF **'}  N {min(ns)}..{max(ns)}  {ve_s}")


def report(world, trace, eng, cfg, seconds, warm, ramp, wall=None):
    """Everything here is MEASURED off the particles. Nothing is a target that was set."""
    from . import engine as E
    cyls = eng.cyl_boxes()
    d = eng.drive
    locked = getattr(d, "locked", False)
    how = ("locked, no ramp" if locked else f"ramp {ramp:.1f}s to x{d.mult:.1f}, then hold")
    print(f"[dry] {eng.name}  {seconds:.1f}s = warm {warm:.0f}s + clip {seconds - warm:.0f}s "
          f"({how})  rev={eng.rev_seconds:.4f}s  "
          f"cycle={eng.cycle_deg:.0f} deg  {len(cyls)} cylinder box(es), {len(eng.boxes)} boxes, "
          f"{len(eng.pistons)} piston(s)  revs in clip {d.revs(seconds) - d.revs(warm):.2f}")
    vpk = max(E.peak_speed(p.fn, eng.period) for p in eng.pistons)
    if locked:
        # peak_speed already measures the piston AT the locked speed - scaling by `mult` here
        # double-counts and printed 1.2 where the truth is 11.2.
        print(f"[dry] SCALED GAS: v_thermal(300K)={C.V_TH:.0f} px/s, peak piston {vpk:.0f} px/s "
              f"-> ratio {C.V_TH / vpk:.1f}  (a real engine is ~100; below ~8 the ports choke)")
    else:
        print(f"[dry] SCALED GAS: v_thermal(300K)={C.V_TH:.0f} px/s, peak piston {vpk:.0f} px/s at base "
              f"-> ratio {C.V_TH / vpk:.1f}, {C.V_TH / (vpk * d.mult):.1f} at x{d.mult:.1f}  "
              f"(a real engine is ~100; below ~8 the ports choke)")
    print(f"[dry] ---- whole run ----")
    _cyl_block(trace, eng, cfg, "")
    hold = [f for f in trace if f["t"] >= warm + ramp]
    if hold:
        print(f"[dry] ---- hold phase (t >= {warm + ramp:.1f}s, x{d.mult:.1f}) ----")
        _cyl_block(hold, eng, cfg, " hold")
    # per-stroke table for the first cylinder, hold phase
    by = {}
    for f in (hold or trace):
        by.setdefault(f["stroke"], []).append(f)
    print(f"[dry] {'stroke (cyl 0, hold)':<22s} {'T min':>7s} {'T max':>7s} {'P min':>7s} {'P max':>7s} {'N max':>7s}")
    for k in ("intake", "compression", "power", "exhaust"):
        if k in by:
            v = by[k]
            print(f"[dry]   {k:<20s} {min(f['T'] for f in v):7.0f} {max(f['T'] for f in v):7.0f} "
                  f"{min(f['p_ideal'] for f in v):7.2f} {max(f['p_ideal'] for f in v):7.2f} "
                  f"{max(f['n_cyl'] for f in v):7d}")
    npk = max(f["n_total"] for f in trace)
    print(f"[dry] population    : {trace[0]['n_total']} -> {trace[-1]['n_total']}, peak {npk}"
          f"{'   ** AT THE CAP - the domain is inflating **' if npk >= world.cap - 64 else ''}")
    n_in = world.n_in + world.n_in_ex + world.n_injected
    n_out = world.n_out_in + world.n_out_ex
    d_dom = trace[-1]["n_total"] - trace[0]["n_total"]
    resid = (n_in - n_out) - d_dom
    print(f"[dry] mass flow     : intake {world.n_in} in / {world.n_out_in} out, "
          f"exhaust {world.n_in_ex} in / {world.n_out_ex} out, injected {world.n_injected}")
    print(f"[dry] mass balance  : (in - out) {n_in - n_out:+d} vs domain change {d_dom:+d}"
          f"   residual {resid:+d}  {'OK' if abs(resid) <= 2 else '** PARTICLES LEAKING **'}")
    print(f"[dry] net through   : {world.n_out_ex - world.n_in_ex} out the exhaust, "
          f"{world.n_in - world.n_out_in} in at the intake")
    if eng.igniters:
        bal = (world.made_co2 == world.reactions and world.made_h2o == 2 * world.reactions)
        print(f"[dry] reactions     : {world.reactions} burnt -> CO2 {world.made_co2}, "
              f"H2O {world.made_h2o}   ledger {'OK' if bal else '** BROKEN **'}")
        ig = [e for e in world.events if e[1] in ("spark", "autoignite", "inject")]
        print(f"[dry] ignition      : {len(ig)} event(s): "
              + ", ".join(f"{e[1]} {e[0]:.2f}s ({e[2]})" for e in ig[:24]))
        end = trace[-1]["counts"]
        print(f"[dry] cyl 0 mix at the end: "
              + "  ".join(f"{C.SPECIES_NAMES[k]}={end[k]}" for k in (0, 1, 2, 3)))
    step = C.V_TH / cfg.substeps_per_sec
    print(f"[dry] tunnelling    : {step:.2f} px per substep at v_thermal   "
          f"geometry rescues: {world.n_clamped}"
          f"{'   ** too many - raise --substeps **' if world.n_clamped > 0.002 * trace[-1]['n_total'] * len(trace) else ''}")
    if wall:
        print(f"[dry] {wall:.1f}s wall clock, {len(trace)} frames simulated "
              f"({wall / seconds:.1f} s per simulated second)")


# ====================================================================== rendering
def _ffmpeg_proc(cfg, tmp):
    ffmpeg = C.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-nostats", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
           *C.encoder_args(cfg), "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=None)


def render_still(eng, cfg, style, t, out_path, seed=7, seconds=10.0, warm=None, mult=3.0):
    from .draw import Frame
    schedule(eng, seconds, warm, mult)
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every)
    fr = Frame(cfg, style, eng)
    _advance(world, cfg, max(0.0, t - 1.0 / cfg.fps))
    world.mark_frame()
    _advance(world, cfg, t)
    from PIL import Image
    img = fr.compose(fr.gas([(world.snapshot(), 1.0)]), world.t)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    Image.fromarray(img).save(out_path)
    s = world.frame_state()
    print(f"[still] t={t:.2f}s  {s['stroke']} {s['deg']:.0f}deg  x{eng.drive.s(t):.2f}  T={s['T']:.0f}K "
          f"P={s['p_ideal']:.2f}bar  N_cyl={s['n_cyl']}  -> {out_path}")
    return out_path


def render_clip(eng, cfg, style, out_path, seconds=10.0, warm=None, mult=3.0, seed=7,
                log_every=60):
    """Warm-up at base speed (not output), then the clip: ramp to `mult` x by the midpoint,
    hold to the end. See the module docstring."""
    from .draw import Frame
    warm, ramp = schedule(eng, seconds, warm, mult)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    world = W.World(eng, seed=seed, coll_every=cfg.coll_every)
    fr = Frame(cfg, style, eng)
    N = int(round(seconds * cfg.fps))
    nwarm = int(round(warm * cfg.fps))
    t0 = time.time()

    for i in range(1, nwarm + 1):
        _advance(world, cfg, i / cfg.fps)
        world.mark_frame()
        world.reset_impulse()
        if log_every and i % log_every == 0:
            st = world.frame_state()
            print(f"[warm] {i}/{nwarm}  {time.time() - t0:6.1f}s  {st['stroke']:<11s} "
                  f"T={st['T']:5.0f}K P={st['p_ideal']:5.2f}bar N={st['n_total']}", flush=True)

    tmp = os.path.splitext(out_path)[0] + ".part.mp4"
    proc = _ffmpeg_proc(cfg, tmp)
    print(f"[render] {eng.name} {cfg.width}x{cfg.height} @{cfg.fps} ({N} frames, ramp {ramp:.1f}s "
          f"to x{mult:.1f}, physics {cfg.substeps_per_sec}/s, ss={cfg.ss})", flush=True)
    try:
        for i in range(N):
            t = warm + i / cfg.fps
            _advance(world, cfg, t)
            gas = fr.gas([(world.snapshot(), 1.0)])
            proc.stdin.write(fr.compose(gas, world.t).tobytes())
            world.mark_frame()
            world.reset_impulse()
            if log_every and (i + 1) % log_every == 0:
                st = world.frame_state()
                el = time.time() - t0
                print(f"[render] {i + 1}/{N}  {el:6.1f}s  x{eng.drive.s(world.t):.2f}  "
                      f"{st['stroke']:<11s} T={st['T']:5.0f}K P={st['p_ideal']:5.2f}bar "
                      f"N={st['n_total']}", flush=True)
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
