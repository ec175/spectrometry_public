"""render.py - the clock, the ffmpeg pipe, and the audio pass.

Two entry points:
  simulate(scene, cfg)      - physics only, no drawing. Answers "does the scene DO its brief?"
                              (how many escapes, when, how the audio schedule falls out) in a
                              second or two. Wind_Tunnel's lesson: a scene with an intent needs
                              a check that measures THAT, not just one that survives a render.
  render_scene(...)         - frames -> ffmpeg -> silent mp4 -> build track -> mux.

The physics rate (cfg.substeps_per_sec) is independent of fps, so a 30 fps preview and a
60 fps final are the SAME simulation - only sampled more often.
"""
from __future__ import annotations

import os
import subprocess
import time

from . import config
from .audio import Score, build_track, mux_track
from .draw import Frame, draw_world


def _drive(world, score, scene, cfg, t, ti):
    """Advance the sim to video time t and hand any new events to the Score."""
    dt = 1.0 / cfg.substeps_per_sec
    while world.t < t - 1e-9:
        world.step(dt)
    while ti < len(world.triggers):
        score.fire(world.triggers[ti]["t"])
        ti += 1
    return ti


def simulate(scene, cfg=None, verbose=True):
    """Run the physics for the scene's duration. Returns (world, score)."""
    cfg = cfg or config.RenderConfig()
    if scene.substeps:
        cfg.substeps_per_sec = scene.substeps
    world = scene.build()
    score = Score(scene.slice_len, scene.song_start)
    n = int(round(scene.duration * cfg.fps))
    ti = 0
    t0 = time.time()
    for i in range(n):
        ti = _drive(world, score, scene, cfg, i / cfg.fps, ti)
    if verbose:
        report(world, score, scene, time.time() - t0)
    return world, score


def report(world, score, scene, wall=None):
    print(f"[sim] {scene.name}  {scene.duration:.1f}s")
    kinds = {}
    for ev in world.triggers:
        k = ev.get("kind", "escape")
        kinds[k] = kinds.get(k, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"
    print(f"[sim] headline events: {len(world.triggers)}  ({breakdown})")
    if world.inner_exits:
        print(f"[sim] inner-shell exits             : {len(world.inner_exits)}")
    print(f"[sim] fell off the bottom            : {world.exits_bottom}"
          f"   live balls at end: {world.n_live()}"
          f"   spawns suppressed by the cap: {world.suppressed_spawns}")
    print(f"[sim] audio: {len(score.segments)} segment(s), "
          f"{score.n_events} event(s), {score.n_extended} extension(s), "
          f"{score.song_seconds():.2f}s of song "
          f"({100.0 * min(1.0, score.song_seconds() / scene.duration):.0f}% of the clip)")
    if world.bounces:
        rs = [b["r"] for b in world.bounces]
        from .audio import blip_freq
        print(f"[sim] bounce blips: {len(world.bounces)} "
              f"({len(world.bounces) / scene.duration:.1f}/s), "
              f"r {min(rs):.1f}-{max(rs):.1f}px -> "
              f"{blip_freq(max(rs)):.0f}-{blip_freq(min(rs)):.0f} Hz")
    if world.triggers:
        ts = ", ".join(f"{e['t']:.2f}" for e in world.triggers)
        print(f"[sim] trigger times: {ts}")
    for st, dur in score.segments:
        print(f"[sim]   play {st:6.2f}s -> {st + dur:6.2f}s  ({dur:.1f}s)")
    if wall:
        print(f"[sim] {wall:.1f}s wall clock")


def render_scene(scene, cfg, out_path, silent=False, log_every=30):
    if scene.substeps:
        cfg.substeps_per_sec = scene.substeps
    ffmpeg = config.find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    base, _ = os.path.splitext(out_path)
    silent_path = base + "_silent.mp4"
    tmp = silent_path + ".part.mp4"

    world = scene.build()
    score = Score(scene.slice_len, scene.song_start)
    fr = Frame(cfg)
    n = int(round(scene.duration * cfg.fps))

    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
           *config.encoder_args(), "-pix_fmt", "yuv420p", tmp]
    print(f"[render] {scene.name} {cfg.width}x{cfg.height} @{cfg.fps} "
          f"({n} frames, physics {cfg.substeps_per_sec}/s, ss={cfg.ss})")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=None)
    ti = 0
    t0 = time.time()
    try:
        for i in range(n):
            t = i / cfg.fps
            ti = _drive(world, score, scene, cfg, t, ti)
            draw_world(fr, world, scene.style, t, score)
            proc.stdin.write(fr.to_rgb().tobytes())
            # trails are sampled per FRAME (after drawing, so the head is not double-drawn)
            for b in world.balls:
                if b.alive:
                    b.trail.append((b.x, b.y))
                    if len(b.trail) > scene.style.trail_len:
                        del b.trail[0]
            world.flashes = [f for f in world.flashes if t - f["t"] < 1.0]
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (n - i - 1)
                print(f"[render] {i+1}/{n}  {el:5.1f}s elapsed  ~{eta:5.1f}s left  "
                      f"balls={world.n_live()}  triggers={len(world.triggers)}", flush=True)
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
    os.replace(tmp, silent_path)
    print(f"[render] silent -> {silent_path}  ({time.time() - t0:.1f}s)")
    report(world, score, scene)

    bounces = world.bounces if scene.bounce_gain > 0 else None
    events = world.triggers if scene.event_gain > 0 else None
    if silent or (not score.segments and not bounces and not events):
        print("[audio] nothing to play - leaving the render silent")
        os.replace(silent_path, out_path)
        print(f"[done] {out_path}")
        return out_path

    song = scene.song if score.segments else ""
    if song and not os.path.exists(song):
        raise FileNotFoundError(f"song not found: {song}")
    wav = base + "_track.wav"
    build_track(score, song, scene.duration, wav, bounces=bounces,
                bounce_gain=scene.bounce_gain, duck=scene.bounce_duck,
                events=events, event_gain=scene.event_gain)
    mux_track(silent_path, wav, out_path)
    print(f"[done] {out_path}")
    return out_path
