# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""render.py - the substep loop, the dry gate, and stills.

THE ENGINE CONVENTION
---------------------
There is deliberately **no plugin system, no registry and no base class**. An engine is
just an object with five things.
If you write a third, follow the same shape and add it to `ENGINES` in `serve.py`:

    engine.NAME               str - "objects"
    engine.PARAMS             list of kit.params.P - drives the sidebar
    engine.bind(cfg, canvas)  called ONCE by Runner before anything else; stash both
    engine.background_rgb()   -> (r, g, b); the Canvas is constructed with it
    engine.post()             -> dict of post-processing settings, passed to
                                 Canvas.to_rgb (see kit/canvas.py for the keys)
    engine.reset()            rebuild initial state from self.p and self.seed
    engine.step(dt)           advance the simulation by exactly dt seconds
    engine.draw(canvas)       paint current state into the Canvas

`Runner` calls `bind`, `background_rgb` and `post` unconditionally, so all three are
required, not optional. (An earlier version of this list said `engine.POST`, a dict, and
omitted `bind` and `background_rgb` - which meant that following it exactly produced an
engine that crashed on its first frame. If you extend this list, scaffold a toy engine
from it and run `cli.py dry` on that engine before trusting it.)

Optionally:

    engine.state_arrays()     dict of name -> ndarray, so the dry gate can look for NaN
                              in the STATE rather than only in the picture. Implement it.
                              A frame can look fine for several seconds after the physics
                              has already gone non-finite.
    engine.check_invariants() -> list of complaints; the dry gate fails on any.
    engine.summary()          dict of cheap scalars (counts, generations) shown in the UI.
    engine.apply_live()       re-read self.p into the RUNNING simulation without
                              rebuilding it, so a slider drag does not restart the scene.
                              Without it, the server has to restart on every edit.


THE SUBSTEP LOOP
----------------
    while t < frame_time: step(1 / SUBSTEP_HZ)

The sim NEVER takes a step of `1/fps`. Because 120 is divisible by 24, 30 and 60, every
frame boundary lands exactly on a substep, so a 30 fps preview and a 60 fps final walk
through the identical sequence of states and merely sample it at different rates. This is
what makes the browser preview a preview rather than a different video.

Do not "optimise" this into a variable-timestep integrator. The reason is not accuracy,
it is REPRODUCIBILITY: the same scene file must produce the same video on a fast machine
and a slow one, today and in a year.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .config import SUBSTEP_HZ
from .canvas import Canvas
from .encoder import VideoWriter, verify_output


def _luma(img):
    a = img.astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def frame_stats(img) -> dict:
    """The handful of numbers every gate in this kit is written against."""
    lum = _luma(img)
    v = img.max(axis=2).astype(np.float32)
    return {
        "luma_mean": float(lum.mean()),
        "luma_max": float(lum.max()),
        "luma_p99": float(np.percentile(lum, 99)),
        "lit_frac": float((v > 0.25 * 255).mean()),
        "blown_frac": float((lum > 250).mean()),
        "black_frac": float((v < 8).mean()),
    }


class Runner:
    """Owns the clock. One per render, one per live stream."""

    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.dt = 1.0 / SUBSTEP_HZ
        self.t = 0.0
        self.canvas = Canvas(cfg, bg=engine.background_rgb())
        engine.bind(cfg, self.canvas)

    def reset(self):
        self.t = 0.0
        self.engine.reset()
        self.canvas.trail = None
        self.canvas.clear()

    def advance_to(self, t_target):
        """Step until `t_target`, exactly. Never overshoots by more than one substep."""
        n = int(math.floor(t_target / self.dt + 1e-9)) - int(math.floor(self.t / self.dt + 1e-9))
        for _ in range(max(0, n)):
            self.engine.step(self.dt)
            self.t += self.dt

    def frame(self):
        self.canvas.clear()
        self.engine.draw(self.canvas)
        return self.canvas.to_rgb(self.engine.post())


def render_scene(engine, cfg, out, seconds, on_progress=None, prefer_nvenc=True):
    """Render to an mp4 and VERIFY it. Returns the verify_output dict.

    Always verifies. A truncated file is produced by several ordinary failures and is not
    obvious from the file merely existing.
    """
    n_frames = int(round(seconds * cfg.fps))
    r = Runner(engine, cfg)
    r.reset()
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with VideoWriter(out, cfg, prefer_nvenc=prefer_nvenc) as w:
        for i in range(n_frames):
            r.advance_to(i / float(cfg.fps))
            w.write(r.frame())
            if on_progress and (i % 10 == 0 or i == n_frames - 1):
                on_progress(i + 1, n_frames, time.time() - t0)
    res = verify_output(out, expect_frames=n_frames, expect_size=(cfg.width, cfg.height))
    res["seconds_elapsed"] = round(time.time() - t0, 1)
    res["frames"] = n_frames
    return res


def render_still(engine, cfg, t, out=None):
    r = Runner(engine, cfg)
    r.reset()
    r.advance_to(t)
    img = r.frame()
    if out:
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        Image.fromarray(img).save(out)
    return img


def dry(engine, cfg, seconds, n=5, verbose=True, extra_checks=None):
    """THE GATE. Run this before every render. It costs a couple of seconds.

    What it catches, in the order the failures actually happen:

    * **non-finite state** - an unsoftened division by zero. The picture can look
      completely normal for seconds after the physics has gone to NaN, because a NaN
      coordinate simply fails to draw. Checked in `engine.state_arrays()`, not in pixels.
    * **a black frame** - the single most common wasted render. Everything ran, nothing
      was visible.
    * **a blown frame** - gains too high; the subject is a white blob.
    * **no motion** - the sim is running but the picture is static, so the video is an
      expensive still.
    * **no progression** - there IS motion but the end looks like the beginning, i.e. the
      scene has no arc. Compared over WINDOWS at each end, not single frames, because one
      frame of a jittery scene differs from another by noise.

    Returns (ok, report). `ok` False means do not render.
    """
    n_frames = int(round(seconds * cfg.fps))
    r = Runner(engine, cfg)
    r.reset()

    h = 1.0 / cfg.fps
    nw = max(3, int(round(min(1.0, seconds / 8.0) * cfg.fps)))
    win_a = [k * h for k in range(nw)]
    win_b = [max(0.0, seconds - (nw - k) * h) for k in range(nw)]
    probes = [seconds * i / float(n) for i in range(n)]
    times = sorted(set(round(x, 6) for x in (win_a + win_b + probes
                                             + [p + h for p in probes])))

    got = {}
    t0 = time.time()
    for t in times:
        r.advance_to(t)
        got[t] = r.frame()
    elapsed = time.time() - t0
    ms = elapsed / max(1, len(times)) * 1000.0

    rep = {
        "engine": getattr(engine, "NAME", "?"),
        "size": [cfg.width, cfg.height],
        "fps": cfg.fps,
        "seconds": seconds,
        "frames": n_frames,
        "ms_per_probe": round(ms, 1),
        "est_render_s": round(ms * 0.55 * n_frames / 1000.0, 1),
        "fail": [],
        "warn": [],
        "stats": [],
    }
    fail, warn = rep["fail"], rep["warn"]

    # --- engine invariants ---------------------------------------------------------------
    if hasattr(engine, "check_invariants"):
        for msg in (engine.check_invariants() or []):
            fail.append(msg)

    # --- non-finite STATE ---------------------------------------------------------------
    if hasattr(engine, "state_arrays"):
        for nm, arr in (engine.state_arrays() or {}).items():
            a = np.asarray(arr)
            if a.size and not np.isfinite(a).all():
                bad = int((~np.isfinite(a)).sum())
                fail.append("%d non-finite values in state[%r]" % (bad, nm))

    # --- per-probe picture --------------------------------------------------------------
    keys = [min(times, key=lambda x, k=k: abs(x - k)) for k in probes]
    for k in keys:
        s = frame_stats(got[k])
        s["t"] = round(k, 3)
        rep["stats"].append({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in s.items()})
        if s["luma_max"] < 24:
            fail.append("t=%.2fs the frame is black (max luma %.0f)" % (k, s["luma_max"]))
        if s["blown_frac"] > 0.06:
            fail.append("t=%.2fs the frame is blown (%.1f%% clipped)"
                        % (k, s["blown_frac"] * 100))
        if s["lit_frac"] < 0.002:
            warn.append("t=%.2fs almost nothing is lit (%.3f%% of pixels)"
                        % (k, s["lit_frac"] * 100))

    # --- motion, at the render's own frame rate ------------------------------------------
    steps = []
    for p in probes:
        a = min(times, key=lambda x: abs(x - p))
        b = min(times, key=lambda x: abs(x - (p + h)))
        if a == b:
            continue
        steps.append(float(np.abs(_luma(got[b]) - _luma(got[a])).mean()))
    if steps:
        rep["motion_per_frame"] = round(float(np.mean(steps)), 4)
        if max(steps) < 0.02:
            fail.append("nothing moves (mean |dframe| = %.4f over %d probes)"
                        % (float(np.mean(steps)), len(steps)))

    # --- progression: does the end differ from the beginning? -----------------------------
    A = np.mean([_luma(got[min(times, key=lambda x: abs(x - t))]) for t in win_a], axis=0)
    B = np.mean([_luma(got[min(times, key=lambda x: abs(x - t))]) for t in win_b], axis=0)
    prog = float(np.abs(B - A).mean())
    rep["progression"] = round(prog, 4)
    if prog < 0.5:
        warn.append("the end looks like the beginning (progression %.3f) - this may be an "
                    "expensive still" % prog)

    if extra_checks:
        extra_checks(rep, got, r)

    rep["ok"] = not fail
    if verbose:
        print("[dry] %s  %dx%d @%d  %.1fs  %d frames" %
              (rep["engine"], cfg.width, cfg.height, cfg.fps, seconds, n_frames))
        print("      %.0f ms/probe -> full render approx %.0fs"
              % (rep["ms_per_probe"], rep["est_render_s"]))
        print("      motion/frame %s   progression %s"
              % (rep.get("motion_per_frame"), rep["progression"]))
        for w_ in warn:
            print("      WARN  " + w_)
        for f_ in fail:
            print("      FAIL  " + f_)
        print("      %s" % ("OK - safe to render" if rep["ok"] else "DO NOT RENDER"))
    return rep["ok"], rep
