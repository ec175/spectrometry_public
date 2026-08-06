"""render.py - frame clock, compositor and the ffmpeg pipe.

Frames are driven by REAL SECONDS (t = i/fps), never by frame index. That is what makes a
540x960 @30 preview and a 1080x1920 @60 final the SAME animation, so a preview is trustworthy -
the same rule as every other engine in this library.

Encoding is atomic: raw RGB24 goes to `<out>.mp4.part.mp4` and the file is renamed onto its real
name only on success. A file bearing its final name is always finalized and playable, which is
what makes the skip-if-exists logic in render_five.ps1 sound.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from . import draw as D
from .config import OUT, encoder_args, find_ffmpeg
from .lines import END_NAMES, END_BUDGET, trace

ALPHA_MIN = 0.02


def solve(spec):
    """Trace only the lines that will actually be drawn.

    A scene allocates its seed slots once at N_max and controls how many are LIT with alpha;
    integrating the dark ones would be pure waste, and in the sparse phases of every scene here
    that is most of them.
    """
    keep = np.nonzero(spec.alphas > ALPHA_MIN)[0]
    if not len(keep):
        z = np.zeros((0, spec.n_steps + 1, 2), np.float32)
        return (z, np.zeros(0, np.int32), np.zeros(0, np.int8),
                np.zeros((0, 3)), np.zeros(0), keep)
    pts, length, ended = trace(
        spec.fieldfn, spec.seeds[keep], ds=spec.ds, n_steps=spec.n_steps,
        sinks=spec.sinks, capture_r=spec.capture_r, bounds=spec.bounds)
    cols = spec.colors(ended) if callable(spec.colors) else np.asarray(spec.colors)[keep]
    return pts, length, ended, cols, spec.alphas[keep], keep


def compose(fr, spec, t):
    """One frame from one FrameSpec. Draw order is lines -> pulses -> markers, and BODIES last
    of all: a solid body is a hole punched in the picture, not an object floating on it."""
    fr.clear()
    fr.dens_tint = spec.dens_tint
    fr.glow_mul, fr.glow2_mul = spec.glow_mul, spec.glow2_mul
    pts, length, ended, cols, alphas, keep = solve(spec)

    if len(length):
        fr.accumulate(pts, length, alphas, sub=3, gain=spec.dens_gain)
        fr.polylines(pts, length, cols, alphas, width=spec.line_width)
        if spec.pulse is not None:
            idx, k0, k1, g = spec.pulse.slices(t, length, spec.ds, spec.n_steps,
                                               seed_ids=keep)
            if len(idx):
                fr.strokes(pts, idx, k0, k1, D.WHITE,
                           g * np.clip(alphas[idx] * 1.5, 0.0, 1.0),
                           width=spec.pulse_width)

    for m in [x for x in spec.markers if x["kind"] != "body"]:
        if m["kind"] == "filled":
            fr.disc(m["x"], m["y"], m["r"], m["color"])
            if m["sign"]:
                fr.glyph(m["x"], m["y"], m["r"], m["sign"],
                         m["glyph"] or (0, 0, 0), width=max(1.8, 0.22 * m["r"]))
        else:
            fr.ring(m["x"], m["y"], m["r"], m["width"], m["color"])
            if m["sign"]:
                fr.glyph(m["x"], m["y"], m["r"] * 0.62, m["sign"], m["color"],
                         width=max(1.6, 0.16 * m["r"]))
    for m in [x for x in spec.markers if x["kind"] == "body"]:
        fr.disc(m["x"], m["y"], m["r"], (0, 0, 0))
        fr.ring(m["x"], m["y"], m["r"], m["width"], m["color"])
    return fr.to_rgb(), (length, ended, alphas)


# --------------------------------------------------------------------------------------
def flux_report(scene, seconds=None, samples=40):
    """Measure the INTENT, not just that it did not crash.

    `dbg`-equivalent for this project. A stable render tells you nothing about whether the
    lines are doing what a field line must do, so this counts fates. END_BUDGET is the number
    that matters: a line that ran out of arclength means capture_r is smaller than ds (it
    stepped over its sink) or the bounds are wrong. Do NOT raise n_steps to make it go away.
    """
    dur = seconds or scene.duration
    tot = np.zeros(len(END_NAMES), dtype=np.int64)
    worst, t_worst = 0.0, 0.0
    hdr = "".join(f"{n:>8}" for n in END_NAMES)
    print(f"  {'t':>6} {'lines':>6}{hdr}")
    for k in range(samples):
        t = dur * k / (samples - 1.0)
        _, length, ended, _, _, _ = solve(scene.spec(t))
        c = np.array([(ended == e).sum() for e in range(len(END_NAMES))], dtype=np.int64)
        tot += c
        frac = c[END_BUDGET] / max(1, len(ended))
        if frac > worst:
            worst, t_worst = frac, t
        if k % max(1, samples // 12) == 0:
            print(f"  {t:6.2f} {len(ended):6d}" + "".join(f"{v:8d}" for v in c))
    n = max(1, tot.sum())
    print("  TOTAL " + "  ".join(f"{nm} {v/n:5.1%}" for nm, v in zip(END_NAMES, tot)))
    print(f"  worst BUDGET fraction {worst:.1%} at t={t_worst:.2f} s"
          + ("   <-- FIX THIS" if worst > 0.02 else "   OK"))
    return tot


def render_stills(scene, cfg, times, prefix=None):
    fr = D.Frame(cfg)
    os.makedirs(OUT, exist_ok=True)
    prefix = prefix or scene.name
    for t in times:
        rgb, _ = compose(fr, scene.spec(t), t)
        p = os.path.join(OUT, f"{prefix}_t{t:05.2f}.png")
        Image.fromarray(rgb).save(p)
        print("[still]", p)


def render_scene(scene, cfg, out=None, seconds=None):
    dur = seconds or scene.duration
    n = int(round(dur * cfg.fps))
    out = out or os.path.join(OUT, f"{scene.name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part.mp4"

    cmd = [find_ffmpeg(), "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "-",
           *encoder_args(), "-pix_fmt", "yuv420p", "-movflags", "+faststart", part]
    print(f"[render] {scene.name}  {cfg.width}x{cfg.height} @{cfg.fps}  "
          f"{dur:.1f}s  {n} frames -> {os.path.basename(out)}")
    fr = D.Frame(cfg)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for i in range(n):
            t = i / cfg.fps
            rgb, _ = compose(fr, scene.spec(t), t)
            proc.stdin.write(rgb.tobytes())
            if i and i % 60 == 0:
                el = time.time() - t0
                sys.stdout.write(f"\r  {i}/{n}  {el/i*1000:.0f} ms/frame  "
                                 f"eta {el/i*(n-i):5.0f}s")
                sys.stdout.flush()
        proc.stdin.close()
        rc = proc.wait()
    except BaseException:
        try:
            proc.stdin.close()
            proc.kill()
        finally:
            if os.path.exists(part):
                os.remove(part)
        raise
    print()
    if rc != 0:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError(f"ffmpeg exited {rc}")
    if os.path.exists(out):
        os.remove(out)
    os.replace(part, out)                     # atomic: a final name is always playable
    el = time.time() - t0
    print(f"[done] {out}   {el:.0f}s  ({el/n*1000:.0f} ms/frame)")
    return out
