"""render.py - the dry report, stills, and the ffmpeg pipe.

Frames are driven by REAL SECONDS (t = i/fps), never by frame index, so a 540x960 @30 preview
and a 1080x1920 @60 final are the same animation and a preview is trustworthy. Library-wide.

Encoding is atomic: raw RGB24 goes to `<out>.mp4.part.mp4` and is renamed onto its real name
only on success, so a file bearing its final name is always playable.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .config import OUT, encoder_args, find_ffmpeg
from .content import EMPTY
from .draw import Frame
from .rgbdelay import ChannelDelay


def dry_report(scene, cfg, samples=31):
    """Measure the INTENT, not just that it did not crash.

    For this format the intent is a DENSITY BAND. Too few lit elements and the frame is an
    empty grid; too many and the bloom integrates to a flat wash with no contrast anywhere
    (Field_Lines mistake 3, and it is the same failure here). This counts what would actually
    be drawn, per frame, without drawing it - about 0.3 s for a whole scene.
    """
    st = scene.content.stats()
    print(f"  lattice   {scene.lat}")
    print(f"  atlas     {scene.atlas}")
    print(f"  content   node_occ {st['node_occ']:.1%}  edge_on {st['edge_on']:.1%}  "
          f"ever_lit {st['ever_lit']:.1%}  beat {st['beat_s']:.3f}s x{st['beats']}  "
          f"per-beat glyph churn {st['beat_churn']:.1%}")
    print(f"  {'t':>6} {'env':>6} {'nodes':>7} {'links':>7} {'boxes':>6} {'total':>7} "
          f"{'arm%':>6}")
    tot, peak, peak_t = [], 0, 0.0
    box_mul = getattr(scene, "box_mul", 0.0)
    for k in range(samples):
        t = scene.duration * k / (samples - 1.0)
        env, ls, le = scene.fields(t)[:3]        # eye scenes return extra fields
        n = nl = nb = 0
        for node, edge, boxes, w in scene.content.blend(t):
            cut = 0.012 / max(scene.gain * env * w, 1e-9)
            n += int(((node != EMPTY) & (ls > cut)).sum())
            nl += int((edge & (le > cut / max(scene.link_mul, 1e-9))).sum())
            bi = np.concatenate(boxes) if boxes else np.zeros(0, np.int64)
            if len(bi) and box_mul > 1e-9:
                nb += int((le[bi] > cut / box_mul).sum())
        total = n + nl + nb
        tot.append(total)
        if total > peak:
            peak, peak_t = total, t
        if k % max(1, samples // 10) == 0:
            # `arm%` is the fraction of sites inside a genuinely lit region, NOT merely above
            # the ambient floor - `ambient` alone puts every site above any small threshold,
            # which made the old cut report 100% on every scene and measured nothing.
            print(f"  {t:6.2f} {env:6.2f} {n:7d} {nl:7d} {nb:6d} {total:7d} "
                  f"{float((ls > 0.35).mean()):6.1%}")
    tot = np.array(tot)
    print(f"  TOTAL  mean {tot.mean():.0f}  median {np.median(tot):.0f}  "
          f"peak {peak} at t={peak_t:.2f}s  min {tot.min()}")
    lo, hi = 400, 12000
    if tot.mean() < lo:
        print(f"  <-- TOO SPARSE (mean {tot.mean():.0f} < {lo}); raise node_p/edge_p or "
              f"lower illum_lo")
    elif peak > hi:
        print(f"  <-- TOO DENSE (peak {peak} > {hi}); the bloom will wash out. Lower "
              f"node_p/edge_p or raise illum_lo")
    else:
        print(f"  density band OK ({lo}..{hi})")
    return tot


def render_stills(scene, cfg, times, prefix=None):
    fr = Frame(cfg, scene.atlas_for(cfg))
    os.makedirs(OUT, exist_ok=True)
    prefix = prefix or scene.name
    for t in times:
        rgb = scene.render_frame(fr, t)
        p = os.path.join(OUT, f"{prefix}_t{t:05.2f}.png")
        Image.fromarray(rgb).save(p)
        print("[still]", p, f"  splatted {fr.splatted}")


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
    fr = Frame(cfg, scene.atlas_for(cfg))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    delay = None
    if getattr(scene, "mono", False):
        delay = ChannelDelay(cfg.fps, getattr(scene, "rgb_delay", None))
        # PRE-ROLL so the buffer is already full at frame 0. Without it the first ~0.2 s
        # comes out greyscale and colour "switches on", which reads as a mistake. The
        # envelope is ~0 that far back, so these frames are nearly black and cost little.
        for k in range(delay.n - 1, 0, -1):
            delay.push(scene.render_mono(fr, -k / cfg.fps))
    try:
        for i in range(n):
            if delay is not None:
                rgb = fr.finish(delay.push(scene.render_mono(fr, i / cfg.fps)),
                                frame_index=i)
            else:
                rgb = scene.render_frame(fr, i / cfg.fps)
            proc.stdin.write(rgb.tobytes())
            if i and i % 30 == 0:
                el = time.time() - t0
                sys.stdout.write(f"\r  {i}/{n}  {el/i*1000:.0f} ms/frame  "
                                 f"eta {el/i*(n-i):5.0f}s   ")
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
    os.replace(part, out)                     # atomic: a final name is always finalized
    el = time.time() - t0
    print(f"[done] {out}   {el:.0f}s  ({el/n*1000:.0f} ms/frame)")
    return out
