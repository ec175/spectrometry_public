# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py - Attractors: particle-swarm attractor shorts.

    python cli.py list
    python cli.py lorenz_swarm --still 3,8,14 --preview
    python cli.py lorenz_swarm --preview
    python cli.py lorenz_swarm                     # 1080x1920 @60 -> out\\<scene>.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at.draw import Canvas                      # noqa: E402
from at.scenes import SCENES                    # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
_NV = None


def find_ffmpeg():
    e = os.environ.get("AT_FFMPEG")
    if e and os.path.exists(e):
        return e
    for name in ("ffmpeg.exe", "ffmpeg"):
        b = os.path.join(ROOT, "bin", name)
        if os.path.exists(b):
            return b
    return "ffmpeg"


def encoder_args():
    global _NV
    if os.environ.get("AT_NVENC", "1") != "0":
        if _NV is None:
            try:
                r = subprocess.run([find_ffmpeg(), "-v", "error", "-f", "lavfi", "-i",
                                    "color=black:size=256x256:rate=30:duration=0.1",
                                    "-c:v", "h264_nvenc", "-f", "null", "-"],
                                   capture_output=True, timeout=25)
                _NV = r.returncode == 0
            except Exception:
                _NV = False
        if _NV:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
                    "-b:v", "0", "-maxrate", "40M", "-bufsize", "80M"]
    return ["-c:v", "libx264", "-crf", "16", "-preset", "medium"]


def run(name, w, h, fps, seconds, stills=None, out=None):
    sc = SCENES[name]()
    cv = Canvas(w, h, palette=sc.palette, trail=sc.trail,
                exposure=sc.exposure * (1.0 if w >= 1080 else 1.6))
    n = int(round(seconds * fps))
    os.makedirs(OUT, exist_ok=True)
    want = set(int(round(s * fps)) for s in (stills or []))

    proc = None
    if not stills:
        part = out + ".part.mp4"
        proc = subprocess.Popen(
            [find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *encoder_args(),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", part], stdin=subprocess.PIPE)
        print(f"[render] {name} {w}x{h}@{fps} {seconds:.0f}s {n} frames")
    t0 = time.time()
    try:
        for i in range(n):
            t = i / fps
            sc.step(1.0 / fps, t)
            cv.decay()
            if hasattr(sc, "ribbon"):
                P, z, u = sc.ribbon(t)
                cv.splat(P, u, z, gain=2.4)
            P, z, u = sc.points(t)
            cv.splat(P, u, z, gain=sc.point_gain)
            rgb = cv.to_rgb()
            if stills:
                if i in want:
                    Image.fromarray(rgb).save(os.path.join(OUT, f"{name}_t{t:05.1f}.png"))
                    print("[still]", t)
            else:
                proc.stdin.write(rgb.tobytes())
                if i and i % 30 == 0:
                    el = time.time() - t0
                    sys.stdout.write(f"\r  {i}/{n} {el/i*1000:.0f} ms/f "
                                     f"eta {el/i*(n-i):5.0f}s   ")
                    sys.stdout.flush()
        if proc:
            proc.stdin.close()
            rc = proc.wait()
    except BaseException:
        if proc:
            try:
                proc.stdin.close()
                proc.kill()
            finally:
                if os.path.exists(out + ".part.mp4"):
                    os.remove(out + ".part.mp4")
        raise
    if proc:
        print()
        if rc != 0:
            os.remove(out + ".part.mp4")
            raise RuntimeError(f"ffmpeg {rc}")
        if os.path.exists(out):
            os.remove(out)
        os.replace(out + ".part.mp4", out)
        print(f"[done] {out}  {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--horizontal", action="store_true",
                    help="16:9 1920x1080 for stock (default is the 9:16 short)")
    ap.add_argument("--still", default="")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    if a.scene == "list":
        for k, v in SCENES.items():
            print(f"  {k:20s} {(v.__doc__ or '').strip().splitlines()[0]}")
        return
    # `draw.py` scales by `0.5 * min(w, h) * 0.94` about the frame centre, so the projection is
    # ISOTROPIC either way - a 16:9 frame fits the attractor to its HEIGHT and lets the object's
    # own x-spread use the extra width. That is the honest conversion: stretching x to fill the
    # frame would distort a 3-D object whose shape is the entire point. Whether the side margins
    # read as composed or as dead frame is a question for a still, not for arithmetic.
    if a.horizontal:
        w, h = (960, 540) if a.preview else (1920, 1080)
    else:
        w, h = (540, 960) if a.preview else (1080, 1920)
    fps = 30 if a.preview else a.fps
    stills = [float(x) for x in a.still.split(",")] if a.still else None
    suffix = ("_preview" if a.preview else "") + ("_h" if a.horizontal else "")
    run(a.scene, w, h, fps, a.seconds, stills,
        os.path.join(OUT, f"{a.scene}{suffix}.mp4"))


if __name__ == "__main__":
    main()
