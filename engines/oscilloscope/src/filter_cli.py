"""filter_cli.py — apply the crtfilm "filmed off a real CRT" look to a rendered mp4.

  .\.venv\Scripts\python.exe filter_cli.py out\loop_braid_final.mp4 out\filmtest_after.mp4 --start 6 --dur 3
  # optional: --before out\filmtest_before.mp4  writes the same untouched clip for A/B

Decodes with ffmpeg to raw RGB, runs osc.crtfilm.FilmLook per frame, re-encodes (CRF 16).
Audio (if any) from the same window is copied onto the output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

from osc.config import encoder_args, find_ffmpeg
from osc.crtfilm import FilmLook


def probe(path, ffmpeg):
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = "ffprobe"
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-show_entries", "format=duration",
         "-of", "json", path], stdout=subprocess.PIPE, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return st["width"], st["height"], fps, float(j["format"]["duration"])


def main():
    ap = argparse.ArgumentParser(description="Film-look CRT filter for rendered scope mp4s.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None, help="seconds (default: to end)")
    ap.add_argument("--before", default=None, help="also write the untouched clip here (A/B)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--glitch", type=float, default=0.0,
                    help="error-prone acquisition level (0=off; ~0.8 subtle, ~1.8 heavy)")
    ap.add_argument("--glitch-seed", type=int, default=11)
    # The chemical-profile GLASS, exposed so any render can take it. These three are what make that
    # format read as a real CRT filmed by a real camera rather than as a green plot: a slight
    # bow, a tinted beveled edge that seats the glass into a bezel, and ambilight spilling the
    # screen's own colour into the surround. Defaults are OFF so existing calls are unchanged.
    ap.add_argument("--bevel", type=float, default=None)
    ap.add_argument("--bevel-frac", type=float, default=0.05)
    ap.add_argument("--ambient", type=float, default=None)
    ap.add_argument("--bow", type=float, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")
    ffmpeg = find_ffmpeg()
    w, h, fps, total = probe(args.input, ffmpeg)
    dur = args.dur if args.dur is not None else max(0.0, total - args.start)
    n = int(round(dur * fps))
    print(f"{args.input}: {w}x{h} @ {fps:g}fps, clip {args.start}s +{dur}s ({n} frames)")

    if args.before:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", str(args.start), "-t", str(dur),
                        "-i", args.input, "-c:v", "libx264", "-crf", "16", "-pix_fmt",
                        "yuv420p", "-an", args.before], check=True)
        print("before clip:", args.before)

    tmp = args.output + ".part.mp4"                # atomic: publish only a finalized file
    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-ss", str(args.start), "-t", str(dur), "-i", args.input,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
         "-ss", str(args.start), "-t", str(dur), "-i", args.input,
         "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
         *encoder_args(), "-pix_fmt", "yuv420p", "-shortest", tmp],
        stdin=subprocess.PIPE)

    fk = {}
    if args.bevel is not None:
        fk.update(bevel=args.bevel, bevel_frac=args.bevel_frac)
    if args.ambient is not None:
        fk.update(ambient_gain=args.ambient)
    if args.bow is not None:
        fk.update(barrel_k=args.bow)
    look = FilmLook(w, h, fps, n, seed=args.seed, **fk)
    glitch = None
    if args.glitch > 0:
        from osc.glitchfx import GlitchFX
        glitch = GlitchFX(w, h, fps, n, seed=args.glitch_seed, level=args.glitch)
    fsize = w * h * 3
    i = 0
    try:
        while True:
            raw = dec.stdout.read(fsize)
            if len(raw) < fsize:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            j = min(i, n - 1)
            if glitch:                   # signal faults happen BEFORE the camera films them
                frame = glitch.pre(frame, j)
            frame = look.process(frame, j)
            if glitch:                   # sensor defects happen IN the camera
                frame = glitch.post(frame, j)
            enc.stdin.write(frame.tobytes())
            if i % 30 == 0:
                print(f"  frame {i}/{n}", flush=True)
            i += 1
        dec.stdout.close()
        enc.stdin.close()
        dec.wait(); enc.wait()
    except BaseException:                # incl. KeyboardInterrupt: never leave a .part
        for p in (dec, enc):
            try:
                p.kill(); p.wait()
            except Exception:
                pass
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if enc.returncode != 0 or i < n - 2:     # tolerate ±2 frames of duration rounding
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"encode failed (exit {enc.returncode}, {i}/{n} frames)")
    os.replace(tmp, args.output)
    print("done:", args.output)


if __name__ == "__main__":
    main()
