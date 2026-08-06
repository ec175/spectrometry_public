"""ascii_video.py — video -> coloured ASCII -> the Oscilloscope CRT "screen" filter.

Decodes an input clip, turns every frame into coloured ASCII (osc.asciivid, ported from the
Ascii_Studio album-icon mapping), runs the result through crtfilm.FilmLook (the "filmed off a
real CRT" SCREEN filter — NOT glitchfx), re-encodes, and muxes the source audio window back on.

  .\.venv\Scripts\python.exe ascii_video.py IN.mp4 out\ascii\hers_ascii.mp4 --start 45 --dur 30
  # quick look:  --preview   (540x960, ~6 s)     raw ascii, no CRT:  --no-film

The full-colour ASCII feeds FilmLook with a NEUTRAL white-phosphor config
(film_config_for_phosphor((255,255,255))) so every hue halates/blooms correctly — the green
phosphor bleed is only right for a single-colour trace. GPU (CuPy+NVENC) is default per the
project policy; OSC_CUPY=0 / OSC_NVENC=0 force CPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from osc.config import encoder_args, find_ffmpeg
from osc.crtfilm import FilmLook, film_config_for_phosphor
from osc.asciivid import AsciiVideo
from filter_cli import probe


def main():
    ap = argparse.ArgumentParser(description="video -> ASCII -> CRT screen filter")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0, help="source in-point (s)")
    ap.add_argument("--dur", type=float, default=None, help="clip length (s; default: to end)")
    ap.add_argument("--seconds", type=float, default=None, help="cap dur (quick test)")
    ap.add_argument("--out-w", type=int, default=1080)
    ap.add_argument("--out-h", type=int, default=1920)
    ap.add_argument("--font-px", type=int, default=16, help="ASCII cell height (smaller = more detail)")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source)")
    ap.add_argument("--glow", type=float, default=1.15, help="CRT halation boost")
    ap.add_argument("--ambient", type=float, default=0.18,
                    help="ambilight: content colour spilling into the surround (0 = white ambient)")
    ap.add_argument("--bow", type=float, default=0.12,
                    help="barrel distortion: how far the glass bows OUT (0.03 = flat CRT, 0 = none)")
    ap.add_argument("--crop-frac", type=float, default=1.0 / 3.0,
                    help="centre slice of source WIDTH to use (0.333 = middle third; 1.0 = whole)")
    ap.add_argument("--hsqueeze", type=float, default=0.8,
                    help="horizontal compression of the slice (0.8 = squeeze 20%%)")
    ap.add_argument("--margin", type=float, default=0.04, help="terminal-border gap (frac of frame)")
    ap.add_argument("--no-film", action="store_true", help="skip the CRT screen filter (raw ASCII)")
    ap.add_argument("--preview", action="store_true", help="540x960 smoke test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--track", action="store_true",
                    help="pan the crop to follow the subject (osc.track) instead of a fixed "
                         "centre slice - the fixed slice renders an empty wall whenever the "
                         "subject walks out of the middle third")
    ap.add_argument("--track-slew", type=float, default=0.32,
                    help="max pan speed, fraction of source width per second")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")
    ffmpeg = find_ffmpeg()
    sw, sh, sfps, total = probe(args.input, ffmpeg)
    fps = args.fps or sfps
    dur = args.dur if args.dur is not None else max(0.0, total - args.start)
    if args.seconds is not None:
        dur = min(dur, args.seconds)
    n = int(round(dur * fps))

    out_w, out_h = args.out_w, args.out_h
    font_px = args.font_px
    if args.preview:
        out_w, out_h, fps = 540, 960, min(fps, 15.0)
        font_px = max(6, args.font_px // 2)
        n = int(round(dur * fps))

    av = AsciiVideo(out_w, out_h, sw, sh, font_px=font_px,
                    crop_frac=args.crop_frac, hsqueeze=args.hsqueeze, margin=args.margin)
    print(f"{args.input}: src {sw}x{sh}@{sfps:g} -> out {out_w}x{out_h}@{fps:g}, "
          f"ASCII {av.cols}x{av.rows} (cell {av.cw}x{av.ch}), {n} frames, "
          f"clip {args.start}s +{dur:g}s, film={'off' if args.no_film else 'on'}")

    pan = None
    if args.track:
        from osc.track import subject_pan
        print("  [track] measuring subject position...", flush=True)
        pan = subject_pan(args.input, args.start, dur, fps=min(fps, 30.0), n_frames=n,
                          crop_frac=args.crop_frac, ffmpeg=ffmpeg,
                          max_px_per_s=args.track_slew)
        print(f"  [track] pan {pan.min():.3f}..{pan.max():.3f} "
              f"(0.5 = the old fixed centre crop), "
              f"total travel {np.abs(np.diff(pan)).sum():.2f} frame-widths", flush=True)

    look = None
    if not args.no_film:
        cfg = film_config_for_phosphor((255, 255, 255), halation_boost=args.glow)
        look = FilmLook(out_w, out_h, fps, n, seed=args.seed, ambient_gain=args.ambient,
                        barrel_k=args.bow, **cfg)

    tmp = args.output + ".part.mp4"
    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-ss", str(args.start), "-t", str(dur), "-i", args.input,
         "-r", f"{fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", f"{fps}", "-i", "pipe:0",
         "-ss", str(args.start), "-t", str(dur), "-i", args.input,
         "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k",
         *encoder_args(), "-pix_fmt", "yuv420p", "-shortest", tmp],
        stdin=subprocess.PIPE)

    fsize = sw * sh * 3
    i = 0
    try:
        while i < n:
            raw = dec.stdout.read(fsize)
            if len(raw) < fsize:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(sh, sw, 3)
            out = av.render(frame, None if pan is None else float(pan[i]))
            if look is not None:
                out = look.process(out, min(i, n - 1))
            enc.stdin.write(out.tobytes())
            if i % 30 == 0:
                print(f"  frame {i}/{n}", flush=True)
            i += 1
        dec.stdout.close()
        enc.stdin.close()
        dec.wait(); enc.wait()
    except BaseException:
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
    if enc.returncode != 0 or i < n - 2:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"encode failed (exit {enc.returncode}, {i}/{n} frames)")
    os.replace(tmp, args.output)
    print("done:", args.output)


if __name__ == "__main__":
    main()
