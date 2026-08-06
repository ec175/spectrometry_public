"""drug_profile.py — render a CRT-scope "drug profile" short (molecule above, spectrum below).

manim-free remake of the vertical drug profiles, drawn as an oscilloscope screen and finished
with crtfilm.FilmLook (the "filmed off a CRT" screen filter) + the `hers` bow-out + ambilight.

  .\.venv\Scripts\python.exe drug_profile.py Glycine
  .\.venv\Scripts\python.exe drug_profile.py Alanine --preview      # 540x960 @15 quick look
  .\.venv\Scripts\python.exe drug_profile.py all                    # Glycine+Alanine+Valine

GPU (CuPy + NVENC) is the project default; OSC_CUPY=0 / OSC_NVENC=0 force CPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from osc.config import OUT, encoder_args, find_ffmpeg
from osc.crtfilm import FilmLook, film_config_for_phosphor
from osc.drugscope import DrugScope
from drug_data import MOLECULES


def render_one(name, *, preview=False, seconds=None, fps=30, bow=0.05, glow=1.2,
               ambient=0.18, seed=7, bevel=0.52, bevel_frac=0.05,
               frame_lo=None, frame_hi=None, warm=24, out_path=None):
    """Render `name`. If frame_lo/frame_hi are given, render ONLY that absolute frame range
    (one parallel lane / chunk of the whole video): the phosphor persistence buffer is first
    warmed up over `warm` frames before frame_lo (decay 0.8/frame -> ~gone in 20), so the chunk
    seams cleanly onto its neighbours. FilmLook is built for the FULL frame count with a fixed
    seed and indexed by ABSOLUTE frame i, so its temporal tracks (shake/flicker/hum/noise) are
    identical across lanes -> no seam."""
    mol = MOLECULES[name]
    w, h = (540, 960) if preview else (1080, 1920)
    fps = 15 if preview else fps
    dur = seconds if seconds is not None else 15.0        # halved 2026-07-22 (transition stays mid)
    n = int(round(dur * fps))
    lo = 0 if frame_lo is None else max(0, frame_lo)
    hi = n if frame_hi is None else min(n, frame_hi)

    scope = DrugScope(mol, w=w, h=h, duration=dur)
    # multi-colour content (green trace + blue/red atoms) -> neutral-white phosphor film config
    cfg = film_config_for_phosphor((255, 255, 255), halation_boost=glow)
    look = FilmLook(w, h, fps, n, seed=seed, ambient_gain=ambient, barrel_k=bow,
                    bevel=bevel, bevel_frac=bevel_frac, **cfg)

    if out_path is None:
        out_path = os.path.join(OUT, f"drug_scope_{name.lower()}{'_preview' if preview else ''}.mp4")
    os.makedirs(os.path.dirname(out_path) or OUT, exist_ok=True)
    tmp = out_path + ".part.mp4"
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-an", *encoder_args(), "-pix_fmt", "yuv420p", tmp]
    print(f"[{name}] {w}x{h}@{fps}  frames {lo}..{hi} of {n}  -> {out_path}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(max(0, lo - warm), lo):            # warm the phosphor (not written)
            scope.render_frame(i, fps)
        for i in range(lo, hi):
            frame = scope.render_frame(i, fps)
            frame = look.process(frame, i)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if i % fps == 0:
                print(f"  frame {i}/{hi}", flush=True)
        proc.stdin.close(); proc.wait()
    except BaseException:
        try:
            proc.kill(); proc.wait()
        except Exception:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if proc.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    os.replace(tmp, out_path)
    print("done:", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="molecule name (Morphine, Indigo, ...) or 'all'")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bow", type=float, default=0.05, help="glass bow-out (subtle main-face bow)")
    ap.add_argument("--bevel", type=float, default=0.42, help="beveled edge ridge amp (0 = off)")
    ap.add_argument("--bevel-frac", type=float, default=0.05, help="bevel band width (frac of min side)")
    ap.add_argument("--glow", type=float, default=1.2)
    ap.add_argument("--ambient", type=float, default=0.18)
    ap.add_argument("--frames", default=None, help="render only absolute frames LO:HI (one lane)")
    ap.add_argument("--warm", type=int, default=24, help="phosphor warm-up frames before LO")
    ap.add_argument("--out", default=None, help="output path override (for a segment/lane)")
    args = ap.parse_args()

    frame_lo = frame_hi = None
    if args.frames:
        frame_lo, frame_hi = (int(x) for x in args.frames.split(":"))

    names = list(MOLECULES) if args.name.lower() == "all" else [args.name]
    for nm in names:
        match = next((k for k in MOLECULES if k.lower() == nm.lower()), None)
        if not match:
            raise SystemExit(f"unknown molecule {nm!r}; have {list(MOLECULES)}")
        render_one(match, preview=args.preview, seconds=args.seconds, fps=args.fps,
                   bow=args.bow, glow=args.glow, ambient=args.ambient,
                   bevel=args.bevel, bevel_frac=args.bevel_frac,
                   frame_lo=frame_lo, frame_hi=frame_hi, warm=args.warm, out_path=args.out)


if __name__ == "__main__":
    main()
