"""render_optimal.py — render the whole chemical-scope set with maximally-utilised GPU lanes.

Strategy: decompose EVERY molecule into contiguous frame CHUNKS, put all chunks in one pool,
and let N GPU lanes (CuPy math + NVENC encode) WORK-STEAL from it. Because the unit of work is
a chunk (not a whole molecule), a lane that finishes early immediately grabs the next chunk of
ANY molecule — so no lane sits idle while work remains and the tail stays balanced (24 chunks /
3 lanes = 8 each, rebalanced live by stealing). Each molecule's chunks are concatenated (lossless
stream copy) the moment its last chunk lands.

Seamless chunk joins: every chunk warms the phosphor `WARM` frames before its start (afterglow
decays ~0.8/frame) and FilmLook is built for the full frame count with a fixed seed indexed by
ABSOLUTE frame, so shake/flicker/hum/noise line up. Chunk boundaries (300/600 of 900) sit in the
steady FTIR / Raman zones, clear of the mid-clip transition, so no seam lands on the scramble.

  .\.venv\Scripts\python.exe render_optimal.py                 # all molecules
  .\.venv\Scripts\python.exe render_optimal.py Morphine Indigo # a subset
  .\.venv\Scripts\python.exe render_optimal.py --lanes 3 --chunks 3 --seconds 30
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from osc.config import OUT, find_ffmpeg
from chemical_data import MOLECULES

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
GPU_ENV = {"OSC_CUPY": "1", "OSC_NVENC": "1"}       # every lane on the GPU (CuPy + NVENC)


def _chunks(n, c):
    """c contiguous frame ranges spanning [0, n)."""
    b = [round(n * j / c) for j in range(c + 1)]
    return [(b[j], b[j + 1]) for j in range(c) if b[j + 1] > b[j]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="molecules (default: all)")
    ap.add_argument("--lanes", type=int, default=3, help="concurrent GPU lanes (NVENC cap ~3)")
    ap.add_argument("--chunks", type=int, default=3, help="frame chunks per molecule")
    ap.add_argument("--seconds", type=float, default=15.0)   # halved 2026-07-22
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=24)
    args = ap.parse_args()

    req = args.names or list(MOLECULES)
    mols = []
    for nm in req:
        m = next((k for k in MOLECULES if k.lower() == nm.lower()), None)
        if not m:
            raise SystemExit(f"unknown molecule {nm!r}; have {list(MOLECULES)}")
        mols.append(m)

    n = int(round(args.seconds * args.fps))
    ranges = _chunks(n, args.chunks)
    seg_dir = os.path.join(OUT, "_segments")
    os.makedirs(seg_dir, exist_ok=True)
    ffmpeg = find_ffmpeg()

    # build the task pool. Heaviest molecules first (more heavy atoms -> slower per frame) so the
    # long poles start early -> a shorter, better-balanced tail (LPT scheduling heuristic).
    mols.sort(key=lambda m: -len(MOLECULES[m]["AT"]))
    tasks = []                                          # (mol, chunk_index, lo, hi)
    for m in mols:
        for ci, (lo, hi) in enumerate(ranges):
            tasks.append((m, ci, lo, hi))

    done = {m: 0 for m in mols}                          # chunks completed per molecule
    ok = {m: True for m in mols}
    finals, lock = [], threading.Lock()
    t0 = time.time()
    print(f"[optimal] {len(mols)} molecules x {len(ranges)} chunks = {len(tasks)} tasks, "
          f"{args.lanes} GPU lanes (work-stealing). molecules(heaviest first): {mols}")

    def seg_path(m, ci):
        return os.path.join(seg_dir, f"{m.lower()}_seg{ci}.mp4")

    def _concat(m):
        listf = os.path.join(seg_dir, f"{m.lower()}_list.txt")
        with open(listf, "w") as fh:
            for ci in range(len(ranges)):
                fh.write(f"file '{os.path.abspath(seg_path(m, ci))}'\n")
        out_path = os.path.join(OUT, f"chemical_scope_{m.lower()}.mp4")
        tmp = out_path + ".cat.mp4"
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                            "-i", listf, "-c", "copy", tmp])
        if r.returncode != 0:
            print(f"[optimal] CONCAT FAILED {m}"); return None
        os.replace(tmp, out_path)
        for ci in range(len(ranges)):
            try:
                os.remove(seg_path(m, ci))
            except OSError:
                pass
        os.remove(listf)
        return out_path

    def run(task):
        m, ci, lo, hi = task
        r = subprocess.run(
            [PY, os.path.join(HERE, "chemical_profile.py"), m,
             "--seconds", str(args.seconds), "--fps", str(args.fps),
             "--frames", f"{lo}:{hi}", "--warm", str(args.warm), "--out", seg_path(m, ci)],
            env={**os.environ, **GPU_ENV}, cwd=HERE)
        good = r.returncode == 0
        with lock:
            if not good:
                ok[m] = False
            done[m] += 1
            print(f"  chunk {m}[{lo}:{hi}] {'ok' if good else 'FAIL'}  "
                  f"({done[m]}/{len(ranges)} of {m}, +{time.time()-t0:.0f}s)", flush=True)
            complete = done[m] == len(ranges)
        if complete and ok[m]:                          # this molecule's last chunk -> concat now
            p = _concat(m)
            if p:
                with lock:
                    finals.append(p)
                print(f"[optimal] DONE {m} -> {p}", flush=True)
        return m, good

    with ThreadPoolExecutor(max_workers=args.lanes) as ex:
        list(ex.map(run, tasks))

    bad = [m for m in mols if not ok[m]]
    print(f"\n[optimal] complete in {time.time()-t0:.0f}s: {len(finals)} rendered"
          + (f", FAILED: {bad}" if bad else ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
