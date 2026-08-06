"""Is the loop seam distinguishable from an ORDINARY frame step? Measured on the shipped MP4.

`loop_check.py` asks the absolute question - how far is frame n from frame 0 - which is the right
question while tuning. It is the wrong question for judging the result, because a viewer never
sees a still: at 60 fps the picture is already changing every frame, and the tracer dashes alone
move ~11 px per frame. A seam is invisible when the jump ACROSS it is no larger than the jumps
either side of it. So this compares:

    d_seam  = |last frame - first frame|          the join the viewer actually sees
    d_typ   = |frame k - frame k+1|               ordinary steps, sampled in the same quiet phase

and reports the ratio. Below ~1 the seam is literally a smaller change than a normal frame step
and cannot be seen. It also runs on the ENCODED file, so h264 quantisation is included rather
than assumed away.

    tools\\seam_check.py [out\\tri_foil_rates_loop.mp4]
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wt.config import find_ffmpeg                    # noqa: E402

mp4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "tri_foil_rates_loop.mp4")
work = os.path.join(ROOT, "scratch", "seam")
os.makedirs(work, exist_ok=True)
for f in os.listdir(work):
    os.remove(os.path.join(work, f))

# Decode to raw frames once; a 10 s 1080x1920 clip is small enough and this avoids any
# seek/keyframe ambiguity about WHICH frame we are comparing.
ff = find_ffmpeg()
proc = subprocess.run([ff, "-v", "error", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                      capture_output=True)
if proc.returncode != 0:
    raise SystemExit(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:400]}")
buf = np.frombuffer(proc.stdout, np.uint8)
W, H = 1080, 1920
n = buf.size // (W * H * 3)
frames = buf[:n * W * H * 3].reshape(n, H, W, 3)
print(f"{os.path.basename(mp4)}: {n} frames decoded at {W}x{H}")


def d(a, b):
    x = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return x.mean(), np.percentile(x, 99.9), x.max()


seam = d(frames[-1], frames[0])
print(f"\n  SEAM   frame {n-1} -> frame 0 : mean {seam[0]:7.3f}   p99.9 {seam[1]:6.1f} "
      f"  max {seam[2]:5.1f}")
print("\n  ordinary frame steps for comparison:")
rows = []
for k in (1, 60, 150, 300, 450, 540, 570, n - 2):
    if 0 <= k < n - 1:
        s = d(frames[k], frames[k + 1])
        rows.append(s)
        print(f"    frame {k:4d} -> {k+1:4d}      : mean {s[0]:7.3f}   p99.9 {s[1]:6.1f} "
              f"  max {s[2]:5.1f}")

quiet = [r for k, r in zip((1, 60, 150, 300, 450, 540, 570, n - 2), rows) if k >= 450]
qm = float(np.mean([r[0] for r in quiet]))
am = float(np.mean([r[0] for r in rows]))
print(f"\n  mean ordinary step, quiet phase : {qm:.3f}")
print(f"  mean ordinary step, whole clip  : {am:.3f}")
lo, hi = min(r[0] for r in rows), max(r[0] for r in rows)
print(f"  ordinary step RANGE             : {lo:.3f} .. {hi:.3f}")
print(f"\n  SEAM / ordinary-step (quiet)    : {seam[0]/max(qm,1e-9):.2f}x")
print(f"  SEAM / ordinary-step (whole)    : {seam[0]/max(am,1e-9):.2f}x")
# The RANGE is the honest bar, not the mean: ordinary steps vary by ~15% among themselves, so a
# seam inside that spread is by definition not distinguishable from one of them.
print("\n  verdict: " + (
    "SEAM IS WITHIN THE RANGE OF ORDINARY FRAME STEPS - not distinguishable"
    if seam[0] <= hi else
    f"seam is {seam[0]/hi:.2f}x the LARGEST ordinary step - inspect it"))

Image.fromarray(frames[0]).save(os.path.join(work, "first.png"))
Image.fromarray(frames[-1]).save(os.path.join(work, "last.png"))
amp = np.clip(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32)) * 6, 0, 255)
Image.fromarray(amp.astype(np.uint8)).save(os.path.join(work, "seam_diff6x.png"))
print(f"\n  wrote {work}\\first.png, last.png, seam_diff6x.png")
