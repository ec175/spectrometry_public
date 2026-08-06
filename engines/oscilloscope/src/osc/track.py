"""track.py - where is the subject, per frame?

`AsciiVideo` takes a slice of the source WIDTH to fit a 16:9 clip into a 9:16 screen. A fixed
centre slice is only correct while the subject stays centred; on a real music video the
performer is off to one side constantly, and the ASCII then spends those seconds rendering an
empty wall while the interesting part of the frame is cropped away.

Two decisions here are load-bearing, and the first build got both wrong.

1. ASK "WHICH WINDOW HOLDS THE MOST ACTION", NOT "WHERE IS THE MEAN".
   The obvious measure - the centroid of the saliency map - barely moves. Energy is spread
   broadly and roughly symmetrically across a frame, so its mean sits near the middle almost
   regardless of content: measured 0.491..0.504 over a 6 s test, i.e. 0.02 frame-widths of
   total travel, which is the fixed centre crop with extra steps. Sliding the ACTUAL crop
   window and scoring what it encloses answers the question being asked, and moves decisively.

2. FRAME PER SHOT. DO NOT PAN ACROSS A CUT.
   A single smoothed track sounds right and is wrong for cut-driven material. This video cuts
   every 1-2 s; a pan that eases toward each new subject spends the whole shot catching up and
   never frames anything. Cuts are detected and the pan is allowed to JUMP on them - there is no
   visual continuity across a cut to protect - while WITHIN a shot it is held near-constant at
   that shot's robust best window. The result reads as a well-framed edit rather than as a
   camera drifting after the action.

SALIENCY = faces + motion + detail + a weak centre prior.
  faces   Haar frontal + profile detection. THIS IS THE ONE THAT MATTERS, and it was missing
          from the first working build. Motion and detail alone framed the WALLPAPER: this
          video's high-contrast concentric-ring backdrops out-score a person on both terms, so
          on a checked sample of six the tracker put its window on the rings while the dancer
          stood outside it - wrong on roughly half the frames. A face is what the shot is
          about, and it outweighs everything else when found.
  motion  frame-to-frame absolute difference. Carries the shots with no visible face.
  detail  gradient magnitude. Finds the subject when the shot is static and motion is ~0;
          without it a locked-off frame yields a flat map and the estimate collapses to centre.
  prior   broad gaussian at frame centre, so a featureless shot degrades gracefully to the old
          behaviour instead of to a random edge.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np


def _decode_small(path, start, dur, w, h, fps, ffmpeg="ffmpeg"):
    cmd = [ffmpeg, "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", path]
    if dur:
        cmd += ["-t", f"{dur:.4f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


def _median1d(x, k):
    if k < 3 or len(x) < k:
        return x
    k |= 1
    pad = np.pad(x, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=-1)


def _face_columns(G, w, every=3, min_frac=0.055):
    """-> (n, w) column energy from detected faces, 0 where none found.

    Runs on a 2x upscale of the tracking decode (Haar needs a bit more than 192x108 to fire),
    every `every` frames, and holds the last hit for the gap - faces do not move far in 0.1 s
    and detection is the expensive term here.
    """
    try:
        import cv2
    except Exception:
        return None
    hc = cv2.data.haarcascades
    fronts = [cv2.CascadeClassifier(os.path.join(hc, f)) for f in
              ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]
    fronts = [c for c in fronts if not c.empty()]
    if not fronts:
        return None

    n = len(G)
    out = np.zeros((n, w), np.float32)
    xs = (np.arange(w) + 0.5) / w
    last = None
    for i in range(n):
        if i % every == 0:
            g = np.clip(G[i], 0, 255).astype(np.uint8)
            g = cv2.resize(g, (w * 2, G.shape[1] * 2), interpolation=cv2.INTER_LINEAR)
            g = cv2.equalizeHist(g)
            hits = []
            for c in fronts:
                for (x, y, fw, fh) in c.detectMultiScale(g, 1.15, 5,
                                                         minSize=(int(w * 2 * 0.05),) * 2):
                    hits.append((x + fw * 0.5, fw))
            # a mirrored pass catches profiles facing the other way (Haar profile is one-sided)
            gm = cv2.flip(g, 1)
            for (x, y, fw, fh) in fronts[-1].detectMultiScale(gm, 1.15, 5,
                                                              minSize=(int(w * 2 * 0.05),) * 2):
                hits.append((g.shape[1] - (x + fw * 0.5), fw))
            last = hits
        if not last:
            continue
        for cx2, fw2 in last:
            cx = cx2 / (w * 2.0)
            sig = max(0.02, (fw2 / (w * 2.0)) * 0.75)
            out[i] += (fw2 / (w * 2.0)) * np.exp(-0.5 * ((xs - cx) / sig) ** 2)
    return out


def subject_pan(path, start=0.0, dur=None, fps=30.0, n_frames=None,
                crop_frac=1.0 / 3.0, ffmpeg="ffmpeg",
                w=192, h=108, motion_w=1.0, detail_w=0.45, prior_w=0.10,
                face_w=3.0, cut_thresh=3.2, temp=0.06, max_px_per_s=0.32, ease=0.30,
                min_shot=0.40):
    """-> (n_frames,) crop-centre x as a fraction of source width, in [half, 1-half].

    `max_px_per_s` limits drift WITHIN a shot (fraction of width per second).
    `ease` is the seconds allowed to settle after a cut.
    """
    G = _decode_small(path, start, dur, w, h, fps, ffmpeg)
    n = len(G)
    if n == 0:
        return np.full(max(1, n_frames or 1), 0.5)
    if n == 1:
        return np.full(max(1, n_frames or 1), 0.5)

    # ---- saliency -------------------------------------------------------------------
    mo = np.empty_like(G)
    mo[1:] = np.abs(np.diff(G, axis=0))
    mo[0] = mo[1]
    gx = np.zeros_like(G)
    gy = np.zeros_like(G)
    gx[:, :, 1:-1] = 0.5 * (G[:, :, 2:] - G[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (G[:, 2:, :] - G[:, :-2, :])
    det = np.hypot(gx, gy)

    def _nrm(a):
        return a / np.maximum(a.reshape(n, -1).max(axis=1).reshape(n, 1, 1), 1e-6)

    xs = (np.arange(w) + 0.5) / w
    prior = np.exp(-0.5 * ((xs - 0.5) / 0.34) ** 2)[None, None, :]
    col = (motion_w * _nrm(mo) + detail_w * _nrm(det) + prior_w * prior).sum(axis=1)  # (n,w)

    # TRIED AND REVERTED: subtracting a wide moving average from `col` ("score only what stands
    # out locally"). It sounds right and it measured WORSE - the ring backdrops are PERIODIC, so
    # they carry strong local contrast and a high-pass sharpens their edges instead of cancelling
    # them. On the six-frame check it pushed the worst case further onto the wallpaper (0.24 ->
    # 0.18) and nearly doubled total pan travel (13.5 -> 25.2 frame-widths, i.e. jitter). Broad
    # texture is only suppressible here by knowing what a SUBJECT is, which is the face term.
    col /= np.maximum(col.max(axis=1, keepdims=True), 1e-6)

    # Faces dominate wherever they are found. Normalising `col` first is what makes `face_w` a
    # meaningful ratio rather than a number that has to be re-tuned per clip.
    fc = _face_columns(G, w)
    if fc is not None and fc.any():
        fc /= np.maximum(fc.max(axis=1, keepdims=True), 1e-6)
        col = col + face_w * fc
        subject_pan.faces_found = float((fc.max(axis=1) > 0).mean())
    else:
        subject_pan.faces_found = 0.0

    # ---- best WINDOW per frame (soft-argmax over window positions) -------------------
    ww = max(1, int(round(crop_frac * w)))
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(col, axis=1)], axis=1)
    score = cs[:, ww:] - cs[:, :-ww]                       # (n, w-ww+1) enclosed energy
    centres = (np.arange(score.shape[1]) + 0.5 * ww) / w
    z = score / np.maximum(score.max(axis=1, keepdims=True), 1e-6)
    wgt = np.exp((z - 1.0) / max(temp, 1e-3))              # soft-argmax: smooth, no hard jumps
    raw = (wgt * centres[None, :]).sum(1) / np.maximum(wgt.sum(1), 1e-9)

    # ---- shot boundaries -------------------------------------------------------------
    fd = np.abs(np.diff(G.reshape(n, -1), axis=0)).mean(axis=1)
    fd = np.concatenate([[0.0], fd])

    # A CUT MUST ALSO BE A LOCAL PEAK, AND SHOTS HAVE A MINIMUM LENGTH.
    #
    # Thresholding the raw frame difference against a multiple of its median is only safe when
    # most of the clip is calm. In a strobing passage every frame clears the threshold, so every
    # frame becomes its own one-frame "shot" with its own framing, and the pan teleports on each
    # one. That is exactly what produced the shake reported at 11-12 s: measured median shot
    # length there was 0.03 s - a single frame - with 102 of 119 shots under half a second.
    #
    # Requiring a local maximum rejects a sustained high-difference plateau (a strobe) while
    # still catching a genuine cut, which is a spike. `min_shot` then enforces that no real edit
    # is shorter than ~0.4 s, collapsing whatever survives into one framing decision.
    cand = np.nonzero(fd > cut_thresh * (np.median(fd) + 1e-6))[0]
    gap = max(1, int(round(min_shot * fps)))
    cuts, last = [], -10 ** 9
    for c in cand:
        if not (0 < c < n):
            continue
        lo, hi = max(0, c - 2), min(n, c + 3)
        if fd[c] < fd[lo:hi].max():          # not a local peak -> a plateau, not a cut
            continue
        if c - last < gap:                   # too soon after the last accepted cut
            continue
        cuts.append(int(c))
        last = c
    bounds = sorted(set([0] + cuts + [n]))

    # ---- per-shot framing -------------------------------------------------------------
    out = np.empty(n)
    slew = max_px_per_s / fps
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= 0:
            continue
        seg = raw[a:b]
        target = float(np.median(seg))                     # robust framing for this shot
        if b - a <= 3:
            out[a:b] = target
            continue
        s = _median1d(seg, min(9, (b - a) | 1))
        # hold near the shot's framing; allow only slow drift for a subject that walks
        s = 0.72 * target + 0.28 * s
        for i in range(1, len(s)):
            s[i] = np.clip(s[i], s[i - 1] - slew, s[i - 1] + slew)
        # ease in from the previous shot's last value over `ease` seconds
        k = min(len(s), max(1, int(ease * fps)))
        if a > 0 and k > 1:
            u = np.linspace(0.0, 1.0, k)
            u = u * u * (3 - 2 * u)
            s[:k] = out[a - 1] * (1 - u) + s[:k] * u
        out[a:b] = s

    half = 0.5 * crop_frac
    out = np.clip(out, half, 1.0 - half)
    if n_frames and n_frames != n:
        out = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, n), out)
    return out
