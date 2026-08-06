"""audio.py — decode a song to mono, precompute per-band magnitudes, and mux audio onto a
silent render. Pure numpy + ffmpeg (no manim). Ported from Ascii_Studio\\asciivid\\audio.py.

CRITICAL sync rule (inherited): the `start` used to build the bands MUST equal the `-ss` used
in `mux_audio`, so the trace and the audio begin at the same instant in the song.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .config import find_ffmpeg


def decode_mono(path, start, dur, sr, ffmpeg=None):
    """ffmpeg-decode a clip to a mono float32 numpy array at sample-rate sr."""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build_bands(samples, sr, fps, n_bands, n_frames=None, nfft=2048,
                f_lo=35.0, f_hi=12000.0, attack=1.0, decay=0.35, floor=0.0,
                spacing="log", tilt=True):
    """[n_frames, n_bands] band magnitudes in ~0..1, perceptual-weighted, globally normalised,
    with per-band attack/decay so it reads like an EQ. Ported from asciivid.audio.build_bars."""
    hop = sr / fps
    half = nfft // 2
    if n_frames is None:
        n_frames = max(1, int(len(samples) / hop))
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    if spacing == "linear":
        edges = np.linspace(f_lo, f_hi, n_bands + 1)
    else:
        edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bands + 1)
    band_bins = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(n_bands)]
    for b in range(n_bands):
        if len(band_bins[b]) == 0:
            band_bins[b] = np.array([int(np.argmin(np.abs(freqs - edges[b])))])
    pweight = np.sqrt(np.clip(freqs, 1.0, None)) if tilt else np.ones_like(freqs)

    inst = np.zeros((n_frames, n_bands), dtype=np.float32)
    padded = np.pad(samples, (half, half))
    for i in range(n_frames):
        c = int(i * hop) + half
        seg = padded[c - half:c + half] * win
        mag = np.abs(np.fft.rfft(seg)) * pweight
        for b in range(n_bands):
            inst[i, b] = mag[band_bins[b]].mean()

    db = 20.0 * np.log10(inst + 1e-6)
    lo = np.percentile(db, 20.0)
    hi = np.percentile(db, 99.0)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    out = np.zeros_like(norm)
    prev = norm[0].copy() if n_frames else np.zeros(n_bands, np.float32)
    for i in range(n_frames):
        cur = norm[i]
        rate = np.where(cur > prev, attack, decay)
        prev = prev + rate * (cur - prev)
        out[i] = prev
    out = floor + (1.0 - floor) * out
    return out.astype(np.float32)


def mux_audio(silent_video, song, out_path, start=0.0, fade_in=0.4, ffmpeg=None):
    """Mux the song onto a silent render with an afade-in, copy video (start == bands start!).
    Atomic: writes <out>.part.mp4 and renames on success, so out_path is never a torso."""
    ffmpeg = ffmpeg or find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".part.mp4"
    cmd = [ffmpeg, "-y", "-ss", str(start), "-i", song, "-i", silent_video,
           "-map", "1:v:0", "-map", "0:a:0", "-c:v", "copy",
           "-af", f"afade=t=in:st=0:d={fade_in}", "-c:a", "aac", "-b:a", "192k",
           "-shortest", tmp]
    try:
        subprocess.run(cmd, check=True)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, out_path)
    return out_path
