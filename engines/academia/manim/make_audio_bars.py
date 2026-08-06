"""
make_audio_bars.py — precompute an audio-reactive equalizer feature track for the
VerticalProfile top "light bar".

Decodes a clip of an mp3 (bundled ffmpeg) starting at --start seconds, then builds a
[n_frames, n_bars] array of band magnitudes in 0..1, one row per VIDEO frame at
--fps (default 30). The manim scene reads this by REAL time (seconds), so the render
fps is irrelevant — features stay synced at any quality.

The clip start here MUST match the -ss used when muxing the audio onto the final
video, so bars and sound line up (both start at the same point in the song).

Usage (run with the manim .venv python — needs numpy + scipy):
  .venv\\Scripts\\python.exe make_audio_bars.py "<song.mp3>" out.npy \
      --start 17 --dur 50 --bars 32 --fps 30
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.path.join(HERE, "bin", "ffmpeg.exe")


def decode_mono(path, start, dur, sr):
    """ffmpeg-decode a clip to a mono float32 numpy array at sample-rate sr."""
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def build_bars(samples, sr, fps, n_bars, n_frames=None, nfft=2048,
               f_lo=35.0, f_hi=12000.0, attack=1.0, decay=0.30, floor=0.06,
               lead_ms=0.0, spacing="log", tilt=True):
    """[n_frames, n_bars] band magnitudes in 0..1, with perceptual weighting,
    global normalisation, and per-band attack/decay so it reads like an EQ.

    Sync notes (fixes the perceived "bars lag the audio"):
      • attack=1.0  -> bars jump up INSTANTLY on transients (no rise lag).
      • lead_ms     -> the analysis window is sampled this many ms AHEAD of the
                       frame, so bars ANTICIPATE the sound, cancelling the small
                       residual delay from decay smoothing + AAC encoder priming.
    Band range (f_lo..f_hi) lets one call make a BASS track and another the rest."""
    hop = sr / fps
    lead = int(sr * lead_ms / 1000.0)
    half = nfft // 2
    if n_frames is None:
        n_frames = max(0, int((len(samples) - lead) / hop))
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    if spacing == "linear":
        edges = np.linspace(f_lo, f_hi, n_bars + 1)
    else:
        edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bars + 1)
    band_bins = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0]
                 for b in range(n_bars)]
    for b in range(n_bars):                       # nearest-bin fallback for empty bands
        if len(band_bins[b]) == 0:
            band_bins[b] = np.array([int(np.argmin(np.abs(freqs - edges[b])))])
    pweight = np.sqrt(np.clip(freqs, 1.0, None)) if tilt else np.ones_like(freqs)

    inst = np.zeros((n_frames, n_bars), dtype=np.float32)
    padded = np.pad(samples, (half, half + lead))
    for i in range(n_frames):
        c = int(i * hop) + half + lead            # look-ahead centre
        seg = padded[c - half:c + half] * win
        mag = np.abs(np.fft.rfft(seg)) * pweight
        for b in range(n_bars):
            inst[i, b] = mag[band_bins[b]].mean()

    # dB compress, then normalise on robust percentiles
    db = 20.0 * np.log10(inst + 1e-6)
    lo = np.percentile(db, 20.0)
    hi = np.percentile(db, 99.0)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # per-band attack/decay smoothing (instant rise by default, smooth fall)
    out = np.zeros_like(norm)
    prev = norm[0].copy() if n_frames else np.zeros(n_bars, np.float32)
    for i in range(n_frames):
        cur = norm[i]
        rate = np.where(cur > prev, attack, decay)
        prev = prev + rate * (cur - prev)
        out[i] = prev
    out = floor + (1.0 - floor) * out             # keep a minimum lit height
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=17.0)
    ap.add_argument("--dur", type=float, default=50.0)
    ap.add_argument("--bars", type=int, default=32)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--f-lo", type=float, default=35.0, dest="f_lo",
                    help="low edge of the analysed band (Hz) — e.g. 35 for a BASS track")
    ap.add_argument("--f-hi", type=float, default=12000.0, dest="f_hi",
                    help="high edge of the analysed band (Hz) — e.g. 350 for a BASS track")
    a = ap.parse_args()
    if not os.path.exists(FFMPEG):
        sys.exit(f"bundled ffmpeg not found at {FFMPEG}")
    samples = decode_mono(a.song, a.start, a.dur, a.sr)
    feat = build_bars(samples, a.sr, a.fps, a.bars, f_lo=a.f_lo, f_hi=a.f_hi)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, feat)
    print(f"saved {a.out}  shape={feat.shape}  ({feat.shape[0] / a.fps:.1f}s @ {a.fps}fps)")


if __name__ == "__main__":
    main()
