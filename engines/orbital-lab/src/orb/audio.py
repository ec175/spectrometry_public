# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""audio.py - the song's measured structure: onsets, BPM, and per-frame envelopes.

⚠ ONE DECODE. ONE CLOCK. This is the rule the whole module exists to enforce, and it is the
most likely reason music sync has failed in this library before.

`prepare()` decodes the source file to **one PCM WAV** in `audio\\`, and that WAV is then the
single audio source for all three consumers:

    the analysis here  ->  the marking viewer  ->  the final mux

The naive shape has three decoders and therefore three clocks: ffmpeg decodes for analysis,
the browser decodes the original with `decodeAudioData`, and the mux seeks the compressed file
again. A browser's MP3/AAC decode may or may not strip the encoder delay (~25 ms, which is one
and a half frames at 60), and `-ss` on a compressed stream can land on a different sample than
the analysis decode did. Each error is small, none of them cancel, and the result is a sync
that is two frames early EVERYWHERE - which reads as "it just doesn't feel locked" with
nothing specific to point at. PCM has no delay ambiguity, so one WAV makes the markers, the
onsets and the encoded track share one sample clock by construction.

⚠ NO NEW DEPENDENCY. This project has no venv of its own on purpose and runs on
`Stock_Video\\Particle_Fields\\.venv` (see `orb/_pf.py`); a `pip install librosa` there would
change the resolution of an engine already gated next door. numpy + scipy + ffmpeg is enough.
The onset detector is ported from `Spectrometry_MP4\\Elements\\tools\\audio_sync.py`, which
has been carrying four shipped films since 2026-08-07.

⚠ THE GRID IS NOT THE CLOCK. `bpm` is returned as a CROSS-CHECK on the onsets, never as the
thing to snap to. A BPM grid drifts against a real recording within a few bars; Elements'
standing rule is that every marker is a real measured onset. The one place the grid is
allowed to matter is telling you that an onset list is nonsense.
"""
from __future__ import annotations

import json
import os
import subprocess
import wave

import numpy as np

#: Analysis rate. 22050 is Elements' and it is not arbitrary: the top band only needs to
#: separate hats from body, and halving the rate halves the STFT cost for no measurable
#: change in onset times.
SR = 22050
#: 128 samples = 5.8 ms per flux frame, which is a third of a video frame at 60 fps. The hop
#: is the FLOOR on marker precision, so it has to be well under one frame or the snap in the
#: viewer quantises to something coarser than the thing it is snapping for.
HOP = 128
WIN = 1024

#: Playback / delivery rate for the WAV every consumer shares.
OUT_SR = 44100

#: Band edges, in Hz. `low` is the 808 and the kick, `high` is hats and transients, `mid` is
#: what is left - vocal and synth body.
BAND_LOW = 160.0
BAND_HIGH = 2000.0

#: The envelope's normalising percentile. NOT the max: one transient (a crash, a clipped
#: 808) would set the scale for the whole clip and leave everything else at a fraction of the
#: range. 97 keeps the loud half of the song near the top of the range while letting the few
#: hottest hits clip against 1.0.
#:
#: ⚠ AND IT IS ONE SCALAR OVER THE WHOLE SECTION, NEVER A ROLLING WINDOW. That is
#: OUTPUT_RULES' "derive the colour range from a stable reference, never the frame's own
#: peak" applied to audio - the "jelly" defect. A rolling normalisation makes a quiet passage
#: react exactly as hard as a drop, so the picture pumps continuously and the drop lands on
#: nothing.
ENV_PCTL = 97.0


def _ffmpeg() -> str:
    from ._pf import find_ffmpeg
    return find_ffmpeg()


# --------------------------------------------------------------------------------------
# the one decode
# --------------------------------------------------------------------------------------
def prepare(src: str, out_dir: str, slug: str) -> dict:
    """Decode `src` to ONE 16-bit PCM WAV at `OUT_SR`. Returns its path and true duration.

    The WHOLE file is decoded, not a section: the trim is a decision Clancy makes in the
    viewer by listening, so the viewer needs the whole song to explore. The section is
    applied later, ONCE, by sample-slicing this WAV - never by a second `-ss` on the mp3.
    """
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, f"{slug}.wav")
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", src,
           "-ac", "2", "-ar", str(OUT_SR), "-c:a", "pcm_s16le", wav]
    subprocess.run(cmd, check=True)
    with wave.open(wav, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
    return dict(wav=wav, sr=sr, frames=n, duration=n / float(sr))


def read_mono(wav: str, t0: float = 0.0, t1: float = None) -> np.ndarray:
    """Mono float32 at `SR`, sliced from the shared WAV BY SAMPLE INDEX.

    ⚠ Sliced, not re-seeked. `t0` becomes `int(t0 * sr)` against the same PCM every other
    consumer reads, so the analysis' idea of "12.517 s" is the viewer's idea of it to the
    sample. Decimation to `SR` is a plain stride after an anti-alias filter, so the time base
    is exact rather than resampled.
    """
    from scipy import signal
    with wave.open(wav, "rb") as w:
        sr, nch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        i0 = max(0, int(round(t0 * sr)))
        i1 = n if t1 is None else min(n, int(round(t1 * sr)))
        w.setpos(i0)
        raw = w.readframes(max(0, i1 - i0))
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    q = int(round(sr / float(SR)))
    if q > 1:
        x = signal.decimate(x, q, ftype="fir", zero_phase=True).astype(np.float32)
    return np.ascontiguousarray(x)


# --------------------------------------------------------------------------------------
# spectral flux, three bands
# --------------------------------------------------------------------------------------
def flux(x: np.ndarray):
    """(t, {band: flux}) - half-wave-rectified LOG-spectral flux.

    ⚠ LOG, not linear, and that is the difference between a detector that finds a hat under a
    loud 808 and one that only ever finds the 808. `log1p(mag * 100)` compresses the dynamic
    range so a 40 dB-quieter transient still produces a comparable RISE, which is what an
    onset actually is. A linear flux is dominated by whatever is loudest in the mix.
    """
    n = (len(x) - WIN) // HOP
    if n < 8:
        raise ValueError("audio too short to analyse")
    win = np.hanning(WIN).astype(np.float32)
    fr = np.lib.stride_tricks.as_strided(
        x, (n, WIN), (x.strides[0] * HOP, x.strides[0])) * win
    mag = np.abs(np.fft.rfft(fr, axis=1))
    logm = np.log1p(mag * 100.0)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    d = np.maximum(0.0, np.diff(logm, axis=0))
    idx = dict(
        low=np.where(freqs < BAND_LOW)[0],
        mid=np.where((freqs >= BAND_LOW) & (freqs <= BAND_HIGH))[0],
        high=np.where(freqs > BAND_HIGH)[0],
    )
    t = (np.arange(1, n) * HOP + WIN / 2.0) / SR
    out = {k: d[:, v].sum(1).astype(np.float32) for k, v in idx.items()}
    out["full"] = d.sum(1).astype(np.float32)
    return t.astype(np.float32), out


def pick(t: np.ndarray, f: np.ndarray, min_sep=0.09, k=1.2, win_s=0.45):
    """[(time, strength)] - local maxima above an ADAPTIVE threshold.

    ⚠ ADAPTIVE, because a fixed threshold picks the loud half of a song and nothing from the
    intro. The threshold is a local mean + k*sigma over a `win_s` window, so a quiet passage
    is judged against itself.

    ⚠ `min_sep` IS A REAL MUSICAL CONSTRAINT, not smoothing. Two flux frames 6 ms apart are
    one drum hit with a rebound, not two events, and letting both through puts two markers
    where a listener hears one. 90 ms is comfortably under a 16th note at any tempo this
    engine will see. When two candidates collide the STRONGER one wins - taking the first
    would systematically place every marker on the attack's leading edge rather than its peak.
    """
    fn = f / (f.max() + 1e-9)
    w = max(1, int(win_s / (HOP / float(SR))))
    pad = np.pad(fn, w, mode="edge")
    c = np.cumsum(pad, dtype=np.float64)
    c2 = np.cumsum(pad.astype(np.float64) ** 2)
    m = 2 * w
    mean = (c[m:] - c[:-m]) / m
    var = np.maximum(0.0, (c2[m:] - c2[:-m]) / m - mean ** 2)
    thr = (mean + k * np.sqrt(var))[:len(fn)]
    out = []
    for i in range(1, len(fn) - 1):
        if fn[i] > thr[i] and fn[i] >= fn[i - 1] and fn[i] >= fn[i + 1]:
            ti, si = float(t[i]), float(fn[i])
            if not out or ti - out[-1][0] >= min_sep:
                out.append((ti, si))
            elif si > out[-1][1]:
                out[-1] = (ti, si)
    return out


def bpm(f: np.ndarray, lo=60.0, hi=190.0) -> float:
    """Autocorrelation tempo, as a SANITY CHECK on the onsets and nothing more.

    Folded across the octave: a detector that locks to the half or double of the real tempo
    is the classic failure, and reporting 83 when the song is 166 would send someone looking
    for a bug in the onsets instead.
    """
    g = f - f.mean()
    ac = np.correlate(g, g, "full")[len(g):]
    fps = SR / float(HOP)
    best, out = 0.0, 0.0
    for lag in range(1, len(ac)):
        b = 60.0 * fps / lag
        if b < lo:
            break
        if b <= hi and ac[lag] > best:
            best, out = float(ac[lag]), b
    return round(out, 2)


# --------------------------------------------------------------------------------------
# the per-frame envelopes - what the renderer actually reads
# --------------------------------------------------------------------------------------
def envelopes(x: np.ndarray, duration: float, fps: int, attack=0.010, release=0.120):
    """{band: array of one value per VIDEO frame}, each in roughly [0, 1].

    ⚠ ONE VALUE PER FRAME, PRECOMPUTED, and that is what keeps the engine's contract intact.
    Orbital_Lab is closed-form in `t` with no integrator, so a backward seek is free and a
    still at any instant is exact. An envelope evaluated by a running filter AT RENDER TIME
    would quietly introduce history and break both - a still at t=9 would differ depending on
    whether the renderer had passed through t=8. As a precomputed array indexed by frame it
    stays a pure function of `t`, exactly as `fit_scales` already is.

    ⚠ ASYMMETRIC SMOOTHING, and the asymmetry is the whole point. A fast attack keeps the hit
    ON the transient - a symmetric filter delays the peak by half the window, which at 120 ms
    is seven frames late and is precisely the "feels behind" failure. The slow release is what
    makes a kick read as a body that recoils and settles rather than a one-frame twitch, and
    a decay is the one shape OUTPUT_RULES' "no sudden motion" will accept.
    """
    from scipy import signal
    nfr = int(round(duration * fps))
    out = {}
    bands = dict(low=(None, BAND_LOW), mid=(BAND_LOW, BAND_HIGH), high=(BAND_HIGH, None))
    for name, (f0, f1) in bands.items():
        y = x
        if f0:
            b, a = signal.butter(2, f0 / (SR / 2.0), "high")
            y = signal.lfilter(b, a, y)
        if f1 and f1 < SR / 2.0 * 0.99:
            b, a = signal.butter(2, f1 / (SR / 2.0), "low")
            y = signal.lfilter(b, a, y)
        # RMS over one video frame, then the asymmetric follower.
        per = max(1, int(round(SR / float(fps))))
        m = nfr * per
        yy = np.zeros(m, np.float32)
        yy[:min(m, len(y))] = y[:min(m, len(y))]
        r = np.sqrt((yy.reshape(nfr, per).astype(np.float64) ** 2).mean(axis=1))
        ka = 1.0 - np.exp(-1.0 / max(attack * fps, 1e-6))
        kr = 1.0 - np.exp(-1.0 / max(release * fps, 1e-6))
        e = np.zeros(nfr, np.float64)
        v = 0.0
        for i, ri in enumerate(r):
            v += (ri - v) * (ka if ri > v else kr)
            e[i] = v
        s = np.percentile(e, ENV_PCTL) if e.size else 1.0
        out[name] = np.clip(e / max(s, 1e-9), 0.0, 1.0).astype(np.float32)
    return out


def spectrogram(wav: str, png: str, width=2400, height=256) -> str:
    """One `showspectrumpic` PNG of the whole WAV, for the viewer's strip.

    ⚠ NOT DECORATION. Elements' hardest-won sync lesson is *"When a sync feels off,
    SPECTROGRAM FIRST, then onsets"* - its helium film's low-band picker locked onto a 13 s
    synth pad and every marker was wrong in a way the onset list looked fine about. A picture
    of the spectrum is the only cheap way to see that the detector and the music disagree.
    """
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", wav,
           "-lavfi", f"showspectrumpic=s={width}x{height}:legend=0:scale=log:color=intensity",
           "-frames:v", "1", png]
    subprocess.run(cmd, check=True)
    return png


def analyse(src: str, out_dir: str, slug: str, fps=60) -> dict:
    """The whole layer-1 pass. Writes the WAV, the spectrogram and `<slug>.analysis.json`."""
    prep = prepare(src, out_dir, slug)
    x = read_mono(prep["wav"])
    t, fx = flux(x)
    ons = {k: pick(t, v, min_sep=(0.13 if k == "low" else 0.09)) for k, v in fx.items()}
    env = envelopes(x, prep["duration"], fps)
    png = spectrogram(prep["wav"], os.path.join(out_dir, f"{slug}.spectrum.png"))
    doc = dict(
        slug=slug, source=os.path.abspath(src), wav=os.path.abspath(prep["wav"]),
        spectrum=os.path.abspath(png), duration=prep["duration"], fps=fps,
        sr_analysis=SR, hop=HOP, env_pctl=ENV_PCTL,
        bpm={k: bpm(v) for k, v in fx.items()},
        onsets={k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in ons.items()},
        envelopes={k: [round(float(z), 4) for z in v] for k, v in env.items()},
    )
    jp = os.path.join(out_dir, f"{slug}.analysis.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    doc["path"] = jp
    return doc
