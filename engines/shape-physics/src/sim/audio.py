"""audio.py - EVENT-TRIGGERED progressive song playback, plus decode/mux.

`decode_*` and `mux_track` are ported from Oscilloscope\\osc\\audio.py (which was itself
ported from Ascii_Studio) - same ffmpeg invocation, same atomic <out>.part.mp4 -> rename.

The NEW piece is `Score`, and it inverts the library's usual relationship with audio. Every
other audio-reactive project here precomputes a band track from the song and lets the RENDER
read it (the "build start == mux -ss" sync rule). Here the SIMULATION decides when the song
plays: each event plays the next half-second of the song, so the song is revealed in order
across the clip.

THE OVERLAP RULE (the mistake this class exists to avoid). Videos of this kind usually
retrigger the sample on every event, so two events inside one interval stack two copies of
the audio a few frames apart - a flam, or a hard cut back to the sample start. Instead:

    event while nothing is playing  -> start a segment, play `slice` seconds
    event while a segment IS playing -> EXTEND that segment by another `slice`

An extended segment is one continuous read of the song, so there is no overlap and no abrupt
stop - the music simply keeps going for as long as events keep arriving. `Score` is causal
(it only ever looks at events already fired), which is what lets the renderer use the very
same object for the on-screen "audio is live" pulse and for the final audio track.
"""
from __future__ import annotations

import math
import os
import subprocess
import wave

import numpy as np

from .config import find_ffmpeg

SR = 44100


class Score:
    """Non-overlapping progressive playback schedule. Fire events in time order."""

    def __init__(self, slice_len: float = 0.5, song_start: float = 0.0):
        self.slice = float(slice_len)
        self.song_start = float(song_start)
        self.segments: list[list[float]] = []   # [[t_video, duration], ...]
        self.play_until = -1e9
        self.n_events = 0
        self.n_extended = 0

    def fire(self, t: float) -> None:
        self.n_events += 1
        if self.segments and t < self.play_until - 1e-9:
            self.segments[-1][1] += self.slice     # extend, do NOT restart
            self.play_until += self.slice
            self.n_extended += 1
        else:
            self.segments.append([float(t), self.slice])
            self.play_until = t + self.slice

    def active(self, t: float) -> bool:
        """Is audio sounding at video time t? (used for the on-screen pulse)"""
        for st, dur in reversed(self.segments):
            if st <= t < st + dur:
                return True
            if st <= t:
                return False
        return False

    def resolved(self) -> list[tuple[float, float, float]]:
        """[(t_video, t_song, duration), ...] - song offsets run CONSECUTIVELY, so the clip
        walks forward through the song no matter how the events clumped."""
        off = self.song_start
        out = []
        for st, dur in self.segments:
            out.append((st, off, dur))
            off += dur
        return out

    def song_seconds(self) -> float:
        return sum(d for _, d in self.segments)


# --------------------------------------------------------------------------- bounce blips
# Procedurally synthesised, no sample assets. The library precedent is Piano\audio.py (numpy
# additive synth); this is the same idea, much smaller: a short pitched tick with an
# exponential decay, pitched by BALL SIZE - smaller ball, higher pitch, like a real object.

BLIP_F_REF = 500.0      # Hz at r = R_REF
BLIP_R_REF = 26.0       # world px (the first ball's radius)
BLIP_EXP = 0.70         # f = F_REF * (R_REF / r) ** EXP


def blip_freq(r: float) -> float:
    return BLIP_F_REF * (BLIP_R_REF / max(r, 1e-3)) ** BLIP_EXP


def blip(freq: float, sr=SR, dur=0.055, attack=0.0018) -> np.ndarray:
    """One bounce tick: fundamental + a little 2nd harmonic under a fast exponential decay.
    The short attack ramp is not cosmetic - starting a sine at full amplitude is a step, and
    a step is a click on top of the tick you actually wanted."""
    n = max(4, int(dur * sr))
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t / (dur * 0.28)).astype(np.float32)
    na = max(1, int(attack * sr))
    env[:na] *= np.linspace(0.0, 1.0, na, dtype=np.float32)
    w = (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)).astype(np.float32)
    return w * env


def mix_bounces(out: np.ndarray, bounces, sr=SR, gain=0.13, duck=0.55, score=None,
                v_ref=520.0, pan=0.45) -> int:
    """Sum a blip per bounce into `out` ([n, 2] float32). Returns how many landed.

    - amplitude tracks IMPACT speed, so the track breathes with the action instead of
      metronoming; a hard hit is louder than a graze.
    - `duck` attenuates blips while the song is sounding. Past ~20 s the music is nearly
      continuous, and un-ducked ticks fight it.
    - blips are cached per quantised pitch (a semitone grid) - a 45 s clip has thousands of
      bounces but only a couple of dozen distinct radii.
    """
    cache: dict[int, np.ndarray] = {}
    n_out = len(out)
    placed = 0
    for ev in bounces:
        f = blip_freq(ev["r"])
        key = int(round(12.0 * math.log2(f / 55.0)))          # nearest semitone
        w = cache.get(key)
        if w is None:
            w = blip(55.0 * 2.0 ** (key / 12.0), sr=sr)
            cache[key] = w
        amp = gain * min(1.0, (ev["v"] / v_ref) ** 0.8)
        if score is not None and score.active(ev["t"]):
            amp *= duck
        i0 = int(ev["t"] * sr)
        i1 = min(i0 + len(w), n_out)
        if i1 <= i0:
            continue
        seg = w[: i1 - i0] * amp
        # gentle stereo placement from where the bounce happened
        u = max(-1.0, min(1.0, (ev.get("x", 540.0) - 540.0) / 540.0)) * pan
        out[i0:i1, 0] += seg * (1.0 - 0.5 * u)
        out[i0:i1, 1] += seg * (1.0 + 0.5 * u)
        placed += 1
    return placed


def mix_events(out: np.ndarray, events, sr=SR, gain=0.30, ratio=0.5, dur=0.20) -> int:
    """A distinct, longer, LOWER voice for the scene's headline event (escaped / scored /
    destroyed). An octave below the ball's own bounce pitch at `ratio` 0.5, so it reads as the
    same object doing something bigger rather than as an unrelated sound."""
    cache: dict[int, np.ndarray] = {}
    n_out = len(out)
    placed = 0
    for ev in events:
        f = blip_freq(ev.get("r", BLIP_R_REF)) * ratio
        key = int(round(12.0 * math.log2(max(f, 20.0) / 55.0)))
        w = cache.get(key)
        if w is None:
            w = blip(55.0 * 2.0 ** (key / 12.0), sr=sr, dur=dur)
            cache[key] = w
        i0 = int(ev["t"] * sr)
        i1 = min(i0 + len(w), n_out)
        if i1 <= i0:
            continue
        out[i0:i1, 0] += w[: i1 - i0] * gain
        out[i0:i1, 1] += w[: i1 - i0] * gain
        placed += 1
    return placed


def decode_stereo(path, start, dur, sr=SR, ffmpeg=None) -> np.ndarray:
    """ffmpeg-decode a clip to a float32 [n, 2] array at sample-rate sr."""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-ss", str(start), "-t", str(dur),
           "-i", path, "-ac", "2", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    return a[: (len(a) // 2) * 2].reshape(-1, 2).copy()


def build_track(score: Score, song: str, total_seconds: float, wav_path: str,
                sr=SR, fade=0.008, ffmpeg=None, bounces=None,
                bounce_gain=0.13, duck=0.55, events=None, event_gain=0.0) -> str | None:
    """Paste each scheduled segment of the song onto a silent bed of the clip's length, then
    sum in the bounce blips.

    A short (default 8 ms) fade sits at each segment's head and tail: the cut lands wherever
    the events fell, which is mid-waveform, and an un-faded cut clicks. The fade is inside
    the segment, so an EXTENDED segment stays one continuous read - the whole point of Score.
    """
    segs = score.resolved()
    if not segs and not bounces and not events:
        return None
    n_out = int(total_seconds * sr) + sr // 4
    out = np.zeros((n_out, 2), dtype=np.float32)

    buf = None
    if segs and song:
        need = score.song_seconds() + 0.25
        buf = decode_stereo(song, score.song_start, need, sr=sr, ffmpeg=ffmpeg)
        if len(buf) == 0:
            raise RuntimeError(f"decoded 0 samples from {song!r} at {score.song_start}s")
    nf = max(1, int(fade * sr))
    for t_video, t_song, dur in (segs if buf is not None else []):
        n = int(dur * sr)
        i0 = int((t_song - score.song_start) * sr)
        chunk = buf[i0:i0 + n]
        if len(chunk) < n:                       # ran off the end of the decoded window
            chunk = np.pad(chunk, ((0, n - len(chunk)), (0, 0)))
        env = np.ones(n, dtype=np.float32)
        k = min(nf, n // 2)
        env[:k] = np.linspace(0.0, 1.0, k, dtype=np.float32)
        env[n - k:] = np.linspace(1.0, 0.0, k, dtype=np.float32)
        j0 = int(t_video * sr)
        j1 = min(j0 + n, n_out)
        if j1 > j0:
            out[j0:j1] += chunk[: j1 - j0] * env[: j1 - j0, None]

    if bounces:
        mix_bounces(out, bounces, sr=sr, gain=bounce_gain, duck=duck, score=score)
    if events and event_gain > 0:
        mix_events(out, events, sr=sr, gain=event_gain)

    pcm = np.clip(out, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)) or ".", exist_ok=True)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return wav_path


def mux_track(silent_video: str, wav_path: str, out_path: str, ffmpeg=None) -> str:
    """Attach the built track to the silent render (video stream copied, no re-encode).
    Atomic: writes <out>.part.mp4 and renames on success, so out_path is never a torso."""
    ffmpeg = ffmpeg or find_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".part.mp4"
    cmd = [ffmpeg, "-y", "-i", silent_video, "-i", wav_path,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-shortest", tmp]
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
