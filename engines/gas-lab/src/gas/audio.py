# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""audio.py - ONE decode, ONE clock, and the firing schedule it implies.

Orbital_Lab paid for the first rule (S-MUSIC, rule 1): three decoders - analysis, a player,
and a second `-ss` on the mp3 at mux time - are three clocks, and the sync comes out uniformly
a frame or two late. So the mp3 is decoded EXACTLY once, to `audio\\<slug>.wav`, and that WAV
is the source for the analysis here, for the marking page, and for the final mux.

WHAT IT MEASURES, in order:
  1. where the music starts (this file carries 5.87 s of near-silence at its head)
  2. the beat: period and phase, by a comb filter over the broadband onset envelope
  3. THE GRID. The vocal hits in Flashing Lights do not sit on beats - they sit on a
     TRIPLET subdivision, one third of a beat (0.221 s) apart. Every measured gap between
     consecutive vocal onsets in the chosen window is 1, 2 or 3 of those units. So the grid,
     not the beat, is the clock the engine locks to.
  4. the vocal onsets, each labelled with its position on that grid (0, 1 or 2 of 3).

WHAT IT DECIDES: `plan()` picks the firing period as a whole number of grid units and the
phase that lands the most combustions on real vocal onsets. A whole number of units is the
point - it means ONE CONSTANT ENGINE SPEED hits the music, instead of a drive that lurches
between 0.17 s and 0.66 s per bang trying to chase every syllable.

No lyrics are transcribed anywhere in this module. It measures energy, not words; which word
is which is a LABEL, applied by `scratch\\mark_words.py` (or the default in `sync.py`).
"""
from __future__ import annotations

import json
import os
import subprocess
import wave

import numpy as np

from . import config as C

AUDIO_DIR = os.path.join(C.ROOT, "audio")
SR_A = 22050          # analysis rate
HOP, NFFT = 256, 2048


# ------------------------------------------------------------------ the one decode
def decode(src: str, slug: str, force=False) -> str:
    """mp3 -> audio\\<slug>.wav, 44.1 kHz stereo. THE only decode. Returns the wav path."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    out = os.path.join(AUDIO_DIR, slug + ".wav")
    if force or not os.path.exists(out):
        r = subprocess.run([C.find_ffmpeg(), "-y", "-v", "error", "-i", src,
                            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"decode failed: {r.stderr[:400]}")
    return out


def mono(wav_path: str, sr=SR_A):
    """The analysis signal, derived from the SAME wav the mux uses - never a second decode
    of the mp3."""
    with wave.open(wav_path, "rb") as w:
        n, ch, sw, fr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    assert sw == 2, f"expected 16-bit pcm, got {sw * 8}-bit"
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != fr:                       # integer decimation, good enough for onsets
        k = int(round(fr / sr))
        a = a[: (a.size // k) * k].reshape(-1, k).mean(axis=1)
        sr = fr / k
    return a, float(sr)


# ------------------------------------------------------------------ spectral frames
def _stft(x, sr):
    win = np.hanning(NFFT).astype(np.float32)
    nfr = max(0, (x.size - NFFT) // HOP)
    S = np.empty((nfr, NFFT // 2 + 1), np.float32)
    for i in range(nfr):
        S[i] = np.abs(np.fft.rfft(x[i * HOP:i * HOP + NFFT] * win))
    return S, np.fft.rfftfreq(NFFT, 1.0 / sr), np.arange(nfr) * HOP / sr, sr / HOP


def _flux(b, half=87):
    d = np.maximum(np.diff(b, prepend=b[0]), 0)
    idx = np.arange(0, d.size, 4)
    med = np.array([np.median(d[max(0, i - half):i + half + 1]) for i in idx])
    return d / (np.interp(np.arange(d.size), idx, med) + 1e-6)


def _peaks(f, thresh, min_gap, fps):
    gap, out, i = max(1, int(min_gap * fps)), [], 1
    while i < f.size - 1:
        if f[i] > thresh and f[i] >= f[i - 1] and f[i] >= f[i + 1]:
            out.append(i)
            i += gap
        else:
            i += 1
    return np.array(out, dtype=int)


# ------------------------------------------------------------------ analysis
def analyse(wav_path: str, window, bpm_range=(80.0, 105.0), verbose=True) -> dict:
    """Beat, triplet grid, and the vocal onsets inside `window` (song seconds)."""
    x, sr = mono(wav_path)
    S, fq, t, fps = _stft(x, sr)
    rms = np.sqrt((S ** 2).sum(axis=1))
    voc = S[:, (fq >= 300) & (fq < 3500)].sum(axis=1)

    lvl = rms / max(rms.max(), 1e-9)
    start = float(t[np.argmax(lvl > 0.02)])

    # ---- beat: comb filter over the broadband onset envelope, music only ----
    env = np.maximum(np.diff(rms, prepend=rms[0]), 0)
    m0 = t >= start + 0.5
    env = env / (env[m0].mean() + 1e-9)
    env[~m0] = 0.0
    i0 = int(start * fps)
    best = None
    for bpm in np.arange(bpm_range[0], bpm_range[1], 0.05):
        per = 60.0 / bpm * fps
        for ph in np.arange(0.0, per, max(1.0, per / 48)):
            idx = np.round(np.arange(ph, env.size - i0, per)).astype(int) + i0
            idx = idx[idx < env.size - 1]
            sc = sum(env[max(0, i - 1):i + 2].max() for i in idx) / max(1, len(idx))
            if best is None or sc > best[0]:
                best = (sc, float(bpm), float((ph + i0) / fps), float(per / fps))
    _, bpm, beat_phase, beat = best

    # ---- vocal onsets inside the window ----
    p95 = float(np.percentile(voc[m0], 95))
    fv = _flux(voc)
    sel = (t >= window[0] - 1.0) & (t <= window[1] + 1.0)
    idx = _peaks(np.where(sel, fv, 0.0), 4.0, 0.15, fps)
    idx = [i for i in idx if voc[i] > 0.42 * p95 and window[0] <= t[i] <= window[1]]
    onsets = [(float(t[i]), float(voc[i] / p95)) for i in idx]

    # ---- THE GRID: a triplet of the beat, phase fitted to the onsets themselves ----
    unit = beat / 3.0
    ts = np.array([o[0] for o in onsets])
    if ts.size >= 4:
        # the phase that minimises the onsets' distance to the nearest grid line
        cand = np.arange(0.0, unit, unit / 240)
        err = [np.abs(((ts - c + unit / 2) % unit) - unit / 2).mean() for c in cand]
        gphase = float(cand[int(np.argmin(err))])
        gerr = float(min(err))
    else:
        gphase, gerr = 0.0, 0.0
    # each onset's slot on the grid, and which of the 3 sub-positions of a beat it is
    ev = []
    for tt, st in onsets:
        k = int(round((tt - gphase) / unit))
        ev.append(dict(t=tt, strength=round(st, 3), slot=k, sub=k % 3,
                       snap=round(gphase + k * unit, 4)))

    d = np.diff(ts) if ts.size > 1 else np.array([])
    gaps = np.round(d / unit).astype(int) if d.size else d
    out = dict(
        wav=os.path.basename(wav_path), music_start=round(start, 3),
        bpm=round(bpm, 2), beat=round(beat, 4), beat_phase=round(beat_phase, 3),
        unit=round(unit, 4), grid_phase=round(gphase, 4), grid_err_ms=round(gerr * 1000, 1),
        window=[float(window[0]), float(window[1])], events=ev,
        gap_histogram={int(g): int((gaps == g).sum()) for g in sorted(set(gaps.tolist()))}
        if d.size else {},
    )
    if verbose:
        print(f"[audio] music starts {start:.3f}s | {bpm:.2f} BPM, beat {beat:.4f}s")
        print(f"[audio] GRID unit {unit:.4f}s (beat/3), phase {gphase:.4f}s, "
              f"onsets sit {gerr * 1000:.0f} ms from it on average")
        print(f"[audio] {len(ev)} vocal onsets in {window[0]:.1f}-{window[1]:.1f}s; "
              f"gaps in grid units: {out['gap_histogram']}")
        print(f"[audio] sub-position counts: "
              f"{ {s: sum(1 for e in ev if e['sub'] == s) for s in (0, 1, 2)} }")
    return out


# ------------------------------------------------------------------ the firing plan
def plan_words(an: dict, min_gap=0.20, max_gap=0.80, verbose=True) -> dict:
    """ONE COMBUSTION PER WORD - Clancy, 2026-09-20: *"each word should have its own stroke and
    flash. The speed/rate of the previous video was nearly perfect! this is way too fast."*

    The constant-rate plan below fires on every grid unit (0.221 s), which lands a bang on
    every vocal onset but also on the two thirds of grid lines that carry no word: 45
    combustions where 19 words exist. This one fires on the ONSETS THEMSELVES. That is 19
    combustions averaging 0.53 s apart, against the previous radial's 0.665 s - the rate he
    called nearly perfect - and every bang is a word.

    The cost is that the crank no longer turns at a constant rate: the gaps run 1, 2, 3 or 5
    grid units, so it surges with the hook. sync.SplineDrive carries that as a monotone curve
    with an exact derivative, so the moving-wall reflection stays right; what it is NOT is a
    torque an iron crank could deliver, and that belongs in the honesty list."""
    ev = an["events"]
    unit = an["unit"]
    # MERGE what is tighter than a grid unit. The two words of the hook really are one unit
    # (0.221 s) apart and must each keep their own stroke, but the onset detector also finds
    # pairs 0.139 s apart inside a single syllable, and those would have made the crank surge
    # 8x. Keep the stronger of any such pair.
    ev = sorted(ev, key=lambda e: e["t"])
    kept = []
    for e in ev:
        if kept and e["t"] - kept[-1]["t"] < min_gap:
            if e["strength"] > kept[-1]["strength"]:
                kept[-1] = e
            continue
        kept.append(e)
    fires = [dict(t=round(e["t"], 4), on_vocal=True, sub=int(e["sub"]),
                  strength=float(e["strength"])) for e in kept]
    # FILL the holes. Where the vocal leaves a gap the engine would have to crawl through,
    # put an ordinary combustion in it - no word, no flash, just a bang that keeps the crank
    # from stalling. Without this the drive surged 6x between the tightest pair and the
    # widest rest.
    filled = []
    for i, f in enumerate(fires):
        filled.append(f)
        if i + 1 < len(fires):
            g = fires[i + 1]["t"] - f["t"]
            n_extra = int(g // max_gap)
            for k in range(1, n_extra + 1):
                filled.append(dict(t=round(f["t"] + g * k / (n_extra + 1), 4),
                                   on_vocal=False, sub=-1, strength=0.0))
    fires = filled
    gaps = [fires[i + 1]["t"] - fires[i]["t"] for i in range(len(fires) - 1)]
    out = dict(mode="words", period_units=0, period=round(sum(gaps) / max(1, len(gaps)), 4),
               phase=0.0, n_fires=len(fires), fires=fires, covered=len(fires),
               of_onsets=len(ev), score=0.0,
               gap_min=round(min(gaps), 4), gap_max=round(max(gaps), 4))
    if verbose:
        print(f"[plan] ONE BANG PER WORD: {len(fires)} combustions in "
              f"{an['window'][1] - an['window'][0]:.1f}s, mean gap {out['period']:.3f}s "
              f"(min {out['gap_min']:.3f}, max {out['gap_max']:.3f}); the previous radial "
              f"fired every 0.665 s")
        print(f"[plan] crank {2.5 * out['period']:.3f}s/rev on average, "
              f"{2.5 * out['gap_min']:.3f}-{2.5 * out['gap_max']:.3f}s at the extremes")
        cnt = {s_: sum(1 for f in fires if f["sub"] == s_) for s_ in (0, 1, 2)}
        nw = sum(1 for f in fires if f["on_vocal"])
        print(f"[plan] by grid sub-position: {cnt}; {nw} carry a word, "
              f"{len(fires) - nw} are ordinary bangs filling the vocal's gaps")
    return out



def plan_from_words(an: dict, words_path=None, fill=0.70, verbose=True) -> dict:
    """THE SHIPPED PLAN. Combustions and flashes are no longer the same list.

    Clancy marked the clip in `tools\\mark` and the marks settled what the onset grid could
    not: all four words he named sit on ONE sub-position, and inside the span he marked as
    echoes both sub series belong to the same word. So:

      COMBUSTIONS  one on each PRIMARY word (he asked for a stroke per word), plus ordinary
                   bangs filling any gap wider than `fill` so the crank keeps a steady 0.6 s
                   - the rate he called nearly perfect.
      FLASHES      on every word event. Primaries at full strength; echoes dimmer, and NOT
                   given a combustion of their own, because two bangs 0.14 s apart is a lurch
                   and an echo is not a new stroke.

    Splitting the two is the whole point. Welding them together is what made the first cut
    fire 45 times.
    """
    import json as _json
    wp = words_path or os.path.join(AUDIO_DIR, "flashing_lights.words.json")
    with open(wp, encoding="utf-8") as f:
        wdoc = _json.load(f)
    words = wdoc["words"]
    span = an["window"][1] - an["window"][0]

    prim = [w for w in words if w["primary"]]
    fires = [dict(t=w["t"], on_vocal=True, sub=-1, strength=w["strength"], kind=w["kind"])
             for w in prim]
    # fill, including before the first word and after the last, so the engine is already
    # turning when the clip opens and does not stall when it ends
    edges = [0.0] + [f["t"] for f in fires] + [span]
    for i in range(len(edges) - 1):
        a_, b_ = edges[i], edges[i + 1]
        g = b_ - a_
        n_extra = int(g // fill)
        for k in range(1, n_extra + 1):
            t = a_ + g * k / (n_extra + 1)
            if 0.02 < t < span - 0.02:
                fires.append(dict(t=round(t, 4), on_vocal=False, sub=-1, strength=0.0, kind=""))
    fires.sort(key=lambda f: f["t"])
    flashes = [dict(t=w["t"], kind=w["kind"], strength=w["strength"],
                    primary=bool(w["primary"])) for w in words]
    gaps = [fires[i + 1]["t"] - fires[i]["t"] for i in range(len(fires) - 1)]
    out = dict(mode="marks", period_units=0,
               period=round(sum(gaps) / max(1, len(gaps)), 4), phase=0.0,
               n_fires=len(fires), fires=fires, flashes=flashes,
               covered=sum(1 for f in fires if f["on_vocal"]), of_onsets=len(words), score=0.0,
               gap_min=round(min(gaps), 4), gap_max=round(max(gaps), 4),
               words_source=wdoc.get("source"))
    if verbose:
        nw = sum(1 for f in fires if f["on_vocal"])
        print(f"[plan] FROM HIS MARKS: {len(fires)} combustions, mean {out['period']:.3f}s "
              f"(min {out['gap_min']:.3f}, max {out['gap_max']:.3f}); the previous radial "
              f"fired every 0.665 s")
        print(f"[plan] {nw} combustions carry a word; "
              f"{len(flashes)} flashes ({sum(1 for f in flashes if f['primary'])} words, "
              f"{sum(1 for f in flashes if not f['primary'])} echoes)")
    return out


def plan(an: dict, period_units: int | None = None, verbose=True) -> dict:
    """Choose a CONSTANT firing period, as a whole number of grid units, and the phase that
    puts the most combustions on real vocal onsets.

    A whole number of units is the whole point: the engine then turns at one steady speed and
    still lands on the music. Chasing every syllable would swing the crank speed 3.8x between
    consecutive bangs, which is a lurch, not an engine."""
    unit, win = an["unit"], an["window"]
    ev = an["events"]
    ts = np.array([e["t"] for e in ev])
    wt = np.array([e["strength"] for e in ev])
    span = win[1] - win[0]
    best = None
    for P in ([period_units] if period_units else [1, 2, 3]):
        per = P * unit
        for ph in np.arange(0.0, per, unit / 24):
            fires = np.arange(win[0] + ph, win[1] + 1e-9, per)
            if fires.size < 4:
                continue
            hit = np.array([np.min(np.abs(ts - f)) if ts.size else 9.9 for f in fires])
            on = hit < 0.5 * unit
            # score: weighted onsets covered, minus a small cost for bangs on nothing
            sc = float(wt[[int(np.argmin(np.abs(ts - f))) for f in fires[on]]].sum()) - 0.15 * (~on).sum()
            if best is None or sc > best[0]:
                best = (sc, P, float(ph), fires, on)
    sc, P, ph, fires, on = best
    fired = []
    for f, o in zip(fires, on):
        j = int(np.argmin(np.abs(ts - f))) if ts.size else -1
        fired.append(dict(t=round(float(f), 4), on_vocal=bool(o),
                          sub=int(ev[j]["sub"]) if o else -1,
                          strength=float(ev[j]["strength"]) if o else 0.0))
    out = dict(period_units=P, period=round(P * unit, 4), phase=round(ph, 4),
               n_fires=len(fired), fires=fired,
               covered=int(sum(1 for f in fired if f["on_vocal"])),
               of_onsets=len(ev), score=round(sc, 2))
    if verbose:
        rev = out["period"] * 5 / 2.0       # 5 fires per 2 crank revolutions
        print(f"[plan] fire every {P} unit(s) = {out['period']:.4f}s  -> {rev:.4f}s per crank rev "
              f"({5.0 / rev:.2f}x the radial's 5 s base)")
        print(f"[plan] {out['n_fires']} combustions in {span:.1f}s; {out['covered']} land on a "
              f"vocal onset, of {len(ev)} onsets present")
        cnt = {s: sum(1 for f in fired if f["sub"] == s) for s in (0, 1, 2)}
        print(f"[plan] combustions by grid sub-position: {cnt}")
    return out


def build(src: str, slug: str, window, period_units=None, force=False, mode="marks") -> dict:
    """Decode once, analyse, plan, and write audio\\<slug>.json. Returns the whole thing."""
    wav = decode(src, slug, force=force)
    an = analyse(wav, window)
    pl = (plan_from_words(an) if mode == "marks"
          else plan_words(an) if mode == "words" else plan(an, period_units))
    doc = dict(slug=slug, src=os.path.basename(src), **an, plan=pl)
    with open(os.path.join(AUDIO_DIR, slug + ".json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"[audio] -> {os.path.join(AUDIO_DIR, slug + '.json')}")
    return doc


def load(slug: str) -> dict:
    with open(os.path.join(AUDIO_DIR, slug + ".json"), encoding="utf-8") as f:
        return json.load(f)


def labels(slug: str) -> dict:
    """The word labels, if the marking page has written any. `{sub: 'flashing'|'lights'|''}`
    keyed by the grid sub-position 0/1/2. Absent means sync.py's default applies."""
    p = os.path.join(AUDIO_DIR, slug + ".labels.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}
