"""
build_montage.py — prepare all assets for the MusicProfile montage (music-visualizer
template). Run with the manim .venv python (needs numpy + scipy; uses bundled ffmpeg
and curl for thumbnails).

Produces, under  montage_assets/<name>/ :
  top_all.npy   [n_frames, 32]  mid/high-frequency EQ track  -> TOP light bar
  bot_all.npy   [n_frames, 20]  bass/percussion EQ track     -> BOTTOM light bar
  audio.m4a     the 3 song clips concatenated (15s each)     -> muxed onto the render
  thumbs/<id>.jpg  YouTube thumbnails for the carousel
  montage.json  manifest the scene reads (per-segment song/artist/molecule/shade/thumb)

The SAME clip starts + 15s lengths are used for BOTH the features and the audio, so the
bars stay synced to the muxed track. The bass/treble split puts kick+bass on the bottom
bar and everything else on the top bar.

Usage:
  .venv\\Scripts\\python.exe build_montage.py
  .venv\\Scripts\\python.exe build_montage.py [MontageName]   # omit name to build ALL
"""
import json
import os
import subprocess
import sys

import numpy as np

from make_audio_bars import FFMPEG, build_bars, decode_mono

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "montage_assets")
SONG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "YT_Downloader", "output"))

FPS = 30
SEG_SECS = 15
SEG_FRAMES = FPS * SEG_SECS                       # 450 frames per segment
SR = 22050
N_TOP = 32                                        # top bar (mid/high)
N_BOT = 20                                        # bottom bar (bass)
XFADE = 0.5                                       # audio crossfade between songs (s)
FADE = 0.4                                        # global fade in / fade out (s)
LEAD_MS = 120.0                                   # pull the bars this many ms AHEAD of the sound

# extra thumbnails that flank the targets in the scrolling carousel reel (shared pool)
REEL_EXTRA = ["rtL5oMyBHPs", "A2VpR8HahKc", "RBtlPT23PTM", "8mGBaXPlri8",
              "kn6-c223DUU", "ewRjZoRtu0Y", "x7_jYYISAbM", "D3VpepnO0gw", "q7qdpyjHyNk"]

# ---- the montages. Each = 3 molecule/song pairs; shade order is the palette order. -----
MONTAGES = {
    "AspirinTylenolCaffeine": [   # first template (red, blue, purple)
        {"song": "Hey Kids", "artist": "Molina feat. Late Verlane", "mol": "Aspirin",
         "shade": "red", "id": "D3VpepnO0gw", "start": 17.0,
         "file": "Molina_-_Hey_Kids_feat._Late_Verlane [D3VpepnO0gw].mp3"},
        {"song": "Duvet", "artist": "Bôa", "mol": "Acetaminophen",
         "shade": "blue", "id": "Uoox9fpmDP0", "start": 30.0,
         "file": "Boa_-_Duvet_Official_Video [Uoox9fpmDP0]_6-212.mp3"},
        {"song": "The Perfect Girl", "artist": "Mareux", "mol": "Caffeine",
         "shade": "purple", "id": "q7qdpyjHyNk", "start": 45.0,
         "file": "Mareux_-_The_Perfect_Girl_Official_Music_Video [q7qdpyjHyNk].mp3"},
    ],
    "DopamineSerotoninNicotine": [   # palette switched: purple, blue, red
        {"song": "Girl Like Me", "artist": "PinkPantheress", "mol": "Dopamine",
         "shade": "purple", "id": "qRAJowgHqxA", "start": 20.0,
         "file": "PinkPantheress_-_Girl_Like_Me_Official_Video [qRAJowgHqxA].mp3"},
        {"song": "Resonance", "artist": "Midwest Emo Version", "mol": "Serotonin",
         "shade": "blue", "id": "LdVtUo-xngw", "start": 40.0,
         "file": "resonance_midwest_emo_version [LdVtUo-xngw].mp3"},
        {"song": "Sirius", "artist": "The Alan Parsons Project", "mol": "Nicotine",
         "shade": "red", "id": "OkC_oi0ksuw", "start": 30.0,
         "file": "The_Alan_Parsons_Project_-_Sirius_Official_Audio [OkC_oi0ksuw]_10-116.mp3"},
    ],
    "ProlineValineLysine": [   # palette switched: purple, blue, red
        {"song": "One More Time", "artist": "Daft Punk", "mol": "Proline",
         "shade": "purple", "id": "A2VpR8HahKc", "start": 35.0,
         "file": "Daft_Punk_-_One_More_Time_Official_Audio [A2VpR8HahKc].mp3"},
        {"song": "Want to Love", "artist": "Just Raw", "mol": "Valine",
         "shade": "blue", "id": "rDAAs3Ka3Mo", "start": 20.0,
         "file": "Want_To_Love_Just_Raw [rDAAs3Ka3Mo].mp3"},
        {"song": "Always Forever", "artist": "Cults", "mol": "Lysine",
         "shade": "red", "id": "x7_jYYISAbM", "start": 30.0,
         "file": "Cults_-_Always_Forever [x7_jYYISAbM].mp3"},
    ],
}


def fetch_thumb(vid, thumbs_dir):
    dst = os.path.join(thumbs_dir, f"{vid}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 2000:
        return dst
    url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
    subprocess.run(["curl", "-sL", "-o", dst, url], check=True)
    return dst


def main(name):
    SEGMENTS = MONTAGES[name]
    out = os.path.join(OUT_ROOT, name)
    thumbs = os.path.join(out, "thumbs")
    os.makedirs(thumbs, exist_ok=True)
    print(f"\n=== building montage: {name} ===")

    # 1) thumbnails (targets + reel pool)
    for vid in [s["id"] for s in SEGMENTS] + REEL_EXTRA:
        fetch_thumb(vid, thumbs)
    print(f"thumbnails -> {thumbs}")

    # 2) crossfaded audio. Each clip is SEG_SECS + XFADE long; acrossfade overlaps the
    #    songs so the change is smooth (no hard cut). Because (clip_len - XFADE) == 15s,
    #    each NEW song starts exactly on a 15s boundary -> zero drift vs the features.
    TOTAL = len(SEGMENTS) * SEG_SECS                  # 45
    audio = os.path.join(out, "audio.m4a")
    cmd = [FFMPEG, "-y", "-v", "error"]
    for s in SEGMENTS:
        cmd += ["-ss", str(s["start"]), "-t", str(SEG_SECS + XFADE),
                "-i", os.path.join(SONG_DIR, s["file"])]
    raw_total = len(SEGMENTS) * (SEG_SECS + XFADE) - (len(SEGMENTS) - 1) * XFADE
    fc = f"[0:a]afade=t=in:st=0:d={FADE}[a0];"
    prev = "a0"
    for i in range(1, len(SEGMENTS)):
        fc += f"[{prev}][{i}:a]acrossfade=d={XFADE}[a{i}];"
        prev = f"a{i}"
    fc += f"[{prev}]afade=t=out:st={raw_total - FADE:.3f}:d={FADE}[a]"
    cmd += ["-filter_complex", fc, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", audio]
    subprocess.run(cmd, check=True)
    print(f"audio -> {audio}  (~{raw_total:.1f}s, {XFADE}s crossfades)")

    # 3) features FROM THE FINAL AUDIO, so the bars match the muxed track exactly
    #    (including the crossfades). LEAD_MS pulls the bars a few ms ahead of the sound.
    samples = decode_mono(audio, 0.0, TOTAL + 1.0, SR)
    nf = FPS * TOTAL
    top_all = build_bars(samples, SR, FPS, N_TOP, n_frames=nf, f_lo=200.0, f_hi=12000.0,
                         spacing="log", tilt=True, lead_ms=LEAD_MS)
    bot_all = build_bars(samples, SR, FPS, N_BOT, n_frames=nf, f_lo=30.0, f_hi=200.0,
                         spacing="linear", tilt=False, decay=0.34, lead_ms=LEAD_MS)
    np.save(os.path.join(out, "top_all.npy"), top_all)
    np.save(os.path.join(out, "bot_all.npy"), bot_all)
    print(f"features -> top{top_all.shape} bot{bot_all.shape}  (lead {LEAD_MS:.0f}ms)")

    # 4) manifest
    manifest = {
        "name": name, "fps": FPS, "seg_frames": SEG_FRAMES,
        "top_npy": os.path.join(out, "top_all.npy"),
        "bot_npy": os.path.join(out, "bot_all.npy"),
        "audio": audio, "thumbs_dir": thumbs,
        "reel": [s["id"] for s in SEGMENTS] + REEL_EXTRA,
        "segments": [{"song": s["song"], "artist": s["artist"], "mol": s["mol"],
                      "shade": s["shade"], "thumb": f"{s['id']}.jpg", "start": s["start"]}
                     for s in SEGMENTS],
    }
    with open(os.path.join(out, "montage.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest -> {os.path.join(out, 'montage.json')}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MONTAGES)
    for nm in targets:
        main(nm)
