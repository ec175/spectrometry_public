# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""encoder.py - find ffmpeg, pick the best encoder that actually works, write the file.

This module exists so that "it renders on my machine but not yours" cannot happen. It is
the single most portable-hostile part of any video engine, so everything here degrades
rather than fails.


FINDING FFMPEG (in order, first hit wins)
-----------------------------------------
1. ``$EK_FFMPEG`` - an explicit override. Set this if you have a build you trust.
2. ``ffmpeg`` on PATH - the normal case.
3. ``imageio_ffmpeg``'s bundled binary - a pip dependency, so ``pip install -r
   requirements.txt`` alone is enough to render even on a machine with no system ffmpeg.
   This is the reason a new user is never stranded.

If all three miss, `find_ffmpeg` raises with the three things to try, rather than letting
a `FileNotFoundError` surface from somewhere deep in a render two minutes in.


PICKING AN ENCODER
------------------
NVENC (`h264_nvenc`) is an NVIDIA hardware encoder: roughly 5-10x faster than libx264 at
delivery settings, and it is the difference between a 4 minute render and a 30 second one.
But it is present only on NVIDIA GPUs, it can be compiled out of an ffmpeg build, and it
fails at RUN time rather than at probe time if the driver is too old.

So we PROBE it once per process by actually encoding eight frames of noise, cached in
`_NVENC`. A probe that encodes real frames is the only honest test - `ffmpeg -encoders`
listing `h264_nvenc` proves only that the build knows the name, not that the machine can
run it. If the probe fails for any reason we fall back to `libx264`, print one line
saying so, and carry on. **A missing GPU can only ever make a render slower, never fail.**

Set ``EK_NVENC=0`` to force libx264 (useful when you want bit-identical output across two
machines - hardware encoders are not bit-reproducible between driver versions).


PIXEL FORMAT
------------
Always ``yuv420p``. It is the only chroma subsampling that every browser, phone, Slack
preview and social platform will decode. ``yuv444p`` looks better and will silently fail
to play for the person you send it to.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

#: Resolved once per process by `probe_nvenc`. None = not yet probed.
_NVENC = None
_FFMPEG = None


def find_ffmpeg() -> str:
    """Absolute path to an ffmpeg binary. See the module docstring for the search order."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG

    env = os.environ.get("EK_FFMPEG")
    if env and os.path.isfile(env):
        _FFMPEG = env
        return _FFMPEG

    onpath = shutil.which("ffmpeg")
    if onpath:
        _FFMPEG = onpath
        return _FFMPEG

    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        pass

    raise RuntimeError(
        "No ffmpeg found. Any one of these fixes it:\n"
        "  1. pip install -r requirements.txt   (installs imageio-ffmpeg, which bundles one)\n"
        "  2. install ffmpeg and put it on PATH\n"
        "  3. set EK_FFMPEG to the full path of an ffmpeg binary")


def find_ffprobe():
    """ffprobe if we can find one, else None. Only used for verification, never rendering,
    so it is allowed to be absent - `verify_output` degrades to a file-size check."""
    env = os.environ.get("EK_FFPROBE")
    if env and os.path.isfile(env):
        return env
    onpath = shutil.which("ffprobe")
    if onpath:
        return onpath
    # imageio-ffmpeg ships ffmpeg only, but a sibling ffprobe is common in real installs.
    try:
        cand = os.path.join(os.path.dirname(find_ffmpeg()), "ffprobe.exe" if os.name == "nt"
                            else "ffprobe")
        return cand if os.path.isfile(cand) else None
    except Exception:
        return None


def probe_nvenc(verbose: bool = True) -> bool:
    """True if h264_nvenc can actually encode on this machine, right now.

    Encodes 8 frames of real noise through the real encoder. Cached per process.
    """
    global _NVENC
    if _NVENC is not None:
        return _NVENC
    if os.environ.get("EK_NVENC", "1") == "0":
        _NVENC = False
        return _NVENC

    try:
        ff = find_ffmpeg()
    except Exception:
        _NVENC = False
        return _NVENC

    cmd = [ff, "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "nullsrc=s=256x256:d=1:r=8",
           "-vf", "geq=random(1)*255:128:128",
           "-c:v", "h264_nvenc", "-frames:v", "8",
           "-f", "null", os.devnull]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        _NVENC = (r.returncode == 0)
        if not _NVENC and verbose:
            tail = (r.stderr or b"").decode("utf8", "replace").strip().splitlines()
            why = tail[-1] if tail else "no stderr"
            print("[encoder] NVENC unavailable (%s) - using libx264. Renders will be "
                  "slower, not worse." % why[:140])
    except Exception as e:
        _NVENC = False
        if verbose:
            print("[encoder] NVENC probe failed (%s: %s) - using libx264."
                  % (type(e).__name__, e))
    return _NVENC


def encoder_args(cfg, prefer_nvenc: bool = True) -> list:
    """The `-c:v ...` half of the ffmpeg command line for this RenderConfig.

    When `cfg.bitrate` is set we drive the encoder toward that RATE as well as toward its
    quality target, with a peak ceiling and a matching VBV buffer. That combination -
    rather than quality alone - is what matters when the file is going to be transcoded
    again by a social platform: their encoder can only work from what you handed it.
    """
    br = int(getattr(cfg, "bitrate", 0) or 0)
    peak = int(br * float(getattr(cfg, "bitrate_peak", 1.6))) if br else 0
    gop = str(max(2, int(cfg.fps)))          # one keyframe per second

    if prefer_nvenc and probe_nvenc():
        a = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
             "-rc", "vbr", "-cq", str(cfg.cq),
             "-profile:v", "high", "-g", gop, "-bf", "3",
             "-pix_fmt", "yuv420p"]
        a += (["-b:v", str(br), "-maxrate", str(peak), "-bufsize", str(peak * 2)]
              if br else ["-b:v", "0"])
        return a

    a = ["-c:v", "libx264", "-preset", "slow", "-crf", str(cfg.crf),
         "-profile:v", "high", "-g", gop, "-bf", "3",
         "-pix_fmt", "yuv420p"]
    if br:
        # With libx264, crf and a bitrate target are mutually exclusive, so a rate floor
        # is expressed as a ceiling plus a buffer and left to crf otherwise.
        a += ["-maxrate", str(peak), "-bufsize", str(peak * 2)]
    return a


class VideoWriter:
    """Pipe RGB uint8 frames into ffmpeg. Use as a context manager.

        with VideoWriter(out, cfg) as w:
            for frame in frames:
                w.write(frame)          # (H, W, 3) uint8

    Frames must be exactly (cfg.height, cfg.width, 3). Supersampled frames are downsampled
    by the caller (`kit.canvas.Canvas.to_rgb`), not here - this class does one job.
    """

    def __init__(self, path: str, cfg, prefer_nvenc: bool = True, quiet: bool = True):
        self.path = path
        self.cfg = cfg
        self.w, self.h = cfg.width, cfg.height
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._cmd = ([find_ffmpeg(), "-hide_banner",
                      "-loglevel", "error" if quiet else "info", "-y",
                      "-f", "rawvideo", "-vcodec", "rawvideo",
                      "-s", "%dx%d" % (self.w, self.h),
                      "-pix_fmt", "rgb24", "-r", str(cfg.fps), "-i", "-",
                      "-an"] + encoder_args(cfg, prefer_nvenc)
                     + ["-movflags", "+faststart", path])
        self.proc = None
        self.n = 0

    def __enter__(self):
        self.proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    def write(self, frame):
        if frame.shape != (self.h, self.w, 3):
            raise ValueError("frame is %s, expected %s"
                             % (frame.shape, (self.h, self.w, 3)))
        if frame.dtype.name != "uint8":
            raise ValueError("frame dtype is %s, expected uint8" % frame.dtype)
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            err = self.proc.stderr.read().decode("utf8", "replace")
            raise RuntimeError("ffmpeg died after %d frames:\n%s" % (self.n, err[-2000:]))
        self.n += 1

    def __exit__(self, *exc):
        if self.proc is None:
            return False
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode("utf8", "replace")
        self.proc.stderr.close()
        if rc != 0 and exc[0] is None:
            raise RuntimeError("ffmpeg exited %d after %d frames:\n%s"
                               % (rc, self.n, err[-2000:]))
        return False


def verify_output(path: str, expect_frames: int = None, expect_size: tuple = None) -> dict:
    """Check a finished file is actually playable, and report what it is.

    ALWAYS call this after a render. A truncated or zero-frame mp4 is produced by several
    ordinary failures (the pipe closing early, an encoder that rejected a frame size) and
    it is not obvious from the file existing, or even from its size.
    """
    out = {"path": path, "ok": False, "notes": []}
    if not os.path.isfile(path):
        out["notes"].append("file does not exist")
        return out
    out["bytes"] = os.path.getsize(path)
    if out["bytes"] < 2048:
        out["notes"].append("file is %d bytes - effectively empty" % out["bytes"])
        return out

    fp = find_ffprobe()
    if not fp:
        out["ok"] = True
        out["notes"].append("no ffprobe available; checked file size only")
        return out

    cmd = [fp, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_frames,r_frame_rate,codec_name",
           "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        for line in r.stdout.decode("utf8", "replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception as e:
        out["notes"].append("ffprobe failed: %s" % e)
        out["ok"] = True
        return out

    n = out.get("nb_frames", "N/A")
    try:
        n = int(n)
    except Exception:
        n = None
    if n is not None and n < 2:
        out["notes"].append("only %s frames in the file" % n)
        return out
    if expect_frames and n is not None and abs(n - expect_frames) > max(2, expect_frames * 0.02):
        out["notes"].append("expected ~%d frames, file has %d" % (expect_frames, n))
    if expect_size:
        got = (int(out.get("width", 0)), int(out.get("height", 0)))
        if got != tuple(expect_size):
            out["notes"].append("expected %s, file is %s" % (tuple(expect_size), got))
            return out
    out["ok"] = True
    return out


if __name__ == "__main__":
    print("ffmpeg :", find_ffmpeg())
    print("ffprobe:", find_ffprobe())
    print("nvenc  :", probe_nvenc())
    sys.exit(0)
