"""render.py — drive the frame clock, stroke each scene's beams onto the Scope, pipe to ffmpeg.

manim-free equivalent of `manim render`: owns the clock + the rawvideo pipe. Frames are driven
by REAL seconds (t = i/fps), so a low-fps preview and a 30-fps final stay in sync.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .config import OUT, RenderConfig, encoder_args, find_ffmpeg
from .scope import Scope


def _finalize(tmp, out_path, returncode):
    """Atomic publish: the encode writes to <out>.part.mp4; only a fully-finalized file is
    renamed onto the real name. An interrupted/failed render leaves NO unfinalized final
    (the old gotcha) — at worst a .part.mp4 that is removed here or safely ignorable."""
    if returncode != 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg pipe failed (exit {returncode})")
    os.replace(tmp, out_path)
    return out_path


def render_scene(scene, out_path, cfg: RenderConfig, seconds=None, progress=True):
    seconds = seconds if seconds is not None else scene.duration
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ffmpeg = find_ffmpeg()
    n_frames = max(1, int(round(seconds * cfg.fps)))
    tmp = out_path + ".part.mp4"
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "pipe:0",
        "-an", *encoder_args(), "-pix_fmt", "yuv420p", tmp,
    ]
    cfg.decay = getattr(scene, "decay", cfg.decay)       # scenes may set their own afterglow

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    scope = Scope(cfg)
    try:
        return _render_frames(scene, scope, proc, cfg, seconds, n_frames, tmp, out_path,
                              progress)
    except BaseException:                     # incl. KeyboardInterrupt: never leave a .part
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _render_frames(scene, scope, proc, cfg, seconds, n_frames, tmp, out_path, progress):
    # A scene exposing frame_traces()/pos() is a TRAIL scene (sub-frame stepped head +
    # persistence). A scene exposing beams(t) draws whole curves each frame.
    trail = hasattr(scene, "frame_traces") or hasattr(scene, "pos")
    try:
        if trail:
            subs = getattr(scene, "substeps", 90)
            trail_gain = getattr(scene, "trail_gain", 0.9)
            base_head = getattr(scene, "head_gain", 1.6)
            tw = getattr(scene, "trail_width", 3)

            def draw(t, ts):
                """Stroke one frame onto the scope; returns the grid state to compose with."""
                scope.new_frame()
                if hasattr(scene, "frame_traces"):
                    for pts, color, gain, ss in scene.frame_traces(t, ts):
                        scope.beam(pts, color=color, gain=gain, speed_shade=ss, width=tw)
                    h = scene.head_at(t) if hasattr(scene, "head_at") else None
                    if h is not None:
                        scope.head(h[0], color=h[1], gain=h[2])
                    if hasattr(scene, "flash_traces"):     # brief non-persistent overlays
                        for pts, color, gain in scene.flash_traces(t):
                            scope.beam_transient(pts, color=color, gain=gain, width=tw)
                    return scene.grid_state(t) if hasattr(scene, "grid_state") else 0.0
                pts = scene.pos(ts)
                scope.beam(pts, gain=trail_gain, speed_shade=False, width=tw)
                hg = scene.head_gain_at(t) if hasattr(scene, "head_gain_at") else base_head
                if hg > 0:
                    scope.head(pts[-1], gain=hg)
                return 0.0

            # warmup: pre-roll the loop's tail into the phosphor (not written) so the first output
            # frame's afterglow matches the last -> a seamless loop even with full-brightness endpoints
            n_warm = int(round(getattr(scene, "warmup", 0.0) * cfg.fps))
            for i in range(n_warm):
                t = seconds - (n_warm - i) / cfg.fps       # ends exactly one frame before the seam
                draw(t, np.linspace(t, t + 1 / cfg.fps, subs))
            for i in range(n_frames):
                t = i / cfg.fps
                gs = draw(t, np.linspace(t, (i + 1) / cfg.fps, subs))
                proc.stdin.write(scope.render(grid_state=gs).tobytes())
                if progress and i % cfg.fps == 0:
                    print(f"  frame {i}/{n_frames}  (t={t:.1f}s)", flush=True)
            proc.stdin.close(); proc.wait()
            return _finalize(tmp, out_path, proc.returncode)

        for i in range(n_frames):
            t = i / cfg.fps
            scope.new_frame()
            for pts, gain, closed in scene.beams(t):
                scope.beam(pts, gain=gain, closed=closed)
            gs = scene.grid_state(t) if hasattr(scene, "grid_state") else 0.0
            proc.stdin.write(scope.render(grid_state=gs).tobytes())
            if progress and i % cfg.fps == 0:
                print(f"  frame {i}/{n_frames}  (t={t:.1f}s)", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    return _finalize(tmp, out_path, proc.returncode)
