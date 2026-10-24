# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""cli.py - the entry point.

    python cli.py list
    python cli.py <scene> --dry                     flux report, no drawing (~2 s)
    python cli.py <scene> --still 0,4,7.5,12        PNGs from the same scene clock
    python cli.py <scene> --preview                 540x960 @30
    python cli.py <scene>                           1080x1920 @60 -> out\\<scene>.mp4

--dry FIRST, ALWAYS. It answers the only question a render cannot: are the lines terminating
where field lines have to terminate? Tune against it, not against renders.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl.config import OUT, RenderConfig          # noqa: E402
from fl.render import flux_report, render_scene, render_stills  # noqa: E402
from fl.scenes import SCENES                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--still", default="")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--ss", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.scene == "list":
        for k, v in SCENES.items():
            # A scene without a docstring is legal; `list` must not die on one.
            doc = next(iter((v.__doc__ or "").strip().splitlines()), "")
            print(f"  {k:16s} {v.duration:4.0f}s  {doc}")
        return

    if a.scene not in SCENES:
        raise SystemExit(f"unknown scene {a.scene!r}; try `cli.py list`")
    scene = SCENES[a.scene]()

    if a.dry:
        print(f"[dry] {scene.name}  {a.seconds or scene.duration:.1f}s")
        flux_report(scene, a.seconds)
        return

    # The scene declares the reference frame it was authored in (`REF`, default 1080x1920), so
    # a 16:9 scene renders 16:9 and a 9:16 scene is completely unaffected.
    ref_w, ref_h = getattr(scene, "REF", (1080, 1920))
    kw = {"ref_w": ref_w, "ref_h": ref_h}
    if a.fps:
        kw["fps"] = a.fps
    if a.ss:
        kw["ss"] = a.ss
    if a.width:
        kw["width"] = a.width
        # aspect comes from the SCENE, not a hardcoded 1920/1080
        kw["height"] = int(round(a.width * ref_h / ref_w))
    elif not a.preview:
        # WITHOUT THIS a 16:9 scene renders 9:16. Setting ref_w/ref_h alone only fixes `scale`;
        # width/height keep the dataclass defaults (1080x1920), and the preview path masked it
        # because `preview()` derives its size from the ref. Cost of learning that: a full
        # 10-scene batch rendered in the wrong aspect. Preview deliberately leaves width unset
        # so `preview()` can halve the scene's own reference.
        kw["width"], kw["height"] = ref_w, ref_h
    cfg = RenderConfig.preview(**kw) if a.preview else RenderConfig(**kw)

    if a.still:
        times = [float(x) for x in a.still.split(",")]
        render_stills(scene, cfg, times,
                      prefix=scene.name + ("_pv" if a.preview else ""))
        return

    out = a.out or os.path.join(
        OUT, f"{scene.name}{'_preview' if a.preview else ''}.mp4")
    render_scene(scene, cfg, out, a.seconds)


if __name__ == "__main__":
    main()
