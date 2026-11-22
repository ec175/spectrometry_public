# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""cli.py - drive the BIO engine from a terminal.

    .\.venv\Scripts\python.exe cli.py list
    .\.venv\Scripts\python.exe cli.py params bio
    .\.venv\Scripts\python.exe cli.py dry    bio --scene scenes/v4/engulfment.json
    .\.venv\Scripts\python.exe cli.py still  bio --t 0 4 8 --out out/bio
    .\.venv\Scripts\python.exe cli.py render bio --scene scenes/v4/engulfment.json \
        --quality 2160 --orientation vertical --fps 60

This project was split out of `..\..\tools\Engine_Kit\` on 2026-08-27: that one is a
two-engine localhost web app being kept for later, this one is the BIO engine as a standalone
render engine, with v4 behaviour as the default. There is no browser UI here - the CLI IS the
interface, and `kit\jobs.py` is carried over unused so the two trees stay diffable.

`dry` is a SUBCOMMAND, not a flag, matching the rest of Clancy's library. RUN IT BEFORE
EVERY RENDER. It costs a couple of seconds and it catches the four failures that
otherwise cost you the whole render: non-finite state, a black frame, a blown frame, and
a scene where nothing moves.

The browser UI calls exactly these code paths, so anything you can do here you can do
there and vice versa.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kit.config import OUT, QUALITY, RenderConfig             # noqa: E402
from kit.render import dry, render_scene, render_still        # noqa: E402
from kit.scene import ENGINES, build_engine, load_scene       # noqa: E402


def _cfg(a, scene):
    zoom = float(scene.get("zoom", 1.0))
    centre = tuple(scene.get("centre", (0.0, 0.0)))
    if a.live:
        return RenderConfig.live(a.orientation, zoom=zoom, centre=centre)
    return RenderConfig.tier(a.quality, a.orientation, a.fps, ss=a.ss,
                             zoom=zoom, centre=centre)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cli.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="the engines and the saved scenes")
    p = sub.add_parser("params", help="print an engine's parameter schema")
    p.add_argument("engine")

    for nm in ("dry", "still", "render"):
        p = sub.add_parser(nm)
        p.add_argument("engine", nargs="?", default=None,
                       help="engine name; omit if --scene names one")
        p.add_argument("--scene", default=None, help="path to a scene .json")
        p.add_argument("--seconds", type=float, default=None)
        p.add_argument("--seed", type=int, default=None)
        # Defaults match the SHIPPED format. Every approved Bio final is
        # `<name>_2160_vertical_60fps.mp4`, so 1080/horizontal - Engine_Kit's defaults, chosen
        # for a fast browser preview - would have meant typing two flags on every real render
        # and silently getting the wrong thing when you forgot.
        p.add_argument("--quality", default="2160", choices=tuple(sorted(QUALITY)))
        p.add_argument("--orientation", default="vertical",
                       choices=("horizontal", "vertical"))
        p.add_argument("--fps", type=int, default=60, choices=(30, 60))
        p.add_argument("--ss", type=int, default=2, help="supersample; 1 is faster/jaggier")
        p.add_argument("--live", action="store_true", help="use the low-res preview tier")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override one parameter; repeatable")
        if nm == "dry":
            p.add_argument("--n", type=int, default=5)
        if nm == "still":
            p.add_argument("--t", type=float, nargs="+", default=[0.0])
            p.add_argument("--out", default=None)
        if nm == "render":
            p.add_argument("--out", default=None)
            p.add_argument("--no-nvenc", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "list":
        print("engines:")
        for k, m in ENGINES.items():
            print("  %-9s %s" % (k, m.BLURB))
        sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")
        names = sorted(f for f in os.listdir(sd) if f.endswith(".json")) \
            if os.path.isdir(sd) else []
        print("\nscenes (%d):" % len(names))
        for f in names:
            try:
                d = json.load(open(os.path.join(sd, f), encoding="utf8"))
                print("  %-28s %-8s %.1fs  %s" % (f, d.get("engine", "?"),
                                                  d.get("seconds", 0),
                                                  d.get("title", "")))
            except Exception as e:
                print("  %-28s UNREADABLE (%s)" % (f, e))
        return 0

    if a.cmd == "params":
        m = ENGINES[a.engine]
        group = None
        for spec in m.PARAMS:
            if spec.group != group:
                group = spec.group
                print("\n[%s]" % group)
            rng = ("  %s..%s" % (spec.lo, spec.hi)) if spec.lo is not None else ""
            ch = ("  {%s}" % "|".join(spec.choices)) if spec.choices else ""
            print("  %-18s %-7s default=%-10s%s%s" % (spec.name, spec.type,
                                                      spec.default, rng, ch))
            if spec.help:
                print("      %s" % spec.help)
        return 0

    scene = load_scene(a.scene) if a.scene else {}
    if a.engine:
        scene["engine"] = a.engine
    if not scene.get("engine"):
        ap.error("give an engine name or a --scene that names one")
    for kv in a.set:
        if "=" not in kv:
            ap.error("--set wants KEY=VALUE, got %r" % kv)
        k, v = kv.split("=", 1)
        scene.setdefault("params", {})[k] = v
    if a.seed is not None:
        scene["seed"] = a.seed
    seconds = a.seconds if a.seconds is not None else float(scene.get("seconds", 10.0))

    cfg = _cfg(a, scene)
    eng = build_engine(scene)

    if a.cmd == "dry":
        ok, rep = dry(eng, cfg, seconds, n=a.n)
        s = eng.summary() if hasattr(eng, "summary") else {}
        for w in s.get("warnings", []):
            print("      NOTE  " + w)
        return 0 if ok else 1

    if a.cmd == "still":
        base = a.out or os.path.join(OUT, "%s_still" % scene["engine"])
        for t in a.t:
            path = "%s_t%05.2f.png" % (base, t)
            render_still(build_engine(scene), cfg, t, path)
            print("wrote %s" % path)
        return 0

    ok, rep = dry(eng, cfg, seconds, n=3, verbose=True)
    if not ok:
        print("\nGATE FAILED - not rendering. Fix the failures above, or force with "
              "--set on a parameter that changes them.")
        return 1

    out = a.out or os.path.join(
        OUT, "%s_%s_%s_%d.mp4" % (scene["engine"], a.quality, a.orientation, a.fps))

    def prog(i, n, el):
        rate = i / max(el, 1e-6)
        sys.stdout.write("\r  %d/%d frames  %.1f fps  eta %.0fs   "
                         % (i, n, rate, (n - i) / max(rate, 1e-6)))
        sys.stdout.flush()

    res = render_scene(build_engine(scene), cfg, out, seconds,
                       on_progress=prog, prefer_nvenc=not a.no_nvenc)
    print()
    print("  %s  %s  %s bytes  %s frames  %.1fs elapsed"
          % ("OK" if res["ok"] else "PROBLEM", res["path"], res.get("bytes"),
             res.get("nb_frames"), res["seconds_elapsed"]))
    for n_ in res["notes"]:
        print("  NOTE " + n_)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
