# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""scene.py - the engine registry and the scene file format.

A SCENE IS A PLAIN JSON FILE
---------------------------
Readable, diffable, hand-editable, and safe to keep in version control:

    {
      "title":   "Colony bloom",
      "engine":  "bio",
      "seconds": 12.0,
      "seed":    7,
      "zoom":    1.0,
      "centre":  [0.0, 0.0],
      "params":  { "growth_rate": 0.22, "tension": 0.55 },
      "objects": [ ... ]                     # objects engine only
    }

Only `engine` is required; everything else falls back to the engine's own defaults. A key
that an engine does not recognise is DROPPED WITH A WARNING rather than silently kept -
see `kit.params.coerce_all` for why that is the safer of the two failure modes.

Saving from the browser writes exactly this format, so a scene you tuned with sliders can
be opened in a text editor, and a scene you wrote in a text editor loads in the browser.


THE REGISTRY
------------
`ENGINES` is a hand-written dict. There is no auto-discovery and no plugin loader, on
purpose: one engine does not need a framework, and an explicit dict is one line to read
and one line to extend. To add a third engine, write the module and add it here.
"""
from __future__ import annotations

import json
import os

import bio as _bio

# Bio_Lab was split out of Engine_Kit on 2026-08-27 and carries the BIO engine only, so the
# registry has one entry. Engine_Kit's `objects` engine stays there; the shape of this dict
# is unchanged so a second engine is still one line to add.
ENGINES = {
    _bio.NAME: _bio,
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SCENE_DIR = os.path.join(ROOT, "scenes")


def engine_info():
    """What the UI needs to draw the bubbles and the sidebar, for every engine."""
    from .params import schema
    out = []
    for name, m in ENGINES.items():
        out.append({
            "name": name,
            "title": m.TITLE,
            "blurb": m.BLURB,
            "params": schema(m.PARAMS),
            "has_objects": hasattr(m, "OBJECT_DEFAULTS"),
            "object_defaults": getattr(m, "OBJECT_DEFAULTS", None),
        })
    return out


def new_scene(engine: str) -> dict:
    from .params import defaults
    m = ENGINES[engine]
    d = {"title": "Untitled", "engine": engine, "seconds": 10.0, "seed": 7,
         "zoom": 1.0, "centre": [0.0, 0.0], "params": defaults(m.PARAMS)}
    if hasattr(m, "default_objects"):
        d["objects"] = m.default_objects()
    return d


def build_engine(scene: dict):
    """Instantiate the engine a scene names. The one place scenes turn into objects."""
    name = scene.get("engine")
    if name not in ENGINES:
        raise KeyError("unknown engine %r; have %s" % (name, sorted(ENGINES)))
    m = ENGINES[name]
    kw = {"params": scene.get("params") or {}, "seed": int(scene.get("seed", 7))}
    if hasattr(m, "OBJECT_DEFAULTS"):
        kw["objects"] = scene.get("objects")
    # A scene may drive parameters over time. Only offered to engines that advertise the
    # facility, so the shape of this function stays the same for one that does not have it.
    if scene.get("ramps"):
        if not getattr(m, "SUPPORTS_RAMPS", False):
            raise ValueError("engine %r does not support a 'ramps' block" % name)
        kw["ramps"] = scene["ramps"]
    return m.make(**kw)


def load_scene(path: str) -> dict:
    p = path
    if not os.path.isfile(p):
        alt = os.path.join(SCENE_DIR, path if path.endswith(".json") else path + ".json")
        if os.path.isfile(alt):
            p = alt
        else:
            raise FileNotFoundError("no scene at %r or %r" % (path, alt))
    with open(p, encoding="utf8") as f:
        d = json.load(f)
    if "engine" not in d:
        raise ValueError("%s has no 'engine' key" % p)
    d["_path"] = p
    return d


def save_scene(scene: dict, path: str = None) -> str:
    d = {k: v for k, v in scene.items() if not k.startswith("_")}
    p = path or scene.get("_path")
    if not p:
        safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                       for c in str(scene.get("title", "untitled")).lower())
        p = os.path.join(SCENE_DIR, (safe or "untitled") + ".json")
    if not os.path.isabs(p):
        p = os.path.join(SCENE_DIR, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf8") as f:
        json.dump(d, f, indent=2)
    return p


def list_scenes():
    if not os.path.isdir(SCENE_DIR):
        return []
    out = []
    for f in sorted(os.listdir(SCENE_DIR)):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(SCENE_DIR, f), encoding="utf8") as fh:
                d = json.load(fh)
            out.append({"file": f, "title": d.get("title", f),
                        "engine": d.get("engine", "?"),
                        "seconds": d.get("seconds", 0)})
        except Exception as e:
            out.append({"file": f, "title": "UNREADABLE: %s" % e, "engine": "?",
                        "seconds": 0})
    return out
