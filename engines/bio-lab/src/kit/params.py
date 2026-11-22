# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""params.py - the parameter schema. ONE declaration drives three things.

An engine declares a list of `P(...)` specs. From that single list you get:

1. the **sidebar** in the browser - every control, its range, its grouping, its help text;
2. the **defaults** a scene starts from;
3. **validation** - a scene file with a typo or an out-of-range value is rejected with a
   readable message instead of producing a black video forty minutes later.

This is the reason adding a parameter is a one-line change. Add a `P(...)` to the engine's
`PARAMS`, read `self.p["your_name"]` in the sim, and the slider appears with no HTML, no
JavaScript and no server change. **That is the main thing to know before you extend
either engine.**

TYPES
-----
    float   slider. Needs `lo`, `hi`; `step` optional (defaults to 1/200 of the range).
    int     slider, integer steps.
    bool    toggle.
    choice  dropdown. Needs `choices`.
    colour  colour picker. Value is "#rrggbb".
    text    free text box. For the few values that are a LIST rather than one choice -
            `palettes` names one ramp per culture, and a dropdown cannot say "these
            three, in this order".
    image   file picker over `assets/<folder>/`, plus an upload button that puts new
            files in that same folder. Value is a path relative to assets/, or "" for
            none. Set `folder="backgrounds"` for background pickers.

GROUPS
------
`group` is the sidebar section heading. Order of first appearance sets section order, so
put the parameters someone actually reaches for in the first group. A group named
"Advanced" is collapsed by default in the UI.
"""
from __future__ import annotations

TYPES = ("float", "int", "bool", "choice", "colour", "image", "text")


class P:
    """One parameter. Keep `help` short - it renders as a tooltip, not a manual."""

    __slots__ = ("name", "type", "default", "lo", "hi", "step", "choices",
                 "label", "group", "help", "folder")

    def __init__(self, name, type="float", default=0.0, lo=None, hi=None, step=None,
                 choices=None, label=None, group="Main", help="", folder="sprites"):
        if type not in TYPES:
            raise ValueError("bad param type %r for %r; have %s" % (type, name, TYPES))
        if type in ("float", "int") and (lo is None or hi is None):
            raise ValueError("numeric param %r needs lo and hi" % name)
        if type == "choice" and not choices:
            raise ValueError("choice param %r needs choices" % name)
        self.name = name
        self.type = type
        self.default = default
        self.lo = lo
        self.hi = hi
        self.step = step if step is not None else (
            1 if type == "int" else (round((hi - lo) / 200.0, 6) if lo is not None else None))
        self.choices = list(choices) if choices else None
        self.label = label or name.replace("_", " ")
        self.group = group
        self.help = help
        #: For `image` params only: which assets/ subfolder the picker lists and uploads
        #: into. Without this every upload landed in assets/sprites/, so a background
        #: image could be chosen but never added.
        self.folder = folder

    def to_dict(self):
        d = {"name": self.name, "type": self.type, "default": self.default,
             "label": self.label, "group": self.group, "help": self.help}
        if self.lo is not None:
            d.update(lo=self.lo, hi=self.hi, step=self.step)
        if self.choices:
            d["choices"] = self.choices
        if self.type == "image":
            d["folder"] = self.folder
        return d

    def coerce(self, v):
        """Convert and clamp one incoming value, raising on anything unusable."""
        if self.type == "text":
            return "" if v is None else str(v)
        if self.type == "float":
            f = float(v)
            if f != f:
                raise ValueError("%s is NaN" % self.name)
            return min(max(f, self.lo), self.hi)
        if self.type == "int":
            return int(min(max(round(float(v)), self.lo), self.hi))
        if self.type == "bool":
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        if self.type == "choice":
            s = str(v)
            if s not in self.choices:
                raise ValueError("%s must be one of %s, got %r"
                                 % (self.name, self.choices, s))
            return s
        if self.type == "colour":
            s = str(v).strip()
            if not (len(s) == 7 and s[0] == "#"):
                raise ValueError("%s must be '#rrggbb', got %r" % (self.name, v))
            int(s[1:], 16)                      # raises if not hex
            return s.lower()
        return str(v)


def defaults(specs) -> dict:
    return {p.name: p.default for p in specs}


def schema(specs) -> list:
    return [p.to_dict() for p in specs]


def coerce_all(specs, values, strict: bool = False) -> dict:
    """Merge `values` onto the defaults, coercing and clamping each.

    Unknown keys are DROPPED with a note rather than raising, unless `strict`. That is
    deliberate: a scene file written against a newer version of an engine should still
    open in an older one, minus the parameter that does not exist yet. Silently accepting
    them into the params dict would be worse - the sim would ignore them and the user
    would think they had taken effect.
    """
    by_name = {p.name: p for p in specs}
    out = defaults(specs)
    unknown = []
    for k, v in (values or {}).items():
        p = by_name.get(k)
        if p is None:
            unknown.append(k)
            continue
        out[k] = p.coerce(v)
    if unknown:
        msg = "ignored %d unknown parameter(s): %s" % (len(unknown), ", ".join(sorted(unknown)))
        if strict:
            raise ValueError(msg)
        print("[params] " + msg)
    return out


def hex_to_rgb(s) -> tuple:
    s = str(s).lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
