"""
render_scenes.py — keyframe-timeline mp4s of nest_egg.html.

Each scene is a timeline of steps:
  ('move', {slider: target, ...}, seconds)   tween one or more sliders (ease in-out)
  ('hold', seconds)                           freeze on current values
  ('set',  {slider: value, 'grad': bool})     instantaneous change (no frames)

Per scene you choose a y-axis basis:
  'nominal' -> axis fits the blue (nominal) peak
  'real'    -> axis fits the red (real) peak  (blue clips off the top = "zoom on red")
The axis is locked across the whole scene (proportions stay fixed) by measuring the
tallest basis-peak over every frame, then pinning FIXED_YMAX to it + headroom.

Usage:
  python render_scenes.py                       # all 3 scenes, youtube
  python render_scenes.py --format short        # all 3 scenes, shortform
  python render_scenes.py --scene 1 --fps 6     # quick preview of scene 1
"""
import argparse, os
import imageio.v2 as imageio
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = "file:///" + os.path.join(HERE, "nest_egg.html").replace("\\", "/")
STAGES = {"youtube": "#stageYt", "short": "#stageSf"}

# shared starting state for all three renders
DEFAULTS = {"retLow": 5, "retHigh": 8, "inf": 3.8, "con": 50000, "gr": 2,
            "ageStart": 20, "ageEnd": 50, "retire": 50, "take": 8000}
DEFAULT_GRAD = False

# ---- scene timelines -------------------------------------------------------
HOLD = 0.6                                  # brief pause beat
END = [("move", dict(DEFAULTS), 1.2),       # glide every slider back to default
       ("set", {"grad": False}), ("hold", 0.8)]

SCENE1 = [
    ("move", {"retHigh": 10}, 2.0), ("hold", HOLD),
    ("move", {"retHigh": 8},  2.0), ("hold", HOLD),
    ("move", {"inf": 3.0},    1.5), ("hold", HOLD),
    ("move", {"inf": 6.0},    2.0), ("hold", HOLD),
    ("move", {"inf": 3.8},    1.5), ("hold", HOLD),
]
SCENE2 = [
    ("move", {"con": 10000}, 2.5), ("hold", HOLD),
    ("move", {"con": 25000}, 3.0), ("hold", HOLD),         # slowly
    ("set",  {"grad": True, "gr": 0}),
    ("move", {"gr": 7}, 2.5), ("hold", HOLD),
    ("move", {"gr": 5}, 1.5), ("hold", HOLD),
    ("move", {"ageStart": 25}, 1.5), ("hold", HOLD),
    ("move", {"retire": 60, "ageEnd": 60}, 2.5), ("hold", HOLD),   # both at once
]
SCENE3 = [
    ("move", {"retire": 60}, 2.0), ("hold", HOLD),
    ("move", {"retire": 40}, 2.5), ("hold", HOLD),
    ("move", {"retire": 50}, 2.0), ("hold", HOLD),
    ("move", {"take": 6000},  1.5), ("hold", HOLD),
    ("move", {"take": 15000}, 2.5), ("hold", HOLD),
    ("move", {"take": 8000},  2.0), ("hold", HOLD),
]
# (name, steps, y-basis, show_nominal, show_legend)
SCENES = {
    1: ("nest_egg_scene1", SCENE1 + END, "real", True,  True),
    2: ("nest_egg_scene2", SCENE2 + END, "real", False, False),
    3: ("nest_egg_scene3", SCENE3 + END, "real", False, False),
}

def ease(t):  # smooth in-out
    return t * t * (3 - 2 * t)

def build_frames(steps, fps):
    state = dict(DEFAULTS); grad = DEFAULT_GRAD
    frames = []
    for step in steps:
        if step[0] == "hold":
            for _ in range(max(1, round(step[1] * fps))):
                frames.append((dict(state), grad))
        elif step[0] == "set":
            ch = dict(step[1])
            if "grad" in ch:
                grad = bool(ch.pop("grad"))
            state.update(ch)
        elif step[0] == "move":
            targets, dur = step[1], step[2]
            n = max(1, round(dur * fps))
            starts = {k: state[k] for k in targets}
            for i in range(1, n + 1):
                e = ease(i / n)
                for k, tv in targets.items():
                    state[k] = starts[k] + (tv - starts[k]) * e
                frames.append((dict(state), grad))
    return frames

def apply_state(page, values, grad):
    page.evaluate(
        "(st)=>{ for(const k in st.v){ const e=document.getElementById(k); if(e) e.value=st.v[k]; }"
        " document.getElementById('grad').checked=!!st.grad; draw(); }",
        {"v": values, "grad": grad})

def render_scene(page, sel, name, steps, basis, show_nominal, show_legend, fps, fmt):
    frames = build_frames(steps, fps)
    peak_var = "__retireReal" if basis == "real" else "__retireNominal"
    page.evaluate("(o)=>{window.YBASIS=o.b; window.SHOW_NOMINAL=o.n; window.SHOW_DOTLEGEND=o.l;}",
                  {"b": basis, "n": show_nominal, "l": show_legend})

    # measure pass: lock axis to the tallest retirement-dot value across the whole scene
    page.evaluate("()=>{window.FIXED_YMAX=0;}")
    mx = 0.0
    for vals, grad in frames:
        apply_state(page, vals, grad)
        m = page.evaluate(f"()=>window.{peak_var}") or 0
        mx = max(mx, m)
    ymax = mx * 1.12
    page.evaluate("(m)=>{window.FIXED_YMAX=m;}", ymax)

    out = os.path.join(HERE, f"{name}_{fmt}.mp4")
    print(f"  [{name}] basis={basis} locked y={ymax:,.0f}  {len(frames)} frames "
          f"({len(frames)/fps:.1f}s)")
    stage = page.locator(sel)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for i, (vals, grad) in enumerate(frames):
            apply_state(page, vals, grad)
            arr = imageio.imread(stage.screenshot())[..., :3]
            h, w = arr.shape[:2]
            writer.append_data(arr[: h - h % 2, : w - w % 2])
            if i % 60 == 0:
                print(f"     frame {i+1}/{len(frames)}")
    finally:
        writer.close()
    print(f"  wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=list(STAGES), default="youtube")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scene", type=int, choices=[1, 2, 3], default=None,
                    help="render just one scene (default: all)")
    ap.add_argument("--scale", type=int, default=1)
    args = ap.parse_args()

    sel = STAGES[args.format]
    todo = [args.scene] if args.scene else [1, 2, 3]
    print(f"Format {args.format} ({sel}) @ {args.fps}fps; scenes {todo}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 2000, "height": 1200},
                                device_scale_factor=args.scale)
        page.goto(HTML)
        page.wait_for_function("typeof draw === 'function'")
        page.evaluate("()=>{fitAll(); draw();}")
        for s in todo:
            name, steps, basis, show_nominal, show_legend = SCENES[s]
            render_scene(page, sel, name, steps, basis, show_nominal, show_legend,
                         args.fps, args.format)
        browser.close()

if __name__ == "__main__":
    main()
