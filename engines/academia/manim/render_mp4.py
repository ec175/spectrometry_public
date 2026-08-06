"""
render_mp4.py — pixel-perfect mp4 of nest_egg.html with one slider swept.

Drives the actual HTML in headless Chromium, sets a master slider across a range
of values frame-by-frame, screenshots one of the fixed-size render stages, and
encodes to mp4.

Stages / formats
----------------
--format youtube  -> #stageYt, 1920x1080 (sliders + graph side by side)   [default]
--format short    -> #stageSf, 1080x1920 (graph over center, sliders below)

Examples
--------
# default demo: inflation 0 -> 10% and back, YouTube 1080p
python render_mp4.py

# shortform version of the same sweep
python render_mp4.py --format short --out inflation_short.mp4

# monthly withdrawal sweep, one direction, holding inflation at 4
python render_mp4.py --param take --from 0 --to 20000 --seconds 6 --no-pingpong --hold inf=4 --out withdrawal.mp4

Slider ids: retLow, retHigh, inf, con, gr (gradient growth), ageStart, ageEnd,
            retire (retirement age), take (monthly withdrawal), grad (checkbox 0/1)
"""
import argparse, os
import numpy as np
import imageio.v2 as imageio
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = "file:///" + os.path.join(HERE, "nest_egg.html").replace("\\", "/")

STAGES = {"youtube": "#stageYt", "short": "#stageSf"}

def set_value(page, param, v):
    """Set a master slider/checkbox and fire its event so draw() runs."""
    if param == "grad":
        page.evaluate(
            "(v)=>{const e=document.getElementById('grad');"
            "e.checked=!!v;e.dispatchEvent(new Event('change'));}", int(v))
    else:
        page.evaluate(
            "([id,v])=>{const e=document.getElementById(id);"
            "e.value=v;e.dispatchEvent(new Event('input'));"
            "e.dispatchEvent(new Event('change'));}", [param, v])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="inf", help="master slider id to sweep")
    ap.add_argument("--from", dest="lo", type=float, default=0.0)
    ap.add_argument("--to", dest="hi", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=5.0, help="sweep duration (one direction)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--format", choices=list(STAGES), default="youtube",
                    help="render stage / aspect ratio")
    ap.add_argument("--pingpong", dest="pingpong", action="store_true", default=True)
    ap.add_argument("--no-pingpong", dest="pingpong", action="store_false")
    ap.add_argument("--hold", action="append", default=[],
                    help="other sliders to fix, e.g. --hold inf=4 --hold grad=1")
    ap.add_argument("--steady", dest="steady", action="store_true", default=True,
                    help="lock the y-axis so it doesn't rescale during the sweep (default on)")
    ap.add_argument("--no-steady", dest="steady", action="store_false")
    ap.add_argument("--ymax", type=float, default=None,
                    help="force a specific y-axis max (overrides --steady auto)")
    ap.add_argument("--scale", type=int, default=1,
                    help="device scale factor (1 = native; canvas already renders at 2x)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    holds = [(k, float(v)) for k, _, v in (h.partition("=") for h in args.hold)]

    n = max(2, int(round(args.seconds * args.fps)))
    up = np.linspace(args.lo, args.hi, n)
    values = np.concatenate([up, up[::-1]]) if args.pingpong else up

    out_name = args.out or f"nest_egg_{args.param}_{args.format}.mp4"
    out = out_name if os.path.isabs(out_name) else os.path.join(HERE, out_name)
    sel = STAGES[args.format]

    print(f"Loading {HTML}")
    print(f"Format '{args.format}' ({sel}); sweeping '{args.param}' {args.lo}->{args.hi} "
          f"({'ping-pong, ' if args.pingpong else ''}{len(values)} frames @ {args.fps}fps)")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 2000, "height": 1200},
                                device_scale_factor=args.scale)
        page.goto(HTML)
        page.wait_for_function("typeof draw === 'function'")
        page.evaluate("()=>{fitAll(); draw();}")
        for k, val in holds:
            set_value(page, k, val)

        # steady y-axis
        if args.ymax is not None:
            page.evaluate("(m)=>{window.FIXED_YMAX=m; draw();}", args.ymax)
            print(f"  y-axis fixed at {args.ymax:,.0f}")
        elif args.steady:
            # lock to the tallest nominal peak across the sweep, with a little headroom
            mx = 0.0
            for v in up:
                set_value(page, args.param, float(v))
                m = page.evaluate("()=>window.__dataMaxY") or 0
                mx = max(mx, m)
            ymax = mx * 1.06
            page.evaluate("(m)=>{window.FIXED_YMAX=m;}", ymax)
            print(f"  steady y-axis locked at peak: {ymax:,.0f}")

        stage = page.locator(sel)
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=None)
        try:
            for i, v in enumerate(values):
                set_value(page, args.param, float(v))
                png = stage.screenshot()           # exact-dimension element capture
                arr = imageio.imread(png)[..., :3]
                h, w = arr.shape[:2]               # crop to even dims (h264 requirement)
                arr = arr[: h - (h % 2), : w - (w % 2)]
                writer.append_data(arr)
                if i % 20 == 0:
                    print(f"  frame {i+1}/{len(values)}")
        finally:
            writer.close()
            browser.close()

    size = os.path.getsize(out) / 1e6
    print(f"\nWrote {out}  ({size:.1f} MB, {len(values)} frames, "
          f"{len(values)/args.fps:.1f}s)")

if __name__ == "__main__":
    main()
