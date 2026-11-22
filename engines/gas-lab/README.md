# Gas Lab

Follow the motion of gas through animated piston engines. The scene presets include four stroke, diesel, two stroke, and radial layouts.

[All engines](../README.md) · [Video recipes](../../docs/render-recipes.md)

<img src="../../docs/media/gas-lab.webp" width="220" alt="Gas Lab preview">

## Render a scene

With a virtual environment active and ffmpeg installed, run these commands from the repository root.

```sh
cd engines/gas-lab/src
python -m pip install -r requirements.txt
python cli.py single_sohc --preview
```

Try `diesel`, `two_stroke`, or `radial`. Run `python cli.py list` to browse the engine and parts scenes. The `radial_flash` preset ships with its timing plan and requires source audio at `audio/flashing_lights.wav` for the soundtrack.
