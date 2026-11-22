# Repository checks

From the repository root, run the structural checks with Python 3.10 or newer.

```sh
python -m compileall -q engines tools tests
python -m unittest discover -s tests -v
```

The Wind Tunnel solver checks use the Wind Tunnel environment. Run them from its `src` folder after installing its requirements.

```sh
python tools/sod_check.py
python tools/shock_check.py
```

If you edit a catalogue source, regenerate the Markdown with `python tools/build_catalogs.py` from the repository root.
