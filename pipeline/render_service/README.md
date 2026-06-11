# render_service — host-side manim render bridge for n8n

n8n runs in a Linux Docker container and can't invoke the host's manim (it needs
the project `.venv` + MiKTeX + bundled ffmpeg, run non-elevated). This tiny
stdlib HTTP service is the bridge: n8n calls it, it locates an existing render or
runs one, and copies the finished MP4 into `..\incoming\` (mounted into the
container at `/incoming`).

## Start it

```powershell
# NON-administrator PowerShell (MiKTeX won't compile LaTeX when elevated)
cd "C:\Users\ethan\Desktop\Research_Library_Scripts\n8n\render_service"
.\run.ps1                       # http://0.0.0.0:8765, quality -qh
.\run.ps1 -Quality l            # fast low-res (good for testing the render path)
```

Leave it running while you use the posting pipeline. Stop with Ctrl+C (or close
the window).

## Auth
`token.txt` holds a shared secret. Every request must send header
`X-Token: <contents of token.txt>`. Delete the file to disable auth.

## Endpoints
| method | path | purpose |
|--------|------|---------|
| GET  | `/health` | paths + venv/MiKTeX sanity |
| GET  | `/molecules` | renderable scene classes + molecule names (66 total) |
| POST | `/ensure` | body `{scene_class}` or `{molecule, template?}`; stages MP4 to `incoming/`. Add `"force":true` to re-render, `"async":true` for a job id. |
| GET  | `/jobs/<id>` | poll an async render |

`/ensure` is **synchronous by default** — if a render is needed it blocks for
minutes, so raise the n8n HTTP Request node's timeout. Use `"async":true` +
`/jobs/<id>` polling for long renders.

## How n8n reaches it
From the container, the host is `host.docker.internal`. Verified:
`http://host.docker.internal:8765/ensure`. The staged file appears at
`/incoming/<scene_class>.mp4` inside the container.

## Mapping
- molecule → `Stacked_<CamelCase>` (e.g. "vitamin c" → `Stacked_VitaminC`).
- Renders go to `manim\renders\videos\spectro_scenes\<res>\<class>.mp4`; the
  service copies the newest match into `incoming\<class>.mp4`.
