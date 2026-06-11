# n8n — short-form video batch pipeline

Self-hosted n8n (community edition) for batching short-form videos to
Instagram Reels, TikTok, and YouTube Shorts.

## Run it

```powershell
cd "C:\Users\ethan\Desktop\Research_Library_Scripts\n8n"
docker compose up -d      # start
docker compose logs -f    # watch logs
docker compose down       # stop (keeps data in the n8n_data volume)
docker compose build      # rebuild after editing the Dockerfile
```

- **Editor UI:** http://localhost:5678
- **Login:** user `ethan`, password is in `.env` (`N8N_BASIC_AUTH_PASSWORD`).
- State (workflows, credentials, encryption key) lives in the `n8n_data`
  Docker volume — survives `down`/`up`. Credentials are encrypted with
  `N8N_ENCRYPTION_KEY` in `.env`; **back that key up**, losing it makes saved
  platform tokens unreadable.

## Folder layout

| Path | Container path | Purpose |
|------|----------------|---------|
| `./local-files` | `/files` | working dir — drop source MP4s, muxed output lands here |
| `./incoming`    | `/incoming` | render hand-off — `render_service` stages finished MP4s here |
| `../` (project root) | `/library` (read-only) | whole `Research_Library_Scripts` tree, so n8n can read source renders directly (manim outputs at `/library/Video Rendering/Chemistry/manim/renders/...`) |

File reads in workflows are whitelisted to `/files;/incoming;/library`
(`N8N_RESTRICT_FILE_ACCESS_TO`). Anything outside those is blocked by n8n.

ffmpeg 8.1.1 + ffprobe are baked into the image (see `Dockerfile`), so the
"add a sound" step runs via an **Execute Command** node — no host install.

## Pipeline (planned)

1. **Ingest** — pick up rendered MP4s (e.g. from
   `Video Rendering\...\renders`) into `/files`.
2. **Add sound** — ffmpeg muxes a royalty-free / owned track onto the video.
   ⚠️ Native IG/TikTok *trending* audio CANNOT be added via API — those
   libraries are licensed and only attach from inside the phone apps. API
   posting only accepts audio already baked into the file.
3. **AI metadata** — an LLM node generates title / description / hashtags per
   video so you don't type them by hand.
4. **Post** — platform nodes upload + publish:
   - **YouTube Shorts:** YouTube Data API v3 (built-in n8n node).
   - **Instagram Reels:** Meta Graph API (Business/Creator acct + FB Page).
   - **TikTok:** Content Posting API (approved developer app).

## Workflows

- **YT Shorts — single upload (test)** — minimal smoke test (manual → read → upload).
- **Batch shorts — molecules → multi-platform** — the real pipeline:
  form (list molecules + template) → AI parse → AI caption/tags → render-if-missing
  (`render_service`) → read MP4 → parallel post (YouTube live; TikTok/Reels scaffolded).
  Regenerate with `node workflows/build_workflow.js`, then re-import via
  `docker exec n8n-shorts n8n import:workflow --input=/files/<staged>.json`.
  Uploads are **Private** until you flip `privacyStatus` in the YouTube node.

## Status

- [x] n8n + ffmpeg running in Docker (WSL2 backend)
- [x] YouTube connected (OAuth, Testing mode — refresh weekly)
- [x] Anthropic key connected (Header Auth credential)
- [x] Host render bridge built + verified (`render_service/`)
- [x] Batch workflow built + imported
- [ ] End-to-end test run
- [ ] TikTok + Instagram Reels credentials + public hosting (Tailscale)
