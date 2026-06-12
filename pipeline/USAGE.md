# Spectrometry Shorts â€” Pipeline Usage (detailed)

The single source of truth for **what this pipeline is, how to run it, and how to
maintain it**. Read this first when picking the project back up.

---

## 1. What it does

You type molecule names into a form; the pipeline **finds-or-renders** each
molecule's vertical (9:16) manim spectroscopy video, writes an AI caption, bakes
in a royalty-free music track, and **posts to the platforms you tick** â€” plus
deposits an audio-free copy into a Google Drive folder for manual TikTok posting.
Everything runs locally, orchestrated by self-hosted n8n.

---

## 2. Components (where everything lives)

| Piece | What | Location / address |
|---|---|---|
| **n8n** (Docker, +ffmpeg) | Orchestrator | `localhost:5678` (login `<your-user>`) â€” also `https://<your-tunnel-host>:8443/` |
| **Render service** | Renders/locates manim videos, stages to `/incoming`. **Renders 9:16 vertical** (`-r 1080,1920`). | host Python, port 8765, `n8n\render_service\` |
| **manim project** | The 43 `Stacked_<Molecule>` scenes | `Video Rendering\Chemistry\manim\spectro_scenes.py` |
| **Claude (Anthropic)** | AI titles/descriptions/tags | n8n "Header Auth" credential |
| **`/music`** | Royalty-free tracks to mux (â‰ˆ45) | `n8n\music\` |
| **Google Drive** | TikTok deposit folder + dedup | Folder 1 = `<DRIVE_FOLDER_ID>` |
| **Tailscale Funnel** | Public HTTPS for OAuth + music page | `â€¦<your-tunnel-host>:8443` |
| **Repo** | Legal pages + sanitized tooling | `github.com/<you>/<repo>` |

---

## 3. Day-to-day usage

### Start it (after a reboot)
Run in a **non-admin** PowerShell:
```
<repo>/pipeline\start_pipeline.ps1
```
Brings up Docker â†’ n8n â†’ render service â†’ funnel and prints the URLs.
Stop with `stop_pipeline.ps1` (or `docker compose down`).

### Publish a batch
1. Open **http://localhost:5678** â†’ open the **`Shorts_Uploader`** workflow.
2. Click **Execute** â†’ fill the form:
   - **Molecules** â€” free text; optionally `<molecule> and <song>` per item, comma/newline separated. e.g. `Dopamine and lofi chill, Serotonin, CBD`
   - **Template** â€” `Stacked (all methods)` (the others are placeholders).
   - **YouTube visibility** â€” `Draft (private)` / `Unlisted` / `Public (post)`.
   - **Platforms** â˜‘ â€” `YouTube` / `X` / `Reels` / `TikTok` (default all off â€” tick what you want).
3. Submit. Only the **ticked** platforms post; each branch is independent (one
   failing never blocks the others).

### What each platform does
- **YouTube** â€” uploads the muxed (audio) video via the Data API.
- **X** â€” chunked-uploads + posts the muxed video (pay-per-use â‰ˆ$0.01/post).
- **Reels** â€” placeholder (Meta shelved).
- **TikTok** â€” does NOT API-post; **deposits the silent original** into Drive
  Folder 1 (replacing any same-named file) so you publish by hand with a real
  trending sound.

### Refresh the music library (occasional)
Open **http://localhost:5678/webhook/music-find** (or the funnel URL) â†’ a page of
popular royalty-free Jamendo tracks with audio previews + checkboxes â†’ tick â†’
they download into `/music`. Use a track by typing its filename (no `.mp3`) as a
song. With no song, the mux picks a track that matches; otherwise the video keeps
its original (silent) audio.

---

## 4. The flow (per molecule)

```
Form â†’ AI parse â†’ AI caption â†’ Build metadata â†’ Render: ensure (renders if missing)
     â†’ Mux audio â†’ Read MP4 â†’ â”¬ Post YouTube?  â†’ Upload â†’ YouTube
                              â”œ Post X?        â†’ Post to X
                              â”œ Post Reels?    â†’ (placeholder)
                              â”” Post TikTok?   â†’ find existing in Drive â†’
                                                 replace-or-upload original to Folder 1
```
Render-on-demand, AI captions, audio mux, checkbox gating, replace-in-place Drive
deposit, and clear `VIDEO POSTING LIMIT` vs `RATE/USAGE LIMIT` error labels are
all built in.

---

## 5. Platform status & limits

| Platform | State | Limit / note |
|---|---|---|
| YouTube | âœ… live | daily upload cap (low for young channels; resets ~24h). OAuth in "Testing" â†’ re-sign-in ~weekly. |
| X | âœ… live | pay-per-use â‰ˆ$0.01/post â€” keep **credits** topped up in developer.x.com. |
| TikTok | âœ… via Drive | manual publish from Folder 1 (app awaiting audit for API auto-post). |
| Reels/IG | â¸ï¸ shelved | needs $12 verification + app review + public hosting. |

---

## 6. Bulk posting (outside the form)

Ad-hoc bulk posts are done with container node scripts that reuse the proven X
OAuth1 upload + ffmpeg mux (e.g. post every rendered molecule). Pattern:
`docker exec n8n-shorts node /files/<script>.js` (keys come from `.env`).

---

## 7. Maintenance / troubleshooting

- **After changing a credential** â†’ `docker compose restart n8n` (n8n caches creds).
- **Webhook (music page) not live** â†’ `n8n update:workflow --id=<id> --active=true` then restart n8n.
- **Horizontal videos** â†’ the render service must pass `-r 1080,1920`; cached
  landscape renders need `{"force":true}` on `/ensure` to re-render vertical.
- **Drive "unauthorized" on OAuth** â†’ do the sign-in via the **funnel URL** (origin must match `N8N_EDITOR_BASE_URL`).
- **Code nodes** need `NODE_FUNCTION_ALLOW_BUILTIN=*` + `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`; the sandbox lacks `fetch`/`URL` (require them from `https`/`url`).
- **Inspect a failed run** â†’ decode `execution_data.data` with `flatted` (read-only).

---

## 8. Pending / future

- Publish the Google app to **Production** (stop weekly YouTube re-auth).
- Wire TikTok API auto-post once the app is audited.
- Decide on Instagram/Reels ($12 + review + funnel hosting).
- **New molecules** (e.g. nifedipine, felodipine) need new `Stacked_<Molecule>`
  manim scenes (structure + per-method spectral data) before they can render.
- Future video formats â†’ new folders under Drive Folder 2
  (`<DRIVE_FOLDER2_ID>`).


