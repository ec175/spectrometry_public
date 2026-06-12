# spectrometry_public

Public home for **Spectrometry Shorts** — a personal, self-hosted pipeline that
batches short-form educational chemistry videos and publishes them to social
platforms (YouTube, with TikTok and Instagram Reels in progress). It also holds
the creator/end-user licensing documents the platform developer apps require.

This repo holds two things:

### 1. Legal pages (GitHub Pages)
Privacy Policy and Terms of Service for the automation app, served at:

- Privacy: <https://ec175.github.io/spectrometry_public/privacy.html>
- Terms:   <https://ec175.github.io/spectrometry_public/terms.html>

These are the URLs referenced by the social-platform developer apps.

### 2. Pipeline tooling — [`pipeline/`](pipeline/)

**📖 [pipeline/USAGE.md](pipeline/USAGE.md)** — the detailed how-to: workflows,
the form, every platform, music + Drive integrations, limits, troubleshooting.

A sanitized copy of the automation stack:

- **`docker-compose.yml` / `Dockerfile`** — self-hosted [n8n](https://n8n.io)
  with `ffmpeg` baked in.
- **`render_service/`** — a small stdlib HTTP service that renders/locates the
  source videos on the host and stages them for n8n (the host↔container bridge).
- **`workflows/build_workflow.js`** — generates the n8n batch workflow
  (form → AI captions → render-if-missing → multi-platform post) as importable
  JSON.

#### Configuration
Nothing secret lives here. Copy `pipeline/.env.example` to `.env` and fill in
your own values (encryption key, basic-auth password, tunnel URL). Credential
IDs and the render-service token are read from environment variables — see the
top of `build_workflow.js`.

```
cp pipeline/.env.example pipeline/.env   # then edit
cd pipeline && docker compose up -d
```

> Personal project. The published videos and any source scene code are not part
> of this repository.
