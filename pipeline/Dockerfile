# n8n + ffmpeg, so the "add a sound" step can mux audio into your MP4s
# right inside the workflow (Execute Command node) without a host install.
#
# The n8n base image is Alpine but ships WITHOUT a package manager (apk was
# stripped for hardening), so we can't `apk add ffmpeg`. Instead we copy a
# self-contained static ffmpeg/ffprobe from a dedicated build image.
FROM mwader/static-ffmpeg:latest AS ffmpeg

FROM n8nio/n8n:latest
USER root
COPY --from=ffmpeg /ffmpeg  /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe
USER node
