"""
render_service — host-side HTTP bridge so the Dockerized n8n can render manim
molecule videos that don't exist yet, and always get the finished MP4 staged
into n8n's `incoming/` handoff folder.

Why this exists: n8n runs in a Linux container and cannot invoke the host's
manim (separate .venv + MiKTeX + bundled ffmpeg, must run non-elevated). So n8n
calls this service over HTTP (host.docker.internal:8765); the service locates an
existing render or runs one, then copies the result into n8n/incoming.

Stdlib only (no pip installs) so it never collides with the global numpy<2 pin
or the manim venv. Rendering is shelled out to the manim venv's own python.

Endpoints
  GET  /health                      -> service + paths sanity
  GET  /molecules                   -> list of renderable scene classes
  POST /ensure                      -> body: {scene_class | molecule, template?, force?, async?}
                                       sync (default): blocks, returns {status:"ready", file, rendered}
                                       async: returns {job_id}; poll /jobs/<id>
  GET  /jobs/<id>                    -> {status: queued|rendering|ready|error, file?, log_tail}

Auth: optional shared token. If render_service/token.txt exists, every request
must send header  X-Token: <that value>.
"""
import json, os, re, glob, shutil, subprocess, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---- paths (override via env if the tree ever moves) -----------------------
HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT     = os.path.dirname(HERE)                      # ...\Research_Library_Scripts\n8n
ROOT        = os.path.dirname(PROJECT)                   # ...\Research_Library_Scripts
MANIM_DIR   = os.environ.get("MANIM_DIR",
                os.path.join(ROOT, "Video Rendering", "Chemistry", "manim"))
VENV_PY     = os.path.join(MANIM_DIR, ".venv", "Scripts", "python.exe")
SCENES_FILE = os.path.join(MANIM_DIR, "spectro_scenes.py")
FFMPEG_BIN  = os.path.join(MANIM_DIR, "bin")
MIKTEX_BIN  = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                           "Programs", "MiKTeX", "miktex", "bin", "x64")
RENDERS_DIR = os.path.join(MANIM_DIR, "renders", "videos", "spectro_scenes")
INCOMING    = os.path.join(PROJECT, "incoming")
TOKEN_FILE  = os.path.join(HERE, "token.txt")
QUALITY     = os.environ.get("RENDER_QUALITY", "h")     # manim l/m/h/p/k
RENDER_TIMEOUT = int(os.environ.get("RENDER_TIMEOUT", "1800"))   # seconds per render
HOST        = os.environ.get("RENDER_HOST", "0.0.0.0")
PORT        = int(os.environ.get("RENDER_PORT", "8765"))

TOKEN = ""
if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "r", encoding="utf-8") as fh:
        TOKEN = fh.read().strip()

_jobs = {}                 # job_id -> dict(status, file, log, scene_class)
_jobs_lock = threading.Lock()
_render_lock = threading.Lock()   # serialize renders (one GPU/LaTeX at a time)


# ---- helpers ---------------------------------------------------------------
def scene_classes():
    """Every `class X(...)` in spectro_scenes.py that isn't a private helper."""
    try:
        with open(SCENES_FILE, "r", encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return []
    names = re.findall(r"^class\s+([A-Za-z_]\w*)\s*\(", src, re.MULTILINE)
    return [n for n in names if not n.startswith("_")]


def resolve_class(body):
    """Accept an explicit scene_class, or build one from molecule (+template)."""
    sc = (body.get("scene_class") or "").strip()
    if sc:
        return sc
    mol = (body.get("molecule") or "").strip()
    if not mol:
        return ""
    # "vitamin c" -> "VitaminC"; default template is the per-molecule stacked composite
    camel = "".join(p.capitalize() for p in re.split(r"[\s_\-]+", mol) if p)
    tmpl = (body.get("template") or "stacked").strip().lower()
    if tmpl in ("stacked", "", "all", "allmethods"):
        return f"Stacked_{camel}"
    return body.get("scene_class", "")          # method templates are molecule-agnostic


def find_existing(scene_class):
    """Newest rendered MP4 for this class, or None."""
    hits = glob.glob(os.path.join(RENDERS_DIR, "*", f"{scene_class}.mp4"))
    hits = [h for h in hits if "partial_movie_files" not in h]
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def stage(src, scene_class):
    os.makedirs(INCOMING, exist_ok=True)
    dst = os.path.join(INCOMING, f"{scene_class}.mp4")
    shutil.copy2(src, dst)
    return dst


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def run_render(scene_class, log):
    """Shell out to the manim venv. Returns the produced mp4 path or raises."""
    env = dict(os.environ)
    env["PATH"] = FFMPEG_BIN + os.pathsep + MIKTEX_BIN + os.pathsep + env.get("PATH", "")
    cmd = [VENV_PY, "-m", "manim", f"-q{QUALITY}", "--fps", "30",
           SCENES_FILE, scene_class]
    log.append("RUN: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=MANIM_DIR, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=RENDER_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"render timed out after {RENDER_TIMEOUT}s")
    log.extend((proc.stdout or "").splitlines()[-40:])
    if proc.returncode != 0:
        raise RuntimeError(f"manim exited {proc.returncode}")
    out = find_existing(scene_class)
    if not out:
        raise RuntimeError("render finished but no output mp4 found")
    return out


def do_ensure(scene_class, force, log):
    if not force:
        existing = find_existing(scene_class)
        if existing:
            dst = stage(existing, scene_class)
            return {"status": "ready", "rendered": False,
                    "source": existing, "file": dst,
                    "incoming_name": os.path.basename(dst),
                    "size": file_size(dst)}
    with _render_lock:                                  # one render at a time
        out = run_render(scene_class, log)
    dst = stage(out, scene_class)
    return {"status": "ready", "rendered": True,
            "source": out, "file": dst,
            "incoming_name": os.path.basename(dst),
            "size": file_size(dst)}


def worker(job_id, scene_class, force):
    job = _jobs[job_id]
    try:
        job["status"] = "rendering"
        res = do_ensure(scene_class, force, job["log"])
        job.update(res)
    except Exception as e:                              # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)


# ---- HTTP ------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        if not TOKEN:
            return True
        return self.headers.get("X-Token", "") == TOKEN

    def log_message(self, *a):                          # quieter console
        pass

    def do_GET(self):
        if not self._auth_ok():
            return self._send(401, {"error": "bad token"})
        if self.path == "/health":
            return self._send(200, {
                "ok": True,
                "manim_dir": MANIM_DIR,
                "venv_python": VENV_PY,
                "venv_exists": os.path.exists(VENV_PY),
                "scenes_file_exists": os.path.exists(SCENES_FILE),
                "miktex_exists": os.path.exists(os.path.join(MIKTEX_BIN, "latex.exe")),
                "incoming": INCOMING,
                "quality": QUALITY,
            })
        if self.path in ("/molecules", "/scenes"):
            cls = scene_classes()
            return self._send(200, {
                "count": len(cls),
                "scene_classes": cls,
                "molecules": [c[len("Stacked_"):] for c in cls if c.startswith("Stacked_")],
            })
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/jobs/", 1)[1]
            with _jobs_lock:
                job = _jobs.get(jid)
            if not job:
                return self._send(404, {"error": "no such job"})
            return self._send(200, {
                "job_id": jid, "status": job["status"],
                "scene_class": job.get("scene_class"),
                "file": job.get("file"), "rendered": job.get("rendered"),
                "incoming_name": job.get("incoming_name"),
                "size": job.get("size"),
                "error": job.get("error"),
                "log_tail": job["log"][-12:],
            })
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth_ok():
            return self._send(401, {"error": "bad token"})
        if self.path != "/ensure":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or "{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json body"})

        scene_class = resolve_class(body)
        if not scene_class:
            return self._send(400, {"error": "need scene_class or molecule"})
        if scene_class not in scene_classes():
            return self._send(404, {
                "error": f"unknown scene class '{scene_class}'",
                "hint": "GET /molecules for the list",
            })
        force = bool(body.get("force"))

        if body.get("async"):
            jid = uuid.uuid4().hex[:12]
            with _jobs_lock:
                _jobs[jid] = {"status": "queued", "log": [], "scene_class": scene_class}
            threading.Thread(target=worker, args=(jid, scene_class, force),
                             daemon=True).start()
            return self._send(202, {"job_id": jid, "status": "queued",
                                    "scene_class": scene_class})

        # synchronous: blocks until the render is done (raise n8n's HTTP timeout)
        log = []
        try:
            res = do_ensure(scene_class, force, log)
            res["scene_class"] = scene_class
            res["log_tail"] = log[-12:]
            return self._send(200, res)
        except Exception as e:                          # noqa: BLE001
            return self._send(500, {"status": "error", "scene_class": scene_class,
                                    "error": str(e), "log_tail": log[-20:]})


def main():
    print(f"render_service on http://{HOST}:{PORT}  (manim={MANIM_DIR})")
    print(f"  token auth: {'ON' if TOKEN else 'OFF'}   quality: -q{QUALITY}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
