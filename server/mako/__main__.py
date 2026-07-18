"""The "Build a Mako" dev server — `python -m server.mako` from the repo root.

Serves `web/` statically on :8077 (with correct MIME types for the WASM
bundle files) and implements the SPEC's API:

    POST /api/mako/spawn        design.json → compile (or cache-hit) → meta

Compile artifacts land in `web/mako/builds/<hash12>/` (gitignored) and are
served statically from there; a dir with a `meta.json` is a complete bundle
and is returned without recompiling. Concurrent identical requests share a
per-hash lock. Stdlib only — the heavy lifting is `builder.py` (manta).
"""

from __future__ import annotations

import json
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .builder import build_bundle, design_hash, validate_design

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = ROOT / "web"
BUILDS_DIR = WEB_DIR / "mako" / "builds"
PORT = 8077

# One lock per design hash so concurrent identical spawns build once;
# `_locks_guard` protects the dict itself.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(hash12: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(hash12, threading.Lock())


class MakoHandler(SimpleHTTPRequestHandler):
    """Static `web/` file server + the spawn endpoint."""

    # ES modules and WASM need exact MIME types or the browser refuses them.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    # ---- API ---------------------------------------------------------------

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/mako/spawn":
            self._json(404, {"ok": False, "errors": ["unknown endpoint"]})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            design = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._json(200, {"ok": False,
                             "errors": [f"design is not valid JSON: {e}"]})
            return

        canonical, errors = validate_design(design)
        if errors:
            self._json(200, {"ok": False, "errors": errors})
            return

        name = str(design.get("name") or "mako")[:80] if isinstance(
            design, dict) else "mako"
        hash12 = design_hash(canonical)
        out_dir = BUILDS_DIR / hash12
        meta_path = out_dir / "meta.json"

        def _respond_cached() -> bool:
            if not meta_path.exists():
                return False
            meta = json.loads(meta_path.read_text())
            meta["name"] = name          # cosmetic; not part of the hash
            self._json(200, {"ok": True, "base": f"/mako/builds/{hash12}/",
                             "meta": meta})
            return True

        if _respond_cached():
            return
        if not shutil.which("emcc"):
            self._json(200, {"ok": False, "errors": [
                "emcc is not on PATH — `source ~/emsdk/emsdk_env.sh` and "
                "restart the server"]})
            return

        with _lock_for(hash12):
            if _respond_cached():        # a racing request built it first
                return
            try:
                meta = build_bundle(canonical, out_dir, name=name,
                                    compile=True)
            except Exception as e:       # emission or emcc failure
                self._json(500, {"ok": False,
                                 "errors": [f"compile stage failed: {e}"]})
                return
        self._json(200, {"ok": True, "base": f"/mako/builds/{hash12}/",
                         "meta": meta})

    # ---- plumbing ------------------------------------------------------------

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than the default: skip per-file GET spam, keep the rest.
        if args and str(args[0]).startswith("GET") and " 200 " in str(args):
            return
        super().log_message(fmt, *args)


def main() -> None:
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("", PORT), MakoHandler)
    emcc = "found" if shutil.which("emcc") else \
        "MISSING (source ~/emsdk/emsdk_env.sh)"
    print(f"mako compile server on http://localhost:{PORT}/  "
          f"(web root {WEB_DIR}, emcc {emcc})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
