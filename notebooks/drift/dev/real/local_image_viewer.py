#!/usr/bin/env python3
"""Small local image browser for active drift-development outputs.

Usage:
    python notebooks/drift/dev/real/local_image_viewer.py \
        --root notebooks/drift/dev/outputs --host 0.0.0.0 --port 8765

The server uses only Python's standard library. It lists PNG/JPEG/GIF/WebP/TIFF
files below the root and serves a browser UI with thumbnails, keyboard
navigation, and image metadata.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.tif', '.tiff', '.bmp'}


def _safe_relative(root: Path, rel: str) -> Path:
    root = root.resolve()
    candidate = (root / unquote(rel)).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError('path escapes image root')
    return candidate


def _scan_images(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        rows.append({'path': rel, 'name': path.name, 'size': stat.st_size})
    return rows


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>QuantEM Drift Image Viewer</title>
<style>
:root { color-scheme: dark; --bg: #101214; --panel: #171a1d; --line: #2b3035; --text: #e7eaee; --muted: #9aa4af; --accent: #77b7ff; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }
.app { display: grid; grid-template-columns: 330px 1fr; min-height: 100vh; }
aside { border-right: 1px solid var(--line); background: var(--panel); min-height: 100vh; overflow: auto; }
header { position: sticky; top: 0; z-index: 2; padding: 12px; background: var(--panel); border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 8px; font-size: 16px; font-weight: 650; }
input { width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--line); background: #0d0f11; color: var(--text); }
.list { padding: 8px; display: grid; gap: 6px; }
.item { display: grid; grid-template-columns: 56px 1fr; gap: 9px; align-items: center; padding: 6px; border: 1px solid transparent; border-radius: 7px; cursor: pointer; color: inherit; text-decoration: none; }
.item:hover, .item.active { border-color: var(--accent); background: #111923; }
.thumb { width: 56px; height: 44px; object-fit: cover; background: #050607; border: 1px solid var(--line); border-radius: 4px; }
.name { font-size: 12px; line-height: 1.25; overflow-wrap: anywhere; }
.meta { color: var(--muted); font-size: 11px; margin-top: 3px; }
main { display: grid; grid-template-rows: auto 1fr; min-width: 0; }
.toolbar { display: flex; gap: 10px; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); background: #0d0f11; }
button { background: #1b222a; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; cursor: pointer; }
button:hover { border-color: var(--accent); }
.path { color: var(--muted); font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stage { overflow: auto; display: grid; place-items: start center; padding: 18px; }
.stage.fit { place-items: center; }
#image { max-width: none; max-height: none; image-rendering: auto; background: #050607; }
.stage.fit #image { max-width: 100%; max-height: calc(100vh - 78px); width: auto; height: auto; object-fit: contain; }
.empty { color: var(--muted); padding: 24px; }
@media (max-width: 900px) { .app { grid-template-columns: 1fr; } aside { min-height: 36vh; max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--line); } }
</style>
</head>
<body>
<div class=\"app\">
<aside>
<header>
<h1>QuantEM Drift Image Viewer</h1>
<input id=\"filter\" placeholder=\"Filter images\" autocomplete=\"off\" />
<div class=\"meta\" id=\"count\"></div>
</header>
<div class=\"list\" id=\"list\"></div>
</aside>
<main>
<div class=\"toolbar\">
<button id=\"prev\">Prev</button>
<button id=\"next\">Next</button>
<button id=\"fit\">Fit</button>
<button id=\"actual\">1:1</button>
<div class=\"path\" id=\"path\"></div>
</div>
<div class=\"stage fit\" id=\"stage\"><div class=\"empty\">Loading images...</div></div>
</main>
</div>
<script>
const IMAGES = __IMAGES__;
let filtered = IMAGES.slice();
let idx = 0;
let fit = true;
const list = document.getElementById('list');
const filter = document.getElementById('filter');
const count = document.getElementById('count');
const stage = document.getElementById('stage');
const pathLabel = document.getElementById('path');
function fmt(bytes) { if (bytes < 1024) return bytes + ' B'; if (bytes < 1048576) return (bytes/1024).toFixed(1)+' KB'; return (bytes/1048576).toFixed(1)+' MB'; }
function imageUrl(p) { return '/image?path=' + encodeURIComponent(p); }
function renderList() {
  list.innerHTML = '';
  filtered.forEach((row, i) => {
    const a = document.createElement('a'); a.className = 'item' + (i === idx ? ' active' : ''); a.href = '#';
    const img = document.createElement('img'); img.className = 'thumb'; img.src = imageUrl(row.path);
    const wrap = document.createElement('div');
    const name = document.createElement('div'); name.className = 'name'; name.textContent = row.path;
    const meta = document.createElement('div'); meta.className = 'meta'; meta.textContent = fmt(row.size);
    wrap.append(name, meta); a.append(img, wrap);
    a.onclick = (e) => { e.preventDefault(); idx = i; render(); };
    list.append(a);
  });
  count.textContent = filtered.length + ' image' + (filtered.length === 1 ? '' : 's');
}
function renderImage() {
  if (!filtered.length) { stage.innerHTML = '<div class=\"empty\">No matching images</div>'; pathLabel.textContent = ''; return; }
  idx = Math.max(0, Math.min(idx, filtered.length - 1));
  const row = filtered[idx];
  stage.className = 'stage' + (fit ? ' fit' : '');
  stage.innerHTML = '';
  const img = document.createElement('img'); img.id = 'image'; img.src = imageUrl(row.path); img.alt = row.path;
  stage.append(img); pathLabel.textContent = row.path + ' · ' + fmt(row.size);
}
function render() { renderList(); renderImage(); document.querySelector('.item.active')?.scrollIntoView({block: 'nearest'}); }
function step(delta) { if (!filtered.length) return; idx = (idx + delta + filtered.length) % filtered.length; render(); }
filter.oninput = () => { const q = filter.value.toLowerCase(); filtered = IMAGES.filter(r => r.path.toLowerCase().includes(q)); idx = 0; render(); };
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('fit').onclick = () => { fit = true; renderImage(); };
document.getElementById('actual').onclick = () => { fit = false; renderImage(); };
document.onkeydown = (e) => { if (e.key === 'ArrowLeft') step(-1); if (e.key === 'ArrowRight') step(1); if (e.key === 'f') { fit = !fit; renderImage(); } };
render();
</script>
</body>
</html>
"""


class ImageViewerHandler(BaseHTTPRequestHandler):
    root: Path
    images: list[dict[str, object]]

    def log_message(self, fmt: str, *args: object) -> None:
        print('%s - %s' % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/':
            body = HTML_PAGE.replace('__IMAGES__', json.dumps(self.images))
            self._send(200, body.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if parsed.path == '/image':
            qs = parse_qs(parsed.query)
            rel = qs.get('path', [''])[0]
            try:
                path = _safe_relative(self.root, rel)
            except ValueError:
                self._send(403, b'forbidden', 'text/plain')
                return
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                self._send(404, b'not found', 'text/plain')
                return
            mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self._send(200, path.read_bytes(), mime)
            return
        self._send(404, b'not found', 'text/plain')

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('notebooks/drift/dev/outputs'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f'image root does not exist: {root}')
    handler = type('ConfiguredImageViewerHandler', (ImageViewerHandler,), {'root': root, 'images': _scan_images(root)})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'serving {len(handler.images)} images from {root}')
    print(f'open http://{args.host}:{args.port}/')
    server.serve_forever()


if __name__ == '__main__':
    main()
