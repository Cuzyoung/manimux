#!/usr/bin/env python3
"""Serve a model selector for the per-model TensorBoard instances."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODELS = (
    ("Pi05", 1, "Pi05 action flow-matching runs"),
    ("LingBot Action", 2, "Action-only LingBot VLA2 runs"),
    ("LingBot Native Depth", 3, "Depth and video distillation runs"),
    ("Xiaomi Robotics 1", 4, "XR1 flow, frequency and choice losses"),
    ("GR00T N1.7", 5, "GR00T action diffusion runs"),
)


def render_index(base_port: int) -> bytes:
    cards = "\n".join(
        f'<a class="card" data-port="{base_port + offset}" href="#">'
        f"<strong>{name}</strong><span>{description}</span></a>"
        for name, offset, description in MODELS
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YAM Training Dashboard</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#f5f6f8;color:#172033}}
main{{max-width:920px;margin:64px auto;padding:0 24px}}
h1{{font-size:32px;margin-bottom:8px}}p{{color:#667085;margin-bottom:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
.card{{display:flex;flex-direction:column;gap:8px;padding:22px;border-radius:14px;
background:white;color:inherit;text-decoration:none;box-shadow:0 2px 12px #10182812;
border:1px solid #e6e8ec}}.card:hover{{border-color:#f59e0b;transform:translateY(-1px)}}
.card strong{{font-size:18px}}.card span{{font-size:14px;color:#667085}}
</style></head><body><main><h1>YAM 训练曲线</h1>
<p>先选择模型，再在 TensorBoard 中选择该模型的实验。</p>
<div class="grid">{cards}</div></main>
<script>for(const a of document.querySelectorAll('.card')){{
a.href=location.protocol+'//'+location.hostname+':'+a.dataset.port+'/'}}</script>
</body></html>""".encode()


class Handler(BaseHTTPRequestHandler):
    base_port = 16006

    def do_GET(self) -> None:
        if self.path == "/healthz":
            body = json.dumps({"status": "ok", "models": len(MODELS)}).encode()
            content_type = "application/json"
        elif self.path in ("/", "/index.html"):
            body = render_index(self.base_port)
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=16006)
    args = parser.parse_args()
    Handler.base_port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"training dashboard index: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
