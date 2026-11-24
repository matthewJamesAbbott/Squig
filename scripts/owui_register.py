#!/usr/bin/env python3
"""Register the DDC library with Open WebUI via its admin API.

Registers both native extension points (OWUI 0.13.x has no addon framework):

  1. OpenAI (API key) connection -> http://172.17.0.1:8072/v1
     Adds a "ddc-library" model to the OWUI model picker. Chatting with it
     runs the reader's /v1/ask (hybrid retrieval + cited answer).
  2. MCP tool-server connection  -> http://172.17.0.1:8072/mcp
     All 13 DDC tools (search_library, ask_library, get_chunk, get_source,
     library_status, browse_ddc, find_books, get_page, rescan_library,
     log_hit, hits_recall, hits_pending, hits_mark) become toggleable in
     any chat.

Idempotent: re-running detects the existing entries and leaves them alone.

Usage:
    python owui_register.py --key <OWUI admin API key> \
        [--owui http://localhost:3000]

Get the key: OWUI -> Settings (gear, top right) -> Profile -> API Keys ->
"New". It can be revoked after this runs.

Prereq: the bridge must be up:  ~/base/reader/run.sh owui start
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

OPENAI_BASE = "http://172.17.0.1:8072/v1"   # as seen from inside the OWUI container
MCP_URL = "http://172.17.0.1:8072/mcp"
MCP_PATH = ""   # OWUI's backend ignores "path" for MCP and uses "url" verbatim
MCP_INFO_ID = "ddc-library"
API_KEY_VALUE = "ddc"      # the bridge accepts any key (local network)
MODEL_ID = "ddc-library"


def _req(key: str, base: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        detail = e.read()[:400].decode(errors="replace")
        raise SystemExit(f"OWUI HTTP {e.code} on {method} {path}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="OWUI admin API key")
    ap.add_argument("--owui", default="http://localhost:3000")
    args = ap.parse_args()
    base = args.owui.rstrip("/")

    # sanity: the bridge must be reachable from here
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=10) as r:
            print("OWUI:", r.read().decode().strip()[:60])
    except Exception as e:
        raise SystemExit(f"cannot reach OWUI at {base}: {e}")

    # ---- 1. OpenAI connection (the ddc-library model) -------------------
    oa = _req(args.key, base, "GET", "/openai/config")
    urls = list(oa.get("OPENAI_API_BASE_URLS") or [])
    keys = list(oa.get("OPENAI_API_KEYS") or [])
    configs = dict(oa.get("OPENAI_API_CONFIGS") or {})

    idx = next((i for i, u in enumerate(urls)
                if u.rstrip("/") == OPENAI_BASE.rstrip("/")), None)
    if idx is None:
        idx = len(urls)
        urls.append(OPENAI_BASE)
        if len(keys) > idx:
            keys[idx] = API_KEY_VALUE
        else:
            keys += [""] * (idx - len(keys) + 1)
            keys[idx] = API_KEY_VALUE
    if str(idx) not in configs:
        configs[str(idx)] = {
            "enable": True, "tags": [], "prefix_id": "",
            "model_ids": [MODEL_ID],
            "connection_type": "external", "auth_type": "bearer",
            "forward_cookies": False, "passthrough_params": [],
        }
    _req(args.key, base, "POST", "/openai/config/update", {
        "ENABLE_OPENAI_API": oa.get("ENABLE_OPENAI_API", True),
        "OPENAI_API_BASE_URLS": urls,
        "OPENAI_API_KEYS": keys,
        "OPENAI_API_CONFIGS": configs,
    })
    print(f"openai connection: ddc-library at {OPENAI_BASE} (index {idx})")

    # ---- 2. MCP tool-server connection ----------------------------------
    ts = _req(args.key, base, "GET", "/api/v1/configs/tool_servers") \
         .get("TOOL_SERVER_CONNECTIONS") or []
    have = any(c.get("type") == "mcp"
               and (c.get("info") or {}).get("id") == MCP_INFO_ID for c in ts)
    if have:
        print("mcp connection: already present")
    else:
        ts.append({
            "url": MCP_URL, "path": MCP_PATH, "type": "mcp",
            "auth_type": "none", "forward_cookies": False,
            "headers": None, "key": None,
            "config": {"enable": True},
            "info": {"id": MCP_INFO_ID},
        })
        _req(args.key, base, "POST", "/api/v1/configs/tool_servers",
             {"TOOL_SERVER_CONNECTIONS": ts})
        print(f"mcp connection: {MCP_URL}{MCP_PATH}")

    # ---- 3. verify -------------------------------------------------------
    models = _req(args.key, base, "GET", "/api/v1/models")
    if isinstance(models, dict):
        models = models.get("data") or []
    ids = [m.get("id") for m in (models or [])
           if isinstance(m, dict) and MODEL_ID in str(m.get("id"))]
    if ids:
        print(f"verified: model picker now contains {ids}")
    else:
        print(f"WARNING: {MODEL_ID!r} not found in GET /api/v1/models — "
              "check the OWUI logs / refresh the model list")
    ts = _req(args.key, base, "GET", "/api/v1/configs/tool_servers") \
         .get("TOOL_SERVER_CONNECTIONS") or []
    print("tool servers:",
          [(c.get("type"), (c.get("info") or {}).get("id") or c.get("url"))
           for c in ts])
    print("\ndone. In OWUI: the 'ddc-library' model should now appear in the "
          "model picker; the 13 DDC tools appear in the tools menu of any chat.")


if __name__ == "__main__":
    main()
