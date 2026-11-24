#!/home/matt/miniconda3/envs/marker_env/bin/python
"""ddc_owui — Open WebUI bridge for the DDC library (port 8072).

Open WebUI 0.13.x dropped the old addon-container framework; its native
extension points are (a) OpenAI-compatible API connections and (b) MCP
tool-server connections. This service provides both in one process:

  * /v1/models + /v1/chat/completions — an OpenAI-compatible shim that
    registers a "ddc-library" model in OWUI's model picker. Chatting with
    that model runs the reader's /v1/chat — the agent: the LLM reasons
    between searches (hybrid retrieval + cited synthesis). No
    function-calling required on the OWUI side. DDC_OWUI_AGENT=0 falls
    back to the one-shot /v1/ask.
  * /mcp (and /) — streamable-HTTP MCP server exposing the same tools as
    mcp_server.py (search_library, ask_library, get_chunk, get_source,
    library_status, browse_ddc, find_books, get_page, rescan_library,
    log_hit, hits_recall, hits_pending, hits_mark) for OWUI's MCP
    connections — toggle them in any squigFast chat. The root path is served too, because OWUI's
    backend ignores the "path" field of a stored MCP connection and
    posts to the connection URL's root.

Run:
    /home/matt/miniconda3/envs/marker_env/bin/python ~/base/reader/ddc_owui.py
    (env: DDC_OWUI_PORT default 8072, DDC_OWUI_HOST default 0.0.0.0,
          READER_URL default http://localhost:8071)

Register in Open WebUI (admin UI, Settings → Connections):
  * OpenAI (API key): base URL http://172.17.0.1:8072/v1  (from inside the
    OWUI container, the host is 172.17.0.1), any non-empty API key,
    model ids: ddc-library
  * MCP: http://172.17.0.1:8072/mcp
"""
import asyncio
import json
import os
import time
import uuid

import requests
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from mcp_server import create_server

READER = os.environ.get("READER_URL", "http://localhost:8071")
HOST = os.environ.get("DDC_OWUI_HOST", "0.0.0.0")
PORT = int(os.environ.get("DDC_OWUI_PORT", "8072"))
MODEL = "ddc-library"
# /v1/chat = the agent (reasons between searches — deeper, slower);
# /v1/ask = one-shot RAG (fast). DDC_OWUI_AGENT=0 falls back to one-shot.
USE_AGENT = os.environ.get("DDC_OWUI_AGENT", "1") != "0"
ASK_PATH = "/v1/chat" if USE_AGENT else "/v1/ask"
ASK_TIMEOUT = int(os.environ.get("DDC_OWUI_ASK_TIMEOUT", "600" if USE_AGENT else "300"))


# ---------------------------------------------------------------------------
# /v1/ask -> OpenAI chat completion mapping
# ---------------------------------------------------------------------------

def _ask_reader(q: str, history: list[dict]) -> dict:
    r = requests.post(f"{READER}{ASK_PATH}",
                      json={"q": q, "history": history}, timeout=ASK_TIMEOUT)
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", r.text[:300])
        except ValueError:
            detail = r.text[:300]
        raise RuntimeError(f"reader {ASK_PATH} HTTP {r.status_code}: {detail}")
    return r.json()


def _content_of(m: dict) -> str:
    """Flatten an OpenAI message content (string or parts list) to text."""
    c = m.get("content")
    if isinstance(c, list):  # content parts: [{"type":"text","text":...}]
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _split_messages(messages) -> tuple[str | None, list[dict]]:
    """Last user message -> the question; earlier user/assistant turns ->
    history (the reader's /v1/ask multi-turn input)."""
    if not isinstance(messages, list):
        return None, []
    last_user = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i] or {}
        if m.get("role") == "user":
            last_user = i
            break
    if last_user is None:
        return None, []
    q = _content_of(messages[last_user]).strip()
    history = []
    for m in messages[:last_user]:
        m = m or {}
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue  # system / tool messages: the answer engine has its own
        history.append({"role": role, "content": _content_of(m)})
    return (q or None), history


def _content_from_ask(j: dict) -> str:
    """Answer + numbered sources + coverage, self-contained for a chat UI."""
    out = [j.get("answer") or ""]
    sources = j.get("sources") or []
    if sources:
        out.append("\n---\n")
        for s in sources:
            out.append(f"[{s['n']}] {s['title']} p.{s['page']}  {s.get('path') or ''}")
    cov = j.get("coverage") or {}
    if cov:
        out.append(f"\n(searched {cov['books_indexed']}/{cov['books_total']} books "
                   f"in the index)")
    return "\n".join(out)


def _sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


async def _stream_sse(content: str, include_usage: bool):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(choices):
        return {"id": cid, "object": "chat.completion.chunk",
                "created": created, "model": MODEL, "choices": choices}

    yield _sse(chunk([{"index": 0,
                       "delta": {"role": "assistant", "content": content},
                       "finish_reason": None}]))
    yield _sse(chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]))
    if include_usage:
        yield _sse({**chunk([]),
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                              "total_tokens": 0}})
    yield "data: [DONE]\n\n"


def _completion(content: str) -> dict:
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": MODEL,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0}}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

async def healthz(request):
    return JSONResponse({"ok": True, "model": MODEL, "reader": READER})


async def index(request):
    return JSONResponse({
        "service": "ddc-owui-bridge",
        "endpoints": {
            "GET /v1/models": "OpenAI model list (registers ddc-library in OWUI)",
            "POST /v1/chat/completions": "OpenAI chat -> reader /v1/ask (RAG)",
            "POST /mcp": "streamable-HTTP MCP (5 library tools)",
            "POST /": "same MCP endpoint (OWUI connects to the URL root)",
        },
    })


async def list_models(request):
    return JSONResponse({"object": "list", "data": [{
        "id": MODEL, "object": "model", "created": 0, "owned_by": "ddc",
    }]})


async def chat_completions(request):
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):
        return JSONResponse({"error": {"message": "invalid JSON body",
                                       "type": "invalid_request_error"}},
                            status_code=400)
    if not isinstance(body, dict):
        body = {}
    q, history = _split_messages(body.get("messages"))
    if not q or len(q) < 3:
        content = ("I answer questions about the local DDC book library — "
                   "ask me something, e.g. 'which section covers the history "
                   "of the Roman Republic?'.")
    else:
        try:
            j = await asyncio.to_thread(_ask_reader, q, history)
        except RuntimeError as e:
            return JSONResponse({"error": {"message": str(e),
                                           "type": "api_error",
                                           "code": "reader_unavailable"}},
                                status_code=502)
        except requests.RequestException as e:
            return JSONResponse({"error": {"message": f"reader unreachable ({e})",
                                           "type": "api_error",
                                           "code": "reader_unreachable"}},
                                status_code=502)
        content = _content_from_ask(j)

    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {})
                             .get("include_usage"))
        return StreamingResponse(_stream_sse(content, include_usage),
                                 media_type="text/event-stream")
    return JSONResponse(_completion(content))


# ---------------------------------------------------------------------------
# app assembly: shim routes + the FastMCP /mcp route, one lifespan
# ---------------------------------------------------------------------------

mcp_server = create_server()
mcp_app = mcp_server.streamable_http_app()  # creates the session manager
mcp_route = next(r for r in mcp_app.routes if isinstance(r, Route))  # /mcp
# mcp 1.x builds the /mcp route around a path-agnostic ASGI wrapper
# (StreamableHTTPASGIApp -> session_manager.handle_request), so the same
# instance can be mounted at "/" as well. OWUI's backend connects to the
# stored connection URL's root (it ignores the "path" field), so a URL
# saved without /mcp still works.
mcp_root = Route("/", mcp_route.endpoint)


async def lifespan(app):
    # A mounted Starlette sub-app never receives lifespan events, so run the
    # MCP session manager's lifecycle in the parent's lifespan (exactly what
    # FastMCP's own app does: lifespan=lambda app: session_manager.run()).
    async with mcp_server.session_manager.run():
        yield


app = Starlette(routes=[
    Route("/info", index),
    Route("/healthz", healthz),
    Route("/v1/models", list_models),
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    mcp_route,  # POST/GET/DELETE /mcp (streamable HTTP MCP)
    mcp_root,   # same MCP endpoint at / (OWUI posts to the URL root)
], lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
