#!/usr/bin/env python3
"""
DDC Book Reader — browse the DDC library and read books.

Walks the DDC directory tree (~/DDC by default) like a library:
  - browse folders down to the books
  - a "book" is a *.parquet with a final_title column (one row per page);
    recipe/ingredient parquets are ignored
  - the reader shows the OCR text of each page, and can render the original
    PDF page (PyMuPDF) on demand so you can check the OCR / see images

Page images are rendered lazily from the PDF that sits next to the parquet
(same name, .pdf) and cached on disk under ./cache/.

One machine with the Qdrant indexer (~/DDC/indexer):
    - /api/semantic      semantic search over the ddc-books Qdrant collection
                         (query embedded with the SAME bge-m3 model the
                         indexer used; hits map back to book+page here)
    - /api/chat          RAG answers: hybrid retrieval (dense bge-m3 + BM25
                         full-text, RRF-fused in Qdrant) + cited synthesis
                         from an OpenAI-compatible LLM (the "onecat" vLLM
                         service by default); every claim cites a numbered
                         page that deep-links into the reader
    - /api/index/status  Qdrant / Ollama / indexer health + progress
    - /api/index/start   run index_books.py in the background (appends to
    - /api/index/stop    the shared index_run.log, exactly like a CLI run)

Stable machine API (for the user's RAG, for LLM agents, for ddq + MCP):
    - GET  /v1/status             health + index coverage
    - POST /v1/search             canonical retrieval: hybrid|dense|keyword,
                                  filters (ddc_coarse, book_id, ...), every
                                  result self-describing with a locator
                                  (local path + page) and a coverage block
    - GET  /v1/books              book catalog (files on disk, not the index):
                                  no params -> DDC section tree; ?ddc=<code>
                                  -> books in a section; ?q=<keyword> ->
                                  title/author search. Per-book index status.
    - GET  /v1/page/{book_id}/{page_num}
                                   full text of one page, read from the source
                                   parquet (works for unindexed books)
    - POST /v1/rescan               re-walk the DDC tree now (the catalog
                                   auto-rescans every CATALOG_RESCAN_INTERVAL
                                   seconds, default 60; this is for right
                                   after a classification run)
    - POST /v1/ask                one-shot RAG: one search, one cited answer
    - POST /v1/chat               agent answer: the LLM reasons between
                                  searches (tools: search_library, get_chunk,
                                  get_source, library_status, browse_ddc,
                                  find_books, get_page), reformulates,
                                  scopes to DDC, pulls full pages, then
                                  synthesizes a cited answer. Slowest, deepest.
    - POST /api/agent             same agent, streamed as SSE for the dashboard
    - GET  /v1/chunk/{point_id}   one chunk by its deterministic Qdrant id
    - GET  /v1/source/{book_id}   book metadata + its index coverage
    - /v1/index/status|start|stop indexer control (mirrors /api/index/*)
    - GET  /v1/hits/status         hit-log (MySQL) health + per-layer counts
    - POST /v1/hits                record one hit (layer user|agent)
    - GET  /v1/hits?layer=&q=...   query the two-layer research log
    - GET  /v1/hits/pending        the loop's frontier (needs-verify, FIFO)
    - POST /v1/hits/mark           advance a hit (status/confidence/note/refs)
    Also exposed natively to agents as MCP tools by mcp_server.py
    (search_library, ask_library, get_chunk, get_source, library_status,
    log_hit, hits_recall, hits_pending, hits_mark)
    and as the ddq command-line client.

Environment (all optional):
    DDC_ROOT, READER_PORT, CACHE_MAX_FILES, CACHE_MAX_BYTES, BOOK_LRU
    QDRANT_URL      default http://localhost:6333
    EMBED_BASE_URL  default http://localhost:11434  (GPU index worker)
    EMBED_QUERY_BASE_URL default http://127.0.0.1:11435 (dedicated CPU
    query-embed worker, managed by query-ollama.sh; keeps search fast while
    the indexer is mid-batch; falls back to EMBED_BASE_URL when down)
    EMBED_MODEL     default bge-m3   (must match what built the index)
    INDEXER_DIR     default ~/DDC/indexer
    LLM_BASE_URL    default http://10.90.38.19:8000  (onecat vLLM container)
    LLM_MODEL       default squigFast
    LLM_TIMEOUT     default 240 (s)
    HITS_AUTOLOG    default 1  (0 = stop auto-recording searches/asks)
    HITS_DB_HOST/PORT/USER/PASSWORD/NAME/TIMEOUT  default
        127.0.0.1 / 3306 / matt / "" / squig_research / 5
        (the MySQL hit database; see hits_db.py)

Run:
    /home/matt/miniconda3/envs/marker_env/bin/python ~/base/reader/app.py
    -> http://localhost:8071
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

import fitz  # PyMuPDF
import pyarrow.parquet as pq
import requests
from flask import Blueprint, Flask, Response, abort, jsonify, render_template, request

import hits_db  # local module: MySQL research log (two layers: user / agent)
HITS_AUTOLOG = os.environ.get("HITS_AUTOLOG", "1") != "0"

# --- hit-log plumbing -------------------------------------------------------
# Every search/ask/agent answer/chunk/page-read is logged to the MySQL hit
# database in the BACKGROUND: a single worker drains a bounded queue, so
# MySQL can never block a request or break one; a dead DB costs one
# rate-limited log line per minute, not one per request.
#
# layer='user' here on purpose: anything arriving through this API is the
# USER's work (via the dashboard, the OWUI bridge, or an MCP client). The
# agent's own off-time research records itself as layer='agent' through
# POST /v1/hits (the log_hit MCP tool).
import queue

_hits_queue: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
_hits_err_at = 0.0
_hits_err_msg = ""


def _autolog(hit: dict, refs: list[tuple[str, str]] | None = None) -> None:
    """Queue one hit for the background writer. Never blocks, never raises."""
    if not HITS_AUTOLOG:
        return
    try:
        _hits_queue.put_nowait({"hit": hit, "refs": refs})
    except queue.Full:
        pass  # the writer is far behind — drop rather than slow the reader


def _hits_worker() -> None:
    while True:
        item = _hits_queue.get()
        try:
            hits_db.record_hit(item["hit"], item.get("refs"))
        except Exception as e:
            global _hits_err_at, _hits_err_msg
            msg = str(e).splitlines()[0]
            now = time.time()
            if msg != _hits_err_msg and now - _hits_err_at > 60:
                print(f"[hits] {msg}", flush=True)
                _hits_err_at, _hits_err_msg = now, msg


def _hits_refs(pts: list[dict]) -> list[tuple[str, str]]:
    """(book, point) refs from raw Qdrant points, for hit_refs."""
    refs: list[tuple[str, str]] = []
    for p in pts:
        pl = p.get("payload") or {}
        if pl.get("book_id"):
            refs.append(("book", str(pl["book_id"])))
        if p.get("id") is not None:
            refs.append(("chunk", str(p["id"])))
    return refs


def _hits_summary(pts: list[dict]) -> str:
    if not pts:
        return "no results"
    top = (pts[0].get("payload") or {})
    return (f"{len(pts)} results · top: {top.get('book_title', '?')} "
            f"p.{top.get('page_num')} "
            f"({round(float(pts[0].get('score', 0.0)), 3)})")

DDC_ROOT = Path(os.environ.get("DDC_ROOT", str(Path.home() / "DDC"))).resolve()
PORT = int(os.environ.get("READER_PORT", "8071"))
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_MAX_FILES = int(os.environ.get("CACHE_MAX_FILES", "1500"))
CACHE_MAX_BYTES = int(os.environ.get("CACHE_MAX_BYTES", str(600 * 1024 * 1024)))
BOOK_LRU = int(os.environ.get("BOOK_LRU", "12"))   # full books held in memory

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
INDEXER_DIR = Path(os.environ.get("INDEXER_DIR",
                                  str(Path.home() / "DDC" / "indexer"))).resolve()
INDEX_RUN_LOG = INDEXER_DIR / "index_run.log"
SEMANTIC_K = 8          # default hits for /api/semantic
INDEXED_TTL = 90        # s: how long to cache the "books indexed" count

# Answer engine (the G in RAG): OpenAI-compatible chat endpoint.
# Default = the "onecat" vLLM service (Qwen3-27B, TP=2 on the V100s) inside
# its LXD container; override with LLM_BASE_URL / LLM_MODEL.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://10.90.38.19:8000")
LLM_MODEL = os.environ.get("LLM_MODEL", "squigFast")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "240"))
CHAT_K = 12             # pages retrieved for synthesis
CHAT_PAGE_CHARS = 1500  # per-page text budget inside the LLM context

# Dedicated interactive query-embed worker: a second, CPU Ollama (managed by
# query-ollama.sh) that never queues behind the indexer's GPU batch jobs, so
# search/ask stay fast while a full index run is in progress. Falls back to
# EMBED_BASE_URL when the dedicated worker is down.
EMBED_QUERY_BASE_URL = os.environ.get("EMBED_QUERY_BASE_URL",
                                      "http://127.0.0.1:11435")
QUERY_EMBED_CACHE_MAX = 512   # same question -> same vector (bge-m3 is deterministic)

CHAT_SYSTEM = """You are a research assistant for a personal library of 632 books, organized by Dewey Decimal Classification.

Answer using ONLY the numbered source pages provided. Rules:
- Support every substantive claim with a bracketed citation like [3] or [1][4].
- If the sources give conflicting accounts, present both, each with its citation.
- If the sources do not contain the answer, say so in one sentence, then briefly summarize what the sources DO say that is closest.
- Quote short phrases only where the exact wording matters. No knowledge outside the sources.
- Plain markdown, concise; short structured answers with headings or bullets when the question warrants it."""

META_COLS = ["final_title", "authors", "publisher", "pub_date", "ddc", "summary"]
PAGE_COLS = ["page_num", "text", "is_index"] + META_COLS

app = Flask(__name__)
app.json.sort_keys = False

_books_lock = threading.Lock()   # book LRU + summaries
_pdf_lock = threading.Lock()     # fitz is not thread-safe
_books: "OrderedDict[str, 'BookData']" = OrderedDict()
_summaries: dict[str, dict] = {}  # str(path) -> {"summary":..., "mtime":..., "size":...}
GLOBAL = {"ready": False, "count": 0, "lock": threading.Lock(), "books": [],
          "by_book_id": {}, "last_rescan": 0.0}
                              # by_book_id: book_id -> summary (semantic hits)
RESCAN_INTERVAL = int(os.environ.get("CATALOG_RESCAN_INTERVAL", "60"))


# book_id must stay identical to the indexer's, so we import it (single
# source of truth) and fall back to an identical hash if the indexer dir
# is missing.
sys.path.insert(0, str(INDEXER_DIR))
try:
    from index_books import COLLECTION, book_id_of  # noqa: E402
except Exception:  # pragma: no cover
    COLLECTION = "ddc-books"

    def book_id_of(final_title, authors, pub_date):
        raw = f"{final_title or ''}|{authors or ''}|{pub_date or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
class BookData:
    """One book fully loaded (page nums, text, is_index, meta)."""

    def __init__(self, path: Path):
        # image_path (stored figures from EPUB/TXT ingestion) is optional in
        # old parquets — fall back to the base columns if it's missing.
        try:
            rows = pq.read_table(path, columns=PAGE_COLS + ["image_path"]).to_pylist()
        except Exception:
            rows = pq.read_table(path, columns=PAGE_COLS).to_pylist()
        if not rows:
            raise ValueError("parquet has no rows")
        self.path = path
        self.page_nums: list[int] = []
        self.texts: dict[int, str] = {}
        self.is_index: dict[int, bool] = {}
        self.image_paths: dict[int, list[str]] = {}  # page -> stored images
        for r in rows:
            pn = int(r["page_num"])
            self.page_nums.append(pn)
            self.texts[pn] = r["text"] or ""
            self.is_index[pn] = bool(r["is_index"])
            ips = r.get("image_path") or []
            if ips:
                self.image_paths[pn] = [p for p in ips if p]
        r0 = rows[0]
        self.meta = {k: (str(r0[k]) if r0[k] is not None else "") for k in META_COLS}
        self.pdf = path.with_suffix(".pdf")
        self.pdf_pages: int | None = None


def is_book_parquet(path: Path) -> bool:
    """Book parquets have a final_title column; recipe/ingredient ones don't."""
    try:
        return "final_title" in set(pq.read_schema(path).names)
    except Exception:
        return False


def safe_path(rel: str) -> Path:
    """Resolve a relative path under DDC_ROOT, rejecting traversal outside."""
    p = (DDC_ROOT / (rel or "").strip().lstrip("/")).resolve()
    if p != DDC_ROOT and DDC_ROOT not in p.parents:
        abort(404, description="path outside library root")
    return p


def book_for(path: Path) -> BookData:
    key = str(path)
    with _books_lock:
        b = _books.get(key)
        if b is not None:
            _books.move_to_end(key)
            return b
    b = BookData(path)
    with _books_lock:
        _books[key] = b
        while len(_books) > BOOK_LRU:
            _books.popitem(last=False)
    return b


def book_summary(path: Path) -> dict:
    """Cheap per-book listing info: footer (page count) + first row (meta)."""
    st = path.stat()
    key = str(path)
    with _books_lock:
        c = _summaries.get(key)
        if c and c["mtime"] == st.st_mtime and c["size"] == st.st_size:
            return c["summary"]
    pf = pq.ParquetFile(path)
    r0 = pf.read_row_group(0).to_pylist()[0]
    meta = {k: (str(r0[k]) if r0.get(k) is not None else "") for k in
            ("final_title", "authors", "publisher", "pub_date", "ddc")}
    rel = str(path.relative_to(DDC_ROOT))
    s = {
        "name": path.name,
        "title": meta["final_title"] or path.stem,
        "authors": meta["authors"],
        "pub_date": meta["pub_date"],
        "publisher": meta["publisher"],
        "ddc": meta["ddc"],
        "ddc_path": "/".join(rel.split("/")[:-1]),  # DDC section (not filename)
        "book_id": book_id_of(meta["final_title"], meta["authors"],
                              meta["pub_date"]),
        "pages": int(pf.metadata.num_rows),
        "has_pdf": path.with_suffix(".pdf").exists(),
        "path": rel,
    }
    with _books_lock:
        _summaries[key] = {"summary": s, "mtime": st.st_mtime, "size": st.st_size}
    return s


# ---------------------------------------------------------------------------
# Catalog (files on disk) — DDC browse, book search, direct page reads
#
# These read the library itself (GLOBAL index + the parquets), NOT the Qdrant
# vector index: they work for books the indexer hasn't reached yet.
# ---------------------------------------------------------------------------
def _book_matches_ddc(book: dict, ddc: str) -> bool:
    """Prefix match against the DDC section path or the book's leaf DDC code.

    ddc can be a code or path at any depth: "500", "500_Science", "540",
    "540_Chemistry", a leaf like "641", or a full path prefix.
    """
    ddc = (ddc or "").strip().strip("/")
    if not ddc:
        return True
    path = book["path"]
    if path.startswith(ddc):
        return True
    if any(seg.startswith(ddc) for seg in path.split("/")[:-1]):
        return True
    leaf = (book.get("ddc") or "").strip()
    return bool(leaf) and leaf.startswith(ddc)


def _catalog_books(ddc: str = "", q: str = "") -> list[dict]:
    """Filter the on-disk catalog; annotate each book with its index status
    (chunks in Qdrant vs pages on disk)."""
    q = (q or "").strip().lower()
    with GLOBAL["lock"]:
        books = list(GLOBAL["books"])
    if ddc:
        books = [b for b in books if _book_matches_ddc(b, ddc)]
    if q:
        books = [b for b in books
                 if q in (b["title"] + " " + b["authors"]
                          + " " + b.get("publisher", "")).lower()]
    # keyword search: by title; pure browse: DDC order (path)
    books.sort(key=lambda b: (b["title"].lower() if q else b["path"]))
    counts = _book_chunk_counts()
    for b in books:
        b["chunks_indexed"] = counts.get(b["book_id"], 0)
        b["indexed"] = b["chunks_indexed"] > 0
    return books


def _ddc_tree() -> list[dict]:
    """What the library actually holds: coarse DDC class -> tens section ->
    book counts, straight from the on-disk catalog."""
    tree: dict = {}
    with GLOBAL["lock"]:
        books = list(GLOBAL["books"])
    for b in books:
        segs = b["path"].split("/")[:-1]
        coarse = segs[0] if segs else "(unclassified)"
        tens = segs[1] if len(segs) > 1 else None
        c = tree.setdefault(coarse, {"ddc": coarse, "books": 0, "sections": {}})
        c["books"] += 1
        if tens:
            c["sections"][tens] = c["sections"].get(tens, 0) + 1
    out = []
    for coarse in sorted(tree):
        c = tree[coarse]
        c["sections"] = [{"ddc": k, "books": v}
                         for k, v in sorted(c["sections"].items())]
        out.append(c)
    return out


_rescan_lock = threading.Lock()   # at most one catalog scan in flight


def rescan_catalog() -> dict:
    """Re-walk the DDC tree and swap in a fresh book catalog.

    Re-entrant: builds the new lists first, then swaps them atomically, so
    the API keeps serving the previous catalog while a scan is in flight.
    Deleted books disappear, new/changed ones appear (book_summary caches
    per mtime/size, so a rescan only re-reads files that changed)."""
    if not _rescan_lock.acquire(blocking=False):
        return {"rescanning": True, "books": GLOBAL["count"]}
    try:
        t0 = time.time()
        books: list[dict] = []
        by_id: dict[str, dict] = {}
        for p in sorted(DDC_ROOT.rglob("*.parquet")):
            try:
                if is_book_parquet(p):
                    s = book_summary(p)
                    books.append(s)
                    by_id[s["book_id"]] = s
            except Exception:
                pass
        with GLOBAL["lock"]:
            GLOBAL["books"] = books
            GLOBAL["by_book_id"] = by_id
            GLOBAL["count"] = len(books)
            GLOBAL["ready"] = True
            GLOBAL["last_rescan"] = time.time()
        out = {"rescanning": False, "books": len(books),
               "seconds": round(time.time() - t0, 1)}
        print(f"[reader] catalog rescanned: {len(books)} books in "
              f"{out['seconds']}s", flush=True)
        return out
    finally:
        _rescan_lock.release()


def _rescan_loop() -> None:
    """Periodically refresh the on-disk catalog so books classified or
    renamed since the last scan show up in browse/find/page without a
    reader restart."""
    while True:
        time.sleep(RESCAN_INTERVAL)
        try:
            rescan_catalog()
        except Exception as e:  # noqa: BLE001 — the loop must survive
            print(f"[reader] rescan failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# PDF page rendering
# ---------------------------------------------------------------------------
def evict_cache() -> None:
    try:
        files = sorted(CACHE_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        for f in files:
            if len(files) <= CACHE_MAX_FILES and total <= CACHE_MAX_BYTES:
                break
            sz = f.stat().st_size
            f.unlink(missing_ok=True)
            files.remove(f)
            total -= sz
    except OSError:
        pass


_PDF_DOCS: "OrderedDict[str, fitz.Document]" = OrderedDict()
_PDF_MAX_DOCS = 4  # large PDFs are expensive to open; keep a few warm


def _open_pdf(path: Path) -> fitz.Document:
    """Open a PDF, keeping a small LRU of open docs (called under _pdf_lock:
    fitz is not thread-safe). Re-parsing a 2000-page PDF on every page
    request is what made page-image requests queue up and 'hang'."""
    key = str(path)
    doc = _PDF_DOCS.get(key)
    if doc is None:
        doc = fitz.open(str(path))
        _PDF_DOCS[key] = doc
        while len(_PDF_DOCS) > _PDF_MAX_DOCS:
            _, old = _PDF_DOCS.popitem(last=False)  # popitem returns (key, doc)
            old.close()
    else:
        _PDF_DOCS.move_to_end(key)
    return doc


def render_pdf_page(pdf: Path, page: int, dpi: int) -> bytes:
    """Render (and cache) one PDF page as PNG. Raises on error."""
    st = pdf.stat()
    key = hashlib.sha1(f"{pdf}|{st.st_mtime}|{st.st_size}|{page}|{dpi}".encode()).hexdigest()
    out = CACHE_DIR / f"{key}.png"
    if out.exists():
        return out.read_bytes()
    with _pdf_lock:
        if out.exists():  # another worker got there first
            return out.read_bytes()
        doc = _open_pdf(pdf)
        if page < 1 or page > doc.page_count:
            raise RuntimeError(f"PDF has {doc.page_count} page(s); page {page} not found")
        m = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        png = doc[page - 1].get_pixmap(matrix=m, alpha=False).tobytes("png")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".png.part")
    tmp.write_bytes(png)
    tmp.rename(out)
    evict_cache()
    return png


def compose_page_images(paths: list[str]) -> bytes | None:
    """Stack the stored images of one page (EPUB/TXT figures/cover) into a
    single PNG — the non-PDF answer to 'render this page'. Returns None if
    none of the paths is a readable image."""
    from PIL import Image

    ims = []
    for p in paths[:4]:  # a page rarely holds more than a few figures
        try:
            if not os.path.exists(p):
                continue
            im = Image.open(p)
            im = im.convert("RGB")
            if im.width > 1200:  # keep responses light
                im = im.resize((1200, max(1, int(im.height * 1200 / im.width))))
            ims.append(im)
        except Exception:
            continue
    if not ims:
        return None
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in ims:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height
    import io as _io
    buf = _io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Semantic search (Qdrant) + indexer control
# ---------------------------------------------------------------------------
def _snippet(text: str, q: str, width: int = 300) -> str:
    """Center a snippet on the first query term found in the page text."""
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    words = re.findall(r"[a-z0-9]{4,}", q.lower())[:12]
    low = text.lower()
    pos = -1
    for w in words:
        i = low.find(w)
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        return text[:width] + "…"
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def _book_chunk_counts() -> Counter:
    """book_id -> how many of its chunks are in Qdrant, cached for
    INDEXED_TTL seconds. One scroll pass (10k/batch) instead of a count call
    per book — cheap at the current scale; revisit at millions of points."""
    now = time.time()
    if now - _indexed_cache["ts"] < INDEXED_TTL:
        return _indexed_cache["value"]
    counts: Counter = Counter()
    try:
        offset = 0
        while True:
            r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                              json={"limit": 10000, "offset": offset,
                                    "with_payload": ["book_id"]}, timeout=30)
            r.raise_for_status()
            res = r.json()["result"]
            for p in res["points"]:
                bid = p["payload"].get("book_id")
                if bid:
                    counts[bid] += 1
            offset = res.get("next_page_offset")
            if offset is None:
                break
        _indexed_cache.update(ts=now, value=counts)
    except requests.RequestException:
        pass
    return _indexed_cache["value"]


def _indexed_books() -> int:
    """Distinct book_ids in Qdrant (cached)."""
    return len(_book_chunk_counts())


_indexed_cache = {"ts": 0.0, "value": 0}


def _ddc_index_progress_file() -> Path | None:
    """Live progress file written by index_books.py (same contract as the
    pipeline GUI): ~/.ddc-progress/index_<root>.json"""
    try:
        d = Path(os.environ.get("DDC_PROGRESS_DIR",
                                str(Path.home() / ".ddc-progress")))
        slug = re.sub(r"[^A-Za-z0-9]+", "_", DDC_ROOT.name).strip("_") or "root"
        return d / f"index_{slug}.json"
    except Exception:
        return None


def _recent_pace(samples, window_s: float = 300.0):
    """Units/second from a [(ts, value), ...] series (last `window_s`,
    falling back to the whole series). None when not enough data."""
    try:
        s = [[float(a), float(b)] for a, b in (samples or [])]
    except (TypeError, ValueError):
        return None
    if len(s) < 2:
        return None
    last_ts, last_v = s[-1]
    ref = s[0]
    for p in s:
        if last_ts - p[0] <= window_s:
            ref = p
            break
    for c in (ref, s[0]):
        dt = last_ts - c[0]
        if dt < 10:
            continue
        dv = last_v - c[1]
        if dv > 0:
            return dv / dt
    return None


def indexer_status() -> dict:
    """Is index_books.py running? Where is it exactly (book + page), how far
    does it have to go, and the ETA? Prefers the machine-readable progress
    file the indexer writes every book/embed batch; falls back to parsing
    the shared log (old indexer builds)."""
    out = {"running": False, "progress": None, "last_done": None, "log_mtime": None}
    try:
        # [.] keeps pgrep from matching shells whose own command line
        # contains the literal pattern (e.g. when checking from the CLI).
        r = subprocess.run(["pgrep", "-f", "index_books[.]py --root"],
                           capture_output=True, text=True, timeout=5)
        pids = [x for x in r.stdout.split() if x and x != str(os.getpid())]
        out["running"] = bool(pids)
    except Exception:
        pass
    if INDEX_RUN_LOG.exists():
        try:
            out["log_mtime"] = INDEX_RUN_LOG.stat().st_mtime
            lines = INDEX_RUN_LOG.read_text(errors="ignore").splitlines()
            for line in reversed(lines):
                m = re.search(r"Book (\d+)/(\d+)", line)
                if m and out["progress"] is None:
                    out["progress"] = f"book {m.group(1)}/{m.group(2)}"
                if "Done:" in line and out["last_done"] is None:
                    out["last_done"] = line.split("INFO", 1)[-1].strip()[:160]
                if out["progress"] and out["last_done"]:
                    break
        except OSError:
            pass
    # --- live progress file (page-level, written by the indexer itself) ---
    pf = _ddc_index_progress_file()
    if pf is not None and pf.exists():
        try:
            d = json.loads(pf.read_text())
            age = time.time() - float(d.get("last_update") or 0)
            alive = False
            if d.get("pid"):
                try:
                    os.kill(int(d["pid"]), 0)
                    alive = True
                except (OSError, ValueError, TypeError):
                    pass
            state = d.get("state") or "running"
            if state == "running" and not alive:
                state = "stalled"
            if age < 300 or state == "running":
                pace = _recent_pace(d.get("samples"))
                rem = d.get("remaining_pages")
                eta_s = (rem / pace) if (pace and rem) else None
                out["detail"] = {
                    "state": state, "age_s": round(age), "pid": d.get("pid"),
                    "books_done": d.get("done_books"),
                    "books_total": d.get("total_books"),
                    "pages_done": d.get("pages_done"),
                    "pages_total": d.get("total_pages"),
                    "pages_embedded": d.get("pages_embedded"),
                    "remaining_pages": rem,
                    "current_book": d.get("current_book"),
                    "current_book_pages_done": d.get("current_book_pages_done"),
                    "current_book_pages": d.get("current_book_pages"),
                    "counts": d.get("counts"),
                    "pace_pages_per_min": round(pace * 60, 1) if pace else None,
                    "eta_s": round(eta_s) if eta_s else None,
                }
                det = out["detail"]
                if state in ("running", "quiet") and det.get("current_book"):
                    pages = (f" · page {det['current_book_pages_done']}/"
                             f"{det['current_book_pages']}"
                             if det.get("current_book_pages") else "")
                    pct = (f" · {100.0 * det['pages_done'] / det['pages_total']:.1f}% of pages"
                           if det.get("pages_total") and det.get("pages_done") is not None
                           else "")
                    eta = (f" · ETA ~{eta_s / 3600:.1f}h" if eta_s else "")
                    out["progress"] = (f"book {det.get('books_done')}/"
                                       f"{det.get('books_total')} of "
                                       f"“{det['current_book']}”{pages}{pct}{eta}")
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Query embedding (dedicated interactive worker + cache)
# ---------------------------------------------------------------------------
_query_embed_cache: "OrderedDict[tuple, list[float]]" = OrderedDict()
_embed_lock = threading.Lock()


def _embed_once(url: str, q: str) -> list[float]:
    r = requests.post(f"{url}/api/embed",
                      json={"model": EMBED_MODEL, "input": q}, timeout=180)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def _embed_query(q: str) -> list[float]:
    """Embed a query with the SAME bge-m3 model that built the index.

    Prefers the dedicated interactive worker (EMBED_QUERY_BASE_URL) so
    search never queues behind the indexer's batch jobs; falls back to the
    main embed endpoint. Results are cached: bge-m3 is deterministic, so a
    repeated question (or a follow-up rephrasing it) is instant.
    """
    urls = list(dict.fromkeys([EMBED_QUERY_BASE_URL, EMBED_BASE_URL]))
    with _embed_lock:
        for url in urls:
            key = (EMBED_MODEL, url, q)
            cached = _query_embed_cache.get(key)
            if cached is not None:
                _query_embed_cache.move_to_end(key)
                return cached
    last_err: Exception | None = None
    for url in urls:
        try:
            vec = _embed_once(url, q)
            with _embed_lock:
                _query_embed_cache[(EMBED_MODEL, url, q)] = vec
                while len(_query_embed_cache) > QUERY_EMBED_CACHE_MAX:
                    _query_embed_cache.popitem(last=False)
            return vec
        except requests.RequestException as e:
            last_err = e
    raise last_err  # pragma: no cover (both endpoints down)


# ---------------------------------------------------------------------------
# Retrieval (dense / keyword / hybrid) + answer engine (the G in RAG)
# ---------------------------------------------------------------------------
# Filterable payload fields for /v1/search (exact keyword matches).
V1_KEYWORD_FIELDS = ("book_id", "ddc_coarse", "ddc_tens", "ddc", "pub_date")


def _v1_filter(filters: dict) -> dict | None:
    must = []
    for key in V1_KEYWORD_FIELDS:
        if filters.get(key):
            must.append({"key": key, "match": {"value": str(filters[key])}})
    if filters.get("min_page") is not None or filters.get("max_page") is not None:
        rng = {}
        if filters.get("min_page") is not None:
            rng["gte"] = int(filters["min_page"])
        if filters.get("max_page") is not None:
            rng["lte"] = int(filters["max_page"])
        must.append({"key": "page_num", "range": rng})
    return {"must": must} if must else None


def _search(q: str, k: int, mode: str = "hybrid",
            filters: dict | None = None) -> tuple[list[dict], int, int]:
    """Retrieve chunks. Returns (points, embed_ms, qdrant_ms).

    mode:
      hybrid  — dense bge-m3 + BM25 full-text, RRF-fused by Qdrant (default)
      dense   — bge-m3 cosine only
      keyword — BM25 full-text only (no embedding; instant, no GPU needed)
    """
    flt = _v1_filter(filters or {})
    t0 = time.time()
    if mode in ("hybrid", "dense"):
        qvec = _embed_query(q)
        embed_ms = int((time.time() - t0) * 1000)
    else:
        qvec, embed_ms = None, 0
    t1 = time.time()
    if mode == "keyword":
        body: dict = {"query": {"text": q}, "using": "text",
                      "limit": k, "with_payload": True}
    elif mode == "dense":
        body = {"query": qvec, "limit": k, "with_payload": True}
    else:  # hybrid
        leg = {"query": {"text": q}, "using": "text",
               "limit": min(60, max(20, k * 3))}
        body = {"query": qvec, "prefetch": [leg], "fusion": "rrf",
                "limit": k, "with_payload": True}
    if flt:
        body["filter"] = flt
        if mode == "hybrid":
            body["prefetch"][0]["filter"] = flt
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()["result"]["points"], embed_ms, int((time.time() - t1) * 1000)


def _hybrid_search(q: str, k: int, ddc: str = "") -> list[dict]:
    """Back-compat wrapper used by /api/chat."""
    flt = {"ddc_coarse": ddc} if ddc else None
    return _search(q, k, "hybrid", flt)[0]


_points_cache = {"ts": 0.0, "value": None}


def _points_count() -> int | None:
    """Exact chunk count, cached for 10 s (exact count is a full scan)."""
    now = time.time()
    if _points_cache["value"] is not None and now - _points_cache["ts"] < 10:
        return _points_cache["value"]
    try:
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                          json={"exact": True}, timeout=5)
        r.raise_for_status()
        v = r.json()["result"]["count"]
        _points_cache.update(ts=now, value=v)
        return v
    except requests.RequestException:
        return _points_cache["value"]


def _coverage() -> dict:
    """How complete is the index — shown in every search/answer response so
    you always know what fraction of the library a result set covers."""
    with GLOBAL["lock"]:
        total = GLOBAL["count"]
    return {"books_indexed": _indexed_books(), "books_total": total,
            "chunks_indexed": _points_count()}


def _locator_for(pl: dict) -> dict | None:
    """Where is this chunk on disk? Payload `path` (newer ingests) or the
    local book index (points written before the payload carried it)."""
    path = pl.get("path")
    if not path:
        local = GLOBAL["by_book_id"].get(pl.get("book_id") or "") or {}
        path = local.get("path")
    if not path:
        return None
    return {"path": path, "page": pl.get("page_num")}


def _result_from_point(p: dict, q: str, include_text: bool = False) -> dict:
    pl = p.get("payload") or {}
    out = {
        "id": str(p.get("id")),
        "score": round(float(p.get("score", 0.0)), 4),
        "source_type": pl.get("source_type", "book"),
        "book_id": pl.get("book_id"),
        "doc_title": pl.get("book_title") or "?",
        "authors": pl.get("authors") or "",
        "pub_date": pl.get("pub_date") or "",
        "ddc": pl.get("ddc") or "",
        "page_num": pl.get("page_num"),
        "locator": _locator_for(pl),
        "snippet": _snippet(pl.get("text") or "", q),
    }
    if include_text:
        out["text"] = pl.get("text") or ""
    return out


def _build_context(hits: list[dict]) -> tuple[list[dict], str]:
    """Number the retrieved pages for the LLM.

    Returns (sources, context): sources carry the metadata the UI needs to
    deep-link each cited page into the reader; context is the numbered text
    the model must cite.
    """
    with GLOBAL["lock"]:
        by_bid = dict(GLOBAL["by_book_id"])
    sources: list[dict] = []
    parts: list[str] = []
    for i, p in enumerate(hits, 1):
        pl = p.get("payload") or {}
        bid = pl.get("book_id") or ""
        local = by_bid.get(bid) or {}
        title = pl.get("book_title") or local.get("title") or "?"
        pn = pl.get("page_num")
        text = " ".join((pl.get("text") or "").split())
        if len(text) > CHAT_PAGE_CHARS:
            text = text[:CHAT_PAGE_CHARS].rsplit(" ", 1)[0] + "…"
        sources.append({"n": i, "title": title, "page": pn,
                        "path": pl.get("path") or local.get("path"),
                        "has_pdf": bool(local.get("has_pdf"))})
        parts.append(f"[{i}] {title}, page {pn}\n{text}")
    return sources, "\n\n".join(parts)


def _llm_ask(messages: list[dict]) -> str:
    """One chat completion on the OpenAI-compatible LLM endpoint."""
    r = requests.post(f"{LLM_BASE_URL}/v1/chat/completions",
                      json={"model": LLM_MODEL, "messages": messages,
                            "temperature": 0.2, "max_tokens": 1024,
                            "chat_template_kwargs": {"enable_thinking": False}},
                      timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


# ---------------------------------------------------------------------------
# Agent chat — multi-step reasoning over the library
#
# /api/chat + /v1/ask are one-shot: one search, one answer. The agent gives
# the LLM the library as tools (search_library / get_chunk / get_source /
# library_status) and lets it reason BETWEEN searches: inspect the snippets,
# reformulate the query, scope to a DDC section, pull full page text, then
# synthesize a cited answer. Every page it sees gets a stable [n] citation
# number for the whole turn; the final answer cites only numbers it saw.
#
# Shared by POST /api/agent (SSE stream for the dashboard) and POST /v1/chat
# (stable JSON for the OWUI bridge, ddq, and other machines).
# ---------------------------------------------------------------------------

AGENT_MAX_ITERS = 6        # LLM turns that may issue tool calls
AGENT_MAX_TOKENS = 2048    # per LLM turn
AGENT_MAX_SOURCES = 40     # citation-registry cap per turn
AGENT_SNIPPET_CHARS = 260  # per hit inside a tool result
AGENT_CHUNK_CHARS = 4000   # full page-text cap inside get_chunk
AGENT_TOOL_CHARS = 6000    # cap on one tool result string
AGENT_TOTAL_TIMEOUT = 240  # s: whole agent run

AGENT_SYSTEM = """You are the research assistant for a private library of {books_total} books, organized by Dewey Decimal Classification. {books_indexed} are currently indexed and searchable ({chunks_indexed} pages); the index grows over time, so a negative answer may just be a coverage gap — say so when it matters.

You have seven tools:
- search_library: the workhorse. Default mode is hybrid (dense + keyword, RRF-fused). Reformulate between searches: different wording, broader or narrower, an exact phrase with mode "keyword", or scope with ddc (e.g. "500_Science"). Two to six searches is normal — do NOT stop at the first pass when the question deserves it.
- get_chunk: full text of one page, by the point id from a search. Use it when a snippet is too thin to answer from.
- get_source: book metadata (title, authors, DDC, pages, index coverage) by book_id.
- browse_ddc: the library organized by Dewey Decimal section. No ddc arg -> the section tree with book counts (use this to discover what the library holds on a topic before searching). With a ddc code or prefix ('500', '540', '540_Chemistry') -> the books in that section, each with its book_id.
- find_books: find books by title/author keyword (e.g. 'fermentation', 'Stamets'). Works even for books not yet indexed. Returns book_ids.
- get_page: read the full text of one specific page of one book, by book_id + 1-based page_num. Use it to pull an exact page you've already located (from a search hit, a book listing, or the user) rather than relying on a snippet.
- library_status: index health + coverage.

When to use the catalog tools (browse_ddc / find_books / get_page) vs search_library:
- "What do you have on X?" / "list books about X" -> browse_ddc (find the right DDC section) then browse_ddc with that ddc, or find_books for a title keyword.
- A specific fact you can read off one known page -> get_page directly.
- An open research question -> search_library, reformulating between searches.
- You can combine: search_library to locate a book+page, then get_page for the exact wording.

How to work: 1) search first. 2) read the snippets. 3) if they are weak, off-topic, or only partial, reformulate and search again — in that turn's content write one short line (<=15 words) saying what you found and why you're searching differently. 4) pull full text only where it matters. 5) answer.

Citations:
- Tool output numbers every page it returns as [n]. Cite ONLY numbers you saw in tool output, inline, like [3] or [1][4].
- If sources conflict, present both, each with its citation.
- If the index has nothing on a topic, say exactly that in one sentence — do not fill the gap from your own knowledge.
- Earlier conversation turns may contain [n] references; those numbers refer to that turn's own sources, not yours.

Plain markdown, concise, no preamble — the user asked one question."""


class AgentError(Exception):
    """Carries an HTTP status out of the agent loop (SSE + JSON routes)."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "search_library",
        "description": "Search the library index (parquet pages in Qdrant). "
                       "Hybrid (dense bge-m3 + BM25) by default; keyword mode "
                       "for exact phrases (instant, no embedding).",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "the query"},
                "k": {"type": "integer",
                      "description": "results, 1-20 (default 8)"},
                "mode": {"type": "string",
                         "enum": ["hybrid", "dense", "keyword"]},
                "ddc": {"type": "string",
                        "description": "scope to a DDC section, e.g. 500_Science"},
                "book_id": {"type": "string",
                            "description": "restrict to one book (from results)"}},
            "required": ["q"]}},
    },
    {"type": "function", "function": {
        "name": "get_chunk",
        "description": "Full text + metadata of one page, by its Qdrant point "
                       "id (the point_id from search_library results).",
        "parameters": {
            "type": "object",
            "properties": {"point_id": {"type": "integer"}},
            "required": ["point_id"]}},
    },
    {"type": "function", "function": {
        "name": "get_source",
        "description": "Book metadata + how much of it is indexed, by book_id.",
        "parameters": {
            "type": "object",
            "properties": {"book_id": {"type": "string"}},
            "required": ["book_id"]}},
    },
    {"type": "function", "function": {
        "name": "library_status",
        "description": "Health of the machine + how complete the index is "
                       "right now.",
        "parameters": {"type": "object"}},
    },
    {"type": "function", "function": {
        "name": "browse_ddc",
        "description": "Browse the library by Dewey Decimal section, straight "
                       "from the on-disk catalog (works for books not yet in "
                       "the search index). No ddc: the section tree with book "
                       "counts. With ddc (a code or prefix: '500', "
                       "'500_Science', '540', '540_Chemistry'): the books in "
                       "that section, with book_ids for get_page.",
        "parameters": {
            "type": "object",
            "properties": {
                "ddc": {"type": "string",
                        "description": "section code or path prefix; "
                                       "omit for the section tree"},
                "limit": {"type": "integer",
                          "description": "max books, 1-40 (default 20)"}}},
    }},
    {"type": "function", "function": {
        "name": "find_books",
        "description": "Find books in the catalog by title/author keyword "
                       "('Stamets', 'fermentation') — files on disk, so it "
                       "works even for books not yet in the vector index. "
                       "Returns title, authors, book_id, section, pages, "
                       "index status.",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string",
                      "description": "keyword in title/author"},
                "limit": {"type": "integer",
                          "description": "max books, 1-20 (default 10)"}},
            "required": ["q"]}},
    },
    {"type": "function", "function": {
        "name": "get_page",
        "description": "Read the full text of one page of one book, from the "
                       "source file (not the search index). book_id from "
                       "browse_ddc / find_books / search_library; page_num is "
                       "the 1-based page from those results.",
        "parameters": {
            "type": "object",
            "properties": {
                "book_id": {"type": "string"},
                "page_num": {"type": "integer"}},
            "required": ["book_id", "page_num"]}},
    },
]


def _agent_ctx() -> dict:
    with GLOBAL["lock"]:
        by_bid = dict(GLOBAL["by_book_id"])
    return {"sources": [], "by_page": {}, "searches": {},
            "by_bid": by_bid, "steps": []}


def _agent_register(ctx: dict, pts: list[dict]) -> list:
    """Assign [n] to pages not yet cited this turn. Returns [n] per point."""
    nums = []
    for p in pts:
        pl = p.get("payload") or {}
        key = (pl.get("book_id") or "", pl.get("page_num"))
        n = ctx["by_page"].get(key)
        if n is None and len(ctx["sources"]) < AGENT_MAX_SOURCES:
            n = len(ctx["sources"]) + 1
            local = ctx["by_bid"].get(key[0]) or {}
            ctx["sources"].append({
                "n": n,
                "book_id": key[0],
                "title": pl.get("book_title") or local.get("title") or "?",
                "page": key[1],
                "path": pl.get("path") or local.get("path"),
                "has_pdf": bool(local.get("has_pdf"))})
            ctx["by_page"][key] = n
        nums.append(n)
    return nums


def _agent_tool_search(ctx: dict, args: dict) -> tuple[str, dict]:
    q = str(args.get("q") or "").strip()
    if len(q) < 2:
        return "q is too short — use at least 2 characters.", {"q": q, "hits": 0}
    try:
        k = min(20, max(1, int(args.get("k", 8))))
    except (TypeError, ValueError):
        k = 8
    mode = args.get("mode") or "hybrid"
    if mode not in ("hybrid", "dense", "keyword"):
        mode = "hybrid"
    ddc = str(args.get("ddc") or "")
    book_id = str(args.get("book_id") or "")
    skey = (q.casefold(), mode, ddc, book_id)
    if skey in ctx["searches"]:
        prev = ctx["searches"][skey]
        nums = " ".join(f"[{n}]" for n in prev if n)
        return (f"same search as before (try different wording, mode, or scope). "
                f"pages already found: {nums or 'none'}.",
                {"q": q, "mode": mode, "ddc": ddc, "hits": len(prev),
                 "summary": "repeat search — reformulated"})
    filters = {}
    if ddc:
        filters["ddc_coarse"] = ddc
    if book_id:
        filters["book_id"] = book_id
    pts, _, _ = _search(q, k, mode, filters)
    if not pts:
        ctx["searches"][skey] = []
        return ("no results — reformulate: different wording, keyword mode for "
                "exact phrases, or drop/fix the DDC scope.",
                {"q": q, "mode": mode, "ddc": ddc, "hits": 0})
    nums = _agent_register(ctx, pts)
    ctx["searches"][skey] = nums
    lines = []
    for p, n in zip(pts, nums):
        pl = p.get("payload") or {}
        title = pl.get("book_title") or "?"
        snip = _snippet(pl.get("text") or "", q, AGENT_SNIPPET_CHARS)
        cite = f"[{n}] " if n else ""
        lines.append(f"{cite}{title} p.{pl.get('page_num')}  "
                     f"(score {round(float(p.get('score', 0.0)), 3)}, "
                     f"point_id {p.get('id')})\n   {snip}")
    out = "\n\n".join(lines)
    if len(out) > AGENT_TOOL_CHARS:
        out = out[:AGENT_TOOL_CHARS].rsplit("\n", 1)[0] + "\n…(truncated)"
    top = pts[0].get("payload") or {}
    return out, {"q": q, "mode": mode, "ddc": ddc, "hits": len(pts),
                 "summary": (f"{len(pts)} hits · top: {top.get('book_title', '?')} "
                             f"p.{top.get('page_num')} "
                             f"({round(float(pts[0].get('score', 0.0)), 3)})")}


def _agent_tool_chunk(ctx: dict, args: dict) -> tuple[str, dict]:
    try:
        pid = int(args.get("point_id"))
    except (TypeError, ValueError):
        return "point_id must be an integer.", {"point_id": args.get("point_id")}
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}/points/{pid}",
                         params={"with_payload": True}, timeout=15)
        r.raise_for_status()
        p = r.json()["result"]
    except requests.RequestException as e:
        return f"error: Qdrant unreachable ({e}).", {"point_id": pid}
    if not p or not p.get("payload"):
        return f"point {pid} not found in {COLLECTION}.", {"point_id": pid}
    pl = p.get("payload") or {}
    n = _agent_register(ctx, [p])[0]
    title = pl.get("book_title") or "?"
    text = " ".join((pl.get("text") or "").split())
    head = (f"[{n}] {title} p.{pl.get('page_num')}  "
            f"({pl.get('authors') or '?'}, {pl.get('pub_date') or '?'}, "
            f"ddc {pl.get('ddc') or '?'})\n" if n else f"{title} p.{pl.get('page_num')}\n")
    if len(text) > AGENT_CHUNK_CHARS:
        text = text[:AGENT_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"
    return head + text, {"point_id": pid,
                         "summary": f"{title} p.{pl.get('page_num')} · {len(text)} chars"}


def _agent_tool_source(ctx: dict, args: dict) -> tuple[str, dict]:
    book_id = str(args.get("book_id") or "")
    local = ctx["by_bid"].get(book_id)
    if not local:
        return (f"book_id {book_id!r} not in the local library — "
                f"use search_library first.", {"book_id": book_id})
    chunks = _book_chunk_counts().get(book_id, 0)
    out = (f"{local.get('title')}  ({local.get('authors', '')})\n"
           f"  ddc {local.get('ddc', '')} · {local.get('pages', '?')} pages · "
           f"pdf {'yes' if local.get('has_pdf') else 'no'}\n"
           f"  chunks_indexed: {chunks}")
    return out, {"book_id": book_id, "summary": f"{local.get('title', '?')}"}


def _agent_tool_status(ctx: dict, args: dict) -> tuple[str, dict]:
    ix = indexer_status()
    cov = _coverage()
    out = (f"indexer: {'running — ' + str(ix.get('progress')) if ix.get('running') else 'stopped'}\n"
           f"coverage: {cov['books_indexed']}/{cov['books_total']} books, "
           f"{cov['chunks_indexed']} chunks")
    return out, {"summary": f"{cov['books_indexed']}/{cov['books_total']} books indexed"}


def _agent_book_line(b: dict) -> str:
    idx = (f"{b['chunks_indexed']}/{b['pages']} pages indexed"
           if b["chunks_indexed"] < b["pages"] else "fully indexed")
    return (f"{b['title']}  ({b['authors'] or '?'}, {b.get('pub_date') or '?'})\n"
            f"   {b.get('ddc_path', '?')} · {b['pages']} pages · {idx} · "
            f"book_id {b['book_id']}")


def _agent_tool_browse(ctx: dict, args: dict) -> tuple[str, dict]:
    """browse_ddc: the section tree (no ddc) or the books in one section."""
    ddc = str(args.get("ddc") or "").strip()
    try:
        limit = min(40, max(1, int(args.get("limit", 20))))
    except (TypeError, ValueError):
        limit = 20
    if not ddc:
        tree = _ddc_tree()
        lines = ["library (on disk):"]
        for c in tree:
            lines.append(f"{c['ddc']} — {c['books']} books")
            for s in c["sections"]:
                lines.append(f"   {s['ddc']} ({s['books']})")
        out = "\n".join(lines)
        if len(out) > AGENT_TOOL_CHARS:
            out = out[:AGENT_TOOL_CHARS].rsplit("\n", 1)[0] + "\n…(truncated)"
        return out, {"summary": f"{len(tree)} DDC sections"}
    books = _catalog_books(ddc=ddc)[:limit]
    if not books:
        return (f"no books in {ddc!r} — use browse_ddc with no args to see "
                f"what sections exist.", {"summary": "empty section"})
    out = (f"{len(books)} book(s) in {ddc}:\n\n"
           + "\n\n".join(_agent_book_line(b) for b in books))
    if len(out) > AGENT_TOOL_CHARS:
        out = out[:AGENT_TOOL_CHARS].rsplit("\n", 1)[0] + "\n…(truncated)"
    return out, {"summary": f"{len(books)} books in {ddc}"}


def _agent_tool_find(ctx: dict, args: dict) -> tuple[str, dict]:
    q = str(args.get("q") or "").strip()
    if len(q) < 2:
        return "q is too short — use at least 2 characters.", {}
    try:
        limit = min(20, max(1, int(args.get("limit", 10))))
    except (TypeError, ValueError):
        limit = 10
    books = _catalog_books(q=q)[:limit]
    if not books:
        return (f"no books matching {q!r} in the catalog — try a shorter "
                f"keyword, or browse_ddc to explore sections.",
                {"summary": "no matches"})
    out = (f"{len(books)} book(s) matching {q!r}:\n\n"
           + "\n\n".join(_agent_book_line(b) for b in books))
    if len(out) > AGENT_TOOL_CHARS:
        out = out[:AGENT_TOOL_CHARS].rsplit("\n", 1)[0] + "\n…(truncated)"
    return out, {"summary": f"{len(books)} books match {q!r}"}


def _agent_tool_page(ctx: dict, args: dict) -> tuple[str, dict]:
    book_id = str(args.get("book_id") or "")
    try:
        page = int(args.get("page_num"))
    except (TypeError, ValueError):
        return "page_num must be an integer (1-based).", {}
    local = ctx["by_bid"].get(book_id)
    if not local:
        return (f"book_id {book_id!r} not in the catalog — use browse_ddc or "
                f"find_books first.", {"book_id": book_id})
    try:
        rows = pq.read_table(safe_path(local["path"]),
                             columns=["page_num", "text"],
                             filters=[("page_num", "=", page)]).to_pylist()
    except Exception as e:
        return f"page read failed: {e}", {"page_num": page}
    if not rows:
        return (f"page {page} not in this book ({local['pages']} pages, "
                f"1-{local['pages']}).", {"page_num": page})
    text = " ".join((rows[0].get("text") or "").split())
    n = _agent_register(ctx, [{"payload": {"book_id": book_id,
                                           "page_num": page,
                                           "book_title": local["title"],
                                           "path": local["path"]}}])[0]
    head = (f"[{n}] {local['title']} p.{page} of {local['pages']}  "
            f"({local.get('authors') or '?'}, {local.get('ddc_path') or '?'})\n"
            if n else f"{local['title']} p.{page}\n")
    if not text:
        out = head + ("scanned page — no text layer. " +
                      (f"PDF: {local['path'].rsplit('.', 1)[0]}.pdf"
                       if local.get("has_pdf") else "no PDF available"))
    else:
        if len(text) > AGENT_CHUNK_CHARS:
            text = text[:AGENT_CHUNK_CHARS].rsplit(" ", 1)[0] + "…"
        out = head + text
    return out, {"page_num": page, "summary": f"{local['title']} p.{page}"}


_AGENT_DISPATCH = {
    "search_library": _agent_tool_search,
    "get_chunk": _agent_tool_chunk,
    "get_source": _agent_tool_source,
    "library_status": _agent_tool_status,
    "browse_ddc": _agent_tool_browse,
    "find_books": _agent_tool_find,
    "get_page": _agent_tool_page,
}


def _llm_tools(messages: list[dict], use_tools: bool = True) -> dict:
    """One chat completion; returns the raw assistant message (content and/or
    tool_calls)."""
    body = {"model": LLM_MODEL, "messages": messages,
            "temperature": 0.2, "max_tokens": AGENT_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False}}
    if use_tools:
        body["tools"] = AGENT_TOOLS
    r = requests.post(f"{LLM_BASE_URL}/v1/chat/completions", json=body,
                      timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return (r.json()["choices"][0] or {}).get("message") or {}


def _do_agent_chat(body: dict):
    """The agent loop. A generator: yields ("step", step) for every tool call
    as it happens, then ("result", payload) — or raises AgentError."""
    q = (body.get("q") or "").strip()
    ddc = (body.get("ddc") or "").strip()
    if len(q) < 3:
        raise AgentError(400, "q must be at least 3 characters")
    t0 = time.time()
    cov = _coverage()
    system = AGENT_SYSTEM.format(books_total=cov["books_total"],
                                 books_indexed=cov["books_indexed"],
                                 chunks_indexed=cov["chunks_indexed"])
    ctx = _agent_ctx()
    messages = [{"role": "system", "content": system}]
    for m in (body.get("history") or [])[-8:]:
        role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
        content = str(m.get("content") or "")[:2500]
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user",
                     "content": (f"(scope to DDC section {ddc})\n" if ddc else "") + q})

    answer, iterations = None, 0
    for _ in range(AGENT_MAX_ITERS):
        iterations += 1
        if time.time() - t0 > AGENT_TOTAL_TIMEOUT:
            break
        try:
            msg = _llm_tools(messages)
        except requests.RequestException as e:
            raise AgentError(503, (f"answer engine unreachable ({e}) — LLM at "
                                   f"{LLM_BASE_URL} down?"))
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            break
        messages.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (ValueError, TypeError):
                args = {}
            executor = _AGENT_DISPATCH.get(name)
            t1 = time.time()
            if executor is None:
                result, meta = f"error: unknown tool {name!r}.", {}
            else:
                try:
                    result, meta = executor(ctx, args)
                except Exception as e:  # keep the loop alive; the model adapts
                    result, meta = f"error: {e}", {}
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": result})
            step = {"i": len(ctx["steps"]) + 1, "tool": name,
                    "args": {k: args.get(k) for k in
                             ("q", "mode", "ddc", "book_id", "point_id",
                              "page_num", "limit")
                             if args.get(k) is not None},
                    "summary": meta.get("summary") or result[:120],
                    "ms": int((time.time() - t1) * 1000)}
            if (msg.get("content") or "").strip():
                step["note"] = str(msg["content"]).strip()[:300]
            ctx["steps"].append(step)
            yield ("step", step)
    if answer is None:
        # budget exhausted while still issuing tool calls: force a synthesis
        # turn without tools so the user always gets an answer.
        try:
            msg = _llm_tools(messages, use_tools=False)
            answer = (msg.get("content") or "").strip()
        except requests.RequestException as e:
            raise AgentError(503, (f"answer engine unreachable ({e}) — LLM at "
                                   f"{LLM_BASE_URL} down?"))
    if not answer:
        answer = "(the answer engine returned no text — try a shorter question)"
    _autolog({"layer": "user", "query": q, "tool": "agent",
              "ddc": ddc or None,
              "book_ids": sorted({s["book_id"] for s in ctx["sources"]
                                  if s.get("book_id")}) or None,
              "n_results": len(ctx["sources"]),
              "result_summary": (f"{len(ctx['steps'])} tool steps · "
                                 f"{len(ctx['sources'])} pages cited"),
              "answer": answer})
    yield ("result", {
        "answer": answer, "sources": ctx["sources"], "steps": ctx["steps"],
        "iterations": iterations, "model": LLM_MODEL,
        "gen_ms": int((time.time() - t0) * 1000), "coverage": _coverage(),
    })


def _do_agent_chat_sync(body: dict) -> dict:
    """JSON-route wrapper: run the agent to completion, return the payload."""
    for kind, payload in _do_agent_chat(body):
        if kind == "result":
            return payload
    raise AgentError(502, "agent produced no answer")


def _agent_stream(body: dict):
    """SSE generator for the dashboard: step events live, then one done event."""
    try:
        for kind, payload in _do_agent_chat(body):
            event = "step" if kind == "step" else "done"
            yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
    except AgentError as e:
        yield f"event: error\ndata: {json.dumps({'status': e.status, 'error': e.msg})}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def api_health():
    n_cache = len(list(CACHE_DIR.glob("*.png"))) if CACHE_DIR.exists() else 0
    return jsonify(ok=True, root=str(DDC_ROOT), cache_files=n_cache)


@app.get("/api/browse")
def api_browse():
    rel = request.args.get("path", "")
    p = safe_path(rel)
    if not p.is_dir():
        abort(404, description="not a directory")
    dirs: list[str] = []
    books: list[dict] = []
    for entry in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        elif entry.suffix.lower() == ".parquet" and is_book_parquet(entry):
            try:
                books.append(book_summary(entry))
            except Exception:
                pass
    return jsonify({
        "name": DDC_ROOT.name if p == DDC_ROOT else p.name,
        "parent": "" if p.parent == DDC_ROOT else str(p.relative_to(DDC_ROOT).parent),
        "path": rel,
        "dirs": dirs,
        "books": books,
    })


@app.get("/api/book")
def api_book():
    p = safe_path(request.args.get("path", ""))
    if not (p.is_file() and p.suffix.lower() == ".parquet") or not is_book_parquet(p):
        abort(404, description="not a book parquet")
    b = book_for(p)
    pdf_pages = 0
    if b.pdf.exists():
        with _pdf_lock:
            try:
                pdf_pages = _open_pdf(b.pdf).page_count
            except Exception:
                pdf_pages = 0
    return jsonify({
        "path": request.args.get("path"),
        "title": b.meta.get("final_title") or p.stem,
        "authors": b.meta.get("authors"),
        "publisher": b.meta.get("publisher"),
        "pub_date": b.meta.get("pub_date"),
        "ddc": b.meta.get("ddc"),
        "summary": b.meta.get("summary"),
        "page_nums": b.page_nums,
        "is_index": b.is_index,
        "has_pdf": b.pdf.exists(),
        "has_images": bool(b.image_paths),
        "pdf_pages": pdf_pages,
    })


@app.get("/api/page")
def api_page():
    p = safe_path(request.args.get("path", ""))
    try:
        page = int(request.args.get("page", "0"))
    except ValueError:
        abort(400, description="page must be an integer")
    b = book_for(p)
    if page not in b.texts:
        abort(404, description=f"page {page} not in book (has {len(b.page_nums)} pages)")
    return jsonify(page_num=page, text=b.texts[page], is_index=b.is_index[page])


@app.get("/api/page-image")
def api_page_image():
    p = safe_path(request.args.get("path", ""))
    try:
        page = int(request.args.get("page", "0"))
        dpi = min(320, max(72, int(request.args.get("dpi", "150"))))
    except ValueError:
        abort(400, description="page/dpi must be integers")
    b = book_for(p)
    if not b.pdf.exists():
        # Non-PDF books (EPUB/TXT ingestion): no page to render — fall back
        # to the images stored on this page (figures, cover) if any.
        imgs = b.image_paths.get(page, [])
        if imgs:
            png = compose_page_images(imgs)
            if png is not None:
                resp = Response(png, mimetype="image/png")
                resp.headers["Cache-Control"] = "public, max-age=86400"
                return resp
        abort(404, description="no page image (no PDF sibling and no stored image on this page)")
    try:
        png = render_pdf_page(b.pdf, page, dpi)
    except RuntimeError as e:
        abort(404, description=str(e))
    except Exception as e:
        abort(500, description=f"render failed: {e}")
    resp = Response(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.get("/api/search")
def api_search_book():
    """Search the text of one book; returns matching pages + snippets."""
    p = safe_path(request.args.get("path", ""))
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify(q=q, count=0, results=[])
    b = book_for(p)
    out: list[dict] = []
    for pn in b.page_nums:
        t = b.texts.get(pn, "")
        if not t:
            continue
        i = t.lower().find(q)
        if i >= 0:
            out.append({
                "page": pn,
                "snippet": t[max(0, i - 60): i + len(q) + 90].replace("\n", " "),
            })
        if len(out) >= 50:
            break
    return jsonify(q=q, count=len(out), results=out)


@app.get("/api/search-books")
def api_search_books():
    """Global title/author search (index built in the background at startup)."""
    q = request.args.get("q", "").strip().lower()
    with GLOBAL["lock"]:
        ready, count, books = GLOBAL["ready"], GLOBAL["count"], list(GLOBAL["books"])
    if q:
        books = [b for b in books
                 if q in (b["title"] + " " + b["authors"]).lower()][:30]
    return jsonify(ready=ready, count=count, books=books)


@app.get("/api/semantic")
def api_semantic():
    """Semantic search over the Qdrant index built by the indexer.

    The query is embedded with the SAME model used for the index (EMBED_MODEL,
    default bge-m3), so vectors are comparable. Hits are mapped back to local
    books via book_id so the UI can jump straight to book + page.
    """
    q = request.args.get("q", "").strip()
    try:
        k = min(25, max(1, int(request.args.get("k", str(SEMANTIC_K)))))
    except ValueError:
        k = SEMANTIC_K
    if len(q) < 2:
        return jsonify(q=q, count=0, hits=[])
    try:
        pts, _embed_ms, _q_ms = _search(q, k, "dense")
    except requests.RequestException as e:
        abort(503, description=f"retrieval failed ({e}) — Qdrant/Ollama down?")
    with GLOBAL["lock"]:
        by_bid = dict(GLOBAL["by_book_id"])
    hits = []
    for p in pts:
        pl = p.get("payload") or {}
        bid = pl.get("book_id")
        local = by_bid.get(bid) or {}
        hits.append({
            "score": round(float(p.get("score", 0.0)), 4),
            "book_id": bid,
            "title": pl.get("book_title") or local.get("title") or "?",
            "authors": pl.get("authors") or "",
            "ddc": pl.get("ddc") or "",
            "page_num": pl.get("page_num"),
            "snippet": _snippet(pl.get("text") or "", q),
            "path": pl.get("path") or local.get("path"),
        })
    return jsonify(q=q, count=len(hits),
                   books_indexed=_indexed_books(), coverage=_coverage(),
                   hits=hits)


@app.get("/api/index/status")
def api_index_status():
    """One-machine status: Qdrant, Ollama, indexer process, index coverage."""
    q = {"ok": False, "points": None}
    try:
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                          json={"exact": True}, timeout=5)
        r.raise_for_status()
        q = {"ok": True, "points": r.json()["result"]["count"]}
    except requests.RequestException:
        pass
    o = {"ok": False, "model_ready": False}
    try:
        r = requests.get(f"{EMBED_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [(m.get("name") or "").split(":")[0]
                 for m in r.json().get("models", [])]
        o = {"ok": True, "model_ready": EMBED_MODEL.split(":")[0] in names}
    except requests.RequestException:
        pass
    with GLOBAL["lock"]:
        total = GLOBAL["count"]
    return jsonify(
        qdrant=q,
        ollama=o,
        indexer=indexer_status(),
        books_total=total,
        books_indexed=_indexed_books() if q["ok"] else 0,
        collection=COLLECTION,
        model=EMBED_MODEL,
    )


@app.post("/api/index/start")
def api_index_start():
    """Start the indexer in the background (appends to the shared run log)."""
    script = INDEXER_DIR / "index_books.py"
    if not script.is_file():
        abort(500, description=f"indexer script not found: {script}")
    if indexer_status()["running"]:
        abort(409, description="indexer is already running")
    logf = open(INDEX_RUN_LOG, "ab")
    subprocess.Popen(
        [sys.executable, "-u", str(script), "--root", str(DDC_ROOT)],
        stdout=logf, stderr=subprocess.STDOUT, cwd=str(INDEXER_DIR),
        start_new_session=True, close_fds=True)
    return jsonify(started=True, log=str(INDEX_RUN_LOG))


@app.post("/api/index/stop")
def api_index_stop():
    # [.] so the pattern can never match this viewer's own command line
    # or a shell quoting the pattern.
    r = subprocess.run(["pkill", "-TERM", "-f", "index_books[.]py --root"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        abort(404, description="indexer is not running")
    return jsonify(stopped=True)


def _do_chat(body: dict) -> dict:
    """RAG pipeline: hybrid retrieval + cited LLM synthesis.
    Shared by POST /api/chat and POST /v1/ask."""
    q = (body.get("q") or "").strip()
    ddc = (body.get("ddc") or "").strip()
    if len(q) < 3:
        abort(400, description="q must be at least 3 characters")
    try:
        k = min(24, max(4, int(body.get("k", CHAT_K))))
    except (TypeError, ValueError):
        k = CHAT_K
    t0 = time.time()
    try:
        hits = _hybrid_search(q, k, ddc)
    except requests.RequestException as e:
        abort(503, description=f"retrieval failed ({e}) — Qdrant/Ollama down?")
    retrieval_ms = int((time.time() - t0) * 1000)
    if not hits:
        _autolog({"layer": "user", "query": q, "tool": "ask",
                  "ddc": ddc or None, "n_results": 0,
                  "result_summary": "no results"})
        return {"answer": ("The index has no pages matching that yet "
                           "(the indexer may still be running)."),
                "sources": [], "model": LLM_MODEL,
                "retrieval_ms": retrieval_ms, "gen_ms": 0,
                "coverage": _coverage()}
    sources, context = _build_context(hits)
    msgs = [{"role": "system", "content": CHAT_SYSTEM}]
    for m in (body.get("history") or [])[-8:]:
        role = m.get("role") if m.get("role") in ("user", "assistant") else "user"
        msgs.append({"role": role, "content": str(m.get("content") or "")[:4000]})
    msgs.append({"role": "user",
                 "content": "Question: " + q +
                            "\n\nNumbered source pages (cite them):\n\n" + context})
    t1 = time.time()
    try:
        answer = _llm_ask(msgs)
    except requests.RequestException as e:
        abort(503, description=(f"answer engine unreachable ({e}) — LLM at "
                                f"{LLM_BASE_URL} down?"))
    gen_ms = int((time.time() - t1) * 1000)
    _autolog({"layer": "user", "query": q, "tool": "ask",
              "ddc": ddc or None,
              "book_ids": sorted({(p.get("payload") or {}).get("book_id")
                                  for p in hits
                                  if (p.get("payload") or {}).get("book_id")}) or None,
              "point_ids": [str(p.get("id")) for p in hits
                            if p.get("id") is not None],
              "n_results": len(hits),
              "result_summary": _hits_summary(hits),
              "answer": answer},
             refs=_hits_refs(hits))
    return {"answer": answer, "sources": sources, "model": LLM_MODEL,
            "retrieval_ms": retrieval_ms, "gen_ms": gen_ms,
            "coverage": _coverage()}


@app.post("/api/chat")
def api_chat():
    """Ask the library a question: hybrid retrieval + cited LLM synthesis.

    JSON body: {"q": "...", "ddc": "400_Language" (optional),
                "history": [{"role": "user"|"assistant", "content": "..."}]}
    Returns:   {"answer": markdown with [n] citations,
                "sources": [{n, title, page, path, has_pdf}...],
                "model", "retrieval_ms", "gen_ms", "coverage"}
    """
    return jsonify(_do_chat(request.get_json(silent=True) or {}))


@app.post("/api/agent")
def api_agent():
    """Agent answer: the LLM reasons between searches (reformulates, scopes,
    pulls full pages) and cites [n] pages from what it actually found.

    Same body as /api/chat ({q, ddc?, history?}), streamed as SSE:
      event: step  — each tool call as it happens (live reasoning trace)
      event: done  — {"answer", "sources", "steps", ...}
                     (same shape as /api/chat plus the step trace)
      event: error — {"status", "error"}
    """
    body = request.get_json(silent=True) or {}
    return Response(_agent_stream(body),
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"},
                    mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# /v1 — the stable machine API (for your RAG, for me, for the CLI + MCP)
#
# /api/* is the human UI's contract; /v1 is the stable, versioned contract
# that tools link against. Both share the same retrieval + answer engine.
# ---------------------------------------------------------------------------
v1 = Blueprint("v1", __name__, url_prefix="/v1")


@v1.get("/status")
def v1_status():
    """One-shot health + coverage: is everything up, and how complete is
    the index that searches are running against right now."""
    q = {"ok": False, "points": None}
    try:
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                          json={"exact": True}, timeout=5)
        r.raise_for_status()
        q = {"ok": True, "points": r.json()["result"]["count"]}
    except requests.RequestException:
        pass
    o = {"ok": False, "model_ready": False}
    try:
        r = requests.get(f"{EMBED_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [(m.get("name") or "").split(":")[0]
                 for m in r.json().get("models", [])]
        o = {"ok": True, "model_ready": EMBED_MODEL.split(":")[0] in names}
    except requests.RequestException:
        pass
    qe = {"ok": False}
    try:
        r = requests.get(f"{EMBED_QUERY_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        qe = {"ok": True, "url": EMBED_QUERY_BASE_URL}
    except requests.RequestException:
        pass
    with GLOBAL["lock"]:
        total = GLOBAL["count"]
        ready = GLOBAL["ready"]
    return jsonify(
        service="ddc-library", version=1,
        root=str(DDC_ROOT),
        qdrant=q, ollama=o, query_embedder=qe, llm=LLM_MODEL,
        indexer=indexer_status(),
        books_total=total, ready=ready,
        coverage=_coverage(),
        collection=COLLECTION, embed_model=EMBED_MODEL,
    )


@v1.post("/search")
def v1_search():
    """The canonical retrieval endpoint.

    Body: {"q": "...", "k": 10, "mode": "hybrid"|"dense"|"keyword",
           "filters": {"book_id","ddc_coarse","ddc_tens","ddc","pub_date",
                        "min_page","max_page"},
           "fields": ["snippet","text"]}
    Every result carries a `locator` (local path + page) so any client can
    fetch the original artifact. `coverage` says how complete the index is.
    """
    body = request.get_json(silent=True) or {}
    q = (body.get("q") or "").strip()
    try:
        k = min(50, max(1, int(body.get("k", 10))))
    except (TypeError, ValueError):
        k = 10
    mode = body.get("mode") or "hybrid"
    if mode not in ("hybrid", "dense", "keyword"):
        abort(400, description=f"mode must be hybrid|dense|keyword (got {mode!r})")
    if len(q) < 2:
        return jsonify(q=q, k=k, mode=mode, results=[], coverage=_coverage())
    filters = body.get("filters") or {}
    t0 = time.time()
    try:
        pts, embed_ms, q_ms = _search(q, k, mode, filters)
    except requests.RequestException as e:
        abort(503, description=f"retrieval failed ({e}) — Qdrant/Ollama down?")
    include_text = "text" in (body.get("fields") or [])
    results = [_result_from_point(p, q, include_text) for p in pts]
    _autolog({"layer": "user", "query": q, "tool": "search", "mode": mode,
              "ddc": str(filters.get("ddc_coarse") or filters.get("ddc") or "") or None,
              "book_ids": sorted({r["book_id"] for r in results
                                  if r.get("book_id")}) or None,
              "point_ids": [r["id"] for r in results],
              "n_results": len(results),
              "result_summary": _hits_summary(pts)},
             refs=_hits_refs(pts))
    return jsonify(q=q, k=k, mode=mode, results=results,
                   coverage=_coverage(),
                   ms={"embed": embed_ms, "qdrant": q_ms,
                       "total": int((time.time() - t0) * 1000)})


@v1.post("/ask")
def v1_ask():
    """RAG answer: hybrid retrieval + cited LLM synthesis (same engine as
    POST /api/chat). One search, one answer — the fast path. Body:
    {"q", "ddc"?, "k"?, "history"?}."""
    return jsonify(_do_chat(request.get_json(silent=True) or {}))


@v1.post("/chat")
def v1_chat():
    """Agent answer: the LLM reasons between searches. It gets the library
    as tools (search_library, get_chunk, get_source, library_status), so it
    inspects results, reformulates queries, scopes to DDC sections, and
    pulls full page text before synthesizing. Slower than /v1/ask, deeper.

    Body: {"q", "ddc"?, "history"?}  (same as /v1/ask minus k)
    Returns: {"answer": markdown with [n] citations,
              "sources": [{n, title, page, path, has_pdf}...],
              "steps": [{i, tool, args, summary, ms, note?}...],
              "iterations", "model", "gen_ms", "coverage"}
    The OWUI bridge (ddc_owui.py) routes its ddc-library model here."""
    try:
        return jsonify(_do_agent_chat_sync(request.get_json(silent=True) or {}))
    except AgentError as e:
        abort(e.status, description=e.msg)


@v1.get("/chunk/<int:point_id>")
def v1_chunk(point_id):
    """One chunk by its deterministic Qdrant point id: full text + meta +
    locator. Stable id => cacheable / referenceable across calls."""
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}/points/{point_id}",
                         params={"with_payload": True}, timeout=15)
        r.raise_for_status()
        p = r.json()["result"]
    except requests.RequestException as e:
        abort(503, description=f"Qdrant unreachable ({e})")
    if not p or not p.get("payload"):
        abort(404, description=f"point {point_id} not in {COLLECTION}")
    out = _result_from_point(p, "")
    out["text"] = (p.get("payload") or {}).get("text") or ""
    _autolog({"layer": "user", "tool": "chunk",
              "book_ids": [out["book_id"]] if out.get("book_id") else None,
              "point_ids": [str(point_id)],
              "n_results": 1,
              "result_summary": f"{out['doc_title']} p.{out['page_num']}"})
    return jsonify(out)


@v1.get("/source/<book_id>")
def v1_source(book_id):
    """A source (book) by book_id: local metadata + page count + how many of
    its chunks are currently in the index."""
    with GLOBAL["lock"]:
        local = GLOBAL["by_book_id"].get(book_id)
    if not local:
        abort(404, description=f"book_id {book_id!r} not in the local library")
    out = {"source_type": "book", **local}
    out["chunks_indexed"] = _book_chunk_counts().get(book_id, 0)
    return jsonify(out)


@v1.get("/books")
def v1_books():
    """The book catalog — the files on disk, not the vector index (works for
    books not yet indexed).

    No params      -> the DDC section tree with book counts.
    ?ddc=<code>    -> books in that DDC section (prefix match at any depth:
                      "500", "500_Science", "540", "540_Chemistry", a path…)
    ?q=<keyword>   -> title/author search (combinable with ddc).
    ?limit&offset  -> paging (limit 1-200, default 50).
    Every book carries chunks_indexed + ddc_path + book_id."""
    ddc = (request.args.get("ddc") or "").strip()
    q = (request.args.get("q") or "").strip()
    try:
        limit = min(200, max(1, int(request.args.get("limit", "50"))))
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        abort(400, description="limit/offset must be integers")
    with GLOBAL["lock"]:
        ready = GLOBAL["ready"]
    if not ddc and not q:
        return jsonify(ready=ready, tree=_ddc_tree())
    books = _catalog_books(ddc, q)
    total = len(books)
    return jsonify(ready=ready, ddc=ddc, q=q, total=total,
                   offset=offset, limit=limit,
                   books=books[offset:offset + limit])


@v1.get("/page/<book_id>/<int:page_num>")
def v1_page(book_id: str, page_num: int):
    """Full text of one page of one book, read from the source parquet (not
    the search index — works for unindexed books). page_num is 1-based, as in
    search results and the reader UI."""
    with GLOBAL["lock"]:
        local = GLOBAL["by_book_id"].get(book_id)
    if not local:
        abort(404, description=(f"book_id {book_id!r} not in the catalog — "
                                f"use /v1/books or /v1/search to find it"))
    p = safe_path(local["path"])
    try:
        t = pq.read_table(p, columns=["page_num", "text"],
                          filters=[("page_num", "=", page_num)])
        rows = t.to_pylist()
    except Exception as e:
        abort(500, description=f"page read failed: {e}")
    if not rows:
        abort(404, description=(f"page {page_num} not in this book "
                                f"({local['pages']} pages, 1-"
                                f"{local['pages']})"))
    text = " ".join((rows[0].get("text") or "").split())
    out = {"book_id": book_id, "title": local["title"],
           "authors": local["authors"], "pub_date": local.get("pub_date", ""),
           "ddc": local["ddc"], "ddc_path": local.get("ddc_path", ""),
           "path": local["path"], "page_num": page_num,
           "pages_total": local["pages"], "has_pdf": local["has_pdf"],
           "text": text, "text_chars": len(text)}
    if not text:
        out["note"] = ("scanned page — no text layer. " +
                       (f"PDF available: {local['path'].rsplit('.', 1)[0]}.pdf"
                        if local["has_pdf"] else
                        "no PDF available for this book"))
    _autolog({"layer": "user", "tool": "page",
              "book_ids": [book_id],
              "ddc": local.get("ddc") or None,
              "n_results": 1,
              "result_summary": f"{local['title']} p.{page_num}"})
    return jsonify(out)


@v1.get("/image")
def v1_image():
    """Rendered page: ?path=&page=&dpi= (same as /api/page-image)."""
    return api_page_image()


@v1.post("/rescan")
def v1_rescan():
    """Re-walk the DDC tree now, so books classified/renamed/deleted since
    the last scan appear immediately in /v1/books, /v1/page and the MCP
    browse_ddc / find_books / get_page tools. The catalog also refreshes
    itself every CATALOG_RESCAN_INTERVAL seconds (default 60)."""
    r = rescan_catalog()
    return jsonify(r), (202 if r.get("rescanning") else 200)


@v1.get("/index/status")
def v1_index_status():
    return api_index_status()


@v1.post("/index/start")
def v1_index_start():
    return api_index_start()


@v1.post("/index/stop")
def v1_index_stop():
    return api_index_stop()


# ---------------------------------------------------------------------------
# Hit database (MySQL, squig_research) — the two-layer research log.
#
# layer='user'  — things the USER asked to look at (also auto-recorded by
#                 every /v1 search/ask/agent/chunk/page above when
#                 HITS_AUTOLOG=1).
# layer='agent' — the system's OWN research (off-time roaming): recorded
#                 explicitly through POST /v1/hits (the log_hit MCP tool),
#                 so it never gets lost under the user's session.
# ---------------------------------------------------------------------------
@v1.get("/hits/status")
def v1_hits_status():
    """Hit-log health: is MySQL reachable, and how many hits each layer
    holds."""
    ok, where = hits_db.ping()
    out: dict = {"ok": ok, "where": where if ok else hits_db._last_error["msg"]}
    if ok:
        try:
            out["layers"] = hits_db.layer_counts()
        except hits_db.HitDBError as e:
            out["layers"] = {}
            out["error"] = str(e)
    return jsonify(out)


@v1.post("/hits")
def v1_hits_record():
    """Record one hit in the research log.

    Body: {"layer": "user"|"agent" (default "agent"),
           "query"?, "note"?, "answer"?, "result_summary"?,
           "tool"?, "mode"?, "ddc"?, "session"?,
           "book_ids": [...], "point_ids": [...], "n_results": 0,
           "refs": [{"type": "book"|"chunk"|"ddc", "id": "..."}],
           -- the v2 research-loop fields --
           "page_start"?, "page_end"?, "topic"? (comma-separated tags),
           "cross_refs"? ([{"book_id","page","ddc","label"}]),
           "status"? (found|needs-verify|verified|done, default found),
           "confidence"? (low|medium|high, default medium),
           "session_id"? (groups related hits from one research walk)}
    Returns {"id": <new hit id>}."""
    body = request.get_json(silent=True) or {}
    if not any(body.get(k) for k in ("query", "note", "answer", "result_summary")):
        abort(400, description="record something: at least one of "
                               "query / note / answer / result_summary")
    hit = {k: body.get(k) for k in
           ("layer", "session", "query", "tool", "mode", "ddc",
            "book_ids", "point_ids", "n_results", "result_summary",
            "note", "answer",
            "page_start", "page_end", "topic", "cross_refs",
            "status", "confidence", "session_id")}
    refs = [(str(r.get("type") or "book")[:8], str(r.get("id")))
            for r in (body.get("refs") or []) if isinstance(r, dict)]
    try:
        hid = hits_db.record_hit(hit, refs or None)
    except hits_db.HitDBError as e:
        abort(400 if "must be one of" in str(e) else 503, description=str(e))
    return jsonify(id=hid)


@v1.get("/hits")
def v1_hits_recall():
    """Query the research log, newest first.

    ?layer=user|agent&session=&session_id=&book=<book_id>&q=&ddc=
    &status=found|needs-verify|verified|done&topic=<tag substring>
    &cross_ref_book=<value inside cross_refs>&limit=1-100&offset=0
    Returns {"hits": [{id, ts, layer, session, query, tool, mode, ddc,
    book_ids, n_results, result_summary, note, answer (truncated),
    page_start, page_end, topic, cross_refs (parsed list), status,
    confidence, session_id}...],
             "total", "layers": {layer: count}}."""
    try:
        rows, total, counts = hits_db.recall_hits(
            layer=(request.args.get("layer") or "").strip(),
            session=(request.args.get("session") or "").strip(),
            session_id=(request.args.get("session_id") or "").strip(),
            book=(request.args.get("book") or "").strip(),
            q=(request.args.get("q") or "").strip(),
            ddc=(request.args.get("ddc") or "").strip(),
            status=(request.args.get("status") or "").strip(),
            topic=(request.args.get("topic") or "").strip(),
            cross_ref_book=(request.args.get("cross_ref_book") or "").strip(),
            limit=int(request.args.get("limit", "20")),
            offset=int(request.args.get("offset", "0")))
    except ValueError:
        abort(400, description="limit/offset must be integers")
    except hits_db.HitDBError as e:
        abort(503, description=f"hit log unreachable: {e}")
    return jsonify(hits=rows, total=total, layers=counts)


@v1.get("/hits/pending")
def v1_hits_pending():
    """The automation loop's frontier pull: every hit stuck at
    status='needs-verify', oldest first (FIFO), all fields including
    cross_refs. ?limit=1-100 (default 10)."""
    try:
        limit = int(request.args.get("limit", "10"))
    except ValueError:
        abort(400, description="limit must be an integer")
    try:
        rows = hits_db.hits_pending(limit)
    except hits_db.HitDBError as e:
        abort(503, description=f"hit log unreachable: {e}")
    return jsonify(hits=rows, count=len(rows))


@v1.post("/hits/mark")
def v1_hits_mark():
    """Advance one hit without rewriting it (the loop's bookkeeping call).

    Body: {"hit_id": 14, "status"?: found|needs-verify|verified|done,
           "confidence"?: low|medium|high,
           "note_append"?: "went back and verified: …",
           "cross_refs_append"??: [{"book_id","page","ddc","label"}]}
    status/confidence overwrite the column; note_append is appended
    (\\n separated); cross_refs_append merges into the existing array,
    deduped by (book_id, page). Returns {"hit": <the updated hit>}."""
    body = request.get_json(silent=True) or {}
    if not body.get("hit_id"):
        abort(400, description="hit_id is required")
    try:
        hit = hits_db.hits_mark(
            int(body["hit_id"]),
            status=body.get("status"),
            confidence=body.get("confidence"),
            note_append=body.get("note_append"),
            cross_refs_append=body.get("cross_refs_append"))
    except (TypeError, ValueError):
        abort(400, description="hit_id must be an integer")
    except hits_db.HitDBError as e:
        code = 404 if "not found" in str(e) else (
            400 if "must be one of" in str(e) else 503)
        abort(code, description=str(e))
    return jsonify(hit=hit)


app.register_blueprint(v1)


@app.get("/<path:subpath>")
def spa(subpath):
    """SPA catch-all: /book/<path>, /browse/<path>, … all serve the app,
    which routes client-side (deep links / hard refreshes work). API routes
    above are exact/static rules, so they win over this catch-all."""
    return render_template("index.html")


if __name__ == "__main__":
    if not DDC_ROOT.is_dir():
        raise SystemExit(f"DDC root not found: {DDC_ROOT}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=rescan_catalog, daemon=True).start()
    threading.Thread(target=_rescan_loop, daemon=True).start()
    threading.Thread(target=hits_db.ensure_schema, daemon=True).start()
    threading.Thread(target=_hits_worker, daemon=True).start()
    print(f"[reader] DDC root: {DDC_ROOT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
