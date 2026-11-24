#!/home/matt/miniconda3/envs/marker_env/bin/python
"""MCP server for the DDC library.

Exposes the /v1 API (http://localhost:8071) as native MCP tools so any
MCP-capable agent (OpenCode, Claude Desktop, ...) can search, read and
ask the local library without knowing the HTTP contract:

    search_library  — hybrid/dense/keyword chunk search with DDC scoping
    ask_library     — RAG answer with [n] citations
    get_chunk       — full text of one chunk by its point id
    get_source      — book metadata + index coverage for a book_id
    library_status  — health + how complete the index is
    browse_ddc      — the DDC section tree, or the books in one section
    find_books      — catalog search by title/author keyword
    get_page        — full text of one page of a book, by book_id + page
    rescan_library  — force an immediate re-scan of the DDC tree (the
                      catalog auto-refreshes every minute; this is for
                      right after a classification run)
    log_hit         — record a hit in the MySQL research log (two layers:
                      user's work / the agent's own off-time research)
    hits_recall     — query that log: what was searched, what was found
    hits_pending    — the loop's frontier pull (needs-verify, FIFO)
    hits_mark       — advance a hit (status / confidence / note / refs)

Point at a different viewer with READER_URL (default http://localhost:8071).

Run (stdio transport — the usual way MCP clients launch servers):
    /home/matt/base/reader/mcp_server.py

Open WebUI uses the same tools over HTTP: `ddc_owui.py` mounts them at
http://<host>:8072/mcp (streamable HTTP) — register that URL in OWUI's
admin UI under Connections → MCP.

OpenCode example (~/.config/opencode/opencode.json, already installed via
`opencode mcp add ddc --global -- ...`):
    "mcp": {
      "servers": {
        "ddc": {
          "type": "local",
          "command": ["/home/matt/miniconda3/envs/marker_env/bin/python",
                      "/home/matt/base/reader/mcp_server.py"]
        }
      }
    }
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

BASE = os.environ.get("READER_URL", "http://localhost:8071")


def create_server(base: str | None = None) -> FastMCP:
    """Build the FastMCP server with the eight library tools.

    Shared by the stdio entry point below (OpenCode) and ddc_owui.py
    (mounted at /mcp for Open WebUI's MCP connections).

    dns-rebinding protection is off: the HTTP transport serves a local
    single-machine network (Open WebUI's container reaches it as
    172.17.0.1), and the allowlist would otherwise 421 non-localhost
    Host headers.
    """
    base = base or BASE
    mcp = FastMCP("ddc-library",
                  transport_security=TransportSecuritySettings(
                      enable_dns_rebinding_protection=False))

    def _call(method: str, path: str, body=None) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read()).get("error", "")
            except Exception:
                err = e.reason
            raise RuntimeError(f"HTTP {e.code}: {err}")

    def _fmt_results(j: dict, show_text: bool = False) -> str:
        if not j["results"]:
            return "no results"
        lines = []
        for i, r in enumerate(j["results"], 1):
            loc = r.get("locator") or {}
            lines.append(
                f"{i}. {r['doc_title']} p.{r['page_num']}  "
                f"(score {r['score']}, point_id {r['id']})\n"
                f"   path: {loc.get('path') or '?'}\n"
                f"   {r['snippet']}")
            if show_text and r.get("text"):
                lines.append(f"   ---\n{r['text']}\n   ---")
        cov = j.get("coverage") or {}
        if cov:
            lines.append(f"[{j['mode']}] coverage: {cov['books_indexed']}/"
                         f"{cov['books_total']} books, "
                         f"{cov['chunks_indexed']} chunks")
        return "\n\n".join(lines)

    @mcp.tool()
    def search_library(q: str, k: int = 10, mode: str = "hybrid",
                       ddc: str = "", book_id: str = "",
                       full_text: bool = False) -> str:
        """Search the local DDC book library (parquet pages indexed in Qdrant).

        Args:
            q: the query (natural language, or exact phrases for keyword mode)
            k: number of results (1-50, default 10)
            mode: hybrid (dense bge-m3 + BM25, RRF-fused; default),
                  dense (embeddings only), keyword (BM25 only — instant, no GPU)
            ddc: optional top-level DDC scope, e.g. "500_Science"
            book_id: optional book_id to search inside one book
            full_text: include the full text of each chunk (not just a snippet)

        Returns numbered hits: title, page, local file path, point_id, snippet.
        Use the point_id with get_chunk for the full chunk.
        """
        if mode not in ("hybrid", "dense", "keyword"):
            raise ValueError(f"mode must be hybrid|dense|keyword (got {mode!r})")
        filters = {}
        if ddc:
            filters["ddc_coarse"] = ddc
        if book_id:
            filters["book_id"] = book_id
        j = _call("POST", "/v1/search",
                  {"q": q, "k": min(50, max(1, k)), "mode": mode,
                   "filters": filters,
                   "fields": ["text"] if full_text else []})
        return _fmt_results(j, full_text)

    @mcp.tool()
    def ask_library(q: str, ddc: str = "", k: int = 12) -> str:
        """Ask the library a question. Retrieves the most relevant pages
        (hybrid search) and synthesizes an answer with [n] citations,
        constrained to the retrieved sources.

        Args:
            q: the question
            ddc: optional top-level DDC scope, e.g. "500_Science"
            k: pages retrieved for synthesis (4-24, default 12)

        Returns the answer (markdown with [n] citations) plus the numbered
        source list (title, page, local path).
        """
        j = _call("POST", "/v1/ask", {"q": q, "ddc": ddc,
                                      "k": min(24, max(4, k))})
        out = [j["answer"], "", "sources:"]
        for s in j.get("sources") or []:
            out.append(f"  [{s['n']}] {s['title']} p.{s['page']}  {s.get('path') or ''}")
        cov = j.get("coverage") or {}
        if cov:
            out.append(f"coverage: {cov['books_indexed']}/{cov['books_total']} books")
        return "\n".join(out)

    @mcp.tool()
    def get_chunk(point_id: int) -> str:
        """Fetch the full text + metadata of one chunk by its Qdrant point id
        (the stable `id` from search_library results)."""
        j = _call("GET", f"/v1/chunk/{int(point_id)}")
        loc = j.get("locator") or {}
        head = (f"{j['doc_title']} p.{j['page_num']}  "
                f"(author: {j.get('authors') or '?'}, "
                f"pub: {j.get('pub_date') or '?'}, ddc {j.get('ddc') or '?'})\n"
                f"path: {loc.get('path') or '?'}\n\n")
        return head + j.get("text", "")

    @mcp.tool()
    def get_source(book_id: str) -> str:
        """Book metadata by book_id (from search_library results): title,
        authors, DDC, local path, total pages, and how many of its chunks
        are currently in the index."""
        j = _call("GET", f"/v1/source/{book_id}")
        return (f"{j['title']}  ({j.get('authors', '')})\n"
                f"  ddc {j.get('ddc', '')} · {j.get('pages', '?')} pages · "
                f"pdf {'yes' if j.get('has_pdf') else 'no'}\n"
                f"  path: {j.get('path', '?')}\n"
                f"  chunks_indexed: {j.get('chunks_indexed')}")

    @mcp.tool()
    def library_status() -> str:
        """Health of the one machine (Qdrant, embedders, LLM, indexer) and
        index coverage: how many of the library's books/chunks are searchable
        right now."""
        j = _call("GET", "/v1/status")
        cov = j.get("coverage") or {}
        ix = j.get("indexer") or {}
        lines = [
            f"qdrant: {'up' if (j.get('qdrant') or {}).get('ok') else 'DOWN'} "
            f"({(j.get('qdrant') or {}).get('points')} points)",
            f"ollama: {'up' if (j.get('ollama') or {}).get('ok') else 'DOWN'}",
            f"query-embedder: {'up' if (j.get('query_embedder') or {}).get('ok') else 'down (falling back to GPU)'}",
            f"llm: {j.get('llm')}",
            f"indexer: {'running — ' + str(ix.get('progress')) if ix.get('running') else 'stopped'}",
            f"coverage: {cov.get('books_indexed')}/{cov.get('books_total')} books, "
            f"{cov.get('chunks_indexed')} chunks",
        ]
        return "\n".join(lines)

    def _fmt_book_line(i: int, b: dict) -> str:
        idx = (f"{b['chunks_indexed']}/{b['pages']} pages indexed"
               if b["chunks_indexed"] < b["pages"] else "fully indexed")
        return (f"{i}. {b['title']}  ({b['authors'] or '?'}, "
                f"{b.get('pub_date') or '?'})\n"
                f"   {b.get('ddc_path', '?')} · {b['pages']} pages · {idx} · "
                f"book_id {b['book_id']}")

    @mcp.tool()
    def browse_ddc(ddc: str = "", limit: int = 40) -> str:
        """Browse the DDC library by Dewey Decimal section — read straight
        from the on-disk catalog, so it also lists books not yet in the
        search index.

        No args: the section tree of what the library holds (top-level DDC
        class -> tens section -> book counts). With ddc (a code or path
        prefix: "500", "500_Science", "540", "540_Chemistry"): the books in
        that section.

        Args:
            ddc: DDC section code or path prefix ("" = show the section tree)
            limit: max books to list (1-200, default 40)

        Returns title, authors, book_id, section, pages, index status.
        Use a book_id with get_page (read a page) or search_library
        (search inside the book)."""
        query = urllib.parse.urlencode({"limit": min(200, max(1, int(limit)))})
        if ddc:
            query += "&" + urllib.parse.urlencode({"ddc": ddc})
        j = _call("GET", f"/v1/books?{query}")
        if "tree" in j:
            lines = ["library (on disk):"]
            for c in j["tree"]:
                lines.append(f"{c['ddc']} — {c['books']} books")
                for s in c["sections"]:
                    lines.append(f"   {s['ddc']} ({s['books']})")
            return "\n".join(lines)
        if not j["books"]:
            return (f"no books in DDC section {ddc!r} — browse_ddc with no "
                    f"args shows what sections exist.")
        lines = [f"{j['total']} book(s) in {j['ddc']} (showing {len(j['books'])}):", ""]
        lines += [_fmt_book_line(i, b) for i, b in enumerate(j["books"], 1)]
        return "\n".join(lines)

    @mcp.tool()
    def find_books(q: str, limit: int = 20) -> str:
        """Find books in the catalog by title/author keyword ("Stamets",
        "fermentation", "wine") — files on disk, so it works for books that
        are NOT in the vector index yet.

        Args:
            q: keyword to match against book titles and authors
            limit: max results (1-100, default 20)

        Returns title, authors, book_id, DDC section, pages, index status.
        Use a book_id with get_page (read a page) or search_library."""
        query = urllib.parse.urlencode(
            {"q": q, "limit": min(100, max(1, int(limit)))})
        j = _call("GET", f"/v1/books?{query}")
        if not j["books"]:
            return (f"no books in the catalog match {q!r} — try a shorter "
                    f"keyword, or browse_ddc to see what sections exist.")
        lines = [f"{j['total']} book(s) match {q!r} (showing {len(j['books'])}):", ""]
        lines += [_fmt_book_line(i, b) for i, b in enumerate(j["books"], 1)]
        return "\n".join(lines)

    @mcp.tool()
    def get_page(book_id: str, page_num: int) -> str:
        """Read one page of a book in full, from the source file (not the
        search index — works for books not yet indexed).

        Args:
            book_id: from browse_ddc / find_books / search_library results
            page_num: the 1-based page number from those results
                      (e.g. "p.42" -> 42)

        Returns the page text, or a note that the page is scanned (no text
        layer) with the PDF path."""
        j = _call("GET", f"/v1/page/{book_id}/{int(page_num)}")
        head = (f"{j['title']} p.{j['page_num']} of {j['pages_total']}  "
                f"({j.get('authors') or '?'}, {j.get('ddc_path') or j.get('ddc') or '?'})\n"
                f"path: {j['path']}\n\n")
        if not j.get("text"):
            return head + j.get("note", "this page has no text")
        return head + j["text"]

    @mcp.tool()
    def rescan_library() -> str:
        """Re-scan the DDC tree on disk so newly classified/renamed books
        (or deletions) show up in browse_ddc / find_books / get_page
        immediately. The catalog auto-refreshes every minute anyway — use
        this right after a classification run when you can't wait.

        Returns the total book count and how long the scan took (or a note
        that a scan is already in flight)."""
        j = _call("POST", "/v1/rescan")
        if j.get("rescanning"):
            return (f"a catalog rescan is already running — "
                    f"{j.get('books', '?')} books in the current catalog; "
                    f"browse_ddc again in a moment.")
        return (f"catalog rescanned: {j['books']} books ({j['seconds']}s) — "
                f"browse_ddc / find_books / get_page now see the current tree.")

    @mcp.tool()
    def log_hit(query: str = "", note: str = "", layer: str = "agent",
                tool: str = "", ddc: str = "", session: str = "",
                book_ids: list[str] | None = None,
                point_ids: list[str] | None = None,
                n_results: int = 0, result_summary: str = "",
                answer: str = "",
                page_start: int | None = None,
                page_end: int | None = None,
                topic: str = "",
                cross_refs: list[dict] | None = None,
                status: str = "",
                confidence: str = "",
                session_id: str = "") -> str:
        """Record a hit in the MySQL research log (the hit database over
        the library). Two layers, kept separate on purpose:
          layer='user'  — what the USER asked to look at (active work)
          layer='agent' — YOUR OWN research (off-time, autonomous roaming)
        Use layer='agent' (the default) when you explore the library on
        your own, so your wandering never gets lost under the user's
        session; use layer='user' to log what you searched for them and
        what it found. Searches/asks made through the other tools are
        already auto-logged as layer='user' — log_hit is for the things
        that happen OUTSIDE those calls (your own research, web findings
        you want to attach to the library, notes).

        Args:
            query: the search/question as issued ("" if this is only a note)
            note: what you learned, why you searched, a pointer — free text
            layer: 'agent' (default) or 'user'
            tool: what produced the hit (e.g. 'search_library', 'get_page',
                  'web', 'manual')
            ddc: DDC scope if one was used, e.g. "500_Science"
            session: optional id grouping related hits (legacy session col)
            book_ids: book_ids the hit touched (from search/browse results)
            point_ids: Qdrant point ids (chunk refs) the hit touched
            n_results: how many results a search returned (0 if n/a)
            result_summary: one line, e.g. "12 results · top: X p.42"
            answer: the answer produced, if any
            page_start: first page number actually used (research-loop field)
            page_end: last page number used (page_start if a single page)
            topic: comma-separated topic tags, e.g.
                   "affinage,proteolysis,alpine-cheese"
            cross_refs: links to other books/domains this hit points at;
                   each {"book_id": str|null, "page": int|null,
                   "ddc": str|null, "label": str}
            status: 'found' (default) | 'needs-verify' (needs a check in
                   another domain) | 'verified' | 'done'
            confidence: 'medium' (default) | 'low' | 'high'
            session_id: groups related hits from one research walk

        Returns the new hit id."""
        body = {"layer": layer,
                "query": query or None, "note": note or None,
                "tool": tool or None, "ddc": ddc or None,
                "session": session or None,
                "book_ids": book_ids or None,
                "point_ids": point_ids or None,
                "n_results": n_results or None,
                "result_summary": result_summary or None,
                "answer": answer or None,
                "page_start": page_start,
                "page_end": page_end,
                "topic": topic or None,
                "cross_refs": cross_refs or None,
                "status": status or None,
                "confidence": confidence or None,
                "session_id": session_id or None}
        refs = [("book", b) for b in (book_ids or [])]
        refs += [("chunk", str(p)) for p in (point_ids or [])]
        if ddc:
            refs.append(("ddc", ddc))
        if refs:
            body["refs"] = [{"type": t, "id": i} for t, i in refs]
        j = _call("POST", "/v1/hits", body)
        return f"logged hit {j['id']} (layer={layer}, " \
               f"status={status or 'found'})"

    @mcp.tool()
    def hits_recall(layer: str = "", q: str = "", book: str = "",
                    ddc: str = "", session: str = "", status: str = "",
                    topic: str = "", cross_ref_book: str = "",
                    session_id: str = "", limit: int = 20) -> str:
        """Query the research log — what was searched/asked before, and what
        the agent found on its own (off-time research). Newest first.
        Use it to pick up where you left off: before re-searching, check
        whether the question (or something close) was already asked or
        researched.

        Args:
            layer: 'user' (the user's work) or 'agent' (its own research);
                   "" = both
            q: substring to match inside the recorded queries
            book: book_id — only hits that touched that book
            ddc: substring to match inside DDC scopes
            session: exact legacy session id
            status: exact status — 'found' | 'needs-verify' | 'verified'
                   | 'done' (e.g. pull only the verified findings)
            topic: substring of a topic tag, e.g. "affinage"
            cross_ref_book: value inside cross_refs — matches a book_id,
                   a ddc, or a label (e.g. "579" finds hits that link out
                   to Microbiology)
            session_id: exact research-walk session id (groups related hits)
            limit: max hits to return (1-100, default 20)

        Returns each hit (id, ts, layer, tool, query, ddc, n_results,
        result_summary, note, answer truncated to 280 chars, page span,
        topic tags, status, confidence, session_id, cross_refs) plus how
        many hits each layer holds in total."""
        params = {"layer": layer, "q": q, "book": book, "ddc": ddc,
                  "session": session, "status": status, "topic": topic,
                  "cross_ref_book": cross_ref_book, "session_id": session_id,
                  "limit": min(100, max(1, int(limit)))}
        query = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v not in ("", None)})
        j = _call("GET", f"/v1/hits?{query}")
        if not j["hits"]:
            return (f"no hits recorded yet (layers: {j.get('layers', {})}) — "
                    f"the log fills as searches/asks happen and as "
                    f"log_hit is used.")
        lines = [f"{j['total']} hit(s) total · layers {j.get('layers', {})}:"]
        for h in j["hits"]:
            head = (f"#{h['id']} {h['ts'][:19]} [{h['layer']}] "
                    f"{h.get('tool') or '?'}")
            if h.get("query"):
                head += f" · {h['query']}"
            if h.get("n_results") is not None:
                head += f" · {h['n_results']} results"
            if h.get("page_start") is not None:
                pages = (f"pp.{h['page_start']}"
                         if h.get("page_end") in (None, h["page_start"])
                         else f"pp.{h['page_start']}-{h['page_end']}")
                head += f" · {pages}"
            head += f" · status={h.get('status') or 'found'}"
            if h.get("confidence") and h["confidence"] != "medium":
                head += f" · conf={h['confidence']}"
            lines.append(head)
            if h.get("ddc"):
                lines.append(f"   ddc: {h['ddc']}")
            if h.get("topic"):
                lines.append(f"   topic: {h['topic']}")
            if h.get("session_id"):
                lines.append(f"   session: {h['session_id']}")
            if h.get("cross_refs"):
                refs = []
                for r in h["cross_refs"]:
                    bits = [f"{r.get('ddc') or r.get('book_id') or '?'}"]
                    if r.get("page") is not None:
                        bits.append(f"p.{r['page']}")
                    if r.get("label"):
                        bits.append(f"({r['label']})")
                    refs.append(" ".join(bits))
                lines.append(f"   cross_refs: {'; '.join(refs)}")
            if h.get("result_summary"):
                lines.append(f"   {h['result_summary']}")
            if h.get("note"):
                lines.append(f"   note: {h['note']}")
            if h.get("answer"):
                lines.append(f"   answer: {h['answer']}")
        return "\n".join(lines)

    @mcp.tool()
    def hits_pending(limit: int = 10) -> str:
        """The research loop's frontier pull: every hit currently stuck at
        status='needs-verify', OLDEST FIRST (FIFO — work the queue in the
        order it was created), with all fields including cross_refs.

        The loop calls this, works through the items (verify each cross-
        ref / open question), then advances them with hits_mark
        (status='verified' or 'done', plus note_append with the detail),
        and writes new hits via log_hit.

        Args:
            limit: max hits to return (1-100, default 10)

        Returns the pending queue, or "no pending hits — frontier is
        clear" with the per-layer totals."""
        j = _call("GET", f"/v1/hits/pending?limit={min(100, max(1, int(limit)))}")
        if not j["hits"]:
            return "no pending hits — the needs-verify frontier is clear."
        lines = [f"{j['count']} pending hit(s), oldest first (FIFO):"]
        for h in j["hits"]:
            head = (f"#{h['id']} {h['ts'][:19]} [{h['layer']}] "
                    f"{h.get('tool') or '?'}")
            if h.get("query"):
                head += f" · {h['query']}"
            lines.append(head)
            if h.get("ddc"):
                lines.append(f"   ddc: {h['ddc']}")
            if h.get("cross_refs"):
                refs = []
                for r in h["cross_refs"]:
                    bits = [f"{r.get('ddc') or r.get('book_id') or '?'}"]
                    if r.get("page") is not None:
                        bits.append(f"p.{r['page']}")
                    if r.get("label"):
                        bits.append(f"({r['label']})")
                    refs.append(" ".join(bits))
                lines.append(f"   cross_refs: {'; '.join(refs)}")
            if h.get("note"):
                lines.append(f"   note: {h['note']}")
            lines.append(f"   -> when done: hits_mark(hit_id={h['id']}, "
                         f"status='verified'|'done', note_append=...)")
        return "\n".join(lines)

    @mcp.tool()
    def hits_mark(hit_id: int, status: str = "", confidence: str = "",
                  note_append: str = "",
                  cross_refs_append: list[dict] | None = None) -> str:
        """Advance one hit without rewriting it (the loop's bookkeeping
        call). Use after working through a pending item:

            hits_mark(14, status='verified', confidence='high',
                      note_append='verified: 579.6 covers P. freud. p.12')

        Args:
            hit_id: the hit to update (from hits_pending / hits_recall)
            status: overwrite the status — 'found' | 'needs-verify' |
                    'verified' | 'done'
            confidence: overwrite the confidence — 'low' | 'medium' |
                    'high'
            note_append: appended to the existing note (newline-separated) —
                    for "I went back and verified, here's the detail"
            cross_refs_append: merge more links into the existing
                    cross_refs array (deduped by book_id+page); each
                    {"book_id", "page", "ddc", "label"}

        At least one of status / confidence / note_append /
        cross_refs_append must be given. Returns the updated hit."""
        body = {"hit_id": hit_id}
        if status:
            body["status"] = status
        if confidence:
            body["confidence"] = confidence
        if note_append:
            body["note_append"] = note_append
        if cross_refs_append:
            body["cross_refs_append"] = cross_refs_append
        if len(body) == 1:
            raise ValueError("give at least one of status / confidence / "
                             "note_append / cross_refs_append")
        j = _call("POST", "/v1/hits/mark", body)
        h = j["hit"]
        return (f"hit {h['id']} updated: status={h['status']}, "
                f"confidence={h['confidence']}")

    return mcp


if __name__ == "__main__":
    create_server().run()  # stdio
