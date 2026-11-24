# DDC Book Reader — the one machine

A Flask viewer for the DDC book library (`~/DDC/`, 643 parquets, one row per
page) that is also the **front end for the semantic-search indexer**
(`~/DDC/indexer`) **and the RAG answer engine**: one UI for browsing, reading,
full-text search, semantic search, *asked questions with cited answers*,
index status, and starting/stopping the indexer.

It also serves the **stable `/v1` machine API** — the link-in contract for
your RAG, for LLM agents (via the bundled **MCP server**), and for the
**`ddq` CLI** — plus a dedicated CPU query-embed worker that keeps search
fast (~0.2 s) while the indexer is mid-batch on the GPU.

```
~/DDC (parquet + PDF + PNG, ~23 GB, read-only)
   │
   ├─► ~/base/reader  (this app, Flask, port 8071, no GPU needed to read)
   │       │  reads parquets + renders PDF pages lazily
   │       │  /api/*         human UI (browse/read/ask/search/indexer)
   │       │  /v1/*          stable machine API (search/ask/chat/books/page/
   │       │                 chunk/source/status)
   │       │                 ◄── ddq (CLI)  and  mcp_server.py (MCP tools)
   │       │  queries        ──► query-embed Ollama (CPU, port 11435,
   │       │                     fast + cached; falls back to the GPU worker)
   │       │                   └► Qdrant (ddc-books: cosine vectors + BM25
   │       │                      text index, hybrid RRF in one call)
   │       │  /api/chat, /v1/ask ──► cited synthesis from the "onecat"
   │       │                      vLLM service (OpenAI-compatible, Qwen3-27B)
   │       │  /api/index/start|stop ──► index_books.py (background process,
   │       │                            appends to the shared index_run.log)
   │
   └─► ~/DDC/indexer  (index_books.py: parquet → bge-m3 → Qdrant)
            Ollama on the P2000 (port 11434) · Qdrant in Docker (port 6333)
            answer LLM: onecat vLLM in its LXD container (port 8000)
```

## Run

```bash
# After a reboot, bring the whole stack up with one shot:
~/base/reader/start-all.sh      # gpu power -> qdrant -> indexer -> gui -> reader -> owui
                                # (MariaDB + MCP need no start — see below)

# ...or individually:
~/base/gui/run.sh               # pipeline dashboard (OCR/index progress) -> :8070
~/base/reader/run.sh            # start the viewer (or: status / stop / logs)
# -> http://localhost:8071

~/base/reader/run.sh owui       # start the Open WebUI bridge (port 8072)
```

**What starts itself after a reboot, and what doesn't:**

| piece | auto after reboot? | how to bring up |
| --- | --- | --- |
| MariaDB (hit DB) | yes — systemd-enabled | nothing |
| MCP (log_hit / hits_recall / …) | yes — no daemon: OpenCode launches `mcp_server.py` per session (stdio); OWUI's `/mcp` is served by the bridge | nothing |
| Qdrant (`ddc-qdrant`) | **no** (restart-policy `no`) | `docker start ddc-qdrant` |
| Indexer supervisor | **no** | `~/DDC/indexer/run_until_done.sh` (nohup; flock-guarded, no-ops if the scan is done) |
| Pipeline GUI (`:8070`) | **no** | `~/base/gui/run.sh start` |
| Reader (`:8071`) | **no** | `~/base/reader/run.sh start` |
| OWUI bridge (`:8072`) | **no** | `~/base/reader/run.sh owui start` |

`start-all.sh` does the five manual ones in order and prints the resulting
state of the whole stack.

**Pipeline progress (OCR + Qdrant indexing):** both `~/base/ocr.py` and
`~/DDC/indexer/index_books.py` write a machine-readable state file to
`~/.ddc-progress/` as they work (per chunk / per book / per embed batch).
The pipeline GUI's "Pipeline progress" section (`/api/pipeline`, :8070) and
the reader's `/v1/index/status` (+ the agent's status tool) turn those into
live *where it is / how far to go / pace / ETA* — for runs started from the
GUI, the CLI, or the supervisor alike. See `~/DDC/indexer/README.md`,
section "Live progress (dashboard)", for the file contract.

**GPU power limits (V100 150W/300W):** the dashboard's "⚡ Power limits"
bar (and the per-GPU buttons on each GPU card) throttle the V100s with
`nvidia-smi -pl` — 150W keeps them cooler, 300W (factory default) runs them
flat-out. Changing a limit needs root, so one-time passwordless sudo for
`nvidia-smi` is installed in `/etc/sudoers.d/nvidia-smi`
(`matt ALL=(root) NOPASSWD: /usr/bin/nvidia-smi`). The last choice is saved
to `~/base/gui/config.json` (`gpu_power` map) and re-applied on every boot
by `start-all.sh` step 0 (`~/base/gui/set_gpu_power.sh`) — the limit itself
resets to the card's factory value on every power cycle.

**squig LLM load/unload (frees the V100s for OCR):** the LLM that serves
OpenCode + OWUI (vLLM, `http://10.90.38.19:8000`) runs inside the LXC
container `onecat` and holds both V100s' VRAM. The dashboard's "🧠 squig
LLM" card (Services area) shows which profile is loaded, live health,
uptime and V100 VRAM, and can **stop / unload** it (SIGTERM, then SIGKILL —
reports how many GB of VRAM were freed) or **load** a profile:

| profile | script | what it is |
| --- | --- | --- |
| squigFast | `~/scripts/squigFast` | qwen3.8-27B, fast, no thinking, 131k ctx |
| squigFastThink | `~/scripts/squigFastThink` | qwen3.8-27B, thinking, 150k ctx |
| squigCode | `~/scripts/squigCode` | orca, coding, 262k ctx |
| squigChat | `~/scripts/squigChat` | llama.cpp, Qwen3.8-Flash-Next (on the host) |

All profiles bind port 8000, so only one can be loaded at a time. Load =
the dashboard spawns `bash ~/scripts/<name>` detached (survives a GUI
restart), logging to `~/squig-llm.log` (a "log tail" box on the card shows
the last lines; model weights take a minute or two to load). Unloading it
also stops the LLM that chat sessions use — reload it afterwards. The last
loaded profile is remembered in `config.json` (`squig.last`); auto-loading
at boot is a commented-out line at the bottom of `start-all.sh`.

- Python: `/home/matt/miniconda3/envs/marker_env/bin/python`
  (override with `PIPELINE_PYTHON` for run.sh).
- **No GPU needed for the viewer itself.** It only needs the GPU if you use
  semantic search or the indexer (both go through Ollama).

The viewer is read-only with respect to `~/DDC`: it never writes to the
library. Rendered page images are cached under `~/base/reader/cache/`
(LRU, default ≤1500 files / 600 MB).

## Views

- **Browse** — folder tree of the DDC classification; each folder lists its
  subfolders and books.
- **Read** — text-first: the OCR text is shown immediately; the rendered PDF
  page is fetched only when you switch the view to `pdf` / `split` (opt-in,
  rendered with PyMuPDF and disk-cached). Pages without OCR text (scanned /
  image-only) show "— no OCR text on this page —".
- **Ask** (`ask the library` in the header) — RAG answers. You ask a natural
  question; the app does **hybrid retrieval** (a dense bge-m3 vector search
  *and* a BM25 full-text search, RRF-fused by Qdrant itself) and hands the top
  ~12 pages to the LLM, which answers **only from those pages** and cites them
  as `[1]`, `[2]`, …. Every citation is clickable and the **Sources** panel
  deep-links each cited page straight into the reader (split view), so every
  claim is one click from the original. You can scope a question to one top
  DDC section (e.g. `500_Science`), and short follow-ups work via an in-page
  history.
- **Search** (header box, `titles` / `semantic` toggle):
  - `titles` — substring match over all 632 book titles/authors
    (in-memory index built in the background at startup).
  - `semantic` — embeds your query with the **same bge-m3 model** that built
    the Qdrant index, runs a cosine search over `ddc-books`, and deep-links
    each hit straight to that book **and page** in the reader.
- **Index status strip** (browse view) — Qdrant health + point count,
  Ollama + model readiness, live indexer progress, and **Index new books /
  Stop indexer** buttons that run `index_books.py` exactly like the CLI.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | ok + root path |
| `GET /api/browse?path=` | one DDC folder (dirs + book summaries) |
| `GET /api/book?path=` | one book (page list + meta + has_pdf) |
| `GET /api/page?path=&page=` | OCR text of one page |
| `GET /api/page-image?path=&page=&dpi=` | rendered PDF page (PNG, disk-cached) |
| `GET /api/search?path=&q=` | in-book text search (page + snippet) |
| `GET /api/search-books?q=` | global title/author search |
| `GET /api/semantic?q=&k=` | dense semantic search over Qdrant (default k=8) |
| `POST /api/chat` | **RAG answer**: hybrid retrieval (dense + BM25, RRF) + cited LLM synthesis. Body `{q, ddc?, history?}` → `{answer, sources[], model, retrieval_ms, gen_ms}` |
| `GET /api/index/status` | Qdrant / Ollama / indexer health + coverage |
| `POST /api/index/start` | start `index_books.py` in the background (409 if running) |
| `POST /api/index/stop` | SIGTERM the indexer (404 if not running) |
| `GET /book/<path>?p=N`, `GET /browse/<path>`, `GET /ask` | SPA deep links (refresh-safe) |

## `/v1` — the stable machine API

`/api/*` is the human UI's contract; **`/v1` is the versioned contract that
tools link against** (your RAG, LLM agents, the CLI). Same retrieval + answer
engine, JSON in/out, every result self-describing (each hit carries a
`locator` with the local file path + page and a `coverage` block saying how
complete the index is, so a client always knows what fraction of the library
it searched).

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/status` | one-shot health (Qdrant, Ollama, query-embedder, LLM, indexer) + index coverage |
| `POST /v1/search` | **canonical retrieval**. Body `{q, k=10, mode: hybrid\|dense\|keyword, filters: {ddc_coarse, ddc_tens, ddc, pub_date, book_id, min_page, max_page}, fields: ["text"]?}` → `{results: [{id (point id), score, doc_title, authors, pub_date, ddc, page_num, locator: {path, page}, snippet, text?}], coverage, ms}` |
| `POST /v1/ask` | RAG answer (same engine as `/api/chat`): `{q, ddc?, k?, history?}` → `{answer, sources[], model, retrieval_ms, gen_ms, coverage}`. One search, one answer — the fast path |
| `POST /v1/chat` | **agent answer**: the LLM gets the library as tools (`search_library`, `get_chunk`, `get_source`, `library_status`, `browse_ddc`, `find_books`, `get_page`) and reasons *between* searches — reformulates, scopes to DDC, pulls full pages — then synthesizes. Same body as `/v1/ask` (minus `k`) → `{answer, sources[], steps[], iterations, model, gen_ms, coverage}`. The dashboard's "research" mode (`POST /api/agent`, SSE) and the OWUI `ddc-library` model run this |
| `GET /v1/books` | **book catalog** — files on disk, not the vector index (works for unindexed books). No params → DDC section tree with book counts; `?ddc=500\|500_Science\|540\|641` (code or path prefix, any depth) → books in that section; `?q=keyword` → title/author search (combinable); `?limit&offset` paging. Every book carries `chunks_indexed`, `ddc_path`, `book_id` |
| `GET /v1/page/{book_id}/{page_num}` | **page retrieval**: full text of one page, read from the source parquet (not the index). `page_num` is 1-based, as in search results; scanned pages return `text: ""` + a note with the PDF path |
| `GET /v1/chunk/{point_id}` | one chunk by its **deterministic** Qdrant point id (stable ⇒ cacheable / referenceable) |
| `GET /v1/source/{book_id}` | book metadata + local path + how many of its chunks are indexed |
| `GET /v1/image?path=&page=&dpi=` | rendered PDF page (same as `/api/page-image`) |
| `GET /v1/index/status` · `POST /v1/index/start` · `POST /v1/index/stop` | indexer control (mirrors `/api/index/*`) |
| `GET /v1/hits/status` | hit-log (MySQL) health + per-layer counts |
| `POST /v1/hits` | record one hit in the research log: `{layer: "user"\|"agent", query?, note?, answer?, result_summary?, tool?, ddc?, session?, session_id?, book_ids?, point_ids?, n_results?, refs?: [{type, id}], page_start?, page_end?, topic?, cross_refs?: [{book_id,page,ddc,label}], status? (found\|needs-verify\|verified\|done), confidence? (low\|medium\|high)}` → `{id}` |
| `GET /v1/hits?layer=&q=&book=&ddc=&session=&session_id=&status=&topic=&cross_ref_book=&limit=&offset=` | query the research log, newest first → `{hits, total, layers}` (each hit includes the v2 fields, `cross_refs` parsed as a list) |
| `GET /v1/hits/pending?limit=` | the loop's frontier: `status='needs-verify'`, oldest first (FIFO) → `{hits, count}` |
| `POST /v1/hits/mark` | advance a hit without rewriting: `{hit_id, status?, confidence?, note_append?, cross_refs_append?: [{book_id,page,ddc,label}]}` → `{hit}` |

Modes: `hybrid` (default) = dense bge-m3 + BM25 full-text, RRF-fused by
Qdrant; `dense` = embeddings only; `keyword` = BM25 only (instant, needs no
embedder at all — handy when the GPU is busy or down).

### `ddq` — the CLI

```bash
~/base/reader/ddq status
~/base/reader/ddq search "mildew in grape vines"            # hybrid
~/base/reader/ddq search "Table 8" --mode keyword           # instant
~/base/reader/ddq search "wine flavour" --ddc 300_Social_Sciences -k 5 --text
~/base/reader/ddq ask "How does fermentation affect wine flavour?"     # one-shot
~/base/reader/ddq chat "What do you have about cheese making?"         # agent (reasons between searches)
~/base/reader/ddq chunk 2464535952654574015
~/base/reader/ddq source 197b485b0d7582f0
# every command: --json for machine-readable output
```

### MCP server — native tools for LLM agents

`mcp_server.py` (stdio) exposes the API as thirteen tools: `search_library`,
`ask_library`, `get_chunk`, `get_source`, `library_status`, `browse_ddc`
(the DDC section tree, or the books in a section), `find_books` (catalog
search by title/author), `get_page` (full text of one page of a book),
`rescan_library` (force an immediate catalog re-scan),
`log_hit` (record a hit in the MySQL research log — see below),
`hits_recall` (query that log), `hits_pending` (the loop's `needs-verify`
frontier, FIFO) and `hits_mark` (advance a hit: status/confidence/note/refs).
Installed
globally in OpenCode via:

```bash
opencode mcp add ddc --global -- /home/matt/miniconda3/envs/marker_env/bin/python /home/matt/base/reader/mcp_server.py
opencode mcp list        # ->  ✓ ddc  connected
```

(which writes to `~/.config/opencode/opencode.json`):

```json
"mcp": {
  "servers": {
    "ddc": {
      "type": "local",
      "command": ["/home/matt/miniconda3/envs/marker_env/bin/python",
                  "/home/matt/base/reader/mcp_server.py"]
    }
  }
}
```

`READER_URL` overrides the base URL on both `ddq` and the MCP server
(default `http://localhost:8071`).

## Hit database (MySQL) — the two-layer research log

A MySQL (MariaDB) database on this machine — **`squig_research`** — keeps a
durable log of what gets searched and found in the library. It exists so
research survives: the library will keep growing, and during *off time* (when
you're not at the machine) the system should still be able to search the
library and record what it found, without that getting lost or mixed up with
your own session.

**Two layers, kept separate on purpose** (`hits.layer`):

| layer | meaning |
| --- | --- |
| `user` | things **you** asked to look at (active work). Auto-recorded by every search/ask/agent/chunk/page call through the reader (`HITS_AUTOLOG=1`, the default) |
| `agent` | the **system's own** research (off-time, autonomous roaming). Recorded explicitly via `POST /v1/hits` / the `log_hit` MCP tool |

The split is the point: while you're working on something, that work stays in
`layer='user'` and never gets buried under the agent's own exploration — and
you can audit what the agent did on its own without sifting through your
session.

**Schema** (created + upgraded by the reader at startup,
`hits_db.ensure_schema`; the v2 columns/indexes are applied idempotently on
every boot, so an old DB upgrades itself):
`hits(id, ts, layer, session, query, tool, mode, ddc, book_ids, point_ids,
n_results, result_summary, note, answer,` **`page_start, page_end, topic,
cross_refs, status, confidence, session_id`** `)` plus
`hit_refs(hit_id, ref_type [book|chunk|ddc], ref_id)` — the cross-ref table
is what makes "which hits touched both book A and book B" a simple join, i.e.
the cross-domain links the research is about.

### v2 — the autonomous research loop

The v2 columns turn the log into a work queue the agent can roam on off
time. Each hit now carries:

| column | meaning |
| --- | --- |
| `page_start` / `page_end` | the page span actually used (`pp.45-47`) |
| `topic` | comma-separated tags (`"affinage,proteolysis,alpine-cheese"`) |
| `cross_refs` | JSON array of `{"book_id","page","ddc","label"}` — links out to other books/domains |
| `status` | `found` → `needs-verify` → `verified` → `done` (CHECK-enforced) |
| `confidence` | `low` / `medium` / `high` (CHECK-enforced) |
| `session_id` | groups related hits from one research walk |

**The loop:** `hits_pending` pulls the frontier — every hit stuck at
`status='needs-verify'`, **oldest first (FIFO)**. The agent works each item
(verifies its cross-refs / open questions), then `hits_mark` advances it
(`status='verified'`/`'done'`, `confidence`, `note_append` with the detail,
`cross_refs_append` to add more links, deduped by book_id+page) — without
rewriting the hit. New findings are written with `log_hit`.

```
log_hit(status='needs-verify', cross_refs=[...])   # plant the frontier
   ↓
hits_pending()        # what's next? (FIFO)
   ↓  work the item, then
hits_mark(id, status='verified', confidence='high', note_append=...)
   ↓
log_hit(...)          # new leads it found
```

**Wiring:**

- Auto-log (user layer): a background thread in the reader drains a bounded
  queue, so MySQL can never block or break a search/answer; a dead DB costs
  one rate-limited log line per minute, not one per request.
- `log_hit` (MCP tool): record a hit explicitly — the agent's off-time
  research (default `layer='agent'`), or anything outside the normal
  search/ask calls (web findings attached to the library, notes). Takes all
  v2 fields.
- `hits_recall` (MCP tool): query the log by layer / query / book / ddc /
  **status / topic / cross_ref_book / session_id** — "what have we already
  found on X?", "what's still unverified?".
- `hits_pending` (MCP tool): the frontier pull (`needs-verify`, FIFO).
- `hits_mark` (MCP tool): advance a hit (status / confidence / note_append /
  cross_refs_append).

> MariaDB note: no partial indexes (Postgres) and no FTS5 (SQLite) on this
> server, so `status`/`topic` get plain B-tree indexes and the filters use
> `=` / `LIKE`. The table is a research log (small), and the one hot query —
> `status = 'needs-verify'` — is index-served.

**MySQL setup** (done): MariaDB on `127.0.0.1:3306`; the `matt` MySQL user
holds `ALL` on `squig_research` (but not global CREATE, which is fine — the
DB pre-exists and the reader only creates tables). Override the connection
with `HITS_DB_HOST` / `HITS_DB_PORT` / `HITS_DB_USER` / `HITS_DB_PASSWORD` /
`HITS_DB_NAME` / `HITS_DB_TIMEOUT` (defaults: `127.0.0.1` / `3306` / `matt` /
empty / `squig_research` / `5`). `HITS_AUTOLOG=0` stops the auto-recording
while leaving the API + MCP tools intact.

```bash
mysql -u matt squig_research -e "SELECT layer, COUNT(*) FROM hits GROUP BY layer"
mysql -u matt squig_research -e "SELECT id,status FROM hits WHERE status='needs-verify'"
curl -s localhost:8071/v1/hits/status
curl -s localhost:8071/v1/hits/pending
curl -s "localhost:8071/v1/hits?status=needs-verify"
~/base/reader/ddq hits --layer agent
```

The original v1 rows (cheese-aging hits #11–#15) were migrated to v2 by the
one-off `migrate_hits_v2.py` (parses `pp.X-Y` from the notes into
`page_start`/`page_end`, sets `status`/`topic`, and plants the #14
cross-domain link as the first `needs-verify` frontier item).

## Open WebUI bridge (port 8072)

Open WebUI 0.13.x removed the old addon-container framework; its native
extension points are **OpenAI-compatible API connections** and **MCP
tool-server connections**. `ddc_owui.py` provides both in one process on
port 8072:

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | OpenAI model list — registers **`ddc-library`** in the OWUI model picker |
| `POST /v1/chat/completions` | OpenAI chat (streaming + not) → the reader's `/v1/chat` — the **agent** (reasons between searches, cited answer). Pick `ddc-library` from the model dropdown and chat — no function-calling needed. Set `DDC_OWUI_AGENT=0` to fall back to the one-shot `/v1/ask` |
| `POST /mcp` | streamable-HTTP MCP with the same thirteen tools as `mcp_server.py` — toggle them in any squigFast chat |
| `POST /` | the same MCP endpoint at the URL root (OWUI gotcha below) |
| `GET /info` / `GET /healthz` | service info / health |

```bash
~/base/reader/run.sh owui            # start the bridge (status / stop / logs)
```

**Register with Open WebUI** (one-shot, idempotent — it reads the current
config and only adds what's missing):

```bash
~/base/reader/run.sh owui start
python3 ~/squig/scripts/owui_register.py \
    --key <OWUI admin API key>        # Settings -> Profile -> API Keys -> New
```

This adds: (1) an OpenAI (API key) connection at `http://172.17.0.1:8072/v1`
so `ddc-library` appears in the model picker, and (2) an MCP tool-server
connection at `http://172.17.0.1:8072/mcp` so the thirteen tools appear in the
tools menu. (From inside the OWUI container the host is the Docker bridge
gateway `172.17.0.1`; from the host itself use `127.0.0.1:8072`.) Manual
equivalent: OWUI admin → Settings → Connections → add an *OpenAI (API key)*
connection (base URL `http://172.17.0.1:8072/v1`, any non-empty key, model
ids `ddc-library`) and an *MCP* connection (`http://172.17.0.1:8072/mcp`).

Notes:

- **OWUI MCP connections use only the `url` field.** The backend
  (`connect_mcp_server` in `open_webui/utils/tools.py`) connects to
  `connection["url"]` verbatim and ignores `path`. So the stored MCP URL
  must be the *full* endpoint — `http://172.17.0.1:8072/mcp`. If it is
  saved without the path (e.g. `http://172.17.0.1:8072`), OWUI posts to
  the URL root; the bridge mounts the MCP app at `/` as well, so that
  still works. Symptom of the old mismatch: tools show in the menu but
  `GET /api/v1/tools/id/server:mcp:ddc-library/specs` → 502
  "Unable to load tools". `owui_register.py` now writes the full URL.
- **Container networking for integrations.** Any URL an OWUI integration
  uses is resolved *from inside the OWUI container*, not the host:
  `localhost`/`127.0.0.1` = the OWUI container itself (its backend listens
  on 8080 in there!), the host = `172.17.0.1`, sibling containers = their
  names. Example: with SearXNG in a sibling `searxng` container, the web
  search setting must be `http://searxng:8080/search` (OWUI appends
  `?q=…&format=json` itself). `http://127.0.0.1:8080` silently hits OWUI's
  own backend and the search fails with a JSON/content-type error.
  Fix: `POST /api/v1/retrieval/config/update` with
  `{"web": {"SEARXNG_QUERY_URL": "http://searxng:8080/search"}}`
  (the field is nested under `web`).
- `ddc-library` is a *model*, not a RAG-augmented one — OWUI's own RAG
  doesn't apply (no knowledge base attached), so every question goes
  straight to the answer engine, which does its own retrieval + citation.
- The bridge accepts any API key (LAN-local); it is a thin translator, all
  intelligence stays in the reader's `/v1`.

## Environment

| Var | Default | Meaning |
| --- | --- | --- |
| `DDC_ROOT` | `~/DDC` | library root |
| `READER_PORT` | `8071` | HTTP port |
| `CACHE_MAX_FILES` / `CACHE_MAX_BYTES` | `1500` / `600 MB` | page-image cache limits |
| `BOOK_LRU` | `12` | full parquets held in memory |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant |
| `EMBED_BASE_URL` | `http://localhost:11434` | Ollama (GPU index worker) |
| `EMBED_QUERY_BASE_URL` | `http://127.0.0.1:11435` | dedicated CPU query-embed worker (kept by `query-ollama.sh`); search falls back to `EMBED_BASE_URL` when it's down |
| `EMBED_MODEL` | `bge-m3` | must match the model that built the index |
| `INDEXER_DIR` | `~/DDC/indexer` | where `index_books.py` + `index_run.log` live |
| `LLM_BASE_URL` | `http://10.90.38.19:8000` | answer LLM (OpenAI-compatible), onecat vLLM |
| `LLM_MODEL` | `squigFast` | model id served by the LLM |
| `LLM_TIMEOUT` | `240` | LLM request timeout (s) |
| `HITS_AUTOLOG` | `1` | auto-record searches/asks/pages to the hit DB (`0` = off) |
| `HITS_DB_HOST` / `HITS_DB_PORT` | `127.0.0.1` / `3306` | hit database (MySQL/MariaDB) |
| `HITS_DB_USER` / `HITS_DB_PASSWORD` | `matt` / empty | hit DB credentials |
| `HITS_DB_NAME` / `HITS_DB_TIMEOUT` | `squig_research` / `5` | hit DB name / connect timeout (s) |

`book_id` is imported from `index_books.py` (single source of truth:
`sha256("final_title|authors|pub_date")[:16]`) so semantic hits and RAG
sources always map back to local books.

## Dependencies (the other services)

- **Qdrant** (collection `ddc-books`, 1024-dim cosine, plus a BM25 text index
  on the `text` payload field):
  `cd ~/DDC/indexer && docker compose up -d`
- **Ollama** (bge-m3, P2000 — the *index* worker):
  `CUDA_VISIBLE_DEVICES=2 OLLAMA_NUM_PARALLEL=1 ollama serve` (port 11434)
- **Query-embed Ollama** (CPU, port 11435 — the *interactive* worker):
  `~/base/reader/query-ollama.sh start|stop|status|restart`.
  The GPU worker embeds the indexer's big batches one at a time, so a
  query queued behind it takes ~14 s mid-run. This CPU worker serves
  interactive queries in ~0.2 s (same bge-m3 model; CPU and GPU embeddings
  are bit-identical, so one vector space). It also caches query embeddings
  in the app (bge-m3 is deterministic), so repeated/follow-up questions
  skip embedding entirely.
- **Answer LLM** (onecat vLLM, Qwen3-27B in its LXD container):
  OpenAI-compatible `POST /v1/chat/completions` at `http://10.90.38.19:8000`

Reading + title search need none of these. `/v1/search` needs Qdrant;
`hybrid`/`dense` modes additionally need an embedder (the CPU query worker,
else the GPU worker) while **`keyword` mode needs only Qdrant** — instant,
no embedder. The indexer needs Qdrant + the GPU Ollama; **ask** needs all
of the above. If one is down the UI shows it in the status strip and the
affected endpoint returns a 503. Full indexer docs:
`~/DDC/indexer/README.md`.
