# The DDC Book System — what it does

*Report, 2026-11-25. Companion docs: `PIPELINE.md` (pipeline detail),
`reader/README.md` (reader + /v1 + MCP + hit DB, very detailed),
`~/DDC/indexer/README.md` (indexer + live-progress contract).*

## 1. What it is

One machine that turns an eclectic physical/digital book collection (a ~10.9 TB
Veracrypt volume at `/media/veracrypt2/Books` — 222k HTML, 7k EPUB, 5.2k MOBI,
1.4k LIT, 961 DOC, 546 DJVU, 539 CHM, 489 RTF, 563 comics, 50 MHT, 34 AZW3, …)
into:

1. a **DDC-classified digital library** (`~/DDC/`, Dewey Decimal tree) where
   every book is a 16-column Parquet file (one row per page) sitting next to
   its **original source file** and its page images — self-contained,
   grep-able, re-embeddable;
2. a **web reader** (browse the DDC tree, read text or rendered pages in a
   split view, search inside books);
3. a **search index** — every content page embedded (bge-m3, 1024-d) into
   Qdrant with a BM25 full-text index on top, so one call does hybrid
   (vector + keyword) retrieval;
4. a **RAG answer engine** — ask the library a question; it retrieves pages,
   answers only from them, and cites them `[1] [2] …` with deep-links back
   into the reader;
5. **agent interfaces** — 13 MCP tools for LLM agents (OpenCode, Open WebUI),
   a stable versioned `/v1` machine API, and a `ddq` CLI;
6. a **five-stage pipeline** (Ingest → Enrich → Classify → Validate → Index)
   with a **dashboard** to run, monitor (live progress + ETA), and control it,
   plus service controls for Qdrant, Ollama, the LLM, and GPU power.

Ingest handles **every common book format** (section 3) — format is
auto-detected per file, so a folder of mixed formats just works.

## 2. Architecture

```
/media/veracrypt2/Books  (source collection, read-only)
        │  any supported format, mixed per folder
        ▼
[1] ocr.py ────────────────┐  format autodetect: magic → extension → text sniff
    Ingest → Parquet       │  · PDF/DJVU/CBZ/CBR → marker OCR (GPU)  ┐
    (~/converted/<cat>)    │  · EPUB/TXT/RTF/DOCX/HTML/MHT/FB2/CHM → CPU extractors
                           │  · Kindle/Office/DJVU/comics converted first (calibre,
                           │    LibreOffice, djvups+ps2pdf, zip/7z/unrar+PIL)
        ▼                  │  one parquet per book (16 cols, row = page) + original
[2] enrich_ddc_collection.py  + images copied beside it
    one LLM call per book (squigFast) → final_title/authors/publisher/
    pub_date/ddc/summary (+ is_index TOC detection, local heuristic)
        ▼
[3] rename_ddc_collection.py   names: "DDC - title - publisher - author - date"
    Classify → copies source + parquet + images into the DDC tree,
    rewrites image_path, verifies every file
        ▼
[4] validate_ddc_tree.py  read-only audit (schema, pages, source sibling,
                          images resolve, numeric DDC, duplicates)
        ▼
[5] index_books.py  content pages → bge-m3 (Ollama) → Qdrant "ddc-books"
    (deterministic point ids; idempotent; self-healing re-embeds)

~/DDC  ◄──────────────────────────────────────────────────────────────┐
   │                                                                  │
   ├─► reader (Flask :8071)  — browse/read/ask/search UI + /v1 API    │
   │       queries → CPU query-embed (Ollama :11435) or GPU worker    │
   │                        (Ollama :11434) → Qdrant :6333 (RRF)      │
   │       ask/chat → squigFast vLLM (http://10.90.38.19:8000)        │
   │       hits  → MariaDB :3306 "squig_research" (research log)      │
   │       ◄── ddq CLI, mcp_server.py (13 tools, stdio)               │
   │                                                                  │
   ├─► OWUI bridge (:8072) — "ddc-library" model + MCP /mcp for       │
   │      Open WebUI                                                │
   │                                                                  │
   └─► pipeline GUI/dashboard (:8070) — run & monitor all 5 stages,  │
          GPU cards + power limits, service controls (Qdrant/Ollama/ │
          squig LLM load/unload), live progress + ETA                │
                                                                      │
squigFast vLLM (qwen3.8-27B) in LXC "onecat" — the LLM that serves   │
OpenCode, the dashboard's chat, enrichment, and the answer engine —  ┘
occupies BOTH V100s; the Quadro P2000 is the OCR/index card
```

## 3. Supported book formats

Stage 1 (`ocr.py`) ingests **any mix of these in one folder**. Detection
order per file: **magic bytes → extension → text sniff**, so mislabelled
files still get the right extractor (the source volume is full of them —
e.g. its `.azw3` files are actually MOBI with a 64-byte custom prefix;
both cases are handled).

| Format | Extensions | Path |
|---|---|---|
| PDF | `.pdf` | **marker OCR on GPU** — one row/page, text + page images |
| EPUB | `.epub` (`.opub`) | CPU extractor, images preserved |
| Plain text | `.txt` `.text` `.textbook` | CPU |
| RTF | `.rtf` | `striprtf` → text |
| DOCX | `.docx` | `mammoth` → HTML (data-URI images) → text + images |
| HTML | `.html` `.htm` | shared HTML block parser (images preserved) |
| MHTML | `.mht` `.mhtml` | MHT = MIME email (stdlib `email`) → HTML blocks |
| FictionBook | `.fb2` | lxml XML → text |
| CHM (Windows help) | `.chm` | `pychm`/libchm → per-topic HTML in TOC order (falls back to natural-sorted archive enumeration when no HHC TOC) |
| Kindle | `.mobi` `.azw` `.azw3` `.kfx` `.lit` | **calibre `ebook-convert` → EPUB** (then EPUB path; images preserved) |
| Office legacy | `.doc` | **LibreOffice headless → DOCX** (then DOCX path) |
| OpenDocument | `.odt` `.ods` `.odp` | LibreOffice headless → DOCX |
| DJVU | `.djvu` | **the cheat code**: `djvups` (PostScript, page-batched) + `ps2pdf` (ghostscript) → PDF → marker OCR |
| Comics | `.cbz` `.cbr` | zip / `7z` / `unrar` extract + PIL → PDF (natural panel order) → marker OCR |

Notes:

- **Converted files are intermediates.** The parquet's `source_file` and the
  tree sibling are always the **original** (`.djvu`, `.mobi`, `.doc`, …),
  never the temp. Conversion workdirs live in `<out>/_conv/<rel>/<stem>/`,
  are kept on failure (resume reuses them) and deleted on success.
- **Scanned books (PDF/DJVU/CBZ/CBR) get real text + real pages.** The
  per-page OCR text is captured from marker's *post-processed* pipeline
  (reading-order, tables, de-hyphenated), not just the PDF's embedded text
  layer — which is empty for a scan, so before this fix DJVU/comics ingested
  with blank `text`. Because the original `.djvu`/`.cbz`/`.cbr` is not a
  renderable PDF sibling, the full-page raster is also saved (one JPEG/page
  in `images/`) so the reader's split view shows the page.
- **Text formats use nominal pages**: re-paginated into `--words-per-page`
  (default 300) pages that break only *between* paragraphs. The reader's
  split view works for these via stored page images; there is no PDF
  sibling for them.
- **Install** (once, done): `bash ~/base/install_formats.sh` → pip
  `striprtf` `mammoth` `pychm` into `marker_env` + apt `djvulibre-bin`
  `calibre` `libchm-dev`. Tools already on the box: LibreOffice, ghostscript
  (`ps2pdf`), `7z`, `unrar`, pypdf, PIL.
- PDF/DJVU/comics need the GPU (marker); everything else is CPU and runs on
  any core — multi-GPU fan-out via `ocr.py --shard I/N` for PDF-like files.

## 4. The five-stage pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 · Ingest | `~/base/ocr.py` | Auto-detected per-file extraction/conversion → Parquet (above). Copies the original source next to each parquet. |
| 2 · Enrich | `~/base/enrich_ddc_collection.py` | One LLM call per book (squigFast) adds `final_title / authors / publisher / pub_date / ddc / summary` in place; marks TOC/index pages `is_index` by local heuristic. Resumable (skips books with `ddc` set). Must run before stage 3. |
| 3 · Classify | `~/base/rename_ddc_collection.py` | Names everything `DDC - title - publisher - author - date`, collision check (`--dry-run` first, `--execute` to copy), copies original + parquet + images into the DDC tree, rewrites `image_path`, verifies after copy. |
| 4 · Validate | `~/base/validate_ddc_tree.py` | Read-only audit: 16-column schema order, page counts, source sibling present, every image reference resolves, numeric DDC, duplicate book identities. Exit 0 = clean. |
| 5 · Index | `~/DDC/indexer/index_books.py` | Embeds content pages (`is_index == False`) with bge-m3 (Ollama) into Qdrant `ddc-books`. Deterministic point ids ⇒ re-runs replace, not duplicate; skips books already fully indexed; repairs partial/stale books. |

Typical CLI flow for a new category (any mix of supported formats):

```bash
PY=/home/matt/miniconda3/envs/marker_env/bin/python
$PY ~/base/ocr.py --input ~/Books --output ~/converted/Books --device cuda:2
$PY ~/base/enrich_ddc_collection.py ~/converted/Books
$PY ~/base/rename_ddc_collection.py --source ~/converted/Books --dest ~/DDC --dry-run
$PY ~/base/rename_ddc_collection.py --source ~/converted/Books --dest ~/DDC --execute
$PY ~/base/validate_ddc_tree.py ~/DDC
$PY ~/DDC/indexer/index_books.py --root ~/DDC
```

…or click the cards on the dashboard (:8070). Conventions: DDC tree layout
`~/DDC/<NNN>_<century>/<NNN>_<section>/…` with ancestor-chain validation and
`<NNN>_Unclassified` fallback; `book_id = sha256("final_title|authors|pub_date")[:16]`
(single source of truth in `index_books.py`); enrichment LLM replies are
parsed from `content` (not `reasoning`).

## 5. Services & links

| Service | Link | What it is |
|---|---|---|
| **Pipeline dashboard** | <http://localhost:8070> (LAN: `192.168.0.53:8070`) | Run & monitor all stages (section 7) |
| **Reader** | <http://localhost:8071> | Browse / read / ask / search UI |
| **Machine API** | <http://localhost:8071/v1> | Stable versioned endpoints: `/v1/search`, `/v1/ask`, `/v1/chat` (agent), `/v1/books` (catalog), `/v1/page/{book_id}/{n}`, `/v1/chunk/{id}`, `/v1/source/{id}`, `/v1/status`, `/v1/index/*`, `/v1/hits*`, `POST /v1/rescan` |
| **OWUI bridge** | <http://localhost:8072> | `ddc-library` in the Open WebUI model picker (`/v1/models`, `/v1/chat/completions`) + the 13 MCP tools over streamable HTTP (`POST /mcp` and `POST /`); from inside the OWUI container use `http://172.17.0.1:8072` |
| **Open WebUI** | <http://localhost:3000> | Docker container; pick the `ddc-library` model or toggle the 13 `ddc` tools in any chat (registration: `scripts/owui_register.py`) |
| Qdrant | <http://localhost:6333> | `ddc-books`: 1024-d cosine + BM25 text index |
| Ollama (GPU index worker) | <http://localhost:11434> | bge-m3 embeddings for the indexer, on the P2000 |
| Ollama (CPU query worker) | <http://localhost:11435> | keeps interactive search ~0.2 s while the indexer runs (falls back to the GPU worker; `keyword` mode needs neither) |
| MariaDB | `127.0.0.1:3306` | `squig_research` — the two-layer research/hit log |
| Answer LLM (vLLM) | <http://10.90.38.19:8000/v1> | `squigFast` (qwen3.8-27B, 131k ctx) in LXC `onecat` — also serves OpenCode; profiles: squigFast / squigFastThink / squigCode / squigChat (load/unload from the dashboard) |
| MCP (OpenCode) | stdio, per session | `opencode mcp add ddc --global -- ~/miniconda3/envs/marker_env/bin/python ~/base/reader/mcp_server.py` |

Startup after reboot: `~/base/reader/start-all.sh` (GPU power → Qdrant →
indexer supervisor → GUI → reader → OWUI bridge). MariaDB is systemd-enabled;
Qdrant/GUI/reader/bridge need the script (or the `run.sh` helpers).

## 6. The reader (:8071)

- **Browse** — the DDC folder tree, folders and books.
- **Read** — text-first; the rendered PDF page is fetched only when you
  switch to `pdf` / `split` (PyMuPDF, LRU disk cache ≤ 1500 files / 600 MB).
  Works for **every ingested format**: PDFs render from the source; converted
  and text books show their stored page images.
- **Ask** ("ask the library") — hybrid retrieval (bge-m3 dense + BM25,
  RRF-fused by Qdrant) → top ~12 pages → LLM answers *only from them* with
  `[1] [2]` citations; every citation deep-links into the reader split view.
  Scope to a DDC section; in-page follow-up history.
- **Search** — `titles` (substring over all book titles/authors) or
  `semantic` (dense search deep-linking to book *and* page).
- **Index strip** — Qdrant/Ollama/indexer health, live progress,
  *Index new books / Stop*.

**MCP tools (13)** — `search_library` (hybrid/dense/keyword + DDC/date/page
filters), `ask_library`, `get_chunk`, `get_source`, `library_status`,
`browse_ddc` (tree or books-in-section), `find_books` (catalog search),
`get_page` (full page text), `rescan_library` (force catalog re-scan; the
catalog also auto-rescans every 60 s), plus the research-log quartet:
`log_hit`, `hits_recall`, `hits_pending` (the `needs-verify` frontier,
FIFO), `hits_mark` (advance a hit: status/confidence/notes/cross-refs).
The same 13 are served to Open WebUI via the bridge's `/mcp`.

**`ddq` CLI** — `~/base/reader/ddq status | search | ask | chat | chunk |
source | hits` (every command `--json`; `READER_URL` overrides the base).

**Research log (MariaDB `squig_research`)** — two layers kept separate:
`user` (auto-logged from every search/ask/page through the reader) and
`agent` (the system's own off-time research via `log_hit`). v2 columns turn it
into a work queue: `status` (found → needs-verify → verified → done),
`confidence`, `topic` tags, `page_start/page_end`, `cross_refs` (JSON links to
other books/domains, deduped, joinable via `hit_refs`). The autonomous loop:
`log_hit(status='needs-verify')` → `hits_pending()` → work it → `hits_mark(…)`
→ new leads. Currently: 24 user + 6 agent hits.

## 7. The dashboard (:8070)

- **Stage cards** — 1 · Ingest (any book format → Parquet), 2 · Rename into
  the DDC tree, 3 · Recipes extraction (`extract_recipes.py`), 4 · Index →
  Qdrant. Each card: input path, per-task GPU selection, Start/Stop, live log
  tail. Tasks run as subprocesses in their own process groups, logs in
  `~/base/gui/logs/`.
- **Pipeline progress** — `ocr.py` and `index_books.py` write machine-readable
  state to `~/.ddc-progress/` as they work; the dashboard turns that into live
  *where it is / how far to go / pace / ETA* (works for GUI- or CLI-started
  runs alike).
- **GPU cards** — per-GPU VRAM/utilisation; power-limit buttons
  (`nvidia-smi -pl`, e.g. 150 W / 300 W on the V100s, one-time passwordless
  sudo, choice saved to `config.json` and re-applied at boot).
- **Services** — Qdrant (docker compose start/stop), Ollama (pick GPU, pull
  models), and the **squig LLM card**: which profile is loaded, health,
  uptime, V100 VRAM, log tail, and **stop/unload** (frees both V100s' VRAM —
  this is the "LLM offline" switch for heavy OCR) or **load a profile**.
- **GPU pinning done right** — the GUI launches GPU tasks with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>` where `<n>` is the
  *nvidia-smi* index, so the selected physical GPU is always the one used.

## 8. GPUs & OCR

| nvidia-smi index | Card | torch (default order) | Role |
|---|---|---|---|
| 0 | Tesla V100-SXM2 32 GB | `cuda:0` | **vLLM squigFast** (tensor-parallel with GPU 2) — do not touch |
| 1 | **Quadro P2000 5 GB** | `cuda:2` | **marker OCR + Ollama index worker** |
| 2 | Tesla V100-SXM2 32 GB | `cuda:1` | **vLLM squigFast** — do not touch |

Gotcha (bit us once, 2026-11-25): torch's default *performance-based* ordering
does **not** match nvidia-smi's PCI ordering — `--device cuda:1` points at a
VLLM-occupied V100 (≈380 MB free ⇒ marker OOMs at model load), while the
actual OCR card is `cuda:2`. When running `ocr.py` by hand, either use
`--device cuda:2` for the P2000 or pin like the GUI does
(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`). **Verified 2026-11-25:
the full marker model set (6 models, fp16) loads on the P2000 with 2.56 GiB
headroom** using ocr.py's small-GPU batch tuning — scanned books and comics
OCR fine there. If a 3000-page monster or heavy colour work OOMs it, use a
free V100 (`--device cuda:0` after unloading the squig LLM, or the incoming
16 GB V100 when it arrives).

## 9. Current status (2026-11-25)

- **Library**: 650 book parquets, ~23 GB, in the DDC tree; Qdrant `ddc-books`
  = **186,938 points** (650/650 books indexed). Reader/dashboard/MCP/OWUI
  bridge all up.
- **Full-format batch test** (`/tmp/opencode/batchtest`, one real book per
  format, 17 books): 14/17 ingested and verified through stage 1
  (txt, rtf, html, htm, chm, docx, doc, odt, mht, fb2, epub, lit, mobi,
  azw3 — parquets + original siblings + correct word counts). The 3
  marker-OCR books (`.djvu` 75 pp, `.cbz` 121 pp, `.cbr` 4 pp) were converted
  to PDF successfully and are finishing OCR on the P2000 — first run hit the
  device-ordering OOM above.
- **Known tree state**: ~520 pre-existing books reference page images in
  deleted staging dirs (their text/parquet is fine; re-OCR repairs images);
  5 pre-existing duplicate identities (validator lists them).
- **Stages 2–4 on the batch** (enrich → classify into a throwaway tree →
  validate) are the remaining end-to-end step for the new formats; stage 5
  then indexes the survivors. Nothing blocks this except LLM time.

## 10. File map

| Path | Role |
|---|---|
| `~/base/ocr.py` | stage 1 ingest (format dispatch, marker, resume, sharding, progress) |
| `~/base/booktext.py` | format detection + all CPU text extractors (txt/epub/rtf/docx/html/mht/fb2/chm), nominal pages |
| `~/base/converters.py` | external-tool conversions (calibre, LibreOffice, djvups+ps2pdf, zip/7z/unrar+PIL) + `quick_page_count` |
| `~/base/install_formats.sh` | one-shot deps installer (idempotent) |
| `~/base/enrich_ddc_collection.py` | stage 2 (LLM metadata) |
| `~/base/rename_ddc_collection.py` | stage 3 (classify into DDC tree) |
| `~/base/validate_ddc_tree.py` | stage 4 (audit) |
| `~/DDC/indexer/index_books.py` | stage 5 (Qdrant indexing) |
| `~/base/gui/` | dashboard (:8070): `app.py`, `templates/`, `config.json`, `run.sh`, `set_gpu_power.sh` |
| `~/base/reader/` | reader (:8071): `app.py`, `mcp_server.py` (13 MCP tools), `ddc_owui.py` (bridge :8072), `ddq` (CLI), `hits_db.py`, `query-ollama.sh` (CPU embed worker), `start-all.sh` |
| `~/base/PIPELINE.md`, `~/base/SYSTEM.md`, `~/base/reader/README.md`, `~/DDC/indexer/README.md` | documentation |
| `~/.ddc-progress/` | live pipeline state files (ocr_*.json, index_*.json) |
| `~/squig-llm.log` | vLLM load log; `~/scripts/squig*` | LLM profile launchers |
