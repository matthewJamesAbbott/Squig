# squig — an eclectic book collection, turned into a searchable DDC library

One machine that turns a messy, mixed-format book collection (PDFs, EPUBs,
MOBIs, RTFs, HTML help files, scanned DJVUs, comics…) into:

1. a **DDC-classified digital library** — every book a 16-column Parquet
   file (one row per page) sitting beside its **original source file** and
   its page images, in a Dewey-Decimal folder tree;
2. a **web reader** — browse the tree, read text or rendered pages in a
   split view, search inside a book;
3. a **hybrid search index** — every content page embedded (bge-m3,
   1024-d) into Qdrant with a BM25 keyword index on top;
4. a **RAG answer engine** — ask the library a question; it retrieves pages,
   answers *only from them*, and cites them `[1] [2]` with deep-links back
   into the reader;
5. **agent interfaces** — 13 MCP tools for LLM agents (OpenCode, Open
   WebUI), a stable versioned `/v1` machine API, and a `ddq` CLI.

All of it is driven by a **five-stage pipeline** (Ingest → Enrich →
Classify → Validate → Index) with a **dashboard** to run, monitor and
control it.

> Full narrative report: [`base/SYSTEM.md`](base/SYSTEM.md). Per-component
> deep dives: [`base/PIPELINE.md`](base/PIPELINE.md),
> [`base/reader/README.md`](base/reader/README.md),
> [`indexer/README.md`](indexer/README.md).

---

## Repository layout

```
base/                      the pipeline + the two web apps
├── ocr.py                 stage 1 · Ingest: any book format → Parquet
├── booktext.py            format autodetection + all CPU text extractors
├── converters.py          external-tool conversions (calibre/LibreOffice/…)
├── install_formats.sh     one-shot dependency installer (idempotent)
├── enrich_ddc_collection.py   stage 2 · Enrich (one LLM call per book)
├── rename_ddc_collection.py   stage 3 · Classify into the DDC tree
├── validate_ddc_tree.py       stage 4 · read-only audit
├── extract_recipes.py           (optional) LLM recipe extraction
├── PIPELINE.md  SYSTEM.md       documentation
├── gui/                   the dashboard (Flask, port 8070)
└── reader/                the reader + /v1 API + MCP + bridge (Flask, 8071/8072)
    ├── app.py                 reader web app (browse/read/ask/search)
    ├── mcp_server.py          13 MCP tools (stdio, for OpenCode)
    ├── ddc_owui.py            Open WebUI bridge: "ddc-library" model + /mcp
    ├── ddq                    CLI client for the /v1 API
    ├── hits_db.py             two-layer research log (MariaDB)
    └── start-all.sh           bring every service up after a reboot

indexer/                   stage 5 · Index: Parquet → Qdrant
├── index_books.py             embed content pages (Ollama bge-m3)
├── repair_failed.py  verify_index.py  run_until_done.sh
├── docker-compose.yml         Qdrant container
└── OPEN_WEBUI_SETUP.md

scripts/                   setup helpers + the LLM
├── owui_register.py         one-shot: registers ddc-library + the 13 MCP
│                            tools in Open WebUI (idempotent, stdlib-only)
├── squig_research.sql       MariaDB schema for the research/hit log
│                            (idempotent; the reader also auto-creates it)
└── squigFast / squigFastThink / squigCode / squigChat
                             vLLM profile launchers for the answer LLM
```

The **book data itself is not in this repo** — it lives in a DDC tree at
`~/DDC` (Parquet + source files + images, tens of GB). Everything above is
code + docs.

---

## The five-stage pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 · Ingest | `base/ocr.py` | Per-file format autodetect (magic bytes → extension → text sniff) → one Parquet per book (16 columns, one row per page). PDF/DJVU/CBZ/CBR go through **marker OCR on GPU**; everything else through **CPU text extractors** (re-paginated into nominal 300-word pages); Kindle/Office/DJVU/comics are **converted first** by external tools. Copies the **original** source file beside each Parquet. |
| 2 · Enrich | `base/enrich_ddc_collection.py` | One LLM call per book adds `final_title / authors / publisher / pub_date / ddc / summary`; marks TOC pages `is_index` by local heuristic. In place, resumable. |
| 3 · Classify | `base/rename_ddc_collection.py` | Names everything `DDC - title - publisher - author - date`, copies original + Parquet + images into the DDC tree, rewrites `image_path`, verifies. `--dry-run` first, `--execute` to commit. |
| 4 · Validate | `base/validate_ddc_tree.py` | Read-only audit: schema, page counts, source sibling, image refs resolve, numeric DDC, duplicate identities. Exit 0 = clean. |
| 5 · Index | `indexer/index_books.py` | Embeds content pages with bge-m3 (Ollama) into Qdrant `ddc-books`. Deterministic point ids ⇒ idempotent; skips what's done; repairs partial books. |

Typical CLI flow for a new category:

```bash
PY=~/miniconda3/envs/marker_env/bin/python
$PY ~/base/ocr.py --input ~/Books --output ~/converted/Books --device cuda:2
$PY ~/base/enrich_ddc_collection.py ~/converted/Books
$PY ~/base/rename_ddc_collection.py --source ~/converted/Books --dest ~/DDC --dry-run
$PY ~/base/rename_ddc_collection.py --source ~/converted/Books --dest ~/DDC --execute
$PY ~/base/validate_ddc_tree.py ~/DDC
$PY ~/DDC/indexer/index_books.py --root ~/DDC
```

…or click the cards on the dashboard (port 8070).

---

## Supported book formats

Stage 1 ingests **any mix of these in one folder**. Detection order per
file is **magic bytes → extension → text sniff**, so mislabelled files still
get the right extractor (real collections are full of them).

| Format | Extensions | Path |
|---|---|---|
| PDF | `.pdf` | **marker OCR on GPU** — one row/page, text + page images |
| EPUB | `.epub` `.opub` | CPU extractor, images preserved |
| Plain text | `.txt` `.text` `.textbook` | CPU |
| RTF | `.rtf` | `striprtf` → text |
| DOCX | `.docx` | `mammoth` → HTML (data-URI images) → text + images |
| HTML | `.html` `.htm` | shared HTML block parser (images preserved) |
| MHTML | `.mht` `.mhtml` | MIME mail (stdlib `email`) → HTML blocks |
| FictionBook | `.fb2` | lxml XML → text |
| CHM | `.chm` | `pychm`/libchm → per-topic HTML in TOC order (natural-sorted archive enumeration when no HHC TOC) |
| Kindle | `.mobi` `.azw` `.azw3` `.azw4` `.kfx` `.lit` | **calibre `ebook-convert` → EPUB** (then EPUB path) |
| Office legacy | `.doc` | **LibreOffice headless → DOCX** (then DOCX path) |
| OpenDocument | `.odt` `.ods` `.odp` | LibreOffice headless → DOCX |
| DJVU | `.djvu` | `djvups` (PS, page-batched) + `ps2pdf` → PDF → marker OCR |
| Comics | `.cbz` `.cbr` | zip / `7z` / `unrar` extract + PIL → PDF (natural panel order) → marker OCR |

Notes:

- **Converted files are intermediates.** The Parquet's `source_file` and the
  tree sibling are always the **original** (`.djvu`, `.mobi`, `.doc`, …).
  Conversion workdirs live in `<out>/_conv/<rel>/<stem>/`, kept on failure
  (resume reuses them), deleted on success.
- **Text formats use nominal pages**: re-paginated into `--words-per-page`
  (default 300) pages that break only *between* paragraphs. The reader's
  split view works for these via stored page images; there is no PDF sibling.
- **Scanned books (PDF/DJVU/CBZ/CBR)** get real per-page OCR text from
  marker's post-processed pipeline (not just the embedded text layer, which
  is empty for a scan). Because the original `.djvu`/`.cbz`/`.cbr` is not a
  renderable PDF, the full-page raster is also saved (one JPEG/page) so the
  reader's split view shows the page.
- **Install deps** (once): `bash base/install_formats.sh` → pip
  `striprtf` `mammoth` `pychm` + apt `djvulibre-bin` `calibre` `libchm-dev`.
  Already needed on most boxes: LibreOffice, ghostscript (`ps2pdf`), `7z`,
  `unrar`, pypdf, PIL.
- PDF/DJVU/comics need the GPU (marker); everything else is CPU. Multi-GPU
  fan-out via `ocr.py --shard I/N` for PDF-like files.

---

## Services & links

| Service | Link | What it is |
|---|---|---|
| **Pipeline dashboard** | `http://localhost:8070` | Run & monitor all 5 stages, GPU cards + power limits, service controls (Qdrant / Ollama / LLM load-unload), live progress + ETA |
| **Reader** | `http://localhost:8071` | Browse / read / ask / search UI |
| **Machine API** | `http://localhost:8071/v1` | Stable versioned endpoints: `/v1/search`, `/v1/ask`, `/v1/chat`, `/v1/books`, `/v1/page/{id}/{n}`, `/v1/chunk/{id}`, `/v1/source/{id}`, `/v1/status`, `/v1/index/*`, `POST /v1/rescan` |
| **OWUI bridge** | `http://localhost:8072` | `ddc-library` in the Open WebUI model picker + the 13 MCP tools over streamable HTTP (`POST /mcp`) — wire it up with `scripts/owui_register.py` |
| **Open WebUI** | `http://localhost:3000` | Docker; pick `ddc-library` or toggle the 13 tools in any chat |
| Qdrant | `http://localhost:6333` | `ddc-books`: 1024-d cosine + BM25 text index |
| Ollama (GPU) | `http://localhost:11434` | bge-m3 embeddings for the indexer |
| Ollama (CPU) | `http://localhost:11435` | keeps interactive search ~0.2 s while the indexer runs |
| MariaDB | `127.0.0.1:3306` | `squig_research` — the two-layer research/hit log (schema: `scripts/squig_research.sql`) |
| Answer LLM (vLLM) | `http://<llm-host>:8000/v1` | `squigFast` (Qwen3-27B) — also serves the dashboard's chat and enrichment |

Startup after reboot: `base/reader/start-all.sh`.

### Wiring Open WebUI (one-shot, idempotent)

```bash
~/base/reader/run.sh owui start                      # the bridge (:8072)
python3 ~/squig/scripts/owui_register.py --key <OWUI admin API key>
```

That registers (a) an OpenAI connection at `http://172.17.0.1:8072/v1` so
`ddc-library` appears in the model picker (chatting with it runs the RAG
agent), and (b) an MCP tool-server connection at
`http://172.17.0.1:8072/mcp` so all 13 tools can be toggled in any chat.
Key: OWUI → Settings → Profile → API Keys → New (revocable afterwards).

### Provisioning the research DB (optional)

The reader creates the MariaDB schema itself at startup; for out-of-band
provisioning (fresh machine, other users):

```bash
mysql -u matt -p < ~/squig/scripts/squig_research.sql
```

### MCP tools (13)

`search_library`, `ask_library`, `get_chunk`, `get_source`,
`library_status`, `browse_ddc`, `find_books`, `get_page`,
`rescan_library`, plus the research-log quartet: `log_hit`, `hits_recall`,
`hits_pending`, `hits_mark`.

Register with OpenCode:

```bash
opencode mcp add ddc --global -- \
  ~/miniconda3/envs/marker_env/bin/python ~/base/reader/mcp_server.py
```

### `ddq` CLI

```bash
~/base/reader/ddq status
~/base/reader/ddq search "protein folding" --k 8
~/base/reader/ddq ask "how does CRISPR work?" --ddc 570
```

Every command `--json`; `READER_URL` overrides the base.

---

## GPUs & OCR

The OCR card must be a *free* GPU. On the reference machine the two 32 GB
V100s are occupied by the vLLM LLM, and a Quadro P2000 is the OCR/index card.

> **Gotcha** (bit us once): torch's default *performance-based* device
> ordering does **not** match `nvidia-smi`'s PCI ordering. `--device cuda:1`
> may point at a busy V100 while the free card is `cuda:2`. The dashboard
> pins GPUs correctly (`CUDA_DEVICE_ORDER=PCI_BUS_ID
> CUDA_VISIBLE_DEVICES=<smi-index>` + `cuda:0`), so prefer it. When running
> `ocr.py` by hand, verify the mapping with `nvidia-smi` +
> `torch.cuda.get_device_properties` first.

The full marker model set (6 models, fp16) loads on a 5 GB P2000 with
~2.5 GB headroom using `ocr.py`'s small-GPU batch tuning. For very large /
colour-heavy books, use a bigger card.

---

## Deployment notes

- **Python**: everything runs from one env with `marker-pdf`, `surya-ocr`,
  `striprtf`, `mammoth`, `pychm`, `pypdf`, `pymupdf`, `pandas`, `pyarrow`.
  The scripts resolve their interpreter from
  `PIPELINE_PYTHON` (default `~/miniconda3/envs/marker_env/bin/python`).
- **Paths**: the code defaults to a `~/base` + `~/DDC` layout
  (`BASE_DIR = ~/base`, `DDC_ROOT = ~/DDC`). This repo ships `base/` and
  `indexer/` separately — either clone to `~/squig` and symlink
  (`ln -s ~/squig/base ~/base`, `ln -s ~/squig/indexer ~/DDC/indexer`), or
  override with the env vars below.
- **Key env overrides**: `DDC_ROOT`, `READER_PORT`, `QDRANT_URL`,
  `EMBED_BASE_URL`, `EMBED_MODEL`, `LLM_BASE_URL`, `LLM_MODEL`,
  `PIPELINE_PYTHON`, `DDC_PROGRESS_DIR`, `HITS_DB_*` (MariaDB),
  `READER_URL` (for `ddq` / MCP / bridge).
- **Book data is not committed.** The DDC tree (`~/DDC`) and any
  `~/converted/` staging dirs are generated by the pipeline. The MariaDB
  `squig_research` database is provisioned by `scripts/squig_research.sql`
  (or auto-created by the reader).
