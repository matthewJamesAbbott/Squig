# DDC Book Library → Qdrant Indexer

Indexes the DDC book library (`~/DDC/`) into a [Qdrant](https://qdrant.tech/)
vector database so books can be searched semantically (e.g. from
[Open WebUI](./OPEN_WEBUI_SETUP.md)).

- Each `*.parquet` under `~/DDC/` is one book; each row is one page.
- Page text is embedded with **bge-m3** (BAAI, 1024-dim, 8192-token context)
  served by **Ollama**.
- Points land in the Qdrant collection `ddc-books` (cosine distance).
- TOC/index pages (`is_index == true`) are excluded.
- **Text-only ingestion: images are NOT embedded.** The parquets carry
  `image_path` / `image_name` / `image_bbox` columns and the extracted PNGs
  live in a per-book folder next to each `.parquet` (alongside the original
  PDF), but only `text` + metadata columns are read. Pages whose text is
  empty (image-only / scanned pages, ~21% of pages) are **skipped** — with
  no image ingestion they have nothing to embed and would only pollute
  search results. If image search is wanted later, use a multimodal model
  (e.g. `bge-visualized` or a CLIP variant) as a second collection; the
  image files are all still on disk.

## Files

| File | Purpose |
| --- | --- |
| `index_books.py` | The indexing pipeline (Python 3.11, pyarrow + requests only) |
| `run_until_done.sh` | **Unattended supervisor**: keeps the indexer alive across crashes/reboots; exits when a full run completes or the indexer is stopped |
| `verify_index.py` | Spot-check helper: point counts, payload/vector inspection, sample semantic search |
| `docker-compose.yml` | Qdrant service (HTTP 6333, gRPC 6334, persistent volume) |
| `OPEN_WEBUI_SETUP.md` | How to hook the index into Open WebUI |
| `failed.jsonl` | (generated) one JSON line per failed file: `{filename, error, timestamp}` |

> **Qdrant version note:** written against Qdrant **1.20.x**, which moved a
> few REST endpoints (create collection: `PUT /collections/{name}`; payload
> index: `PUT /collections/{name}/index`; upsert: `PUT /collections/{name}/points`;
> search: `POST /collections/{name}/points/query`). The script is pinned to
> these; on Qdrant ≤1.19 use the old POST/`indexes`/`points/search` paths.

## Quick start

```bash
# 1. Qdrant (once)
cd ~/DDC/indexer && docker compose up -d

# 2. Ollama on the P2000 (terminal 2; keep running)
CUDA_VISIBLE_DEVICES=2 OLLAMA_NUM_PARALLEL=1 ollama serve
#    (first time only:)  ollama pull bge-m3

# 3. Verify the embedder returns 1024-dim vectors
curl -s localhost:11434/api/embed -d '{"model":"bge-m3","input":"hello"}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["embeddings"][0]), "dims")'

# 4. Dry run (no embedding, no writes)
python index_books.py --root ~/DDC --limit 5 --dry-run

# 5. Index everything
python index_books.py --root ~/DDC
```

Use the conda env with the right packages:

```bash
/home/matt/miniconda3/envs/marker_env/bin/python index_books.py --root ~/DDC
```

## Re-indexing / adding new books

The pipeline is **idempotent, resumable and self-healing**:

- `book_id` = first 16 hex chars of `sha256("final_title|authors|pub_date")`.
  **Duplicate-title disambiguation:** when several parquets hash to the same
  `book_id` (the same book OCR'd from two PDFs, or misfiled under two DDC
  sub-dirs — and in this library, four different books whose OCR produced
  `Unknown|Unknown|Unknown`), they would otherwise share *point* ids too
  (`point_id` hashes `book_id:page_num`), overwrite each other's points, and
  can never satisfy the exact-count skip check — every run would delete and
  re-embed all of them, forever. So at the start of a run the files are
  grouped by base id: the first file in sort order keeps the plain id
  (matching whatever is already in Qdrant — zero re-embedding cost for
  one-offs), each further file gets the hash seeded with its
  root-relative path. Deterministic across runs, logged as
  `book_id disambiguated (...)`.
- Point ids are deterministic (`uint64` hash of `book_id:page_num`), so
  re-indexing a book **replaces** its points instead of duplicating them.
- A book is skipped only when Qdrant holds *exactly* the number of points
  the current pipeline would write. Fewer points (a run died mid-upsert) or
  more (an older pipeline version, e.g. one that also embedded empty-text
  pages) ⇒ the book's points are deleted and rebuilt from scratch. The
  delete is two-pass: a filter delete (uses the `book_id` payload index),
  then a by-point-id delete of any stragglers the follow-up count still
  sees — observed live under optimizer load, where a plain filter delete
  once left 96 points behind.
- If a run crashes at book 3000/5000, just run it again: the first 2999 are
  skipped in milliseconds, the half-finished 3000th is detected and rebuilt,
  and the run continues from 3001.

So after adding new books: `python index_books.py --root ~/DDC` again —
already-indexed books are skipped, new ones are added.

**Unattended long-haul runs** — for the months-long ingestion (new parquets
landing periodically, machine may reboot, Ollama may blip), run the
supervisor instead of a bare pipeline:

```bash
nohup ~/DDC/indexer/run_until_done.sh > ~/DDC/indexer/supervisor.log 2>&1 &
tail -f ~/DDC/indexer/supervisor.log
# stop:  pkill -f "run_until_done[.]sh"   (or the UI "Stop indexer" button)
```

It follows an already-running indexer, restarts the pipeline 60 s after any
crash (the self-heal scan picks up exactly where it died), and exits by
itself once a full run completes (rc 0) or the indexer is intentionally
stopped.

## Live progress (dashboard)

The pipeline writes a small machine-readable state file as it works, so the
pipeline dashboard (`~/base/gui`, http://localhost:8070 — "Pipeline
progress" section + `/api/pipeline`) and the reader's `/v1/index/status`
can show **exactly which book/page it is on, how far it has to go, the
current pace, and an ETA** — for runs started from the GUI, the CLI, or the
supervisor alike.

- File: `~/.ddc-progress/index_<root-name>.json`
  (env `DDC_PROGRESS_DIR` to relocate; written atomically, fail-soft — a
  progress-write error can never kill the index run).
- Updated after every book and (throttled to ~3 s) after every embed batch
  inside a book, so the in-book position is live too.
- Contents: state (`running`/`done`/`failed`/`stalled`),
  `total_books`/`done_books`, `total_pages`/`pages_done`/`remaining_pages`
  (page counts come from parquet metadata up front, so the exact total
  workload is known before the first embed), `current_book` +
  in-book page position, per-status `counts`, and rolling
  `[(ts, pages_embedded), …]` sample series (last 64) that the dashboards
  turn into pace + ETA (page-based when a page pace exists, book-based
  otherwise).
- The dashboard also falls back to parsing the shared `index_run.log` when
  a *pre-progress-file* indexer build is still running, and flags a
  progress file as `stalled` when its writer process is gone.

(The OCR pipeline, `~/base/ocr.py`, writes the same contract:
`~/.ddc-progress/ocr_<outdir>.json`, updated per chunk/file.)

To force re-indexing one book (e.g. after a metadata fix), delete its points
first, then re-run:

```bash
curl -s -X POST http://localhost:6333/collections/ddc-books/points/delete \
  -d '{"filter":{"must":[{"key":"book_id","match":{"value":"<book_id>"}}]}}'
```

## Collection schema and scale configuration

Payload of every point (written by `index_books.py`):

| Field | Meaning |
| --- | --- |
| `text` | the page's OCR text (also the BM25 full-text index field) |
| `page_num` | 1-based page number |
| `book_id` | deterministic `sha256(final_title\|authors\|pub_date)[:16]` |
| `book_title`, `authors`, `publisher`, `pub_date`, `ddc` | book metadata |
| `ddc_coarse`, `ddc_tens` | top-level DDC section (folder) + tens-level (filterable) |
| `source_type` | always `"book"` for now; JSTOR (`"journal_article"`) and newspapers (`"newspaper"`) will use the same collection and schema |
| `path` | library-relative file path (e.g. `500_Science/.../book.parquet`) — search hits are self-locating without a mapping service |

Payload indexes (all filterable fields, so filters stay fast at scale):
`book_id` (keyword), `ddc_coarse`, `ddc_tens`, `pub_date`, `source_type`
(keyword), `page_num` (integer), `text` (text, BM25 scoring).

Storage configuration applied to the collection (persisted in Qdrant):

- `hnsw_config.on_disk = true` — HNSW graphs live on disk (mmap-backed);
  RAM stays bounded as the collection grows.
- `on_disk_payload = true` (default for this collection).
- `optimizer_config.memmap_threshold = 100000` — any optimized segment with
  >100k vectors keeps its vectors mmap'd instead of RAM-resident. No-op at
  book scale; automatic at JSTOR/newspaper scale.

## When the data grows (JSTOR, then newspapers)

The agreed vectorization policy: **vectorize the high-signal text, BM25
(full-text) everything.**

- **Books (now):** every text page → bge-m3 vector + BM25. ~643 books,
  ~170k points, ~4 GB.
- **JSTOR (year 1):** OCR + parquet each article, then ingest with
  `source_type: "journal_article"`. Vectors: abstract + first ~2 pages only;
  the full article body goes to the BM25 leg (the Qdrant text index on
  `text` needs no per-vector storage). Add payload fields: `journal`,
  `year`, `doi`, `date` (ISO, for range filters — newspaper-era needs it
  too).
- **Newspapers (year 1+):** BM25 + date is the primary retrieval path
  (that's how you find a 1978 headline); optional cheap vectors for
  front-page/high-signal pages only.

Before a big ingestion lands, apply quantization to the dense leg so 1M+
vectorized sources stay in RAM (the box has 188 GB; 300M float32 1024-dim
vectors = 1.2 TB, but binary-quantized = ~38 GB of codes + rescore from the
mmap'd originals):

```bash
curl -X PATCH http://localhost:6333/collections/ddc-books \
  -d '{"quantization_config":{"binary":{"always_ram":true}}}'
```

(cheap to apply at ~200k points; do it *before* the collection is large.)
Also plan storage: the Qdrant volume is on the root disk (~470 GB total,
watch free space) — at JSTOR scale the volume likely moves to one of the
11 TB veracrypt volumes (recreate the compose volume there).

## GPU notes (P2000 vs V100)

The script never talks to the GPU directly — only to Ollama over
`http://localhost:11434`. GPU selection happens entirely at `ollama serve`
startup, so switching GPUs is just a different terminal:

| GPU | nvidia-smi index | CUDA index | Startup command |
| --- | --- | --- | --- |
| V100 (PCI 02:00) | 0 | 0 | `CUDA_VISIBLE_DEVICES=0 OLLAMA_NUM_PARALLEL=1 ollama serve` |
| V100 (PCI 85:00) | 2 | 1 | `CUDA_VISIBLE_DEVICES=1 OLLAMA_NUM_PARALLEL=1 ollama serve` |
| P2000 | 1 | 2 | `CUDA_VISIBLE_DEVICES=2 OLLAMA_NUM_PARALLEL=1 ollama serve` |

> **Note:** CUDA orders devices by performance, *not* by `nvidia-smi` PCI
> order, so the P2000 is CUDA index **2** even though `nvidia-smi` shows it
> at index 1. Verified with `torch.cuda.get_device_name(i)`.

Ollama is installed user-locally (no root needed): binary at
`~/.local/ollama/ollama`, on PATH via `~/.local/bin/ollama`.

- **P2000 (current):** 5 GB VRAM, fine for bge-m3 (~1.2 GB). `EMBED_BATCH=32`
  pages/request is safe; keep `OLLAMA_NUM_PARALLEL=1`.
  Measured: **~180 s/book (~0.4 s/page)** → ~32 h for the current 643 books.
- **V100 (32 GB, later):** you can raise `EMBED_BATCH` in `index_books.py`
  (e.g. 64–128) and/or `OLLAMA_NUM_PARALLEL` to go faster. Nothing else
  changes — the script only talks to the Ollama HTTP API. A V100 is roughly
  15× faster on bge-m3, so a full 5000-book run is ~2 h instead of ~4 days.
  To switch mid-project: stop the P2000 `ollama serve`, start it on the
  V100, and re-run the indexer — already-indexed books are skipped, so a
  partially-finished run simply continues on the faster GPU.

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `EMBED_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `EMBED_MODEL` | `bge-m3` | embedding model (must match what built the index) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |

Example: `EMBED_BASE_URL=http://localhost:11434 python index_books.py`

CLI flags: `--root DIR` (default `~/DDC`), `--limit N` (first N files, 0 = all),
`--only SUBSTR` (only files whose name contains SUBSTR, repeatable — useful
for testing or re-indexing one book), `--dry-run` (scan only, no
embedding/writes; still reports which books are already indexed).

### Ollama crash resilience

If Ollama dies mid-run (it did once, killing the whole `ollama serve`
process), the embedder detects the connection error and **polls
`/api/tags` for up to 15 min** per attempt before giving up on a book
(5 attempts total). So a crash + restart within ~15 min costs no books.
If it stays down longer, the affected books are logged to `failed.jsonl`
and the run continues; re-run the indexer after Ollama is back (partial
books are detected by the exact-count check and rebuilt automatically —
`repair_failed.py` is only needed to inspect the plan or clear the log).

## Storage / memory

The collection is already configured for growth (see *Collection schema and
scale configuration* above): HNSW on disk, payload on disk, and segments
>100k vectors are mmap'd automatically. So RAM stays bounded even at JSTOR
scale — the constraint that bites first at scale is **disk**, not RAM
(300M float32 vectors = 1.2 TB before quantization; see the checklist above).

At current book scale (~170k points) the whole collection is only a few GB
and Qdrant's container footprint is ~1–2 GB RAM.

## Failure handling

A bad file (unreadable parquet, Ollama down mid-run, …) is logged, appended
to `failed.jsonl` as `{filename, error, timestamp}`, and the run continues.

Non-book parquets that live in the tree (e.g. `recipes.parquet` /
`ingredients.parquet` produced by `extract_recipes.py`) have no `final_title`
column; they are detected by schema and **skipped, not failed**.

### Re-indexing failed books

If a book failed *after* some of its pages were already upserted, it holds
**partial points** in Qdrant. The exact-count check above catches this
automatically: the next run deletes the partial points and re-indexes the
book, so a plain re-run is usually all you need. `repair_failed.py` remains
as a manual tool (inspect the plan with `--no-delete`, or clear
`failed.jsonl` without re-running the whole library):

```bash
python repair_failed.py            # delete partial points for every failed
                                   # book + clear failed.jsonl
python repair_failed.py --no-delete  # just show the plan
python index_books.py --root ~/DDC   # now indexes exactly the missing books;
                                     # everything else is skipped in minutes
```

If Ollama was simply down, most failed books have *no* points yet (they died
on their first embed batch) and the repair step only clears the log.

## Front end: the Flask viewer (`~/base/reader`)

The viewer at `~/base/reader` (http://localhost:8071) is wired into this
indexer — it is the front end for the whole system:

- **Semantic search** in the header (`/api/semantic`): the query is embedded
  with the same `EMBED_MODEL` (bge-m3) that built the index, so the vectors
  are directly comparable; hits deep-link into the book *and page* in the
  reader.
- **Status strip** on the browse page (`/api/index/status`): Qdrant health +
  point count, Ollama + model readiness, and live indexer progress
  (parsed from `index_run.log`).
- **Indexer control** (`/api/index/start`, `/api/index/stop`): starts/stops
  `index_books.py` with the same command line and shared `index_run.log` as
  a CLI run. Because of the self-healing exact-count check, stopping and
  restarting from the UI is safe — a half-finished book is rebuilt on the
  next start.
