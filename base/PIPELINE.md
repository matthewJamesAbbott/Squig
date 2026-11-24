# Book pipeline: any book format → searchable DDC library

Five stages, each an independent script (dashboard cards on :8070, or CLI):

| Stage | Script | What it does |
|---|---|---|
| 1 · Ingest | `ocr.py` | **Format autodetected per file** (magic bytes → extension → text sniff), so a folder of mixed formats just works and a mislabelled file still gets the right extractor. Three paths: **PDF / DJVU / CBZ / CBR** → marker OCR on GPU (one parquet per book, one row/page, text + images into a shared `images/` dir; DJVU is first converted to PDF via `djvups`+`ps2pdf` (batched, pypdf merge) and comics via image extraction + PIL); **EPUB / TXT / RTF / DOCX / HTML / MHT / FB2 / CHM** → CPU text extractors (`booktext.py`) re-paginated into **nominal pages** of `--words-per-page` (default 300) that break only *between* paragraphs (no mid-paragraph splits unless a paragraph alone exceeds 1.5× target); **Kindle (.mobi/.azw/.azw3/.kfx/.lit) → calibre ebook-convert → EPUB** and **DOC/ODT → LibreOffice headless → DOCX** (external tools in `converters.py`, run `install_formats.sh` once). Copies the **original** source file next to the parquet. Multi-GPU fan-out via `--shard I/N` (PDF-like only; text formats convert on any core). |
| 2 · Enrich | `enrich_ddc_collection.py` | One LLM call per book (squigFast, `http://10.90.38.19:8000/v1`) adds `final_title / authors / publisher / pub_date / ddc / summary` in place; marks TOC/index pages `is_index` locally (dot-leader + trailing-INDEX heuristics). Output = the 16-column DDC schema. Resumable: re-runs skip books whose `ddc` is already set. **Must run before stage 3** — unenriched parquets are skipped there. |
| 3 · Classify | `rename_ddc_collection.py` | Reads enriched parquets, names everything `DDC - title - publisher - author - date`, checks destination collisions (skip unless `--overwrite`), copies the original source file (PDF / EPUB / TXT / any converter original like .djvu/.mobi/.doc) + parquet + images + index files into the DDC tree, rewrites `image_path` to the new destination, verifies every file after the copy. |
| 4 · Validate | `validate_ddc_tree.py` | Read-only audit of the whole tree: schema order, page counts, source-file sibling, every image reference resolves on disk, numeric DDC, duplicate book identities. Exit 0 = clean. |
| 5 · Index | `indexer/index_books.py` | Embeds content pages (`is_index == False`) with bge-m3 (Ollama) into Qdrant. Skips books already fully indexed (exact point-count check), so re-runs only touch new/changed books. |

Typical flow for a new category (e.g. `Psychology`, any mix of the supported formats):

```bash
PY=/home/matt/miniconda3/envs/marker_env/bin/python
# 1 · Ingest — format autodetected per file; text formats are fast CPU jobs,
#    PDF/DJVU/comics use the GPU. For text-based inputs add --words-per-page
#    250 etc. to tune the nominal pages.
$PY /home/matt/base/ocr.py --input ~/Psychology --output ~/converted/Psychology --device cuda:1
$PY /home/matt/base/enrich_ddc_collection.py ~/converted/Psychology
$PY /home/matt/base/rename_ddc_collection.py --source ~/converted/Psychology --dest ~/DDC --dry-run
$PY /home/matt/base/rename_ddc_collection.py --source ~/converted/Psychology --dest ~/DDC --execute
$PY /home/matt/base/validate_ddc_tree.py ~/DDC
$PY /home/matt/DDC/indexer/index_books.py --root ~/DDC
```

## Known tree state (2026-11-24)

- ~520 pre-existing books have `image_path` references into the old
  `/home/matt/converted/…` staging dirs, which were deleted after the earlier
  (pre-fix) classifications — their text/PDF/parquet are fine, the images are
  gone unless those books are re-OCR'd. Stage 3 since then copies images into
  the tree, so new books are self-contained.
- 5 duplicate identities (same book filed twice) were pre-existing; the
  validator lists them.

## Conventions

- DDC tree: `~/DDC/<NNN>_<century>/<NNN>_<section>/…`; `find_ddc_directory`
  validates the ancestor chain and falls back to `<NNN>_Unclassified`.
- Enrichment LLM: `squigFast` via vLLM in the LXC (ecat); the model also
  returns a `reasoning` field — only `content` is parsed.
- The GUI (`~/base/gui`) tracks every stage as a task; enrichment/rename/
  index defaults live in `config.json`.
