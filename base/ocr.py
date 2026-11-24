#!/usr/bin/env python3
"""
convert_books_parquet.py
========================
Converts books to structured Parquet files for LLM/RAG use.
Formats (autodetected per file: magic bytes -> extension -> text sniff):
  PDF      — marker OCR on GPU (text + embedded images per physical page)
  EPUB     — spine-order text + figures, re-paginated into nominal pages
  TXT      — paragraphs re-paginated into nominal pages (~--words-per-page
             words, breaking only between paragraphs)
  RTF      — striprtf -> text path
  DOCX     — mammoth -> text + images path
  HTML/MHT — lxml -> text + images path
  FB2      — lxml XML -> text path (FictionBook, common in RU collections)
  CHM      — pychm -> per-topic HTML -> text path
  Kindle   — .mobi/.azw/.azw3/.kfx/.lit -> calibre ebook-convert -> EPUB
  DOC/ODT  — LibreOffice headless -> DOCX (system tools via converters.py)
  DJVU     — djvups (PS) + ps2pdf -> PDF -> marker OCR (converters.py)
  CBZ/CBR  -> PIL -> PDF -> marker OCR (converters.py)
Each parquet file = one book. Each row = one page.
Columns: source_file, book_title, page_num, page_width, page_height,
         text, image_bbox, image_path, image_name
The text extractors live in booktext.py; system-tool conversions in
converters.py (run install_formats.sh once for the extras).
Usage:
    python convert_books_parquet.py --input /path/to/books/ --output /data/parquet/
    python convert_books_parquet.py --input /path/to/books/ --output /data/parquet/ --resume
    python convert_books_parquet.py --input book.epub --output /data/parquet/
    python convert_books_parquet.py --input book.txt --output /data/parquet/ --words-per-page 250

Multi-GPU (N workers, one per GPU): split the PDF list round-robin with
--shard I/N and pin each worker to its own GPU (the GUI does this for you;
from the CLI, use CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=0 python convert_books_parquet.py -i IN -o OUT --shard 0/2 -d cuda:0
    CUDA_VISIBLE_DEVICES=2 python convert_books_parquet.py -i IN -o OUT --shard 1/2 -d cuda:0
Each worker writes its own checkpoint shard (conversion_checkpoint.json.g<i>)
and progress file (~/.ddc-progress/ocr_<out>_g<i>.json); --resume merges all
shards, so no file is ever converted twice.
"""
import os
import sys
import json
import time
import argparse
import gc
import traceback
import io
import re
import shutil
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional
# ── MUST be before any torch/marker imports ──────────────────────────────────
import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
# ─────────────────────────────────────────────────────────────────────────────
# Small-GPU batch tuning — marker's defaults OOM on a 5GB card (Quadro P2000).
# Effective batch = these values x --batch (DEFAULT_BATCH). On dense pages the
# detection/layout models (DETR, ~1000 queries/page) dominate VRAM, so keep
# the base values at 1. These are defaults only; real environment variables
# always win (os.environ.setdefault). Raise them for a bigger GPU.
for _k, _v in {
    "DETECTOR_BATCH_SIZE":    "1",
    "LAYOUT_BATCH_SIZE":      "1",
    "ORDER_BATCH_SIZE":       "2",
    "RECOGNITION_BATCH_SIZE": "2",
    "TABLE_REC_BATCH_SIZE":   "1",
    "TEXIFY_BATCH_SIZE":      "2",
    "SURYA_OCR_DPI":          "128",
    "SURYA_TABLE_DPI":        "128",
    # Keep large allocations in mmap'd memory so they don't touch glibc's
    # free lists (a mixed PDFium/MuPDF/OpenCV/torch workload can otherwise
    # hit "corrupted double-linked list" aborts after many pages).
    "MALLOC_MMAP_THRESHOLD_": "131072",
}.items():
    os.environ.setdefault(_k, _v)
# ─────────────────────────────────────────────────────────────────────────────
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
# ── Schema (one row = one page, single text field) ───────────────────────────
PAGE_SCHEMA = pa.schema([
    pa.field("source_file",            pa.string()),
    pa.field("book_title",             pa.string()),
    pa.field("page_num",               pa.int32()),
    pa.field("page_width",             pa.float32()),
    pa.field("page_height",            pa.float32()),
    # Single text field — all text from the page concatenated
    pa.field("text",                   pa.string()),
    # Image fields — each is a list matching images on the page
    pa.field("image_bbox",             pa.list_(pa.list_(pa.float32()))),
    pa.field("image_path",             pa.list_(pa.string())),
    pa.field("image_name",             pa.list_(pa.string())),
])
# ── Config ────────────────────────────────────────────────────────────────────
def pick_default_device() -> str:
    """Pick the CUDA GPU with the most free memory (falls back to CPU).

    GPU indices change when hardware changes (CUDA orders by speed, not by
    nvidia-smi index), so auto-pick the freest device instead of hardcoding.
    """
    try:
        if torch.cuda.device_count() == 0:
            return "cpu"
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_idx, best_free = i, free
        return f"cuda:{best_idx}"
    except Exception:
        return "cpu"

DEFAULT_DEVICE    = pick_default_device()  # Auto-pick freest GPU
DEFAULT_CHUNK     = 20
DEFAULT_BATCH     = 2  # Effective batch = env base value x this; OOM retry drops it to 1
DEFAULT_DTYPE     = torch.float16
CHECKPOINT_FILE   = "conversion_checkpoint.json"
HEADING_RATIO     = 1.4   # font size ratio above page median = heading
# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "partial": {}}

def load_shard_checkpoints(output_dir: Path) -> dict:
    """Merge the base checkpoint with any .g<i> shard files (multi-GPU runs).

    Each sharded worker only writes its OWN shard file (no write races), so
    a plain single-process run and sharded runs share completion state:
    anything completed by any worker — or by an earlier single-process run —
    is skipped by all. 'failed' is unioned too, so the progress 'failed
    files' count is accurate no matter which worker recorded it."""
    state = {"completed": [], "failed": [], "partial": {}}
    paths = [output_dir / CHECKPOINT_FILE]
    paths += sorted(output_dir.glob(f"{CHECKPOINT_FILE}.g*"))
    for p in paths:
        s = load_checkpoint(p)
        for k in s.get("completed", []):
            if k not in state["completed"]:
                state["completed"].append(k)
        for k in s.get("failed", []):
            if k not in state["failed"]:
                state["failed"].append(k)
        state["partial"].update(s.get("partial", {}))
    return state

def save_checkpoint(path: Path, state: dict):
    # with_name (not with_suffix): shard files like conversion_checkpoint.json.g0
    # must not all share one .tmp name, or concurrent saves race on the rename.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)
# ── Live progress (dashboard: ~/base/gui + /api/pipeline) ───────────────────
# Machine-readable state, written atomically after every chunk and file so
# the UI can show which file/page we are on, how far we have to go, and a
# live ETA — for runs started from the GUI or from the CLI alike.
# Never raises: progress is a nicety, the conversion is the job.
PROGRESS_DIR = Path(os.environ.get(
    "DDC_PROGRESS_DIR", str(Path.home() / ".ddc-progress")))

def progress_path_for(output_dir: Path, shard: "tuple | None" = None) -> Path:
    """One progress file per output dir: ~/.ddc-progress/ocr_<outdir>.json.

    A sharded (multi-GPU) worker gets its own file, ocr_<outdir>_g<i>.json —
    the dashboard groups them by the shared base name and sums them into one
    card. Unsharded runs keep the plain name (back-compat with older runs).
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", output_dir.name).strip("_") or "out"
    if shard:
        return PROGRESS_DIR / f"ocr_{slug}_g{shard[0]}.json"
    return PROGRESS_DIR / f"ocr_{slug}.json"

def save_progress(path: Path, state: dict):
    try:
        state = dict(state)
        state["pid"] = os.getpid()
        state["last_update"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        tmp.replace(path)
    except Exception:
        pass
# ── PDF utilities ─────────────────────────────────────────────────────────────
def get_page_count(pdf_path: Path) -> int:
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return 0
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        pass
    return 99999
def get_book_list(input_path: Path) -> list:
    """Find every convertible book under input_path.

    Native:  *.pdf *.epub *.opub *.txt *.text *.textbook *.rtf *.docx
             *.html *.htm *.mht *.mhtml *.fb2 *.chm
    Converted (system tool): *.mobi *.azw *.azw3 *.kfx *.lit *.doc *.odt
             *.ods *.odp *.djvu *.cbz *.cbr
    Plus extensionless files whose magic bytes sniff as any of the above
    (detect_format). Sorted for deterministic sharding."""
    if input_path.is_file():
        return [input_path]
    from booktext import detect_format
    known_exts = {
        ".pdf", ".epub", ".opub", ".txt", ".text", ".textbook",
        ".rtf", ".docx", ".html", ".htm", ".mht", ".mhtml", ".fb2",
        ".chm",
        ".mobi", ".azw", ".azw3", ".kfx", ".lit",
        ".doc", ".odt", ".ods", ".odp",
        ".djvu", ".cbz", ".cbr",
    }
    files = []
    seen = set()
    for p in sorted(input_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in known_exts:
            files.append(p)
            seen.add(p)
        elif not p.suffix and p.stat().st_size > 0:
            if detect_format(p) != "unknown":
                files.append(p)
                seen.add(p)
                log(f"  sniffed format from magic bytes: {p.name}")
    files = sorted(p for p in set(files))
    if not files:
        log(f"ERROR: No books (pdf/epub/txt/rtf/docx/html/chm/mobi/doc/"
            f"odt/djvu/cbz/...) found in {input_path}")
        sys.exit(1)
    return files
# ── Text extraction ───────────────────────────────────────────────────────────
def resolve_hyphens(text: str) -> str:
    """Join hyphenated line-breaks: 'imme-\ndiately' -> 'immediately'"""
    return re.sub(r"-\s*\n\s*(\S)", r"\1", text)
def extract_page_text(page_dict: dict) -> str:
    """
    Extract all text from a pdftext page dict as a single string.
    Iterates blocks → lines → spans, concatenating all text in reading order.
    """
    blocks = page_dict.get("blocks", [])
    if not blocks:
        return ""
    words = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t and t != "\n":
                    words.append(t)
                sz = span.get("font", {}).get("size", 0)
    text = " ".join(words)
    text = resolve_hyphens(text)
    return text.strip()
# ── Image extraction ──────────────────────────────────────────────────────────
def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
def extract_page_images(pdf_path: Path, page_num: int, page_width: float, page_height: float, img_dir: Optional[Path] = None, book_title: str = "", keep_full_page: bool = False) -> list:
    """
    Extract embedded images from a PDF page using pymupdf.
    Handles stencils (monochrome images) and saves to img_dir.

    keep_full_page: for books whose PDF is a raster of pages (DJVU / comics
    converted upstream) the ORIGINAL source is not a renderable PDF sibling,
    so the full-page image IS the page render the reader's split view needs —
    it is kept (as the single largest image) instead of skipped.
    """
    try:
        import fitz
    except ImportError:
        return []
    records = []
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            doc.close()
            return []
        log(f"    page {page_num}: checking for images")
        raw_images = list(page.get_images(full=True))
        log(f"    page {page_num}: found {len(raw_images)} embedded image(s)")
        # Resolve each embedded image to (img, xref, rect, coverage) first
        infos = []
        for img in raw_images:
            xref = img[0]
            smask = img[1]
            # Handle stencils / monochrome: xref==0 means real xref is in smask
            if xref == 0:
                xref = smask
            if xref == 0:
                log(f"    page {page_num}: no valid xref, skipping")
                continue
            # Get placement rects — try both xref values
            rects = page.get_image_rects(xref)
            try:
                rects2 = page.get_image_rects(img[0])
                if rects2:
                    rects = rects2
            except Exception:
                pass
            if not rects:
                log(f"    page {page_num}: no placement rects for xref {xref}")
                if img[0] == 0:
                    continue
            rect = rects[0]
            coverage = (rect.width * rect.height) / page_area
            infos.append((img, xref, rect, coverage))
        # Full-page raster: keep the largest image (it IS the page) and drop
        # the rest; for normal PDFs the PDF sibling is rendered on demand.
        full_xref = None
        if keep_full_page:
            big = [i for i in infos if i[3] >= 0.8]
            if big:
                full_xref = max(big, key=lambda i: i[3])[1]
                log(f"    page {page_num}: full-page image xref {full_xref} kept (page render)")
        seen_xrefs = set()
        for img, xref, rect, coverage in infos:
            smask = img[1]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            if full_xref is not None and xref != full_xref:
                continue
            if coverage >= 0.8:
                if full_xref is None:
                    log(f"    page {page_num}: xref {xref} coverage {coverage:.0%}, skipping as full-page scan")
                    continue
            # Extract — handle both regular images and stencils
            extract_xref = xref if img[0] != 0 else smask
            if extract_xref == 0:
                extract_xref = xref
            try:
                base_img = doc.extract_image(extract_xref)
                raw = base_img["image"]
            except Exception as e:
                log(f"    page {page_num}: extract_image({extract_xref}) failed: {e}")
                continue
            safe_book = re.sub(r"[^\w]", "", book_title)
            is_full = (full_xref is not None and xref == full_xref)
            # Full-page color rasters are much smaller as JPEG; small
            # figures stay PNG.
            ext = "jpg" if is_full else "png"
            name = f"{safe_book}_p{page_num + 1}_img_{xref}.{ext}"
            try:
                pil = Image.open(io.BytesIO(raw))
            except Exception as e:
                log(f"    page {page_num}: PIL open failed: {e}")
                continue
            if img_dir is not None:
                save_path = img_dir / name
                if is_full and pil.mode not in ("RGB", "L"):
                    pil = pil.convert("RGB")
                pil.save(str(save_path), format="JPEG" if is_full else "PNG",
                         quality=85, optimize=True)
                fsize = save_path.stat().st_size
                log(f"    page {page_num}: saved {name} ({fsize / 1024:.0f} KB)")
                file_path = str(save_path)
            else:
                file_path = ""
            records.append({
                "bbox": [float(rect.x0), float(rect.y0),
                         float(rect.x1), float(rect.y1)],
                "path": file_path,
                "name": name,
            })
        if not records:
            log(f"    page {page_num}: no images extracted")
        else:
            log(f"    page {page_num}: extracted {len(records)} image(s)")
        doc.close()
    except Exception as e:
        log(f"    page {page_num}: error: {e}")
    return records
# ── Core conversion ───────────────────────────────────────────────────────────

# Helper to check if convert_single_pdf supports do_table_recognition
_CONVERT_SINGE_PDF_SIGNATURE = None

def _supports_table_recognition_arg():
    global _CONVERT_SINGE_PDF_SIGNATURE
    if _CONVERT_SINGE_PDF_SIGNATURE is None:
        from marker.convert import convert_single_pdf
        sig = inspect.signature(convert_single_pdf)
        _CONVERT_SINGE_PDF_SIGNATURE = "do_table_recognition" in sig.parameters
    return _CONVERT_SINGE_PDF_SIGNATURE

# ── Per-page OCR text from marker ────────────────────────────────────────────
# convert_single_pdf() flattens marker's fully-processed pages (embedded text
# + OCR + layout + reading order + tables + equations) into ONE markdown
# string, which loses page boundaries. merge_lines() is the last step that
# still carries a pnum per block, so we patch it (in marker.convert's
# namespace) to record the flat block list as it is produced, and build the
# per-page `text` column from that. Without this, `text` would come only from
# the PDF's embedded text layer — which is empty for scanned books, exactly
# the case marker exists for (DJVU / comics / old scans came back blank).
_MARKER_CAPTURE: list = []

def _install_marker_capture():
    """Idempotent. Safe to call every run; no-op if marker is missing."""
    try:
        import marker.convert as mc
    except ImportError:
        return
    if getattr(mc, "_ddc_page_capture", False):
        return
    try:
        _orig_merge_lines = mc.merge_lines
    except AttributeError:
        log("  WARNING: marker.convert.merge_lines not found — per-page OCR "
            "text unavailable (marker upgrade?); text falls back to the "
            "embedded layer")
        return
    def _capturing_merge_lines(blocks):
        out = _orig_merge_lines(blocks)
        _MARKER_CAPTURE.append(out)
        return out
    mc.merge_lines = _capturing_merge_lines
    mc._ddc_page_capture = True

def _captured_page_texts() -> dict:
    """{chunk-relative pnum: text} from the last convert_single_pdf() call.
    Figure/Image blocks (whose 'text' is a ![...](...) placeholder) are
    dropped, and stray image placeholders are stripped from the rest."""
    if not _MARKER_CAPTURE:
        return {}
    img_pat = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    texts: dict = {}
    for b in _MARKER_CAPTURE[-1]:
        if b.block_type in ("Figure", "Image"):
            continue
        t = img_pat.sub(" ", b.text or "")
        if t.strip():
            texts.setdefault(b.pnum, []).append(t.strip())
    return {p: resolve_hyphens("\n".join(ts)).strip() for p, ts in texts.items()}

def convert_chunk_to_pages(
    pdf_path: Path,
    model_lst,
    start_page: int,
    chunk_size: int,
    batch_multiplier: int,
    img_dir: Optional[Path] = None,
    book_title: str = "",
    do_table_recognition: bool = True,
    keep_full_page: bool = False,
) -> Optional[list]:
    from marker.convert import convert_single_pdf
    import pdftext.extraction as pe
    
    # Build kwargs dynamically to avoid version errors
    kwargs = {
        "max_pages": chunk_size,
        "start_page": start_page,
        "batch_multiplier": batch_multiplier,
        "langs": ["English"],
    }
    
    # Only pass do_table_recognition if the installed marker version supports it
    if _supports_table_recognition_arg():
        kwargs["do_table_recognition"] = do_table_recognition
    
    _MARKER_CAPTURE.clear()
    try:
        _markdown, _images, meta = convert_single_pdf(
            str(pdf_path),
            model_lst,
            **kwargs
        )
    except Exception as e:
        log(f"    marker error (tables={do_table_recognition}): {e}")
        return None
    page_texts = _captured_page_texts()
        
    page_range = list(range(start_page, start_page + chunk_size))
    try:
        raw_pages = pe.dictionary_output(str(pdf_path), page_range=page_range, workers=1)
    except Exception as e:
        log(f"    pdftext error: {e}")
        raw_pages = []
        
    raw_lookup = {start_page + i: rp for i, rp in enumerate(raw_pages)}
    actual_count = meta.get("pages", chunk_size)
    
    pages = []
    for i in range(actual_count):
        abs_page = start_page + i
        raw_page = raw_lookup.get(abs_page, {})
        # marker's processed per-page text (embedded layer + OCR, reading
        # order, tables) — falls back to the raw embedded layer when the
        # capture is unavailable (older marker or empty page).
        page_text   = page_texts.get(i) or extract_page_text(raw_page)
        
        img_records = extract_page_images(
            pdf_path, abs_page,
            float(raw_page.get("width", 0)),
            float(raw_page.get("height", 0)),
            img_dir=img_dir,
            book_title=book_title,
            keep_full_page=keep_full_page,
        )
        
        if img_records:
            img_bboxes = [r["bbox"] for r in img_records]
            img_paths  = [r["path"] for r in img_records]
            img_names  = [r["name"] for r in img_records]
        else:
            img_bboxes = []; img_paths = []; img_names = []
            
        pages.append({
            "page_num":      abs_page + 1,
            "page_width":    float(raw_page.get("width",  0)),
            "page_height":   float(raw_page.get("height",0)),
            "text":          page_text,
            "image_bbox":    img_bboxes,
            "image_path":    img_paths,
            "image_name":    img_names,
        })
        
    return pages if pages else None

def pages_to_arrow_table(pages: list, source_file: str, book_title: str) -> pa.Table:
    cols = ["source_file","book_title","page_num","page_width","page_height",
            "text","image_bbox","image_path","image_name"]
    rows = {k: [] for k in cols}
    for page in pages:
        rows["source_file"].append(source_file)
        rows["book_title"].append(book_title)
        rows["page_num"].append(page["page_num"])
        rows["page_width"].append(page["page_width"])
        rows["page_height"].append(page["page_height"])
        rows["text"].append(page["text"])
        rows["image_bbox"].append(page["image_bbox"])
        rows["image_path"].append(page["image_path"])
        rows["image_name"].append(page["image_name"])
    return pa.Table.from_pydict(rows, schema=PAGE_SCHEMA)
# ── HTML reconstruction reference ─────────────────────────────────────────────
def build_page_html(page_row: dict) -> str:
    """
    Reconstruct one page as positioned HTML.
    page_row is a dict matching the parquet schema (call table.to_pylist()[n]).
    """
    import base64
    w = page_row.get("page_width")  or 800
    h = page_row.get("page_height") or 1100
    DISPLAY_W = 800
    scale     = DISPLAY_W / w if w else 1.0
    display_h = int(h * scale)
    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  .page {{ position:relative; width:{DISPLAY_W}px; height:{display_h}px;
            background:white; border:1px solid #ccc; font-family:serif; overflow:hidden; }}
  .blk  {{ position:absolute; overflow:hidden; line-height:1.3; }}
  .head {{ font-weight:bold; }}
  .img  {{ }}
  .img img {{ max-width:100%; max-height:100%; }}
</style></head><body><div class="page">"""]
    para_texts  = page_row.get("paragraph_text", [])
    para_bboxes = page_row.get("paragraph_bbox", [])
    para_fsize  = page_row.get("paragraph_font_size", [])
    para_heading = page_row.get("paragraph_is_heading", [])
    para_order  = page_row.get("paragraph_reading_order", [])
    for idx in range(len(para_texts)):
        x1,y1,x2,y2 = [v*scale for v in para_bboxes[idx]]
        bw = max(x2-x1, 10)
        bh = max(y2-y1, 10)
        fs = para_fsize[idx] * scale
        cls = "blk head" if para_heading[idx] else "blk"
        txt = para_texts[idx].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts.append(
            f'<div class="{cls}" style="left:{x1:.1f}px;top:{y1:.1f}px;'
            f'width:{bw:.1f}px;height:{bh:.1f}px;font-size:{fs:.1f}px;">{txt}</div>'
        )
    img_bboxes = page_row.get("image_bbox", [])
    img_paths  = page_row.get("image_path", [])
    img_names  = page_row.get("image_name", [])
    for idx in range(len(img_paths)):
        img_path = img_paths[idx]
        if img_path and Path(img_path).exists():
            raw = open(img_path, "rb").read()
            b64 = base64.b64encode(raw).decode()
        else:
            b64 = ""
        x1,y1,x2,y2 = [v*scale for v in img_bboxes[idx]]
        if any(v > 0 for v in [x1,y1,x2,y2]):
            style = f"left:{x1:.1f}px;top:{y1:.1f}px;width:{max(x2-x1,10):.1f}px;height:{max(y2-y1,10):.1f}px;"
        else:
            style = "position:relative;display:block;max-width:100%;margin:4px 0;"
        parts.append(
            f'<div class="blk img" style="{style}">'
            f'<img src="data:image/png;base64,{b64}" alt="{img_names[idx]}"></div>'
        )
    parts.append("</div></body></html>")
    return "\n".join(parts)
# ── Per-book conversion ───────────────────────────────────────────────────────
def convert_pdf_to_parquet(
    pdf_path: Path,
    model_lst,
    output_dir: Path,
    input_root: Path,
    chunk_size: int,
    batch_multiplier: int,
    resume_chunk: int = 0,
    on_chunk=None,
    source_file: str | None = None,
    original_path: Path | None = None,
) -> bool:
    try:
        rel = pdf_path.relative_to(input_root)
    except ValueError:
        rel = Path(pdf_path.name)
        
    out_subdir = output_dir / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path   = out_subdir / (pdf_path.stem + ".parquet")
    img_dir    = out_subdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    source_file = source_file or str(pdf_path)
    book_title  = pdf_path.stem.replace("_", " ").replace("-", " ")
    
    total_pages = get_page_count(pdf_path)
    log(f"  → {out_path}")
    log(f"  Pages (estimated): {total_pages}")
    
    writer        = None
    failed_chunks = []
    chunk_start   = resume_chunk * chunk_size
    chunk_num     = resume_chunk
    
    # Check if we can disable tables via argument
    can_disable_tables = _supports_table_recognition_arg()

    try:
        while chunk_start < total_pages:
            log(f"  Chunk {chunk_num}: pages {chunk_start}–{chunk_start + chunk_size - 1}")
            t0    = time.time()
            pages = None
            
            # --- ATTEMPT 1: Standard (Tables ON) ---
            try:
                pages = convert_chunk_to_pages(
                    pdf_path, model_lst, chunk_start, chunk_size, batch_multiplier, 
                    img_dir=img_dir, book_title=book_title, do_table_recognition=True,
                    keep_full_page=original_path is not None)
            except torch.cuda.OutOfMemoryError:
                log(f"  OOM — retrying chunk {chunk_num} at batch_multiplier=1 with tables")
                torch.cuda.empty_cache(); gc.collect()
                try:
                    pages = convert_chunk_to_pages(
                        pdf_path, model_lst, chunk_start, chunk_size, 1, 
                        img_dir=img_dir, book_title=book_title, do_table_recognition=True,
                        keep_full_page=original_path is not None)
                except Exception as e2:
                    log(f"  Retry 1 failed: {e2}")
                    pages = None
            except Exception as e:
                log(f"  Chunk {chunk_num} error (tables): {e}")
                pages = None

            # --- ATTEMPT 2: Fallback (Tables OFF if supported, else lower batch) ---
            if pages is None:
                if can_disable_tables:
                    log(f"  ⚠️  Table recognition failed/hung. Retrying without table recognition...")
                    torch.cuda.empty_cache(); gc.collect()
                    try:
                        # Try with Tables OFF and Batch Multiplier 1 for extra safety
                        pages = convert_chunk_to_pages(
                            pdf_path, model_lst, chunk_start, chunk_size, 1,
                            img_dir=img_dir, book_title=book_title, do_table_recognition=False,
                            keep_full_page=original_path is not None)
                    except Exception as e3:
                        log(f"  ⚠️  Retry 2 failed (no tables, batch 1): {e3}")
                        pages = None
                else:
                    # If older marker version, just try with lower batch multiplier again
                    log(f"  ⚠️  Retry with lower batch (batch_multiplier=1)...")
                    torch.cuda.empty_cache(); gc.collect()
                    try:
                        pages = convert_chunk_to_pages(
                            pdf_path, model_lst, chunk_start, chunk_size, 1,
                            img_dir=img_dir, book_title=book_title, do_table_recognition=True,
                            keep_full_page=original_path is not None)
                    except Exception as e4:
                        log(f"  ⚠️  Retry 2 failed (low batch): {e4}")
                        pages = None

            # --- WRITING LOGIC ---
            if pages:
                table = pages_to_arrow_table(pages, source_file, book_title)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, PAGE_SCHEMA)
                writer.write_table(table)
                elapsed = time.time() - t0
                log(f"  Chunk {chunk_num}: {len(pages)} pages written in {elapsed:.1f}s")
                
                if len(pages) < chunk_size:
                    log("  Reached end of document.")
                    if on_chunk:
                        on_chunk(chunk_num)
                    break
            else:
                log(f"  Chunk {chunk_num}: no pages extracted — skipping")
                failed_chunks.append(chunk_num)
                
            if on_chunk:
                on_chunk(chunk_num)
            torch.cuda.empty_cache(); gc.collect()
            chunk_start += chunk_size
            chunk_num   += 1
            
    finally:
        if writer:
            writer.close()
            
    if not out_path.exists() or out_path.stat().st_size == 0:
        log(f"  WARNING: Output file missing or empty")
        return False
        
    log(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Copy the original source next to the parquet so the output dir is
    # self-contained for the rename/DDC step, even if the source dir
    # moves or is re-pointed later. (For a converted book — e.g. DJVU
    # rasterized to PDF — that is the ORIGINAL file, .djvu, not the
    # temporary PDF.)
    orig = Path(original_path) if original_path else pdf_path
    pdf_copy = out_subdir / (orig.stem + orig.suffix.lower())
    src_size = orig.stat().st_size
    if src_size > 0 and (not pdf_copy.exists() or pdf_copy.stat().st_size != src_size):
        shutil.copy2(orig, pdf_copy)
        log(f"  Copied source: {pdf_copy.name} ({src_size / 1e6:.1f} MB)")

    if failed_chunks:
        log(f"  WARNING: {len(failed_chunks)} failed chunks: {failed_chunks}")
        return False
        
    return True
# ── Main ──────────────────────────────────────────────────────────────────────
def parse_shard(spec: str | None) -> tuple[int, int]:
    """'I/N' (e.g. '0/2', '1/2') -> (i, N). None/'0/1' -> (0, 1)."""
    if not spec:
        return 0, 1
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", spec)
    if not m:
        raise SystemExit(f"ERROR: --shard must be I/N (e.g. 0/2): {spec!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or i >= n:
        raise SystemExit(f"ERROR: bad --shard {spec!r} (need 0 <= I < N, N >= 1)")
    return i, n

def main():
    parser = argparse.ArgumentParser(description="PDF → Parquet pipeline for LLM/RAG")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--device", "-d", default=DEFAULT_DEVICE)
    parser.add_argument("--chunk",  "-c", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--batch",  "-b", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--shard", "-s", default=None, metavar="I/N",
                        help="run one worker of a multi-GPU job: I/N (e.g. 0/2, "
                             "1/2). Splits the PDF list round-robin across the N "
                             "workers (deterministic); each worker uses its own "
                             "checkpoint shard (.g<i>) and progress file (_g<i>). "
                             "Combine with CUDA_VISIBLE_DEVICES to pin each worker "
                             "to its own GPU.")
    parser.add_argument("--job", default=None, metavar="ID",
                        help="job id shared by all workers of one multi-GPU run "
                             "(the GUI generates one). Recorded in the progress "
                             "file so the dashboard merges exactly this job's "
                             "shards into one card — even shards that finish early.")
    parser.add_argument("--words-per-page", "-w", type=int, default=300,
                        help="nominal words per page for text-based books "
                             "(txt/epub, which have no physical pages). Pages "
                             "accumulate whole paragraphs up to this count and "
                             "break at the next paragraph boundary (default 300 "
                             "= one printed book page). PDFs ignore this.")
    args = parser.parse_args()
    shard_i, shard_n = parse_shard(args.shard)
    job_id = str(args.job).strip() or None
    input_path  = Path(args.input).resolve()
    output_dir  = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / (
        CHECKPOINT_FILE if shard_n == 1 else f"{CHECKPOINT_FILE}.g{shard_i}")
    if not input_path.exists():
        log(f"ERROR: {input_path} does not exist"); sys.exit(1)
    input_root = input_path if input_path.is_dir() else input_path.parent
    from booktext import detect_format, estimate_pages, \
        convert_textbook_to_parquet
    books      = get_book_list(input_path)
    all_books  = books
    if shard_n > 1:
        # Round-robin by sorted position: deterministic across restarts and
        # a decent spread (file sizes vary, but any worker can pick up any
        # leftover file on a later resume).
        books = [p for k, p in enumerate(all_books) if k % shard_n == shard_i]
        log(f"Shard       : worker {shard_i}/{shard_n} -> {len(books)} of "
            f"{len(all_books)} books")
    # Resume merges the base checkpoint with every shard file, so files
    # completed by a sibling worker (or an earlier plain run) are skipped.
    state      = (load_shard_checkpoints(output_dir) if args.resume
                  else {"completed": [], "failed": [], "partial": {}})
    # Per-file format (autodetect once; reused for dispatch + page pre-scan)
    from converters import FMT_TO_CONVERTER, _TARGET_FMT, \
        convert as conv_file
    fmt_of: dict[str, str] = {}
    for p in books:
        fmt_of[str(p)] = detect_format(p)
        if fmt_of[str(p)] == "unknown":
            log(f"  WARN: unrecognized format, will try as pdf: {p.name}")
            fmt_of[str(p)] = "pdf"
    def eff_fmt(p: Path) -> str:
        """Format after any external conversion (djvu/cbz/cbr -> pdf,
        mobi/azw3 -> epub, doc/odt -> docx). Used to decide GPU model
        loading and the dispatch path."""
        f = fmt_of.get(str(p), "pdf")
        conv = FMT_TO_CONVERTER.get(f)
        return _TARGET_FMT.get(conv, f) if conv else f
    _counts = {}
    for f in fmt_of.values():
        _counts[f] = _counts.get(f, 0) + 1
    log(f"Books found : {len(books)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(_counts.items()))})")
    log(f"Output dir  : {output_dir}")

    # ── live progress for the dashboard (page-level, for a real ETA) ──
    # Per-file page counts up front (pypdf/fitz, no GPU): exact total
    # workload, and "pages left" at any point.
    # Sharded workers report on their own file (ocr_<out>_g<i>.json); the
    # dashboard groups by base name and sums the shards into one card.
    progress_path = progress_path_for(output_dir,
                                      (shard_i, shard_n) if shard_n > 1 else None)
    # Which physical GPU is this worker pinned to? (The GUI pins each
    # worker via CUDA_VISIBLE_DEVICES; used to label the dashboard.)
    _cvis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_index = int(_cvis) if re.fullmatch(r"\d+", _cvis.strip()) else None
    file_page_count = {}
    for p in books:
        try:
            if p.stat().st_size > 0:
                fmt = fmt_of.get(str(p), "pdf")
                if fmt == "pdf":
                    file_page_count[str(p)] = get_page_count(p)
                else:
                    # nominal pages for text/epub (fast CPU scan, no GPU)
                    file_page_count[str(p)] = estimate_pages(
                        p, fmt, args.words_per_page)
        except Exception:
            file_page_count[str(p)] = 0
    total_pages_all = sum(file_page_count.values())
    # Files already converted (resume) count as done from the start.
    pages_done_base = sum(file_page_count.get(k, 0) for k in state["completed"])
    ocr_samples: list = []
    def done_files_count() -> int:
        # SHARD-LOCAL count: state["completed"] is merged across shards on
        # resume (and each shard re-saves the merged state), so len() would
        # count sibling workers' files too.
        return sum(1 for p in books if str(p) in state["completed"])
    prog = {
        "process": "ocr",
        "input": str(input_path), "output": str(output_dir),
        "state": "running",
        "shard": [shard_i, shard_n] if shard_n > 1 else None,
        "job": job_id,
        "gpu": gpu_index,
        "total_files": len(books),
        "done_files": done_files_count(),
        "failed_files": sum(1 for p in books if str(p) in state["failed"]),
        "total_pages": total_pages_all,
        "pages_done": pages_done_base,
        "current_file": None, "current_file_pages": 0,
        "current_file_pages_done": 0,
        "samples": ocr_samples,
    }
    def emit():
        save_progress(progress_path, prog)
    def finish(st: str):
        prog["state"] = st
        prog["current_file"] = None
        emit()
    emit()
    try:
        _idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        _gname = torch.cuda.get_device_name(_idx)
    except Exception:
        _gname = ""
    log(f"Device      : {args.device}" + (f" ({_gname})" if _gname else ""))
    log(f"Chunk size  : {args.chunk} pages")
    log(f"Batch mult  : {args.batch}")
    log(f"Words/page  : {args.words_per_page} (text-based nominal pages)")
    log(f"Resume      : {args.resume}")

    # Marker models are only needed for PDFs. A text/epub-only job (or a
    # sharded worker that got no PDFs) converts on CPU without loading them.
    model_lst = None
    def load_models():
        nonlocal model_lst
        from marker.models import load_all_models
        log("Loading models (once for entire batch)...")
        _install_marker_capture()
        torch.cuda.empty_cache(); gc.collect()
        model_lst = load_all_models(device=args.device, dtype=DEFAULT_DTYPE)
        dev_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        try:
            free, total = torch.cuda.mem_get_info(dev_idx)
            log(f"VRAM after model load: {free/1e9:.1f}GB free of {total/1e9:.1f}GB")
        except Exception:
            pass
        return model_lst
    if not any(eff_fmt(p) == "pdf" for p in books):
        log("No PDFs (or PDF-converted books) in this batch — "
            "GPU models not loaded.")
    
    completed = skipped = failed = 0
    for i, book_path in enumerate(books):
        key = str(book_path)
        if key in state["completed"]:
            log(f"[{i+1}/{len(books)}] SKIP (done): {book_path.name}")
            skipped += 1
            prog["done_files"] = done_files_count()
            emit()
            continue
        if book_path.stat().st_size == 0:
            log(f"[{i+1}/{len(books)}] SKIP (empty file): {book_path.name}")
            skipped += 1
            emit()
            continue
        fmt = fmt_of.get(key, "pdf")
        resume_chunk = state["partial"].get(key, 0)
        label = f"RESUME chunk {resume_chunk}" if resume_chunk else "Processing"
        log(f"[{i+1}/{len(books)}] {label} ({fmt}): {book_path.name}")
        t0      = time.time()
        success = False

        fpages = file_page_count.get(key, 0)
        prog["current_file"] = book_path.name
        prog["current_file_pages"] = fpages
        prog["current_file_pages_done"] = min(resume_chunk * args.chunk, fpages)
        prog["pages_done"] = pages_done_base + prog["current_file_pages_done"]
        emit()

        def on_chunk(n, _key=key):
            # Checkpoint after every chunk so a crash costs at most one chunk
            state["partial"][_key] = n + 1
            save_checkpoint(checkpoint_path, state)
            # live progress (chunk granularity: a chunk takes 30–90 s)
            prog["current_file_pages_done"] = min((n + 1) * args.chunk, fpages)
            prog["pages_done"] = pages_done_base + prog["current_file_pages_done"]
            ocr_samples.append([time.time(), prog["pages_done"]])
            del ocr_samples[:-64]
            emit()

        try:
            # External conversion first (Kindle / DOC / ODT / DJVU / CBZ /
            # CBR): the result lands in output_dir/_conv/<rel>/<stem>/ and
            # is ingested like its native target (pdf -> marker OCR,
            # epub/docx -> text path). The ORIGINAL file is what gets
            # recorded in source_file and copied into the tree later. The
            # workdir is kept on failure so a resume can reuse the
            # conversion; removed once the book is done.
            conv_path, conv_wd = None, None
            efmt = eff_fmt(book_path)
            if efmt != fmt:
                try:
                    rel = book_path.relative_to(input_root)
                except ValueError:
                    rel = Path(book_path.name)
                conv_wd = output_dir / "_conv" / rel.parent / book_path.stem
                log(f"  Converting {fmt} -> {efmt} (external tool) ...")
                conv_path, _ = conv_file(book_path, conv_wd, fmt=fmt, log=log)
                log(f"  Converted: {conv_path.name}")
            target = conv_path or book_path

            if efmt == "pdf":
                if model_lst is None:
                    load_models()
                success = convert_pdf_to_parquet(
                    pdf_path=target, model_lst=model_lst,
                    output_dir=output_dir, input_root=input_root,
                    chunk_size=args.chunk, batch_multiplier=args.batch,
                    resume_chunk=resume_chunk, on_chunk=on_chunk,
                    source_file=None if conv_path is None else key,
                    original_path=None if conv_path is None else book_path,
                )
            else:
                # Text-based formats (txt/epub/rtf/docx/html/mht/fb2/chm
                # + converted ones): CPU extraction, nominal pages, no
                # chunks (one atomic unit; a crash just re-does the whole
                # file, which takes seconds).
                success = convert_textbook_to_parquet(
                    target, output_dir, input_root,
                    words_per_page=args.words_per_page, log=log,
                    original_path=None if conv_path is None else book_path,
                )
                if success:
                    on_chunk(0)
            if success and conv_wd is not None:
                shutil.rmtree(conv_wd, ignore_errors=True)
        except Exception as e:
            log(f"  FATAL: {e}"); traceback.print_exc()
        if success:
            state["completed"].append(key)
            state["partial"].pop(key, None)
            if key in state["failed"]:
                state["failed"].remove(key)
            completed += 1
            pages_done_base += fpages
            log(f"  Done in {time.time()-t0:.0f}s")
        else:
            state["failed"].append(key)
            failed += 1
            log(f"  Failed after {time.time()-t0:.0f}s")
        save_checkpoint(checkpoint_path, state)
        prog["done_files"] = done_files_count()
        prog["failed_files"] = sum(1 for p in books if str(p) in state["failed"])
        ocr_samples.append([time.time(), prog["pages_done"]])
        del ocr_samples[:-64]
        emit()
        torch.cuda.empty_cache(); gc.collect()
    log("=" * 60)
    log(f"Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    if state["failed"]:
        log("Failed files:")
        for f in state["failed"]: log(f"  {f}")
    finish("done")
    if model_lst is not None:
        del model_lst
    torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()

