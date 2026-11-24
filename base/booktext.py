#!/usr/bin/env python3
"""
booktext.py — multi-format text extraction for the DDC pipeline.
=================================================================
Converts text-based books into the same per-page Parquet schema that
ocr.py produces for PDFs, so every downstream stage (enrich ->
classify/rename -> validate -> index) works unchanged.

Natively handled here: TXT, EPUB/OEPUB, RTF, DOCX, HTML/MHT, FB2, CHM.
Formats that need a system tool first (Kindle, DOC/ODT, DJVU, CBZ/CBR)
are converted by converters.py into one of these (or into a PDF for the
marker OCR path) before landing here.

Each row = one NOMINAL page:
  - a page is ~--words-per-page words (default 300, the nominal count of
    a printed book page), accumulated paragraph by paragraph
  - breaks only happen BETWEEN paragraphs (paragraph-aware), never inside
    one, unless a single paragraph alone exceeds 1.5x the target — then it
    is split at sentence boundaries (word boundaries as a last resort)
  - images (EPUB figures/cover, DOCX/HTML figures) ride along on the page
    where they appear in reading order and are saved next to the parquet
    exactly like PDF images: <images>/<book>_p<page>_img_<n>.<ext>

Format autodetection: magic bytes first (%PDF, EPUB zip mimetype, OLE,
{\\rtf, ITSF, DJVU, MOBI, <html...), then extension, then a text sniff.
ocr.py calls detect_format() per file, so a file called "book.pdf" that
is actually an EPUB (or an extensionless file) still gets the right
extractor.

Dependencies: stdlib + lxml + ebooklib + PIL + pyarrow (native), plus
striprtf / mammoth / pychm for rtf / docx / chm (lazy imports).
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Same schema ocr.py writes (import it so the two can never drift).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from ocr import PAGE_SCHEMA  # noqa: E402  (ocr.py imports torch at top)
except Exception:  # torch unavailable -> identical fallback schema
    PAGE_SCHEMA = pa.schema([
        pa.field("source_file",            pa.string()),
        pa.field("book_title",             pa.string()),
        pa.field("page_num",               pa.int32()),
        pa.field("page_width",             pa.float32()),
        pa.field("page_height",            pa.float32()),
        pa.field("text",                   pa.string()),
        pa.field("image_bbox",             pa.list_(pa.list_(pa.float32()))),
        pa.field("image_path",             pa.list_(pa.string())),
        pa.field("image_name",             pa.list_(pa.string())),
    ])

SUPPORTED_FORMATS = ("pdf", "epub", "txt", "rtf", "docx", "html", "mht",
                     "fb2", "chm")
# Formats this module extracts natively (ocr.py dispatches on these).
TEXT_FORMATS = ("txt", "epub", "rtf", "docx", "html", "mht", "fb2", "chm")
EXT_TO_FORMAT = {
    ".pdf": "pdf", ".epub": "epub", ".opub": "epub",
    ".txt": "txt", ".text": "txt", ".textbook": "txt",
    ".rtf": "rtf",
    ".docx": "docx",
    ".html": "html", ".htm": "html",
    ".mht": "mht", ".mhtml": "mht",
    ".fb2": "fb2",
    ".chm": "chm",
    ".ods": "odt", ".odp": "odt",  # office family, converted like odt
}
# Formats that need converters.py (external tool) before extraction.
CONVERTIBLE_FORMATS = ("mobi", "azw3", "doc", "odt", "djvu", "cbz", "cbr")

# Nominal page geometry for text pages (display/HTML reconstruction uses
# these; a real PDF keeps its own dimensions).
NOMINAL_PAGE_W = 800.0
NOMINAL_PAGE_H = 1100.0
# An inline image counts this many words of page budget.
IMAGE_WORDS = 25

_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"
_EPUB_MIMETYPE = b"application/epub+zip"
_AZW3_MIMETYPE = b"application/vnd.amazon.ebook"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # .doc / old Office
_RTF_MAGIC = b"{\\rtf"
_CHM_MAGIC = b"ITSF"
_MOBI_MAGIC = b"MOBI"  # signature; checked at offsets 8/12/78 and BOOK-pair@60


# ── Format detection ──────────────────────────────────────────────────────────
def detect_format(path: Path) -> str:
    """Format string for one file, or 'unknown'.

    Native:   pdf | epub | txt | rtf | docx | html | mht | fb2 | chm
    Convert:  mobi | azw3 | doc | odt | djvu | cbz | cbr  (see
              converters.py — these need an external tool first)

    Order: magic bytes -> extension -> text sniff. Magic wins over
    extension so mislabelled files still get the right extractor."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return "unknown"
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return "unknown"

    fmt = _magic_format(head, path)
    if fmt:
        return fmt
    fmt = EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt:
        return fmt
    # Convertible extensions not covered by a magic (kfx/lit/cbz/cbr by
    # name only — their mags are tool-specific or absent).
    ext = path.suffix.lower()
    # Kindle family: many library exports carry non-standard prefixes or
    # mislabels (this collection's .azw3 files are actually MOBI), so the
    # extension is a trusted fallback — ebook-convert reads them all.
    if ext in (".kfx", ".azw", ".mobi", ".azw3", ".azw4", ".lit"):
        return "mobi"
    if ext == ".cbr":
        return "cbr"
    if head.startswith(_ZIP_MAGIC):
        return "unknown"  # zip but no epub/azw3/odf/docx marker
    if _looks_like_text(head):
        return "txt"
    return "unknown"


def _magic_format(head: bytes, path: Path) -> str:
    """Magic-byte sniff. Returns a format or '' (undecided)."""
    if head.startswith(_PDF_MAGIC):
        return "pdf"
    if head.startswith(_OLE_MAGIC):
        return "doc"
    if head.startswith(_RTF_MAGIC):
        return "rtf"
    if head.startswith(_CHM_MAGIC):
        return "chm"
    if head[:5] == b"AT&TF" or head[:3] == b"DJV":
        return "djvu"
    # MOBI (PalmDOC): the "MOBI" signature sits at offset 12 in the
    # classic 16-byte header ("Book"@8), at 78 in the full PalmDB header,
    # and at 64 in the Kindle-library PDB variant (BOOK@60, 64-byte
    # custom prefix with the file title). Test the known positions rather
    # than one fixed offset.
    if (head[8:12] == _MOBI_MAGIC or head[12:16] == _MOBI_MAGIC
            or head[78:82] == _MOBI_MAGIC
            or (head[60:64] == b"BOOK" and head[64:68] == _MOBI_MAGIC)):
        return "mobi"
    if head.lstrip().startswith(b"MIME-Version:"):
        return "mht"
    low = head.lower()
    if b"<html" in low[:512] or b"<!doctype html" in low[:512]:
        return "html"
    if b"fictionbook" in low:
        return "fb2"
    if head.startswith(_ZIP_MAGIC):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                mt = b""
                if "mimetype" in names:
                    with z.open("mimetype") as m:
                        mt = m.read(64).strip()
                if mt == _EPUB_MIMETYPE:
                    return "epub"
                if mt == _AZW3_MIMETYPE:
                    return "azw3"
                if "word/document.xml" in names:
                    return "docx"
                if "content.xml" in names:
                    return "odt"
                if any(n.lower().endswith((".png", ".jpg", ".jpeg"))
                       for n in names[:20]):
                    return "cbz"
        except Exception:
            pass  # some zip, not one we know — fall through
    return ""


def _looks_like_text(sample: bytes) -> bool:
    """Heuristic: decodable and almost no control bytes -> text."""
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            sample.decode("latin-1")
        except Exception:
            return False
    controls = sum(1 for b in sample if b < 9 or (13 < b < 32) or b == 127)
    return controls <= max(1, len(sample) // 16)


# ── Text reading ──────────────────────────────────────────────────────────────
def read_text(path: Path) -> str:
    """Read a .txt (or any text file) with encoding fallbacks.

    Order: utf-8-sig (handles the BOM most exports carry) -> utf-16
    (some Windows exports) -> latin-1 (never fails)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("latin-1", errors="replace")


# ── TXT -> blocks ─────────────────────────────────────────────────────────────
def text_blocks(text: str) -> list[dict]:
    """Split raw text into paragraph blocks (a block = one dict).

    A paragraph is a block separated from the next by at least one blank
    line (the universal TXT convention). Inside a paragraph, hard line
    breaks become spaces (prose layout); all whitespace is normalized."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff").strip()
    blocks = []
    for para in re.split(r"\n[ \t]*\n+", text):
        p = " ".join(para.split())
        if p:
            blocks.append({"kind": "text", "text": p})
    return blocks


# ── EPUB -> blocks (+metadata) ────────────────────────────────────────────────
# Block tags: a paragraph-level element whose text we take wholesale.
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
               "blockquote", "pre", "figcaption", "dt", "dd"}
_SKIP_TAGS = {"script", "style", "nav", "head", "title", "meta", "link",
              "template", "noscript"}


def epub_blocks(path: Path) -> tuple[list[dict], dict]:
    """Walk an EPUB's spine in reading order -> text + image blocks.

    Returns (blocks, meta) where meta = {"title": str, "authors": str}
    from the package metadata (for logging; the pipeline's own metadata
    comes from the enrichment LLM)."""
    from ebooklib import epub as _epub
    import ebooklib as _ebooklib

    book = _epub.read_epub(str(path))
    meta = {"title": "", "authors": ""}
    try:
        t = book.get_metadata("DC", "title")
        if t and t[0][0]:
            meta["title"] = str(t[0][0])
        a = book.get_metadata("DC", "creator")
        if a:
            meta["authors"] = ", ".join(str(x[0]) for x in a if x[0])
    except Exception:
        pass

    docs = {it.get_id(): it for it in book.get_items_of_type(_ebooklib.ITEM_DOCUMENT)}
    blocks: list[dict] = []

    for idref, _linear in (book.spine or []):
        item = docs.get(idref)
        if item is None:
            continue
        try:
            content = item.get_content()
        except Exception:
            continue
        try:
            from lxml import etree
            root = etree.fromstring(content)
        except Exception:
            # Fallback: crude tag strip (better than dropping the document)
            blocks.extend(text_blocks(_crude_strip(content)))
            continue
        _walk(root, book, item, blocks)

    return blocks, meta


def _crude_strip(html_bytes) -> str:
    html = html_bytes.decode("utf-8", errors="replace") if isinstance(html_bytes, bytes) \
        else str(html_bytes)
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", html)


def _is_nav_item(item) -> bool:
    """True for the TOC/nav document (properties='nav' in the manifest)."""
    try:
        props = item.get_property("properties") or ""
    except Exception:
        props = ""
    if "nav" in str(props).split():
        return True
    return re.search(r"(^|[_\-/])nav(\.|-)|(^|[_\-/])toc(\.|-)",
                     (item.get_name() or "").lower()) is not None


def _walk(root, book, item, blocks: list[dict]) -> None:
    """Depth-first in document order; emit text for the deepest block
    elements and images wherever they sit."""
    from lxml import etree

    if _is_nav_item(item):
        return

    def visit(el: "etree._Element") -> None:
        tag = etree.QName(el).localname if isinstance(
            getattr(el, "tag", None), str) else None
        if tag in _SKIP_TAGS:
            return
        if tag == "img":
            _emit_image(el, book, item, blocks)
            return
        if tag in _BLOCK_TAGS:
            has_block_child = any(
                isinstance(c.tag, str)
                and etree.QName(c).localname in _BLOCK_TAGS
                for c in el
            )
            if not has_block_child:
                # Deepest block element: take its full text (nested inline
                # markup comes along; nested images contribute no text).
                txt = " ".join("".join(el.itertext()).split())
                if txt:
                    blocks.append({"kind": "text", "text": txt})
            for c in el:
                visit(c)
            return
        for c in el:
            visit(c)

    for c in root:
        visit(c)


def _emit_image(el, book, item, blocks: list[dict]) -> None:
    """Resolve <img src> relative to this document inside the EPUB and
    record the image block (raw bytes + extension, no page yet)."""
    from lxml import etree
    src = el.get("src") or el.get("href") \
        or el.get("{http://www.w3.org/1999/xlink}href")
    if not src:
        return
    src = src.split("#")[0].split("?")[0].strip()
    if not src:
        return

    # Resolve relative to the document's own directory in the package:
    # doc "OEBPS/ch1.xhtml" + src "images/f.png" -> "OEBPS/images/f.png"
    doc_dir = ""
    try:
        doc_name = item.get_name() or ""
        if "/" in doc_name:
            doc_dir = doc_name.rsplit("/", 1)[0]
    except Exception:
        pass
    if src.startswith("/"):
        candidate = src.lstrip("/")
    elif doc_dir:
        candidate = _normalize_relpath(f"{doc_dir}/{src}")
    else:
        candidate = _normalize_relpath(src)

    img_item = None
    for it in book.get_items():
        if (it.get_name() or "") == candidate:
            img_item = it
            break
    if img_item is None:
        # Fallback: match by basename (loose packaging, case differences)
        base = candidate.rsplit("/", 1)[-1].lower()
        for it in book.get_items():
            n = (it.get_name() or "")
            if n.lower() == candidate.lower() \
                    or n.rsplit("/", 1)[-1].lower() == base:
                img_item = it
                break
    if img_item is None:
        return

    try:
        data = img_item.get_content()
    except Exception:
        return
    if not data:
        return
    ext = _image_ext(data, img_item.get_name() or "")
    if _too_small(data):  # bullets, rules — not a real figure
        return
    blocks.append({"kind": "image", "ext": ext, "data": data,
                   "ref": candidate})


def _normalize_relpath(p: str) -> str:
    parts = []
    for part in p.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _image_ext(data: bytes, name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".webp"):
        if ext == ".webp":
            return ".png"  # rename regex accepts png/jpg/gif/tiff
        return ext
    if data[:4] == b"\x89PNG":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return ".png"


def _too_small(data: bytes) -> bool:
    """True for tiny decorative bits (bullets, rules) — a real figure is
    meaningfully larger than 48px on its short side."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
        return min(w, h) < 48
    except Exception:
        return False


# ── HTML -> blocks (shared by html / mht / docx / chm) ───────────────────────
def html_to_blocks(content, base_dir: Optional[Path] = None) -> list[dict]:
    """HTML (full document or fragment) -> text + image blocks, in document
    order. <img> srcs are data-URIs or files under base_dir (when given)."""
    import base64
    from urllib.parse import unquote, urlparse
    from lxml import html as _lhtml

    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    try:
        root = _lhtml.fromstring(content)
    except Exception:
        # Broken markup: crude strip beats dropping the content.
        return text_blocks(_crude_strip(content))

    def resolve(src: str):
        src = src.split("#")[0].split("?")[0].strip()
        if not src:
            return None, None
        if src.startswith("data:"):
            m = re.match(r"data:[\w/+.-]+;base64,(.+)$", src, re.S)
            if not m:
                return None, None
            try:
                data = base64.b64decode(m.group(1))
            except Exception:
                return None, None
            return data, _image_ext(data, "")
        if base_dir is None or src.startswith(("http://", "https://")):
            return None, None
        if src.startswith("file://"):
            try:
                src = unquote(urlparse(src).path)
            except Exception:
                pass
        p = Path(src)
        if not p.is_absolute():
            p = Path(base_dir) / src
        try:
            data = p.resolve().read_bytes()
        except OSError:
            return None, None
        return data, _image_ext(data, p.name)

    blocks: list[dict] = []

    def visit(el) -> None:
        tag = el.tag if isinstance(el.tag, str) else None
        if tag in _SKIP_TAGS:
            return
        if tag == "img":
            src = el.get("src") or el.get("href") \
                or el.get("{http://www.w3.org/1999/xlink}href")
            if src:
                data, ext = resolve(src)
                if data and not _too_small(data):
                    blocks.append({"kind": "image", "ext": ext, "data": data,
                                   "ref": src})
            return
        if tag in _BLOCK_TAGS:
            has_block_child = any(
                isinstance(c.tag, str) and c.tag in _BLOCK_TAGS for c in el)
            if not has_block_child:
                txt = " ".join("".join(el.itertext()).split())
                if txt:
                    blocks.append({"kind": "text", "text": txt})
            for c in el:
                visit(c)
            return
        for c in el:
            visit(c)

    visit(root)
    return blocks


def _decode_html(raw: bytes) -> str:
    """Decode an HTML file, honouring its declared charset (cp1251 Russian
    books included), with the usual fallbacks otherwise."""
    m = re.search(rb"charset\s*=\s*[\"']?([\w-]+)", raw[:8192], re.I)
    encs = []
    if m:
        encs.append(m.group(1).decode("ascii", "ignore"))
    encs += ["utf-8-sig", "utf-16", "latin-1"]
    for enc in encs:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


# ── RTF / DOCX / HTML / MHT / FB2 / CHM -> blocks ────────────────────────────
def rtf_blocks(path: Path) -> list[dict]:
    raw = Path(path).read_bytes().decode("utf-8", errors="replace")
    try:
        from striprtf.striprtf import rtf_to_text  # striprtf >= 0.0.2x
        text = rtf_to_text(raw)
    except ImportError:
        import striprtf  # older API
        text = striprtf.rtf.striprtf(raw)
    return text_blocks(text)


def docx_blocks(path: Path) -> list[dict]:
    import mammoth
    with open(path, "rb") as f:
        # mammoth 1.x: images.data_uri is a ready-made <img> converter
        result = mammoth.convert_to_html(
            f, convert_image=mammoth.images.data_uri)
    return html_to_blocks(result.value, base_dir=Path(path).parent)


def html_blocks(path: Path) -> list[dict]:
    return html_to_blocks(_decode_html(Path(path).read_bytes()),
                          base_dir=Path(path).parent)


def mht_blocks(path: Path) -> list[dict]:
    """MHT/MHTML: a MIME multipart file (often one text/html part)."""
    import email
    from email import policy
    raw = Path(path).read_bytes()
    try:
        msg = email.message_from_bytes(raw, policy=policy.default)
    except Exception:
        return html_blocks(Path(path))  # not really MIME — treat as HTML
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(payload)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            parts.append(payload)
    blocks: list[dict] = []
    for p in parts:
        blocks.extend(html_to_blocks(p, base_dir=Path(path).parent))
    return blocks


def fb2_blocks(path: Path) -> list[dict]:
    """FictionBook 2.0 (common in Russian collections): simple XML."""
    from lxml import etree
    raw = Path(path).read_bytes()
    root = etree.fromstring(raw)
    blocks: list[dict] = []
    body = root.find(".//{http://www.gribiko.ru/l/fb2}body")
    if body is None:
        body = root.find(".//body")
    if body is None:
        body = root
    for el in body.iter():
        local = etree.QName(el).localname \
            if isinstance(getattr(el, "tag", None), str) else None
        if local in ("p", "title", "subtitle", "cite", "epigraph"):
            txt = " ".join("".join(el.itertext()).split())
            if txt:
                blocks.append({"kind": "text", "text": txt})
    if not blocks:  # namespaced-less or odd file: crude fallback
        blocks = text_blocks(_crude_strip(_crude_strip_xml(raw)))
    return blocks


def _crude_strip_xml(raw: bytes) -> str:
    xml = raw.decode("utf-8", errors="replace")
    xml = re.sub(r"<(empty-line|image|author-info)[^>]*/?>", " ", xml, flags=re.I)
    return re.sub(r"<[^>]+>", " ", xml)


def chm_blocks(path: Path) -> list[dict]:
    """CHM (Windows help, ITSF): extract each HTML topic in TOC order.

    Uses the `chm` package (pip name: pychm) — a SWIG wrapper around
    libchm: CHMFile.LoadCHM / ResolveObject / RetrieveObject."""
    from chm.chm import CHMFile

    f = CHMFile()
    if not f.LoadCHM(str(path)):
        raise RuntimeError("could not open CHM archive")
    enc = f.GetEncoding()  # codec name or None

    def decode(data: bytes) -> str:
        for e in ([enc] if enc else []) + ["utf-8", "latin-1"]:
            try:
                return data.decode(e)
            except (UnicodeDecodeError, UnicodeError, LookupError):
                continue
        return data.decode("latin-1", errors="replace")

    def fetch(url: str):
        if not url or url.startswith("/#"):
            return None
        if not url.startswith("/"):
            url = "/" + url
        try:
            res, ui = f.ResolveObject(url.encode("utf-8"))
            if res != 0:
                return None
            size, data = f.RetrieveObject(ui)
            return data if size else None
        except Exception:
            return None

    urls = _chm_topic_urls(f)
    if not urls:
        urls = _chm_enumerate_html(f)
    blocks: list[dict] = []
    for u in urls:
        data = fetch(u)
        if data:
            blocks.extend(html_to_blocks(decode(data)))
    f.CloseCHM()
    return blocks


def _chm_topic_urls(f) -> list[str]:
    """Topic URLs in TOC order, parsed from the HHC (contents) file.
    HHC is HTML-ish: <LI><OBJECT ...><param name="URL" value="..."> or
    <object data="...">."""
    tree = f.GetTopicsTree()
    if not tree:
        return []
    html = tree.decode("utf-8", errors="replace")
    urls = re.findall(
        r'(?:<param\s+name=["\']URL["\']\s+value=["\']([^"\']+)["\']'
        r'|<object[^>]+data=["\']([^"\']+)["\'])', html, re.I)
    out, seen = [], set()
    for m in urls:
        u = (m[0] or m[1] or "").strip()
        low = u.lower()
        if u and u not in seen and not low.startswith(
                ("http:", "https:", "file:", "javascript:", "about:", "#")) \
                and not low.endswith((".hhc", ".hhk")):
            seen.add(u)
            out.append(u)
    return out


def _chm_enumerate_html(f) -> list[str]:
    """Fallback when there is no HHC TOC: enumerate all .htm[l] entries."""
    from chm import chmlib
    found: list[str] = []

    def cb(ctx, ui, context):
        try:
            p = ui.path.decode("utf-8", "replace") \
                if isinstance(ui.path, (bytes, bytearray)) \
                else str(ui.path)
        except Exception:
            p = ""
        if p.lower().endswith((".htm", ".html")) and not p.startswith("/#"):
            found.append(p)
        return chmlib.CHM_ENUMERATOR_CONTINUE

    try:
        chmlib.chm_enumerate(f.file, chmlib.CHM_ENUMERATE_NORMAL, cb, None)
    except Exception:
        pass
    # libchm's enumeration order is index (hash) order — natural-sort so
    # sections come out in a sane, deterministic sequence.
    found.sort(key=lambda u: [int(t) if t.isdigit() else t.lower()
                              for t in re.split(r"(\d+)", u)])
    return found


def extract_blocks(path: Path) -> tuple[list[dict], dict]:
    """Native formats -> (blocks, meta). meta carries title/authors when the
    container has them (EPUB); {} otherwise. Raises ValueError for
    non-native formats (pdf and the converters.py formats)."""
    fmt = detect_format(path)
    if fmt == "epub":
        return epub_blocks(path)
    if fmt == "rtf":
        return rtf_blocks(path), {}
    if fmt == "docx":
        return docx_blocks(path), {}
    if fmt == "html":
        return html_blocks(path), {}
    if fmt == "mht":
        return mht_blocks(path), {}
    if fmt == "fb2":
        return fb2_blocks(path), {}
    if fmt == "chm":
        return chm_blocks(path), {}
    if fmt == "txt":
        return text_blocks(read_text(path)), {}
    raise ValueError(f"no native extractor for format '{fmt}'")


# ── Pagination (nominal word count, paragraph-aware) ─────────────────────────
def _word_count(s: str) -> int:
    return len(s.split())


def split_long_text(text: str, target: int) -> list[str]:
    """Split one oversized paragraph at sentence boundaries, then at word
    boundaries if a single sentence still exceeds the budget."""
    sents = re.split(r"(?<=[.!?…])\s+", text.strip())
    pieces: list[str] = []
    cur, cur_w = "", 0
    for s in sents:
        if _word_count(s) > target:
            if cur:
                pieces.append(cur)
                cur, cur_w = "", 0
            words = s.split()
            while len(words) > target:
                pieces.append(" ".join(words[:target]))
                words = words[target:]
            s, cur = " ".join(words), ""
            cur_w = _word_count(s)
            continue
        if cur and cur_w + _word_count(s) > target:
            pieces.append(cur)
            cur, cur_w = "", 0
        cur = f"{cur} {s}".strip()
        cur_w = _word_count(cur)
    if cur:
        pieces.append(cur)
    return pieces


def paginate(blocks: list[dict], target_words: int) -> list[list[dict]]:
    """Group blocks into nominal pages of ~target_words words.

    Rules (paragraph-aware):
      - a page accumulates whole paragraphs until it reaches target_words,
        then breaks at the next paragraph boundary
      - a paragraph that would push the page past 1.5x target starts a new
        page instead (no mid-paragraph breaks in normal text)
      - a single paragraph > 1.5x target is split at sentence/word
        boundaries (rare: wall-of-text exports)
      - images count IMAGE_WORDS of budget; they never start a page, they
        ride on the current one until the hard cap"""
    target = max(50, int(target_words))
    hard_max = int(target * 1.5)
    pages: list[list[dict]] = []
    cur: list[dict] = []
    cur_w = 0

    def flush() -> None:
        nonlocal cur, cur_w
        if cur:
            pages.append(cur)
            cur, cur_w = [], 0

    for b in blocks:
        if b.get("kind") == "image":
            if cur and cur_w + IMAGE_WORDS > hard_max:
                flush()
            cur.append(b)
            cur_w += IMAGE_WORDS
            continue
        w = _word_count(b["text"])
        if w > hard_max:
            if cur:
                flush()
            for piece in split_long_text(b["text"], target):
                cur.append({"kind": "text", "text": piece})
                cur_w += _word_count(piece)
                if cur_w >= target:
                    flush()
            continue
        if cur and cur_w + w > hard_max:
            flush()
        cur.append(b)
        cur_w += w
        if cur_w >= target:
            flush()
    flush()
    return pages


# ── Page count estimate (dashboard pre-scan) ─────────────────────────────────
def estimate_pages(path: Path, fmt: str, words_per_page: int) -> int:
    """How many nominal pages this file will produce (fast, CPU-only).
    0 = unknown (the dashboard counts the file exactly at processing
    time)."""
    path = Path(path)
    try:
        if fmt in TEXT_FORMATS:
            blocks, _ = extract_blocks(path)
            return max(1, len(paginate(blocks, words_per_page)))
        if fmt != "pdf":  # convertible formats (djvu/cbz/cbr/...)
            from converters import quick_page_count
            return quick_page_count(path, fmt)
        return 0
    except Exception:
        return 0


# ── Conversion to the pipeline Parquet schema ────────────────────────────────
def convert_textbook_to_parquet(
    path: Path,
    output_dir: Path,
    input_root: Path,
    words_per_page: int = 300,
    log=print,
    original_path: Optional[Path] = None,
) -> bool:
    """One text-based book -> one pipeline-parquet + images/.

    Accepts any native text format (txt/epub/rtf/docx/html/mht/fb2/chm),
    including a file produced by converters.py (e.g. a converted .docx or
    .epub) — `original_path` (the user's actual file, e.g. .doc/.mobi)
    is what gets recorded in source_file and copied next to the parquet.

    Mirrors ocr.py's PDF layout:
      <out>/<rel parent>/<stem>.parquet
      <out>/<rel parent>/<stem>.<original ext>  (source copy)
      <out>/<rel parent>/images/<book>_p<page>_img_<n>.<ext>
    Returns True on success."""
    path = Path(path)
    original_path = Path(original_path) if original_path else path
    fmt = detect_format(path)
    if fmt not in TEXT_FORMATS:
        log(f"  ERROR: {path.name} is not a text book (detected: {fmt})")
        return False

    try:
        blocks, meta = extract_blocks(path)
    except Exception as e:
        log(f"  ERROR: extraction failed for {path.name}: {e}")
        return False
    if meta.get("title"):
        log(f"  {fmt.upper()} metadata: {meta['title']}"
            + (f" — {meta['authors']}" if meta.get("authors") else ""))

    if not blocks:
        log(f"  WARNING: no text extracted from {path.name}")
        return False

    pages = paginate(blocks, words_per_page)
    n_images = sum(1 for p in pages for b in p if b.get("kind") == "image")
    n_words = sum(_word_count(b["text"]) for p in pages for b in p
                  if b.get("kind") == "text")
    log(f"  Text blocks: {len(blocks)}  pages (nominal "
        f"{int(words_per_page)} w/page): {len(pages)}  "
        f"words: {n_words}  images: {n_images}")

    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    out_subdir = Path(output_dir) / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / (path.stem + ".parquet")
    img_dir = out_subdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    source_file = str(original_path)
    book_title = path.stem.replace("_", " ").replace("-", " ")
    safe_book = re.sub(r"[^\w]", "", book_title)  # ocr.py image naming

    rows = {"source_file": [], "book_title": [], "page_num": [],
            "page_width": [], "page_height": [], "text": [],
            "image_bbox": [], "image_path": [], "image_name": []}
    img_seq = 0
    for page_no, page_blocks in enumerate(pages, start=1):
        texts: list[str] = []
        img_bboxes, img_paths, img_names = [], [], []
        for b in page_blocks:
            if b.get("kind") == "text":
                texts.append(b["text"])
                continue
            img_seq += 1
            ext = b.get("ext") or ".png"
            name = f"{safe_book}_p{page_no}_img_{img_seq}{ext}"
            save_path = img_dir / name
            try:
                from PIL import Image
                if ext == ".png" and b.get("ext") in (".webp", None, ""):
                    with Image.open(io.BytesIO(b["data"])) as im:
                        im.convert("RGB").save(str(save_path), format="PNG")
                else:
                    save_path.write_bytes(b["data"])
            except Exception as e:  # fall back to raw bytes
                save_path.write_bytes(b["data"])
                log(f"    WARN: image {name}: {e}")
            img_bboxes.append([0.0, 0.0, 0.0, 0.0])
            img_paths.append(str(save_path))
            img_names.append(name)

        rows["source_file"].append(source_file)
        rows["book_title"].append(book_title)
        rows["page_num"].append(page_no)
        rows["page_width"].append(NOMINAL_PAGE_W)
        rows["page_height"].append(NOMINAL_PAGE_H)
        rows["text"].append("\n\n".join(texts))
        rows["image_bbox"].append(img_bboxes)
        rows["image_path"].append(img_paths)
        rows["image_name"].append(img_names)

    table = pa.Table.from_pydict(rows, schema=PAGE_SCHEMA)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_path)

    # Copy the ORIGINAL source (user's actual file, e.g. .doc/.mobi when
    # this came from a converter) next to the parquet so the output dir is
    # self-contained for the rename/DDC step.
    src_copy = out_subdir / (original_path.stem + original_path.suffix.lower())
    src_size = original_path.stat().st_size
    if src_size > 0 and (not src_copy.exists()
                         or src_copy.stat().st_size != src_size):
        shutil.copy2(original_path, src_copy)
        log(f"  Copied source: {src_copy.name} ({src_size / 1e6:.2f} MB)")

    log(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return True
