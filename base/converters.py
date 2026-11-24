#!/usr/bin/env python3
"""
converters.py — external-tool conversions for the DDC pipeline.
================================================================
Formats that need a system tool before the native extractors can
handle them. Each converter writes the result into a per-book work
directory (same stem as the original, so output naming is preserved)
and returns (converted_path, target_format):

  Kindle: .mobi .azw .azw3 .kfx .lit  -> ebook-convert -> .epub  (epub path,
                                                        images preserved)
  Office: .doc .odt .ods .odp         -> soffice headless -> .docx
                                                        (docx text+images path)
  DJVU:   .djvu                       -> djvups (PS) + ps2pdf -> .pdf,
                                                batched + pypdf merge (marker OCR path)
  Comics: .cbz .cbr                   -> zip/7z/unrar + PIL -> .pdf
                                                        (marker OCR path)

Tools are invoked lazily (a text-only batch never touches calibre).
convert() returns None for formats the pipeline handles natively
(pdf/epub/txt/rtf/docx/html/mht/fb2/chm).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

# original extension -> converter name
EXT_TO_CONVERTER = {
    ".mobi": "kindle", ".azw": "kindle", ".azw3": "kindle",
    ".kfx": "kindle", ".lit": "kindle",
    ".doc": "office", ".odt": "office", ".ods": "office", ".odp": "office",
    ".djvu": "djvu",
    ".cbz": "comic", ".cbr": "comic",
}
# detected format (booktext.detect_format) -> converter name
FMT_TO_CONVERTER = {
    "mobi": "kindle", "azw": "kindle", "azw3": "kindle",
    "kfx": "kindle", "lit": "kindle",
    "doc": "office", "odt": "office",
    "djvu": "djvu",
    "cbz": "comic", "cbr": "comic",
}
CONVERTER_TIMEOUT = {  # seconds; a 3000-page djvu rasterizes for minutes
    "kindle": 900, "office": 600, "djvu": 3600, "comic": 1800,
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".webp")
# Formats ocr.py treats as PDF-like (marker OCR) once converted.
PDF_LIKE = {"djvu", "cbz", "cbr"}


def _run(cmd: list[str], timeout: int, log=print) -> subprocess.CompletedProcess:
    log(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _tool(name: str, log=print) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(
            f"'{name}' not found on PATH — run install_formats.sh")
    return p


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t
            for t in re.split(r"(\d+)", name.lower())]


# ── Kindle -> EPUB ────────────────────────────────────────────────────────────
def kindle_to_epub(path: Path, workdir: Path, log=print) -> Path:
    out = workdir / (path.stem + ".epub")
    r = _run([_tool("ebook-convert", log), str(path), str(out)],
             CONVERTER_TIMEOUT["kindle"], log)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ebook-convert failed: {r.stderr[-500:]}")
    return out


# ── DOC/ODT/... -> DOCX ───────────────────────────────────────────────────────
def office_to_docx(path: Path, workdir: Path, log=print) -> Path:
    # Unique LO profile per workdir so parallel jobs don't fight the lock.
    # Explicit export filter name: some LO builds (Debian minimal) don't
    # register the bare "docx" target alias.
    env = f"-env:UserInstallation=file://{workdir / 'lo_profile'}"
    r = _run([_tool("soffice", log), "--headless", "--norestore", env,
              "--convert-to", "docx:MS Word 2007 XML",
              "--outdir", str(workdir), str(path)],
             CONVERTER_TIMEOUT["office"], log)
    out = workdir / (path.stem + ".docx")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"soffice convert failed: {r.stderr[-500:]}")
    return out


# ── DJVU -> PDF (the cheat code) ──────────────────────────────────────────────
def _djvu_page_count(path: Path, log=print) -> int:
    """Exact page count via djvused's 'n' command (script on stdin)."""
    r = subprocess.run([_tool("djvused", log), str(path)],
                       input="n\n", capture_output=True, text=True, timeout=120)
    m = re.search(r"\d+", r.stdout or "")
    return int(m.group(0)) if m else 0


def djvu_to_pdf(path: Path, workdir: Path, log=print) -> Path:
    """djvu -> PDF via djvups (PostScript) + ps2pdf (ghostscript).

    Rendered in page batches: djvups' PS output is huge (a 75-page scan
    produced 370 MB of .ps), so each batch becomes its own PDF which is
    then merged with pypdf. Batches keep the temp footprint bounded for
    3000-page monsters."""
    djvups, ps2pdf = _tool("djvups", log), _tool("ps2pdf", log)
    out = workdir / (path.stem + ".pdf")
    try:
        n = _djvu_page_count(path, log)
    except Exception:
        n = 0
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    batch = 50
    for start in range(0, max(n, 1), batch):
        end = min(start + batch, n) if n else start + batch
        plist = f"{start + 1}-{end}" if n else ""
        ps = workdir / f"{path.stem}_b{start // batch}.ps"
        batch_pdf = workdir / f"{path.stem}_b{start // batch}.pdf"
        r = _run([djvups, "-gamma=1", "-colormatch=no", "-mode=color",
                  "-level=2", *([f"-page={plist}"] if plist else []),
                  str(path), str(ps)], CONVERTER_TIMEOUT["djvu"], log)
        if r.returncode != 0 or not ps.exists():
            raise RuntimeError(f"djvups failed: {r.stderr[-500:]}")
        r = _run([ps2pdf, "-dPDFSETTINGS=/ebook", str(ps), str(batch_pdf)],
                 CONVERTER_TIMEOUT["djvu"], log)
        if r.returncode != 0 or not batch_pdf.exists():
            raise RuntimeError(f"ps2pdf failed: {r.stderr[-500:]}")
        ps.unlink(missing_ok=True)
        writer.append(str(batch_pdf))
        batch_pdf.unlink(missing_ok=True)
        if n and end >= n:
            break
    if not writer.pages:
        raise RuntimeError("djvu->pdf produced no pages")
    with open(out, "wb") as f:
        writer.write(f)
    return out


def _images_to_pdf(imgs: list[Path], out: Path, log=print) -> Path:
    """Bundle page images into one PDF with PIL (png2pdf isn't packaged
    on Debian). Images are opened lazily and streamed, so even a 500-page
    comic stays memory-friendly."""
    from PIL import Image
    ims = []
    for p in imgs:
        im = Image.open(p)
        if im.mode not in ("RGB", "L", "CMYK"):
            im = im.convert("RGB")
        ims.append(im)
    ims[0].save(str(out), "PDF", save_all=True, append_images=ims[1:])
    return out


# ── CBZ / CBR -> PDF ──────────────────────────────────────────────────────────
def comic_to_pdf(path: Path, workdir: Path, log=print) -> Path:
    pages = workdir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".cbz":
        with zipfile.ZipFile(path) as z:
            z.extractall(pages)
    else:  # cbr — "RAR" in name only: real RAR, 7z, even zip all show up
        # as .cbr. 7z reads them all; unrar (rar only) is the fallback.
        if shutil.which("7z"):
            r = _run(["7z", "x", "-y", f"-o{pages}", str(path)],
                     CONVERTER_TIMEOUT["comic"], log)
        else:
            r = _run([_tool("unrar", log), "x", "-y", str(path), str(pages)],
                     CONVERTER_TIMEOUT["comic"], log)
        if r.returncode != 0:
            raise RuntimeError(f"comic extraction failed: {r.stderr[-500:]}")
    imgs = sorted(
        (p for p in pages.rglob("*") if p.suffix.lower() in IMG_EXTS),
        key=lambda p: _natural_key(p.name))
    if not imgs:
        raise RuntimeError("no images found in the comic archive")
    out = workdir / (path.stem + ".pdf")
    _images_to_pdf(imgs, out, log)
    shutil.rmtree(pages, ignore_errors=True)
    return out


# ── Dispatch ──────────────────────────────────────────────────────────────────
_CONVERTERS = {"kindle": kindle_to_epub, "office": office_to_docx,
               "djvu": djvu_to_pdf, "comic": comic_to_pdf}
_TARGET_FMT = {"kindle": "epub", "office": "docx", "djvu": "pdf",
               "comic": "pdf"}


def convert(path: Path, workdir: Path, fmt: str | None = None, log=print):
    """Convert one file if its format needs an external tool.

    `fmt` is the detected format (preferred — the file may be mislabelled);
    without it the extension is used. Returns (converted_path,
    target_format) or None when the file is handled natively (no
    conversion needed). Raises RuntimeError when a conversion fails."""
    conv = (FMT_TO_CONVERTER.get(fmt) if fmt
            else EXT_TO_CONVERTER.get(path.suffix.lower()))
    if conv is None:
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    out = _CONVERTERS[conv](path, workdir, log)
    return out, _TARGET_FMT[conv]


def quick_page_count(path: Path, fmt: str) -> int:
    """Cheap page estimate for not-yet-converted books (dashboard pre-scan).
    0 = unknown (the file will be counted exactly when it is processed)."""
    try:
        if fmt == "djvu":
            return _djvu_page_count(path)
        if fmt == "cbz":
            with zipfile.ZipFile(path) as z:
                return sum(1 for n in z.namelist()
                           if Path(n).suffix.lower() in IMG_EXTS)
        if fmt == "cbr":
            # Same tool order as comic_to_pdf: 7z reads every .cbr variant,
            # unrar only real RAR.
            if shutil.which("7z"):
                r = _run(["7z", "l", str(path)], 120)
                out = r.stdout or ""
                # The file list sits between the two dashed separator rows;
                # the "N files" summary comes after the second one (and the
                # header words are locale-dependent, so don't rely on them).
                lines = out.splitlines()
                seps = [i for i, l in enumerate(lines)
                        if l.strip().startswith("-----")]
                count = (sum(1 for l in lines[seps[0] + 1:seps[1]]
                             if l.strip()) if len(seps) >= 2 else 0)
                if count == 0:
                    m = re.search(r"\((\d+) files?\)", out, re.I) \
                        or re.search(r"Files = (\d+)", out)
                    count = int(m.group(1)) if m else 0
                return count
            r = _run([_tool("unrar", log=lambda *_: None), "lb", str(path)],
                     120)
            return max(0, (r.stdout or "").count("\n") - 1) \
                if r.returncode == 0 else 0
    except Exception:
        pass
    return 0
