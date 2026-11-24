#!/usr/bin/env python3
"""Parquet -> Qdrant indexing pipeline for the DDC book library.

Walks a root directory for *.parquet files (one file per book, one row per
page), embeds page text with an Ollama model (default: bge-m3, 1024-dim) and
upserts points into a Qdrant collection (default: ddc-books, cosine).

Idempotent / resumable / self-healing:
  - book_id = first 16 hex chars of sha256("final_title|authors|pub_date").
    When several parquets share the same book_id (same book OCR'd twice, or
    misfiled under two DDC sub-dirs), the first file in sort order keeps
    that id and each further file gets the hash seeded with its relative
    path — otherwise duplicates would share point ids and re-embed forever.
  - point ids are deterministic uint64s of (book_id, page_num), so re-running
    a book REPLACES its points instead of duplicating them
  - a book is skipped only when Qdrant holds EXACTLY the points the current
    pipeline would write. Fewer points (a run died mid-upsert => partial
    book) or more (an older pipeline version, e.g. one that also embedded
    empty-text pages) => the book's points are deleted and rebuilt, so
    crashed or stale books repair themselves on the next re-run
  - non-book parquets without a final_title column (e.g. recipes.parquet /
    ingredients.parquet) are skipped, not counted as failures

Usage:
    python index_books.py --root ~/DDC                 # index everything
    python index_books.py --root ~/DDC --limit 5       # first 5 files only
    python index_books.py --root ~/DDC --limit 5 --dry-run

Environment:
    EMBED_BASE_URL  Ollama base URL      (default http://localhost:11434)
    EMBED_MODEL     embedding model name (default bge-m3)
    QDRANT_URL      Qdrant base URL      (default http://localhost:6333)

Failures on individual files are logged and appended to
<indexer dir>/failed.jsonl as {filename, error, timestamp}; the run
continues with the next file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Defaults / tunables
# ---------------------------------------------------------------------------
DEFAULT_ROOT = Path.home() / "DDC"
DEFAULT_EMBED_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_QDRANT_URL = "http://localhost:6333"

COLLECTION = "ddc-books"
VECTOR_DIM = 1024
EMBED_BATCH = 32     # pages per /api/embed request (safe on a P2000)
UPSERT_BATCH = 64    # points per Qdrant upsert
EMBED_TIMEOUT = 300  # s per embed request (first call includes model load)
QDRANT_TIMEOUT = 60  # s per Qdrant request
EMBED_RETRIES = 5
RETRY_BACKOFFS = [2, 8, 30, 60, 120]  # s between embed retries
OLLAMA_DOWN_WAIT = 900  # s to wait for Ollama to come back after a connection error
ESTIMATE_AFTER = 3           # number of indexed books before showing ETA

# ---------------------------------------------------------------------------
# Progress file — machine-readable state for the dashboard (~/base/gui) and
# any other watcher. Written atomically after every book and (throttled)
# after every embed batch inside a book, so the UI can show exactly which
# book/page we are on, how far we have to go, and a live ETA. Never raises:
# progress is a nicety, the index run is the job.
# ---------------------------------------------------------------------------
PROGRESS_DIR = Path(os.environ.get(
    "DDC_PROGRESS_DIR", str(Path.home() / ".ddc-progress")))


def progress_path_for(root: Path) -> Path:
    """One progress file per library root: ~/.ddc-progress/index_<root>.json."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_") or "root"
    return PROGRESS_DIR / f"index_{slug}.json"


def write_progress(path: Path, state: dict) -> None:
    try:
        state = dict(state)
        state["pid"] = os.getpid()
        state["last_update"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - progress must never kill the run
        pass

# Parquet columns we actually need (avements: skip the heavy image columns)
META_KEYS = ("final_title", "authors", "publisher", "pub_date", "ddc")
PAGE_COLUMNS = list(META_KEYS) + ["text", "page_num", "is_index"]

log = logging.getLogger("index_books")


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------
def book_id_of(final_title: str, authors: str, pub_date: str) -> str:
    """Stable 16-hex-char book identity (spec: sha256 of title|authors|pub_date)."""
    raw = f"{final_title or ''}|{authors or ''}|{pub_date or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def point_id_for(book_id: str, page_num: int) -> int:
    """Deterministic uint64 Qdrant point id, stable across re-runs.

    (Qdrant only accepts uint64 or UUID point ids, so the string
    'book_id_page' is hashed to a uint64; same inputs -> same id always.)
    """
    raw = f"{book_id}:{page_num}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def assign_book_ids(files: list[Path], root: Path) -> dict[Path, str | None]:
    """Per-file book_id, unique even when several parquets describe the same
    book (same title|authors|pub_date from different source files — the same
    book OCR'd twice, or misfiled under two DDC sub-dirs).

    Without this, duplicates share one book_id *and* their point ids
    (point ids hash book_id + page_num), so they overwrite each other's
    points and can never satisfy the exact-count skip check: every run
    deletes and re-embeds all of them, forever.

    Disambiguation stays stable across runs (the exact-count skip relies on
    that): the first file in sort order keeps the plain
    sha256(title|authors|pub_date) id — matching whatever is already in
    Qdrant — and each additional file gets that hash seeded with its path
    relative to root. Same tree -> same ids, always.
    """
    base_of: dict[Path, str] = {}
    by_base: dict[str, list[Path]] = {}
    for f in files:
        try:
            t = pq.read_table(f, columns=["final_title", "authors", "pub_date"])
            rows = t.to_pylist()
            if not rows:
                continue
            meta = {k: (str(rows[0][k]) if rows[0][k] is not None else "")
                    for k in ("final_title", "authors", "pub_date")}
        except Exception:  # noqa: BLE001 - unreadable / not a book: no id
            continue
        base = book_id_of(meta["final_title"], meta["authors"], meta["pub_date"])
        base_of[f] = base
        by_base.setdefault(base, []).append(f)
    out: dict[Path, str | None] = {}
    for base, group in by_base.items():
        for pos, f in enumerate(sorted(group)):
            if pos == 0:
                out[f] = base
            else:
                rel = str(f.relative_to(root))
                out[f] = hashlib.sha256(f"{base}|{rel}".encode("utf-8")).hexdigest()[:16]
                log.info("book_id disambiguated (shares title with %s): %s -> %s",
                         base, rel, out[f])
    return out


# ---------------------------------------------------------------------------
# Clients (plain REST, no SDK)
# ---------------------------------------------------------------------------
class Qdrant:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def _req(self, method: str, path: str, **kwargs):
        r = self.session.request(method, self.base + path,
                                 timeout=QDRANT_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json() if r.content else None

    def healthy(self) -> bool:
        try:
            r = self.session.get(self.base + "/healthz", timeout=5)
            return r.status_code in (200, 405)
        except requests.RequestException:
            return False

    def ensure_collection(self) -> None:
        """Create the collection (and a payload index on book_id) if missing."""
        existing = self._req("GET", "/collections")["result"]["collections"]
        if any(c["name"] == COLLECTION for c in existing):
            return
        # Qdrant >= 1.20 uses PUT for collection creation (POST is 405).
        self._req("PUT", f"/collections/{COLLECTION}", json={
            "vectors": {"size": VECTOR_DIM, "distance": "Cosine"},
        })
        log.info("Created collection '%s' (%d-dim, cosine)", COLLECTION, VECTOR_DIM)
        self.ensure_book_id_index()

    def ensure_book_id_index(self) -> None:
        """Keyword index on payload book_id so existence checks are O(1)."""
        try:
            # Qdrant >= 1.20: PUT /collections/{name}/index (singular).
            self._req("PUT", f"/collections/{COLLECTION}/index", json={
                "field_name": "book_id",
                "field_schema": "keyword",
            })
            log.info("Created keyword index on payload field 'book_id'")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                return  # index already exists
            raise

    def book_exists_count(self, book_id: str) -> int:
        data = self._req("POST", f"/collections/{COLLECTION}/points/count", json={
            "filter": {"must": [{"key": "book_id", "match": {"value": book_id}}]},
            "exact": True,
        })
        return data["result"]["count"]

    def book_exists(self, book_id: str) -> bool:
        return self.book_exists_count(book_id) > 0

    def delete_book_points(self, book_id: str) -> int:
        """Delete all points of one book (e.g. partials left by a failed run).

        Two passes: the filter delete uses the book_id payload index and is
        instant, but under optimizer load a follow-up count can still see
        points that are settling. Stragglers are then removed by explicit
        point id — a by-id delete cannot be skipped by a stale filter scan.
        """
        self._req("POST", f"/collections/{COLLECTION}/points/delete", json={
            "filter": {"must": [{"key": "book_id", "match": {"value": book_id}}]},
            "wait": True,
        })
        for attempt in range(12):
            gone = self.book_exists_count(book_id)
            if not gone:
                return 0
            if attempt >= 2:
                ids: list = []
                offset = None
                while True:
                    body = {"filter": {"must": [{"key": "book_id",
                                                 "match": {"value": book_id}}]},
                            "limit": 1024, "with_payload": False,
                            "with_vector": False}
                    if offset is not None:
                        body["offset"] = offset
                    data = self._req("POST",
                                     f"/collections/{COLLECTION}/points/scroll",
                                     json=body)
                    pts = data["result"]["points"]
                    ids.extend(p["id"] for p in pts)
                    offset = data["result"].get("next_page_offset")
                    if not offset:
                        break
                if ids:
                    log.info("deleting %d straggler point(s) of %s by id",
                             len(ids), book_id)
                    self._req("POST", f"/collections/{COLLECTION}/points/delete",
                              json={"points": ids, "wait": True})
            time.sleep(0.5)
        gone = self.book_exists_count(book_id)
        if gone:
            raise RuntimeError(f"delete_book_points({book_id}) left {gone} points")
        return 0

    def upsert(self, points: list[dict]) -> None:
        # Qdrant >= 1.20: upsert is PUT /points (POST /points is now "retrieve").
        self._req("PUT", f"/collections/{COLLECTION}/points",
                  json={"points": points})

    def total_points(self) -> int:
        data = self._req("POST", f"/collections/{COLLECTION}/points/count",
                         json={"exact": True})
        return data["result"]["count"]


class Embedder:
    def __init__(self, base_url: str, model: str):
        self.root = base_url.rstrip("/")
        self.embed_url = self.root + "/api/embed"
        self.model = model
        self.session = requests.Session()

    def model_available(self) -> bool:
        r = self.session.get(self.root + "/api/tags", timeout=15)
        r.raise_for_status()
        # /api/tags reports "bge-m3:latest" style names; match the base name.
        wanted = self.model.split(":")[0]
        return any((m.get("name") or "").split(":")[0] == wanted
                   for m in r.json().get("models", []))

    def wait_for_ollama(self, timeout: float = OLLAMA_DOWN_WAIT) -> bool:
        """Poll /api/tags until Ollama answers again (e.g. it crashed and is
        being restarted). Returns True once reachable, False after timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.session.get(self.root + "/api/tags", timeout=10)
                return True
            except requests.RequestException:
                time.sleep(15)
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, retrying transient failures."""
        last: Exception | None = None
        for attempt in range(EMBED_RETRIES):
            try:
                r = self.session.post(self.embed_url,
                                      json={"model": self.model, "input": texts},
                                      timeout=EMBED_TIMEOUT)
                r.raise_for_status()
                vecs = r.json()["embeddings"]
                if len(vecs) != len(texts):
                    raise ValueError(f"expected {len(texts)} vectors, got {len(vecs)}")
                return vecs
            except (requests.RequestException, ValueError, KeyError) as e:
                last = e
                # Connection-level failure => Ollama is likely down; give it
                # up to 15 min to come back (crash + restart) before counting
                # this as a failed attempt.
                if isinstance(e, requests.ConnectionError):
                    log.warning("Ollama unreachable (attempt %d/%d): %s "
                                "-- waiting up to %ds for it to come back",
                                attempt + 1, EMBED_RETRIES, e, OLLAMA_DOWN_WAIT)
                    if self.wait_for_ollama():
                        log.info("Ollama is back; retrying immediately")
                        continue
                wait = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
                log.warning("embed batch failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt + 1, EMBED_RETRIES, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"embedding failed after {EMBED_RETRIES} attempts: {last}")


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def ddc_dirs(path: Path, root: Path) -> tuple[str, str]:
    """Return (ddc_coarse, ddc_tens) from the file's location under root."""
    parts = path.relative_to(root).parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "", ""


def embeddable_page_count(path: Path) -> int:
    """Pages process_file would actually embed for this file: content rows
    (is_index is False) with non-empty text — same filter as the pipeline.
    0 for non-book parquets and unreadable files (fail-soft: progress math
    must never raise)."""
    try:
        if "final_title" not in pq.read_schema(path).names:
            return 0
        rows = pq.read_table(path, columns=["is_index", "text"]).to_pylist()
        return sum(1 for r in rows
                   if r.get("is_index") is False and (r.get("text") or "").strip())
    except Exception:  # noqa: BLE001
        return 0


def process_file(path: Path, root: Path, qdrant: Qdrant | None,
                 embedder: Embedder | None, dry_run: bool,
                 book_id: str | None = None, on_progress=None) -> dict:
    """Index one parquet file.

    Returns a dict: status ('indexed' | 'skipped' | 'dry' | 'not_a_book'),
    book_id, title, pages_total, pages_indexed.
    Raises on any read/embed/upsert error (handled by the caller).
    book_id: pre-assigned by main() (disambiguated when several parquets
    share the same title); None -> derive from the title/authors/pub_date.
    on_progress(done_pages, total_pages) is called after every embed batch
    (page-level live progress for the dashboard).
    """
    # Non-book parquets that live in the tree (recipes.parquet /
    # ingredients.parquet from extract_recipes.py, ...) have no book
    # columns. Skip them instead of treating them as failures.
    schema_names = set(pq.read_schema(path).names)
    if "final_title" not in schema_names:
        return {"status": "not_a_book", "book_id": None, "title": path.name,
                "pages_total": 0, "pages_indexed": 0}

    table = pq.read_table(path, columns=PAGE_COLUMNS)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("parquet contains no rows")

    r0 = rows[0]
    meta = {k: (str(r0[k]) if r0[k] is not None else "") for k in META_KEYS}
    if book_id is None:
        book_id = book_id_of(meta["final_title"], meta["authors"], meta["pub_date"])
    ddc_coarse, ddc_tens = ddc_dirs(path, root)

    # Keep only content pages (is_index marks TOC/index pages to exclude),
    # and drop empty-text pages: without image ingestion a page that only
    # contains a figure (scanned page, image_path set but text blank) has
    # nothing to embed and would only pollute search results.
    content = [r for r in rows if r.get("is_index") is False]
    pages = [r for r in content if (r.get("text") or "").strip()]

    result = {
        "status": "dry" if dry_run else "indexed",
        "book_id": book_id,
        "title": meta["final_title"],
        "pages_total": len(rows),
        "pages_indexed": len(pages),
        "pages_skipped_empty": len(content) - len(pages),
        "replaced_points": 0,
    }

    # Skip only when Qdrant holds exactly what this pipeline would write.
    # Fewer points => a run died mid-upsert (partial book); more points =>
    # an older pipeline version (e.g. empty-text pages were embedded). In
    # both cases a plain re-upsert would leave orphan points behind, so the
    # book's points are deleted and rebuilt from scratch.
    have = qdrant.book_exists_count(book_id)
    if have > 0 and have == len(pages):
        result["status"] = "skipped"
        return result
    if have > 0:
        result["replaced_points"] = have
        if not dry_run:
            qdrant.delete_book_points(book_id)

    if dry_run:
        return result

    if not pages:
        # Image-only book: nothing to embed without image ingestion.
        # Logged, not a failure; it will be re-checked on the next run.
        result["status"] = "empty"
        return result

    texts = [r["text"] for r in pages]  # all non-empty strings (filtered above)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embedder.embed(texts[i:i + EMBED_BATCH]))
        if on_progress:
            on_progress(min(i + EMBED_BATCH, len(texts)), len(texts))

    points = []
    for page, vector in zip(pages, vectors):
        points.append({
            "id": point_id_for(book_id, int(page["page_num"])),
            "vector": vector,
            "payload": {
                "text": page["text"] or "",
                "page_num": int(page["page_num"]),
                "book_id": book_id,
                "book_title": meta["final_title"],
                "authors": meta["authors"],
                "publisher": meta["publisher"],
                "pub_date": meta["pub_date"],
                "ddc": meta["ddc"],
                "ddc_coarse": ddc_coarse,
                "ddc_tens": ddc_tens,
                # Self-describing hits: source kind + local pointer, so any
                # client can go from a search hit to the original file
                # without a separate mapping service. (Points written before
                # these fields exist still resolve via the viewer's local
                # book index.)
                "source_type": "book",
                "path": str(path.relative_to(root)),
            },
        })

    for i in range(0, len(points), UPSERT_BATCH):
        qdrant.upsert(points[i:i + UPSERT_BATCH])

    return result


def record_failure(failed_path: Path, filename: str, error: Exception) -> None:
    rec = {
        "filename": filename,
        "error": f"{type(error).__name__}: {error}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(failed_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index DDC book parquets into Qdrant.")
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="root directory to scan for *.parquet (default: ~/DDC)")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N files (0 = all)")
    p.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                   help="only process files whose name contains SUBSTR "
                        "(may be given multiple times)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would happen without embedding or upserting")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        log.error("root directory does not exist: %s", root)
        return 2
    embed_base = os.environ.get("EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL)
    model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    qdrant_url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)

    files = sorted(root.rglob("*.parquet"))
    if args.only:
        files = [f for f in files if any(s in f.name for s in args.only)]
        log.info("Filter --only %s -> %d file(s)", args.only, len(files))
    if args.limit > 0:
        files = files[:args.limit]
    log.info("Found %d parquet file(s) under %s%s",
             len(files), root, f", limiting to first {args.limit}" if args.limit else "")
    # Unique per-file book ids (disambiguates parquets that share a title).
    file_book_ids = assign_book_ids(files, root)

    # --- live progress for the dashboard (~/base/gui + /api/pipeline) -----
    # Per-file EMBEDDABLE page counts (content pages with non-empty text —
    # exactly what the pipeline embeds): the true total workload, known
    # before the first embed, so the ETA is in pages, not books. Only BOOK
    # files count — the tree also holds non-book parquets (e.g. recipe/
    # ingredient outputs in a sub-dir, ~570k rows here) that are skipped as
    # not_a_book; including them would inflate the total ~3x and wreck the
    # ETA. (~7 s for the whole tree.)
    progress_path = progress_path_for(root)
    file_pages = [embeddable_page_count(f) for f in files]
    all_pages = sum(file_pages)
    # "Real work" per file, refined by the Qdrant pre-scan below to 0 for
    # files already fully indexed — those skip in milliseconds, so counting
    # their pages in "remaining" would inflate the ETA ~3x during the skip
    # burst. Until the pre-scan runs, everything counts as work (upper
    # bound, so the first emitted file is conservative).
    file_work = list(file_pages)
    suffix_pages = [0] * (len(files) + 1)
    for _j in range(len(files) - 1, -1, -1):
        suffix_pages[_j] = suffix_pages[_j + 1] + file_work[_j]

    samples: list = []          # [ts, pages_embedded] for the pace window
    book_samples: list = []     # [ts, books_done] for the book-level pace
    prog = {
        "process": "indexer", "root": str(root),
        "state": "running",
        "total_books": len(files), "done_books": 0,
        "total_pages": all_pages, "pages_done": 0, "pages_embedded": 0,
        "remaining_pages": all_pages,
        "current_book": None, "current_book_pages": 0,
        "current_book_pages_done": 0,
        "counts": {}, "samples": samples, "book_samples": book_samples,
    }

    def emit() -> None:
        write_progress(progress_path, prog)

    def finish(state: str) -> None:
        prog["state"] = state
        prog["current_book"] = None
        emit()

    if not files:
        finish("done")
        return 0
    emit()

    qdrant = Qdrant(qdrant_url)
    if not qdrant.healthy():
        log.error("Qdrant not reachable at %s (start it with: cd ~/DDC/indexer && docker compose up -d)",
                  qdrant_url)
        finish("failed")
        return 2

    # Work pre-scan: ask Qdrant which books already hold EXACTLY what this
    # pipeline would write (the same check process_file uses to skip).
    # Those files skip in milliseconds — drop them from "pages left" so the
    # ETA reflects the work that is actually remaining, not the skip burst.
    # Fail-soft: an unknown count just means the file keeps counting as
    # work. A prediction can rarely miss (e.g. a count that looks exact but
    # is mid-settling) — then one book's pages get embedded anyway while
    # already counted as 0; the error self-heals on the next run.
    n_skip_files = 0
    for i, f in enumerate(files):
        bid = file_book_ids.get(f)
        if file_pages[i] == 0 or bid is None:
            file_work[i] = 0
            continue
        try:
            have = qdrant.book_exists_count(bid)
        except Exception:  # noqa: BLE001 - unknown: count as work
            continue
        if have > 0 and have == file_pages[i]:
            file_work[i] = 0
            n_skip_files += 1
    for _j in range(len(files) - 1, -1, -1):
        suffix_pages[_j] = suffix_pages[_j + 1] + file_work[_j]
    n_book_files = sum(1 for p in file_pages if p > 0)
    prog["remaining_pages"] = sum(file_work)
    log.info("Work pre-scan: %d of %d book files already fully indexed (will "
             "skip); ~%d of %d embeddable pages are real work",
             n_skip_files, n_book_files, sum(file_work), all_pages)
    emit()

    embedder: Embedder | None = None
    if not args.dry_run:
        try:
            embedder = Embedder(embed_base, model)
            embedder.model_available()
        except (requests.RequestException, KeyError) as e:
            log.error("Ollama not usable at %s: %s", embed_base, e)
            log.error("Start it (pick a free GPU) via the GUI at http://localhost:8070, "
                      "or manually: OLLAMA_NUM_PARALLEL=1 ollama serve")
            finish("failed")
            return 2
        if not embedder.model_available():
            log.error("Model '%s' not found in Ollama at %s. Pull it with: ollama pull %s",
                      model, embed_base, model)
            finish("failed")
            return 2
        qdrant.ensure_collection()
    else:
        # Dry runs still do read-only book_exists checks (so the summary's
        # "already indexed" count is accurate); Qdrant must be up.
        log.info("[dry-run] no embedding, no upserts")

    failed_path = Path(__file__).resolve().parent / "failed.jsonl"
    n = len(files)
    t0 = time.monotonic()
    counts = {"indexed": 0, "skipped": 0, "dry": 0, "empty": 0,
              "not_a_book": 0, "failed": 0}
    total_pages = 0
    total_replaced = 0
    eta_shown = False

    _throttle = {"t": 0.0}

    def on_progress(done, total):
        # Page-level progress inside the current book (throttled: an embed
        # batch every few seconds is plenty for a 5s-refreshing dashboard).
        prog["current_book_pages_done"] = done
        if time.time() - _throttle["t"] >= 3.0:
            _throttle["t"] = time.time()
            emit()

    for i, path in enumerate(files, 1):
        prog["current_book"] = path.name
        prog["current_book_pages"] = file_pages[i - 1]
        prog["current_book_pages_done"] = 0
        emit()
        try:
            res = process_file(path, root, qdrant, embedder, args.dry_run,
                               book_id=file_book_ids.get(path),
                               on_progress=on_progress)
            counts[res["status"]] += 1
            if res["status"] == "indexed":
                total_pages += res["pages_indexed"]
                total_replaced += res.get("replaced_points", 0)
            if res["status"] == "skipped":
                log.info('Book %d/%d: skip: already indexed -- "%s"', i, n, res["title"])
            elif res["status"] == "dry":
                extra = (f", would replace {res['replaced_points']} stale points"
                         if res.get("replaced_points") else "")
                log.info('Book %d/%d [DRY-RUN]: "%s" book_id=%s (%d pages, %d to index%s)',
                         i, n, res["title"], res["book_id"],
                         res["pages_total"], res["pages_indexed"], extra)
            elif res["status"] == "empty":
                log.info('Book %d/%d: skip: image-only book, no text pages -- "%s"',
                         i, n, res["title"])
            elif res["status"] == "not_a_book":
                log.info('Book %d/%d: skip: not a book parquet (no final_title) -- %s',
                         i, n, path.name)
            else:
                extra = (f", {res['pages_skipped_empty']} empty-text skipped"
                         if res.get("pages_skipped_empty") else "")
                if res.get("replaced_points"):
                    extra += (f", replaced {res['replaced_points']} "
                              f"stale/partial points")
                log.info('Book %d/%d: "%s" (%d pages, %d indexed%s)',
                         i, n, res["title"], res["pages_total"],
                         res["pages_indexed"], extra)
        except Exception as e:  # noqa: BLE001 - one bad file must not kill the run
            counts["failed"] += 1
            record_failure(failed_path, str(path), e)
            log.error("Book %d/%d FAILED: %s -- %s", i, n, path.name, e)
            res = {"status": "failed", "pages_indexed": 0}

        prog["done_books"] = i
        prog["counts"] = dict(counts)
        prog["pages_done"] += file_pages[i - 1]
        if res["status"] == "indexed":
            prog["pages_embedded"] += res["pages_indexed"]
        prog["remaining_pages"] = suffix_pages[i]
        samples.append([time.time(), prog["pages_embedded"]])
        del samples[:-64]
        book_samples.append([time.time(), i])
        del book_samples[:-64]
        emit()

        # ETA based on the first few indexed books
        if not args.dry_run and not eta_shown and counts["indexed"] == ESTIMATE_AFTER:
            per_book = (time.monotonic() - t0) / counts["indexed"]
            remaining = n - i
            if remaining > 0:
                eta_min = per_book * remaining / 60
                log.info("Pace: ~%.0fs/book -> estimated ~%.0f min for %d remaining file(s)",
                         per_book, eta_min, remaining)
                eta_shown = True

    elapsed = time.monotonic() - t0
    log.info("-" * 60)
    if args.dry_run:
        log.info("Dry run complete: %d file(s) would be indexed, "
                 "%d already indexed, %d image-only, %d not book parquets, %d failed",
                 counts["dry"], counts["skipped"], counts["empty"],
                 counts["not_a_book"], counts["failed"])
    else:
        try:
            points = qdrant.total_points()
        except requests.RequestException:
            points = -1
        log.info("Done: %d book(s) indexed, %d skipped (already indexed), "
                 "%d image-only, %d not book parquets, %d failed, "
                 "%d pages indexed, %d stale point(s) replaced, "
                 "in %.1f min (%.1f s/book)",
                 counts["indexed"], counts["skipped"], counts["empty"],
                 counts["not_a_book"],
                 counts["failed"],
                 total_pages, total_replaced, elapsed / 60,
                 (elapsed / counts["indexed"]) if counts["indexed"] else 0.0)
        if points >= 0:
            log.info("Qdrant collection '%s' now holds %d point(s)", COLLECTION, points)
    if counts["failed"]:
        log.info("Failed files recorded in %s", failed_path)
    rc = 1 if counts["failed"] == n and n > 0 else 0
    finish("done" if rc == 0 else "failed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
