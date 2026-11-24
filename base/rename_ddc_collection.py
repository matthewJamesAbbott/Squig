#!/usr/bin/env python3
"""
Rename and classify files from a source directory into a DDC-structured destination.

Naming convention: DDC - title - publisher - date - author.<suffix>

Pipeline position: this is the CLASSIFICATION stage. Parquets must have
been enriched first (enrich_ddc_collection.py adds final_title/authors/
publisher/pub_date/ddc/summary/is_index); parquets without those columns
are skipped with a message pointing at the enrichment stage.

For each source category directory, the script:
  1. Validates + normalizes metadata from parquet files
  2. Builds a new normalized name from metadata fields
  3. Checks destination for collisions (skip unless --overwrite)
  4. Renames PDF, parquet, image files, and index text files
  5. Updates parquet columns (source_file, image_name, image_path -> full destination paths)
  6. Copies everything into the correct DDC classification subdirectory
  7. Post-copy validation: verifies all destination files exist
"""

import os
import sys
import re
import shutil
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – adjust these for each source category
# ---------------------------------------------------------------------------

# Source root: the directory containing .pdf, .parquet, image dirs, and indexes/
SOURCE_ROOT = Path("/home/matt/converted/Dictionaries")

# Destination root: the DDC classification tree
DEST_ROOT = Path("/home/matt/DDC")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_name(name: str, max_len: int = 200) -> str:
    """Remove characters illegal for filenames and shorten."""
    name = name.strip()
    name = name.replace("/", "-").replace("\\", "-")
    name = name.replace(":", "-").replace("?", "")
    name = name.replace("*", "").replace('"', "'")
    name = name.replace("<", "").replace(">", "")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("- ").strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip("- ")
    return name


def build_new_name(ddc, title, publisher, author, date) -> str:
    """Build the new filename: DDC - title - publisher - author - date"""
    parts = []
    for val in [ddc, title, publisher, author, date]:
        s = str(val).strip() if val and str(val) != "Unknown" else ""
        if s:
            parts.append(s)

    if not parts:
        parts = ["Unknown"]

    return sanitize_name(" - ".join(parts))


def is_nonempty(val):
    """Check if a value (possibly array/list/scalar) is non-empty."""
    if isinstance(val, (list, np.ndarray)):
        return len(val) > 0
    return bool(val)


def norm_meta(val) -> str:
    """Normalize a metadata value into a predictable string for filenames.

    Handles str, list/tuple/np arrays (pyarrow lists via as_py()), numpy
    scalars, NaN and None. Multi-value fields are comma-joined.
    Missing/unknown -> "Unknown".
    """
    if val is None:
        return "Unknown"
    as_py = getattr(val, "as_py", None)
    if callable(as_py):
        try:
            val = as_py()
        except Exception:
            pass
    if isinstance(val, float) and pd.isna(val):
        return "Unknown"
    if isinstance(val, (list, tuple, np.ndarray)):
        parts = [norm_meta(v) for v in val]
        parts = [p for p in parts if p != "Unknown"]
        return ", ".join(parts) if parts else "Unknown"
    s = re.sub(r"\s+", " ", str(val).strip())
    return s if s and s.lower() not in ("unknown", "nan") else "Unknown"


def ddc_span(num: int) -> int:
    """The range a DDC node covers. Century nodes (100, 200, ...) span 100;
    tens nodes (130, 520, ...) span 10; anything deeper spans 1.
    (0 means the 000 century, since int() drops the zero padding.)"""
    if num == 0:
        return 100
    s = str(num)
    if num % (10 ** (len(s) - 1)) == 0:
        return 10 ** (len(s) - 1)
    return 10


def find_ddc_directory(dest_root: Path, ddc_number: str) -> Path | None:
    """
    Find the matching DDC directory by truncating to 10-quantized level.

    DDC tree structure:
        DEST_ROOT/
          500_Science/
            520_Astronomy/
            530_Physics/

    For DDC 523 -> truncate to 520 -> find 520_Astronomy/
    For DDC 125 -> truncate to 120 -> find 120_Epistemology/
    For DDC 520 -> match 520 exactly -> find 520_Astronomy/

    Truncation: 523 -> 520, 115 -> 110, 100 -> 100
    Formula: int(ddc) // 10 * 10, zero-padded to 3 digits.
    """
    ddc_str = str(ddc_number).strip()
    if not ddc_str or ddc_str == "Unknown" or not ddc_str.replace(".", "").replace("-", "").isdigit():
        return None

    # Take the integer part (before any decimal)
    ddc_int_str = ddc_str.split(".")[0].split("-")[0]
    ddc_int = int(ddc_int_str)

    # Truncate to nearest 10: 523 -> 520, 115 -> 110, 100 -> 100
    truncated = ddc_int // 10 * 10
    truncated_str = f"{truncated:03d}"

    # Collect candidate directories named <num>_*, anywhere in the tree.
    # Only accept candidates whose numeric ancestor chain is a valid DDC
    # hierarchy for this book (each ancestor must cover the next level),
    # so a duplicate DDC number sitting under the wrong century can't win.
    def candidates_for(num_str: str):
        found = []
        for entry in dest_root.rglob("*"):
            if not entry.is_dir():
                continue
            m = re.match(r"^(\d+)_", entry.name)
            if not m or m.group(1) != num_str:
                continue
            rel = entry.relative_to(dest_root)
            nums = []
            for part in rel.parts:
                pm = re.match(r"^(\d+)_", part)
                if not pm:
                    nums = None
                    break
                nums.append(int(pm.group(1)))
            if not nums or nums[-1] != int(num_str):
                continue
            if all(a <= b < a + ddc_span(a) for a, b in zip(nums, nums[1:])):
                found.append(entry)
        return found

    # Tens-level match first (523 -> 520); fall back to century (523 -> 500).
    for num in (truncated, ddc_int // 100 * 100):
        cands = candidates_for(f"{num:03d}")
        if not cands:
            continue
        # Deterministic selection: most specific (deepest) level wins;
        # among equals, alphabetical by full path.
        cands.sort(key=lambda p: (len(p.relative_to(dest_root).parts), str(p)))
        deepest = max(len(p.relative_to(dest_root).parts) for p in cands)
        ties = [p for p in cands if len(p.relative_to(dest_root).parts) == deepest]
        if len(ties) > 1:
            print(f"  WARN: {len(ties)} DDC directories match {num:03d}; "
                  f"choosing {ties[0].name} (deterministic tie-break)")
        return ties[0]

    return None


def find_images_directory(base_dir: Path, pdf_stem: str) -> Path | None:
    """Find the image subdirectory that matches the PDF stem."""
    candidate = base_dir / pdf_stem
    if candidate.is_dir():
        image_files = [f for f in candidate.iterdir() if f.is_file()]
        if image_files:
            return candidate
    return None


def update_list_column(df, col_name, rename_map):
    """
    Update a column that contains lists/arrays of strings.
    For each element in each list, apply the rename_map.
    Returns the updated column as a list.
    """
    results = []
    for val in df[col_name]:
        if isinstance(val, (list, np.ndarray)):
            results.append([rename_map.get(str(v), str(v)) for v in val if v])
        elif isinstance(val, str) and val:
            results.append(rename_map.get(val, val))
        else:
            results.append(val if val is not None else [])
    return results


def process_category(source_dir: Path, dest_root: Path, dry_run: bool = True,
                     overwrite: bool = False):
    """Process one source category directory."""
    print(f"\n{'='*70}")
    print(f"Processing: {source_dir}")
    print(f"{'='*70}")

    parquet_files = list(source_dir.glob("*.parquet"))
    parquet_files.sort()

    if not parquet_files:
        print(f"  No parquet files found in {source_dir}")
        return 0

    processed = 0

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)

        if len(df) == 0:
            print(f"  SKIP (empty parquet): {parquet_path.name}")
            continue

        # The original source file (PDF for marker OCR, or .epub/.txt/.rtf/
        # .docx/.html/.fb2/.chm for the text extractors, or the ORIGINAL
        # file for converter formats — e.g. .djvu/.mobi/.doc) is usually
        # next to the parquet; ocr.py records its full path in the
        # `source_file` column — use that as fallback.
        pdf_path = None
        for ext in (".pdf", ".epub", ".opub", ".txt", ".text", ".textbook",
                    ".rtf", ".docx", ".html", ".htm", ".mht", ".mhtml",
                    ".fb2", ".chm", ".mobi", ".azw", ".azw3", ".kfx",
                    ".lit", ".doc", ".odt", ".ods", ".odp", ".djvu",
                    ".cbz", ".cbr"):
            cand = parquet_path.with_suffix(ext)
            if cand.exists():
                pdf_path = cand
                break
        if pdf_path is None and "source_file" in df.columns:
            cand = str(df["source_file"].iloc[0] or "").strip()
            if cand and Path(cand).exists():
                pdf_path = Path(cand)
        if pdf_path is None:
            print(f"  SKIP (no matching source file): "
                  f"{parquet_path.name}")
            continue

        # ------------------------------------------------------------------
        # Validate + normalize metadata (enrichment stage output)
        # ------------------------------------------------------------------
        if "ddc" not in df.columns:
            print(f"  SKIP (no enrichment columns — run "
                  f"enrich_ddc_collection.py first): {parquet_path.name}")
            continue
        missing = [c for c in ("final_title", "authors", "publisher",
                               "pub_date") if c not in df.columns]
        if missing:
            print(f"  WARN: missing columns {missing}; using fallbacks")

        row = df.iloc[0]
        ddc = norm_meta(row.get("ddc"))
        if not re.fullmatch(r"\d{1,4}(\.\d+)?", ddc):
            print(f"  SKIP (ddc is not numeric: {ddc!r}): {parquet_path.name}")
            continue

        title = norm_meta(row.get("final_title", row.get("book_title", "Unknown")))
        publisher = norm_meta(row.get("publisher"))
        authors = norm_meta(row.get("authors"))
        pub_date = norm_meta(row.get("pub_date"))

        # Long author lists get truncated out of the 200-char filename,
        # eating the year; use "first author et al." instead (the parquet
        # keeps the full author list).
        author_name = authors
        if len(authors) > 60:
            author_name = authors.split(",")[0].strip() + " et al."
        new_base = build_new_name(ddc, title, publisher, author_name, pub_date)

        # Fallback: if all metadata is Unknown, use the original PDF stem
        if new_base == "Unknown":
            new_base = sanitize_name(parquet_path.stem)

        print(f"\n  --- Document ---")
        print(f"  Original PDF   : {parquet_path.stem}")
        print(f"  New base name  : {new_base}")
        print(f"  DDC            : {ddc}")
        print(f"  Title          : {title}")
        print(f"  Publisher      : {publisher}")
        print(f"  Date           : {pub_date}")
        print(f"  Authors        : {authors}")

        # ------------------------------------------------------------------
        # Determine destination DDC directory
        # ------------------------------------------------------------------
        dest_subdir = find_ddc_directory(dest_root, str(ddc))

        if not dest_subdir:
            print(f"  WARN: Could not find DDC directory for {ddc}")
            ddc_hundred = str(ddc)[:1] + "00" if str(ddc).isdigit() else "000"
            dest_subdir = find_ddc_directory(dest_root, ddc_hundred)
            if not dest_subdir:
                dest_subdir = dest_root / f"{ddc}_Unclassified"

        print(f"  Dest directory : {dest_subdir}")

        # ------------------------------------------------------------------
        # Build destination paths
        # ------------------------------------------------------------------
        new_pdf_name = f"{new_base}{pdf_path.suffix.lower()}"
        new_parquet_name = f"{new_base}.parquet"
        dest_pdf = dest_subdir / new_pdf_name
        dest_parquet = dest_subdir / new_parquet_name

        # ------------------------------------------------------------------
        # Find and rename image directory + images
        # ------------------------------------------------------------------
        pdf_stem = parquet_path.stem
        img_dir = find_images_directory(source_dir, pdf_stem)
        img_prefix = ""
        if img_dir is None:
            # ocr.py layout: every book in this dir shares one images/ dir;
            # files are named <sanitized-stem>_p<page>_img_<xref>.<ext>
            # (same sanitization ocr.py applies to the pdf stem).
            shared = source_dir / "images"
            prefix = re.sub(r"[^\w]", "",
                            pdf_stem.replace("_", " ").replace("-", " ")) + "_"
            if shared.is_dir() and any(f.is_file() and f.name.startswith(prefix)
                                       for f in shared.iterdir()):
                img_dir, img_prefix = shared, prefix

        new_img_dir_name = new_base
        new_img_dir = dest_subdir / new_img_dir_name

        img_rename_map = {}

        if img_dir:
            # In the shared images/ layout (ocr.py) img_prefix filters to this
            # book's images only; in a per-book dir it is "" (no filtering).
            img_files = [f for f in img_dir.iterdir()
                         if f.is_file() and f.name.startswith(img_prefix)]
            print(f"  Image directory: {img_dir.name} ({len(img_files)} images)")

            for img_file in img_files:
                old_name = img_file.name
                match = re.match(
                    r"^(?:.*?_)?p(\d+)_img_(\d+)\.(png|jpg|jpeg|tiff?|gif)$",
                    old_name, re.IGNORECASE
                )
                if match:
                    page_num = match.group(1)
                    img_num = match.group(2)
                    ext = match.group(3)
                    new_img_name = f"{new_base}_p{page_num}_img_{img_num}.{ext}"
                else:
                    ext = old_name.rsplit(".", 1)[-1]
                    new_img_name = f"{new_base}_{old_name}"

                img_rename_map[old_name] = new_img_name

            # Update parquet columns for image renames
            if "image_name" in df.columns:
                df["image_name"] = update_list_column(df, "image_name", img_rename_map)

            if "image_path" in df.columns:
                # image_path stores full paths into the SOURCE images dir;
                # rewrite each entry to point at the book's destination
                # image dir so the parquet is self-contained after the copy.
                def _rewrite_paths(v):
                    items = (list(v) if isinstance(v, (list, np.ndarray))
                             else [v] if v is not None and v != "" else [])
                    out = []
                    for p in items:
                        if isinstance(p, str) and p:
                            base = os.path.basename(p)
                            out.append(str(new_img_dir / img_rename_map[base])
                                       if base in img_rename_map else p)
                        else:
                            out.append(p)
                    return out

                df["image_path"] = [_rewrite_paths(v) for v in df["image_path"]]

        else:
            print(f"  No image directory found for {pdf_stem}")

        # ------------------------------------------------------------------
        # Update parquet: source_file column (strip path, keep new name)
        # ------------------------------------------------------------------
        df["source_file"] = new_pdf_name

        # ------------------------------------------------------------------
        # Index text files
        # ------------------------------------------------------------------
        index_dir = source_dir / "indexes"
        new_index_name = f"{new_base}_indexes.txt"

        old_index = index_dir / f"{pdf_stem}_indexes.txt" if index_dir.exists() else None
        dest_index = None

        if old_index and old_index.exists():
            dest_index = dest_subdir / "indexes" / new_index_name
            print(f"  Index file     : {old_index.name} -> {new_index_name}")

        # ------------------------------------------------------------------
        # Collision check: refuse to clobber existing destination files
        # unless --overwrite is given. Checked BEFORE anything is written,
        # so a book's copy is all-or-nothing.
        # ------------------------------------------------------------------
        dest_files = [dest_pdf, dest_parquet]
        if img_dir:
            dest_files += [new_img_dir / n for n in img_rename_map.values()]
        if dest_index:
            dest_files.append(dest_index)
        collisions = [p for p in dest_files if p.exists()]
        if collisions:
            for p in collisions:
                print(f"  COLLISION: {p}")
            if not overwrite:
                print(f"  SKIP (destination exists — use --overwrite to replace)")
                continue
            print(f"  Overwriting {len(collisions)} existing file(s)")

        # ------------------------------------------------------------------
        # Execute operations
        # ------------------------------------------------------------------
        if dry_run:
            print(f"  [DRY RUN] Would copy/rename:")
            print(f"    {pdf_path.name} -> {dest_pdf}")
            print(f"    {parquet_path.name} -> {dest_parquet}")
            if img_dir:
                print(f"    {img_dir.name}/ -> {new_img_dir}/ "
                      f"(rename {len(img_rename_map)} images)")
            if dest_index:
                idx_parent = old_index.parent.name if old_index else ""
                print(f"    {idx_parent}/{old_index.name} -> indexes/{new_index_name}")
        else:
            dest_subdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_path, dest_pdf)
            df.to_parquet(dest_parquet, index=False)

            if img_dir:
                os.makedirs(new_img_dir, exist_ok=True)
                for old_name, new_name in img_rename_map.items():
                    shutil.copy2(img_dir / old_name, new_img_dir / new_name)

            if dest_index:
                os.makedirs(dest_subdir / "indexes", exist_ok=True)
                shutil.copy2(old_index, dest_index)

            # Post-copy validation: every destination asset must exist.
            missing = [p for p in dest_files if not p.exists()]
            if missing:
                print(f"  WARN: {len(missing)} destination file(s) missing after copy:")
                for p in missing[:10]:
                    print(f"    {p}")
            else:
                print(f"  [DONE] Processed to {dest_subdir} "
                      f"({len(dest_files)} files verified)")

        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Rename DDC collection files to standardized naming convention."
    )
    parser.add_argument(
        "--source", "-s", default=None,
        help="Source root directory (default: %(default)s)"
    )
    parser.add_argument(
        "--dest", "-d", default=None,
        help="Destination DDC root (default: %(default)s)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview changes without making them"
    )
    parser.add_argument(
        "--execute", action="store_true", default=False,
        help="Actually perform the rename/copy operations"
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Replace destination files that already exist (default: skip with COLLISION)"
    )

    args = parser.parse_args()

    # resolve(): rewritten image_path values are built from dest_root and
    # must be absolute (the reader/validator resolve stored paths against
    # CWD; relative CLI args would poison every parquet that gets copied).
    source_root = (Path(args.source) if args.source else SOURCE_ROOT).resolve()
    dest_root = (Path(args.dest) if args.dest else DEST_ROOT).resolve()

    dry_run = not args.execute
    if args.dry_run:
        dry_run = True

    if not source_root.is_dir():
        print(f"ERROR: Source directory does not exist: {source_root}")
        sys.exit(1)

    if not dest_root.is_dir():
        if dry_run:
            print(f"NOTE: Destination does not exist yet, will be created on execute: {dest_root}")
        else:
            dest_root.mkdir(parents=True)
            print(f"Created destination: {dest_root}")

    print(f"Source : {source_root}")
    print(f"Dest   : {dest_root}")
    print(f"Mode   : {'DRY RUN' if dry_run else 'EXECUTE'}")

    total = 0
    root_parquet = list(source_root.glob("*.parquet"))
    has_subdirs_with_data = False

    for subdir in source_root.iterdir():
        if subdir.is_dir() and subdir.name != "indexes":
            if list(subdir.glob("*.parquet")):
                has_subdirs_with_data = True
                break

    if root_parquet and not has_subdirs_with_data:
        total = process_category(source_root, dest_root, dry_run, args.overwrite)
    else:
        for subdir in sorted(source_root.iterdir()):
            if not subdir.is_dir() or subdir.name == "indexes":
                continue
            if list(subdir.glob("*.parquet")):
                total += process_category(subdir, dest_root, dry_run, args.overwrite)

        if root_parquet:
            total += process_category(source_root, dest_root, dry_run, args.overwrite)

    print(f"\n{'='*70}")
    print(f"Total documents processed: {total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
