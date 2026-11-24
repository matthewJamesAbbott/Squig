#!/usr/bin/env python3
"""
Validate the DDC tree (stage 4 of the pipeline).

Checks every book parquet under the destination root:
  1. schema: the 16-column DDC layout, in order
  2. pages: at least one row
  3. source_file: the sibling PDF exists next to the parquet
  4. images: every image_path / image_name reference resolves to a real file
  5. metadata: ddc is numeric, final_title is not Unknown
  6. duplicates: same (final_title, authors, pub_date) appearing more than
     once in the tree (different files that are really the same book)

Read-only. Exit code 0 = all clean, 1 = problems found.

Usage:  validate_ddc_tree.py [dest_root]
"""

import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = "/home/matt/DDC"

SCHEMA = ["source_file", "book_title", "page_num", "page_width",
          "page_height", "text", "image_bbox", "image_path", "image_name",
          "final_title", "authors", "publisher", "pub_date", "ddc",
          "summary", "is_index"]


def aslist(v):
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return [v]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    if not root.is_dir():
        print(f"No such directory: {root}")
        sys.exit(1)

    parquets = sorted(root.rglob("*.parquet"))
    books = 0
    clean = 0
    failed = []
    identities = {}

    for f in parquets:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            failed.append((f, [f"unreadable parquet: {e}"]))
            continue
        if "final_title" not in df.columns:
            continue  # not a book (e.g. intermediate/ingredient parquet)

        books += 1
        issues = []

        if list(df.columns) != SCHEMA:
            extra = [c for c in df.columns if c not in SCHEMA]
            missing = [c for c in SCHEMA if c not in df.columns]
            issues.append(f"schema drift (missing {missing}, extra {extra})")
        if len(df) == 0:
            issues.append("zero pages")

        sf = str(df["source_file"].iloc[0]) if "source_file" in df.columns else ""
        if sf and not (f.parent / sf).exists():
            issues.append(f"source_file {sf!r} missing next to parquet")

        ddc = str(df["ddc"].iloc[0]) if "ddc" in df.columns else "Unknown"
        if not re.fullmatch(r"\d{1,4}(\.\d+)?", ddc):
            issues.append(f"ddc not numeric: {ddc!r}")

        title = str(df["final_title"].iloc[0]) if "final_title" in df.columns else ""
        if title == "Unknown":
            issues.append("final_title is Unknown")
        if title and title != "Unknown":
            key = (title.lower(),
                   str(df["authors"].iloc[0]).lower() if "authors" in df.columns else "",
                   str(df["pub_date"].iloc[0]) if "pub_date" in df.columns else "")
            identities.setdefault(key, []).append(f.name)

        refs = miss = 0
        img_dir = f.parent / f.stem  # per-book image dir (same base name)
        for col in ("image_path", "image_name"):
            if col not in df.columns:
                continue
            for val in df[col]:
                for p in aslist(val):
                    if not p:
                        continue
                    refs += 1
                    target = Path(p) if Path(p).is_absolute() else f.parent / p
                    if col == "image_name" and not target.exists():
                        target = img_dir / p  # bare names live in the book dir
                    if not target.exists():
                        miss += 1
        if miss:
            issues.append(f"{miss}/{refs} image references missing on disk")

        if issues:
            failed.append((f, issues))
        else:
            clean += 1

    dupes = {k: v for k, v in identities.items() if len(v) > 1}

    print(f"Tree       : {root}")
    print(f"Books      : {books}")
    print(f"Clean      : {clean}")
    print(f"With issues: {len(failed)}")
    for f, issues in failed:
        rel = f.relative_to(root)
        print(f"  FAIL {rel}")
        for i in issues:
            print(f"       - {i}")
    print(f"\nDuplicate identities (same book, multiple files): {len(dupes)}")
    for (t, a, d), names in dupes.items():
        print(f"  DUP  {t!r} / {a!r} / {d!r}:")
        for n in names:
            print(f"       - {n}")

    sys.exit(0 if not failed and not dupes else 1)


if __name__ == "__main__":
    main()
