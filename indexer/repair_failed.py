#!/usr/bin/env python3
"""Repair books that failed in a previous index_books.py run.

A book that failed mid-way may hold *partial* points in Qdrant. Because the
indexer skips any book whose book_id already exists, those partial points
would make a plain re-run skip the very books we want to (re)index.

This script:
  1. reads the unique failed filenames from failed.jsonl
  2. recomputes each book's book_id (identically to the indexer)
  3. reports how many points each book already has in Qdrant
  4. deletes those points (default; --no-delete to plan only)
  5. clears failed.jsonl (default; --keep-log to preserve it)

Then re-run:  python index_books.py --root ~/DDC
It will index exactly the repaired books; all others are skipped
(already indexed) and finish in a few minutes.

Usage:
    python repair_failed.py                    # delete partial points + clear log
    python repair_failed.py --no-delete        # just show the plan
    python repair_failed.py --keep-log         # leave failed.jsonl in place
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_books import COLLECTION, META_KEYS, Qdrant, book_id_of  # noqa: E402


def read_meta(path: Path) -> dict | None:
    """Return the meta dict, or None if the file is not a book parquet."""
    if "final_title" not in set(pq.read_schema(path).names):
        return None
    table = pq.read_table(path, columns=list(META_KEYS))
    row = table.to_pylist()[0]
    return {k: (str(row[k]) if row[k] is not None else "") for k in META_KEYS}


def main() -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Re-queue failed books for re-indexing.")
    p.add_argument("--failed-file", default=str(here / "failed.jsonl"))
    p.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--no-delete", action="store_true",
                   help="show the plan but do not delete any points")
    p.add_argument("--keep-log", action="store_true",
                   help="do not clear failed.jsonl after processing")
    args = p.parse_args()

    failed_path = Path(args.failed_file)
    if not failed_path.exists():
        print(f"No failure log at {failed_path} - nothing to repair.")
        return 0
    recs = [json.loads(line) for line in failed_path.read_text().splitlines() if line.strip()]
    files: list[str] = []
    for r in recs:
        if r["filename"] not in files:
            files.append(r["filename"])
    print(f"{len(recs)} failure record(s) -> {len(files)} unique book(s)")
    if not files:
        return 0

    q = Qdrant(args.qdrant_url)
    bad = 0
    total_deleted = 0
    for f in files:
        path = Path(f)
        try:
            meta = read_meta(path)
        except Exception as e:
            print(f"  READ-FAIL {path.name} -- {e}")
            bad += 1
            continue
        if meta is None:
            print(f"  IGNORED  {path.name}  (not a book parquet, indexer now skips it)")
            continue
        bid = book_id_of(meta["final_title"], meta["authors"], meta["pub_date"])
        title = (meta["final_title"] or "?")[:64]
        try:
            before = q.book_exists_count(bid)
        except Exception as e:
            print(f"  COUNT-FAIL {path.name} -- {e}")
            bad += 1
            continue
        if before:
            if args.no_delete:
                print(f"  {title:64s} {bid}  {before} partial points  (would delete)")
            else:
                try:
                    q.delete_book_points(bid)
                    after = q.book_exists_count(bid)
                    print(f"  {title:64s} {bid}  deleted {before} partial points "
                          f"-> {after} remaining")
                    total_deleted += before - after
                except Exception as e:
                    print(f"  DELETE-FAIL {path.name} -- {e}")
                    bad += 1
                    continue
        else:
            print(f"  {title:64s} {bid}  no points in Qdrant  (nothing to delete)")

    print(f"\n{len(files) - bad}/{len(files)} book(s) ready for re-indexing, "
          f"{total_deleted} partial point(s) deleted.")
    if bad:
        print(f"WARNING: {bad} book(s) had errors - fix and re-run this script.")

    if not args.no_delete and not args.keep_log and not bad:
        failed_path.write_text("")
        print(f"Cleared {failed_path.name} (re-run the indexer now: "
              f"python index_books.py --root ~/DDC)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
