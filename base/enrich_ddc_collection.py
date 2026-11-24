#!/usr/bin/env python3
"""
Enrich converted OCR parquets IN PLACE with book metadata.

One LLM call per book adds: final_title, authors, publisher, pub_date,
ddc, summary.  is_index (TOC / front & back matter) is detected locally
with a text heuristic — no LLM involved.

Output schema matches the DDC tree (16 columns) that index_books.py and
the reader expect:
    source_file, book_title, page_num, page_width, page_height, text,
    image_bbox, image_path, image_name, final_title, authors, publisher,
    pub_date, ddc, summary, is_index

Usage:
    enrich_ddc_collection.py [input_dir] [--api-base URL] [--model NAME]
                             [--dry-run] [--category NAME]

input_dir defaults to /home/matt/converted; parquets are found
recursively (images/ subdirs are not parquets, so they're ignored).
Resume: a parquet is skipped when it already has all 5 META columns and
ddc is not "Unknown".
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = "/home/matt/converted"
DEFAULT_API_BASE = os.environ.get("VLLM_BASE", "http://10.90.38.19:8000/v1")
DEFAULT_MODEL = "squigFast"

META_KEYS = ("final_title", "authors", "publisher", "pub_date", "ddc")
SCHEMA_ORDER = ["source_file", "book_title", "page_num", "page_width",
                "page_height", "text", "image_bbox", "image_path",
                "image_name", "final_title", "authors", "publisher",
                "pub_date", "ddc", "summary", "is_index"]
DEFAULTS = {"final_title": "Unknown", "authors": "Unknown",
            "publisher": "Unknown", "pub_date": "Unknown",
            "ddc": "Unknown", "summary": ""}

CONTEXT_PAGES = 20   # first N pages of text sent to the LLM
MAX_WORDS = 1500     # word cap on that combined sample
MAX_RETRIES = 3
RETRY_DELAY = 2

PROMPT = """You are extracting metadata for one book. The text sample is from the beginning of the book; the filename may also contain useful information.

Filename: {filename}

Return ONLY a JSON object with exactly these keys:
{{"final_title": "...", "authors": "...", "publisher": "...", "pub_date": "...", "ddc": "...", "summary": "..."}}

Rules:
- final_title: the real book title. No series name, no year, no publisher, no edition.
- authors: as printed on the title page, comma-separated.
- publisher: as printed. No location.
- pub_date: a 4-digit year if visible, otherwise "Unknown".
- ddc: a 3-digit Dewey Decimal classification for this book (e.g. 150 Psychology, 155 Social psychology, 370 Education). If unsure, give the closest plausible 3-digit code.
- summary: one or two neutral sentences describing what the book covers and its audience.
- If a field cannot be determined, use "Unknown" (for summary, an empty string).

Text Sample:
{text}"""

# ── is_index heuristic (local, no LLM) ───────────────────────────────────────
# A page is TOC/front/back-matter when:
#  1. it sits in the first quarter of the book and contains at least 3
#     dot-leader + page-number sequences ("..... 12") anywhere in its text
#     (TXT/EPUB TOCs often pack many entries on one line), or its first line
#     is a CONTENTS/INDEX/PLATES-style heading; or
#  2. it is at/after a standalone "INDEX" heading in the last fifth of the
#     book (everything from there on is the index section).
LEADER_ANY_RE = re.compile(r"\.{3,}\s*\d{1,4}\b")
FRONT_HEADING_RE = re.compile(
    r"^\s*(CONTENTS?|INDEX|SUBJECT INDEX|PLATES|LIST OF (FIGURES|TABLES|"
    r"PLATES|ABBREVIATIONS|SYMBOLS))\b", re.I)
BACK_HEADING = {"INDEX", "SUBJECT INDEX", "NAME INDEX", "GENERAL INDEX"}


def compute_is_index(texts):
    """texts: list of page strings (may contain None). Returns list[bool]."""
    n = len(texts)
    flags = [False] * n
    if n == 0:
        return flags

    front_end = max(10, n // 4)
    back_start = n - max(5, n // 5)
    index_started = False

    for i, raw in enumerate(texts):
        t = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if not lines:
            continue

        if i < front_end and not index_started:
            stripped = lines[0].strip()
            hits = len(LEADER_ANY_RE.findall(t))
            if hits >= 3:
                flags[i] = True
            elif FRONT_HEADING_RE.match(stripped) and len(t) < 3000:
                flags[i] = True

        if i >= back_start:
            if not index_started:
                first = lines[0].strip().upper()
                if first in BACK_HEADING or first.startswith("INDEX OF"):
                    index_started = True
            if index_started:
                flags[i] = True

    return flags


# ── LLM ──────────────────────────────────────────────────────────────────────
def get_model_name(client, requested):
    if requested and requested != "auto":
        return requested
    try:
        models = client.models.list()
        return models.data[0].id if models.data else DEFAULT_MODEL
    except Exception as e:
        print(f"  WARNING: model autodetect failed ({e}); using {DEFAULT_MODEL}")
        return DEFAULT_MODEL


def combine_sample_text(df, pages=CONTEXT_PAGES, max_words=MAX_WORDS):
    texts = df.iloc[:pages]["text"].tolist()
    valid = [t for t in texts if pd.notna(t) and isinstance(t, str) and t.strip()]
    words = " ".join(valid).split()
    return " ".join(words[:max_words])


def call_llm(client, model, filename, text_sample):
    prompt = PROMPT.format(filename=filename, text=text_sample)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            msg = resp.choices[0].message
            # reasoning models carry an extra `reasoning` field; the answer
            # is always in `content`
            content = (msg.content or "").strip()
            if not content:
                content = (getattr(msg, "reasoning", None) or "").strip()

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                start, end = content.find("{"), content.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(content[start:end])
            print(f"  WARNING: no JSON in response: {content[:150]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue  # reasoning models sometimes chatter; retry
        except Exception as e:
            print(f"  attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return dict(DEFAULTS)


def clean_meta(raw):
    meta = {}
    for key in DEFAULTS:
        val = raw.get(key)
        val = "Unknown" if (val is None or val == "nan" or val == "") else str(val).strip()
        meta[key] = val
    if meta["summary"] == "Unknown":
        meta["summary"] = ""
    d = meta["ddc"]
    m = re.search(r"\d{3}", d or "")
    meta["ddc"] = m.group(0) if m else "Unknown"
    return meta


# ── file handling ────────────────────────────────────────────────────────────
def is_done(path: Path) -> bool:
    """A parquet counts as enriched when it has all 5 META columns and
    ddc/final_title are real values (not Unknown)."""
    try:
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(path).names)
        if not set(META_KEYS).issubset(cols):
            return False
        rows = pq.read_table(path, columns=["final_title", "ddc"]).to_pylist()
        if not rows:
            return False
        return (str(rows[0].get("ddc") or "Unknown") != "Unknown"
                and str(rows[0].get("final_title") or "Unknown") != "Unknown")
    except Exception:
        return False


def process_file(path: Path, client, model, dry_run=False):
    name = path.name

    if is_done(path):
        print(f"  SKIP (already enriched): {name}")
        return "skipped"

    df = pd.read_parquet(path)
    if df.empty:
        print(f"  SKIP (empty): {name}")
        return "skipped"

    text_sample = combine_sample_text(df)
    if not text_sample.strip():
        print(f"  SKIP (no text to sample): {name}")
        return "skipped"

    print(f"  -> LLM metadata for: {name}")
    meta = clean_meta(call_llm(client, model, path.stem, text_sample))
    print(f"     title: {meta['final_title']}  |  ddc: {meta['ddc']}  |  {meta['authors']}")

    for key in META_KEYS:
        df[key] = meta[key]
    df["summary"] = meta["summary"]
    df["is_index"] = compute_is_index(df["text"].tolist())

    if dry_run:
        print(f"  DRY: would write {name}")
        return "dry"

    # Reorder to the canonical 16-column DDC schema, then atomic replace.
    cols = [c for c in SCHEMA_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    out = df[cols + extra]
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    n_idx = int(out["is_index"].sum())
    print(f"  OK: {name} (is_index pages: {n_idx}/{len(out)})")
    return "enriched"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("input_dir", nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--category", default="",
                    help="only process <input_dir>/<category>/**")
    args = ap.parse_args()

    root = Path(args.input_dir)
    base = root / args.category if args.category else root
    if not base.is_dir():
        print(f"No such directory: {base}")
        sys.exit(1)

    files = sorted(base.rglob("*.parquet"))
    if not files:
        print(f"No .parquet files under {base}")
        sys.exit(0)
    print(f"Enriching {len(files)} parquet(s) under {base}")
    print(f"API: {args.api_base}  dry_run: {args.dry_run}\n")

    client = None
    model = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(base_url=args.api_base, api_key="not-needed")
        model = get_model_name(client, args.model)
        print(f"Model: {model}\n")

    counts = {}
    for f in files:
        try:
            r = process_file(f, client, model, args.dry_run)
        except Exception as e:
            print(f"  FAILED: {f.name}: {e}")
            r = "failed"
        counts[r] = counts.get(r, 0) + 1

    print(f"\nDone: {counts}")


if __name__ == "__main__":
    main()
