#!/home/matt/miniconda3/envs/marker_env/bin/python
"""One-off v2 migration for the original cheese-aging hits (#11-#15).

Their structured data was buried in `note` (v1 had no page/topic/status
columns). This script:
  * parses pp.X-Y / pp.X,Y,Z / p.X patterns from note + result_summary
    -> page_start / page_end
  #14 (alpine cross-domain): cross_refs -> P. freudenreichii in 579,
                              status='needs-verify'
  #15 (domain map):          topic='cheese,domain-map,ddc-catalog',
                              status='found'
  #12, #13:                  status='verified' (solid), topic from the note
  #11 (tooling test):        status='done'
Idempotent: safe to re-run (fields are set, not appended).
"""
import json
import re

import hits_db


def parse_pages(*texts):
    """(page_start, page_end) from pp.45-47 / pp.39,50,51 / p.45 patterns."""
    for text in texts:
        if not text:
            continue
        # pp.45-47 or pp.39,50,51
        m = re.search(r"pp\.?\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)", text)
        if m:
            chunk = m.group(1)
            if re.search(r"[-–]", chunk):
                a, b = (int(x) for x in re.split(r"[-–]", chunk, maxsplit=1))
                return a, max(a, b)
            nums = [int(x) for x in re.findall(r"\d+", chunk)]
            if nums:
                return min(nums), max(nums)
        # single page: "p.45" (word-boundary p. so '20-24C' can't match)
        m = re.search(r"\bp\.(\d+)", text)
        if m:
            return int(m.group(1)), int(m.group(1))
        # "PAGES: 39 (acid), 50 (...), 51 (...)"
        m = re.search(r"PAGES:\s*([^\n]+)", text)
        if m:
            nums = [int(x) for x in re.findall(r"\b\d{1,4}\b", m.group(1))]
            if nums:
                return min(nums), max(nums)
    return None, None


def main():
    updates = {
        11: {"status": "done"},
        12: {"status": "verified",
             "topic": "affinage,proteolysis,ripening,alpine-cheese,"
                      "propionic-fermentation"},
        13: {"status": "verified",
             "topic": "home-aging,temperature-humidity,turning,waxing,"
                      "practical-parameters"},
        14: {"status": "needs-verify",
             "cross_refs": [{"book_id": None, "page": None, "ddc": "579",
                             "label": "P. freudenreichii taxonomy/physiology"}],
             "topic": "alpine-cheese,eyes,propionic-fermentation,"
                      "p-freudenreichii,cross-domain"},
        15: {"status": "found",
             "topic": "cheese,domain-map,ddc-catalog"},
    }
    rows, _, _ = hits_db.recall_hits(limit=100)
    by_id = {r["id"]: r for r in rows}
    for hid, spec in updates.items():
        row = by_id.get(hid)
        if not row:
            print(f"#{hid}: not found — skipped")
            continue
        page_start, page_end = parse_pages(row.get("note"),
                                           row.get("result_summary"))
        # set the fields with a direct update through the same connection
        # helper (hits_mark only overwrites, it can't set page/topic)
        conn = hits_db._connect()
        try:
            with conn.cursor() as cur:
                fields, args = [], []
                for k, v in (("page_start", page_start),
                             ("page_end", page_end),
                             ("status", spec.get("status")),
                             ("topic", spec.get("topic")),
                             ("cross_refs",
                              (json.dumps(spec["cross_refs"])
                               if spec.get("cross_refs") else None))):
                    if v is not None:
                        fields.append(f"{k}=%s")
                        args.append(v)
                if fields:
                    cur.execute(f"UPDATE hits SET {', '.join(fields)} "
                                f"WHERE id=%s", args + [hid])
        finally:
            conn.close()
        print(f"#{hid}: pages={page_start}-{page_end} "
              f"status={spec.get('status')} topic={spec.get('topic')}")
    print("done — verify with: hits_recall / hits_pending / hits_mark")


if __name__ == "__main__":
    main()
