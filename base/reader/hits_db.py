#!/home/matt/miniconda3/envs/marker_env/bin/python
"""Hit database (MySQL) — the two-layer research log for the DDC library.

Two layers, kept separate on purpose:
  layer='user'  — things the USER asked to look at (active work).
  layer='agent' — the SYSTEM's own research (off-time, autonomous roaming).

The point of the split: while you're working on something, that work stays
in its own layer and never gets lost under the agent's own exploration —
and you can audit what the agent did on its own without sifting through
your session.

Every hit = one search/ask/page-read event: what was queried, which tool,
which DDC scope, which books/chunks it touched (hit_refs), what it found,
and optionally the answer + the agent's note on why it searched.

v2 (the autonomous research loop): each hit can now carry
  page_start / page_end  — the page span actually used
  topic                  — comma-separated topic tags
  cross_refs             — JSON array of {"book_id","page","ddc","label"}
                           pointing at other books/domains (the links)
  status                 — found -> needs-verify -> verified -> done
  confidence             — low / medium / high
  session_id             — groups related hits from one research walk
The loop's frontier pull is hits_pending() (status='needs-verify', oldest
first); the loop advances items with hits_mark() without rewriting them.

Env (all optional — the module fails soft so the reader keeps working
before MySQL exists / while it's down):
  HITS_DB_HOST    default 127.0.0.1
  HITS_DB_PORT    default 3306
  HITS_DB_USER    default matt
  HITS_DB_PASSWORD default ""
  HITS_DB_NAME    default squig_research   (created by the user; the `matt`
                   MySQL user holds ALL on it but not global CREATE)
  HITS_DB_TIMEOUT default 5
  HITS_AUTOLOG    default 1  (0 = stop auto-recording searches/asks)
"""
import json
import os
import time

import pymysql

HOST = os.environ.get("HITS_DB_HOST", "127.0.0.1")
PORT = int(os.environ.get("HITS_DB_PORT", "3306"))
USER = os.environ.get("HITS_DB_USER", "matt")
PASSWORD = os.environ.get("HITS_DB_PASSWORD", "")
NAME = os.environ.get("HITS_DB_NAME", "squig_research")
CONNECT_TIMEOUT = int(os.environ.get("HITS_DB_TIMEOUT", "5"))

VALID_STATUS = ("found", "needs-verify", "verified", "done")
VALID_CONFIDENCE = ("low", "medium", "high")

TABLES = f"""
CREATE TABLE IF NOT EXISTS hits (
  id             BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  ts             DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  layer          VARCHAR(8)    NOT NULL DEFAULT 'user',
  session        VARCHAR(80)   NULL,
  query          TEXT          NULL,
  tool           VARCHAR(40)   NULL,
  mode           VARCHAR(16)   NULL,
  ddc            VARCHAR(120)  NULL,
  book_ids       JSON          NULL,
  point_ids      JSON          NULL,
  n_results      INT UNSIGNED  NULL,
  result_summary VARCHAR(500)  NULL,
  note           TEXT          NULL,
  answer         MEDIUMTEXT    NULL,
  -- v2: autonomous research-loop fields
  page_start     INT           NULL,
  page_end       INT           NULL,
  topic          TEXT          NULL,  -- comma-separated tags, e.g. "affinage,proteolysis,alpine-cheese"
  cross_refs     TEXT          NULL,  -- JSON array of {{"book_id","page","ddc","label"}}
  status         TEXT          NOT NULL DEFAULT 'found'
               CHECK (status IN ('found','needs-verify','verified','done')),
  confidence     TEXT          NOT NULL DEFAULT 'medium'
               CHECK (confidence IN ('low','medium','high')),
  session_id     TEXT          NULL,  -- groups related hits from one research walk
  INDEX idx_layer_ts (layer, ts),
  INDEX idx_session (session),
  INDEX idx_ts (ts),
  INDEX idx_hits_status (status),
  INDEX idx_hits_ddc (ddc),
  INDEX idx_hits_topic (topic),
  INDEX idx_hits_session_id (session_id)
) ENGINE=InnoDB;

-- cross-refs: which books / chunks / DDC codes each hit touched.
-- "hits that touched both book A and book B" is a join on this table —
-- the cross-domain links the research is about.
CREATE TABLE IF NOT EXISTS hit_refs (
  hit_id   BIGINT UNSIGNED NOT NULL,
  ref_type VARCHAR(8)      NOT NULL,
  ref_id   VARCHAR(80)     NOT NULL,
  PRIMARY KEY (hit_id, ref_type, ref_id),
  CONSTRAINT fk_refs_hit FOREIGN KEY (hit_id)
    REFERENCES hits(id) ON DELETE CASCADE,
  INDEX idx_ref (ref_type, ref_id)
) ENGINE=InnoDB;
"""

# Upgrade path for DBs created before v2: ADD COLUMN IF NOT EXISTS /
# CREATE INDEX IF NOT EXISTS make these safe no-ops to run at every
# startup. MariaDB notes: no partial indexes (Postgres) and no FTS5
# (SQLite) — status/topic get plain B-tree indexes and the filters use
# = / LIKE. The table stays small (a research log, not a warehouse) and
# the one hot query is status = 'needs-verify', which the index serves.
UPGRADES = [
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS page_start INT NULL",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS page_end INT NULL",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS topic TEXT NULL",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS cross_refs TEXT NULL",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS status TEXT NOT NULL "
    "DEFAULT 'found' CHECK (status IN ('found','needs-verify','verified','done'))",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS confidence TEXT NOT NULL "
    "DEFAULT 'medium' CHECK (confidence IN ('low','medium','high'))",
    "ALTER TABLE hits ADD COLUMN IF NOT EXISTS session_id TEXT NULL",
    "CREATE INDEX IF NOT EXISTS idx_hits_status ON hits(status)",
    "CREATE INDEX IF NOT EXISTS idx_hits_ddc ON hits(ddc)",
    "CREATE INDEX IF NOT EXISTS idx_hits_topic ON hits(topic)",
    "CREATE INDEX IF NOT EXISTS idx_hits_session_id ON hits(session_id)",
]

# every column select-able in the log (used by recall/pending/mark)
_SELECT = """SELECT h.id, h.ts, h.layer, h.session, h.query, h.tool,
                    h.mode, h.ddc, h.book_ids, h.point_ids, h.n_results,
                    h.result_summary, h.note, h.answer,
                    h.page_start, h.page_end, h.topic, h.cross_refs,
                    h.status, h.confidence, h.session_id
             FROM hits h"""

_last_error = {"msg": "not connected yet"}
_schema_ok = False


class HitDBError(Exception):
    """Raised on record/recall failures; carries a human-readable cause."""


def _connect(database: str = NAME):
    return pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD,
        database=database, charset="utf8mb4",
        connect_timeout=CONNECT_TIMEOUT, read_timeout=15,
        write_timeout=15, autocommit=True)


def _parse_rows(rows, cols) -> list[dict]:
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("book_ids", "point_ids", "cross_refs"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, ValueError):
                    pass
        if d.get("answer"):
            d["answer"] = str(d["answer"])[:280]
        d["ts"] = str(d["ts"])
        d["id"] = int(d["id"])
        out.append(d)
    return out


def ensure_schema() -> bool:
    """Best-effort: make sure the database + tables exist, then apply the
    v2 column/index upgrades. Called once at reader startup in a daemon
    thread.

    The usual deployment: the user pre-created the DB (e.g. squig_research)
    and granted the app user ALL on it but not global CREATE, so we connect
    straight to the DB and just create the tables. If the DB does not exist
    yet and the user CAN create databases, we create it; otherwise we report
    the access error and the reader keeps working without the log."""
    global _schema_ok
    try:
        conn = _connect()
    except pymysql.err.OperationalError as e:
        if not (e.args and e.args[0] == 1049):  # 1049 = unknown database
            _last_error["msg"] = str(e).split("\n")[0]
            print(f"[hits] schema ensure failed: {_last_error['msg']}", flush=True)
            return False
        conn = _connect(database="")
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {NAME} "
                        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.close()
        conn = _connect()
    try:
        with conn.cursor() as cur:
            for stmt in TABLES.split(";"):
                s = "\n".join(l for l in stmt.splitlines()
                              if l.strip() and not l.strip().startswith("--"))
                if s.strip():
                    cur.execute(s)
            for stmt in UPGRADES:
                cur.execute(stmt)
        conn.close()
        _schema_ok = True
        _last_error["msg"] = ""
        print(f"[hits] schema ensured in {NAME}@{HOST}:{PORT}", flush=True)
        return True
    except Exception as e:
        _last_error["msg"] = str(e).split("\n")[0]
        print(f"[hits] schema ensure failed: {_last_error['msg']}", flush=True)
        return False


def ping() -> tuple[bool, str]:
    try:
        conn = _connect()
        conn.close()
        _last_error["msg"] = ""
        return True, f"{NAME}@{HOST}:{PORT}"
    except Exception as e:
        _last_error["msg"] = str(e).split("\n")[0]
        return False, _last_error["msg"]


def layer_counts() -> dict:
    """How many hits each layer holds (user / agent)."""
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT layer, COUNT(*) FROM hits GROUP BY layer")
                return {l: n for l, n in cur.fetchall()}
        finally:
            conn.close()
    except Exception as e:
        raise HitDBError(f"{NAME}@{HOST}:{PORT}: {str(e).splitlines()[0]}") from e


def _check_enum(value, valid, what: str):
    if value is None:
        return None
    v = str(value)
    if v not in valid:
        raise HitDBError(f"{what} must be one of {list(valid)} (got {v!r})")
    return v


def record_hit(hit: dict, refs: list[tuple[str, str]] | None = None) -> int:
    """Insert one hit (+ its refs). Returns the new hit id.

    hit keys (all optional): layer, session, query, tool, mode, ddc,
    book_ids, point_ids, n_results, result_summary, note, answer,
    and the v2 loop fields: page_start, page_end, topic, cross_refs
    (list of {book_id,page,ddc,label} dicts -> stored as a JSON string),
    status (found|needs-verify|verified|done, default 'found'),
    confidence (low|medium|high, default 'medium'), session_id.
    refs: list of (ref_type, ref_id), ref_type in book|chunk|ddc.
    """
    layer = str(hit.get("layer") or "user")[:8]
    if layer not in ("user", "agent"):
        layer = "user"
    # or-default: an explicit NULL into a NOT NULL column is error 1048,
    # while omitting the column would apply the DEFAULT — so coerce
    status = _check_enum(hit.get("status"), VALID_STATUS, "status") or "found"
    confidence = (_check_enum(hit.get("confidence"), VALID_CONFIDENCE,
                              "confidence") or "medium")
    cross_refs = hit.get("cross_refs")
    if cross_refs is not None:
        cross_refs = json.dumps(cross_refs)
    page_start = int(hit["page_start"]) if hit.get("page_start") is not None else None
    page_end = int(hit["page_end"]) if hit.get("page_end") is not None else None
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO hits
                       (layer, session, query, tool, mode, ddc, book_ids,
                        point_ids, n_results, result_summary, note, answer,
                        page_start, page_end, topic, cross_refs, status,
                        confidence, session_id)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                               %s,%s,%s,%s,%s,%s,%s)""",
                    (layer,
                     (str(hit.get("session"))[:80] if hit.get("session") else None),
                     hit.get("query"),
                     (str(hit.get("tool"))[:40] if hit.get("tool") else None),
                     (str(hit.get("mode"))[:16] if hit.get("mode") else None),
                     (str(hit.get("ddc"))[:120] if hit.get("ddc") else None),
                     json.dumps(hit.get("book_ids") or None),
                     json.dumps(hit.get("point_ids") or None),
                     int(hit["n_results"]) if hit.get("n_results") is not None else None,
                     (str(hit.get("result_summary"))[:500]
                      if hit.get("result_summary") else None),
                     hit.get("note"),
                     hit.get("answer"),
                     page_start, page_end,
                     hit.get("topic"),
                     cross_refs,
                     status, confidence,
                     (str(hit.get("session_id"))[:200]
                      if hit.get("session_id") else None)))
                hid = cur.lastrowid
                if refs:
                    cur.executemany(
                        "INSERT IGNORE INTO hit_refs (hit_id, ref_type, ref_id) "
                        "VALUES (%s,%s,%s)",
                        [(hid, rt[:8], str(rid)[:80]) for rt, rid in refs])
            return int(hid)
        finally:
            conn.close()
    except Exception as e:
        raise HitDBError(f"{NAME}@{HOST}:{PORT}: {str(e).splitlines()[0]}") from e


def recall_hits(layer: str = "", session: str = "", book: str = "",
                q: str = "", ddc: str = "", status: str = "", topic: str = "",
                cross_ref_book: str = "", session_id: str = "",
                limit: int = 20, offset: int = 0) -> tuple[list[dict], int, dict]:
    """Query the log, newest first.

    Filters: layer (user|agent), session (exact), book (book_id via
    hit_refs), q (substring of the query), ddc (substring),
    status (exact: found|needs-verify|verified|done), topic (substring
    of the comma-separated tags), cross_ref_book (substring match inside
    the cross_refs JSON — matches a book_id, a ddc, or a label value),
    session_id (exact).
    Returns (hits, total, layer_counts)."""
    limit = min(100, max(1, int(limit)))
    offset = max(0, int(offset))
    where, args = ["1=1"], []
    if layer:
        where.append("h.layer = %s"); args.append(layer[:8])
    if session:
        where.append("h.session = %s"); args.append(session[:80])
    if ddc:
        where.append("h.ddc LIKE %s"); args.append(f"%{ddc}%")
    if q:
        where.append("h.query LIKE %s"); args.append(f"%{q}%")
    if status:
        where.append("h.status = %s"); args.append(status[:16])
    if topic:
        where.append("h.topic LIKE %s"); args.append(f"%{topic}%")
    if cross_ref_book:
        where.append("h.cross_refs LIKE %s"); args.append(f"%{cross_ref_book}%")
    if session_id:
        where.append("h.session_id = %s"); args.append(session_id[:200])
    if book:
        where.append("""h.id IN (SELECT hit_id FROM hit_refs
                                  WHERE ref_type='book' AND ref_id=%s)""")
        args.append(book[:80])
    where_sql = " AND ".join(where)
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM hits h WHERE {where_sql}", args)
                total = int(cur.fetchone()[0])
                cur.execute(
                    f"""{_SELECT} WHERE {where_sql}
                        ORDER BY h.ts DESC, h.id DESC
                        LIMIT %s OFFSET %s""",
                    args + [limit, offset])
                rows = _parse_rows(cur.fetchall(), [c[0] for c in cur.description])
                cur.execute("SELECT layer, COUNT(*) FROM hits GROUP BY layer")
                counts = {l: n for l, n in cur.fetchall()}
            return rows, total, counts
        finally:
            conn.close()
    except Exception as e:
        raise HitDBError(f"{NAME}@{HOST}:{PORT}: {str(e).splitlines()[0]}") from e


def hits_pending(limit: int = 10) -> list[dict]:
    """The automation loop's frontier pull: every hit stuck at
    status='needs-verify', oldest first (FIFO — work the queue in the
    order it was created), all fields including cross_refs."""
    limit = min(100, max(1, int(limit)))
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""{_SELECT} WHERE h.status = 'needs-verify'
                        ORDER BY h.ts ASC, h.id ASC LIMIT %s""",
                    (limit,))
                return _parse_rows(cur.fetchall(), [c[0] for c in cur.description])
        finally:
            conn.close()
    except Exception as e:
        raise HitDBError(f"{NAME}@{HOST}:{PORT}: {str(e).splitlines()[0]}") from e


def hits_mark(hit_id: int, status: str | None = None,
              confidence: str | None = None, note_append: str | None = None,
              cross_refs_append: list | None = None) -> dict:
    """Advance one hit without rewriting it (the loop's bookkeeping call).

    status / confidence     — overwrite the column when provided
    note_append             — append to the existing note (\\n separated),
                              e.g. "went back and verified: …"
    cross_refs_append       — merge into the existing cross_refs JSON array,
                              deduped by (book_id, page)
    Returns the updated hit (all fields)."""
    status = _check_enum(status, VALID_STATUS, "status")
    confidence = _check_enum(confidence, VALID_CONFIDENCE, "confidence")
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT note, cross_refs FROM hits WHERE id=%s",
                            (int(hit_id),))
                row = cur.fetchone()
                if not row:
                    raise HitDBError(f"hit {hit_id} not found")
                sets, args = [], []
                if status is not None:
                    sets.append("status=%s"); args.append(status)
                if confidence is not None:
                    sets.append("confidence=%s"); args.append(confidence)
                if note_append:
                    old = str(row[0]) if row[0] else ""
                    sets.append("note=%s")
                    args.append((old + "\n" + str(note_append)) if old
                                else str(note_append))
                if cross_refs_append:
                    old_refs = json.loads(row[1]) if row[1] else []
                    if not isinstance(old_refs, list):
                        old_refs = [old_refs]
                    seen = {(r.get("book_id"), r.get("page"))
                            for r in old_refs if isinstance(r, dict)}
                    for r in cross_refs_append:
                        if not isinstance(r, dict):
                            continue
                        key = (r.get("book_id"), r.get("page"))
                        if key not in seen:
                            old_refs.append(r)
                            seen.add(key)
                    sets.append("cross_refs=%s")
                    args.append(json.dumps(old_refs))
                if sets:
                    cur.execute(f"UPDATE hits SET {', '.join(sets)} "
                                f"WHERE id=%s", args + [int(hit_id)])
                cur.execute(f"{_SELECT} WHERE h.id=%s", (int(hit_id),))
                return _parse_rows(cur.fetchall(),
                                   [c[0] for c in cur.description])[0]
        finally:
            conn.close()
    except HitDBError:
        raise
    except Exception as e:
        raise HitDBError(f"{NAME}@{HOST}:{PORT}: {str(e).splitlines()[0]}") from e
