-- squig_research — the two-layer research / hit log for the DDC system.
--
-- MariaDB / MySQL schema. This is the standalone source of truth for the
-- database that base/reader/hits_db.py connects to (the reader auto-logs
-- every search/ask/page here as layer='user'; the system's own off-time
-- research writes layer='agent' via the log_hit MCP tool).
--
-- The reader creates the tables itself at startup (CREATE TABLE IF NOT
-- EXISTS + idempotent v2 upgrades), so running this file is only needed if
-- you want to provision the DB out-of-band. It is safe to run repeatedly.
--
--   mysql -u matt -p < squig_research.sql
--   (connection defaults in hits_db.py: HITS_DB_HOST/PORT/USER/NAME env,
--    default 127.0.0.1:3306, user "matt", db "squig_research")

CREATE DATABASE IF NOT EXISTS squig_research
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE squig_research;

-- One row per logged interaction. `layer` keeps the human (user) log and
-- the agent's autonomous research log separate; `status` turns the agent
-- rows into a work queue (found -> needs-verify -> verified -> done).
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

-- ---------------------------------------------------------------------------
-- Upgrade path for DBs created before v2: every statement below is a no-op
-- on a fresh database (ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT
-- EXISTS) and back-fills the research-loop fields on an old one. Safe to
-- run on every deploy, mirroring hits_db.py's UPGRADES list.
-- ---------------------------------------------------------------------------
ALTER TABLE hits ADD COLUMN IF NOT EXISTS page_start INT NULL;
ALTER TABLE hits ADD COLUMN IF NOT EXISTS page_end INT NULL;
ALTER TABLE hits ADD COLUMN IF NOT EXISTS topic TEXT NULL;
ALTER TABLE hits ADD COLUMN IF NOT EXISTS cross_refs TEXT NULL;
ALTER TABLE hits ADD COLUMN IF NOT EXISTS status TEXT NOT NULL
  DEFAULT 'found' CHECK (status IN ('found','needs-verify','verified','done'));
ALTER TABLE hits ADD COLUMN IF NOT EXISTS confidence TEXT NOT NULL
  DEFAULT 'medium' CHECK (confidence IN ('low','medium','high'));
ALTER TABLE hits ADD COLUMN IF NOT EXISTS session_id TEXT NULL;
CREATE INDEX IF NOT EXISTS idx_hits_status ON hits(status);
CREATE INDEX IF NOT EXISTS idx_hits_ddc ON hits(ddc);
CREATE INDEX IF NOT EXISTS idx_hits_topic ON hits(topic);
CREATE INDEX IF NOT EXISTS idx_hits_session_id ON hits(session_id);
