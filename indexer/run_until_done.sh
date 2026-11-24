#!/usr/bin/env bash
# run_until_done.sh — unattended supervisor for the DDC indexer.
#
# For the long-haul: new parquets land over months (books, then JSTOR,
# then newspapers) and the machine may reboot or the indexer may crash.
# This script keeps the indexer alive and lets it finish:
#
#   - if an indexer is already running (e.g. started from the UI), it
#     follows it and reports when it finishes
#   - otherwise it runs `index_books.py --root ~/DDC`, appending to the
#     shared index_run.log exactly like a manual run
#   - clean exit (0)  -> full scan completed, supervisor stops
#   - SIGTERM (143)   -> intentional stop (UI "Stop indexer"), supervisor
#                        stops
#   - any other exit  -> crash: wait 60 s, start again. index_books.py is
#                        self-healing (exact point-count check per book),
#                        so a restart re-checks everything and only
#                        re-embeds what is missing/partial.
#
# Usage:
#   nohup ~/DDC/indexer/run_until_done.sh > ~/DDC/indexer/supervisor.log 2>&1 &
#   tail -f ~/DDC/indexer/supervisor.log
# Stop:
#   pkill -f "run_until_done[.]sh"          # supervisor SIGTERMs its child
#   # (or use the UI "Stop indexer" button; the supervisor exits on 143)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PIPELINE_PYTHON:-/home/matt/miniconda3/envs/marker_env/bin/python}"
ROOT="${DDC_ROOT:-$HOME/DDC}"
LOG="$DIR/index_run.log"
LOCK="/tmp/ddc-indexer.lock"

# Single supervisor instance (also guards against double starts from the UI)
exec 9>"$LOCK"
flock -n 9 || { echo "another supervisor is already running"; exit 0; }

running_pid() { pgrep -f "index_books[.]py --root" | head -1 || true; }

# Already running (e.g. started from the UI) -> follow, don't double-start
p="$(running_pid)"
if [ -n "$p" ]; then
  echo "indexer already running (pid $p); following $LOG ..."
  while kill -0 "$p" 2>/dev/null; do
    sleep 15
    p="$(running_pid)"
    [ -n "$p" ] || break
  done
  echo "indexer (pid $p) finished."
  exit 0
fi

stop_child() { kill "$child" 2>/dev/null; }
trap 'stop_child; exit 0' INT TERM

while true; do
  nohup "$PY" -u "$DIR/index_books.py" --root "$ROOT" >> "$LOG" 2>&1 &
  child=$!
  echo "started indexer pid $child (log: $LOG)"
  wait "$child"
  rc=$?
  case "$rc" in
    0)   echo "full run completed (rc=0) — nothing left to index. Supervisor stopping."; exit 0 ;;
    143) echo "indexer stopped by signal (SIGTERM) — supervisor stopping."; exit 0 ;;
    *)   echo "indexer exited rc=$rc — crash? Restarting in 60 s (self-heal will re-check all books)." ;;
  esac
  sleep 60
done
