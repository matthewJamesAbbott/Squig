#!/usr/bin/env bash
# DDC Book Pipeline GUI — start/stop/status helper.
#   ./run.sh            start if not running
#   ./run.sh status     check
#   ./run.sh stop       stop
#   ./run.sh logs       tail the GUI log
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PIPELINE_PYTHON:-$HOME/miniconda3/envs/marker_env/bin/python}"
PORT="${GUI_PORT:-8070}"
LOG="$DIR/gui.log"
PIDFILE="$DIR/gui.pid"

is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-start}" in
  status)
    if is_running; then echo "running (pid $(cat "$PIDFILE"), http://localhost:$PORT)";
    else echo "not running"; fi
    ;;
  stop)
    if is_running; then kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"; echo "stopped";
    else echo "not running"; fi
    ;;
  logs)
    tail -n 50 -f "$LOG"
    ;;
  start)
    if is_running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    nohup "$PY" "$DIR/app.py" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if curl -sf "http://localhost:$PORT/api/health" > /dev/null; then
      echo "started: http://localhost:$PORT (pid $(cat "$PIDFILE"))"
    else
      echo "started process (pid $(cat "$PIDFILE")) but health check failed — check $LOG"
    fi
    ;;
  *)
    echo "usage: $0 [start|stop|status|logs]"; exit 1
    ;;
esac
