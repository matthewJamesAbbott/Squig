#!/usr/bin/env bash
# DDC Book Reader — start/stop/status helper.
#
#   ./run.sh [start|stop|status|logs]        the Flask viewer (port 8071)
#   ./run.sh owui [start|stop|status|logs]   the Open WebUI bridge (port 8072)
#
# The bridge (ddc_owui.py) exposes /v1/models + /v1/chat/completions
# (OpenAI-compatible, registers the ddc-library model in OWUI) and /mcp
# (streamable-HTTP MCP with the 13 DDC library tools).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PIPELINE_PYTHON:-$HOME/miniconda3/envs/marker_env/bin/python}"

# service name: "reader" (default) or "owui"
case "${1:-}" in
  owui) shift; SVC="owui" ;;
  *)    SVC="reader" ;;
esac

if [[ "$SVC" == "owui" ]]; then
  APP="$DIR/ddc_owui.py"
  PORT="${DDC_OWUI_PORT:-8072}"
  HEALTH="/healthz"
  LOG="$DIR/ddc-owui.log"
  PIDFILE="$DIR/ddc-owui.pid"
else
  APP="$DIR/app.py"
  PORT="${READER_PORT:-8071}"
  HEALTH="/api/health"
  LOG="$DIR/reader.log"
  PIDFILE="$DIR/reader.pid"
fi

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
    nohup "$PY" "$APP" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if curl -sf "http://localhost:$PORT$HEALTH" > /dev/null; then
      echo "started: http://localhost:$PORT (pid $(cat "$PIDFILE"))"
    else
      echo "started process (pid $(cat "$PIDFILE")) but health check failed — check $LOG"
    fi
    ;;
  *)
    echo "usage: $0 [reader|owui] [start|stop|status|logs]"; exit 1
    ;;
esac
