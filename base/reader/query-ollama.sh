#!/usr/bin/env bash
# Manage the dedicated CPU query-embed Ollama (port 11435).
#
# The indexer's GPU worker (11434) embeds big batches with
# OLLAMA_NUM_PARALLEL=1, so an interactive query queued behind it takes
# ~14 s mid-run. This CPU worker keeps search/ask at ~0.2 s while the
# index is being built. Same bge-m3 model: CPU and GPU embeddings are
# bit-identical, so one vector space.
#
#   ./query-ollama.sh start | stop | status | restart
set -euo pipefail
HOST=127.0.0.1
PORT=11435
LOG="${LOG:-/home/matt/base/reader/ollama-query.log}"
PIDFILE="${PIDFILE:-/home/matt/base/reader/ollama-query.pid}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama/ollama}"

alive() {
  curl -s -m 3 "http://$HOST:$PORT/api/tags" >/dev/null 2>&1
}

case "${1:-status}" in
  start)
    if alive; then
      echo "already running on $HOST:$PORT"
      exit 0
    fi
    CUDA_VISIBLE_DEVICES= OLLAMA_HOST="$HOST:$PORT" \
    OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=30m \
      nohup "$OLLAMA_BIN" serve >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 20); do
      alive && { echo "started on $HOST:$PORT (pid $(cat "$PIDFILE"))"; exit 0; }
      sleep 0.5
    done
    echo "failed to start — see $LOG" >&2; exit 1
    ;;
  stop)
    # Kill by listener on the port (there are two `ollama serve` processes:
    # the GPU index worker on 11434 and this one — pidfile alone isn't enough)
    pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null || true
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    sleep 1
    echo "stopped"
    ;;
  status)
    if alive; then
      echo "up on $HOST:$PORT"
      curl -s -m 3 "http://$HOST:$PORT/api/tags" | \
        /home/matt/miniconda3/envs/marker_env/bin/python -c \
        "import json,sys; print('models:', [m['name'] for m in json.load(sys.stdin)['models']])" \
        || true
    else
      echo "down"
      exit 1
    fi
    ;;
  restart)
    "$0" stop; sleep 1; "$0" start
    ;;
  *)
    echo "usage: $0 start|stop|status|restart"; exit 2
    ;;
esac
