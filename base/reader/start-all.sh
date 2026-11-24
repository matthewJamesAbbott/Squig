#!/usr/bin/env bash
# start-all.sh — bring the whole DDC stack up after a reboot (one shot).
#
#   1. qdrant       docker container (restart-policy is 'no', so it does
#                   NOT come back by itself)
#   2. indexer      unattended supervisor (flock-guarded: no-ops if one is
#                   already running, or if the scan is already complete)
#   3. gui          pipeline dashboard on :8070 (OCR/index progress + controls),
#                   preceded by re-applying the saved GPU power limits
#                   (nvidia-smi -pl resets to factory default on reboot)
#   4. reader       Flask app on :8071 (./run.sh start)
#   5. owui bridge  Open WebUI bridge on :8072 (./run.sh owui start)
#
# NOT started here, on purpose:
#   - MariaDB       systemd-enabled, comes up on its own (hit DB)
#   - MCP           there is no MCP daemon: OpenCode launches mcp_server.py
#                   itself per session (stdio), and OWUI's /mcp is served
#                   by the bridge (step 5)
#   - squig LLM     the vLLM model on :8000 (OpenCode/OWUI) — managed from
#                   the dashboard (Services → squig LLM); it is often left
#                   unloaded so the V100s can do OCR. See the commented
#                   step at the bottom of this file to auto-load it.
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "0/5 gpu power limits (last chosen in the dashboard)"
"$HOME/base/gui/set_gpu_power.sh" || true   # fail-soft: preference, not a service

echo "1/5 qdrant"
docker start ddc-qdrant 2>/dev/null || echo "   (docker not up yet? check: systemctl status docker)"

echo "2/5 indexer supervisor"
nohup "$HOME/DDC/indexer/run_until_done.sh" >> "$HOME/DDC/indexer/supervisor.log" 2>&1 &

echo "3/5 gui dashboard (:8070)"
"$HOME/base/gui/run.sh" start

echo "4/5 reader (:8071)"
"$DIR/run.sh" start

echo "5/5 owui bridge (:8072)"
"$DIR/run.sh" owui start

echo
echo "stack state:"
nvidia-smi --query-gpu=index,name,power.limit,temperature.gpu --format=csv,noheader \
  | sed 's/^/  gpu /'
docker ps --format '  qdrant: {{.Names}} {{.Status}}' | grep ddc-qdrant || echo "  qdrant: NOT RUNNING"
pgrep -f "run_until_done[.]sh" >/dev/null && echo "  indexer supervisor: running" || echo "  indexer supervisor: not running (fine if the scan is complete)"
"$HOME/base/gui/run.sh" status
"$DIR/run.sh" status
"$DIR/run.sh" owui status
curl -sf localhost:8071/v1/hits/status >/dev/null && echo "  hit DB (MariaDB): reachable" || echo "  hit DB (MariaDB): unreachable"
curl -sf -m 3 http://10.90.38.19:8000/health >/dev/null 2>&1 \
  && echo "  squig LLM (:8000): loaded" || echo "  squig LLM (:8000): unloaded (V100s free for OCR)"

# Optional: load the squig LLM at boot. It is normally managed from the
# dashboard (Services → squig LLM), which unloads it while OCR work runs.
# nohup "$HOME/scripts/squigFastThink" >> "$HOME/squig-llm.log" 2>&1 &
