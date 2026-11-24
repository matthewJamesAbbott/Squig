#!/usr/bin/env bash
# Re-apply the GPU power limits last chosen in the dashboard.
#
# nvidia-smi -pl values reset to the card's factory default every time the
# GPU loses power (reboot / GPU reset), so start-all.sh runs this at boot.
# The desired limits live in ~/base/gui/config.json under "gpu_power":
#     "gpu_power": {"0": 150, "2": 150}
# (the dashboard writes the map whenever you change a limit).
# Fail-soft: missing config, missing sudo rule or a transient error just
# prints a line — boot must not stop because of a power preference.
set -u
PY="${PIPELINE_PYTHON:-$HOME/miniconda3/envs/marker_env/bin/python}"
CFG="$HOME/base/gui/config.json"
[ -f "$CFG" ] || exit 0
"$PY" - "$CFG" <<'EOF'
import json, subprocess, sys
cfg = json.load(open(sys.argv[1]))
limits = cfg.get("gpu_power") or {}
if not limits:
    sys.exit(0)
for idx, watts in sorted(limits.items(), key=lambda kv: int(kv[0])):
    r = subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(idx),
                        "-pl", str(int(watts))],
                       capture_output=True, text=True, timeout=15)
    out = "ok" if r.returncode == 0 else \
        "FAILED: " + ((r.stderr or r.stdout or f"exit {r.returncode}").strip()[:90])
    print(f"   GPU{idx} power limit -> {watts}W: {out}")
EOF
