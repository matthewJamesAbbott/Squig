#!/usr/bin/env python3
"""
DDC Book Pipeline — Flask control panel.

One web UI (default http://localhost:8070) that runs, monitors and logs:

  1. Ingest:     ~/base/ocr.py                    any book format -> Parquet
                                                        (format autodetected per file;
                                                         marker GPU for PDF/DJVU/CBZ/CBR;
                                                         CPU text extractors, nominal
                                                         pages, for the rest; calibre/
                                                         LibreOffice converters in between)
  2. Rename:     ~/base/rename_ddc_collection.py  move into the DDC tree
  3. Recipes:    ~/base/extract_recipes.py        LLM recipe/ingredient extraction
  4. Index:      ~/DDC/indexer/index_books.py     Parquet -> Qdrant (bge-m3)

plus service controls for Qdrant (docker compose), Ollama (GPU pick + pull)
and GPU power limits (nvidia-smi -pl — e.g. 150W/300W on the V100s, so they
can be throttled to stay cool or run flat-out when needed).

Pipeline progress: ocr.py and index_books.py each write a machine-readable
state file to ~/.ddc-progress/ (index_<root>.json / ocr_<outdir>.json) as
they work, and /api/pipeline turns those into live "where it is / how far
to go / pace / ETA" for the page — for runs started here or from the CLI.

Every task runs as its own subprocess in its own process group with its own
log file under ~/base/gui/logs/. GPU selection is per-task: the GUI uses
nvidia-smi (PCI) indices and launches GPU tasks with
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>
so the selected physical GPU is always the one used (independent of CUDA's
default performance-based ordering).

Run:
    /home/matt/miniconda3/envs/marker_env/bin/python ~/base/gui/app.py
"""

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Locations / constants
# ---------------------------------------------------------------------------
HOME = Path.home()
BASE_DIR = HOME / "base"
DDC_DIR = HOME / "DDC"
INDEXER_DIR = DDC_DIR / "indexer"
GUI_DIR = Path(__file__).resolve().parent
LOG_DIR = GUI_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CONFIG_FILE = GUI_DIR / "config.json"
TASKS_FILE = GUI_DIR / "tasks.json"

PY = os.environ.get("PIPELINE_PYTHON", str(HOME / "miniconda3/envs/marker_env/bin/python"))
OLLAMA_BIN = os.environ.get("OLLAMA_BIN", str(HOME / ".local/ollama/ollama"))
SUDO_BIN = os.environ.get("PIPELINE_SUDO", "sudo")  # -pl (power limits) needs root
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL_DEFAULT = "http://localhost:11434"
GUI_PORT = int(os.environ.get("GUI_PORT", "8070"))

# Live pipeline progress: ocr.py and index_books.py each write a small
# machine-readable JSON state file after every chunk/book/embed-batch
# (atomic replace, fail-soft on the writer side). /api/pipeline reads them,
# which is why the dashboard shows "where it is + how far to go + ETA" for
# runs started HERE or from the CLI/supervisor alike.
PROGRESS_DIR = Path(os.environ.get("DDC_PROGRESS_DIR", str(HOME / ".ddc-progress")))

COLLECTION = "ddc-books"

# The user's vLLM (OpenAI-compatible) LLM runs in the `onecat` LXC container.
# Verify:  lxc exec onecat -- ss -ltn  /  curl http://10.90.38.19:8000/v1/models
VLLM_BASE = "http://10.90.38.19:8000/v1"
VLLM_MODEL = "squigFast"

DEFAULTS = {
    "ocr": {
        "input": str(HOME / "Psychology"),
        "output": str(HOME / "converted" / "Psychology"),
        "gpu": "auto",
        # Multi-GPU: one worker process per selected GPU. "auto" = freest
        # GPU not already used by another worker in the same job. Check
        # several to fan the PDF list out round-robin (ocr.py --shard).
        "gpus": ["auto"],
        "chunk": 20,
        "batch": 2,
        "resume": True,
        # marker small-GPU tuning (blank = script default)
        "env": {
            "DETECTOR_BATCH_SIZE": "", "LAYOUT_BATCH_SIZE": "",
            "ORDER_BATCH_SIZE": "", "RECOGNITION_BATCH_SIZE": "",
            "TABLE_REC_BATCH_SIZE": "", "TEXIFY_BATCH_SIZE": "",
            "SURYA_OCR_DPI": "", "SURYA_TABLE_DPI": "",
        },
    },
    "rename": {
        "source": str(HOME / "converted" / "Psychology"),
        "dest": str(DDC_DIR),
        "execute": False,
        "overwrite": False,
    },
    "enrich": {
        "input": str(HOME / "converted"),
        "category": "",
        "dry_run": False,
        "api_base": VLLM_BASE,
        "model": "squigFast",
    },
    "recipes": {
        "input": str(DDC_DIR / "600_Technology" / "640_Home_and_family_management"),
        "output": str(HOME / "recipes_out"),
        "llm_base_1": VLLM_BASE,
        "llm_base_2": "",
        "llm_model": VLLM_MODEL,
        "max_concurrent": 8,
        "batch_size": 5,
        "flush_interval": 5,
        "max_files": "",
        "start_file": "",
        "resume": True,
    },
    "index": {
        "root": str(DDC_DIR),
        "limit": 0,
        "dry_run": False,
        "embed_base_url": OLLAMA_URL_DEFAULT,
        "embed_model": "bge-m3",
        "qdrant_url": QDRANT_URL,
        "ollama_gpu": "auto",
        "relocate_ollama": False,
    },
    "ollama": {
        "gpu": "auto",
        "num_parallel": 1,
        "host": "0.0.0.0:11434",
        "pull_model": "bge-m3",
    },
    # squig LLM (the model serving OpenCode + OWUI on :8000): which profile
    # was last loaded from the dashboard. Only one profile can be loaded at
    # a time (they all bind port 8000).
    "squig": {"last": "squigFastThink"},
    # Last power limits chosen in the dashboard ({"0": 150, "2": 300}).
    # nvidia-smi -pl resets to the card default on every power cycle, so
    # start-all.sh re-applies this map at boot (base/gui/set_gpu_power.sh).
    "gpu_power": {},
}

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Config persistence (last-used form values)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            for k, v in saved.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


CONFIG = load_config()


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def _fnum(x: str):
    """'300.00' -> 300.0 ; '[N/A]' / '' / None -> None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def nvidia_smi_gpus() -> list:
    """One entry per GPU, indexed like `nvidia-smi` (PCI order).

    Includes power state (limit / min-max range / current draw) and temp so
    the dashboard can show + change power limits (nvidia-smi -pl)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,pci.bus_id,memory.used,memory.total,"
             "utilization.gpu,power.limit,power.default_limit,power.min_limit,"
             "power.max_limit,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        (idx, name, bdf, used, total, util, p_lim, p_def,
         p_min, p_max, p_draw, temp) = [p.strip() for p in line.split(",")]
        used, total = int(used), int(total)
        gpus.append({
            "index": int(idx),
            "name": name,
            "pci": bdf,
            "mem_used_mb": used,
            "mem_total_mb": total,
            "mem_free_mb": total - used,
            "util_pct": int(util),
            "power_limit_w": _fnum(p_lim),
            "power_default_w": _fnum(p_def),
            "power_min_w": _fnum(p_min),
            "power_max_w": _fnum(p_max),
            "power_draw_w": _fnum(p_draw),
            "temp_c": _fnum(temp),
        })
    return gpus


def pick_gpu(choice: str, gpus: list) -> Optional[dict]:
    """Resolve a GUI gpu choice ('auto' | 'cpu' | str(smi index)) to a GPU."""
    if choice in ("cpu", "CPU"):
        return None
    if choice in ("auto", ""):
        return max(gpus, key=lambda g: g["mem_free_mb"]) if gpus else None
    for g in gpus:
        if str(g["index"]) == str(choice):
            return g
    return None


def gpu_env(gpu: Optional[dict]) -> dict:
    """Env that pins a subprocess to exactly one physical GPU (or none).

    CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA indices match nvidia-smi's, so
    CUDA_VISIBLE_DEVICES=<smi index> + device 'cuda:0' is unambiguous.
    """
    if gpu is None:
        return {}
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
    }


def gpu_label(gpu: Optional[dict]) -> str:
    return f"GPU{gpu['index']} {gpu['name'].split('-')[0]}" if gpu else "CPU"


# ---------------------------------------------------------------------------
# GPU power limits (nvidia-smi -pl): throttle to stay cool, or flat-out
# ---------------------------------------------------------------------------
def sudo_nvidia_sm_ok() -> bool:
    """Can we run nvidia-smi as root without a password prompt?

    Read-only queries work unprivileged, but changing a power limit
    (-pl / -rgpl) needs root, and the dashboard is a headless nohup process,
    so this must be passwordless (NOPASSWD sudoers rule for nvidia-smi)."""
    try:
        r = subprocess.run([SUDO_BIN, "-n", "nvidia-smi", "-L"],
                           capture_output=True, text=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def _pl_cmd(index: int, watts_w: float) -> list:
    """The nvidia-smi command that sets one GPU's power limit (in watts).

    Note: this driver (535.x) has no `-rgpl` reset flag, so 'default' is
    implemented as `-pl <power.default_limit>` in the caller."""
    return [SUDO_BIN, "-n", "nvidia-smi", "-i", str(index),
            "-pl", str(int(round(watts_w)))]


def gpu_set_power(target, watts) -> dict:
    """Set a power limit on one GPU or a group of them.

    target: an int smi index (0/1/2…) or the string 'v100s' (every GPU whose
            name contains V100 — the machine's two big cards).
    watts:  a number, validated against the card's own min/max, or
            'default' to reset to the factory limit.
    """
    gpus = nvidia_smi_gpus()
    if str(target).lower() == "v100s":
        targets = [g["index"] for g in gpus if "v100" in g["name"].lower()]
        if not targets:
            return {"ok": False, "targets": [], "error": "no V100 GPUs found"}
    else:
        targets = [int(target)]
    is_default = str(watts).strip().lower() in ("default", "reset", "")
    results = []
    for i in targets:
        g = next((x for x in gpus if x["index"] == i), None)
        w = watts
        if is_default:
            # driver 535 has no -rgpl: 'default' = -pl <factory limit>
            if g is None or g.get("power_default_w") is None:
                return {"ok": False, "targets": targets,
                        "error": f"GPU{i}: factory power limit unknown"}
            w = g["power_default_w"]
        if g:
            lo, hi = g.get("power_min_w"), g.get("power_max_w")
            if lo is not None and hi is not None and not (lo <= float(w) <= hi):
                return {"ok": False, "targets": targets,
                        "error": f"GPU{i}: {w}W is outside this card's "
                                 f"allowed range {lo:.0f}–{hi:.0f}W"}
        try:
            r = subprocess.run(_pl_cmd(i, w),
                               capture_output=True, text=True, timeout=15)
        except Exception as e:
            return {"ok": False, "targets": targets,
                    "error": f"GPU{i}: {e}"}
        ok = r.returncode == 0
        err = ""
        if not ok:
            lines = (r.stderr or r.stdout or "").strip().splitlines()
            err = lines[0] if lines else f"exit code {r.returncode}"
        results.append({"gpu": i, "ok": ok, "error": None if ok else err})
    ok_all = all(r_["ok"] for r_ in results)
    if any("password" in (r_["error"] or "").lower() for r_ in results):
        import getpass
        user = getpass.getuser()
        return {"ok": False, "targets": targets, "results": results,
                "error": "passwordless sudo for nvidia-smi is not configured — "
                         f"run once:  echo '{user} ALL=(root) NOPASSWD: "
                         "/usr/bin/nvidia-smi' | sudo tee "
                         "/etc/sudoers.d/nvidia-smi && sudo chmod 440 "
                         "/etc/sudoers.d/nvidia-smi"}
    # report the limits as they stand now (verification, not the request)
    after = {str(g["index"]): g["power_limit_w"]
             for g in nvidia_smi_gpus() if g["index"] in targets}
    return {"ok": ok_all, "targets": targets, "results": results,
            "limits": after, "watts": None if is_default else int(watts)}


# ---------------------------------------------------------------------------
# Task manager
# ---------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    kind: str
    label: str
    cmd: list
    cwd: str
    env_extra: dict
    log_path: str
    pid: int
    started_at: float
    status: str = "running"          # running | success | failed | stopped
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    proc: object = field(default=None, repr=False, compare=False)

    def public(self) -> dict:
        d = {
            "id": self.id, "kind": self.kind, "label": self.label,
            "cmd": self.cmd, "cwd": self.cwd, "env_extra": self.env_extra,
            "log_path": self.log_path, "pid": self.pid,
            "started_at": self.started_at, "status": self.status,
            "finished_at": self.finished_at, "returncode": self.returncode,
        }
        d["elapsed_s"] = (self.finished_at or time.time()) - self.started_at
        d["progress"] = task_progress(self)
        return d


# "Book 12/643:" (index), "[ 3/9] file" (ocr/recipes)
PROGRESS_PATTERNS = {
    "index": re.compile(r"Book (\d+)/(\d+)"),
    "ocr": re.compile(r"\[ *(\d+)/(\d+)\]"),
    "recipes": re.compile(r"\[ *(\d+)/(\d+)\]"),
}


def task_progress(t: "Task"):
    """Progress for a running task: {current,total,pct[,eta_s,current_item]}.

    index/ocr tasks prefer the machine-readable progress file the process
    itself writes (page-level + ETA); anything else falls back to parsing
    the log tail.
    """
    if t.status != "running":
        return None
    if t.kind in ("index", "ocr"):
        p = _task_progress_from_files(t)
        if p:
            return p
    pat = PROGRESS_PATTERNS.get(t.kind)
    if not pat:
        return None
    try:
        size = os.path.getsize(t.log_path)
        with open(t.log_path, "rb") as f:
            f.seek(max(0, size - 16384))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    m = None
    for m in pat.finditer(tail):
        pass
    if not m:
        return None
    cur, total = int(m.group(1)), int(m.group(2))
    if total <= 0:
        return None
    return {"current": cur, "total": total, "pct": round(100.0 * cur / total, 1)}


def _task_progress_from_files(t: "Task"):
    """Progress file for this task's index/ocr run (matched by --root /
    --output), if the file exists and is not ancient. None otherwise.

    Sharded OCR workers carry `--shard i/N` in their cmd and write
    ocr_<out>_g<i>.json — the row shows that worker's own shard."""
    cmd = t.cmd
    flag = "--root" if t.kind == "index" else "--output"
    target = None
    if flag in cmd:
        i = cmd.index(flag)
        if i + 1 < len(cmd):
            target = cmd[i + 1]
    if not target:
        return None
    slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(target).name).strip("_")
    prefix = "index" if t.kind == "index" else "ocr"
    fname = f"{prefix}_{slug or ('root' if prefix == 'index' else 'out')}.json"
    if "--shard" in cmd:
        si = cmd.index("--shard")
        m = re.match(r"(\d+)/\d+", cmd[si + 1]) if si + 1 < len(cmd) else None
        if m:
            fname = re.sub(r"\.json$", f"_g{m.group(1)}.json", fname)
    f = PROGRESS_DIR / fname
    try:
        d = json.loads(f.read_text())
    except Exception:
        return None
    if (time.time() - (d.get("last_update") or 0)) > 600 and d.get("state") != "running":
        return None
    view = _progress_view(f, d, time.time())
    total = d.get("total_books") or d.get("total_files") or 0
    done = d.get("done_books") or d.get("done_files") or 0
    pct = view["pct_pages"] if view["pct_pages"] is not None else view["pct_items"]
    if not total or pct is None:
        return None
    out = {"current": done, "total": total, "pct": pct,
           "pct_basis": "pages" if view["pct_pages"] is not None else "items"}
    if view.get("eta_s"):
        out["eta_s"] = view["eta_s"]
    if view.get("current"):
        out["current_item"] = view["current"]
    return out


TASKS: dict = {}
TASKS_LOCK = threading.Lock()


def _load_task_history() -> None:
    """Restore finished tasks from disk (running ones are dead after restart)."""
    if not TASKS_FILE.exists():
        return
    try:
        for d in json.loads(TASKS_FILE.read_text())[-200:]:
            t = Task(**{k: d[k] for k in (
                "id", "kind", "label", "cmd", "cwd", "env_extra", "log_path",
                "pid", "started_at", "status", "finished_at", "returncode")})
            t.status = "stopped" if d["status"] == "running" else d["status"]
            TASKS[t.id] = t
    except Exception:
        pass


def _persist_tasks() -> None:
    with TASKS_LOCK:
        items = sorted(TASKS.values(), key=lambda t: t.started_at, reverse=True)
    TASKS_FILE.write_text(json.dumps([t.public() for t in items[:200]], indent=1))


def _poller() -> None:
    """Mark finished subprocesses."""
    while True:
        time.sleep(1)
        changed = False
        with TASKS_LOCK:
            items = list(TASKS.values())
        for t in items:
            if t.status == "running" and t.proc is not None:
                rc = t.proc.poll()
                if rc is not None:
                    t.finished_at = time.time()
                    t.returncode = rc
                    t.status = "stopped" if rc < 0 else ("success" if rc == 0 else "failed")
                    changed = True
        if changed:
            _persist_tasks()


def _kill_task(t: Task, sig: int) -> None:
    try:
        os.killpg(t.proc.pid, sig)
    except Exception:
        pass


def start_task(kind: str, params: dict):
    """Start a task — or, for a multi-GPU OCR job, one worker task per GPU.

    Returns a Task (single-worker kinds) or a list of Tasks (fanned-out OCR
    job). Each worker is a fully independent, resumable process with its own
    log, checkpoint shard and progress file; the dashboard merges them."""
    gpus = nvidia_smi_gpus()
    specs: list = []  # (cmd, env_extra, label)
    env_extra: dict = {}
    cwd = str(GUI_DIR)

    if kind == "ocr":
        input_path = Path(params["input"]).expanduser()
        output_dir = Path(params["output"]).expanduser()
        if not input_path.exists():
            raise ValueError(f"input does not exist: {input_path}")
        # One worker per selected GPU. 'auto' resolves to the freest GPU not
        # already claimed by another worker of this job.
        choices = params.get("gpus")
        if choices is None:
            choices = [params.get("gpu", "auto")]
        if isinstance(choices, (str, int)):
            choices = [choices]
        choices = [str(c).strip() for c in choices if str(c).strip()] or ["auto"]
        picked: list = []
        used: set = set()
        for ch in choices:
            if ch in ("auto", ""):
                free = [g for g in gpus if g["index"] not in used]
                gpu = max(free, key=lambda g: g["mem_free_mb"]) if free else None
            elif ch in ("cpu", "CPU"):
                gpu = None
            else:
                gpu = pick_gpu(ch, gpus)
            key = gpu["index"] if gpu else "cpu"
            if key in used:
                continue
            used.add(key)
            picked.append(gpu)
        if not picked:
            raise ValueError("no GPUs available for this job")
        n_workers = len(picked)
        # A job id ties the shard progress files together, so the dashboard
        # can merge a sharded job (including shards that finish early) into
        # one card without mixing in older, unrelated runs of the same dir.
        job_id = uuid.uuid4().hex[:8] if n_workers > 1 else None
        env_custom = {k: str(v).strip() for k, v in (params.get("env") or {}).items()
                      if str(v).strip()}
        for k, gpu in enumerate(picked):
            device = "cuda:0" if gpu else "cpu"
            wcmd = [PY, str(BASE_DIR / "ocr.py"),
                    "--input", str(input_path), "--output", str(output_dir),
                    "--device", device,
                    "--chunk", str(int(params.get("chunk", 20))),
                    "--batch", str(int(params.get("batch", 2))),
                    "--words-per-page", str(int(params.get("words_per_page", 300)))]
            if n_workers > 1:
                wcmd += ["--shard", f"{k}/{n_workers}", "--job", job_id]
            if params.get("resume"):
                wcmd.append("--resume")
            wenv = dict(gpu_env(gpu))
            wenv.update(env_custom)
            suffix = f" · worker {k}/{n_workers}" if n_workers > 1 else ""
            wlabel = f"OCR: {input_path.name} -> {output_dir.name} [{gpu_label(gpu)}{suffix}]"
            specs.append((wcmd, wenv, wlabel))

    elif kind == "rename":
        source = Path(params["source"]).expanduser()
        dest = Path(params["dest"]).expanduser()
        if not source.is_dir():
            raise ValueError(f"source directory does not exist: {source}")
        if not dest.is_dir():
            raise ValueError(f"destination directory does not exist: {dest}")
        cmd = [PY, str(BASE_DIR / "rename_ddc_collection.py"),
               "--source", str(source), "--dest", str(dest)]
        cmd.append("--execute" if params.get("execute") else "--dry-run")
        if params.get("overwrite"):
            cmd.append("--overwrite")
        label = f"Rename: {source.name} -> {dest.name} [{'EXECUTE' if params.get('execute') else 'dry-run'}]"
        specs.append((cmd, env_extra, label))

    elif kind == "enrich":
        input_dir = Path(params.get("input", "/home/matt/converted")).expanduser()
        if not input_dir.is_dir():
            raise ValueError(f"input directory does not exist: {input_dir}")
        category = str(params.get("category", "")).strip()
        if category and not (input_dir / category).is_dir():
            raise ValueError(f"category directory does not exist: {input_dir / category}")
        cmd = [PY, str(BASE_DIR / "enrich_ddc_collection.py"), str(input_dir)]
        if category:
            cmd += ["--category", category]
        cmd += ["--api-base",
                str(params.get("api_base", VLLM_BASE)).strip() or VLLM_BASE]
        model = str(params.get("model", "auto")).strip() or "auto"
        if model != "auto":
            cmd += ["--model", model]
        if params.get("dry_run"):
            cmd.append("--dry-run")
        scope = f"{input_dir.name}/{category}" if category else input_dir.name
        label = f"Enrich: {scope} [{'dry-run' if params.get('dry_run') else 'execute'}]"
        specs.append((cmd, env_extra, label))

    elif kind == "recipes":
        input_dir = Path(params["input"]).expanduser()
        output_dir = Path(params["output"]).expanduser()
        if not input_dir.is_dir():
            raise ValueError(f"input directory does not exist: {input_dir}")
        env_extra.update({
            "LLM_BASE_1": str(params.get("llm_base_1", "")).strip(),
            "LLM_BASE_2": str(params.get("llm_base_2", "")).strip(),
            "LLM_MODEL": str(params.get("llm_model", "squig")).strip() or "squig",
            "RECIPE_MAX_CONCURRENT": str(int(params.get("max_concurrent", 8))),
            "RECIPE_BATCH_SIZE": str(int(params.get("batch_size", 5))),
            "RECIPE_FLUSH_INTERVAL": str(int(params.get("flush_interval", 5))),
        })
        cmd = [PY, str(BASE_DIR / "extract_recipes.py"),
               str(input_dir), str(output_dir)]
        if str(params.get("start_file", "")).strip():
            cmd += ["--start-file", str(params["start_file"]).strip()]
        if str(params.get("max_files", "")).strip():
            cmd += ["--max-files", str(int(params["max_files"]))]
        if params.get("resume"):
            cmd.append("--resume")
        label = f"Recipes: {input_dir.name} -> {output_dir.name}"
        specs.append((cmd, env_extra, label))

    elif kind == "index":
        root = Path(params["root"]).expanduser()
        if not root.is_dir():
            raise ValueError(f"root directory does not exist: {root}")
        limit = int(params.get("limit", 0) or 0)
        env_extra.update({
            "EMBED_BASE_URL": str(params.get("embed_base_url", OLLAMA_URL_DEFAULT)).strip(),
            "EMBED_MODEL": str(params.get("embed_model", "bge-m3")).strip(),
            "QDRANT_URL": str(params.get("qdrant_url", QDRANT_URL)).strip(),
        })
        cmd = [PY, str(INDEXER_DIR / "index_books.py"), "--root", str(root)]
        if limit > 0:
            cmd += ["--limit", str(limit)]
        if params.get("dry_run"):
            cmd.append("--dry-run")
        label = f"Index: {root.name} (limit {limit or 'all'})"
        # Optional: make sure Ollama (the embedder) is up on the chosen GPU.
        if params.get("relocate_ollama") and not params.get("dry_run"):
            note = ensure_ollama(str(params.get("ollama_gpu", "auto") or "auto"))
            label += f" [ollama {note['status']} → GPU{note.get('gpu')}]"
        specs.append((cmd, env_extra, label))

    elif kind == "ollama_pull":
        model = str(params.get("model", "bge-m3")).strip()
        if not model:
            raise ValueError("model name required")
        cmd = [OLLAMA_BIN, "pull", model]
        label = f"Ollama pull: {model}"
        specs.append((cmd, env_extra, label))

    elif kind == "qdrant":
        action = params.get("action", "up")
        cmd = ["docker", "compose", "-f", str(INDEXER_DIR / "docker-compose.yml"),
               "up" if action == "up" else "down", "-d"]
        cwd = str(INDEXER_DIR)
        label = f"Qdrant {'start' if action == 'up' else 'stop'}"
        specs.append((cmd, env_extra, label))

    else:
        raise ValueError(f"unknown task kind: {kind}")

    if not specs:
        raise ValueError(f"no work to do for kind {kind!r}")

    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    rand = uuid.uuid4().hex[:6]
    n_specs = len(specs)
    tasks: list = []
    for k, (cmd, wenv, label) in enumerate(specs):
        t = Task(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            label=label,
            cmd=cmd,
            cwd=cwd,
            env_extra=wenv,
            log_path=str(LOG_DIR / (f"{kind}_{ts}_{rand}.log" if n_specs == 1
                                    else f"{kind}_{ts}_{rand}_g{k}.log")),
            pid=0,
            started_at=time.time(),
        )
        full_env = os.environ.copy()
        full_env.update(wenv)
        with open(t.log_path, "ab") as logf:
            logf.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}  {label}\n".encode())
            logf.write(f"# $ {' '.join(shlex.quote(c) for c in cmd)}\n".encode())
            if wenv:
                for ek, ev in wenv.items():
                    logf.write(f"# env {ek}={ev}\n".encode())
            logf.flush()
            t.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd=cwd, env=full_env, start_new_session=True)
            t.pid = t.proc.pid
        with TASKS_LOCK:
            TASKS[t.id] = t
        tasks.append(t)
    _persist_tasks()
    return tasks[0] if n_specs == 1 else tasks


def stop_task(task_id: str) -> bool:
    with TASKS_LOCK:
        t = TASKS.get(task_id)
    if not t or t.status != "running":
        return False
    threading.Thread(target=_escalating_kill, args=(t,), daemon=True).start()
    return True


def _escalating_kill(t: Task) -> None:
    _kill_task(t, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if t.proc.poll() is not None:
            return
    _kill_task(t, signal.SIGKILL)


_load_task_history()
threading.Thread(target=_poller, daemon=True).start()


# ---------------------------------------------------------------------------
# Services: Ollama / Qdrant
# ---------------------------------------------------------------------------
OLLAMA_STATE = {"pid": None}  # pid of the `ollama serve` we launched


def ollama_up(base_url: str = OLLAMA_URL_DEFAULT) -> dict:
    try:
        r = requests.get(f"{base_url}/api/version", timeout=3)
        return {"up": True, "version": r.json().get("version", "?")}
    except Exception:
        return {"up": False, "version": None}


def ollama_models(base_url: str = OLLAMA_URL_DEFAULT) -> list:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_gpu() -> Optional[int]:
    """nvidia-smi index of the GPU the running ollama process is on (or None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        uuids = {}
        for line in out.strip().splitlines():
            pid, name, guuid = [p.strip() for p in line.split(",")]
            if "ollama" in name.lower() or "runner" in name.lower():
                uuids[guuid] = True
        if not uuids:
            return None
        up = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        for line in up.strip().splitlines():
            idx, guuid = [p.strip() for p in line.split(",")]
            if guuid in uuids:
                return int(idx)
    except Exception:
        pass
    return None


def spawn_ollama(gpu: Optional[dict], host: str = "0.0.0.0:11434",
                 num_parallel: int = 1) -> subprocess.Popen:
    """Launch `ollama serve` pinned to `gpu` (or CPU) and record its pid."""
    env = os.environ.copy()
    env.update(gpu_env(gpu))
    env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    env["OLLAMA_HOST"] = host
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    log_path = LOG_DIR / "ollama-serve.log"
    with open(log_path, "ab") as logf:
        logf.write(f"\n# {datetime.now():%Y-%m-%d %H:%M:%S} start "
                   f"gpu={gpu_label(gpu)} host={host} num_parallel={num_parallel}\n".encode())
        logf.flush()
        proc = subprocess.Popen([OLLAMA_BIN, "serve"], stdout=logf,
                                stderr=subprocess.STDOUT, env=env,
                                cwd=str(HOME), start_new_session=True)
    OLLAMA_STATE["pid"] = proc.pid
    return proc


def wait_ollama_down(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ollama_up()["up"]:
            return True
        time.sleep(0.5)
    return not ollama_up()["up"]


def stop_ollama_managed() -> bool:
    """Stop an Ollama server that this GUI started. True if we sent the signal."""
    if OLLAMA_STATE["pid"]:
        try:
            os.kill(OLLAMA_STATE["pid"], signal.SIGTERM)
            OLLAMA_STATE["pid"] = None
            return True
        except OSError:
            OLLAMA_STATE["pid"] = None
    return False


def ensure_ollama(gpu_choice: str, host: Optional[str] = None,
                  num_parallel: Optional[int] = None) -> dict:
    """Make sure Ollama is up on the requested GPU.

    - already up on that GPU (or 'auto' matches)  -> leave it alone
    - up on a different GPU and managed here      -> restart on the requested one
    - up on a different GPU but started elsewhere -> ValueError (use force stop)
    - down                                        -> start it on the requested one

    Returns {"status": "already"|"started"|"restarted", "gpu": <smi index>}.
    """
    host = host or str(CONFIG["ollama"].get("host", "0.0.0.0:11434"))
    num_parallel = int(num_parallel or CONFIG["ollama"].get("num_parallel", 1))
    want = pick_gpu(gpu_choice, nvidia_smi_gpus())

    if ollama_up()["up"]:
        cur = ollama_gpu()
        on_target = (
            ((gpu_choice in ("auto", "")) and (want is None or cur == want["index"]))
            or (want is not None and cur == want["index"])
        )
        if on_target:
            return {"status": "already", "gpu": cur}
        if stop_ollama_managed():
            if not wait_ollama_down():
                raise ValueError("Ollama did not shut down in time — check Services → force stop")
        else:
            raise ValueError(
                f"Ollama is running on GPU {cur} but was not started by this GUI. "
                f"Use Services → force stop first, or untick 'relocate Ollama'.")
    spawn_ollama(want, host, num_parallel)
    for _ in range(20):  # wait for it to bind
        if ollama_up()["up"]:
            break
        time.sleep(0.5)
    return {"status": "restarted", "gpu": ollama_gpu()}


# ---------------------------------------------------------------------------
# squig LLM — the model that serves OpenCode + OWUI on port 8000
# ---------------------------------------------------------------------------
# Launch scripts live in ~/scripts: three vLLM servers inside the LXC
# container 'onecat' (squigFast / squigFastThink / squigCode) and one
# llama.cpp server on the host (squigChat). They all bind port 8000, so
# only one can be loaded at a time. Unloading frees the V100s' VRAM for
# OCR work; reloading re-claims it. The GUI spawns `bash ~/scripts/<name>`
# detached (own session) so the server survives a GUI restart; its output
# goes to ~/squig-llm.log.

LXC_CONT = os.environ.get("SQUIG_LXC", "onecat")
SQUIG_LOG = HOME / "squig-llm.log"

SQUIG_PROFILES = {
    # name: (launch script in ~/scripts, where it runs, what it is)
    "squigFast":      ("squigFast",      "lxc",  "qwen3.8-27B · fast · no thinking · 131k ctx"),
    "squigFastThink": ("squigFastThink", "lxc",  "qwen3.8-27B · thinking · 150k ctx"),
    "squigCode":      ("squigCode",      "lxc",  "orca · coding · 262k ctx"),
    "squigChat":      ("squigChat",      "host", "llama.cpp · Qwen3.8-Flash-Next · on the host"),
}


def lxc_cont_state() -> Optional[str]:
    """'running' / 'stopped' / … for the LXC container, or None (no lxc)."""
    try:
        r = subprocess.run(["lxc", "list", LXC_CONT, "--format", "json"],
                           capture_output=True, text=True, timeout=8)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout or "[]")
        return data[0].get("status") if data else None
    except Exception:
        return None


def lxc_exec(args: list, timeout: int = 10):
    """Run a command inside the LXC container. -> (rc, combined output)."""
    try:
        r = subprocess.run(["lxc", "exec", LXC_CONT, "--"] + args,
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 1, str(e)


def _squig_procs():
    """Find the LLM server process. -> (kind, pid, cmdline).

    kind is 'vllm' (inside the LXC container), 'llama' (llama-server on the
    host) or None when nothing is loaded."""
    rc, out = lxc_exec(["pgrep", "-af", "vllm.entrypoints.openai.api_server"],
                       timeout=6)
    if rc == 0 and out.strip():
        first = out.strip().splitlines()[0]
        return "vllm", int(first.split(None, 1)[0]), first
    try:
        r = subprocess.run(["pgrep", "-af", "llama-server"],
                           capture_output=True, text=True, timeout=6)
        if r.returncode == 0 and r.stdout.strip():
            first = r.stdout.strip().splitlines()[0]
            return "llama", int(first.split(None, 1)[0]), first
    except Exception:
        pass
    return None, None, None


def _detect_squig_profile(cmdline: str) -> str:
    """Which ~/scripts launch script does this process come from?"""
    if "llama-server" in cmdline:
        return "squigChat"
    if "/models/orca" in cmdline:
        return "squigCode"
    if '"enable_thinking":true' in cmdline or "--max-model-len 150000" in cmdline:
        return "squigFastThink"
    return "squigFast"


def _served_model_name(exec_health) -> Optional[str]:
    """Model id from /v1/models, via the given in-scope curl. -> str|None."""
    rc, out = exec_health(["curl", "-s", "-m", "3", "http://127.0.0.1:8000/v1/models"])
    if rc != 0:
        return None
    try:
        data = json.loads(out)
        items = data.get("data") or data.get("models") or []
        return items[0].get("id") if items else None
    except Exception:
        return None


def squig_llm_status() -> dict:
    """Live state of the squig LLM for the dashboard."""
    st = {"state": "stopped", "profile": None, "pid": None, "uptime": None,
          "served": None, "health": False,
          "container": lxc_cont_state(),
          "api": VLLM_BASE,
          "log": str(SQUIG_LOG),
          "profiles": {k: v[2] for k, v in SQUIG_PROFILES.items()},
          "last": (CONFIG.get("squig") or {}).get("last")}
    kind, pid, cmd = _squig_procs()
    if kind is None:
        return st
    in_cont = kind == "vllm"

    def _host_curl(args_, **_k):
        r = subprocess.run(args_, capture_output=True, text=True, timeout=6)
        return r.returncode, (r.stdout or "")

    curl = lxc_exec if in_cont else _host_curl
    st.update(state="starting", pid=pid,
              profile="squigChat" if in_cont is False else _detect_squig_profile(cmd))
    rc, _out = curl(["curl", "-sf", "-m", "3", "http://127.0.0.1:8000/health"])
    st["health"] = rc == 0
    st["state"] = "running" if rc == 0 else "starting"
    if st["health"]:
        st["served"] = _served_model_name(curl)
    try:
        ps = lxc_exec(["ps", "-o", "etime=", "-p", str(pid)], timeout=6) if in_cont \
            else (0, subprocess.run(["ps", "-o", "etime=", "-p", str(pid)],
                                    capture_output=True, text=True,
                                    timeout=6).stdout)
        if ps[0] == 0:
            st["uptime"] = ps[1].strip()
    except Exception:
        pass
    return st


def squig_llm_stop() -> dict:
    """Unload the LLM: SIGTERM (then SIGKILL) the server, report freed VRAM."""
    kind, pid, cmd = _squig_procs()
    if kind is None:
        return {"ok": True, "profile": None, "freed_mb": 0,
                "note": "no LLM running — V100s already free"}
    in_cont = kind == "vllm"
    profile = "squigChat" if in_cont is False else _detect_squig_profile(cmd)
    before = {g["index"]: g["mem_used_mb"] for g in nvidia_smi_gpus()}
    kill = (lambda sig, pat: lxc_exec(["pkill", sig, "-f", pat], timeout=6)) \
        if in_cont else (
            lambda sig, pat: (subprocess.run(["pkill", sig, "-f", pat],
                                             capture_output=True, text=True,
                                             timeout=6).returncode, ""))
    if in_cont:
        kill("-TERM", "vllm.entrypoints.openai.api_server")
        kill("-TERM", "VLLM::")          # EngineCore / worker children
    else:
        kill("-TERM", "llama-server")
    for _ in range(20):                   # vLLM can take a while to free VRAM
        if _squig_procs()[0] is None:
            break
        time.sleep(1)
    if _squig_procs()[0] is not None:     # still alive -> force
        (kill("-KILL", "vllm") if in_cont else kill("-KILL", "llama-server"))
        time.sleep(2)
    gone = _squig_procs()[0] is None
    after = {g["index"]: g["mem_used_mb"] for g in nvidia_smi_gpus()}
    freed = sum(max(0, before[i] - after.get(i, 0)) for i in before)
    return {"ok": gone, "profile": profile, "freed_mb": freed,
            "note": "unloaded — V100s free for OCR" if gone
            else "process still alive after SIGKILL — check the log"}


def squig_llm_start(profile: str) -> dict:
    """Load a model profile: spawn `bash ~/scripts/<name>` detached."""
    if profile not in SQUIG_PROFILES:
        return {"ok": False, "error": f"unknown profile '{profile}'"}
    script_name, where, _desc = SQUIG_PROFILES[profile]
    kind, pid, cmd = _squig_procs()
    if kind is not None:
        cur = "squigChat" if kind == "llama" else _detect_squig_profile(cmd)
        return {"ok": False,
                "error": f"{cur} is already running (pid {pid}) — stop it "
                         f"first; only one model can hold port 8000"}
    if where == "lxc" and lxc_cont_state() != "running":
        return {"ok": False,
                "error": f"LXC container '{LXC_CONT}' is not running — "
                         f"start it first (button on the left)"}
    script = HOME / "scripts" / script_name
    if not script.exists():
        return {"ok": False, "error": f"launch script not found: {script}"}
    try:
        with open(SQUIG_LOG, "ab") as logf:
            proc = subprocess.Popen(
                ["bash", str(script)], stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                cwd=str(HOME))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        CONFIG.setdefault("squig", {})["last"] = profile
        save_config(CONFIG)
    except Exception:
        pass
    return {"ok": True, "profile": profile, "pid": proc.pid,
            "note": "loading — weights take a minute or two; watch the state "
                    "and the log below"}


def lxc_cont_start() -> dict:
    r = subprocess.run(["lxc", "start", LXC_CONT],
                       capture_output=True, text=True, timeout=60)
    return {"ok": r.returncode == 0,
            "error": None if r.returncode == 0
            else (r.stderr or r.stdout or "").strip()[:200]}


def squig_llm_log_tail(n: int = 12) -> list:
    if not SQUIG_LOG.exists():
        return []
    try:
        return SQUIG_LOG.read_text(errors="replace").splitlines()[-n:]
    except Exception:
        return []


def services_snapshot() -> dict:
    snap = {"qdrant": {}, "ollama": {}, "gpus": nvidia_smi_gpus(),
            "power": {"sudo_ok": sudo_nvidia_sm_ok()},
            "squig": squig_llm_status()}
    try:
        r = requests.get(f"{QDRANT_URL}/", timeout=3)
        snap["qdrant"] = {"up": True, "version": r.json().get("version")}
        try:
            c = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
                              json={"exact": True}, timeout=5).json()
            snap["qdrant"]["points"] = c["result"]["count"]
        except Exception:
            snap["qdrant"]["points"] = None
    except Exception:
        snap["qdrant"] = {"up": False}

    o = ollama_up()
    snap["ollama"] = {
        "up": o["up"],
        "version": o["version"],
        "models": ollama_models() if o["up"] else [],
        "gpu_smi_index": ollama_gpu() if o["up"] else None,
        "managed": OLLAMA_STATE["pid"] is not None and o["up"],
        "managed_alive": False,
    }
    if OLLAMA_STATE["pid"]:
        try:
            os.kill(OLLAMA_STATE["pid"], 0)
            snap["ollama"]["managed_alive"] = True
        except OSError:
            OLLAMA_STATE["pid"] = None
    return snap


# ---------------------------------------------------------------------------
# Pipeline progress (OCR + Qdrant indexing): "where is it, how far to go"
# ---------------------------------------------------------------------------
def _recent_rate(samples, window_s: float = 300.0):
    """Units/second from a [(ts, value), ...] series.

    Prefers the most recent `window_s` (current pace, robust to the
    fast-skip stretches at the start of an index run); falls back to the
    whole series. None when there is not enough data or nothing advanced.
    """
    try:
        s = [[float(a), float(b)] for a, b in (samples or [])]
    except (TypeError, ValueError):
        return None
    if len(s) < 2:
        return None
    last_ts, last_v = s[-1]
    window_ref = s[0]
    for p in s:
        if last_ts - p[0] <= window_s:
            window_ref = p
            break
    for ref in (window_ref, s[0]):
        dt = last_ts - ref[0]
        if dt < 10:
            continue
        dv = last_v - ref[1]
        if dv > 0:
            return dv / dt
    return None


def _progress_view(f: Path, d: dict, now: float) -> dict:
    """Normalize one progress file into the shape the UI wants, computing
    freshness (is the writer process alive?), pace and ETA."""
    last = float(d.get("last_update") or 0)
    age = max(0.0, now - last)
    pid = d.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError, TypeError):
            alive = False
    state = d.get("state") or "running"
    if state == "running" and not alive:
        state = "stalled"       # writer died without finishing
    if state == "running" and alive and age >= 120:
        state = "quiet"         # alive, no updates (long book / model load)
    active = state == "running" and alive and age < 120

    total_items = d.get("total_books") or d.get("total_files") or 0
    done_items = d.get("done_books") or d.get("done_files") or 0
    total_pages = d.get("total_pages") or 0
    pages_done = d.get("pages_done")
    if pages_done is None:
        pages_done = d.get("pages_embedded")
    remaining_pages = d.get("remaining_pages")
    if remaining_pages is None:
        remaining_pages = max(0, total_pages - (pages_done or 0)) if total_pages else None

    pace_pages = _recent_rate(d.get("samples"))
    pace_books = _recent_rate(d.get("book_samples"))
    # ETA: in pages when we have a page pace (accurate across books of very
    # different size), otherwise in books.
    if remaining_pages and pace_pages:
        eta_s, eta_basis = remaining_pages / pace_pages, "pages"
    elif total_items > done_items and pace_books:
        eta_s, eta_basis = (total_items - done_items) / pace_books, "books"
    else:
        eta_s, eta_basis = None, None

    return {
        "file": f.name, "state": state, "active": active, "alive": alive,
        "age_s": round(age), "pid": pid,
        "total_books": d.get("total_books"), "done_books": d.get("done_books"),
        "total_files": d.get("total_files"), "done_files": d.get("done_files"),
        "failed_files": d.get("failed_files"), "counts": d.get("counts"),
        "pct_items": round(100.0 * done_items / total_items, 1) if total_items else None,
        "total_pages": total_pages or None, "pages_done": pages_done,
        "remaining_pages": remaining_pages,
        "pct_pages": round(100.0 * (pages_done or 0) / total_pages, 1)
                     if (total_pages and pages_done is not None) else None,
        "current": d.get("current_book") or d.get("current_file"),
        "current_pages": d.get("current_book_pages") or d.get("current_file_pages") or 0,
        "current_pages_done": d.get("current_book_pages_done") or d.get("current_file_pages_done") or 0,
        "pace_pages_per_min": round(pace_pages * 60, 1) if pace_pages else None,
        "pace_books_per_min": round(pace_books * 60, 2) if pace_books else None,
        "eta_s": round(eta_s) if eta_s else None, "eta_basis": eta_basis,
        "input": d.get("input"), "output": d.get("output"), "root": d.get("root"),
        "last_update": last,
    }


def _indexer_log_view() -> Optional[dict]:
    """Fallback for an indexer that predates the progress file: the process
    is alive, so derive position + book pace from the shared index_run.log
    (timestamped 'Book N/M' lines). Page-level numbers need the new writer."""
    try:
        r = subprocess.run(["pgrep", "-f", "index_books[.]py --root"],
                           capture_output=True, text=True, timeout=5)
        pids = [x for x in r.stdout.split() if x and x != str(os.getpid())]
    except Exception:
        return None
    if not pids:
        return None
    log_path = INDEXER_DIR / "index_run.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="ignore").splitlines()[-400:]
    except OSError:
        return None
    entries = []
    for line in lines:
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO Book (\d+)/(\d+):", line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            continue
        entries.append((ts, int(m.group(2)), int(m.group(3)), line))
    if not entries:
        return {"state": "running", "active": True, "alive": True, "pid": int(pids[0]),
                "age_s": 0, "total_books": None, "done_books": 0,
                "current": "(starting up)", "note": "live from the shared log"}
    last_ts, done, total, last_line = entries[-1]
    age = time.time() - last_ts
    ref = entries[0]
    for e in entries:
        if last_ts - e[0] <= 300:
            ref = e
            break
    pace_books = None
    if last_ts - ref[0] >= 20:
        pace_books = (done - ref[1]) / (last_ts - ref[0])
    eta_min = (total - done) / pace_books / 60 if (pace_books and total > done) else None
    m2 = re.search(r'Book \d+/\d+: "(.*)"', last_line)
    return {
        "state": "running" if age < 300 else "quiet", "active": age < 120,
        "alive": True, "pid": int(pids[0]), "age_s": round(age),
        "total_books": total, "done_books": done,
        "pct_items": round(100.0 * done / total, 1) if total else None,
        "total_pages": None, "pages_done": None, "remaining_pages": None,
        "pct_pages": None, "current": m2.group(1) if m2 else "(between books)",
        "current_pages": 0, "current_pages_done": 0,
        "pace_pages_per_min": None,
        "pace_books_per_min": round(pace_books * 60, 2) if pace_books else None,
        "eta_s": round(eta_min * 60) if eta_min else None, "eta_basis": "books",
        "last_update": last_ts,
        "note": "live from the shared log — restart the indexer after updating "
                "index_books.py for page-level progress + page ETA",
    }


def _shard_group_key(name: str, d: dict) -> tuple:
    """Group progress files belonging to one job: the base filename (shard
    suffix `_g<i>` stripped) + the job id. Sharded jobs started from the GUI
    carry a job id on every shard; plain single-process runs have job None
    and always stand alone."""
    base = re.sub(r"_g\d+$", "", name)
    return (base, d.get("job"))


def _merge_shard_views(names: list, datas: list, now: float) -> dict:
    """One dashboard card for a sharded (multi-GPU) job: files/pages summed
    across shards, a per-worker breakdown, the summed pace, one ETA.

    Shards keep their job id, so a shard that already FINISHED stays in the
    group and counts as done. State: any live worker keeps the card
    'running'; otherwise quiet > stalled > failed > done."""
    views = [_progress_view(PROGRESS_DIR / n, d, now)
             for n, d in zip(names, datas)]
    states = [v["state"] for v in views]
    if "running" in states:
        state = "running"
    elif "quiet" in states:
        state = "quiet"
    elif "stalled" in states:
        state = "stalled"
    elif "failed" in states:
        state = "failed"
    else:
        state = "done"

    total_files = sum(d.get("total_files") or 0 for d in datas)
    done_files = sum(d.get("done_files") or 0 for d in datas)
    failed_files = sum(d.get("failed_files") or 0 for d in datas)
    total_pages = sum(d.get("total_pages") or 0 for d in datas)
    pages_done = sum(d.get("pages_done") or 0 for d in datas)
    remaining = 0
    for d in datas:
        rem = d.get("remaining_pages")
        if rem is None:
            rem = max(0, (d.get("total_pages") or 0) - (d.get("pages_done") or 0))
        remaining += int(rem or 0)
    # Pages/min is additive across parallel workers — sum the per-shard
    # rates instead of trying to merge their sample series.
    pace = sum(r for r in (_recent_rate(d.get("samples")) for d in datas) if r)
    latest = max(views, key=lambda v: v.get("last_update") or 0)
    eta_s = (remaining / pace) if (pace and remaining > 0) else None
    workers = []
    for i, (d, v) in enumerate(zip(datas, views)):
        workers.append({
            "shard": i, "gpu": d.get("gpu"),
            "file": d.get("current_file") or d.get("current_book"),
            "pages_done": (d.get("current_file_pages_done")
                           or d.get("current_book_pages_done") or 0),
            "pages": (d.get("current_file_pages")
                      or d.get("current_book_pages") or 0),
            "state": v["state"], "age_s": v["age_s"], "pid": d.get("pid"),
        })
    return {
        "file": " + ".join(names),
        "state": state, "active": any(v["active"] for v in views),
        "alive": any(v["alive"] for v in views),
        "age_s": min(v["age_s"] for v in views),
        "pid": latest.get("pid"), "pids": [d.get("pid") for d in datas],
        "total_books": sum(d.get("total_books") or 0 for d in datas) or None,
        "done_books": sum(d.get("done_books") or 0 for d in datas) or None,
        "total_files": total_files or None,
        "done_files": done_files or None,
        "failed_files": failed_files or None,
        "counts": None,
        "pct_items": round(100.0 * done_files / total_files, 1) if total_files else None,
        "total_pages": total_pages or None,
        "pages_done": pages_done or None,
        "remaining_pages": remaining or None,
        "pct_pages": round(100.0 * pages_done / total_pages, 1)
                      if (total_pages and pages_done) else None,
        "current": latest.get("current"),
        "current_pages": latest.get("current_pages") or 0,
        "current_pages_done": latest.get("current_pages_done") or 0,
        "pace_pages_per_min": round(pace * 60, 1) if pace else None,
        "pace_books_per_min": None,
        "eta_s": round(eta_s) if eta_s else None,
        "eta_basis": "pages" if eta_s else None,
        "input": datas[0].get("input"), "output": datas[0].get("output"),
        "root": datas[0].get("root"),
        "shards": len(names), "workers": workers,
        "last_update": max(v.get("last_update") or 0 for v in views),
    }


def pipeline_snapshot() -> dict:
    """Live state of the OCR + indexing pipeline for the dashboard.

    Precedence per indexer: a fresh progress file wins; an alive process
    whose file has not been updated in 2 min (old build) falls back to the
    shared log; a dead process leaves its file as the last known state.
    Sharded (multi-GPU) jobs are merged into one card per job; an older
    finished run of a dir that has a live sharded job is hidden."""
    now = time.time()
    out = {"indexer": None, "ocr": [], "dir": str(PROGRESS_DIR)}
    files: dict = {}
    if PROGRESS_DIR.exists():
        for f in PROGRESS_DIR.glob("*.json"):
            try:
                files[f.name] = json.loads(f.read_text())
            except Exception:
                continue

    ocr_files = {n: d for n, d in files.items()
                 if n.startswith("ocr_") or d.get("process") == "ocr"}
    ocr_groups: dict = {}
    for n, d in ocr_files.items():
        ocr_groups.setdefault(_shard_group_key(n, d), []).append(n)
    active_bases = set()
    for key, names in ocr_groups.items():
        if any(_progress_view(PROGRESS_DIR / n, ocr_files[n], now)["state"]
               in ("running", "quiet", "stalled") for n in names):
            active_bases.add(key[0])
    for key, names in ocr_groups.items():
        names = sorted(names)
        if len(names) > 1:
            out["ocr"].append(_merge_shard_views(
                names, [ocr_files[n] for n in names], now))
            continue
        n = names[0]
        view = _progress_view(PROGRESS_DIR / n, ocr_files[n], now)
        if (view["state"] in ("done", "failed") and key[1] is None
                and key[0] in active_bases):
            continue  # earlier plain run of a dir whose sharded job is live
        out["ocr"].append(view)
    out["ocr"].sort(key=lambda v: -(v.get("last_update") or 0))

    idx = {n: d for n, d in files.items()
           if n.startswith("index_") or d.get("process") == "indexer"}
    if idx:
        idx_groups: dict = {}
        for n, d in idx.items():
            idx_groups.setdefault(_shard_group_key(n, d), []).append(n)
        best = max(idx_groups.items(),
                   key=lambda kv: max(idx[n].get("last_update") or 0
                                       for n in kv[1]))
        names = sorted(best[1])
        if len(names) > 1:
            out["indexer"] = _merge_shard_views(
                names, [idx[n] for n in names], now)
        else:
            name = names[0]
            d = idx[name]
            view = _progress_view(PROGRESS_DIR / name, d, now)
            if d.get("state") == "running" and view["alive"] and \
                    (now - (d.get("last_update") or 0)) > 120:
                lv = _indexer_log_view()
                out["indexer"] = lv or view
            else:
                out["indexer"] = view
    else:
        out["indexer"] = _indexer_log_view()
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/pipeline")
def api_pipeline():
    """Live OCR + Qdrant-indexing progress: where each run is, how far it
    has to go (files/pages), the current pace, and an ETA. Works for runs
    started from this GUI or from the CLI/supervisor (the processes write
    the progress files themselves)."""
    return jsonify(pipeline_snapshot())


@app.get("/api/config")
def get_config():
    return jsonify(CONFIG)


@app.post("/api/config")
def post_config():
    global CONFIG
    data = request.get_json(force=True)
    for k, v in data.items():
        if k in CONFIG:
            if isinstance(CONFIG[k], dict) and isinstance(v, dict):
                CONFIG[k].update(v)
            else:
                CONFIG[k] = v
    save_config(CONFIG)
    return jsonify(CONFIG)


@app.get("/api/gpus")
def api_gpus():
    return jsonify(nvidia_smi_gpus())


@app.post("/api/gpus/power")
def api_gpu_power():
    """Set a GPU power limit (nvidia-smi -pl, as root).

    body: {"gpu": 0 | 2 | "v100s", "watts": 150 | 300 | "default"}
    Remembers the result in config.json (gpu_power map) so start-all.sh can
    re-apply it after a reboot — the limit resets on every power cycle."""
    data = request.get_json(force=True)
    target = data.get("gpu", "v100s")
    watts = data.get("watts")
    if watts in (None, ""):
        return jsonify({"error": "watts required (a number, or 'default')"}), 400
    try:
        res = gpu_set_power(target, watts)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    if res.get("ok"):
        try:
            saved = CONFIG.setdefault("gpu_power", {})
            for idx in (res.get("targets") or []):
                lim = (res.get("limits") or {}).get(str(idx))
                if lim is not None:
                    saved[str(idx)] = int(lim)
            save_config(CONFIG)
        except Exception:
            pass  # persisting the preference is a nicety
    return jsonify(res)


@app.get("/api/services")
def api_services():
    return jsonify(services_snapshot())


@app.get("/api/squig-llm")
def api_squig_llm():
    """State of the squig LLM (which profile, running/loading/stopped,
    health, uptime, container state, profiles available)."""
    return jsonify(squig_llm_status())


@app.get("/api/squig-llm/log")
def api_squig_llm_log():
    n = request.args.get("lines", 12, type=int)
    return jsonify({"lines": squig_llm_log_tail(max(1, min(n, 200)))})


@app.post("/api/squig-llm")
def api_squig_llm_ctrl():
    """Load/unload the squig LLM.

    body: {"action": "stop"}                       unload whatever is loaded
           {"action": "start", "profile": <name>}  load a profile
           {"action": "container-start"}           start the LXC container
    """
    data = request.get_json(force=True)
    action = data.get("action")
    if action == "stop":
        res = squig_llm_stop()
    elif action == "start":
        res = squig_llm_start(str(data.get("profile") or "").strip())
    elif action == "container-start":
        res = lxc_cont_start()
    else:
        return jsonify({"error": "action must be 'stop', 'start' or 'container-start'"}), 400
    return jsonify(res)


@app.post("/api/services/ollama")
def api_ollama_control():
    data = request.get_json(force=True)
    action = data.get("action")
    if action in ("stop", "stop-force"):
        if not ollama_up()["up"] and action != "stop-force":
            return jsonify({"status": "not running"})
        if stop_ollama_managed():
            return jsonify({"status": "stopping"})
        if action == "stop-force":
            # kill whatever is serving on the Ollama port, managed or not
            try:
                subprocess.run(["pkill", "-TERM", "-f", "ollama serve"],
                               capture_output=True, timeout=10)
            except Exception:
                pass
            if wait_ollama_down():
                return jsonify({"status": "stopped"})
            try:
                subprocess.run(["pkill", "-KILL", "-f", "ollama serve"],
                               capture_output=True, timeout=10)
            except Exception:
                pass
            return jsonify({"status": "stopped (SIGKILL)"})
        return jsonify({"status": "external",
                        "message": "Ollama is running but was not started from this GUI. "
                                   "Use 'force stop' to kill it, then Start here."})
    if action == "start":
        if ollama_up()["up"]:
            return jsonify({"status": "already running",
                            "gpu_smi_index": ollama_gpu()})
        gpus = nvidia_smi_gpus()
        gpu = pick_gpu(data.get("gpu", "auto"), gpus)
        host = str(data.get("host", "0.0.0.0:11434")).strip() or "0.0.0.0:11434"
        num_parallel = int(data.get("num_parallel", 1))
        spawn_ollama(gpu, host, num_parallel)
        # give it a moment to bind
        time.sleep(2)
        up = ollama_up()["up"]
        if not up:
            return jsonify({"status": "starting", "pid": OLLAMA_STATE["pid"],
                            "note": "still binding; check /api/services"})
        return jsonify({"status": "started", "pid": OLLAMA_STATE["pid"],
                        "gpu_smi_index": ollama_gpu(),
                        "expected_gpu": gpu["index"] if gpu else None})
    return jsonify({"error": "unknown action"}), 400


@app.post("/api/services/qdrant")
def api_qdrant_control():
    data = request.get_json(force=True)
    try:
        t = start_task("qdrant", {"action": data.get("action", "up")})
        return jsonify(t.public())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/tasks")
def api_tasks():
    with TASKS_LOCK:
        items = sorted(TASKS.values(), key=lambda t: t.started_at, reverse=True)
    return jsonify([t.public() for t in items[:50]])


@app.post("/api/tasks")
def api_tasks_start():
    data = request.get_json(force=True)
    kind = data.get("kind")
    params = data.get("params", {})
    # persist last-used params
    if kind in CONFIG and isinstance(CONFIG[kind], dict):
        save_cfg = dict(params)
        save_cfg.pop("env", None)
        if kind == "ocr":
            save_cfg["env"] = {**(CONFIG["ocr"].get("env") or {}),
                               **(params.get("env") or {})}
        CONFIG[kind].update(save_cfg)
        save_config(CONFIG)
    try:
        result = start_task(kind, params)
        tasks = result if isinstance(result, list) else [result]
        return jsonify({"tasks": [t.public() for t in tasks]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/tasks/<task_id>/stop")
def api_tasks_stop(task_id):
    ok = stop_task(task_id)
    return jsonify({"stopped": ok})


@app.get("/api/tasks/<task_id>/log")
def api_task_log(task_id):
    with TASKS_LOCK:
        t = TASKS.get(task_id)
    if not t:
        return jsonify({"error": "no such task"}), 404
    after = int(request.args.get("after", 0))
    try:
        size = os.path.getsize(t.log_path)
    except OSError:
        return jsonify({"chunk": "", "size": 0, "done": t.status != "running"})
    if after > size:
        after = size
    with open(t.log_path, "rb") as f:
        f.seek(after)
        chunk = f.read(min(size - after, 1_000_000)).decode("utf-8", "replace")
    return jsonify({"chunk": chunk, "size": size, "done": t.status != "running",
                    "status": t.status})


@app.get("/api/browse")
def api_browse():
    """Simple directory browser for path pickers."""
    path = Path(request.args.get("path", str(HOME))).expanduser()
    try:
        path = path.resolve()
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        dirs = [p.name for p in entries if p.is_dir()][:500]
        pdfs = len(list(path.glob("*.pdf")))
        parquets = len(list(path.glob("*.parquet")))
        return jsonify({
            "path": str(path),
            "parent": str(path.parent) if path.parent != path else None,
            "dirs": dirs,
            "pdfs": pdfs,
            "parquets": parquets,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "python": PY, "base": str(BASE_DIR),
                    "ddc": str(DDC_DIR)})


@app.post("/api/search")
def api_search():
    """Semantic search over the ddc-books collection (embed via Ollama, query Qdrant)."""
    data = request.get_json(force=True)
    query = str(data.get("query", "")).strip()
    top_k = max(1, min(50, int(data.get("top_k", 8) or 8)))
    if not query:
        return jsonify({"error": "empty query"}), 400
    emb_url = str(CONFIG["index"].get("embed_base_url", OLLAMA_URL_DEFAULT)).strip()
    model = str(CONFIG["index"].get("embed_model", "bge-m3")).strip()
    qurl = str(CONFIG["index"].get("qdrant_url", QDRANT_URL)).strip()
    try:
        r = requests.post(f"{emb_url}/api/embed",
                          json={"model": model, "input": query}, timeout=120)
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
    except Exception as e:
        return jsonify({"error": f"embedding failed (is Ollama up with {model}?): {e}"}), 502
    try:
        r = requests.post(f"{qurl}/collections/{COLLECTION}/points/query",
                          json={"query": vec, "limit": top_k, "with_payload": True,
                                "with_vectors": False}, timeout=60)
        r.raise_for_status()
        res = r.json()["result"]
        # Qdrant 1.20: raw-vector query -> {"result": {"points": [...]}};
        # some query variants return a bare list.
        hits = res["points"] if isinstance(res, dict) else res
    except Exception as e:
        return jsonify({"error": f"qdrant query failed: {e}"}), 502
    results = []
    for h in hits:
        pl = h.get("payload") or {}
        results.append({
            "score": round(float(h.get("score", 0.0)), 4),
            "book_title": pl.get("book_title"),
            "authors": pl.get("authors"),
            "pub_date": pl.get("pub_date"),
            "ddc": pl.get("ddc"),
            "ddc_coarse": pl.get("ddc_coarse"),
            "page_num": pl.get("page_num"),
            "book_id": pl.get("book_id"),
            "text": (pl.get("text") or "")[:800],
        })
    return jsonify({"query": query, "results": results})


@app.get("/api/llm-test")
def api_llm_test():
    """Ping an OpenAI-compatible endpoint's /models (for the recipe LLM fields)."""
    url = str(request.args.get("url", "")).strip().rstrip("/")
    if not url:
        return jsonify({"error": "url required"}), 400
    try:
        r = requests.get(f"{url}/models", timeout=8)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"})


from werkzeug.exceptions import HTTPException  # noqa: E402


@app.errorhandler(Exception)
def handle_error(e):
    """Return JSON for API errors instead of Werkzeug's HTML page."""
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    app.logger.exception("unhandled error")
    return jsonify(error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=GUI_PORT, threaded=True)
