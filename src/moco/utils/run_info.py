"""Provenance written into every output directory: resolved config, git state, SLURM job."""
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_state() -> dict:
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}


def save_run_info(out_dir: Path, cfg: DictConfig) -> None:
    """Write the resolved config.yaml and append one provenance line per (re)start to run_info.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    info = {
        **git_state(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    with open(out_dir / "run_info.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")
