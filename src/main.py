"""src/main.py
Top-level orchestrator – spawns the actual training run in a subprocess so that
GitHub Actions (or any scheduler) can launch many runs independently.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # type: ignore
    # Validate mode ------------------------------------------------------------
    if cfg.mode not in ("trial", "full"):
        raise ValueError("mode must be 'trial' or 'full'")

    # Prepare subprocess command ----------------------------------------------
    original_cwd = Path(get_original_cwd())

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    print("Launching training subprocess:\n", " ".join(cmd))
    subprocess.run(cmd, cwd=original_cwd, check=True)


if __name__ == "__main__":
    main()
