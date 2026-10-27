# -*- coding: utf-8 -*-
"""端点生物物理对照的一键运行、支持断点续跑的控制器。 / One-command, restart-safe controller for the endpoint biophysical control."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def run(name, arguments):
    cmd = [sys.executable, str(HERE / name), *arguments]
    print("\n" + "=" * 88 + "\nRUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work"))
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--analysis-only", action="store_true")
    args = p.parse_args()
    common = ["--project-root", args.project_root]
    if not args.analysis_only:
        run("generate_profiles.py", common)
        if args.prepare_only:
            print("[READY] Profiles generated. Rerun without --prepare-only to start NEURON.")
            return
        run("run_lfp_profiles.py", [*common, "--max-workers", str(args.max_workers)])
    run("analyze_profiles.py", common)


if __name__ == "__main__":
    main()
