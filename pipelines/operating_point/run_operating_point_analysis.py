"""Run the shared E/I and regression analysis at m=0.90 and/or m=0.95."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPOSITORY_ROOT / "pipelines" / "analysis"


def run(script: str, env: dict[str, str]):
    path = ANALYSIS_DIR / script
    print("\n" + "=" * 88)
    print("RUN", path.name, flush=True)
    subprocess.run([sys.executable, str(path)], cwd=REPOSITORY_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work"),
    )
    parser.add_argument("--only-m", choices=("0.90", "0.95"))
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    allen = project / "ALLEN"
    conditions = (args.only_m,) if args.only_m else ("0.90", "0.95")
    for m_value in conditions:
        env = os.environ.copy()
        env.update({
            "PROJECT_ROOT": str(project),
            "ALLEN_ROOT": str(allen),
            "TRIALS_ROOT": str(allen / "file" / "add_trial" / f"m_{m_value}"),
            "SAVE_ROOT_REAL": str(allen / "save" / f"m_{m_value}" / "final_real"),
            "SAVE_ROOT_ER": str(allen / "save" / f"m_{m_value}" / "final_ER"),
            "RESULTS_TAG": f"m_{m_value}",
            "MAX_WORKERS": str(args.max_workers),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")])
            ),
        })
        print("\n" + "#" * 88)
        print(f"OPERATING-POINT ANALYSIS: m={m_value}")
        print("#" * 88)
        run("compute_trial_ei_ratios.py", env)
        run("compute_main_exponents.py", env)
        run("fit_main_regressions.py", env)

    print("[ALL COMPLETE] operating-point analyses")


if __name__ == "__main__":
    main()
