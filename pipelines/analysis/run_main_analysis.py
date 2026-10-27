"""运行主要兴奋/抑制比、频谱、回归和子集验证分析。 / Run the primary E/I, spectral, regression, and subset-validation analyses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(script: str, env: dict[str, str]):
    path = HERE / script
    print("\n" + "=" * 88)
    print("RUN", path.name, flush=True)
    subprocess.run([sys.executable, str(path)], cwd=REPOSITORY_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work"),
    )
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--skip-subset-validation", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(project_root),
        "ALLEN_ROOT": str(project_root / "ALLEN"),
        "TRIALS_ROOT": str(project_root / "ALLEN" / "file" / "final_trial"),
        "MAX_WORKERS": str(args.max_workers),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")])
        ),
    })

    run("compute_trial_ei_ratios.py", env)
    run("compute_main_exponents.py", env)
    run("fit_main_regressions.py", env)
    if not args.skip_subset_validation:
        run("validate_stratified_subset.py", env)
    print("[ALL COMPLETE] primary statistical analysis")


if __name__ == "__main__":
    main()
