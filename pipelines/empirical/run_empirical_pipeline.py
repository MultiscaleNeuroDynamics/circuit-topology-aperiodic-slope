"""运行公开ds003380丙泊酚验证流程。不重新分发OpenNeuro数据；设置PIG_DATA_ROOT或通过--data-root指定本地ds003380下载目录。 / Run the public ds003380 propofol validation pipeline.

The OpenNeuro data are not redistributed. Set ``PIG_DATA_ROOT`` or pass
``--data-root`` to the local ds003380 download.
"""

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
    parser.add_argument("--data-root", type=Path, required=True, help="Local OpenNeuro ds003380 directory.")
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")),
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--skip-workflow-figure", action="store_true")
    parser.add_argument("--pig-image", type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not (data_root / "participants.tsv").is_file():
        raise FileNotFoundError(f"participants.tsv was not found under {data_root}")

    results_root = args.project_root.resolve() / "ALLEN" / "results" / "empirical_propofol"
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(args.project_root.resolve()),
        "PIG_DATA_ROOT": str(data_root),
        "PIG_RESULTS_ROOT": str(results_root),
    })
    if args.pig_image:
        env["PIG_IMAGE_PATH"] = str(args.pig_image.resolve())

    run("inspect_openneuro_dataset.py", env)
    if args.inspect_only:
        return
    run("extract_propofol_features.py", env)
    run("analyze_propofol_effects.py", env)
    run("plot_propofol_results.py", env)
    if not args.skip_workflow_figure:
        run("plot_dataset_workflow.py", env)
    print(f"[ALL COMPLETE] empirical results: {results_root}")


if __name__ == "__main__":
    main()
