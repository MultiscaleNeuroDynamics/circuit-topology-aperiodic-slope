#!/usr/bin/env python3
"""根据版本控制中的MICrONS输入准备生成数据工作区。 / Prepare the generated workspace from version-controlled MICrONS inputs."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from circuit_topology.repo_paths import ALLEN_ROOT, DATA_ROOT


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required input directory is missing: {source}")
    for source_file in source.rglob("*"):
        if source_file.is_file():
            copy_file(source_file, destination / source_file.relative_to(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ALLEN_ROOT.parent,
        help="Generated workspace root (default: repository/work).",
    )
    args = parser.parse_args()

    # 为按同一组终端指令启动的子进程设置此值。 / Set this for child processes launched from the same shell instructions.
    workspace_root = args.workspace_root.resolve()
    allen_root = workspace_root / "ALLEN"
    file_root = allen_root / "file"
    for path in (file_root, allen_root / "save", allen_root / "results"):
        path.mkdir(parents=True, exist_ok=True)

    for name in (
        "proofread_cells_celltypes_auto.csv",
        "proofread_cells_celltypes_all.csv",
        "proofread_cells_celltypes_manual.csv",
        "proofread_cells_metadata.csv",
        "proofread_cells_synapses.csv",
        "aibs_metamodel_celltypes_v661_full.csv",
    ):
        copy_file(DATA_ROOT / name, file_root / name)

    copy_tree(DATA_ROOT / "patch", file_root / "patch")
    copy_tree(DATA_ROOT / "morphologies", file_root / "select_swc")
    copy_tree(DATA_ROOT / "representative_morphologies", file_root / "swc")
    copy_tree(DATA_ROOT / "synapse_segment_mapping", file_root / "synapse_segment_mapping")

    print(f"Workspace prepared at: {workspace_root}")
    print(f"For subsequent commands set PROJECT_ROOT={workspace_root}")


if __name__ == "__main__":
    main()
