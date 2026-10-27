# -*- coding: utf-8 -*-
import os
import re
import glob
import shutil
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    str(REPOSITORY_ROOT / "work")
)
ALLEN_ROOT = os.environ.get("ALLEN_ROOT", os.path.join(PROJECT_ROOT, "ALLEN"))
for module_root in (str(REPOSITORY_ROOT / "src"), PROJECT_ROOT, ALLEN_ROOT):
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
from circuit_topology.real_synapse_mapping import build_clean_synapse_segment_maps


FINAL_TRIAL_DIR = os.environ.get(
    "TRIALS_ROOT",
    os.path.join(ALLEN_ROOT, "file", "final_trial")
)
PATCH_DIR = os.environ.get(
    "PATCH_DIR",
    os.path.join(ALLEN_ROOT, "file", "patch")
)
SELECT_SWC_ROOT = os.environ.get(
    "SELECT_SWC_ROOT",
    os.path.join(ALLEN_ROOT, "file", "select_swc")
)

# 文件处理细节。 / File-handling details.
LAYER_TO_FOLDER = {
    "L2/3": "L2_3",
    "L4": "L4",
    "L5": "L5",
    "L6": "L6",
}

LAYER_TO_NEWNAME = {
    "L2/3": "23.swc",
    "L4": "4.swc",
    "L5": "5.swc",
    "L6": "6.swc",
}


def list_er_trial_dirs(root_dir: str):
    ds = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and re.match(r"^ER_trial\d+$", name):
            ds.append(p)

    def _trial_num(path):
        m = re.search(r"ER_trial(\d+)$", os.path.basename(path))
        return int(m.group(1)) if m else 10**9

    ds.sort(key=_trial_num)
    return ds


def pick_random_swc(layer_key: str, select_root: str, rng: random.Random):
    folder = LAYER_TO_FOLDER[layer_key]
    layer_dir = os.path.join(select_root, folder)
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(f"Layer folder not found: {layer_dir}")

    swcs = sorted(glob.glob(os.path.join(layer_dir, "*.swc")))
    if not swcs:
        raise FileNotFoundError(f"No .swc files found in: {layer_dir}")

    return rng.choice(swcs)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(seed: int = 0):
    trial_dirs = list_er_trial_dirs(FINAL_TRIAL_DIR)
    if not trial_dirs:
        raise RuntimeError(f"No ER_trial<number> folders found under: {FINAL_TRIAL_DIR}")

    for outdir in trial_dirs:
        ensure_dir(outdir)

        expected = [
            os.path.join(outdir, "23.swc"),
            os.path.join(outdir, "4.swc"),
            os.path.join(outdir, "5.swc"),
            os.path.join(outdir, "6.swc"),
            os.path.join(outdir, "syn_to_segment_L2_3.csv"),
            os.path.join(outdir, "syn_to_segment_L4.csv"),
            os.path.join(outdir, "syn_to_segment_L5.csv"),
            os.path.join(outdir, "syn_to_segment_L6.csv"),
        ]
        if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in expected):
            print(f"[SKIP] {os.path.basename(outdir)}: SWCs and REAL maps already complete")
            continue

        # 确定性的逐试次选择：即使跳过前面已完成的试次，两个m / Deterministic per-trial selection: corresponding trials in both m
        # 条件中相对应的试次仍获得相同SWC形态。 / conditions receive the same SWCs, even when earlier trials are skipped.
        match = re.search(r"ER_trial(\d+)$", os.path.basename(outdir))
        trial_number = int(match.group(1)) if match else 0
        rng = random.Random(seed + trial_number)

        # 可重复随机化细节。 / Reproducible randomization detail.
        SWC_PATHS = {}
        picked = {}
        for layer in ["L2/3", "L4", "L5", "L6"]:
            swc_path = pick_random_swc(layer, SELECT_SWC_ROOT, rng)
            SWC_PATHS[layer] = swc_path
            picked[layer] = swc_path

        # 路径与配置。 / Paths and configuration.
        build_clean_synapse_segment_maps(
            PATCH_DIR,
            SWC_PATHS,
            outdir
        )

        for layer, swc_src in picked.items():
            new_name = LAYER_TO_NEWNAME[layer]
            dst = os.path.join(outdir, new_name)
            shutil.copy2(swc_src, dst)

        print(f"[OK] {os.path.basename(outdir)}  SWCs:",
              {k: os.path.basename(v) for k, v in picked.items()})


if __name__ == "__main__":
    main(seed=0)
