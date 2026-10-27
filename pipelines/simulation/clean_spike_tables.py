# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ALLEN_ROOT = Path(os.environ.get("ALLEN_ROOT", PROJECT_ROOT / "ALLEN")).resolve()

BASE_DIR = os.environ.get(
    "TRIALS_ROOT",
    str(ALLEN_ROOT / "file" / "final_trial")
)

TARGET_FILES = [
    "microns_patch_synapse_spikes_ER_ms.csv",
    "microns_patch_synapse_spikes_ER_ms_EImatched.csv",
    "microns_patch_synapse_spikes_real_ms.csv",
]


def list_er_trials(root):
    return sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and re.match(r"^ER_trial\d+$", d)
    )


def nan_to_empty_inplace(csv_path):
    df = pd.read_csv(csv_path, header=None, dtype=str)

    # 将pandas NaN转换为空字符串。 / pandas NaN -> ""
    df = df.fillna("")

    df = df.replace(
        to_replace=r"(?i)^nan$",
        value="",
        regex=True
    )

    df.to_csv(csv_path, index=False, header=False)
    print(f"[OK] cleaned NaN -> empty: {csv_path}")


if __name__ == "__main__":
    trials = list_er_trials(BASE_DIR)
    if not trials:
        raise RuntimeError(f"No ER_trial found under {BASE_DIR}")

    for tdir in trials:
        marker = os.path.join(tdir, ".nan_cleaned.ok")
        existing_targets = [
            os.path.join(tdir, fname) for fname in TARGET_FILES
            if os.path.isfile(os.path.join(tdir, fname))
        ]
        if (
            os.path.isfile(marker)
            and existing_targets
            and all(os.path.getmtime(marker) >= os.path.getmtime(p) for p in existing_targets)
        ):
            print(f"[SKIP] already cleaned: {tdir}")
            continue
        for fname in TARGET_FILES:
            fpath = os.path.join(tdir, fname)
            if os.path.isfile(fpath):
                nan_to_empty_inplace(fpath)
            else:
                print(f"[SKIP] not found: {fpath}")
        with open(marker, "w", encoding="utf-8") as f:
            f.write("NaN fields normalized to empty strings.\n")

    print("[COMPLETE] NaN fields normalized to empty strings.")
