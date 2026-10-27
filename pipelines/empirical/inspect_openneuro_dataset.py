# -*- coding: utf-8 -*-
"""分析前建立本地OpenNeuro ds003380 BIDS数据集清单。 / Inventory the local OpenNeuro ds003380 BIDS dataset before analysis."""

from pathlib import Path
import os
import tarfile
import pandas as pd
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ROOT = Path(os.environ.get("PIG_DATA_ROOT", REPOSITORY_ROOT / "data" / "external" / "ds003380")).resolve()
OUT_DIR = Path(os.environ.get(
    "PIG_RESULTS_ROOT",
    PROJECT_ROOT / "ALLEN" / "results" / "empirical_propofol",
)).resolve() / "inspection"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------
# ------------------------------------------------------------
def safe_extract_tar(tar_path: Path, extract_dir: Path):
    def is_within_directory(directory, target):
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = extract_dir / member.name
            if not is_within_directory(extract_dir, member_path):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(extract_dir)


print("=" * 120)
print("[STEP 0] Extract tar.gz files")
print("=" * 120)

tar_files = sorted(ROOT.glob("sub-*/eeg/*_edf.tar.gz"))
print(f"[FOUND] tar.gz files = {len(tar_files)}")

for tar_path in tar_files:
    eeg_dir = tar_path.parent
    existing_edfs = list(eeg_dir.glob("*.edf"))
    if existing_edfs:
        print(f"[SKIP] already has EDF: {tar_path}")
        continue

    print(f"[EXTRACT] {tar_path}")
    safe_extract_tar(tar_path, eeg_dir)

print("[DONE] extraction")


# ------------------------------------------------------------
# 2. participants.tsv参与者表。 / 2. participants.tsv
# ------------------------------------------------------------
print("\n" + "=" * 120)
print("[STEP 1] participants.tsv")
print("=" * 120)

participants_path = ROOT / "participants.tsv"
if participants_path.exists():
    participants = pd.read_csv(participants_path, sep="\t")
    print("[participants.tsv]")
    print(participants.head(20))
    print("\n[COLUMNS]")
    print(list(participants.columns))
    participants.to_csv(OUT_DIR / "00_participants.csv", index=False, encoding="utf-8-sig")
else:
    participants = pd.DataFrame()
    print("[WARN] participants.tsv not found")


# ------------------------------------------------------------
# ------------------------------------------------------------
print("\n" + "=" * 120)
print("[STEP 2] channels.tsv summary")
print("=" * 120)

all_channels = []

for ch_path in sorted(ROOT.glob("sub-*/eeg/channels.tsv")):
    sub = ch_path.parts[-3]
    try:
        ch = pd.read_csv(ch_path, sep="\t")
        ch.insert(0, "subject", sub)
        ch.insert(1, "channels_tsv", str(ch_path))
        all_channels.append(ch)
    except Exception as e:
        print(f"[WARN] failed to read {ch_path}: {e}")

if all_channels:
    all_channels = pd.concat(all_channels, ignore_index=True)
    all_channels.to_csv(OUT_DIR / "00_channels_all_subjects.csv", index=False, encoding="utf-8-sig")

    print("[CHANNEL COLUMNS]")
    print(list(all_channels.columns))

    print("\n[CHANNEL HEAD]")
    print(all_channels.head(40).to_string(index=False))

    for col in ["name", "type", "units", "status", "description", "sampling_frequency"]:
        if col in all_channels.columns:
            print(f"\n[VALUE COUNTS] {col}")
            print(all_channels[col].value_counts(dropna=False).head(50))
else:
    all_channels = pd.DataFrame()
    print("[WARN] no channels.tsv found")


# ------------------------------------------------------------
# 文件处理细节。 / File-handling details.
# ------------------------------------------------------------
print("\n" + "=" * 120)
print("[STEP 3] EDF inventory")
print("=" * 120)

try:
    import mne
except Exception as e:
    raise RuntimeError(
    ) from e

edf_files = sorted(ROOT.glob("sub-*/eeg/*.edf"))
print(f"[FOUND] EDF files = {len(edf_files)}")

edf_rows = []

for edf_path in edf_files:
    sub = edf_path.parts[-3]
    fname = edf_path.name

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        nchan = len(raw.ch_names)
        n_times = raw.n_times
        duration_sec = n_times / sfreq

        edf_rows.append({
            "subject": sub,
            "edf_file": str(edf_path),
            "edf_name": fname,
            "sfreq": sfreq,
            "n_channels": nchan,
            "n_times": n_times,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "channel_names": "|".join(raw.ch_names),
        })

        print(f"[EDF] {sub} | {fname} | sfreq={sfreq:.1f} Hz | "
              f"nchan={nchan} | duration={duration_sec/60:.2f} min")
        print("      channels:", raw.ch_names)

    except Exception as e:
        print(f"[WARN] failed to read EDF: {edf_path}")
        print(f"       {type(e).__name__}: {e}")

edf_info = pd.DataFrame(edf_rows)
edf_info.to_csv(OUT_DIR / "00_edf_inventory.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("[SAVED]")
print("=" * 120)
print(OUT_DIR / "00_participants.csv")
print(OUT_DIR / "00_channels_all_subjects.csv")
print(OUT_DIR / "00_edf_inventory.csv")

print("\n[NEXT]")
print("1. Review 00_edf_inventory.csv for readable files and recording duration.")
print("2. Review 00_channels_all_subjects.csv for channel labels and status.")
print("3. Confirm that all intended subjects and states are represented.")
print("4. Then run extract_propofol_features.py.")
