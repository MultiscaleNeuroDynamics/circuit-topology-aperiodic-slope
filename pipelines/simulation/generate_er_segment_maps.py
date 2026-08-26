# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ALLEN_ROOT = Path(os.environ.get("ALLEN_ROOT", PROJECT_ROOT / "ALLEN")).resolve()

# =========================
# =========================
FINAL_TRIAL_DIR = os.environ.get(
    "TRIALS_ROOT",
    str(ALLEN_ROOT / "file" / "final_trial")
)

SYN_EE_PATH = os.path.join(
    os.environ.get(
        "PATCH_DIR",
        str(ALLEN_ROOT / "file" / "patch")
    ),
    "syn_ee.csv"
)

ER_SPIKE_NAME = "microns_patch_synapse_spikes_ER_ms.csv"

MAP_NAMES = {
    "L2/3": "syn_to_segment_L2_3.csv",
    "L4":   "syn_to_segment_L4.csv",
    "L5":   "syn_to_segment_L5.csv",
    "L6":   "syn_to_segment_L6.csv",
}

OUT_NAMES = {
    "L2/3": "syn_to_segment_L2_3_ER.csv",
    "L4":   "syn_to_segment_L4_ER.csv",
    "L5":   "syn_to_segment_L5_ER.csv",
    "L6":   "syn_to_segment_L6_ER.csv",
}

LAYER_ORDER = ["L2/3", "L4", "L5", "L6"]


# =========================
# =========================
def list_trial_dirs(root_dir: str, pattern=r"^ER_trial\d+$"):
    ds = []
    if not os.path.isdir(root_dir):
        return ds
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and re.match(pattern, name):
            ds.append(p)
    ds.sort()
    return ds

def read_csv_skip_first_line(path: str) -> pd.DataFrame:
    # Accept either headerless maps or maps with a textual first row.
    df = pd.read_csv(path, header=None, dtype=str).fillna("")
    if df.shape[1] >= 2:
        c0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        c1 = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df = df.loc[c0.notna() & c1.notna()].copy()
    return df
def read_csv_no_first_line(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.fillna("")
    return df
def load_ee_set(syn_ee_path: str) -> set:
    df = pd.read_csv(syn_ee_path, header=None, dtype=str).fillna("")
    ee = set(df.iloc[:, 0].astype(str).str.strip().tolist())
    ee.discard("")
    return ee

def spike_signature(row_vals) -> tuple:
    return tuple("" if (x is None or str(x).lower() == "nan") else str(x) for x in row_vals)

def group_er_synapses(er_spike_path: str):
    df = read_csv_no_first_line(er_spike_path)
    if df.shape[1] < 1:
        return [], []

    syn_ids = df.iloc[:, 0].astype(str).str.strip().tolist()
    spikes = df.iloc[:, 1:].values

    gE = defaultdict(list)
    gI = defaultdict(list)

    for sid, sp in zip(syn_ids, spikes):
        sid = str(sid).strip()
        sig = spike_signature(sp)

        if sid.startswith("10"):
            gE[sig].append(sid)
        elif sid.startswith("00"):
            gI[sig].append(sid)

    groups_E = list(gE.values())
    groups_I = list(gI.values())
    groups_E.sort(key=len, reverse=True)
    groups_I.sort(key=len, reverse=True)
    return groups_E, groups_I

def load_segment_capacities(map_paths: dict, ee_set: set):
    cap_E = {ly: defaultdict(int) for ly in map_paths.keys()}
    cap_I = {ly: defaultdict(int) for ly in map_paths.keys()}
    total_rows = {}

    for layer, p in map_paths.items():
        m = read_csv_skip_first_line(p)
        if m.shape[1] < 2:
            total_rows[layer] = 0
            continue

        total_rows[layer] = len(m)
        syn_col = m.iloc[:, 0].astype(str).str.strip()
        seg_col = m.iloc[:, 1].astype(int, errors="ignore")

        seg_list = []
        for v in seg_col.tolist():
            try:
                seg_list.append(int(v))
            except Exception:
                seg_list.append(None)

        for sid, seg in zip(syn_col.tolist(), seg_list):
            if seg is None:
                continue
            if sid in ee_set:
                cap_E[layer][seg] += 1
            else:
                cap_I[layer][seg] += 1

    return cap_E, cap_I, total_rows

def layer_remaining(cap_layer: dict) -> int:
    return int(sum(int(v) for v in cap_layer.values()))

def choose_best_single_layer(need: int, cap_by_layer: dict, layer_order: list):
    best = None
    best_rem = -1
    for ly in layer_order:
        rem = layer_remaining(cap_by_layer[ly])
        if rem >= need and rem > best_rem:
            best = ly
            best_rem = rem
    return best

def allocate_to_layer_segments(syn_ids: list, layer: str, cap_layer: dict, out_assign: dict):
    if layer not in out_assign:
        out_assign[layer] = []

    segs_sorted = sorted(cap_layer.keys(), key=lambda s: cap_layer[s], reverse=True)

    idx = 0
    n = len(syn_ids)
    for seg in segs_sorted:
        if idx >= n:
            break
        cap = int(cap_layer[seg])
        if cap <= 0:
            continue
        take = min(cap, n - idx)
        for k in range(take):
            out_assign[layer].append((syn_ids[idx + k], int(seg)))
        cap_layer[seg] -= take
        idx += take

    if idx < n:
        return syn_ids[idx:]
    return []

def allocate_groups_soft(groups: list, cap_by_layer: dict, layer_order: list):
    out_assign = {ly: [] for ly in layer_order}
    unallocated = []

    for g in groups:
        if not g:
            continue
        need = len(g)

        single = choose_best_single_layer(need, cap_by_layer, layer_order)
        if single is not None:
            leftover = allocate_to_layer_segments(g, single, cap_by_layer[single], out_assign)
            if leftover:
                g = leftover
            else:
                continue

        remaining_ids = g
        layers_sorted = sorted(layer_order, key=lambda ly: layer_remaining(cap_by_layer[ly]), reverse=True)
        for ly in layers_sorted:
            if not remaining_ids:
                break
            if layer_remaining(cap_by_layer[ly]) <= 0:
                continue
            remaining_ids = allocate_to_layer_segments(remaining_ids, ly, cap_by_layer[ly], out_assign)

        if remaining_ids:
            unallocated.extend(remaining_ids)

    return out_assign, unallocated

def write_mapping(out_path: str, pairs: list):
    df = pd.DataFrame(pairs, columns=["synapse_id", "segment_id"])
    df.to_csv(out_path, index=False, header=False, encoding="utf-8")


# =========================
# Main execution.
# =========================
if __name__ == "__main__":
    if not os.path.isfile(SYN_EE_PATH):
        raise FileNotFoundError(f"syn_ee.csv not found: {SYN_EE_PATH}")

    ee_set = load_ee_set(SYN_EE_PATH)
    print(f"[INFO] EE_SET size = {len(ee_set)}")

    trial_dirs = list_trial_dirs(FINAL_TRIAL_DIR)
    if not trial_dirs:
        raise RuntimeError(f"No ER_trialXXXX dirs found in: {FINAL_TRIAL_DIR}")

    for tdir in trial_dirs:
        tname = os.path.basename(tdir.rstrip("/"))
        print(f"\n===== {tname} =====")

        expected_er_maps = [os.path.join(tdir, OUT_NAMES[ly]) for ly in LAYER_ORDER]
        if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in expected_er_maps):
            print(f"[SKIP] {tname}: all four ER mapping tables already complete")
            continue

        er_spike_path = os.path.join(tdir, ER_SPIKE_NAME)
        if not os.path.isfile(er_spike_path):
            print(f"[SKIP] missing ER spike file: {er_spike_path}")
            continue

        # mapping paths
        map_paths = {ly: os.path.join(tdir, MAP_NAMES[ly]) for ly in LAYER_ORDER}
        missing_maps = [ly for ly, p in map_paths.items() if not os.path.isfile(p)]
        if missing_maps:
            print(f"[SKIP] missing mapping files in {tname}: {missing_maps}")
            continue

        # 1) capacities
        try:
            capE, capI, total_rows = load_segment_capacities(map_paths, ee_set)
        except Exception as e:
            print(f"[SKIP] load capacities failed: {e}")
            continue

        # 2) group ER synapses
        try:
            groupsE, groupsI = group_er_synapses(er_spike_path)
        except Exception as e:
            print(f"[SKIP] group ER synapses failed: {e}")
            continue

        total_erE = sum(len(g) for g in groupsE)
        total_erI = sum(len(g) for g in groupsI)
        total_capE = sum(layer_remaining(capE[ly]) for ly in LAYER_ORDER)
        total_capI = sum(layer_remaining(capI[ly]) for ly in LAYER_ORDER)

        print(f"[INFO] ER totals: E={total_erE}, I={total_erI}")
        print(f"[INFO] CAP totals: E={total_capE}, I={total_capI}")
        for ly in LAYER_ORDER:
            print(f"  - {ly}: rows={total_rows.get(ly,0)}, capE={layer_remaining(capE[ly])}, capI={layer_remaining(capI[ly])}")

        # 3) allocate (soft)
        assignE, unallocE = allocate_groups_soft(groupsE, capE, LAYER_ORDER)
        assignI, unallocI = allocate_groups_soft(groupsI, capI, LAYER_ORDER)

        if unallocE or unallocI:
            print(f"[WARN] unallocated: E={len(unallocE)}, I={len(unallocI)} (details)")

        for ly in LAYER_ORDER:
            pairs = []
            pairs.extend(assignE.get(ly, []))
            pairs.extend(assignI.get(ly, []))

            outp = os.path.join(tdir, OUT_NAMES[ly])
            try:
                write_mapping(outp, pairs)
                print(f"[SAVE] {ly}: {len(pairs)} rows -> {outp}")
            except Exception as e:
                print(f"[WARN] save failed ({ly}): {e}")

    print("\n[COMPLETE] generated ER mapping tables for all trials (soft mode).")
