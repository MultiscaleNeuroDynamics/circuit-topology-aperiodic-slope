# -*- coding: utf-8 -*-
import os
import re
import csv
from pathlib import Path
import pandas as pd
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ALLEN_ROOT = Path(os.environ.get("ALLEN_ROOT", PROJECT_ROOT / "ALLEN")).resolve()

# =========================================================
# =========================================================
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)

def list_trial_dirs(base_dir):
    ds = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and re.match(r"ER_trial\d+", name):
            ds.append(p)
    ds.sort()
    return ds

# =========================================================
# 1) Real: soma spike -> synapse spike
# =========================================================
def _load_patch_synapses(syn_ee_path: str, syn_ie_path: str) -> pd.DataFrame:
    if not os.path.exists(syn_ee_path):
        raise FileNotFoundError(f"Missing syn_ee.csv: {syn_ee_path}")
    if not os.path.exists(syn_ie_path):
        raise FileNotFoundError(f"Missing syn_ie.csv: {syn_ie_path}")

    ee = pd.read_csv(syn_ee_path)
    ie = pd.read_csv(syn_ie_path)

    need_cols = {"id", "pre_pt_root_id"}
    for tag, df in [("syn_ee", ee), ("syn_ie", ie)]:
        if not need_cols.issubset(df.columns):
            raise ValueError(f"{tag} requires columns {need_cols}; observed {list(df.columns)}")

    syn_df = pd.concat([ee, ie], axis=0, ignore_index=True)

    syn_df = syn_df.drop_duplicates(subset=["id"], keep="first").copy()

    syn_df["id"] = syn_df["id"].astype(np.int64)
    syn_df["pre_pt_root_id"] = syn_df["pre_pt_root_id"].astype(np.int64)

    return syn_df


def build_real_synapse_spike_table_from_patch(
    soma_spike_path: str,
    syn_ee_path: str,
    syn_ie_path: str,
    out_syn_spike_path: str,
    unit: str = "ms"
):
    # Load input data.
    soma_df = pd.read_csv(soma_spike_path, header=None, dtype=str)
    neuron_ids = soma_df.iloc[:, 0].astype(np.int64).to_numpy()

    times_raw = soma_df.iloc[:, 1:].to_numpy(dtype=str)
    times_raw = np.where(pd.isna(times_raw), "", times_raw)

    id_to_times = {int(nid): list(times_raw[i]) for i, nid in enumerate(neuron_ids)}

    print(f"[REAL] soma rows: {len(neuron_ids)} | spike columns: {times_raw.shape[1] if times_raw.size else 0}")

    # Load input data.
    syn_df = _load_patch_synapses(syn_ee_path, syn_ie_path)
    print(f"[REAL] unique patch synapses (EE+IE): {len(syn_df)}")

    # Synapse-processing details.
    mask_pre_in_soma = syn_df["pre_pt_root_id"].isin(neuron_ids)
    syn_sub = syn_df.loc[mask_pre_in_soma, ["id", "pre_pt_root_id"]].copy()

    print(f"[REAL] synapses with a presynaptic soma row: {len(syn_sub)}")

    # Synapse-processing details.
    rows = []
    for row in syn_sub.itertuples(index=False):
        syn_id = int(row.id)
        pre_id = int(row.pre_pt_root_id)
        times = id_to_times.get(pre_id, [])
        rows.append([syn_id] + list(times))

    if not rows:
        raise RuntimeError("No REAL synapses could be matched to presynaptic spike rows.")

    max_len = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_len:
            r.extend([""] * (max_len - len(r)))

    syn_spike_df = pd.DataFrame(rows).fillna("")
    ensure_dir(out_syn_spike_path)
    syn_spike_df.to_csv(out_syn_spike_path, index=False, header=False)
    print(f"[REAL] saved {out_syn_spike_path} | rows={len(syn_spike_df)} | columns={syn_spike_df.shape[1]}")

# =========================================================
# 2) ER: soma spike + (EE/IE edges weight) -> synapse spike
# =========================================================
def make_syn_id(prefix2: str, rng: np.random.Generator, used_suffix: set):
    while True:
        suffix = int(rng.integers(0, 100_000_000))
        if suffix not in used_suffix:
            used_suffix.add(suffix)
            return f"{prefix2}{suffix:08d}"

def load_edge_df(path, tag):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {tag} edge table: {path}")
    df = pd.read_csv(path)
    need = {"pre_id", "post_id", "weight"}
    if not need.issubset(df.columns):
        raise ValueError(f"{tag} requires columns {need}; observed {list(df.columns)}")
    df = df.copy()
    df["pre_id"]  = df["pre_id"].astype(np.int64)
    df["post_id"] = df["post_id"].astype(np.int64)
    df["weight"]  = df["weight"].astype(np.int64)
    df = df[df["weight"] > 0]
    return df

def build_er_synapse_spike_table(
    soma_spike_path,
    ee_edge_path,
    ie_edge_path,
    out_syn_spike_path,
    rng_seed=20251213
):
    print("[ER] loading soma spike table")
    soma_df = pd.read_csv(soma_spike_path, header=None, dtype=str)
    if soma_df.shape[1] < 2:
        raise RuntimeError("ER soma spike table must contain an ID column and at least one spike column.")

    neuron_ids = soma_df.iloc[:, 0].astype(np.int64).to_numpy()
    times_raw = soma_df.iloc[:, 1:].to_numpy(dtype=str)
    times_raw = np.where(pd.isna(times_raw), "", times_raw)

    id_to_times = {int(nid): list(times_raw[i]) for i, nid in enumerate(neuron_ids)}
    print(f"[ER] soma rows: {len(neuron_ids)} | spike columns: {times_raw.shape[1]}")

    print("[ER] loading weighted edge tables")
    ee_df = load_edge_df(ee_edge_path, "EE(E->E)")
    ie_df = load_edge_df(ie_edge_path, "IE(I->E)")
    print(f"[ER] EE edges: {len(ee_df)} | synapse multiplicity: {int(ee_df['weight'].sum())}")
    print(f"[ER] IE edges: {len(ie_df)} | synapse multiplicity: {int(ie_df['weight'].sum())}")

    ensure_dir(out_syn_spike_path)

    rng = np.random.default_rng(rng_seed)
    used_suffix = set()

    n_spike_cols = times_raw.shape[1]
    n_out_cols = 1 + n_spike_cols

    print("[ER] expanding weighted edges to synapse-level rows")
    total_rows = [0]

    with open(out_syn_spike_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        def write_from_edges(edge_df, prefix2, tag):
            miss_pre = 0
            zero_spike_pre = 0

            for row in edge_df.itertuples(index=False):
                pre_id = int(row.pre_id)
                w = int(row.weight)

                times = id_to_times.get(pre_id, None)
                if times is None:
                    miss_pre += 1
                    continue

                if all((t == "" or str(t).lower() == "nan") for t in times):
                    zero_spike_pre += 1

                for _ in range(w):
                    syn_id = make_syn_id(prefix2, rng, used_suffix)
                    out_row = [syn_id] + times

                    if len(out_row) != n_out_cols:
                        if len(out_row) < n_out_cols:
                            out_row.extend([""] * (n_out_cols - len(out_row)))
                        else:
                            out_row = out_row[:n_out_cols]

                    writer.writerow(out_row)
                    total_rows[0] += 1

            print(f"[ER][{tag}] missing presynaptic rows={miss_pre}, zero-spike rows={zero_spike_pre}")

        write_from_edges(ee_df, prefix2="10", tag="EE(E-pre)")
        write_from_edges(ie_df, prefix2="00", tag="IE(I-pre)")

    print(f"[ER] saved: {out_syn_spike_path}")
    print(f"[ER] output synapse rows: {total_rows[0]}")
    print(f"[ER] output columns: {n_out_cols} (one synapse ID plus {n_spike_cols} spike columns)")

# =========================================================
# File-handling details.
# =========================================================
if __name__ == "__main__":
    base = os.environ.get(
        "TRIALS_ROOT",
        str(ALLEN_ROOT / "file" / "final_trial")
    )

    patch_dir = os.environ.get(
        "PATCH_DIR",
        str(ALLEN_ROOT / "file" / "patch")
    )
    syn_ee_path = os.path.join(patch_dir, "syn_ee.csv")
    syn_ie_path = os.path.join(patch_dir, "syn_ie.csv")

    trial_dirs = list_trial_dirs(base)
    if len(trial_dirs) == 0:
        raise RuntimeError(f"No ER_trial directories found under: {base}")

    print("Trial directories found:", len(trial_dirs))
    for d in trial_dirs:
        print("  -", os.path.basename(d))

    global_real_soma = os.path.join(base, "microns_patch_spikes_real_ms.csv")

    for i, tdir in enumerate(trial_dirs):
        print("\n" + "="*80)
        print(f"[Trial {i+1}/{len(trial_dirs)}] {os.path.basename(tdir)}")

        real_soma_path = os.path.join(tdir, "microns_patch_spikes_real_ms.csv")
        if not os.path.exists(real_soma_path):
            real_soma_path = global_real_soma  # fallback

        er_soma_path = os.path.join(tdir, "microns_patch_spikes_ER_ms.csv")
        ee_edge_path = os.path.join(tdir, "microns_patch_edges_ER_weights.csv")
        ie_edge_path = os.path.join(tdir, "microns_patch_edges_ER_IE_weights.csv")

        out_real_synspk = os.path.join(tdir, "microns_patch_synapse_spikes_real_ms.csv")
        out_er_synspk   = os.path.join(tdir, "microns_patch_synapse_spikes_ER_ms.csv")

        if os.path.isfile(out_real_synspk) and os.path.getsize(out_real_synspk) > 0:
            print(f"[SKIP Real] exists: {out_real_synspk}")
        else:
            build_real_synapse_spike_table_from_patch(
                soma_spike_path=real_soma_path,
                syn_ee_path=syn_ee_path,
                syn_ie_path=syn_ie_path,
                out_syn_spike_path=out_real_synspk,
                unit="ms"
            )

        if os.path.isfile(out_er_synspk) and os.path.getsize(out_er_synspk) > 0:
            print(f"[SKIP ER] exists: {out_er_synspk}")
        else:
            build_er_synapse_spike_table(
                soma_spike_path=er_soma_path,
                ee_edge_path=ee_edge_path,
                ie_edge_path=ie_edge_path,
                out_syn_spike_path=out_er_synspk,
                rng_seed=20251213 + i
            )

    print("[COMPLETE] synapse-level spike tables generated for all trials.")
