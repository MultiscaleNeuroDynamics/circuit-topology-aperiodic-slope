# -*- coding: utf-8 -*-
"""计算REAL和ER配对试次的实际突触兴奋/抑制比。REAL突触依据MICrONS突触前细胞类型分类；ER突触依据编码中的兴奋/抑制标识符前缀分类。输出每个配对试次中各拓扑的事件计数比。 / Compute realised synaptic E/I ratios for paired REAL and ER trials.

REAL synapses are classified from their presynaptic MICrONS cell type. ER
synapses use the encoded excitatory/inhibitory identifier prefix. The output
contains topology-specific event-count ratios for every paired trial.
"""

import os
import re
import time
import threading
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

# -------------------------
# 配置。 / Configuration.
# -------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ALLEN_ROOT = PROJECT_ROOT / "ALLEN"
BASE_DIR = str(Path(os.environ.get("TRIALS_ROOT", ALLEN_ROOT / "file" / "final_trial")).resolve())

# REAL突触标识符到突触前根标识符的映射。 / REAL synapse identifier to presynaptic root identifier.
PROOFREAD_SYNAPSE_TABLE_PATH = str(ALLEN_ROOT / "file" / "proofread_cells_synapses.csv")
# 突触前根标识符到细胞类型的映射。 / Presynaptic root identifier to cell type.
PROOFREAD_CELLTYPE_PATH = str(ALLEN_ROOT / "file" / "proofread_cells_celltypes_all.csv")

REAL_CSV_NAME = "microns_patch_synapse_spikes_real_ms.csv"
ER_CSV_NAME   = "microns_patch_synapse_spikes_ER_ms.csv"

RESULTS_TAG = os.environ.get("RESULTS_TAG", "").strip("/\\")
RESULTS_BASE = ALLEN_ROOT / "results"
if RESULTS_TAG:
    RESULTS_BASE = RESULTS_BASE / RESULTS_TAG
OUT_DIR = str(RESULTS_BASE / "ei_ratio_stats")
os.makedirs(OUT_DIR, exist_ok=True)

PRINT_MEM_EVERY_SEC = 5


# -------------------------
# 加载输入数据。 / Load input data.
# -------------------------
def _read_total_mem_bytes_linux():
    # 读取系统总内存，单位为kB。 / /proc/meminfo: MemTotal:  xxxx kB
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        return None
    return None


def _read_proc_rss_bytes_linux(pid=None):
    # 读取指定进程的常驻内存，单位为kB。 / /proc/<pid>/status: VmRSS: xxxx kB
    if pid is None:
        pid = os.getpid()
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        return None
    return None


def get_mem_usage_ratio():
    try:
        import psutil  # noqa
        p = psutil.Process(os.getpid())
        rss = int(p.memory_info().rss)
        total = int(psutil.virtual_memory().total)
        ratio = rss / total if total > 0 else None
        return rss, total, ratio
    except Exception:
        pass

    total = _read_total_mem_bytes_linux()
    rss = _read_proc_rss_bytes_linux()
    if total is None or rss is None or total <= 0:
        return None, None, None
    return rss, total, rss / total


def mem_monitor(stop_event, every_sec=5):
    while not stop_event.is_set():
        rss, total, ratio = get_mem_usage_ratio()
        if ratio is None:
            print("[MEM] (unavailable)")
        else:
            rss_mb = rss / (1024 ** 2)
            total_mb = total / (1024 ** 2)
            print(f"[MEM] RSS={rss_mb:.1f} MB / Total={total_mb:.0f} MB  ({ratio*100:.2f}%)")
        stop_event.wait(every_sec)


# -------------------------
# -------------------------
def list_trials(root_dir, pattern=r"^ER_trial\d+$"):
    ds = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and re.match(pattern, name):
            ds.append(p)
    ds.sort()
    return ds


def load_synapse_to_pre_root_map(path):
    df = pd.read_csv(path, dtype=str)
    if df.shape[0] == 0:
        return {}

    if "synapse_id" in df.columns:
        syn_col = "synapse_id"
    else:
        syn_col = df.columns[0]

    if "pre_pt_root_id" not in df.columns:
        raise ValueError(f"Synapse table is missing pre_pt_root_id: {path}")

    sid = df[syn_col].astype(str).str.strip()
    pre = df["pre_pt_root_id"].astype(str).str.strip()

    m = (sid != "") & (pre != "")
    sid = sid[m]
    pre = pre[m]

    return dict(zip(sid.tolist(), pre.tolist()))


def load_root_to_celltype_map(path):
    df = pd.read_csv(path, header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError(f"Cell-type table must contain at least two columns: {path}")
    rid = df.iloc[:, 0].astype(str).str.strip()
    ctp = df.iloc[:, 1].astype(str).str.strip()
    m = (rid != "") & (ctp != "")
    rid = rid[m]
    ctp = ctp[m]
    return dict(zip(rid.tolist(), ctp.tolist()))


def count_spikes_from_table(csv_path):
    df = pd.read_csv(csv_path, header=None, dtype=str)
    if df.shape[1] < 2:
        ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
        return ids, np.zeros(ids.shape[0], dtype=int)

    ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    sp = df.iloc[:, 1:].replace(r"^\s*$", np.nan, regex=True)
    spike_counts = sp.notna().sum(axis=1).astype(int).to_numpy()
    return ids, spike_counts


def compute_ei_ratio_from_counts(e_spikes, i_spikes):
    e_spikes = int(e_spikes)
    i_spikes = int(i_spikes)
    if i_spikes == 0:
        return np.inf if e_spikes > 0 else np.nan
    return float(e_spikes) / float(i_spikes)


def paired_cohen_dz(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan
    d = x - y
    sd = d.std(ddof=1)
    return np.nan if sd == 0 else float(d.mean() / sd)


# -------------------------
# -------------------------
_SYN2PRE = None      # 突触标识符到突触前根标识符的映射。 / synapse_id -> pre_pt_root_id
_ROOT2TYPE = None    # 根标识符到细胞类型的映射。 / root_id -> celltype


def init_worker(synapse_table_path, celltype_path):
    global _SYN2PRE, _ROOT2TYPE
    _SYN2PRE = load_synapse_to_pre_root_map(synapse_table_path)
    _ROOT2TYPE = load_root_to_celltype_map(celltype_path)


# -------------------------
# -------------------------
def process_one_trial(trial_dir):
    global _SYN2PRE, _ROOT2TYPE
    if _SYN2PRE is None or _ROOT2TYPE is None:
        raise RuntimeError("Worker maps not initialized.")

    trial_name = os.path.basename(trial_dir.rstrip("/"))

    real_path = os.path.join(trial_dir, REAL_CSV_NAME)
    er_path   = os.path.join(trial_dir, ER_CSV_NAME)

    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"[{trial_name}] missing: {real_path}")
    if not os.path.isfile(er_path):
        raise FileNotFoundError(f"[{trial_name}] missing: {er_path}")

    # REAL真实拓扑。 / -------- REAL --------
    real_ids, real_counts = count_spikes_from_table(real_path)

    real_E = 0
    real_I = 0
    real_unknown_syn = 0

    for sid, cnt in zip(real_ids, real_counts):
        s = str(sid).strip()

        pre_root = _SYN2PRE.get(s, None)
        if pre_root is None:
            real_unknown_syn += 1
            continue

        ctp = _ROOT2TYPE.get(str(pre_root).strip(), None)
        if ctp is None:
            real_unknown_syn += 1
            continue

        if ctp == "excitatory_neuron":
            real_E += int(cnt)
        elif ctp == "inhibitory_neuron":
            real_I += int(cnt)
        else:
            real_unknown_syn += 1

    real_ei = compute_ei_ratio_from_counts(real_E, real_I)

    # ER随机拓扑。 / -------- ER --------
    er_ids, er_counts = count_spikes_from_table(er_path)

    er_E = 0
    er_I = 0
    er_unknown_syn = 0
    for sid, cnt in zip(er_ids, er_counts):
        s = str(sid).strip()
        prefix = s[:2] if len(s) >= 2 else ""
        if prefix == "10":
            er_E += int(cnt)
        elif prefix == "00":
            er_I += int(cnt)
        else:
            er_unknown_syn += 1

    er_ei = compute_ei_ratio_from_counts(er_E, er_I)

    return {
        "trial": trial_name,

        "REAL_E_spikes": real_E,
        "REAL_I_spikes": real_I,
        "REAL_EI_ratio": real_ei,
        "REAL_unknown_synapses": real_unknown_syn,

        "ER_E_spikes": er_E,
        "ER_I_spikes": er_I,
        "ER_EI_ratio": er_ei,
        "ER_unknown_synapses": er_unknown_syn,
    }


# -------------------------
# 执行主程序。 / Main execution.
# -------------------------
if __name__ == "__main__":
    mp.freeze_support()

    trial_dirs = list_trials(BASE_DIR, pattern=r"^ER_trial\d+$")
    trial_dirs=trial_dirs[:]
    if not trial_dirs:
        raise RuntimeError(f"No ER_trial directories found under {BASE_DIR}")

    # 内存管理细节。 / Memory-management details.
    stop_evt = threading.Event()
    th = threading.Thread(target=mem_monitor, args=(stop_evt, PRINT_MEM_EVERY_SEC), daemon=True)
    th.start()

    # 并行执行。 / Parallel execution.
    max_workers = int(os.environ.get("MAX_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
    print(f"[Parallel] max_workers = {max_workers}")
    print(f"[Trials] N = {len(trial_dirs)}")

    rows = []
    errors = []

    t0 = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(PROOFREAD_SYNAPSE_TABLE_PATH, PROOFREAD_CELLTYPE_PATH)
        ) as ex:
            futs = {ex.submit(process_one_trial, td): td for td in trial_dirs}

            for fut in as_completed(futs):
                td = futs[fut]
                name = os.path.basename(td)
                try:
                    row = fut.result()
                    rows.append(row)
                    print(f"[Done] {name} | REAL_EI={row['REAL_EI_ratio']:.4g} | ER_EI={row['ER_EI_ratio']:.4g}")
                except Exception as e:
                    msg = f"[ERROR] {name}: {e}"
                    errors.append(msg)
                    print(msg)

        if errors:
            raise RuntimeError("One or more trial-level E/I calculations failed:\n" + "\n".join(errors))

    finally:
        # 内存管理细节。 / Memory-management details.
        stop_evt.set()
        th.join(timeout=1.0)

    t1 = time.time()
    print(f"[Time] total = {t1 - t0:.2f} s")

    # 输出处理。 / Output handling.
    df = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    out_csv = os.path.join(OUT_DIR, "EI_ratio_per_trial_REAL_vs_ER.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")

    # -------------------------
    # -------------------------
    x = df["REAL_EI_ratio"].to_numpy(dtype=float)
    y = df["ER_EI_ratio"].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m]
    y2 = y[m]

    report_path = os.path.join(OUT_DIR, "paired_ttest_EI_ratio_REAL_vs_ER.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Paired comparison of EI ratio (REAL vs ER) ===\n\n")
        f.write(f"Base dir: {BASE_DIR}\n")
        f.write(f"Synapse table (REAL): {PROOFREAD_SYNAPSE_TABLE_PATH}\n")
        f.write(f"Celltype map  (REAL): {PROOFREAD_CELLTYPE_PATH}\n")
        f.write("REAL label rule: synapse_id -> pre_pt_root_id -> celltype\n")
        f.write("ER label rule: synapse_id prefix '10'->E, '00'->I\n\n")
        f.write(f"All trials found: {len(df)}\n")
        f.write(f"Trials used (finite EI ratios): {int(m.sum())}\n\n")

        if x2.size < 2:
            f.write("Not enough valid paired samples for t-test.\n")
        else:
            t, p = stats.ttest_rel(x2, y2, nan_policy="omit")
            dz = paired_cohen_dz(x2, y2)

            f.write(f"REAL EI ratio: mean={np.mean(x2):.6g}, sd={np.std(x2, ddof=1):.6g}\n")
            f.write(f"ER   EI ratio: mean={np.mean(y2):.6g}, sd={np.std(y2, ddof=1):.6g}\n\n")
            f.write(f"Paired t-test: t={t:.6g}, p={p:.6g}\n")
            f.write(f"Cohen dz    : dz={dz:.6g}\n")

    print(f"[Saved] {report_path}")

    if x2.size >= 2:
        t, p = stats.ttest_rel(x2, y2, nan_policy="omit")
        print(f"\n[Paired t-test] n={x2.size}, t={t:.4g}, p={p:.4g}")
    else:
        print("\n[Paired t-test] valid pairs < 2, skipped.")
