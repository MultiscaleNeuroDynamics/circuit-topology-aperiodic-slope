# -*- coding: utf-8 -*-
"""为MICrONS拓扑生成分层LFP。 / Generate layer-resolved LFPs for the MICrONS topology."""

import os
import json
import sys
import time
import gc
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import LFPy
from neuron import h

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from circuit_topology.cortical_neuron import cortical_neuron as init_neuron

DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")
).resolve()
DEFAULT_ALLEN_ROOT = Path(
    os.environ.get("ALLEN_ROOT", DEFAULT_PROJECT_ROOT / "ALLEN")
).resolve()


# ==================================================
# ==================================================
ELECTRODE_POS = (669.60, -100.00, 882.58)  # 三维坐标，单位为微米。 / (x, y, z) in µm


# ==================================================
# 构建任务输入。 / Build task inputs.
# ==================================================
def load_ee_set(syn_ee_path: str) -> set:
    df = pd.read_csv(syn_ee_path, header=None, dtype=str).fillna("")
    ee = set(df.iloc[:, 0].astype(str).str.strip().tolist())
    ee.discard("")
    return ee


def build_synapse_spikes_and_ei_from_ee_set(
    spike_file: str,
    ee_set: set
):
    spike_df = pd.read_csv(spike_file)
    syn_ids = spike_df.iloc[:, 0].astype(int).values

    spike_times_by_syn = {}
    syn_is_inhibitory = {}

    for idx, syn_id in enumerate(syn_ids):
        syn_id_int = int(syn_id)
        row = spike_df.iloc[idx, 1:]
        times = row.dropna().astype(float).values
        spike_times_by_syn[syn_id_int] = times

        syn_is_inhibitory[syn_id_int] = (str(syn_id_int) not in ee_set)

    return syn_ids, spike_times_by_syn, syn_is_inhibitory


# ==================================================
# ==================================================
def addsyns_from_microns(connection_data,
                         spike_times_by_syn,
                         syn_is_inhibitory,
                         parameters):
    SYANPSE_FAILURE_RATE = 0.0

    synParams = []
    synTimes = []
    synTypes = []  # 0表示兴奋性，1表示抑制性。 / 0 = E, 1 = I

    for row in connection_data:
        syn_id = int(row[0])
        seg_idx = int(row[1])

        if syn_id not in spike_times_by_syn:
            continue

        ts_all = spike_times_by_syn[syn_id]
        if ts_all.size == 0:
            continue

        spikeSelection = np.random.binomial(ts_all.size, 1 - SYANPSE_FAILURE_RATE)
        if spikeSelection == 0:
            continue
        ts = np.random.choice(ts_all, spikeSelection, replace=False)
        ts = np.sort(ts).tolist()

        is_inh = syn_is_inhibitory.get(syn_id, False)

        if is_inh:
            params = parameters["iSynParams"].copy()
            synTypes.append(1)
        else:
            params = parameters["eSynParams"].copy()
            synTypes.append(0)

        # 突触处理细节。 / Synapse-processing details.
        params["record_current"] = False
        params["idx"] = seg_idx

        synParams.append(params)
        synTimes.append(ts)

    return synParams, synTimes, synTypes


# ==================================================
# ==================================================
def run_one_cell(parameters,
                 connection_file,
                 spike_times_by_syn,
                 syn_is_inhibitory,
                 savePath,
                 T_MAX,
                 swc_path,
                 dt=0.05,
                 use_cvode=True):

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    print(f"[{os.path.basename(swc_path)}] using point electrode at "
          f"(x={ELECTRODE_POS[0]:.2f}, y={ELECTRODE_POS[1]:.2f}, z={ELECTRODE_POS[2]:.2f}) µm")

    nrnM = init_neuron(swc_path, parameters)

    cell = LFPy.Cell(
        nrnM.sectionList,
        v_init=parameters["biophys_pars"]["pas_mem_pars"]["erev_leak"],
        celsius=37.0,
        dt=dt
    )
    cell.tstop = T_MAX

    if use_cvode:
        h.load_file("stdrun.hoc")
        h.cvode.active(1)
        h.cvode.atol(1e-6)

    # 兼容无表头映射及首行为文本的映射。 / Accept both headerless maps and maps with a textual first row.
    connection_df = pd.read_csv(connection_file, header=None, dtype=str)
    c0 = pd.to_numeric(connection_df.iloc[:, 0], errors="coerce")
    c1 = pd.to_numeric(connection_df.iloc[:, 1], errors="coerce")
    valid = c0.notna() & c1.notna()
    connection_data = np.column_stack([
        c0.loc[valid].astype(np.int64).to_numpy(),
        c1.loc[valid].astype(np.int64).to_numpy(),
    ])
    if connection_data.size == 0:
        raise RuntimeError(f"No valid [synapse_id, segment_id] rows in {connection_file}")

    # 突触处理细节。 / Synapse-processing details.
    synParams, synTimes, synTypes = addsyns_from_microns(
        connection_data, spike_times_by_syn, syn_is_inhibitory, parameters
    )

    syn_objects = []
    for p, ts in zip(synParams, synTimes):
        s = LFPy.Synapse(cell, **p)
        s.set_spike_times(np.array(ts, dtype=float))
        syn_objects.append(s)

    ex, ey, ez = ELECTRODE_POS
    electrode_parameters = {
        "sigma": 0.3,
        "x": np.array([ex]),
        "y": np.array([ey]),
        "z": np.array([ez]),
        "method": "pointsource" 
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)

    print(f"[{os.path.basename(swc_path)}] start simulate, synapses = {len(syn_objects)}")
    cell.simulate(rec_vmem=True, rec_imem=True)
    print(f"[{os.path.basename(swc_path)}] simulation finished")

    # LFP等于映射矩阵M乘以膜电流imem。 / -------- LFP = M @ imem --------
    imem = cell.imem  # 形状为分段数乘以时间点数。 / (n_seg, n_t)

    if hasattr(electrode, "get_transformation_matrix"):
        M = electrode.get_transformation_matrix()
    elif hasattr(electrode, "calc_mapping"):
        try:
            M = electrode.calc_mapping(cell)
        except TypeError:
            M = electrode.calc_mapping()
    else:
        raise AttributeError(
            "The installed LFPy electrode object exposes neither "
            "get_transformation_matrix() nor calc_mapping()."
        )

    lfp_all = M @ imem
    lfp = np.asarray(lfp_all)[0, :]

    t = cell.tvec.copy()
    v_soma = cell.somav.copy()

    cell.strip_hoc_objects()
    del electrode
    del cell

    # 保存生成的输出。 / Save generated output.
    data = np.vstack([t, lfp, v_soma]).astype(np.float32)
    np.save(os.path.join(savePath, "_lfp.npy"), data)

    print(f"[{os.path.basename(swc_path)}] saved LFP to {savePath}")
    return os.path.join(savePath, "_lfp.npy")


# ==================================================
# ==================================================
def run_microns_layer(
    layer_name,
    swc_file,
    conn_file,
    parameters,
    spike_times_by_syn,
    syn_is_inhibitory,
    save_root
):
    T_MAX = parameters["tmax"]
    # save_root已以皮层层名结尾；再次追加层名 / ``save_root`` already ends with the layer name.  Appending it again
    # 会生成历史trial/layer/layer结构，破坏续跑 / creates the legacy ``trial/layer/layer`` layout and breaks restart
    # 检测，因此新运行均直接写入trial/layer。 / detection, so all new runs write to ``trial/layer`` directly.
    savePath = save_root

    return run_one_cell(
        parameters=parameters,
        connection_file=conn_file,
        spike_times_by_syn=spike_times_by_syn,
        syn_is_inhibitory=syn_is_inhibitory,
        savePath=savePath,
        T_MAX=T_MAX,
        swc_path=swc_file,
        dt=0.01,
        use_cvode=True
    )


# ============================================================
# ============================================================
def _read_cpu_times():
    with open("/proc/stat", "r") as f:
        line = f.readline()
    parts = line.split()
    if parts[0] != "cpu":
        return None
    vals = list(map(int, parts[1:8]))
    user, nice, system, idle, iowait, irq, softirq = vals
    idle_all = idle + iowait
    non_idle = user + nice + system + irq + softirq
    total = idle_all + non_idle
    return total, idle_all

def _cpu_percent_linux(prev):
    cur = _read_cpu_times()
    if cur is None:
        return None, prev
    total, idle = cur
    if prev is None:
        return 0.0, cur
    prev_total, prev_idle = prev
    dt = total - prev_total
    di = idle - prev_idle
    if dt <= 0:
        return 0.0, cur
    cpu = (dt - di) / dt * 100.0
    return cpu, cur

def _mem_linux_gb():
    mt = None
    ma = None
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mt = int(line.split()[1])  # 千字节。 / kB
            elif line.startswith("MemAvailable:"):
                ma = int(line.split()[1])  # 千字节。 / kB
            if mt is not None and ma is not None:
                break
    if mt is None or ma is None:
        return None
    total_gb = mt / (1024**2)
    avail_gb = ma / (1024**2)
    used_gb = total_gb - avail_gb
    pct = used_gb / total_gb * 100.0 if total_gb > 0 else 0.0
    return used_gb, total_gb, pct

def _gpu_mem_line():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if not out:
            return None
        parts = []
        for line in out.splitlines():
            arr = [x.strip() for x in line.split(",")]
            if len(arr) >= 5:
                idx, name, used, total, util = arr[:5]
                parts.append(f"GPU{idx} {name} {used}/{total}MiB util={util}%")
        return " | ".join(parts) if parts else None
    except Exception:
        return None

def monitor_resources(stop_event: mp.Event, interval_sec: float = 20.0):
    prev = None
    while not stop_event.is_set():
        ts = time.strftime("%H:%M:%S")
        cpu, prev = _cpu_percent_linux(prev)
        mem = _mem_linux_gb()
        load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        if mem is not None:
            used_gb, total_gb, pct = mem
            msg = (f"[MON {ts}] CPU={cpu:5.1f}% | "
                   f"MEM={used_gb:6.2f}/{total_gb:.2f}GB ({pct:4.1f}%) | "
                   f"load={load[0]:.2f},{load[1]:.2f},{load[2]:.2f}")
        else:
            msg = f"[MON {ts}] CPU={cpu:5.1f}% | MEM=? | load={load[0]:.2f},{load[1]:.2f},{load[2]:.2f}"
        gpu = _gpu_mem_line()
        if gpu:
            msg += " | " + gpu
        print(msg, flush=True)
        stop_event.wait(interval_sec)


# ============================================================
# ============================================================
def run_one_sim_task(task: dict):
    """任务字典字段：试次名与目录、皮层层名、SWC文件、连接文件、参数、兴奋到兴奋连接集合，以及全局保存根目录。 / 
    task dict fields:
      - trial_name, trial_dir
      - layer_name, swc_file, conn_file
      - parameters
      - ee_set
      - save_root_global
    """
    import os
    import gc

    trial_name = task["trial_name"]
    trial_dir = task["trial_dir"]
    layer_name = task["layer_name"]
    swc_file = task["swc_file"]
    conn_file = task["conn_file"]
    parameters = task["parameters"]
    ee_set = task["ee_set"]
    save_root_global = task["save_root_global"]

    # ==================================================
    # 输出处理。 / Output handling.
    # ==================================================
    save_root = os.path.join(save_root_global, trial_name, layer_name)
    out_lfp_path = os.path.join(save_root, "_lfp.npy")

    if os.path.isfile(out_lfp_path):
        print(f"[SKIP] {trial_name} | {layer_name} -> exists: {out_lfp_path}", flush=True)
        return (trial_name, layer_name, out_lfp_path)

    # ==================================================
    # ==================================================
    spike_file = os.path.join(trial_dir, "microns_patch_synapse_spikes_real_ms.csv")
    if not os.path.isfile(spike_file):
        raise FileNotFoundError(f"[{trial_name}|{layer_name}] missing spike_file: {spike_file}")

    # 输入处理。 / Input handling.
    if not os.path.isfile(swc_file):
        raise FileNotFoundError(f"[{trial_name}|{layer_name}] missing swc_file: {swc_file}")
    if not os.path.isfile(conn_file):
        raise FileNotFoundError(f"[{trial_name}|{layer_name}] missing conn_file: {conn_file}")

    os.makedirs(save_root, exist_ok=True)

    # ==================================================
    # 构建任务输入。 / Build task inputs.
    # ==================================================
    syn_ids, spike_times_by_syn, syn_is_inhibitory = build_synapse_spikes_and_ei_from_ee_set(
        spike_file=spike_file,
        ee_set=ee_set
    )

    out = run_microns_layer(
        layer_name=layer_name,
        swc_file=swc_file,
        conn_file=conn_file,
        parameters=parameters,
        spike_times_by_syn=spike_times_by_syn,
        syn_is_inhibitory=syn_is_inhibitory,
        save_root=save_root
    )

    del syn_ids, spike_times_by_syn, syn_is_inhibitory
    gc.collect()

    return (trial_name, layer_name, out)


# ==================================================
# 构建任务输入。 / Build task inputs.
# ==================================================
if __name__ == "__main__":
    mp.freeze_support()

    # ==================================================
    # 并行执行。 / Parallel execution.
    # ==================================================
    MAX_PARALLEL = int(os.environ.get("MAX_WORKERS", "10"))
    MONITOR_INTERVAL_SEC = 5.0

    # ==================================================
    # 路径与配置。 / Paths and configuration.
    # ==================================================
    PATCH_DIR = os.environ.get(
        "PATCH_DIR",
        str(DEFAULT_ALLEN_ROOT / "file" / "patch")
    )
    SYN_EE_PATH = os.path.join(PATCH_DIR, "syn_ee.csv")
    param_file = os.environ.get(
        "PARAM_FILE",
        str(DEFAULT_ALLEN_ROOT / "file" / "swc" / "parameter.json")
    )

    trials_root = os.environ.get(
        "TRIALS_ROOT",
        str(DEFAULT_ALLEN_ROOT / "file" / "final_trial")
    )
    save_root_global = os.environ.get(
        "SAVE_ROOT_REAL",
        str(DEFAULT_ALLEN_ROOT / "save" / "final_real")
    )
    os.makedirs(save_root_global, exist_ok=True)

    # ==================================================
    # 加载输入数据。 / Load input data.
    # ==================================================
    if not os.path.isfile(param_file):
        raise FileNotFoundError(f"missing param_file: {param_file}")
    with open(param_file, "r", encoding="utf-8") as f:
        parameters = json.load(f)

    # ==================================================
    # 加载输入数据。 / Load input data.
    # ==================================================
    if not os.path.isfile(SYN_EE_PATH):
        raise FileNotFoundError(f"missing SYN_EE_PATH: {SYN_EE_PATH}")
    ee_set = load_ee_set(SYN_EE_PATH)
    print(f"[INFO] EE_SET size = {len(ee_set)}")

    # ==================================================
    # ==================================================
    def list_trial_dirs(base_dir):
        ds = []
        for name in os.listdir(base_dir):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                ds.append(p)
        ds.sort()
        return ds

    if not os.path.isdir(trials_root):
        raise FileNotFoundError(f"missing trials_root: {trials_root}")

    trial_dirs = list_trial_dirs(trials_root)
    if not trial_dirs:
        raise RuntimeError(f"No trial directories found under: {trials_root}")

    print(f"Trial directories found: {len(trial_dirs)}")
    for d in trial_dirs:
        print("  -", os.path.basename(d))

    # ==================================================
    # 构建任务输入。 / Build task inputs.
    # ==================================================
    tasks = []
    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir.rstrip("/"))

        layers_template = [
            ("L2_3", os.path.join(trial_dir, "23.swc"), os.path.join(trial_dir, "syn_to_segment_L2_3.csv")),
            ("L4",   os.path.join(trial_dir, "4.swc"),  os.path.join(trial_dir, "syn_to_segment_L4.csv")),
            ("L5",   os.path.join(trial_dir, "5.swc"),  os.path.join(trial_dir, "syn_to_segment_L5.csv")),
            ("L6",   os.path.join(trial_dir, "6.swc"),  os.path.join(trial_dir, "syn_to_segment_L6.csv")),
        ]

        for layer_name, swc_file, conn_file in layers_template:
            tasks.append({
                "trial_name": trial_name,
                "trial_dir": trial_dir,
                "layer_name": layer_name,
                "swc_file": swc_file,
                "conn_file": conn_file,
                "parameters": parameters,
                "ee_set": ee_set,
                "save_root_global": save_root_global,
            })

    print(f"\nTotal sim tasks = {len(tasks)} (trials={len(trial_dirs)} × layers=4)")
    print(f"MAX_PARALLEL = {MAX_PARALLEL}\n")

    # ==================================================
    # ==================================================
    stop_event = mp.Event()
    mon_proc = mp.Process(target=monitor_resources, args=(stop_event, MONITOR_INTERVAL_SEC), daemon=True)
    mon_proc.start()

    # ==================================================
    # ==================================================
    results = []
    errors = []

    t0 = time.time()
    try:
        with ProcessPoolExecutor(max_workers=min(MAX_PARALLEL, len(tasks))) as ex:
            futures = [ex.submit(run_one_sim_task, task) for task in tasks]

            done = 0
            total = len(futures)
            for fu in as_completed(futures):
                done += 1
                try:
                    trial_name, layer_name, out_path = fu.result()
                    results.append((trial_name, layer_name, out_path))
                    print(f"[DONE {done:03d}/{total}] {trial_name} | {layer_name} -> {out_path}", flush=True)
                except Exception as e:
                    errors.append(e)
                    print(f"[ERROR {done:03d}/{total}] {repr(e)}", flush=True)

    finally:
        stop_event.set()
        try:
            mon_proc.join(timeout=2.0)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"\nWall time: {dt/60:.2f} min")

    if errors:
        print("\nSome tasks failed. First 5 errors:")
        for e in errors[:5]:
            print("  -", repr(e))
        raise RuntimeError(f"{len(errors)} task(s) failed.")

    # ==================================================
    # 输出处理。 / Output handling.
    # ==================================================
    layer_order = {"L2_3": 0, "L4": 1, "L5": 2, "L6": 3}
    results.sort(key=lambda x: (x[0], layer_order.get(x[1], 99)))

    print("\n==============================")
    print("All tasks finished. Outputs:")
    print("==============================")
    cur = None
    for trial_name, layer_name, out_path in results:
        if cur != trial_name:
            cur = trial_name
            print(f"\n[{trial_name}]")
        print(f"  {layer_name}: {out_path}")
