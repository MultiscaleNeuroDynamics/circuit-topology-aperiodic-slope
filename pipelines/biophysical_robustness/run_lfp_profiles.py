# -*- coding: utf-8 -*-
"""在四个端点试次上运行五组REAL/ER生物物理配置，共160项任务。 / Run five REAL/ER biophysical profiles on four endpoint trials (160 tasks)."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import LFPy
import numpy as np
import pandas as pd
from neuron import h

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from circuit_topology.cortical_neuron_biophys import cortical_neuron

TRIAL_IDS = (1, 4, 97, 100)
LAYERS = {
    "L2_3": ("23.swc", "syn_to_segment_L2_3.csv", "syn_to_segment_L2_3_ER.csv"),
    "L4": ("4.swc", "syn_to_segment_L4.csv", "syn_to_segment_L4_ER.csv"),
    "L5": ("5.swc", "syn_to_segment_L5.csv", "syn_to_segment_L5_ER.csv"),
    "L6": ("6.swc", "syn_to_segment_L6.csv", "syn_to_segment_L6_ER.csv"),
}
ELECTRODE_POS = (669.60, -100.00, 882.58)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work"))
    p.add_argument("--trials-root")
    p.add_argument("--profiles-root")
    p.add_argument("--output-root")
    p.add_argument("--max-workers", type=int, default=10)
    return p.parse_args()


def load_ee_set(path: Path):
    df = pd.read_csv(path, header=None, dtype=str).fillna("")
    return set(df.iloc[:, 0].str.strip())


def read_spikes(path: Path, topology: str, ee_set: set[str]):
    # 遵循原表约定：REAL可能含表头，ER不含表头。 / Match the original table conventions: REAL may contain a header, ER is headerless.
    df = pd.read_csv(path, dtype=str) if topology == "REAL" else pd.read_csv(path, header=None, dtype=str)
    result, inhibitory = {}, {}
    for _, row in df.iterrows():
        sid_text = str(row.iloc[0]).strip()
        try:
            sid = int(sid_text)
        except ValueError:
            continue
        vals = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().to_numpy(float)
        result[sid] = vals
        inhibitory[sid] = (str(sid) not in ee_set) if topology == "REAL" else sid_text.zfill(10).startswith("00")
    return result, inhibitory


def stable_delay(profile: str, topology: str, trial: str, layer: str, syn_id: int, low: float, high: float):
    token = f"{profile}|{topology}|{trial}|{layer}|{syn_id}".encode()
    integer = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    u = integer / float(2**64 - 1)
    return low + u * (high - low)


def run_task(task):
    profile, topology = task["profile"], task["topology"]
    trial, layer = task["trial"], task["layer"]
    out = Path(task["output"]) / profile / topology / trial / layer / "_lfp.npy"
    if out.is_file() and out.stat().st_size > 0:
        return "SKIP", profile, topology, trial, layer, str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(Path(task["parameter_json"]).read_text(encoding="utf-8"))
    ee_set = load_ee_set(Path(task["syn_ee_path"]))
    spike_file = Path(task["trial_dir"]) / (
        "microns_patch_synapse_spikes_real_ms.csv" if topology == "REAL"
        else "microns_patch_synapse_spikes_ER_ms.csv")
    spike_times, inhibitory = read_spikes(spike_file, topology, ee_set)
    conn = pd.read_csv(task["connection_file"], header=None, dtype=str)
    c0, c1 = pd.to_numeric(conn.iloc[:, 0], errors="coerce"), pd.to_numeric(conn.iloc[:, 1], errors="coerce")
    valid = c0.notna() & c1.notna()
    connection_data = zip(c0[valid].astype(np.int64), c1[valid].astype(np.int64))

    nrn = cortical_neuron(task["swc_file"], parameters)
    cell = LFPy.Cell(nrn.sectionList,
                     v_init=parameters["biophys_pars"]["pas_mem_pars"]["erev_leak"],
                     celsius=37.0, dt=0.01)
    cell.tstop = parameters["tmax"]
    h.load_file("stdrun.hoc"); h.cvode.active(1); h.cvode.atol(1e-6)
    meta = parameters["sensitivity_meta"]
    synapses = []
    for syn_id, segment_id in connection_data:
        syn_id, segment_id = int(syn_id), int(segment_id)
        times = spike_times.get(syn_id)
        if times is None or len(times) == 0:
            continue
        synpars = parameters["iSynParams" if inhibitory.get(syn_id, False) else "eSynParams"].copy()
        synpars.update({"idx": segment_id, "record_current": False})
        syn = LFPy.Synapse(cell, **synpars)
        delay = stable_delay(profile, topology, trial, layer, syn_id,
                             float(meta["delay_min_ms"]), float(meta["delay_max_ms"]))
        shifted = times + delay
        shifted = shifted[shifted <= float(parameters["tmax"])]
        syn.set_spike_times(shifted)
        synapses.append(syn)
    ex, ey, ez = ELECTRODE_POS
    electrode = LFPy.RecExtElectrode(cell, sigma=0.3, x=np.array([ex]), y=np.array([ey]),
                                     z=np.array([ez]), method="pointsource")
    cell.simulate(rec_vmem=True, rec_imem=True)
    if hasattr(electrode, "get_transformation_matrix"):
        matrix = electrode.get_transformation_matrix()
    elif hasattr(electrode, "calc_mapping"):
        try:
            matrix = electrode.calc_mapping(cell)
        except TypeError:
            matrix = electrode.calc_mapping()
    else:
        raise AttributeError("RecExtElectrode exposes neither transformation-matrix API")
    lfp = np.asarray(matrix @ cell.imem)[0]
    data = np.vstack([cell.tvec.copy(), lfp, cell.somav.copy()]).astype(np.float32)
    np.save(out, data)
    cell.strip_hoc_objects()
    del synapses, electrode, cell, nrn
    gc.collect()
    return "DONE", profile, topology, trial, layer, str(out)


def main():
    mp.freeze_support()
    args = parse_args()
    project = Path(args.project_root).resolve(); allen = project / "ALLEN"
    trials = Path(args.trials_root or allen / "file/final_trial").resolve()
    profiles = Path(args.profiles_root or allen / "file/biophys_endpoint_sensitivity").resolve()
    output = Path(args.output_root or allen / "save/m_0.99/biophys_endpoint_sensitivity").resolve()
    syn_ee = allen / "file/patch/syn_ee.csv"
    tasks = []
    for profile_json in sorted(profiles.glob("profile_*.json")):
        profile = profile_json.stem
        for topology in ("REAL", "ER"):
            for tid in TRIAL_IDS:
                trial = f"ER_trial{tid:04d}"; trial_dir = trials / trial
                for layer, (swc, real_map, er_map) in LAYERS.items():
                    task = {
                        "profile": profile, "parameter_json": str(profile_json), "topology": topology,
                        "trial": trial, "trial_dir": str(trial_dir), "layer": layer,
                        "swc_file": str(trial_dir / swc),
                        "connection_file": str(trial_dir / (real_map if topology == "REAL" else er_map)),
                        "syn_ee_path": str(syn_ee), "output": str(output),
                    }
                    for key in ("swc_file", "connection_file", "syn_ee_path"):
                        if not Path(task[key]).is_file():
                            raise FileNotFoundError(task[key])
                    tasks.append(task)
    if len(tasks) != 160:
        raise RuntimeError(f"Expected 160 tasks from five profiles, got {len(tasks)}")
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                status, profile, topology, trial, layer, path = future.result()
                print(f"[{status} {count:03d}/160] {profile} {topology} {trial} {layer} -> {path}", flush=True)
            except Exception as exc:
                errors.append(repr(exc)); print(f"[ERROR {count:03d}/160] {exc!r}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} tasks failed; first errors: {errors[:5]}")
    print(f"[DONE] all 160 endpoint sensitivity LFP files: {output}")


if __name__ == "__main__":
    main()
