# -*- coding: utf-8 -*-
"""生成40试次、m=0.99的保持度数DP对照。保留原1至100试次标识符，选取约等间距的40个标识符，复用相应兴奋/抑制比，独立搜索测得m_eff匹配论文目标的实现，并将DP结果写入独立目录树。下游旧脚本将随机对照称为ER；为避免改动已验证的NEURON代码，DP目录内使用历史*_ER_*兼容文件名，但其数据为DP，记录于dp_metadata.json和输出根目录名。 / Generate the 40-trial m=0.99 degree-preserving (DP) control.

This script deliberately keeps the original 1..100 trial identifiers.  It
selects 40 approximately equally spaced identifiers, reuses the corresponding
E/I values, independently searches for a realization whose measured m_eff
matches the manuscript target, and writes DP outputs into an isolated tree.

The downstream legacy scripts call a randomized control "ER".  To avoid risky
changes to the already validated NEURON code, DP files in the isolated DP tree
use the legacy *_ER_* compatibility filenames.  They are DP data, as recorded
in dp_metadata.json and the output root name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
SOURCE_SCRIPT = REPOSITORY_ROOT / "pipelines" / "simulation" / "generate_main_trials.py"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")
)
DEFAULT_ALLEN_ROOT = Path(os.environ.get("ALLEN_ROOT", DEFAULT_PROJECT_ROOT / "ALLEN"))

# 按等间距取整选择原论文试次标识符， / np.rint(np.linspace(1, 100, 40)).  These are original manuscript trial IDs,
# 而非重新赋予1至40的标识符。 / not newly assigned 1..40 identifiers.
SELECTED_TRIAL_IDS = (
    1, 4, 6, 9, 11, 14, 16, 19, 21, 24,
    26, 29, 31, 34, 37, 39, 42, 44, 47, 49,
    52, 54, 57, 59, 62, 64, 67, 70, 72, 75,
    77, 80, 82, 85, 87, 90, 92, 95, 97, 100,
)


def load_manuscript_module():
    source = SOURCE_SCRIPT
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing manuscript dynamics script: {source}\n"
            "The release must include pipelines/simulation/generate_main_trials.py."
        )
    spec = importlib.util.spec_from_file_location("manuscript_m099", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _edge_list_from_matrix(W: np.ndarray):
    """返回W[post, pre]对应的突触前、突触后及权重三列记录。 / Return [pre, post, weight] rows for W[post, pre]."""
    posts, pres = np.nonzero(W)
    return [
        [int(pre), int(post), float(W[post, pre])]
        for post, pre in zip(posts, pres)
    ]


def degree_preserving_rewire(
    W: np.ndarray,
    seed: int,
    nswap_factor: int = 20,
    allow_self_loops: bool = False,
):
    """对W[post, pre]实施有向双边交换，将u→v与x→y改为u→y与x→v，保持每个节点的二值入度和出度。将权重保留于来源边，同时保持每个源节点的总出强度、突触总数及全局连接重数分布。 / Directed double-edge swap for W[post, pre].

    A swap (u->v, x->y) -> (u->y, x->v) preserves every node's binary
    in-degree and out-degree.  Keeping each weight with its source edge also
    preserves every source node's total outgoing strength, the total synapse
    count, and the global multiplicity distribution.
    """
    edges = _edge_list_from_matrix(np.asarray(W))
    if len(edges) < 2:
        return np.asarray(W).copy(), 0, 0

    rng = np.random.default_rng(seed)
    occupied = {(u, v) for u, v, _ in edges}
    target_swaps = int(max(1, nswap_factor * len(edges)))
    max_attempts = int(max(1000, target_swaps * 50))
    accepted = 0
    attempts = 0

    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        a, b = rng.choice(len(edges), size=2, replace=False)
        u, v, wu = edges[int(a)]
        x, y, wx = edges[int(b)]
        if u == x or v == y:
            continue
        if not allow_self_loops and (u == y or x == v):
            continue
        if (u, y) in occupied or (x, v) in occupied:
            continue

        occupied.remove((u, v))
        occupied.remove((x, y))
        occupied.add((u, y))
        occupied.add((x, v))
        edges[int(a)] = [u, y, wu]
        edges[int(b)] = [x, v, wx]
        accepted += 1

    out = np.zeros_like(W)
    for pre, post, weight in edges:
        out[post, pre] = weight
    return out, accepted, attempts


def assert_dp_invariants(W_real: np.ndarray, W_dp: np.ndarray, label: str):
    A = np.asarray(W_real)
    B = np.asarray(W_dp)
    checks = {
        "shape": A.shape == B.shape,
        "edge_count": int(np.count_nonzero(A)) == int(np.count_nonzero(B)),
        "total_weight": bool(np.isclose(A.sum(), B.sum())),
        "binary_indegree": np.array_equal((A > 0).sum(axis=1), (B > 0).sum(axis=1)),
        "binary_outdegree": np.array_equal((A > 0).sum(axis=0), (B > 0).sum(axis=0)),
        "out_strength": np.allclose(A.sum(axis=0), B.sum(axis=0)),
        "weight_multiset": np.array_equal(
            np.sort(A[A > 0].astype(float)), np.sort(B[B > 0].astype(float))
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"DP invariant failure for {label}: {failed}")
    return {name: bool(ok) for name, ok in checks.items()}


def calibrate_A_for_dp(
    module,
    G_dp,
    rho_dp: float,
    initial_A: float,
    output_root: Path,
    calibration_tol: float = 0.002,
    initial_relative_span: float = 0.20,
    max_iterations: int = 12,
    max_expansions: int = 6,
):
    """在内存中校准DP的A，不创建A_calibration目录。遵循m=0.90/0.95代码的安全单侧过程：各候选值使用相同五个种子，以测得m_eff的中位数为目标，A可从1.19116降低但不能增大。 / Calibrate DP A in memory; never create an A_calibration directory.

    This mirrors the safe one-sided procedure used by the m=0.90/0.95 code:
    each candidate is evaluated with the same five seeds, the median measured
    m_eff is targeted, and A may decrease from 1.19116 but never increase.
    """
    history_path = output_root / "dp_A_calibration_history_m0.99.csv"
    selected_path = output_root / "dp_A_selected_m0.99.json"
    if selected_path.is_file() and selected_path.stat().st_size > 0:
        with open(selected_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        selected_A = float(saved["A_dp_calibrated"])
        if selected_A > float(initial_A) + 1e-12:
            raise RuntimeError(
                f"Saved unsafe A_dp={selected_A} exceeds initial A={initial_A}"
            )
        print(f"[CAL DP SKIP] reuse A_dp={selected_A:.8f} from {selected_path}")
        return selected_A

    target_m = 0.99
    seeds = [91001, 91002, 91003, 91004, 91005]
    sim_kwargs = dict(
        tmax_sec=22.0,
        dt=0.001,
        warmup_time=2.0,
        refractory_period=0.001,
        external_rate=1,
        syn_delay_mean=0.001,
    )
    history = []
    cache = {}

    def evaluate(A):
        A = float(A)
        key = round(A, 12)
        if key in cache:
            return cache[key]
        values = []
        external_values = []
        rho_eff = float(rho_dp) / A
        for seed in seeds:
            _, meff, ext_total = module.simulate_event_driven_branching(
                G_dp,
                rho=rho_eff,
                m0=target_m,
                name=f"CAL_DP_A{A:.8f}_seed{seed}",
                seed=int(seed),
                **sim_kwargs,
            )
            if np.isfinite(meff) and meff > 0:
                values.append(float(meff))
            external_values.append(int(ext_total))
        row = {
            "network": "DP",
            "target_m": target_m,
            "A": A,
            "rho": float(rho_dp),
            "rho_eff": rho_eff,
            "n_seeds": len(values),
            "median_meff": float(np.median(values)) if values else np.nan,
            "mean_meff": float(np.mean(values)) if values else np.nan,
            "sd_meff": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "mean_external_inputs": (
                float(np.mean(external_values)) if external_values else np.nan
            ),
        }
        row["abs_error"] = (
            abs(row["median_meff"] - target_m)
            if np.isfinite(row["median_meff"])
            else np.inf
        )
        history.append(row)
        cache[key] = row
        print(
            f"[CAL DP] A={A:.8f}, median m_eff={row['median_meff']:.6f}, "
            f"error={row['abs_error']:.6f}"
        )
        return row

    low_A = max(1e-8, float(initial_A) * (1.0 - initial_relative_span))
    low = evaluate(low_A)
    high = evaluate(float(initial_A))
    expansions = 0
    while expansions < max_expansions:
        low_y = low["median_meff"]
        high_y = high["median_meff"]
        if np.isfinite(low_y) and np.isfinite(high_y) and low_y <= target_m <= high_y:
            break
        if np.isfinite(low_y) and low_y > target_m:
            high = low
            low_A = max(1e-8, low_A * 0.85)
            low = evaluate(low_A)
        elif np.isfinite(high_y) and high_y < target_m:
            print(
                "[CAL DP WARNING] target is above m_eff at initial A; "
                "the safe one-sided search will not increase A."
            )
            break
        else:
            break
        expansions += 1

    bracketed = (
        np.isfinite(low["median_meff"])
        and np.isfinite(high["median_meff"])
        and low["median_meff"] <= target_m <= high["median_meff"]
    )
    if bracketed:
        for _ in range(max_iterations):
            mid = evaluate(0.5 * (low["A"] + high["A"]))
            if mid["abs_error"] <= calibration_tol:
                break
            if mid["median_meff"] < target_m:
                low = mid
            else:
                high = mid
    else:
        print("[CAL DP WARNING] target was not bracketed; selecting closest safe A")

    finite = [row for row in history if np.isfinite(row["abs_error"])]
    if not finite:
        raise RuntimeError("DP A calibration produced no finite m_eff")
    best = min(finite, key=lambda row: row["abs_error"])
    selected_A = float(best["A"])
    if selected_A > float(initial_A) + 1e-12:
        raise RuntimeError("DP calibration attempted to increase A above initial value")

    # 文件直接写入m_0.99_DP40，不创建A_calibration目录， / Files are written directly in m_0.99_DP40.  No A_calibration directory
    # 也不创建虚构试次目录。 / and no fake trial directory are created.
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    selected = {
        "target_m": target_m,
        "A_dp_initial": float(initial_A),
        "A_dp_calibrated": selected_A,
        "median_meff": float(best["median_meff"]),
        "abs_error": float(best["abs_error"]),
        "rho_calibration_DP": float(rho_dp),
        "calibration_topology_seed": 10001,
        "calibration_seeds": seeds,
        "search_constraint": "A_dp <= A_dp_initial; downward-only",
        "history_file": str(history_path),
    }
    with open(selected_path, "w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2, ensure_ascii=False)
    print(
        f"[CAL DP SELECTED] A_dp={selected_A:.8f}, "
        f"median m_eff={best['median_meff']:.6f}"
    )
    return selected_A


def find_original_trial_root(allen_root: Path, explicit: str | None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            allen_root / "file" / "final_trial",
            allen_root / "file" / "add_trial",
            allen_root / "file" / "add_trial" / "m_0.99",
            allen_root / "file" / "trails",
        ]
    )
    for root in candidates:
        if all((root / f"ER_trial{i:04d}").is_dir() for i in SELECTED_TRIAL_IDS):
            return root
    listed = "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not locate all selected manuscript trial directories. Checked:\n  - " + listed
    )


def build_microns_matrices(module, allen_root: Path):
    """安全重定向三个路径后运行原始矩阵构建器。论文函数含硬编码路径；仅在内存中替换临时函数副本中的字面路径赋值，不编辑论文文件本身。 / Run the original matrix builder after safely redirecting its three paths.

    The manuscript function contains hard-coded paths.  We execute a temporary
    function clone with only those literal path assignments replaced in memory;
    the manuscript file itself is never edited.
    """
    import inspect
    import re

    fn_source = inspect.getsource(module.build_microns_EE_EI_IE_matrices)
    replacements = {
        "celltype_path": allen_root / "file" / "proofread_cells_celltypes_auto.csv",
        "meta_path": allen_root / "file" / "proofread_cells_metadata.csv",
        "syn_path": allen_root / "file" / "proofread_cells_synapses.csv",
    }
    missing = [str(path) for path in replacements.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required MICrONS source CSV files are missing:\n  - "
            + "\n  - ".join(missing)
        )
    lines = fn_source.splitlines()
    replaced = set()
    for index, line in enumerate(lines):
        for variable, path in replacements.items():
            match = re.match(rf"^(\s*){re.escape(variable)}\s*=.*$", line)
            if match:
                lines[index] = f"{match.group(1)}{variable} = {str(path)!r}"
                replaced.add(variable)
    not_replaced = sorted(set(replacements) - replaced)
    if not_replaced:
        raise RuntimeError(
            "Could not redirect hard-coded MICrONS paths in the manuscript "
            f"matrix builder: {not_replaced}"
        )
    namespace = dict(module.__dict__)
    exec("\n".join(lines), namespace)
    return namespace["build_microns_EE_EI_IE_matrices"]()


def copy_trial_assets(original_trial: Path, output_trial: Path):
    output_trial.mkdir(parents=True, exist_ok=True)
    names = (
        "23.swc", "4.swc", "5.swc", "6.swc",
        "syn_to_segment_L2_3.csv", "syn_to_segment_L4.csv",
        "syn_to_segment_L5.csv", "syn_to_segment_L6.csv",
        "microns_patch_spikes_real_ms.csv",
    )
    for name in names:
        src = original_trial / name
        dst = output_trial / name
        if src.is_file() and src.stat().st_size > 0 and not dst.exists():
            shutil.copy2(src, dst)


def read_original_trial_settings(original_trial: Path, trial_id: int):
    """读取论文兴奋/抑制比，必要时重建其精确网格值。 / Read the manuscript E/I ratio; reconstruct its exact grid value if needed."""
    settings = {}
    summary = original_trial / "summary.txt"
    if summary.is_file():
        with open(summary, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                settings[key.strip()] = value.strip()

    used_ei_formula = "EI_ratio" not in settings
    ei_ratio = (
        float(settings["EI_ratio"])
        if not used_ei_formula
        else float(np.linspace(1.0 / 3.0, 3.0, 100)[trial_id - 1])
    )
    return {
        "EI_ratio": ei_ratio,
        "settings_source": str(summary) if not used_ei_formula else "100_trial_EI_grid_formula",
        "used_EI_formula": used_ei_formula,
    }


def save_spike_dict(module, spikes_e, spikes_i, n_e, n_i, output_path: Path):
    e_ids = np.arange(n_e, dtype=int)
    i_ids = np.arange(n_i, dtype=int)
    e_dict = module.build_spike_dict_from_spike_times(spikes_e, e_ids, id_offset=0)
    i_dict = module.build_spike_dict_from_spike_times(spikes_i, i_ids, id_offset=n_e)
    merged = {}
    for data in (e_dict, i_dict):
        for key, values in data.items():
            merged.setdefault(key, []).extend(values)
    module.save_spikes_ms_csv_from_dict(merged, str(output_path))


def run_one_dp_trial(task):
    trial_id = int(task["trial_id"])
    output_root = Path(task["output_root"])
    original_root = Path(task["original_root"])
    allen_root = Path(task["allen_root"])
    trial_out = output_root / f"ER_trial{trial_id:04d}"
    trial_src = original_root / f"ER_trial{trial_id:04d}"
    trial_out.mkdir(parents=True, exist_ok=True)

    required = (
        "microns_patch_spikes_ER_ms.csv",
        "microns_patch_edges_ER_weights.csv",
        "microns_patch_edges_ER_IE_weights.csv",
        "dp_metadata.json",
    )
    if all((trial_out / name).is_file() and (trial_out / name).stat().st_size > 0 for name in required):
        print(f"[SKIP DP dynamics] original trial {trial_id:04d} already complete", flush=True)
        with open(trial_out / "dp_metadata.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    module = load_manuscript_module()
    W_EE, W_EI, W_IE, ee_ids, i_ids = build_microns_matrices(module, allen_root)
    n_e = int(W_EE.shape[0])
    n_i = int(W_IE.shape[1]) if W_IE.ndim == 2 else 0

    # 拓扑随机化可重复，但动力学种子有意 / Topology randomization is reproducible, but dynamical seeds are deliberately
    # 独立于REAL：比较按测得的m_eff匹配，而非按种子匹配。 / independent from REAL: comparison is matched on measured m_eff, not seed.
    topology_seed_ee = 10000 + trial_id
    topology_seed_ie = 4000 + trial_id
    W_EE_dp, swaps_ee, attempts_ee = degree_preserving_rewire(
        W_EE, seed=topology_seed_ee, nswap_factor=int(task["nswap_factor"]),
        allow_self_loops=False,
    )
    W_IE_dp, swaps_ie, attempts_ie = degree_preserving_rewire(
        W_IE, seed=topology_seed_ie, nswap_factor=int(task["nswap_factor"]),
        allow_self_loops=True,
    )
    checks_ee = assert_dp_invariants(W_EE, W_EE_dp, "EE")
    checks_ie = assert_dp_invariants(W_IE, W_IE_dp, "IE")

    rho_dp, _ = module.spectral_radius(W_EE_dp, name=f"DP_EE_trial{trial_id:04d}")
    G_dp = module.matrix_to_digraph(W_EE_dp)
    sim_kwargs = dict(
        tmax_sec=22.0,
        dt=0.001,
        warmup_time=2.0,
        refractory_period=0.001,
        external_rate=1,
        syn_delay_mean=0.001,
    )

    original_settings = read_original_trial_settings(trial_src, trial_id)
    base_seed_dp = 6001 + 1000 * trial_id
    A_dp = float(task["A_dp"])
    spikes_dp_e, meff_dp, inner_trial, selected_seed, ext_total = module.run_until_meff_close(
        G_dp,
        rho_eff=rho_dp / A_dp,
        m0=0.99,
        tol=5e-4,
        max_trials=200,
        base_seed=base_seed_dp,
        name=f"DP_trial{trial_id:04d}",
        sim_kwargs=sim_kwargs,
    )
    if not np.isfinite(meff_dp) or abs(float(meff_dp) - 0.99) >= 5e-4:
        raise RuntimeError(
            f"DP trial {trial_id:04d} failed m_eff matching: "
            f"m_eff={meff_dp}, target=0.99, tolerance=5e-4"
        )

    _, E_bins = module.bin_counts_E(
        spikes_dp_e, dt=0.001, t_start=2.0, t_end=22.0, N_E=n_e
    )
    ei_ratio = float(original_settings["EI_ratio"])
    spikes_dp_i = module.generate_ER_I_spikes_by_E_bins(
        E_bin_counts=E_bins,
        N_I=n_i,
        dt=0.001,
        t_start=2.0,
        EI_ratio=ei_ratio,
        seed=3000 + trial_id,
    )

    # 兼容命名：下游已验证的ER脚本仅在 / Compatibility names: downstream validated ER scripts operate only inside
    # 独立DP根目录内运行，因此不会触及已有REAL/ER论文文件。 / the isolated DP root, so existing REAL/ER manuscript files cannot be touched.
    save_spike_dict(
        module, spikes_dp_e, spikes_dp_i, n_e, n_i,
        trial_out / "microns_patch_spikes_ER_ms.csv",
    )
    module.save_er_edges_weights_EE(
        W_EE_dp, str(trial_out / "microns_patch_edges_ER_weights.csv")
    )
    module.save_er_edges_weights_IE(
        W_IE_dp,
        I_id_offset=n_e,
        out_path=str(trial_out / "microns_patch_edges_ER_IE_weights.csv"),
    )
    copy_trial_assets(trial_src, trial_out)

    metadata = {
        "topology": "DP_directed_degree_preserving",
        "source_manuscript_trial_id": trial_id,
        "source_trial_directory": str(trial_src),
        "m0": 0.99,
        "EI_ratio": ei_ratio,
        "A_dp": A_dp,
        "rho_dp": float(rho_dp),
        "m_eff_dp": float(meff_dp),
        "m_eff_target": 0.99,
        "m_eff_tolerance": 5e-4,
        "DP_base_seed_independent_stream": int(base_seed_dp),
        "DP_selected_seed": int(selected_seed),
        "DP_inner_trial": int(inner_trial),
        "EI_settings_source": original_settings["settings_source"],
        "used_EI_formula": bool(original_settings["used_EI_formula"]),
        "external_inputs_dp": int(ext_total),
        "topology_seed_EE_matches_ER_schedule": int(topology_seed_ee),
        "topology_seed_IE_matches_ER_schedule": int(topology_seed_ie),
        "accepted_swaps_EE": int(swaps_ee),
        "attempted_swaps_EE": int(attempts_ee),
        "accepted_swaps_IE": int(swaps_ie),
        "attempted_swaps_IE": int(attempts_ie),
        "invariants_EE": checks_ee,
        "invariants_IE": checks_ie,
        "legacy_filename_note": "Files named ER in this isolated root contain DP data.",
    }
    with open(trial_out / "dp_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def write_selection_manifest(path: Path, original_root: Path):
    rows = []
    for trial_id in SELECTED_TRIAL_IDS:
        trial_root = original_root / f"ER_trial{trial_id:04d}"
        original = read_original_trial_settings(trial_root, trial_id)
        rows.append(
            {
                "source_manuscript_trial_id": trial_id,
                "source_trial_name": f"ER_trial{trial_id:04d}",
                "EI_ratio_from_manuscript_trial": float(original["EI_ratio"]),
                "DP_independent_base_seed_schedule": 6001 + 1000 * trial_id,
                "ER_topology_seed_schedule": 10000 + trial_id,
                "settings_source": original["settings_source"],
                "used_EI_formula": bool(original["used_EI_formula"]),
                "source_directory": str(trial_root),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--original-trials-root", default=os.environ.get("ORIGINAL_TRIALS_ROOT"))
    parser.add_argument(
        "--output-root",
        default=os.environ.get("DP_TRIALS_ROOT"),
        help="Default: ALLEN/file/add_trial/m_0.99_DP40",
    )
    parser.add_argument(
        "--A-dp", type=float,
        help="Explicitly override calibration. Normally omit this option.",
    )
    parser.add_argument(
        "--initial-A-dp", type=float, default=1.19116,
        help="Safe upper bound and starting value for automatic DP calibration.",
    )
    parser.add_argument("--nswap-factor", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 1)))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    allen_root = Path(os.environ.get("ALLEN_ROOT", project_root / "ALLEN"))
    original_root = find_original_trial_root(allen_root, args.original_trials_root)
    output_root = Path(args.output_root) if args.output_root else (
        allen_root / "file" / "add_trial" / "m_0.99_DP40"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.A_dp is None:
        # 正式生成试次前，在一个确定性的代表性DP拓扑上校准， / Calibrate on one deterministic representative DP topology before any
        # 不向试次文件夹写入任何内容。 / formal trial is generated.  Nothing is written into a trial folder.
        manuscript = load_manuscript_module()
        W_EE, _, _, _, _ = build_microns_matrices(manuscript, allen_root)
        W_cal_dp, _, _ = degree_preserving_rewire(
            W_EE, seed=10001, nswap_factor=int(args.nswap_factor),
            allow_self_loops=False,
        )
        assert_dp_invariants(W_EE, W_cal_dp, "EE calibration topology")
        rho_cal_dp, _ = manuscript.spectral_radius(W_cal_dp, name="CAL_DP_EE")
        G_cal_dp = manuscript.matrix_to_digraph(W_cal_dp)
        selected_A_dp = calibrate_A_for_dp(
            manuscript,
            G_cal_dp,
            rho_cal_dp,
            initial_A=float(args.initial_A_dp),
            output_root=output_root,
        )
    else:
        selected_A_dp = float(args.A_dp)
        if selected_A_dp > float(args.initial_A_dp) + 1e-12:
            raise ValueError(
                f"Explicit A_dp={selected_A_dp} exceeds safe initial upper bound "
                f"{args.initial_A_dp}"
            )
        print(f"[CAL DP OVERRIDE] using explicit A_dp={selected_A_dp:.8f}")

    manifest_rows = write_selection_manifest(
        output_root / "selected_40_from_manuscript_100.csv", original_root
    )
    tasks = []
    for row in manifest_rows:
        tasks.append(
            {
                "trial_id": int(row["source_manuscript_trial_id"]),
                "ei_ratio": float(row["EI_ratio_from_manuscript_trial"]),
                "output_root": str(output_root),
                "original_root": str(original_root),
                "allen_root": str(allen_root),
                "A_dp": float(selected_A_dp),
                "nswap_factor": int(args.nswap_factor),
            }
        )

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as executor:
        futures = {executor.submit(run_one_dp_trial, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[DP DONE {done:02d}/40] source trial "
                    f"{int(result['source_manuscript_trial_id']):04d} "
                    f"m_eff={result['m_eff_dp']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append((task["trial_id"], repr(exc)))
                print(f"[DP ERROR] source trial {task['trial_id']:04d}: {exc!r}", flush=True)

    if results:
        pd.DataFrame(results).sort_values("source_manuscript_trial_id").to_csv(
            output_root / "dp40_dynamics_summary_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if errors:
        raise RuntimeError(f"{len(errors)} DP trial(s) failed: {errors[:5]}")

    pd.DataFrame(results).sort_values("source_manuscript_trial_id").to_csv(
        output_root / "dp40_dynamics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[DP DYNAMICS COMPLETE] {output_root}")


if __name__ == "__main__":
    main()
