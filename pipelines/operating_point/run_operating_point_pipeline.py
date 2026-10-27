# -*- coding: utf-8 -*-
"""运行并验证m=0.90和0.95的完整动力学后处理流程。 / Run and validate the complete post-dynamics pipeline for m=0.90 and 0.95."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ALLEN_ROOT = PROJECT_ROOT / "ALLEN"
PIPELINE_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = PIPELINE_DIR.parent / "simulation"
M_VALUES = ("0.90", "0.95")

SCRIPTS = {
    "generate_dynamics": PIPELINE_DIR / "generate_operating_point_trials.py",
    "maps_real": SIMULATION_DIR / "assign_trial_morphologies.py",
    "synapse_spikes": SIMULATION_DIR / "expand_soma_spikes_to_synapses.py",
    "clean_nan": SIMULATION_DIR / "clean_spike_tables.py",
    "maps_er": SIMULATION_DIR / "generate_er_segment_maps.py",
    "neuron_real": SIMULATION_DIR / "run_real_lfp.py",
    "neuron_er": SIMULATION_DIR / "run_er_lfp.py",
}

BASE_DYNAMICS_FILES = (
    "microns_patch_spikes_real_ms.csv",
    "microns_patch_spikes_ER_ms.csv",
    "microns_patch_edges_ER_weights.csv",
    "microns_patch_edges_ER_IE_weights.csv",
)
REAL_MAPS = (
    "syn_to_segment_L2_3.csv",
    "syn_to_segment_L4.csv",
    "syn_to_segment_L5.csv",
    "syn_to_segment_L6.csv",
)
ER_MAPS = tuple(name.replace(".csv", "_ER.csv") for name in REAL_MAPS)
SWCS = ("23.swc", "4.swc", "5.swc", "6.swc")
SYNAPSE_SPIKES = (
    "microns_patch_synapse_spikes_real_ms.csv",
    "microns_patch_synapse_spikes_ER_ms.csv",
)


def trial_dirs(root: Path):
    result = []
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and re.fullmatch(r"ER_trial\d+", path.name):
                result.append(path)
    return sorted(result, key=lambda p: int(re.search(r"\d+", p.name).group()))


def validate_files(root: Path, required, stage: str, expected_trials=40):
    trials = trial_dirs(root)
    if len(trials) != expected_trials:
        raise RuntimeError(
            f"[{stage}] expected {expected_trials} trial directories under {root}, "
            f"found {len(trials)}"
        )
    missing = []
    empty = []
    for trial in trials:
        for name in required:
            path = trial / name
            if not path.is_file():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                empty.append(str(path))
    if missing or empty:
        msg = [f"[{stage}] validation failed"]
        if missing:
            msg.append(f"missing ({len(missing)}): " + "; ".join(missing[:10]))
        if empty:
            msg.append(f"empty ({len(empty)}): " + "; ".join(empty[:10]))
        raise RuntimeError("\n".join(msg))
    print(f"[VALID {stage}] {len(trials)} trials; all required files present and non-empty")


def all_trial_files_complete(root: Path, required, expected_trials=40):
    trials = trial_dirs(root)
    return (
        len(trials) == expected_trials
        and all(
            (trial / name).is_file() and (trial / name).stat().st_size > 0
            for trial in trials
            for name in required
        )
    )


def validate_project_dependencies():
    required = (
        ALLEN_ROOT / "file" / "patch" / "syn_ee.csv",
        ALLEN_ROOT / "file" / "patch" / "syn_ie.csv",
        ALLEN_ROOT / "file" / "swc" / "parameter.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    swc_root = ALLEN_ROOT / "file" / "select_swc"
    for folder in ("L2_3", "L4", "L5", "L6"):
        layer_dir = swc_root / folder
        if not layer_dir.is_dir() or not any(layer_dir.glob("*.swc")):
            missing.append(f"{layer_dir}/*.swc")

    if missing:
        raise FileNotFoundError(
            "Project dependency preflight failed:\n  - " + "\n  - ".join(missing)
        )
    print("[VALID dependencies] patch tables, parameter.json, SWC pools and mapping module found")


def expected_lfp_candidates(save_root: Path, expected_trials=40):
    """返回每项模拟任务的当前及历史输出路径。 / Return the current and legacy output paths for every simulation task."""
    return [
        (
            save_root / f"ER_trial{trial_idx:04d}" / layer / "_lfp.npy",
            save_root / f"ER_trial{trial_idx:04d}" / layer / layer / "_lfp.npy",
        )
        for trial_idx in range(1, expected_trials + 1)
        for layer in ("L2_3", "L4", "L5", "L6")
    ]


def existing_lfp(paths):
    return next(
        (path for path in paths if path.is_file() and path.stat().st_size > 0),
        None,
    )


def validate_lfp_tree(save_root: Path, stage: str, expected_trials=40):
    groups = expected_lfp_candidates(save_root, expected_trials)
    missing = [str(paths[0]) for paths in groups if existing_lfp(paths) is None]
    empty = [str(path) for paths in groups for path in paths if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(
            f"[{stage}] missing={len(missing)}, empty={len(empty)}; "
            f"examples={(missing + empty)[:10]}"
        )
    print(f"[VALID {stage}] {len(groups)} non-empty _lfp.npy files")


def lfp_tree_complete(save_root: Path, expected_trials=40):
    return all(
        existing_lfp(paths) is not None
        for paths in expected_lfp_candidates(save_root, expected_trials)
    )


def lfp_tree_status(save_root: Path, expected_trials=40):
    groups = expected_lfp_candidates(save_root, expected_trials)
    complete = [paths for paths in groups if existing_lfp(paths) is not None]
    empty = [path for paths in groups for path in paths if path.is_file() and path.stat().st_size == 0]
    return len(complete), len(groups), empty


def run_neuron_stage_without_overwrite(
        save_root: Path, script_key: str, stage: str, env, expected_trials=40):
    """续跑NEURON阶段，不替换当前或历史输出。 / Resume a NEURON stage without replacing current or legacy outputs."""
    complete_count, expected_count, empty = lfp_tree_status(save_root, expected_trials)
    if complete_count == expected_count and not empty:
        print(f"[SKIP stage] all {expected_count} {stage} _lfp.npy files already exist")
        return
    if empty:
        raise RuntimeError(
            f"[SAFE STOP {stage}] found {len(empty)} empty LFP files: "
            f"{[str(path) for path in empty[:10]]}"
        )

    # 当前模拟器跳过直接路径；对于旧版双层目录结果， / Current simulators skip the direct path. For a result produced by an older
    # 创建临时直接符号链接，使其也被跳过。 / double-layer layout, create a temporary direct symlink so it is also skipped.
    created_links = []
    try:
        for direct_path, legacy_path in expected_lfp_candidates(save_root, expected_trials):
            if direct_path.exists() or not (legacy_path.is_file() and legacy_path.stat().st_size > 0):
                continue
            direct_path.parent.mkdir(parents=True, exist_ok=True)
            direct_path.symlink_to(legacy_path)
            created_links.append(direct_path)
        print(f"[RESUME {stage}] preserving {complete_count}/{expected_count} completed outputs")
        run_script(script_key, env)
    finally:
        for link in created_links:
            try:
                if link.is_symlink():
                    link.unlink()
            except OSError as exc:
                print(f"[WARN] could not remove temporary skip link {link}: {exc}")


def run_script(script_key: str, env, extra_args=()):
    script = Path(SCRIPTS[script_key])
    if not script.is_file():
        raise FileNotFoundError(script)
    print("\n" + "=" * 88)
    print(f"RUN {script.name}")
    print(f"TRIALS_ROOT={env['TRIALS_ROOT']}")
    subprocess.run(
        [sys.executable, str(script), *extra_args],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


def env_for_m(m_value: str, max_workers: int):
    env = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["ALLEN_ROOT"] = str(ALLEN_ROOT)
    env["TRIALS_ROOT"] = str(ALLEN_ROOT / "file" / "add_trial" / f"m_{m_value}")
    env["PATCH_DIR"] = str(ALLEN_ROOT / "file" / "patch")
    env["SELECT_SWC_ROOT"] = str(ALLEN_ROOT / "file" / "select_swc")
    env["PARAM_FILE"] = str(ALLEN_ROOT / "file" / "swc" / "parameter.json")
    env["SAVE_ROOT_REAL"] = str(ALLEN_ROOT / "save" / f"m_{m_value}" / "final_real")
    env["SAVE_ROOT_ER"] = str(ALLEN_ROOT / "save" / f"m_{m_value}" / "final_ER")
    env["MECHANISM_ROOT"] = str(REPOSITORY_ROOT / "mechanisms")
    env["MAX_WORKERS"] = str(max_workers)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-neuron", action="store_true",
        help="After preprocessing, run both costly REAL and ER NEURON/LFP stages."
    )
    parser.add_argument(
        "--only-m", choices=M_VALUES,
        help="Process only one m condition; default processes both."
    )
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    validate_project_dependencies()
    selected_m = (args.only_m,) if args.only_m else M_VALUES
    for m_value in selected_m:
        env = env_for_m(m_value, args.max_workers)
        root = Path(env["TRIALS_ROOT"])
        print("\n" + "#" * 88)
        print(f"POST-DYNAMICS PIPELINE: m={m_value}")
        print("#" * 88)

        if not all_trial_files_complete(root, BASE_DYNAMICS_FILES):
            run_script(
                "generate_dynamics", env,
                ("--only-m", m_value, "--max-workers", str(args.max_workers)),
            )

        validate_files(root, BASE_DYNAMICS_FILES, "dynamics inputs")

        if all_trial_files_complete(root, SWCS + REAL_MAPS):
            print("[SKIP stage] SWCs and REAL mappings already complete for all 40 trials")
        else:
            run_script("maps_real", env)
        validate_files(root, SWCS + REAL_MAPS, "SWC and REAL mappings")

        if all_trial_files_complete(root, SYNAPSE_SPIKES):
            print("[SKIP stage] REAL/ER synapse spike tables already complete for all 40 trials")
        else:
            run_script("synapse_spikes", env)
        validate_files(root, SYNAPSE_SPIKES, "synapse spike tables")

        # NaN清理识别标记，仅重写新增或变更的文件。 / NaN cleaning is marker-aware and only rewrites files that are new or changed.
        run_script("clean_nan", env)
        validate_files(root, SYNAPSE_SPIKES, "cleaned synapse spike tables")

        if all_trial_files_complete(root, ER_MAPS):
            print("[SKIP stage] ER mappings already complete for all 40 trials")
        else:
            run_script("maps_er", env)
        validate_files(root, ER_MAPS, "ER mappings")

        if args.run_neuron:
            real_save = Path(env["SAVE_ROOT_REAL"])
            er_save = Path(env["SAVE_ROOT_ER"])
            run_neuron_stage_without_overwrite(
                real_save, "neuron_real", "REAL", env
            )
            validate_lfp_tree(real_save, "REAL NEURON/LFP")
            run_neuron_stage_without_overwrite(
                er_save, "neuron_er", "ER", env
            )
            validate_lfp_tree(er_save, "ER NEURON/LFP")
        else:
            print(
                f"[READY m={m_value}] preprocessing complete. "
                "Use --run-neuron to launch the expensive simulations."
            )


if __name__ == "__main__":
    main()
