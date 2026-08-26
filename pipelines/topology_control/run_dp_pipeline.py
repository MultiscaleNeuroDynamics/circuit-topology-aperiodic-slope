# -*- coding: utf-8 -*-
"""One-command, restart-safe m=0.99 DP40 dynamics-to-LFP pipeline.

All DP work is isolated under m_0.99_DP40 and final_DP40. Existing primary
REAL and ER outputs are read-only and are never used as output targets.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PIPELINE_DIR.parents[1]
SIMULATION_DIR = REPOSITORY_ROOT / "pipelines" / "simulation"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")
)
SELECTED_TRIAL_IDS = (
    1, 4, 6, 9, 11, 14, 16, 19, 21, 24,
    26, 29, 31, 34, 37, 39, 42, 44, 47, 49,
    52, 54, 57, 59, 62, 64, 67, 70, 72, 75,
    77, 80, 82, 85, 87, 90, 92, 95, 97, 100,
)

SCRIPTS = {
    "generate_dp": PIPELINE_DIR / "generate_dp_trials.py",
    "maps_real": SIMULATION_DIR / "assign_trial_morphologies.py",
    "synapse_spikes": SIMULATION_DIR / "expand_soma_spikes_to_synapses.py",
    "clean_nan": SIMULATION_DIR / "clean_spike_tables.py",
    "maps_control": SIMULATION_DIR / "generate_er_segment_maps.py",
    "neuron_control": SIMULATION_DIR / "run_er_lfp.py",
}

DP_DYNAMICS_FILES = (
    "microns_patch_spikes_ER_ms.csv",
    "microns_patch_edges_ER_weights.csv",
    "microns_patch_edges_ER_IE_weights.csv",
    "dp_metadata.json",
)
REAL_ASSETS = (
    "23.swc", "4.swc", "5.swc", "6.swc",
    "syn_to_segment_L2_3.csv", "syn_to_segment_L4.csv",
    "syn_to_segment_L5.csv", "syn_to_segment_L6.csv",
    "microns_patch_spikes_real_ms.csv",
)
SYNAPSE_SPIKES = (
    "microns_patch_synapse_spikes_real_ms.csv",
    "microns_patch_synapse_spikes_ER_ms.csv",
)
CONTROL_MAPS = (
    "syn_to_segment_L2_3_ER.csv", "syn_to_segment_L4_ER.csv",
    "syn_to_segment_L5_ER.csv", "syn_to_segment_L6_ER.csv",
)
LAYERS = ("L2_3", "L4", "L5", "L6")


def expected_trial_dirs(root: Path):
    return [root / f"ER_trial{i:04d}" for i in SELECTED_TRIAL_IDS]


def files_complete(root: Path, filenames):
    return all(
        trial.is_dir()
        and all((trial / name).is_file() and (trial / name).stat().st_size > 0 for name in filenames)
        for trial in expected_trial_dirs(root)
    )


def validate_files(root: Path, filenames, label: str):
    missing = []
    empty = []
    for trial in expected_trial_dirs(root):
        for name in filenames:
            path = trial / name
            if not path.is_file():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                empty.append(str(path))
    if missing or empty:
        raise RuntimeError(
            f"[{label}] missing={len(missing)}, empty={len(empty)}; "
            f"examples={(missing + empty)[:8]}"
        )
    print(f"[VALID {label}] 40 source trials, {len(filenames)} files per trial")


def locate_lfp(save_root: Path, trial_id: int, layer: str):
    # The validated legacy script may use either layout depending on version.
    candidates = (
        save_root / f"ER_trial{trial_id:04d}" / layer / "_lfp.npy",
        save_root / f"ER_trial{trial_id:04d}" / layer / layer / "_lfp.npy",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def lfp_complete(save_root: Path):
    return all(
        locate_lfp(save_root, trial_id, layer) is not None
        for trial_id in SELECTED_TRIAL_IDS
        for layer in LAYERS
    )


def validate_lfp(save_root: Path):
    missing = [
        f"ER_trial{trial_id:04d}/{layer}"
        for trial_id in SELECTED_TRIAL_IDS
        for layer in LAYERS
        if locate_lfp(save_root, trial_id, layer) is None
    ]
    if missing:
        raise RuntimeError(f"[DP LFP] missing {len(missing)}/160: {missing[:10]}")
    print(f"[VALID DP LFP] 160 non-empty files under {save_root}")


def run_script(key: str, env, extra_args=()):
    path = Path(SCRIPTS[key])
    if not path.is_file():
        raise FileNotFoundError(path)
    print("\n" + "=" * 88)
    print(f"RUN {path.name}")
    subprocess.run(
        [sys.executable, str(path), *extra_args],
        cwd=str(PIPELINE_DIR),
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--original-trials-root", default=os.environ.get("ORIGINAL_TRIALS_ROOT"))
    parser.add_argument("--run-neuron", action="store_true")
    parser.add_argument(
        "--A-dp", type=float,
        help="Optional explicit override; omit to calibrate DP automatically.",
    )
    parser.add_argument("--initial-A-dp", type=float, default=1.19116)
    parser.add_argument("--nswap-factor", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 1)))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    allen_root = Path(os.environ.get("ALLEN_ROOT", project_root / "ALLEN"))
    trials_root = allen_root / "file" / "add_trial" / "m_0.99_DP40"
    save_root = allen_root / "save" / "m_0.99" / "final_DP40"

    if save_root.resolve() in {
        (allen_root / "save" / "m_0.99" / "final_real").resolve(),
        (allen_root / "save" / "m_0.99" / "final_ER").resolve(),
    }:
        raise RuntimeError("Safety stop: DP output points to a manuscript REAL/ER directory")

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(project_root),
            "ALLEN_ROOT": str(allen_root),
            "TRIALS_ROOT": str(trials_root),
            "DP_TRIALS_ROOT": str(trials_root),
            "PATCH_DIR": str(allen_root / "file" / "patch"),
            "SELECT_SWC_ROOT": str(allen_root / "file" / "select_swc"),
            "PARAM_FILE": str(allen_root / "file" / "swc" / "parameter.json"),
            # Legacy control simulator reads SAVE_ROOT_ER.  It is redirected to
            # the isolated DP directory, never to the manuscript final_ER tree.
            "SAVE_ROOT_ER": str(save_root),
            "MAX_WORKERS": str(args.max_workers),
            "MECHANISM_ROOT": str(REPOSITORY_ROOT / "mechanisms"),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")])
            ),
        }
    )
    if args.original_trials_root:
        env["ORIGINAL_TRIALS_ROOT"] = str(args.original_trials_root)

    if not files_complete(trials_root, DP_DYNAMICS_FILES):
        generator_args = [
            "--project-root", str(project_root),
            "--initial-A-dp", str(args.initial_A_dp),
            "--nswap-factor", str(args.nswap_factor),
            "--max-workers", str(args.max_workers),
        ]
        if args.A_dp is not None:
            generator_args.extend(["--A-dp", str(args.A_dp)])
        run_script(
            "generate_dp",
            env,
            extra_args=tuple(generator_args),
        )
    else:
        print("[SKIP] all 40 DP dynamics outputs already exist")
    validate_files(trials_root, DP_DYNAMICS_FILES, "DP dynamics")

    if not files_complete(trials_root, REAL_ASSETS):
        run_script("maps_real", env)
    validate_files(trials_root, REAL_ASSETS, "SWC/REAL compatibility assets")

    if not files_complete(trials_root, SYNAPSE_SPIKES):
        run_script("synapse_spikes", env)
    validate_files(trials_root, SYNAPSE_SPIKES, "DP synapse spikes")

    run_script("clean_nan", env)
    validate_files(trials_root, SYNAPSE_SPIKES, "clean DP synapse spikes")

    if not files_complete(trials_root, CONTROL_MAPS):
        run_script("maps_control", env)
    validate_files(trials_root, CONTROL_MAPS, "DP synapse-to-segment maps")

    if not args.run_neuron:
        print(
            "[READY] DP dynamics and preprocessing complete. "
            "Run again with --run-neuron to produce 160 LFP files."
        )
        return

    if lfp_complete(save_root):
        print("[SKIP] all 160 DP _lfp.npy files already exist")
    else:
        # The worker-level code skips each existing _lfp.npy.  Therefore an
        # interrupted partial DP run resumes without overwriting completed LFPs.
        run_script("neuron_control", env)
    validate_lfp(save_root)
    print(f"[ALL COMPLETE] m=0.99 DP40 LFP: {save_root}")


if __name__ == "__main__":
    main()
