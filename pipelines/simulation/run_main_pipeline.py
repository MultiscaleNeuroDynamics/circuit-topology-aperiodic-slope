"""主要100试次模拟流程的支持续跑控制器。生成动力学，构建形态和突触输入，并可选启动高成本NEURON/LFP模拟；不覆盖已有完整输出。 / Restart-safe controller for the 100-trial primary simulation pipeline.

The controller generates dynamics, constructs morphology/synapse inputs, and
optionally launches the expensive NEURON/LFP simulations. Existing complete
outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
LAYERS = ("L2_3", "L4", "L5", "L6")
PREPARED_FILES = (
    "23.swc", "4.swc", "5.swc", "6.swc",
    "syn_to_segment_L2_3.csv", "syn_to_segment_L4.csv",
    "syn_to_segment_L5.csv", "syn_to_segment_L6.csv",
    "microns_patch_synapse_spikes_real_ms.csv",
    "microns_patch_synapse_spikes_ER_ms.csv",
    "syn_to_segment_L2_3_ER.csv", "syn_to_segment_L4_ER.csv",
    "syn_to_segment_L5_ER.csv", "syn_to_segment_L6_ER.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work"),
        help="Generated workspace root (default: <repository>/work).",
    )
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--m", type=float, default=0.99)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--run-neuron", action="store_true",
        help="Run the expensive REAL and ER LFP simulations after preprocessing.",
    )
    return parser.parse_args()


def trial_directories(root: Path):
    pattern = re.compile(r"ER_trial(\d+)$")
    result = []
    if root.is_dir():
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match:
                result.append((int(match.group(1)), path))
    return [path for _, path in sorted(result)]


def trial_files_complete(root: Path, n_runs: int, filenames) -> bool:
    trials = trial_directories(root)
    if len(trials) != n_runs:
        return False
    return all(
        all((trial / name).is_file() and (trial / name).stat().st_size > 0 for name in filenames)
        for trial in trials
    )


def dynamics_complete(root: Path, n_runs: int) -> bool:
    required = (
        "microns_patch_spikes_real_ms.csv",
        "microns_patch_spikes_ER_ms.csv",
        "microns_patch_edges_real_weights.csv",
        "microns_patch_edges_ER_weights.csv",
        "microns_patch_edges_ER_IE_weights.csv",
        "summary.txt",
    )
    return trial_files_complete(root, n_runs, required)


def lfp_path(root: Path, trial_name: str, layer: str):
    candidates = (
        root / trial_name / layer / "_lfp.npy",
        root / trial_name / layer / layer / "_lfp.npy",
    )
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def validate_lfp(root: Path, trials: list[Path], label: str):
    missing = [
        f"{trial.name}/{layer}"
        for trial in trials for layer in LAYERS
        if lfp_path(root, trial.name, layer) is None
    ]
    if missing:
        raise RuntimeError(f"[{label}] missing {len(missing)} LFP files; examples: {missing[:10]}")
    print(f"[VALID {label}] {len(trials) * len(LAYERS)} non-empty LFP files")


def run(script: str, env: dict[str, str], *extra: str):
    path = PIPELINE_DIR / script
    print("\n" + "=" * 88)
    print("RUN", path.name, flush=True)
    subprocess.run([sys.executable, str(path), *extra], check=True, env=env, cwd=REPOSITORY_ROOT)


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    allen_root = project_root / "ALLEN"
    trials_root = allen_root / "file" / "final_trial"
    real_root = allen_root / "save" / "final_real"
    er_root = allen_root / "save" / "final_ER"

    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(project_root),
        "ALLEN_ROOT": str(allen_root),
        "TRIALS_ROOT": str(trials_root),
        "PATCH_DIR": str(allen_root / "file" / "patch"),
        "SELECT_SWC_ROOT": str(allen_root / "file" / "select_swc"),
        "PARAM_FILE": str(allen_root / "file" / "swc" / "parameter.json"),
        "SAVE_ROOT_REAL": str(real_root),
        "SAVE_ROOT_ER": str(er_root),
        "MAX_WORKERS": str(args.max_workers),
        "MECHANISM_ROOT": str(REPOSITORY_ROOT / "mechanisms"),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")])
        ),
    })

    if dynamics_complete(trials_root, args.n_runs):
        print(f"[SKIP] all {args.n_runs} primary dynamics trials are complete")
    else:
        existing = trial_directories(trials_root)
        if existing:
            raise RuntimeError(
                "Partial primary dynamics output exists. The generator is intentionally not "
                "started because it could replace partial trial data. Move the partial tree "
                "aside or complete it explicitly before rerunning this controller."
            )
        run(
            "generate_main_trials.py", env,
            "--output-root", str(trials_root),
            "--n-runs", str(args.n_runs),
            "--m", str(args.m),
            "--max-workers", str(args.max_workers),
        )
    if not dynamics_complete(trials_root, args.n_runs):
        raise RuntimeError("Primary dynamics validation failed")

    if not trial_files_complete(trials_root, args.n_runs, PREPARED_FILES):
        run("assign_trial_morphologies.py", env)
        run("expand_soma_spikes_to_synapses.py", env)
        run("clean_spike_tables.py", env)
        run("generate_er_segment_maps.py", env)
    if not trial_files_complete(trials_root, args.n_runs, PREPARED_FILES):
        raise RuntimeError("Morphology and synapse preprocessing validation failed")
    print(f"[VALID preprocessing] {args.n_runs} trials")

    if not args.run_neuron:
        print("[READY] preprocessing is complete; add --run-neuron to generate LFP files")
        return

    trials = trial_directories(trials_root)
    if any(lfp_path(real_root, trial.name, layer) is None for trial in trials for layer in LAYERS):
        run("run_real_lfp.py", env)
    validate_lfp(real_root, trials, "REAL LFP")

    if any(lfp_path(er_root, trial.name, layer) is None for trial in trials for layer in LAYERS):
        run("run_er_lfp.py", env)
    validate_lfp(er_root, trials, "ER LFP")
    print("[ALL COMPLETE] primary 100-trial dynamics-to-LFP pipeline")


if __name__ == "__main__":
    main()
