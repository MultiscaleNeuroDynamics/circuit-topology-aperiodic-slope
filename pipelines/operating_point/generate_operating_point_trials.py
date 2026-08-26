#!/usr/bin/env python3
"""Generate calibrated 40-trial robustness datasets at m=0.90 and m=0.95.

The event-driven dynamics, MICrONS matrix construction, ER control, and trial
writer are imported from the primary generator so all operating points share
one authoritative implementation.  This module adds only operating-point gain
calibration and condition-level scheduling.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pipelines.simulation.generate_main_trials import (
    ALLEN_ROOT,
    build_ER_control_uniform_synapses,
    build_microns_EE_EI_IE_matrices,
    matrix_to_digraph,
    run_one_trial,
    simulate_event_driven_branching,
    spectral_radius,
)


def _worker_run_one_trial(kwargs):
    """Execute one shared dynamics trial in a process-pool worker."""
    return run_one_trial(**kwargs)


def calibrate_A_for_target_m(
        G, rho, target_m, initial_A, name,
        sim_kwargs, seeds,
        calibration_tol=0.002,
        max_iterations=12,
        initial_relative_span=0.20,
        max_expansions=6):
    """
    Calibrate A in rho_eff=rho/A so that the median measured m_eff is close
    to target_m. Candidate A values are evaluated with identical fixed seeds.

    For safety, the search is strictly one-sided: A is allowed to decrease
    from the original value but is never allowed to exceed initial_A. Because
    m_eff normally increases monotonically with A, bisection is used after a
    valid bracket is found. The complete search history is returned.
    """
    seeds = [int(s) for s in seeds]
    history = []
    cache = {}

    def evaluate(A):
        A = float(A)
        key = round(A, 12)
        if key in cache:
            return cache[key]

        values = []
        ext_values = []
        rho_eff = float(rho) / A
        for seed in seeds:
            _, meff, ext_total = simulate_event_driven_branching(
                G, rho=rho_eff, m0=float(target_m),
                name=f"CAL_{name}_A{A:.8f}_seed{seed}",
                seed=seed, **sim_kwargs
            )
            if np.isfinite(meff) and meff > 0:
                values.append(float(meff))
            ext_values.append(int(ext_total))

        if not values:
            median_meff = np.nan
            mean_meff = np.nan
            sd_meff = np.nan
        else:
            median_meff = float(np.median(values))
            mean_meff = float(np.mean(values))
            sd_meff = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        row = {
            "network": name,
            "target_m": float(target_m),
            "A": A,
            "rho": float(rho),
            "rho_eff": rho_eff,
            "n_seeds": len(values),
            "median_meff": median_meff,
            "mean_meff": mean_meff,
            "sd_meff": sd_meff,
            "abs_error": abs(median_meff - target_m) if np.isfinite(median_meff) else np.inf,
            "mean_external_inputs": float(np.mean(ext_values)) if ext_values else np.nan,
        }
        history.append(row)
        cache[key] = row
        print(
            f"[CAL {name}] target={target_m:.3f}, A={A:.8f}, "
            f"median m_eff={median_meff:.6f}, error={row['abs_error']:.6f}"
        )
        return row

    # Safety constraint: never increase A above the value calibrated for m=0.99.
    low_A = max(1e-8, float(initial_A) * (1.0 - initial_relative_span))
    high_A = float(initial_A)
    low = evaluate(low_A)
    high = evaluate(high_A)

    # If needed, expand downward only. Never increase above initial_A because
    # a larger A increases g_branch=m0*A/rho and may approach supercriticality.
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
                f"[CAL {name}] target cannot be reached by decreasing A: "
                f"m_eff at initial A={initial_A:.8f} is {high_y:.6f}, "
                f"below target {target_m:.6f}. No upward search will be attempted."
            )
            break
        else:
            break
        expansions += 1

    low_y = low["median_meff"]
    high_y = high["median_meff"]
    bracketed = (
        np.isfinite(low_y) and np.isfinite(high_y)
        and low_y <= target_m <= high_y
    )

    if bracketed:
        for _ in range(max_iterations):
            mid_A = 0.5 * (low["A"] + high["A"])
            mid = evaluate(mid_A)
            if mid["abs_error"] <= calibration_tol:
                break
            if mid["median_meff"] < target_m:
                low = mid
            else:
                high = mid
    else:
        print(
            f"[CAL {name}] WARNING: target m was not bracketed; "
            "the closest tested A not exceeding the original value will be used."
        )

    finite_history = [h for h in history if np.isfinite(h["abs_error"])]
    if not finite_history:
        raise RuntimeError(f"Calibration failed for {name}: no finite m_eff")
    best = min(finite_history, key=lambda x: x["abs_error"])
    if best["A"] > float(initial_A) + 1e-12:
        raise RuntimeError(
            f"Unsafe calibration result for {name}: A={best['A']} exceeds "
            f"initial_A={initial_A}"
        )
    print(
        f"[CAL {name}] selected A={best['A']:.8f}, "
        f"median m_eff={best['median_meff']:.6f}, "
        f"target={target_m:.6f}"
    )
    return float(best["A"]), pd.DataFrame(history)


def calibrate_real_and_er_A(target_m, initial_Areal, initial_Aer,
                            calibration_dir,
                            tmax_sec, dt, warmup_time,
                            refractory_period, external_rate,
                            syn_delay_mean):
    """Calibrate REAL and ER A values on fixed representative networks."""
    os.makedirs(calibration_dir, exist_ok=True)

    W_EE_real, _, _, _, _ = build_microns_EE_EI_IE_matrices()
    W_er = build_ER_control_uniform_synapses(W_EE_real, seed=10001)
    rho_real, _ = spectral_radius(W_EE_real, name="CAL_MICRONS_EE")
    rho_er, _ = spectral_radius(W_er, name="CAL_ER_EE")
    G_real = matrix_to_digraph(W_EE_real)
    G_er = matrix_to_digraph(W_er)

    sim_kwargs = dict(
        tmax_sec=tmax_sec,
        dt=dt,
        warmup_time=warmup_time,
        refractory_period=refractory_period,
        external_rate=external_rate,
        syn_delay_mean=syn_delay_mean,
    )

    # Fixed seeds make every candidate A directly comparable.
    real_seeds = [91001, 91002, 91003, 91004, 91005]
    er_seeds = [92001, 92002, 92003, 92004, 92005]

    Areal, hist_real = calibrate_A_for_target_m(
        G_real, rho_real, target_m, initial_Areal, "REAL",
        sim_kwargs, real_seeds
    )
    Aer, hist_er = calibrate_A_for_target_m(
        G_er, rho_er, target_m, initial_Aer, "ER",
        sim_kwargs, er_seeds
    )

    history = pd.concat([hist_real, hist_er], ignore_index=True)
    history.to_csv(
        os.path.join(calibration_dir, f"A_calibration_history_m{target_m:.2f}.csv"),
        index=False, encoding="utf-8-sig"
    )
    selected = pd.DataFrame([{
        "target_m": float(target_m),
        "Areal_initial": float(initial_Areal),
        "Aer_initial": float(initial_Aer),
        "Areal_calibrated": float(Areal),
        "Aer_calibrated": float(Aer),
        "calibration_seeds_per_network": 5,
        "calibration_target_tolerance": 0.002,
    }])
    selected.to_csv(
        os.path.join(calibration_dir, f"A_calibrated_values_m{target_m:.2f}.csv"),
        index=False, encoding="utf-8-sig"
    )
    return Areal, Aer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate m=0.90 and m=0.95 operating-point trials.")
    parser.add_argument("--output-root", type=Path, default=ALLEN_ROOT / "file" / "add_trial")
    parser.add_argument("--only-m", type=float, choices=(0.90, 0.95))
    parser.add_argument("--n-runs", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", max(1, (os.cpu_count() or 4) - 1))))
    args = parser.parse_args()
    base_out_root = str(args.output_root.resolve())

    # Sensitivity conditions: m=0.90 and m=0.95, 40 trials per condition.
    # The same 40 evenly spaced E/I ratios are used for both m conditions.
    M_VALUES = [float(args.only_m)] if args.only_m is not None else [0.90, 0.95]
    N_RUNS = int(args.n_runs)
    INITIAL_AREAL = 1.19116
    INITIAL_AER = 1.030544
    EI_min = 1.0 / 3.0  # 1:3
    EI_max = 3.0  # 3:1

    EI_ratios = np.linspace(EI_min, EI_max, N_RUNS, dtype=float)

    # Configuration.
    USE_THREADS = False
    max_workers = int(args.max_workers)
    Executor = ThreadPoolExecutor if USE_THREADS else ProcessPoolExecutor

    for m_value in M_VALUES:
        # Separate directories prevent m=0.90 and m=0.95 from overwriting
        # each other's per-trial files and summary tables.
        m_tag = f"m_{m_value:.2f}"
        out_root = os.path.join(base_out_root, m_tag)
        os.makedirs(out_root, exist_ok=True)

        required = (
            "microns_patch_spikes_real_ms.csv",
            "microns_patch_spikes_ER_ms.csv",
            "microns_patch_edges_ER_weights.csv",
            "microns_patch_edges_ER_IE_weights.csv",
        )
        trial_dirs = [os.path.join(out_root, f"ER_trial{k:04d}") for k in range(1, N_RUNS + 1)]
        complete = all(
            os.path.isdir(folder)
            and all(os.path.isfile(os.path.join(folder, name)) and os.path.getsize(os.path.join(folder, name)) > 0 for name in required)
            for folder in trial_dirs
        )
        if complete:
            print(f"[SKIP] all {N_RUNS} dynamics trials for m={m_value:.2f} are complete")
            continue
        existing = [folder for folder in trial_dirs if os.path.isdir(folder)]
        if existing:
            raise RuntimeError(
                f"Partial m={m_value:.2f} dynamics output exists under {out_root}. "
                "Generation stopped to avoid replacing completed trials."
            )

        print("\n" + "=" * 80)
        print(f"Starting {N_RUNS} trials for m0={m_value:.2f}")
        print(f"Output directory: {out_root}")

        # Stage 1: recalibrate the two hyperparameters for this target m.
        # Formal 40-trial generation starts only after calibration finishes.
        calibration_dir = os.path.join(out_root, "A_calibration")
        Areal_calibrated, Aer_calibrated = calibrate_real_and_er_A(
            target_m=float(m_value),
            initial_Areal=INITIAL_AREAL,
            initial_Aer=INITIAL_AER,
            calibration_dir=calibration_dir,
            tmax_sec=22.0,
            dt=0.001,
            warmup_time=2.0,
            refractory_period=0.001,
            external_rate=1,
            syn_delay_mean=0.001,
        )
        print(
            f"[CALIBRATED m0={m_value:.2f}] "
            f"Areal={Areal_calibrated:.8f}, Aer={Aer_calibrated:.8f}"
        )

        results = [None] * N_RUNS
        completed = 0

        partial_path = os.path.join(
            out_root, f"trial_summary_m{m_value:.2f}_{N_RUNS}runs_partial.csv"
        )
        final_path = os.path.join(
            out_root, f"trial_summary_m{m_value:.2f}_{N_RUNS}runs.csv"
        )

        futures = []
        with Executor(max_workers=max_workers) as ex:
            for k in range(1, N_RUNS + 1):
                # Keep the same E/I value at trial k for both m conditions.
                ei_local = float(EI_ratios[k - 1])

                kwargs = dict(
                    trial_idx=int(k),
                    base_out_dir=out_root,
                    m0=float(m_value),
                    tol=5e-4,
                    max_trials=200,
                    tmax_sec=22.0,
                    dt=0.001,
                    warmup_time=2.0,
                    refractory_period=0.001,
                    external_rate=1,
                    syn_delay_mean=0.001,
                    Areal=float(Areal_calibrated),
                    Aer=float(Aer_calibrated),
                    EI_ratio=ei_local,
                    show_plot=False
                )

                futures.append(ex.submit(_worker_run_one_trial, kwargs))

            for fut in as_completed(futures):
                row = fut.result()

                if isinstance(row, dict) and "trial_idx" in row:
                    idx = int(row["trial_idx"]) - 1
                    if 0 <= idx < N_RUNS:
                        results[idx] = row
                else:
                    for j in range(N_RUNS):
                        if results[j] is None:
                            results[j] = row
                            break

                completed += 1

                if (completed % 20) == 0:
                    df_tmp = pd.DataFrame([r for r in results if r is not None])
                    if "trial_idx" in df_tmp.columns:
                        df_tmp = df_tmp.sort_values("trial_idx")
                    df_tmp.insert(1, "m0_requested", float(m_value))
                    df_tmp.to_csv(partial_path, index=False, encoding="utf-8-sig")
                    print(f"[partial m0={m_value:.2f}] {completed}/{N_RUNS} saved -> {partial_path}")

        df = pd.DataFrame([r for r in results if r is not None])
        if "trial_idx" in df.columns:
            df = df.sort_values("trial_idx")
        df.insert(1, "m0_requested", float(m_value))
        df.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\n[summary m0={m_value:.2f}] saved -> {final_path}")
