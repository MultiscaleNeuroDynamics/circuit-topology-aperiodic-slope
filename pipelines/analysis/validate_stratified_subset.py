# -*- coding: utf-8 -*-
"""
E/I-stratified subset analysis for REAL versus ER topology.

Purpose
-------
1. Use all available paired trials as the primary analysis.
2. Select a fixed, reproducible subset of 40 paired trials spanning the full
   E/I range, for planning computationally expensive sensitivity simulations.
3. Repeat stratified sampling many times to evaluate whether conclusions based
   on 40 trials are stable rather than dependent on one favourable subset.

Models
------
M1: beta ~ EI_ratio
M2: beta ~ topology
M3: beta ~ EI_ratio + topology
M4: beta ~ EI_ratio * topology

REAL is the reference topology. Standard errors and 95% confidence intervals
are clustered by trial because REAL and ER observations from the same trial
form a pair.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================================
# 1. User settings
# ============================================================================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
RESULTS_ROOT = PROJECT_ROOT / "ALLEN" / "results"
EI_CSV = str(RESULTS_ROOT / "ei_ratio_stats" / "EI_ratio_per_trial_REAL_vs_ER.csv")
EXP_CSV = str(RESULTS_ROOT / "fooof_group_compare" / "fooof_exponent_REAL_and_ER_two_columns.csv")
OUT_DIR = str(RESULTS_ROOT / "stratified_subset_validation")

N_SELECT = 40
N_RESAMPLES = 1000
RANDOM_SEED = 20260813
ALPHA = 0.05


MODEL_SPECS = [
    ("M1_EI_only", "beta ~ EI_ratio"),
    ("M2_topology_only", "beta ~ C(topology)"),
    ("M3_EI_plus_topology", "beta ~ EI_ratio + C(topology)"),
    ("M4_interaction", "beta ~ EI_ratio * C(topology)"),
]

TOPOLOGY_TERM = "C(topology)[T.ER]"
INTERACTION_TERM = "EI_ratio:C(topology)[T.ER]"


# ============================================================================
# 2. Read and strictly align paired trial data
# ============================================================================
def load_paired_data(ei_csv, exp_csv):
    df_ei = pd.read_csv(ei_csv)
    df_exp = pd.read_csv(exp_csv)

    required_ei = ["REAL_EI_ratio", "ER_EI_ratio"]
    required_exp = ["REAL_exponent", "ER_exponent"]
    missing = [c for c in required_ei if c not in df_ei.columns]
    missing += [c for c in required_exp if c not in df_exp.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Prefer an explicit trial-key merge. If neither table has a trial column,
    # row alignment is used and clearly recorded by a generated trial ID.
    if "trial" in df_ei.columns and "trial" in df_exp.columns:
        if df_ei["trial"].astype(str).duplicated().any():
            raise RuntimeError("Duplicate trial IDs found in the E/I table")
        if df_exp["trial"].astype(str).duplicated().any():
            raise RuntimeError("Duplicate trial IDs found in the exponent table")

        left = df_ei[["trial"] + required_ei].copy()
        right = df_exp[["trial"] + required_exp].copy()
        left["trial"] = left["trial"].astype(str)
        right["trial"] = right["trial"].astype(str)

        only_ei = sorted(set(left["trial"]) - set(right["trial"]))
        only_exp = sorted(set(right["trial"]) - set(left["trial"]))
        if only_ei or only_exp:
            raise RuntimeError(
                "Trial IDs do not match between files. "
                f"Only in EI table: {only_ei[:10]}; "
                f"only in exponent table: {only_exp[:10]}"
            )
        paired = left.merge(right, on="trial", how="inner", validate="one_to_one")
    else:
        if len(df_ei) != len(df_exp):
            raise RuntimeError(
                f"Row mismatch without a shared trial column: "
                f"EI={len(df_ei)}, exponent={len(df_exp)}"
            )
        warnings.warn(
            "No shared trial column was found. The two input files are assumed "
            "to have exactly the same row order.",
            RuntimeWarning,
        )
        paired = pd.DataFrame({
            "trial": [f"trial_{i:04d}" for i in range(len(df_ei))],
            "REAL_EI_ratio": df_ei["REAL_EI_ratio"].values,
            "ER_EI_ratio": df_ei["ER_EI_ratio"].values,
            "REAL_exponent": df_exp["REAL_exponent"].values,
            "ER_exponent": df_exp["ER_exponent"].values,
        })

    numeric_cols = required_ei + required_exp
    for col in numeric_cols:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")

    valid = np.isfinite(paired[numeric_cols]).all(axis=1)
    n_removed = int((~valid).sum())
    if n_removed:
        warnings.warn(f"Removed {n_removed} trials containing missing/non-finite values")
    paired = paired.loc[valid].copy().reset_index(drop=True)

    if len(paired) < N_SELECT:
        raise RuntimeError(
            f"Only {len(paired)} complete paired trials remain; "
            f"cannot select {N_SELECT}."
        )

    # A topology-neutral value used only for stratification/selection.
    paired["mean_EI"] = (
        paired["REAL_EI_ratio"] + paired["ER_EI_ratio"]
    ) / 2.0
    return paired


def paired_to_long(paired):
    real = paired[["trial", "REAL_EI_ratio", "REAL_exponent"]].rename(
        columns={"REAL_EI_ratio": "EI_ratio", "REAL_exponent": "beta"}
    )
    real["topology"] = "REAL"

    er = paired[["trial", "ER_EI_ratio", "ER_exponent"]].rename(
        columns={"ER_EI_ratio": "EI_ratio", "ER_exponent": "beta"}
    )
    er["topology"] = "ER"

    long_df = pd.concat([real, er], ignore_index=True)
    long_df["trial"] = long_df["trial"].astype(str)
    long_df["topology"] = pd.Categorical(
        long_df["topology"], categories=["REAL", "ER"]
    )
    return long_df[["trial", "topology", "EI_ratio", "beta"]]


# ============================================================================
# 3. Prespecified fixed selection: one representative per E/I stratum
# ============================================================================
def assign_equal_count_strata(paired, n_strata):
    """Assign nearly equal-sized rank strata without qcut duplicate-edge issues."""
    ranked = paired.sort_values(["mean_EI", "trial"]).copy().reset_index(drop=True)
    # array_split guarantees that all trials are assigned and group sizes differ
    # by no more than one.
    index_groups = np.array_split(np.arange(len(ranked)), n_strata)
    ranked["EI_stratum"] = -1
    for stratum, indices in enumerate(index_groups, start=1):
        ranked.loc[indices, "EI_stratum"] = stratum
    ranked["EI_stratum"] = ranked["EI_stratum"].astype(int)
    return ranked


def select_fixed_representatives(stratified):
    """Select the observation closest to each stratum's median mean_EI."""
    chosen_rows = []
    for stratum, group in stratified.groupby("EI_stratum", sort=True):
        target = group["mean_EI"].median()
        candidates = group.assign(distance_to_stratum_median=(group["mean_EI"] - target).abs())
        # Sorting by trial provides deterministic tie breaking.
        selected = candidates.sort_values(
            ["distance_to_stratum_median", "trial"]
        ).iloc[0]
        chosen_rows.append(selected)

    selected = pd.DataFrame(chosen_rows).sort_values("EI_stratum").reset_index(drop=True)
    selected["selection_rule"] = "nearest_to_stratum_median"
    return selected


# ============================================================================
# 4. Cluster-robust regressions with 95% confidence intervals
# ============================================================================
def fit_one_model(formula, long_df):
    ordinary = smf.ols(formula=formula, data=long_df).fit()
    robust = ordinary.get_robustcov_results(
        cov_type="cluster",
        groups=long_df["trial"],
        use_correction=True,
        df_correction=True,
    )
    return ordinary, robust


def coefficient_table(robust, model_name, alpha=ALPHA):
    ci = np.asarray(robust.conf_int(alpha=alpha))
    return pd.DataFrame({
        "model": model_name,
        "term": robust.model.exog_names,
        "coef": robust.params,
        "se_cluster": robust.bse,
        "ci95_low": ci[:, 0],
        "ci95_high": ci[:, 1],
        "t": robust.tvalues,
        "p": robust.pvalues,
        "significant_0.05": np.asarray(robust.pvalues) < 0.05,
    })


def run_model_set(long_df, analysis_name):
    all_coef = []
    fit_rows = []

    for model_name, formula in MODEL_SPECS:
        ordinary, robust = fit_one_model(formula, long_df)
        tab = coefficient_table(robust, model_name)
        tab.insert(0, "analysis", analysis_name)
        all_coef.append(tab)

        fit_rows.append({
            "analysis": analysis_name,
            "model": model_name,
            "formula": formula,
            "n_obs": int(robust.nobs),
            "n_trials": int(long_df["trial"].nunique()),
            "r2": float(ordinary.rsquared),
            "adj_r2": float(ordinary.rsquared_adj),
            "aic": float(ordinary.aic),
            "bic": float(ordinary.bic),
        })

    return pd.concat(all_coef, ignore_index=True), pd.DataFrame(fit_rows)


def extract_term(coef_df, model, term):
    row = coef_df[(coef_df["model"] == model) & (coef_df["term"] == term)]
    if len(row) != 1:
        return None
    return row.iloc[0]


def topology_attenuation(coef_df, analysis_name):
    """Describe change in the absolute topology coefficient from M2 to M3."""
    m2 = extract_term(coef_df, "M2_topology_only", TOPOLOGY_TERM)
    m3 = extract_term(coef_df, "M3_EI_plus_topology", TOPOLOGY_TERM)
    if m2 is None or m3 is None:
        return None

    if np.isclose(m2["coef"], 0.0):
        attenuation = np.nan
    else:
        attenuation = 100.0 * (abs(m2["coef"]) - abs(m3["coef"])) / abs(m2["coef"])

    return {
        "analysis": analysis_name,
        "M2_topology_coef": m2["coef"],
        "M2_ci95_low": m2["ci95_low"],
        "M2_ci95_high": m2["ci95_high"],
        "M2_p": m2["p"],
        "M3_adjusted_topology_coef": m3["coef"],
        "M3_ci95_low": m3["ci95_low"],
        "M3_ci95_high": m3["ci95_high"],
        "M3_p": m3["p"],
        "absolute_coef_attenuation_percent": attenuation,
    }


# ============================================================================
# 5. Repeated stratified sampling of 40 paired trials
# ============================================================================
TRACKED_TERMS = [
    ("M1_EI_only", "EI_ratio", "M1_EI"),
    ("M2_topology_only", TOPOLOGY_TERM, "M2_topology"),
    ("M3_EI_plus_topology", "EI_ratio", "M3_EI"),
    ("M3_EI_plus_topology", TOPOLOGY_TERM, "M3_topology"),
    ("M4_interaction", INTERACTION_TERM, "M4_interaction"),
]


def repeated_stratified_analysis(stratified, n_resamples, seed):
    rng = np.random.default_rng(seed)
    result_rows = []
    selection_rows = []

    grouped_indices = [
        group.index.to_numpy()
        for _, group in stratified.groupby("EI_stratum", sort=True)
    ]

    for repetition in range(1, n_resamples + 1):
        selected_indices = [rng.choice(indices) for indices in grouped_indices]
        sample = stratified.loc[selected_indices].copy()
        long_sample = paired_to_long(sample)

        try:
            coef_df, _ = run_model_set(long_sample, f"resample_{repetition:04d}")
        except Exception as exc:
            warnings.warn(f"Resample {repetition} failed: {exc}")
            continue

        for _, selected_row in sample.iterrows():
            selection_rows.append({
                "resample": repetition,
                "EI_stratum": int(selected_row["EI_stratum"]),
                "trial": selected_row["trial"],
                "mean_EI": selected_row["mean_EI"],
            })

        for model, term, label in TRACKED_TERMS:
            row = extract_term(coef_df, model, term)
            if row is None:
                continue
            result_rows.append({
                "resample": repetition,
                "effect": label,
                "model": model,
                "term": term,
                "coef": row["coef"],
                "se_cluster": row["se_cluster"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "p": row["p"],
                "significant_0.05": bool(row["p"] < 0.05),
                "positive": bool(row["coef"] > 0),
            })

    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    if results.empty:
        raise RuntimeError("All repeated stratified analyses failed")
    return results, selections


def summarize_resamples(results, full_coef):
    full_lookup = {}
    for model, term, label in TRACKED_TERMS:
        row = extract_term(full_coef, model, term)
        if row is not None:
            full_lookup[label] = float(row["coef"])

    summaries = []
    for effect, group in results.groupby("effect", sort=False):
        full_value = full_lookup.get(effect, np.nan)
        if np.isfinite(full_value) and not np.isclose(full_value, 0.0):
            same_direction = np.mean(np.sign(group["coef"]) == np.sign(full_value))
        else:
            same_direction = np.nan

        summaries.append({
            "effect": effect,
            "model": group["model"].iloc[0],
            "term": group["term"].iloc[0],
            "n_successful_resamples": len(group),
            "full_100_trial_coef": full_value,
            "mean_subset_coef": group["coef"].mean(),
            "median_subset_coef": group["coef"].median(),
            "subset_coef_q025": group["coef"].quantile(0.025),
            "subset_coef_q975": group["coef"].quantile(0.975),
            "same_direction_as_full_proportion": same_direction,
            "p_below_0.05_proportion": group["significant_0.05"].mean(),
            "median_p": group["p"].median(),
            "median_ci_width": (group["ci95_high"] - group["ci95_low"]).median(),
        })
    return pd.DataFrame(summaries)


# ============================================================================
# 6. Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paired = load_paired_data(EI_CSV, EXP_CSV)
    stratified = assign_equal_count_strata(paired, N_SELECT)
    fixed20 = select_fixed_representatives(stratified)

    full_long = paired_to_long(paired)
    fixed20_long = paired_to_long(fixed20)

    # Save source/selection tables before fitting models.
    paired.to_csv(
        os.path.join(OUT_DIR, "all_paired_trials_with_mean_EI.csv"),
        index=False, encoding="utf-8-sig"
    )
    stratified.to_csv(
        os.path.join(OUT_DIR, "all_trials_with_EI_strata.csv"),
        index=False, encoding="utf-8-sig"
    )
    fixed20.to_csv(
        os.path.join(OUT_DIR, "selected_40_trials.csv"),
        index=False, encoding="utf-8-sig"
    )
    fixed20_long.to_csv(
        os.path.join(OUT_DIR, "selected_40_long_table.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Primary full-data analysis and fixed 20-trial reference analysis.
    full_coef, full_fit = run_model_set(full_long, "all_trials_primary")
    sub_coef, sub_fit = run_model_set(fixed20_long, "fixed_stratified_20")
    coefficient_results = pd.concat([full_coef, sub_coef], ignore_index=True)
    fit_results = pd.concat([full_fit, sub_fit], ignore_index=True)

    coefficient_results.to_csv(
        os.path.join(OUT_DIR, "full_vs_fixed20_coefficients_with_CI.csv"),
        index=False, encoding="utf-8-sig"
    )
    fit_results.to_csv(
        os.path.join(OUT_DIR, "full_vs_fixed20_model_fit.csv"),
        index=False, encoding="utf-8-sig"
    )

    attenuation_rows = []
    for coef, name in [
        (full_coef, "all_trials_primary"),
        (sub_coef, "fixed_stratified_20"),
    ]:
        row = topology_attenuation(coef, name)
        if row is not None:
            attenuation_rows.append(row)
    pd.DataFrame(attenuation_rows).to_csv(
        os.path.join(OUT_DIR, "topology_effect_before_after_EI_adjustment.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Repeated sampling assesses stability; it does not search for a significant subset.
    resample_results, resample_selections = repeated_stratified_analysis(
        stratified, N_RESAMPLES, RANDOM_SEED
    )
    resample_summary = summarize_resamples(resample_results, full_coef)

    resample_results.to_csv(
        os.path.join(OUT_DIR, "stratified40_resample_all_results.csv"),
        index=False, encoding="utf-8-sig"
    )
    resample_selections.to_csv(
        os.path.join(OUT_DIR, "stratified40_resample_selected_trials.csv"),
        index=False, encoding="utf-8-sig"
    )
    resample_summary.to_csv(
        os.path.join(OUT_DIR, "stratified40_stability_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 88)
    print("ANALYSIS COMPLETED")
    print(f"Complete paired trials: {len(paired)}")
    print(f"Fixed E/I-stratified subset: {len(fixed20)} trials")
    print(f"Repeated stratified samples requested: {N_RESAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {OUT_DIR}")

    print(f"\nSelected {N_SELECT} trials (one representative from each E/I stratum):")
    print(
        fixed20[["EI_stratum", "trial", "mean_EI", "REAL_EI_ratio", "ER_EI_ratio"]]
        .to_string(index=False)
    )

    print("\nKey coefficients: all trials versus fixed stratified 20")
    key_mask = (
        ((coefficient_results["model"] == "M1_EI_only") &
         (coefficient_results["term"] == "EI_ratio")) |
        ((coefficient_results["model"] == "M2_topology_only") &
         (coefficient_results["term"] == TOPOLOGY_TERM)) |
        ((coefficient_results["model"] == "M3_EI_plus_topology") &
         (coefficient_results["term"].isin(["EI_ratio", TOPOLOGY_TERM]))) |
        ((coefficient_results["model"] == "M4_interaction") &
         (coefficient_results["term"] == INTERACTION_TERM))
    )
    print(
        coefficient_results.loc[
            key_mask,
            ["analysis", "model", "term", "coef", "ci95_low", "ci95_high", "p"]
        ].to_string(index=False)
    )

    print("\nRepeated 20-trial sampling stability summary:")
    print(resample_summary.to_string(index=False))
    print("\nImportant: use the 100-trial analysis for formal inference. The 20-trial")
    print("subset is intended to plan expensive m/topology/physiology sensitivity runs.")


if __name__ == "__main__":
    main()
