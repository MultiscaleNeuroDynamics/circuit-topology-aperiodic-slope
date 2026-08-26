# -*- coding: utf-8 -*-
"""Run matched-channel, animal-level propofol effect analyses.

The analysis residualizes prespecified state covariates without including
region, balances cortical and thalamic channel counts by subsampling, and uses
exact animal-level sign-flip inference for the regional contrast.
"""

from pathlib import Path
import itertools
import json
import os
import gc

# Reduce native-library instability on Windows / old conda stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Use non-interactive backend before importing pyplot.
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel


# ============================================================
# Paths and configuration.
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ROOT = Path(os.environ.get("PIG_DATA_ROOT", REPOSITORY_ROOT / "data" / "external" / "ds003380")).resolve()
RESULTS_ROOT = Path(os.environ.get(
    "PIG_RESULTS_ROOT",
    PROJECT_ROOT / "ALLEN" / "results" / "empirical_propofol",
)).resolve()
IN_DIR = RESULTS_ROOT / "step2_features"
OUT_DIR = RESULTS_ROOT / "step3_statistics"
OUT_DIR.mkdir(exist_ok=True, parents=True)

INPUT_CSV = IN_DIR / "05_epoch_channel_features_fooof_with_delta_Ising.csv"

MAIN_MODEL_NAME = "R7_MAIN_PC123_baseline_IsingBetaC_signalQC_matchedChannels"

STATE_ORDER = ["propofol_early", "propofol_late"]


# ============================================================
# ============================================================

APPLY_SIGNAL_CHANNEL_QC = True

SIGNAL_QC_LOWER_QUANTILE_BY_REGION = 0.02
MIN_CHANNEL_SIGNAL_PASS_FRACTION = 0.70

APPLY_FOOOF_R2_HARD_QC = True
MIN_FOOOF_R2 = 0.80

CHANNEL_COL_CANDIDATES = [
    "channel", "channel_id", "channel_num", "chan", "ch", "ch_num",
    "edf_channel", "channel_name", "chan_name", "electrode", "electrode_id",
]

# Matched-channel subsampling controls the different numbers of cortical and thalamic channels
# without putting channel count into the residualization model. For each animal/state/draw,
# the same number of cortex and thalamus channels is sampled, using k = min(n_ctx, n_thal).
MATCHED_CHANNEL_SUBSAMPLING = True

# Fast deterministic matched-channel inference.
# The animal-level draw-averaged value under uniform matched-channel subsampling is
# mathematically equal to the equal-channel mean within each region. This avoids
# materializing 5000 draw × animal tables for every model, which can crash old
# Windows/conda stacks with exit code 0xC0000005.
FAST_DRAW_AVERAGED_MATCHING = True

# Only used if FAST_DRAW_AVERAGED_MATCHING = False.
N_MATCHED_CHANNEL_DRAWS = 5000
MATCHED_CHANNEL_RANDOM_SEED = 20260630
MIN_MATCHED_CHANNELS_PER_REGION = 1

# Plotting many model-level figures can also trigger native backend crashes.
# Main CSV outputs are unaffected. Keep only the main-model plots by default.
SAVE_PER_MODEL_DIAGNOSTIC_PLOTS = False
SAVE_MAIN_MODEL_DIAGNOSTIC_PLOTS = True



# ============================================================
# ============================================================

def exact_signflip_test(diffs):
    """
    Exact two-sided sign-flip permutation.

    Null:
        animal-level paired differences are exchangeable under sign flip.

    Input:
        diffs = thalamus residual - cortex residual, one value per animal.

    Return:
        observed mean difference, exact two-sided p, n animals.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)

    if n == 0:
        return np.nan, np.nan, 0

    obs = float(np.mean(diffs))

    if n <= 20:
        vals = []
        for signs in itertools.product([-1, 1], repeat=n):
            signs = np.asarray(signs, dtype=float)
            vals.append(np.mean(signs * diffs))
        vals = np.asarray(vals)
        p = (np.sum(np.abs(vals) >= abs(obs)) + 1.0) / (len(vals) + 1.0)
        return obs, float(p), int(n)

    rng = np.random.default_rng(123)
    vals = []
    for _ in range(20000):
        signs = rng.choice([-1, 1], size=n)
        vals.append(np.mean(signs * diffs))
    vals = np.asarray(vals)
    p = (np.sum(np.abs(vals) >= abs(obs)) + 1.0) / (len(vals) + 1.0)
    return obs, float(p), int(n)


def zscore_safe(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)

    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x, dtype=float)

    return (x - m) / s


def prepare_numeric_covariates(df, covariates):
    valid = []

    for c in covariates:
        if c not in df.columns:
            continue

        x = pd.to_numeric(df[c], errors="coerce").values
        finite = np.isfinite(x)

        if finite.sum() < 10:
            continue

        if np.nanstd(x[finite]) <= 1e-12:
            continue

        valid.append(c)

    return valid




def first_existing_column(df, candidates):
    """Return the first existing column from a candidate list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_signal_amplitude_columns(df):
    """Infer columns that can indicate whether a waveform/channel is effectively present.

    We avoid delta_* columns because a negative delta is a physiological change, not low signal.
    We also avoid relative band power columns because they are compositional variables.
    """
    include_tokens = [
        "log_rms", "rms",
        "log_total_power", "total_power",
        "median_log10_psd", "log10_psd", "median_psd",
    ]
    exclude_tokens = [
        "delta_", "_rel", "relative", "pc", "ising", "beta", "exponent",
        "offset", "residual", "fitted", "z_",
    ]

    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in include_tokens) and not any(tok in lc for tok in exclude_tokens):
            x = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(x).sum() >= 10 and np.nanstd(x[np.isfinite(x)]) > 1e-12:
                cols.append(c)

    priority = {"log_rms": 0, "rms": 1, "log_total_power": 2, "total_power": 3, "median_log10_psd": 4, "log10_psd": 5, "median_psd": 6}
    cols = sorted(cols, key=lambda c: min([priority[t] for t in priority if t in c.lower()] + [99]))
    return cols


def infer_fooof_r2_columns(df):
    """Infer FOOOF / aperiodic fit R² columns, excluding Ising/model R²."""
    cols = []
    for c in df.columns:
        lc = c.lower()
        has_r2 = ("r2" in lc) or ("r_squared" in lc) or ("rsquared" in lc)
        looks_fooof = ("fooof" in lc) or ("aperiodic" in lc) or ("specparam" in lc)
        bad = ("ising" in lc) or ("residual" in lc) or ("model" in lc) or ("coef" in lc)
        if has_r2 and looks_fooof and not bad:
            x = pd.to_numeric(df[c], errors="coerce")
            if np.isfinite(x).sum() >= 10:
                cols.append(c)

    if not cols:
        for c in df.columns:
            lc = c.lower()
            if lc in {"r2", "fit_r2", "fooof_r2", "aperiodic_r2", "r_squared", "rsquared"}:
                x = pd.to_numeric(df[c], errors="coerce")
                if np.isfinite(x).sum() >= 10:
                    cols.append(c)

    return cols


def add_channel_and_qc_covariates(df, out_dir):
    """Apply low-signal channel QC and attach channel-count / fit-R² covariates.

    The function is intentionally column-name tolerant, because upstream step2 files may use
    slightly different names for RMS, total power, PSD median, channel, and FOOOF R².
    """
    use = df.copy()
    n0 = len(use)

    channel_col = first_existing_column(use, CHANNEL_COL_CANDIDATES)
    if channel_col is not None:
        use["qc_channel"] = use[channel_col].astype(str)
    else:
        # This keeps the code runnable, but channel-count control becomes unavailable.
        use["qc_channel"] = "unknown_channel"
        print("[QC WARN] No channel column found. Channel-count control will be uninformative.")

    signal_cols = infer_signal_amplitude_columns(use)
    r2_cols = infer_fooof_r2_columns(use)

    qc_info = {
        "n_rows_before_qc": int(n0),
        "channel_col": channel_col,
        "signal_amplitude_columns_used": signal_cols,
        "fooof_r2_columns_used": r2_cols,
        "apply_signal_channel_qc": bool(APPLY_SIGNAL_CHANNEL_QC),
        "signal_qc_lower_quantile_by_region": float(SIGNAL_QC_LOWER_QUANTILE_BY_REGION),
        "min_channel_signal_pass_fraction": float(MIN_CHANNEL_SIGNAL_PASS_FRACTION),
        "apply_fooof_r2_hard_qc": bool(APPLY_FOOOF_R2_HARD_QC),
        "min_fooof_r2": float(MIN_FOOOF_R2),
    }

    # --------------------------------------------------------
    # 1) FOOOF R² as row-level fit-quality covariate
    # --------------------------------------------------------
    if r2_cols:
        r2_mat = []
        for c in r2_cols:
            x = pd.to_numeric(use[c], errors="coerce").astype(float)
            # Some pipelines save R² as percent. Convert if needed.
            finite = np.isfinite(x)
            if finite.sum() and np.nanmedian(x[finite]) > 1.5:
                x = x / 100.0
            r2_mat.append(x.values)
        r2_mat = np.vstack(r2_mat).T
        use["fit_r2_for_control"] = np.nanmean(r2_mat, axis=1)
    else:
        use["fit_r2_for_control"] = np.nan
        print("[QC WARN] No FOOOF/aperiodic R² column found. R² control will be skipped.")

    # --------------------------------------------------------
    # 2) Low-signal / almost-flat waveform QC
    #    Unit-free strategy: convert each available amplitude/power variable into
    #    a region-wise percentile rank, then use the median percentile score.
    # --------------------------------------------------------
    if signal_cols:
        rank_cols = []
        for c in signal_cols:
            x = pd.to_numeric(use[c], errors="coerce").astype(float)
            # If the column is RMS/total_power in linear units, rank still works.
            tmp = pd.DataFrame({"region": use["region"].astype(str).values, "x": x.values})
            # pct=True gives 0-1 percentile rank within each region.
            r = tmp.groupby("region")["x"].rank(method="average", pct=True)
            rc = f"qc_rank__{c}"
            use[rc] = r.values
            rank_cols.append(rc)

        use["qc_signal_score"] = use[rank_cols].median(axis=1, skipna=True)
        use["qc_signal_row_pass"] = np.isfinite(use["qc_signal_score"]) & (
            use["qc_signal_score"] >= SIGNAL_QC_LOWER_QUANTILE_BY_REGION
        )
    else:
        use["qc_signal_score"] = np.nan
        use["qc_signal_row_pass"] = True
        print("[QC WARN] No amplitude/power/PSD-median column found. Low-signal channel QC cannot be applied.")

    # Channel-level pass fraction. This targets stable bad channels, not isolated epochs.
    channel_group_cols = ["animal_id", "subject", "group", "state_label", "region", "qc_channel"]
    use["qc_channel_signal_pass_fraction"] = (
        use.groupby(channel_group_cols)["qc_signal_row_pass"]
        .transform("mean")
        .astype(float)
    )
    use["qc_channel_signal_ok"] = use["qc_channel_signal_pass_fraction"] >= MIN_CHANNEL_SIGNAL_PASS_FRACTION

    # Hard R² QC can be disabled if it removes too many data points.
    if APPLY_FOOOF_R2_HARD_QC and r2_cols:
        use["qc_fooof_r2_ok"] = np.isfinite(use["fit_r2_for_control"]) & (use["fit_r2_for_control"] >= MIN_FOOOF_R2)
    else:
        use["qc_fooof_r2_ok"] = True

    use["qc_keep"] = True
    if APPLY_SIGNAL_CHANNEL_QC:
        use["qc_keep"] &= use["qc_channel_signal_ok"].astype(bool)
    use["qc_keep"] &= use["qc_fooof_r2_ok"].astype(bool)

    qc_channel_summary = (
        use.groupby(channel_group_cols)
        .agg(
            n_rows=("qc_keep", "size"),
            signal_pass_fraction=("qc_signal_row_pass", "mean"),
            signal_score_median=("qc_signal_score", "median"),
            fit_r2_median=("fit_r2_for_control", "median"),
            channel_signal_ok=("qc_channel_signal_ok", "max"),
            channel_kept=("qc_keep", "max"),
        )
        .reset_index()
    )

    if out_dir is not None:
        with open(out_dir / "QC_detected_columns_and_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(qc_info, f, ensure_ascii=False, indent=2)
        qc_channel_summary.to_csv(out_dir / "QC_channel_signal_and_fit_summary.csv", index=False, encoding="utf-8-sig")

    before_counts = (
        use.groupby(["state_label", "region"])
        .agg(n_rows_before=("qc_keep", "size"), n_channels_before=("qc_channel", "nunique"))
        .reset_index()
    )

    use = use[use["qc_keep"].astype(bool)].copy()

    after_counts = (
        use.groupby(["state_label", "region"])
        .agg(n_rows_after=("qc_keep", "size"), n_channels_after=("qc_channel", "nunique"))
        .reset_index()
    )

    qc_counts = before_counts.merge(after_counts, on=["state_label", "region"], how="left")
    for c in ["n_rows_after", "n_channels_after"]:
        qc_counts[c] = qc_counts[c].fillna(0).astype(int)

    if out_dir is not None:
        qc_counts.to_csv(out_dir / "QC_counts_before_after_by_state_region.csv", index=False, encoding="utf-8-sig")

    print("[QC] detected channel column:", channel_col)
    print("[QC] signal columns used:", signal_cols if signal_cols else "NONE")
    print("[QC] FOOOF R² columns used:", r2_cols if r2_cols else "NONE")
    print(f"[QC] rows before={n0}, after={len(use)}, removed={n0 - len(use)}")
    print("[QC] counts by state/region:")
    print(qc_counts.to_string(index=False))

    # --------------------------------------------------------
    # 3) Attach technical covariates to the retained rows.
    # --------------------------------------------------------
    region_group_cols = ["animal_id", "subject", "group", "state_label", "region"]
    use["valid_channel_count_in_region"] = (
        use.groupby(region_group_cols)["qc_channel"].transform("nunique").astype(float)
    )
    use["region_mean_fit_r2"] = (
        use.groupby(region_group_cols)["fit_r2_for_control"].transform("mean").astype(float)
    )
    use["region_median_signal_score"] = (
        use.groupby(region_group_cols)["qc_signal_score"].transform("median").astype(float)
    )

    return use

def build_design_matrix(df, numeric_covariates=None, categorical_covariates=None):
    numeric_covariates = numeric_covariates or []
    categorical_covariates = categorical_covariates or []

    X_parts = []
    names = []

    # intercept
    X_parts.append(np.ones((len(df), 1), dtype=float))
    names.append("Intercept")

    # numeric
    for c in numeric_covariates:
        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        z = zscore_safe(x)
        X_parts.append(z.reshape(-1, 1))
        names.append("z_" + c)

    # categorical dummies
    for c in categorical_covariates:
        if c not in df.columns:
            continue

        s = df[c].astype(str).fillna("NA")
        dummies = pd.get_dummies(s, prefix=c, drop_first=True)

        if dummies.shape[1] == 0:
            continue

        X_parts.append(dummies.values.astype(float))
        names.extend(list(dummies.columns))

    X = np.hstack(X_parts)
    return X, names


def residualize(df, y_col, numeric_covariates, categorical_covariates, model_name):
    use = df.copy()

    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use = use[np.isfinite(use[y_col])].copy()

    numeric_covariates = prepare_numeric_covariates(use, numeric_covariates)

    X, names = build_design_matrix(
        use,
        numeric_covariates=numeric_covariates,
        categorical_covariates=categorical_covariates
    )

    y = use[y_col].values.astype(float)

    # pseudo-inverse handles collinearity
    beta = np.linalg.pinv(X) @ y
    yhat = X @ beta
    resid = y - yhat

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    out = use.copy()
    out["residual"] = resid
    out["fitted"] = yhat
    out["residual_model"] = model_name

    coef = pd.DataFrame({
        "model": model_name,
        "term": names,
        "coef": beta,
        "n_rows": len(out),
        "r2": r2,
        "y_col": y_col,
    })

    return out, coef, numeric_covariates


def aggregate_animal_region(resid_df, value_col="residual"):

    group_cols = ["animal_id", "subject", "group", "state_label", "region"]

    # Input validation.
    missing_group_cols = [c for c in group_cols if c not in resid_df.columns]
    if missing_group_cols:
        raise RuntimeError(f"aggregate_animal_region missing grouping columns: {missing_group_cols}")

    if value_col not in resid_df.columns:
        raise RuntimeError(f"aggregate_animal_region missing value column: {value_col}")

    agg_dict = {
        "n_rows": (value_col, "size"),
        "residual_mean": (value_col, "mean"),
        "residual_sd": (value_col, "std"),
    }

    # QC / technical summaries for checking and optional second-level interpretation.
    if "qc_channel" in resid_df.columns:
        agg_dict["n_valid_channels"] = ("qc_channel", "nunique")
    if "fit_r2_for_control" in resid_df.columns:
        agg_dict["fooof_r2_mean"] = ("fit_r2_for_control", "mean")
    if "region_mean_fit_r2" in resid_df.columns:
        agg_dict["region_mean_fit_r2"] = ("region_mean_fit_r2", "mean")
    if "qc_signal_score" in resid_df.columns:
        agg_dict["qc_signal_score_mean"] = ("qc_signal_score", "mean")
    if "region_median_signal_score" in resid_df.columns:
        agg_dict["region_median_signal_score"] = ("region_median_signal_score", "mean")

    optional_aggs = {
        "raw_delta_exponent_mean": ("delta_exponent_vs_fentanyl", "mean"),
        "baseline_exponent_mean": ("baseline_exponent", "mean"),
        "delta_offset_mean": ("delta_offset_vs_fentanyl", "mean"),

        "delta_PC1_mean": ("delta_effective_PC1_vs_fentanyl", "mean"),
        "delta_PC2_mean": ("delta_effective_PC2_vs_fentanyl", "mean"),
        "delta_PC3_mean": ("delta_effective_PC3_vs_fentanyl", "mean"),

        "baseline_ising_beta_mean": ("baseline_ising_beta_param", "mean"),
        "delta_ising_beta_mean": ("delta_ising_beta_param_vs_fentanyl", "mean"),

        "baseline_ising_c_mean": ("baseline_ising_c_param", "mean"),
        "delta_ising_c_mean": ("delta_ising_c_param_vs_fentanyl", "mean"),

        "baseline_ising_dist_crit_mean": ("baseline_ising_dist_crit", "mean"),
        "delta_ising_dist_crit_mean": ("delta_ising_dist_crit_vs_fentanyl", "mean"),
    }

    for out_col, (in_col, func) in optional_aggs.items():
        if in_col in resid_df.columns:
            agg_dict[out_col] = (in_col, func)

    summary = (
        resid_df
        .groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    # Input validation.
    missing_after = [c for c in group_cols if c not in summary.columns]
    if missing_after:
        raise RuntimeError(
            f"aggregate_animal_region output is missing columns: {missing_after}\n"
            f"summary columns = {list(summary.columns)}"
        )

    return summary
def animal_level_paired_test(summary, model_name, analysis_name, out_dir):
    rows = []
    diff_tables = []

    for state in STATE_ORDER:
        tmp = summary[summary["state_label"] == state].copy()

        wide = tmp.pivot_table(
            index=["animal_id", "group"],
            columns="region",
            values="residual_mean",
            aggfunc="mean"
        ).reset_index()

        if "cortex" not in wide.columns or "thalamus" not in wide.columns:
            continue

        wide = wide.dropna(subset=["cortex", "thalamus"]).copy()

        if wide.empty:
            continue

        wide["thalamus_minus_cortex_residual"] = wide["thalamus"] - wide["cortex"]
        wide["model"] = model_name
        wide["analysis"] = analysis_name
        wide["state"] = state

        # exact sign-flip
        obs, p_perm, n = exact_signflip_test(wide["thalamus_minus_cortex_residual"].values)

        if len(wide) >= 2:
            tt = ttest_rel(wide["thalamus"], wide["cortex"], nan_policy="omit")
            tval = float(tt.statistic)
            p_t = float(tt.pvalue)
        else:
            tval, p_t = np.nan, np.nan

        rows.append({
            "model": model_name,
            "analysis": analysis_name,
            "state": state,
            "n_animals": int(len(wide)),
            "mean_resid_cortex": float(wide["cortex"].mean()),
            "mean_resid_thalamus": float(wide["thalamus"].mean()),
            "mean_thalamus_minus_cortex_residual": float(wide["thalamus_minus_cortex_residual"].mean()),
            "median_thalamus_minus_cortex_residual": float(wide["thalamus_minus_cortex_residual"].median()),
            "t_paired_reference": tval,
            "p_paired_ttest_reference": p_t,
            "p_signflip_exact_MAIN": p_perm,
            "all_diffs_positive": bool((wide["thalamus_minus_cortex_residual"] > 0).all()),
        })

        diff_tables.append(wide)

        wide.to_csv(
            out_dir / f"diff_table_{model_name}_{analysis_name}_{state}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    tests = pd.DataFrame(rows)

    if diff_tables:
        diffs = pd.concat(diff_tables, ignore_index=True)
    else:
        diffs = pd.DataFrame()

    return tests, diffs


def plot_residual_paired(summary, model_name, analysis_name, out_dir):
    """
    Paired residual plot, one file per state.
    """
    for state in STATE_ORDER:
        tmp = summary[summary["state_label"] == state].copy()

        wide = tmp.pivot_table(
            index=["animal_id", "group"],
            columns="region",
            values="residual_mean",
            aggfunc="mean"
        ).reset_index()

        if "cortex" not in wide.columns or "thalamus" not in wide.columns:
            continue

        wide = wide.dropna(subset=["cortex", "thalamus"]).copy()
        if wide.empty:
            continue

        fig, ax = plt.subplots(figsize=(4.4, 4.8), dpi=220)

        for _, r in wide.iterrows():
            ax.plot([0, 1], [r["cortex"], r["thalamus"]], color="0.72", lw=1.2)
            ax.scatter([0, 1], [r["cortex"], r["thalamus"]], s=36)

        ax.axhline(0, color="0.2", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cortex", "Thalamus"])
        ax.set_ylabel("Residual ΔFOOOF exponent")
        ax.set_title(f"{model_name}\n{analysis_name}: {state}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / f"paired_residual_{model_name}_{analysis_name}_{state}.png")
        plt.close(fig)


def plot_residual_diff_bar(all_diffs, out_dir):
    """
    Summary plot of animal-level residual differences.
    """
    if all_diffs.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)

    plot_df = all_diffs[all_diffs["model"].eq(MAIN_MODEL_NAME)].copy()
    if plot_df.empty:
        plot_df = all_diffs.copy()

    labels = []
    means = []
    xs = []

    for i, (analysis, state) in enumerate([
        ("all_available_pre_ischemia", "propofol_early"),
        ("all_available_pre_ischemia", "propofol_late"),
        ("sedation_only", "propofol_early"),
        ("sedation_only", "propofol_late"),
    ]):
        g = plot_df[(plot_df["analysis"] == analysis) & (plot_df["state"] == state)]
        if g.empty:
            continue

        x = len(xs)
        vals = g["thalamus_minus_cortex_residual"].values.astype(float)

        xs.append(x)
        labels.append(f"{analysis}\n{state}")
        means.append(np.mean(vals))

        # animal points
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=34, alpha=0.8)
        ax.plot([x - 0.18, x + 0.18], [np.mean(vals), np.mean(vals)], color="0.1", lw=2)

    ax.axhline(0, color="0.2", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Thalamus - cortex residual ΔFOOOF exponent")
    ax.set_title("Animal-level residual differences")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "MAIN_animal_level_residual_differences.png")
    plt.close(fig)




def _stable_seed(*parts, base_seed=MATCHED_CHANNEL_RANDOM_SEED):
    """Deterministic seed from strings without relying on Python's randomized hash."""
    s = "|".join(str(p) for p in parts)
    val = base_seed
    for ch in s:
        val = (val * 131 + ord(ch)) % (2 ** 32 - 1)
    return int(val)


def channel_level_summary_for_matching(resid_df, value_col="residual"):
    """Collapse epoch rows to channel means before channel matching.

    This makes each selected channel contribute equally, instead of letting channels with
    more retained epochs dominate the regional mean.
    """
    required_cols = ["animal_id", "subject", "group", "state_label", "region", "qc_channel", value_col]
    missing = [c for c in required_cols if c not in resid_df.columns]
    if missing:
        raise RuntimeError(f"channel_level_summary_for_matching missing columns: {missing}")

    agg_dict = {
        "channel_value_mean": (value_col, "mean"),
        "channel_value_sd": (value_col, "std"),
        "n_epoch_rows": (value_col, "size"),
    }
    if "fit_r2_for_control" in resid_df.columns:
        agg_dict["channel_fit_r2_mean"] = ("fit_r2_for_control", "mean")
    if "qc_signal_score" in resid_df.columns:
        agg_dict["channel_signal_score_mean"] = ("qc_signal_score", "mean")

    ch = (
        resid_df
        .groupby(["animal_id", "subject", "group", "state_label", "region", "qc_channel"])
        .agg(**agg_dict)
        .reset_index()
    )
    return ch


def _one_matched_channel_draw(channel_df, state, rng):
    """One random matched-channel draw for one state.

    For each animal, sample k channels from cortex and k channels from thalamus,
    where k = min(n_cortex, n_thalamus). Return one paired row per animal.
    """
    tmp = channel_df[channel_df["state_label"] == state].copy()
    rows = []

    for (animal_id, group), g in tmp.groupby(["animal_id", "group"]):
        ctx = g[g["region"] == "cortex"].copy()
        thal = g[g["region"] == "thalamus"].copy()

        if ctx.empty or thal.empty:
            continue

        ctx_channels = np.asarray(ctx["qc_channel"].unique())
        thal_channels = np.asarray(thal["qc_channel"].unique())
        k = int(min(len(ctx_channels), len(thal_channels)))

        if k < MIN_MATCHED_CHANNELS_PER_REGION:
            continue

        if len(ctx_channels) > k:
            sel_ctx = rng.choice(ctx_channels, size=k, replace=False)
        else:
            sel_ctx = ctx_channels

        if len(thal_channels) > k:
            sel_thal = rng.choice(thal_channels, size=k, replace=False)
        else:
            sel_thal = thal_channels

        ctx_sel = ctx[ctx["qc_channel"].isin(sel_ctx)]
        thal_sel = thal[thal["qc_channel"].isin(sel_thal)]

        ctx_val = float(ctx_sel["channel_value_mean"].mean())
        thal_val = float(thal_sel["channel_value_mean"].mean())

        rows.append({
            "animal_id": animal_id,
            "group": group,
            "state": state,
            "cortex": ctx_val,
            "thalamus": thal_val,
            "thalamus_minus_cortex_residual": thal_val - ctx_val,
            "n_matched_channels_per_region": k,
            "selected_cortex_channels": ";".join(map(str, sorted(sel_ctx))),
            "selected_thalamus_channels": ";".join(map(str, sorted(sel_thal))),
            "n_cortex_channels_available": int(len(ctx_channels)),
            "n_thalamus_channels_available": int(len(thal_channels)),
        })

    return pd.DataFrame(rows)


def _paired_rows_from_equal_channel_means(channel_df, state):
    """Exact draw-averaged matched-channel paired rows for one state.

    For a fixed animal/state, each draw samples k=min(n_ctx,n_thal) channels per region.
    Under uniform sampling without replacement, the expected selected-channel mean equals
    the mean over all available channels in that region. Therefore the draw-averaged
    matched-channel animal difference can be computed exactly without simulating draws.
    """
    tmp = channel_df[channel_df["state_label"] == state].copy()
    rows = []

    for (animal_id, subject, group), g in tmp.groupby(["animal_id", "subject", "group"]):
        ctx = g[g["region"] == "cortex"].copy()
        thal = g[g["region"] == "thalamus"].copy()

        if ctx.empty or thal.empty:
            continue

        n_ctx = int(ctx["qc_channel"].nunique())
        n_thal = int(thal["qc_channel"].nunique())
        k = int(min(n_ctx, n_thal))

        if k < MIN_MATCHED_CHANNELS_PER_REGION:
            continue

        ctx_val = float(ctx["channel_value_mean"].mean())
        thal_val = float(thal["channel_value_mean"].mean())

        rows.append({
            "animal_id": animal_id,
            "subject": subject,
            "group": group,
            "state": state,
            "cortex": ctx_val,
            "thalamus": thal_val,
            "thalamus_minus_cortex_residual": thal_val - ctx_val,
            "n_draws_contributing": int(N_MATCHED_CHANNEL_DRAWS) if not FAST_DRAW_AVERAGED_MATCHING else 1,
            "median_n_matched_channels_per_region": float(k),
            "n_cortex_channels_available": float(n_ctx),
            "n_thalamus_channels_available": float(n_thal),
        })

    return pd.DataFrame(rows)


def _make_main_test_row_from_drawavg(drawavg, model_name, analysis_name, state):
    st = drawavg[drawavg["state"] == state].copy()
    if st.empty:
        return None

    vals = st["thalamus_minus_cortex_residual"].values.astype(float)
    obs, p_perm, n = exact_signflip_test(vals)

    if len(st) >= 2:
        tt = ttest_rel(st["thalamus"], st["cortex"], nan_policy="omit")
        tval = float(tt.statistic)
        p_t = float(tt.pvalue)
    else:
        tval, p_t = np.nan, np.nan

    row = {
        "model": model_name,
        "analysis": analysis_name,
        "state": state,
        "n_animals": int(len(st)),
        "n_draws": int(N_MATCHED_CHANNEL_DRAWS) if not FAST_DRAW_AVERAGED_MATCHING else 1,
        "mean_resid_cortex": float(st["cortex"].mean()),
        "mean_resid_thalamus": float(st["thalamus"].mean()),
        "mean_thalamus_minus_cortex_residual": float(st["thalamus_minus_cortex_residual"].mean()),
        "median_thalamus_minus_cortex_residual": float(st["thalamus_minus_cortex_residual"].median()),
        "t_paired_reference": tval,
        "p_paired_ttest_reference": p_t,
        "p_signflip_exact_MAIN": p_perm,
        "all_diffs_positive": bool((vals > 0).all()),
        "median_n_matched_channels_per_region": float(st["median_n_matched_channels_per_region"].median()),
        "median_n_cortex_channels_available": float(st["n_cortex_channels_available"].median()),
        "median_n_thalamus_channels_available": float(st["n_thalamus_channels_available"].median()),

        # In fast mode these describe the exact draw-averaged matched-channel effect,
        # rather than a Monte-Carlo distribution over random draws.
        "draw_effect_mean": float(st["thalamus_minus_cortex_residual"].mean()),
        "draw_effect_median": float(st["thalamus_minus_cortex_residual"].mean()),
        "draw_effect_ci2p5": np.nan,
        "draw_effect_ci97p5": np.nan,
        "draw_p_median": p_perm,
        "draw_prop_p_le_0p05": float(p_perm <= 0.05) if np.isfinite(p_perm) else np.nan,
        "draw_prop_all_diffs_positive": float((vals > 0).all()),
        "matched_channel_mode": "exact_draw_averaged_equal_channel_mean" if FAST_DRAW_AVERAGED_MATCHING else "monte_carlo_draws",
    }
    return row


def matched_channel_subsampling_test(resid_df, model_name, analysis_name, out_dir):
    """Matched-channel inference with memory-safe exact draw-averaged computation.

    The previous implementation materialized every draw × animal row for every model.
    On old Windows/conda environments this can trigger native memory crashes. The main
    matched-channel test only needs the draw-averaged animal difference, which is exactly
    obtainable from channel-level equal means. This function preserves the main output
    filenames and columns used by the plotting script.
    """
    if not MATCHED_CHANNEL_SUBSAMPLING:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    channel_df = channel_level_summary_for_matching(resid_df, value_col="residual")
    channel_df.to_csv(
        out_dir / f"matched_channel_level_summary_{model_name}_{analysis_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Fast exact draw-averaged main inference
    # --------------------------------------------------------
    if FAST_DRAW_AVERAGED_MATCHING:
        drawavg_parts = []
        for state in STATE_ORDER:
            st = _paired_rows_from_equal_channel_means(channel_df, state)
            if st.empty:
                continue
            st["model"] = model_name
            st["analysis"] = analysis_name
            drawavg_parts.append(st)

            # Representative draw table: in exact mode this is the draw-averaged table.
            st.to_csv(
                out_dir / f"matched_channel_representative_draw_diff_table_{model_name}_{analysis_name}_{state}.csv",
                index=False,
                encoding="utf-8-sig"
            )

        drawavg_diffs = pd.concat(drawavg_parts, ignore_index=True) if drawavg_parts else pd.DataFrame()

        main_rows = []
        if not drawavg_diffs.empty:
            for state in STATE_ORDER:
                row = _make_main_test_row_from_drawavg(drawavg_diffs, model_name, analysis_name, state)
                if row is not None:
                    main_rows.append(row)

        main_tests = pd.DataFrame(main_rows)
        draw_tests = pd.DataFrame()  # intentionally not materialized in fast mode

        if not main_tests.empty:
            main_tests.to_csv(
                out_dir / f"matched_channel_MAIN_tests_{model_name}_{analysis_name}.csv",
                index=False,
                encoding="utf-8-sig"
            )
        if not drawavg_diffs.empty:
            drawavg_diffs.to_csv(
                out_dir / f"matched_channel_drawavg_animal_differences_{model_name}_{analysis_name}.csv",
                index=False,
                encoding="utf-8-sig"
            )

        return main_tests, drawavg_diffs, draw_tests

    # --------------------------------------------------------
    # Original Monte-Carlo fallback: kept for optional diagnostics only.
    # --------------------------------------------------------
    rng = np.random.default_rng(_stable_seed(model_name, analysis_name))

    draw_test_rows = []
    diff_rows = []

    for draw_idx in range(int(N_MATCHED_CHANNEL_DRAWS)):
        for state in STATE_ORDER:
            wide = _one_matched_channel_draw(channel_df, state, rng)
            if wide.empty:
                continue

            wide["draw"] = draw_idx
            wide["model"] = model_name
            wide["analysis"] = analysis_name
            diff_rows.append(wide)

            vals = wide["thalamus_minus_cortex_residual"].values.astype(float)
            obs, p_perm, n = exact_signflip_test(vals)

            if len(wide) >= 2:
                tt = ttest_rel(wide["thalamus"], wide["cortex"], nan_policy="omit")
                tval = float(tt.statistic)
                p_t = float(tt.pvalue)
            else:
                tval, p_t = np.nan, np.nan

            draw_test_rows.append({
                "model": model_name,
                "analysis": analysis_name,
                "state": state,
                "draw": draw_idx,
                "n_animals": int(len(wide)),
                "mean_resid_cortex": float(wide["cortex"].mean()),
                "mean_resid_thalamus": float(wide["thalamus"].mean()),
                "mean_thalamus_minus_cortex_residual": float(np.mean(vals)),
                "median_thalamus_minus_cortex_residual": float(np.median(vals)),
                "t_paired_reference": tval,
                "p_paired_ttest_reference": p_t,
                "p_signflip_exact_per_draw": p_perm,
                "all_diffs_positive": bool((vals > 0).all()),
                "mean_n_matched_channels_per_region": float(wide["n_matched_channels_per_region"].mean()),
            })

    draw_tests = pd.DataFrame(draw_test_rows)
    diffs = pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame()

    if not draw_tests.empty:
        draw_tests.to_csv(
            out_dir / f"matched_channel_draw_tests_{model_name}_{analysis_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
    if not diffs.empty:
        diffs.to_csv(
            out_dir / f"matched_channel_draw_animal_differences_{model_name}_{analysis_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    main_rows = []
    drawavg_diffs_all = []

    if not diffs.empty:
        drawavg = (
            diffs
            .groupby(["model", "analysis", "state", "animal_id", "group"])
            .agg(
                cortex=("cortex", "mean"),
                thalamus=("thalamus", "mean"),
                thalamus_minus_cortex_residual=("thalamus_minus_cortex_residual", "mean"),
                n_draws_contributing=("draw", "nunique"),
                median_n_matched_channels_per_region=("n_matched_channels_per_region", "median"),
                n_cortex_channels_available=("n_cortex_channels_available", "median"),
                n_thalamus_channels_available=("n_thalamus_channels_available", "median"),
            )
            .reset_index()
        )

        for state in STATE_ORDER:
            row = _make_main_test_row_from_drawavg(drawavg, model_name, analysis_name, state)
            if row is not None:
                main_rows.append(row)
                drawavg_diffs_all.append(drawavg[drawavg["state"] == state].copy())

    main_tests = pd.DataFrame(main_rows)
    drawavg_diffs = pd.concat(drawavg_diffs_all, ignore_index=True) if drawavg_diffs_all else pd.DataFrame()

    if not main_tests.empty:
        main_tests.to_csv(
            out_dir / f"matched_channel_MAIN_tests_{model_name}_{analysis_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
    if not drawavg_diffs.empty:
        drawavg_diffs.to_csv(
            out_dir / f"matched_channel_drawavg_animal_differences_{model_name}_{analysis_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    return main_tests, drawavg_diffs, draw_tests

def plot_matched_channel_draw_effects(matched_draw_tests, out_dir):
    """Quick diagnostic plot for draw-level matched-channel effects of the main model."""
    if matched_draw_tests.empty:
        return

    plot_df = matched_draw_tests[matched_draw_tests["model"].eq(MAIN_MODEL_NAME)].copy()
    if plot_df.empty:
        return

    for analysis in plot_df["analysis"].dropna().unique():
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
        sub = plot_df[plot_df["analysis"] == analysis].copy()
        xs = []
        labels = []
        for i, state in enumerate(STATE_ORDER):
            vals = sub[sub["state"] == state]["mean_thalamus_minus_cortex_residual"].dropna().values
            if len(vals) == 0:
                continue
            xs.append(i)
            labels.append(state)
            ax.boxplot(vals, positions=[i], widths=0.46, showfliers=False)
            ax.scatter(np.full(len(vals), i), vals, s=4, alpha=0.08)
        ax.axhline(0, color="0.25", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Matched-channel Thalamus - cortex residual ΔFOOOF exponent")
        ax.set_title(f"Matched-channel draw effects: {analysis}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"MAIN_matched_channel_draw_effect_distribution_{analysis}.png")
        plt.close(fig)


# ============================================================
# Load input data.
# ============================================================

print("=" * 120)
print("[LOAD]")
print("=" * 120)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Feature table not found: {INPUT_CSV}\n"
        "Run extract_propofol_features.py before the statistical analysis."
    )

df = pd.read_csv(INPUT_CSV)

print(f"[INPUT] {INPUT_CSV}")
print(f"[ROWS] {len(df)}")
print(f"[COLUMNS] {len(df.columns)}")

df = df[df["state_label"].isin(STATE_ORDER)].copy()

required = [
    "animal_id",
    "subject",
    "group",
    "state_label",
    "region",
    "delta_exponent_vs_fentanyl",
]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise RuntimeError(f"Input feature table is missing required columns: {missing_required}")

df = df[df["region"].isin(["cortex", "thalamus"])].copy()
df["delta_exponent_vs_fentanyl"] = pd.to_numeric(df["delta_exponent_vs_fentanyl"], errors="coerce")
df = df[np.isfinite(df["delta_exponent_vs_fentanyl"])].copy()

print(f"[PROP ROWS BEFORE QC] {len(df)}")
print(df.groupby(["state_label", "region"])["delta_exponent_vs_fentanyl"].agg(["count", "mean", "std"]).to_string())

# ============================================================
# ============================================================

print("\n" + "=" * 120)
print("[BASE QC: LOW-AMPLITUDE CHANNELS + FOOOF R2 HARD FILTER]")
print("=" * 120)

df = add_channel_and_qc_covariates(df, OUT_DIR)

print(f"[PROP ROWS AFTER QC] {len(df)}")
print(df.groupby(["state_label", "region"])["delta_exponent_vs_fentanyl"].agg(["count", "mean", "std"]).to_string())

# Check whether the updated step2 Ising full-state signal QC is present.
if "ising_signal_ok" in df.columns:
    print("\n[CHECK] Ising full-state signal QC pass rate after base QC:")
    print(df.groupby(["state_label", "region"])["ising_signal_ok"].mean().to_string())
if "ising_control_ok" in df.columns:
    print("\n[CHECK] Ising-control rows available after base QC:")
    print(df.groupby(["state_label", "region"])["ising_control_ok"].mean().to_string())


# ============================================================
# ============================================================

base_covs = [
    "baseline_exponent",

]

pc_covs = [
    "delta_effective_PC1_vs_fentanyl",
    "delta_effective_PC2_vs_fentanyl",
    "delta_effective_PC3_vs_fentanyl",
]

direct_covs = [
    "delta_log_rms_vs_fentanyl",
    "delta_log_total_power_vs_fentanyl",
    "delta_delta_rel_vs_fentanyl",
    "delta_gamma_rel_vs_fentanyl",
    "delta_ABP_mean_vs_fentanyl",
    "delta_ICP_mean_vs_fentanyl",
]

ising_beta_c_covs = [
    "baseline_ising_beta_param",
    "delta_ising_beta_param_vs_fentanyl",
    "baseline_ising_c_param",
    "delta_ising_c_param_vs_fentanyl",
]

ising_distance_covs = [
    "baseline_ising_dist_crit",
    "delta_ising_dist_crit_vs_fentanyl",
]

# FOOOF R² is primarily handled by hard QC above.
# It is included only in a sensitivity model, not in the main model, because R² can be partly
# downstream of region/state-related spectral changes. Channel count is NOT used as a covariate;
# it is controlled by matched-channel subsampling below.
fit_r2_sensitivity_covs = [
    "fit_r2_for_control",
]

CONTROL_MODELS = {
    "R0_RAW_no_residualization": {
        "numeric": [],
        "categorical": [],
        "ising_qc": False,
        "description": "Raw delta exponent, no residualization",
    },

    "R1_effective_PC123": {
        "numeric": pc_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": False,
        "description": "Residualize by state/group and effective-state PC1-PC3",
    },

    "R2_PC123_baseline_offset": {
        "numeric": pc_covs + base_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": False,
        "description": "Residualize by effective-state PC1-PC3, baseline exponent, delta offset",
    },

    "R3_direct_covariates_baseline_offset": {
        "numeric": direct_covs + base_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": False,
        "description": "Residualize by direct physiological/spectral covariates, baseline exponent, delta offset",
    },

    "R4_PC123_baseline_offset_IsingBetaC_no_channelR2": {
        "numeric": pc_covs + base_covs + ising_beta_c_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": True,
        "description": "Sensitivity: residualize by PC1-PC3, baseline exponent, adaptive-Ising beta/c; no channel-count/R2 controls",
    },

    "R5_PC123_baseline_offset_IsingDistance": {
        "numeric": pc_covs + base_covs + ising_distance_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": True,
        "description": "Residualize by PC1-PC3, baseline exponent, delta offset, adaptive-Ising critical distance",
    },

    "R6_direct_covariates_baseline_offset_IsingBetaC": {
        "numeric": direct_covs + base_covs + ising_beta_c_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": True,
        "description": "Residualize by direct covariates, baseline exponent, adaptive-Ising beta/c",
    },

    MAIN_MODEL_NAME: {
        "numeric": pc_covs + base_covs + ising_beta_c_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": True,
        "description": "MAIN QC: residualize by PC1-PC3, baseline exponent, adaptive-Ising beta/c; channel count controlled by matched-channel subsampling",
    },

    "R8_sensitivity_PC123_baseline_IsingBetaC_fitR2": {
        "numeric": pc_covs + base_covs + ising_beta_c_covs + fit_r2_sensitivity_covs,
        "categorical": ["state_label", "group"],
        "ising_qc": True,
        "description": "Sensitivity: same as main plus FOOOF R2 as an explicit covariate",
    },
}


# ============================================================
# 4. Run residualization + animal-level exact sign-flip
# ============================================================

print("\n" + "=" * 120)
print("[ANIMAL-LEVEL RESIDUAL PERMUTATION MAIN TEST]")
print("=" * 120)

all_test_rows = []
all_diff_rows = []
all_summary_rows = []
all_coef_rows = []
all_matched_test_rows = []
all_matched_diff_rows = []
all_matched_draw_rows = []
model_meta_rows = []

for model_name, spec in CONTROL_MODELS.items():
    print(f"\n[MODEL] {model_name}")
    print(f"        {spec['description']}")

    use = df.copy()

    # Ising-control models only use rows with valid adaptive-Ising controls.
    # In the updated step2, ising_control_ok already includes:
    #   full-state Ising signal QC + current/baseline ACF fit QC.
    if spec.get("ising_qc", False):
        if "ising_control_ok" in use.columns:
            use = use[use["ising_control_ok"].astype(bool)].copy()

    # R0 raw: no residualization
    if model_name == "R0_RAW_no_residualization":
        resid_df = use.copy()
        resid_df["residual"] = resid_df["delta_exponent_vs_fentanyl"].values
        resid_df["fitted"] = 0.0
        resid_df["residual_model"] = model_name

        coef_df = pd.DataFrame([{
            "model": model_name,
            "term": "RAW_no_fit",
            "coef": np.nan,
            "n_rows": len(resid_df),
            "r2": np.nan,
            "y_col": "delta_exponent_vs_fentanyl",
        }])

        used_covs = []

    else:
        resid_df, coef_df, used_covs = residualize(
            use,
            y_col="delta_exponent_vs_fentanyl",
            numeric_covariates=spec["numeric"],
            categorical_covariates=spec["categorical"],
            model_name=model_name,
        )

    coef_df["description"] = spec["description"]
    coef_df.to_csv(
        OUT_DIR / f"coefficients_{model_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_coef_rows.append(coef_df)

    print(f"        rows used = {len(resid_df)}")
    print(f"        animals   = {resid_df['animal_id'].nunique()}")
    print(f"        covariates used = {used_covs}")

    summary = aggregate_animal_region(resid_df, value_col="residual")
    summary["model"] = model_name
    summary["description"] = spec["description"]

    summary.to_csv(
        OUT_DIR / f"animal_state_region_residual_summary_{model_name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_summary_rows.append(summary)

    # all animals
    tests_all, diffs_all = animal_level_paired_test(
        summary,
        model_name=model_name,
        analysis_name="all_available_pre_ischemia",
        out_dir=OUT_DIR
    )

    # sedation only
    summary_sed = summary[summary["group"] == "sedation"].copy()
    tests_sed, diffs_sed = animal_level_paired_test(
        summary_sed,
        model_name=model_name,
        analysis_name="sedation_only",
        out_dir=OUT_DIR
    )

    all_test_rows.append(tests_all)
    all_test_rows.append(tests_sed)

    all_diff_rows.append(diffs_all)
    all_diff_rows.append(diffs_sed)

    # Plotting many model-level figures can trigger native backend crashes in old Windows stacks.
    # Keep only main-model diagnostics by default; CSV outputs are unaffected.
    if SAVE_PER_MODEL_DIAGNOSTIC_PLOTS or (SAVE_MAIN_MODEL_DIAGNOSTIC_PLOTS and model_name == MAIN_MODEL_NAME):
        plot_residual_paired(summary, model_name, "all_available_pre_ischemia", OUT_DIR)
        plot_residual_paired(summary_sed, model_name, "sedation_only", OUT_DIR)
        plt.close("all")

    # Matched-channel subsampling: controls cortex/thalamus channel-count imbalance
    matched_tests_all, matched_diffs_all, matched_draws_all = matched_channel_subsampling_test(
        resid_df,
        model_name=model_name,
        analysis_name="all_available_pre_ischemia",
        out_dir=OUT_DIR
    )

    resid_df_sed = resid_df[resid_df["group"] == "sedation"].copy()
    matched_tests_sed, matched_diffs_sed, matched_draws_sed = matched_channel_subsampling_test(
        resid_df_sed,
        model_name=model_name,
        analysis_name="sedation_only",
        out_dir=OUT_DIR
    )

    all_matched_test_rows.append(matched_tests_all)
    all_matched_test_rows.append(matched_tests_sed)
    all_matched_diff_rows.append(matched_diffs_all)
    all_matched_diff_rows.append(matched_diffs_sed)
    all_matched_draw_rows.append(matched_draws_all)
    all_matched_draw_rows.append(matched_draws_sed)

    model_meta_rows.append({
        "model": model_name,
        "description": spec["description"],
        "n_rows_used": len(resid_df),
        "n_animals_used": resid_df["animal_id"].nunique(),
        "numeric_covariates_requested": "|".join(spec["numeric"]),
        "numeric_covariates_used": "|".join(used_covs),
        "categorical_covariates": "|".join(spec["categorical"]),
        "ising_qc": spec.get("ising_qc", False),
    })

    # Memory cleanup between models. This is useful on old Windows/conda stacks.
    del use, resid_df, summary, summary_sed
    gc.collect()
    plt.close("all")


# ============================================================
# Save generated output.
# ============================================================

tests = pd.concat(all_test_rows, ignore_index=True) if all_test_rows else pd.DataFrame()
diffs = pd.concat(all_diff_rows, ignore_index=True) if all_diff_rows else pd.DataFrame()
summaries = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()
coefs = pd.concat(all_coef_rows, ignore_index=True) if all_coef_rows else pd.DataFrame()
meta = pd.DataFrame(model_meta_rows)
matched_tests = pd.concat(all_matched_test_rows, ignore_index=True) if all_matched_test_rows else pd.DataFrame()
matched_diffs = pd.concat(all_matched_diff_rows, ignore_index=True) if all_matched_diff_rows else pd.DataFrame()
matched_draw_tests = pd.concat(all_matched_draw_rows, ignore_index=True) if all_matched_draw_rows else pd.DataFrame()

tests_path = OUT_DIR / "MAIN_animal_level_residual_permutation_tests_unmatched_reference.csv"
diffs_path = OUT_DIR / "MAIN_animal_level_residual_difference_tables_unmatched_reference.csv"
summaries_path = OUT_DIR / "MAIN_animal_state_region_residual_summary_all_models.csv"
coefs_path = OUT_DIR / "MAIN_residualization_coefficients_all_models.csv"
meta_path = OUT_DIR / "MAIN_residualization_model_metadata.csv"
matched_tests_path = OUT_DIR / "MAIN_matched_channel_subsampling_tests.csv"
matched_diffs_path = OUT_DIR / "MAIN_matched_channel_drawavg_animal_difference_tables.csv"
matched_draw_tests_path = OUT_DIR / "MAIN_matched_channel_draw_level_tests.csv"

tests.to_csv(tests_path, index=False, encoding="utf-8-sig")
diffs.to_csv(diffs_path, index=False, encoding="utf-8-sig")
summaries.to_csv(summaries_path, index=False, encoding="utf-8-sig")
coefs.to_csv(coefs_path, index=False, encoding="utf-8-sig")
meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
matched_tests.to_csv(matched_tests_path, index=False, encoding="utf-8-sig")
matched_diffs.to_csv(matched_diffs_path, index=False, encoding="utf-8-sig")
matched_draw_tests.to_csv(matched_draw_tests_path, index=False, encoding="utf-8-sig")

plot_residual_diff_bar(diffs, OUT_DIR)
plot_matched_channel_draw_effects(matched_draw_tests, OUT_DIR)


# ============================================================
# ============================================================

print("\n" + "=" * 120)
print("[UNMATCHED REFERENCE RESULTS: EXACT SIGN-FLIP ON ANIMAL-LEVEL RESIDUAL DIFFERENCES]")
print("=" * 120)

if not tests.empty:
    print(tests.to_string(index=False))
else:
    print("[WARN] Unmatched reference tests returned no rows.")

print("\n" + "=" * 120)
print("[MATCHED-CHANNEL SUBSAMPLING RESULTS: MAIN INFERENCE]")
print("=" * 120)

if not matched_tests.empty:
    print(matched_tests.to_string(index=False))
else:
    print("[WARN] Matched-channel tests returned no rows.")

print("\n" + "=" * 120)
print("[MAIN MODEL ONLY: MATCHED-CHANNEL SUBSAMPLING]")
print("=" * 120)

main_tests = matched_tests[matched_tests["model"] == MAIN_MODEL_NAME].copy() if not matched_tests.empty else pd.DataFrame()
if not main_tests.empty:
    print(main_tests.to_string(index=False))
else:
    print("[WARN] Main matched-channel model returned no tests.")

print("\n" + "=" * 120)
print("[MAIN MODEL ANIMAL DIFF TABLES: DRAW-AVERAGED MATCHED CHANNELS]")
print("=" * 120)

main_diffs = matched_diffs[matched_diffs["model"] == MAIN_MODEL_NAME].copy() if not matched_diffs.empty else pd.DataFrame()
if not main_diffs.empty:
    print(main_diffs.to_string(index=False))
else:
    print("[WARN] Main matched-channel model returned no animal differences.")


# ============================================================
# Save generated output.
# ============================================================

run_summary = {
    "input_csv": str(INPUT_CSV),
    "out_dir": str(OUT_DIR),
    "main_model": MAIN_MODEL_NAME,
    "main_test": (
        "Residualize epoch-level delta exponent without region term; "
        "aggregate residuals to channel means; "
        "match cortex/thalamus channel counts by random subsampling within each animal/state; "
        "average animal differences over matched draws; "
        "test thalamus-minus-cortex residual differences by exact sign-flip permutation."
    ),
    "n_input_rows_propofol": int(len(df)),
    "n_models": int(len(CONTROL_MODELS)),
    "state_order": STATE_ORDER,
    "qc_settings": {
        "apply_signal_channel_qc": APPLY_SIGNAL_CHANNEL_QC,
        "signal_qc_lower_quantile_by_region": SIGNAL_QC_LOWER_QUANTILE_BY_REGION,
        "min_channel_signal_pass_fraction": MIN_CHANNEL_SIGNAL_PASS_FRACTION,
        "apply_fooof_r2_hard_qc": APPLY_FOOOF_R2_HARD_QC,
        "min_fooof_r2": MIN_FOOOF_R2,
        "matched_channel_subsampling": MATCHED_CHANNEL_SUBSAMPLING,
        "n_matched_channel_draws": N_MATCHED_CHANNEL_DRAWS,
        "matched_channel_random_seed": MATCHED_CHANNEL_RANDOM_SEED,
        "min_matched_channels_per_region": MIN_MATCHED_CHANNELS_PER_REGION,
        "fast_draw_averaged_matching": FAST_DRAW_AVERAGED_MATCHING,
        "save_per_model_diagnostic_plots": SAVE_PER_MODEL_DIAGNOSTIC_PLOTS,
        "save_main_model_diagnostic_plots": SAVE_MAIN_MODEL_DIAGNOSTIC_PLOTS,
        "fit_r2_sensitivity_covariates": fit_r2_sensitivity_covs,
        "ising_control_ok_includes_full_state_signal_qc": "ising_signal_ok" in df.columns,
    },
    "outputs": {
        "unmatched_reference_tests": str(tests_path),
        "unmatched_reference_diffs": str(diffs_path),
        "matched_channel_tests": str(matched_tests_path),
        "matched_channel_diffs": str(matched_diffs_path),
        "matched_channel_draw_tests": str(matched_draw_tests_path),
        "summaries": str(summaries_path),
        "coefs": str(coefs_path),
        "meta": str(meta_path),
    },
}

with open(OUT_DIR / "run_summary_step3_animal_level_residual_permutation.json", "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 120)
print("[DONE]")
print("=" * 120)
print(f"OUT_DIR = {OUT_DIR}")
print("Key outputs:")
print("  MAIN_matched_channel_subsampling_tests.csv")
print("  MAIN_matched_channel_drawavg_animal_difference_tables.csv")
print("  MAIN_matched_channel_draw_level_tests.csv")
print("  MAIN_animal_level_residual_permutation_tests_unmatched_reference.csv")
print("  MAIN_animal_level_residual_difference_tables_unmatched_reference.csv")
print("  MAIN_animal_state_region_residual_summary_all_models.csv")
print("  MAIN_residualization_coefficients_all_models.csv")
print("  MAIN_residualization_model_metadata.csv")
print("  MAIN_animal_level_residual_differences.png")
print(f"  paired_residual_{MAIN_MODEL_NAME}_all_available_pre_ischemia_propofol_early.png")
print(f"  paired_residual_{MAIN_MODEL_NAME}_all_available_pre_ischemia_propofol_late.png")
print(f"  paired_residual_{MAIN_MODEL_NAME}_sedation_only_propofol_early.png")
