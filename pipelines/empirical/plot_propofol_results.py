# -*- coding: utf-8 -*-
r"""Create the publication-ready horizontal propofol result figure.

Revised visual design:
    a: raw propofol-induced aperiodic exponent change, Ctx vs Thal paired plot
    b: Δ adaptive-Ising beta distribution by region, instead of Fentanyl→Propofol line plots
    c: residualization forest plot, instead of a line-heavy ladder
    d: adjusted paired residuals, Ctx vs Thal paired plot

Important design choices:
- The figure canvas is set to the same media-box size as figure1.pdf:
    758.352 x 412.304 pt = 10.5327 x 5.7264 inch
  Therefore it should be included in LaTeX exactly like figure1:
    \\includegraphics[width=\\textwidth]{figurefinal.pdf}
- The PDF is saved without bbox_inches='tight', so the output page size remains fixed.

Reviewer-statistics revision:
- reports paired Cohen's dz and small-sample-corrected Hedges' gz;
- reports 95% bootstrap CIs for the animal-level mean contrast and dz;
- performs leave-one-animal-out sensitivity analyses;
- reports both the canonical exhaustive sign-flip p-value and the earlier
  conservative plus-one value.  The plus-one correction is needed for random
  Monte-Carlo permutations, but not when all 2**n sign patterns are enumerated.
"""

from pathlib import Path
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.text import Text


# ============================================================
# 0. Paths and fixed analysis choices
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ROOT = Path(os.environ.get("PIG_DATA_ROOT", REPOSITORY_ROOT / "data" / "external" / "ds003380")).resolve()
RESULTS_ROOT = Path(os.environ.get(
    "PIG_RESULTS_ROOT",
    PROJECT_ROOT / "ALLEN" / "results" / "empirical_propofol",
)).resolve()

STEP2_DIR = RESULTS_ROOT / "step2_features"
STEP3_DIR = RESULTS_ROOT / "step3_statistics"

EPOCH_CSV = STEP2_DIR / "05_epoch_channel_features_fooof_with_delta_Ising.csv"
TEST_CSV = STEP3_DIR / "MAIN_matched_channel_subsampling_tests.csv"
DIFF_CSV = STEP3_DIR / "MAIN_matched_channel_drawavg_animal_difference_tables.csv"
SUMMARY_CSV = STEP3_DIR / "MAIN_animal_state_region_residual_summary_all_models.csv"

OUT_DIR = RESULTS_ROOT / "publication_figure"
OUT_DIR.mkdir(exist_ok=True, parents=True)

STATE = "propofol_early"
ANALYSIS = "all_available_pre_ischemia"

# Base controls for all figure panels:
# - FOOOF R² hard filtering is already done in step2 and repeated in step3;
# - low-amplitude/low-power epoch-channel QC is reproduced here only for panel b;
# - matched-channel subsampling is used for panel b so β change is not driven by 9 cortex vs 2 thalamus channels.
MIN_FOOOF_R2 = 0.80
SIGNAL_QC_LOWER_QUANTILE_BY_REGION = 0.02
MIN_CHANNEL_SIGNAL_PASS_FRACTION = 0.70
N_MATCHED_CHANNEL_DRAWS_FOR_BETA = 5000
MATCHED_CHANNEL_RANDOM_SEED_FOR_BETA = 20260630

# Match the media-box size of figure1.pdf.
FIG_W_IN = 758.352 / 72.0
FIG_H_IN = 412.304 / 72.0

# The figure is inserted into the manuscript at approximately 7.16 inches
# while its fixed PDF media-box width is 10.5327 inches. Matplotlib font
# sizes are therefore enlarged so that the effective size after LaTeX
# downscaling remains safely above the editor's 10-pt requirement.
FINAL_INSERT_WIDTH_IN = 7.16
MANUSCRIPT_SCALE = FINAL_INSERT_WIDTH_IN / FIG_W_IN
MIN_FINAL_FONT_PT = 10.0

def source_pt(final_pt):
    """Convert a desired final manuscript font size to source-PDF points."""
    return float(final_pt) / MANUSCRIPT_SCALE

# Effective final manuscript font targets.
FONT_BASE = source_pt(10.3)
FONT_LABEL = source_pt(10.4)
FONT_TITLE = source_pt(10.5)
FONT_TICK = source_pt(10.1)
FONT_LEGEND = source_pt(10.1)
FONT_STATS = source_pt(10.1)
FONT_STATS_COMPACT = source_pt(10.0)
FONT_STAR = source_pt(12.0)
FONT_PANEL = source_pt(12.5)

FINAL_MODEL_CANDIDATES = [
    # Main displayed final model keeps the same conceptual label as before:
    # + Ising 1−κI.  All rows are now from the matched-channel/QC/R²-filtered
    # result tables, so QC + matched channels + FOOOF R² are treated as base controls.
    "R5_PC123_baseline_IsingDistance",
    "R5_PC123_baseline_offset_IsingDistance",
    # Fallbacks if only the beta/c main model is available.
    "R7_MAIN_PC123_baseline_IsingBetaC_signalQC_matchedChannels",
    "R8_sensitivity_PC123_baseline_IsingBetaC_fitR2",
]

MODEL_GROUPS_FOR_LADDER = [
    ["R0_RAW_no_residualization"],
    ["R1_effective_PC123"],
    ["R2_PC123_baseline", "R2_PC123_baseline_offset"],
    ["R5_PC123_baseline_IsingDistance", "R5_PC123_baseline_offset_IsingDistance"],
]

MODEL_DISPLAY = {
    "R0_RAW_no_residualization": "Raw",
    "R1_effective_PC123": "+ state PCs",
    "R2_PC123_baseline": "+ baseline",
    "R2_PC123_baseline_offset": "+ baseline",
    "R5_PC123_baseline_IsingDistance": "+ Ising 1−κ$_I$",
    "R5_PC123_baseline_offset_IsingDistance": "+ Ising 1−κ$_I$",
    "R7_MAIN_PC123_baseline_IsingBetaC_signalQC_matchedChannels": "+ Ising β/c",
    "R8_sensitivity_PC123_baseline_IsingBetaC_fitR2": "+ Ising β/c",
}

COLORS = {
    "cortex": "#3B6EA8",
    "thalamus": "#D88C3A",
    "line": "#C6C6C6",
    "text": "#111111",
    "axis": "#333333",
    "zero": "#8A8A8A",
    "main": "#111111",
    "light": "#EAEAEA",
    "dots": "#7A7A7A",
}


# ============================================================
# 1. Style helpers
# ============================================================

def set_pub_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.05,
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.major.size": 3.8,
        "ytick.major.size": 3.8,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(color=COLORS["axis"], labelcolor=COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])


def add_panel_label_fig(fig, ax, label, dx=-0.026, dy=0.044):
    """Place panel letters in figure coordinates so they do not collide with axis labels/titles."""
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        label,
        ha="left",
        va="top",
        fontsize=FONT_PANEL,
        fontweight="bold",
        color=COLORS["text"],
    )


def p_label(p):
    if not np.isfinite(p):
        return "p = n.a."
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def exact_p_label(p):
    """Compact label for the exhaustive two-sided sign-flip p-value."""
    if not np.isfinite(p):
        return r"$p_{exact}$ = n.a."
    if p < 0.001:
        return r"$p_{exact}$ < 0.001"
    return rf"$p_{{exact}}$ = {p:.3f}"


def compact_effect_annotation(vals, seed=20260813):
    """Two compact figure lines; full precision remains in the CSV outputs."""
    vals = finite_values(vals)
    mean_val, p_exact, _, n = exact_signflip_one_sample(vals)
    dz, _ = paired_standardized_effect(vals)
    ci_lo, ci_hi = bootstrap_ci(vals, n_boot=20000, seed=seed)
    line1 = rf"Thal$-$Ctx = {mean_val:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}]"
    line2 = rf"$d_z$ = {dz:.2f}, {exact_p_label(p_exact)}, $n$ = {n}"
    return line1, line2, {
        "mean": mean_val,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cohen_dz": dz,
        "p_exact": p_exact,
        "n": n,
    }


def p_star(p):
    if not np.isfinite(p):
        return ""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def finite_values(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def jitter_positions(n, width=0.035):
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def set_padded_ylim(ax, vals, pad_frac=0.16, min_span=0.1):
    vals = finite_values(vals)
    if len(vals) == 0:
        return
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    span = max(ymax - ymin, min_span)
    ax.set_ylim(ymin - span * pad_frac, ymax + span * pad_frac)


def pivot_region_pairs(df, value_col, group_cols=("animal_id", "group")):
    wide = (
        df.pivot_table(
            index=list(group_cols),
            columns="region",
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    if "cortex" not in wide.columns or "thalamus" not in wide.columns:
        return pd.DataFrame()
    return wide.dropna(subset=["cortex", "thalamus"]).copy()


def exact_signflip_one_sample(vals):
    """Two-sided exhaustive sign-flip test.

    Because all 2**n sign patterns are enumerated, the canonical exact p-value
    is count / 2**n; the Monte-Carlo ``+1`` correction is not needed.  The
    conservative plus-one value is returned separately for continuity with
    earlier analyses (for n=6 its minimum is 3/65 = 0.0462).
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    obs = float(np.mean(vals))
    null_vals = []
    for signs in itertools.product([-1, 1], repeat=n):
        null_vals.append(np.mean(np.asarray(signs, dtype=float) * vals))
    null_vals = np.asarray(null_vals, dtype=float)
    extreme = int(np.sum(np.abs(null_vals) >= abs(obs) - 1e-15))
    p_exact = extreme / len(null_vals)
    p_plus1 = (extreme + 1.0) / (len(null_vals) + 1.0)
    return obs, float(p_exact), float(p_plus1), int(n)


def bootstrap_ci(vals, n_boot=10000, ci=95, seed=20260629):
    vals = finite_values(vals)
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def paired_standardized_effect(vals):
    """Cohen's dz and its small-sample corrected counterpart Hedges' gz."""
    vals = finite_values(vals)
    n = len(vals)
    if n < 2:
        return np.nan, np.nan
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan, np.nan
    dz = float(np.mean(vals) / sd)
    # Hedges correction using df=n-1; accurate and stable for the present n.
    correction = 1.0 - 3.0 / (4.0 * (n - 1.0) - 1.0)
    return dz, float(correction * dz)


def bootstrap_effect_ci(vals, n_boot=20000, ci=95, seed=20260813):
    """Percentile bootstrap CIs for the paired mean and Cohen's dz."""
    vals = finite_values(vals)
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_mean = np.empty(n_boot, dtype=float)
    boot_dz = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot_mean[i] = np.mean(sample)
        if n >= 2:
            sd = np.std(sample, ddof=1)
            if np.isfinite(sd) and sd > 0:
                boot_dz[i] = np.mean(sample) / sd
    alpha = (100.0 - ci) / 2.0
    mean_ci = tuple(np.percentile(boot_mean, [alpha, 100.0 - alpha]).astype(float))
    finite_dz = boot_dz[np.isfinite(boot_dz)]
    dz_ci = (
        tuple(np.percentile(finite_dz, [alpha, 100.0 - alpha]).astype(float))
        if len(finite_dz) else (np.nan, np.nan)
    )
    return mean_ci, dz_ci


def summarize_small_sample_effect(vals, label, model, seed=20260813):
    """One audit-ready row for a paired/one-sample animal-level contrast."""
    vals = finite_values(vals)
    mean_val, p_exact, p_plus1, n = exact_signflip_one_sample(vals)
    dz, gz = paired_standardized_effect(vals)
    mean_ci, dz_ci = bootstrap_effect_ci(vals, seed=seed)
    return {
        "contrast": label,
        "model": model,
        "n_animals": n,
        "mean_difference": mean_val,
        "mean_difference_bootstrap_ci95_lo": mean_ci[0],
        "mean_difference_bootstrap_ci95_hi": mean_ci[1],
        "cohen_dz": dz,
        "cohen_dz_bootstrap_ci95_lo": dz_ci[0],
        "cohen_dz_bootstrap_ci95_hi": dz_ci[1],
        "hedges_gz_small_sample_corrected": gz,
        "p_signflip_exhaustive_two_sided": p_exact,
        "p_signflip_plus1_conservative": p_plus1,
        "n_sign_patterns": int(2 ** n) if n else 0,
    }


def leave_one_animal_out_sensitivity(vals_df, value_col, label, model):
    """Repeat the effect after omitting each animal in turn."""
    d = vals_df[["animal_id", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[np.isfinite(d[value_col])].drop_duplicates("animal_id")
    rows = []
    for omitted in d["animal_id"].tolist():
        kept = d.loc[d["animal_id"] != omitted, value_col].to_numpy(dtype=float)
        row = summarize_small_sample_effect(
            kept,
            label=label,
            model=model,
            seed=20260813 + len(rows),
        )
        row["omitted_animal_id"] = omitted
        row["all_remaining_same_direction"] = bool(
            len(kept) and np.all(np.sign(kept) == np.sign(np.mean(kept)))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_reviewer_requested_statistics(diffs, final_model):
    """Effect sizes, uncertainty, and leave-one-animal-out sensitivity.

    These analyses directly address the reviewer requests concerning the small
    biological sample, effect-size reporting, confidence intervals, and
    sensitivity to individual animals.  The unit of inference remains the
    animal; channel/epoch observations are never treated as independent n.
    """
    model_order = ["R0_RAW_no_residualization", final_model]
    summary_rows = []
    loo_tables = []
    animal_tables = []

    for i, model in enumerate(dict.fromkeys(model_order)):
        gd = get_model_diff_values(diffs, model)
        if gd.empty:
            print(f"[WARN] no animal-level difference rows for {model}")
            continue
        value_col = "thalamus_minus_cortex_residual"
        vals = pd.to_numeric(gd[value_col], errors="coerce")
        label = "Thalamus minus cortex: raw delta exponent" if model.startswith("R0_") else (
            "Thalamus minus cortex: adjusted residualized delta exponent"
        )
        summary_rows.append(
            summarize_small_sample_effect(vals, label, model, seed=20260813 + i)
        )
        loo_tables.append(
            leave_one_animal_out_sensitivity(gd, value_col, label, model)
        )
        keep = gd[["animal_id", "group", value_col]].copy()
        keep["contrast"] = label
        keep["model"] = model
        animal_tables.append(keep)

    summary = pd.DataFrame(summary_rows)
    loo = pd.concat(loo_tables, ignore_index=True) if loo_tables else pd.DataFrame()
    animals = pd.concat(animal_tables, ignore_index=True) if animal_tables else pd.DataFrame()
    return summary, loo, animals


def choose_first_available_model(df_with_model_col, candidates, model_role="model"):
    available = set(df_with_model_col["model"].astype(str))
    for m in candidates:
        if m in available:
            return m
    raise RuntimeError(
        f"Could not identify the {model_role} model. Candidates:\n"
        + "\n".join(candidates)
        + "\nAvailable models:\n"
        + "\n".join(sorted(available))
    )


def choose_ladder_models(tests):
    available = set(tests["model"].astype(str))
    models = []
    for candidates in MODEL_GROUPS_FOR_LADDER:
        chosen = None
        for m in candidates:
            if m in available:
                chosen = m
                break
        if chosen is None:
            print(f"[WARN] missing ladder model candidates: {candidates}")
        else:
            models.append(chosen)
    return models


def get_test_row(tests, model, analysis=ANALYSIS, state=STATE):
    q = tests[
        (tests["model"] == model)
        & (tests["analysis"] == analysis)
        & (tests["state"] == state)
    ].copy()
    if q.empty:
        return None
    return q.iloc[0]


def audit_final_font_sizes(fig, min_final_pt=MIN_FINAL_FONT_PT):
    """Report any visible text that would fall below 10 pt after LaTeX scaling."""
    fig.canvas.draw()
    issues = []
    for obj in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        txt = obj.get_text()
        if txt is None or not str(txt).strip() or not obj.get_visible():
            continue
        source_size = float(obj.get_fontsize())
        final_size = source_size * MANUSCRIPT_SCALE
        if final_size + 1e-9 < min_final_pt:
            issues.append((str(txt).replace("\n", " / "), source_size, final_size))

    if issues:
        print("\n[FONT AUDIT] Text below the final 10-pt requirement:")
        for txt, src_pt, final_pt in sorted(issues, key=lambda x: x[2]):
            print(f"  final={final_pt:.2f} pt | source={src_pt:.2f} pt | {txt}")
    else:
        print(
            f"\n[FONT AUDIT] PASS: all visible text is >= {min_final_pt:.1f} pt "
            f"after manuscript scaling (scale={MANUSCRIPT_SCALE:.3f})."
        )


def save_fixed_canvas(fig, stem="figurefinal"):
    # Do NOT use bbox_inches='tight': keeping fixed page size makes LaTeX scaling identical to figure1.pdf.
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}",
            facecolor="white",
            edgecolor="white",
        )


# ============================================================
# 2. Data preparation
# ============================================================

def compute_raw_animal_means(epoch_df):
    d = epoch_df.copy()
    d = d[d["state_label"] == STATE].copy()
    d = d[d["region"].isin(["cortex", "thalamus"])].copy()
    d["delta_exponent_vs_fentanyl"] = pd.to_numeric(d["delta_exponent_vs_fentanyl"], errors="coerce")
    d = d[np.isfinite(d["delta_exponent_vs_fentanyl"])].copy()

    return (
        d.groupby(["animal_id", "group", "state_label", "region"])
        .agg(
            delta_mean=("delta_exponent_vs_fentanyl", "mean"),
            n_epochs=("delta_exponent_vs_fentanyl", "size"),
        )
        .reset_index()
    )



def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_signal_amplitude_columns_for_plot(df):
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
    return cols


def apply_base_qc_for_beta_panel(df):
    """Reproduce the base QC for panel b.

    Panels a/c/d read the matched-channel step3 outputs directly. Panel b is computed
    from step2 Ising beta columns, so we apply the same base logic here:
    FOOOF R² hard filtering, low-amplitude channel QC, and updated Ising signal/fitting QC.
    """
    d = df.copy()

    channel_col = first_existing_column(
        d,
        [
            "channel", "channel_id", "channel_num", "chan", "ch", "ch_num",
            "edf_channel", "channel_name", "chan_name", "electrode", "electrode_id",
        ],
    )
    if channel_col is None:
        d["qc_channel"] = "unknown_channel"
    else:
        d["qc_channel"] = d[channel_col].astype(str)

    if "fooof_r2" in d.columns:
        r2 = pd.to_numeric(d["fooof_r2"], errors="coerce")
        d = d[np.isfinite(r2) & (r2 >= MIN_FOOOF_R2)].copy()

    signal_cols = infer_signal_amplitude_columns_for_plot(d)
    if signal_cols:
        rank_cols = []
        for c in signal_cols:
            x = pd.to_numeric(d[c], errors="coerce")
            tmp = pd.DataFrame({"region": d["region"].astype(str).values, "x": x.values})
            r = tmp.groupby("region")["x"].rank(method="average", pct=True)
            rc = f"qc_rank__{c}"
            d[rc] = r.values
            rank_cols.append(rc)
        d["qc_signal_score"] = d[rank_cols].median(axis=1, skipna=True)
        d["qc_signal_row_pass"] = np.isfinite(d["qc_signal_score"]) & (
            d["qc_signal_score"] >= SIGNAL_QC_LOWER_QUANTILE_BY_REGION
        )
        group_cols = ["animal_id", "group", "state_label", "region", "qc_channel"]
        if "subject" in d.columns:
            group_cols.insert(1, "subject")
        d["qc_channel_signal_pass_fraction"] = d.groupby(group_cols)["qc_signal_row_pass"].transform("mean")
        d = d[d["qc_channel_signal_pass_fraction"] >= MIN_CHANNEL_SIGNAL_PASS_FRACTION].copy()

    # This is the key update after adding Ising full-state signal QC in step2.
    if "ising_control_ok" in d.columns:
        d = d[d["ising_control_ok"].fillna(False).astype(bool)].copy()
    else:
        if "ising_signal_ok" in d.columns:
            d = d[d["ising_signal_ok"].fillna(False).astype(bool)].copy()
        if "ising_fit_success" in d.columns:
            d = d[d["ising_fit_success"].fillna(False).astype(bool)].copy()

    return d


def _stable_seed_for_beta(*parts, base_seed=MATCHED_CHANNEL_RANDOM_SEED_FOR_BETA):
    s = "|".join(str(p) for p in parts)
    val = base_seed
    for ch in s:
        val = (val * 131 + ord(ch)) % (2 ** 32 - 1)
    return int(val)


def compute_statewise_ising_beta_means(epoch_df):
    """Return matched-channel animal-level Δ adaptive-Ising beta for panel b.

    The returned table is already the panel-b data: one row per animal × region,
    with cortex/thalamus channel counts balanced by random subsampling.
    """
    d = epoch_df.copy()
    d = d[d["state_label"].eq(STATE)].copy()
    d = d[d["region"].isin(["cortex", "thalamus"])].copy()

    need_cols = ["ising_beta_param", "baseline_ising_beta_param"]
    for c in need_cols:
        if c not in d.columns:
            print(f"[WARN] panel b cannot find {c}; returning empty beta table.")
            return pd.DataFrame(columns=["animal_id", "group", "region", "delta_beta"])
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[np.isfinite(d["ising_beta_param"]) & np.isfinite(d["baseline_ising_beta_param"])].copy()
    d = apply_base_qc_for_beta_panel(d)
    if d.empty:
        return pd.DataFrame(columns=["animal_id", "group", "region", "delta_beta"])

    d["delta_beta_epoch"] = d["ising_beta_param"] - d["baseline_ising_beta_param"]

    channel_group_cols = ["animal_id", "group", "region", "qc_channel"]
    ch = (
        d.groupby(channel_group_cols)
        .agg(delta_beta_channel=("delta_beta_epoch", "mean"), n_rows=("delta_beta_epoch", "size"))
        .reset_index()
    )

    rng = np.random.default_rng(_stable_seed_for_beta("panel_b", STATE, ANALYSIS))
    draw_rows = []

    for draw in range(int(N_MATCHED_CHANNEL_DRAWS_FOR_BETA)):
        for (animal_id, group), g in ch.groupby(["animal_id", "group"]):
            ctx = g[g["region"] == "cortex"].copy()
            thal = g[g["region"] == "thalamus"].copy()
            if ctx.empty or thal.empty:
                continue

            ctx_channels = np.asarray(ctx["qc_channel"].unique())
            thal_channels = np.asarray(thal["qc_channel"].unique())
            k = int(min(len(ctx_channels), len(thal_channels)))
            if k < 1:
                continue

            sel_ctx = rng.choice(ctx_channels, size=k, replace=False) if len(ctx_channels) > k else ctx_channels
            sel_thal = rng.choice(thal_channels, size=k, replace=False) if len(thal_channels) > k else thal_channels

            ctx_val = float(ctx[ctx["qc_channel"].isin(sel_ctx)]["delta_beta_channel"].mean())
            thal_val = float(thal[thal["qc_channel"].isin(sel_thal)]["delta_beta_channel"].mean())

            draw_rows.append({
                "draw": draw,
                "animal_id": animal_id,
                "group": group,
                "region": "Cortex",
                "delta_beta": ctx_val,
                "n_matched_channels_per_region": k,
                "n_cortex_channels_available": int(len(ctx_channels)),
                "n_thalamus_channels_available": int(len(thal_channels)),
            })
            draw_rows.append({
                "draw": draw,
                "animal_id": animal_id,
                "group": group,
                "region": "Thalamus",
                "delta_beta": thal_val,
                "n_matched_channels_per_region": k,
                "n_cortex_channels_available": int(len(ctx_channels)),
                "n_thalamus_channels_available": int(len(thal_channels)),
            })

    draws = pd.DataFrame(draw_rows)
    if draws.empty:
        return pd.DataFrame(columns=["animal_id", "group", "region", "delta_beta"])

    # Save draw-level values for auditability.
    draws.to_csv(OUT_DIR / "figurefinal_beta_delta_matched_draw_level_values.csv", index=False, encoding="utf-8-sig")

    out = (
        draws.groupby(["animal_id", "group", "region"])
        .agg(
            delta_beta=("delta_beta", "mean"),
            n_draws=("draw", "nunique"),
            median_n_matched_channels_per_region=("n_matched_channels_per_region", "median"),
            median_n_cortex_channels_available=("n_cortex_channels_available", "median"),
            median_n_thalamus_channels_available=("n_thalamus_channels_available", "median"),
        )
        .reset_index()
    )
    out.to_csv(OUT_DIR / "figurefinal_beta_delta_matched_animal_values.csv", index=False, encoding="utf-8-sig")
    return out


def compute_beta_delta_by_region(state_beta):
    # New path: compute_statewise_ising_beta_means already returns matched-channel Δβ values.
    if "delta_beta" in state_beta.columns:
        return state_beta.copy()
    return pd.DataFrame(columns=["animal_id", "group", "region", "delta_beta"])


def get_model_diff_values(diffs, model):
    gd = diffs[
        (diffs["model"] == model)
        & (diffs["analysis"] == ANALYSIS)
        & (diffs["state"] == STATE)
    ].copy()
    if gd.empty:
        return pd.DataFrame()
    return gd


# ============================================================
# 3. Panel plotting functions
# ============================================================

def draw_panel_a(ax, matched_diffs, tests):
    """Panel a uses draw-averaged matched-channel R0 values.

    R0 is raw delta exponent after the base pipeline:
    signal QC + FOOOF R² hard filtering + matched-channel subsampling.
    Therefore panel a is still visually the raw propofol-induced change,
    but it is computed from the same QC/matched data base as panels c/d.
    """
    wide = get_model_diff_values(matched_diffs, "R0_RAW_no_residualization")

    for _, r in wide.iterrows():
        ax.plot([0, 1], [r["cortex"], r["thalamus"]], color=COLORS["line"], lw=1.25, zorder=1)
        ax.scatter(0, r["cortex"], s=58, color=COLORS["cortex"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(1, r["thalamus"], s=58, color=COLORS["thalamus"], edgecolor="white", linewidth=0.8, zorder=3)

    # Keep the main-text figure visually lean: show only the contrast and the
    # canonical exhaustive p-value. Effect sizes, confidence intervals, n, and
    # leave-one-animal-out results are reported in the caption/text and CSVs.
    vals = wide["thalamus_minus_cortex_residual"].to_numpy(dtype=float)
    mean_val, p_exact, _, _ = exact_signflip_one_sample(vals)
    ax.set_title(
        "Raw propofol-induced change\n"
        + rf"Thal$-$Ctx = {mean_val:.3f}, {exact_p_label(p_exact)}",
        pad=6,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ctx", "Thal"])
    ax.set_ylabel("Δ aperiodic exponent\n(propofol − fentanyl)")
    ax.axhline(0, color=COLORS["zero"], lw=1.0, ls="--")
    ax.set_xlim(-0.35, 1.35)
    set_padded_ylim(ax, wide[["cortex", "thalamus"]].values.ravel(), pad_frac=0.14)
    clean_ax(ax)


def draw_panel_b(ax, beta_delta_df):
    # Region-wise Δβ dot plot: less line-heavy than the original fentanyl→propofol paired plot.
    region_order = ["Cortex", "Thalamus"]
    region_colors = {"Cortex": COLORS["cortex"], "Thalamus": COLORS["thalamus"]}
    xmap = {"Cortex": 0, "Thalamus": 1}
    all_vals = []

    stats = []
    for region in region_order:
        vals = beta_delta_df.loc[beta_delta_df["region"] == region, "delta_beta"].astype(float).values
        vals = finite_values(vals)
        all_vals.extend(vals.tolist())

        x0 = xmap[region]
        jit = jitter_positions(len(vals), width=0.055)
        ax.scatter(
            np.full(len(vals), x0) + jit,
            vals,
            s=58,
            color=region_colors[region],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )

        mean_val, p_val, p_plus1, n = exact_signflip_one_sample(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, seed=20260629 + x0)
        ax.errorbar(
            x0,
            mean_val,
            yerr=np.array([[mean_val - ci_lo], [ci_hi - mean_val]]),
            fmt="o",
            color=COLORS["main"],
            ecolor=COLORS["main"],
            elinewidth=1.6,
            capsize=4,
            markersize=5.0,
            zorder=5,
        )

        star = p_star(p_val)
        stats.append({
            "region": region,
            "n_animals": n,
            "mean_delta_beta": mean_val,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_signflip_exhaustive_delta_beta_vs_zero": p_val,
            "p_signflip_plus1_conservative_delta_beta_vs_zero": p_plus1,
        })

        label = rf"mean $\Delta\kappa_I$={mean_val:.4f}" + "\n" + p_label(p_val)
        ax.text(
            x0, 0.995, label,
            ha="center", va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_STATS,
            linespacing=0.92,
            clip_on=False,
            zorder=8,
        )

        if star:
            ax.text(
                x0, 1.105, star,
                ha="center", va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=FONT_STAR,
                clip_on=False,
                zorder=9,
            )

    ax.axhline(0, color=COLORS["zero"], lw=1.0, ls="--", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cortex", "Thalamus"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Δ adaptive-Ising κ$_I$\n(propofol − fentanyl)")
    ax.set_title(
        "Adaptive-Ising κ$_I$ change",
        y=1.145,
        pad=0,
    )
    # Extra headroom keeps the highest observations and error bars below
    # the mean/p annotations placed at the top of the axes.
    set_padded_ylim(ax, all_vals, pad_frac=0.58, min_span=0.006)
    clean_ax(ax)
    return pd.DataFrame(stats)


def draw_panel_c(ax, tests, diffs, ladder_models):
    # Horizontal forest plot: one row per residualization model.
    # Significance stars are placed at the right edge of each CI row,
    # not above the points, to avoid collision with the title.
    rows = []
    for i, m in enumerate(ladder_models):
        gd = get_model_diff_values(diffs, m)
        vals = finite_values(gd["thalamus_minus_cortex_residual"].values) if not gd.empty else np.array([])
        mean_val = float(np.mean(vals)) if len(vals) else np.nan
        ci_lo, ci_hi = bootstrap_ci(vals, seed=20260629 + i)

        _, p_val, p_plus1, n = exact_signflip_one_sample(vals)
        dz, _ = paired_standardized_effect(vals)

        rows.append({
            "model": m,
            "display_model": MODEL_DISPLAY.get(m, m),
            "y": len(ladder_models) - 1 - i,
            "vals": vals,
            "mean": mean_val,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p_val,
            "p_plus1_conservative": p_plus1,
            "cohen_dz": dz,
            "n_animals": n,
        })

    # Set x-limits before placing stars, leaving a small right-side margin
    # for significance markers.
    vals_for_xlim = finite_values(
        [v for r in rows for v in r["vals"].tolist()]
        + [r["ci_lo"] for r in rows]
        + [r["ci_hi"] for r in rows]
    )
    if len(vals_for_xlim):
        xmin, xmax = vals_for_xlim.min(), vals_for_xlim.max()
        span = max(xmax - xmin, 0.35)
        x_left = xmin - 0.16 * span
        x_right = xmax + 0.48 * span
        ax.set_xlim(x_left, x_right)
    else:
        x_left, x_right = -0.1, 1.0
        ax.set_xlim(x_left, x_right)

    x0, x1 = ax.get_xlim()
    xspan = x1 - x0

    # Highlight the fully adjusted Ising-distance model row.
    if rows:
        final_y = rows[-1]["y"]
        ax.axhspan(final_y - 0.42, final_y + 0.42, color=COLORS["light"], zorder=-2)

    for r in rows:
        y = r["y"]
        vals = r["vals"]

        if len(vals):
            jit = jitter_positions(len(vals), width=0.055)
            ax.scatter(
                vals,
                np.full(len(vals), y) + jit,
                s=34,
                color=COLORS["dots"],
                alpha=0.62,
                linewidth=0,
                zorder=2,
            )

        if np.isfinite(r["mean"]):
            ax.errorbar(
                r["mean"],
                y,
                xerr=np.array([[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]]),
                fmt="o",
                color=COLORS["main"],
                ecolor=COLORS["main"],
                elinewidth=1.8,
                capsize=4,
                markersize=5.6,
                zorder=5,
            )

        # Keep the forest plot uncluttered; exact values are in the caption,
        # text, and exported statistics table.
        star = p_star(r["p"])
        if star:
            ax.text(
                x1 - 0.035 * xspan, y, star,
                ha="center", va="center", fontsize=FONT_STAR,
                color=COLORS["text"], zorder=6,
            )

    ax.axvline(0, color=COLORS["zero"], lw=1.0, ls="--", zorder=0)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["display_model"] for r in rows])
    ax.set_xlabel("Thalamus − cortex residualized Δ exponent")
    ax.set_title("Residualization forest plot", pad=6)

    clean_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return pd.DataFrame([{k: v for k, v in r.items() if k != "vals"} for r in rows])



def draw_panel_d(ax, matched_diffs, tests, final_model):
    """Panel d uses draw-averaged matched-channel values for the final model."""
    wide = get_model_diff_values(matched_diffs, final_model)

    for _, r in wide.iterrows():
        ax.plot([0, 1], [r["cortex"], r["thalamus"]], color=COLORS["line"], lw=1.25, zorder=1)
        ax.scatter(0, r["cortex"], s=58, color=COLORS["cortex"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(1, r["thalamus"], s=58, color=COLORS["thalamus"], edgecolor="white", linewidth=0.8, zorder=3)

    vals = wide["thalamus_minus_cortex_residual"].to_numpy(dtype=float)
    mean_val, p_exact, _, n = exact_signflip_one_sample(vals)
    ci_lo, ci_hi = bootstrap_ci(vals, n_boot=20000, seed=20260814)
    dz, _ = paired_standardized_effect(vals)
    displayed = {
        "mean": mean_val, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "cohen_dz": dz, "p_exact": p_exact, "n": n,
    }
    ax.set_title(
        "Adjusted residuals\n"
        + rf"Thal$-$Ctx = {mean_val:.3f}, {exact_p_label(p_exact)}",
        pad=6,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ctx", "Thal"])
    ax.set_ylabel("Residualized Δ exponent", labelpad=2)
    ax.axhline(0, color=COLORS["zero"], lw=1.0, ls="--")
    ax.set_xlim(-0.35, 1.35)
    set_padded_ylim(ax, wide[["cortex", "thalamus"]].values.ravel(), pad_frac=0.14)
    clean_ax(ax)

    out = wide[["animal_id", "group", "cortex", "thalamus"]].copy()
    out["thalamus_minus_cortex_residualized_delta"] = out["thalamus"] - out["cortex"]
    out["mean_thalamus_minus_cortex_residual"] = displayed["mean"]
    out["bootstrap_ci95_lo"] = displayed["ci_lo"]
    out["bootstrap_ci95_hi"] = displayed["ci_hi"]
    out["cohen_dz"] = displayed["cohen_dz"]
    out["p_signflip_exhaustive_two_sided"] = displayed["p_exact"]
    out["n_animals"] = displayed["n"]
    out["model"] = final_model
    return out


# ============================================================
# 4. Main
# ============================================================

def main():
    set_pub_style()

    for p in [EPOCH_CSV, TEST_CSV, DIFF_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    epoch = pd.read_csv(EPOCH_CSV)
    tests = pd.read_csv(TEST_CSV)
    diffs = pd.read_csv(DIFF_CSV)

    final_model = choose_first_available_model(diffs, FINAL_MODEL_CANDIDATES, "final Ising-distance model")
    ladder_models = choose_ladder_models(tests)

    state_beta = compute_statewise_ising_beta_means(epoch)
    beta_delta = compute_beta_delta_by_region(state_beta)

    print("=" * 100)
    print("[LOAD]")
    print("=" * 100)
    print(f"EPOCH rows  : {len(epoch)}")
    print(f"TEST rows   : {len(tests)}")
    print(f"DIFF rows   : {len(diffs)}")
    print(f"STATE       : {STATE} displayed as Propofol")
    print(f"FINAL MODEL : {final_model}")
    print(f"LADDER      : {ladder_models}")
    print(f"FIG SIZE    : {FIG_W_IN:.3f} x {FIG_H_IN:.3f} inch, matched to figure1.pdf")
    print(f"LATEX SCALE : {MANUSCRIPT_SCALE:.3f}; final minimum font target = {MIN_FINAL_FONT_PT:.1f} pt")
    print(f"OUT_DIR     : {OUT_DIR}")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")

    outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.0, 1.0],
        left=0.145,
        right=0.985,
        bottom=0.125,
        top=0.915,
        hspace=0.46,
    )

    # Use an explicit empty middle column. This makes panel b narrower
    # than panel a and creates a stable visual gap that is not consumed by
    # panel b's long y-axis label.
    top = outer[0].subgridspec(
        1, 3,
        width_ratios=[1.05, 0.25, 0.95],
        wspace=0.00,
    )

    bottom = outer[1].subgridspec(
        1, 2,
        width_ratios=[1.78, 1.00],
        wspace=0.24,
    )

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 2])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    draw_panel_a(ax_a, diffs, tests)
    beta_stats = draw_panel_b(ax_b, beta_delta)
    forest_stats = draw_panel_c(ax_c, tests, diffs, ladder_models)
    adjusted_stats = draw_panel_d(ax_d, diffs, tests, final_model)

    # Reviewer-requested animal-level statistics. These are kept separate from
    # the plotting tables so the manuscript statistics are transparent and
    # directly auditable.
    effect_stats, loo_stats, animal_contrasts = build_reviewer_requested_statistics(
        diffs, final_model
    )

    add_panel_label_fig(fig, ax_a, "a", dx=-0.032, dy=0.055)
    add_panel_label_fig(fig, ax_b, "b", dx=-0.032, dy=0.055)
    add_panel_label_fig(fig, ax_c, "c", dx=-0.032, dy=0.055)
    add_panel_label_fig(fig, ax_d, "d", dx=-0.032, dy=0.055)

    audit_final_font_sizes(fig)
    save_fixed_canvas(fig, "figurefinal")
    plt.close(fig)

    forest_stats.to_csv(OUT_DIR / "figurefinal_residualization_stats.csv", index=False, encoding="utf-8-sig")
    beta_stats.to_csv(OUT_DIR / "figurefinal_beta_delta_stats.csv", index=False, encoding="utf-8-sig")
    adjusted_stats.to_csv(OUT_DIR / "figurefinal_adjusted_paired_residuals_stats.csv", index=False, encoding="utf-8-sig")
    effect_stats.to_csv(
        OUT_DIR / "reviewer_requested_effect_sizes_and_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loo_stats.to_csv(
        OUT_DIR / "reviewer_requested_leave_one_animal_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    animal_contrasts.to_csv(
        OUT_DIR / "reviewer_requested_animal_level_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("[DONE]")
    print("=" * 100)
    print(f"Saved matched-size grouped figure to:\n{OUT_DIR}")
    print("Files:")
    for p in [OUT_DIR / "figurefinal.pdf", OUT_DIR / "figurefinal.png", OUT_DIR / "figurefinal.svg"]:
        print("  " + str(p))

    print("\nAdaptive-Ising beta delta stats:")
    print(beta_stats.to_string(index=False))
    print("\nResidualization forest stats:")
    print(forest_stats.to_string(index=False))
    print("\nAdjusted effect stats:")
    print(adjusted_stats.to_string(index=False))
    print("\nReviewer-requested effect sizes and confidence intervals:")
    print(effect_stats.to_string(index=False))
    print("\nLeave-one-animal-out sensitivity:")
    print(loo_stats.to_string(index=False))


if __name__ == "__main__":
    main()
