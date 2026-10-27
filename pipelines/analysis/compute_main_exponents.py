# -*- coding: utf-8 -*-
"""估计各试次REAL与ER的配对非周期指数并生成汇总图。每个试次内汇总各皮层层的LFP，在1至100 Hz拟合FOOOF，并输出回归流程所需的配对指数表、质量控制结果和描述性结果。 / Estimate paired REAL/ER aperiodic exponents and create summary figures.

Layer-resolved LFPs are aggregated within each trial. FOOOF is fitted over
1--100 Hz, and the paired exponent table consumed by the regression pipeline
is written together with quality-control and descriptive outputs.
"""

import os
import re
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

from scipy.signal import butter, sosfiltfilt, welch, resample
from scipy import stats
from fooof import FOOOF

import statsmodels.api as sm


# =========================
# 基本工具。 / Basic utilities
# =========================
def bandpass_filter(x, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def estimate_fs_from_t_ms(t_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    if t_ms.size < 3:
        raise ValueError("At least three time samples are required to estimate sampling rate.")
    dt_ms = float(np.median(np.diff(t_ms)))
    if dt_ms <= 0:
        raise ValueError("LFP time samples must be strictly increasing.")
    return 1000.0 / dt_ms


def load_lfp_npy(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"{path} must have shape (at least 2, time samples).")
    t_ms = np.asarray(arr[0, :], dtype=float)
    lfp = np.asarray(arr[1, :], dtype=float)
    return t_ms, lfp


def safe_align_truncate(signals, times):
    m = int(min(len(x) for x in signals))
    signals2 = [x[:m] for x in signals]
    times2 = [t[:m] for t in times]
    return signals2, times2[0]


def resample_to_fs(x, fs_in, fs_out):
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Input and output sampling rates must be positive.")
    n_out = int(round(x.shape[0] * fs_out / fs_in))
    if n_out < 10:
        raise ValueError("Resampled signal would contain fewer than ten samples.")
    y = resample(x, n_out, axis=0)
    return y, float(fs_out)


def smooth_ma(x, win=5):
    x = np.asarray(x, float)
    if win is None or int(win) <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    half = win // 2
    y = np.empty_like(x)
    for i in range(x.size):
        i0 = max(0, i - half)
        i1 = min(x.size, i + half + 1)
        y[i] = np.mean(x[i0:i1])
    return y


def compute_psd_1_100(
    eeg, fs,
    bp_low=0.1, bp_high=120.0,
    nperseg=2048, noverlap=1024,
    smooth_win_lin=1,
    smooth_win_log=9,
    eps=1e-20
):
    eeg = np.asarray(eeg, dtype=np.float64)
    eeg_filt = bandpass_filter(eeg, bp_low, bp_high, fs, order=4)

    f, Pxx = welch(eeg_filt, fs=fs, nperseg=nperseg, window="hann", noverlap=noverlap)

    mask = (f >= 0.1) & (f <= 120.0)
    f1 = f[mask]
    p1 = Pxx[mask]

    p_lin = smooth_ma(p1, smooth_win_lin)
    logp = np.log10(np.maximum(p_lin, eps))
    logp_s = smooth_ma(logp, smooth_win_log)
    p_smooth = 10.0 ** logp_s
    return f1, p_smooth


def run_fooof(
    f_range, psd_range, out_json_path, out_png_path,
    peak_width_limits=(1, 8),
    max_n_peaks=0,
    min_peak_height=0.10,
    aperiodic_mode="fixed",
    fit_range=(1, 100)
):
    fm_obj = FOOOF(
        peak_width_limits=list(peak_width_limits),
        max_n_peaks=int(max_n_peaks),
        min_peak_height=float(min_peak_height),
        aperiodic_mode=str(aperiodic_mode),
        verbose=False
    )
    fm_obj.fit(f_range, psd_range, list(fit_range))

    out_dir = os.path.dirname(out_json_path)
    os.makedirs(out_dir, exist_ok=True)
    name_noext = os.path.splitext(os.path.basename(out_json_path))[0]
    fm_obj.save(file_name=name_noext, file_path=out_dir, save_results=True)

    # 可选的逐试次调试图。 / optional per-trial debug plot
    plt.figure(figsize=(7, 4))
    fm_obj.plot()
    plt.title("FOOOF fit (1–100 Hz)")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300)
    plt.close()

    ap = fm_obj.aperiodic_params_
    if ap is None or len(ap) < 2:
        raise RuntimeError("FOOOF did not return a valid fixed aperiodic fit.")
    offset = float(ap[0])
    exponent = float(ap[1])
    r2 = float(fm_obj.r_squared_) if getattr(fm_obj, "r_squared_", None) is not None else np.nan
    return offset, exponent, r2


def list_trials(root_dir, pattern=r"^ER_trial\d+$", max_n=100):
    ds = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and re.match(pattern, name):
            ds.append(p)
    ds.sort()
    return ds[:int(max_n)] if max_n is not None else ds


def read_EI_ratio_from_summary(trial_dir, fname="summary.txt"):
    path = os.path.join(trial_dir, fname)
    if not os.path.isfile(path):
        return np.nan

    key_pat = re.compile(r"\b(EI_ratio|ER_EI_ratio)\b", re.IGNORECASE)
    num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if not key_pat.search(s):
                    continue
                nums = num_pat.findall(s)
                if nums:
                    try:
                        return float(nums[0])
                    except Exception:
                        return np.nan
    except Exception:
        return np.nan

    return np.nan


# =========================
# 试次处理。 / Trial processing
# =========================
def process_one_trial(trial_dir, out_tag="REAL"):
    trial_name = os.path.basename(trial_dir.rstrip("/"))

    def locate_layer_lfp(layer):
        # 第一个候选路径采用标准发布目录结构；第二个路径 / The first candidate is the canonical release layout.  The second
        # 用于读取历史双层目录结构中的已有结果， / keeps existing results from the historical double-layer layout
        # 无需复制或覆盖这些结果。 / readable without copying or overwriting them.
        candidates = (
            os.path.join(trial_dir, layer, "_lfp.npy"),
            os.path.join(trial_dir, layer, layer, "_lfp.npy"),
        )
        return next((path for path in candidates if os.path.isfile(path)), None)

    layer_paths = {layer: locate_layer_lfp(layer) for layer in ("L2_3", "L4", "L5", "L6")}
    for layer, path in layer_paths.items():
        if path is None:
            raise FileNotFoundError(
                f"[{trial_name}] missing {layer} LFP in canonical or legacy layout"
            )

    times, signals = [], []
    for k in ["L2_3", "L4", "L5", "L6"]:
        t_ms, lfp = load_lfp_npy(layer_paths[k])
        times.append(t_ms)
        signals.append(lfp)

    signals, t_ms_ref = safe_align_truncate(signals, times)
    fs_in = estimate_fs_from_t_ms(t_ms_ref)

    eeg_raw = signals[0] + signals[1] + signals[2] + signals[3]
    eeg_rs, fs_rs = resample_to_fs(eeg_raw, fs_in=fs_in, fs_out=1000.0)

    f_range, psd_range = compute_psd_1_100(
        eeg_rs, fs=fs_rs,
        bp_low=0.1, bp_high=120.0,
        nperseg=2048, noverlap=1024,
        smooth_win_lin=1,
        smooth_win_log=9,
        eps=1e-20
    )

    out_json = os.path.join(trial_dir, f"fooof_1_100_fixed_{out_tag}.json")
    out_png  = os.path.join(trial_dir, f"fooof_1_100_fixed_{out_tag}.png")

    offset, exponent, r2 = run_fooof(
        f_range, psd_range,
        out_json_path=out_json,
        out_png_path=out_png,
        peak_width_limits=(1, 8),
        max_n_peaks=0,
        min_peak_height=0.10,
        aperiodic_mode="fixed",
        fit_range=(1, 100)
    )

    return dict(trial=trial_name, offset=offset, exponent=exponent, r2=r2)


# =========================
# 稳健配对统计。 / Robust paired stats
# =========================
def paired_cohen_dz(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan
    d = x - y
    sd = d.std(ddof=1)
    return np.nan if sd == 0 else float(d.mean() / sd)


def paired_stats(x, y, n_perm=20000, seed=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    out = {"n": int(x.size)}
    if x.size < 2:
        out.update({"t": np.nan, "p_t": np.nan, "w": np.nan, "p_w": np.nan, "p_perm": np.nan, "dz": np.nan})
        return out

    d = x - y

    t, p = stats.ttest_rel(x, y, nan_policy="omit")
    out["t"] = float(t)
    out["p_t"] = float(p)

    try:
        w, pw = stats.wilcoxon(d, alternative="two-sided")
        out["w"] = float(w)
        out["p_w"] = float(pw)
    except Exception:
        out["w"] = np.nan
        out["p_w"] = np.nan

    rng = np.random.default_rng(seed)
    t_obs = abs(float(stats.ttest_1samp(d, 0.0).statistic))
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), d.size), replace=True)
    d_perm = signs * d[None, :]
    mean_perm = d_perm.mean(axis=1)
    sd_perm = d_perm.std(axis=1, ddof=1)
    t_perm = np.abs(mean_perm / (sd_perm / np.sqrt(d.size)))
    out["p_perm"] = float((np.sum(t_perm >= t_obs) + 1.0) / (t_perm.size + 1.0))

    out["dz"] = paired_cohen_dz(x, y)
    out["mean_real"] = float(np.mean(x)); out["sd_real"] = float(np.std(x, ddof=1))
    out["mean_er"]   = float(np.mean(y)); out["sd_er"]   = float(np.std(y, ddof=1))
    out["mean_d"]    = float(np.mean(d)); out["sd_d"]    = float(np.std(d, ddof=1))
    return out


# =========================
# 发表用绘图工具。 / Publication plot helpers
# =========================
def nature_palette():
    return {"REAL": "#1f77b4", "ER": "#ff7f0e", "INK": "#1a1a1a"}


def add_pvalue_bracket(ax, x1, x2, y, h, text, lw=1.0, color="#1a1a1a", fs=None):
    if fs is None:
        fs = plt.rcParams.get("font.size", 15)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=color, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fs, color=color)


def plot_paired_panel(ax, real_vals, er_vals, ylabel, pval, colors):
    real_vals = np.asarray(real_vals, float)
    er_vals   = np.asarray(er_vals, float)
    m = np.isfinite(real_vals) & np.isfinite(er_vals)
    real_vals = real_vals[m]
    er_vals   = er_vals[m]

    x1, x2 = 1.0, 2.0
    rng = np.random.default_rng(0)
    jit = (rng.random(real_vals.size) - 0.5) * 0.08

    for i in range(real_vals.size):
        ax.plot([x1, x2], [real_vals[i], er_vals[i]],
                color=colors["INK"], alpha=0.10, lw=0.6, zorder=1)

    ax.scatter(np.full(real_vals.size, x1) + jit, real_vals,
               s=18, alpha=0.30, color=colors["REAL"], edgecolors="none", zorder=2)
    ax.scatter(np.full(er_vals.size, x2) + jit, er_vals,
               s=18, alpha=0.30, color=colors["ER"], edgecolors="none", zorder=2)

    def mean_ci(a):
        mu = float(np.mean(a))
        if a.size < 2:
            return mu, np.nan
        se = float(stats.sem(a, nan_policy="omit"))
        return mu, 1.96 * se

    mu1, ci1 = mean_ci(real_vals)
    mu2, ci2 = mean_ci(er_vals)

    for x, mu, ci, c in [(x1, mu1, ci1, colors["REAL"]), (x2, mu2, ci2, colors["ER"])]:
        ax.errorbar([x], [mu], yerr=[ci],
                    fmt="o", ms=6.8, mfc=c, mec=colors["INK"], mew=0.6,
                    ecolor=colors["INK"], elinewidth=1.3, capsize=3.0, capthick=1.3,
                    zorder=4)

    ax.set_xlim(0.55, 2.45)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["MICrONs", "ER"])
    ax.set_ylabel(ylabel)

    ymax = float(np.max([np.max(real_vals), np.max(er_vals)]))
    ymin = float(np.min([np.min(real_vals), np.min(er_vals)]))
    yr = ymax - ymin if ymax > ymin else 1.0
    y = ymax + 0.12 * yr
    h = 0.03 * yr
    add_pvalue_bracket(ax, x1, x2, y, h, f"p = {pval:.2g}", fs=None)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def ols_fit_with_inference(x, y):
    """OLS模型：y ~ 1 + x。返回包含斜率、截距及推断结果的fit_info字典，以及statsmodels模型结果。 / 
    OLS: y ~ 1 + x
    Returns:
      fit_info dict (slope/intercept + inference)
      model (statsmodels result)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        raise RuntimeError("At least three finite observations are required for OLS.")

    X = sm.add_constant(x)  # 设计矩阵的常数项与自变量。 / [1, x]
    model = sm.OLS(y, X).fit()

    intercept = float(model.params[0])
    slope = float(model.params[1])

    se_intercept = float(model.bse[0])
    se_slope = float(model.bse[1])

    t_intercept = float(model.tvalues[0])
    t_slope = float(model.tvalues[1])

    p_intercept = float(model.pvalues[0])
    p_slope = float(model.pvalues[1])

    ci = model.conf_int(alpha=0.05)  # 95%置信区间。 / 95% CI
    ci_intercept_lo, ci_intercept_hi = float(ci[0, 0]), float(ci[0, 1])
    ci_slope_lo, ci_slope_hi = float(ci[1, 0]), float(ci[1, 1])

    fit_info = {
        "n": int(x.size),
        "df_resid": float(model.df_resid),
        "intercept": intercept,
        "slope": slope,
        "se_intercept": se_intercept,
        "se_slope": se_slope,
        "t_intercept": t_intercept,
        "t_slope": t_slope,
        "p_intercept": p_intercept,
        "p_slope": p_slope,
        "ci95_intercept_lo": ci_intercept_lo,
        "ci95_intercept_hi": ci_intercept_hi,
        "ci95_slope_lo": ci_slope_lo,
        "ci95_slope_hi": ci_slope_hi,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }
    return fit_info, model


def plot_delta_vs_Ilevel(ax, I_level, d_exp, colors, legend_outside=True):
    """关系图：横轴为兴奋/抑制比I_level，纵轴为指数差ER减REAL；散点按纵轴正负着色，添加statsmodels的OLS回归线，将图例置于图外。返回fit_info字典。 / 
    Relationship plot:
    x = EI ratio (I_level)
    y = Δexponent (ER - REAL)

    - scatter colored by sign of y
    - OLS regression line (statsmodels)
    - legend outside

    Returns fit_info dict.
    """
    x = np.asarray(I_level, float)
    y = np.asarray(d_exp, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        raise RuntimeError("At least three finite observations are required for log-log OLS.")

    pos = y > 0
    neg = y < 0
    zer = ~(pos | neg)

    if np.any(pos):
        ax.scatter(x[pos], y[pos], s=18, alpha=0.35, color=colors["ER"], edgecolors="none", zorder=2)
    if np.any(neg):
        ax.scatter(x[neg], y[neg], s=18, alpha=0.35, color=colors["REAL"], edgecolors="none", zorder=2)
    if np.any(zer):
        ax.scatter(x[zer], y[zer], s=18, alpha=0.35, color=colors["INK"], edgecolors="none", zorder=2)

    ax.axhline(0.0, color=colors["INK"], lw=1.0, alpha=0.25, zorder=1)

    fit_info, model = ols_fit_with_inference(x, y)

    x_line = np.linspace(fit_info["x_min"], fit_info["x_max"], 200)
    y_line = fit_info["slope"] * x_line + fit_info["intercept"]
    ax.plot(x_line, y_line, color=colors["INK"], lw=1.4, alpha=0.85, zorder=3)

    ax.set_xlabel("EI ratio")
    ax.set_ylabel(r"$\Delta$ exponent")

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=colors["ER"], markeredgecolor="none", alpha=0.85,
               label="ER steeper"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=colors["REAL"], markeredgecolor="none", alpha=0.85,
               label="MICrONs steeper"),
    ]

    if legend_outside:
        ax.legend(
            handles=handles,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    else:
        ax.legend(handles=handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    return fit_info


# =========================
# 主程序。 / Main
# =========================
if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("PROJECT_ROOT", repository_root / "work")).resolve()
    root = str(project_root / "ALLEN")

    real_root = os.environ.get("SAVE_ROOT_REAL", os.path.join(root, "save", "final_real"))
    er_root = os.environ.get("SAVE_ROOT_ER", os.path.join(root, "save", "final_ER"))
    summary_root = os.environ.get("TRIALS_ROOT", os.path.join(root, "file", "final_trial"))

    if not os.path.isdir(real_root):
        raise RuntimeError(f"REAL LFP root does not exist: {real_root}")
    if not os.path.isdir(er_root):
        raise RuntimeError(f"ER LFP root does not exist: {er_root}")
    if not os.path.isdir(summary_root):
        raise RuntimeError(f"Trial-summary root does not exist: {summary_root}")

    real_trials = list_trials(real_root, pattern=r"^ER_trial\d+$", max_n=100)
    er_trials   = list_trials(er_root,   pattern=r"^ER_trial\d+$", max_n=100)
    if len(real_trials) == 0 or len(er_trials) == 0:
        raise RuntimeError("No paired REAL/ER LFP trial directories were found.")

    real_map = {os.path.basename(p): p for p in real_trials}
    er_map   = {os.path.basename(p): p for p in er_trials}

    summary_trials = list_trials(summary_root, pattern=r"^ER_trial\d+$", max_n=1000)
    summary_map = {os.path.basename(p): p for p in summary_trials}

    common_names = sorted(set(real_map.keys()) & set(er_map.keys()))
    if len(common_names) == 0:
        raise RuntimeError("REAL and ER LFP roots have no trial identifiers in common.")

    results_tag = os.environ.get("RESULTS_TAG", "").strip("/\\")
    results_base = os.path.join(root, "results", results_tag) if results_tag else os.path.join(root, "results")
    out_dir = os.path.join(results_base, "fooof_group_compare")
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 逐试次运行FOOOF。 / Run FOOOF per trial
    # =========================
    real_rows, er_rows = [], []
    for i, name in enumerate(common_names, start=1):
        print(f"[{i:02d}/{len(common_names)}] {name}")
        r = process_one_trial(real_map[name], out_tag="REAL")
        e = process_one_trial(er_map[name], out_tag="ER")
        real_rows.append(r)
        er_rows.append(e)
        print(f"  REAL exp={r['exponent']:.4f}, r2={r['r2']:.4f}")
        print(f"  ER   exp={e['exponent']:.4f}, r2={e['r2']:.4f}")
        gc.collect()

    df_real = pd.DataFrame(real_rows).sort_values("trial").reset_index(drop=True)
    df_er   = pd.DataFrame(er_rows).sort_values("trial").reset_index(drop=True)
    if not np.all(df_real["trial"].values == df_er["trial"].values):
        raise RuntimeError("REAL and ER exponent tables are not aligned by trial identifier.")

    df_real.to_csv(os.path.join(out_dir, "fooof_trial_params_REAL.csv"), index=False, encoding="utf-8-sig")
    df_er.to_csv(os.path.join(out_dir, "fooof_trial_params_ER.csv"), index=False, encoding="utf-8-sig")
    df_two_columns = df_real[["trial", "exponent"]].rename(
        columns={"exponent": "REAL_exponent"}
    ).merge(
        df_er[["trial", "exponent"]].rename(columns={"exponent": "ER_exponent"}),
        on="trial",
        validate="one_to_one",
    )
    df_two_columns.to_csv(
        os.path.join(out_dir, "fooof_exponent_REAL_and_ER_two_columns.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # =========================
    # 统计分析：配对非周期指数。 / Stats (paired exponent)
    # =========================
    stats_exp = paired_stats(df_real["exponent"].values, df_er["exponent"].values, n_perm=20000, seed=0)

    pd.DataFrame([{"metric": "aperiodic_exponent", **stats_exp}]).to_csv(
        os.path.join(out_dir, "fooof_group_stats_exponent_PAIRED_REAL_vs_ER.csv"),
        index=False, encoding="utf-8-sig"
    )

    # =========================
    # 构建配对表并加入兴奋/抑制比。 / Build pair table + EI_ratio
    # =========================
    rows = []
    missing_summary = 0
    er_exp_lookup = dict(zip(df_er["trial"].astype(str).values, df_er["exponent"].astype(float).values))

    for _, rr in df_real.iterrows():
        trial = str(rr["trial"])
        if trial not in er_exp_lookup:
            continue

        exp_real = float(rr["exponent"])
        exp_er = float(er_exp_lookup[trial])

        if trial in summary_map:
            I_level = read_EI_ratio_from_summary(summary_map[trial], fname="summary.txt")
        else:
            I_level = np.nan
            missing_summary += 1

        rows.append({
            "trial": trial,
            "exponent_REAL": exp_real,
            "exponent_ER": exp_er,
            "d_exponent_ER_minus_REAL": exp_er - exp_real,
            "I_level": I_level,
        })

    df_pair = pd.DataFrame(rows)
    df_pair = df_pair[
        np.isfinite(df_pair["I_level"]) &
        np.isfinite(df_pair["d_exponent_ER_minus_REAL"])
    ].copy()

    pair_csv = os.path.join(out_dir, "pair_table_exponent_and_EI_from_summary.csv")
    df_pair.to_csv(pair_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {pair_csv}")
    print(f"[Info] valid pairs for Fig3: n={len(df_pair)}")
    if missing_summary > 0:
        print(f"[Warn] missing summary folders in {summary_root}: {missing_summary} trials")

    # =========================
    # 固定字体为Times New Roman。 / Font lock: Times New Roman
    # =========================
    font_path = os.environ.get("MANUSCRIPT_FONT_FILE", "")
    if font_path and os.path.isfile(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    else:
        plt.rcParams["font.family"] = "serif"

    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    print("[FontCheck] rcParams['font.family'] =", plt.rcParams.get("font.family"))

    colors = nature_palette()

    # =========================
    # 图2：配对非周期指数。 / Figure2: paired exponent
    # =========================
    fig2, ax2 = plt.subplots(figsize=(3.0, 3.0), dpi=600)
    plot_paired_panel(
        ax2,
        df_real["exponent"].values,
        df_er["exponent"].values,
        ylabel="Aperiodic exponent",
        pval=stats_exp["p_t"],
        colors=colors
    )
    fig2.tight_layout(pad=0.15)

    fig2_png = os.path.join(out_dir, "fooof_exponent_REAL_vs_ER.png")
    fig2_pdf = os.path.join(out_dir, "fooof_exponent_REAL_vs_ER.pdf")
    fig2.savefig(fig2_png, dpi=600, bbox_inches="tight", pad_inches=0)
    fig2.savefig(fig2_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)

    print(f"[Saved] {fig2_png}")
    print(f"[Saved] {fig2_pdf}")

    # =========================
    # 图3：指数差与抑制水平的关系。 / Figure3: Δexponent vs I level
    # =========================
    fig3, ax3 = plt.subplots(figsize=(3.0, 3.0), dpi=600)
    fig3.subplots_adjust(right=0.78)

    fit_info = plot_delta_vs_Ilevel(
        ax3,
        df_pair["I_level"].values,
        df_pair["d_exponent_ER_minus_REAL"].values,
        colors=colors,
        legend_outside=True
    )

    fig3.tight_layout(pad=0.15)

    fig3_png = os.path.join(out_dir, "fooof_delta_exponent_vs_Ilevel.png")
    fig3_pdf = os.path.join(out_dir, "fooof_delta_exponent_vs_Ilevel.pdf")
    fig3.savefig(fig3_png, dpi=600, bbox_inches="tight", pad_inches=0)
    fig3.savefig(fig3_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig3)

    print(f"[Saved] {fig3_png}")
    print(f"[Saved] {fig3_pdf}")

    # =========================
    # 保存图3的OLS拟合信息，新增统计推断结果。 / Save Fig3 OLS fit info (NEW, with inference)
    # =========================
    fit_csv = os.path.join(out_dir, "fig3_ols_fit_params.csv")
    pd.DataFrame([{
        **fit_info,
        "x_label": "EI ratio",
        "y_label": "Δ exponent (ER - MICrONs)",
        "model": "OLS(y ~ 1 + x)",
        "test": "two-sided t-test for coefficients (statsmodels)",
        "alpha": 0.05,
    }]).to_csv(fit_csv, index=False, encoding="utf-8-sig")

    fit_txt = os.path.join(out_dir, "fig3_ols_fit_params.txt")
    with open(fit_txt, "w", encoding="utf-8") as f:
        f.write("OLS fit for Fig3: y = intercept + slope * x\n")
        f.write(f"n = {fit_info['n']}\n")
        f.write(f"R^2 = {fit_info['r2']:.10g} (adj R^2 = {fit_info['adj_r2']:.10g})\n")
        f.write(f"x range = [{fit_info['x_min']:.10g}, {fit_info['x_max']:.10g}]\n\n")

        f.write("Slope (two-sided t-test):\n")
        f.write(f"  slope = {fit_info['slope']:.10g}\n")
        f.write(f"  SE = {fit_info['se_slope']:.10g}\n")
        f.write(f"  t = {fit_info['t_slope']:.10g}\n")
        f.write(f"  p = {fit_info['p_slope']:.10g}\n")
        f.write(f"  95% CI = [{fit_info['ci95_slope_lo']:.10g}, {fit_info['ci95_slope_hi']:.10g}]\n\n")

        f.write("Intercept (two-sided t-test):\n")
        f.write(f"  intercept = {fit_info['intercept']:.10g}\n")
        f.write(f"  SE = {fit_info['se_intercept']:.10g}\n")
        f.write(f"  t = {fit_info['t_intercept']:.10g}\n")
        f.write(f"  p = {fit_info['p_intercept']:.10g}\n")
        f.write(f"  95% CI = [{fit_info['ci95_intercept_lo']:.10g}, {fit_info['ci95_intercept_hi']:.10g}]\n\n")

        f.write("x = EI ratio; y = Δ exponent (ER - MICrONs)\n")

    print(f"[Saved] {fit_csv}")
    print(f"[Saved] {fit_txt}")
    print("\nDone.")
