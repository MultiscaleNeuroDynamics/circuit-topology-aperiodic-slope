# -*- coding: utf-8 -*-
"""Read-only m=0.99 REAL/ER/DP LFP + E/I analysis for the matched 40 trials."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, resample, sosfiltfilt, welch
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

try:
    from fooof import FOOOF
except ImportError as exc:
    raise SystemExit(
        "Missing package 'fooof'. Activate the neuron environment and install it first."
    ) from exc


TRIAL_IDS = [
    1, 4, 6, 9, 11, 14, 16, 19, 21, 24,
    26, 29, 31, 34, 37, 39, 42, 44, 47, 49,
    52, 54, 57, 59, 62, 64, 67, 70, 72, 75,
    77, 80, 82, 85, 87, 90, 92, 95, 97, 100,
]
LAYERS = ("L2_3", "L4", "L5", "L6")
TOPOLOGIES = ("REAL", "DP", "ER")


def parse_args():
    repository_root = Path(__file__).resolve().parents[2]
    default_project = os.environ.get("PROJECT_ROOT", repository_root / "work")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=default_project)
    p.add_argument("--real-lfp-root")
    p.add_argument("--er-lfp-root")
    p.add_argument("--dp-lfp-root")
    p.add_argument("--original-trials-root")
    p.add_argument("--dp-trials-root")
    p.add_argument("--output-root")
    p.add_argument("--force-fooof", action="store_true", help="Recompute cached FOOOF rows")
    p.add_argument("--save-fit-plots", action="store_true", help="Save 120 diagnostic fits")
    return p.parse_args()


def trial_name(trial_id: int) -> str:
    return f"ER_trial{trial_id:04d}"


def locate_lfp(root: Path, trial_id: int, layer: str) -> Path | None:
    base = root / trial_name(trial_id) / layer
    candidates = (base / "_lfp.npy", base / layer / "_lfp.npy")
    valid = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if len(valid) > 1:
        # Prefer the double-layer layout used by the current simulators.
        return candidates[1]
    return valid[0] if valid else None


def choose_lfp_root(explicit: str | None, candidates: list[Path], topology: str) -> Path:
    """Select the first candidate containing the first matched trial/layer LFP."""
    if explicit:
        return Path(explicit).resolve()
    for candidate in candidates:
        candidate = candidate.resolve()
        if locate_lfp(candidate, TRIAL_IDS[0], LAYERS[0]) is not None:
            print(f"[PATH] {topology} LFP root: {candidate}")
            return candidate
    checked = "\n".join(f"- {p.resolve()}" for p in candidates)
    raise FileNotFoundError(
        f"Could not auto-detect the {topology} LFP root. Checked:\n{checked}\n"
        f"Use --{topology.lower()}-lfp-root to specify it explicitly."
    )


def validate_inputs(lfp_roots: dict[str, Path], original_root: Path, dp_root: Path):
    problems = []
    resolved = {}
    for topo, root in lfp_roots.items():
        for tid in TRIAL_IDS:
            for layer in LAYERS:
                p = locate_lfp(root, tid, layer)
                if p is None:
                    problems.append(f"missing LFP: {topo}/{trial_name(tid)}/{layer}/_lfp.npy")
                else:
                    resolved[(topo, tid, layer)] = p

    for tid in TRIAL_IDS:
        src = original_root / trial_name(tid)
        for name in ("microns_patch_synapse_spikes_real_ms.csv", "microns_patch_synapse_spikes_ER_ms.csv"):
            p = src / name
            if not p.is_file() or p.stat().st_size == 0:
                problems.append(f"missing source spike table: {p}")
        p = dp_root / trial_name(tid) / "microns_patch_synapse_spikes_ER_ms.csv"
        if not p.is_file() or p.stat().st_size == 0:
            problems.append(f"missing DP spike table: {p}")

    if problems:
        preview = "\n".join(f"- {x}" for x in problems[:30])
        raise RuntimeError(f"Input validation failed ({len(problems)} problems):\n{preview}")
    print("[VALID] 480 non-empty LFP files and all matched spike tables are present.")
    return resolved


def load_lfp(path: Path):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 10:
        raise ValueError(f"Unexpected LFP shape {arr.shape}: {path}")
    t = np.asarray(arr[0], dtype=float)
    x = np.asarray(arr[1], dtype=float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError(f"Non-finite LFP data: {path}")
    return t, x


def moving_average(x, width=9):
    if width <= 1:
        return x
    width = int(width) | 1
    return np.convolve(x, np.ones(width) / width, mode="same")


def fit_one_lfp(layer_paths: list[Path], save_plot: Path | None = None):
    times, signals = zip(*(load_lfp(p) for p in layer_paths))
    n = min(map(len, signals))
    t = np.asarray(times[0][:n], float)
    if any(len(tt) < n or not np.allclose(tt[:n], t, rtol=0, atol=1e-7) for tt in times):
        raise ValueError(f"Layer time vectors do not match: {layer_paths[0].parent.parent}")
    dt_ms = float(np.median(np.diff(t)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("Invalid time step")
    fs_in = 1000.0 / dt_ms
    signal = np.sum(np.vstack([x[:n] for x in signals]), axis=0)

    # Preserve the settings used for the revised 1--100 Hz manuscript analysis.
    nyquist = fs_in / 2.0
    if nyquist <= 120.0:
        raise ValueError(f"Sampling rate {fs_in:g} Hz is too low for the 120-Hz prefilter")
    sos = butter(4, [0.1, 120.0], btype="bandpass", fs=fs_in, output="sos")
    signal = sosfiltfilt(sos, signal)
    n_out = int(round(signal.size * 1000.0 / fs_in))
    signal = resample(signal, n_out)
    freqs, power = welch(signal, fs=1000.0, window="hann", nperseg=2048, noverlap=1024)
    mask = (freqs >= 0.1) & (freqs <= 120.0)
    freqs = freqs[mask]
    power = power[mask]
    power = 10.0 ** moving_average(np.log10(np.maximum(power, 1e-20)), 9)

    fm = FOOOF(
        peak_width_limits=[1, 8], max_n_peaks=0, min_peak_height=0.10,
        aperiodic_mode="fixed", verbose=False,
    )
    fm.fit(freqs, power, [1, 100])
    ap = np.asarray(fm.aperiodic_params_, float)
    if ap.size < 2 or not np.all(np.isfinite(ap[:2])):
        raise RuntimeError("FOOOF failed to return finite fixed-mode parameters")
    if save_plot is not None:
        save_plot.parent.mkdir(parents=True, exist_ok=True)
        fm.plot()
        plt.title("FOOOF fit (1–100 Hz)")
        plt.tight_layout()
        plt.savefig(save_plot, dpi=180)
        plt.close()
    return float(ap[0]), float(ap[1]), float(fm.r_squared_), float(fm.error_)


def load_synapse_maps(allen_root: Path):
    syn_path = allen_root / "file" / "proofread_cells_synapses.csv"
    type_path = allen_root / "file" / "proofread_cells_celltypes_all.csv"
    syn = pd.read_csv(syn_path, dtype=str)
    syn_col = "synapse_id" if "synapse_id" in syn.columns else syn.columns[0]
    if "pre_pt_root_id" not in syn.columns:
        raise RuntimeError(f"pre_pt_root_id is absent from {syn_path}")
    syn2pre = dict(zip(syn[syn_col].str.strip(), syn["pre_pt_root_id"].str.strip()))
    types = pd.read_csv(type_path, header=None, dtype=str)
    root2type = dict(zip(types.iloc[:, 0].str.strip(), types.iloc[:, 1].str.strip()))
    return syn2pre, root2type


def spike_counts(path: Path):
    df = pd.read_csv(path, header=None, dtype=str)
    ids = df.iloc[:, 0].fillna("").str.strip().to_numpy()
    if df.shape[1] == 1:
        return ids, np.zeros(len(ids), dtype=int)
    values = df.iloc[:, 1:].replace(r"^\s*$", np.nan, regex=True)
    return ids, values.notna().sum(axis=1).to_numpy(dtype=int)


def ei_from_real(path: Path, syn2pre: dict, root2type: dict):
    ids, counts = spike_counts(path)
    e = i = unknown = 0
    for sid, count in zip(ids, counts):
        pre = syn2pre.get(sid)
        cell_type = root2type.get(pre) if pre is not None else None
        label = str(cell_type).lower()
        if "excitatory" in label:
            e += int(count)
        elif "inhibitory" in label:
            i += int(count)
        else:
            unknown += 1
    return e, i, unknown


def ei_from_control(path: Path):
    ids, counts = spike_counts(path)
    e = i = unknown = 0
    for sid, count in zip(ids, counts):
        sid = sid.strip()
        if sid.startswith("10"):
            e += int(count)
        elif sid.startswith("00"):
            i += int(count)
        else:
            unknown += 1
    return e, i, unknown


def ratio(e, i):
    return float(e) / float(i) if i else (math.inf if e else math.nan)


def holm(pvalues):
    out = np.full(len(pvalues), np.nan)
    ok = np.isfinite(pvalues)
    if np.any(ok):
        out[ok] = multipletests(np.asarray(pvalues)[ok], method="holm")[1]
    return out


def paired_stats(long_df: pd.DataFrame, value: str):
    wide = long_df.pivot(index="trial", columns="topology", values=value)
    rows = []
    for a, b in (("REAL", "DP"), ("REAL", "ER"), ("DP", "ER")):
        sub = wide[[a, b]].dropna()
        x, y = sub[a].to_numpy(float), sub[b].to_numpy(float)
        d = x - y
        sem = stats.sem(d) if len(d) > 1 else np.nan
        ci = stats.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=sem) if len(d) > 1 else (np.nan, np.nan)
        t_res = stats.ttest_rel(x, y)
        try:
            w_res = stats.wilcoxon(x, y, alternative="two-sided")
            wp = float(w_res.pvalue)
        except ValueError:
            wp = np.nan
        rows.append({
            "metric": value, "contrast": f"{a}-{b}", "n_pairs": len(d),
            f"mean_{a}": np.mean(x), f"mean_{b}": np.mean(y),
            "mean_difference": np.mean(d), "ci95_low": ci[0], "ci95_high": ci[1],
            "paired_t": t_res.statistic, "paired_t_p": t_res.pvalue,
            "wilcoxon_p": wp,
            "cohen_dz": np.mean(d) / np.std(d, ddof=1) if np.std(d, ddof=1) else np.nan,
        })
    ans = pd.DataFrame(rows)
    ans["paired_t_p_holm"] = holm(ans["paired_t_p"].to_numpy(float))
    ans["wilcoxon_p_holm"] = holm(ans["wilcoxon_p"].to_numpy(float))
    return ans


def robust_fit(formula: str, data: pd.DataFrame):
    model = smf.ols(formula, data=data).fit()
    robust = model.get_robustcov_results(cov_type="cluster", groups=data["trial"])
    ci = np.asarray(robust.conf_int(), float)
    coef = pd.DataFrame({
        "term": robust.model.exog_names, "coef": robust.params, "se_cluster": robust.bse,
        "t": robust.tvalues, "p": robust.pvalues,
        "ci95_low": ci[:, 0], "ci95_high": ci[:, 1],
    })
    fit = {"formula": formula, "n_obs": robust.nobs, "n_trials": data["trial"].nunique(),
           "r2": model.rsquared, "adj_r2": model.rsquared_adj, "aic": model.aic, "bic": model.bic}
    return model, robust, coef, fit


def linear_contrast(robust, weights: dict[str, float], label: str):
    names = robust.model.exog_names
    vector = np.array([weights.get(name, 0.0) for name in names], float)
    test = robust.t_test(vector)
    ci = np.asarray(test.conf_int(), float).ravel()
    return {"contrast": label, "estimate": float(np.asarray(test.effect).ravel()[0]),
            "se_cluster": float(np.asarray(test.sd).ravel()[0]),
            "t": float(np.asarray(test.tvalue).ravel()[0]),
            "p": float(np.asarray(test.pvalue).ravel()[0]),
            "ci95_low": ci[0], "ci95_high": ci[1]}


def run_models(df: pd.DataFrame, out: Path):
    data = df.copy()
    data["topology"] = pd.Categorical(data["topology"], categories=["REAL", "DP", "ER"])
    specs = {
        "M1_EI_only": "exponent ~ EI_ratio",
        "M2_topology_only": "exponent ~ C(topology)",
        "M3_EI_plus_topology": "exponent ~ EI_ratio + C(topology)",
        "M4_interaction": "exponent ~ EI_ratio * C(topology)",
    }
    coef_all, fits, contrasts = [], [], []
    for name, formula in specs.items():
        _, rob, coef, fit = robust_fit(formula, data)
        coef.insert(0, "model", name)
        coef_all.append(coef)
        fit["model"] = name
        fits.append(fit)
        if name in ("M2_topology_only", "M3_EI_plus_topology"):
            contrasts.extend([
                {"model": name, **linear_contrast(rob, {"C(topology)[T.DP]": -1}, "REAL-DP")},
                {"model": name, **linear_contrast(rob, {"C(topology)[T.ER]": -1}, "REAL-ER")},
                {"model": name, **linear_contrast(rob, {"C(topology)[T.DP]": 1, "C(topology)[T.ER]": -1}, "DP-ER")},
            ])
        if name == "M4_interaction":
            contrasts.extend([
                {"model": name, **linear_contrast(rob, {"EI_ratio": 1}, "REAL_EI_slope")},
                {"model": name, **linear_contrast(rob, {"EI_ratio": 1, "EI_ratio:C(topology)[T.DP]": 1}, "DP_EI_slope")},
                {"model": name, **linear_contrast(rob, {"EI_ratio": 1, "EI_ratio:C(topology)[T.ER]": 1}, "ER_EI_slope")},
                {"model": name, **linear_contrast(rob, {"EI_ratio:C(topology)[T.DP]": -1}, "REAL-DP_slope")},
                {"model": name, **linear_contrast(rob, {"EI_ratio:C(topology)[T.ER]": -1}, "REAL-ER_slope")},
                {"model": name, **linear_contrast(rob, {"EI_ratio:C(topology)[T.DP]": 1, "EI_ratio:C(topology)[T.ER]": -1}, "DP-ER_slope")},
            ])
    coef_df = pd.concat(coef_all, ignore_index=True)
    contrast_df = pd.DataFrame(contrasts)
    for model_name in contrast_df["model"].unique():
        idx = contrast_df["model"].eq(model_name)
        contrast_df.loc[idx, "p_holm_within_model"] = holm(contrast_df.loc[idx, "p"].to_numpy(float))
    coef_df.to_csv(out / "model_coefficients_cluster_trial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).to_csv(out / "model_fit_metrics.csv", index=False, encoding="utf-8-sig")
    contrast_df.to_csv(out / "model_planned_contrasts.csv", index=False, encoding="utf-8-sig")
    return coef_df, contrast_df


def make_plots(df: pd.DataFrame, out: Path):
    colors = {"REAL": "#1976D2", "DP": "#2E7D32", "ER": "#D32F2F"}
    wide = df.pivot(index="trial", columns="topology", values="exponent")[list(TOPOLOGIES)]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for _, row in wide.iterrows():
        ax.plot(range(3), row.values, color="0.75", lw=0.7, alpha=0.6)
    for pos, topo in enumerate(TOPOLOGIES):
        vals = wide[topo].dropna().to_numpy()
        ax.scatter(np.full(vals.size, pos), vals, s=20, alpha=0.75, color=colors[topo], zorder=3)
        ax.errorbar(pos, vals.mean(), yerr=stats.sem(vals), fmt="o", ms=8, color="black", capsize=4, zorder=4)
    ax.set_xticks(range(3), TOPOLOGIES)
    ax.set_ylabel("Aperiodic exponent (1–100 Hz)")
    ax.set_title("Matched 40-trial topology comparison")
    fig.tight_layout()
    fig.savefig(out / "paired_exponent_REAL_DP_ER.png", dpi=300)
    fig.savefig(out / "paired_exponent_REAL_DP_ER.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, topo in zip(axes, TOPOLOGIES):
        sub = df[df["topology"] == topo]
        x, y = sub["EI_ratio"].to_numpy(float), sub["exponent"].to_numpy(float)
        ax.scatter(x, y, color=colors[topo], alpha=0.8, s=28)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid, intercept + slope * grid, color="black", lw=1.5)
        ax.set_title(topo)
        ax.set_xlabel("Realized synaptic E/I ratio")
    axes[0].set_ylabel("Aperiodic exponent (1–100 Hz)")
    fig.tight_layout()
    fig.savefig(out / "exponent_vs_EI_by_topology.png", dpi=300)
    fig.savefig(out / "exponent_vs_EI_by_topology.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    project = Path(args.project_root).resolve()
    allen = project / "ALLEN"
    # Original manuscript m=0.99 outputs predate the m-specific directory and
    # normally live directly under ALLEN/save.  The second candidates retain
    # compatibility with later directory layouts.
    lfp_roots = {
        "REAL": choose_lfp_root(args.real_lfp_root, [
            allen / "save" / "final_real",
            allen / "save" / "m_0.99" / "final_real",
        ], "REAL"),
        "DP": choose_lfp_root(args.dp_lfp_root, [
            allen / "save" / "m_0.99" / "final_DP40",
            allen / "save" / "final_DP40",
        ], "DP"),
        "ER": choose_lfp_root(args.er_lfp_root, [
            allen / "save" / "final_ER",
            allen / "save" / "m_0.99" / "final_ER",
        ], "ER"),
    }
    original_root = Path(args.original_trials_root or allen / "file" / "final_trial").resolve()
    dp_root = Path(args.dp_trials_root or allen / "file" / "add_trial" / "m_0.99_DP40").resolve()
    out = Path(args.output_root or allen / "results" / "m_0.99" / "real_er_dp40").resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = validate_inputs(lfp_roots, original_root, dp_root)

    config = {"project_root": str(project), "lfp_roots": {k: str(v) for k, v in lfp_roots.items()},
              "original_trials_root": str(original_root), "dp_trials_root": str(dp_root),
              "output_root": str(out), "trial_ids": TRIAL_IDS,
              "fooof": {"fit_range_hz": [1, 100], "aperiodic_mode": "fixed", "max_n_peaks": 0}}
    (out / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    cache = out / "fooof_trial_params_REAL_DP_ER_partial.csv"
    rows = []
    if cache.is_file() and not args.force_fooof:
        rows = pd.read_csv(cache).to_dict("records")
    done = {(str(r["topology"]), int(r["trial_id"])) for r in rows
            if np.isfinite(pd.to_numeric(r.get("exponent"), errors="coerce"))}
    total = len(TOPOLOGIES) * len(TRIAL_IDS)
    for topo in TOPOLOGIES:
        for tid in TRIAL_IDS:
            if (topo, tid) in done:
                print(f"[SKIP FOOOF] {topo} {trial_name(tid)}")
                continue
            plot = out / "fooof_fits" / f"{topo}_{trial_name(tid)}.png" if args.save_fit_plots else None
            offset, exponent, r2, error = fit_one_lfp([paths[(topo, tid, layer)] for layer in LAYERS], plot)
            rows.append({"trial": trial_name(tid), "trial_id": tid, "topology": topo,
                         "offset": offset, "exponent": exponent, "fooof_r2": r2, "fooof_error": error})
            pd.DataFrame(rows).drop_duplicates(["topology", "trial_id"], keep="last").to_csv(
                cache, index=False, encoding="utf-8-sig")
            print(f"[FOOOF {len(done)+1:03d}/{total}] {topo} {trial_name(tid)} exponent={exponent:.6g} R2={r2:.4f}")
            done.add((topo, tid))
    fooof_df = pd.DataFrame(rows).drop_duplicates(["topology", "trial_id"], keep="last")
    if len(fooof_df) != total:
        raise RuntimeError(f"Expected {total} unique FOOOF rows, got {len(fooof_df)}")

    syn2pre, root2type = load_synapse_maps(allen)
    ei_rows = []
    for tid in TRIAL_IDS:
        name = trial_name(tid)
        specifications = {
            "REAL": (original_root / name / "microns_patch_synapse_spikes_real_ms.csv", "real"),
            "ER": (original_root / name / "microns_patch_synapse_spikes_ER_ms.csv", "control"),
            "DP": (dp_root / name / "microns_patch_synapse_spikes_ER_ms.csv", "control"),
        }
        for topo, (path, kind) in specifications.items():
            e, i, unknown = ei_from_real(path, syn2pre, root2type) if kind == "real" else ei_from_control(path)
            ei_rows.append({"trial": name, "trial_id": tid, "topology": topo,
                            "E_spikes": e, "I_spikes": i, "EI_ratio": ratio(e, i),
                            "unknown_synapse_rows": unknown})
    ei_df = pd.DataFrame(ei_rows)
    long_df = fooof_df.merge(ei_df, on=["trial", "trial_id", "topology"], validate="one_to_one")
    if len(long_df) != total or not np.all(np.isfinite(long_df[["exponent", "EI_ratio"]])):
        raise RuntimeError("The final 120-row table is incomplete or contains non-finite exponent/EI values")
    long_df["topology"] = pd.Categorical(long_df["topology"], categories=list(TOPOLOGIES), ordered=True)
    long_df = long_df.sort_values(["trial_id", "topology"])
    long_df.to_csv(out / "trial_level_REAL_DP_ER.csv", index=False, encoding="utf-8-sig")

    exponent_pairs = paired_stats(long_df, "exponent")
    ei_pairs = paired_stats(long_df, "EI_ratio")
    exponent_pairs.to_csv(out / "paired_tests_exponent.csv", index=False, encoding="utf-8-sig")
    ei_pairs.to_csv(out / "paired_tests_EI_ratio.csv", index=False, encoding="utf-8-sig")
    coef, contrasts = run_models(long_df, out)
    make_plots(long_df, out)

    descriptives = long_df.groupby("topology", observed=True).agg(
        n=("exponent", "size"), exponent_mean=("exponent", "mean"), exponent_sd=("exponent", "std"),
        EI_mean=("EI_ratio", "mean"), EI_sd=("EI_ratio", "std"),
        fooof_r2_mean=("fooof_r2", "mean"), fooof_r2_min=("fooof_r2", "min"),
    ).reset_index()
    descriptives.to_csv(out / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    lines = ["m=0.99 REAL vs DP vs ER (matched 40 trials)", "=" * 58, "", "DESCRIPTIVES",
             descriptives.to_string(index=False), "", "PAIRED EXPONENT TESTS",
             exponent_pairs.to_string(index=False), "", "PLANNED MODEL CONTRASTS",
             contrasts.to_string(index=False), "", "All regression SEs are clustered by original trial.",
             "Pairwise p values include Holm correction within each family."]
    (out / "RESULTS_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Results saved under: {out}")
    print(exponent_pairs[["contrast", "mean_difference", "ci95_low", "ci95_high", "paired_t_p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
