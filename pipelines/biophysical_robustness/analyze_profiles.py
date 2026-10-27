# -*- coding: utf-8 -*-
"""针对五个参数配置和基线开展FOOOF及描述性端点稳健性分析。 / FOOOF and descriptive endpoint robustness analysis for five profiles + baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fooof import FOOOF
from scipy.signal import butter, resample, sosfiltfilt, welch

TRIAL_IDS = (1, 4, 97, 100)
LOW_IDS, HIGH_IDS = (1, 4), (97, 100)
LAYERS = ("L2_3", "L4", "L5", "L6")


def parse_args():
    repository_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", repository_root / "work"))
    p.add_argument("--output-root")
    p.add_argument("--force-fooof", action="store_true")
    return p.parse_args()


def locate(root, trial, layer):
    base = root / trial / layer
    for path in (base / "_lfp.npy", base / layer / "_lfp.npy"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(base / "_lfp.npy")


def fit_lfp(paths):
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    n = min(a.shape[1] for a in arrays)
    t = arrays[0][0, :n].astype(float)
    x = np.sum([a[1, :n].astype(float) for a in arrays], axis=0)
    fs = 1000.0 / np.median(np.diff(t))
    x = sosfiltfilt(butter(4, [0.1, 120], btype="bandpass", fs=fs, output="sos"), x)
    x = resample(x, int(round(len(x) * 1000.0 / fs)))
    f, p = welch(x, fs=1000.0, window="hann", nperseg=2048, noverlap=1024)
    mask = (f >= 0.1) & (f <= 120); f, p = f[mask], p[mask]
    logp = np.log10(np.maximum(p, 1e-20)); logp = np.convolve(logp, np.ones(9) / 9, mode="same")
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=0, min_peak_height=.10,
               aperiodic_mode="fixed", verbose=False)
    fm.fit(f, 10 ** logp, [1, 100])
    return float(fm.aperiodic_params_[1]), float(fm.r_squared_), float(fm.error_)


def load_maps(allen):
    syn = pd.read_csv(allen / "file/proofread_cells_synapses.csv", dtype=str)
    sid = "synapse_id" if "synapse_id" in syn.columns else syn.columns[0]
    types = pd.read_csv(allen / "file/proofread_cells_celltypes_all.csv", header=None, dtype=str)
    return dict(zip(syn[sid].str.strip(), syn["pre_pt_root_id"].str.strip())), dict(zip(types[0].str.strip(), types[1].str.strip()))


def counts(path):
    df = pd.read_csv(path, header=None, dtype=str)
    ids = df.iloc[:, 0].fillna("").str.strip()
    n = df.iloc[:, 1:].replace(r"^\s*$", np.nan, regex=True).notna().sum(axis=1)
    return zip(ids, n)


def actual_ei(path, topology, syn2pre, root2type):
    e = i = 0
    for sid, n in counts(path):
        if topology == "REAL":
            label = str(root2type.get(syn2pre.get(sid), "")).lower()
            if "excitatory" in label: e += int(n)
            elif "inhibitory" in label: i += int(n)
        else:
            if sid.startswith("10"): e += int(n)
            elif sid.startswith("00"): i += int(n)
    return e / i


def main():
    args = parse_args(); project = Path(args.project_root).resolve(); allen = project / "ALLEN"
    simulation_root = allen / "save/m_0.99/biophys_endpoint_sensitivity"
    out = Path(args.output_root or allen / "results/m_0.99/biophys_endpoint_sensitivity").resolve()
    out.mkdir(parents=True, exist_ok=True)
    roots = {"baseline": {"REAL": allen / "save/final_real", "ER": allen / "save/final_ER"}}
    for profile_dir in sorted(simulation_root.glob("profile_*")):
        roots[profile_dir.name] = {"REAL": profile_dir / "REAL", "ER": profile_dir / "ER"}
    if len(roots) != 6:
        raise RuntimeError(f"Expected baseline + 5 profiles, found {list(roots)}")
    cache = out / "fooof_endpoint_partial.csv"
    rows = [] if args.force_fooof or not cache.is_file() else pd.read_csv(cache).to_dict("records")
    done = {(r["profile"], r["topology"], int(r["trial_id"])) for r in rows}
    for profile, topo_roots in roots.items():
        for topology, root in topo_roots.items():
            for tid in TRIAL_IDS:
                key = (profile, topology, tid)
                if key in done:
                    print(f"[SKIP FOOOF] {key}"); continue
                trial = f"ER_trial{tid:04d}"
                exponent, r2, error = fit_lfp([locate(root, trial, layer) for layer in LAYERS])
                rows.append({"profile": profile, "topology": topology, "trial": trial,
                             "trial_id": tid, "endpoint": "low" if tid in LOW_IDS else "high",
                             "exponent": exponent, "fooof_r2": r2, "fooof_error": error})
                pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8-sig")
                print(f"[FOOOF] {key} exponent={exponent:.5f} R2={r2:.4f}")
    df = pd.DataFrame(rows).drop_duplicates(["profile", "topology", "trial_id"], keep="last")
    if len(df) != 48:
        raise RuntimeError(f"Expected 48 FOOF rows, found {len(df)}")
    syn2pre, root2type = load_maps(allen); original = allen / "file/final_trial"
    ei = {}
    for topology in ("REAL", "ER"):
        for tid in TRIAL_IDS:
            trial = f"ER_trial{tid:04d}"
            filename = "microns_patch_synapse_spikes_real_ms.csv" if topology == "REAL" else "microns_patch_synapse_spikes_ER_ms.csv"
            ei[(topology, tid)] = actual_ei(original / trial / filename, topology, syn2pre, root2type)
    df["EI_ratio"] = [ei[(r.topology, int(r.trial_id))] for r in df.itertuples()]
    df.to_csv(out / "endpoint_trial_results.csv", index=False, encoding="utf-8-sig")

    summaries = []
    for profile in roots:
        slopes, changes = {}, {}
        for topology in ("REAL", "ER"):
            sub = df[(df.profile == profile) & (df.topology == topology)].sort_values("EI_ratio")
            slopes[topology] = float(np.polyfit(sub.EI_ratio, sub.exponent, 1)[0])
            changes[topology] = float(sub[sub.endpoint == "high"].exponent.mean() - sub[sub.endpoint == "low"].exponent.mean())
        summaries.append({
            "profile": profile, "REAL_slope": slopes["REAL"], "ER_slope": slopes["ER"],
            "slope_interaction_ER_minus_REAL": slopes["ER"] - slopes["REAL"],
            "REAL_high_minus_low": changes["REAL"], "ER_high_minus_low": changes["ER"],
            "difference_in_differences_ER_minus_REAL": changes["ER"] - changes["REAL"],
            "REAL_negative": slopes["REAL"] < 0, "ER_negative": slopes["ER"] < 0,
            "ER_more_negative_than_REAL": slopes["ER"] < slopes["REAL"],
            "passes_directional_criterion": slopes["REAL"] < 0 and slopes["ER"] < slopes["REAL"],
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "profile_robustness_summary.csv", index=False, encoding="utf-8-sig")
    profiles = summary[summary.profile != "baseline"]
    report = [
        "Biophysical endpoint sensitivity (m=0.99)", "=" * 55,
        "Four original trials: 0001, 0004, 0097, 0100",
        "Five perturbation profiles are robustness scenarios, not independent replicates.", "",
        summary.to_string(index=False), "",
        f"New profiles retaining negative REAL slope: {int(profiles.REAL_negative.sum())}/5",
        f"New profiles retaining negative ER slope: {int(profiles.ER_negative.sum())}/5",
        f"New profiles retaining ER-more-negative interaction: {int(profiles.ER_more_negative_than_REAL.sum())}/5",
        f"New profiles passing full directional criterion: {int(profiles.passes_directional_criterion.sum())}/5",
    ]
    (out / "RESULTS_SUMMARY.txt").write_text("\n".join(report), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(summary)); ax.axhline(0, color="black", lw=.8)
    ax.plot(x, summary.REAL_slope, "o-", label="REAL")
    ax.plot(x, summary.ER_slope, "o-", label="ER")
    ax.set_xticks(x, summary.profile, rotation=30, ha="right")
    ax.set_ylabel("Endpoint E/I–aperiodic exponent slope"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "biophys_profile_slopes.png", dpi=300); fig.savefig(out / "biophys_profile_slopes.pdf"); plt.close(fig)
    print(f"[DONE] analysis: {out}\n" + "\n".join(report[-4:]))


if __name__ == "__main__":
    main()
