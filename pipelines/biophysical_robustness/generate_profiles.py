# -*- coding: utf-8 -*-
"""Generate five deterministic joint biophysical perturbation profiles."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np


PROFILE_U = np.array([
    [0.1, 0.5, 0.9, 0.3, 0.7, 0.1, 0.5, 0.9, 0.3, 0.7, 0.1],
    [0.3, 0.9, 0.5, 0.7, 0.1, 0.5, 0.9, 0.3, 0.7, 0.1, 0.5],
    [0.5, 0.1, 0.7, 0.9, 0.3, 0.9, 0.3, 0.7, 0.1, 0.5, 0.9],
    [0.7, 0.3, 0.1, 0.5, 0.9, 0.7, 0.1, 0.5, 0.9, 0.3, 0.7],
    [0.9, 0.7, 0.3, 0.1, 0.5, 0.3, 0.7, 0.1, 0.5, 0.9, 0.3],
], dtype=float)

NAMES = [
    "e_weight", "i_weight", "e_tau1", "e_tau2", "i_tau1", "i_tau2",
    "Ra", "g_leak", "cm", "erev_shift_mV", "delay_mean_ms",
]


def parse_args():
    repository_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", repository_root / "work"))
    p.add_argument("--parameter-file")
    p.add_argument("--output-root")
    return p.parse_args()


def scale(u, lo, hi):
    return float(lo + u * (hi - lo))


def numeric_key(d, candidates, label):
    for key in candidates:
        if key in d and isinstance(d[key], (int, float)):
            return key
    raise KeyError(f"Cannot find numeric {label}; checked keys {candidates}; available={list(d)}")


def main():
    args = parse_args()
    project = Path(args.project_root).resolve()
    parameter_file = Path(args.parameter_file or project / "ALLEN/file/swc/parameter.json").resolve()
    output = Path(args.output_root or project / "ALLEN/file/biophys_endpoint_sensitivity").resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(parameter_file.read_text(encoding="utf-8"))
    e = baseline["eSynParams"]
    i = baseline["iSynParams"]
    pas = baseline["biophys_pars"]["pas_mem_pars"]
    e_weight_key = numeric_key(e, ["weight", "gmax", "conductance"], "E synaptic weight")
    i_weight_key = numeric_key(i, ["weight", "gmax", "conductance"], "I synaptic weight")
    e_tau1_key = numeric_key(e, ["tau1"], "E tau1")
    e_tau2_key = numeric_key(e, ["tau2"], "E tau2")
    i_tau1_key = numeric_key(i, ["tau1"], "I tau1")
    i_tau2_key = numeric_key(i, ["tau2"], "I tau2")

    baseline_cm = float(pas.get("cm", 1.0))
    rows = []
    for idx, u in enumerate(PROFILE_U, start=1):
        multipliers = {name: scale(value, 0.8, 1.2) for name, value in zip(NAMES[:9], u[:9])}
        erev_shift = scale(u[9], -5.0, 5.0)
        delay_mean = scale(u[10], 0.5, 2.0)
        delay_half_width = 0.25

        pars = copy.deepcopy(baseline)
        pe, pi = pars["eSynParams"], pars["iSynParams"]
        pp = pars["biophys_pars"]["pas_mem_pars"]
        pe[e_weight_key] = float(e[e_weight_key]) * multipliers["e_weight"]
        pi[i_weight_key] = float(i[i_weight_key]) * multipliers["i_weight"]
        pe[e_tau1_key] = float(e[e_tau1_key]) * multipliers["e_tau1"]
        pe[e_tau2_key] = float(e[e_tau2_key]) * multipliers["e_tau2"]
        pi[i_tau1_key] = float(i[i_tau1_key]) * multipliers["i_tau1"]
        pi[i_tau2_key] = float(i[i_tau2_key]) * multipliers["i_tau2"]
        if not pe[e_tau1_key] < pe[e_tau2_key]:
            pe[e_tau1_key] = 0.8 * pe[e_tau2_key]
        if not pi[i_tau1_key] < pi[i_tau2_key]:
            pi[i_tau1_key] = 0.8 * pi[i_tau2_key]
        pp["Ra"] = float(pas["Ra"]) * multipliers["Ra"]
        pp["g_leak"] = float(pas["g_leak"]) * multipliers["g_leak"]
        pp["cm"] = baseline_cm * multipliers["cm"]
        pp["erev_leak"] = float(pas["erev_leak"]) + erev_shift
        pars["sensitivity_meta"] = {
            "profile": f"profile_{idx:02d}",
            "design": "deterministic_stratified_joint_perturbation",
            "multiplicative_range": [0.8, 1.2],
            "erev_shift_mV": erev_shift,
            "delay_min_ms": max(0.0, delay_mean - delay_half_width),
            "delay_max_ms": delay_mean + delay_half_width,
            "delay_rule": "one deterministic fixed latency per synapse",
        }
        path = output / f"profile_{idx:02d}.json"
        path.write_text(json.dumps(pars, indent=2), encoding="utf-8")
        row = {"profile": f"profile_{idx:02d}", "parameter_json": str(path)}
        row.update({f"multiplier_{k}": v for k, v in multipliers.items()})
        row.update({
            "E_weight": pe[e_weight_key], "I_weight": pi[i_weight_key],
            "E_tau1": pe[e_tau1_key], "E_tau2": pe[e_tau2_key],
            "I_tau1": pi[i_tau1_key], "I_tau2": pi[i_tau2_key],
            "Ra": pp["Ra"], "g_leak": pp["g_leak"], "cm": pp["cm"],
            "erev_leak": pp["erev_leak"],
            "delay_min_ms": pars["sensitivity_meta"]["delay_min_ms"],
            "delay_max_ms": pars["sensitivity_meta"]["delay_max_ms"],
        })
        rows.append(row)

    with (output / "biophys_parameter_profiles.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "baseline_parameter_snapshot.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"[DONE] generated 5 profiles under {output}")


if __name__ == "__main__":
    main()
