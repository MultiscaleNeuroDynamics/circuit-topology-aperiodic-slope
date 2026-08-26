#!/usr/bin/env python3
"""Assemble manuscript-ready Tables 1--3 from generated analysis outputs.

This script performs no new model fitting.  It extracts the already estimated
cluster-robust coefficients, operating-point interactions, biophysical
robustness ranges, and three-topology slope contrasts into concise CSV files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required generated result is missing: {path}\n"
            "Run the corresponding pipeline listed in the top-level README first."
        )
    return path


def interaction_row(results_root: Path, label: str) -> dict:
    coefficients = pd.read_csv(
        require(results_root / "mixed_ei_topology_test" /
                "cluster_robust_coefficients_with_95CI.csv")
    )
    model = coefficients[coefficients["model"].eq("M4_interaction")].copy()
    slope = model[model["term"].eq("EI_ratio")]
    interaction = model[
        model["term"].str.contains("EI_ratio:C\\(topology\\)\\[T\\.ER\\]", regex=True)
    ]
    if len(slope) != 1 or len(interaction) != 1:
        raise RuntimeError(f"Could not identify the M4 terms for {label}.")
    slope = slope.iloc[0]
    interaction = interaction.iloc[0]
    return {
        "Analysis": label,
        "MICrONS_EI_slope": slope["coef"],
        "MICrONS_slope_CI95_low": slope["ci95_low"],
        "MICrONS_slope_CI95_high": slope["ci95_high"],
        "ER_minus_MICrONS_slope_interaction": interaction["coef"],
        "Interaction_CI95_low": interaction["ci95_low"],
        "Interaction_CI95_high": interaction["ci95_high"],
        "Interaction_p": interaction["p"],
        "Directional_result": "negative interaction",
    }


def build_table2(allen_results: Path) -> pd.DataFrame:
    rows = [
        interaction_row(allen_results, "m=0.99"),
        interaction_row(allen_results / "m_0.95", "m=0.95"),
        interaction_row(allen_results / "m_0.90", "m=0.90"),
    ]

    profiles = pd.read_csv(require(
        allen_results / "m_0.99" / "biophys_endpoint_sensitivity" /
        "profile_robustness_summary.csv"
    ))
    profiles = profiles[profiles["profile"].astype(str).str.startswith("profile_")].copy()
    if profiles.empty:
        raise RuntimeError("No perturbed biophysical profiles were found.")
    passed = int(
        profiles["passes_directional_criterion"].astype(str).str.lower()
        .isin(("true", "1", "yes")).sum()
    )
    rows.append({
        "Analysis": "Five joint biophysical profiles, m=0.99",
        "MICrONS_EI_slope": f"{profiles['REAL_slope'].min():.3f} to {profiles['REAL_slope'].max():.3f}",
        "MICrONS_slope_CI95_low": np.nan,
        "MICrONS_slope_CI95_high": np.nan,
        "ER_minus_MICrONS_slope_interaction": (
            f"{profiles['slope_interaction_ER_minus_REAL'].min():.3f} to "
            f"{profiles['slope_interaction_ER_minus_REAL'].max():.3f}"
        ),
        "Interaction_CI95_low": np.nan,
        "Interaction_CI95_high": np.nan,
        "Interaction_p": np.nan,
        "Directional_result": f"criterion retained in {passed}/{len(profiles)} profiles",
    })
    return pd.DataFrame(rows)


def build_table3(allen_results: Path) -> pd.DataFrame:
    contrasts = pd.read_csv(require(
        allen_results / "m_0.99" / "real_er_dp40" / "model_planned_contrasts.csv"
    ))
    m4 = contrasts[contrasts["model"].eq("M4_interaction")].copy()
    by_label = {str(row["contrast"]): row for _, row in m4.iterrows()}
    slope_labels = {"MICrONS": "REAL_EI_slope", "DP": "DP_EI_slope", "ER": "ER_EI_slope"}
    pair_labels = {"MICrONS": "REAL-DP_slope", "DP": "REAL-ER_slope", "ER": "DP-ER_slope"}
    rows = []
    for topology in ("MICrONS", "DP", "ER"):
        slope = by_label.get(slope_labels[topology])
        pair = by_label.get(pair_labels[topology])
        if slope is None or pair is None:
            raise RuntimeError(f"Missing M4 contrast required for the {topology} Table 3 row.")
        rows.append({
            "Topology": topology,
            "EI_slope": slope["estimate"],
            "EI_slope_CI95_low": slope["ci95_low"],
            "EI_slope_CI95_high": slope["ci95_high"],
            "Pairwise_slope_contrast": pair["contrast"].replace("REAL", "MICrONS").replace("_slope", ""),
            "Contrast_estimate": pair["estimate"],
            "Contrast_CI95_low": pair["ci95_low"],
            "Contrast_CI95_high": pair["ci95_high"],
            "Contrast_p_Holm": pair["p_holm_within_model"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")),
    )
    args = parser.parse_args()
    results = args.project_root.resolve() / "ALLEN" / "results"
    output = results / "manuscript_tables"
    output.mkdir(parents=True, exist_ok=True)

    table1_source = require(
        results / "mixed_ei_topology_test" / "Table1_coefficients_with_95CI.csv"
    )
    shutil.copy2(table1_source, output / "Table1_cluster_robust_regressions.csv")
    build_table2(results).to_csv(
        output / "Table2_operating_point_and_biophysical_robustness.csv",
        index=False, encoding="utf-8-sig",
    )
    build_table3(results).to_csv(
        output / "Table3_three_topology_slope_contrasts.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"[COMPLETE] manuscript tables written to: {output}")


if __name__ == "__main__":
    main()
