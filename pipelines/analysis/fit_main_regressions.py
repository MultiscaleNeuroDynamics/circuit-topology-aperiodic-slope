# -*- coding: utf-8 -*-
"""
Mixed-group test: does EI explain beta, and does topology still matter after
controlling EI?

- Build long table: 2N rows (REAL + ER)
- Regressions with trial-cluster robust SE
- Report coefficient, trial-cluster robust SE, t, p and 95% CI for EVERY term
- Models:
  M1: beta ~ EI
  M2: beta ~ topology
  M3: beta ~ EI + topology
  M4: beta ~ EI * topology

Important: the 95% confidence intervals are calculated from the same
trial-cluster-robust covariance matrix used for SE, t and p. The E/I
definition, model formulas and inferential procedure are otherwise unchanged.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================================================
# 1) Paths (row-aligned)
# =========================================================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
RESULTS_ROOT = PROJECT_ROOT / "ALLEN" / "results"
RESULTS_TAG = os.environ.get("RESULTS_TAG", "").strip("/\\")
if RESULTS_TAG:
    RESULTS_ROOT = RESULTS_ROOT / RESULTS_TAG
ei_csv = str(RESULTS_ROOT / "ei_ratio_stats" / "EI_ratio_per_trial_REAL_vs_ER.csv")
exp_csv = str(RESULTS_ROOT / "fooof_group_compare" / "fooof_exponent_REAL_and_ER_two_columns.csv")
out_dir = str(RESULTS_ROOT / "mixed_ei_topology_test")
os.makedirs(out_dir, exist_ok=True)


# =========================================================
# 2) Load
# =========================================================
df_ei = pd.read_csv(ei_csv)
df_exp = pd.read_csv(exp_csv)

if len(df_ei) != len(df_exp):
    raise RuntimeError(f"Row mismatch: EI={len(df_ei)} vs exponent={len(df_exp)}")

need_ei_cols = ["REAL_EI_ratio", "ER_EI_ratio"]
need_exp_cols = ["REAL_exponent", "ER_exponent"]
miss = (
    [c for c in need_ei_cols if c not in df_ei.columns]
    + [c for c in need_exp_cols if c not in df_exp.columns]
)
if miss:
    raise RuntimeError(f"Missing columns: {miss}")

# Use the original trial identifiers whenever available. Otherwise, preserve
# the original row-wise REAL--ER pairing with an index-derived identifier.
if "trial" in df_ei.columns:
    trial_ids = df_ei["trial"].astype(str).values
elif "trial" in df_exp.columns:
    trial_ids = df_exp["trial"].astype(str).values
else:
    trial_ids = np.array(
        [f"trial_{i:04d}" for i in range(len(df_ei))], dtype=str
    )


# =========================================================
# 3) Build long table (2N rows)
# =========================================================
real_beta = pd.to_numeric(df_exp["REAL_exponent"], errors="coerce")
er_beta = pd.to_numeric(df_exp["ER_exponent"], errors="coerce")
real_ei = pd.to_numeric(df_ei["REAL_EI_ratio"], errors="coerce")
er_ei = pd.to_numeric(df_ei["ER_EI_ratio"], errors="coerce")

df_long = pd.DataFrame(
    {
        "trial": np.concatenate([trial_ids, trial_ids]),
        "topology": np.array(
            ["REAL"] * len(trial_ids) + ["ER"] * len(trial_ids), dtype=str
        ),
        "EI_ratio": np.concatenate([real_ei.values, er_ei.values]),
        "beta": np.concatenate([real_beta.values, er_beta.values]),
    }
)

df_long = df_long[
    np.isfinite(df_long["EI_ratio"]) & np.isfinite(df_long["beta"])
].copy()
df_long["topology"] = pd.Categorical(
    df_long["topology"], categories=["REAL", "ER"]
)
df_long.to_csv(
    os.path.join(out_dir, "long_table_REAL_ER.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(
    f"[Info] long rows = {len(df_long)} "
    f"(should be ~{2 * len(trial_ids)}), "
    f"unique trials = {df_long['trial'].nunique()}"
)


# =========================================================
# 4) Fit models with trial-cluster robust SE and 95% CI
# =========================================================
def fit_cluster_ols(formula, data, cluster_col="trial"):
    """Fit OLS and return both conventional and cluster-robust results."""
    model = smf.ols(formula=formula, data=data).fit()
    robust = model.get_robustcov_results(
        cov_type="cluster", groups=data[cluster_col]
    )
    return model, robust


def summarize_robust(robust, name, alpha=0.05):
    """Return all terms with cluster-robust inference and matching 95% CIs."""
    ci = np.asarray(robust.conf_int(alpha=alpha), dtype=float)
    out = pd.DataFrame(
        {
            "model": name,
            "term": robust.model.exog_names,
            "coef": robust.params,
            "se_cluster": robust.bse,
            "ci95_low": ci[:, 0],
            "ci95_high": ci[:, 1],
            "t": robust.tvalues,
            "p": robust.pvalues,
        }
    )
    return out


specs = [
    ("M1_EI_only", "beta ~ EI_ratio"),
    ("M2_topology_only", "beta ~ C(topology)"),
    ("M3_EI_plus_topology", "beta ~ EI_ratio + C(topology)"),
    ("M4_interaction", "beta ~ EI_ratio * C(topology)"),
]

all_tabs = []
fit_info = []

for name, formula in specs:
    model, robust = fit_cluster_ols(formula, df_long, cluster_col="trial")
    all_tabs.append(summarize_robust(robust, name, alpha=0.05))
    fit_info.append(
        {
            "model": name,
            "formula": formula,
            "n_obs": int(robust.nobs),
            "n_trials": int(df_long["trial"].nunique()),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "aic": float(model.aic),
            "bic": float(model.bic),
        }
    )

    print("\n" + "=" * 75)
    print(f"===== {name}: {formula} (cluster=trial) =====")
    print(robust.summary())

coef_df = pd.concat(all_tabs, axis=0, ignore_index=True)
fit_df = pd.DataFrame(fit_info)

# Full-precision statistical output.
coef_path = os.path.join(out_dir, "cluster_robust_coefficients_with_95CI.csv")
fit_path = os.path.join(out_dir, "model_fit_metrics.csv")
coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
fit_df.to_csv(fit_path, index=False, encoding="utf-8-sig")

# Manuscript-ready output. It retains every coefficient, including intercepts,
# so the reviewer request is met without selectively reporting terms.
table1_df = coef_df.copy()
table1_df["estimate_95CI"] = table1_df.apply(
    lambda row: (
        f"{row['coef']:.3f} "
        f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
    ),
    axis=1,
)
table1_df["p_display"] = table1_df["p"].map(
    lambda x: f"{x:.3e}" if np.isfinite(x) else "NA"
)
table1_path = os.path.join(out_dir, "Table1_coefficients_with_95CI.csv")
table1_df.to_csv(table1_path, index=False, encoding="utf-8-sig")

print(f"\n[Saved] {coef_path}")
print(f"[Saved] {fit_path}")
print(f"[Saved] {table1_path}")


# =========================================================
# 5) Key comparisons, now including matching 95% CIs
# =========================================================
def get_term_row(data, model_name, term_name):
    sub = data[(data["model"] == model_name) & (data["term"] == term_name)]
    if len(sub) != 1:
        return None
    return sub.iloc[0]


def print_term(label, row):
    if row is None:
        print(f"[{label}] term not found (check term naming)")
        return
    print(
        f"[{label}] coef={row['coef']:.6g}, "
        f"SE_cluster={row['se_cluster']:.6g}, "
        f"95% CI=[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}], "
        f"p={row['p']:.3g}"
    )


term_topology = "C(topology)[T.ER]"
row_m2 = get_term_row(coef_df, "M2_topology_only", term_topology)
row_m3 = get_term_row(coef_df, "M3_EI_plus_topology", term_topology)

print("\n" + "=" * 75)
print("===== Topology(ER) effect before/after controlling EI =====")
print_term("M2", row_m2)
print_term("M3", row_m3)

term_interaction = "EI_ratio:C(topology)[T.ER]"
row_m4_interaction = get_term_row(
    coef_df, "M4_interaction", term_interaction
)
print("\n===== Interaction EI_ratio × topology(ER) =====")
print_term("M4", row_m4_interaction)

print("\nDone.")
