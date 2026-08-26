# -*- coding: utf-8 -*-
"""Extract aperiodic and adaptive-Ising features from OpenNeuro ds003380.

The script performs signal and FOOOF quality control, estimates state-level
adaptive-Ising covariates, and creates the epoch/channel table used by the
animal-level propofol analysis.
"""

from pathlib import Path
import os
import re
import json
import math
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch, detrend, resample_poly, butter, sosfiltfilt
from scipy.optimize import least_squares
from scipy.stats import ttest_rel

import mne
from tqdm import tqdm

try:
    import scipy.misc
    import scipy.special
    scipy.misc.factorial = scipy.special.factorial
    scipy.misc.comb = scipy.special.comb
except Exception:
    pass

import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from fooof import FOOOF
except Exception as e:
    raise RuntimeError(
        "The 'fooof' package is required. Create the environment from "
        "the repository environment.yml file."
    ) from e


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
OUT_DIR = RESULTS_ROOT / "step2_features"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PARTICIPANTS_TSV = ROOT / "participants.tsv"

STATE_MAP = {
    "2": "fentanyl",
    "4": "propofol_early",
    "5": "propofol_late",
}
KEEP_STATES = set(STATE_MAP.keys())

INCLUDE_ISCHEMIA_PRE_ISCHEMIA = True

EPOCH_SEC = 20.0
EPOCH_STEP_SEC = 20.0
TARGET_FS = 500.0

PSD_NPERSEG_SEC = 4.0
FOOOF_F_RANGE = [1.0, 100.0]
FOOOF_PEAK_WIDTH_LIMITS = [1.0, 12.0]
FOOOF_MAX_N_PEAKS = 8
FOOOF_MIN_PEAK_HEIGHT = 0.05
FOOOF_PEAK_THRESHOLD = 2.0
FOOOF_APERIODIC_MODE = "fixed"

# FOOOF QC
MIN_FOOOF_R2 = 0.80
MAX_FOOOF_ERROR = 0.25
EXPONENT_RANGE = (-2.0, 6.0)

# Adaptive-Ising inference from alpha-band ACF
ISING_ALPHA_BAND = (8.0, 13.0)
ISING_ACF_MAX_LAG_SEC = 0.50
ISING_FIT_START_LAG_SEC = 0.0
ISING_FILTER_ORDER = 4

ISING_BETA_BOUNDS = (0.50, 0.9999)
ISING_C_BOUNDS = (1e-6, 1.0)
ISING_MIN_R2 = 0.50

# Full-state signal QC before adaptive-Ising ACF fitting.
# Purpose: remove/flag channels whose raw waveform is too small or whose
# broadband power is too low, because alpha-band ACF fitting standardizes the
# signal and would otherwise allow very low-amplitude channels to enter the fit.
ISING_SIGNAL_QC_MIN_FULL_SEC = 30.0
ISING_SIGNAL_QC_MIN_STD = 1e-12
ISING_SIGNAL_QC_MIN_RMS = 1e-12
ISING_SIGNAL_QC_MIN_LOG_TOTAL_POWER_1_100 = -30.0
ISING_SIGNAL_QC_MIN_MEDIAN_LOG10_PSD_1_90 = -30.0

# Additional data-driven low-power exclusion within state × region.
# This catches channels that are technically nonzero but are extreme low-power
# outliers relative to the same region/state. Set to 0 to disable.
ISING_SIGNAL_QC_LOW_QUANTILE = 0.02
ISING_SIGNAL_QC_MIN_GROUP_N_FOR_QUANTILE = 8

PHYS_CH = {
    1: "Temp",
    2: "ABP",
    3: "ECG",
    4: "CVP",
    5: "ICP",
}
CORTEX_CH = list(range(6, 15))   # Chan 6–14
THALAMUS_CH = [15, 16]           # Chan 15–16
SIGNAL_CH = CORTEX_CH + THALAMUS_CH


# ============================================================
# File-handling details.
# ============================================================

def parse_state_from_name(fname):
    m = re.search(r"state([0-9]+(?:\.[0-9]+)?)([a-z]*)", fname)
    if not m:
        return None
    num = m.group(1)
    suffix = m.group(2)
    return num + suffix if suffix else num


def parse_animal_from_name(fname):
    m = re.search(r"(P_[0-9]+)", fname)
    return m.group(1) if m else None


def edf_chan_name(ch_num):
    return "Chan %d" % int(ch_num)


def channel_region(ch_num):
    ch_num = int(ch_num)
    if ch_num in CORTEX_CH:
        return "cortex"
    if ch_num in THALAMUS_CH:
        return "thalamus"
    return "physiology"


# ============================================================
# ============================================================

def maybe_resample(data, fs, target_fs):
    """
    data: n_channels × n_times
    """
    if fs <= target_fs * 1.05:
        return data, fs

    up = int(target_fs)
    down = int(fs)
    g = math.gcd(up, down)
    up //= g
    down //= g

    data2 = resample_poly(data, up=up, down=down, axis=1)
    fs2 = fs * up / down
    return data2, fs2


def band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))


def interpolate_line_noise(freqs, psd, bands=((48.0, 52.0),)):
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float).copy()
    y = np.log10(np.maximum(psd, 1e-300))

    for lo, hi in bands:
        bad = (freqs >= lo) & (freqs <= hi)
        if bad.sum() == 0:
            continue

        left = np.where(freqs < lo)[0]
        right = np.where(freqs > hi)[0]
        if len(left) == 0 or len(right) == 0:
            continue

        i1 = left[-1]
        i2 = right[0]
        x1, x2 = freqs[i1], freqs[i2]
        y1, y2 = y[i1], y[i2]

        if x2 <= x1:
            continue

        y[bad] = y1 + (y2 - y1) * (freqs[bad] - x1) / (x2 - x1)

    return np.power(10.0, y)


def bandpass_sos(x, fs, lo, hi, order=4):
    x = np.asarray(x, dtype=float)

    nyq = fs / 2.0
    lo = max(lo, 0.01)
    hi = min(hi, nyq * 0.95)

    if hi <= lo:
        return np.full_like(x, np.nan)

    sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def autocorr_fft_norm(x, max_lag_samples):
    """
    Fast normalized unbiased autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max_lag_samples + 20:
        return None

    x = x - np.mean(x)
    sd = np.std(x)
    if not np.isfinite(sd) or sd <= 0:
        return None
    x = x / sd

    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()

    fx = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(fx * np.conj(fx), n=nfft)[:max_lag_samples + 1]

    denom = np.arange(n, n - max_lag_samples - 1, -1)
    acf = acf / denom

    if not np.isfinite(acf[0]) or acf[0] <= 0:
        return None

    acf = acf / acf[0]
    return acf


# ============================================================
# 3. FOOOF aperiodic exponent
# ============================================================

def fit_fooof_fixed(freqs, psd):
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    ok = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0) & (psd > 0)
    freqs = freqs[ok]
    psd = psd[ok]

    if len(freqs) < 20:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    psd = interpolate_line_noise(freqs, psd, bands=((48.0, 52.0),))

    fm = FOOOF(
        peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
        max_n_peaks=FOOOF_MAX_N_PEAKS,
        min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
        peak_threshold=FOOOF_PEAK_THRESHOLD,
        aperiodic_mode=FOOOF_APERIODIC_MODE,
        verbose=False,
    )

    try:
        fm.fit(freqs, psd, FOOOF_F_RANGE)
        ap = fm.aperiodic_params_

        if ap is None or len(ap) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan, False

        offset = float(ap[0])
        exponent = float(ap[1])
        r2 = float(fm.r_squared_) if fm.r_squared_ is not None else np.nan
        err = float(fm.error_) if fm.error_ is not None else np.nan
        n_peaks = int(len(fm.peak_params_)) if fm.peak_params_ is not None else 0

        return exponent, offset, r2, err, n_peaks, True

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False


def compute_epoch_features_fooof(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < int(fs * 5):
        return None

    x = detrend(x, type="linear")

    rms = float(np.sqrt(np.mean(x ** 2)))
    std = float(np.std(x))
    abs_mean = float(np.mean(np.abs(x)))

    nperseg = int(PSD_NPERSEG_SEC * fs)
    nperseg = min(nperseg, len(x))
    if nperseg < int(fs):
        return None

    freqs, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        average="median",
    )

    exponent, offset, r2, err, n_peaks, success = fit_fooof_fixed(freqs, psd)

    total_power = band_power(freqs, psd, 1.0, 100.0)
    delta_power = band_power(freqs, psd, 1.0, 4.0)
    theta_power = band_power(freqs, psd, 4.0, 8.0)
    alpha_power = band_power(freqs, psd, 8.0, 13.0)
    beta_power = band_power(freqs, psd, 13.0, 30.0)
    gamma_power = band_power(freqs, psd, 30.0, 80.0)

    eps = 1e-30

    return {
        "fooof_exponent": exponent,
        "fooof_flatness": -exponent if np.isfinite(exponent) else np.nan,
        "fooof_offset": offset,
        "fooof_r2": r2,
        "fooof_error": err,
        "fooof_n_peaks": n_peaks,
        "fooof_success": bool(success),

        "rms": rms,
        "std": std,
        "abs_mean": abs_mean,

        "total_power_1_100": total_power,
        "delta_power": delta_power,
        "theta_power": theta_power,
        "alpha_power": alpha_power,
        "beta_band_power": beta_power,
        "gamma_power": gamma_power,

        "delta_rel": delta_power / (total_power + eps),
        "theta_rel": theta_power / (total_power + eps),
        "alpha_rel": alpha_power / (total_power + eps),
        "beta_band_rel": beta_power / (total_power + eps),
        "gamma_rel": gamma_power / (total_power + eps),

        "log_rms": np.log(rms + eps),
        "log_total_power": np.log(total_power + eps) if np.isfinite(total_power) else np.nan,
    }


# ============================================================
# 4. Adaptive-Ising alpha-band ACF fitting
# ============================================================

def empty_adaptive_ising_output():
    """Default adaptive-Ising output for failed/skipped fits."""
    return {
        "ising_beta_param": np.nan,
        "ising_c_param": np.nan,
        "ising_gamma": np.nan,
        "ising_omega_sample": np.nan,
        "ising_freq_hz": np.nan,
        "ising_dist_crit": np.nan,
        "ising_acf_r2": np.nan,
        "ising_fit_success": False,
    }


def compute_full_state_signal_qc_for_ising(x, fs):
    """
    Full-state signal QC before adaptive-Ising fitting.

    The adaptive-Ising ACF fit standardizes the alpha-band signal. Therefore,
    amplitude information would be removed before fitting. This QC step checks
    the unstandardized full-state channel signal and flags near-flat / very
    low-power recordings before they enter the Ising model.
    """
    eps = 1e-30
    out = {
        "ising_signal_ok_initial": False,
        "ising_signal_qc_reason": "not_checked",
        "ising_full_n_samples": 0,
        "ising_full_duration_sec": np.nan,
        "ising_full_std": np.nan,
        "ising_full_rms": np.nan,
        "ising_full_abs_mean": np.nan,
        "ising_full_total_power_1_100": np.nan,
        "ising_full_alpha_power": np.nan,
        "ising_full_log_rms": np.nan,
        "ising_full_log_total_power_1_100": np.nan,
        "ising_full_log_alpha_power": np.nan,
        "ising_full_median_log10_psd_1_90": np.nan,
    }

    try:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        out["ising_full_n_samples"] = int(len(x))
        out["ising_full_duration_sec"] = float(len(x) / fs) if fs > 0 else np.nan

        if len(x) < int(fs * ISING_SIGNAL_QC_MIN_FULL_SEC):
            out["ising_signal_qc_reason"] = "too_short"
            return out

        x = detrend(x, type="linear")

        std = float(np.std(x))
        rms = float(np.sqrt(np.mean(x ** 2)))
        abs_mean = float(np.mean(np.abs(x)))

        out.update({
            "ising_full_std": std,
            "ising_full_rms": rms,
            "ising_full_abs_mean": abs_mean,
            "ising_full_log_rms": float(np.log(rms + eps)) if np.isfinite(rms) else np.nan,
        })

        reasons = []
        if (not np.isfinite(std)) or std <= ISING_SIGNAL_QC_MIN_STD:
            reasons.append("std_too_low")
        if (not np.isfinite(rms)) or rms <= ISING_SIGNAL_QC_MIN_RMS:
            reasons.append("rms_too_low")

        nperseg = int(PSD_NPERSEG_SEC * fs)
        nperseg = min(nperseg, len(x))
        if nperseg < int(fs):
            reasons.append("too_short_for_psd")
        else:
            freqs, psd = welch(
                x,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density",
                average="median",
            )

            ok = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0) & (psd > 0)
            freqs = freqs[ok]
            psd = psd[ok]

            if len(freqs) < 20:
                reasons.append("psd_invalid")
            else:
                psd_clean = interpolate_line_noise(freqs, psd, bands=((48.0, 52.0),))
                total_power = band_power(freqs, psd_clean, 1.0, 100.0)
                alpha_power = band_power(freqs, psd_clean, ISING_ALPHA_BAND[0], ISING_ALPHA_BAND[1])
                fmask = (freqs >= 1.0) & (freqs <= 90.0)
                median_log10_psd = (
                    float(np.nanmedian(np.log10(np.maximum(psd_clean[fmask], eps))))
                    if fmask.sum() else np.nan
                )

                out.update({
                    "ising_full_total_power_1_100": total_power,
                    "ising_full_alpha_power": alpha_power,
                    "ising_full_log_total_power_1_100": (
                        float(np.log(total_power + eps)) if np.isfinite(total_power) else np.nan
                    ),
                    "ising_full_log_alpha_power": (
                        float(np.log(alpha_power + eps)) if np.isfinite(alpha_power) else np.nan
                    ),
                    "ising_full_median_log10_psd_1_90": median_log10_psd,
                })

                if (
                    (not np.isfinite(out["ising_full_log_total_power_1_100"]))
                    or out["ising_full_log_total_power_1_100"] <= ISING_SIGNAL_QC_MIN_LOG_TOTAL_POWER_1_100
                ):
                    reasons.append("total_power_too_low")

                if (
                    (not np.isfinite(median_log10_psd))
                    or median_log10_psd <= ISING_SIGNAL_QC_MIN_MEDIAN_LOG10_PSD_1_90
                ):
                    reasons.append("median_psd_too_low")

        if reasons:
            out["ising_signal_qc_reason"] = ";".join(reasons)
            out["ising_signal_ok_initial"] = False
        else:
            out["ising_signal_qc_reason"] = "ok_initial"
            out["ising_signal_ok_initial"] = True

        return out

    except Exception as e:
        out["ising_signal_qc_reason"] = "qc_exception:" + str(e)[:80]
        return out


def apply_state_region_low_power_qc(ising_state):
    """
    Add state × region quantile-based low-power QC to the Ising state table.

    The absolute QC catches near-flat signals. This additional relative QC
    catches extreme low-power outliers within the same state and region.
    """
    if ising_state is None or ising_state.empty:
        return ising_state

    out = ising_state.copy()

    out["ising_signal_ok_quantile"] = True
    out["ising_signal_qc_log_rms_qthr"] = np.nan
    out["ising_signal_qc_log_total_power_qthr"] = np.nan

    # Start from absolute/full-state QC; quantile QC can only turn OK into False.
    out["ising_signal_ok"] = out["ising_signal_ok_initial"].fillna(False).astype(bool)
    out["ising_signal_qc_reason_final"] = out["ising_signal_qc_reason"].astype(str)

    q = float(ISING_SIGNAL_QC_LOW_QUANTILE)
    if q <= 0:
        return out

    value_cols = [
        "ising_full_log_rms",
        "ising_full_log_total_power_1_100",
    ]

    for (state_label, region), idx in out.groupby(["state_label", "region"]).groups.items():
        idx = list(idx)
        g = out.loc[idx]

        if len(g) < ISING_SIGNAL_QC_MIN_GROUP_N_FOR_QUANTILE:
            continue

        ok_quantile = pd.Series(True, index=g.index)
        thresholds = {}

        for c in value_cols:
            x = pd.to_numeric(g[c], errors="coerce")
            finite = np.isfinite(x.values)
            if finite.sum() < ISING_SIGNAL_QC_MIN_GROUP_N_FOR_QUANTILE:
                continue

            thr = float(np.nanquantile(x.values[finite], q))
            thresholds[c] = thr
            ok_quantile &= x >= thr

        if not thresholds:
            continue

        out.loc[g.index, "ising_signal_ok_quantile"] = ok_quantile.values
        if "ising_full_log_rms" in thresholds:
            out.loc[g.index, "ising_signal_qc_log_rms_qthr"] = thresholds["ising_full_log_rms"]
        if "ising_full_log_total_power_1_100" in thresholds:
            out.loc[g.index, "ising_signal_qc_log_total_power_qthr"] = thresholds["ising_full_log_total_power_1_100"]

        bad_idx = g.index[~ok_quantile.values]
        if len(bad_idx):
            out.loc[bad_idx, "ising_signal_qc_reason_final"] = (
                out.loc[bad_idx, "ising_signal_qc_reason_final"].astype(str)
                + ";low_state_region_power_quantile"
            )

    out["ising_signal_ok"] = (
        out["ising_signal_ok_initial"].fillna(False).astype(bool)
        & out["ising_signal_ok_quantile"].fillna(False).astype(bool)
    )

    # Any record failing signal QC is treated as unusable for Ising-control models.
    out.loc[~out["ising_signal_ok"], "ising_fit_success"] = False

    return out


def adaptive_ising_acf(lags_sample, beta, c):
    beta = float(beta)
    c = float(c)

    gamma = (1.0 - beta) / 2.0
    omega_sq = beta * c - ((1.0 - beta) ** 2) / 4.0

    if omega_sq <= 0 or gamma < 0:
        return None

    omega = np.sqrt(omega_sq)
    tau = np.asarray(lags_sample, dtype=float)

    C = np.exp(-gamma * tau) * (
        np.cos(omega * tau) + (gamma / omega) * np.sin(omega * tau)
    )
    return C


def infer_adaptive_ising_from_alpha_acf(x, fs):
    out = empty_adaptive_ising_output()

    try:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < int(fs * 30):
            return out

        x = detrend(x, type="linear")

        xa = bandpass_sos(
            x,
            fs,
            ISING_ALPHA_BAND[0],
            ISING_ALPHA_BAND[1],
            order=ISING_FILTER_ORDER
        )

        xa = xa[np.isfinite(xa)]
        if len(xa) < int(fs * 30):
            return out

        sd = np.std(xa)
        if not np.isfinite(sd) or sd <= 0:
            return out

        xa = (xa - np.mean(xa)) / sd

        max_lag = int(ISING_ACF_MAX_LAG_SEC * fs)
        start_lag = int(ISING_FIT_START_LAG_SEC * fs)

        acf = autocorr_fft_norm(xa, max_lag)
        if acf is None:
            return out

        lags = np.arange(len(acf), dtype=float)
        fit_mask = np.isfinite(acf) & (lags >= start_lag) & (lags <= max_lag)

        t = lags[fit_mask]
        y = acf[fit_mask]

        if len(y) < 20:
            return out

        init_grid = [
            (0.90, 0.005),
            (0.95, 0.008),
            (0.98, 0.010),
            (0.99, 0.012),
            (0.995, 0.020),
            (0.80, 0.050),
            (0.70, 0.100),
        ]

        weight = 1.0 / np.sqrt(1.0 + t)

        def residual_fun(par):
            beta, c = par
            pred = adaptive_ising_acf(t, beta, c)
            if pred is None or np.any(~np.isfinite(pred)):
                return np.ones_like(y) * 1e3
            return (pred - y) * weight

        best = None

        for init in init_grid:
            try:
                res = least_squares(
                    residual_fun,
                    x0=np.asarray(init, dtype=float),
                    bounds=(
                        [ISING_BETA_BOUNDS[0], ISING_C_BOUNDS[0]],
                        [ISING_BETA_BOUNDS[1], ISING_C_BOUNDS[1]],
                    ),
                    max_nfev=3000,
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8,
                )

                beta_hat, c_hat = res.x
                pred = adaptive_ising_acf(t, beta_hat, c_hat)
                if pred is None or np.any(~np.isfinite(pred)):
                    continue

                ss_res = np.sum((y - pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                if best is None or ss_res < best["ss_res"]:
                    best = {
                        "beta": float(beta_hat),
                        "c": float(c_hat),
                        "r2": float(r2),
                        "ss_res": float(ss_res),
                    }

            except Exception:
                continue

        if best is None:
            return out

        beta = best["beta"]
        c = best["c"]
        gamma = (1.0 - beta) / 2.0
        omega_sq = beta * c - ((1.0 - beta) ** 2) / 4.0

        if omega_sq <= 0:
            return out

        omega = np.sqrt(omega_sq)
        freq_hz = omega / (2.0 * np.pi) * fs

        out.update({
            "ising_beta_param": beta,
            "ising_c_param": c,
            "ising_gamma": gamma,
            "ising_omega_sample": float(omega),
            "ising_freq_hz": float(freq_hz),
            "ising_dist_crit": float(1.0 - beta),
            "ising_acf_r2": best["r2"],
            "ising_fit_success": bool(best["r2"] >= ISING_MIN_R2),
        })

        return out

    except Exception:
        return out


# ============================================================
# Statistical analysis.
# ============================================================

def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x)
    return (x - m) / s


def exact_signflip_test(diffs):
    """
    animal-level paired differences:
        diff = thalamus_delta - cortex_delta
    two-sided exact sign-flip p
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
            vals.append(np.mean(np.asarray(signs) * diffs))
        vals = np.asarray(vals)
        p = (np.sum(np.abs(vals) >= abs(obs)) + 1.0) / (len(vals) + 1.0)
        return obs, float(p), n

    rng = np.random.default_rng(123)
    vals = []
    for _ in range(20000):
        signs = rng.choice([-1, 1], size=n)
        vals.append(np.mean(signs * diffs))
    vals = np.asarray(vals)
    p = (np.sum(np.abs(vals) >= abs(obs)) + 1.0) / (len(vals) + 1.0)
    return obs, float(p), n


def fit_cluster_ols(formula, data, cluster_col="animal_id"):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]}
    )


# ============================================================
# 6. Build EDF inventory
# ============================================================

participants = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
part_map = participants.set_index("participant_id").to_dict("index")

edf_files = sorted(ROOT.glob("sub-*/eeg/*.edf"))

inventory_rows = []
for p in edf_files:
    subject = p.parts[-3]
    fname = p.name

    state = parse_state_from_name(fname)
    animal_from_file = parse_animal_from_name(fname)
    animal_from_participants = part_map.get(subject, {}).get("animal_id", None)

    # File-handling details.
    animal_id = animal_from_file if animal_from_file is not None else animal_from_participants
    group = part_map.get(subject, {}).get("group", "")

    inventory_rows.append({
        "subject": subject,
        "animal_id": animal_id,
        "group": group,
        "edf_path": str(p),
        "edf_name": fname,
        "state_raw": state,
        "state_label": STATE_MAP.get(state, ""),
    })

inv = pd.DataFrame(inventory_rows)

inv_main = inv[inv["state_raw"].isin(KEEP_STATES)].copy()
if not INCLUDE_ISCHEMIA_PRE_ISCHEMIA:
    inv_main = inv_main[inv_main["group"].eq("sedation")].copy()

inv_main = inv_main.sort_values(["animal_id", "subject", "state_raw", "edf_name"]).reset_index(drop=True)

inv.to_csv(OUT_DIR / "01_all_edf_inventory.csv", index=False, encoding="utf-8-sig")
inv_main.to_csv(OUT_DIR / "01_main_state245_inventory.csv", index=False, encoding="utf-8-sig")

print("=" * 120)
print("[INVENTORY]")
print("=" * 120)
print(f"All EDF files = {len(inv)}")
print(f"Main state2/4/5 EDF = {len(inv_main)}")
print(inv_main[["subject", "animal_id", "group", "state_raw", "state_label", "edf_name"]].to_string(index=False))


# ============================================================
# 7. Feature extraction: FOOOF epoch-level + Ising state-level
# ============================================================

feature_rows = []
ising_rows = []

print("\n" + "=" * 120)
print("[FOOOF EPOCH FEATURES + STATE-LEVEL ADAPTIVE-ISING]")
print("=" * 120)

for _, meta in tqdm(inv_main.iterrows(), total=len(inv_main), desc="EDF"):
    edf_path = Path(meta["edf_path"])

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    except Exception as e:
        print(f"[WARN] failed to read {edf_path}: {e}")
        continue

    fs = float(raw.info["sfreq"])
    if fs < 200:
        print(f"[SKIP] low fs={fs}: {edf_path}")
        continue

    need_nums = list(range(1, 17))
    need_chs = [edf_chan_name(i) for i in need_nums]
    present = [ch for ch in need_chs if ch in raw.ch_names]

    raw.pick_channels(present, ordered=True)
    data = raw.get_data()
    data, fs2 = maybe_resample(data, fs, TARGET_FS)

    present_nums = []
    for ch in present:
        m = re.search(r"Chan\s+([0-9]+)", ch)
        present_nums.append(int(m.group(1)) if m else None)

    n_times = data.shape[1]

    # ------------------------------------------------------------
    # 7.1 state-level adaptive-Ising inference on full state
    # ------------------------------------------------------------
    for ch_num in SIGNAL_CH:
        if ch_num not in present_nums:
            continue

        idx = present_nums.index(ch_num)
        x_full = data[idx, :]

        # Full-state signal QC is applied before alpha-band standardization and
        # ACF fitting. Low-amplitude / low-power channels are not sent into the
        # adaptive-Ising fit.
        ising_signal_qc = compute_full_state_signal_qc_for_ising(x_full, fs2)
        if bool(ising_signal_qc.get("ising_signal_ok_initial", False)):
            ising_feats = infer_adaptive_ising_from_alpha_acf(x_full, fs2)
        else:
            ising_feats = empty_adaptive_ising_output()

        row = {
            "subject": meta["subject"],
            "animal_id": meta["animal_id"],
            "group": meta["group"],
            "edf_name": meta["edf_name"],
            "state_raw": meta["state_raw"],
            "state_label": meta["state_label"],
            "channel_num": ch_num,
            "channel_name": edf_chan_name(ch_num),
            "region": channel_region(ch_num),
            "fs_used": fs2,
            "duration_sec": n_times / fs2,
        }
        row.update(ising_signal_qc)
        row.update(ising_feats)
        ising_rows.append(row)

    # ------------------------------------------------------------
    # 7.2 epoch-level FOOOF + covariates
    # ------------------------------------------------------------
    epoch_len = int(EPOCH_SEC * fs2)
    epoch_step = int(EPOCH_STEP_SEC * fs2)

    if n_times < epoch_len:
        continue

    for start in range(0, n_times - epoch_len + 1, epoch_step):
        stop = start + epoch_len
        epoch_idx = int(start // epoch_step)
        t0 = start / fs2
        t1 = stop / fs2

        # physiology channel covariates
        phys_feats = {}
        for ch_num, phys_name in PHYS_CH.items():
            if ch_num in present_nums:
                idx = present_nums.index(ch_num)
                xx = data[idx, start:stop]
                xx = xx[np.isfinite(xx)]
                phys_feats[f"{phys_name}_mean"] = float(np.nanmean(xx)) if len(xx) else np.nan
                phys_feats[f"{phys_name}_std"] = float(np.nanstd(xx)) if len(xx) else np.nan
            else:
                phys_feats[f"{phys_name}_mean"] = np.nan
                phys_feats[f"{phys_name}_std"] = np.nan

        for ch_num in SIGNAL_CH:
            if ch_num not in present_nums:
                continue

            idx = present_nums.index(ch_num)
            x = data[idx, start:stop]

            feats = compute_epoch_features_fooof(x, fs2)
            if feats is None:
                continue

            out = {
                "subject": meta["subject"],
                "animal_id": meta["animal_id"],
                "group": meta["group"],
                "edf_name": meta["edf_name"],
                "state_raw": meta["state_raw"],
                "state_label": meta["state_label"],

                "channel_num": ch_num,
                "channel_name": edf_chan_name(ch_num),
                "region": channel_region(ch_num),

                "epoch_idx": epoch_idx,
                "epoch_start_sec": t0,
                "epoch_stop_sec": t1,
                "fs_used": fs2,
                "epoch_sec": EPOCH_SEC,
            }
            out.update(feats)
            out.update(phys_feats)
            feature_rows.append(out)

features = pd.DataFrame(feature_rows)
ising_state = pd.DataFrame(ising_rows)

# Apply state × region low-power quantile QC after all state/channel rows are available.
# The saved Ising table therefore contains both absolute and relative signal-QC flags.
ising_state = apply_state_region_low_power_qc(ising_state)

features.to_csv(OUT_DIR / "02_epoch_channel_features_fooof_raw.csv", index=False, encoding="utf-8-sig")
ising_state.to_csv(OUT_DIR / "02_state_channel_adaptive_ising_raw.csv", index=False, encoding="utf-8-sig")

print("\n[SAVED RAW FEATURES]")
print(OUT_DIR / "02_epoch_channel_features_fooof_raw.csv")
print(OUT_DIR / "02_state_channel_adaptive_ising_raw.csv")
print(f"FOOOF epoch rows = {len(features)}")
print(f"Ising state rows = {len(ising_state)}")


# ============================================================
# 8. QC + merge Ising to epoch features
# ============================================================

df = features.copy()

df = df[df["fooof_success"].astype(bool)].copy()
df = df[np.isfinite(df["fooof_exponent"])].copy()
df = df[df["fooof_r2"] >= MIN_FOOOF_R2].copy()
df = df[df["fooof_error"] <= MAX_FOOOF_ERROR].copy()
df = df[(df["fooof_exponent"] > EXPONENT_RANGE[0]) & (df["fooof_exponent"] < EXPONENT_RANGE[1])].copy()

# merge state-level Ising covariates
ising_cols = [
    "subject",
    "animal_id",
    "edf_name",
    "state_label",
    "channel_num",
    "ising_beta_param",
    "ising_c_param",
    "ising_gamma",
    "ising_omega_sample",
    "ising_freq_hz",
    "ising_dist_crit",
    "ising_acf_r2",
    "ising_fit_success",
    "ising_signal_ok_initial",
    "ising_signal_ok_quantile",
    "ising_signal_ok",
    "ising_signal_qc_reason",
    "ising_signal_qc_reason_final",
    "ising_full_std",
    "ising_full_rms",
    "ising_full_abs_mean",
    "ising_full_total_power_1_100",
    "ising_full_alpha_power",
    "ising_full_log_rms",
    "ising_full_log_total_power_1_100",
    "ising_full_log_alpha_power",
    "ising_full_median_log10_psd_1_90",
]
ising_merge = ising_state[ising_cols].copy()

df = df.merge(
    ising_merge,
    on=["subject", "animal_id", "edf_name", "state_label", "channel_num"],
    how="left"
)

df.to_csv(OUT_DIR / "03_epoch_channel_features_fooof_QC_with_stateIsing.csv",
          index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("[QC]")
print("=" * 120)
print(f"rows after FOOOF QC = {len(df)}")
if not df.empty:
    print(df.groupby(["state_label", "region"])["fooof_exponent"].agg(["count", "mean", "std"]).to_string())

print("\n[ISING STATE FIT QC]")
if not ising_state.empty:
    print(ising_state.groupby(["state_label", "region"])["ising_acf_r2"].agg(["count", "mean", "std"]).to_string())
    print("\n[ISING SIGNAL QC PASS RATE]")
    if "ising_signal_ok" in ising_state.columns:
        print(ising_state.groupby(["state_label", "region"])["ising_signal_ok"].mean().to_string())
    print("\n[ISING SUCCESS RATE AFTER SIGNAL QC]")
    print(ising_state.groupby(["state_label", "region"])["ising_fit_success"].mean().to_string())


# ============================================================
# 9. Effective-state PCA PC1–PC3
# ============================================================

score_cols = [
    "log_rms",
    "log_total_power",
    "delta_rel",
    "theta_rel",
    "alpha_rel",
    "beta_band_rel",
    "gamma_rel",
    "ABP_mean",
    "ABP_std",
    "ECG_std",
    "ICP_mean",
    "ICP_std",
]

score_cols = [c for c in score_cols if c in df.columns]

score_df = df[score_cols].replace([np.inf, -np.inf], np.nan).copy()
for c in score_cols:
    score_df[c] = score_df[c].fillna(score_df[c].median())

X = StandardScaler().fit_transform(score_df.values)
n_pc = min(3, X.shape[1])
pca = PCA(n_components=n_pc)
PC = pca.fit_transform(X)

for i in range(n_pc):
    df[f"effective_PC{i+1}"] = PC[:, i]

df["effective_state_score"] = df["effective_PC1"]

pca_info = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(n_pc)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
})
pca_info.to_csv(OUT_DIR / "04_effective_state_pca_info.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("[EFFECTIVE STATE PCA]")
print("=" * 120)
print(pca_info.to_string(index=False))

df.to_csv(OUT_DIR / "04_epoch_channel_features_fooof_QC_with_PC123_Ising.csv",
          index=False, encoding="utf-8-sig")


# ============================================================
# 10. Baseline and delta features
# ============================================================

base_agg = {
    "baseline_exponent": ("fooof_exponent", "mean"),
    "baseline_flatness": ("fooof_flatness", "mean"),
    "baseline_offset": ("fooof_offset", "mean"),
    "baseline_PC1": ("effective_PC1", "mean"),
    "baseline_log_rms": ("log_rms", "mean"),
    "baseline_log_total_power": ("log_total_power", "mean"),
    "baseline_delta_rel": ("delta_rel", "mean"),
    "baseline_gamma_rel": ("gamma_rel", "mean"),
    "baseline_ABP_mean": ("ABP_mean", "mean"),
    "baseline_ICP_mean": ("ICP_mean", "mean"),

    "baseline_ising_beta_param": ("ising_beta_param", "mean"),
    "baseline_ising_c_param": ("ising_c_param", "mean"),
    "baseline_ising_gamma": ("ising_gamma", "mean"),
    "baseline_ising_omega_sample": ("ising_omega_sample", "mean"),
    "baseline_ising_freq_hz": ("ising_freq_hz", "mean"),
    "baseline_ising_dist_crit": ("ising_dist_crit", "mean"),
    "baseline_ising_acf_r2": ("ising_acf_r2", "mean"),
    "baseline_ising_success_rate": ("ising_fit_success", "mean"),
    "baseline_ising_signal_ok_rate": ("ising_signal_ok", "mean"),
    "baseline_ising_signal_ok_initial_rate": ("ising_signal_ok_initial", "mean"),
    "baseline_ising_signal_ok_quantile_rate": ("ising_signal_ok_quantile", "mean"),
    "baseline_ising_full_rms": ("ising_full_rms", "mean"),
    "baseline_ising_full_log_rms": ("ising_full_log_rms", "mean"),
    "baseline_ising_full_log_total_power_1_100": ("ising_full_log_total_power_1_100", "mean"),
}

if "effective_PC2" in df.columns:
    base_agg["baseline_PC2"] = ("effective_PC2", "mean")
if "effective_PC3" in df.columns:
    base_agg["baseline_PC3"] = ("effective_PC3", "mean")

base = (
    df[df["state_label"] == "fentanyl"]
    .groupby(["animal_id", "channel_num"])
    .agg(**base_agg)
    .reset_index()
)

df2 = df.merge(base, on=["animal_id", "channel_num"], how="left")

df2["delta_exponent_vs_fentanyl"] = df2["fooof_exponent"] - df2["baseline_exponent"]
df2["delta_flatness_vs_fentanyl"] = df2["fooof_flatness"] - df2["baseline_flatness"]
df2["delta_offset_vs_fentanyl"] = df2["fooof_offset"] - df2["baseline_offset"]

for pc in ["PC1", "PC2", "PC3"]:
    col = f"effective_{pc}"
    bcol = f"baseline_{pc}"
    if col in df2.columns and bcol in df2.columns:
        df2[f"delta_effective_{pc}_vs_fentanyl"] = df2[col] - df2[bcol]

for c in [
    "log_rms",
    "log_total_power",
    "delta_rel",
    "gamma_rel",
    "ABP_mean",
    "ICP_mean",
]:
    b = f"baseline_{c}"
    if c in df2.columns and b in df2.columns:
        df2[f"delta_{c}_vs_fentanyl"] = df2[c] - df2[b]

for c in [
    "ising_beta_param",
    "ising_c_param",
    "ising_gamma",
    "ising_omega_sample",
    "ising_freq_hz",
    "ising_dist_crit",
]:
    b = f"baseline_{c}"
    if c in df2.columns and b in df2.columns:
        df2[f"delta_{c}_vs_fentanyl"] = df2[c] - df2[b]

# Now includes full-state signal QC for both the current propofol state and
# its fentanyl baseline. This prevents low-amplitude channels from entering
# adaptive-Ising-control analyses through standardized ACF fits.
df2["ising_control_ok"] = (
    df2["ising_signal_ok"].fillna(False).astype(bool)
    & (df2["baseline_ising_signal_ok_rate"] >= 1.0)
    & df2["ising_fit_success"].fillna(False).astype(bool)
    & (df2["ising_acf_r2"] >= ISING_MIN_R2)
    & (df2["baseline_ising_acf_r2"] >= ISING_MIN_R2)
)

df2.to_csv(OUT_DIR / "05_epoch_channel_features_fooof_with_delta_Ising.csv",
           index=False, encoding="utf-8-sig")


# ============================================================
# 11. Subject × region × state summary
# ============================================================

agg_dict = {
    "n_epochs": ("fooof_exponent", "size"),
    "exponent_mean": ("fooof_exponent", "mean"),
    "exponent_sd": ("fooof_exponent", "std"),
    "flatness_mean": ("fooof_flatness", "mean"),
    "offset_mean": ("fooof_offset", "mean"),
    "fooof_r2_mean": ("fooof_r2", "mean"),
    "fooof_error_mean": ("fooof_error", "mean"),

    "delta_exponent_mean": ("delta_exponent_vs_fentanyl", "mean"),
    "delta_flatness_mean": ("delta_flatness_vs_fentanyl", "mean"),
    "delta_offset_mean": ("delta_offset_vs_fentanyl", "mean"),
    "baseline_exponent_mean": ("baseline_exponent", "mean"),

    "effective_PC1_mean": ("effective_PC1", "mean"),
    "delta_effective_PC1_mean": ("delta_effective_PC1_vs_fentanyl", "mean"),

    "ising_beta_mean": ("ising_beta_param", "mean"),
    "ising_c_mean": ("ising_c_param", "mean"),
    "ising_dist_crit_mean": ("ising_dist_crit", "mean"),
    "ising_acf_r2_mean": ("ising_acf_r2", "mean"),
    "ising_success_rate": ("ising_fit_success", "mean"),
    "ising_signal_ok_rate": ("ising_signal_ok", "mean"),
    "ising_full_rms_mean": ("ising_full_rms", "mean"),
    "ising_full_log_rms_mean": ("ising_full_log_rms", "mean"),
    "ising_full_log_total_power_mean": ("ising_full_log_total_power_1_100", "mean"),

    "baseline_ising_beta_mean": ("baseline_ising_beta_param", "mean"),
    "delta_ising_beta_mean": ("delta_ising_beta_param_vs_fentanyl", "mean"),
    "baseline_ising_c_mean": ("baseline_ising_c_param", "mean"),
    "delta_ising_c_mean": ("delta_ising_c_param_vs_fentanyl", "mean"),
    "baseline_ising_dist_crit_mean": ("baseline_ising_dist_crit", "mean"),
    "delta_ising_dist_crit_mean": ("delta_ising_dist_crit_vs_fentanyl", "mean"),
    "baseline_ising_signal_ok_rate_mean": ("baseline_ising_signal_ok_rate", "mean"),
}

if "effective_PC2" in df2.columns:
    agg_dict["effective_PC2_mean"] = ("effective_PC2", "mean")
    agg_dict["delta_effective_PC2_mean"] = ("delta_effective_PC2_vs_fentanyl", "mean")
if "effective_PC3" in df2.columns:
    agg_dict["effective_PC3_mean"] = ("effective_PC3", "mean")
    agg_dict["delta_effective_PC3_mean"] = ("delta_effective_PC3_vs_fentanyl", "mean")

sub_region = (
    df2
    .groupby(["animal_id", "subject", "group", "state_label", "region"])
    .agg(**agg_dict)
    .reset_index()
)

sub_region.to_csv(OUT_DIR / "06_subject_region_state_summary_fooof_Ising.csv",
                  index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("[SUBJECT × REGION × STATE SUMMARY]")
print("=" * 120)
print(sub_region.to_string(index=False))


# ============================================================
# 12. Robustness A: animal-level sign-flip permutation
# ============================================================

paired_rows = []

for analysis_name, summary_df in [
    ("all_available_pre_ischemia", sub_region),
    ("sedation_only", sub_region[sub_region["group"] == "sedation"].copy()),
]:
    for state in ["propofol_early", "propofol_late"]:
        tmp = summary_df[summary_df["state_label"] == state].copy()
        wide = tmp.pivot_table(
            index=["animal_id", "group"],
            columns="region",
            values="delta_exponent_mean",
            aggfunc="mean"
        ).reset_index()

        if "cortex" not in wide.columns or "thalamus" not in wide.columns:
            continue

        wide = wide.dropna(subset=["cortex", "thalamus"]).copy()
        wide["thalamus_minus_cortex_delta_exponent"] = wide["thalamus"] - wide["cortex"]

        if len(wide) >= 2:
            tt = ttest_rel(wide["thalamus"], wide["cortex"], nan_policy="omit")
            tval = float(tt.statistic)
            p_t = float(tt.pvalue)
        else:
            tval, p_t = np.nan, np.nan

        obs, p_perm, n = exact_signflip_test(wide["thalamus_minus_cortex_delta_exponent"].values)

        paired_rows.append({
            "analysis": analysis_name,
            "state": state,
            "n_animals": int(len(wide)),
            "mean_delta_exponent_cortex": float(wide["cortex"].mean()) if len(wide) else np.nan,
            "mean_delta_exponent_thalamus": float(wide["thalamus"].mean()) if len(wide) else np.nan,
            "mean_thalamus_minus_cortex": float(wide["thalamus_minus_cortex_delta_exponent"].mean()) if len(wide) else np.nan,
            "t_paired": tval,
            "p_paired_ttest": p_t,
            "p_signflip_exact": p_perm,
        })

        wide.to_csv(
            OUT_DIR / f"07_paired_animal_delta_exponent_{analysis_name}_{state}.csv",
            index=False,
            encoding="utf-8-sig"
        )

paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT_DIR / "07_animal_level_signflip_tests_all_and_sedation.csv",
              index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("[ROBUSTNESS A/B: ANIMAL-LEVEL SIGN-FLIP + SEDATION-ONLY]")
print("=" * 120)
print(paired.to_string(index=False))


# ============================================================
# 13. Robustness C/D: regression controls
# ============================================================

reg = df2[df2["state_label"].isin(["propofol_early", "propofol_late"])].copy()

for c in [
    "delta_exponent_vs_fentanyl",
    "baseline_exponent",
    "delta_offset_vs_fentanyl",
    "delta_effective_PC1_vs_fentanyl",
]:
    reg = reg[np.isfinite(reg[c])].copy()

# z-score covariates
z_cols = [
    "baseline_exponent",
    "delta_offset_vs_fentanyl",
    "delta_effective_PC1_vs_fentanyl",
]

if "delta_effective_PC2_vs_fentanyl" in reg.columns:
    z_cols.append("delta_effective_PC2_vs_fentanyl")
if "delta_effective_PC3_vs_fentanyl" in reg.columns:
    z_cols.append("delta_effective_PC3_vs_fentanyl")

direct_delta_cols = [
    "delta_log_rms_vs_fentanyl",
    "delta_log_total_power_vs_fentanyl",
    "delta_delta_rel_vs_fentanyl",
    "delta_gamma_rel_vs_fentanyl",
    "delta_ABP_mean_vs_fentanyl",
    "delta_ICP_mean_vs_fentanyl",
]

direct_delta_cols = [c for c in direct_delta_cols if c in reg.columns]

ising_cov_cols = [
    "baseline_ising_beta_param",
    "delta_ising_beta_param_vs_fentanyl",
    "baseline_ising_c_param",
    "delta_ising_c_param_vs_fentanyl",
    "baseline_ising_dist_crit",
    "delta_ising_dist_crit_vs_fentanyl",
    "baseline_ising_freq_hz",
    "delta_ising_freq_hz_vs_fentanyl",
]

ising_cov_cols = [c for c in ising_cov_cols if c in reg.columns]

for c in direct_delta_cols + ising_cov_cols:
    if np.isfinite(reg[c]).sum() > 10:
        z_cols.append(c)

for c in sorted(set(z_cols)):
    reg[f"z_{c}"] = zscore(reg[c].values)

reg.to_csv(OUT_DIR / "08_regression_input_epoch_level.csv",
           index=False, encoding="utf-8-sig")


formulas = {
    "M0_region_only":
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label)",

    "M1_PC1":
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl",

    "M2_PC123":
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl "
        "+ z_delta_effective_PC2_vs_fentanyl "
        "+ z_delta_effective_PC3_vs_fentanyl",

    "M3_PC123_baseline_offset":
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl "
        "+ z_delta_effective_PC2_vs_fentanyl "
        "+ z_delta_effective_PC3_vs_fentanyl "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl",

    "M5_region_x_PC1":
        "delta_exponent_vs_fentanyl ~ C(region) * z_delta_effective_PC1_vs_fentanyl "
        "+ C(state_label) + z_baseline_exponent + z_delta_offset_vs_fentanyl",
}

# direct covariate model
direct_terms = []
for raw_c in direct_delta_cols:
    zc = f"z_{raw_c}"
    if zc in reg.columns:
        direct_terms.append(zc)

if direct_terms:
    formulas["M4_direct_covariates_baseline_offset"] = (
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl + "
        + " + ".join(direct_terms)
    )

# adaptive-Ising controls
if (
    "z_baseline_ising_beta_param" in reg.columns
    and "z_delta_ising_beta_param_vs_fentanyl" in reg.columns
    and "z_baseline_ising_c_param" in reg.columns
    and "z_delta_ising_c_param_vs_fentanyl" in reg.columns
):
    formulas["M6_PC123_baseline_offset_adaptiveIsing_beta_c"] = (
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl "
        "+ z_delta_effective_PC2_vs_fentanyl "
        "+ z_delta_effective_PC3_vs_fentanyl "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl "
        "+ z_baseline_ising_beta_param "
        "+ z_delta_ising_beta_param_vs_fentanyl "
        "+ z_baseline_ising_c_param "
        "+ z_delta_ising_c_param_vs_fentanyl"
    )

if (
    "z_baseline_ising_dist_crit" in reg.columns
    and "z_delta_ising_dist_crit_vs_fentanyl" in reg.columns
):
    formulas["M7_PC123_baseline_offset_isingDistance"] = (
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl "
        "+ z_delta_effective_PC2_vs_fentanyl "
        "+ z_delta_effective_PC3_vs_fentanyl "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl "
        "+ z_baseline_ising_dist_crit "
        "+ z_delta_ising_dist_crit_vs_fentanyl"
    )

if direct_terms and "M6_PC123_baseline_offset_adaptiveIsing_beta_c" in formulas:
    formulas["M8_direct_covariates_offset_adaptiveIsing_beta_c"] = (
        "delta_exponent_vs_fentanyl ~ C(region) + C(state_label) "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl "
        "+ " + " + ".join(direct_terms) + " "
        "+ z_baseline_ising_beta_param "
        "+ z_delta_ising_beta_param_vs_fentanyl "
        "+ z_baseline_ising_c_param "
        "+ z_delta_ising_c_param_vs_fentanyl"
    )

if "z_delta_ising_beta_param_vs_fentanyl" in reg.columns:
    formulas["M9_region_x_deltaIsingBeta"] = (
        "delta_exponent_vs_fentanyl ~ C(region) * z_delta_ising_beta_param_vs_fentanyl "
        "+ C(state_label) "
        "+ z_delta_effective_PC1_vs_fentanyl "
        "+ z_delta_effective_PC2_vs_fentanyl "
        "+ z_delta_effective_PC3_vs_fentanyl "
        "+ z_baseline_exponent + z_delta_offset_vs_fentanyl"
    )


term_rows = []

print("\n" + "=" * 120)
print("[ROBUSTNESS C/D: REGRESSION CONTROLS INCLUDING ADAPTIVE-ISING]")
print("=" * 120)

for name, formula in formulas.items():
    needed_z_terms = re.findall(r"z_[A-Za-z0-9_]+", formula)
    missing = [c for c in needed_z_terms if c not in reg.columns]

    if missing:
        print(f"[SKIP] {name}, missing: {missing}")
        continue

    use_data = reg.copy()

    if ("ising" in name.lower()) or ("Ising" in name):
        if "ising_control_ok" in use_data.columns:
            use_data = use_data[use_data["ising_control_ok"].astype(bool)].copy()

    vars_in_formula = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    vars_in_formula = [v for v in vars_in_formula if v in use_data.columns]
    for v in vars_in_formula:
        if use_data[v].dtype.kind in "fc":
            use_data = use_data[np.isfinite(use_data[v])].copy()

    if len(use_data) < 20 or use_data["animal_id"].nunique() < 2:
        print(f"[SKIP] {name}, not enough data after QC")
        continue

    try:
        fit = fit_cluster_ols(formula, use_data, cluster_col="animal_id")

        with open(OUT_DIR / f"08_{name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(fit.summary()))

        ci = fit.conf_int()
        for term in fit.params.index:
            term_rows.append({
                "model": name,
                "term": term,
                "coef": float(fit.params[term]),
                "se": float(fit.bse[term]),
                "t": float(fit.tvalues[term]),
                "p": float(fit.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "n_rows": int(fit.nobs),
                "n_animals": int(use_data["animal_id"].nunique()),
                "r2": float(fit.rsquared),
                "formula": formula,
            })

    except Exception as e:
        print(f"[WARN] model failed {name}: {e}")

terms = pd.DataFrame(term_rows)
terms.to_csv(OUT_DIR / "08_model_terms_fooof_delta_exponent_IsingControl.csv",
             index=False, encoding="utf-8-sig")

key_terms = terms[terms["term"].str.contains("region", case=False, na=False)].copy()

print("\n[KEY REGION TERMS]")
print(key_terms.to_string(index=False))


# ============================================================
# 14. Figures
# ============================================================

plt.rcParams["font.family"] = "Arial"

# paired plots
for analysis_name, summary_df in [
    ("all", sub_region),
    ("sedation_only", sub_region[sub_region["group"] == "sedation"].copy()),
]:
    for state in ["propofol_early", "propofol_late"]:
        tmp = summary_df[summary_df["state_label"] == state].copy()
        wide = tmp.pivot_table(
            index=["animal_id", "group"],
            columns="region",
            values="delta_exponent_mean",
            aggfunc="mean"
        ).reset_index()

        if "cortex" not in wide.columns or "thalamus" not in wide.columns:
            continue

        wide = wide.dropna(subset=["cortex", "thalamus"]).copy()
        if wide.empty:
            continue

        fig, ax = plt.subplots(figsize=(4.2, 4.5), dpi=220)

        for _, rr in wide.iterrows():
            ax.plot([0, 1], [rr["cortex"], rr["thalamus"]], color="0.75", lw=1.2)
            ax.scatter([0, 1], [rr["cortex"], rr["thalamus"]], s=32)

        ax.axhline(0, color="0.2", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cortex", "Thalamus"])
        ax.set_ylabel("Δ FOOOF exponent vs fentanyl")
        ax.set_title(f"{analysis_name}: {state.replace('_', ' ')}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"09_paired_delta_exponent_{analysis_name}_{state}.png")
        plt.close(fig)

# Δexponent vs effective PC1
plot_df = reg.copy()
if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=220)

    for region, g in plot_df.groupby("region"):
        ax.scatter(
            g["delta_effective_PC1_vs_fentanyl"],
            g["delta_exponent_vs_fentanyl"],
            s=10,
            alpha=0.35,
            label=region,
        )

        gx = g["delta_effective_PC1_vs_fentanyl"].values
        gy = g["delta_exponent_vs_fentanyl"].values
        ok = np.isfinite(gx) & np.isfinite(gy)
        if ok.sum() > 5:
            slope, intercept = np.polyfit(gx[ok], gy[ok], 1)
            xs = np.linspace(np.nanpercentile(gx[ok], 2), np.nanpercentile(gx[ok], 98), 50)
            ax.plot(xs, slope * xs + intercept, lw=2)

    ax.axhline(0, color="0.2", lw=0.8, ls="--")
    ax.set_xlabel("Δ effective-state PC1 vs fentanyl")
    ax.set_ylabel("Δ FOOOF exponent vs fentanyl")
    ax.set_title("FOOOF Δexponent vs effective-state PC1")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_delta_exponent_vs_effective_PC1.png")
    plt.close(fig)

# Δexponent vs ΔIsing beta
if "delta_ising_beta_param_vs_fentanyl" in plot_df.columns:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=220)

    for region, g in plot_df.groupby("region"):
        ax.scatter(
            g["delta_ising_beta_param_vs_fentanyl"],
            g["delta_exponent_vs_fentanyl"],
            s=10,
            alpha=0.35,
            label=region,
        )

        gx = g["delta_ising_beta_param_vs_fentanyl"].values
        gy = g["delta_exponent_vs_fentanyl"].values
        ok = np.isfinite(gx) & np.isfinite(gy)
        if ok.sum() > 5:
            slope, intercept = np.polyfit(gx[ok], gy[ok], 1)
            xs = np.linspace(np.nanpercentile(gx[ok], 2), np.nanpercentile(gx[ok], 98), 50)
            ax.plot(xs, slope * xs + intercept, lw=2)

    ax.axhline(0, color="0.2", lw=0.8, ls="--")
    ax.set_xlabel("Δ adaptive-Ising β vs fentanyl")
    ax.set_ylabel("Δ FOOOF exponent vs fentanyl")
    ax.set_title("FOOOF Δexponent vs adaptive-Ising β")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_delta_exponent_vs_delta_IsingBeta.png")
    plt.close(fig)

# Ising beta/c summary plot
ising_plot = sub_region[sub_region["state_label"].isin(["fentanyl", "propofol_early", "propofol_late"])].copy()
if not ising_plot.empty:
    for ycol, ylabel, fname in [
        ("ising_beta_mean", "adaptive-Ising β", "09_state_region_IsingBeta.png"),
        ("ising_c_mean", "adaptive-Ising c", "09_state_region_IsingC.png"),
        ("ising_dist_crit_mean", "1 - adaptive-Ising β", "09_state_region_IsingDistance.png"),
    ]:
        if ycol not in ising_plot.columns:
            continue

        fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=220)

        state_order = ["fentanyl", "propofol_early", "propofol_late"]
        x_map = {s: i for i, s in enumerate(state_order)}

        for region, g in ising_plot.groupby("region"):
            means = []
            xs = []
            for s in state_order:
                vals = g[g["state_label"] == s][ycol].values
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    xs.append(x_map[s])
                    means.append(np.mean(vals))
            ax.plot(xs, means, marker="o", lw=2, label=region)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["fentanyl", "early", "late"])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by state/region")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(OUT_DIR / fname)
        plt.close(fig)


# ============================================================
# 15. Save run summary
# ============================================================

run_summary = {
    "root": str(ROOT),
    "out_dir": str(OUT_DIR),

    "epoch_sec": EPOCH_SEC,
    "epoch_step_sec": EPOCH_STEP_SEC,
    "target_fs": TARGET_FS,

    "fooof_f_range": FOOOF_F_RANGE,
    "fooof_peak_width_limits": FOOOF_PEAK_WIDTH_LIMITS,
    "fooof_max_n_peaks": FOOOF_MAX_N_PEAKS,
    "fooof_min_peak_height": FOOOF_MIN_PEAK_HEIGHT,
    "fooof_peak_threshold": FOOOF_PEAK_THRESHOLD,
    "fooof_aperiodic_mode": FOOOF_APERIODIC_MODE,
    "min_fooof_r2": MIN_FOOOF_R2,
    "max_fooof_error": MAX_FOOOF_ERROR,

    "ising_alpha_band": ISING_ALPHA_BAND,
    "ising_acf_max_lag_sec": ISING_ACF_MAX_LAG_SEC,
    "ising_beta_bounds": ISING_BETA_BOUNDS,
    "ising_c_bounds": ISING_C_BOUNDS,
    "ising_min_r2": ISING_MIN_R2,
    "ising_signal_qc_min_full_sec": ISING_SIGNAL_QC_MIN_FULL_SEC,
    "ising_signal_qc_min_std": ISING_SIGNAL_QC_MIN_STD,
    "ising_signal_qc_min_rms": ISING_SIGNAL_QC_MIN_RMS,
    "ising_signal_qc_min_log_total_power_1_100": ISING_SIGNAL_QC_MIN_LOG_TOTAL_POWER_1_100,
    "ising_signal_qc_min_median_log10_psd_1_90": ISING_SIGNAL_QC_MIN_MEDIAN_LOG10_PSD_1_90,
    "ising_signal_qc_low_quantile": ISING_SIGNAL_QC_LOW_QUANTILE,
    "ising_signal_qc_min_group_n_for_quantile": ISING_SIGNAL_QC_MIN_GROUP_N_FOR_QUANTILE,
    "ising_note": (
        "Adaptive-Ising parameters are inferred from state-level alpha-band "
        "autocorrelation, treating the alpha-band channel signal as empirical "
        "observation of m(t). beta/c are fitted in resampled sample-lag units."
    ),

    "n_raw_epoch_features": int(len(features)),
    "n_QC_epoch_features": int(len(df)),
    "n_state_ising_rows": int(len(ising_state)),
    "n_animals_QC": int(df["animal_id"].nunique()) if len(df) else 0,
}

with open(OUT_DIR / "run_summary_step2_fooof_ising_control.json", "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)


print("\n" + "=" * 120)
print("[DONE]")
print("=" * 120)
print(f"OUT_DIR = {OUT_DIR}")
print("Key outputs:")
print("  02_epoch_channel_features_fooof_raw.csv")
print("  02_state_channel_adaptive_ising_raw.csv  # includes Ising full-state signal QC flags")
print("  03_epoch_channel_features_fooof_QC_with_stateIsing.csv")
print("  04_effective_state_pca_info.csv")
print("  05_epoch_channel_features_fooof_with_delta_Ising.csv")
print("  06_subject_region_state_summary_fooof_Ising.csv")
print("  07_animal_level_signflip_tests_all_and_sedation.csv")
print("  08_model_terms_fooof_delta_exponent_IsingControl.csv")
print("  09_paired_delta_exponent_all_propofol_early.png")
print("  09_paired_delta_exponent_all_propofol_late.png")
print("  09_paired_delta_exponent_sedation_only_propofol_early.png")
print("  09_delta_exponent_vs_effective_PC1.png")
print("  09_delta_exponent_vs_delta_IsingBeta.png")
print("  09_state_region_IsingBeta.png")
print("  09_state_region_IsingC.png")
print("  09_state_region_IsingDistance.png")
