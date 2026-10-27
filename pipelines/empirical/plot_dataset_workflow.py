# -*- coding: utf-8 -*-
"""图3示意图：丙泊酚验证流程。布局为a镇静流程与分析状态、b猪ECoG/EThG记录布局、c代表性原始波形、d PSD与FOOOF非周期拟合；a占据首行全宽，b/c/d紧凑排列于第二行。区域配色为皮层蓝色、丘脑橙色；状态2绿色、状态4紫色、其他灰色。最终7.16英寸论文宽度下所有可见图中文字至少10磅；d采用两个并排近正方形坐标轴，纵轴完整包含FOOOF拟合以避免截断；a报告各状态可用动物数及状态2到4的配对动物数。输入为pig.png及ds003380 EDF文件；输出为原文列出的PNG和PDF流程图。 / 
Figure 3 schematic: propofol validation workflow.

Layout:
    a) Sedation protocol and analysis states
    b) Pig ECoG/EThG recording layout
    c) Representative raw waveforms
    d) PSD and FOOOF aperiodic fits

Layout structure:
    a a a
    b c d

Color logic:
    Region colors:
        Cortex   = blue
        Thalamus = orange
    State colors:
        State 2  = green
        State 4  = purple
        Other    = gray

Key changes:
    1) All visible in-figure text is audited to remain at or above 10 pt at the final 7.16-inch manuscript width.
    2) Panel d uses two near-square PSD axes side by side, matching the reference style.
    3) Panel d y-limits include the full FOOOF aperiodic fits to avoid clipping.
    4) Cortex/Thalamus are blue/orange; State 2/State 4 are green/purple.
    5) Panel a spans the full top row; b/c/d are compact second-row panels.
    6) Panel a reports available animal n for each state and paired n for State 2→4.

Inputs:
    pig.png
    OpenNeuro ds003380 EDF files

Outputs:
    figure3_propofol_workflow_abcd_v13_10pt.png
    figure3_propofol_workflow_abcd_v13_10pt.pdf
"""

import os
import re
import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.signal import welch, detrend, resample_poly

try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False

try:
    from fooof import FOOOF
    FOOOF_AVAILABLE = True
except Exception:
    FOOOF_AVAILABLE = False


# ============================================================
# 用户设置。 / User settings
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", REPOSITORY_ROOT / "work")).resolve()
ROOT = Path(os.environ.get("PIG_DATA_ROOT", REPOSITORY_ROOT / "data" / "external" / "ds003380")).resolve()
RESULTS_ROOT = Path(os.environ.get(
    "PIG_RESULTS_ROOT",
    PROJECT_ROOT / "ALLEN" / "results" / "empirical_propofol",
)).resolve()
WORKFLOW_DIR = RESULTS_ROOT / "workflow_figure"
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
PIG_IMG = os.environ.get("PIG_IMAGE_PATH", str(REPOSITORY_ROOT / "data" / "external" / "pig.png"))

OUT_PNG = WORKFLOW_DIR / "propofol_workflow.png"
OUT_PDF = WORKFLOW_DIR / "propofol_workflow.pdf"
DPI = 600

# 匹配IEEE/EMBC双栏论文的最终印刷尺寸。 / Match the final printed size used by IEEE/EMBC two-column manuscripts.
# v12略微增大图内字体，因为图2插入论文后 / v12 uses slightly larger in-figure fonts because Fig. 2 looked undersized
# 显得过小。 / after insertion into the manuscript.
# 由于LaTeX使用按正文宽度插图的命令，超过10英寸的 / Because LaTeX uses \includegraphics[width=\textwidth]{...}, a 10+ inch
# 源图会缩小，导致所有标签过小。使 / source figure is downscaled and all labels become too small. Keeping the
# PDF接近最终正文全宽可保持预期字体大小。 / PDF close to the final full-text width preserves the intended font sizes.
FIG_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 5.15

ROTATE_PIG_180 = True

# 代表性数据选择。 / Representative data selection
REPRESENT_ANIMAL = None
PREFERRED_ANIMALS = ["P_737", "P_743", "P_746", "P_752", "P_728", "P_738"]

REP_CORTEX_CH = None
REP_THAL_CH = None

REP_CORTEX_CH_CANDIDATES = [9, 10, 6, 7, 8, 13, 14, 11, 12]
REP_THAL_CH_CANDIDATES = [15, 16]

MIN_SIGNAL_STD = 1e-12
MIN_MEDIAN_LOG10_PSD = -30.0
MAX_WITHIN_CHANNEL_LOGPSD_RANGE = 12.0

# 原始波形面板。 / Raw waveform panel
RAW_DURATION_SEC = 5.0
RAW_SEARCH_START_SEC = 10.0
RAW_SEARCH_STEP_SEC = 2.0

# PSD与FOOOF面板。 / PSD / FOOOF panel
PSD_START_SEC = 20.0
PSD_USE_SEC = 120.0

TARGET_FS = 500.0
PSD_NPERSEG_SEC = 4.0

FOOOF_F_RANGE = [1.0, 90.0]
FOOOF_PEAK_WIDTH_LIMITS = [1.0, 12.0]
FOOOF_MAX_N_PEAKS = 8
FOOOF_MIN_PEAK_HEIGHT = 0.05
FOOOF_PEAK_THRESHOLD = 2.0
FOOOF_APERIODIC_MODE = "fixed"

# 样本量来自Frasch等人在Scientific Data 2021中的表3。 / Sample sizes from Frasch et al. Scientific Data 2021, Table 3.
# 这些是各状态的可用数据集数；主要状态2到状态4比较 / These are available datasets per state; the main State 2→State 4 contrast is paired
# 在接受丙泊酚的动物中配对，因此n=6。 / across the propofol-exposed animals, hence n=6.
STATE_SAMPLE_N = {
    "State 1": 11,
    "State 2": 11,
    "State 3": 6,
    "State 4": 6,
    "State 5": 4,
}
MAIN_CONTRAST_N = 6


# ============================================================
# 图形样式。 / Figure style
# ============================================================

# 区域颜色：皮层与丘脑。 / Region colors: cortex / thalamus
CORTEX_COLOR = "#3B6EA8"     # 蓝色。 / blue
THAL_COLOR   = "#D97706"     # 橙色。 / orange

# 状态颜色：状态2与状态4。 / State colors: state 2 / state 4
STATE2_LINE  = "#2E7D32"     # 绿色。 / green
STATE4_LINE  = "#8E5EA2"     # 紫色。 / purple
STATE2_FILL  = "#DDEEDC"     # 浅绿色。 / light green
STATE4_FILL  = "#E9D9F0"     # 浅紫色。 / light purple
GRAY_STATE   = "#E8E8E8"

DARK  = "#222222"
MID   = "#666666"
LIGHT = "#D4D4D4"

# 字体大小按最终7.16英寸宽的印刷图指定。 / Font sizes are specified for the final 7.16-inch-wide printed figure.
# 所有可见文字至少为10磅，面板字母更大。 / Every visible text object is kept at or above 10 pt; panel letters are larger.
FONT_PANEL = 13.5
FONT_TITLE = 10.5
FONT_TEXT = 10.0
FONT_TICK = 10.0
FONT_ANNOT = 10.0
FONT_LEGEND = 10.0
MIN_FINAL_FONT_PT = 10.0

plt.rcParams.update({
    # 使用类似Times的衬线字体，匹配论文正文及其他图。 / Use a Times-like serif font to match the manuscript body and other figures.
    # 在Windows上解析为Times New Roman，否则matplotlib / On Windows this will resolve to Times New Roman; otherwise matplotlib
    # 回退到兼容的衬线字体。 / falls back to a compatible serif font.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FONT_TEXT,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_TEXT,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 猪与电极的几何布局。 / Pig and electrode geometry
# ============================================================

PIG_CX = 0.650
PIG_CY = 0.505
# 最大绘图框；实际显示宽高根据 / Maximum drawing box. The actual displayed width/height are computed
# 裁剪图像宽高比及坐标轴物理宽高比计算， / from the cropped image aspect ratio and the physical axes aspect ratio,
# 防止猪头图像拉伸或压缩。 / preventing the pig head from being stretched or compressed.
PIG_MAX_W = 0.90
PIG_MAX_H = 0.76
PIG_ALPHA = 0.30

ELEC_CX = 0.650
ELEC_BREGMA_Y = 0.360
ELEC_SX = 0.0068
ELEC_SY = 0.0100


# ============================================================
# 通用绘图辅助函数。 / General drawing helpers
# ============================================================

def add_arrow(ax, xy1, xy2, color=MID, lw=1.0, ms=9, style="-|>", zorder=5):
    arr = FancyArrowPatch(
        xy1, xy2,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def add_round_box(ax, x, y, w, h, fc, ec="#D8D8D8", lw=0.8, r=0.018, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def crop_white_margin(img, white_thresh=0.985):
    arr = img.copy()

    if arr.ndim == 2:
        gray = arr
    else:
        gray = arr[..., :3].mean(axis=2)

    mask = gray < white_thresh
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]


def map_electrode(x_mm, ap_mm):
    return ELEC_CX + x_mm * ELEC_SX, ELEC_BREGMA_Y + ap_mm * ELEC_SY


def draw_label_line(ax, text, y, color=MID):
    """绘制左侧解剖标签和不重叠的引导线。标签按固定锚点右对齐，线从小间隙后开始，使Temporal和Occipital等较长文字不与水平线重叠。 / Draw a left anatomical label followed by a non-overlapping guide line.

    The label is right-aligned to a fixed anchor and the line starts after a
    small gap, so longer words such as ``Temporal`` and ``Occipital`` never
    overlap the horizontal guide line.
    """
    x_text_right = 0.245
    x_line_start = 0.265
    x_line_end = 0.435

    ax.text(
        x_text_right, y, text,
        ha="right", va="center",
        fontsize=FONT_TEXT,
        color=color,
        zorder=10
    )

    ax.plot(
        [x_line_start, x_line_end],
        [y, y],
        color="#D8D8D8",
        lw=0.65,
        zorder=1
    )


def image_extent_preserve_aspect(ax, img, cx, cy, max_w, max_h):
    """返回保持原图宽高比的图像范围。matplotlib的aspect='auto'会填满指定范围，面板几何变化时可能使猪头显得过窄；此处同时根据裁剪图像和坐标轴物理宽高比调整。 / Return an image extent that preserves the source-image aspect ratio.

    Matplotlib's ``aspect='auto'`` fills the requested extent and can make the
    pig head appear unnaturally narrow when the panel geometry changes. Here
    the extent is adjusted using both the cropped image aspect ratio and the
    physical axes aspect ratio.
    """
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()

    axes_ratio = bbox.width / max(bbox.height, 1e-9)
    image_ratio = img.shape[1] / max(img.shape[0], 1)

    h = max_h
    w = h * image_ratio / max(axes_ratio, 1e-9)

    if w > max_w:
        w = max_w
        h = w * axes_ratio / max(image_ratio, 1e-9)

    return [
        cx - w / 2,
        cx + w / 2,
        cy - h / 2,
        cy + h / 2,
    ]


def add_panel_label(ax, letter, x=0.000, y=1.000):
    """将面板字母置于绘图区边缘或外侧，避免a/b/c/d占用实际绘图空间，使整行更紧凑。 / Panel letters placed just outside/at the plotting block.

    This avoids using real plotting space for the a/b/c/d labels and makes
    the row look more compact.
    """
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FONT_PANEL,
        weight="bold",
        color=DARK,
        clip_on=False,
        zorder=30,
    )


# ============================================================
# 数据辅助函数。 / Data helpers
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


def maybe_resample(data, fs, target_fs):
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


def robust_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.nanstd(x))


def robust_rms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.sqrt(np.nanmean(x ** 2)))


def zscore_for_display(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    x = x - np.nanmean(x)
    s = np.nanstd(x)

    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x)

    return x / s


def compute_psd_and_fooof(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < int(fs * 5):
        return None

    if np.nanstd(x) <= MIN_SIGNAL_STD:
        return None

    x = detrend(x, type="linear")

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

    ok = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 0) & (psd > 0)
    freqs = freqs[ok]
    psd = psd[ok]

    if len(freqs) < 20:
        return None

    psd_clean = interpolate_line_noise(freqs, psd, bands=((48.0, 52.0),))

    fmask = (freqs >= FOOOF_F_RANGE[0]) & (freqs <= FOOOF_F_RANGE[1])

    median_log10_psd = (
        float(np.nanmedian(np.log10(np.maximum(psd_clean[fmask], 1e-300))))
        if fmask.sum() else np.nan
    )

    out = {
        "freqs": freqs,
        "psd": psd_clean,
        "success": False,
        "exponent": np.nan,
        "offset": np.nan,
        "r2": np.nan,
        "median_log10_psd": median_log10_psd,
        "freqs_fit": None,
        "aperiodic_log10": None,
        "full_fit_log10": None,
    }

    if not FOOOF_AVAILABLE:
        return out

    fm = FOOOF(
        peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
        max_n_peaks=FOOOF_MAX_N_PEAKS,
        min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
        peak_threshold=FOOOF_PEAK_THRESHOLD,
        aperiodic_mode=FOOOF_APERIODIC_MODE,
        verbose=False,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fm.fit(freqs, psd_clean, FOOOF_F_RANGE)

        ap = fm.aperiodic_params_

        if ap is None or len(ap) < 2:
            return out

        offset = float(ap[0])
        exponent = float(ap[1])

        f_fit = np.asarray(fm.freqs, dtype=float)
        ap_log10 = offset - exponent * np.log10(f_fit)

        out.update({
            "success": True,
            "exponent": exponent,
            "offset": offset,
            "r2": float(fm.r_squared_) if fm.r_squared_ is not None else np.nan,
            "freqs_fit": f_fit,
            "aperiodic_log10": ap_log10,
            "full_fit_log10": np.asarray(fm.fooofed_spectrum_, dtype=float)
            if fm.fooofed_spectrum_ is not None else None,
        })

        return out

    except Exception:
        return out


def build_edf_inventory(root):
    edf_files = sorted(root.glob("sub-*/eeg/*.edf"))
    rows = []

    for p in edf_files:
        rows.append({
            "subject": p.parts[-3],
            "edf_path": str(p),
            "edf_name": p.name,
            "animal_id": parse_animal_from_name(p.name),
            "state_raw": parse_state_from_name(p.name),
        })

    return rows


def choose_state_pair_inventory(inv_rows):
    by_animal = {}

    for r in inv_rows:
        animal = r["animal_id"]
        state = r["state_raw"]

        if animal is None or state is None:
            continue

        by_animal.setdefault(animal, {})[state] = r

    if REPRESENT_ANIMAL is not None:
        animals = [REPRESENT_ANIMAL]
    else:
        animals = list(PREFERRED_ANIMALS)
        animals += [a for a in sorted(by_animal.keys()) if a not in animals]

    pairs = []

    for animal in animals:
        if animal in by_animal and "2" in by_animal[animal] and "4" in by_animal[animal]:
            pairs.append((animal, by_animal[animal]["2"], by_animal[animal]["4"]))

    if not pairs:
        raise RuntimeError("No animal with both state2 and state4 was found.")

    return pairs


def load_state_channels(edf_meta, channels):
    if not MNE_AVAILABLE:
        raise RuntimeError("mne is not available. Please install mne or run in the analysis environment.")

    edf_path = Path(edf_meta["edf_path"])
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    fs = float(raw.info["sfreq"])

    need_chs = [edf_chan_name(ch) for ch in channels]
    present = [ch for ch in need_chs if ch in raw.ch_names]

    if len(present) == 0:
        raise RuntimeError(f"No requested channels found in {edf_path.name}")

    raw.pick_channels(present, ordered=True)

    data = raw.get_data()
    data, fs2 = maybe_resample(data, fs, TARGET_FS)

    out = {}

    for i, ch_name in enumerate(present):
        m = re.search(r"Chan\s+([0-9]+)", ch_name)
        ch_num = int(m.group(1))
        out[ch_num] = data[i, :]

    return out, fs2


def get_psd_segment(x, fs):
    n = len(x)

    start = int(PSD_START_SEC * fs)
    dur = int(PSD_USE_SEC * fs)

    if start + dur > n:
        start = 0
        dur = min(n, int(PSD_USE_SEC * fs))

    stop = min(n, start + dur)

    return x[start:stop]


def signal_summary(x, fs):
    std = robust_std(x)
    rms = robust_rms(x)

    psd_fit = compute_psd_and_fooof(get_psd_segment(x, fs), fs)

    if psd_fit is None:
        median_log10_psd = np.nan
        fooof_success = False
        exponent = np.nan
    else:
        median_log10_psd = psd_fit.get("median_log10_psd", np.nan)
        fooof_success = bool(psd_fit.get("success", False))
        exponent = psd_fit.get("exponent", np.nan)

    valid = (
        np.isfinite(std)
        and std > MIN_SIGNAL_STD
        and np.isfinite(rms)
        and rms > MIN_SIGNAL_STD
        and np.isfinite(median_log10_psd)
        and median_log10_psd > MIN_MEDIAN_LOG10_PSD
    )

    return {
        "std": std,
        "rms": rms,
        "median_log10_psd": median_log10_psd,
        "fooof_success": fooof_success,
        "exponent": exponent,
        "valid": valid,
    }


def evaluate_candidate(state2_data, state4_data, fs, cortex_ch, thal_ch):
    keys = [
        ("cortex_state2", state2_data.get(cortex_ch, None)),
        ("cortex_state4", state4_data.get(cortex_ch, None)),
        ("thal_state2", state2_data.get(thal_ch, None)),
        ("thal_state4", state4_data.get(thal_ch, None)),
    ]

    summaries = {}

    for key, x in keys:
        if x is None:
            return None

        summaries[key] = signal_summary(x, fs)

    vals = [s["median_log10_psd"] for s in summaries.values()]
    valid_flags = [s["valid"] for s in summaries.values()]
    success_count = sum(s["fooof_success"] for s in summaries.values())

    vals_arr = np.asarray(vals, dtype=float)

    if np.any(~np.isfinite(vals_arr)):
        min_log = -999
        log_range = 999
    else:
        min_log = float(np.min(vals_arr))
        log_range = float(np.max(vals_arr) - np.min(vals_arr))

    cortex_range = abs(
        summaries["cortex_state2"]["median_log10_psd"]
        - summaries["cortex_state4"]["median_log10_psd"]
    )

    thal_range = abs(
        summaries["thal_state2"]["median_log10_psd"]
        - summaries["thal_state4"]["median_log10_psd"]
    )

    range_penalty = 0.20 * log_range + 0.35 * max(cortex_range, thal_range)

    valid_bonus = 100.0 if all(valid_flags) else 0.0
    success_bonus = 1.5 * success_count

    score = valid_bonus + min_log + success_bonus - range_penalty

    if summaries["thal_state4"]["median_log10_psd"] <= MIN_MEDIAN_LOG10_PSD:
        score -= 100.0

    if thal_range > MAX_WITHIN_CHANNEL_LOGPSD_RANGE:
        score -= 80.0

    return {
        "score": score,
        "summaries": summaries,
        "all_valid": all(valid_flags),
        "cortex_ch": cortex_ch,
        "thal_ch": thal_ch,
    }


def find_representative_window(x, fs, duration_sec=RAW_DURATION_SEC):
    x = np.asarray(x, dtype=float)
    n = len(x)

    win = int(duration_sec * fs)
    step = int(RAW_SEARCH_STEP_SEC * fs)
    start0 = int(RAW_SEARCH_START_SEC * fs)

    if n < win:
        return x, 0.0

    starts = list(range(start0, n - win + 1, step))

    if not starts:
        starts = list(range(0, n - win + 1, step))

    rows = []

    for st in starts:
        seg = x[st:st + win]
        seg = seg[np.isfinite(seg)]

        if len(seg) < win * 0.80:
            continue

        sd = np.nanstd(seg)
        rms = np.sqrt(np.nanmean(seg ** 2))

        if not np.isfinite(sd) or sd <= MIN_SIGNAL_STD:
            continue

        if not np.isfinite(rms) or rms <= MIN_SIGNAL_STD:
            continue

        rows.append((st, rms, sd))

    if not rows:
        return x[:win], 0.0

    rms_vals = np.asarray([r[1] for r in rows])
    med = np.nanmedian(rms_vals)

    best = min(rows, key=lambda r: abs(np.log(r[1] + 1e-300) - np.log(med + 1e-300)))
    st = best[0]

    return x[st:st + win], st / fs


def load_representative_data():
    inv = build_edf_inventory(ROOT)
    pairs = choose_state_pair_inventory(inv)

    cortex_candidates = [REP_CORTEX_CH] if REP_CORTEX_CH is not None else REP_CORTEX_CH_CANDIDATES
    thal_candidates = [REP_THAL_CH] if REP_THAL_CH is not None else REP_THAL_CH_CANDIDATES

    needed_channels = sorted(set(cortex_candidates + thal_candidates))

    best = None

    for animal, state2_meta, state4_meta in pairs:
        try:
            state2_data, fs2 = load_state_channels(state2_meta, needed_channels)
            state4_data, fs4 = load_state_channels(state4_meta, needed_channels)

            if abs(fs2 - fs4) > 1e-6:
                continue

            fs = fs2

            for cortex_ch in cortex_candidates:
                if cortex_ch not in state2_data or cortex_ch not in state4_data:
                    continue

                for thal_ch in thal_candidates:
                    if thal_ch not in state2_data or thal_ch not in state4_data:
                        continue

                    eva = evaluate_candidate(state2_data, state4_data, fs, cortex_ch, thal_ch)

                    if eva is None:
                        continue

                    candidate = {
                        "animal_id": animal,
                        "state2_meta": state2_meta,
                        "state4_meta": state4_meta,
                        "state2_data": state2_data,
                        "state4_data": state4_data,
                        "fs": fs,
                        "cortex_ch": cortex_ch,
                        "thal_ch": thal_ch,
                        "score": eva["score"],
                        "summaries": eva["summaries"],
                        "all_valid": eva["all_valid"],
                    }

                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

            if best is not None and best["all_valid"] and animal in PREFERRED_ANIMALS:
                break

        except Exception as e:
            print(f"[WARN] Failed candidate animal {animal}: {e}")
            continue

    if best is None:
        raise RuntimeError("No valid representative animal/channel combination found.")

    cortex_ch = best["cortex_ch"]
    thal_ch = best["thal_ch"]
    fs = best["fs"]

    example = {
        "animal_id": best["animal_id"],
        "state2_meta": best["state2_meta"],
        "state4_meta": best["state4_meta"],
        "fs": fs,
        "cortex_ch": cortex_ch,
        "thal_ch": thal_ch,
        "score": best["score"],
        "summaries": best["summaries"],
        "data": {
            "state2": {
                cortex_ch: best["state2_data"][cortex_ch],
                thal_ch: best["state2_data"][thal_ch],
            },
            "state4": {
                cortex_ch: best["state4_data"][cortex_ch],
                thal_ch: best["state4_data"][thal_ch],
            },
        }
    }

    example["raw"] = {}

    for region_name, ch in [("Cortex", cortex_ch), ("Thalamus", thal_ch)]:
        example["raw"][region_name] = {}

        for state_name in ["state2", "state4"]:
            seg, t0 = find_representative_window(
                example["data"][state_name][ch],
                fs,
                duration_sec=RAW_DURATION_SEC
            )

            example["raw"][region_name][state_name] = {
                "trace": seg,
                "t0": t0,
                "channel": ch,
            }

    example["psd"] = {}

    for region_name, ch in [("Cortex", cortex_ch), ("Thalamus", thal_ch)]:
        example["psd"][region_name] = {}

        for state_name in ["state2", "state4"]:
            x_psd = get_psd_segment(example["data"][state_name][ch], fs)
            example["psd"][region_name][state_name] = compute_psd_and_fooof(x_psd, fs)

    return example


# ============================================================
# 面板a：实验流程。 / Panel a: protocol
# ============================================================

def draw_state_timeline_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_label(ax, "a", x=0.000, y=0.995)

    # 已注释的文字绘制调用。 / ax.text(
    # 标题位置与文字：镇静流程和分析状态。 / 0.50, 0.985, "Sedation protocol and analysis states",
    # 水平居中、垂直顶端对齐。 / ha="center", va="top",
    # 字体大小9.8磅。 / fontsize=9.8,
    # 粗体字重。 / weight="bold",
    # 深色文字。 / color=DARK
    # )

    states = [
        ("State 1", "Isoflurane\nsedation",     STATE_SAMPLE_N["State 1"], GRAY_STATE),
        ("State 2", "Fentanyl\nbaseline",      STATE_SAMPLE_N["State 2"], STATE2_FILL),
        ("State 3", "Propofol\ninduction/BSP", STATE_SAMPLE_N["State 3"], GRAY_STATE),
        ("State 4", "Moderate\npropofol",      STATE_SAMPLE_N["State 4"], STATE4_FILL),
        ("State 5", "60 min\nmoderate",       STATE_SAMPLE_N["State 5"], GRAY_STATE),
    ]

    # 加宽框体、减少内部留白，使面板a紧凑。 / Wider boxes and less internal whitespace; keeps panel a compact.
    x_start, x_end = 0.080, 0.920
    # 增高框体，为状态、样本数和描述提供更多垂直空间。 / Taller boxes: more vertical room for State / n / description.
    # 增加高度时降低纵坐标，使顶部保持靠近流程引导线。 / Keep the top near the protocol guide line by lowering y while increasing h.
    y = 0.420
    h = 0.340
    gap = 0.018

    n = len(states)
    w = (x_end - x_start - gap * (n - 1)) / n

    centers = []

    for i, (state, desc, n_animals, fc) in enumerate(states):
        x = x_start + i * (w + gap)

        # 不显示框体边界；状态2和状态4通过填充 / No visible box border. State 2 and State 4 are indicated by their fill
        # 颜色及后续标签区分，而非通过轮廓线区分。 / colors and downstream labels, rather than by an outline.
        add_round_box(
            ax, x, y, w, h,
            fc=fc,
            ec="none",
            lw=0.0,
            r=0.018,
            zorder=2
        )

        ax.text(
            x + w / 2, y + h * 0.74, state,
            ha="center", va="center",
            fontsize=FONT_TITLE,
            weight="bold",
            color=DARK,
            zorder=4
        )

        # 该状态的可用动物数；状态2总体可用n=11， / Available animal count for this state. State 2 has n=11 available overall,
        # 而状态2到状态4比较采用配对n=6。 / while the State 2→4 contrast uses paired n=6 animals.
        if state == "State 2":
            n_color = STATE2_LINE
        elif state == "State 4":
            n_color = STATE4_LINE
        else:
            n_color = MID
        ax.text(
            x + w / 2, y + h * 0.515, f"n={n_animals}",
            ha="center", va="center",
            fontsize=FONT_TEXT,
            weight="bold" if state in {"State 2", "State 4"} else "normal",
            color=n_color,
            zorder=4
        )

        ax.text(
            x + w / 2, y + h * 0.285, desc,
            ha="center", va="center",
            fontsize=FONT_TEXT,
            color=DARK,
            linespacing=0.92,
            zorder=4
        )

        centers.append((x + w / 2, y + h / 2))

    for i in range(len(centers) - 1):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]

        add_arrow(
            ax,
            (x1 + w * 0.48, y1),
            (x2 - w * 0.48, y2),
            color="#8D8D8D",
            lw=0.9,
            ms=7
        )

    ax.text(
        0.080, 0.860, "Pre-ischemic sedation period",
        ha="left", va="center",
        fontsize=FONT_TEXT,
        color=MID
    )

    ax.plot(
        [0.080, 0.920],
        [0.825, 0.825],
        color="#CFCFCF",
        lw=0.8
    )

    state2_center = centers[1]
    state4_center = centers[3]
    y_brace = 0.315

    ax.plot(
        [state2_center[0], state2_center[0], state4_center[0], state4_center[0]],
        [y_brace + 0.026, y_brace, y_brace, y_brace + 0.026],
        color=DARK,
        lw=1.15
    )

    ax.text(
        (state2_center[0] + state4_center[0]) / 2,
        y_brace - 0.055,
        f"Main contrast: State 4 − State 2 (paired n={MAIN_CONTRAST_N})",
        ha="center", va="top",
        fontsize=FONT_TEXT,
        color=DARK
    )



# ============================================================
# 面板b：猪的布局。 / Panel b: pig layout
# ============================================================

def draw_pig_electrode_panel(ax, pig_img_path):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    img = plt.imread(pig_img_path)

    if ROTATE_PIG_180:
        img = np.rot90(img, 2)

    img = crop_white_margin(img, white_thresh=0.985)

    pig_extent = image_extent_preserve_aspect(
        ax,
        img,
        cx=PIG_CX,
        cy=PIG_CY,
        max_w=PIG_MAX_W,
        max_h=PIG_MAX_H,
    )

    ax.imshow(
        img,
        extent=pig_extent,
        aspect="auto",
        alpha=PIG_ALPHA,
        zorder=0
    )

    add_panel_label(ax, "b", x=0.000, y=1.000)

    # 已注释的文字绘制调用。 / ax.text(
    # 标题位置与文字：猪ECoG/EThG记录布局。 / 0.50, 0.985, "Pig ECoG/EThG recording layout",
    # 水平居中、垂直顶端对齐。 / ha="center", va="top",
    # 字体大小9.3磅。 / fontsize=9.3,
    # 粗体字重。 / weight="bold",
    # 深色文字。 / color=DARK
    # )

    y_mid_top = map_electrode(0, 32)[1] + 0.12
    y_mid_bottom = map_electrode(0, -10)[1] - 0.12

    ax.plot(
        [ELEC_CX, ELEC_CX],
        [y_mid_bottom, y_mid_top],
        color=LIGHT,
        lw=0.8,
        ls="--",
        zorder=1
    )

    bregma = map_electrode(0, 0)

    ax.add_patch(Circle(
        bregma,
        0.006,
        facecolor=DARK,
        edgecolor="white",
        lw=0.5,
        zorder=7
    ))

    ax.text(
        bregma[0] + 0.145,
        bregma[1] + 0.040,
        "Bregma",
        ha="left",
        va="top",
        fontsize=FONT_TEXT,
        color=MID,
        zorder=10
    )

    ref_y = map_electrode(0, 34)[1]

    ref_pts = [
        (ELEC_CX - 0.022, ref_y),
        (ELEC_CX + 0.022, ref_y),
        (ELEC_CX - 0.040, ref_y - 0.020),
        (ELEC_CX + 0.040, ref_y - 0.020),
    ]

    for x, y in ref_pts:
        ax.add_patch(Circle(
            (x, y),
            0.0058,
            facecolor="white",
            edgecolor="#6D6D6D",
            lw=0.8,
            zorder=6
        ))

    ax.text(
        ELEC_CX,
        ref_y + 0.060,
        "Reference\nnasal bone",
        ha="center",
        va="center",
        fontsize=FONT_TEXT,
        color=MID,
        zorder=10
    )

    ecog_used = {
        "6":  (-12, +30),
        "7":  (+12, +30),
        "8":  (+12, +20),
        "9":  (-12, +10),
        "10": (+12, +10),
        "11": (-24, -10),
        "12": (+24, -10),
        "13": (-12, -10),
        "14": (+12, -10),
    }

    ethg = {
        "RTN": (-9, +2),
        "LD":  (-5, -2),
    }

    label_offsets = {
        "6":  (-0.032, 0.050),
        "7":  (0.032, 0.050),
        "8":  (0.050, 0.010),
        "9":  (-0.052, 0.010),
        "10": (0.052, 0.010),
        "11": (-0.055, -0.038),
        "12": (0.055, -0.038),
        "13": (-0.040, 0.052),
        "14": (0.040, -0.055),
    }

    draw_label_line(ax, "Frontal",   map_electrode(0, +30)[1])
    draw_label_line(ax, "Parietal",  map_electrode(0, +20)[1])
    draw_label_line(ax, "Central",   map_electrode(0, +10)[1])
    draw_label_line(ax, "Temporal",  map_electrode(0, -10)[1] + 0.018)
    draw_label_line(ax, "Occipital", map_electrode(0, -10)[1] - 0.018)

    for ch, (x_mm, ap_mm) in ecog_used.items():
        x, y = map_electrode(x_mm, ap_mm)

        ax.add_patch(Circle(
            (x, y),
            0.0142,
            facecolor=CORTEX_COLOR,
            edgecolor="white",
            lw=0.8,
            zorder=8
        ))

        dx, dy = label_offsets[ch]

        ax.text(
            x + dx,
            y + dy,
            ch,
            ha="center",
            va="center",
            fontsize=FONT_TEXT,
            color=CORTEX_COLOR,
            zorder=9
        )

    ethg_xy = {}

    for nucleus, (x_mm, ap_mm) in ethg.items():
        x, y = map_electrode(x_mm, ap_mm)
        ethg_xy[nucleus] = (x, y)

        ax.add_patch(Circle(
            (x, y),
            0.024,
            facecolor="none",
            edgecolor=THAL_COLOR,
            lw=1.0,
            alpha=0.85,
            zorder=8
        ))

        ax.add_patch(Circle(
            (x, y),
            0.0155,
            facecolor=THAL_COLOR,
            edgecolor="white",
            lw=0.8,
            zorder=9
        ))

    rtn_x, rtn_y = ethg_xy["RTN"]
    ld_x, ld_y = ethg_xy["LD"]

    ax.text(
        rtn_x - 0.125,
        rtn_y + 0.010,
        "RTN (Ch.15)",
        ha="right",
        va="center",
        fontsize=FONT_TEXT,
        color=THAL_COLOR,
        zorder=10
    )

    ax.text(
        ld_x + 0.105,
        ld_y - 0.015,
        "LD (Ch.16)",
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=THAL_COLOR,
        zorder=10
    )

    legend_y = 0.075

    ax.add_patch(Circle(
        (0.085, legend_y),
        0.0105,
        facecolor=CORTEX_COLOR,
        edgecolor="white",
        lw=0.7,
        zorder=10
    ))

    ax.text(
        0.110, legend_y,
        "ECoG cortex, Ch. 6–14",
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=CORTEX_COLOR
    )

    ax.add_patch(Circle(
        (0.085, legend_y - 0.052),
        0.0105,
        facecolor=THAL_COLOR,
        edgecolor="white",
        lw=0.7,
        zorder=10
    ))

    ax.text(
        0.110, legend_y - 0.052,
        "EThG thalamus, Ch. 15–16",
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=THAL_COLOR
    )


# ============================================================
# 面板c：原始波形。 / Panel c: raw waveforms
# ============================================================

def draw_panel_raw_waveforms(ax, example):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_label(ax, "c", x=0.000, y=1.000)

    # 已注释的文字绘制调用。 / ax.text(
    #     0.50, 0.985,
    # 标题文字：代表性原始波形。 / "Representative raw waveforms",
    # 水平居中对齐。 / ha="center",
    # 垂直顶端对齐。 / va="top",
    # 字体大小9.3磅。 / fontsize=9.3,
    # 粗体字重。 / weight="bold",
    # 深色文字。 / color=DARK
    # )

    if example is None:
        ax.text(
            0.5, 0.5,
            "No EDF data loaded.\nCheck ROOT / mne / fooof.",
            ha="center",
            va="center",
            fontsize=FONT_TEXT,
            color=MID
        )
        return

    fs = example["fs"]
    animal_id = example["animal_id"]
    cortex_ch = example["cortex_ch"]
    thal_ch = example["thal_ch"]

    rows = [
        ("Cortex",   "state2", STATE2_LINE, 0.780, "State 2"),
        ("Cortex",   "state4", STATE4_LINE, 0.620, "State 4"),
        ("Thalamus", "state2", STATE2_LINE, 0.380, "State 2"),
        ("Thalamus", "state4", STATE4_LINE, 0.220, "State 4"),
    ]

    # 将轨迹左移，消除面板b和c之间明显的空隙。 / Pull traces leftward to remove the apparent b-c gutter.
    # 对z分数轨迹进行截断并略微压缩垂直间距， / The z-scored traces are also clipped and slightly compressed vertically
    # 避免四行相互重叠。 / so the four rows do not overlap.
    x_trace0 = 0.315
    x_trace1 = 0.975
    amp = 0.025

    ax.text(
        0.030, 0.700,
        "Cortex",
        ha="left",
        va="center",
        fontsize=FONT_TITLE,
        weight="bold",
        color=CORTEX_COLOR
    )

    ax.text(
        0.030, 0.300,
        "Thalamus",
        ha="left",
        va="center",
        fontsize=FONT_TITLE,
        weight="bold",
        color=THAL_COLOR
    )

    for region, state, color, y0, label in rows:
        trace = example["raw"][region][state]["trace"]
        trace = zscore_for_display(trace)
        trace = np.clip(trace, -2.55, 2.55)

        if len(trace) == 0:
            continue

        t = np.arange(len(trace)) / fs
        max_pts = 2200
        step = max(1, int(np.ceil(len(t) / max_pts)))

        t = t[::step]
        trace = trace[::step]

        tx = x_trace0 + (x_trace1 - x_trace0) * (t / max(t[-1], 1e-9))
        ty = y0 + amp * trace

        ax.plot(tx, ty, color=color, lw=0.82, alpha=0.95)

        ax.plot(
            [x_trace0, x_trace1],
            [y0, y0],
            color="#E7E7E7",
            lw=0.55,
            zorder=0
        )

        ax.text(
            0.285, y0,
            label,
            ha="right",
            va="center",
            fontsize=FONT_TEXT,
            color=color
        )

    scalebar_sec = 1.0
    bar_w = (x_trace1 - x_trace0) * scalebar_sec / RAW_DURATION_SEC
    bar_x0 = x_trace1 - bar_w
    bar_y = 0.070

    ax.plot([bar_x0, x_trace1], [bar_y, bar_y], color=DARK, lw=1.0)

    ax.text(
        (bar_x0 + x_trace1) / 2,
        bar_y - 0.030,
        "1 s",
        ha="center",
        va="top",
        fontsize=FONT_TEXT,
        color=DARK
    )

    ax.text(
        0.975, 0.955,
        f"{animal_id} | Ctx {cortex_ch} | Thal {thal_ch}",
        ha="right",
        va="top",
        fontsize=FONT_TEXT,
        color=MID
    )


# ============================================================
# 面板d：PSD与FOOOF。 / Panel d: PSD + FOOOF
# ============================================================

def _beta_text(fit2, fit4):
    b2 = fit2.get("exponent", np.nan) if fit2 is not None else np.nan
    b4 = fit4.get("exponent", np.nan) if fit4 is not None else np.nan

    s2 = rf"S2 $\beta$={b2:.2f}" if np.isfinite(b2) else r"S2 $\beta$=NA"
    s4 = rf"S4 $\beta$={b4:.2f}" if np.isfinite(b4) else r"S4 $\beta$=NA"

    return s2 + "\n" + s4


def _plot_psd_fit_axis(ax, fit2, fit4, title, show_ylabel=True):
    ax.set_xscale("log")

    raw_y = []
    fit_y = []

    for fit, color in [
        (fit2, STATE2_LINE),
        (fit4, STATE4_LINE),
    ]:
        if fit is None:
            continue

        freqs = fit["freqs"]
        psd = fit["psd"]

        mask = (freqs >= 1.0) & (freqs <= 90.0)
        f = freqs[mask]
        y = np.log10(np.maximum(psd[mask], 1e-300))

        # 原始PSD：细实线。 / raw PSD: thin solid
        ax.plot(
            f, y,
            color=color,
            lw=0.85,
            alpha=0.35,
            clip_on=False
        )

        raw_y.extend(list(y[np.isfinite(y)]))

        # FOOOF非周期拟合：虚线。 / FOOOF aperiodic fit: dashed
        if fit.get("success", False) and fit.get("freqs_fit", None) is not None:
            f_fit = np.asarray(fit["freqs_fit"])
            ap_y = np.asarray(fit["aperiodic_log10"])

            fit_mask = (f_fit >= 1.0) & (f_fit <= 90.0) & np.isfinite(ap_y)

            ax.plot(
                f_fit[fit_mask],
                ap_y[fit_mask],
                color=color,
                lw=1.55,
                ls="--",
                alpha=0.98,
                clip_on=False
            )

            fit_y.extend(list(ap_y[fit_mask]))

    ax.set_title(title, fontsize=FONT_TITLE, color=DARK, pad=3.0, linespacing=0.95)
    ax.set_xlim(1, 100)

    # --------------------------------------------------------
    # 纵轴范围。 / y-limit:
    # 原始PSD使用稳健范围，FOOOF拟合须完整包含。 / raw PSD uses robust range, FOOOF fits must be fully included.
    # --------------------------------------------------------
    raw_y = np.asarray(raw_y, dtype=float)
    raw_y = raw_y[np.isfinite(raw_y)]

    fit_y = np.asarray(fit_y, dtype=float)
    fit_y = fit_y[np.isfinite(fit_y)]

    y_low_candidates = []
    y_high_candidates = []

    if len(raw_y):
        y_low_candidates.append(np.nanpercentile(raw_y, 1.0))
        y_high_candidates.append(np.nanpercentile(raw_y, 99.0))

    if len(fit_y):
        y_low_candidates.append(np.nanmin(fit_y))
        y_high_candidates.append(np.nanmax(fit_y))

    if y_low_candidates and y_high_candidates:
        lo = float(np.nanmin(y_low_candidates))
        hi = float(np.nanmax(y_high_candidates))

        span = max(hi - lo, 0.8)
        ax.set_ylim(lo - 0.16 * span, hi + 0.20 * span)

    # 参考样式几何布局：近正方形PSD坐标轴。 / reference-like geometry: near-square PSD axes
    try:
        ax.set_box_aspect(1.0)
    except Exception:
        pass

    ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=FONT_TICK, length=2.6, width=0.85, pad=2.0)

    ax.tick_params(axis="y", labelleft=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(False)

    # beta文字标注。 / beta text
    ax.text(
        0.98, 0.96,
        _beta_text(fit2, fit4),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_TEXT,
        color=DARK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0)
    )


def draw_panel_psd_fooof(ax, example):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_label(ax, "d", x=0.000, y=1.000)

    # 已注释的文字绘制调用。 / ax.text(
    #     0.50, 0.985,
    # 标题文字：PSD与FOOOF非周期拟合。 / "PSD and FOOOF aperiodic fits",
    # 水平居中对齐。 / ha="center",
    # 垂直顶端对齐。 / va="top",
    # 字体大小9.3磅。 / fontsize=9.3,
    # 粗体字重。 / weight="bold",
    # 深色文字。 / color=DARK
    # )

    if example is None:
        ax.text(
            0.5, 0.5,
            "No EDF data loaded.\nPSD/FOOOF panel not drawn.",
            ha="center",
            va="center",
            fontsize=FONT_TEXT,
            color=MID
        )
        return

    cortex_ch = example["cortex_ch"]
    thal_ch = example["thal_ch"]

    # 两个近正方形PSD坐标轴并排。 / Two near-square PSD axes side by side.
    ax_ctx = ax.inset_axes([0.105, 0.210, 0.375, 0.620])
    ax_tha = ax.inset_axes([0.590, 0.210, 0.375, 0.620])

    _plot_psd_fit_axis(
        ax_ctx,
        example["psd"]["Cortex"]["state2"],
        example["psd"]["Cortex"]["state4"],
        f"Cortex\nECoG Ch.{cortex_ch}",
        show_ylabel=True
    )

    _plot_psd_fit_axis(
        ax_tha,
        example["psd"]["Thalamus"]["state2"],
        example["psd"]["Thalamus"]["state4"],
        f"Thalamus\nEThG Ch.{thal_ch}",
        show_ylabel=False
    )

    # 在两个方形坐标轴上方放置紧凑图例。 / Compact legend above the two square axes
    legend_y = 0.945

    ax.plot(
        [0.120, 0.170],
        [legend_y, legend_y],
        color=STATE2_LINE,
        lw=1.4,
        transform=ax.transAxes,
        clip_on=False
    )

    ax.text(
        0.180, legend_y,
        "S2",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=DARK
    )

    ax.plot(
        [0.350, 0.400],
        [legend_y, legend_y],
        color=STATE4_LINE,
        lw=1.4,
        transform=ax.transAxes,
        clip_on=False
    )

    ax.text(
        0.410, legend_y,
        "S4",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=DARK
    )

    ax.plot(
        [0.565, 0.615],
        [legend_y, legend_y],
        color="#555555",
        lw=1.2,
        ls="--",
        transform=ax.transAxes,
        clip_on=False
    )

    ax.text(
        0.625, legend_y,
        "Fit",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_TEXT,
        color=DARK
    )


    ax.text(
        0.535, 0.055,
        "Frequency (Hz)",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=FONT_TEXT,
        color=DARK
    )

    ax.text(
        0.020, 0.520,
        "Power spectral density",
        transform=ax.transAxes,
        ha="center", va="center",
        rotation=90,
        fontsize=FONT_TEXT,
        color=DARK
    )


# ============================================================
# 字体检查。 / Font audit
# ============================================================

def audit_minimum_font_size(fig, minimum_pt=MIN_FINAL_FONT_PT):
    """任一可见文字小于期刊要求时立即报错。 / Fail early if any visible text is smaller than the journal requirement."""
    violations = []
    for artist in fig.findobj(match=plt.Text):
        label = artist.get_text().strip()
        if not artist.get_visible() or not label:
            continue
        size = float(artist.get_fontsize())
        if size + 1e-9 < minimum_pt:
            violations.append((label.replace("\n", " / "), size))

    if violations:
        details = "\n".join(f"  {size:.2f} pt: {label}" for label, size in violations)
        raise RuntimeError(
            f"Figure contains text below {minimum_pt:.1f} pt:\n{details}"
        )


# ============================================================
# 主程序。 / Main
# ============================================================

def main():
    if not os.path.exists(PIG_IMG):
        raise FileNotFoundError(
            f"Cannot find {PIG_IMG}. Put pig.png in the same folder as this script "
            f"or set PIG_IMG to the full image path."
        )

    example = None

    try:
        example = load_representative_data()

        print("[INFO] Loaded representative data:")
        print(f"  animal_id  = {example['animal_id']}")
        print(f"  cortex_ch  = {example['cortex_ch']}")
        print(f"  thal_ch    = {example['thal_ch']}")
        print(f"  score      = {example['score']:.3f}")
        print(f"  state2     = {example['state2_meta']['edf_name']}")
        print(f"  state4     = {example['state4_meta']['edf_name']}")
        print(f"  fs         = {example['fs']}")

        print("\n[INFO] Candidate signal summaries:")
        for k, s in example["summaries"].items():
            print(
                f"  {k:15s} | valid={s['valid']} "
                f"| std={s['std']:.3e} "
                f"| rms={s['rms']:.3e} "
                f"| median_log10_psd={s['median_log10_psd']:.3f} "
                f"| fooof={s['fooof_success']} "
                f"| beta={s['exponent']:.3f}"
            )

    except Exception as e:
        print("[WARN] Could not load representative EDF data.")
        print(f"       {e}")
        print("       Panels c/d will show placeholders.")

    # a横跨全宽，b/c/d位于第二行。 / a spans full width; b/c/d are in second row.
    # d最大，以容纳两个近正方形PSD坐标轴。 / d is largest to allow two near-square PSD axes.
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=150)

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.24, 1.16, 1.70],
        height_ratios=[1, 1.24],
        wspace=0.045,
        hspace=0.000
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    draw_state_timeline_panel(ax_a)
    draw_pig_electrode_panel(ax_b, PIG_IMG)
    draw_panel_raw_waveforms(ax_c, example)
    draw_panel_psd_fooof(ax_d, example)

    plt.subplots_adjust(
        left=0.040,
        right=0.992,
        top=0.970,
        bottom=0.065
    )

    audit_minimum_font_size(fig)

    fig.savefig(
        OUT_PNG,
        dpi=DPI,
        facecolor="white"
    )

    fig.savefig(
        OUT_PDF,
        dpi=DPI,
        facecolor="white"
    )

    plt.close(fig)

    print(f"[SAVED] {OUT_PNG}")
    print(f"[SAVED] {OUT_PDF}")


if __name__ == "__main__":
    main()
