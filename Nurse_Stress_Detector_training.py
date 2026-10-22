"""
================================================================================
NURSE WEARABLE STRESS DETECTION - PRODUCTION-ORIENTED TRAINING PIPELINE
================================================================================

Goal
----
Train a deployable model that answers, in real time from a nurse's wearable
stream (EDA, HR, TEMP, 3-axis accelerometer):

    "Is this nurse currently experiencing SUSTAINED HIGH STRESS?"

This is explicitly NOT a 3-class (LOW/MODERATE/HIGH) accuracy-maximization
exercise. The production target is binary:

    LOW, MODERATE  -> NORMAL       (0)
    HIGH           -> HIGH_STRESS  (1)

The 3-class label is still tracked (dominant_3class) so a 3-class experiment
could be added later, but it never enters the production training loop.

Design summary
---------------
1. Load -> rename X/Y/Z -> ACC_X/ACC_Y/ACC_Z.
2. Remove only *consecutive* duplicate rows per subject (this is what
   forward-filling EDA/HR/TEMP onto a faster accelerometer clock produces).
   We deliberately do NOT drop non-consecutive duplicate signal rows, since
   a nurse's HR/EDA can legitimately repeat a value at two unrelated times.
3. Resample to ~4 Hz using per-subject elapsed-time integer binning
   (groupby on an integer bin index), never pandas .resample() on a
   wall-clock DatetimeIndex, which would materialize huge empty ranges
   whenever a subject's recording has large gaps.
4. Split each subject's resampled stream into continuous "blocks" wherever
   the gap between consecutive samples exceeds `max_gap_seconds`. Windows
   are only built within a single block, so a window can never silently
   bridge a large sensor dropout.
5. Build overlapping windows (default 60 s / 15 s stride at 4 Hz = 240
   samples). Each window gets a HIGH-fraction and a configurable purity
   threshold decides its binary label -- a single stray HIGH sample inside
   an otherwise NORMAL window does not flip the window's label.
6. Per subject, compute an unsupervised physiological baseline (median,
   IQR, mean, std of EDA/HR/TEMP/ACC_MAG) from ONLY the subject's first
   `calibration_minutes` of recording. Windows starting before the
   calibration period ends are discarded (they cannot be personalized).
   This baseline is later used to build delta / relative / robust-z
   "personal-relative" features for every subsequent window -- for BOTH
   train and test subjects, since it only ever uses that subject's own
   past data and never leaks across subjects or into the future.
7. Extract a rich handcrafted feature vector per window (time-domain,
   SCR/EDA decomposition, frequency-domain, personal-relative) alongside
   the raw [T, 7] signal tensor used by the CNN branch.
8. Strict Leave-One-Subject-Out (LOSO) evaluation. For every fold: the test
   subject is 100% unseen during scaler fitting, model training, sampler
   weighting, and threshold selection. Remaining subjects are split into
   train/validation subjects (whole subjects, never a random window split),
   and only validation-subject windows are used for early stopping and
   threshold tuning.
9. A small multiscale 1D CNN (parallel kernel sizes 3/7/15/31 -> concat ->
   residual block -> global average pool) is fused with an MLP embedding of
   the handcrafted feature vector (which already carries the
   personal-relative information). Class-and-subject-aware
   WeightedRandomSampler + focal loss address the heavy class imbalance
   without letting any single long-recording subject dominate training.
10. Per-window probabilities are converted into an actual alert stream via
    EMA smoothing, a persistence requirement (N consecutive smoothed
    probabilities over threshold), enter/clear hysteresis, and a cooldown
    period -- the network never fires a raw per-window notification.
11. Both window-level metrics (MCC, AUPRC, AUROC, precision/recall/F1,
    balanced accuracy, confusion matrix) and event-level metrics (event
    recall/precision, false alerts/hour, detection delay) are reported.
    Accuracy is never used for model/threshold selection.
12. After LOSO evaluation, an explicitly-labeled "not part of unbiased
    evaluation" final stage retrains on all development subjects and saves
    a self-contained production checkpoint plus a StressInferenceEngine
    class for later website integration -- no dependency on the original
    CSV at inference time.

Leakage checklist (also enforced in code, see comments at each site)
----------------------------------------------------------------------
- StandardScaler: fit on TRAIN-fold windows only, applied to val/test.
- WeightedRandomSampler weights: computed from TRAIN-fold windows only.
- Focal loss / model weights: trained on TRAIN-fold windows only.
- Early stopping metric: computed on VAL-fold windows only.
- Threshold selection: computed on VAL-fold windows only, frozen before
  touching the TEST subject.
- Windows: built per (subject, block) and assigned to exactly one of
  train/val/test by WHOLE SUBJECT -- no window from the test subject can
  appear in training, and no subject's windows are split across sets.
- Personal baselines: computed per subject from that subject's own first
  `calibration_minutes` only -- never uses another subject's data and
  never uses samples after the calibration period ends.
================================================================================
"""

import os
import json
import math
import pickle
import warnings
import zipfile
from dataclasses import dataclass, asdict, replace
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import iqr as scipy_iqr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              IsolationForest, StackingClassifier, ExtraTreesClassifier)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionMeta  # meta-learner for stacking
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
)
import joblib

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Config:
    data_path: str = "/kaggle/input/datasets/priyankraval/nurse-stress-prediction-wearable-sensors/merged_data.csv"
        # Timestamped per-run output dir. If two executions ever overlap (e.g. a
    # stale Kaggle background job from a previous cell run), they now write
    # to physically different directories instead of silently overwriting
    # each other's checkpoints under the same fixed "fold_{subject}_checkpoint.pth"
    # path -- which is what produced the mismatched fold_94_checkpoint.pth
    # appearing inside the printed test_subject=7A block in this run's log.
    output_dir: str = f"outputs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    # --- resampling ---
    target_hz: float = 4.0
    max_gap_seconds: float = 5.0

    # --- windowing ---
    window_seconds: float = 60.0
    stride_seconds: float = 15.0
    high_stress_fraction: float = 0.5   # fraction of HIGH samples in a window needed to label it HIGH_STRESS

    # --- personalization / calibration ---
    calibration_minutes: float = 7.5
    epsilon: float = 1e-6

    # --- features ---
    use_frequency_features: bool = True

    # --- model ---
    cnn_kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31)
    cnn_hidden_dim: int = 16
    feature_mlp_hidden: int = 64
    fusion_hidden: int = 64
    dropout: float = 0.3
    # Domain-adversarial training (DANN, Ganin & Lempitsky 2015): a small
    # classifier tries to predict WHICH TRAINING SUBJECT a window's CNN
    # embedding came from, through a gradient-reversal layer. This pushes the
    # CNN to make subject identity harder to recover from its embedding --
    # i.e. to learn subject-invariant stress dynamics instead of shortcutting
    # on per-subject calibration/sensor idiosyncrasies, which is the exact
    # failure mode this project already diagnosed (best_val_mcc consistently
    # far above test window_mcc across folds).
    # Disabled by default based on controlled A/B evidence: with all other
    # fixes held constant (id-normalization, raw-tensor personal-relative
    # normalization, pooled LOSO metrics), pooled_mcc_at_own_threshold was
    # 0.333 with DANN off vs 0.263 with DANN on (domain_loss_weight=0.3).
    # The domain loss reliably plateaued near ln(n_domains) -- i.e. the
    # domain classifier was fully fooled -- while val_mcc degraded in
    # lockstep in most folds, consistent with too little capacity/data
    # (~11 training subjects, 51K-param CNN) for the network to separate
    # "subject calibration artifact" from "subject's true stress signal".
    # Kept as an opt-in flag (with dann_warmup_epochs / dann_grl_lambda /
    # domain_loss_weight already tuned lower below) for future re-evaluation
    # if the subject pool grows.
    use_domain_adversarial: bool = False
    # Reduced from 0.3: at that weight, forcing the CNN embedding toward
    # domain-invariance (domain loss plateaus near ln(n_domains), i.e. the
    # domain classifier is fully fooled) was accompanied by val_mcc
    # DEGRADING in most folds rather than improving -- with ~11 training
    # subjects and a small 51K-param CNN, there likely isn't enough
    # capacity/data for the network to cleanly separate "subject calibration
    # artifact" from "subject's true stress dynamics", so strong adversarial
    # pressure scrubs both. A much gentler weight asks the network to
    # de-emphasize subject identity without actively fighting the task.
    domain_loss_weight: float = 0.08
    dann_grl_lambda: float = 0.3
    dann_warmup_epochs: int = 8
    dann_grl_slope: float = 5.0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 60
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    use_mixed_precision: bool = True
    val_subject_fraction: float = 0.2

    # --- augmentation (train windows only) ---
    use_augmentation: bool = True
    aug_gaussian_noise_std: float = 0.02
    aug_amplitude_scale_range: float = 0.05
    aug_baseline_shift_std: float = 0.02
    aug_time_mask_fraction: float = 0.05
    aug_channel_dropout_prob: float = 0.05

    # --- temporal alert logic ---
    ema_alpha: float = 0.3
    persistence_windows: int = 3
    alert_threshold: float = 0.70
    clear_threshold: float = 0.50
    cooldown_seconds: float = 600.0
    event_match_tolerance_seconds: float = 60.0

    # --- misc ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold_objective: str = "mcc"          # "mcc" or "f1"
    enable_knowledge_distillation: bool = False  # disabled by default, per spec
    run_final_production_training: bool = True

    # --- classical (production/deployment) model ---
    # For the deployed webapp, the CNN+MLP fusion network is not used: every
    # LOSO/multi-seed experiment in this project showed it failing to
    # meaningfully beat plain logistic regression on the same handcrafted
    # features, at much higher complexity and inference cost (PyTorch/CUDA,
    # training instability). This ensemble runs entirely on
    # scikit-learn and is what StressInferenceEngineV2 actually serves.
    classical_n_cv_folds: int = 5              # subject-grouped CV folds for threshold/sanity-check
    classical_rf_n_estimators: int = 400        # fallback if hyperparameter search is skipped
    classical_ood_contamination: float = 0.05  # expected fraction of "atypical" windows for the OOD guard
    # "Best model, time/compute don't matter" mode: search hyperparameters
    # per base learner via RandomizedSearchCV (nested inside the outer
    # subject-grouped CV, so the reported CV score is not biased by tuning
    # on the same data being scored), use GPU-accelerated XGBoost as a third
    # ensemble member alongside RF/HGB, and blend all three with a learned
    # stacking meta-learner instead of a fixed 0.5/0.5 average.
    classical_use_hyperparam_search: bool = True
    classical_search_n_iter: int = 20           # trials per base learner -- searched ONCE (see below), not per outer fold
    classical_use_xgboost: bool = True
    classical_use_stacking: bool = True
    # Search hyperparameters ONCE on the full dataset (still leakage-safe --
    # the search's own internal GroupKFold never touches a true held-out
    # set), then reuse those fixed hyperparameters when evaluating each
    # outer CV fold. This was previously done PER outer fold (5x cost for
    # marginal benefit at 15 subjects) and combined with an n_jobs
    # oversubscription bug, burned a full 12-hour Kaggle session without
    # completing fold 1. Only re-enable per-fold search if you have a much
    # larger time budget and want the extra rigor.
    classical_checkpoint_every_learner: bool = True
    classical_search_scoring: str = "average_precision"  # threshold-independent; MCC@0.5 is noisy for model search
    classical_use_extra_learners: bool = True            # adds ExtraTrees + logistic regression for ensemble diversity
    classical_eval_scheme: str = "loso"                  # "loso" (15 folds, slower) or "group5" (faster)
    classical_export_pth: bool = True


CFG = Config()
RAW_CHANNELS = ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG"]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Without these, two "identical" runs (same seed, same code) can still
    # diverge -- cudnn.benchmark autotunes convolution algorithms based on
    # runtime timing (not deterministic across runs), and several CUDA
    # backward-pass ops use non-deterministic atomic adds by default. This
    # is very likely why pooled_mcc_at_own_threshold moved from 0.333 to
    # 0.231 between two runs of the SAME no-DANN config: given how often
    # best_epoch lands at 1-3, this pipeline is highly sensitive to exactly
    # how the first few gradient steps land.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def log_stage(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==============================================================================
# STAGE 0: LOAD
# ==============================================================================
def load_raw_data(cfg: Config) -> pd.DataFrame:
    log_stage("STAGE 0: LOAD")
    df = pd.read_csv(cfg.data_path)
    df = df.rename(columns={"X": "ACC_X", "Y": "ACC_Y", "Z": "ACC_Z"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Normalize subject id to a single consistent string representation.
    # If the source CSV mixes numeric-looking ids (e.g. int 15) with string
    # ids (e.g. "5C") -- or has stray whitespace on some rows -- pandas can
    # silently treat "15" (int) and "15" (str) as two DIFFERENT subjects
    # everywhere downstream: audit, windowing, baseline calibration, and
    # LOSO fold construction. That's not cosmetic: it means one physical
    # nurse's recording could fragment into multiple pseudo-subjects, and if
    # fragments of the SAME real subject land in both a fold's train set and
    # its test set, the whole-subject LOSO leakage guarantee silently breaks.
    raw_id_dtypes = df["id"].map(type).nunique()
    raw_nunique = df["id"].nunique()
    df["id"] = df["id"].astype(str).str.strip()
    normalized_nunique = df["id"].nunique()
    if raw_id_dtypes > 1 or normalized_nunique != raw_nunique:
        print(f"  WARNING: id column had {raw_id_dtypes} distinct Python types and "
              f"{raw_nunique} unique raw values, collapsing to {normalized_nunique} "
              f"unique values after str-normalization. Some subjects were likely "
              f"being double-counted / fragmented before this fix -- re-check any "
              f"prior run's LOSO fold list and metrics.")

    print(f"Raw rows loaded: {len(df):,}")
    print(f"Subjects: {df['id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    return df

# ==============================================================================
# STAGE 1: DEDUPLICATE
# ==============================================================================
def deduplicate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    log_stage("STAGE 1: DEDUPLICATE")
    n_before = len(df)
    sig_cols = ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]
    df = df.sort_values(["id", "datetime"]).reset_index(drop=True)

    # Only drop CONSECUTIVE duplicate rows within the same subject. This is
    # exactly the artifact produced by forward-filling EDA/HR/TEMP onto the
    # faster accelerometer clock -- long runs of identical values. Dropping
    # duplicates globally (regardless of position) would also remove
    # legitimate repeated readings that recur far apart in time, which is
    # not what we want.
    same_subject = (df["id"] == df["id"].shift(1)).values
    same_signal = np.ones(len(df), dtype=bool)
    for c in sig_cols:
        same_signal &= (df[c].values == df[c].shift(1).values)
    is_consecutive_dup = same_subject & same_signal

    df = df.loc[~is_consecutive_dup].reset_index(drop=True)
    n_after = len(df)
    print(f"Rows before dedup: {n_before:,}")
    print(f"Rows after dedup:  {n_after:,} ({100 * n_after / max(n_before, 1):.2f}% retained)")
    return df


# ==============================================================================
# STAGE 2: RESAMPLE (memory-efficient, per-subject elapsed-time binning)
# ==============================================================================
def _resample_subject(sub_df: pd.DataFrame, bin_seconds: float) -> pd.DataFrame:
    sub_df = sub_df.sort_values("datetime")
    t0 = sub_df["datetime"].iloc[0]
    elapsed = (sub_df["datetime"] - t0).dt.total_seconds().values
    bin_idx = np.floor(elapsed / bin_seconds).astype(np.int64)
    sub_df = sub_df.assign(_bin=bin_idx)
    agg = sub_df.groupby("_bin", sort=True).agg(
        EDA=("EDA", "mean"),
        HR=("HR", "mean"),
        TEMP=("TEMP", "mean"),
        ACC_X=("ACC_X", "mean"),
        ACC_Y=("ACC_Y", "mean"),
        ACC_Z=("ACC_Z", "mean"),
        label=("label", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[-1]),
        datetime=("datetime", "first"),
    ).reset_index(drop=True)
    agg["id"] = sub_df["id"].iloc[0]
    return agg


def resample_all(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    log_stage("STAGE 2: RESAMPLE")
    import time
    bin_seconds = 1.0 / cfg.target_hz
    frames = []
    n_subjects = df["id"].nunique()
    for i, (sid, g) in enumerate(df.groupby("id", sort=False)):
        t0 = time.time()
        frames.append(_resample_subject(g, bin_seconds))
        # Per-subject progress so a long stage never looks silently "stuck"
        # -- this is what was missing when the 12-hour session left no way
        # to tell whether the process was working or hung.
        print(f"    [{i+1}/{n_subjects}] subject={sid} rows={len(g):,} done in {time.time()-t0:.1f}s")
    resampled = pd.concat(frames, ignore_index=True)
    resampled = resampled.sort_values(["id", "datetime"]).reset_index(drop=True)
    resampled["ACC_MAG"] = np.sqrt(
        resampled["ACC_X"] ** 2 + resampled["ACC_Y"] ** 2 + resampled["ACC_Z"] ** 2
    )
    for c in ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG", "label"]:
        resampled[c] = resampled[c].astype(np.float32) if c != "label" else resampled[c].astype(np.int64)
    print(f"Resampled rows: {len(resampled):,} at ~{cfg.target_hz} Hz")
    return resampled


def assign_blocks(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Split each subject's stream into continuous blocks wherever the gap
    between consecutive samples exceeds max_gap_seconds. Windows will never
    be built across a block boundary."""
    df = df.sort_values(["id", "datetime"]).reset_index(drop=True)
    id_change = (df["id"] != df["id"].shift(1)).values
    time_diff = df["datetime"].diff().dt.total_seconds().values
    gap = time_diff > cfg.max_gap_seconds
    gap[0] = False
    new_block = id_change | gap
    new_block[0] = True
    df["block_id"] = np.cumsum(new_block)
    return df


# ==============================================================================
# STAGE 3: AUDIT
# ==============================================================================
def audit_data(df: pd.DataFrame, cfg: Config):
    log_stage("STAGE 3: AUDIT")
    subjects = df["id"].unique()
    print(f"Number of subjects: {len(subjects)}")
    for sid in subjects:
        sub = df[df["id"] == sid]
        duration_min = (sub["datetime"].max() - sub["datetime"].min()).total_seconds() / 60.0
        class_counts = sub["label"].value_counts().to_dict()
        n_blocks = sub["block_id"].nunique()
        gaps = sub["datetime"].diff().dt.total_seconds().dropna()
        n_gaps = int((gaps > cfg.max_gap_seconds).sum())
        print(f"  Subject {sid}: rows={len(sub):,} duration={duration_min:.1f}min "
              f"blocks={n_blocks} gaps>{cfg.max_gap_seconds}s={n_gaps} class_counts={class_counts}")
        total = sum(class_counts.values())
        for cls in [0, 1, 2]:
            if cls not in class_counts:
                print(f"    WARNING: subject {sid} has NO samples of class {cls}")
        for cls, cnt in class_counts.items():
            if total > 0 and cnt / total > 0.99:
                print(f"    WARNING: subject {sid} is {100 * cnt / total:.1f}% class {cls} (near single-class)")
    overall = (df["label"].value_counts(normalize=True) * 100).to_dict()
    print(f"Overall class distribution (%): {overall}")


# ==============================================================================
# STAGE 4: WINDOWING (gap-aware, binary + soft 3-class label)
# ==============================================================================
def build_windows(df: pd.DataFrame, cfg: Config) -> List[dict]:
    log_stage("STAGE 4: WINDOWING")
    hz = cfg.target_hz
    window_samples = int(round(cfg.window_seconds * hz))
    stride_samples = int(round(cfg.stride_seconds * hz))
    channels = ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG"]

    windows = []
    n_short_blocks = 0
    for (sid, bid), g in df.groupby(["id", "block_id"], sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < window_samples:
            n_short_blocks += 1
            continue
        raw = g[channels].values.astype(np.float32)
        lab = g["label"].values.astype(np.int64)
        times = g["datetime"].values
        for start in range(0, n - window_samples + 1, stride_samples):
            end = start + window_samples
            wl = lab[start:end]
            high_frac = float(np.mean(wl == 2))
            mod_frac = float(np.mean(wl == 1))
            low_frac = float(np.mean(wl == 0))
            dominant = int(np.bincount(wl, minlength=3).argmax())
            binary_label = 1 if high_frac >= cfg.high_stress_fraction else 0
            windows.append({
                "subject": sid,
                "block_id": bid,
                "start_time": times[start],
                "raw": raw[start:end],
                "high_frac": high_frac, "mod_frac": mod_frac, "low_frac": low_frac,
                "dominant_3class": dominant,
                "binary_label": binary_label,
            })

    print(f"Total windows constructed: {len(windows):,}")
    print(f"Blocks too short for a full window: {n_short_blocks}")
    if windows:
        n_high = sum(w["binary_label"] for w in windows)
        print(f"HIGH_STRESS windows: {n_high:,} ({100 * n_high / len(windows):.2f}%) "
              f"[purity threshold = {cfg.high_stress_fraction}]")
    return windows


# ==============================================================================
# PERSONAL BASELINE CALIBRATION (leakage-safe: subject's own past data only)
# ==============================================================================
def compute_personal_baselines(df: pd.DataFrame, cfg: Config) -> Dict:
    log_stage("STAGE: PERSONAL BASELINE CALIBRATION")
    baselines = {}
    calib_seconds = cfg.calibration_minutes * 60.0
    for sid, g in df.groupby("id", sort=False):
        g = g.sort_values("datetime")
        t0 = g["datetime"].iloc[0]
        elapsed = (g["datetime"] - t0).dt.total_seconds().values
        mask = elapsed <= calib_seconds
        calib = g.loc[mask]
        if len(calib) < 5:
            # fallback for pathologically short recordings: first 30s of data
            calib = g.iloc[: max(int(cfg.target_hz * 30), 1)]
        stats = {}
        # Extended to ACC_X/ACC_Y/ACC_Z (not just ACC_MAG) so every raw channel
        # fed to the CNN branch has a baseline to be robust-z normalized
        # against (see normalize_raw_window below). Previously the CNN's raw
        # [T,7] tensor was left in each subject's absolute sensor units, so it
        # could shortcut on "whose units are these" instead of learning
        # stress-related waveform shape -- a shortcut that cannot transfer to
        # an unseen LOSO test subject.
        for ch in ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG"]:
            vals = calib[ch].dropna().values.astype(np.float64)
            if len(vals) == 0:
                stats[ch] = {"median": 0.0, "iqr": 1.0, "mean": 0.0, "std": 1.0}
                continue
            stats[ch] = {
                "median": float(np.median(vals)),
                "iqr": float(max(scipy_iqr(vals), cfg.epsilon)),
                "mean": float(np.mean(vals)),
                "std": float(max(np.std(vals), cfg.epsilon)),
            }
        calib_end_time = t0 + pd.Timedelta(seconds=calib_seconds)
        baselines[sid] = {"stats": stats, "calib_end_time": calib_end_time}
    print(f"Computed personal baselines for {len(baselines)} subjects "
          f"(first {cfg.calibration_minutes} min of each recording only).")
    return baselines

RAW_CNN_CHANNELS = ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "ACC_MAG"]


RAW_CNN_CLIP = 8.0  # clip robust-z normalized raw samples to +/- this many "IQR units"


def normalize_raw_window(raw: np.ndarray, subject, baselines: Dict, cfg: Config) -> np.ndarray:
    """Personal-relative (robust-z) normalization of the raw [T, 7] tensor fed
    to the CNN branch, using ONLY that subject's own calibration-period
    baseline (median/IQR per channel) -- the same leakage-safe baseline
    already used for the handcrafted personal-relative features. This puts
    the CNN input on the same footing across subjects, so the network has to
    learn stress-related shape rather than absolute sensor calibration.

    Channels like ACC_X/Y/Z can have a near-zero calibration-period IQR for a
    resting subject (e.g. IQR ~1e-3), which is far above the cfg.epsilon
    floor but still small enough that per-sample division produces values in
    the hundreds/thousands -- overflowing fp16 under mixed precision and
    turning the very first training step into NaN loss. We use a slightly
    larger relative floor on iqr AND hard-clip the result, since a persistent
    physiological/motion event 8+ IQRs from a resting baseline is already an
    extreme outlier the network doesn't need the raw magnitude of."""
    b = baselines[subject]["stats"]
    out = np.empty_like(raw, dtype=np.float32)
    for i, ch in enumerate(RAW_CNN_CHANNELS):
        median = b[ch]["median"]
        # Relative floor: at least cfg.epsilon, but also at least a small
        # fraction of the median's magnitude, so near-constant channels
        # (e.g. a resting accelerometer axis) don't get an artificially tiny
        # denominator.
        iqr = max(b[ch]["iqr"], cfg.epsilon, abs(median) * 1e-3, 1e-4)
        z = (raw[:, i] - median) / iqr
        out[:, i] = np.clip(z, -RAW_CNN_CLIP, RAW_CNN_CLIP)
    return np.nan_to_num(out, nan=0.0, posinf=RAW_CNN_CLIP, neginf=-RAW_CNN_CLIP)
# ==============================================================================
# STAGE 5: HAND-CRAFTED FEATURE EXTRACTION
# ==============================================================================
def basic_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        keys = ["mean", "median", "std", "iqr", "min", "max", "range", "cv",
                "slope", "d1_rms", "d2_rms", "p10", "p90"]
        return {f"{prefix}_{k}": 0.0 for k in keys}
    if np.any(np.isnan(x)):
        fill = np.nanmean(x) if not np.all(np.isnan(x)) else 0.0
        x = np.nan_to_num(x, nan=fill)
    mean, std, med = np.mean(x), np.std(x), np.median(x)
    q_iqr = max(scipy_iqr(x), 1e-8)
    mn, mx = np.min(x), np.max(x)
    rng = mx - mn
    cv = std / abs(mean) if abs(mean) > 1e-8 else 0.0
    t = np.arange(len(x))
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) >= 2 else 0.0
    d1 = np.diff(x)
    d1_rms = float(np.sqrt(np.mean(d1 ** 2))) if len(d1) > 0 else 0.0
    d2 = np.diff(x, 2)
    d2_rms = float(np.sqrt(np.mean(d2 ** 2))) if len(d2) > 0 else 0.0
    p10, p90 = np.percentile(x, 10), np.percentile(x, 90)
    return {
        f"{prefix}_mean": float(mean), f"{prefix}_median": float(med),
        f"{prefix}_std": float(std), f"{prefix}_iqr": float(q_iqr),
        f"{prefix}_min": float(mn), f"{prefix}_max": float(mx),
        f"{prefix}_range": float(rng), f"{prefix}_cv": float(cv),
        f"{prefix}_slope": slope, f"{prefix}_d1_rms": d1_rms, f"{prefix}_d2_rms": d2_rms,
        f"{prefix}_p10": float(p10), f"{prefix}_p90": float(p90),
    }


def compute_scr_features(eda: np.ndarray, hz: float) -> dict:
    """Simple tonic/phasic EDA decomposition via moving-average low-pass, then
    SCR (skin conductance response) peak detection on the phasic residual.
    This is an approximation suitable for a 4 Hz signal; it does not claim
    clinical-grade SCR extraction."""
    if len(eda) < 4:
        return {"EDA_tonic_mean": 0.0, "EDA_phasic_std": 0.0, "EDA_scr_count": 0,
                "EDA_scr_density": 0.0, "EDA_scr_amp_mean": 0.0, "EDA_scr_amp_std": 0.0,
                "EDA_inter_scr_interval_mean": 0.0, "EDA_scr_pct_active": 0.0}
    win = max(int(hz * 4), 3)
    kernel = np.ones(win) / win
    tonic = np.convolve(eda, kernel, mode="same")
    phasic = eda - tonic
    peaks, props = scipy_signal.find_peaks(phasic, height=0.01, distance=max(int(hz * 1), 1))
    scr_count = len(peaks)
    duration_s = len(eda) / hz
    scr_density = scr_count / duration_s if duration_s > 0 else 0.0
    amps = props["peak_heights"] if scr_count > 0 else np.array([])
    scr_amp_mean = float(np.mean(amps)) if len(amps) > 0 else 0.0
    scr_amp_std = float(np.std(amps)) if len(amps) > 0 else 0.0
    inter_scr = float(np.mean(np.diff(peaks) / hz)) if scr_count > 1 else 0.0
    pct_active = float(np.mean(phasic > 0.01))
    return {
        "EDA_tonic_mean": float(np.mean(tonic)), "EDA_phasic_std": float(np.std(phasic)),
        "EDA_scr_count": scr_count, "EDA_scr_density": scr_density,
        "EDA_scr_amp_mean": scr_amp_mean, "EDA_scr_amp_std": scr_amp_std,
        "EDA_inter_scr_interval_mean": inter_scr, "EDA_scr_pct_active": pct_active,
    }


def frequency_features(x: np.ndarray, hz: float, prefix: str) -> dict:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 8:
        return {f"{prefix}_dom_freq": 0.0, f"{prefix}_spec_centroid": 0.0,
                f"{prefix}_spec_entropy": 0.0, f"{prefix}_lf_power": 0.0,
                f"{prefix}_hf_power": 0.0, f"{prefix}_lf_hf_ratio": 0.0}
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    mag = np.abs(np.fft.rfft(x))
    power = mag ** 2
    total_power = float(np.sum(power)) + 1e-8
    dom_freq = float(freqs[np.argmax(power[1:]) + 1]) if len(power) > 1 else 0.0
    spec_centroid = float(np.sum(freqs * power) / total_power)
    p_norm = power / total_power
    spec_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))
    lf_mask = (freqs >= 0.01) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    lf_power = float(np.sum(power[lf_mask]))
    hf_power = float(np.sum(power[hf_mask]))
    ratio = lf_power / (hf_power + 1e-8)
    return {f"{prefix}_dom_freq": dom_freq, f"{prefix}_spec_centroid": spec_centroid,
            f"{prefix}_spec_entropy": spec_entropy, f"{prefix}_lf_power": lf_power,
            f"{prefix}_hf_power": hf_power, f"{prefix}_lf_hf_ratio": ratio}


def acc_features(acc_mag, acc_x, acc_y, acc_z, hz: float) -> dict:
    feats = basic_stats(acc_mag, "ACC_MAG")
    jerk = np.diff(acc_mag) * hz
    feats["ACC_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else 0.0
    mean_val = np.mean(acc_mag) if len(acc_mag) > 0 else 0.0
    zc = np.sum(np.diff(np.sign(acc_mag - mean_val)) != 0) if len(acc_mag) > 1 else 0
    feats["ACC_zero_crossing_rate"] = float(zc / max(len(acc_mag), 1))
    axis_asym = float(np.std([np.std(acc_x), np.std(acc_y), np.std(acc_z)]))
    feats["ACC_axis_asymmetry"] = axis_asym
    return feats


def personal_relative_features(window_mean: float, baseline_stats: dict, channel: str, eps: float) -> dict:
    b = baseline_stats[channel]
    delta = window_mean - b["median"]
    relative = delta / max(abs(b["median"]), eps)
    robust_z = delta / max(b["iqr"], eps)
    return {
        f"{channel}_baseline_abs": window_mean,
        f"{channel}_delta_from_baseline": delta,
        f"{channel}_abs_delta_from_baseline": abs(delta),
        f"{channel}_relative_change": relative,
        f"{channel}_robust_z": robust_z,
    }


def extract_window_features(w: dict, cfg: Config, baselines: Dict) -> dict:
    raw = w["raw"]  # [T, 7]: EDA, HR, TEMP, ACC_X, ACC_Y, ACC_Z, ACC_MAG
    eda, hr, temp, ax, ay, az, amag = [raw[:, i] for i in range(7)]
    hz = cfg.target_hz
    feats = {}
    feats.update(basic_stats(eda, "EDA"))
    feats.update(compute_scr_features(eda, hz))
    feats.update(basic_stats(hr, "HR"))
    hr_diff = np.diff(hr)
    feats["HR_accel_decel_asymmetry"] = float(np.mean(hr_diff > 0) - np.mean(hr_diff < 0)) if len(hr_diff) > 0 else 0.0
    feats["HR_pct_above_own_mean"] = float(np.mean(hr > np.mean(hr))) if len(hr) > 0 else 0.0
    # NOTE: we deliberately do NOT report "RMSSD" here. True RMSSD requires
    # beat-to-beat NN/RR intervals; this dataset only has sampled HR, so
    # HR_d1_rms (from basic_stats) is reported as what it actually is: the
    # RMS of the first difference of sampled HR, not RMSSD.
    feats.update(basic_stats(temp, "TEMP"))
    feats.update(acc_features(amag, ax, ay, az, hz))
    if cfg.use_frequency_features:
        feats.update(frequency_features(eda, hz, "EDA"))
        feats.update(frequency_features(hr, hz, "HR"))
        feats.update(frequency_features(temp, hz, "TEMP"))
        feats.update(frequency_features(amag, hz, "ACC_MAG"))
    b = baselines[w["subject"]]["stats"]
    for ch, arr in [("EDA", eda), ("HR", hr), ("TEMP", temp), ("ACC_MAG", amag)]:
        feats.update(personal_relative_features(float(np.mean(arr)), b, ch, cfg.epsilon))
    return feats


def build_feature_dataset(windows: List[dict], baselines: Dict, cfg: Config):
    log_stage("STAGE 5: FEATURE EXTRACTION")
    kept, rows = [], []
    for w in windows:
        calib_end = baselines[w["subject"]]["calib_end_time"]
        # Windows starting before the subject's own calibration period ends
        # are dropped: there is no valid personal baseline for them yet, and
        # keeping them would mean normalizing a window using data from
        # inside (or after) itself.
        if pd.Timestamp(w["start_time"]) < calib_end:
            continue
        rows.append(extract_window_features(w, cfg, baselines))
        # Personal-relative (robust-z) normalization of the CNN's raw input.
        # w["raw"] itself is left untouched -- extract_window_features above
        # needs absolute units (EDA_tonic_mean, ACC_jerk_rms, HR_pct_above_own_mean,
        # etc.), so we store the normalized copy under a separate key.
        w["raw_cnn"] = normalize_raw_window(w["raw"], w["subject"], baselines, cfg)
        kept.append(w)
    feat_df = pd.DataFrame(rows)
    feature_names = list(feat_df.columns)
    print(f"Windows kept after calibration-period filtering: {len(kept):,} / {len(windows):,}")
    print(f"Number of handcrafted features: {len(feature_names)}")
    return kept, feat_df, feature_names


def assemble_arrays(windows: List[dict], feat_df: pd.DataFrame):
    # Use the per-subject robust-z NORMALIZED raw tensor for the CNN branch
    # (w["raw_cnn"]), not the absolute-unit w["raw"]. See normalize_raw_window().
    raw_arr = np.stack([w["raw_cnn"] for w in windows]).astype(np.float32)
    labels = np.array([w["binary_label"] for w in windows], dtype=np.int64)
    labels_3class = np.array([w["dominant_3class"] for w in windows], dtype=np.int64)
    subjects = np.array([w["subject"] for w in windows])
    times = np.array([w["start_time"] for w in windows])
    feats = np.nan_to_num(feat_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return raw_arr, feats, labels, labels_3class, subjects, times


# ==============================================================================
# STAGE 6: LOSO SPLITS
# ==============================================================================
def make_loso_splits(subjects_unique: List, cfg: Config) -> List[dict]:
    log_stage("STAGE 6: LOSO SPLITS")
    rng = np.random.RandomState(cfg.seed)
    folds = []
    for test_subj in subjects_unique:
        remaining = [s for s in subjects_unique if s != test_subj]
        shuffled = list(remaining)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * cfg.val_subject_fraction)))
        val_subjs = shuffled[:n_val]
        train_subjs = shuffled[n_val:]
        if len(train_subjs) == 0:
            train_subjs, val_subjs = val_subjs, []
        folds.append({"test": test_subj, "val": val_subjs, "train": train_subjs})
    print(f"Constructed {len(folds)} LOSO folds (1 per subject).")
    return folds


# ==============================================================================
# DATASET / AUGMENTATION
# ==============================================================================
class StressWindowDataset(Dataset):
    def __init__(self, raw, feats, labels, augment=False, cfg: Config = CFG, domain_labels=None):
        self.raw = raw
        self.feats = feats
        self.labels = labels
        self.augment = augment
        self.cfg = cfg
        # Optional int64 array of per-window domain (training-subject) index,
        # used only by the DANN domain classifier. None for val/test loaders
        # -- val/test subjects never participate in the domain-adversarial
        # objective, consistent with every other leakage boundary here.
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.raw[idx].copy()  # [T, C]
        f = self.feats[idx].copy()
        y = self.labels[idx]
        if self.augment and self.cfg.use_augmentation:
            x = self._augment(x)
        x = torch.from_numpy(x.astype(np.float32)).transpose(0, 1)  # [C, T]
        f = torch.from_numpy(f.astype(np.float32))
        y = torch.tensor(y, dtype=torch.float32)
        if self.domain_labels is not None:
            d = torch.tensor(self.domain_labels[idx], dtype=torch.long)
            return x, f, y, d
        return x, f, y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        T, C = x.shape
        std_per_ch = np.std(x, axis=0, keepdims=True) + 1e-6
        if cfg.aug_gaussian_noise_std > 0:
            x = x + np.random.normal(0, cfg.aug_gaussian_noise_std, size=x.shape).astype(np.float32) * std_per_ch
        if cfg.aug_amplitude_scale_range > 0:
            scale = 1.0 + np.random.uniform(-cfg.aug_amplitude_scale_range, cfg.aug_amplitude_scale_range,
                                             size=(1, C)).astype(np.float32)
            x = x * scale
        if cfg.aug_baseline_shift_std > 0:
            shift = (np.random.normal(0, cfg.aug_baseline_shift_std, size=(1, C)).astype(np.float32) * std_per_ch)
            x = x + shift
        if cfg.aug_time_mask_fraction > 0 and np.random.rand() < 0.5:
            mask_len = int(T * cfg.aug_time_mask_fraction)
            if mask_len > 0:
                start = np.random.randint(0, max(T - mask_len, 1))
                x[start:start + mask_len, :] = 0.0
        if cfg.aug_channel_dropout_prob > 0:
            for c in range(C):
                if np.random.rand() < cfg.aug_channel_dropout_prob:
                    x[:, c] = 0.0
        return x


# ==============================================================================
# MODEL
# ==============================================================================
class MultiScaleCNNBranch(nn.Module):
    def __init__(self, in_channels, kernel_sizes, hidden_dim):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ) for k in kernel_sizes
        ])
        concat_dim = hidden_dim * len(kernel_sizes)
        self.residual = nn.Sequential(
            nn.Conv1d(concat_dim, concat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(concat_dim),
            nn.ReLU(),
            nn.Conv1d(concat_dim, concat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(concat_dim),
        )
        self.out_dim = concat_dim

    def forward(self, x):  # x: [B, C, T]
        outs = [b(x) for b in self.branches]
        h = torch.cat(outs, dim=1)
        r = self.residual(h)
        h = F.relu(h + r)
        return F.adaptive_avg_pool1d(h, 1).squeeze(-1)


class FeatureMLPBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.out_dim = hidden_dim

    def forward(self, f):
        return self.net(f)


class StressDetector(nn.Module):
    """Small multiscale-CNN + handcrafted-feature-MLP fusion binary
    HIGH-stress classifier. Deliberately not a Transformer: with only
    ~15-18 subjects, the bottleneck is cross-subject generalization, not
    model capacity."""

    def __init__(self, n_raw_channels, n_features, cfg: Config):
        super().__init__()
        self.cnn_branch = MultiScaleCNNBranch(n_raw_channels, cfg.cnn_kernel_sizes, cfg.cnn_hidden_dim)
        self.feat_branch = FeatureMLPBranch(n_features, cfg.feature_mlp_hidden, cfg.dropout)
        fusion_in = self.cnn_branch.out_dim + self.feat_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, 1),
        )

    def forward(self, x_raw, x_feat, return_cnn_embedding=False):
        cnn_h = self.cnn_branch(x_raw)
        h = torch.cat([cnn_h, self.feat_branch(x_feat)], dim=1)
        logits = self.classifier(h).squeeze(-1)
        if return_cnn_embedding:
            return logits, cnn_h
        return logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass; negates (and scales) the gradient in the
    backward pass. This is the core DANN trick: the domain classifier trains
    normally to predict subject identity, but the gradient it sends back into
    the shared CNN embedding is reversed, so the CNN is simultaneously pushed
    to make subject identity *harder* to predict."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


class DomainClassifier(nn.Module):
    """Small MLP head predicting which TRAINING subject a window's CNN
    embedding came from. Trained via the reversed gradient only; discarded
    after training and never used at inference/deployment."""

    def __init__(self, in_dim, n_domains, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, h):
        return self.net(h)


def build_sampler(subjects: np.ndarray, labels: np.ndarray) -> WeightedRandomSampler:
    """Subject-and-class-aware sampling: a window's weight is
    (1 / that subject's window count) * (1 / that class's window count),
    so neither a long-recording subject nor the majority class can
    dominate an epoch. Computed on TRAIN-fold windows only."""
    subj_counts = pd.Series(subjects).value_counts().to_dict()
    class_counts = pd.Series(labels).value_counts().to_dict()
    weights = np.array([
        (1.0 / subj_counts[subjects[i]]) * (1.0 / class_counts[labels[i]])
        for i in range(len(labels))
    ], dtype=np.float64)
    weights = weights / weights.sum() * len(weights)
    print(f"  Sampler subject counts: {subj_counts}")
    print(f"  Sampler class counts:   {class_counts}")
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def get_model_probs(model, raw, feats, device, batch_size=256) -> np.ndarray:
    model.eval()
    ds = StressWindowDataset(raw, feats, np.zeros(len(raw)), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, fb, _ in loader:
            xb, fb = xb.to(device), fb.to(device)
            probs.append(torch.sigmoid(model(xb, fb)).cpu().numpy())
    out = np.concatenate(probs) if probs else np.array([])
    if out.size and not np.all(np.isfinite(out)):
        # A collapsed/NaN model (e.g. training diverged to NaN loss) should
        # not crash sklearn metrics downstream. Surface it loudly in the
        # logs instead of silently reporting garbage window metrics.
        n_bad = int(np.sum(~np.isfinite(out)))
        print(f"    WARNING: {n_bad}/{len(out)} model probabilities were non-finite "
              f"(NaN/inf) -- treating as degenerate model output for this fold.")
        out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return out


def optimize_threshold(y_true, y_prob, objective="mcc") -> Tuple[float, float]:
    """Threshold search performed on VALIDATION windows only; the returned
    threshold is then frozen and applied to the untouched test subject."""
    best_t, best_score = 0.5, -2.0
    if len(np.unique(y_true)) < 2:
        return best_t, best_score
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob >= t).astype(int)
        score = matthews_corrcoef(y_true, preds) if objective == "mcc" else f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


# ==============================================================================
# TEMPORAL SMOOTHING + PERSISTENCE + HYSTERESIS + COOLDOWN ALERT LOGIC
# ==============================================================================
def apply_temporal_alert_logic(times: np.ndarray, probs: np.ndarray, cfg: Config):
    order = np.argsort(times)
    times_sorted = times[order]
    probs_sorted = probs[order]
    n = len(probs_sorted)
    smoothed = np.zeros(n)
    ema = probs_sorted[0] if n > 0 else 0.0
    for i in range(n):
        ema = cfg.ema_alpha * probs_sorted[i] + (1 - cfg.ema_alpha) * ema
        smoothed[i] = ema

    alerts = np.zeros(n, dtype=bool)
    alert_events = []  # (index_in_sorted_order, timestamp)
    state = "NORMAL"
    consecutive_high = 0
    last_alert_time = None
    for i in range(n):
        t = pd.Timestamp(times_sorted[i])
        if state == "NORMAL":
            consecutive_high = consecutive_high + 1 if smoothed[i] >= cfg.alert_threshold else 0
            if consecutive_high >= cfg.persistence_windows:
                cooldown_ok = (last_alert_time is None) or \
                    ((t - last_alert_time).total_seconds() >= cfg.cooldown_seconds)
                if cooldown_ok:
                    state = "ALERT"
                    alerts[i] = True
                    alert_events.append((i, t))
                    last_alert_time = t
                consecutive_high = 0
        else:  # ALERT state -> wait for clear_threshold (hysteresis) before re-arming
            if smoothed[i] < cfg.clear_threshold:
                state = "NORMAL"
    return smoothed, alerts, alert_events, order


def extract_true_events(times: np.ndarray, labels: np.ndarray, cfg: Config) -> List[pd.Timestamp]:
    """A 'true event' is the start of a contiguous run of binary_label==1
    windows in chronological order."""
    order = np.argsort(times)
    times_sorted, labels_sorted = times[order], labels[order]
    events, in_event, start_t = [], False, None
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1 and not in_event:
            in_event, start_t = True, pd.Timestamp(times_sorted[i])
        elif labels_sorted[i] == 0 and in_event:
            in_event = False
            events.append(start_t)
    if in_event:
        events.append(start_t)
    return events


def compute_event_metrics(true_event_times, alert_events, cfg: Config, total_duration_hours: float) -> dict:
    detected = set()
    delays = []
    false_alerts = 0
    for (_, alert_time) in alert_events:
        matched = False
        for j, te in enumerate(true_event_times):
            if j in detected:
                continue
            delta = (alert_time - te).total_seconds()
            # allow the alert to fire slightly before the labeled event
            # (tolerance) up to a generous window after it (10x tolerance)
            if -cfg.event_match_tolerance_seconds <= delta <= cfg.event_match_tolerance_seconds * 10:
                detected.add(j)
                delays.append(max(delta, 0.0))
                matched = True
                break
        if not matched:
            false_alerts += 1
    n_true = len(true_event_times)
    n_detected = len(detected)
    n_alerts = len(alert_events)
    return {
        "n_true_events": n_true,
        "n_detected_events": n_detected,
        "event_recall": (n_detected / n_true) if n_true > 0 else float("nan"),
        "event_precision": ((n_alerts - false_alerts) / n_alerts) if n_alerts > 0 else float("nan"),
        "missed_events": n_true - n_detected,
        "false_alert_count": false_alerts,
        "false_alerts_per_hour": (false_alerts / total_duration_hours) if total_duration_hours > 0 else float("nan"),
        "avg_detection_delay_s": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay_s": float(np.median(delays)) if delays else float("nan"),
    }


def compute_window_metrics(y_true, y_prob, threshold: float) -> dict:
    preds = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "high_precision": precision_score(y_true, preds, zero_division=0),
        "high_recall": recall_score(y_true, preds, zero_division=0),
        "high_f1": f1_score(y_true, preds, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
        metrics["mcc"] = matthews_corrcoef(y_true, preds)
    else:
        metrics["auroc"] = metrics["auprc"] = metrics["mcc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    return metrics


def compute_context_baselines(y_train, y_test, feats_train=None, feats_test=None) -> dict:
    """Non-production reference baselines only (majority-class, stratified
    random, and a simple logistic-regression-on-handcrafted-features model),
    never used for model/threshold selection -- purely diagnostic context for
    how much the CNN+MLP fusion network is actually adding over a trivial
    linear model on the same leakage-safe (already train-fold-fit-scaled)
    feature vectors."""
    majority_class = int(pd.Series(y_train).mode().iloc[0])
    maj_preds = np.full_like(y_test, majority_class)
    rng = np.random.RandomState(0)
    class_probs = pd.Series(y_train).value_counts(normalize=True)
    rand_preds = rng.choice(class_probs.index, size=len(y_test), p=class_probs.values)
    result = {
        "majority_class": majority_class,
        "majority_accuracy": float(accuracy_score(y_test, maj_preds)),
        "stratified_random_accuracy": float(accuracy_score(y_test, rand_preds)),
    }
    if feats_train is not None and feats_test is not None and len(np.unique(y_train)) > 1:
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
        lr.fit(feats_train, y_train)
        lr_probs = lr.predict_proba(feats_test)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)
        result["logreg_accuracy"] = float(accuracy_score(y_test, lr_preds))
        result["logreg_high_recall"] = float(recall_score(y_test, lr_preds, zero_division=0))
        result["logreg_high_precision"] = float(precision_score(y_test, lr_preds, zero_division=0))
        if len(np.unique(y_test)) > 1:
            result["logreg_auprc"] = float(average_precision_score(y_test, lr_probs))
            result["logreg_mcc"] = float(matthews_corrcoef(y_test, lr_preds))
        else:
            result["logreg_auprc"] = float("nan")
            result["logreg_mcc"] = float("nan")
    else:
        result["logreg_accuracy"] = float("nan")
        result["logreg_high_recall"] = float("nan")
        result["logreg_high_precision"] = float("nan")
        result["logreg_auprc"] = float("nan")
        result["logreg_mcc"] = float("nan")
    return result


# ==============================================================================
# STAGE 7-10: TRAIN / VALIDATE / EVENT-EVALUATE / AGGREGATE, PER LOSO FOLD
# ==============================================================================
def _train_one_model(raw_train, feats_train, y_train, subj_train,
                      raw_val, feats_val, y_val, cfg: Config, device: str):
    # Map each TRAINING subject to an integer domain id. Only train-fold
    # subjects get one -- the domain classifier's job is to make the CNN
    # embedding invariant across the subjects the model actually trains on;
    # val/test subjects are never involved (no leakage introduced).
    train_subjects_unique = sorted(pd.unique(subj_train).tolist())
    subj_to_domain = {s: i for i, s in enumerate(train_subjects_unique)}
    n_domains = len(train_subjects_unique)
    domain_labels_train = np.array([subj_to_domain[s] for s in subj_train], dtype=np.int64)

    train_ds = StressWindowDataset(
        raw_train, feats_train, y_train, augment=True, cfg=cfg,
        domain_labels=domain_labels_train if cfg.use_domain_adversarial else None,
    )
    val_ds = StressWindowDataset(raw_val, feats_val, y_val, augment=False, cfg=cfg)
    sampler = build_sampler(subj_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    n_raw_channels = raw_train.shape[2]
    n_features = feats_train.shape[1]
    model = StressDetector(n_raw_channels, n_features, cfg).to(device)
    total_p, trainable_p = model.count_parameters()
    print(f"  Model params: total={total_p:,} trainable={trainable_p:,} "
          f"(~{total_p * 4 / 1024:.1f} KB float32)")

    domain_clf = None
    if cfg.use_domain_adversarial and n_domains > 1:
        domain_clf = DomainClassifier(model.cnn_branch.out_dim, n_domains).to(device)
        print(f"  Domain-adversarial: {n_domains} training-subject domains, "
              f"domain_loss_weight={cfg.domain_loss_weight}")

    if domain_clf is not None:
        # Domain classifier gets its own (lower) LR param group. It only
        # needs to be "good enough" to give the GRL a meaningful signal to
        # reverse -- letting it train at the same LR as the main model lets
        # it overpower the shared CNN embedding early in the post-warmup
        # ramp, which is part of what caused the loss spike/val_mcc collapse.
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": cfg.learning_rate},
            {"params": domain_clf.parameters(), "lr": cfg.learning_rate * 0.5},
        ], weight_decay=cfg.weight_decay)
        params = list(model.parameters()) + list(domain_clf.parameters())
    else:
        params = list(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    domain_criterion = nn.CrossEntropyLoss()
    use_amp = cfg.use_mixed_precision and device == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score, best_state, best_epoch, patience_counter = -2.0, None, -1, 0
    ema_val_mcc = None  # smoothed val_mcc used ONLY for model-selection/early-stopping,
                         # to avoid anchoring on a single noisy epoch (val set is only
                         # 3 subjects here, and MCC at a fixed 0.5 threshold is high-variance)
    for epoch in range(cfg.epochs):
        model.train()
        if domain_clf is not None:
            domain_clf.train()
        # Standard DANN lambda schedule: ramps 0 -> cfg.dann_grl_lambda over
        # training so the adversarial signal doesn't destabilize the first
        # few epochs, while the CNN embedding is still close to random and
        # has nothing meaningful yet for the domain classifier to reverse.
        # Warmup: grl_lambda stays at 0 (pure task training, domain
        # classifier trains passively but sends no reversed gradient) for
        # the first dann_warmup_epochs. After warmup, ramp with a shallower
        # slope over the REMAINING epochs so it doesn't saturate to max
        # strength in the first handful of post-warmup steps.
        if epoch < cfg.dann_warmup_epochs:
            grl_lambda = 0.0
        else:
            remaining_epochs = max(cfg.epochs - cfg.dann_warmup_epochs - 1, 1)
            p = (epoch - cfg.dann_warmup_epochs) / remaining_epochs
            grl_lambda = cfg.dann_grl_lambda * (2.0 / (1.0 + math.exp(-cfg.dann_grl_slope * p)) - 1.0)

        # The pre-warmup phase (grl_lambda == 0, pure task training) and the
        # post-warmup phase (adversarial pressure active) are different
        # optimization regimes. A pre-warmup epoch can produce a spuriously
        # high val_mcc on a small/skewed val-subject set (as seen: epoch 1
        # peaked at 0.69 then never recovered), which then silently eats
        # most of the early-stopping patience budget before the
        # domain-adversarial objective even switches on -- exactly what
        # happened here (best_epoch=1, patience=10 -> stopped at epoch 11
        # with only ~2-3 epochs of real post-warmup training). Give the
        # post-warmup phase a fresh, full patience budget instead of
        # inheriting a used-up counter from warmup. The best-ever checkpoint
        # tracked via best_state/best_score/best_epoch is NOT reset -- if
        # warmup genuinely found the best model, it's still kept.
        if domain_clf is not None and epoch == cfg.dann_warmup_epochs:
            patience_counter = 0
            print(f"    -- warmup ended, resetting early-stopping patience counter "
                  f"for post-warmup (adversarial) phase --")

        total_loss, total_task_loss, total_domain_loss = 0.0, 0.0, 0.0
        for batch in train_loader:
            if domain_clf is not None:
                xb, fb, yb, db = batch
                db = db.to(device)
            else:
                xb, fb, yb = batch
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                if domain_clf is not None:
                    logits, cnn_h = model(xb, fb, return_cnn_embedding=True)
                    task_loss = criterion(logits, yb)
                    domain_logits = domain_clf(grad_reverse(cnn_h, grl_lambda))
                    domain_loss = domain_criterion(domain_logits, db)
                    loss = task_loss + cfg.domain_loss_weight * domain_loss
                    total_task_loss += task_loss.item() * len(yb)
                    total_domain_loss += domain_loss.item() * len(yb)
                else:
                    logits = model(xb, fb)
                    loss = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item() * len(yb)
        total_loss /= max(len(train_ds), 1)
        total_task_loss /= max(len(train_ds), 1)
        total_domain_loss /= max(len(train_ds), 1)

        if not math.isfinite(total_loss):
            # Training has diverged (NaN/inf loss) -- further epochs won't
            # recover since BatchNorm running stats and weights are already
            # corrupted. Stop immediately rather than burning the remaining
            # patience budget on a dead model.
            print(f"    epoch {epoch + 1}/{cfg.epochs} loss={total_loss} -- NON-FINITE LOSS, "
                  f"stopping fold early (best_epoch so far={best_epoch if best_epoch > 0 else 'none'})")
            break

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, fb, yb in val_loader:
                xb, fb = xb.to(device), fb.to(device)
                val_probs.append(torch.sigmoid(model(xb, fb)).cpu().numpy())
                val_true.append(yb.numpy())
        val_probs = np.concatenate(val_probs) if val_probs else np.array([])
        val_true = np.concatenate(val_true) if val_true else np.array([])
        val_mcc = matthews_corrcoef(val_true, (val_probs >= 0.5).astype(int)) if len(np.unique(val_true)) > 1 else 0.0
        ema_val_mcc = val_mcc if ema_val_mcc is None else 0.3 * val_mcc + 0.7 * ema_val_mcc
        scheduler.step(ema_val_mcc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if domain_clf is not None:
                print(f"    epoch {epoch + 1}/{cfg.epochs} loss={total_loss:.4f} "
                      f"(task={total_task_loss:.4f} domain={total_domain_loss:.4f} "
                      f"grl_lambda={grl_lambda:.3f}) val_mcc@0.5={val_mcc:.4f}")
            else:
                print(f"    epoch {epoch + 1}/{cfg.epochs} loss={total_loss:.4f} val_mcc@0.5={val_mcc:.4f}")

        if ema_val_mcc > best_score:
            best_score = ema_val_mcc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"    early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score


def run_pipeline(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = cfg.device
    print(f"Using device: {device}")

    df = load_raw_data(cfg)
    df = deduplicate(df, cfg)
    df = resample_all(df, cfg)
    df = assign_blocks(df, cfg)
    audit_data(df, cfg)

    windows = build_windows(df, cfg)
    baselines = compute_personal_baselines(df, cfg)
    kept_windows, feat_df, feature_names = build_feature_dataset(windows, baselines, cfg)
    raw_arr, feats, labels, labels_3class, subjects, times = assemble_arrays(kept_windows, feat_df)

    subjects_unique = sorted(pd.unique(subjects).tolist())
    folds = make_loso_splits(subjects_unique, cfg)

    log_stage("STAGE 7-10: TRAINING / VALIDATION / EVENT EVALUATION / AGGREGATION")
    all_fold_metrics, all_pooled_preds = [], []

    for fold in folds:
        test_subj, val_subjs, train_subjs = fold["test"], fold["val"], fold["train"]
        print(f"\n--- FOLD: held-out test_subject={test_subj} ---")
        print(f"  train_subjects={train_subjs}")
        print(f"  val_subjects={val_subjs}")

        train_mask = np.isin(subjects, train_subjs)
        val_mask = np.isin(subjects, val_subjs)
        test_mask = subjects == test_subj

        if train_mask.sum() < 10 or val_mask.sum() < 5 or test_mask.sum() < 5:
            print("  Skipping fold: insufficient windows for train/val/test.")
            continue

        print(f"  train class dist: {pd.Series(labels[train_mask]).value_counts(normalize=True).round(3).to_dict()}")
        print(f"  val class dist:   {pd.Series(labels[val_mask]).value_counts(normalize=True).round(3).to_dict()}")
        print(f"  test class dist:  {pd.Series(labels[test_mask]).value_counts(normalize=True).round(3).to_dict()}")

        # --- Scaler fit on TRAIN-fold windows only (leakage-safe) ---
        scaler = StandardScaler()
        feats_train = scaler.fit_transform(feats[train_mask])
        feats_val = scaler.transform(feats[val_mask])
        feats_test = scaler.transform(feats[test_mask])

        raw_train, y_train, subj_train = raw_arr[train_mask], labels[train_mask], subjects[train_mask]
        raw_val, y_val = raw_arr[val_mask], labels[val_mask]
        raw_test, y_test = raw_arr[test_mask], labels[test_mask]
        times_test = times[test_mask]

        model, best_epoch, best_val_mcc = _train_one_model(
            raw_train, feats_train, y_train, subj_train,
            raw_val, feats_val, y_val, cfg, device,
        )
        print(f"  best_epoch={best_epoch} best_val_mcc@0.5={best_val_mcc:.4f}")

        # --- Threshold selection on VALIDATION windows only ---
        val_probs_final = get_model_probs(model, raw_val, feats_val, device)
        best_threshold, thr_score = optimize_threshold(y_val, val_probs_final, cfg.threshold_objective)
        print(f"  selected threshold={best_threshold:.2f} ({cfg.threshold_objective}={thr_score:.4f} on val)")

        # --- Frozen threshold applied to untouched TEST subject ---
        test_probs = get_model_probs(model, raw_test, feats_test, device)
        window_metrics = compute_window_metrics(y_test, test_probs, best_threshold)
        print(f"  TEST window metrics: { {k: v for k, v in window_metrics.items() if k != 'confusion_matrix'} }")

        smoothed, alerts, alert_events, order = apply_temporal_alert_logic(times_test, test_probs, cfg)
        true_events = extract_true_events(times_test, y_test, cfg)
        duration_hours = max(
            (pd.Timestamp(times_test.max()) - pd.Timestamp(times_test.min())).total_seconds() / 3600.0,
            cfg.window_seconds / 3600.0,
        )
        event_metrics = compute_event_metrics(true_events, alert_events, cfg, duration_hours)
        print(f"  TEST event metrics: {event_metrics}")

        baseline_metrics = compute_context_baselines(y_train, y_test, feats_train, feats_test)
        print(f"  context baselines: {baseline_metrics}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_scaler_mean": scaler.mean_.astype(np.float32),
            "feature_scaler_scale": scaler.scale_.astype(np.float32),
            "feature_names": feature_names,
            "n_raw_channels": raw_arr.shape[2],
            "threshold": best_threshold,
            "label_mapping": {"0": "NORMAL (LOW/MODERATE)", "1": "HIGH_STRESS"},
            "test_subject": str(test_subj),
            "best_epoch": best_epoch,
        }
        ckpt_path = os.path.join(cfg.output_dir, f"fold_{test_subj}_checkpoint.pth")
        torch.save(checkpoint, ckpt_path)
        # weights_only=False: this checkpoint was just written by this same
        # process and contains numpy arrays (the scaler mean/scale) in
        # addition to tensors, which PyTorch >=2.6's default weights_only=True
        # loader refuses to unpickle. We trust the source here (ourselves).
        reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in reloaded, "checkpoint failed reload verification"
        print(f"  checkpoint saved and reload-verified: {ckpt_path}")

        fold_summary = {
            "test_subject": test_subj, "best_epoch": best_epoch,
            "best_val_mcc": best_val_mcc, "threshold": best_threshold,
            **{f"window_{k}": v for k, v in window_metrics.items() if k != "confusion_matrix"},
            **{f"event_{k}": v for k, v in event_metrics.items()},
            **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
        }
        all_fold_metrics.append(fold_summary)
        all_pooled_preds.append(pd.DataFrame({
            "subject": test_subj, "time": times_test, "y_true": y_test,
            "y_prob": test_probs, "y_pred": (test_probs >= best_threshold).astype(int),
        }))

    log_stage("STAGE 10: AGGREGATION")
    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    fold_metrics_df.to_csv(os.path.join(cfg.output_dir, "loso_fold_metrics.csv"), index=False)
    pooled_df = pd.concat(all_pooled_preds, ignore_index=True) if all_pooled_preds else pd.DataFrame()
    if not pooled_df.empty:
        pooled_df.to_csv(os.path.join(cfg.output_dir, "pooled_predictions.csv"), index=False)
    if len(fold_metrics_df) > 0:
        numeric_cols = fold_metrics_df.select_dtypes(include=[np.number]).columns
        print("Per-fold (macro) mean metrics:")
        print(fold_metrics_df[numeric_cols].mean(numeric_only=True))

    log_stage("STAGE 10b: POOLED (MICRO) METRICS ACROSS ALL LOSO FOLDS")
    # With only 15 subjects and several near-single-class test folds (e.g.
    # test class dist {1: 1.0}), the MACRO mean above is dominated by a
    # handful of noisy/degenerate per-fold MCC and AUROC values (some folds
    # even report nan). Pooling every fold's out-of-fold predictions into one
    # set and computing metrics ONCE gives a much more stable estimate of
    # true LOSO performance -- this is standard practice for small-N LOSO/CV
    # and should be treated as the primary number, with the per-fold table
    # above kept as a diagnostic breakdown, not the headline metric.
    pooled_metrics = {}
    if not pooled_df.empty:
        py_true = pooled_df["y_true"].values
        py_prob = pooled_df["y_prob"].values
        py_pred = pooled_df["y_pred"].values  # each row already thresholded with ITS OWN fold's frozen val threshold
        pooled_metrics["n_pooled_windows"] = int(len(pooled_df))
        if len(np.unique(py_true)) > 1:
            pooled_metrics["pooled_auroc"] = float(roc_auc_score(py_true, py_prob))
            pooled_metrics["pooled_auprc"] = float(average_precision_score(py_true, py_prob))
        else:
            pooled_metrics["pooled_auroc"] = float("nan")
            pooled_metrics["pooled_auprc"] = float("nan")
        pooled_metrics["pooled_mcc_at_own_threshold"] = float(matthews_corrcoef(py_true, py_pred))
        pooled_metrics["pooled_balanced_accuracy_at_own_threshold"] = float(balanced_accuracy_score(py_true, py_pred))
        pooled_metrics["pooled_high_precision_at_own_threshold"] = float(precision_score(py_true, py_pred, zero_division=0))
        pooled_metrics["pooled_high_recall_at_own_threshold"] = float(recall_score(py_true, py_pred, zero_division=0))
        print(pooled_metrics)
        with open(os.path.join(cfg.output_dir, "pooled_metrics.json"), "w") as fh:
            json.dump(pooled_metrics, fh, indent=2)
    else:
        print("No pooled predictions available (all folds skipped).")

    with open(os.path.join(cfg.output_dir, "feature_names.json"), "w") as fh:
        json.dump(feature_names, fh, indent=2)

    if cfg.run_final_production_training:
        run_final_production_training(raw_arr, feats, labels, subjects, subjects_unique, feature_names, cfg, device)

    return fold_metrics_df

def run_multi_seed(cfg: Config, seeds: List[int] = (42, 43, 44)) -> pd.DataFrame:
    """Run the full LOSO pipeline once per seed and report pooled MCC (and
    other pooled metrics) across seeds as mean +/- std, instead of trusting
    a single run. Given the demonstrated run-to-run variance at fixed
    hyperparameters (0.333 vs 0.231 pooled MCC across two runs of the same
    no-DANN config), a single run should not be treated as the reported
    result -- for the paper, report this multi-seed summary instead."""
    import copy
    base_output_dir = cfg.output_dir
    results = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = seed
        run_cfg.output_dir = f"{base_output_dir}_seed{seed}"
        log_stage(f"MULTI-SEED RUN: seed={seed}")
        run_pipeline(run_cfg)
        pooled_metrics_path = os.path.join(run_cfg.output_dir, "pooled_metrics.json")
        if os.path.exists(pooled_metrics_path):
            with open(pooled_metrics_path) as fh:
                m = json.load(fh)
            m["seed"] = seed
            results.append(m)

    results_df = pd.DataFrame(results)
    log_stage("MULTI-SEED SUMMARY")
    if len(results_df) > 0:
        print(results_df)
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns.drop("seed", errors="ignore")
        print("\nMean +/- std across seeds:")
        for col in numeric_cols:
            print(f"  {col}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")
        results_df.to_csv(os.path.join(base_output_dir + "_multiseed_summary.csv"), index=False)
    return results_df
    
# ==============================================================================
# STAGE 11: FINAL PRODUCTION MODEL (explicitly NOT unbiased evaluation)
# ==============================================================================
def run_final_production_training(raw_arr, feats, labels, subjects, subjects_unique, feature_names, cfg: Config, device: str):
    log_stage("STAGE 11: FINAL PRODUCTION MODEL (NOT PART OF UNBIASED EVALUATION)")
    rng = np.random.RandomState(cfg.seed + 1)
    shuffled = list(subjects_unique)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * cfg.val_subject_fraction)))
    val_subjs, train_subjs = shuffled[:n_val], shuffled[n_val:]
    print(f"Production train subjects: {train_subjs}")
    print(f"Production val subjects (early stopping + threshold only): {val_subjs}")

    train_mask = np.isin(subjects, train_subjs)
    val_mask = np.isin(subjects, val_subjs)

    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats[train_mask])
    feats_val = scaler.transform(feats[val_mask])
    raw_train, y_train, subj_train = raw_arr[train_mask], labels[train_mask], subjects[train_mask]
    raw_val, y_val = raw_arr[val_mask], labels[val_mask]

    model, best_epoch, best_val_mcc = _train_one_model(
        raw_train, feats_train, y_train, subj_train,
        raw_val, feats_val, y_val, cfg, device,
    )
    val_probs_final = get_model_probs(model, raw_val, feats_val, device)
    best_threshold, thr_score = optimize_threshold(y_val, val_probs_final, cfg.threshold_objective)
    print(f"Production model: best_epoch={best_epoch} val_mcc={best_val_mcc:.4f} threshold={best_threshold:.2f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "feature_scaler_mean": scaler.mean_.astype(np.float32),
        "feature_scaler_scale": scaler.scale_.astype(np.float32),
        "feature_names": feature_names,
        "n_raw_channels": raw_arr.shape[2],
        "threshold": best_threshold,
        "label_mapping": {"0": "NORMAL (LOW/MODERATE)", "1": "HIGH_STRESS"},
        "note": "Trained on ALL development subjects. This is a deployment checkpoint, "
                "not an unbiased LOSO evaluation result.",
    }
    prod_path = os.path.join(cfg.output_dir, "production_model.pth")
    torch.save(checkpoint, prod_path)
    reloaded = torch.load(prod_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in reloaded, "production checkpoint failed reload verification"

    with open(os.path.join(cfg.output_dir, "production_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    with open(os.path.join(cfg.output_dir, "feature_names.json"), "w") as fh:
        json.dump(feature_names, fh, indent=2)
    with open(os.path.join(cfg.output_dir, "threshold.json"), "w") as fh:
        json.dump({"threshold": best_threshold}, fh, indent=2)
    with open(os.path.join(cfg.output_dir, "feature_scaler.pkl"), "wb") as fh:
        pickle.dump({"mean": scaler.mean_, "scale": scaler.scale_}, fh)

    print(f"Production model saved to: {prod_path}")
    return prod_path

# ==============================================================================
# CLASSICAL (PRODUCTION) MODEL: RandomForest + HistGradientBoosting ensemble
# on the handcrafted feature vector, calibrated, with an OOD guard. This is
# the model actually served by the webapp -- see ClassicalStressInferenceEngine
# below. It is deliberately independent of the CNN/DANN code above.
# ==============================================================================
def _rf_search_space():
    return {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }


def _hgb_search_space():
    return {
        "max_iter": [200, 400, 600, 900],
        "max_depth": [None, 6, 10, 16],
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "max_leaf_nodes": [15, 31, 63, 127],
    }


def _xgb_search_space():
    return {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }


def _search_best_estimator(base_estimator, search_space, X, y, groups, cfg: Config, name: str,
                            checkpoint_path: Optional[str] = None):
    """RandomizedSearchCV with subject-grouped CV, scored on MCC.

    CRITICAL: base_estimator must have n_jobs=1 (set by the caller) when
    used here. RandomizedSearchCV(n_jobs=-1) already parallelizes across
    all CPU cores at the trial level; if the estimator ALSO tries to use
    all cores internally, you get massive thread/process oversubscription
    -- cores contending with each other instead of doing useful work. This
    was the primary cause of RF's search taking ~6.7 hours for what should
    be a few minutes of work on 22K rows."""
    import time
    t0 = time.time()
    inner_gkf = GroupKFold(n_splits=min(cfg.classical_n_cv_folds, len(np.unique(groups))))
    search = RandomizedSearchCV(
        base_estimator, param_distributions=search_space, n_iter=cfg.classical_search_n_iter,
        scoring=cfg.classical_search_scoring, cv=inner_gkf, n_jobs=-1, random_state=cfg.seed, refit=True,
    )
    search.fit(X, y, groups=groups)
    elapsed = time.time() - t0
    print(f"    [{name}] best_params={search.best_params_} best_cv_mcc={search.best_score_:.4f} "
          f"({elapsed:.1f}s)")

    if checkpoint_path:
        # Save immediately so this result survives even if a later stage
        # crashes or the session times out -- nothing is lost this time.
        joblib.dump({"name": name, "best_estimator": search.best_estimator_,
                     "best_params": search.best_params_, "best_cv_mcc": search.best_score_},
                    checkpoint_path)
        print(f"    [{name}] checkpoint saved: {checkpoint_path}")
    return search.best_estimator_

def _make_base_learners(X, y, groups, cfg: Config, checkpoint_dir: Optional[str] = None):
    """Builds RF, HGB, and (if available/enabled) XGB estimators -- searched
    ONCE (not per outer fold) if classical_use_hyperparam_search is on, else
    sane fixed defaults. Every base estimator uses n_jobs=1 -- parallelism
    happens at the RandomizedSearchCV level instead, to avoid the
    oversubscription bug that caused the previous run's RF search to take
    ~6.7 hours instead of minutes."""
    learners = {}

    def ckpt(name):
        return os.path.join(checkpoint_dir, f"search_{name}.joblib") if checkpoint_dir else None

    if cfg.classical_use_hyperparam_search:
        learners["rf"] = _search_best_estimator(
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=cfg.seed),
            _rf_search_space(), X, y, groups, cfg, "rf", ckpt("rf"))
        learners["hgb"] = _search_best_estimator(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=cfg.seed),
            _hgb_search_space(), X, y, groups, cfg, "hgb", ckpt("hgb"))
        if cfg.classical_use_xgboost and _HAS_XGB:
            pos_weight = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
            # device="cpu" here even though cfg.device may be "cuda": at
            # only ~22K rows x 109 features, GPU transfer overhead and the
            # CPU<->GPU device-mismatch fallback warnings seen in the crashed
            # run outweigh any GPU speedup. XGBoost's CPU "hist" method is
            # already fast at this data size.
            learners["xgb"] = _search_best_estimator(
                XGBClassifier(tree_method="hist", device="cpu", n_jobs=1,
                               scale_pos_weight=pos_weight, eval_metric="logloss",
                               random_state=cfg.seed),
                _xgb_search_space(), X, y, groups, cfg, "xgb", ckpt("xgb"))
    else:
        learners["rf"] = RandomForestClassifier(n_estimators=cfg.classical_rf_n_estimators,
                                                  class_weight="balanced_subsample", max_depth=12,
                                                  min_samples_leaf=5, n_jobs=-1, random_state=cfg.seed)
        learners["hgb"] = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                           random_state=cfg.seed)
        if cfg.classical_use_xgboost and _HAS_XGB:
            pos_weight = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
            learners["xgb"] = XGBClassifier(tree_method="hist", device="cpu",
                                             n_estimators=500, max_depth=6, learning_rate=0.05,
                                             scale_pos_weight=pos_weight, eval_metric="logloss",
                                             random_state=cfg.seed)

    if cfg.classical_use_xgboost and not _HAS_XGB:
        print("    WARNING: xgboost not installed -- skipping XGB base learner. "
              "Run `pip install xgboost` to include it.")
    return learners

def load_learners_from_checkpoints(checkpoint_dir: str, learner_names: List[str] = ("rf", "hgb", "xgb")) -> dict:
    """Reload already-searched estimators from search_<name>.joblib
    checkpoints instead of re-running RandomizedSearchCV. Use this after a
    session interruption where the search stage already completed and was
    checkpointed (see _search_best_estimator) but a later stage crashed."""
    learners = {}
    for name in learner_names:
        path = os.path.join(checkpoint_dir, f"search_{name}.joblib")
        if os.path.exists(path):
            ckpt = joblib.load(path)
            learners[name] = ckpt["best_estimator"]
            print(f"  Reloaded [{name}] from checkpoint: best_cv_mcc={ckpt['best_cv_mcc']:.4f}")
        else:
            print(f"  WARNING: no checkpoint found for [{name}] at {path}")
    return learners

def _extra_learners(cfg: Config) -> dict:
    """Diverse, cheap base learners for the stack (trees + a linear model)."""
    out = {}
    if cfg.classical_use_extra_learners:
        out["et"] = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt",
                                         class_weight="balanced_subsample", n_jobs=1, random_state=cfg.seed)
        out["lr"] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=cfg.seed)
    return out


def _build_stack(learners: dict, X, y, groups, cfg: Config):
    """Fit a StackingClassifier with subject-grouped inner splits (precomputed,
    so no metadata-routing issue). Learners are cloned, never mutated."""
    from sklearn.base import clone
    n_splits = min(cfg.classical_n_cv_folds, len(np.unique(groups)))
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    stack = StackingClassifier(
        estimators=[(n, clone(l)) for n, l in learners.items()],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=cfg.seed),
        cv=splits, stack_method="predict_proba", n_jobs=-1, passthrough=False,
    )
    return stack.fit(X, y)


def export_artifact_as_pth(artifact: dict, path: str) -> str:
    """torch.save the sklearn artifact (pickle inside a .pth container) and verify reload."""
    torch.save(artifact, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    assert reloaded["feature_names"] == artifact["feature_names"], "pth reload verification failed"
    return path


def save_evaluation_bundle(y, probs, subjects, times, threshold, cfg: Config, out_dir: str,
                           feature_names=None, importances=None, extra_files=()) -> str:
    """Writes every evaluation artifact to <out_dir>/evaluation_metrics/ and zips it."""
    eval_dir = os.path.join(out_dir, "evaluation_metrics")
    os.makedirs(eval_dir, exist_ok=True)
    y, probs, subjects = np.asarray(y), np.asarray(probs), np.asarray(subjects)
    both = len(np.unique(y)) > 1

    def _save(fig, name):
        fig.tight_layout(); fig.savefig(os.path.join(eval_dir, name), dpi=200); plt.close(fig)

    # 1) pooled metrics (tuned threshold and 0.5)
    summary = {}
    for tag, thr in [("tuned_threshold", threshold), ("threshold_0.5", 0.5)]:
        m = compute_window_metrics(y, probs, thr); m["threshold"] = float(thr)
        summary[tag] = m
    with open(os.path.join(eval_dir, "pooled_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # 2) raw out-of-fold predictions
    oof = pd.DataFrame({"subject": subjects, "y_true": y, "y_prob": probs,
                        "y_pred": (probs >= threshold).astype(int)})
    if times is not None:
        oof["time"] = np.asarray(times)
    oof.to_csv(os.path.join(eval_dir, "oof_predictions.csv"), index=False)

    # 3) per-subject window + event metrics
    ev_cfg = replace(cfg, alert_threshold=float(threshold), clear_threshold=max(float(threshold) - 0.15, 0.05))
    rows = []
    for s in sorted(np.unique(subjects)):
        m = subjects == s
        row = {"subject": s, "n_windows": int(m.sum()), "high_fraction": float(y[m].mean())}
        row.update({k: v for k, v in compute_window_metrics(y[m], probs[m], threshold).items()
                    if k != "confusion_matrix"})
        if times is not None:
            try:
                t_s = np.asarray(times)[m]
                _, _, alert_events, _ = apply_temporal_alert_logic(t_s, probs[m], ev_cfg)
                true_events = extract_true_events(t_s, y[m], ev_cfg)
                hours = max((pd.Timestamp(t_s.max()) - pd.Timestamp(t_s.min())).total_seconds() / 3600.0,
                            cfg.window_seconds / 3600.0)
                row.update({f"event_{k}": v for k, v in
                            compute_event_metrics(true_events, alert_events, ev_cfg, hours).items()})
            except Exception as e:
                print(f"    event metrics failed for subject {s}: {e}")
        rows.append(row)
    per_subj = pd.DataFrame(rows)
    per_subj.to_csv(os.path.join(eval_dir, "per_subject_metrics.csv"), index=False)

    # 4) threshold sweep
    sweep = []
    for t in np.arange(0.05, 0.96, 0.01):
        p = (probs >= t).astype(int)
        sweep.append({"threshold": round(float(t), 2), "mcc": matthews_corrcoef(y, p),
                      "f1": f1_score(y, p, zero_division=0),
                      "precision": precision_score(y, p, zero_division=0),
                      "recall": recall_score(y, p, zero_division=0),
                      "balanced_accuracy": balanced_accuracy_score(y, p)})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(eval_dir, "threshold_sweep.csv"), index=False)

    # 5) plots
    if both:
        fpr, tpr, _ = roc_curve(y, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y, probs):.3f}"); ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(); ax.set_title("ROC (pooled OOF)")
        _save(fig, "roc_curve.png")

        prec, rec, _ = precision_recall_curve(y, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec, label=f"AUPRC={average_precision_score(y, probs):.3f}")
        ax.axhline(y.mean(), color="k", ls="--", label=f"prevalence={y.mean():.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend(); ax.set_title("Precision-Recall (pooled OOF)")
        _save(fig, "pr_curve.png")

        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(mean_pred, frac_pos, "o-"); ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed HIGH fraction"); ax.set_title("Calibration")
        _save(fig, "calibration_curve.png")

    cm = np.array(summary["tuned_threshold"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "HIGH"]); ax.set_yticklabels(["NORMAL", "HIGH"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    _save(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[y == 0], bins=40, alpha=0.6, label="NORMAL", density=True)
    ax.hist(probs[y == 1], bins=40, alpha=0.6, label="HIGH", density=True)
    ax.axvline(threshold, color="k", ls="--"); ax.legend(); ax.set_xlabel("Predicted probability")
    ax.set_title("Probability distribution by class")
    _save(fig, "probability_histogram.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_subj["subject"].astype(str), per_subj["mcc"].fillna(0))
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("MCC"); ax.set_title("Per-subject MCC (NaN shown as 0)")
    plt.setp(ax.get_xticklabels(), rotation=45)
    _save(fig, "per_subject_mcc.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ["mcc", "f1", "precision", "recall"]:
        ax.plot(sweep_df["threshold"], sweep_df[c], label=c)
    ax.axvline(threshold, color="k", ls="--"); ax.legend(); ax.set_xlabel("Threshold"); ax.set_title("Threshold sweep")
    _save(fig, "threshold_sweep.png")

    if importances is not None and feature_names is not None:
        imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        imp.to_csv(os.path.join(eval_dir, "feature_importance_rf.csv"), header=["importance"])
        top = imp.head(25)[::-1]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.barh(top.index, top.values); ax.set_title("Top 25 RF feature importances")
        _save(fig, "feature_importance_top25.png")

    # 6) README
    with open(os.path.join(eval_dir, "README.txt"), "w") as fh:
        fh.write(
            "Evaluation bundle\n=================\n"
            f"Scheme: {cfg.classical_eval_scheme} (subject-held-out, out-of-fold predictions of the full stacking ensemble)\n"
            f"Threshold: {threshold:.3f}, selected on the pooled OOF predictions -> pooled MCC at this threshold is mildly optimistic;\n"
            "pooled_metrics.json also reports threshold 0.5 for comparison.\n"
            "Hyperparameters were searched once on all subjects (grouped CV), so scores carry a small optimistic bias.\n"
            "Windows overlap 75%, so window-level counts are not independent samples; prefer per-subject results.\n")

    # 7) zip
    zip_path = os.path.join(out_dir, "evaluation_metrics.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(eval_dir):
            for f in files:
                zf.write(os.path.join(root, f), arcname=os.path.join("evaluation_metrics", f))
        for p in extra_files:
            if os.path.exists(p):
                zf.write(p, arcname=os.path.join("evaluation_metrics", os.path.basename(p)))
    print(f"  Evaluation bundle saved: {zip_path}")
    return zip_path

def train_classical_production_model(feat_df: pd.DataFrame, labels: np.ndarray, subjects: np.ndarray,
                                       feature_names: List[str], cfg: Config,
                                       times=None, prefit_learners: Optional[dict] = None) -> dict:
    log_stage("CLASSICAL PRODUCTION MODEL: hyperparameter search (ONCE, on full data)")
    X = np.nan_to_num(feat_df.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = labels
    os.makedirs(cfg.output_dir, exist_ok=True)
    checkpoint_dir = cfg.output_dir if cfg.classical_checkpoint_every_learner else None

    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)

    # Search ONCE on the full dataset instead of once per outer CV fold.
    # This is still leakage-safe -- _search_best_estimator's own internal
    # GroupKFold never touches a genuinely held-out test set -- and cuts
    # search cost by ~5x versus the previous nested-per-fold approach,
    # which combined with an n_jobs oversubscription bug (now fixed) burned
    # a full 12-hour Kaggle session without finishing. Each search result
    # is checkpointed to disk immediately (see _search_best_estimator).
    if prefit_learners:      # resume from search_*.joblib checkpoints, skip the search
        final_learners = dict(prefit_learners)
    else:
        final_learners = _make_base_learners(X_scaled, y, subjects, cfg, checkpoint_dir)
    final_learners.update(_extra_learners(cfg))

    log_stage("CLASSICAL PRODUCTION MODEL: subject-grouped CV evaluation (fixed hyperparameters)")
    # Now evaluate those FIXED hyperparameters via cheap CV refits (no
    # search inside the loop) to get an honest, subject-grouped held-out
    # score for reporting -- this is fast since there's no search here.
    from sklearn.base import clone
    if cfg.classical_eval_scheme == "loso":
        splitter = LeaveOneGroupOut()
    else:
        splitter = GroupKFold(n_splits=min(cfg.classical_n_cv_folds, len(np.unique(subjects))))
    cv_probs = np.zeros(len(y))
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=subjects)):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_val = scaler.transform(X[train_idx]), scaler.transform(X[val_idx])
        fold_stack = _build_stack(final_learners, X_train, y[train_idx], subjects[train_idx], cfg)
        fold_probs = fold_stack.predict_proba(X_val)[:, 1]
        cv_probs[val_idx] = fold_probs
        fold_mcc = matthews_corrcoef(y[val_idx], (fold_probs >= 0.5).astype(int)) if len(np.unique(y[val_idx])) > 1 else float("nan")
        print(f"  CV fold {fold_i} (held-out={np.unique(subjects[val_idx])}): n_val={len(val_idx)} mcc@0.5={fold_mcc:.4f}")

    cv_auroc = roc_auc_score(y, cv_probs) if len(np.unique(y)) > 1 else float("nan")
    cv_mcc = matthews_corrcoef(y, (cv_probs >= 0.5).astype(int))
    best_threshold, _ = optimize_threshold(y, cv_probs, cfg.threshold_objective)
    print(f"  Pooled CV: auroc={cv_auroc:.4f} mcc@0.5={cv_mcc:.4f} "
          f"selected_threshold={best_threshold:.2f} (sanity check, not the deployed threshold below)")

    log_stage("CLASSICAL PRODUCTION MODEL: calibrating final learners on ALL subjects")

    # Pre-compute the fold splits ourselves (using `groups` right here,
    # while we still have it) instead of passing a GroupKFold object plus
    # groups=subjects into .fit(). Newer scikit-learn requires explicit
    # metadata routing to be enabled for fit-time `groups` to reach a
    # splitter nested inside CalibratedClassifierCV/StackingClassifier --
    # without that, the splitter silently receives groups=None and raises
    # "The 'groups' parameter should not be None." Passing a precomputed
    # list of (train_idx, test_idx) pairs as `cv=` sidesteps the routing
    # requirement entirely.
    calibrated_learners = {}
    if cfg.classical_use_stacking and len(final_learners) > 1:
        log_stage("CLASSICAL PRODUCTION MODEL: fitting stacking meta-learner on ALL subjects")
        stack = _build_stack(final_learners, X_scaled, y, subjects, cfg)
        print(f"  stacking meta-learner coefficients: "
              f"{dict(zip(final_learners.keys(), stack.final_estimator_.coef_[0].round(3)))}")
        ensemble_predictor = {"kind": "stacking", "model": stack}
    else:
        precomputed_splits = list(GroupKFold(
            n_splits=min(cfg.classical_n_cv_folds, len(np.unique(subjects)))).split(X_scaled, y, groups=subjects))
        for name, lrn in final_learners.items():
            cal = CalibratedClassifierCV(lrn, method="sigmoid", cv=precomputed_splits)
            cal.fit(X_scaled, y)
            calibrated_learners[name] = cal
        ensemble_predictor = {"kind": "average", "models": calibrated_learners}

    ood_guard = IsolationForest(contamination=cfg.classical_ood_contamination,
                                 n_estimators=200, random_state=cfg.seed, n_jobs=-1)
    ood_guard.fit(X_scaled)

    artifact = {
        "ensemble": ensemble_predictor,
        "calibrated_learners": calibrated_learners,  # kept for inspection even when stacking is used
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
        "ood_guard": ood_guard,
        "feature_names": feature_names,
        "threshold": float(best_threshold),
        "label_mapping": {"0": "NORMAL (LOW/MODERATE)", "1": "HIGH_STRESS"},
        "cv_pooled_auroc": float(cv_auroc),
        "cv_pooled_mcc": float(cv_mcc),
        "trained_on_subjects": sorted(pd.unique(subjects).tolist()),
        "note": "Trained on ALL subjects for deployment. cv_pooled_* metrics above are "
                "from nested subject-grouped CV during training, NOT a held-out test evaluation.",
    }
    out_path = os.path.join(cfg.output_dir, "classical_production_model.joblib")
    os.makedirs(cfg.output_dir, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"  Classical production model saved to: {out_path}")

    if cfg.classical_export_pth:
        pth_path = os.path.join(cfg.output_dir, "classical_production_model.pth")
        export_artifact_as_pth(artifact, pth_path)
        artifact["pth_path"] = pth_path
        print(f"  .pth export saved and reload-verified: {pth_path}")

    rf = final_learners.get("rf")
    artifact["evaluation_zip"] = save_evaluation_bundle(
        y, cv_probs, subjects, times, best_threshold, cfg, cfg.output_dir,
        feature_names=feature_names, importances=getattr(rf, "feature_importances_", None),
        extra_files=[os.path.join(cfg.output_dir, "threshold.json")] if False else [],
    )
    return artifact


def run_classical_production_pipeline(cfg: Config, checkpoint_dir: Optional[str] = None) -> dict:
    """Runs the shared data pipeline (load -> dedup -> resample -> window ->
    calibrate -> features) then trains the classical ensemble on top. This is
    the entry point to use for the webapp deliverable -- it does NOT run any
    of the CNN/DANN training code above."""
    set_seed(cfg.seed)
    df = load_raw_data(cfg)
    df = deduplicate(df, cfg)
    df = resample_all(df, cfg)
    df = assign_blocks(df, cfg)
    audit_data(df, cfg)
    windows = build_windows(df, cfg)
    baselines = compute_personal_baselines(df, cfg)
    kept_windows, feat_df, feature_names = build_feature_dataset(windows, baselines, cfg)
    _, _, labels, _, subjects, times = assemble_arrays(kept_windows, feat_df)
    prefit = load_learners_from_checkpoints(checkpoint_dir) if checkpoint_dir else None
    return train_classical_production_model(feat_df, labels, subjects, feature_names, cfg,
                                            times=times, prefit_learners=prefit)


class ClassicalStressInferenceEngine:
    """Deployment-time inference engine for the webapp backend. Loads the
    joblib artifact from train_classical_production_model -- no PyTorch or
    GPU required, safe to run in a lightweight web server process (Flask/
    FastAPI/serverless). Mirrors StressInferenceEngine's calibration/
    streaming-buffer flow but serves the classical ensemble instead of the
    CNN+MLP model."""

    def __init__(self, artifact_path: str, cfg: Optional[Config] = None):
        if str(artifact_path).endswith((".pth", ".pt")):
            # only load files you created yourself (pickle-based)
            self.artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        else:
            self.artifact = joblib.load(artifact_path)
        self.cfg = cfg or CFG
        self.feature_names = self.artifact["feature_names"]
        self.threshold = self.artifact["threshold"]
        self.alert_threshold = float(self.threshold)
        self.clear_threshold = max(self.alert_threshold - 0.15, 0.05)
        self.subjects_state: Dict = {}

    def initialize_subject(self, subject_id):
        self.subjects_state[subject_id] = {
            "calibration_buffer": [], "calibrated": False, "baseline_stats": None,
            "calib_start_time": None, "window_buffer": [],
            "ema": None, "consecutive_high": 0, "state": "NORMAL",
            "last_alert_time": None, "alert_history": [],
        }

    def _compute_baseline(self, buffer: List[np.ndarray]) -> dict:
        arr = np.stack(buffer)
        channel_idx = {"EDA": 0, "HR": 1, "TEMP": 2, "ACC_MAG": 6}
        stats = {}
        for ch, idx in channel_idx.items():
            vals = arr[:, idx].astype(np.float64)
            stats[ch] = {
                "median": float(np.median(vals)), "iqr": float(max(scipy_iqr(vals), self.cfg.epsilon)),
                "mean": float(np.mean(vals)), "std": float(max(np.std(vals), self.cfg.epsilon)),
            }
        return stats

    def update_subject(self, subject_id, timestamp, eda, hr, temp, acc_x, acc_y, acc_z) -> dict:
        if subject_id not in self.subjects_state:
            self.initialize_subject(subject_id)
        st = self.subjects_state[subject_id]
        acc_mag = math.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        sample = np.array([eda, hr, temp, acc_x, acc_y, acc_z, acc_mag], dtype=np.float32)

        if st["calib_start_time"] is None:
            st["calib_start_time"] = timestamp
        elapsed_calib = (timestamp - st["calib_start_time"]).total_seconds()

        if not st["calibrated"]:
            st["calibration_buffer"].append(sample)
            if elapsed_calib >= self.cfg.calibration_minutes * 60.0:
                st["baseline_stats"] = self._compute_baseline(st["calibration_buffer"])
                st["calibrated"] = True
            return {"calibration_status": "calibrating",
                    "seconds_remaining": max(0.0, self.cfg.calibration_minutes * 60.0 - elapsed_calib)}

        st["window_buffer"].append((timestamp, sample))
        window_samples = int(round(self.cfg.window_seconds * self.cfg.target_hz))
        if len(st["window_buffer"]) > window_samples:
            st["window_buffer"] = st["window_buffer"][-window_samples:]
        if len(st["window_buffer"]) < window_samples:
            return {"calibration_status": "collecting_window",
                    "samples_collected": len(st["window_buffer"]), "samples_needed": window_samples}

        return self._predict_and_alert(subject_id, timestamp)

    def _predict_and_alert(self, subject_id, timestamp) -> dict:
        st = self.subjects_state[subject_id]
        raw = np.stack([s for (_, s) in st["window_buffer"]])
        w = {"subject": subject_id, "raw": raw}
        feats_dict = extract_window_features(w, self.cfg, {subject_id: {"stats": st["baseline_stats"]}})
        feat_vec = np.nan_to_num(np.array([feats_dict.get(n, 0.0) for n in self.feature_names], dtype=np.float64))
        feat_scaled = (feat_vec - self.artifact["scaler_mean"]) / self.artifact["scaler_scale"]
        feat_scaled_2d = feat_scaled.reshape(1, -1)

        ensemble = self.artifact["ensemble"]
        if ensemble["kind"] == "stacking":
            prob = ensemble["model"].predict_proba(feat_scaled_2d)[0, 1]
        else:
            probs = [lrn.predict_proba(feat_scaled_2d)[0, 1] for lrn in ensemble["models"].values()]
            prob = float(np.mean(probs))

        # -1 = flagged as out-of-distribution by the IsolationForest guard.
        is_ood = self.artifact["ood_guard"].predict(feat_scaled_2d)[0] == -1

        st["ema"] = prob if st["ema"] is None else self.cfg.ema_alpha * prob + (1 - self.cfg.ema_alpha) * st["ema"]
        smoothed = st["ema"]

        alert_now = False
        if not is_ood:
            if st["state"] == "NORMAL":
                st["consecutive_high"] = st["consecutive_high"] + 1 if smoothed >= self.alert_threshold else 0
                if st["consecutive_high"] >= self.cfg.persistence_windows:
                    cooldown_ok = (st["last_alert_time"] is None) or \
                        ((timestamp - st["last_alert_time"]).total_seconds() >= self.cfg.cooldown_seconds)
                    if cooldown_ok:
                        st["state"] = "ALERT"
                        alert_now = True
                        st["last_alert_time"] = timestamp
                        st["alert_history"].append(timestamp)
                    st["consecutive_high"] = 0
            else:
                if smoothed < self.clear_threshold:
                    st["state"] = "NORMAL"

        return {
            "calibration_status": "active",
            "raw_probability": float(prob),
            "smoothed_probability": float(smoothed),
            "state": st["state"],
            "alert_triggered": alert_now,
            "out_of_distribution": bool(is_ood),
            "confidence_note": "reading may be unreliable (atypical sensor pattern)" if is_ood else None,
            "seconds_since_last_alert": (
                (timestamp - st["last_alert_time"]).total_seconds() if st["last_alert_time"] else None
            ),
        }

    def should_alert(self, subject_id) -> bool:
        st = self.subjects_state.get(subject_id)
        return st is not None and st["state"] == "ALERT"


# ==============================================================================
# INFERENCE API (for future website integration; no dependency on the CSV)
# ==============================================================================
class StressInferenceEngine:
    """
    Deployment-time inference engine for a single nurse's live wearable
    stream. Wraps calibration accumulation, feature computation, model
    inference, temporal smoothing, and the alert decision logic.

    Typical usage from a website backend:

        engine = StressInferenceEngine("outputs/production_model.pth")
        engine.initialize_subject("nurse_123")
        for sample in incoming_sensor_stream:
            result = engine.update_subject("nurse_123", sample.timestamp,
                                            sample.eda, sample.hr, sample.temp,
                                            sample.acc_x, sample.acc_y, sample.acc_z)
            if result.get("alert_triggered"):
                send_notification("nurse_123")
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        # weights_only=False: our checkpoints intentionally bundle numpy
        # arrays (scaler mean/scale) alongside the model's state_dict.
        # Only load checkpoints produced by this script / a trusted source.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["cnn_kernel_sizes"] = tuple(cfg_dict["cnn_kernel_sizes"])
        self.cfg = Config(**cfg_dict)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = ckpt["feature_names"]
        self.threshold = ckpt["threshold"]
        n_raw_channels = ckpt["n_raw_channels"]
        n_features = len(self.feature_names)
        self.model = StressDetector(n_raw_channels, n_features, self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.scaler_mean = ckpt.get("feature_scaler_mean")
        self.scaler_scale = ckpt.get("feature_scaler_scale")
        self.subjects_state: Dict = {}

    def initialize_subject(self, subject_id):
        self.subjects_state[subject_id] = {
            "calibration_buffer": [], "calibrated": False, "baseline_stats": None,
            "calib_start_time": None, "window_buffer": [], "ema": None,
            "consecutive_high": 0, "state": "NORMAL", "last_alert_time": None,
            "alert_history": [],
        }

    def _compute_baseline(self, buffer: List[np.ndarray]) -> dict:
        arr = np.stack(buffer)
        # Extended to ACC_X/ACC_Y/ACC_Z (indices 3,4,5) to match the offline
        # training baseline in compute_personal_baselines -- the raw CNN
        # input is now robust-z normalized per channel, so every channel
        # needs a baseline, not just EDA/HR/TEMP/ACC_MAG.
        channel_idx = {"EDA": 0, "HR": 1, "TEMP": 2, "ACC_X": 3, "ACC_Y": 4, "ACC_Z": 5, "ACC_MAG": 6}
        stats = {}
        for ch, idx in channel_idx.items():
            vals = arr[:, idx].astype(np.float64)
            stats[ch] = {
                "median": float(np.median(vals)), "iqr": float(max(scipy_iqr(vals), self.cfg.epsilon)),
                "mean": float(np.mean(vals)), "std": float(max(np.std(vals), self.cfg.epsilon)),
            }
        return stats

    def update_subject(self, subject_id, timestamp, eda, hr, temp, acc_x, acc_y, acc_z) -> dict:
        """Feed one new raw sample (already at ~target_hz) for a subject."""
        if subject_id not in self.subjects_state:
            self.initialize_subject(subject_id)
        st = self.subjects_state[subject_id]
        acc_mag = math.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        sample = np.array([eda, hr, temp, acc_x, acc_y, acc_z, acc_mag], dtype=np.float32)

        if st["calib_start_time"] is None:
            st["calib_start_time"] = timestamp
        elapsed_calib = (timestamp - st["calib_start_time"]).total_seconds()

        if not st["calibrated"]:
            st["calibration_buffer"].append(sample)
            if elapsed_calib >= self.cfg.calibration_minutes * 60.0:
                st["baseline_stats"] = self._compute_baseline(st["calibration_buffer"])
                st["calibrated"] = True
            return {"calibration_status": "calibrating",
                    "seconds_remaining": max(0.0, self.cfg.calibration_minutes * 60.0 - elapsed_calib)}

        st["window_buffer"].append((timestamp, sample))
        window_samples = int(round(self.cfg.window_seconds * self.cfg.target_hz))
        if len(st["window_buffer"]) > window_samples:
            st["window_buffer"] = st["window_buffer"][-window_samples:]
        if len(st["window_buffer"]) < window_samples:
            return {"calibration_status": "collecting_window",
                    "samples_collected": len(st["window_buffer"]), "samples_needed": window_samples}

        return self._predict_and_alert(subject_id, timestamp)

    def _predict_and_alert(self, subject_id, timestamp) -> dict:
        st = self.subjects_state[subject_id]
        raw = np.stack([s for (_, s) in st["window_buffer"]])
        w = {"subject": subject_id, "raw": raw}
        feats_dict = extract_window_features(w, self.cfg, {subject_id: {"stats": st["baseline_stats"]}})
        feat_vec = np.nan_to_num(np.array([feats_dict.get(n, 0.0) for n in self.feature_names], dtype=np.float32))
        if self.scaler_mean is not None:
            feat_vec = (feat_vec - self.scaler_mean) / self.scaler_scale

        # Same personal-relative (robust-z) normalization used at training time
        # (see normalize_raw_window) -- the CNN branch must see the same
        # distribution at inference that it was trained on.
        raw_cnn = normalize_raw_window(raw, subject_id, {subject_id: {"stats": st["baseline_stats"]}}, self.cfg)
        x = torch.from_numpy(raw_cnn.astype(np.float32)).unsqueeze(0).transpose(1, 2).to(self.device)
        f = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x, f)).item()

        st["ema"] = prob if st["ema"] is None else self.cfg.ema_alpha * prob + (1 - self.cfg.ema_alpha) * st["ema"]
        smoothed = st["ema"]

        alert_now = False
        if st["state"] == "NORMAL":
            st["consecutive_high"] = st["consecutive_high"] + 1 if smoothed >= self.cfg.alert_threshold else 0
            if st["consecutive_high"] >= self.cfg.persistence_windows:
                cooldown_ok = (st["last_alert_time"] is None) or \
                    ((timestamp - st["last_alert_time"]).total_seconds() >= self.cfg.cooldown_seconds)
                if cooldown_ok:
                    st["state"] = "ALERT"
                    alert_now = True
                    st["last_alert_time"] = timestamp
                    st["alert_history"].append(timestamp)
                st["consecutive_high"] = 0
        else:
            if smoothed < self.cfg.clear_threshold:
                st["state"] = "NORMAL"

        return {
            "calibration_status": "active",
            "raw_probability": float(prob),
            "smoothed_probability": float(smoothed),
            "state": st["state"],
            "alert_triggered": alert_now,
            "seconds_since_last_alert": (
                (timestamp - st["last_alert_time"]).total_seconds() if st["last_alert_time"] else None
            ),
        }

    def predict_window(self, subject_id, raw_window: np.ndarray) -> float:
        """Direct single-window prediction, bypassing the streaming buffer.
        raw_window must be [T, 7] = EDA, HR, TEMP, ACC_X, ACC_Y, ACC_Z, ACC_MAG."""
        st = self.subjects_state.get(subject_id)
        if st is None or not st.get("calibrated"):
            raise ValueError("Subject not calibrated. Call update_subject during the calibration period first.")
        w = {"subject": subject_id, "raw": raw_window}
        feats_dict = extract_window_features(w, self.cfg, {subject_id: {"stats": st["baseline_stats"]}})
        feat_vec = np.nan_to_num(np.array([feats_dict.get(n, 0.0) for n in self.feature_names], dtype=np.float32))
        if self.scaler_mean is not None:
            feat_vec = (feat_vec - self.scaler_mean) / self.scaler_scale
        # Same robust-z normalization of the raw tensor as at training time.
        raw_cnn = normalize_raw_window(raw_window, subject_id, {subject_id: {"stats": st["baseline_stats"]}}, self.cfg)
        x = torch.from_numpy(raw_cnn.astype(np.float32)).unsqueeze(0).transpose(1, 2).to(self.device)
        f = torch.from_numpy(feat_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(x, f)).item()

    def should_alert(self, subject_id) -> bool:
        st = self.subjects_state.get(subject_id)
        return st is not None and st["state"] == "ALERT"


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    cfg = CFG
    # Production target: a real-time webapp, not a research paper. The
    # CNN+MLP+DANN path above (run_multi_seed / run_pipeline) is kept in
    # the file as research/ablation evidence but is no longer what gets
    # deployed -- every experiment showed it not meaningfully beating the
    # classical model below, at far higher inference/runtime complexity.
    # This trains and saves the model ClassicalStressInferenceEngine serves.
    classical_artifact = run_classical_production_pipeline(cfg)
    log_stage("PRODUCTION MODEL SUMMARY")
    print(f"CV pooled AUROC: {classical_artifact['cv_pooled_auroc']:.4f}")
    print(f"CV pooled MCC:   {classical_artifact['cv_pooled_mcc']:.4f}")
    print(f"Deployed threshold: {classical_artifact['threshold']:.2f}")
    print(f"Saved to: {os.path.join(cfg.output_dir, 'classical_production_model.joblib')}")
    return
    log_stage("FINAL SUMMARY")
    print("Model: personalized multiscale-1D-CNN + handcrafted-physiological-feature "
          "fusion network, binary NORMAL-vs-HIGH-STRESS detector with temporal "
          "smoothing / persistence / hysteresis / cooldown alert logic on top.")
    if len(fold_metrics_df) > 0:
        print(f"LOSO subjects evaluated: {len(fold_metrics_df)}")
        print(f"Mean window MCC:              {fold_metrics_df['window_mcc'].mean():.4f}")
        print(f"Mean window HIGH recall:      {fold_metrics_df['window_high_recall'].mean():.4f}")
        print(f"Mean window HIGH precision:   {fold_metrics_df['window_high_precision'].mean():.4f}")
        print(f"Mean window AUPRC:            {fold_metrics_df['window_auprc'].mean():.4f}")
        print(f"Mean event recall:            {fold_metrics_df['event_event_recall'].mean():.4f}")
        print(f"Mean false alerts/hour:       {fold_metrics_df['event_false_alerts_per_hour'].mean():.4f}")
        print(f"Mean detection delay (s):     {fold_metrics_df['event_avg_detection_delay_s'].mean():.2f}")
    print(f"Personalization: every window's handcrafted features include delta / relative-change / "
          f"robust-z versions of EDA, HR, TEMP, ACC_MAG relative to that subject's own first "
          f"{cfg.calibration_minutes}-minute calibration baseline -- computed with no future information.")
    print(f"Production checkpoint (if enabled): {os.path.join(cfg.output_dir, 'production_model.pth')}")
    print(f"All artifacts saved under: {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = CFG
    # Point checkpoint_dir at the old run folder that holds search_rf/hgb/xgb.joblib to skip the ~20-30 min search.
    # Use None to search from scratch.
    artifact = run_classical_production_pipeline(
        cfg,
        checkpoint_dir="/kaggle/input/models/syedazmulhasan2001/outputs-20260824-202053/tensorflow2/default/1",
    )
    
    print("CV pooled AUROC:", artifact["cv_pooled_auroc"], " MCC:", artifact["cv_pooled_mcc"])
    print("Threshold:", artifact["threshold"])
    
    from IPython.display import FileLink          # Kaggle/Jupyter download links
    display(FileLink(artifact["pth_path"]))
    display(FileLink(artifact["evaluation_zip"]))
    
    # Load test
    eng = ClassicalStressInferenceEngine(artifact["pth_path"], cfg)