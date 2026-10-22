"""
stress_inference.py
===================
Inference helpers for the Nurse Stress Detector.

What this adds on top of ``ClassicalStressInferenceEngine`` (defined in
``Nurse_Stress_Detector_training.py``):

1. ``StridedStressEngine`` -- runs the model once per ``stride_seconds``
   (15 s by default) instead of on every incoming 4 Hz sample. The EMA
   smoothing, persistence count and cooldown in the alert logic were tuned
   and evaluated on windows spaced one stride apart, so predicting at every
   sample would make "3 consecutive windows" mean 0.75 s instead of 45 s.
   It also clears the rolling window whenever the stream has a gap longer
   than ``max_gap_seconds``, mirroring the block-splitting used in training.
2. ``download_model`` -- fetches the published model from Kaggle.
3. A small CLI that replays a CSV recording through the engine.

Usage
-----
    python stress_inference.py --csv merged_data.csv --subject 15 \
        --kaggle-handle <owner>/<model>/<framework>/<variation>/<version>

    python stress_inference.py --csv merged_data.csv --subject 15 \
        --model path/to/classical_production_model.joblib

The CSV needs the columns: datetime, EDA, HR, TEMP, X, Y, Z (and optionally
id). Column names match the public Kaggle copy of the nurse stress dataset.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from Nurse_Stress_Detector_training import CFG, ClassicalStressInferenceEngine

KAGGLE_MODEL_URL = "https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector"
ARTIFACT_NAMES = ("classical_production_model.joblib", "classical_production_model.pth")
SIGNAL_COLS = ["EDA", "HR", "TEMP", "ACC_X", "ACC_Y", "ACC_Z"]


# ------------------------------------------------------------------------------
# Model download
# ------------------------------------------------------------------------------
def download_model(handle: str) -> str:
    """Download a model from Kaggle with kagglehub and return the artifact path.

    ``handle`` is the full Kaggle model handle,
    ``<owner>/<model>/<framework>/<variation>/<version>``. Copy it from the
    "Use this model" / download snippet on the model page:
    https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector
    """
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("kagglehub is not installed. Run: pip install kagglehub") from e

    local_dir = kagglehub.model_download(handle)
    for name in ARTIFACT_NAMES:  # prefer .joblib, fall back to .pth
        for root, _, files in os.walk(local_dir):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f"None of {ARTIFACT_NAMES} found under {local_dir}")


# ------------------------------------------------------------------------------
# Engine with stride + gap handling
# ------------------------------------------------------------------------------
class StridedStressEngine(ClassicalStressInferenceEngine):
    """Classical engine that predicts once per stride and resets on data gaps."""

    def update_subject(self, subject_id, timestamp, eda, hr, temp, acc_x, acc_y, acc_z) -> dict:
        st = self.subjects_state.get(subject_id)
        if st is not None:
            last_ts = st.get("_last_ts")
            if (st.get("calibrated") and last_ts is not None
                    and (timestamp - last_ts).total_seconds() > self.cfg.max_gap_seconds):
                # Sensor dropout: never build a window across it (see assign_blocks in training).
                st["window_buffer"] = []
                st["_since_pred"] = 0
                st["_last_result"] = None
        result = super().update_subject(subject_id, timestamp, eda, hr, temp, acc_x, acc_y, acc_z)
        self.subjects_state[subject_id]["_last_ts"] = timestamp
        return result

    def _predict_and_alert(self, subject_id, timestamp) -> dict:
        st = self.subjects_state[subject_id]
        stride = max(int(round(self.cfg.stride_seconds * self.cfg.target_hz)), 1)
        st["_since_pred"] = st.get("_since_pred", 0) + 1
        last = st.get("_last_result")
        if last is not None and st["_since_pred"] < stride:
            out = dict(last)
            out["alert_triggered"] = False   # alerts only fire on a fresh prediction
            out["fresh_prediction"] = False
            return out
        st["_since_pred"] = 0
        out = super()._predict_and_alert(subject_id, timestamp)
        out["fresh_prediction"] = True
        st["_last_result"] = out
        return out


def load_engine(model_path: str, cfg=CFG) -> StridedStressEngine:
    return StridedStressEngine(model_path, cfg)


# ------------------------------------------------------------------------------
# CSV -> ~4 Hz stream (same preprocessing as training, minus labels)
# ------------------------------------------------------------------------------
def prepare_stream(df: pd.DataFrame, hz: float = 4.0) -> pd.DataFrame:
    """Rename X/Y/Z, drop consecutive duplicate rows, bin to ``hz`` by elapsed time."""
    df = df.rename(columns={"X": "ACC_X", "Y": "ACC_Y", "Z": "ACC_Z"}).copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    same = np.ones(len(df), dtype=bool)
    for c in SIGNAL_COLS:
        same &= (df[c].values == df[c].shift(1).values)
    same[0] = False
    df = df.loc[~same].reset_index(drop=True)

    elapsed = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().values
    bins = np.floor(elapsed * hz).astype(np.int64)
    out = (df.assign(_bin=bins)
             .groupby("_bin", sort=True)
             .agg(datetime=("datetime", "first"), **{c: (c, "mean") for c in SIGNAL_COLS})
             .reset_index(drop=True))
    return out


def replay(engine: StridedStressEngine, stream: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """Feed a prepared stream through the engine; return one row per fresh prediction."""
    engine.initialize_subject(subject_id)
    rows = []
    for r in stream.itertuples(index=False):
        res = engine.update_subject(subject_id, r.datetime, r.EDA, r.HR, r.TEMP,
                                    r.ACC_X, r.ACC_Y, r.ACC_Z)
        if res.get("calibration_status") == "active" and res.get("fresh_prediction"):
            rows.append({"time": r.datetime, **res})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Replay a wearable recording through the stress detector.")
    p.add_argument("--csv", required=True, help="CSV with datetime, EDA, HR, TEMP, X, Y, Z (+ id)")
    p.add_argument("--subject", default=None, help="Subject id to replay (required if the CSV has an 'id' column)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Path to classical_production_model.joblib / .pth")
    src.add_argument("--kaggle-handle", help="Full Kaggle model handle (see README)")
    p.add_argument("--out", default="predictions.csv", help="Where to write the per-window predictions")
    args = p.parse_args(argv)

    model_path = args.model or download_model(args.kaggle_handle)
    print(f"Model: {model_path}")

    df = pd.read_csv(args.csv)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str).str.strip()
        if args.subject is None:
            raise SystemExit(f"--subject is required. Available ids: {sorted(df['id'].unique())}")
        df = df[df["id"] == str(args.subject)]
        if df.empty:
            raise SystemExit(f"No rows for subject {args.subject!r}")
    subject_id = str(args.subject or "subject")

    stream = prepare_stream(df, CFG.target_hz)
    print(f"Stream: {len(stream):,} samples at ~{CFG.target_hz} Hz "
          f"({len(stream) / CFG.target_hz / 60:.1f} min); "
          f"first {CFG.calibration_minutes} min are used for calibration.")

    engine = load_engine(model_path)
    res = replay(engine, stream, subject_id)
    if res.empty:
        raise SystemExit("No predictions produced -- recording is shorter than calibration + one window.")

    res.to_csv(args.out, index=False)
    alerts = res[res["alert_triggered"]]
    print(f"{len(res)} predictions written to {args.out}")
    print(f"Mean smoothed probability: {res['smoothed_probability'].mean():.3f} | "
          f"out-of-distribution windows: {int(res['out_of_distribution'].sum())}")
    print(f"Alerts: {len(alerts)}")
    for t in alerts["time"]:
        print(f"  ALERT at {t}")


if __name__ == "__main__":
    main()
