# Nurse Stress Detection

Real-time detection of **sustained high stress** in nurses from wearable sensor streams (EDA, heart rate, skin temperature, 3-axis accelerometer).

[![Model on Kaggle](https://img.shields.io/badge/Kaggle-Model-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

The trained model is publicly available on Kaggle:
**https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector**

> **Research prototype, not a medical device.** Predictions are not a diagnosis and must not be used for clinical, employment, or disciplinary decisions. See [`MODEL_CARD.md`](MODEL_CARD.md).

---

## What it does

The production task is binary:

| Original label | Model target |
|---|---|
| LOW, MODERATE | `NORMAL` (0) |
| HIGH | `HIGH_STRESS` (1) |

A 60 s window of signal (stride 15 s, resampled to 4 Hz) is turned into a vector of handcrafted features, and a calibrated ensemble outputs `P(HIGH_STRESS)`. A temporal alert layer (EMA smoothing, persistence, hysteresis, cooldown) turns the per-window probabilities into alerts, so the system never fires on a single noisy window.

### Pipeline

1. **Load / clean**: rename accelerometer axes, remove only *consecutive* duplicate rows per subject (an artifact of forward-filling slow sensors onto the accelerometer clock).
2. **Resample** to ~4 Hz by per-subject elapsed-time binning (no wall-clock `resample()`, so large recording gaps don't blow up memory).
3. **Gap-aware windowing**: windows are never built across a gap longer than 5 s.
4. **Personal baseline**: each nurse's first 7.5 minutes define a median/IQR baseline; later windows get delta, relative-change and robust-z features. Windows inside the calibration period are discarded.
5. **Features**: time-domain stats, SCR/EDA tonic-phasic decomposition, frequency-domain features, accelerometer jerk/zero-crossings, personal-relative features.
6. **Model**: subject-grouped hyperparameter search for RandomForest, HistGradientBoosting and XGBoost, plus ExtraTrees and logistic regression, blended by a logistic-regression stacking meta-learner. An IsolationForest guard flags atypical windows.
7. **Alerts**: EMA smoothing → N consecutive windows above threshold → enter/clear hysteresis → cooldown.

A multiscale 1D-CNN + feature-MLP fusion network (with an optional domain-adversarial head) is also included in the training script as research/ablation code. It did not meaningfully beat the classical ensemble, so the classical model is the one that is deployed.

## Repository layout

```
Nurse-Stress-Detection/
├── Nurse_Stress_Detector_training.py   # full pipeline: data → features → LOSO/CV → models → engines
├── stress_inference.py                 # model download, strided streaming engine, CSV replay CLI
├── requirements.txt
├── MODEL_CARD.md                       # intended use, data, evaluation protocol, limitations
├── LICENSE
└── .gitignore
```

## Quick start

### 1. Install

```bash
git clone https://github.com/sabbir-just/Nurse-Stress-Detection.git
cd Nurse-Stress-Detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` is required even for the classical model because the training module imports it at the top level.

### 2. Get the model

Download from the [Kaggle model page](https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector), or let `kagglehub` do it. Copy the full handle (`<owner>/<model>/<framework>/<variation>/<version>`) from the page's *Use this model* snippet:

```python
import kagglehub
path = kagglehub.model_download("syedazmulhasansabbir/nurse-stress-detector/<framework>/<variation>/<version>")
print(path)   # contains classical_production_model.joblib / .pth
```

> Only load model files from sources you trust: `.pth` files are pickle-based, and `.joblib` files are too.
> Use the same major `scikit-learn` / `xgboost` versions as in `requirements.txt` to avoid unpickling errors.

### 3. Run on a recording

The CSV needs the columns `datetime, EDA, HR, TEMP, X, Y, Z` (and optionally `id`), as in the public [Kaggle copy of the dataset](https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors).

```bash
python stress_inference.py \
    --csv merged_data.csv --subject 15 \
    --model path/to/classical_production_model.joblib \
    --out predictions.csv
```

Or download and run in one step with `--kaggle-handle <owner>/<model>/<framework>/<variation>/<version>`.

### 4. Use it from Python (streaming)

```python
import pandas as pd
from stress_inference import load_engine

engine = load_engine("classical_production_model.joblib")
engine.initialize_subject("nurse_123")

for s in incoming_samples:   # ~4 Hz stream
    out = engine.update_subject("nurse_123", pd.Timestamp(s.time),
                                s.eda, s.hr, s.temp, s.acc_x, s.acc_y, s.acc_z)
    if out.get("alert_triggered"):
        notify("nurse_123")
```

During the first 7.5 minutes `update_subject` returns `calibration_status: "calibrating"`; then `"collecting_window"` until 60 s of data is buffered; then `"active"` with `raw_probability`, `smoothed_probability`, `state`, `alert_triggered` and `out_of_distribution`.

`load_engine` returns a `StridedStressEngine`, which predicts once per 15 s stride and resets its window after sensor gaps, matching how the alert logic was tuned. Prefer it over instantiating `ClassicalStressInferenceEngine` directly.

## Training

The script was written for Kaggle. To reproduce:

1. Get the data: [`priyankraval/nurse-stress-prediction-wearable-sensors`](https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors) (`merged_data.csv`).
2. Set `Config.data_path` (top of `Nurse_Stress_Detector_training.py`) to your copy of the CSV.
3. In the `__main__` block, change `checkpoint_dir=` to `None` to run the hyperparameter search from scratch (it points to a Kaggle input folder that only exists in the original session), and remove the Kaggle/IPython `FileLink` lines if you are not running in a notebook.
4. Run:

```bash
python Nurse_Stress_Detector_training.py
```

Outputs go to a timestamped `outputs_YYYYmmdd_HHMMSS/` folder: `classical_production_model.joblib` / `.pth`, per-learner `search_*.joblib` checkpoints, and `evaluation_metrics.zip`.

## Evaluation protocol

- Out-of-fold predictions come from **subject-held-out** evaluation (leave-one-subject-out by default, `classical_eval_scheme="loso"`).
- Reported: MCC, AUPRC, AUROC, precision/recall/F1, balanced accuracy, confusion matrix, per-subject metrics, calibration curve, threshold sweep, and event-level metrics (event recall, false alerts/hour, detection delay). Accuracy is never used for model or threshold selection.
- `evaluation_metrics.zip` contains `pooled_metrics.json`, `per_subject_metrics.csv`, `oof_predictions.csv`, plots, and a README with caveats.

Please read the caveats before quoting numbers: hyperparameters are searched once on all subjects, and the deployed threshold is chosen on pooled out-of-fold predictions, so pooled scores are mildly optimistic. Windows overlap 75 %, so window counts are not independent samples; prefer per-subject results. The dataset has only 15 nurses.

## Dataset and citation

Data: Hosseini, S. et al. *A multimodal sensor dataset for continuous stress detection of nurses in a hospital.* Scientific Data 9 (2022). https://doi.org/10.1038/s41597-022-01361-y. Wearable recordings from 15 nurses over about one week of regular shifts. Please follow the dataset's own license and citation requirements.

## License

Code is released under the [MIT License](LICENSE). The dataset and the Kaggle model page have their own terms.
