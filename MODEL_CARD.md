# Model Card: Nurse Stress Detector

Published model: https://www.kaggle.com/models/syedazmulhasansabbir/nurse-stress-detector

## Summary

Binary classifier that estimates whether a nurse is in a **sustained high-stress** state from a 60-second window of wearable data (EDA, heart rate, skin temperature, 3-axis accelerometer). Output is a calibrated probability, `P(HIGH_STRESS)`, which an alert layer smooths and thresholds.

| | |
|---|---|
| Task | Binary: `NORMAL` (LOW + MODERATE) vs `HIGH_STRESS` |
| Input | 60 s window at ~4 Hz, stride 15 s: EDA, HR, TEMP, ACC_X/Y/Z (+ derived magnitude) |
| Personalization | Per-nurse baseline from the first 7.5 min of recording (median/IQR) |
| Features | Time-domain, SCR/EDA decomposition, frequency-domain, accelerometer, personal-relative |
| Model | Stacking ensemble (RandomForest, HistGradientBoosting, XGBoost, ExtraTrees, logistic regression) with a logistic-regression meta-learner |
| Safeguard | IsolationForest out-of-distribution guard; alerts are suppressed on flagged windows |
| Runtime | scikit-learn / XGBoost; no GPU needed |
| Training script | `Nurse_Stress_Detector_training.py` |

## Intended use

- Research on wearable-based occupational stress detection.
- Prototyping a real-time stress dashboard or notification system for nurses, **with the informed consent of the people being monitored**.

## Out-of-scope use

- Medical diagnosis or any clinical decision.
- Employment, performance, scheduling or disciplinary decisions about individual workers.
- Covert monitoring, or monitoring without consent and clear data-governance rules.
- Populations, devices or settings very different from the training data (see below) without re-validation.

## Training data

Hosseini et al., *A multimodal sensor dataset for continuous stress detection of nurses in a hospital*, Scientific Data (2022), https://doi.org/10.1038/s41597-022-01361-y. Recordings from 15 nurses over about one week of regular shifts. The labels come from the dataset's stress annotations, which are tied to nurse self-reports; they are a proxy for stress, not a clinical measurement. Class balance is heavily skewed toward non-high-stress, and some subjects have very few or no HIGH windows.

## Evaluation

- Subject-held-out evaluation (leave-one-subject-out by default); each nurse is scored by a model that never saw them.
- Metrics: MCC (primary), AUPRC, AUROC, precision/recall/F1, balanced accuracy, calibration, per-subject results, and event-level metrics (event recall, false alerts per hour, detection delay).
- The full set of numbers is produced by the training script in `evaluation_metrics.zip` (`pooled_metrics.json`, `per_subject_metrics.csv`, plots).

Known optimism and variance:

- Hyperparameters were searched once on all subjects with grouped CV.
- The deployed threshold was chosen on pooled out-of-fold predictions.
- Windows overlap 75 %, so window-level counts are not independent.
- With only 15 subjects, per-subject performance varies widely and a few subjects dominate the pooled numbers. Report per-subject results alongside pooled ones.

## Limitations and risks

- **Small cohort**: 15 female nurses from one hospital during the COVID-19 period, wearing one device type. Generalization to other people, units, sensors or sampling rates is untested.
- **Label noise**: self-reported stress is subjective and sparse in time.
- **Needs calibration**: the first 7.5 minutes of each wearer's recording are used as a baseline, so the model assumes that period is representative of that person's resting state.
- **Motion and sensor artifacts** can mimic physiological arousal. The OOD guard helps but does not remove this.
- **Arousal is not distress**: EDA and heart rate also rise with exercise, caffeine, temperature and excitement.
- **False alerts and missed events**: the alert layer trades detection delay against false alarms (see event-level metrics). Do not treat an alert as proof of stress or the absence of an alert as proof of calm.
- **Privacy**: physiological streams are sensitive personal data. Store and transmit them securely, minimize retention, and make participation voluntary.

## Security note

Model artifacts (`.joblib`, `.pth`) are pickle-based. Only load files you trust.

## Contact

Syed Azmul Hasan, Jashore University of Science and Technology (JUST). Issues and questions: use the GitHub issue tracker.
