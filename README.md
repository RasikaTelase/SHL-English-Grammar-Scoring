# SHL English Grammar Scoring Engine

An end-to-end multimodal machine learning system for predicting continuous English grammar proficiency scores (0.0 to 5.0) from spoken audio recordings and speech transcripts.

---

## Executive Summary & Final Audit Metrics

| Metric | Target / Benchmark | Final Model Result | Status |
| :--- | :--- | :--- | :--- |
| **Training Dataset** | 769 WAV audio samples | 769 samples processed | Verified |
| **Test Dataset** | 216 WAV audio samples | 216 samples predicted | Verified |
| **Primary Evaluated Model** | Extra Trees Regressor | 100 estimators, max_depth=8 | Selected |
| **Compulsory Training RMSE** | Evaluated on train split | **0.3761** | Verified |
| **Validation RMSE (80/20 Split)** | Held-out validation split | **0.6838** | Verified |
| **Validation Pearson Correlation ($r$)** | Held-out validation split | **0.8660** | Verified |
| **5-Fold Cross Validation RMSE** | Out-of-fold predictions | **0.7545 ± 0.0450** | Verified |
| **5-Fold CV Pearson Correlation ($r$)**| Out-of-fold predictions | **0.7946** | Verified |
| **Prediction Score Range** | [0.0, 5.0] target score range | **[2.3842, 4.6017]** | Clipped & Validated |

---

## Project Structure & Architecture

```text
AI-Grammar-Scoring/
├── artifacts/
│   ├── features/
│   │   ├── train_audio_features.csv    # 769 rows × 155 cols (Librosa spectral/MFCC)
│   │   ├── test_audio_features.csv     # 216 rows × 155 cols
│   │   ├── train_text_features.csv     # 769 rows × 41 cols (Readability, length, TTR)
│   │   ├── test_text_features.csv      # 216 rows × 41 cols
│   │   ├── train_grammar_features.csv  # 769 rows × 27 cols (Fluency alignment, heuristics)
│   │   ├── test_grammar_features.csv   # 216 rows × 27 cols
│   │   ├── train_tfidf_features.csv    # 769 rows × 101 cols (100 TF-IDF n-grams)
│   │   └── test_tfidf_features.csv     # 216 rows × 101 cols
│   ├── transcripts/
│   │   ├── train_transcripts.csv       # 769 Whisper tiny.en transcripts
│   │   └── test_transcripts.csv        # 216 Whisper tiny.en transcripts
│   └── predictions/
│       └── test_predictions.csv        # 216 test predictions matching test.csv
├── data/
│   ├── train.csv                       # 769 rows (filename, label)
│   ├── test.csv                        # 216 rows (filename, placeholder label -1)
│   └── sample_submission.csv           # 204 rows (Legacy template file)
├── models/
│   ├── best_model.joblib               # Fitted scikit-learn Extra Trees Pipeline
│   └── tfidf_vectorizer.joblib         # Fitted TF-IDF Vectorizer (trained on train set only)
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Step 1: Exploratory Data Analysis
│   ├── 02_audio_features.ipynb        # Step 2: Audio Signal Processing & Feature Extraction
│   ├── 03_transcription.ipynb         # Step 3: Whisper ASR Speech Transcription
│   ├── 04_text_features.ipynb          # Step 4: Text & Linguistic Feature Extraction
│   ├── 05_grammar_features.ipynb       # Step 5: Grammar & Fluency Indicators
│   └── 06_model_training.ipynb        # Step 6: Multimodal Training & Evaluation
└── src/
    ├── audio_features.py               # Audio extraction functions
    ├── transcription.py                # Whisper ASR transcription logic
    ├── text_features.py                 # Text & TF-IDF extraction logic
    ├── grammar_features.py              # Grammar heuristics & fluency alignment
    └── train.py                        # Model training, CV, evaluation & pipeline export
```

---

## Multimodal Pipeline Workflow (Steps 1–7)

1. **Step 1: Dataset Inspection & Setup**
   - Verified 769 training audio samples and 216 test audio samples (mono, 16 kHz PCM WAV, ~35–60s duration). Target labels range from 0.0 to 5.0.

2. **Step 2: Audio Feature Extraction (`src/audio_features.py`)**
   - Extracted 153 acoustic indicators per sample: 20 MFCC means & standard deviations, 7 spectral contrast bands, spectral centroid, rolloff, bandwidth, pitch (F0 statistics), RMS energy, and pause statistics (`duration_sec`, `speech_duration_ratio`, `silence_duration_ratio`, `pause_count`, `mean_pause_duration_sec`).

3. **Step 3: Speech Transcription (`src/transcription.py`)**
   - Transcribed all audio using OpenAI Whisper (`tiny.en`).
   - One sample ([`audio_5055.wav`](file:///d:/AI-Grammar-Scoring/artifacts/transcripts/train_transcripts.csv#L19)) failed transcription due to silent audio; it was preserved at index 17 with explicit status `FAILED_TRANSCRIPTION_AUDIO_5055` without dropping data.

4. **Step 4: Text & Linguistic Feature Extraction (`src/text_features.py`)**
   - Extracted 40 dense text metrics (word count, sentence count, TTR, Hapax Legomena, readability indices, POS ratios, punctuation stats) and 100 word/character TF-IDF n-grams (fitted strictly on training transcripts).

5. **Step 5: Grammar-Specific Features & Fluency Indicators (`src/grammar_features.py`)**
   - Audited external tool availability (`language_tool_python` requires Java, `spacy` model missing). Implemented defensible rule-based heuristics for subject-verb agreement mismatches, double negatives, article misuse, sentence fragments, conjunction ratios, and speech-rate alignment (`speech_rate_wpm`, `articulation_rate_wps`, `pause_per_word_ratio`).

6. **Step 6: Feature Combination, Training & Evaluation (`src/train.py`)**
   - Combined 312 numeric features.
   - Evaluated 6 regression baselines. Selected `ExtraTreesRegressor` (100 estimators, max_depth=8) inside a leakage-free `Pipeline` (`SimpleImputer` + `StandardScaler`).
   - Evaluated 5-Fold Cross-Validation (**CV RMSE = 0.7545**, **CV Pearson $r$ = 0.7946**).
   - Exported model pipeline to `models/best_model.joblib` and predictions for 216 test samples to `artifacts/predictions/test_predictions.csv`.

7. **Step 7: Final Audit & Readiness**
   - Audited all 216 test predictions: 0 NaNs, 0 Infs, score range [2.3842, 4.6017].
   - Verified 204 vs 216 row sample submission discrepancy.

---

## Regression Model Benchmarking

### 1. Model Comparison (80/20 Train/Validation Split)

| Model Architecture | Training RMSE | Validation RMSE | Validation Pearson $r$ |
| :--- | :--- | :--- | :--- |
| **Extra Trees Regressor** | **0.3761** | **0.6838** | **0.8660** |
| **Gradient Boosting Regressor** | 0.3324 | 0.6847 | 0.8658 |
| **HistGradientBoosting Regressor** | 0.3993 | 0.6932 | 0.8614 |
| **Random Forest Regressor** | 0.4003 | 0.7217 | 0.8479 |
| **Ridge Regression (L2)** | 0.4917 | 0.9225 | 0.8110 |
| **Dummy Regressor (Mean Baseline)** | 1.2062 | 1.3596 | 0.0000 |

### 2. Feature Subset Ablation (5-Fold Cross Validation)

| Feature Configuration | Feature Count | 5-Fold CV RMSE | 5-Fold CV Pearson $r$ |
| :--- | :--- | :--- | :--- |
| **Audio Only** | 152 | 0.7945 ± 0.0446 | 0.7688 |
| **Text + Grammar Only** | 61 | 0.8533 ± 0.0136 | 0.7257 |
| **Audio + Text + Grammar** | 212 | 0.7615 ± 0.0443 | 0.7900 |
| **Audio + Text + Grammar + TF-IDF** | **312** | **0.7545 ± 0.0450** | **0.7946** |

---

## Sample Submission Discrepancy Findings

- **`test.csv`**: Contains **216 rows**, matching the 216 audio WAV files present in `Dataset_Final/test/`.
- **`sample_submission.csv`**: Contains **204 rows**, with filenames comprising 105 training samples, 25 test samples, and 99 unaligned filenames.
- **Conclusion**: `sample_submission.csv` is an outdated/unaligned legacy template file. The true evaluation test set consists of the 216 filenames in `test.csv`. Intermediate test predictions have been saved to `artifacts/predictions/test_predictions.csv` (216 rows).

---

## Reproducibility & Execution

To reproduce the model training and prediction pipeline:

```bash
# 1. Activate environment
source .venv/bin/activate  # Or D:\Drone_AI_Brain\venv\Scripts\activate on Windows

# 2. Run feature combination and training test script
python scratch/run_training_test.py

# 3. Execute main training notebook
python scratch/execute_notebook.py
```
