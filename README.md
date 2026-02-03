# 🚢 Titanic Survival Prediction — End-to-End ML + Streamlit + Docker

An ML project that predicts Titanic passenger survival with a **full reproducible pipeline**:

EDA → Feature Engineering → Statistical feature checks → Optuna tuning (XGBoost) → **custom end-to-end pipeline (raw → processed → model)** → Streamlit app → Docker.

---

## 🔗 Live Demo
**Streamlit app:** https://titanic-ml-optuna.streamlit.app/

---

## ✨ Highlights
- ✅ Feature engineering + statistical validation:
  - **Categorical:** chi-square, survival-rate comparison
  - **Numerical:** Mann–Whitney U, Welch’s t-test, Cohen’s d
- ✅ Correlation filtering to reduce duplicated information:
  - Pearson / Spearman (numeric), Cramér’s V (categorical)
- ✅ Feature selection:
  - greedy selection + L1 (Lasso) + permutation importance
- ✅ Optuna hyperparameter tuning + saving best params and **optimal F1 threshold**
- ✅ Sanity check: **shuffle target → ROC-AUC ≈ 0.5** (leakage / “lucky split” check)
- ✅ Deployment:
  - Streamlit app + Docker + Docker Compose

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [How It Works (Step-by-Step)](#-how-it-works-step-by-step)
- [Run Locally](#️-run-locally)
- [Run with Docker](#-run-with-docker)
- [Artifacts](#-artifacts)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📖 Project Overview
**Goal:** build an end-to-end, reproducible ML pipeline to predict `Survived (0/1)`, covering the full path:
raw data → processing → training → artifact saving → inference → UI.

**Modeling approach:**
- Baseline for feature validation: **Logistic Regression**
- Final model: **XGBoost** (good speed/quality trade-off)

**Primary metric during feature work:** **ROC-AUC**  
**Final comparison metrics:** Accuracy / Precision / Recall / F1

**Evaluation setup:**
- Holdout split: `train_test_split(..., stratify=y, random_state=...)`
- Cross-validation: `StratifiedKFold` used with `cross_val_predict` to obtain out-of-fold predictions


---

## 🧱 Repository Structure
```text
.
├─ artifacts/
│  ├─ model_data/
│  └─ features.json
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ 01_EDA_feature_engineering.ipynb
│  ├─ 02_modelling_optuna.ipynb
│  └─ 03_check_pipelines_performance.ipynb
├─ src/
│  ├─ custom_pipeline.py
│  ├─ train_custom.py
│  └─ predict.py
├─ app.py
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
└─ .gitignore
Data folders:

data/raw — original raw dataset

data/processed — processed dataset after feature engineering

Artifacts:

artifacts/ — everything needed for reproducibility: feature metadata, best params, thresholds, etc.

📊 Results
> Metrics are reported using a stratified train/test split and validated with Stratified K-Fold cross-validation (via `cross_val_predict`) to preserve class distribution and reduce variance.

Pipeline Comparison
Full custom pipeline (raw → processed → model):

accuracy: 0.9057

precision: 0.8686

recall: 0.8889

f1: 0.8786

Standard pipeline (processed + standard preprocessing):

accuracy: 0.8945

precision: 0.8543

recall: 0.8743

f1: 0.8642

Difference (custom − standard):

accuracy: +0.0112

precision: +0.0143

recall: +0.0146

f1: +0.0145

✅ Conclusion: the full custom pipeline performs better, so it is kept as the final solution.

🧠 How It Works (Step-by-Step)
1) EDA + Feature Engineering + Feature Selection
Started with exploratory analysis to understand:

numeric/categorical distributions

separability between Survived=1 and Survived=0

Then performed deeper statistical checks (weak signal → iterate back into feature engineering):

Categorical checks

Chi-square test

Survival rate comparison

Numerical checks

Mann–Whitney U (robust to non-normality and outliers)

Welch’s t-test (different variances)

Cohen’s d (effect size)

Next: correlation filtering to remove redundant information:

Pearson / Spearman for numeric features

Cramér’s V for categorical features

Final selection combined a Logistic Regression baseline + ROC-AUC:

Greedy selection (add features one by one and keep only those improving the score)

L1 regularization (Lasso)

Permutation importance (important features reduce the metric when permuted; noisy features ≈ 0 or negative)

Outputs:

processed dataset → data/processed/

feature metadata → artifacts/features.json

2) Modeling + Optuna (XGBoost)
tested multiple models and selected XGBoost

ran a sanity check: shuffle target → expected ROC-AUC ≈ 0.5

if it stays high, it may indicate leakage/bug/overfitting

Optuna tuning → saved best parameters and tuning results

3) Full Custom Pipeline (raw → processed → model)
The “standard” approach trained only on the already processed dataset.
This project also includes a full custom pipeline that:

takes raw input

performs feature engineering inside the pipeline

trains the model using the best Optuna parameters

Files:

src/custom_pipeline.py — pipeline assembly

src/train_custom.py — training the full pipeline

notebooks/03_check_pipelines_performance.ipynb — pipeline comparison

▶️ Run Locally
Install dependencies
pip install -r requirements.txt
# dev dependencies (optional)
pip install -r requirements-dev.txt
Run Streamlit
streamlit run app.py
🐳 Run with Docker
Build
docker build -t titanic-streamlit .
Run
docker run --rm -p 8501:8501 titanic-streamlit
Or with Docker Compose
docker compose up --build
Open:

http://localhost:8501

📦 Artifacts
Stored in artifacts/model_data/:

best_params.json — best Optuna hyperparameters

best_score.json — best objective score (e.g., best_roc_auc_value)

threshold.json — threshold_f1 (threshold that maximizes F1; not fixed at 0.5)

Also:

artifacts/features.json — final selected features + metadata

🛣 Roadmap
✅ Add Streamlit demo link — done

📄 License
MIT — see LICENSE