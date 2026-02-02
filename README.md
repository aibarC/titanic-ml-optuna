
# 🚢 Titanic Survival Prediction — End-to-End ML + Streamlit + Docker

ML-проект для предсказания выживаемости пассажиров Titanic с **полным пайплайном**:  
EDA → Feature Engineering → статистическая проверка фич → Optuna tuning (XGBoost) → **full custom pipeline (raw → processed → model)** → Streamlit app → Docker.

---
The link to see the result:
[titanic->predict survivability](https://titanic-ml-optuna.streamlit.app/)

## ✨ Highlights
- ✅ Feature engineering с проверками (chi-square, Mann–Whitney U, Welch t-test, Cohen’s d)
- ✅ Correlation filtering (Pearson/Spearman, Cramér’s V) для удаления дубликатов информации
- ✅ Feature selection: greedy + L1 (Lasso) + permutation importance
- ✅ Optuna tuning + сохранение лучших параметров и порога под F1
- ✅ Sanity check (shuffle target → метрика ~0.5) против утечек/случайной удачи
- ✅ Streamlit + Docker/Docker Compose для запуска приложения

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [How It Works (Step-by-Step)](#-how-it-works-step-by-step)
- [Run Locally](#-run-locally)
- [Run with Docker](#-run-with-docker)
- [Artifacts](#-artifacts)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📖 Project Overview
Цель: построить воспроизводимый ML-пайплайн, который предсказывает `Survived (0/1)` и включает полный путь:  
данные → обработка → обучение → сохранение артефактов → предикт → UI.

**Modeling approach:**
- Baseline для проверки фич: **Logistic Regression**
- Финальная модель: **XGBoost** (скорость + качество)

**Primary metric during feature work:** **ROC-AUC**  
**Final comparison metrics:** Accuracy / Precision / Recall / F1

---

## 🗂 Repository Structure
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
````

**Data folders:**

* `data/raw` — исходные данные (без обработки)
* `data/processed` — обработанные данные после feature engineering

**Artifacts:**

* `artifacts/` — всё для воспроизводимости: метаданные фич, лучшие параметры, пороги и т.п.

---

## 📊 Results

### Pipeline Comparison

**Full custom pipeline (raw → processed → model):**

* accuracy: **0.9057**
* precision: **0.8686**
* recall: **0.8889**
* f1: **0.8786**

**Standard pipeline (processed + standard preprocessing):**

* accuracy: **0.8945**
* precision: **0.8543**
* recall: **0.8743**
* f1: **0.8642**

**Difference (custom - standard):**

* accuracy: **+0.0112**
* precision: **+0.0143**
* recall: **+0.0146**
* f1: **+0.0145**

✅ Итог: **Full custom pipeline лучше**, поэтому он сохранён как финальный.

---

## 🧠 How It Works (Step-by-Step)

### 1) EDA + Feature Engineering + Feature Selection

Сначала — быстрый визуальный EDA, чтобы понять:

* распределения числовых/категориальных фич
* различимость между `Survived=1` и `Survived=0`

Потом — углублённая статистическая проверка (если слабый сигнал → возвращался в feature engineering):

**Categorical tests**

* Chi-square
* Survival rate comparison

**Numerical tests**

* Mann–Whitney U (не требует нормальности, устойчив к outliers/skew)
* Welch t-test (сравнение средних при разных дисперсиях)
* Cohen’s d (размер эффекта)

Далее корреляции и удаление дубликатов:

* Pearson / Spearman (числовые)
* Cramér’s V (категориальные)

Финальный отбор фич делался через baseline **Logistic Regression** + **ROC-AUC**:

* Greedy selection (добавлял по одной фиче, оставлял только те, что улучшают)
* L1 regularization (Lasso)
* Permutation importance (важные фичи сильно “роняют” метрику при перемешивании; шумовые ≈ 0 или отрицательный вклад)

**Outputs:**

* processed dataset → `data/processed/`
* метаданные фич → `artifacts/features.json`

---

### 2) Modeling + Optuna (XGBoost)

* протестировал несколько моделей и выбрал **XGBoost**
* сделал sanity check: **shuffle target → ожидаемо ~0.5**
  (если не ~0.5 — возможна утечка/переподгон/удача)
* Optuna tuning → сохранил лучшие параметры и результаты

---

### 3) Full Custom Pipeline (raw → processed → model)

Так как “стандартный” пайплайн работал уже на processed, я сделал **full custom pipeline**, который:

1. принимает raw
2. превращает в processed внутри пайплайна
3. обучает модель на best params

Файлы:

* `src/custom_pipeline.py` — сборка пайплайна
* `src/train_custom.py` — обучение full pipeline
* `notebooks/03_check_pipelines_performance.ipynb` — сравнение пайплайнов

---

## ▶️ Run Locally

### Install dependencies

```bash
pip install -r requirements.txt
# dev dependencies (optional)
pip install -r requirements-dev.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### Build

```bash
docker build -t titanic-streamlit .
```

### Run

```bash
docker run --rm -p 8501:8501 titanic-streamlit
```

### Or with Docker Compose

```bash
docker compose up --build
```

Open:

* [http://localhost:8501](http://localhost:8501)

---

## 📦 Artifacts

Файлы в `artifacts/model_data/`:

* `best_params.json` — лучшие гиперпараметры Optuna
* `best_score.json` — лучший objective score Optuna (`best_roc_auc_value`)
* `threshold.json` — `threshold_f1` (порог, который максимизирует F1; не просто 0.5)

Также:

* `artifacts/features.json` — финальные метаданные/набор фич

---

## 🛣 Roadmap

> добавить демо ссылку Streamlit: **[TODO]** ```[done]```

---

## 📄 License

MIT — see `LICENSE`

```
```
