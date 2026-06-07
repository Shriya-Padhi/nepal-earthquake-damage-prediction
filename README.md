# 🌏 Nepal Earthquake Damage Prediction

> Ranked **#210 out of 2,715** on DrivenData, predicting building damage from the 2015 Gorkha earthquake using LightGBM, neural geographic embeddings, and Bayesian hyperparameter optimization.

[![Competition](https://img.shields.io/badge/DrivenData-Richter's%20Predictor-1a6bb5?style=flat-square)](https://www.drivendata.org/competitions/57/nepal-earthquake/)
[![Rank](https://img.shields.io/badge/Rank-%23210%20%2F%202%2C715-gold?style=flat-square)]()
[![Score](https://img.shields.io/badge/Micro--F1-0.7519-brightgreen?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab?style=flat-square&logo=python&logoColor=white)]()
[![Framework](https://img.shields.io/badge/LightGBM%20%2B%20Keras-orange?style=flat-square)]()

---

## Overview

The 2015 Gorkha earthquake killed over 9,000 people and injured 22,000 more across Nepal. In response, the Nepali government conducted a large-scale household survey across affected districts, producing a dataset of **260,601 buildings** with 38 features covering location, construction materials, structural properties, and ownership.

This project predicts each building's **ordinal damage grade**:

| Grade | Meaning |
| :---: | :--- |
| 1 | Low damage |
| 2 | Medium damage |
| 3 | Near-total destruction |

**Competition result: Micro-F1 of 0.7519, ranked #210 out of 2,715 participants (top 8%).**

---

## Methodology

### 1. Exploratory Data Analysis

- Analyzed feature distributions (floor count, building age, land surface condition, plan configuration) across all three damage grades
- Confirmed **no missing values** in the dataset
- Identified significant **class imbalance** where damage grade 2 dominates, grades 1 and 3 are underrepresented

### 2. Feature Engineering

| Technique | Detail |
| :--- | :--- |
| **Volume feature** | `area_percentage × height_percentage` is more expressive of 3D building size than either dimension alone |
| **Winsorization** | Capped `age` and `volume` at the 95th percentile to suppress extreme outliers without dropping rows |
| **Feature pruning** | Removed low-signal secondary usage flags (`has_secondary_use_institution`, `_school`, `_industry`, `_health_post`, `_gov_office`, `_use_police`, `_other`) |
| **Geographic embedding** | Neural autoencoder compresses sparse geo IDs into dense 16-dim spatial representations (see below) |

### 3. Geographic Embedding: Key Feature

Nepal's three-level geographic hierarchy is highly cardinal: `geo_level_3_id` alone has **11,861 unique values**. Naive one-hot encoding produces an unmanageably sparse, high-dimensional feature space.

Instead, we train a **bottleneck autoencoder** that learns to compress `geo_level_3` into a 16-dimensional embedding by reconstructing the coarser levels (`geo_level_2`, `geo_level_1`). This forces the embedding to capture the hierarchical spatial structure of Nepal's administrative geography in a compact, dense form.

```
Input: geo_level_3  (11,861-dim one-hot)
           │
       Dense(16)        ← intermediate embedding layer
      ╱         ╲
Dense(geo2_dim)  Dense(geo1_dim)
  sigmoid           sigmoid
     │                 │
geo_level_2_hat   geo_level_1_hat
```

The 16 values from the intermediate layer replace all three raw geo ID columns in the final feature matrix.

> Inspired by [Goodsea/Richter-s-Eye](https://github.com/Goodsea/Richter-s-Eye).

### 4. Model Iteration

Each row represents a complete submission to the DrivenData leaderboard:

| Model | Key Additions | Micro-F1 |
| :--- | :--- | :---: |
| Logistic Regression | EDA baseline, PCA | 0.5150 |
| Random Forest | Grid search on benchmark code | 0.5815 |
| CatBoost | Visual feature engineering, feature importance | 0.7423 |
| XGBoost | Winsorization, GBDT + DART boosting | 0.7400 |
| LightGBM | Geo-embedding, class resampling | 0.7497 |
| **LightGBM + Optuna** | **5-fold CV, Bayesian hyperparameter tuning** | **0.7519** |

### 5. Hyperparameter Optimization with Optuna

We used **Optuna** (Tree-structured Parzen Estimator) instead of grid search with 50 trials per fold, each maximizing micro-F1 on the held-out split.

Search space:

| Parameter | Range |
| :--- | :--- |
| `num_leaves` | 10 – 500 |
| `max_depth` | 1 – 20 |
| `learning_rate` | 1e-8 – 1.0 (log-uniform) |
| `feature_fraction` | 0.1 – 1.0 |

### 6. Ensemble Strategy

Five LightGBM models (one per CV fold) are combined by **summing raw class probability scores** across all models before taking the argmax. This soft-voting approach is more robust than majority voting and benefits from the diversity of models trained on different data partitions.



## Repository Structure

```
nepal-earthquake-damage-prediction/
│
├── Richter_Predictor.ipynb          # Exploratory notebook with geo-embedding + LightGBM baseline
├── Richter_with_Resampling.py       # LightGBM pipeline with class resampling (F1: 0.7497)
├── Richter_with_PreProcessing.py    # Final pipeline — Optuna HPO + full preprocessing (F1: 0.7519)
├── Report.pdf                       # Full project report (NeurIPS format)
└── README.md
```

---

## Setup & Usage

All scripts were developed and run on **Google Colab** (GPU runtime recommended).

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/nepal-earthquake-damage-prediction.git
```

### 2. Download the data

Download the four CSV files from the [DrivenData competition page](https://www.drivendata.org/competitions/57/nepal-earthquake/data/) and upload them to your Google Drive.

```
MyDrive/
└── data/
    ├── train_values.csv
    ├── train_labels.csv
    ├── test_values.csv
    └── submission_format.csv
```

### 3. Update data paths

At the top of each script, update the file paths to match your Drive location:

```python
train_x = pd.read_csv("/content/drive/MyDrive/data/train_values.csv")
train_y = pd.read_csv("/content/drive/MyDrive/data/train_labels.csv")
test_x  = pd.read_csv("/content/drive/MyDrive/data/test_values.csv")
sub_csv = pd.read_csv("/content/drive/MyDrive/data/submission_format.csv")
```

### 4. Run

Open either script in Colab and run all cells. To reproduce the best score:

```
Richter_with_PreProcessing.py  →  F1: 0.7519  ← best result
Richter_with_Resampling.py     →  F1: 0.7497
```

---

## Dependencies

```
lightgbm
tensorflow
keras
scikit-learn
optuna
pandas
numpy
scipy
```

Install in Colab with:

```bash
pip install optuna lightgbm
```

All other dependencies come pre-installed in the Colab environment.

---

## Results

| Metric | Value |
| :--- | :---: |
| Evaluation metric | Micro-averaged F1 |
| Final leaderboard score | **0.7519** |
| Leaderboard rank | **#210 / 2,715** |
| Top percentile | **Top 8%** |
| Cross-validation | 5-Fold |
| Ensemble method | Soft-vote (probability sum) across 5 models |

---

## Dependencies

```
lightgbm>=3.3
tensorflow>=2.8
keras
scikit-learn>=1.0
optuna>=3.0
pandas>=1.4
numpy>=1.22
scipy>=1.8
```

```bash
pip install -r requirements.txt
```

---

## Team

Built as a course project for **CSCI 567: Machine Learning** at the **University of Southern California**.

| Name | USC Email |
| :--- | :--- |
| Shriya Padhi | spadhi@usc.edu |
| Shambhavi Sinha | sinhasha@usc.edu |
| Anushka Patil | anushkap@usc.edu |

---

## References

- Akiba et al. [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) (2019)
- Ke et al. [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) (NeurIPS 2017)
- Baran [Goodsea/Richter-s-Eye](https://github.com/Goodsea/Richter-s-Eye) geo-embedding architecture inspiration
- [DrivenData Richter's Predictor Competition](https://www.drivendata.org/competitions/57/nepal-earthquake/)
