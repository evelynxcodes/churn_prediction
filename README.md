# Customer Churn Prediction

End-to-end machine learning project predicting telecom customer churn — built for a data science portfolio.

## Overview

| Item | Detail |
|---|---|
| **Dataset** | Synthetic Telco-style (7,043 customers, 20 features) |
| **Task** | Binary classification — will a customer churn? |
| **Best model** | Logistic Regression — AUC-ROC **0.792** |
| **Models compared** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM |

## Project Structure

```
ds_try/
├── data/                          # Generated CSV dataset
│   └── churn_data.csv
├── notebooks/
│   └── churn_prediction_analysis.ipynb   # Full analysis notebook
├── src/
│   ├── data_generator.py          # Synthetic dataset generator
│   ├── preprocessing.py           # Sklearn preprocessing pipeline
│   ├── models.py                  # Model definitions
│   └── evaluation.py              # Metrics, plots, SHAP
├── outputs/
│   ├── figures/                   # All saved plots
│   └── model_comparison.csv       # Metrics table
├── train_pipeline.py              # Main end-to-end script
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python train_pipeline.py
```

Or open the notebook for the full visual walkthrough:

```bash
jupyter notebook notebooks/churn_prediction_analysis.ipynb
```

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.706 | 0.491 | 0.772 | 0.601 | **0.792** |
| Random Forest | 0.699 | 0.485 | 0.775 | 0.596 | 0.788 |
| Gradient Boosting | 0.758 | 0.626 | 0.389 | 0.479 | 0.783 |
| Decision Tree | 0.700 | 0.485 | 0.738 | 0.585 | 0.781 |
| XGBoost | 0.686 | 0.468 | 0.710 | 0.564 | 0.765 |
| LightGBM | 0.701 | 0.485 | 0.686 | 0.568 | 0.761 |

> **Note:** ROC-AUC is the primary metric because the classes are imbalanced (~29% churn rate).

## Key Churn Drivers

1. **Month-to-month contract** — highest churn risk
2. **Short tenure (< 12 months)** — new customers are most vulnerable
3. **Fiber optic internet** — elevated churn vs DSL
4. **Electronic check payment** — correlated with churn
5. **No online security / tech support** — reduces stickiness

## Pipeline Highlights

- **Preprocessing:** `ColumnTransformer` with `OneHotEncoder` + `StandardScaler` wrapped in a sklearn `Pipeline`
- **Class imbalance:** handled via `class_weight="balanced"` in each model
- **Evaluation:** confusion matrix, ROC curves, classification report, 5-fold cross-validation
- **Interpretability:** feature importance + SHAP summary plots for the best model
- **Inference:** `joblib`-serialised pipeline for single-call scoring on new customers
