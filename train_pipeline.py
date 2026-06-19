"""
Customer Churn Prediction — end-to-end training pipeline.

Usage:
    python train_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from pathlib import Path

from src.data_generator import generate_churn_data
from src.preprocessing import build_preprocessor, split_data
from src.models import build_pipeline, get_base_models
from src.evaluation import (
    compute_metrics,
    metrics_table,
    plot_churn_distribution,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_feature_importance,
    plot_shap_summary,
)

MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/churn_data.csv")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  Customer Churn Prediction Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n[1/6] Generating / loading data ...")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"  Loaded {len(df):,} rows from {DATA_PATH}")
    else:
        df = generate_churn_data(n_samples=7043, random_state=42)
        df.to_csv(DATA_PATH, index=False)
        print(f"  Generated {len(df):,} rows → saved to {DATA_PATH}")

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"  Churn rate: {churn_rate:.1%}")

    # 2. Split
    print("\n[2/6] Splitting data (70 / 10 / 20) ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # 3. Build preprocessor + model pipelines
    print("\n[3/6] Building pipelines ...")
    preprocessor = build_preprocessor()
    base_models = get_base_models()
    pipelines = {
        name: build_pipeline(build_preprocessor(), est)
        for name, est in base_models.items()
    }
    print(f"  Models: {list(pipelines.keys())}")

    # 4. Train
    print("\n[4/6] Training ...")
    results = {}
    for name, pipe in pipelines.items():
        print(f"  → {name} ...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_prob)
        print(f"AUC={results[name]['ROC-AUC']:.4f}")

    # 5. Results table
    print("\n[5/6] Results on test set:")
    print()
    df_results = metrics_table(results)
    print(df_results.to_string())
    df_results.to_csv("outputs/model_comparison.csv")

    best_name = df_results.index[0]
    best_pipe = pipelines[best_name]
    print(f"\n  Best model: {best_name}  (AUC={df_results.loc[best_name, 'ROC-AUC']:.4f})")
    joblib.dump(best_pipe, MODELS_DIR / "best_model.pkl")
    print(f"  Saved → {MODELS_DIR}/best_model.pkl")

    # 6. Plots
    print("\n[6/6] Generating plots ...")
    plot_churn_distribution(y_test, save=True)
    plot_roc_curves(pipelines, X_test, y_test, save=True)
    plot_confusion_matrices(pipelines, X_test, y_test, save=True)
    plot_metrics_comparison(results, save=True)
    plot_feature_importance(best_pipe, list(X_train.columns), best_name, save=True)

    # SHAP on a 500-row sample for speed
    print("  Computing SHAP values (sample) ...")
    X_shap = X_test.sample(min(500, len(X_test)), random_state=42)
    plot_shap_summary(best_pipe, X_shap, best_name, save=True)

    print("\nDone! All outputs written to outputs/")


if __name__ == "__main__":
    main()
