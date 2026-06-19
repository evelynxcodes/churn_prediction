"""
Evaluation utilities: metrics, confusion matrix, ROC, feature importance, SHAP.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_churn_distribution(y: pd.Series, save: bool = True):
    counts = y.value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["No Churn (0)", "Churn (1)"], counts.values, color=["#4c72b0", "#dd8452"])
    axes[0].set_title("Churn Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")
    axes[1].pie(
        counts.values, labels=["No Churn", "Churn"],
        autopct="%1.1f%%", colors=["#4c72b0", "#dd8452"], startangle=90
    )
    axes[1].set_title("Churn Rate")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(pipelines: dict, X_test, y_test, save: bool = True):
    n = len(pipelines)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()
    for ax, (name, pipe) in zip(axes, pipelines.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipe, X_test, y_test,
            display_labels=["No Churn", "Churn"],
            cmap="Blues", ax=ax,
        )
        ax.set_title(name)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(pipelines: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in pipelines.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_comparison(results: dict, save: bool = True):
    df = metrics_table(results)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df_plot = df[metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_plot))
    width = 0.15
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df_plot[metric], width, label=metric)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(pipeline, feature_names: list, model_name: str, top_n: int = 20, save: bool = True):
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        print(f"[{model_name}] Feature importance not available.")
        return

    prep = pipeline.named_steps["prep"]
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = feature_names

    imp_df = (
        pd.Series(importances, index=feat_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_df[::-1].plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save:
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURES_DIR / f"feature_importance_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_summary(pipeline, X_sample, model_name: str, save: bool = True):
    try:
        import shap
    except ImportError:
        print("SHAP not installed — skipping SHAP plot.")
        return

    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    X_transformed = prep.transform(X_sample)

    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feat_names)

    if hasattr(clf, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(clf)
        except Exception:
            explainer = shap.Explainer(clf, X_df)
    else:
        explainer = shap.LinearExplainer(clf, X_df)

    shap_values = explainer(X_df)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURES_DIR / f"shap_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
