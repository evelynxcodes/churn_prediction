"""
Model definitions and training helpers.
Each entry in MODELS is a (name, estimator) pair that slots into a sklearn Pipeline
after the preprocessor.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


def get_base_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
            use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=42,
        )
    if _HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, subsample=0.8, class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1,
        )
    return models


def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor), ("clf", estimator)])
