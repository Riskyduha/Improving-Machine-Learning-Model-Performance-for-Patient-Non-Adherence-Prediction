"""Model registry and factory helpers."""
from __future__ import annotations

from typing import Dict, Iterable

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import RANDOM_SEED


def build_baseline_models(random_state: int | None = RANDOM_SEED) -> Dict[str, object]:
    """Return a dictionary of baseline models ready for training."""
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }


def get_model_names(models: Dict[str, object]) -> Iterable[str]:
    """Return sorted model names for reporting."""
    return sorted(models.keys())


def get_baseline_xgb(random_state: int | None = RANDOM_SEED) -> XGBClassifier:
    """Return an XGBoost baseline classifier."""
    return XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
    )
