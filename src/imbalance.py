"""Utilities for handling class imbalance strategies."""
from __future__ import annotations

from typing import Dict

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from .config import RANDOM_SEED


def build_resampling_strategies(random_state: int | None = RANDOM_SEED) -> Dict[str, object]:
    """Return the default resampling strategies used across experiments."""
    return {
        "ROS": RandomOverSampler(random_state=random_state),
        "RUS": RandomUnderSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
    }


def calculate_scale_pos_weight(y) -> float:
    """Compute scale_pos_weight for XGBoost based on class imbalance."""
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y
    negative = (y_arr == 0).sum()
    positive = (y_arr == 1).sum()
    if positive == 0:
        raise ValueError("Cannot compute scale_pos_weight: no positive samples.")
    return negative / positive


def get_weighted_xgb(scale_pos_weight: float, random_state: int | None = RANDOM_SEED) -> XGBClassifier:
    """Return an XGBoost model configured with scale_pos_weight."""
    return XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to the training data only."""
    sm = SMOTE(random_state=RANDOM_SEED)
    return sm.fit_resample(X_train, y_train)
