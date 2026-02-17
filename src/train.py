"""Training orchestration helpers."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED
from .evaluation import compute_classification_report


def run_cross_validation(
    models: Dict[str, object],
    X,
    y,
    preprocess=None,
    n_splits: int = 5,
    random_state: int | None = RANDOM_SEED,
) -> Dict[str, List[Dict[str, float]]]:
    """Train supplied models using stratified CV and return fold-level metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, List[Dict[str, float]]] = {}

    for name, model in models.items():
        results[name] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            steps = []
            if preprocess is not None:
                steps.append(("preprocess", preprocess))
            steps.append(("model", model))
            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
            metrics = compute_classification_report(y_val, y_pred, y_prob)
            results[name].append(metrics)
    return results


def summarize_results(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean and std per metric for each model."""
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, folds in results.items():
        if not folds:
            continue
        keys = folds[0].keys()
        summary[name] = {}
        for key in keys:
            values = np.array([fold[key] for fold in folds], dtype=float)
            summary[name][key] = (float(values.mean()), float(values.std(ddof=1)))
    return summary
