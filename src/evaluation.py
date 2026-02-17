from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_SEED

def compute_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[float] | None = None,
) -> Dict[str, float]:
    """Compute baseline metrics required by the experiments."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    report = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
    }
    if y_prob is not None:
        report["auc"] = roc_auc_score(y_true_arr, np.asarray(list(y_prob)))
    return report


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """Stack fold metrics and append mean/std rows for quick inspection."""
    df = pd.DataFrame(fold_metrics)
    summary = df.agg(["mean", "std"]).rename(index={"mean": "Mean", "std": "Std"})
    return pd.concat([df, summary])


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate a fitted model using probability thresholding."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement predict_proba for evaluation.")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def threshold_search(model, X_test, y_test):
    """Sweep probability thresholds to inspect recall/precision trade-offs."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement predict_proba for threshold search.")

    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    y_prob = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results.append(
            {
                "threshold": float(t),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    return results


def stratified_kfold_evaluation(
    model,
    X,
    y,
    n_splits: int = 10,
    preprocess=None,
    threshold: float = 0.5,
):
    """Run stratified CV and return average metrics across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results: List[Dict[str, float]] = []

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        if preprocess is not None:
            preproc = clone(preprocess)
            preproc.fit(X_train_fold, y_train_fold)
            X_train_proc = preproc.transform(X_train_fold)
            X_val_proc = preproc.transform(X_val_fold)
        else:
            X_train_proc = X_train_fold
            X_val_proc = X_val_fold

        model_clone = clone(model)
        model_clone.fit(X_train_proc, y_train_fold)

        y_prob = model_clone.predict_proba(X_val_proc)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        fold_results.append(
            {
                "accuracy": accuracy_score(y_val_fold, y_pred),
                "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                "f1": f1_score(y_val_fold, y_pred, zero_division=0),
                "auc": roc_auc_score(y_val_fold, y_prob),
            }
        )

    results_df = pd.DataFrame(fold_results)
    mean_results = results_df.mean().to_dict()
    return {metric: float(value) for metric, value in mean_results.items()}


def stratified_kfold_with_smote(
    model,
    X,
    y,
    n_splits: int = 10,
    preprocess=None,
    threshold: float = 0.5,
):
    """Run stratified CV applying SMOTE within each training fold."""
    from .imbalance import apply_smote

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results: List[Dict[str, float]] = []

    for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        if preprocess is not None:
            preproc = clone(preprocess)
            preproc.fit(X_train_fold, y_train_fold)
            X_train_proc = preproc.transform(X_train_fold)
            X_val_proc = preproc.transform(X_val_fold)
        else:
            X_train_proc = X_train_fold
            X_val_proc = X_val_fold

        X_res, y_res = apply_smote(X_train_proc, y_train_fold)
        model_clone = clone(model)
        model_clone.fit(X_res, y_res)

        y_prob = model_clone.predict_proba(X_val_proc)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        fold_results.append(
            {
                "accuracy": accuracy_score(y_val_fold, y_pred),
                "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                "f1": f1_score(y_val_fold, y_pred, zero_division=0),
                "auc": roc_auc_score(y_val_fold, y_prob),
            }
        )

    results_df = pd.DataFrame(fold_results)
    mean_results = results_df.mean().to_dict()
    return {metric: float(value) for metric, value in mean_results.items()}
