from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config import *  # noqa: F403,F401
from src.evaluation import (
    evaluate_model,
    stratified_kfold_evaluation,
    stratified_kfold_with_smote,
    threshold_search,
)
from src.imbalance import apply_smote, calculate_scale_pos_weight, get_weighted_xgb
from src.models import get_baseline_xgb
from src.preprocessing import (
    build_event_level_dataset,
    build_preprocessing_pipeline,
)


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from disk."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Split dataset into stratified train and test sets."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    df = load_data("data/raw/medication_adherence.csv")
    df = build_event_level_dataset(df)
    X_train, X_test, y_train, y_test = split_data(df)

    id_columns = [col for col in ["patient_id"] if col in X_train.columns]
    if id_columns:
        X_train = X_train.drop(columns=id_columns)
        X_test = X_test.drop(columns=id_columns)

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]
    categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
    preprocess = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    baseline_model = get_baseline_xgb()
    baseline_cv = stratified_kfold_evaluation(
        baseline_model,
        X_train,
        y_train,
        n_splits=10,
        preprocess=preprocess,
    )
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocess)),
            ("model", get_baseline_xgb()),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_test = evaluate_model(baseline_pipeline, X_test, y_test)

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    weighted_model = get_weighted_xgb(scale_pos_weight)
    weighted_cv = stratified_kfold_evaluation(
        weighted_model,
        X_train,
        y_train,
        n_splits=10,
        preprocess=preprocess,
    )
    weighted_pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocess)),
            ("model", get_weighted_xgb(scale_pos_weight)),
        ]
    )
    weighted_pipeline.fit(X_train, y_train)
    weighted_test = evaluate_model(weighted_pipeline, X_test, y_test)

    threshold_results = threshold_search(weighted_pipeline, X_test, y_test)

    smote_preprocess = clone(preprocess)
    smote_preprocess.fit(X_train, y_train)
    X_train_processed = smote_preprocess.transform(X_train)
    X_res, y_res = apply_smote(X_train_processed, y_train)
    smote_model = get_baseline_xgb()
    smote_model.fit(X_res, y_res)
    X_test_processed = smote_preprocess.transform(X_test)
    smote_test = evaluate_model(smote_model, X_test_processed, y_test)

    smote_cv = stratified_kfold_with_smote(
        get_baseline_xgb(),
        X_train,
        y_train,
        n_splits=10,
        preprocess=preprocess,
    )

    cv_scores = {
        "baseline": baseline_cv,
        "weighted": weighted_cv,
        "smote": smote_cv,
    }
    best_model_name = max(cv_scores.items(), key=lambda item: item[1]["f1"])[0]

    if best_model_name == "baseline":
        best_test_results = baseline_test
    elif best_model_name == "weighted":
        best_test_results = weighted_test
    else:
        best_test_results = smote_test

    all_results = [
        {"model": "baseline_cv", **baseline_cv},
        {"model": "baseline_test", **baseline_test},
        {"model": "weighted_cv", **weighted_cv},
        {"model": "weighted_test", **weighted_test},
        {"model": "smote_cv", **smote_cv},
        {"model": "smote_test", **smote_test},
        {"model": f"best_model_test_{best_model_name}", **best_test_results},
    ]

    print("Baseline CV:", baseline_cv)
    print("Baseline Test:", baseline_test)
    print("Weighted CV:", weighted_cv)
    print("Weighted Test:", weighted_test)
    print("SMOTE CV:", smote_cv)
    print("SMOTE Test:", smote_test)
    print("Threshold tuning sample:", threshold_results[:5])
    print(f"Best model by CV F1: {best_model_name}")
    print("Best model test metrics:", best_test_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/experiment_results.csv", index=False)


if __name__ == "__main__":
    main()
