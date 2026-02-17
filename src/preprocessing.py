from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def list_raw_datasets(suffix: str = ".csv") -> Iterable[Path]:
    """Yield files inside the raw data directory that match the suffix."""
    return RAW_DATA_DIR.glob(f"*{suffix}")


def load_raw_dataset(filename: str, **read_kwargs: object) -> pd.DataFrame:
    """Load a raw dataset by filename from the raw directory."""
    path = RAW_DATA_DIR / filename
    if not path.exists():  # Guard early to surface missing files quickly.
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path, **read_kwargs)


def save_processed_dataset(df: pd.DataFrame, filename: str) -> Path:
    """Write a processed dataset to disk and return the path."""
    target = PROCESSED_DATA_DIR / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def train_validation_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/validation split when the target is binary."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def aggregate_patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level records into patient-level features."""
    if "patient_id" not in df.columns:
        raise KeyError("Column 'patient_id' is required for patient-level aggregation.")
    if "timestamp" not in df.columns:
        raise KeyError("Column 'timestamp' is required for patient-level aggregation.")

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    response = working.get("reminder_response_time")
    if response is not None:
        working["response_time_clean"] = response.replace(-1, np.nan)
    else:
        working["response_time_clean"] = np.nan

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        group_sorted = group.sort_values("timestamp")
        total_missed = group_sorted["future_non_adherence"].sum()
        missed_rate = group_sorted["future_non_adherence"].mean()

        response_times = group_sorted["response_time_clean"]
        avg_response_time = response_times.mean()
        std_response_time = response_times.std(ddof=0)

        count = len(group_sorted)
        if count > 1:
            idx = np.arange(count)
            adherence_trend = float(np.polyfit(idx, group_sorted["future_non_adherence"], 1)[0])
        else:
            adherence_trend = 0.0

        timestamps = group_sorted["timestamp"].dropna()
        if not timestamps.empty:
            days_active = int((timestamps.max() - timestamps.min()).days) + 1
            window_start = timestamps.max() - pd.Timedelta(days=7)
            last_window = group_sorted[group_sorted["timestamp"] >= window_start]
            last_7_days_missed = last_window["future_non_adherence"].sum()
        else:
            days_active = 0
            last_7_days_missed = 0

        target_value = int(group_sorted["future_non_adherence"].iloc[-1])

        return pd.Series(
            {
                "total_missed": float(total_missed),
                "missed_rate": float(missed_rate),
                "avg_response_time": float(avg_response_time) if not np.isnan(avg_response_time) else 0.0,
                "std_response_time": float(std_response_time) if not np.isnan(std_response_time) else 0.0,
                "adherence_trend": adherence_trend,
                "days_active": float(days_active),
                "last_7_days_missed": float(last_7_days_missed),
                "future_non_adherence": target_value,
            }
        )

    aggregated = working.groupby("patient_id", as_index=False).apply(_aggregate)
    aggregated = aggregated.reset_index(drop=True)
    return aggregated


def build_event_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Construct event-level dataset using cumulative history per patient."""
    required_cols = {"patient_id", "timestamp", "future_non_adherence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], errors="coerce")
    df_sorted = df_sorted.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    rows = []

    for patient_id, df_patient in df_sorted.groupby("patient_id"):
        df_patient = df_patient.reset_index(drop=True)

        cumulative_missed = 0.0
        cumulative_response = 0.0
        cumulative_response_sq = 0.0
        first_timestamp = df_patient.loc[0, "timestamp"]

        for i in range(len(df_patient)):
            current_row = df_patient.iloc[i]

            current_timestamp = current_row["timestamp"]
            if pd.isna(current_timestamp):
                # Skip rows without valid timestamps since history cannot be defined.
                continue

            if i > 0:
                history_span = current_timestamp - first_timestamp
                total_events = i
                missed_rate = cumulative_missed / total_events if total_events else 0.0
                avg_response = (
                    cumulative_response / total_events if total_events else 0.0
                )
                variance = (
                    cumulative_response_sq / total_events - avg_response**2
                    if total_events
                    else 0.0
                )
                std_response = np.sqrt(max(variance, 0.0))

                last_window_start = current_timestamp - pd.Timedelta(days=7)
                recent_history = df_patient.iloc[:i]
                recent_history = recent_history[recent_history["timestamp"] >= last_window_start]
                last_7_days_missed = recent_history["future_non_adherence"].sum()

                rows.append(
                    {
                        "patient_id": patient_id,
                        "event_index": i,
                        "total_missed": cumulative_missed,
                        "missed_rate": missed_rate,
                        "avg_response_time": avg_response,
                        "std_response_time": std_response,
                        "days_active": history_span.days + 1,
                        "last_7_days_missed": last_7_days_missed,
                        "future_non_adherence": current_row["future_non_adherence"],
                    }
                )

            # Update cumulative values after feature extraction
            missed_flag = current_row.get("future_non_adherence", 0)
            response_time = current_row.get("reminder_response_time", np.nan)
            if response_time == -1:
                response_time = np.nan

            cumulative_missed += missed_flag
            if not np.isnan(response_time):
                cumulative_response += response_time
                cumulative_response_sq += response_time**2

    return pd.DataFrame(rows)
