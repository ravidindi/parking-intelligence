"""Train and export the obstruction risk prediction model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import build_hotspot_risk_table, build_ml_training_table
from src.preprocessing import load_and_clean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "jan to may police violation_anonymized791b166.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "police_violation"
MODEL_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "event_frequency",
    "avg_duration",
    "vehicle_types",
    "wrong_parking_count",
    "no_parking_count",
    "main_road_count",
    "footpath_count",
    "double_parking_count",
    "crossing_count",
    "bus_stop_count",
    "traffic_light_count",
    "congestion_violation_count",
    "two_wheeler_count",
    "auto_count",
    "car_count",
    "heavy_vehicle_count",
    "approved_count",
    "sent_to_scita_count",
    "road_closure_count",
    "unique_devices",
    "peak_hour",
    "parking_surge_hour",
    "weekend",
    "vehicle_impact",
]


def train(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train an XGBoost regressor and write model artifacts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    training_path = model_dir / "training_features.csv"
    if training_path.exists():
        ml_df = pd.read_csv(training_path)
        df = None
    else:
        df = load_and_clean(data_path)
        ml_df = build_ml_training_table(df)
    if ml_df.empty:
        raise ValueError("No obstruction rows with junction/zone/hour were available for training.")

    X = _model_matrix(ml_df)
    y = ml_df["risk_score"]

    train_test_split, mean_absolute_error, XGBRegressor, joblib = _load_training_dependencies()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))

    joblib.dump(model, model_dir / "risk_model.pkl")
    joblib.dump(list(X.columns), model_dir / "model_columns.pkl")

    lookup_path = model_dir / "risk_lookup.csv"
    if not lookup_path.exists():
        if df is None:
            df = load_and_clean(data_path)
        hotspots = build_hotspot_risk_table(df).sort_values("risk_score", ascending=False)
        hotspots.to_csv(lookup_path, index=False)

    metrics = {
        "mae": mae,
        "rows": int(len(ml_df)),
        "features": int(X.shape[1]),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }
    (model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _model_matrix(ml_df: pd.DataFrame) -> pd.DataFrame:
    """Keep training compact by using numeric engineered signals, not one column per station."""
    features = ml_df.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
    return features.apply(pd.to_numeric, errors="coerce").fillna(0)


def _load_training_dependencies():
    try:
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor
        import joblib
    except ImportError as exc:
        raise ImportError(
            "Training requires scikit-learn, xgboost, and joblib. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return train_test_split, mean_absolute_error, XGBRegressor, joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the parking obstruction risk model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to the police violation CSV")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory for model artifacts")
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
