"""Feature engineering, risk scoring, and recommendations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing import load_and_clean


PEAK_HOURS = set(range(8, 12)) | set(range(17, 22))
PARKING_SURGE_HOURS = set(range(0, 7)) | set(range(19, 24))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "jan to may police violation_anonymized791b166.csv"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "models" / "police_violation"


@dataclass(frozen=True)
class RiskThresholds:
    high: float = 70.0
    medium: float = 40.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day, month, and peak-hour fields from start_datetime."""
    featured = df.copy()
    if "start_datetime" not in featured.columns:
        raise ValueError("Dataset must contain start_datetime")

    featured["hour"] = featured["start_datetime"].dt.hour
    featured["day_of_week"] = featured["start_datetime"].dt.dayofweek
    featured["month"] = featured["start_datetime"].dt.month
    featured["peak_hour"] = featured["hour"].apply(is_peak_hour).astype(int)
    featured["parking_surge_hour"] = featured["hour"].apply(is_parking_surge_hour).astype(int)
    featured["weekend"] = featured["day_of_week"].isin({5, 6}).astype(int)
    return featured


def is_peak_hour(hour: int | float | None) -> int:
    if pd.isna(hour):
        return 0
    return int(int(hour) in PEAK_HOURS)


def is_parking_surge_hour(hour: int | float | None) -> int:
    if pd.isna(hour):
        return 0
    return int(int(hour) in PARKING_SURGE_HOURS)


def build_hotspot_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the hotspot/hour risk table from cleaned police parking violations."""
    events = _ensure_signal_columns(_obstruction_events(add_time_features(df)))
    if events.empty:
        return _empty_risk_table(["junction", "zone", "hour"])

    risk_df = (
        events.dropna(subset=["junction", "zone", "hour"])
        .groupby(["junction", "zone", "hour"], as_index=False)
        .agg(
            event_frequency=("id", "count"),
            avg_duration=("duration_minutes", "mean"),
            vehicle_types=("veh_type", "nunique"),
            wrong_parking_count=("wrong_parking", "sum"),
            no_parking_count=("no_parking", "sum"),
            main_road_count=("main_road_parking", "sum"),
            footpath_count=("footpath_parking", "sum"),
            double_parking_count=("double_parking", "sum"),
            crossing_count=("near_crossing", "sum"),
            bus_stop_count=("near_bus_stop", "sum"),
            traffic_light_count=("near_traffic_light", "sum"),
            congestion_violation_count=("congestion_violation", "sum"),
            two_wheeler_count=("two_wheeler", "sum"),
            auto_count=("auto_rickshaw", "sum"),
            car_count=("car_vehicle", "sum"),
            heavy_vehicle_count=("heavy_vehicle", "sum"),
            approved_count=("approved_violation", "sum"),
            sent_to_scita_count=("sent_to_scita", "sum"),
            road_closure_count=("requires_road_closure", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            unique_devices=("device_id", "nunique"),
        )
        .fillna(0)
    )

    risk_df["peak_hour"] = risk_df["hour"].apply(is_peak_hour).astype(int)
    risk_df["parking_surge_hour"] = risk_df["hour"].apply(is_parking_surge_hour).astype(int)
    risk_df["vehicle_impact"] = _vehicle_impact(risk_df)
    return calculate_risk_scores(risk_df)


def build_ml_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model training table used by train.py."""
    events = _ensure_signal_columns(_obstruction_events(add_time_features(df)))
    if events.empty:
        columns = [
            "junction",
            "zone",
            "hour",
            "day_of_week",
            "event_frequency",
            "avg_duration",
            "vehicle_types",
            "congestion_violation_count",
            "vehicle_impact",
            "parking_surge_hour",
            "peak_hour",
            "risk_score",
        ]
        return pd.DataFrame(columns=columns)

    ml_df = (
        events.dropna(subset=["junction", "zone", "hour", "day_of_week"])
        .groupby(["junction", "zone", "hour", "day_of_week"], as_index=False)
        .agg(
            event_frequency=("id", "count"),
            avg_duration=("duration_minutes", "mean"),
            vehicle_types=("veh_type", "nunique"),
            wrong_parking_count=("wrong_parking", "sum"),
            no_parking_count=("no_parking", "sum"),
            main_road_count=("main_road_parking", "sum"),
            footpath_count=("footpath_parking", "sum"),
            double_parking_count=("double_parking", "sum"),
            crossing_count=("near_crossing", "sum"),
            bus_stop_count=("near_bus_stop", "sum"),
            traffic_light_count=("near_traffic_light", "sum"),
            congestion_violation_count=("congestion_violation", "sum"),
            two_wheeler_count=("two_wheeler", "sum"),
            auto_count=("auto_rickshaw", "sum"),
            car_count=("car_vehicle", "sum"),
            heavy_vehicle_count=("heavy_vehicle", "sum"),
            approved_count=("approved_violation", "sum"),
            sent_to_scita_count=("sent_to_scita", "sum"),
            road_closure_count=("requires_road_closure", "sum"),
            unique_devices=("device_id", "nunique"),
        )
        .fillna(0)
    )

    ml_df["peak_hour"] = ml_df["hour"].apply(is_peak_hour).astype(int)
    ml_df["parking_surge_hour"] = ml_df["hour"].apply(is_parking_surge_hour).astype(int)
    ml_df["weekend"] = ml_df["day_of_week"].isin({5, 6}).astype(int)
    ml_df["vehicle_impact"] = _vehicle_impact(ml_df)
    ml_df = calculate_risk_scores(
        ml_df,
        weights={
            "event_frequency": 0.42,
            "congestion_violation_count": 0.18,
            "vehicle_impact": 0.12,
            "avg_duration": 0.10,
            "parking_surge_hour": 0.10,
            "approved_count": 0.05,
            "unique_devices": 0.03,
        },
    )
    return ml_df


def calculate_risk_scores(
    risk_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Scale risk features and calculate a 0-100 risk_score."""
    scored = risk_df.copy()
    weights = weights or {
        "event_frequency": 0.42,
        "congestion_violation_count": 0.18,
        "vehicle_impact": 0.12,
        "avg_duration": 0.10,
        "parking_surge_hour": 0.10,
        "approved_count": 0.05,
        "unique_devices": 0.03,
    }

    for column in weights:
        if column not in scored.columns:
            scored[column] = 0
        scored[column] = scored[column].fillna(0)
        scored[f"{column}_scaled"] = _minmax(scored[column])

    total_weight = sum(weights.values()) or 1.0
    score = sum(scored[f"{column}_scaled"] * weight for column, weight in weights.items())
    scored["risk_score"] = (score / total_weight * 100).clip(0, 100)
    scored["risk_level"] = scored["risk_score"].apply(risk_level)
    scored["recommendation"] = scored.apply(recommend, axis=1)
    scored["reason"] = scored.apply(build_reason, axis=1)
    return scored


def risk_level(score: float, thresholds: RiskThresholds = RiskThresholds()) -> str:
    if score >= thresholds.high:
        return "High"
    if score >= thresholds.medium:
        return "Medium"
    return "Low"


def recommend(row: pd.Series, thresholds: RiskThresholds = RiskThresholds()) -> str:
    score = float(row.get("risk_score", 0) or 0)
    hour = int(row.get("hour", 0) or 0)

    if score >= thresholds.high:
        deploy_hour = (hour - 1) % 24
        return f"Deploy response team at {deploy_hour:02d}:30"
    if score >= thresholds.medium:
        return "Increase monitoring"
    return "Normal patrol"


def build_reason(row: pd.Series) -> str:
    reasons = reason_list(row)
    if not reasons:
        return "Limited historical obstruction signal"
    return "; ".join(reasons)


def reason_list(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    frequency = float(row.get("event_frequency", 0) or 0)
    duration = float(row.get("avg_duration", 0) or 0)
    congestion = float(row.get("congestion_violation_count", 0) or 0)
    heavy = float(row.get("heavy_vehicle_count", 0) or 0)

    if frequency >= 1000:
        reasons.append("Heavy parking violation concentration")
    elif frequency >= 100:
        reasons.append("Recurring parking violation cluster")
    elif frequency > 0:
        reasons.append("Recorded parking violation history")

    if is_peak_hour(row.get("hour")):
        reasons.append("Traffic peak-hour impact")

    if is_parking_surge_hour(row.get("hour")):
        reasons.append("Parking enforcement surge-hour pattern")

    if congestion >= 100:
        reasons.append("Main-road, footpath, crossing, or double-parking pattern")
    elif congestion > 0:
        reasons.append("Violation type directly reduces usable road space")

    if duration >= 240:
        reasons.append("Delayed validation/clearance proxy")

    if heavy >= 10:
        reasons.append("Heavy-vehicle parking involvement")

    return reasons


def prepare_model_features(
    rows: pd.DataFrame,
    model_columns: Iterable[str],
) -> pd.DataFrame:
    """One-hot encode rows and align them to trained model columns."""
    feature_rows = rows.drop(columns=["risk_score"], errors="ignore")
    encoded = pd.get_dummies(feature_rows)
    return encoded.reindex(columns=list(model_columns), fill_value=0)


def _obstruction_events(df: pd.DataFrame) -> pd.DataFrame:
    if "is_obstruction_event" not in df.columns:
        raise ValueError("Dataset must contain is_obstruction_event")
    return df[df["is_obstruction_event"].eq(1)].copy()


def _ensure_signal_columns(events: pd.DataFrame) -> pd.DataFrame:
    prepared = events.copy()
    defaults = {
        "wrong_parking": 0,
        "no_parking": 0,
        "main_road_parking": 0,
        "footpath_parking": 0,
        "double_parking": 0,
        "near_crossing": 0,
        "near_bus_stop": 0,
        "near_traffic_light": 0,
        "congestion_violation": 0,
        "two_wheeler": 0,
        "auto_rickshaw": 0,
        "car_vehicle": 0,
        "heavy_vehicle": 0,
        "approved_violation": 0,
        "sent_to_scita": 0,
        "requires_road_closure": False,
        "device_id": "unknown-device",
    }
    for column, value in defaults.items():
        if column not in prepared.columns:
            prepared[column] = value
    return prepared


def _vehicle_impact(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df.get("two_wheeler_count", 0), errors="coerce").fillna(0) * 0.35
        + pd.to_numeric(df.get("auto_count", 0), errors="coerce").fillna(0) * 0.9
        + pd.to_numeric(df.get("car_count", 0), errors="coerce").fillna(0) * 1.0
        + pd.to_numeric(df.get("heavy_vehicle_count", 0), errors="coerce").fillna(0) * 1.8
    )


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(max_value, min_value):
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def _truck_count(values: pd.Series) -> int:
    return int(values.astype("string").str.lower().str.contains("truck", na=False).sum())


def _empty_risk_table(group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
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
        "unique_devices",
        "vehicle_impact",
        "road_closure_count",
        "peak_hour",
        "parking_surge_hour",
        "risk_score",
        "risk_level",
        "recommendation",
        "reason",
    ]
    return pd.DataFrame(columns=group_columns + metric_columns)


def export_feature_tables(data_path: str | Path = DEFAULT_DATA_PATH, out_dir: str | Path = DEFAULT_FEATURE_DIR) -> dict[str, int | str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cleaned = load_and_clean(data_path)
    hotspot_table = build_hotspot_risk_table(cleaned).sort_values(["risk_score", "event_frequency"], ascending=False)
    training_table = build_ml_training_table(cleaned).sort_values(["risk_score", "event_frequency"], ascending=False)

    hotspot_file = out_path / "risk_lookup.csv"
    training_file = out_path / "training_features.csv"
    summary_file = out_path / "feature_summary.json"
    hotspot_table.to_csv(hotspot_file, index=False)
    training_table.to_csv(training_file, index=False)
    summary = {
        "source": str(data_path),
        "cleaned_rows": int(len(cleaned)),
        "hotspot_rows": int(len(hotspot_table)),
        "training_rows": int(len(training_table)),
        "top_hotspot": None if hotspot_table.empty else str(hotspot_table.iloc[0]["junction"]),
        "top_risk_score": 0 if hotspot_table.empty else round(float(hotspot_table.iloc[0]["risk_score"]), 2),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parking-violation feature tables.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to the police violation CSV")
    parser.add_argument("--out-dir", default=str(DEFAULT_FEATURE_DIR), help="Directory for engineered feature tables")
    args = parser.parse_args()
    print(json.dumps(export_feature_tables(args.data, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
