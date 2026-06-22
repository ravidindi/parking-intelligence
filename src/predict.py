"""Prediction and explainability helpers for enforcement recommendations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import (
    build_hotspot_risk_table,
    build_ml_training_table,
    build_reason,
    prepare_model_features,
    reason_list,
    recommend,
    risk_level,
)
from src.preprocessing import load_and_clean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "jan to may police violation_anonymized791b166.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "police_violation"


def predict_risk(
    junction: str,
    hour: int,
    day: int | str | None = None,
    zone: str | None = None,
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Return risk score, reasons, and enforcement recommendation."""
    hour = _validate_hour(hour)
    day_of_week = parse_day(day) if day is not None else None

    ml_df = _training_table(data_path)
    historical_row = _select_historical_row(ml_df, junction, hour, day_of_week, zone)

    model_prediction = _predict_with_model(historical_row, model_dir)
    if model_prediction is not None:
        risk_score = model_prediction["risk_score"]
        shap_reasons = model_prediction.get("shap_reasons", [])
    else:
        risk_score = _historical_risk_score(junction, hour, zone, data_path)
        shap_reasons = []

    row = historical_row.iloc[0].copy()
    row["risk_score"] = risk_score
    row["risk_level"] = risk_level(risk_score)
    row["recommendation"] = recommend(row)
    row["reason"] = "; ".join(shap_reasons or reason_list(row)) or build_reason(row)

    return {
        "junction": junction,
        "zone": None if pd.isna(row.get("zone")) else row.get("zone"),
        "police_station": _clean_string(row.get("zone"), "Unknown"),
        "dispatch_station": _dispatch_station(row.get("zone")),
        "hour": hour,
        "day_of_week": None if day_of_week is None else day_of_week,
        "risk_score": round(float(risk_score), 2),
        "risk_level": row["risk_level"],
        "reason": row["reason"],
        "recommendation": row["recommendation"],
        "features": {
            "event_frequency": int(row.get("event_frequency", 0) or 0),
            "avg_duration": round(float(row.get("avg_duration", 0) or 0), 2),
            "vehicle_types": int(row.get("vehicle_types", 0) or 0),
            "peak_hour": int(row.get("peak_hour", 0) or 0),
        },
    }


def predict_location_risk(
    latitude: float,
    longitude: float,
    hour: int,
    radius_km: float = 2.0,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Return risk for the nearest historical hotspot to a latitude/longitude."""
    hour = _validate_hour(hour)
    lat = _validate_latitude(latitude)
    lon = _validate_longitude(longitude)

    risk_df = _risk_table(data_path)
    if risk_df.empty:
        row = _fallback_row("Unknown", "Unknown", hour, None).iloc[0]
        return _location_response(row, lat, lon, hour, None, "No historical hotspots available")

    risk_df = risk_df.dropna(subset=["latitude", "longitude"]).copy()
    risk_df["distance_km"] = _haversine_km(lat, lon, risk_df["latitude"], risk_df["longitude"])

    hourly = risk_df[risk_df["hour"].eq(hour)].copy()
    candidates = hourly if not hourly.empty else risk_df
    nearby = candidates[candidates["distance_km"].le(radius_km)].copy()
    if nearby.empty:
        nearby = candidates.sort_values("distance_km").head(10).copy()

    chosen = nearby.sort_values(["risk_score", "event_frequency"], ascending=False).head(1).iloc[0]
    match_note = (
        f"Nearest relevant hotspot within {round(float(chosen.distance_km), 2)} km"
        if float(chosen.distance_km) <= radius_km
        else f"No hotspot within {radius_km} km; using nearest available match at {round(float(chosen.distance_km), 2)} km"
    )
    return _location_response(chosen, lat, lon, hour, float(chosen.distance_km), match_note)


def top_hotspots(
    limit: int = 24,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> list[dict[str, Any]]:
    """Return unique corridor hotspots for dashboard ranking and map markers."""
    risk_df = _risk_table(data_path)
    if risk_df.empty:
        return []

    rows = _unique_corridor_rows(risk_df, limit)
    return [_hotspot_record(row) for _, row in rows.iterrows()]


def parse_day(day: int | str) -> int:
    if isinstance(day, int):
        if 0 <= day <= 6:
            return day
        raise ValueError("day must be 0-6, where Monday is 0")

    normalized = str(day).strip().lower()
    names = {
        "monday": 0,
        "mon": 0,
        "tuesday": 1,
        "tue": 1,
        "wednesday": 2,
        "wed": 2,
        "thursday": 3,
        "thu": 3,
        "friday": 4,
        "fri": 4,
        "saturday": 5,
        "sat": 5,
        "sunday": 6,
        "sun": 6,
    }
    if normalized.isdigit():
        return parse_day(int(normalized))
    if normalized in names:
        return names[normalized]
    raise ValueError("day must be a weekday name or 0-6")


def _location_response(
    row: pd.Series,
    requested_latitude: float,
    requested_longitude: float,
    hour: int,
    distance_km: float | None,
    match_note: str,
) -> dict[str, Any]:
    risk_score = float(row.get("risk_score", 0) or 0)
    row = row.copy()
    row["hour"] = hour
    row["risk_score"] = risk_score
    return {
        "requested_latitude": requested_latitude,
        "requested_longitude": requested_longitude,
        "matched_latitude": _clean_float(row.get("latitude")),
        "matched_longitude": _clean_float(row.get("longitude")),
        "distance_km": None if distance_km is None else round(distance_km, 3),
        "junction": _clean_string(row.get("junction"), "Unknown"),
        "zone": _clean_string(row.get("zone"), "Unknown"),
        "police_station": _clean_string(row.get("zone"), "Unknown"),
        "dispatch_station": _dispatch_station(row.get("zone")),
        "hour": hour,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level(risk_score),
        "reason": build_reason(row),
        "recommendation": recommend(row),
        "match_note": match_note,
        "features": {
            "event_frequency": int(row.get("event_frequency", 0) or 0),
            "avg_duration": round(float(row.get("avg_duration", 0) or 0), 2),
            "vehicle_types": int(row.get("vehicle_types", 0) or 0),
            "peak_hour": int(row.get("peak_hour", 0) or 0),
        },
    }


def _hotspot_record(row: pd.Series) -> dict[str, Any]:
    return {
        "junction": _clean_string(row.get("junction"), "Unknown"),
        "zone": _clean_string(row.get("zone"), "Unknown"),
        "police_station": _clean_string(row.get("zone"), "Unknown"),
        "dispatch_station": _dispatch_station(row.get("zone")),
        "hour": int(row.get("hour", 0) or 0),
        "latitude": _clean_float(row.get("latitude")),
        "longitude": _clean_float(row.get("longitude")),
        "risk_score": round(float(row.get("risk_score", 0) or 0), 2),
        "risk_level": _clean_string(row.get("risk_level"), "Low"),
        "event_frequency": int(row.get("event_frequency", 0) or 0),
        "avg_duration": round(float(row.get("avg_duration", 0) or 0), 2),
        "recommendation": _clean_string(row.get("recommendation"), "Normal patrol"),
        "reason": _clean_string(row.get("reason"), "Limited historical obstruction signal"),
        "active_hours": _clean_hours(row.get("active_hours"), row.get("hour", 0)),
        "records_merged": int(row.get("records_merged", 1) or 1),
    }


def _unique_corridor_rows(risk_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    rows = risk_df.dropna(subset=["latitude", "longitude"]).copy()
    if rows.empty:
        return rows

    rows["risk_score"] = pd.to_numeric(rows["risk_score"], errors="coerce").fillna(0)
    rows["event_frequency"] = pd.to_numeric(rows["event_frequency"], errors="coerce").fillna(0)
    rows["avg_duration"] = pd.to_numeric(rows["avg_duration"], errors="coerce").fillna(0)

    representatives: list[pd.Series] = []
    for _, group in rows.groupby(["junction", "zone"], dropna=False):
        ranked = group.sort_values(["risk_score", "event_frequency"], ascending=False)
        representative = ranked.iloc[0].copy()
        total_events = float(group["event_frequency"].sum())
        if total_events > 0:
            representative["avg_duration"] = float((group["avg_duration"] * group["event_frequency"]).sum() / total_events)
        representative["event_frequency"] = int(total_events)
        representative["active_hours"] = sorted({int(hour) for hour in pd.to_numeric(group["hour"], errors="coerce").dropna()})
        representative["records_merged"] = int(len(group))
        representatives.append(representative)

    unique_rows = pd.DataFrame(representatives)
    return unique_rows.sort_values(["risk_score", "event_frequency"], ascending=False).head(limit)


def _clean_hours(value: Any, fallback_hour: Any) -> list[int]:
    if isinstance(value, list):
        hours = value
    elif isinstance(value, tuple):
        hours = list(value)
    else:
        hours = [fallback_hour]
    cleaned = sorted({int(hour) for hour in hours if not pd.isna(hour)})
    return cleaned or [int(fallback_hour or 0)]


def _haversine_km(lat: float, lon: float, lats: pd.Series, lons: pd.Series) -> pd.Series:
    radius = 6371.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(pd.to_numeric(lats, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lons, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return pd.Series(radius * 2 * np.arcsin(np.sqrt(a)), index=lats.index)


def _validate_latitude(latitude: float) -> float:
    value = float(latitude)
    if not -90 <= value <= 90:
        raise ValueError("latitude must be between -90 and 90")
    return value


def _validate_longitude(longitude: float) -> float:
    value = float(longitude)
    if not -180 <= value <= 180:
        raise ValueError("longitude must be between -180 and 180")
    return value


def _clean_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 7)


def _clean_string(value: Any, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    text = str(value)
    return text if text else fallback


def _dispatch_station(value: Any) -> str:
    station = _clean_string(value, "Unknown")
    if station == "Unknown":
        return "Nearest available traffic enforcement unit"
    lowered = station.lower()
    if "station" in lowered:
        return station
    return f"{station} Traffic Police Station"


def _select_historical_row(
    ml_df: pd.DataFrame,
    junction: str,
    hour: int,
    day_of_week: int | None,
    zone: str | None,
) -> pd.DataFrame:
    if ml_df.empty:
        return _fallback_row(junction, zone, hour, day_of_week)

    candidates = ml_df[
        ml_df["junction"].astype("string").str.casefold().eq(str(junction).casefold())
        & ml_df["hour"].eq(hour)
    ].copy()

    if zone:
        zoned = candidates[candidates["zone"].astype("string").str.casefold().eq(str(zone).casefold())]
        if not zoned.empty:
            candidates = zoned

    if day_of_week is not None:
        daily = candidates[candidates["day_of_week"].eq(day_of_week)]
        if not daily.empty:
            candidates = daily

    if candidates.empty:
        candidates = ml_df[
            ml_df["junction"].astype("string").str.casefold().eq(str(junction).casefold())
        ].copy()

    if candidates.empty:
        return _fallback_row(junction, zone, hour, day_of_week)

    chosen = candidates.sort_values(["risk_score", "event_frequency"], ascending=False).head(1).copy()
    chosen["hour"] = hour
    if day_of_week is not None:
        chosen["day_of_week"] = day_of_week
    return chosen


def _predict_with_model(row: pd.DataFrame, model_dir: str | Path) -> dict[str, Any] | None:
    model_dir = Path(model_dir)
    try:
        import joblib

        model = joblib.load(model_dir / "risk_model.pkl")
        columns = joblib.load(model_dir / "model_columns.pkl")
    except Exception:
        return None

    features = prepare_model_features(
        row.drop(columns=["risk_level", "recommendation", "reason"], errors="ignore"),
        columns,
    )
    score = float(model.predict(features)[0])
    return {
        "risk_score": max(0.0, min(100.0, score)),
        "shap_reasons": _shap_reasons(model, features),
    }


def _shap_reasons(model: Any, features: pd.DataFrame) -> list[str]:
    try:
        mpl_config_dir = PROJECT_ROOT / ".mplconfig"
        mpl_config_dir.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import shap

        explanation = shap.Explainer(model)(features)
        values = explanation.values[0]
    except Exception:
        return []

    positive = sorted(
        zip(features.columns, values),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    labels = []
    for column, contribution in positive:
        if contribution <= 0:
            continue
        labels.append(_feature_reason(column))
    return list(dict.fromkeys(labels))


def _feature_reason(column: str) -> str:
    if "event_frequency" in column:
        return "Frequent historical incidents"
    if "avg_duration" in column:
        return "Long obstruction duration"
    if "peak_hour" in column or "hour" in column:
        return "Peak-hour impact"
    if "truck" in column or "vehicle" in column:
        return "Vehicle mix increases clearance complexity"
    if "junction_" in column:
        return "Location has a historical obstruction pattern"
    return "Historical pattern increases risk"


def _historical_risk_score(
    junction: str,
    hour: int,
    zone: str | None,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> float:
    risk_df = _risk_table(data_path)
    if risk_df.empty:
        return 0.0

    matches = risk_df[
        risk_df["junction"].astype("string").str.casefold().eq(str(junction).casefold())
        & risk_df["hour"].eq(hour)
    ]
    if zone:
        zoned = matches[matches["zone"].astype("string").str.casefold().eq(str(zone).casefold())]
        if not zoned.empty:
            matches = zoned

    if matches.empty:
        matches = risk_df[risk_df["junction"].astype("string").str.casefold().eq(str(junction).casefold())]

    if matches.empty:
        return 0.0
    return float(matches.sort_values("risk_score", ascending=False).iloc[0]["risk_score"])


def _risk_table(data_path: str | Path) -> pd.DataFrame:
    return _risk_table_cached(str(Path(data_path).resolve()))


@lru_cache(maxsize=4)
def _risk_table_cached(data_path: str) -> pd.DataFrame:
    lookup_path = _artifact_path(Path(data_path), "risk_lookup.csv")
    if lookup_path.exists():
        return pd.read_csv(lookup_path)
    df = load_and_clean(data_path)
    return build_hotspot_risk_table(df)


def _training_table(data_path: str | Path) -> pd.DataFrame:
    return _training_table_cached(str(Path(data_path).resolve()))


@lru_cache(maxsize=4)
def _training_table_cached(data_path: str) -> pd.DataFrame:
    training_path = _artifact_path(Path(data_path), "training_features.csv")
    if training_path.exists():
        return pd.read_csv(training_path)
    df = load_and_clean(data_path)
    return build_ml_training_table(df)


def _artifact_path(data_path: Path, filename: str) -> Path:
    if data_path.resolve() == DEFAULT_DATA_PATH.resolve():
        return DEFAULT_MODEL_DIR / filename
    return Path("__missing_artifact__") / filename


def _fallback_row(
    junction: str,
    zone: str | None,
    hour: int,
    day_of_week: int | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "junction": junction,
                "zone": zone or "Unknown",
                "hour": hour,
                "day_of_week": 0 if day_of_week is None else day_of_week,
                "event_frequency": 0,
                "avg_duration": 0.0,
                "vehicle_types": 0,
                "truck_count": 0,
                "road_closure_count": 0,
                "peak_hour": int(hour in range(8, 12) or hour in range(17, 22)),
                "risk_score": 0.0,
            }
        ]
    )


def _validate_hour(hour: int) -> int:
    parsed = int(hour)
    if not 0 <= parsed <= 23:
        raise ValueError("hour must be between 0 and 23")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict parking obstruction risk.")
    parser.add_argument("--junction", required=True)
    parser.add_argument("--hour", required=True, type=int)
    parser.add_argument("--day")
    parser.add_argument("--zone")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()

    result = predict_risk(args.junction, args.hour, args.day, args.zone, args.data, args.model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
