"""Data loading and cleaning for the parking intelligence pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


OBSTRUCTION_CAUSES = {"vehicle_breakdown"}
PARKING_KEYWORDS = (
    "parking",
    "parked",
    "illegal parking",
    "wrong parking",
    "stopped vehicle",
    "standing vehicle",
    "abandoned",
    "blocking",
    "obstruction",
    "lane block",
)
NEW_DATASET_COLUMNS = {
    "violation_type",
    "created_datetime",
    "police_station",
    "junction_name",
    "vehicle_type",
}

POLICE_DATETIME_COLUMNS = (
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "data_sent_to_scita_timestamp",
    "validation_timestamp",
)

DATETIME_COLUMNS = (
    "start_datetime",
    "end_datetime",
    "resolved_datetime",
    "closed_datetime",
    "modified_datetime",
)

TEXT_COLUMNS = (
    "event_cause",
    "junction",
    "zone",
    "veh_type",
    "description",
    "reason_breakdown",
    "address",
    "location",
    "violation_type",
    "offence_code",
    "vehicle_type",
    "updated_vehicle_type",
    "police_station",
    "junction_name",
    "validation_status",
)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw events and add obstruction/duration fields."""
    if NEW_DATASET_COLUMNS.issubset(df.columns):
        return clean_police_violation_dataset(df)
    return clean_legacy_traffic_dataset(df)


def clean_police_violation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the police parking-violation dataset for risk feature engineering."""
    cleaned = df.copy()

    for column in TEXT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA, "nan": pd.NA})
            )

    for column in POLICE_DATETIME_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", utc=True)

    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")
    cleaned["start_datetime"] = cleaned["created_datetime"]
    cleaned["effective_end_datetime"] = _first_available_datetime(
        cleaned,
        ("action_taken_timestamp", "validation_timestamp", "closed_datetime", "modified_datetime"),
    )
    cleaned["duration_minutes"] = (
        cleaned["effective_end_datetime"] - cleaned["start_datetime"]
    ).dt.total_seconds() / 60
    cleaned.loc[cleaned["duration_minutes"] < 0, "duration_minutes"] = pd.NA
    cleaned["duration_minutes"] = cleaned["duration_minutes"].clip(upper=1440).fillna(0)

    violation_text = _listish_text(cleaned.get("violation_type", pd.Series("", index=cleaned.index)))
    offence_text = _listish_text(cleaned.get("offence_code", pd.Series("", index=cleaned.index)))
    vehicle_text = _vehicle_series(cleaned).astype("string").fillna("Unknown")

    cleaned["event_cause"] = violation_text.replace("", "parking_violation")
    cleaned["reason_breakdown"] = violation_text
    cleaned["offence_code_text"] = offence_text
    cleaned["address"] = cleaned.get("location", pd.Series(pd.NA, index=cleaned.index))
    cleaned["veh_type"] = vehicle_text
    cleaned["vehicle_category"] = vehicle_text.apply(_vehicle_category)
    cleaned["zone"] = cleaned.get("police_station", pd.Series("Unknown Station", index=cleaned.index)).fillna("Unknown Station")
    cleaned["grid_latitude"] = cleaned["latitude"].round(3)
    cleaned["grid_longitude"] = cleaned["longitude"].round(3)

    junction_name = cleaned.get("junction_name", pd.Series(pd.NA, index=cleaned.index)).astype("string")
    has_named_junction = junction_name.notna() & ~junction_name.str.casefold().isin({"no junction", "nan", "null", ""})
    grid_label = "Grid " + cleaned["grid_latitude"].astype("string") + ", " + cleaned["grid_longitude"].astype("string")
    cleaned["junction"] = junction_name.where(has_named_junction, grid_label).fillna("Unknown Hotspot")
    cleaned["hotspot_key"] = cleaned["junction"].astype("string").str.casefold() + "|" + cleaned["zone"].astype("string").str.casefold()

    cleaned["wrong_parking"] = _contains_text(violation_text, "wrong parking")
    cleaned["no_parking"] = _contains_text(violation_text, "no parking")
    cleaned["main_road_parking"] = _contains_text(violation_text, "main road")
    cleaned["footpath_parking"] = _contains_text(violation_text, "footpath")
    cleaned["double_parking"] = _contains_text(violation_text, "double parking")
    cleaned["near_crossing"] = _contains_text(violation_text, "road crossing")
    cleaned["near_bus_stop"] = _contains_text(violation_text, "bustop|bus stop|school|hospital")
    cleaned["near_traffic_light"] = _contains_text(violation_text, "traffic light|zebra")
    cleaned["congestion_violation"] = (
        cleaned[
            [
                "main_road_parking",
                "footpath_parking",
                "double_parking",
                "near_crossing",
                "near_bus_stop",
                "near_traffic_light",
            ]
        ]
        .any(axis=1)
        .astype(int)
    )
    cleaned["two_wheeler"] = cleaned["vehicle_category"].eq("Two-wheeler").astype(int)
    cleaned["auto_rickshaw"] = cleaned["vehicle_category"].eq("Auto").astype(int)
    cleaned["car_vehicle"] = cleaned["vehicle_category"].eq("Car").astype(int)
    cleaned["heavy_vehicle"] = cleaned["vehicle_category"].eq("Heavy vehicle").astype(int)
    cleaned["approved_violation"] = (
        cleaned.get("validation_status", pd.Series("", index=cleaned.index))
        .astype("string")
        .str.casefold()
        .eq("approved")
        .fillna(False)
        .astype(int)
    )
    cleaned["sent_to_scita"] = _coerce_bool(
        cleaned.get("data_sent_to_scita", pd.Series(False, index=cleaned.index))
    ).astype(int)
    cleaned["requires_road_closure"] = False
    cleaned["is_obstruction_event"] = generate_obstruction_labels(cleaned)
    return cleaned


def clean_legacy_traffic_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the older traffic-event dataset. Kept only for backwards compatibility."""
    cleaned = df.copy()

    for column in TEXT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA, "nan": pd.NA})
            )

    for column in DATETIME_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", utc=True)

    cleaned["event_cause"] = cleaned.get("event_cause", pd.Series(dtype="string")).fillna("unknown")
    cleaned["requires_road_closure"] = _coerce_bool(
        cleaned.get("requires_road_closure", pd.Series(False, index=cleaned.index))
    )

    cleaned["effective_end_datetime"] = _derive_effective_end(cleaned)
    cleaned["duration_minutes"] = (
        cleaned["effective_end_datetime"] - cleaned["start_datetime"]
    ).dt.total_seconds() / 60
    cleaned.loc[cleaned["duration_minutes"] < 0, "duration_minutes"] = pd.NA

    cleaned["is_obstruction_event"] = generate_obstruction_labels(cleaned)
    return cleaned


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper used by training, prediction, and API code."""
    source = Path(path).resolve()
    return _load_and_clean_cached(str(source), source.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_and_clean_cached(path: str, mtime_ns: int) -> pd.DataFrame:
    _ = mtime_ns
    return clean_dataset(load_data(path))


def generate_obstruction_labels(df: pd.DataFrame) -> pd.Series:
    """Return 1 for stationary vehicle / likely illegal parking obstruction rows."""
    cause = df.get("event_cause", pd.Series("", index=df.index)).astype("string").str.lower()
    labels = cause.isin(OBSTRUCTION_CAUSES)

    text_signal = _contains_any_keyword(
        df,
        columns=("event_cause", "description", "reason_breakdown", "address"),
        keywords=PARKING_KEYWORDS,
    )
    return (labels | text_signal).astype(int)


def _listish_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"^\s*\[|\]\s*$", "", regex=True)
        .str.replace('"', "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(",", ";", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _vehicle_series(df: pd.DataFrame) -> pd.Series:
    updated = df.get("updated_vehicle_type", pd.Series(pd.NA, index=df.index))
    original = df.get("vehicle_type", pd.Series(pd.NA, index=df.index))
    return updated.fillna(original).fillna("Unknown")


def _vehicle_category(value: object) -> str:
    text = str(value).lower()
    if any(keyword in text for keyword in ("scooter", "motor cycle", "motorcycle", "moped")):
        return "Two-wheeler"
    if "auto" in text or "maxi-cab" in text:
        return "Auto"
    if any(keyword in text for keyword in ("bus", "lorry", "truck", "hgv", "lgv", "tanker", "tempo")):
        return "Heavy vehicle"
    if any(keyword in text for keyword in ("car", "jeep", "van")):
        return "Car"
    return "Other"


def _contains_text(series: pd.Series, pattern: str) -> pd.Series:
    return series.str.casefold().str.contains(pattern, regex=True, na=False).astype(int)


def _contains_any_keyword(
    df: pd.DataFrame,
    columns: Iterable[str],
    keywords: Iterable[str],
) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="string")
    for column in columns:
        if column in df.columns:
            text = text.str.cat(df[column].astype("string").fillna(""), sep=" ")
    pattern = "|".join(pd.Series(list(keywords)).str.replace(r"\s+", r"\\s+", regex=True))
    return text.str.lower().str.contains(pattern, regex=True, na=False)


def _derive_effective_end(df: pd.DataFrame) -> pd.Series:
    if "start_datetime" not in df.columns:
        raise ValueError("Dataset must contain start_datetime")

    end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for column in ("end_datetime", "resolved_datetime", "closed_datetime", "modified_datetime"):
        if column in df.columns:
            end = end.fillna(df[column])
    return end


def _first_available_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for column in columns:
        if column in df.columns:
            end = end.fillna(df[column])
    return end


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})
