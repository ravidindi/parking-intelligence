"""Operational intelligence helpers for congestion impact and enforcement analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engineering import build_hotspot_risk_table
from src.preprocessing import load_and_clean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "jan to may police violation_anonymized791b166.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "police_violation"
DEFAULT_ANALYSIS_DAYS = 151


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return round(max(minimum, min(maximum, value)), 2)


def estimate_congestion_impact(
    camera: dict[str, Any],
    detection: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    """Estimate lane-capacity loss and delay from live detection plus historical risk."""
    vehicle_count = int(detection.get("vehicle_count", 0) or 0)
    alert_count = int(detection.get("alert_count", 0) or 0)
    observed_seconds = float(camera.get("observed_seconds", 0) or 0)
    alert_threshold = max(1.0, float(camera.get("alert_threshold", 1) or 1))
    risk_score = float(risk.get("risk_score", 0) or 0)
    features = risk.get("features", {}) if isinstance(risk.get("features"), dict) else {}
    avg_duration = float(features.get("avg_duration", 0) or 0)
    event_frequency = int(features.get("event_frequency", 0) or 0)
    restricted = bool(camera.get("restricted_zone", False))

    lane_loss = clamp(
        vehicle_count * 10
        + alert_count * 18
        + (12 if restricted else 0)
        + min(22, risk_score * 0.22)
        + min(12, observed_seconds / alert_threshold * 8),
        maximum=82,
    )
    delay_minutes = round(
        min(
            180.0,
            avg_duration * (0.12 + risk_score / 260)
            + observed_seconds / 60 * 2.2
            + alert_count * 9
            + event_frequency * 1.8,
        ),
        1,
    )
    queue_meters = int(
        min(
            950,
            35
            + lane_loss * 4.3
            + vehicle_count * 32
            + alert_count * 70
            + event_frequency * 11,
        )
    )
    throughput_loss = clamp(lane_loss + risk_score * 0.12 + alert_count * 4, maximum=88)
    impact_score = clamp(
        risk_score * 0.38
        + lane_loss * 0.32
        + alert_count * 18
        + vehicle_count * 3
        + min(14, observed_seconds / alert_threshold * 8),
    )

    if impact_score >= 75:
        severity = "Severe"
        flow_state = "Lane capacity compromised"
        action = "Clear obstruction immediately and hold junction approach"
    elif impact_score >= 55:
        severity = "High"
        flow_state = "Active slowdown likely"
        action = "Dispatch patrol and monitor approach queue"
    elif impact_score >= 35:
        severity = "Moderate"
        flow_state = "Localized friction"
        action = "Increase monitoring"
    else:
        severity = "Low"
        flow_state = "Normal flow"
        action = "Normal patrol"

    return {
        "impact_score": impact_score,
        "severity": severity,
        "flow_state": flow_state,
        "lane_capacity_loss_pct": lane_loss,
        "throughput_loss_pct": throughput_loss,
        "estimated_delay_minutes": delay_minutes,
        "queue_risk_meters": queue_meters,
        "vehicle_count": vehicle_count,
        "alert_count": alert_count,
        "action": action,
        "assumptions": [
            "Impact is a calibrated proxy from stationary duration, vehicle count, zone policy, and historical obstruction risk.",
            "Delay and queue estimates are demo-stage planning signals, not live sensor measurements.",
        ],
    }


def before_after_analytics(limit: int = 12, data_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Project enforcement benefit for top hotspots from artifacts or the historical dataset."""
    risk_df, analysis_days = _risk_table_for_analytics(data_path)
    if risk_df.empty:
        return {
            "summary": {
                "hotspots_analyzed": 0,
                "delay_before_minutes": 0.0,
                "delay_after_minutes": 0.0,
                "delay_saved_minutes": 0.0,
                "risk_reduction_pct": 0.0,
                "patrol_hours_saved": 0.0,
            },
            "corridors": [],
        }

    rows = risk_df.sort_values(["risk_score", "event_frequency"], ascending=False).head(limit)
    corridors = []
    before_total = 0.0
    after_total = 0.0
    before_risk_total = 0.0
    after_risk_total = 0.0

    for _, row in rows.iterrows():
        risk_score = float(row.get("risk_score", 0) or 0)
        event_frequency = int(row.get("event_frequency", 0) or 0)
        avg_duration = float(row.get("avg_duration", 0) or 0)
        weekly_events = event_frequency / analysis_days * 7
        delay_before = round(weekly_events * min(avg_duration, 45), 1)
        reduction_rate = _reduction_rate(risk_score)
        delay_after = round(delay_before * (1 - reduction_rate), 1)
        risk_after = clamp(risk_score * (1 - reduction_rate * 0.72))
        saved = round(delay_before - delay_after, 1)

        before_total += delay_before
        after_total += delay_after
        before_risk_total += risk_score
        after_risk_total += risk_after
        corridors.append(
            {
                "junction": _string(row.get("junction"), "Unknown"),
                "zone": _string(row.get("zone"), "Unknown"),
                "hour": int(row.get("hour", 0) or 0),
                "risk_before": round(risk_score, 2),
                "risk_after": risk_after,
                "delay_before_minutes": delay_before,
                "delay_after_minutes": delay_after,
                "delay_saved_minutes": saved,
                "expected_reduction_pct": round(reduction_rate * 100, 1),
                "action": "Pre-position patrol" if risk_score >= 70 else "Timed monitoring sweep" if risk_score >= 40 else "Routine watch",
            }
        )

    count = len(corridors) or 1
    risk_reduction = 0.0 if before_risk_total == 0 else (before_risk_total - after_risk_total) / before_risk_total * 100
    return {
        "summary": {
            "hotspots_analyzed": len(corridors),
            "delay_before_minutes": round(before_total, 1),
            "delay_after_minutes": round(after_total, 1),
            "delay_saved_minutes": round(before_total - after_total, 1),
            "risk_reduction_pct": round(risk_reduction, 1),
            "patrol_hours_saved": round((before_total - after_total) / 60, 1),
            "average_risk_before": round(before_risk_total / count, 1),
            "average_risk_after": round(after_risk_total / count, 1),
        },
        "corridors": corridors,
    }


def _risk_table_for_analytics(data_path: str | Path) -> tuple[pd.DataFrame, int]:
    lookup_path = DEFAULT_MODEL_DIR / "risk_lookup.csv"
    if Path(data_path).resolve() == DEFAULT_DATA_PATH.resolve() and lookup_path.exists():
        return pd.read_csv(lookup_path), DEFAULT_ANALYSIS_DAYS

    df = load_and_clean(data_path)
    risk_df = build_hotspot_risk_table(df)
    valid_times = df["start_datetime"].dropna() if "start_datetime" in df.columns else None
    if valid_times is not None and not valid_times.empty:
        analysis_days = max(1, int((valid_times.max() - valid_times.min()).days) + 1)
    else:
        analysis_days = DEFAULT_ANALYSIS_DAYS
    return risk_df, analysis_days


def _reduction_rate(risk_score: float) -> float:
    if risk_score >= 70:
        return 0.42
    if risk_score >= 40:
        return 0.32
    return 0.18


def _string(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value)
    return text if text and text.lower() != "nan" else fallback
