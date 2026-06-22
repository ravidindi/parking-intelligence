"""FastAPI endpoint and Swagger docs for parking intelligence."""

from __future__ import annotations

import time
import uuid
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.incidents import IncidentStore
from src.intelligence import before_after_analytics, estimate_congestion_impact
from src.offence_codes import list_offences
from src.predict import predict_location_risk, predict_risk, top_hotspots
from src.realtime_detection import (
    StationaryVehicleTracker,
    annotate_alerts,
    detect_vehicles_safely,
    summarize_detection,
    synthetic_detection,
)


tracker = StationaryVehicleTracker()
incident_store = IncidentStore()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "junction": "SilkBoardJunc",
                    "hour": 19,
                    "day": "Friday",
                    "zone": "South Zone 2",
                },
                {
                    "junction": "GokuldasImagesJunc",
                    "hour": 21,
                },
            ]
        }
    )

    junction: str = Field(
        ...,
        min_length=1,
        description="Junction name from the historical traffic event dataset.",
        examples=["SilkBoardJunc"],
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day in 24-hour format.",
        examples=[19],
    )
    day: Optional[str] = Field(
        default=None,
        description="Optional weekday name or number. Monday is 0, Sunday is 6.",
        examples=["Friday"],
    )
    zone: Optional[str] = Field(
        default=None,
        description="Optional zone filter when a junction exists in multiple zones.",
        examples=["South Zone 2"],
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "junction": "SilkBoardJunc",
                "zone": "South Zone 2",
                "police_station": "South Zone 2",
                "dispatch_station": "South Zone 2 Traffic Police Station",
                "hour": 19,
                "day_of_week": None,
                "risk_score": 35.09,
                "risk_level": "Low",
                "reason": "Frequent historical incidents; Peak-hour impact; Long obstruction duration",
                "recommendation": "Normal patrol",
                "features": {
                    "event_frequency": 2,
                    "avg_duration": 170.49,
                    "vehicle_types": 1,
                    "peak_hour": 1,
                },
            }
        }
    )

    junction: str = Field(description="Requested junction name.")
    zone: Optional[str] = Field(description="Matched historical zone, when available.")
    police_station: Optional[str] = Field(default=None, description="Police station from the historical dataset.")
    dispatch_station: Optional[str] = Field(default=None, description="Recommended enforcement station/unit to notify.")
    hour: int = Field(description="Requested hour of day.")
    day_of_week: Optional[int] = Field(description="Parsed weekday number, Monday is 0.")
    risk_score: float = Field(description="Predicted risk score from 0 to 100.")
    risk_level: str = Field(description="Low, Medium, or High risk bucket.")
    reason: str = Field(description="Explainable reasons behind the risk score.")
    recommendation: str = Field(description="Operational enforcement recommendation.")
    features: dict = Field(description="Historical features used for the prediction.")


class LocationPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Camera/event latitude.")
    longitude: float = Field(..., ge=-180, le=180, description="Camera/event longitude.")
    hour: int = Field(..., ge=0, le=23, description="Hour of day in 24-hour format.")
    radius_km: float = Field(2.0, ge=0.1, le=25, description="Search radius for nearby historical hotspots.")


class LocationPredictionResponse(BaseModel):
    requested_latitude: float
    requested_longitude: float
    matched_latitude: Optional[float]
    matched_longitude: Optional[float]
    distance_km: Optional[float]
    junction: str
    zone: str
    police_station: str
    dispatch_station: str
    hour: int
    risk_score: float
    risk_level: str
    reason: str
    recommendation: str
    match_note: str
    features: dict


class HotspotResponse(BaseModel):
    junction: str
    zone: str
    police_station: str
    dispatch_station: str
    hour: int
    latitude: Optional[float]
    longitude: Optional[float]
    risk_score: float
    risk_level: str
    event_frequency: int
    avg_duration: float
    recommendation: str
    reason: str
    active_hours: list[int] = Field(default_factory=list, description="Hours where this corridor has historical risk rows.")
    records_merged: int = Field(default=1, description="Number of hourly hotspot rows merged into this corridor.")


class RealtimeDetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "camera_id": "cam-silk-board-01",
                "restricted_zone": True,
                "vehicle_count": 1,
                "alert_count": 1,
                "detections": [
                    {
                        "class_name": "car",
                        "confidence": 0.99,
                        "bbox": [80.0, 120.0, 260.0, 260.0],
                        "centroid": [170.0, 190.0],
                        "stationary_seconds": 360.0,
                        "alert": True,
                        "alert_reason": "restricted zone; stationary for 360s",
                        "offence_code": "MV_ACT_177",
                        "offence_label": "No Parking",
                        "offence_fine_amount": 1000,
                        "offence_source": "https://btp.karnataka.gov.in/117/spot-fines/en",
                        "offence_legal_section": "Section 177 of M.V. Act",
                        "offence_subtype": "No-parking zone violation",
                    }
                ],
                "recommendation": "Generate illegal parking alert and dispatch nearest enforcement unit",
            }
        }
    )

    camera_id: str = Field(description="Camera or feed identifier.")
    restricted_zone: bool = Field(description="Whether detections are inside a restricted parking zone.")
    vehicle_count: int = Field(description="Number of detected vehicles.")
    alert_count: int = Field(description="Number of illegal parking/stationary alerts.")
    detections: list[dict[str, Any]] = Field(description="Detected vehicles with boxes and alert metadata.")
    recommendation: str = Field(description="Real-time enforcement recommendation.")


class CameraPolicy(BaseModel):
    camera_id: str = Field(description="Stable camera/feed identifier.")
    label: str = Field(description="Human-readable camera source.")
    zone_type: str = Field(description="Operational zone type for the camera.")
    enforcement_level: str = Field(description="High, Medium, or Low enforcement policy.")
    restricted_zone: bool = Field(description="Whether the source is treated as restricted.")
    observed_seconds: float = Field(ge=0, description="Current observed stationary duration.")
    alert_threshold: float = Field(ge=0, description="Policy threshold for stationary alerts.")
    offence_context: Optional[str] = Field(default=None, description="Road-rule context used for offence code assignment.")


class CongestionImpactRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "camera": {
                    "camera_id": "cam-kr-market-01",
                    "label": "KR Market",
                    "zone_type": "Restricted commercial corridor",
                    "enforcement_level": "High",
                    "restricted_zone": True,
                    "observed_seconds": 210,
                    "alert_threshold": 90,
                },
                "detection": RealtimeDetectionResponse.model_config["json_schema_extra"]["example"],
                "risk": {
                    "requested_latitude": 12.9155,
                    "requested_longitude": 77.6238,
                    "matched_latitude": 12.9155,
                    "matched_longitude": 77.6238,
                    "distance_km": 0.0,
                    "junction": "SilkBoardJunc",
                    "zone": "South Zone 2",
                    "police_station": "South Zone 2",
                    "dispatch_station": "South Zone 2 Traffic Police Station",
                    "hour": 19,
                    "risk_score": 75,
                    "risk_level": "High",
                    "reason": "Frequent historical incidents; Peak-hour impact",
                    "recommendation": "Deploy response team",
                    "match_note": "Nearest relevant hotspot within 0.0 km",
                    "features": {"event_frequency": 3, "avg_duration": 180, "vehicle_types": 1, "peak_hour": 1},
                },
            }
        }
    )

    camera: CameraPolicy
    detection: RealtimeDetectionResponse
    risk: LocationPredictionResponse


class CongestionImpactResponse(BaseModel):
    impact_score: float
    severity: str
    flow_state: str
    lane_capacity_loss_pct: float
    throughput_loss_pct: float
    estimated_delay_minutes: float
    queue_risk_meters: int
    vehicle_count: int
    alert_count: int
    action: str
    assumptions: list[str]


class CorridorBenefit(BaseModel):
    junction: str
    zone: str
    hour: int
    risk_before: float
    risk_after: float
    delay_before_minutes: float
    delay_after_minutes: float
    delay_saved_minutes: float
    expected_reduction_pct: float
    action: str


class BeforeAfterSummary(BaseModel):
    hotspots_analyzed: int
    delay_before_minutes: float
    delay_after_minutes: float
    delay_saved_minutes: float
    risk_reduction_pct: float
    patrol_hours_saved: float
    average_risk_before: float = 0
    average_risk_after: float = 0


class BeforeAfterResponse(BaseModel):
    summary: BeforeAfterSummary
    corridors: list[CorridorBenefit]


class IncidentStatusUpdate(BaseModel):
    status: str = Field(
        ...,
        pattern="^(Open|Dispatched|Resolved|Dismissed)$",
        description="Incident workflow state.",
        examples=["Dispatched"],
    )


class EnforcementEvaluationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "camera": {
                    "camera_id": "cam-kr-market-01",
                    "label": "KR Market",
                    "zone_type": "Restricted commercial corridor",
                    "enforcement_level": "High",
                    "restricted_zone": True,
                    "observed_seconds": 210,
                    "alert_threshold": 90,
                },
                "detection": RealtimeDetectionResponse.model_config["json_schema_extra"]["example"],
                "risk": {
                    "requested_latitude": 12.9155,
                    "requested_longitude": 77.6238,
                    "matched_latitude": 12.9155,
                    "matched_longitude": 77.6238,
                    "distance_km": 0.0,
                    "junction": "SilkBoardJunc",
                    "zone": "South Zone 2",
                    "police_station": "South Zone 2",
                    "dispatch_station": "South Zone 2 Traffic Police Station",
                    "hour": 19,
                    "risk_score": 75,
                    "risk_level": "High",
                    "reason": "Frequent historical incidents; Peak-hour impact",
                    "recommendation": "Deploy response team",
                    "match_note": "Nearest relevant hotspot within 0.0 km",
                    "features": {"event_frequency": 3, "avg_duration": 180, "vehicle_types": 1, "peak_hour": 1},
                },
                "frame_name": "camera-frame.png",
            }
        }
    )

    camera: CameraPolicy
    detection: RealtimeDetectionResponse
    risk: LocationPredictionResponse
    frame_name: Optional[str] = Field(default=None, description="Uploaded frame name used as evidence context.")


class DispatchPlan(BaseModel):
    from_station: str
    target_stop: str
    target_latitude: Optional[float] = None
    target_longitude: Optional[float] = None
    personnel_count: int
    unit_count: int
    eta_minutes: int
    instruction: str
    rationale: list[str]


class EnforcementEvaluationResponse(BaseModel):
    incident_id: str
    created_at: str
    status: str = "Open"
    updated_at: Optional[str] = None
    priority: str
    decision: str
    fused_score: float
    dispatch: str
    dispatch_plan: Optional[DispatchPlan] = None
    reasons: list[str]
    congestion_impact: Optional[CongestionImpactResponse] = None
    evidence: dict[str, Any]


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _build_dispatch_plan(
    *,
    priority: str,
    dispatch_station: str,
    detection: RealtimeDetectionResponse,
    risk: LocationPredictionResponse,
    camera: CameraPolicy,
) -> dict[str, Any]:
    if priority == "Critical":
        personnel = 4
        eta_minutes = 8
    elif priority == "High":
        personnel = 3
        eta_minutes = 12
    elif priority == "Watch":
        personnel = 2
        eta_minutes = 20
    else:
        personnel = 1 if detection.vehicle_count else 0
        eta_minutes = 30

    if detection.alert_count >= 5:
        personnel += 2
    elif detection.alert_count >= 3:
        personnel += 1
    if risk.risk_score >= 85:
        personnel += 1
    if camera.enforcement_level.lower() == "high" and priority in {"Critical", "High"}:
        personnel += 1

    personnel = min(8, personnel)
    unit_count = 0 if personnel == 0 else max(1, (personnel + 1) // 2)
    target_latitude = risk.matched_latitude if risk.matched_latitude is not None else risk.requested_latitude
    target_longitude = risk.matched_longitude if risk.matched_longitude is not None else risk.requested_longitude
    target_stop = f"{risk.junction}, {risk.zone}"
    instruction = (
        f"Dispatch {personnel} personnel in {unit_count} unit(s) from {dispatch_station} "
        f"to {target_stop} within {eta_minutes} minutes."
        if personnel
        else f"No immediate dispatch; keep {target_stop} under scheduled patrol."
    )
    rationale = [
        f"{detection.alert_count} active alert(s), {detection.vehicle_count} vehicle(s) in frame.",
        f"Historical risk {risk.risk_level.lower()} with score {risk.risk_score:.0f}.",
        f"{camera.enforcement_level} enforcement camera policy.",
    ]
    if risk.distance_km is not None:
        rationale.append(f"Matched hotspot is {risk.distance_km:.2f} km from selected coordinates.")

    return {
        "from_station": dispatch_station,
        "target_stop": target_stop,
        "target_latitude": target_latitude,
        "target_longitude": target_longitude,
        "personnel_count": personnel,
        "unit_count": unit_count,
        "eta_minutes": eta_minutes,
        "instruction": instruction,
        "rationale": rationale,
    }


def evaluate_enforcement(payload: EnforcementEvaluationRequest) -> dict[str, Any]:
    detection = payload.detection
    risk = payload.risk
    camera = payload.camera
    dispatch_station = getattr(risk, "dispatch_station", None) or "nearest traffic enforcement unit"

    detection_score = 45 if detection.alert_count else min(20, detection.vehicle_count * 4)
    history_score = risk.risk_score * 0.45
    policy_score = 12 if camera.restricted_zone else 0
    density_score = min(10, detection.vehicle_count * 2)
    duration_score = 10 if camera.observed_seconds >= camera.alert_threshold else 0
    fused_score = _clamp_score(detection_score + history_score + policy_score + density_score + duration_score)

    if fused_score >= 80 or (detection.alert_count > 0 and risk.risk_level == "High"):
        priority = "Critical"
        decision = "Illegal parking risk confirmed"
        dispatch = f"Dispatch {dispatch_station} now"
    elif fused_score >= 60:
        priority = "High"
        decision = "Probable obstruction risk"
        dispatch = f"Dispatch {dispatch_station}"
    elif fused_score >= 35:
        priority = "Watch"
        decision = "Elevated monitoring required"
        dispatch = f"Ask {dispatch_station} to increase monitoring"
    else:
        priority = "Monitor"
        decision = "No immediate enforcement action"
        dispatch = f"Normal patrol by {dispatch_station}"

    dispatch_plan = _build_dispatch_plan(
        priority=priority,
        dispatch_station=dispatch_station,
        detection=detection,
        risk=risk,
        camera=camera,
    )
    if dispatch_plan["personnel_count"]:
        dispatch = dispatch_plan["instruction"]

    reasons: list[str] = []
    if detection.alert_count:
        reasons.append(f"{detection.alert_count} real-time stationary alert(s) detected.")
    elif detection.vehicle_count:
        reasons.append(f"{detection.vehicle_count} vehicle(s) detected without an active stationary alert.")
    else:
        reasons.append("No vehicle alert detected in the current frame.")
    reasons.append(f"Historical risk is {risk.risk_level.lower()} at {risk.junction} with score {risk.risk_score:.0f}.")
    reasons.append(f"{camera.zone_type} policy uses a {camera.alert_threshold:.0f}s threshold; observed {camera.observed_seconds:.0f}s.")
    if risk.distance_km is not None:
        reasons.append(f"Nearest historical hotspot is {risk.distance_km:.2f} km from selected coordinates.")
    reasons.append(f"Recommended enforcement station: {dispatch_station}.")

    created_at = datetime.now(timezone.utc).isoformat()
    incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
    impact = estimate_congestion_impact(
        camera.model_dump(),
        detection.model_dump(),
        risk.model_dump(),
    )
    evidence = {
        "incident_id": incident_id,
        "created_at": created_at,
        "status": "Open",
        "frame_name": payload.frame_name,
        "camera": camera.model_dump(),
        "detection": detection.model_dump(),
        "historical_risk": risk.model_dump(),
        "congestion_impact": impact,
        "dispatch_plan": dispatch_plan,
        "fusion": {
            "fused_score": fused_score,
            "priority": priority,
            "decision": decision,
            "dispatch": dispatch,
            "reasons": reasons,
        },
    }
    return {
        "incident_id": incident_id,
        "created_at": created_at,
        "status": "Open",
        "updated_at": None,
        "priority": priority,
        "decision": decision,
        "fused_score": fused_score,
        "dispatch": dispatch,
        "dispatch_plan": dispatch_plan,
        "reasons": reasons,
        "congestion_impact": impact,
        "evidence": evidence,
    }


def _cors_origins() -> list[str]:
    defaults = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]
    configured = os.getenv("BACKEND_CORS_ORIGINS", "")
    return defaults + [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app():
    app = FastAPI(
        title="Parking Intelligence API",
        description=(
            "Predict parking-induced congestion risk and test Phase 2 real-time "
            "vehicle detection / stationary vehicle alerts."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_origin_regex=os.getenv("BACKEND_CORS_REGEX", r"https://.*\.netlify\.app"),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def swagger_redirect():
        return RedirectResponse(url="/docs")

    @app.get(
        "/health",
        tags=["System"],
        summary="Health check",
        description="Verify that the API service is running.",
    )
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Prediction"],
        summary="Predict junction risk",
        description=(
            "Submit a junction and hour to receive a 0-100 obstruction risk score, "
            "reasons, and an enforcement recommendation."
        ),
    )
    def predict(payload: PredictionRequest) -> dict:
        return predict_risk(
            junction=payload.junction,
            hour=payload.hour,
            day=payload.day,
            zone=payload.zone,
        )

    @app.post(
        "/predict/location",
        response_model=LocationPredictionResponse,
        tags=["Prediction"],
        summary="Predict risk by coordinates",
        description=(
            "Predict historical obstruction risk by latitude/longitude. This avoids ambiguous "
            "place-name matching by selecting nearby historical hotspots using distance."
        ),
    )
    def predict_location(payload: LocationPredictionRequest) -> dict:
        return predict_location_risk(
            latitude=payload.latitude,
            longitude=payload.longitude,
            hour=payload.hour,
            radius_km=payload.radius_km,
        )

    @app.get(
        "/hotspots",
        response_model=list[HotspotResponse],
        tags=["Prediction"],
        summary="List top historical hotspot corridors",
        description=(
            "Return deduplicated corridor hotspots from the police violation dataset. "
            "Multiple hourly rows for the same junction/station are merged for dashboard ranking and map markers."
        ),
    )
    def hotspots(limit: int = 24) -> list[dict]:
        return top_hotspots(limit=limit)

    @app.get(
        "/offences",
        tags=["Offences"],
        summary="List supported parking offence codes",
        description=(
            "Return official/public Bengaluru Traffic Police spot-fine parking offence categories "
            "used by the detector rules. These are not the dataset's internal offence_code labels."
        ),
    )
    def offences() -> list[dict[str, Any]]:
        return list_offences()

    @app.post(
        "/impact/estimate",
        response_model=CongestionImpactResponse,
        tags=["Impact"],
        summary="Quantify parking-induced congestion impact",
        description=(
            "Estimate lane capacity loss, throughput loss, queue risk, and delay by "
            "combining the live frame verdict, selected camera policy, and coordinate-based historical risk."
        ),
    )
    def impact_estimate(payload: CongestionImpactRequest) -> dict[str, Any]:
        return estimate_congestion_impact(
            payload.camera.model_dump(),
            payload.detection.model_dump(),
            payload.risk.model_dump(),
        )

    @app.get(
        "/analytics/before-after",
        response_model=BeforeAfterResponse,
        tags=["Impact"],
        summary="Estimate enforcement before/after benefit",
        description=(
            "Project how targeted enforcement can reduce obstruction delay and risk "
            "for the highest-priority historical corridors."
        ),
    )
    def before_after(limit: int = 12) -> dict[str, Any]:
        return before_after_analytics(limit=limit)

    @app.post(
        "/detect/frame",
        response_model=RealtimeDetectionResponse,
        tags=["Real-Time Detection"],
        summary="Detect vehicles in a camera frame",
        description=(
            "Upload an image frame. The API detects vehicles with YOLO, tracks "
            "stationary duration by camera_id, and raises an alert when a vehicle "
            "is stationary in a restricted zone longer than the threshold. Use "
            "mock_detection=true in Swagger to test the alert flow without a YOLO "
            "model download."
        ),
    )
    async def detect_frame(
        image: UploadFile = File(..., description="Camera frame image: JPEG, PNG, HEIC, or HEIF."),
        camera_id: str = Form("default-camera", description="Stable camera/feed identifier."),
        restricted_zone: bool = Form(False, description="True if this camera covers a no-parking zone."),
        stationary_threshold_seconds: float = Form(
            300.0,
            ge=0,
            description="Seconds before a stationary restricted-zone vehicle becomes an alert.",
        ),
        confidence: float = Form(
            0.25,
            ge=0,
            le=1,
            description="YOLO confidence threshold.",
        ),
        observed_seconds: Optional[float] = Form(
            None,
            ge=0,
            description="Optional test helper that simulates prior stationary duration.",
        ),
        offence_context: Optional[str] = Form(
            None,
            description=(
                "Optional road-rule context used to assign an official offence: no_parking, "
                "main_road, footpath, double_parking, road_crossing, bus_stop_school_hospital, "
                "traffic_light_zebra, towing_car, towing_two_wheeler, towing_mtv, towing_htv."
            ),
        ),
        mock_detection: bool = Form(
            False,
            description="Use one deterministic sample vehicle instead of YOLO inference.",
        ),
    ) -> dict:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        if mock_detection:
            detections = [synthetic_detection()]
        else:
            try:
                detections = detect_vehicles_safely(content, confidence=confidence)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        timestamp = time.time()
        detections = tracker.update(camera_id, detections, timestamp=timestamp)
        if observed_seconds is not None:
            for detection in detections:
                detection.stationary_seconds = observed_seconds

        detections = annotate_alerts(
            detections,
            restricted_zone=restricted_zone,
            stationary_threshold_seconds=stationary_threshold_seconds,
            offence_context=offence_context,
        )
        return summarize_detection(detections, camera_id, restricted_zone)

    @app.post(
        "/detect/reset",
        tags=["Real-Time Detection"],
        summary="Reset stationary tracking state",
        description="Clear tracker state for one camera or all cameras.",
    )
    def reset_detection_state(camera_id: Optional[str] = None) -> dict[str, str | None]:
        tracker.reset(camera_id)
        return {"status": "reset", "camera_id": camera_id}

    @app.post(
        "/enforcement/evaluate",
        response_model=EnforcementEvaluationResponse,
        tags=["Enforcement"],
        summary="Fuse detection and historical risk into an incident decision",
        description=(
            "Combine the live frame verdict, camera policy, stationary threshold, "
            "and coordinate-based historical risk into one operational decision. "
            "The returned evidence package is persisted as JSONL for demo auditability."
        ),
    )
    def enforcement_evaluate(payload: EnforcementEvaluationRequest) -> dict[str, Any]:
        incident = evaluate_enforcement(payload)
        return incident_store.append(incident)

    @app.get(
        "/incidents",
        response_model=list[EnforcementEvaluationResponse],
        tags=["Enforcement"],
        summary="List recent incident evidence packages",
        description="Return recently persisted enforcement evaluations.",
    )
    def incidents(limit: int = 25) -> list[dict[str, Any]]:
        return incident_store.list(limit=limit)

    @app.patch(
        "/incidents/{incident_id}/status",
        response_model=EnforcementEvaluationResponse,
        tags=["Enforcement"],
        summary="Update incident workflow status",
        description="Move an incident through Open, Dispatched, Resolved, or Dismissed for the operations demo.",
    )
    def incident_status(incident_id: str, payload: IncidentStatusUpdate) -> dict[str, Any]:
        updated = incident_store.update(
            incident_id,
            {
                "status": payload.status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Incident not found.")
        if isinstance(updated.get("evidence"), dict):
            updated["evidence"]["status"] = payload.status
        return updated

    @app.get(
        "/incidents/{incident_id}",
        response_model=EnforcementEvaluationResponse,
        tags=["Enforcement"],
        summary="Get one incident evidence package",
        description="Return one persisted incident by id.",
    )
    def incident_detail(incident_id: str) -> dict[str, Any]:
        incident = incident_store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found.")
        return incident

    return app


app = create_app()
