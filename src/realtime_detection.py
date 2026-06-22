"""Real-time vehicle detection and stationary-vehicle alerting."""

from __future__ import annotations

import math
import os
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from src.offence_codes import classify_parking_offence


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]
    centroid: tuple[float, float]
    stationary_seconds: float = 0.0
    alert: bool = False
    alert_reason: str | None = None
    offence_code: str | None = None
    offence_label: str | None = None
    offence_fine_amount: int | None = None
    offence_source: str | None = None
    offence_legal_section: str | None = None
    offence_subtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(value, 2) for value in self.bbox],
            "centroid": [round(self.centroid[0], 2), round(self.centroid[1], 2)],
            "stationary_seconds": round(self.stationary_seconds, 2),
            "alert": self.alert,
            "alert_reason": self.alert_reason,
            "offence_code": self.offence_code,
            "offence_label": self.offence_label,
            "offence_fine_amount": self.offence_fine_amount,
            "offence_source": self.offence_source,
            "offence_legal_section": self.offence_legal_section,
            "offence_subtype": self.offence_subtype,
        }


@dataclass
class Track:
    track_id: str
    centroid: tuple[float, float]
    first_seen: float
    last_seen: float
    stationary_since: float
    class_name: str


class YoloVehicleDetector:
    """Thin wrapper around Ultralytics YOLO."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def detect(self, image: np.ndarray, confidence: float = 0.25) -> list[Detection]:
        model = self._load_model()
        results = model.predict(image, conf=confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(names[class_id]).lower()
                if class_name not in VEHICLE_CLASSES:
                    continue

                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                centroid = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=[float(value) for value in xyxy],
                        centroid=centroid,
                    )
                )

        return detections

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model

            try:
                config_dir = Path(".ultralytics").resolve()
                mpl_config_dir = Path(".mplconfig").resolve()
                config_dir.mkdir(exist_ok=True)
                mpl_config_dir.mkdir(exist_ok=True)
                os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
                os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Real-time detection requires ultralytics. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

            self._model = YOLO(self.model_name)
            return self._model


class StationaryVehicleTracker:
    """Track centroids across camera frames and estimate stationary duration."""

    def __init__(self, movement_tolerance_px: float = 24.0, stale_after_seconds: float = 900.0) -> None:
        self.movement_tolerance_px = movement_tolerance_px
        self.stale_after_seconds = stale_after_seconds
        self._tracks_by_camera: dict[str, dict[str, Track]] = {}
        self._lock = Lock()

    def update(
        self,
        camera_id: str,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> list[Detection]:
        now = timestamp or time.time()
        with self._lock:
            tracks = self._tracks_by_camera.setdefault(camera_id, {})
            self._drop_stale_tracks(tracks, now)

            for detection in detections:
                track = self._match_track(tracks, detection)
                if track is None:
                    track = self._create_track(tracks, detection, now)
                else:
                    distance = _distance(track.centroid, detection.centroid)
                    if distance > self.movement_tolerance_px:
                        track.stationary_since = now
                    track.centroid = detection.centroid
                    track.last_seen = now
                    track.class_name = detection.class_name

                detection.stationary_seconds = max(0.0, now - track.stationary_since)

        return detections

    def reset(self, camera_id: str | None = None) -> None:
        with self._lock:
            if camera_id is None:
                self._tracks_by_camera.clear()
            else:
                self._tracks_by_camera.pop(camera_id, None)

    def _match_track(self, tracks: dict[str, Track], detection: Detection) -> Track | None:
        best_track = None
        best_distance = math.inf
        for track in tracks.values():
            distance = _distance(track.centroid, detection.centroid)
            if distance < best_distance and distance <= self.movement_tolerance_px:
                best_distance = distance
                best_track = track
        return best_track

    def _create_track(self, tracks: dict[str, Track], detection: Detection, now: float) -> Track:
        track_id = f"track-{len(tracks) + 1}-{int(now * 1000)}"
        track = Track(
            track_id=track_id,
            centroid=detection.centroid,
            first_seen=now,
            last_seen=now,
            stationary_since=now,
            class_name=detection.class_name,
        )
        tracks[track_id] = track
        return track

    def _drop_stale_tracks(self, tracks: dict[str, Track], now: float) -> None:
        stale_ids = [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen > self.stale_after_seconds
        ]
        for track_id in stale_ids:
            tracks.pop(track_id, None)


def decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Image decoding requires opencv-python. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        image = _decode_with_pillow(image_bytes)
    return image


def _decode_with_pillow(image_bytes: bytes) -> np.ndarray:
    try:
        import cv2
        from PIL import Image
        from pillow_heif import register_heif_opener
        from io import BytesIO
    except ImportError as exc:
        raise ValueError(
            "Uploaded file is not a readable image. For HEIC/HEIF support install pillow-heif."
        ) from exc

    register_heif_opener()
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def detect_vehicles_safely(image_bytes: bytes, confidence: float = 0.25) -> list[Detection]:
    """Run YOLO in a subprocess so native crashes cannot kill the API server."""
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as image_file:
        image_file.write(image_bytes)
        image_path = image_file.name

    try:
        command = [
            sys.executable,
            "-m",
            "src.yolo_worker",
            image_path,
            str(confidence),
        ]
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "YOLO worker failed"
            raise RuntimeError(message.splitlines()[-1])

        payload = json.loads(result.stdout)
        return [
            Detection(
                class_name=item["class_name"],
                confidence=float(item["confidence"]),
                bbox=[float(value) for value in item["bbox"]],
                centroid=(float(item["centroid"][0]), float(item["centroid"][1])),
            )
            for item in payload.get("detections", [])
        ]
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("YOLO inference timed out. Try mock detection for the live demo.") from exc
    finally:
        Path(image_path).unlink(missing_ok=True)


def annotate_alerts(
    detections: list[Detection],
    restricted_zone: bool,
    stationary_threshold_seconds: float,
    offence_context: str | None = None,
) -> list[Detection]:
    for detection in detections:
        reasons: list[str] = []
        if restricted_zone:
            reasons.append("restricted zone")
        if detection.stationary_seconds >= stationary_threshold_seconds:
            reasons.append(f"stationary for {int(detection.stationary_seconds)}s")

        detection.alert = restricted_zone and detection.stationary_seconds >= stationary_threshold_seconds
        detection.alert_reason = "; ".join(reasons) if detection.alert else None
        offence = classify_parking_offence(
            alert=detection.alert,
            restricted_zone=restricted_zone,
            offence_context=offence_context,
            vehicle_class=detection.class_name,
        )
        if offence:
            detection.offence_code = offence["code"]
            detection.offence_label = offence["label"]
            detection.offence_fine_amount = offence["fine_amount"]
            detection.offence_source = offence["source_url"]
            detection.offence_legal_section = offence["legal_section"]
            detection.offence_subtype = offence["detected_subtype"]
    return detections


def summarize_detection(
    detections: list[Detection],
    camera_id: str,
    restricted_zone: bool,
) -> dict[str, Any]:
    alerts = [detection for detection in detections if detection.alert]
    return {
        "camera_id": camera_id,
        "restricted_zone": restricted_zone,
        "vehicle_count": len(detections),
        "alert_count": len(alerts),
        "detections": [detection.to_dict() for detection in detections],
        "recommendation": _recommend_realtime(alerts),
    }


def synthetic_detection() -> Detection:
    """Deterministic sample detection used by docs/tests without a model download."""
    return Detection(
        class_name="car",
        confidence=0.99,
        bbox=[80.0, 120.0, 260.0, 260.0],
        centroid=(170.0, 190.0),
    )


def model_cache_path(model_name: str) -> Path:
    return Path(model_name).expanduser()


def _recommend_realtime(alerts: list[Detection]) -> str:
    if alerts:
        return "Generate illegal parking alert and dispatch nearest enforcement unit"
    return "No stationary restricted-zone alert"


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])
