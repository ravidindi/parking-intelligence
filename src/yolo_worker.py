"""Subprocess entry point for YOLO inference."""

from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

from src.realtime_detection import YoloVehicleDetector


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is not None:
        return image

    register_heif_opener()
    with path.open("rb") as handle:
        pil_image = Image.open(BytesIO(handle.read())).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m src.yolo_worker <image_path> [confidence]", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplconfig"))

    image_path = Path(sys.argv[1])
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    try:
        image = read_image(image_path)
    except Exception:
        print("Uploaded file is not a readable image.", file=sys.stderr)
        return 3

    detections = YoloVehicleDetector().detect(image, confidence=confidence)
    print(json.dumps({"detections": [detection.to_dict() for detection in detections]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
