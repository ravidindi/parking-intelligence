import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app


class ParkingApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_hotspots_limit(self) -> None:
        response = self.client.get("/hotspots?limit=3")
        self.assertEqual(response.status_code, 200)
        rows = response.json()
        self.assertEqual(len(rows), 3)
        corridor_keys = {(row["junction"], row["police_station"]) for row in rows}
        self.assertEqual(len(corridor_keys), len(rows))
        self.assertIn("active_hours", rows[0])
        self.assertGreaterEqual(rows[0]["records_merged"], 1)

    def test_offences_use_public_traffic_fine_codes(self) -> None:
        response = self.client.get("/offences")
        self.assertEqual(response.status_code, 200)
        codes = {item["code"]: item for item in response.json()}
        self.assertIn("MV_ACT_177", codes)
        self.assertIn("190_CLAUSE_117", codes)
        self.assertEqual(codes["MV_ACT_177"]["legal_section"], "Section 177 of M.V. Act")
        self.assertEqual(codes["190_CLAUSE_117"]["fine_amount"], 1000)
        self.assertEqual(codes["SEC_15_2_RW_SEC_177_TOWING_MTV"]["fine_amount"], 2250)

    def test_coordinate_prediction(self) -> None:
        response = self.client.post(
            "/predict/location",
            json={"latitude": 12.9155, "longitude": 77.6238, "hour": 19, "radius_km": 3},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("risk_score", payload)
        self.assertIn("distance_km", payload)

    def test_detection_and_enforcement_evaluation(self) -> None:
        image_path = Path("test_assets/parked-car-test.png")
        with image_path.open("rb") as handle:
            detection_response = self.client.post(
                "/detect/frame",
                files={"image": ("parked-car-test.png", handle, "image/png")},
                data={
                    "camera_id": "cam-test",
                    "restricted_zone": "true",
                    "stationary_threshold_seconds": "90",
                    "observed_seconds": "210",
                    "offence_context": "main_road",
                    "mock_detection": "true",
                },
            )
        self.assertEqual(detection_response.status_code, 200)
        detection = detection_response.json()
        self.assertEqual(detection["alert_count"], 1)
        alert = detection["detections"][0]
        self.assertEqual(alert["offence_code"], "190_CLAUSE_117")
        self.assertEqual(alert["offence_label"], "Wrong Parking")
        self.assertEqual(alert["offence_legal_section"], "190 Clause 117")
        self.assertEqual(alert["offence_fine_amount"], 1000)

        risk_response = self.client.post(
            "/predict/location",
            json={"latitude": 12.9155, "longitude": 77.6238, "hour": 19, "radius_km": 3},
        )
        self.assertEqual(risk_response.status_code, 200)

        evaluation_response = self.client.post(
            "/enforcement/evaluate",
            json={
                "camera": {
                    "camera_id": "cam-test",
                    "label": "Test Camera",
                    "zone_type": "Restricted test corridor",
                    "enforcement_level": "High",
                    "restricted_zone": True,
                    "observed_seconds": 210,
                    "alert_threshold": 90,
                },
                "detection": detection,
                "risk": risk_response.json(),
                "frame_name": "parked-car-test.png",
            },
        )
        self.assertEqual(evaluation_response.status_code, 200)
        evaluation = evaluation_response.json()
        self.assertIn(evaluation["priority"], {"Critical", "High", "Watch", "Monitor"})
        self.assertIn("evidence", evaluation)
        self.assertEqual(evaluation["status"], "Open")
        self.assertIn("congestion_impact", evaluation)
        self.assertEqual(evaluation["evidence"]["frame_name"], "parked-car-test.png")
        self.assertIn("dispatch_plan", evaluation)
        self.assertGreater(evaluation["dispatch_plan"]["personnel_count"], 0)
        self.assertGreater(evaluation["dispatch_plan"]["unit_count"], 0)
        self.assertIn("Traffic Police Station", evaluation["dispatch_plan"]["from_station"])
        self.assertIn(evaluation["dispatch_plan"]["target_stop"], evaluation["dispatch_plan"]["instruction"])

        status_response = self.client.patch(
            f"/incidents/{evaluation['incident_id']}/status",
            json={"status": "Dispatched"},
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "Dispatched")

    def test_congestion_impact_and_before_after(self) -> None:
        detection = {
            "camera_id": "cam-test",
            "restricted_zone": True,
            "vehicle_count": 2,
            "alert_count": 1,
            "detections": [
                {
                    "class_name": "car",
                    "confidence": 0.99,
                    "bbox": [80, 120, 260, 260],
                    "centroid": [170, 190],
                    "stationary_seconds": 210,
                    "alert": True,
                    "alert_reason": "restricted zone; stationary for 210s",
                }
            ],
            "recommendation": "Generate illegal parking alert and dispatch nearest enforcement unit",
        }
        risk_response = self.client.post(
            "/predict/location",
            json={"latitude": 12.9155, "longitude": 77.6238, "hour": 19, "radius_km": 3},
        )
        self.assertEqual(risk_response.status_code, 200)
        camera = {
            "camera_id": "cam-test",
            "label": "Test Camera",
            "zone_type": "Restricted test corridor",
            "enforcement_level": "High",
            "restricted_zone": True,
            "observed_seconds": 210,
            "alert_threshold": 90,
        }
        impact_response = self.client.post(
            "/impact/estimate",
            json={"camera": camera, "detection": detection, "risk": risk_response.json()},
        )
        self.assertEqual(impact_response.status_code, 200)
        impact = impact_response.json()
        self.assertGreater(impact["impact_score"], 0)
        self.assertIn("lane_capacity_loss_pct", impact)
        self.assertIn("estimated_delay_minutes", impact)

        analytics_response = self.client.get("/analytics/before-after?limit=5")
        self.assertEqual(analytics_response.status_code, 200)
        analytics = analytics_response.json()
        self.assertEqual(analytics["summary"]["hotspots_analyzed"], 5)
        self.assertEqual(len(analytics["corridors"]), 5)
        self.assertIn("delay_saved_minutes", analytics["summary"])


if __name__ == "__main__":
    unittest.main()
